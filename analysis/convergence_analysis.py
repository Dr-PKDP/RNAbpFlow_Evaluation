"""
Assesses whether 20 MC-dropout samples per target is enough for a stable
dispersion estimate, using the real per-sample PDB files already
produced by run_mc_dropout_inference.py (Sample_*.pdb files under
pred_dir/{target_id}/), read the same way and using the same C4'
Kabsch-alignment methodology as analyze_dispersion_vs_error.py, so the
convergence numbers this produces are directly comparable to the
dispersion values already reported.

Requires the raw Sample_*.pdb files to still exist on disk -- if they
were deleted after the original dispersion/error computation, this
script cannot run and a fresh GPU pass would be needed instead.

For each target, computes dispersion using bootstrap resamples (with
replacement) of size 5/10/15/20 drawn from the available samples, and
reports the coefficient of variation (CV) of the dispersion estimate
across many resamples at each size. Sampling WITH replacement is used
deliberately so that the largest subset size (20, matching the total
available) still produces genuine resample-to-resample variation rather
than being forced to select the identical full set every time, which
would happen under without-replacement sampling once size equals the
total available count.

Usage:
    python convergence_analysis.py --pred-dir <dir with target_id/Sample_*.pdb> \
        --results dispersion_error_results_deduped.json \
        --output convergence_diagnostics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


def get_c4_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                if "C4'" in res:
                    coords.append(res["C4'"].coord)
        break
    return np.array(coords)


def kabsch_rmsd(mobile, target):
    if mobile.shape != target.shape:
        return None
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    H = mobile_c.T @ target_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    aligned = (R @ mobile_c.T).T
    return np.sqrt(np.mean(np.sum((aligned - target_c) ** 2, axis=1)))


def mean_pairwise_dispersion(coords_list):
    n = len(coords_list)
    if n < 2:
        return np.nan
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            r = kabsch_rmsd(coords_list[i], coords_list[j])
            if r is not None:
                dists.append(r)
    return np.mean(dists) if dists else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--subset-sizes", default="5,10,15,20")
    parser.add_argument("--n-draws", type=int, default=30)
    parser.add_argument("--output", default="convergence_diagnostics.json")
    args = parser.parse_args()

    subset_sizes = [int(x) for x in args.subset_sizes.split(",")]
    rng = np.random.default_rng(0)
    pred_dir = Path(args.pred_dir)

    with open(args.results) as f:
        data = json.load(f)

    results = []
    n_ok, n_missing, n_too_few, n_shape_mismatch = 0, 0, 0, 0

    for record in data:
        target_id = record["target"]
        target_dir = pred_dir / target_id
        if not target_dir.exists():
            n_missing += 1
            continue

        sample_files = sorted(target_dir.glob("Sample_*.pdb"))
        if len(sample_files) < max(subset_sizes):
            n_too_few += 1
            continue

        try:
            all_coords = [get_c4_coords(f) for f in sample_files]
        except Exception as e:
            print(f"  ERROR reading {target_id}: {e}")
            continue

        ref_shape = all_coords[0].shape
        if any(c.shape != ref_shape for c in all_coords):
            n_shape_mismatch += 1
            continue

        target_result = {"target": target_id, "n_available": len(all_coords)}
        for size in subset_sizes:
            draw_dispersions = []
            for _ in range(args.n_draws):
                idx = rng.choice(len(all_coords), size, replace=True)
                subset = [all_coords[i] for i in idx]
                draw_dispersions.append(mean_pairwise_dispersion(subset))
            draw_dispersions = np.array(draw_dispersions)
            mean_d = draw_dispersions.mean()
            cv = draw_dispersions.std() / mean_d if mean_d > 0 else np.nan
            target_result[f"n{size}_mean_dispersion"] = float(mean_d)
            target_result[f"n{size}_cv"] = float(cv)
        results.append(target_result)
        n_ok += 1

    print(f"Processed: {n_ok}, missing target dir: {n_missing}, "
          f"too few samples on disk: {n_too_few}, shape mismatch: {n_shape_mismatch}")

    if results:
        for size in subset_sizes:
            cvs = [r[f"n{size}_cv"] for r in results if f"n{size}_cv" in r]
            print(f"  n={size}: mean CV across targets = {np.mean(cvs):.4f}, "
                  f"median = {np.median(cvs):.4f}")
        print("\nInterpretation: if CV at n=20 is not meaningfully lower than at "
              "n=5 or n=10, the dispersion estimate has not clearly stabilized by "
              "20 samples. A clear downward trend plateauing by n=15-20 supports "
              "20 samples as adequate.")

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nSaved diagnostics for {len(results)} targets to {args.output}")


if __name__ == "__main__":
    main()
