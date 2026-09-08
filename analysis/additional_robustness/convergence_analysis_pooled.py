"""
Convergence check using the pooled 30-sample set (20 original samples
from Predictions/ + 10 from Predictions_extra10/), via without-
replacement subsetting for n < 30. For n=30, only one subset exists, so
this is reported as a single point estimate.

Usage:
    python convergence_analysis_pooled.py \
        --original-dir Predictions --extra-dir Predictions_extra10 \
        --results dispersion_error_results_deduped.json \
        --output convergence_diagnostics_pooled.json
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
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--extra-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--subset-sizes", default="5,10,15,20,25")
    parser.add_argument("--n-draws", type=int, default=30)
    parser.add_argument("--output", default="convergence_diagnostics_pooled.json")
    args = parser.parse_args()

    subset_sizes = [int(x) for x in args.subset_sizes.split(",")]
    rng = np.random.default_rng(0)
    original_dir = Path(args.original_dir)
    extra_dir = Path(args.extra_dir)

    with open(args.results) as f:
        data = json.load(f)

    results = []
    n_ok, n_missing, n_shape_mismatch = 0, 0, 0

    for record in data:
        target_id = record["target"]
        orig_target_dir = original_dir / target_id
        extra_target_dir = extra_dir / target_id

        if not orig_target_dir.exists() or not extra_target_dir.exists():
            n_missing += 1
            continue

        orig_files = sorted(orig_target_dir.glob("Sample_*.pdb"))
        extra_files = sorted(extra_target_dir.glob("Sample_*.pdb"))
        all_files = orig_files + extra_files

        if len(orig_files) != 20 or len(extra_files) != 10:
            print(f"  UNEXPECTED counts for {target_id}: {len(orig_files)} original, "
                  f"{len(extra_files)} extra -- expected 20 and 10")

        try:
            all_coords = [get_c4_coords(f) for f in all_files]
        except Exception as e:
            print(f"  ERROR reading {target_id}: {e}")
            continue

        ref_shape = all_coords[0].shape
        if any(c.shape != ref_shape for c in all_coords):
            n_shape_mismatch += 1
            continue

        n_pool = len(all_coords)
        target_result = {"target": target_id, "n_pool": n_pool}

        # Without-replacement subsetting -- valid for every size < n_pool
        for size in subset_sizes:
            if size >= n_pool:
                continue
            draw_dispersions = []
            for _ in range(args.n_draws):
                idx = rng.choice(n_pool, size, replace=False)
                subset = [all_coords[i] for i in idx]
                draw_dispersions.append(mean_pairwise_dispersion(subset))
            draw_dispersions = np.array(draw_dispersions)
            mean_d = draw_dispersions.mean()
            cv = draw_dispersions.std() / mean_d if mean_d > 0 else np.nan
            target_result[f"n{size}_mean_dispersion"] = float(mean_d)
            target_result[f"n{size}_cv"] = float(cv)

        # Full pool (n=30): single point estimate, no CV -- only one possible subset
        full_pool_dispersion = mean_pairwise_dispersion(all_coords)
        target_result[f"n{n_pool}_point_estimate"] = float(full_pool_dispersion)

        results.append(target_result)
        n_ok += 1

    print(f"Processed: {n_ok}, missing target dir (original or extra): {n_missing}, "
          f"shape mismatch: {n_shape_mismatch}")

    if results:
        for size in subset_sizes:
            key = f"n{size}_cv"
            cvs = [r[key] for r in results if key in r]
            if cvs:
                print(f"  n={size} (without replacement, from 30-pool): "
                      f"mean CV = {np.mean(cvs):.4f}, median = {np.median(cvs):.4f}")
        full_key = f"n30_point_estimate"
        full_vals = [r[full_key] for r in results if full_key in r]
        if full_vals:
            print(f"  n=30 (full pool, single point estimate, no CV defined): "
                  f"mean dispersion = {np.mean(full_vals):.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nSaved diagnostics for {len(results)} targets to {args.output}")


if __name__ == "__main__":
    main()
