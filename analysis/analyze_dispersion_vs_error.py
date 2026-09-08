"""
Computes per-target ensemble dispersion (spread among repeated
predictions for the same target) and structural error (distance from
predictions to the real ground-truth structure), and reports the
correlation between them.

Targets are included only when the predicted and ground-truth structures
have matching residue counts after Kabsch alignment; others are
excluded.

Usage:
    python analyze_dispersion_vs_error.py <predictions_dir> <ground_truth_dir> \\
        <external_cohort.json>

Writes dispersion_error_results.json with per-target values, for use in
downstream conformal calibration and selective-risk analysis.
"""
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from scipy.stats import spearmanr, pearsonr


def get_c4_coords(pdb_path, chain_id=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    coords = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain:
                if "C4'" in res:
                    coords.append(res["C4'"].coord)
        break
    return np.array(coords)


def kabsch_align(mobile, target):
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    H = mobile_c.T @ target_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return (R @ mobile_c.T).T + target.mean(axis=0)


def rmsd(a, b):
    if a.shape != b.shape:
        return None
    a_aligned = kabsch_align(a, b)
    return np.sqrt(np.mean(np.sum((a_aligned - b) ** 2, axis=1)))


def main() -> None:
    pred_dir, gt_dir, cohort_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

    with open(cohort_path, encoding="utf-8") as f:
        cohort = json.load(f)

    dispersions, errors, target_names, lengths = [], [], [], []
    n_no_files = n_too_few_samples = n_extract_error = n_shape_mismatch = n_used = 0

    for entry in cohort:
        pdb_id, chain = entry["pdb_id"], entry["chain"]
        target_id = f"{pdb_id.lower()}_{chain}"
        target_pred_dir = pred_dir / target_id
        gt_path = gt_dir / f"{pdb_id.upper()}.pdb"

        if not target_pred_dir.exists() or not gt_path.exists():
            n_no_files += 1
            continue

        sample_files = sorted(target_pred_dir.glob("Sample_*.pdb"))
        if len(sample_files) < 2:
            n_too_few_samples += 1
            continue

        try:
            gt_coords = get_c4_coords(gt_path, chain_id=chain)
            sample_coords = [get_c4_coords(f) for f in sample_files]
        except Exception as exc:
            n_extract_error += 1
            print(f"  extract error {target_id}: {exc}")
            continue

        if (any(s.shape != sample_coords[0].shape for s in sample_coords)
                or gt_coords.shape != sample_coords[0].shape):
            n_shape_mismatch += 1
            continue

        pairwise = []
        for i in range(len(sample_coords)):
            for j in range(i + 1, len(sample_coords)):
                r = rmsd(sample_coords[i], sample_coords[j])
                if r is not None:
                    pairwise.append(r)
        if not pairwise:
            continue
        dispersion = np.mean(pairwise)

        errs = [rmsd(s, gt_coords) for s in sample_coords]
        errs = [e for e in errs if e is not None]
        if not errs:
            continue
        error = np.mean(errs)

        dispersions.append(dispersion)
        errors.append(error)
        target_names.append(target_id)
        lengths.append(gt_coords.shape[0])
        n_used += 1

    print(f"No prediction/ground-truth files: {n_no_files}")
    print(f"Too few samples: {n_too_few_samples}")
    print(f"Coordinate extraction error: {n_extract_error}")
    print(f"Shape mismatch: {n_shape_mismatch}")
    print(f"Used: {n_used}")

    if len(dispersions) < 3:
        print("Too few targets for a meaningful correlation.")
        return

    rho, rho_p = spearmanr(dispersions, errors)
    r, r_p = pearsonr(dispersions, errors)
    print(f"\nSpearman correlation: {rho:.3f} (p={rho_p:.4f})")
    print(f"Pearson correlation:  {r:.3f} (p={r_p:.4f})")

    results = [{"target": t, "dispersion": d, "error": e, "length": l}
               for t, d, e, l in zip(target_names, dispersions, errors, lengths)]
    with open("dispersion_error_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote dispersion_error_results.json ({len(results)} targets)")


if __name__ == "__main__":
    main()
