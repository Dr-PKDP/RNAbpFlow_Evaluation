"""
analyze_stochasticity_source.py -- Computes dispersion and error for
Condition A (dropout OFF, generative sampling ON) using the same
residue-level sequence-alignment methodology as analyze_dispersion_vs_error.py,
then compares directly against the existing Condition B data.

The alignment logic is reused unchanged, so the full cohort is recovered in the
same way as in the primary analysis.

Requires the original RCSB metadata file for predicted sequences
(predictions cover the full submitted sequence, so the sequence must
come from the original metadata, not be inferred from the prediction
file itself) -- same file used by the primary analysis:
    RNAbpFlow_training/rna_monomers_2026-09-01.json

Usage:
    python analyze_stochasticity_source.py \
        --dropout-off-dir Predictions_dropout_off \
        --dropout-on-results dispersion_error_results_deduped.json \
        --ground-truth-dir ground_truth_pdbs \
        --metadata RNAbpFlow_training/rna_monomers_2026-09-01.json \
        --output stochasticity_source_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from Bio import Align
from Bio.PDB import PDBParser
from scipy.stats import pearsonr, spearmanr

_NA_LETTER_BY_RESNAME = {
    "DA": "A", "DC": "C", "DG": "G", "DT": "U",
    "A": "A", "C": "C", "G": "G", "U": "U",
}
MAX_MISMATCH_RATE = 0.05


def build_aligner():
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    return aligner


def get_gt_seq_and_coords(pdb_path, chain_id):
    """Same extraction convention as the original featurization pipeline:
    residue accepted only if it is a standard nucleotide type AND has a
    resolved C4' atom."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = list(structure.get_models())[0]
    chain = model[chain_id]
    seq_chars, coords = [], []
    for res in chain:
        if res.id[0] != " ":
            continue
        resname = res.resname.strip()
        if resname not in _NA_LETTER_BY_RESNAME:
            continue
        if "C4'" not in res:
            continue
        seq_chars.append(_NA_LETTER_BY_RESNAME[resname])
        coords.append(res["C4'"].coord)
    return "".join(seq_chars), np.array(coords)


def get_pred_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    coords = []
    for atom in structure.get_atoms():
        if atom.get_name() == "C4'":
            coords.append(atom.coord)
    return np.array(coords)


def align_correspondence(pred_seq, gt_seq):
    aligner = build_aligner()
    alignment = aligner.align(pred_seq, gt_seq)[0]
    aligned_pred, aligned_gt = alignment[0], alignment[1]
    pred_indices, gt_indices = [], []
    n_compared = n_mismatch = 0
    pred_pos = gt_pos = 0
    for pc, gc in zip(aligned_pred, aligned_gt):
        if pc != "-" and gc != "-":
            n_compared += 1
            if pc != gc:
                n_mismatch += 1
            pred_indices.append(pred_pos)
            gt_indices.append(gt_pos)
        if pc != "-":
            pred_pos += 1
        if gc != "-":
            gt_pos += 1
    mismatch_rate = n_mismatch / n_compared if n_compared > 0 else 1.0
    return np.array(pred_indices), np.array(gt_indices), mismatch_rate


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
    if a.shape != b.shape or a.shape[0] < 3:
        return None
    a_aligned = kabsch_align(a, b)
    return np.sqrt(np.mean(np.sum((a_aligned - b) ** 2, axis=1)))


def get_sequences_by_id(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    seqs = {}
    for e in data["entries"]:
        rcsb_id = e.get("rcsb_id")
        for pe in e.get("polymer_entities", []) or []:
            ep = pe.get("entity_poly", {}) or {}
            if ep.get("rcsb_entity_polymer_type") == "RNA":
                seq = ep.get("pdbx_seq_one_letter_code_can")
                if seq:
                    seqs[rcsb_id] = seq.replace("\n", "").upper()
    return seqs


def compute_dispersion_error(pred_dir, gt_dir, targets, sequences):
    """targets: list of (pdb_id, chain) tuples."""
    dispersions, errors, target_names, lengths = [], [], [], []
    n_no_files = n_too_few_samples = n_extract_error = n_alignment_rejected = n_used = 0

    for pdb_id, chain in targets:
        target_id = f"{pdb_id.lower()}_{chain}"
        target_pred_dir = pred_dir / target_id
        gt_path = gt_dir / f"{pdb_id.upper()}.pdb"
        pred_seq = sequences.get(pdb_id.upper())

        if not target_pred_dir.exists() or not gt_path.exists() or pred_seq is None:
            n_no_files += 1
            continue

        sample_files = sorted(target_pred_dir.glob("Sample_*.pdb"))
        if len(sample_files) < 2:
            n_too_few_samples += 1
            continue

        try:
            gt_seq, gt_coords = get_gt_seq_and_coords(gt_path, chain)
            sample_coords_full = [get_pred_coords(f) for f in sample_files]
        except Exception as exc:
            n_extract_error += 1
            print(f"  extract error {target_id}: {exc}")
            continue

        if len(gt_seq) == 0 or any(s.shape[0] != len(pred_seq) for s in sample_coords_full):
            n_extract_error += 1
            continue

        try:
            pred_idx, gt_idx, mismatch = align_correspondence(pred_seq, gt_seq)
        except Exception:
            n_alignment_rejected += 1
            continue
        if mismatch > MAX_MISMATCH_RATE or len(pred_idx) < 10:
            n_alignment_rejected += 1
            print(f"  rejected {target_id}: mismatch={mismatch:.1%}, aligned={len(pred_idx)}")
            continue

        gt_common = gt_coords[gt_idx]
        sample_common = [s[pred_idx] for s in sample_coords_full]

        pairwise = []
        for i in range(len(sample_common)):
            for j in range(i + 1, len(sample_common)):
                r = rmsd(sample_common[i], sample_common[j])
                if r is not None:
                    pairwise.append(r)
        if not pairwise:
            continue
        dispersion = np.mean(pairwise)

        errs = [rmsd(s, gt_common) for s in sample_common]
        errs = [e for e in errs if e is not None]
        if not errs:
            continue
        error = np.mean(errs)

        dispersions.append(dispersion)
        errors.append(error)
        target_names.append(target_id)
        lengths.append(len(gt_idx))
        n_used += 1

    print(f"\nCondition A extraction summary:")
    print(f"  No files/sequence: {n_no_files}")
    print(f"  Too few samples: {n_too_few_samples}")
    print(f"  Extraction error / shape mismatch: {n_extract_error}")
    print(f"  Alignment rejected: {n_alignment_rejected}")
    print(f"  Used: {n_used}")

    return pd.DataFrame({"target": target_names, "dispersion": dispersions,
                          "error": errors, "length": lengths})


def paired_loocv_incremental_value(log_length, log_dispersion, log_error, df=4):
    n = len(log_length)
    data_full = pd.DataFrame({"log_length": log_length})
    formula = f"cr(log_length, df={df}) - 1"
    design_info = patsy.dmatrix(formula, data_full, return_type="dataframe").design_info

    e0_sq, e1_sq = [], []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        d_train = pd.DataFrame({"log_length": [log_length[j] for j in train_idx]})
        X0_train = patsy.build_design_matrices([design_info], d_train, return_type="dataframe")[0]
        y_train = np.array([log_error[j] for j in train_idx])

        m0 = sm.OLS(y_train, X0_train).fit()
        X1_train = X0_train.copy()
        X1_train["log_dispersion"] = [log_dispersion[j] for j in train_idx]
        m1 = sm.OLS(y_train, X1_train).fit()

        d_test = pd.DataFrame({"log_length": [log_length[i]]})
        X0_test = patsy.build_design_matrices([design_info], d_test, return_type="dataframe")[0]
        X1_test = X0_test.copy()
        X1_test["log_dispersion"] = [log_dispersion[i]]

        pred0 = m0.predict(X0_test)[0]
        pred1 = m1.predict(X1_test)[0]
        e0_sq.append((log_error[i] - pred0) ** 2)
        e1_sq.append((log_error[i] - pred1) ** 2)

    e0_sq, e1_sq = np.array(e0_sq), np.array(e1_sq)
    d = e0_sq - e1_sq
    mean_diff = d.mean()
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(d, size=n, replace=True).mean() for _ in range(10000)]
    ci = np.percentile(boot_means, [2.5, 97.5])
    rmse0, rmse1 = np.sqrt(e0_sq.mean()), np.sqrt(e1_sq.mean())
    rel_improvement = (rmse0 - rmse1) / rmse0 * 100
    return {
        "rmse0": rmse0, "rmse1": rmse1, "relative_improvement_pct": rel_improvement,
        "mean_delta_mse": mean_diff, "ci_95": list(ci),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout-off-dir", required=True)
    ap.add_argument("--dropout-on-results", required=True)
    ap.add_argument("--ground-truth-dir", required=True)
    ap.add_argument("--metadata", required=True,
                     help="Original RCSB metadata JSON (e.g. rna_monomers_2026-09-01.json)")
    ap.add_argument("--output", default="stochasticity_source_results.json")
    args = ap.parse_args()

    dropout_off_dir = Path(args.dropout_off_dir)
    ground_truth_dir = Path(args.ground_truth_dir)

    with open(args.dropout_on_results, encoding="utf-8") as f:
        b_raw = json.load(f)
    targets = [tuple(entry["target"].rsplit("_", 1)) for entry in b_raw]
    print(f"Derived {len(targets)} targets from {args.dropout_on_results}")

    print("Loading RCSB metadata for predicted sequences...")
    sequences = get_sequences_by_id(args.metadata)
    print(f"  {len(sequences)} RNA sequences loaded")

    print("Computing Condition A (dropout OFF) dispersion and error...")
    df_a = compute_dispersion_error(dropout_off_dir, ground_truth_dir, targets, sequences)

    df_b = pd.DataFrame(b_raw)
    df_b = df_b.rename(columns={c: c.lower() for c in df_b.columns})

    merged = df_a.merge(df_b, on="target", suffixes=("_a", "_b"))
    print(f"\n{len(merged)} targets present in both conditions (of {len(targets)} total)")

    results = {"n_targets_matched": len(merged), "n_targets_total": len(targets)}

    rho_a, p_rho_a = spearmanr(merged["dispersion_a"], merged["error_a"])
    r_a, p_r_a = pearsonr(merged["dispersion_a"], merged["error_a"])
    results["condition_a_raw_association"] = {
        "n": len(merged), "spearman_rho": rho_a, "spearman_p": p_rho_a,
        "pearson_r": r_a, "pearson_p": p_r_a,
    }

    log_length = np.log(merged["length_a"])
    log_disp_a = np.log(merged["dispersion_a"])
    log_err_a = np.log(merged["error_a"])
    results["condition_a_incremental_value"] = paired_loocv_incremental_value(
        log_length.values, log_disp_a.values, log_err_a.values
    )

    disp_corr_ab, _ = pearsonr(merged["dispersion_a"], merged["dispersion_b"])
    ratio = (merged["dispersion_a"] / merged["dispersion_b"]).values
    results["dispersion_a_vs_b"] = {
        "pearson_r_between_conditions": disp_corr_ab,
        "mean_ratio_a_over_b": float(np.mean(ratio)),
        "median_ratio_a_over_b": float(np.median(ratio)),
    }

    print(json.dumps(results, indent=2, default=float))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
