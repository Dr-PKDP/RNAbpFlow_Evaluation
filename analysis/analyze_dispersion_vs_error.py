"""
Computes per-target ensemble dispersion and structural error against
ground truth, using residue-level sequence alignment to establish
correspondence between the predicted structure (covering the full
input sequence) and the ground-truth structure (covering only resolved
residues), rather than requiring an exact residue-count match.

This recovers targets previously excluded solely because deposited
structures commonly leave disordered regions -- scattered anywhere in
the sequence, not necessarily at the termini -- without coordinates,
while predictions necessarily cover the full submitted sequence.
Alignment quality is checked directly (mismatch rate among aligned
positions) before a target is accepted, so a genuinely different or
misidentified sequence is excluded rather than forced through.

Usage:
    python analyze_dispersion_vs_error.py <predictions_dir> <ground_truth_dir> \\
        <external_cohort.json> <metadata.json>

Writes dispersion_error_results.json with per-target values.
"""
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio import Align
from scipy.stats import spearmanr, pearsonr

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
    """Same extraction convention as the original featurization pipeline
    (build_training_pkl.py's parse_geometry): residue accepted only if
    it is a standard nucleotide type AND has a resolved C4' atom."""
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


def main():
    pred_dir, gt_dir, cohort_path, metadata_path = (
        Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    )

    with open(cohort_path, encoding="utf-8") as f:
        cohort = json.load(f)
    sequences = get_sequences_by_id(metadata_path)

    dispersions, errors, target_names, lengths = [], [], [], []
    n_no_files = n_too_few_samples = n_extract_error = n_alignment_rejected = n_used = 0

    for entry in cohort:
        pdb_id, chain = entry["pdb_id"], entry["chain"]
        target_id = f"{pdb_id.lower()}_{chain}"
        target_pred_dir = pred_dir / target_id
        gt_path = gt_dir / f"{pdb_id.upper()}.pdb"
        pred_seq = sequences.get(pdb_id)

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
        except Exception as exc:
            n_alignment_rejected += 1
            continue
        if mismatch > MAX_MISMATCH_RATE or len(pred_idx) < 10:
            n_alignment_rejected += 1
            print(f"  rejected {target_id}: mismatch={mismatch:.1%}, "
                  f"aligned positions={len(pred_idx)}")
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
        if n_used % 10 == 0:
            print(f"  progress: {n_used} used so far (most recent: {target_id})", flush=True)

    print(f"\nNo files/sequence: {n_no_files}")
    print(f"Too few samples: {n_too_few_samples}")
    print(f"Extraction error / shape mismatch on predictions: {n_extract_error}")
    print(f"Alignment rejected (mismatch too high or too few aligned positions): {n_alignment_rejected}")
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
