"""
Pairwise sequence identity: within-cohort near-duplicate detection, and
identity against RNA3DB training sequences.

Percent identity is global alignment (Bio.Align.PairwiseAligner),
normalized by the longer of the two sequences.

Usage:
    python sequence_leakage_audit.py \
        --results dispersion_error_results_deduped.json \
        --ground-truth-dir ground_truth_pdbs \
        --rna3db-sequences <path> \
        --output leakage_audit_results.json
"""
import argparse
import glob
import json
import os

import numpy as np
from Bio.Align import PairwiseAligner


def extract_sequence_from_pdb(filepath, chain_id=None):
    valid_bases = {"A", "C", "G", "U"}
    seen = set()
    seq = []
    with open(filepath) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            this_chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            if chain_id is not None and this_chain != chain_id:
                continue
            if resname not in valid_bases:
                continue
            key = (this_chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            seq.append((int(resseq), resname))
    seq.sort(key=lambda x: x[0])
    return "".join(b for _, b in seq)


def percent_identity(seq1, seq2, aligner):
    """Percent identity normalized by the longer sequence's length."""
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    alignment = aligner.align(seq1, seq2)[0]
    aligned1, aligned2 = str(alignment[0]), str(alignment[1])
    matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
    return matches / max(len(seq1), len(seq2))


def make_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    return aligner


def load_cohort_sequences(results_path, ground_truth_dir):
    with open(results_path) as f:
        data = json.load(f)
    sequences = {}
    for record in data:
        target = record["target"]
        pdb_id, chain_id = target.split("_", 1) if "_" in target else (target, None)
        fpath = os.path.join(ground_truth_dir, f"{pdb_id.upper()}.pdb")
        if not os.path.exists(fpath):
            continue
        seq = extract_sequence_from_pdb(fpath, chain_id)
        if len(seq) == int(round(record["length"])):
            sequences[target] = seq
    return sequences


def within_cohort_clustering(sequences, thresholds=(0.80, 0.60)):
    aligner = make_aligner()
    targets = list(sequences.keys())
    n = len(targets)
    flagged = {t: [] for t in thresholds}
    print(f"Computing {n*(n-1)//2} pairwise alignments for within-cohort clustering...")
    for i in range(n):
        for j in range(i + 1, n):
            pid = percent_identity(sequences[targets[i]], sequences[targets[j]], aligner)
            for t in thresholds:
                if pid >= t:
                    flagged[t].append((targets[i], targets[j], round(pid, 3)))
    return flagged


def against_training_data(sequences, rna3db_path, thresholds=(0.80, 0.60)):
    if not rna3db_path or not os.path.exists(rna3db_path):
        return None
    with open(rna3db_path) as f:
        training_seqs = json.load(f)  # expects {chain_id: sequence}
    aligner = make_aligner()
    results = {}
    for target, seq in sequences.items():
        best_pid, best_match = 0.0, None
        for train_id, train_seq in training_seqs.items():
            pid = percent_identity(seq, train_seq, aligner)
            if pid > best_pid:
                best_pid, best_match = pid, train_id
        results[target] = {"nearest_training_chain": best_match, "percent_identity": round(best_pid, 3)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--rna3db-sequences", default=None,
                         help="JSON file mapping RNA3DB training chain IDs to sequences.")
    parser.add_argument("--output", default="leakage_audit_results.json")
    args = parser.parse_args()

    print("Extracting cohort sequences...")
    sequences = load_cohort_sequences(args.results, args.ground_truth_dir)
    print(f"Extracted {len(sequences)} sequences with verified matching length.\n")

    print("=== (A) Within-cohort clustering ===")
    flagged = within_cohort_clustering(sequences)
    for t, pairs in flagged.items():
        print(f"Pairs at or above {t*100:.0f}% identity: {len(pairs)}")
        for a, b, pid in pairs[:10]:
            print(f"    {a} <-> {b}: {pid*100:.1f}%")
        if len(pairs) > 10:
            print(f"    ... and {len(pairs)-10} more")

    print("\n=== (B) Against RNA3DB training data ===")
    training_results = against_training_data(sequences, args.rna3db_sequences)
    if training_results is None:
        print("SKIPPED: no --rna3db-sequences file provided or found.")
    else:
        high_identity = [(t, r) for t, r in training_results.items() if r["percent_identity"] >= 0.80]
        print(f"Targets with >=80% identity to a training chain: {len(high_identity)}")
        for t, r in high_identity[:10]:
            print(f"    {t}: {r['percent_identity']*100:.1f}% to {r['nearest_training_chain']}")

    with open(args.output, "w") as f:
        json.dump({
            "within_cohort_flagged": {str(k): v for k, v in flagged.items()},
            "against_training": training_results,
        }, f)
    print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
