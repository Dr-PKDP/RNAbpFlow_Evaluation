"""
Per-target alignment detail against the nearest RNA3DB training
sequence: global and local (Smith-Waterman-style) percent identity,
alignment coverage of both the target and the training sequence, and
80% global-identity exclusion flag.

Does not compute RNA family/Rfam classification (requires Infernal and
the Rfam covariance-model database, not used here).

Usage:
    python training_overlap_detailed.py \
        --results dispersion_error_results_deduped.json \
        --ground-truth-dir ground_truth_pdbs \
        --rna3db-sequences rna3db_train_sequences.json \
        --output training_overlap_detailed.json
"""
import argparse
import json
import os

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


def make_global_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    return aligner


def make_local_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    return aligner


def alignment_detail(seq1, seq2, aligner):
    """Returns matches, aligned_len (non-gap-on-both-sides positions),
    coverage of seq1, coverage of seq2."""
    alignment = aligner.align(seq1, seq2)[0]
    a1, a2 = str(alignment[0]), str(alignment[1])
    matches = sum(1 for a, b in zip(a1, a2) if a == b and a != '-')
    non_gap_both = sum(1 for a, b in zip(a1, a2) if a != '-' and b != '-')
    cov1 = non_gap_both / len(seq1) if len(seq1) > 0 else 0.0
    cov2 = non_gap_both / len(seq2) if len(seq2) > 0 else 0.0
    return matches, non_gap_both, cov1, cov2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--rna3db-sequences", required=True)
    parser.add_argument("--output", default="training_overlap_detailed.json")
    args = parser.parse_args()

    print("NOTE: this script computes sequence-alignment detail only. "
          "RNA family/Rfam classification is NOT computed -- Infernal and "
          "the Rfam covariance-model database are not available in this "
          "environment. Any 'RNA class/family' column would need to be "
          "added separately if that tooling becomes available.\n")

    with open(args.results) as f:
        results = json.load(f)
    with open(args.rna3db_sequences) as f:
        train_seqs = json.load(f)

    global_aligner = make_global_aligner()
    local_aligner = make_local_aligner()

    detailed = []
    for record in results:
        target = record["target"]
        pdb_id, chain_id = target.split("_", 1) if "_" in target else (target, None)
        fpath = os.path.join(args.ground_truth_dir, f"{pdb_id.upper()}.pdb")
        if not os.path.exists(fpath):
            continue
        eval_seq = extract_sequence_from_pdb(fpath, chain_id)
        if len(eval_seq) != int(round(record["length"])):
            continue

        best_pid, best_chain = 0.0, None
        for train_id, train_seq in train_seqs.items():
            matches, aligned_len, cov_eval, cov_train = alignment_detail(
                eval_seq, train_seq, global_aligner)
            pid = matches / max(len(eval_seq), len(train_seq))
            if pid > best_pid:
                best_pid, best_chain = pid, train_id

        if best_chain is None:
            continue

        train_seq = train_seqs[best_chain]
        matches_g, aligned_len_g, cov_eval_g, cov_train_g = alignment_detail(
            eval_seq, train_seq, global_aligner)
        matches_l, aligned_len_l, cov_eval_l, cov_train_l = alignment_detail(
            eval_seq, train_seq, local_aligner)
        local_pid = matches_l / aligned_len_l if aligned_len_l > 0 else 0.0

        detailed.append({
            "target": target,
            "nearest_training_chain": best_chain,
            "global_percent_identity": round(best_pid, 4),
            "global_aligned_length": aligned_len_g,
            "global_coverage_of_target": round(cov_eval_g, 4),
            "global_coverage_of_training_seq": round(cov_train_g, 4),
            "local_percent_identity": round(local_pid, 4),
            "local_aligned_length": aligned_len_l,
            "local_coverage_of_target": round(cov_eval_l, 4),
            "local_coverage_of_training_seq": round(cov_train_l, 4),
            "excluded_at_80pct_threshold": best_pid >= 0.80,
        })
        print(f"  {target}: global={best_pid*100:.1f}% (cov {cov_eval_g*100:.0f}%/{cov_train_g*100:.0f}%), "
              f"local={local_pid*100:.1f}% (cov {cov_eval_l*100:.0f}%/{cov_train_l*100:.0f}%)")

    with open(args.output, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"\nSaved {len(detailed)} detailed records to {args.output}")


if __name__ == "__main__":
    main()
