"""
Compares length between the 82 included cohort targets and the 69
targets removed at the deduplication step, using the ground-truth PDB
files in ground_truth_pdbs/ (151 structures with usable residue
correspondence, post mmCIF-exclusion, pre-deduplication).

Does not cover the 45 structures excluded for being available only in
mmCIF format (no file was downloaded for those).

Usage:
    python compare_excluded_included.py \
        --ground-truth-dir ground_truth_pdbs \
        --results dispersion_error_results_deduped.json \
        --output excluded_vs_included_comparison.json
"""
import argparse
import glob
import json
import os
import statistics


def count_residues_from_pdb(filepath):
    valid_bases = {"A", "C", "G", "U"}
    seen = set()
    with open(filepath) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            if resname not in valid_bases:
                continue
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            seen.add((chain, resseq, icode))
    return len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="excluded_vs_included_comparison.json")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    included_pdbids = {r["target"].split("_")[0].upper() for r in results}
    included_lengths = [r["length"] for r in results]

    all_gt_files = glob.glob(os.path.join(args.ground_truth_dir, "*.pdb"))
    print(f"Ground-truth files found: {len(all_gt_files)}")
    print(f"Included (final cohort) PDB IDs: {len(included_pdbids)}")

    excluded_lengths = []
    n_excluded_files = 0
    for fpath in all_gt_files:
        pdb_id = os.path.basename(fpath).replace(".pdb", "").upper()
        if pdb_id in included_pdbids:
            continue
        try:
            n_res = count_residues_from_pdb(fpath)
            if n_res > 0:
                excluded_lengths.append(n_res)
                n_excluded_files += 1
        except Exception as e:
            print(f"  error reading {fpath}: {e}")

    print(f"Excluded-by-deduplication files characterized: {n_excluded_files}")
    print("\nNOTE: this comparison covers included (82) vs. deduplication-excluded "
          f"({n_excluded_files}) only. The 45 mmCIF-only-excluded structures have no "
          "file in ground_truth_dir and are NOT represented in this comparison -- "
          "any conclusion here is scoped accordingly, not a full picture of all "
          "196 original candidates.")

    print(f"\nIncluded length: mean={statistics.mean(included_lengths):.1f}, "
          f"median={statistics.median(included_lengths):.1f}, n={len(included_lengths)}")
    if excluded_lengths:
        print(f"Deduplication-excluded length: mean={statistics.mean(excluded_lengths):.1f}, "
              f"median={statistics.median(excluded_lengths):.1f}, n={len(excluded_lengths)}")
        try:
            from scipy.stats import mannwhitneyu
            u_stat, p_val = mannwhitneyu(included_lengths, excluded_lengths)
            print(f"Mann-Whitney U test (included vs. dedup-excluded length): p={p_val:.4f}")
        except ImportError:
            print("(scipy not available for formal test)")

    with open(args.output, "w") as f:
        json.dump({"included_lengths": included_lengths,
                    "excluded_lengths": excluded_lengths,
                    "scope_note": "excludes the 45 mmCIF-only structures; "
                                   "covers included vs. deduplication-excluded only"}, f)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
