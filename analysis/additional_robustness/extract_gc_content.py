"""
Extracts GC content per target from the ground-truth PDB files
downloaded by download_ground_truth.py (ground_truth_pdbs/{PDB_ID}.pdb),
the same files used by analyze_dispersion_vs_error.py.

Usage:
    python extract_gc_content.py \
        --results dispersion_error_results_deduped.json \
        --ground-truth-dir ground_truth_pdbs \
        --output dispersion_error_results_with_gc.json

Cross-checks every extracted sequence's length against the length
already recorded for that target; any mismatch is reported and that
target excluded rather than silently trusted.
"""
import argparse
import json
import os


def extract_sequence_from_pdb(filepath, chain_id=None):
    """One base per residue, ordered by residue sequence number, using
    ATOM records only (matches analyze_dispersion_vs_error.py's own
    approach of reading directly from ATOM lines rather than SEQRES,
    since SEQRES can include residues absent from the modeled structure)."""
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


def gc_fraction(seq):
    if not seq:
        return None
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--output", default="dispersion_error_results_with_gc.json")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    n_ok, n_missing, n_mismatch, n_error = 0, 0, 0, 0
    for record in data:
        target = record["target"]
        pdb_id, chain_id = target.split("_", 1) if "_" in target else (target, None)
        fpath = os.path.join(args.ground_truth_dir, f"{pdb_id.upper()}.pdb")

        if not os.path.exists(fpath):
            print(f"  MISSING file for {target}: {fpath}")
            record["gc_content"] = None
            n_missing += 1
            continue

        try:
            seq = extract_sequence_from_pdb(fpath, chain_id)
        except Exception as e:
            print(f"  ERROR on {target}: {e}")
            record["gc_content"] = None
            n_error += 1
            continue

        if len(seq) != int(round(record["length"])):
            print(f"  LENGTH MISMATCH for {target}: extracted {len(seq)} nt, "
                  f"expected {record['length']} nt -- EXCLUDING")
            record["gc_content"] = None
            n_mismatch += 1
            continue

        record["gc_content"] = gc_fraction(seq)
        n_ok += 1

    print(f"\nOK: {n_ok}, missing: {n_missing}, length mismatches: {n_mismatch}, errors: {n_error}")

    gc_values = [r["gc_content"] for r in data if r.get("gc_content") is not None]
    if gc_values:
        import statistics
        print(f"GC content range: min={min(gc_values):.3f}, max={max(gc_values):.3f}, "
              f"mean={statistics.mean(gc_values):.3f}")
        print("Sanity check: typical RNA GC content is roughly 0.3-0.7; if values "
              "cluster at exactly 0 or 1, stop and investigate chain-id handling.")

    complete = [r for r in data if r.get("gc_content") is not None]
    if len(complete) < len(data):
        print(f"\n{len(data) - len(complete)} targets excluded. "
              f"complexity_sensitivity_model.py needs a complete record for every "
              f"target -- do not run it on data with missing values.")

    with open(args.output, "w") as f:
        json.dump(complete, f)
    print(f"\nSaved {len(complete)} complete records to {args.output}")


if __name__ == "__main__":
    main()
