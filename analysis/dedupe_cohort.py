"""
Deduplicates an evaluation cohort by exact RNA sequence before
statistical analysis. Public structure repositories can contain
substantial redundancy from batch depositions of closely related
structures (e.g. the same construct solved under multiple conditions);
treating such near-duplicate structures as independent observations
inflates effective sample size and biases downstream statistics.

Groups results by exact sequence and keeps one representative per
unique sequence.

Usage:
    python dedupe_cohort.py <results.json> <metadata.json>

results.json: list of {"target": str, "dispersion": float, "error": float, ...}
metadata.json: RCSB-format metadata with an `entries` list, used to look
up each target's sequence.
"""
import json
import sys
from collections import defaultdict


def main() -> None:
    results_path, metadata_path = sys.argv[1], sys.argv[2]

    with open(results_path) as f:
        results = json.load(f)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    seq_by_pdbid = {}
    for e in metadata["entries"]:
        rcsb_id = e.get("rcsb_id")
        for pe in e.get("polymer_entities", []) or []:
            ep = pe.get("entity_poly", {}) or {}
            if ep.get("rcsb_entity_polymer_type") == "RNA":
                seq = ep.get("pdbx_seq_one_letter_code_can")
                if seq:
                    seq_by_pdbid[rcsb_id] = seq.replace("\n", "")

    clusters = defaultdict(list)
    for r in results:
        pdb_id = r["target"].split("_")[0].upper()
        seq = seq_by_pdbid.get(pdb_id)
        if seq is None:
            print(f"  no sequence found for {r['target']}, treating as unique")
            seq = r["target"]
        clusters[seq].append(r)

    print(f"Total targets: {len(results)}")
    print(f"Unique sequence clusters: {len(clusters)}")
    cluster_sizes = sorted([len(v) for v in clusters.values()], reverse=True)
    print(f"Largest cluster sizes: {cluster_sizes[:10]}")

    deduped = [sorted(members, key=lambda r: r["target"])[0]
               for members in clusters.values()]

    with open("dispersion_error_results_deduped.json", "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"\nWrote dispersion_error_results_deduped.json ({len(deduped)} independent targets)")


if __name__ == "__main__":
    main()
