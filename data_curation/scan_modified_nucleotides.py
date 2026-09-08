"""
Flags candidates containing modified nucleotides by comparing RCSB's raw
and canonical sequence fields. A difference between the two indicates at
least one modified residue, since RCSB remaps most modified residues to
their standard parent base in the canonical field only.

Note: inosine is not remapped in RCSB's canonical field and is therefore
not detected by this check. It is instead caught downstream by an
explicit residue-name check during featurization.

Usage:
    python scan_modified_nucleotides.py <path-to-metadata.json>

Writes clean_candidate_ids.json and modified_nucleotide_ids.json.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

GRAPHQL_URL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 50


def fetch_raw_sequences(pdb_ids: list[str]) -> dict[str, list[str]]:
    query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        polymer_entities {
          entity_poly {
            pdbx_seq_one_letter_code
            rcsb_entity_polymer_type
          }
        }
      }
    }
    """
    body = json.dumps({"query": query, "variables": {"ids": pdb_ids}}).encode()
    req = urllib.request.Request(GRAPHQL_URL, data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    out = {}
    for entry in data.get("data", {}).get("entries", []) or []:
        rcsb_id = entry["rcsb_id"]
        seqs = []
        for pe in entry.get("polymer_entities", []) or []:
            ep = pe.get("entity_poly", {}) or {}
            if ep.get("rcsb_entity_polymer_type") == "RNA":
                seq = ep.get("pdbx_seq_one_letter_code")
                if seq:
                    seqs.append(seq.replace("\n", ""))
        out[rcsb_id] = seqs
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    entries = data["entries"]
    print(f"Loaded {len(entries)} candidates from {sys.argv[1]}")

    canonical_by_id = {}
    for e in entries:
        rcsb_id = e.get("rcsb_id")
        seqs = []
        for pe in e.get("polymer_entities", []) or []:
            ep = pe.get("entity_poly", {}) or {}
            if ep.get("rcsb_entity_polymer_type") == "RNA":
                seq = ep.get("pdbx_seq_one_letter_code_can")
                if seq:
                    seqs.append(seq.replace("\n", ""))
        canonical_by_id[rcsb_id] = seqs

    all_ids = list(canonical_by_id.keys())
    modified_ids = []
    clean_ids = []
    errors = []

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i:i + BATCH_SIZE]
        try:
            raw_by_id = fetch_raw_sequences(batch)
        except Exception as exc:
            errors.extend(batch)
            print(f"  batch {i}-{i+len(batch)}: request failed ({exc})")
            continue

        for pdb_id in batch:
            raw_seqs = raw_by_id.get(pdb_id, [])
            can_seqs = canonical_by_id.get(pdb_id, [])
            if raw_seqs != can_seqs:
                modified_ids.append(pdb_id)
            else:
                clean_ids.append(pdb_id)

        print(f"  processed {min(i+BATCH_SIZE, len(all_ids))}/{len(all_ids)}")
        time.sleep(0.2)

    print(f"\nClean: {len(clean_ids)}")
    print(f"Contains modified nucleotides: {len(modified_ids)}")
    print(f"Errors: {len(errors)}")

    with open("clean_candidate_ids.json", "w") as f:
        json.dump(clean_ids, f, indent=2)
    with open("modified_nucleotide_ids.json", "w") as f:
        json.dump(modified_ids, f, indent=2)
    if errors:
        with open("scan_errors.json", "w") as f:
            json.dump(errors, f, indent=2)


if __name__ == "__main__":
    main()
