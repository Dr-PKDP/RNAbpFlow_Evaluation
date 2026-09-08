"""
Constructs the held-out external evaluation cohort: candidates released
after the training-set freeze date, with modified nucleotides excluded,
chains resolved, and NMR structures excluded.

NMR structures are excluded from both the training pool and the external
cohort. NMR yields an ensemble of conformers consistent with experimental
restraints rather than a single privileged ground-truth structure, so
selecting one representative model as supervision or evaluation target
introduces an arbitrary choice that this pipeline avoids by exclusion
rather than convention.

Usage:
    python build_external_cohort.py <metadata.json> <clean_candidate_ids.json> \
        <auth_chain_map.json> <freeze_date YYYY-MM-DD> <output.json>
"""
import json
import sys
from datetime import date, datetime


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def is_nmr(entry):
    exptl = entry.get("exptl", []) or []
    return any("NMR" in (x.get("method") or "") for x in exptl)


def main() -> None:
    json_path, clean_ids_path, chain_map_path, freeze_str, out_path = sys.argv[1:6]
    freeze = date.fromisoformat(freeze_str)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(clean_ids_path, encoding="utf-8") as f:
        clean_ids = set(json.load(f))
    with open(chain_map_path, encoding="utf-8") as f:
        chain_map = json.load(f)

    targets = []
    n_nmr_excluded = 0
    for e in data["entries"]:
        rcsb_id = e.get("rcsb_id")
        if rcsb_id not in clean_ids or rcsb_id not in chain_map:
            continue
        if is_nmr(e):
            n_nmr_excluded += 1
            continue
        rel = parse_date((e.get("rcsb_accession_info") or {}).get("initial_release_date"))
        if rel is None or rel <= freeze:
            continue
        targets.append({"pdb_id": rcsb_id, "chain": chain_map[rcsb_id]})

    print(f"NMR structures excluded: {n_nmr_excluded}")
    print(f"External cohort targets: {len(targets)}")

    with open(out_path, "w") as f:
        json.dump(targets, f, indent=2)


if __name__ == "__main__":
    main()
