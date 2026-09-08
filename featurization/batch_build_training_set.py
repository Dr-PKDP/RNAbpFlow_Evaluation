"""
Batch-builds training feature files across the full candidate training
pool: clean (no modified nucleotides), chain-resolved, released before
the freeze date, non-NMR. Writes both the per-target `.pkl` files and a
metadata CSV in the format the training data loader expects.

Resumable: skips any entry that already has an output `.pkl`. Individual
failures (missing sequence data, non-standard residues, network errors)
are logged and skipped rather than aborting the batch.

Usage:
    python batch_build_training_set.py <metadata.json> <clean_candidate_ids.json> \\
        <auth_chain_map.json> <freeze_date YYYY-MM-DD> <output_pkl_dir> <output_csv>
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, ".")
from build_training_pkl import download_pdb, parse_geometry, run_fr3d, \
    build_ss_onehot, build_identity_fields  # noqa: E402


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def is_nmr(entry: dict) -> bool:
    """NMR structures are excluded: they represent an ensemble of
    conformers consistent with experimental restraints rather than a
    single privileged ground-truth structure, making them unsuitable as
    training supervision without an arbitrary conformer-selection
    choice."""
    exptl = entry.get("exptl", []) or []
    return any("NMR" in (x.get("method") or "") for x in exptl)


def main() -> None:
    json_path, clean_ids_path, chain_map_path, freeze_str, out_dir, out_csv = sys.argv[1:7]
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
        if rel is None or rel > freeze:
            continue
        targets.append((rcsb_id, chain_map[rcsb_id]))

    print(f"NMR structures excluded: {n_nmr_excluded}")
    print(f"Training-pool targets to build: {len(targets)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("batch_build_errors.log")

    csv_rows = []
    n_done = n_skipped = n_failed = 0

    for i, (pdb_id, chain) in enumerate(targets):
        pdb_name = f"{pdb_id.lower()}_{chain}"
        pkl_path = out_dir / pdb_name / f"{pdb_name}.pkl"

        if pkl_path.exists():
            n_skipped += 1
            with open(pkl_path, "rb") as f:
                n_res = pickle.load(f)["aatype"].shape[0]
            csv_rows.append((pdb_name, str(pkl_path), n_res))
            continue

        try:
            pdb_dir = Path("/tmp/batch_pdb")
            pdb_dir.mkdir(parents=True, exist_ok=True)
            pdb_file = download_pdb(pdb_id, pdb_dir / f"{pdb_id}.pdb")
            geom_feats, residues, sequence = parse_geometry(pdb_file, chain)
            first_resnum = residues[0].id[1]
            bp_file = run_fr3d(pdb_id)
            ss_feats = build_ss_onehot(bp_file, sequence, first_resnum)
            id_feats = build_identity_fields(len(residues), chain)
            full = {"pdb_name": pdb_name, **geom_feats, **ss_feats, **id_feats}

            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl_path, "wb") as f:
                pickle.dump(full, f)

            csv_rows.append((pdb_name, str(pkl_path), len(residues)))
            n_done += 1
            if n_done % 10 == 0:
                print(f"  progress: {n_done} done, {n_skipped} skipped, "
                      f"{n_failed} failed (most recent: {pdb_name})", flush=True)
        except Exception:
            n_failed += 1
            with open(log_path, "a") as f:
                f.write(f"\n=== {pdb_id} chain {chain} ===\n")
                f.write(traceback.format_exc())
            print(f"  [{i+1}/{len(targets)}] failed: {pdb_id}_{chain}")

    # Column set required by the training data loader's metadata CSV:
    # modeled_na_seq_len drives length-based batching; quaternary_category
    # and num_protein_chains are used as monomer/RNA-only filters and are
    # fixed true values for this single-chain RNA candidate pool;
    # helix/coil/strand_percent and radius_gyration are read but not used
    # in any downstream filtering or loss computation.
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_name", "processed_path", "modeled_na_seq_len",
                          "quaternary_category", "num_protein_chains",
                          "helix_percent", "coil_percent", "strand_percent",
                          "radius_gyration"])
        for pdb_name, path, n_res in csv_rows:
            writer.writerow([pdb_name, path, n_res, "homomer", 0, 0, 0, 0, 0])

    print(f"\nBuilt: {n_done}, already existed: {n_skipped}, failed: {n_failed}")
    print(f"Wrote {out_csv} with {len(csv_rows)} entries")


if __name__ == "__main__":
    main()
