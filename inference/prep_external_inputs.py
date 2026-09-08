"""
Generates base-pair conditioning maps for external-cohort targets and
builds the target list file the inference pipeline consumes.

Usage (run from the inference package's root directory):
    python prep_external_inputs.py <external_cohort.json> <metadata.json> <k_samples>

A per-target timeout is applied to the base-pair map generation step:
secondary-structure prediction algorithms scale super-linearly with
sequence length, and a small number of long sequences (several hundred
nucleotides) can take disproportionately longer than the bulk of a
typical candidate pool. Targets that exceed the timeout are logged and
skipped rather than blocking the batch.
"""
import json
import subprocess
import sys
from pathlib import Path

BPMAPS_SCRIPT = "/opt/bpmaps/generate_bp_maps.py"
INPUTS_DIR = Path("Inputs")
PER_TARGET_TIMEOUT_SECONDS = 180


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
                    seqs[rcsb_id] = seq.replace("\n", "")
    return seqs


def main() -> None:
    cohort_path, json_path, k_samples = sys.argv[1], sys.argv[2], int(sys.argv[3])

    with open(cohort_path, encoding="utf-8") as f:
        cohort = json.load(f)

    sequences = get_sequences_by_id(json_path)

    list_lines = []
    n_done = n_skipped = n_failed = 0

    for entry in cohort:
        pdb_id = entry["pdb_id"]
        target_id = f"{pdb_id.lower()}_{entry['chain']}"
        seq = sequences.get(pdb_id)
        if not seq:
            print(f"  skip {target_id}: no sequence in metadata")
            n_failed += 1
            continue

        target_dir = INPUTS_DIR / target_id
        expected_maps = [target_dir / f"map{i}.npy" for i in (1, 2, 3)]
        if all(m.exists() for m in expected_maps):
            n_skipped += 1
            list_lines.append(f"{target_id} {k_samples}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["python", BPMAPS_SCRIPT, target_id, seq, str(target_dir)],
                check=True, capture_output=True, text=True,
                timeout=PER_TARGET_TIMEOUT_SECONDS,
            )
            list_lines.append(f"{target_id} {k_samples}")
            n_done += 1
            if n_done % 10 == 0:
                print(f"  progress: {n_done} done, {n_skipped} skipped, "
                      f"{n_failed} failed (most recent: {target_id})", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  timeout {target_id} (seq len {len(seq)}): "
                  f"exceeded {PER_TARGET_TIMEOUT_SECONDS}s, skipped")
            n_failed += 1
        except subprocess.CalledProcessError as exc:
            print(f"  failed {target_id}: {exc.stderr[-300:]}")
            n_failed += 1

    with open(INPUTS_DIR / "list.txt", "w") as f:
        f.write("\n".join(list_lines) + "\n")

    print(f"\nDone: {n_done}, skipped: {n_skipped}, failed: {n_failed}")
    print(f"Wrote Inputs/list.txt with {len(list_lines)} targets")


if __name__ == "__main__":
    main()
