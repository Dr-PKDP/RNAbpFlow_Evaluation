"""
Downloads reference structures for external-cohort targets, for
comparing predicted structures against real deposited coordinates.

Note: structures too large for the legacy PDB format (requiring
multi-character author chain IDs) are not available via this endpoint
and will fail; such structures are available only in mmCIF format,
which this script does not currently parse.

Usage:
    python download_ground_truth.py <external_cohort.json> <output_dir>
"""
import json
import sys
import time
import urllib.request
from pathlib import Path


def download_pdb(pdb_id, dest, max_retries=3):
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    last_exc = None
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc


def main() -> None:
    cohort_path, out_dir = sys.argv[1], sys.argv[2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cohort_path, encoding="utf-8") as f:
        cohort = json.load(f)

    n_done = n_skipped = n_failed = 0
    failed_ids = []
    for entry in cohort:
        pdb_id = entry["pdb_id"]
        dest = out_dir / f"{pdb_id.upper()}.pdb"
        if dest.exists():
            n_skipped += 1
            continue
        try:
            download_pdb(pdb_id, dest)
            n_done += 1
        except Exception as exc:
            n_failed += 1
            failed_ids.append(pdb_id)
            print(f"  failed {pdb_id}: {exc}")
        time.sleep(0.2)

    print(f"\nDownloaded: {n_done}, already existed: {n_skipped}, failed: {n_failed}")
    if failed_ids:
        with open(out_dir / "download_failures.json", "w") as f:
            json.dump(failed_ids, f)


if __name__ == "__main__":
    main()
