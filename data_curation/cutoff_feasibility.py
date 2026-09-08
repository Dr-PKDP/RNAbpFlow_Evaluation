"""
Evaluates candidate PDB snapshot freeze dates against training-pool and
external-cohort size tradeoffs.

Reads a metadata file (JSON with an `entries` list, each containing
`rcsb_accession_info.initial_release_date` and `deposit_date`) and
reports, for a grid of candidate cutoff dates, how many entries fall on
each side of the split under both date fields.

`initial_release_date` (the date a structure becomes publicly available)
is used as the operative freeze boundary rather than `deposit_date` (the
date it was submitted), since a structure held under an author-requested
release delay is not usable as training data until it is actually public,
regardless of when it was deposited.

Usage:
    python cutoff_feasibility.py <path-to-metadata.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date, datetime


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def load_entries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data["entries"]
    print(f"Loaded {len(entries)} entries from {path} "
          f"(snapshot_date={data.get('snapshot_date', 'unknown')})")
    return entries


def extract_dates(entries: list[dict]) -> list[tuple[str, date | None, date | None]]:
    out = []
    for e in entries:
        acc = e.get("rcsb_accession_info", {}) or {}
        rel = _parse_date(acc.get("initial_release_date"))
        dep = _parse_date(acc.get("deposit_date"))
        out.append((e.get("rcsb_id", "UNKNOWN"), rel, dep))
    return out


def report_missing(rows: list[tuple[str, date | None, date | None]]) -> None:
    missing_rel = sum(1 for _, rel, _ in rows if rel is None)
    missing_dep = sum(1 for _, _, dep in rows if dep is None)
    if missing_rel or missing_dep:
        print(f"\n{missing_rel} entries missing initial_release_date, "
              f"{missing_dep} missing deposit_date. Excluded from the "
              f"histograms and cutoff table below.")


def year_histogram(rows: list[tuple[str, date | None, date | None]]) -> None:
    rel_years = Counter(rel.year for _, rel, _ in rows if rel is not None)
    dep_years = Counter(dep.year for _, _, dep in rows if dep is not None)
    all_years = sorted(set(rel_years) | set(dep_years))
    print(f"\n{'Year':<6}{'release_date n':<18}{'deposit_date n':<18}")
    for y in all_years:
        print(f"{y:<6}{rel_years.get(y, 0):<18}{dep_years.get(y, 0):<18}")


def cutoff_table(rows: list[tuple[str, date | None, date | None]],
                  candidate_cutoffs: list[str]) -> None:
    print(f"\n{'Cutoff':<12}{'train(rel<=)':<14}{'external(rel>)':<16}"
          f"{'train(dep<=)':<14}{'external(dep>)':<16}")
    for c in candidate_cutoffs:
        cutoff = date.fromisoformat(c)
        train_rel = sum(1 for _, rel, _ in rows if rel is not None and rel <= cutoff)
        ext_rel = sum(1 for _, rel, _ in rows if rel is not None and rel > cutoff)
        train_dep = sum(1 for _, _, dep in rows if dep is not None and dep <= cutoff)
        ext_dep = sum(1 for _, _, dep in rows if dep is not None and dep > cutoff)
        print(f"{c:<12}{train_rel:<14}{ext_rel:<16}{train_dep:<14}{ext_dep:<16}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python cutoff_feasibility.py <path-to-metadata.json>")
        sys.exit(1)
    entries = load_entries(sys.argv[1])
    rows = extract_dates(entries)
    report_missing(rows)
    year_histogram(rows)
    candidate_cutoffs = [
        "2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01",
        "2025-01-01", "2025-07-01", "2026-01-01", "2026-06-01",
    ]
    cutoff_table(rows, candidate_cutoffs)


if __name__ == "__main__":
    main()
