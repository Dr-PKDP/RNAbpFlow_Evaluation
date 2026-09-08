"""
Builds a list.txt-format file containing the 82 cohort targets (from
dispersion_error_results_deduped.json), each with "10" as the requested
sample count.

Usage:
    python generate_extra10_list.py dispersion_error_results_deduped.json extra10_list.txt
"""
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

targets = sorted(d["target"] for d in data)
print(f"{len(targets)} targets found in {sys.argv[1]}")

with open(sys.argv[2], "w") as f:
    for t in targets:
        f.write(f"{t} 10\n")

print(f"Wrote {sys.argv[2]} with {len(targets)} lines, each requesting 10 samples.")
