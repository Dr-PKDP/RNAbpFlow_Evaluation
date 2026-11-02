"""
c3_get_cif.py -- downloads the mmCIF files of the extended-cohort entries (those without a PDB-format file)
and writes the results file the exporter needs.

Usage (from backbones/rnabpflow/):
    python c3_get_cif.py c3_facts.json ground_truth_cif results_c3.json
"""
import json, sys, time, urllib.request
from pathlib import Path

facts = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ok = fail = skip = 0
for pid in sorted({o["pdb_id"].upper() for o in facts}):
    dest = out / f"{pid}.cif"
    if dest.exists() and dest.stat().st_size > 1000: skip += 1; continue
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.cif", dest); ok += 1; break
        except Exception as exc:
            if attempt == 2: print(f"  FAILED {pid}: {exc}"); fail += 1
            else: time.sleep(2 ** attempt)
json.dump([{"target": o["target"]} for o in facts], open(sys.argv[3], "w"))
print(f"downloaded {ok}, already present {skip}, failed {fail}; wrote {sys.argv[3]} with {len(facts)} targets")
