"""
Applies patch 2 from README.md: disables deterministic per-sample
seeding in the inference dataset, required for genuinely independent
repeated samples (e.g. Monte Carlo dropout).

Run once, from the inference package's root directory:
    python apply_dataset_seed_patch.py
"""
import re
from pathlib import Path

TARGET = Path("src/data/dataset.py")
content = TARGET.read_text()

pattern = re.compile(r"seedval\s*=\s*start_seed\s*\*\s*\(sample_id\s*\+\s*1\)\s*\*\s*10")
matches = pattern.findall(content)
print(f"Matches found: {len(matches)}")
if len(matches) != 1:
    print("Expected exactly one match; refusing to patch.")
    raise SystemExit(1)

print(f"Matched: {matches[0]!r}")
new_content = pattern.sub("seedval = -1", content)
TARGET.write_text(new_content)
print(f"Patched {TARGET}.")
