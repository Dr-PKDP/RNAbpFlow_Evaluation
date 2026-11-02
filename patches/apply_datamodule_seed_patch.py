"""
Applies patch 1 described in the top-level README: adds a configurable seed to the training
data loader's length-batcher construction.

Run once, from the training package's root directory:
    python apply_datamodule_seed_patch.py

Uses tolerant whitespace matching rather than an exact string match,
since indentation can vary between code releases. Prints the matched
text before modifying anything, and refuses to proceed if the match is
not unique.
"""
import re
from pathlib import Path

TARGET = Path("src/data/pdb_na_datamodule_base.py")
content = TARGET.read_text()

pattern = re.compile(
    r"(lb\s*=\s*RNALengthBatcher\(\s*"
    r"sampler_cfg\s*=\s*self\.data_cfg\s*,\s*"
    r"metadata_csv\s*=\s*self\.data_train\.csv\s*,\s*"
    r"rank\s*=\s*rank\s*,\s*"
    r"num_replicas\s*=\s*num_replicas\s*)\)",
    re.DOTALL,
)

matches = pattern.findall(content)
print(f"Matches found: {len(matches)}")
if len(matches) != 1:
    print("Expected exactly one match; refusing to patch.")
    raise SystemExit(1)

print("Matched text:")
print(matches[0] + ")")

new_call = matches[0] + ",\n                seed=run_seed,\n            )"
content_patched = pattern.sub(lambda m: new_call, content, count=1)

seed_line = (
    "        run_seed = getattr(self.data_cfg, \"seed\", 123)\n"
)
content_patched = content_patched.replace(
    "        lb = RNALengthBatcher(", seed_line + "        lb = RNALengthBatcher(", 1
)

TARGET.write_text(content_patched)
print(f"Patched {TARGET}.")
