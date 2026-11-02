"""
patch_interpolant_scoped_seed.py -- Modifies _centered_gaussian and
_uniform_so3 in src/data/interpolant_inf.py so that a fixed (non -1)
seedval only pins the initial generative noise/frame, without also
freezing everything downstream in the same process.

WHY THIS IS NEEDED: torch.manual_seed() and np.random.seed() set the
GLOBAL RNG state, not something scoped to the calling function. Without
this patch, giving Condition C (dropout ON, generative frame fixed) a
fixed seedval would ALSO freeze dropout's own mask draws during the
subsequent flow-matching integration loop, since dropout pulls from the
same global torch RNG that was just seeded -- silently defeating the
entire purpose of the condition (byte-identical output regardless of
dropout state, with no way to tell from the output alone).

THE FIX: immediately after consuming the fixed seed to draw the initial
noise/frame, reseed to fresh entropy (torch.seed() / np.random.seed(None)).
Only that one draw is pinned; everything downstream -- including
dropout -- draws from freshly-randomized state again.

This is purely additive: the existing seedval == -1 code path (used
unchanged by Conditions A and B) is untouched. Only the seedval != -1
branch changes, and that branch is otherwise unused elsewhere in this
project.

Run once, from backbones/rnabpflow/:
    python patch_interpolant_scoped_seed.py
"""
import re
from pathlib import Path

TARGET = Path("src/data/interpolant_inf.py")
content = TARGET.read_text()

old_gaussian = (
    "def _centered_gaussian(num_batch, num_res, device, seedval):\n"
    "    # returns randomly-sampled zero-centered Gaussian noise for translations\n"
    "    if seedval != -1:\n"
    "        torch.manual_seed(seedval)\n"
    "    noise = torch.randn(num_batch, num_res, 3, device=device)\n"
    "    return noise - torch.mean(noise, dim=-2, keepdims=True)"
)
new_gaussian = (
    "def _centered_gaussian(num_batch, num_res, device, seedval):\n"
    "    # returns randomly-sampled zero-centered Gaussian noise for translations\n"
    "    if seedval != -1:\n"
    "        torch.manual_seed(seedval)\n"
    "        noise = torch.randn(num_batch, num_res, 3, device=device)\n"
    "        torch.seed()  # PATCHED: reseed to fresh entropy immediately --\n"
    "        # only this noise draw is pinned by seedval, not anything\n"
    "        # downstream (e.g. dropout mask draws later in the same call)\n"
    "    else:\n"
    "        noise = torch.randn(num_batch, num_res, 3, device=device)\n"
    "    return noise - torch.mean(noise, dim=-2, keepdims=True)"
)

old_so3 = (
    "def _uniform_so3(num_batch, num_res, device, seedval):\n"
    "    # returns uniformly-sampled rotations (from U(SO(3)))\n"
    "    if seedval != -1:\n"
    "        np.random.seed(seedval)\n"
    "    return torch.tensor(\n"
    "        Rotation.random(num_batch*num_res).as_matrix(), # random rotation\n"
    "        device=device,\n"
    "        dtype=torch.float32,\n"
    "    ).reshape(num_batch, num_res, 3, 3)"
)
new_so3 = (
    "def _uniform_so3(num_batch, num_res, device, seedval):\n"
    "    # returns uniformly-sampled rotations (from U(SO(3)))\n"
    "    if seedval != -1:\n"
    "        np.random.seed(seedval)\n"
    "        rot_matrices = Rotation.random(num_batch*num_res).as_matrix()\n"
    "        np.random.seed(None)  # PATCHED: same rationale as _centered_gaussian\n"
    "    else:\n"
    "        rot_matrices = Rotation.random(num_batch*num_res).as_matrix()\n"
    "    return torch.tensor(\n"
    "        rot_matrices,\n"
    "        device=device,\n"
    "        dtype=torch.float32,\n"
    "    ).reshape(num_batch, num_res, 3, 3)"
)

n_gaussian = content.count(old_gaussian)
n_so3 = content.count(old_so3)
print(f"_centered_gaussian matches found: {n_gaussian}")
print(f"_uniform_so3 matches found: {n_so3}")

if n_gaussian != 1 or n_so3 != 1:
    print("ERROR: expected exactly one match for each function -- refusing "
          "to patch. Printing the file's current content around each "
          "function for manual inspection:")
    for marker in ("_centered_gaussian", "_uniform_so3"):
        idx = content.find(f"def {marker}")
        print(f"\n--- around 'def {marker}' ---")
        print(content[max(0, idx - 20):idx + 400])
    raise SystemExit(1)

content = content.replace(old_gaussian, new_gaussian)
content = content.replace(old_so3, new_so3)
TARGET.write_text(content)
print(f"\nPatched {TARGET} successfully.")
