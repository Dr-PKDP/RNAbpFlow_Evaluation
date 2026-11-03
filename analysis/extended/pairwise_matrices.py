"""Pooled-sample RMSD matrices read by a8_a9.py and a10.py.

For every target the 50 samples of the primary, replicate and extra runs are
concatenated (20 + 20 + 10) and the Kabsch C4' RMSD is computed between all
pairs (key <target>__M) and against the reference (key <target>__E).
Writes pairwise.npz in the current directory.
"""
import numpy as np

from bund import RUNS, get, gt
from lib import load

out = {}
for t in load().target:
    X = np.concatenate([get(t, run) for run in RUNS], axis=0)
    from bund import rmsd_pairs
    out[f"{t}__M"] = rmsd_pairs(X, X)
    out[f"{t}__E"] = rmsd_pairs(X, gt(t)[None])[:, 0]
np.savez_compressed("pairwise.npz", **out)
print("targets:", len(out) // 2)
