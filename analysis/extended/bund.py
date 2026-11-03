"""Loader for the data bundle written by analysis/export_bundle.py.

The bundle directory (coords.npz, meta.json, bp.npz) is looked up under
$RNA_DATA/bundle, directly or one level down.
"""
import glob
import json
import os

import numpy as np

_ROOT = os.path.join(os.environ.get("RNA_DATA", "data").rstrip("/"), "bundle")
_hits = glob.glob(os.path.join(_ROOT, "coords.npz")) or glob.glob(os.path.join(_ROOT, "*", "coords.npz"))
if not _hits:
    raise FileNotFoundError("coords.npz not found under " + _ROOT + "; run analysis/export_bundle.py first")
BD = os.path.dirname(_hits[0]) + os.sep
META = json.load(open(BD + "meta.json"))
Z = np.load(BD + "coords.npz")
RUNS = ("primary", "dropout_off", "extra10")


def get(t, run, paired=True):
    """Predicted C4' coordinates (n_samples, L, 3); paired=True keeps the residues matched to the reference."""
    X = Z[f"{t}__{run}"].astype(np.float64)
    return X[:, Z[f"{t}__pred_idx"], :] if paired else X


def gt(t):
    """Reference C4' coordinates (L, 3) for the paired residues."""
    return Z[f"{t}__gt"].astype(np.float64)


def rmsd_pairs(A, B):
    """A (m, L, 3), B (n, L, 3) -> (m, n) Kabsch RMSD."""
    m, n, L = len(A), len(B), A.shape[1]
    Ac = A - A.mean(1, keepdims=True)
    Bc = B - B.mean(1, keepdims=True)
    H = np.einsum('mli,nlj->mnij', Ac, Bc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    S2 = S.copy()
    S2[..., 2] *= d
    ga = (Ac ** 2).sum((1, 2))[:, None]
    gb = (Bc ** 2).sum((1, 2))[None, :]
    val = ga + gb - 2 * S2.sum(-1)
    return np.sqrt(np.maximum(val, 0) / L)
