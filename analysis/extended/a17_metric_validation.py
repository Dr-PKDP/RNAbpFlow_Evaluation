"""Validation of the in-house TM-score and lDDT against external reference implementations.
Runs from the directory holding lib.py, bund.py and metrics.py.

TM-score is checked against US-align (https://github.com/pylelab/USalign), which must be
built from source and available as `USalign` on PATH; invoked with the flags
`-mol RNA -atom " C4'" -TMscore 1` so that it scores the same C4'-only, fixed-residue-
correspondence comparison as the in-house tm_score().

lDDT is checked against a direct numpy re-implementation of DeepMind's AlphaFold reference
formula (alphafold/model/lddt.py, Apache 2.0), with the same 15 A inclusion radius, the same
four distance-difference thresholds and per-residue normalization, which is a pure function
of numpy arrays and needs no external tool.
"""
import json
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, ".")
from lib import *
from bund import get, gt
from metrics import tm_score, lddt


def write_pdb(path, coords, atom=" C4'", resname="  A"):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords, 1):
            f.write("ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                    % (i, atom, resname, i, x, y, z))
        f.write("TER\nEND\n")


def usalign_tm(pred, ref, binary="USalign"):
    with tempfile.TemporaryDirectory() as td:
        pp, rp = f"{td}/pred.pdb", f"{td}/ref.pdb"
        write_pdb(pp, pred)
        write_pdb(rp, ref)
        out = subprocess.run([binary, pp, rp, "-mol", "RNA", "-atom", " C4'", "-TMscore", "1"],
                              capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("TM-score="):
            return float(line.split("=")[1].split()[0])
    raise RuntimeError("no TM-score line in US-align output:\n" + out)


def lddt_alphafold_port(pred, true, cutoff=15.0, per_residue=True):
    """Direct numpy port of alphafold/model/lddt.py; booleans cast to float before summing
    (summing np.bool_ arrays with '+' is elementwise OR, not a count -- cast each term first)."""
    dmat_true = np.sqrt(1e-10 + ((true[:, None] - true[None]) ** 2).sum(-1))
    dmat_pred = np.sqrt(1e-10 + ((pred[:, None] - pred[None]) ** 2).sum(-1))
    mask = (dmat_true < cutoff) * (1.0 - np.eye(len(true)))
    dist_l1 = np.abs(dmat_true - dmat_pred)
    score = 0.25 * ((dist_l1 < 0.5).astype(float) + (dist_l1 < 1.0).astype(float)
                     + (dist_l1 < 2.0).astype(float) + (dist_l1 < 4.0).astype(float))
    if per_residue:
        denom = 1e-10 + mask.sum(1)
        return (mask * score).sum(1) / denom
    denom = 1e-10 + mask.sum()
    return (mask * score).sum() / denom


if __name__ == "__main__":
    d = load()
    T = d.target.tolist()
    have_usalign = shutil.which("USalign") is not None
    rows = []
    for t in T:
        X = get(t, "primary")
        G = gt(t)
        mine_lddt = [lddt(x, G) for x in X]
        af_lddt = [float(np.mean(lddt_alphafold_port(x, G, per_residue=True))) for x in X]
        row = dict(target=t, mine_lddt=float(np.mean(mine_lddt)), af_lddt=float(np.mean(af_lddt)))
        if have_usalign:
            mine_tm = [tm_score(x, G) for x in X]
            us_tm = [usalign_tm(x, G) for x in X]
            row["mine_tm"] = float(np.mean(mine_tm))
            row["us_tm"] = float(np.mean(us_tm))
        rows.append(row)
    m = pd.DataFrame(rows)
    json.dump(rows, open("metric_validation.json", "w"))

    print("lDDT vs AlphaFold-reference-formula port, per-residue normalization, N=%d" % len(m))
    print("  Pearson r=%.4f  mean|diff|=%.4f  max|diff|=%.4f" % (
        np.corrcoef(m.mine_lddt, m.af_lddt)[0, 1],
        np.abs(m.mine_lddt - m.af_lddt).mean(), np.abs(m.mine_lddt - m.af_lddt).max()))

    if have_usalign:
        print("TM-score vs US-align (-mol RNA -atom \" C4'\" -TMscore 1), N=%d" % len(m))
        print("  Pearson r=%.4f  Spearman rho=%.4f  mean|diff|=%.4f  max|diff|=%.4f" % (
            np.corrcoef(m.mine_tm, m.us_tm)[0, 1],
            m[["mine_tm", "us_tm"]].corr("spearman").iloc[0, 1],
            np.abs(m.mine_tm - m.us_tm).mean(), np.abs(m.mine_tm - m.us_tm).max()))
    else:
        print("USalign not found on PATH; skipped the TM-score comparison "
              "(build from https://github.com/pylelab/USalign and place the binary on PATH to run it)")
