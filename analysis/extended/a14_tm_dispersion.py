"""Size-normalised replicate of the primary comparison with dispersion in the same metric.

The outcome is the mean C4' TM-score (RNA d0) of the 20 primary samples against the
reference, and the dispersion is the mean pairwise TM-score among the same 20 samples
(all 190 pairs). Model 0 is the 4-df spline of log length, Model 1 adds log of the
mean pairwise TM-score. Writes a14_tmdisp.json (reused when present).
"""
import itertools
import os
import sys

sys.path.insert(0, ".")
from lib import *
from bund import get, gt
from metrics import tm_score
from scipy.stats import spearmanr

d = load()
if os.path.exists("a14_tmdisp.json"):
    m = pd.read_json("a14_tmdisp.json")
else:
    rows = []
    for t in d.target:
        X, G = get(t, "primary"), gt(t)
        ref = np.mean([tm_score(x, G) for x in X])
        pw = np.mean([tm_score(X[i], X[j]) for i, j in itertools.combinations(range(len(X)), 2)])
        rows.append(dict(target=t, tm_ref=ref, tm_pair=float(pw)))
    m = pd.DataFrame(rows)
    m.to_json("a14_tmdisp.json", orient="records")
d = d.merge(m, on="target")
print("Spearman, mean pairwise TM vs RMSD dispersion D: %.3f" % spearmanr(d.tm_pair, d.dispersion)[0])
print("Spearman, TM to reference vs mean pairwise TM:   %.3f" % spearmanr(d.tm_ref, d.tm_pair)[0])
y = np.log(d.tm_ref.values)
S = design(d, "M0")
for lab, z in (("log mean pairwise TM", np.log(d.tm_pair.values)), ("log (1 - mean pairwise TM)", np.log(1 - d.tm_pair.values))):
    X1 = np.column_stack([S, z])
    e0, e1 = loo_fixed(S, y), loo_fixed(X1, y)
    s = summ(e0, e1)
    ci, cc = boot_ci(e0 ** 2 - e1 ** 2), boot_ci(e0 ** 2 - e1 ** 2, d.cl.values)
    print(f"outcome log TM-score to reference, predictor {lab}: RMSE0 {s['rmse0']:.4f} RMSE1 {s['rmse1']:.4f} "
          f"rel {s['rel']:.2f}% dMSE {s['dMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}] cluster [{cc[0]:.4f},{cc[1]:.4f}] "
          f"coef {ols(X1, y)[-1]:.3f}")
