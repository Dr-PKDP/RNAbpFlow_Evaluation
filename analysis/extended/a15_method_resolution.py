"""Structure-determination method and reported resolution as covariates and strata.

Reads the method and resolution columns of manifest/manifest_82_targets.csv.
"""
import os
import sys

sys.path.insert(0, ".")
from lib import *
from scipy.stats import spearmanr

d = load()
man = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "manifest", "manifest_82_targets.csv"))
d = d.merge(man[["target", "method", "resolution_A"]], on="target")
d["em"] = (d.method == "EM").astype(float)
d["logRes"] = np.log(d.resolution_A)
y = d.logE.values
print(d.groupby("method").agg(n=("target", "size"), median_length=("length", "median"), median_E=("error", "median"),
                              median_D=("dispersion", "median"), median_resolution=("resolution_A", "median")).round(2))


def run(label, Z0, Z1):
    e0, e1 = loo_fixed(Z0, y), loo_fixed(Z1, y)
    s = summ(e0, e1)
    ci, cc = boot_ci(e0 ** 2 - e1 ** 2), boot_ci(e0 ** 2 - e1 ** 2, d.cl.values)
    print(f"{label:34s} RMSE0 {s['rmse0']:.4f} RMSE1 {s['rmse1']:.4f} rel {s['rel']:6.2f}% dMSE {s['dMSE']:.4f} "
          f"[{ci[0]:.4f},{ci[1]:.4f}] cluster [{cc[0]:.4f},{cc[1]:.4f}]")


S, D = design(d, "M0"), d.logD.values[:, None]
print("Model 0 RMSE %.4f; + method %.4f; + log resolution %.4f" % (
    np.sqrt(np.mean(loo_fixed(S, y) ** 2)), np.sqrt(np.mean(loo_fixed(np.column_stack([S, d.em]), y) ** 2)),
    np.sqrt(np.mean(loo_fixed(np.column_stack([S, d.logRes]), y) ** 2))))
run("method in both models", np.column_stack([S, d.em]), np.column_stack([S, d.em, D]))
run("log resolution in both models", np.column_stack([S, d.logRes]), np.column_stack([S, d.logRes, D]))
run("method and log resolution in both", np.column_stack([S, d.em, d.logRes]), np.column_stack([S, d.em, d.logRes, D]))
e0 = loo_fixed(S, y)
print("Spearman log resolution vs Model 0 residual %.3f; vs log length %.3f; vs log E %.3f" % (
    spearmanr(d.logRes, e0)[0], spearmanr(d.logRes, d.logL)[0], spearmanr(d.logRes, d.logE)[0]))
pg = e0 ** 2 - loo_fixed(np.column_stack([S, D]), y) ** 2
for k, g in d.groupby("method"):
    print(k, "summed pointwise gain in the primary fit %.3f" % pg[g.index].sum())
    g = g.reset_index(drop=True)
    for df in (3, 4):
        a, b = cv_refit(g, "M0", df), cv_refit(g, "M1", df)
        s, ci = summ(a, b), boot_ci(a ** 2 - b ** 2)
        print(f"  {k} n={len(g)} spline refitted in stratum, df {df}: RMSE0 {s['rmse0']:.3f} RMSE1 {s['rmse1']:.3f} "
              f"rel {s['rel']:.2f}% dMSE {s['dMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]")
