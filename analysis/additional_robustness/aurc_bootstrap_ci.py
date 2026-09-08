"""
Bootstrap confidence intervals for the AURC differences (length-only
vs. dispersion-only, and length-only vs. simplified combined) under
median and trimmed-mean risk.

Usage:
    python aurc_bootstrap_ci.py <results.json>
"""
import json
import sys

import numpy as np


def aurc_for_ranking(scores, errors, risk_fn):
    order = np.argsort(scores)
    sorted_errors = errors[order]
    n = len(errors)
    risks = [risk_fn(sorted_errors[:k]) for k in range(1, n + 1)]
    coverage = np.arange(1, n + 1) / n
    if hasattr(np, "trapezoid"):
        return np.trapezoid(risks, coverage)
    return np.trapz(risks, coverage)


def trimmed_mean(x, frac=0.1):
    x = np.sort(x)
    n = len(x)
    k = int(np.floor(n * frac))
    if k == 0:
        return x.mean()
    return x[k:n - k].mean()


def bootstrap_aurc_diff(length, dispersion, error, risk_fn, n_boot=2000, seed=0):
    n = len(length)
    rng = np.random.default_rng(seed)
    diffs_disp = []
    diffs_comb = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l_b, d_b, e_b = length[idx], dispersion[idx], error[idx]
        combined_b = (np.argsort(np.argsort(l_b)) + np.argsort(np.argsort(d_b))) / 2.0
        aurc_l = aurc_for_ranking(l_b, e_b, risk_fn)
        aurc_d = aurc_for_ranking(d_b, e_b, risk_fn)
        aurc_c = aurc_for_ranking(combined_b, e_b, risk_fn)
        diffs_disp.append(aurc_l - aurc_d)
        diffs_comb.append(aurc_l - aurc_c)
    diffs_disp = np.array(diffs_disp)
    diffs_comb = np.array(diffs_comb)
    return {
        "length_minus_dispersion": {
            "mean": float(diffs_disp.mean()),
            "ci_95": [float(x) for x in np.percentile(diffs_disp, [2.5, 97.5])],
        },
        "length_minus_combined": {
            "mean": float(diffs_comb.mean()),
            "ci_95": [float(x) for x in np.percentile(diffs_comb, [2.5, 97.5])],
        },
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    for risk_name, risk_fn in [("median", np.median),
                                 ("trimmed_mean_10pct", lambda x: trimmed_mean(x, 0.1))]:
        print(f"=== Risk function: {risk_name} ===")
        result = bootstrap_aurc_diff(length, dispersion, error, risk_fn, n_boot=2000, seed=0)
        for comparison, r in result.items():
            print(f"  {comparison}: mean={r['mean']:.3f}, 95% CI={r['ci_95']}")
        print()
