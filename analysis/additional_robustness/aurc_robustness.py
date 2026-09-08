"""
AURC practical-triage comparison under median-risk and trimmed-mean-risk
(in addition to mean-risk), plus a coverage-threshold table
(25%/50%/75% retained).

Usage:
    python aurc_robustness.py <results.json>
"""
import json
import sys

import numpy as np


def aurc_for_ranking(scores, errors, risk_fn):
    """scores: per-target ranking criterion (lower = more trustworthy).
    risk_fn: function taking an array of retained errors and returning a
    scalar risk (mean, median, or trimmed mean)."""
    order = np.argsort(scores)
    sorted_errors = errors[order]
    n = len(errors)
    risks = []
    for k in range(1, n + 1):
        retained = sorted_errors[:k]
        risks.append(risk_fn(retained))
    # integrate via trapezoidal rule over coverage = k/n from 1/n to 1
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


def coverage_threshold_table(scores_dict, errors, thresholds=(0.25, 0.5, 0.75)):
    results = {}
    for name, scores in scores_dict.items():
        order = np.argsort(scores)
        sorted_errors = errors[order]
        n = len(errors)
        row = {}
        for t in thresholds:
            k = max(1, int(round(n * t)))
            row[t] = float(np.mean(sorted_errors[:k]))
        results[name] = row
    return results


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    scores = {
        "length_only": length,
        "dispersion_only": dispersion,
        # combined score: simple rank-average of length and dispersion,
        # matching the spirit of the cross-fitted combined policy without
        # requiring a full refit here (used only for the risk-fn
        # sensitivity, not to replace the primary cross-fitted result)
        "combined": (np.argsort(np.argsort(length)) + np.argsort(np.argsort(dispersion))) / 2.0,
    }

    print("=== AURC under different risk functions ===")
    for risk_name, risk_fn in [("mean", np.mean), ("median", np.median),
                                 ("trimmed_mean_10pct", lambda x: trimmed_mean(x, 0.1))]:
        print(f"\n--- Risk function: {risk_name} ---")
        for name, s in scores.items():
            aurc = aurc_for_ranking(s, error, risk_fn)
            print(f"  {name}: AURC = {aurc:.2f}")

    print("\n=== Coverage-threshold table (mean error among top-k retained) ===")
    table = coverage_threshold_table(scores, error)
    print(f"{'Policy':<18} {'25%':>8} {'50%':>8} {'75%':>8}")
    for name, row in table.items():
        print(f"{name:<18} {row[0.25]:>8.2f} {row[0.5]:>8.2f} {row[0.75]:>8.2f}")


if __name__ == "__main__":
    main()
