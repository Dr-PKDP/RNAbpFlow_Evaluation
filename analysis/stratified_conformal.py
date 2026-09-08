"""
Length-stratified (Mondrian) split-conformal calibration, alongside the
pooled marginal result. A pooled conformal procedure can achieve
approximately nominal overall coverage while performing badly for a
specific subgroup; this checks per-stratum coverage explicitly rather
than assuming the pooled number generalizes.

Uses 2 broad strata (median length split): more strata would risk
per-stratum calibration sets too small for a stable conformal quantile
at this sample size (N=82).

Usage:
    python stratified_conformal.py <results.json>
"""
import json
import sys

import numpy as np


def split_conformal_once(dispersions, errors, alpha, rng):
    n = len(dispersions)
    idx = rng.permutation(n)
    n_cal = n // 2
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
    cal_scores = errors[cal_idx] / dispersions[cal_idx]
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_hat = np.quantile(cal_scores, q_level)
    covered = errors[test_idx] <= q_hat * dispersions[test_idx]
    widths = q_hat * dispersions[test_idx]
    return covered.mean(), widths.mean(), n_cal, len(test_idx)


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])
    length = np.array([d["length"] for d in data])
    n = len(dispersion)
    alpha = 0.1
    n_splits = 50
    rng = np.random.default_rng(0)

    print(f"N = {n}\n")

    print("=== Pooled (marginal) conformal, for reference ===")
    pooled = [split_conformal_once(dispersion, error, alpha, rng) for _ in range(n_splits)]
    pooled_cov = np.mean([p[0] for p in pooled])
    pooled_width = np.mean([p[1] for p in pooled])
    print(f"  Mean coverage: {pooled_cov:.3f} (nominal: {1-alpha:.2f})")
    print(f"  Mean interval width: {pooled_width:.3f}\n")

    print("=== Length-stratified (2 strata, median split) ===")
    median_length = np.median(length)
    strata = {
        "short (<= median)": length <= median_length,
        "long (> median)": length > median_length,
    }

    for name, mask in strata.items():
        d_s, e_s = dispersion[mask], error[mask]
        n_stratum = len(d_s)
        n_cal_typical = n_stratum // 2
        print(f"  {name}: n={n_stratum} (calibration set ~{n_cal_typical} per split)")
        if n_cal_typical < 10:
            print(f"    WARNING: calibration set size {n_cal_typical} is small; "
                  f"conformal quantile may be unstable at this stratum size.")
        results = [split_conformal_once(d_s, e_s, alpha, rng) for _ in range(n_splits)]
        cov = np.mean([r[0] for r in results])
        width = np.mean([r[1] for r in results])
        print(f"    Mean coverage: {cov:.3f} (nominal: {1-alpha:.2f})")
        print(f"    Mean interval width: {width:.3f}")

    print(f"\nInterpretation: if per-stratum coverage differs substantially from the "
          f"pooled {pooled_cov:.3f} (e.g. one stratum well below 90%), the pooled "
          f"result masks non-uniform reliability across the length range.")


if __name__ == "__main__":
    main()
