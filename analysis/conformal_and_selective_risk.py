"""
Split-conformal calibration and selective-risk evaluation, using
per-target ensemble dispersion as a nonconformity score for structural
error.

Usage:
    python conformal_and_selective_risk.py <dispersion_error_results.json>

Input format: a list of {"target": str, "dispersion": float, "error": float}.
"""
import json
import sys

import numpy as np


def split_conformal(dispersions, errors, alpha=0.1, n_splits=50, seed=0):
    """Reports empirical coverage of the dispersion-normalized
    nonconformity score (error / dispersion) averaged across many random
    calibration/test splits, rather than a single split, since a single
    split's coverage estimate is noisy at moderate sample sizes."""
    n = len(dispersions)
    dispersions = np.asarray(dispersions)
    errors = np.asarray(errors)
    rng = np.random.default_rng(seed)

    coverages = []
    q_hats = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        n_cal = n // 2
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        cal_scores = errors[cal_idx] / dispersions[cal_idx]
        q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
        q_hat = np.quantile(cal_scores, q_level)

        covered = errors[test_idx] <= q_hat * dispersions[test_idx]
        coverages.append(covered.mean())
        q_hats.append(q_hat)

    return {
        "nominal_coverage": 1 - alpha,
        "mean_empirical_coverage": float(np.mean(coverages)),
        "std_empirical_coverage": float(np.std(coverages)),
        "mean_q_hat": float(np.mean(q_hats)),
    }


def selective_risk_curve(dispersions, errors, fractions=None):
    """Reports mean error among the most confident (lowest-dispersion)
    fraction of targets, for a range of retention fractions. A useful
    selective uncertainty signal shows decreasing mean error as the
    retention fraction decreases."""
    if fractions is None:
        fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    dispersions = np.asarray(dispersions)
    errors = np.asarray(errors)
    order = np.argsort(dispersions)

    n = len(dispersions)
    results = []
    for frac in fractions:
        k = max(1, int(round(n * frac)))
        kept_idx = order[:k]
        results.append({
            "keep_fraction": frac,
            "n_kept": k,
            "mean_error_among_kept": float(errors[kept_idx].mean()),
        })
    return results


def main() -> None:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    dispersions = [d["dispersion"] for d in data]
    errors = [d["error"] for d in data]

    print(f"N = {len(dispersions)}")

    print("\nSplit-conformal calibration (50 random splits, 90% nominal):")
    result = split_conformal(dispersions, errors, alpha=0.1, n_splits=50)
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nSelective risk curve:")
    curve = selective_risk_curve(dispersions, errors)
    for row in curve:
        print(f"  keep {row['keep_fraction']*100:.0f}% (n={row['n_kept']}): "
              f"mean error = {row['mean_error_among_kept']:.3f}")


if __name__ == "__main__":
    main()
