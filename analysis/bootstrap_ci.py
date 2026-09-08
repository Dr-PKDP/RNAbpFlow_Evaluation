"""
Computes bootstrap 95% confidence intervals for correlation statistics
between an uncertainty signal and structural error, via nonparametric
case resampling. Recommended whenever reporting a correlation on a
modest sample size, particularly an adjusted (e.g. partial) correlation,
where a point estimate alone does not indicate whether the effect is
distinguishable from sampling noise.

Usage:
    python bootstrap_ci.py <results.json> [n_bootstrap]

results.json: list of {"dispersion": float, "error": float, "length": int}
"""
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr


def partial_correlation(x, y, z):
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))


def bootstrap_ci(dispersions, errors, lengths, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dispersions)
    pearson_samples, spearman_samples, partial_samples = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d, e, l = dispersions[idx], errors[idx], lengths[idx]
        if len(set(d)) < 2 or len(set(e)) < 2:
            continue
        pearson_samples.append(pearsonr(d, e)[0])
        spearman_samples.append(spearmanr(d, e)[0])
        try:
            partial_samples.append(partial_correlation(d, e, l))
        except Exception:
            pass
    return (
        np.percentile(pearson_samples, [2.5, 97.5]),
        np.percentile(spearman_samples, [2.5, 97.5]),
        np.percentile(partial_samples, [2.5, 97.5]),
    )


def main() -> None:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    dispersions = np.array([d["dispersion"] for d in data])
    errors = np.array([d["error"] for d in data])
    lengths = np.array([d["length"] for d in data])

    point_pearson = pearsonr(dispersions, errors)[0]
    point_spearman = spearmanr(dispersions, errors)[0]
    point_partial = partial_correlation(dispersions, errors, lengths)

    pearson_ci, spearman_ci, partial_ci = bootstrap_ci(dispersions, errors, lengths, n_boot)

    print(f"N = {len(dispersions)}, {n_boot} bootstrap resamples\n")
    print(f"Pearson r:  {point_pearson:.3f}  95% CI [{pearson_ci[0]:.3f}, {pearson_ci[1]:.3f}]")
    print(f"Spearman rho: {point_spearman:.3f}  95% CI [{spearman_ci[0]:.3f}, {spearman_ci[1]:.3f}]")
    print(f"Partial r (controlling length): {point_partial:.3f}  "
          f"95% CI [{partial_ci[0]:.3f}, {partial_ci[1]:.3f}]")


if __name__ == "__main__":
    main()
