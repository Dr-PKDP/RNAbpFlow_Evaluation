"""
Sensitivity of the dispersion incremental effect to the length spline's
degrees of freedom. df=4 is the primary, pre-specified value; this
reports df=3 and df=5 as a robustness check rather than searching for
the best-fitting df (which would introduce a new, undisclosed tuning
step at this sample size).

Usage:
    python df_sensitivity.py <results.json>
"""
import json
import sys

import numpy as np

from paired_loocv_test import paired_loocv_comparison


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    print(f"N = {len(length)}\n")
    for df in [3, 4, 5]:
        results = paired_loocv_comparison(length, dispersion, error, spline_df=df)
        marker = " (primary)" if df == 4 else ""
        print(f"df={df}{marker}:")
        print(f"  LOOCV RMSE: {results['loocv_rmse_model0']:.4f} -> "
              f"{results['loocv_rmse_model1']:.4f} "
              f"({results['relative_improvement_pct']:+.2f}%)")
        ci = results["bootstrap_95ci_diff"]
        print(f"  Mean squared-error diff: {results['mean_squared_error_diff']:.5f}, "
              f"95% CI [{ci[0]:.5f}, {ci[1]:.5f}]")
        print(f"  Paired permutation p: {results['paired_permutation_p']:.4f}\n")

    print("Interpretation: if the relative improvement and CI direction are similar "
          "across df=3,4,5, the finding is robust to this choice. If magnitude or "
          "direction changes substantially, the finding is more fragile than a "
          "single df=4 result would suggest.")


if __name__ == "__main__":
    main()
