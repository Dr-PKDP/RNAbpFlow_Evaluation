"""
Redesigned nonconformity score for split-conformal calibration, replacing
the E/D ratio (unstable for near-zero-dispersion targets) with a
residual-based score computed on the log-error scale.

Uses genuine leave-one-out residuals as the score -- NOT a same-data
model-fit-then-score approach, which was tested first and found to
undercover (roughly 80-85% instead of the 90% nominal target) because
scoring a model on the exact data used to fit it understates residuals.
LOOCV residuals avoid this: each target's residual comes from a model
that never saw that target.

Score for target i: S_i = |log(E_i) - pred_i|, where pred_i is the
leave-one-out prediction from Model 1 (spline(log length) +
log(dispersion)) fit on the other 81 targets.

Calibration then randomly splits these 82 LOOCV residuals into
calibration/test subsets (50 repeats) to assess coverage and width,
matching the same evaluation protocol as the original E/D-ratio version
for a fair comparison.

Usage:
    python residual_conformal.py <results.json>
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def loocv_predictions(log_length, log_dispersion, log_error, spline_df=4):
    n = len(log_length)
    preds = np.zeros(n)
    full_data = pd.DataFrame({"log_length": log_length})
    basis_info = patsy.dmatrix(f"cr(log_length, df={spline_df}) - 1", full_data,
                                 return_type="dataframe").design_info
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        train_data = pd.DataFrame({"log_length": log_length[train_idx]})
        X_train = patsy.build_design_matrices([basis_info], train_data, return_type="dataframe")[0]
        X_train["log_dispersion"] = log_dispersion[train_idx]
        model = sm.OLS(log_error[train_idx], X_train).fit()

        test_data = pd.DataFrame({"log_length": log_length[[i]]})
        X_test = patsy.build_design_matrices([basis_info], test_data, return_type="dataframe")[0]
        X_test["log_dispersion"] = log_dispersion[[i]]
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        preds[i] = model.predict(X_test).values[0]
    return preds


def run_comparison(length, dispersion, error, alpha=0.1, n_splits=50, seed=0):
    n = len(length)
    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    # Genuine out-of-fold predictions and residuals -- computed ONCE, honestly
    loocv_pred = loocv_predictions(log_length, log_dispersion, log_error)
    scores = np.abs(log_error - loocv_pred)

    median_length = np.median(length)
    short_mask = length <= median_length
    long_mask = length > median_length

    rng = np.random.default_rng(seed)

    def eval_stratum(mask):
        idx_pool = np.where(mask)[0]
        n_pool = len(idx_pool)
        covs, mean_widths, median_widths = [], [], []
        for _ in range(n_splits):
            perm = rng.permutation(idx_pool)
            n_cal = n_pool // 2
            cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
            q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
            q_hat = np.quantile(scores[cal_idx], q_level)
            lower = np.exp(loocv_pred[test_idx] - q_hat)
            upper = np.exp(loocv_pred[test_idx] + q_hat)
            covered = (error[test_idx] >= lower) & (error[test_idx] <= upper)
            widths = upper - lower
            covs.append(covered.mean())
            mean_widths.append(widths.mean())
            median_widths.append(np.median(widths))
        return {
            "mean_coverage": float(np.mean(covs)),
            "mean_width": float(np.mean(mean_widths)),
            "median_width": float(np.median(median_widths)),
        }

    return {
        "pooled": eval_stratum(np.ones(n, dtype=bool)),
        "short": eval_stratum(short_mask),
        "long": eval_stratum(long_mask),
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    print(f"N = {len(length)}\n")
    results = run_comparison(length, dispersion, error)
    for name, r in results.items():
        print(f"{name}: coverage={r['mean_coverage']:.3f}, "
              f"mean width={r['mean_width']:.1f} A, median width={r['median_width']:.1f} A")

    print("\nReference values for the E/D-ratio score: "
          "pooled 89.1% coverage, short-stratum median width 408.9 A, long-stratum "
          "median width 107.8 A.")
