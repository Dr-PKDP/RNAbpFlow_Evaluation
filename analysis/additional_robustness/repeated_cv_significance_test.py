"""
Applies a paired t-test and Wilcoxon signed-rank test directly to the
distribution of per-repeat relative RMSE improvements across repeated
5-fold cross-validation (many random 5-fold partitions).

This test is invalid: apparent significance is an artifact of the
analyst-chosen repeat count, not genuine sample size. Run with varying
repeat counts to see the effect.

Usage:
    python repeated_cv_significance_test.py <results.json>
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats


def fit_predict_fold(log_length_train, log_dispersion_train, log_error_train,
                       log_length_test, log_dispersion_test, spline_df=4):
    train_data = pd.DataFrame({"log_length": log_length_train})
    basis_info = patsy.dmatrix(f"cr(log_length, df={spline_df}) - 1", train_data,
                                 return_type="dataframe").design_info
    X_train = patsy.build_design_matrices([basis_info], train_data, return_type="dataframe")[0]
    X_train_1 = X_train.copy()
    X_train_1["log_dispersion"] = log_dispersion_train
    model_0 = sm.OLS(log_error_train, X_train).fit()
    model_1 = sm.OLS(log_error_train, X_train_1).fit()

    test_data = pd.DataFrame({"log_length": log_length_test})
    X_test = patsy.build_design_matrices([basis_info], test_data, return_type="dataframe")[0]
    X_test_1 = X_test.copy()
    X_test_1["log_dispersion"] = log_dispersion_test
    X_test_1 = X_test_1.reindex(columns=X_train_1.columns, fill_value=0.0)

    pred_0 = model_0.predict(X_test)
    pred_1 = model_1.predict(X_test_1)
    return pred_0.values, pred_1.values


def one_repeat(log_length, log_dispersion, log_error, k, rng, spline_df=4):
    n = len(log_length)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    se_0_all, se_1_all = [], []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        pred_0, pred_1 = fit_predict_fold(
            log_length[train_idx], log_dispersion[train_idx], log_error[train_idx],
            log_length[test_idx], log_dispersion[test_idx], spline_df)
        se_0_all.append((pred_0 - log_error[test_idx]) ** 2)
        se_1_all.append((pred_1 - log_error[test_idx]) ** 2)
    se_0_all = np.concatenate(se_0_all)
    se_1_all = np.concatenate(se_1_all)
    rmse_0 = np.sqrt(se_0_all.mean())
    rmse_1 = np.sqrt(se_1_all.mean())
    relative_improvement_pct = (rmse_0 - rmse_1) / rmse_0 * 100
    return relative_improvement_pct


def run_repeated_cv_test(length, dispersion, error, k=5, n_repeats=100, seed=0):
    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    rng = np.random.default_rng(seed)
    improvements = np.array([
        one_repeat(log_length, log_dispersion, log_error, k, rng)
        for _ in range(n_repeats)
    ])

    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(improvements)
    ttest_stat, ttest_p = stats.ttest_1samp(improvements, 0)

    return {
        "n_repeats": n_repeats,
        "mean_improvement_pct": float(improvements.mean()),
        "sd_improvement_pct": float(improvements.std()),
        "min_pct": float(improvements.min()),
        "max_pct": float(improvements.max()),
        "frac_negative": float((improvements < 0).mean()),
        "wilcoxon_statistic": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
        "ttest_statistic": float(ttest_stat),
        "ttest_p": float(ttest_p),
        "raw_improvements": improvements.tolist(),
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    print(f"N = {len(length)}\n")
    results = run_repeated_cv_test(length, dispersion, error)
    for k, v in results.items():
        if k != "raw_improvements":
            print(f"  {k}: {v}")
