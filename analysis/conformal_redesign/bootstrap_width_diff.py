"""
Bootstrap confidence interval for the paired mean-width difference
between length-only and length-plus-dispersion split-conformal
intervals.

For each of B bootstrap replicates: resample all targets with
replacement, then run the complete three-way split-conformal procedure
on this bootstrap sample for both models, and record the paired
mean-width difference.

Usage:
    python bootstrap_width_diff.py <results.json>
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def fit_model(log_length_train, log_dispersion_train, log_error_train,
               spline_df=4, include_dispersion=True):
    train_data = pd.DataFrame({"log_length": log_length_train})
    basis_info = patsy.dmatrix(f"cr(log_length, df={spline_df}) - 1", train_data,
                                 return_type="dataframe").design_info
    X_train = patsy.build_design_matrices([basis_info], train_data, return_type="dataframe")[0]
    if include_dispersion:
        X_train["log_dispersion"] = log_dispersion_train
    model = sm.OLS(log_error_train, X_train).fit()
    return model, basis_info, X_train.columns


def predict_with(model, basis_info, columns, log_length, log_dispersion, include_dispersion):
    data = pd.DataFrame({"log_length": log_length})
    X = patsy.build_design_matrices([basis_info], data, return_type="dataframe")[0]
    if include_dispersion:
        X["log_dispersion"] = log_dispersion
    X = X.reindex(columns=columns, fill_value=0.0)
    return model.predict(X).values


def one_split_width(log_length, log_dispersion, log_error, alpha, rng, include_dispersion):
    n = len(log_length)
    idx = rng.permutation(n)
    n_train = int(n * 0.4)
    n_cal = int(n * 0.3)
    train_idx = idx[:n_train]
    cal_idx = idx[n_train:n_train + n_cal]
    test_idx = idx[n_train + n_cal:]
    if len(test_idx) < 3 or len(cal_idx) < 3 or len(train_idx) < 5:
        return None

    model, basis_info, columns = fit_model(
        log_length[train_idx], log_dispersion[train_idx], log_error[train_idx],
        include_dispersion=include_dispersion)
    pred_cal = predict_with(model, basis_info, columns, log_length[cal_idx],
                              log_dispersion[cal_idx], include_dispersion)
    cal_scores = np.abs(log_error[cal_idx] - pred_cal)
    n_c = len(cal_idx)
    q_level = min(np.ceil((n_c + 1) * (1 - alpha)) / n_c, 1.0)
    q_hat = np.quantile(cal_scores, q_level)

    pred_test = predict_with(model, basis_info, columns, log_length[test_idx],
                               log_dispersion[test_idx], include_dispersion)
    width = np.exp(pred_test + q_hat) - np.exp(pred_test - q_hat)
    return width.mean()


def mean_width_diff_once(length, dispersion, error, alpha=0.1, n_splits=50, seed=0):
    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)
    rng = np.random.default_rng(seed)

    widths_lo, widths_lpd = [], []
    for _ in range(n_splits):
        w_lo = one_split_width(log_length, log_dispersion, log_error, alpha, rng, False)
        w_lpd = one_split_width(log_length, log_dispersion, log_error, alpha, rng, True)
        if w_lo is not None:
            widths_lo.append(w_lo)
        if w_lpd is not None:
            widths_lpd.append(w_lpd)
    return np.mean(widths_lo) - np.mean(widths_lpd)  # positive = dispersion narrower


def run_bootstrap(length, dispersion, error, n_boot=200, n_splits_per_replicate=50, seed=0):
    n = len(length)
    rng = np.random.default_rng(seed)
    diffs = []
    for b in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        try:
            d = mean_width_diff_once(length[boot_idx], dispersion[boot_idx], error[boot_idx],
                                       n_splits=n_splits_per_replicate, seed=b)
            diffs.append(d)
        except Exception:
            continue
    diffs = np.array(diffs)
    ci = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_valid_replicates": len(diffs),
        "mean_width_diff": float(diffs.mean()),
        "ci_95": [float(ci[0]), float(ci[1])],
        "frac_dispersion_narrower": float((diffs > 0).mean()),
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    print(f"N = {len(length)}\n")
    result = run_bootstrap(length, dispersion, error, n_boot=200, n_splits_per_replicate=50)
    for k, v in result.items():
        print(f"  {k}: {v}")
