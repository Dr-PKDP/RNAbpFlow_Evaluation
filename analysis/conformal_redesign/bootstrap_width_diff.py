"""
Bootstrap confidence interval for the paired width difference between
length-only and length-plus-dispersion split-conformal intervals.

For each of B bootstrap replicates: resample all targets with
replacement, then draw S random three-way partitions of the resampled
targets. Both models are fitted, calibrated and evaluated on the same
partition, so every width difference is paired. The replicate value is
the mean over the S partitions of (length-only width minus
length-plus-dispersion width), computed for the mean and the median
interval width over test targets.

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


def split_widths(log_length, log_dispersion, log_error, alpha, rng):
    """One partition; returns (mean, median) widths for both models."""
    n = len(log_length)
    idx = rng.permutation(n)
    n_train = int(n * 0.4)
    n_cal = int(n * 0.3)
    train_idx = idx[:n_train]
    cal_idx = idx[n_train:n_train + n_cal]
    test_idx = idx[n_train + n_cal:]
    if len(test_idx) < 3 or len(cal_idx) < 3 or len(train_idx) < 5:
        return None
    out = []
    for include_dispersion in (False, True):
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
        out.append((width.mean(), np.median(width)))
    return out


def replicate(length, dispersion, error, alpha, n_splits, seed):
    log_length, log_dispersion, log_error = np.log(length), np.log(dispersion), np.log(error)
    rng = np.random.default_rng(seed)
    d_mean, d_med = [], []
    for _ in range(n_splits):
        r = split_widths(log_length, log_dispersion, log_error, alpha, rng)
        if r is None:
            continue
        (m0, md0), (m1, md1) = r
        d_mean.append(m0 - m1)
        d_med.append(md0 - md1)
    return np.mean(d_mean), np.mean(d_med)


def run_bootstrap(length, dispersion, error, n_boot=500, n_splits_per_replicate=50, seed=0,
                  alpha=0.1):
    n = len(length)
    rng = np.random.default_rng(seed)
    mean_d, med_d = [], []
    for b in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        try:
            a, m = replicate(length[boot_idx], dispersion[boot_idx], error[boot_idx], alpha,
                             n_splits_per_replicate, seed=b)
        except Exception:
            continue
        mean_d.append(a)
        med_d.append(m)
    result = {"n_valid_replicates": len(mean_d)}
    for name, v in (("mean_width_diff", np.array(mean_d)), ("median_width_diff", np.array(med_d))):
        ci = np.percentile(v, [2.5, 97.5])
        result[name] = {"estimate_bootstrap_mean": float(v.mean()),
                        "ci_95": [float(ci[0]), float(ci[1])],
                        "frac_dispersion_narrower": float((v > 0).mean())}
    # the same quantities on the original (unresampled) targets
    a, m = replicate(length, dispersion, error, alpha, 4000, seed=12345)
    result["original_data_4000_splits"] = {"mean_width_diff": float(a), "median_width_diff": float(m)}
    return result


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    print(f"N = {len(length)}\n")
    result = run_bootstrap(length, dispersion, error, n_boot=n_boot)
    print(json.dumps(result, indent=2))
