"""
Split-conformal prediction for the residual-based nonconformity score,
with a genuine three-way partition, repeated over many random splits:
    - TRAIN (40%): fit the point-prediction model.
    - CALIBRATE (30%): compute |log(E) - pred| residuals from the
      TRAIN-fitted model.
    - TEST (30%): evaluate coverage and width.
Run for both a length-only and a length-plus-dispersion point-prediction
model at matched ~90% nominal coverage.

Usage:
    python three_way_split_conformal.py <results.json>
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

# Default: linear interpolation at level ceil((n+1)(1-alpha))/n, which is never below the rank-k order
# statistic. Set EXACT_ORDER_STATISTIC = True (or pass --exact) for the rank-k order statistic itself.
EXACT_ORDER_STATISTIC = "--exact" in sys.argv


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


def one_split(log_length, log_dispersion, log_error, error, alpha, rng, include_dispersion):
    n = len(log_length)
    idx = rng.permutation(n)
    n_train = int(n * 0.4)
    n_cal = int(n * 0.3)
    train_idx = idx[:n_train]
    cal_idx = idx[n_train:n_train + n_cal]
    test_idx = idx[n_train + n_cal:]

    model, basis_info, columns = fit_model(
        log_length[train_idx], log_dispersion[train_idx], log_error[train_idx],
        include_dispersion=include_dispersion)

    pred_cal = predict_with(model, basis_info, columns, log_length[cal_idx],
                              log_dispersion[cal_idx], include_dispersion)
    cal_scores = np.abs(log_error[cal_idx] - pred_cal)
    n_c = len(cal_idx)
    q_level = min(np.ceil((n_c + 1) * (1 - alpha)) / n_c, 1.0)
    q_hat = (np.sort(cal_scores)[int(round(q_level * n_c)) - 1] if EXACT_ORDER_STATISTIC
             else np.quantile(cal_scores, q_level))

    pred_test = predict_with(model, basis_info, columns, log_length[test_idx],
                               log_dispersion[test_idx], include_dispersion)
    lower = np.exp(pred_test - q_hat)
    upper = np.exp(pred_test + q_hat)
    covered = (error[test_idx] >= lower) & (error[test_idx] <= upper)
    widths = upper - lower
    return covered.mean(), widths.mean(), np.median(widths), len(test_idx)


def run_three_way(length, dispersion, error, alpha=0.1, n_splits=200, seed=0):
    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    results = {}
    for name, include_disp in [("length_only", False), ("length_plus_dispersion", True)]:
        rng = np.random.default_rng(seed)
        covs, mean_widths, median_widths, test_sizes = [], [], [], []
        for _ in range(n_splits):
            c, mw, medw, ts = one_split(log_length, log_dispersion, log_error, error,
                                          alpha, rng, include_disp)
            covs.append(c); mean_widths.append(mw); median_widths.append(medw)
            test_sizes.append(ts)
        results[name] = {
            "mean_coverage": float(np.mean(covs)),
            "mean_width": float(np.mean(mean_widths)),
            "median_width": float(np.median(median_widths)),
            "mean_test_size": float(np.mean(test_sizes)),
        }
    return results


if __name__ == "__main__":
    with open([a for a in sys.argv[1:] if not a.startswith('--')][0]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    print(f"N = {len(length)}\n")
    results = run_three_way(length, dispersion, error, n_splits=200)
    for name, r in results.items():
        print(f"{name}: coverage={r['mean_coverage']:.3f}, mean_width={r['mean_width']:.1f} A, "
              f"median_width={r['median_width']:.1f} A, mean_test_n={r['mean_test_size']:.1f}")
