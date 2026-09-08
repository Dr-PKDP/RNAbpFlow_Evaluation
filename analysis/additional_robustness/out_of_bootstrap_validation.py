"""
Target-level out-of-bootstrap validation. For each replicate: resample
targets with replacement (the in-bag set), fit both models on the
in-bag sample, and evaluate on the targets not drawn (the out-of-bag
set). Reports the relative RMSE improvement distribution across
replicates and its stability across increasing replicate counts.

Usage:
    python out_of_bootstrap_validation.py <results.json>
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def fit_predict(log_length_train, log_dispersion_train, log_error_train,
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

    return model_0.predict(X_test).values, model_1.predict(X_test_1).values


def one_replicate(log_length, log_dispersion, log_error, rng, spline_df=4):
    n = len(log_length)
    in_bag_idx = rng.integers(0, n, size=n)
    out_of_bag_idx = np.setdiff1d(np.arange(n), in_bag_idx)
    if len(out_of_bag_idx) < 5:
        return None  # too few OOB targets this replicate, skip (rare)

    pred_0, pred_1 = fit_predict(
        log_length[in_bag_idx], log_dispersion[in_bag_idx], log_error[in_bag_idx],
        log_length[out_of_bag_idx], log_dispersion[out_of_bag_idx], spline_df)

    se_0 = (pred_0 - log_error[out_of_bag_idx]) ** 2
    se_1 = (pred_1 - log_error[out_of_bag_idx]) ** 2
    rmse_0 = np.sqrt(se_0.mean())
    rmse_1 = np.sqrt(se_1.mean())
    return 100 * (rmse_0 - rmse_1) / rmse_0, len(out_of_bag_idx)


def run_out_of_bootstrap(length, dispersion, error, n_boot=1000, seed=0):
    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    rng = np.random.default_rng(seed)
    improvements = []
    oob_sizes = []
    for _ in range(n_boot):
        result = one_replicate(log_length, log_dispersion, log_error, rng)
        if result is not None:
            improvements.append(result[0])
            oob_sizes.append(result[1])

    improvements = np.array(improvements)
    ci = np.percentile(improvements, [2.5, 97.5])
    return {
        "n_valid_replicates": len(improvements),
        "mean_oob_size": float(np.mean(oob_sizes)),
        "mean_improvement_pct": float(improvements.mean()),
        "median_improvement_pct": float(np.median(improvements)),
        "sd_improvement_pct": float(improvements.std()),
        "ci_95_pct": [float(ci[0]), float(ci[1])],
        "frac_negative": float((improvements < 0).mean()),
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    print(f"N = {len(length)}\n")

    print("=== Stability check: does the result stabilize as B increases? ===")
    for n_boot in [200, 500, 1000, 2000]:
        r = run_out_of_bootstrap(length, dispersion, error, n_boot=n_boot, seed=0)
        print(f"  B={n_boot}: mean={r['mean_improvement_pct']:.3f}%, "
              f"SD={r['sd_improvement_pct']:.3f}%, CI={r['ci_95_pct']}")

    print()
    print("=== Final result at B=2000 ===")
    final = run_out_of_bootstrap(length, dispersion, error, n_boot=2000, seed=0)
    for k, v in final.items():
        print(f"  {k}: {v}")
