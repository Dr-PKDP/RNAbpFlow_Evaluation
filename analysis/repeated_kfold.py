"""
Repeated 5-fold cross-validation, as a stability check on the LOOCV
RMSE improvement -- confirms the result is not an artifact of one
particular resampling strategy.

Usage:
    python repeated_kfold.py <results.json> [n_repeats]
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from nested_spline_model import build_design


def kfold_squared_errors(log_length, log_error, log_dispersion, k, seed,
                          spline_df=4, include_dispersion=True):
    n = len(log_length)
    rng = np.random.default_rng(seed)
    fold_assignment = rng.permutation(n) % k

    full_data = pd.DataFrame({"log_length": log_length})
    basis_info = patsy.dmatrix(f"cr(log_length, df={spline_df}) - 1", full_data,
                                return_type="dataframe").design_info

    squared_errors = np.zeros(n)
    for fold in range(k):
        test_idx = np.where(fold_assignment == fold)[0]
        train_idx = np.where(fold_assignment != fold)[0]

        X_train = build_design(log_length[train_idx], log_dispersion[train_idx],
                                spline_df, include_dispersion, False, basis_info)
        y_train = log_error[train_idx]
        model = sm.OLS(y_train, X_train).fit()

        X_test = build_design(log_length[test_idx], log_dispersion[test_idx],
                               spline_df, include_dispersion, False, basis_info)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        preds = model.predict(X_test)
        squared_errors[test_idx] = (preds.values - log_error[test_idx]) ** 2

    return squared_errors


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    n_repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])
    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    improvements = []
    for repeat in range(n_repeats):
        se0 = kfold_squared_errors(log_length, log_error, log_dispersion, k=5,
                                     seed=repeat, include_dispersion=False)
        se1 = kfold_squared_errors(log_length, log_error, log_dispersion, k=5,
                                     seed=repeat, include_dispersion=True)
        rmse0 = np.sqrt(se0.mean())
        rmse1 = np.sqrt(se1.mean())
        improvements.append(100 * (rmse0 - rmse1) / rmse0)

    improvements = np.array(improvements)
    print(f"N = {len(length)}, {n_repeats} repeats of 5-fold CV\n")
    print(f"Relative RMSE improvement (Model 0 -> Model 1) across repeats:")
    print(f"  mean={improvements.mean():.2f}%, std={improvements.std():.2f}%")
    print(f"  min={improvements.min():.2f}%, max={improvements.max():.2f}%")
    print(f"  fraction of repeats showing improvement (>0%): "
          f"{(improvements > 0).mean():.1%}")


if __name__ == "__main__":
    main()
