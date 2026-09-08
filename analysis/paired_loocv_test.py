"""
Out-of-sample significance test for the LOOCV RMSE improvement: a
paired, per-target comparison of out-of-fold squared errors, with a
bootstrap confidence interval and permutation test across targets. Use
this, not an in-sample nested F-test, for the LOOCV difference.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from nested_spline_model import build_design


def per_target_loo_squared_errors(log_length, log_error, log_dispersion, spline_df=4,
                                    include_dispersion=True):
    """Returns the array of per-target OUT-OF-FOLD squared errors (one
    per target), not just their aggregate RMSE -- needed for paired
    inference across targets."""
    n = len(log_length)
    full_data = pd.DataFrame({"log_length": log_length})
    basis_info = patsy.dmatrix(f"cr(log_length, df={spline_df}) - 1", full_data,
                                return_type="dataframe").design_info

    squared_errors = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train = build_design(log_length[train_idx], log_dispersion[train_idx],
                                spline_df, include_dispersion, False, basis_info)
        y_train = log_error[train_idx]
        model = sm.OLS(y_train, X_train).fit()

        X_test = build_design(log_length[[i]], log_dispersion[[i]],
                               spline_df, include_dispersion, False, basis_info)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        pred = model.predict(X_test)[0]
        squared_errors[i] = (pred - log_error[i]) ** 2

    return squared_errors


def paired_loocv_comparison(length, dispersion, error, spline_df=4, n_boot=10000, seed=0):
    log_length = np.log(np.asarray(length))
    log_dispersion = np.log(np.asarray(dispersion))
    log_error = np.log(np.asarray(error))
    n = len(log_length)

    se0 = per_target_loo_squared_errors(log_length, log_error, log_dispersion,
                                         spline_df, include_dispersion=False)
    se1 = per_target_loo_squared_errors(log_length, log_error, log_dispersion,
                                         spline_df, include_dispersion=True)

    d = se0 - se1  # positive means Model 1 (with dispersion) had lower squared error
    rmse0 = np.sqrt(se0.mean())
    rmse1 = np.sqrt(se1.mean())
    mean_d = d.mean()

    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
    ci = np.percentile(boot_means, [2.5, 97.5])

    # Paired permutation test: randomly flip the sign of each target's
    # difference (equivalent to permuting which model "wins" per target
    # under the null of no systematic difference), compare to observed mean.
    signs = rng.choice([-1, 1], size=(n_boot, n))
    perm_means = (signs * d).mean(axis=1)
    perm_p = np.mean(np.abs(perm_means) >= np.abs(mean_d))

    return {
        "loocv_rmse_model0": rmse0,
        "loocv_rmse_model1": rmse1,
        "relative_improvement_pct": 100 * (rmse0 - rmse1) / rmse0,
        "mean_squared_error_diff": mean_d,
        "bootstrap_95ci_diff": tuple(ci),
        "paired_permutation_p": perm_p,
    }


if __name__ == "__main__":
    import json
    import sys
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    results = paired_loocv_comparison(length, dispersion, error, spline_df=4)
    print(f"N = {len(length)}\n")
    for k, v in results.items():
        if isinstance(v, tuple):
            print(f"  {k}: [{v[0]:.5f}, {v[1]:.5f}]")
        else:
            print(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")
