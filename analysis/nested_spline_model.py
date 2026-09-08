"""
Nested spline-model comparison: tests whether MC-dropout dispersion
provides incremental, out-of-sample predictive value for structural
error beyond a nonlinear function of sequence length.

Model 0 (length only):      log(error) ~ spline(log(length), df=4)
Model 1 (length+dispersion): log(error) ~ spline(log(length), df=4) + log(dispersion)
Model 2 (with interaction):  log(error) ~ spline(log(length), df=4) + log(dispersion)
                                          + log(length):log(dispersion)

Comparison is via leave-one-out cross-validated RMSE and an in-sample
nested F-test. The in-sample F-test and the LOOCV RMSE difference test
different questions; for a significance test of the LOOCV improvement,
use paired_loocv_test.py, not this script's F-test.

The spline basis (knot placement) is fixed once from the full dataset;
only coefficients are refit per fold.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


def build_design(log_length, log_dispersion, spline_df=4, include_dispersion=True,
                  include_interaction=False, spline_basis_info=None,
                  interaction_aux_model=None):
    """Builds a design matrix. If spline_basis_info is provided (from an
    earlier patsy call), reuses that basis definition (same knots) for
    out-of-sample prediction; otherwise fits a new one from this data.

    patsy's cr() basis, with auto-intercept suppressed ('- 1'), already
    spans a constant function internally; no separate intercept term is
    added.

    The interaction term (length_x_dispersion) is the raw product
    log_length * log_dispersion, residualized against the full
    main-effects design, so it is orthogonal to every main-effect
    column by construction."""
    data = pd.DataFrame({"log_length": log_length, "log_dispersion": log_dispersion})
    formula = f"cr(log_length, df={spline_df}) - 1"
    if spline_basis_info is None:
        design = patsy.dmatrix(formula, data, return_type="dataframe")
    else:
        design = patsy.build_design_matrices([spline_basis_info], data, return_type="dataframe")[0]
    if include_dispersion:
        design["log_dispersion"] = log_dispersion
    if include_interaction:
        raw_interaction = log_length * log_dispersion
        if interaction_aux_model is None:
            # Fit the residualizing regression on THIS data
            aux_model = sm.OLS(raw_interaction, design).fit()
        else:
            aux_model = interaction_aux_model
        design["length_x_dispersion"] = raw_interaction - aux_model.predict(design)
    return design


def fit_and_loocv(log_length, log_error, log_dispersion, spline_df=4,
                   include_dispersion=True, include_interaction=False):
    """Returns (loocv_rmse, in_sample_model, full_design_matrix)."""
    n = len(log_length)

    # Fix the spline basis ONCE using all data
    full_data = pd.DataFrame({"log_length": log_length})
    full_design_full = patsy.dmatrix(f"cr(log_length, df={spline_df}) - 1", full_data,
                                      return_type="dataframe")
    basis_info = full_design_full.design_info

    # Fix the interaction-residualizing model ONCE using all data too, for
    # the same reason the spline knots are fixed once: the interaction
    # definition itself should not silently change per fold.
    interaction_aux_model = None
    if include_interaction:
        main_design_full = build_design(log_length, log_dispersion, spline_df,
                                         include_dispersion, include_interaction=False,
                                         spline_basis_info=basis_info)
        raw_interaction_full = log_length * log_dispersion
        interaction_aux_model = sm.OLS(raw_interaction_full, main_design_full).fit()

    # Full-data fit (for in-sample significance testing). No sm.add_constant --
    # the spline basis already provides the model's effective intercept.
    X_full = build_design(log_length, log_dispersion, spline_df,
                           include_dispersion, include_interaction, basis_info,
                           interaction_aux_model)
    model_full = sm.OLS(log_error, X_full).fit()

    # Leave-one-out CV
    squared_errors = []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train = build_design(log_length[train_idx], log_dispersion[train_idx],
                                spline_df, include_dispersion, include_interaction,
                                basis_info, interaction_aux_model)
        y_train = log_error[train_idx]
        model = sm.OLS(y_train, X_train).fit()

        X_test = build_design(log_length[[i]], log_dispersion[[i]],
                               spline_df, include_dispersion, include_interaction,
                               basis_info, interaction_aux_model)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        pred = model.predict(X_test)[0]
        squared_errors.append((pred - log_error[i]) ** 2)

    loocv_rmse = np.sqrt(np.mean(squared_errors))
    return loocv_rmse, model_full, X_full


def compare_models(length, dispersion, error, spline_df=4):
    log_length = np.log(np.asarray(length))
    log_dispersion = np.log(np.asarray(dispersion))
    log_error = np.log(np.asarray(error))

    rmse0, model0, X0 = fit_and_loocv(log_length, log_error, log_dispersion,
                                        spline_df, include_dispersion=False)
    rmse1, model1, X1 = fit_and_loocv(log_length, log_error, log_dispersion,
                                        spline_df, include_dispersion=True)
    rmse2, model2, X2 = fit_and_loocv(log_length, log_error, log_dispersion,
                                        spline_df, include_dispersion=True,
                                        include_interaction=True)

    # In-sample nested F-test: Model 0 vs Model 1 (does dispersion help at all?)
    f_test_01 = anova_lm(model0, model1)
    # In-sample nested F-test: Model 1 vs Model 2 (does the interaction help?)
    f_test_12 = anova_lm(model1, model2)

    return {
        "loocv_rmse_model0_length_only": rmse0,
        "loocv_rmse_model1_length_plus_dispersion": rmse1,
        "loocv_rmse_model2_with_interaction": rmse2,
        "loocv_improvement_0_to_1_pct": 100 * (rmse0 - rmse1) / rmse0,
        "loocv_improvement_1_to_2_pct": 100 * (rmse1 - rmse2) / rmse1,
        "insample_pvalue_dispersion_term (model0 vs model1)": f_test_01["Pr(>F)"].iloc[1],
        "insample_pvalue_interaction_term (model1 vs model2)": f_test_12["Pr(>F)"].iloc[1],
        "model1_dispersion_coef": model1.params.get("log_dispersion", np.nan),
        "model1_dispersion_pvalue": model1.pvalues.get("log_dispersion", np.nan),
    }


def main():
    import json
    import sys
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    results = compare_models(length, dispersion, error, spline_df=4)
    print(f"N = {len(length)}\n")
    print("=== Nested spline model comparison (length spline df=4, fixed in advance) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nloocv_improvement_0_to_1_pct > 0 means dispersion improves out-of-sample")
    print("prediction of error beyond a flexible nonlinear length model.")
    print("loocv_improvement_1_to_2_pct > 0 with a significant interaction p-value")
    print("means dispersion's relationship with error depends on sequence length.")


if __name__ == "__main__":
    main()
