"""
Varying-coefficient sensitivity model:

    log(E) = s_4(log L) + [beta_0 + g(log L)] * log(D) + eps

s_4 is the primary 4-df length spline (main effect); g(log L) is a
low-complexity (~2 df) function modulating the dispersion coefficient.

g(log L) is a centered quadratic polynomial (log_length, log_length^2).
The centering value is fixed once from the full dataset and reused for
every train/test fold. The g(log L)*log(D) interaction columns are
residualized against the full main-effects design.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


def build_varying_basis(log_length, center):
    """Centered quadratic polynomial in log_length -- the low-complexity
    (2-parameter) modulator basis. `center` must be fixed once (from the
    full dataset) and passed identically for every fold."""
    lc = log_length - center
    return pd.DataFrame({"vc_lin": lc, "vc_sq": lc ** 2})


def build_design(log_length, log_dispersion, main_basis_info):
    data = pd.DataFrame({"log_length": log_length})
    main_basis = patsy.build_design_matrices([main_basis_info], data,
                                              return_type="dataframe")[0]
    design = main_basis.copy()
    design["log_dispersion"] = log_dispersion
    return design


def fit_varying_coefficient_model(log_length, log_error, log_dispersion, main_spline_df=4):
    center = log_length.mean()
    main_data = pd.DataFrame({"log_length": log_length})
    main_basis_info = patsy.dmatrix(f"cr(log_length, df={main_spline_df}) - 1", main_data,
                                     return_type="dataframe").design_info

    design = build_design(log_length, log_dispersion, main_basis_info)
    varying_basis = build_varying_basis(log_length, center)
    raw_interaction = varying_basis.multiply(log_dispersion, axis=0)
    raw_interaction.columns = ["vc_interact_lin", "vc_interact_sq"]

    aux_model = sm.OLS(raw_interaction, design).fit()
    # aux_model.predict() returns generic column names (0, 1, ...) for a
    # multi-column endog, not the original names -- subtracting two
    # DataFrames with mismatched column names silently produces NaN
    # everywhere via pandas' automatic column alignment, rather than an
    # error. Using .values sidesteps pandas alignment entirely.
    clean_interaction = pd.DataFrame(
        raw_interaction.values - aux_model.predict(design).values,
        columns=raw_interaction.columns, index=raw_interaction.index,
    )

    X_full = pd.concat([design, clean_interaction], axis=1)
    model_main = sm.OLS(log_error, design).fit()
    model_full = sm.OLS(log_error, X_full).fit()

    f_test = anova_lm(model_main, model_full)
    p_value = f_test["Pr(>F)"].iloc[1]
    return model_main, model_full, p_value


def loocv_varying_coefficient(length, dispersion, error, main_spline_df=4):
    log_length = np.log(np.asarray(length))
    log_dispersion = np.log(np.asarray(dispersion))
    log_error = np.log(np.asarray(error))
    n = len(log_length)

    center = log_length.mean()  # fixed once, using the full dataset
    main_data = pd.DataFrame({"log_length": log_length})
    main_basis_info = patsy.dmatrix(f"cr(log_length, df={main_spline_df}) - 1", main_data,
                                     return_type="dataframe").design_info

    # Fix the interaction-residualizing model once, using the full dataset
    full_design = build_design(log_length, log_dispersion, main_basis_info)
    full_varying = build_varying_basis(log_length, center)
    full_raw_interaction = full_varying.multiply(log_dispersion, axis=0)
    full_raw_interaction.columns = ["vc_interact_lin", "vc_interact_sq"]
    aux_model = sm.OLS(full_raw_interaction, full_design).fit()

    se_main = np.zeros(n)
    se_vc = np.zeros(n)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        design_train = build_design(log_length[train_idx], log_dispersion[train_idx], main_basis_info)
        varying_train = build_varying_basis(log_length[train_idx], center)
        raw_interaction_train = varying_train.multiply(log_dispersion[train_idx], axis=0)
        raw_interaction_train.columns = ["vc_interact_lin", "vc_interact_sq"]
        clean_interaction_train = pd.DataFrame(
            raw_interaction_train.values - aux_model.predict(design_train).values,
            columns=raw_interaction_train.columns, index=raw_interaction_train.index,
        )

        X_full_train = pd.concat([design_train, clean_interaction_train], axis=1)
        y_train = log_error[train_idx]
        model_main = sm.OLS(y_train, design_train).fit()
        model_vc = sm.OLS(y_train, X_full_train).fit()

        design_test = build_design(log_length[[i]], log_dispersion[[i]], main_basis_info)
        varying_test = build_varying_basis(log_length[[i]], center)
        raw_interaction_test = varying_test.multiply(log_dispersion[[i]], axis=0)
        raw_interaction_test.columns = ["vc_interact_lin", "vc_interact_sq"]
        clean_interaction_test = pd.DataFrame(
            raw_interaction_test.values - aux_model.predict(design_test).values,
            columns=raw_interaction_test.columns, index=raw_interaction_test.index,
        )

        X_main_test = design_test.reindex(columns=design_train.columns, fill_value=0.0)
        X_full_test = pd.concat([design_test, clean_interaction_test], axis=1)
        X_full_test = X_full_test.reindex(columns=X_full_train.columns, fill_value=0.0)

        se_main[i] = (model_main.predict(X_main_test)[0] - log_error[i]) ** 2
        se_vc[i] = (model_vc.predict(X_full_test)[0] - log_error[i]) ** 2

    return se_main, se_vc


if __name__ == "__main__":
    import json
    import sys
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    model_main, model_vc, p_value = fit_varying_coefficient_model(log_length, log_error, log_dispersion)
    print(f"N = {len(length)}")
    print(f"In-sample nested F-test, varying-coefficient term: p={p_value:.4f}\n")

    se_main, se_vc = loocv_varying_coefficient(length, dispersion, error)
    rmse_main = np.sqrt(se_main.mean())
    rmse_vc = np.sqrt(se_vc.mean())
    print(f"LOOCV RMSE, Model 1 (constant dispersion coefficient): {rmse_main:.4f}")
    print(f"LOOCV RMSE, varying-coefficient model:                 {rmse_vc:.4f}")
    print(f"Relative change: {100*(rmse_main-rmse_vc)/rmse_main:+.2f}%")

    d = se_main - se_vc
    rng = np.random.default_rng(0)
    n = len(d)
    boot_means = np.array([rng.choice(d, n, replace=True).mean() for _ in range(10000)])
    ci = np.percentile(boot_means, [2.5, 97.5])
    print(f"Bootstrap 95% CI for squared-error difference: [{ci[0]:.5f}, {ci[1]:.5f}]")
    print(f"\nInterpretation: if the varying-coefficient model does not improve LOOCV "
          f"RMSE and the CI includes zero, there is no detectable evidence that "
          f"dispersion's effect varies smoothly with length.")
