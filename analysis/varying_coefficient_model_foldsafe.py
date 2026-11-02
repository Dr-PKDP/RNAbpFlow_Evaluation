"""
varying_coefficient_model_foldsafe.py -- Fold-safe re-verification of
varying_coefficient_model.py's LOOCV comparison. Preserves the original
model exactly (same centered-quadratic g(log L), same residualization
technique, same NaN-avoidance workaround), but refits THREE things that
the original computes once on the full 82-target dataset, per fold
instead, using only the 81 training targets each time:

  1. the main 4-df spline knot placement (main_basis_info)
  2. the g(log L) centering value (center = log_length.mean())
  3. the interaction-residualizing auxiliary model (aux_model)

This directly answers whether any of the three affects the reported
LOOCV comparison, matching the same check already completed for the
primary length-only vs length+dispersion model.

Usage (run from backbones/rnabpflow/):
    python3 varying_coefficient_model_foldsafe.py dispersion_error_results_deduped.json
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

# Reuse the original module's basis-building functions directly, so the
# model itself is identical -- only the fold-safety of its inputs changes.
from varying_coefficient_model import build_varying_basis, build_design


def loocv_varying_coefficient_foldsafe(length, dispersion, error, main_spline_df=4):
    log_length = np.log(np.asarray(length))
    log_dispersion = np.log(np.asarray(dispersion))
    log_error = np.log(np.asarray(error))
    n = len(log_length)

    se_main = np.zeros(n)
    se_vc = np.zeros(n)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]

        # --- All three of the following are now fold-specific ---
        center = log_length[train_idx].mean()

        main_data_train = pd.DataFrame({"log_length": log_length[train_idx]})
        main_basis_info = patsy.dmatrix(
            f"cr(log_length, df={main_spline_df}) - 1", main_data_train,
            return_type="dataframe"
        ).design_info

        design_train = build_design(log_length[train_idx], log_dispersion[train_idx], main_basis_info)
        varying_train = build_varying_basis(log_length[train_idx], center)
        raw_interaction_train = varying_train.multiply(log_dispersion[train_idx], axis=0)
        raw_interaction_train.columns = ["vc_interact_lin", "vc_interact_sq"]

        # aux_model refit on training data only (was: fixed once on all 82)
        aux_model = sm.OLS(raw_interaction_train, design_train).fit()
        clean_interaction_train = pd.DataFrame(
            raw_interaction_train.values - aux_model.predict(design_train).values,
            columns=raw_interaction_train.columns, index=raw_interaction_train.index,
        )
        # --- end fold-specific block ---

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


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    print(f"N = {len(length)}")
    print()

    # Re-run the ORIGINAL (pooled-knot/center/aux) version first, for a
    # direct side-by-side comparison against the fold-safe version below.
    from varying_coefficient_model import loocv_varying_coefficient as loocv_original
    se_main_a, se_vc_a = loocv_original(length, dispersion, error)
    rmse_main_a = np.sqrt(se_main_a.mean())
    rmse_vc_a = np.sqrt(se_vc_a.mean())
    d_a = se_main_a - se_vc_a
    rng = np.random.default_rng(0)
    n = len(d_a)
    boot_means_a = np.array([rng.choice(d_a, n, replace=True).mean() for _ in range(10000)])
    ci_a = np.percentile(boot_means_a, [2.5, 97.5])

    print("=" * 70)
    print("(A) Pooled knots, centring value and auxiliary model")
    print("=" * 70)
    print(f"  LOOCV RMSE, constant-coefficient: {rmse_main_a:.4f}")
    print(f"  LOOCV RMSE, varying-coefficient:  {rmse_vc_a:.4f}")
    print(f"  Mean squared-error diff: {d_a.mean():.5f}")
    print(f"  95% bootstrap interval: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print()

    # Fold-safe version
    se_main_b, se_vc_b = loocv_varying_coefficient_foldsafe(length, dispersion, error)
    rmse_main_b = np.sqrt(se_main_b.mean())
    rmse_vc_b = np.sqrt(se_vc_b.mean())
    d_b = se_main_b - se_vc_b
    rng2 = np.random.default_rng(0)
    boot_means_b = np.array([rng2.choice(d_b, n, replace=True).mean() for _ in range(10000)])
    ci_b = np.percentile(boot_means_b, [2.5, 97.5])

    print("=" * 70)
    print("(B) FOLD-SAFE (knots, center, and aux-model all refit per fold")
    print("    using only the 81 training targets)")
    print("=" * 70)
    print(f"  LOOCV RMSE, constant-coefficient: {rmse_main_b:.4f}")
    print(f"  LOOCV RMSE, varying-coefficient:  {rmse_vc_b:.4f}")
    print(f"  Mean squared-error diff: {d_b.mean():.5f}")
    print(f"  95% bootstrap interval: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    print()

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  RMSE diff (A vs B), constant-coef: {rmse_main_a - rmse_main_b:+.4f}")
    print(f"  RMSE diff (A vs B), varying-coef:  {rmse_vc_a - rmse_vc_b:+.4f}")
    print("Paste back this full output.")


if __name__ == "__main__":
    main()
