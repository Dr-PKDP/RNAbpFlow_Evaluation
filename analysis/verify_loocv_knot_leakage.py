"""
verify_loocv_knot_leakage.py -- Tests whether refitting the spline knots
strictly within each LOOCV fold changes the result, compared with knots
computed once from all 82 targets (which lets each held-out target's
log(length) enter its own fold's basis).

Runs both versions and reports both LOOCV RMSE values side by side:
  (A) "pooled-knot" LOOCV: knots computed once from all 82 targets
  (B) "fold-knot" LOOCV: for each fold, knots computed using only the
      81 training targets, then that exact basis applied to the held-
      out target -- the strict out-of-sample procedure

Requires patsy (pip install patsy --break-system-packages if missing).

Usage (run from backbones/rnabpflow/):
    python3 verify_loocv_knot_leakage.py dispersion_error_results_deduped.json
"""
import json
import sys

import numpy as np

try:
    from patsy import dmatrix
except ImportError:
    print("patsy not installed. Run: pip install patsy --break-system-packages")
    sys.exit(1)

from sklearn.linear_model import LinearRegression


def build_design(log_length, df=4, knots_source_idx=None):
    """Builds a restricted cubic spline (cr()) design matrix for
    log_length. If knots_source_idx is given, knot positions are
    computed using ONLY log_length[knots_source_idx] (the fold-safe
    version); otherwise knots use the full input array (the
    pooled/leaky version)."""
    if knots_source_idx is not None:
        # Fit the spline's knot structure using only the training subset,
        # then apply that exact transform to the full array (train + test).
        formula_data = {"x": log_length[knots_source_idx]}
        design_info = dmatrix(f"cr(x, df={df})", formula_data).design_info
        full = dmatrix(design_info, {"x": log_length})
        return np.asarray(full)
    else:
        full = dmatrix(f"cr(x, df={df})", {"x": log_length})
        return np.asarray(full)


def loocv_rmse(log_error, X0_cols_fn, X1_extra_fn, fold_safe):
    """Runs LOOCV for Model 0 (length-only) and Model 1 (length+dispersion).
    X0_cols_fn(train_idx) -> full-N design matrix for length spline
      (fold_safe: knots fit on train_idx only; else on all N)
    X1_extra_fn -> the additional dispersion column(s) to append (already
      full-length, e.g. log(dispersion), no refitting needed since it's
      not a learned transform)
    """
    n = len(log_error)
    resid0 = np.zeros(n)
    resid1 = np.zeros(n)

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        test_idx = np.array([i])

        if fold_safe:
            X0_full = X0_cols_fn(train_idx)
        else:
            X0_full = X0_cols_fn(None)  # knots from all N, computed once outside loop ideally,
                                         # but recomputed here per-fold for a fair apples-to-apples
                                         # timing comparison; the RESULT is identical to computing
                                         # it once outside since it doesn't depend on train_idx.

        X1_full = np.hstack([X0_full, X1_extra_fn])

        m0 = LinearRegression().fit(X0_full[train_idx], log_error[train_idx])
        pred0 = m0.predict(X0_full[test_idx])
        resid0[i] = log_error[i] - pred0[0]

        m1 = LinearRegression().fit(X1_full[train_idx], log_error[train_idx])
        pred1 = m1.predict(X1_full[test_idx])
        resid1[i] = log_error[i] - pred1[0]

    rmse0 = np.sqrt(np.mean(resid0**2))
    rmse1 = np.sqrt(np.mean(resid1**2))
    return rmse0, rmse1


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)

    length = np.array([d["length"] for d in data], dtype=float)
    dispersion = np.array([d["dispersion"] for d in data], dtype=float)
    error = np.array([d["error"] for d in data], dtype=float)

    log_length = np.log(length)
    log_dispersion = np.log(dispersion).reshape(-1, 1)
    log_error = np.log(error)

    n = len(length)
    print(f"N = {n}")
    print()

    print("=" * 70)
    print("(A) POOLED-KNOT LOOCV (knots from all N targets, computed once)")
    print("=" * 70)
    X0_pooled = build_design(log_length, df=4, knots_source_idx=None)

    def X0_fn_pooled(train_idx):
        return X0_pooled  # same matrix regardless of fold -- this is the leak

    rmse0_a, rmse1_a = loocv_rmse(log_error, X0_fn_pooled, log_dispersion, fold_safe=False)
    print(f"  Model 0 (length only) LOOCV RMSE: {rmse0_a:.4f}")
    print(f"  Model 1 (length+dispersion) LOOCV RMSE: {rmse1_a:.4f}")
    print(f"  Relative improvement: {(rmse0_a - rmse1_a) / rmse0_a * 100:.2f}%")
    print()

    print("=" * 70)
    print("(B) FOLD-SAFE LOOCV (knots refit per fold, using only the 81")
    print("    training targets, then applied to the held-out target)")
    print("=" * 70)

    def X0_fn_foldsafe(train_idx):
        return build_design(log_length, df=4, knots_source_idx=train_idx)

    rmse0_b, rmse1_b = loocv_rmse(log_error, X0_fn_foldsafe, log_dispersion, fold_safe=True)
    print(f"  Model 0 (length only) LOOCV RMSE: {rmse0_b:.4f}")
    print(f"  Model 1 (length+dispersion) LOOCV RMSE: {rmse1_b:.4f}")
    print(f"  Relative improvement: {(rmse0_b - rmse1_b) / rmse0_b * 100:.2f}%")
    print()

    print("=" * 70)
    print("COMPARISON WITH THE PRIMARY ANALYSIS (0.3609 / 0.3567)")
    print("=" * 70)
    print(f"  (A) Pooled-knot Model 0: diff = {rmse0_a - 0.3609:+.4f}")
    print(f"  (A) Pooled-knot Model 1: diff = {rmse1_a - 0.3567:+.4f}")
    print(f"  (B) Fold-safe  Model 0: diff = {rmse0_b - 0.3609:+.4f}")
    print(f"  (B) Fold-safe  Model 1: diff = {rmse1_b - 0.3567:+.4f}")


if __name__ == "__main__":
    main()
