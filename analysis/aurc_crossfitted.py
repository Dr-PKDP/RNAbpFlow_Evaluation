"""
Cross-fitted AURC comparison: the combined length+dispersion ranking
score is generated via leave-one-out out-of-fold prediction (each
target's combined score comes from a model that did not see that
target's own outcome), avoiding the optimistic bias of an in-sample
fitted score.

Usage:
    python aurc_crossfitted.py <results.json>
"""
import json
import sys

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from nested_spline_model import build_design
from aurc_comparison import aurc, _trapezoidal, risk_coverage_curve


def cross_fitted_combined_score(length, dispersion, error):
    """Leave-one-out out-of-fold combined risk score: for each target,
    fit Model 1 on the other 81 targets, predict this target's score."""
    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)
    n = len(length)

    full_data = pd.DataFrame({"log_length": log_length})
    basis_info = patsy.dmatrix("cr(log_length, df=4) - 1", full_data,
                                return_type="dataframe").design_info

    oof_scores = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train = build_design(log_length[train_idx], log_dispersion[train_idx],
                                4, True, False, basis_info)
        y_train = log_error[train_idx]
        model = sm.OLS(y_train, X_train).fit()

        X_test = build_design(log_length[[i]], log_dispersion[[i]],
                               4, True, False, basis_info)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        oof_scores[i] = model.predict(X_test)[0]

    return oof_scores


def bootstrap_aurc_diff_crossfitted(length, dispersion, error, combined_scores,
                                      n_boot=2000, seed=0):
    """Bootstraps the AURC difference using the ALREADY-COMPUTED
    cross-fitted combined scores (resampling targets, not refitting the
    cross-fitting procedure itself inside the bootstrap -- refitting
    leave-one-out models inside a bootstrap loop would be extremely
    expensive; resampling the already-valid out-of-fold scores is the
    standard, computationally tractable approach)."""
    rng = np.random.default_rng(seed)
    n = len(length)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        l, e, c = length[idx], error[idx], combined_scores[idx]
        aurc_length = aurc(l, e)
        aurc_combined = aurc(c, e)
        diffs.append(aurc_length - aurc_combined)
    return np.percentile(diffs, [2.5, 97.5])


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    combined_oof = cross_fitted_combined_score(length, dispersion, error)

    aurc_length = aurc(length, error)
    aurc_dispersion = aurc(dispersion, error)
    aurc_combined_oof = aurc(combined_oof, error)

    print(f"N = {len(length)}\n")
    print(f"AURC, length-only ranking:                    {aurc_length:.4f}")
    print(f"AURC, dispersion-only ranking:                 {aurc_dispersion:.4f}")
    print(f"AURC, combined (CROSS-FITTED, out-of-fold):    {aurc_combined_oof:.4f}")
    print(f"(lower is better)\n")

    ci = bootstrap_aurc_diff_crossfitted(length, dispersion, error, combined_oof)
    print(f"Bootstrap 95% CI for (length-only AURC - cross-fitted combined AURC):")
    print(f"  [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Interpretation: POSITIVE values favor the combined model. This is now "
          f"a fair, non-optimistic comparison (no target's own outcome informed its "
          f"own combined score).")


if __name__ == "__main__":
    main()
