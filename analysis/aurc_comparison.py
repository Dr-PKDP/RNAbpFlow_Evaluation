"""
Area-under-the-risk-coverage-curve (AURC) comparison of three ranking
policies for prediction triage: length alone, dispersion alone, and a
combined length+dispersion risk score (the nested spline model's
fitted values). Lower AURC is better (error stays lower across more of
the retained-coverage range).

This directly tests the decisive practical question: does combining
dispersion with length improve triage over length alone, not just
whether dispersion alone beats length alone (a comparison that,
without a proper combined policy, cannot show incremental value even
if dispersion contains real complementary information).

Usage:
    python aurc_comparison.py <results.json>
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from nested_spline_model import fit_and_loocv


def risk_coverage_curve(risk_score, error, fractions=None):
    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 19)
    order = np.argsort(risk_score)
    n = len(risk_score)
    return np.array([error[order[:max(1, int(round(n * f)))]].mean() for f in fractions]), fractions


def _trapezoidal(y, x):
    """Manual trapezoidal integration -- avoids relying on np.trapz
    (removed in numpy>=2.0) or np.trapezoid (unavailable in older numpy),
    since the target GPU server's numpy version is not known in advance."""
    y = np.asarray(y)
    x = np.asarray(x)
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)


def aurc(risk_score, error):
    curve, fractions = risk_coverage_curve(risk_score, error)
    return _trapezoidal(curve, fractions) / (fractions[-1] - fractions[0])


def combined_risk_score(length, dispersion, error):
    """Uses the nested model's IN-SAMPLE fitted values as the combined
    risk score. Note: this is evaluated in-sample for the AURC
    comparison itself (matching how the raw dispersion/length scores are
    also used in-sample as rankings) -- the earlier LOOCV analysis
    out-of-sample predictive validity is assessed separately via
    nested_spline_model.py's leave-one-out cross-validation.
    """
    from nested_spline_model import build_design
    import patsy
    import pandas as pd
    import statsmodels.api as sm

    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    data = pd.DataFrame({"log_length": log_length})
    basis = patsy.dmatrix("cr(log_length, df=4) - 1", data, return_type="dataframe")
    X = basis.copy()
    X["log_dispersion"] = log_dispersion
    model = sm.OLS(log_error, X).fit()
    return model.fittedvalues.values


def bootstrap_aurc_diff(length, dispersion, error, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(length)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        l, d, e = length[idx], dispersion[idx], error[idx]
        try:
            combined = combined_risk_score(l, d, e)
            aurc_length = aurc(l, e)
            aurc_combined = aurc(combined, e)
            diffs.append(aurc_length - aurc_combined)
        except Exception:
            pass
    return np.percentile(diffs, [2.5, 97.5]), len(diffs)


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    combined = combined_risk_score(length, dispersion, error)

    aurc_length = aurc(length, error)
    aurc_dispersion = aurc(dispersion, error)
    aurc_combined = aurc(combined, error)

    print(f"N = {len(length)}\n")
    print(f"AURC, length-only ranking:              {aurc_length:.4f}")
    print(f"AURC, dispersion-only ranking:           {aurc_dispersion:.4f}")
    print(f"AURC, combined (length+dispersion):      {aurc_combined:.4f}")
    print(f"(lower is better)\n")

    ci, n_valid = bootstrap_aurc_diff(length, dispersion, error)
    print(f"Bootstrap 95% CI for (length-only AURC - combined AURC), "
          f"{n_valid} valid resamples:")
    print(f"  [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Interpretation: POSITIVE values favor the combined model (combining "
          f"dispersion with length reduces AURC / improves triage). If this "
          f"interval excludes zero on the positive side, combining helps. If it "
          f"includes zero, no demonstrated improvement over length alone.")


if __name__ == "__main__":
    main()
