"""
extract_paired_diffs.py -- Computes the per-target paired LOOCV
differences d_i = e0i^2 - e1i^2 (Model 0 vs Model 1 squared out-of-fold
residuals), the actual underlying quantity whose mean and bootstrap
interval are already reported in the manuscript (Equation 3). This is
needed to build a distribution plot for Figure 4 (showing the real
spread of d_i across targets, not just two summary bars), rather than
guessing at what the distribution looks like.

Uses the SAME 4-df restricted cubic spline / LOOCV procedure already
verified against the manuscript's reported values (0.3609 / 0.3567),
so the output here is guaranteed consistent with the primary result.

Usage (run from backbones/rnabpflow/):
    python3 extract_paired_diffs.py dispersion_error_results_deduped.json
"""
import json
import sys

import numpy as np
from patsy import dmatrix
from sklearn.linear_model import LinearRegression


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)

    length = np.array([d["length"] for d in data], dtype=float)
    dispersion = np.array([d["dispersion"] for d in data], dtype=float)
    error = np.array([d["error"] for d in data], dtype=float)
    targets = [d.get("target", f"t{i}") for i, d in enumerate(data)]

    log_length = np.log(length)
    log_dispersion = np.log(dispersion).reshape(-1, 1)
    log_error = np.log(error)

    n = len(length)
    X0 = np.asarray(dmatrix("cr(x, df=4)", {"x": log_length}))
    X1 = np.hstack([X0, log_dispersion])

    e0 = np.zeros(n)
    e1 = np.zeros(n)
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        test_idx = np.array([i])

        m0 = LinearRegression().fit(X0[train_idx], log_error[train_idx])
        e0[i] = log_error[i] - m0.predict(X0[test_idx])[0]

        m1 = LinearRegression().fit(X1[train_idx], log_error[train_idx])
        e1[i] = log_error[i] - m1.predict(X1[test_idx])[0]

    d = e0**2 - e1**2  # Equation 3 exactly

    rmse0 = np.sqrt(np.mean(e0**2))
    rmse1 = np.sqrt(np.mean(e1**2))
    print(f"Sanity check -- should match manuscript's 0.3609 / 0.3567:")
    print(f"  Model 0 LOOCV RMSE: {rmse0:.4f}")
    print(f"  Model 1 LOOCV RMSE: {rmse1:.4f}")
    print(f"  Mean d_i: {d.mean():.4f} (should match ~0.0031)")
    print()

    print("Per-target paired differences d_i = e0i^2 - e1i^2:")
    print(json.dumps([{"target": t, "d": float(di)} for t, di in zip(targets, d)]))

    # Save to a file too, for direct use in regenerating Figure 4
    out = [{"target": t, "d": float(di), "e0": float(e0i), "e1": float(e1i)}
           for t, di, e0i, e1i in zip(targets, d, e0, e1)]
    with open("paired_diffs_for_figure4.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved full detail to paired_diffs_for_figure4.json")


if __name__ == "__main__":
    main()
