"""
Diagnoses the counterintuitive conformal interval-width pattern (shorter
RNAs receiving wider calibrated intervals than longer RNAs, despite
having smaller average error) by directly examining the distributions
that drive it: dispersion, error, and their ratio, by length stratum,
plus checking whether the pattern is driven by a small number of
outlier targets (reporting median/IQR alongside the mean).

Usage:
    python diagnose_conformal_width.py <results.json>
"""
import json
import sys

import numpy as np


def describe(name, values):
    print(f"  {name}: mean={np.mean(values):.3f}, median={np.median(values):.3f}, "
          f"IQR=[{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}], "
          f"min={np.min(values):.3f}, max={np.max(values):.3f}")


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])
    length = np.array([d["length"] for d in data])
    targets = [d.get("target", "?") for d in data]

    median_length = np.median(length)
    short_mask = length <= median_length
    long_mask = ~short_mask

    for name, mask in [("SHORT (<= median)", short_mask), ("LONG (> median)", long_mask)]:
        print(f"=== {name}, n={mask.sum()} ===")
        describe("dispersion", dispersion[mask])
        describe("error", error[mask])
        ratio = error[mask] / dispersion[mask]
        describe("error/dispersion ratio", ratio)

        # Real calibration quantile this stratum would produce (90% target,
        # same finite-sample correction as the actual conformal procedure)
        n_stratum = mask.sum()
        q_level = min(np.ceil((n_stratum // 2 + 1) * 0.9) / (n_stratum // 2), 1.0)
        q_hat = np.quantile(ratio, q_level)
        print(f"  Approx. calibration quantile (q_hat) at this stratum's full data: {q_hat:.3f}")
        implied_widths = q_hat * dispersion[mask]
        describe("implied interval width (q_hat * dispersion)", implied_widths)

        # Check for outlier-driven effects: does removing the single most
        # extreme ratio value change q_hat and mean width substantially?
        sorted_idx = np.argsort(ratio)
        trimmed_ratio = ratio[sorted_idx[:-1]] if len(ratio) > 5 else ratio
        trimmed_dispersion = dispersion[mask][sorted_idx[:-1]] if len(ratio) > 5 else dispersion[mask]
        q_hat_trimmed = np.quantile(trimmed_ratio, q_level)
        width_trimmed = (q_hat_trimmed * trimmed_dispersion).mean()
        print(f"  With single most extreme ratio value removed: q_hat={q_hat_trimmed:.3f}, "
              f"mean width={width_trimmed:.3f} (compare to {implied_widths.mean():.3f} with it included)")
        print()

    print("Interpretation:")
    print("  - If dispersion is genuinely much larger in one stratum, and the ratio")
    print("    distributions are otherwise similar, the width difference reflects real,")
    print("    expected behavior of a dispersion-scaled nonconformity score.")
    print("  - If the ratio distributions differ sharply (not just dispersion itself),")
    print("    or if removing one extreme point changes q_hat/width substantially,")
    print("    the reported mean width may be misleading and median/IQR should be")
    print("    reported instead.")


if __name__ == "__main__":
    main()
