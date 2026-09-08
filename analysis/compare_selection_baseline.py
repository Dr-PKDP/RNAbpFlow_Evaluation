"""
Compares selective-risk performance of dispersion-based ranking against
a naive length-based ranking baseline. An uncertainty signal's practical
value for prediction triage should be demonstrated against this
baseline directly, since sequence length is a free, zero-cost
alternative ranking criterion.

Usage:
    python compare_selection_baseline.py <results.json>

results.json: list of {"dispersion": float, "error": float, "length": int}
"""
import json
import sys
import numpy as np


def selective_curve(rank_by, errors, fractions):
    order = np.argsort(rank_by)
    n = len(rank_by)
    return [float(errors[order[:max(1, int(round(n * f)))]].mean()) for f in fractions]


def main() -> None:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    dispersions = np.array([d["dispersion"] for d in data])
    errors = np.array([d["error"] for d in data])
    lengths = np.array([d["length"] for d in data])

    fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    curve_dispersion = selective_curve(dispersions, errors, fractions)
    curve_length = selective_curve(lengths, errors, fractions)

    print(f"{'Keep %':<10}{'Dispersion-ranked':<20}{'Length-ranked':<20}")
    for f, d, l in zip(fractions, curve_dispersion, curve_length):
        print(f"{f*100:<10.0f}{d:<20.3f}{l:<20.3f}")

    reduction_dispersion = 1 - curve_dispersion[-1] / curve_dispersion[0]
    reduction_length = 1 - curve_length[-1] / curve_length[0]
    print(f"\nError reduction at 20% keep -- dispersion: {reduction_dispersion:.1%}, "
          f"length: {reduction_length:.1%}")


if __name__ == "__main__":
    main()
