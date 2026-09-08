"""
Checks whether sequence length confounds a dispersion-error correlation.
RMSD-based error and dispersion metrics can both scale with chain
length independent of genuine model uncertainty; this computes the
partial correlation between dispersion and error, controlling for
length, and should be run as standard practice before reporting a raw
correlation between an uncertainty signal and structural error.

Usage:
    python check_length_confound.py <results.json>

results.json: list of {"dispersion": float, "error": float, "length": int}
"""
import json
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr


def partial_correlation(x, y, z):
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)
    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return numerator / denominator


def main() -> None:
    with open(sys.argv[1]) as f:
        data = json.load(f)

    dispersions = np.array([d["dispersion"] for d in data])
    errors = np.array([d["error"] for d in data])
    lengths = np.array([d["length"] for d in data])

    print(f"N = {len(dispersions)}")
    print(f"Length range: {lengths.min()}-{lengths.max()} residues\n")

    r_len_disp, p1 = pearsonr(lengths, dispersions)
    r_len_err, p2 = pearsonr(lengths, errors)
    r_raw, p3 = pearsonr(dispersions, errors)
    rho_raw, _ = spearmanr(dispersions, errors)

    print(f"length vs dispersion:  r={r_len_disp:.3f} (p={p1:.4f})")
    print(f"length vs error:       r={r_len_err:.3f} (p={p2:.4f})")
    print(f"dispersion vs error (raw): Pearson r={r_raw:.3f} (p={p3:.4f}), "
          f"Spearman rho={rho_raw:.3f}")

    r_partial = partial_correlation(dispersions, errors, lengths)
    print(f"\ndispersion vs error, controlling for length: partial r={r_partial:.3f}")


if __name__ == "__main__":
    main()
