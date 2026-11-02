"""
compare_dropout_variance.py -- Computes pairwise backbone RMSD within a
set of sample .pdb files, to quantitatively compare dropout-on vs
dropout-off dispersion for the same target.

Usage:
    python compare_dropout_variance.py <directory-of-pdb-files>
"""
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


def get_backbone_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    coords = []
    for atom in structure.get_atoms():
        if atom.get_name() == "C4'":
            coords.append(atom.coord)
    return np.array(coords)


def kabsch_align(mobile, target):
    """Superposes `mobile` onto `target` via the Kabsch algorithm (optimal
    rotation minimizing RMSD), removing arbitrary global rotation/
    translation differences before comparing shape. Without this, RMSD is
    dominated by where each sample happens to sit in space, not genuine
    conformational differences."""
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_c = mobile - mobile_center
    target_c = target - target_center
    H = mobile_c.T @ target_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    aligned = (R @ mobile_c.T).T + target_center
    return aligned


def rmsd(a, b):
    if a.shape != b.shape:
        return None
    a_aligned = kabsch_align(a, b)
    return np.sqrt(np.mean(np.sum((a_aligned - b) ** 2, axis=1)))


directory = Path(sys.argv[1])
pdb_files = sorted(directory.glob("Sample_*.pdb"))
print(f"Found {len(pdb_files)} sample files in {directory}")

coords_list = [get_backbone_coords(p) for p in pdb_files]

pairwise = []
for i in range(len(coords_list)):
    for j in range(i + 1, len(coords_list)):
        r = rmsd(coords_list[i], coords_list[j])
        if r is not None:
            pairwise.append(r)

if pairwise:
    print(f"Pairwise C4' RMSD across {len(pairwise)} pairs:")
    print(f"  mean: {np.mean(pairwise):.3f} A")
    print(f"  std:  {np.std(pairwise):.3f} A")
    print(f"  min:  {np.min(pairwise):.3f} A")
    print(f"  max:  {np.max(pairwise):.3f} A")
else:
    print("No valid pairwise comparisons (shape mismatches?).")
