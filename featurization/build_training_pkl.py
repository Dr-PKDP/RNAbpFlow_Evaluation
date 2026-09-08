"""
Builds a complete RNAbpFlow training feature file from a raw PDB
structure and chain ID, for use where the upstream training package's
own featurization code is unavailable.

Combines three components:
  - Atom-level geometry (positions, masks, residue typing), parsed
    directly from the structure file.
  - Base-pair features (`ss`, `onehot`, `custom_ss_mask`), derived from
    FR3D structural annotation.
  - Monomer identity fields (chain/entity/symmetry indices), which are
    fixed constants for single-chain assemblies.

Usage:
    python build_training_pkl.py --pdb-id 1ABC --chain A \\
        --pdb-name 1abc_A --out output/1abc_A.pkl

Requires FR3D-python (github.com/BGSU-RNA/fr3d-python) installed and
importable via the path in FR3D_SCRIPT below, and this project's
training package (`nucleotide_constants`, `vocabulary`, and dataset
utilities) importable from the working directory.
"""
from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

sys.path.insert(0, ".")
from src.data import nucleotide_constants as nc  # noqa: E402
from src.data import vocabulary as vocab  # noqa: E402
from src.data.utils import chain_str_to_int  # noqa: E402

FR3D_SCRIPT = Path("/opt/fr3d-python/fr3d/classifiers/NA_pairwise_interactions.py")
FR3D_DIR = FR3D_SCRIPT.parent.parent.parent
UV_BIN = "uv"  # or the path to a specific uv/python binary if not on PATH

# Nucleic acid types occupy a combined protein+nucleic-acid vocabulary
# (src/data/vocabulary.py), not the narrower 8-symbol table in
# nucleotide_constants.py: lowercase letters, offset by
# protein_restype_num_with_x.
ONEHOT_ORDER = {"A": 0, "U": 1, "G": 2, "C": 3}
_NA_LETTER_BY_RESNAME = {
    "DA": "a", "DC": "c", "DG": "g", "DT": "t",
    "A": "a", "C": "c", "G": "g", "U": "u",
}
UNSUPPORTED_RESIDUE_MSG = (
    "Residue name not in the standard 8-type table. This is likely a "
    "modified nucleotide (e.g. pseudouridine, inosine, dihydrouridine), "
    "which this pipeline does not support. Exclude this structure from "
    "the candidate pool rather than substitute a placeholder value."
)


def download_pdb(pdb_id: str, dest: Path, max_retries: int = 3) -> Path:
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    last_exc = None
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc


def parse_geometry(pdb_file: Path, chain_id: str) -> tuple[dict, list, str]:
    """Returns (feature_dict, residues, one_letter_sequence).

    Raises ValueError listing any non-standard residues found, rather
    than silently dropping or placeholder-filling them.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_file))
    models = list(structure.get_models())
    if len(models) == 0:
        raise ValueError("No models found in structure.")
    if len(models) > 1:
        # Multi-model files (most commonly NMR ensembles) are treated as
        # single structures by using the first model as the
        # representative conformer -- standard practice when a single
        # structure is required from a multi-model file. Projects that
        # exclude NMR structures upstream should not encounter this path
        # for NMR entries.
        print(f"  {len(models)} models found; using model 1.")
    chain = models[0][chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    n_res = len(residues)

    bad = [(i, r.resname.strip()) for i, r in enumerate(residues)
           if r.resname.strip() not in nc.restype_order]
    if bad:
        raise ValueError(f"{len(bad)} non-standard residues found: {bad[:10]}"
                          f"{'...' if len(bad) > 10 else ''}\n{UNSUPPORTED_RESIDUE_MSG}")

    atom_positions = np.zeros((n_res, nc.compact_atom_type_num, 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, nc.compact_atom_type_num), dtype=np.float32)
    atom_b_factors = np.zeros((n_res, nc.compact_atom_type_num), dtype=np.float32)
    aatype = np.zeros((n_res,), dtype=np.int64)
    atom_deoxy = np.zeros((n_res,), dtype=bool)
    sequence_chars = []

    for i, res in enumerate(residues):
        resname = res.resname.strip()
        letter = _NA_LETTER_BY_RESNAME[resname]
        aatype[i] = vocab.restype_order[letter]
        atom_deoxy[i] = resname.startswith("D")
        sequence_chars.append(letter.upper())

        compact_names = nc.restype_name_to_compact_atom_names[resname]
        for slot_idx, atom_name in enumerate(compact_names):
            if not atom_name:
                continue
            if atom_name in res:
                atom = res[atom_name]
                atom_positions[i, slot_idx] = atom.coord
                atom_mask[i, slot_idx] = 1.0
                atom_b_factors[i, slot_idx] = atom.bfactor

    # Backbone reference slot: fixed compact-array index 3, matching the
    # index the training pipeline's own centering/rigid-frame logic
    # applies to this array (that logic indexes by the full 27-atom
    # table's C4' position, which coincides with compact index 3 in this
    # codebase's atom ordering convention).
    c4_slot = np.full((n_res,), 3, dtype=np.int64)
    row_idx = np.arange(n_res)
    bb_mask = atom_mask[row_idx, c4_slot]
    bb_pos_raw = atom_positions[row_idx, c4_slot]
    bb_center = bb_pos_raw.sum(axis=0) / (bb_mask.sum() + 1e-5)

    real_atom_mask = atom_mask.astype(bool)
    atom_positions = atom_positions - bb_center
    atom_positions[~real_atom_mask] = 0.0
    bb_positions = atom_positions[row_idx, c4_slot]

    molecule_type_encoding = np.zeros((n_res, 4), dtype=np.float32)
    is_rna = ~atom_deoxy
    molecule_type_encoding[is_rna, 2] = 1.0
    molecule_type_encoding[atom_deoxy, 1] = 1.0

    feats = {
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "atom_b_factors": atom_b_factors,
        "aatype": aatype,
        "atom_deoxy": atom_deoxy,
        "bb_mask": bb_mask,
        "bb_positions": bb_positions,
        "molecule_type_encoding": molecule_type_encoding,
    }
    return feats, residues, "".join(sequence_chars)


def run_fr3d(pdb_id: str, max_retries: int = 3) -> Path:
    last_error = None
    for attempt in range(max_retries):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            result = subprocess.run(
                [UV_BIN, "run", "python", "fr3d/classifiers/NA_pairwise_interactions.py",
                 "-i", str(input_dir), "-o", str(output_dir),
                 "-c", "basepair_detail", pdb_id],
                capture_output=True, text=True, cwd=str(FR3D_DIR),
            )
            if result.returncode == 0:
                bp_file = output_dir / f"{pdb_id}_basepair_detail.txt"
                content = bp_file.read_text()
                persistent = Path(tempfile.mktemp(suffix=".txt"))
                persistent.write_text(content)
                return persistent

            last_error = (
                f"FR3D failed for {pdb_id} (exit {result.returncode}), "
                f"attempt {attempt+1}/{max_retries}.\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(last_error)


def build_ss_onehot(bp_file: Path, sequence: str, first_resnum: int) -> dict:
    n = len(sequence)

    def build_matrix(type_filter):
        mat = np.zeros((n, n), dtype=np.int64)
        with open(bp_file) as f:
            for line in f:
                parts = re.split(r"\s+", line.strip())
                if len(parts) < 3:
                    continue
                res1, itype, res2 = parts[0], parts[1], parts[2]
                if type_filter is not None and itype not in type_filter:
                    continue
                try:
                    i = int(res1.split("|")[-1]) - first_resnum
                    j = int(res2.split("|")[-1]) - first_resnum
                except (ValueError, IndexError):
                    continue
                if 0 <= i < n and 0 <= j < n and i != j:
                    mat[i, j] = 1
                    mat[j, i] = 1
        return mat

    canonical = build_matrix({"cWW"})
    # The second and third channels use the combined canonical +
    # non-canonical pairing set. This is an approximation to the
    # original training pipeline's channel definitions: it exactly
    # reproduces canonical pairing (channel 0) but does not fully
    # recover the original non-canonical channel distinction, which
    # could not be determined from the released training artifacts.
    # See featurization/README.md for the empirical precision/recall
    # characteristics of this approximation.
    all_pairs = build_matrix(None)
    ss = np.stack([canonical, all_pairs, all_pairs], axis=-1)

    onehot = np.zeros((n, 4), dtype=np.int32)
    for i, base in enumerate(sequence.upper()):
        if base in ONEHOT_ORDER:
            onehot[i, ONEHOT_ORDER[base]] = 1

    return {"ss": ss, "onehot": onehot, "custom_ss_mask": ss.copy()}


def build_identity_fields(n_res: int, chain_id: str) -> dict:
    """Fixed for single-chain, single-entity monomer assemblies."""
    return {
        "atom_chain_id_mask": np.ones((n_res,), dtype=np.int64),
        "atom_chain_indices": np.full((n_res,), chain_str_to_int(chain_id), dtype=np.int64),
        "asym_id": np.ones((n_res,), dtype=np.float64),
        "sym_id": np.ones((n_res,), dtype=np.float64),
        "entity_id": np.ones((n_res,), dtype=np.float64),
        "modeled_idx": np.arange(n_res, dtype=np.int64),
        "na_modeled_idx": np.arange(n_res, dtype=np.int64),
        "inter_chain_interacting_idx": np.array([], dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-id", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--pdb-name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pdb_file = download_pdb(args.pdb_id, Path(f"/tmp/{args.pdb_id}.pdb"))
    geom_feats, residues, sequence = parse_geometry(pdb_file, args.chain)

    first_resnum = residues[0].id[1]
    bp_file = run_fr3d(args.pdb_id)
    ss_feats = build_ss_onehot(bp_file, sequence, first_resnum)
    id_feats = build_identity_fields(len(residues), args.chain)

    full = {"pdb_name": args.pdb_name, **geom_feats, **ss_feats, **id_feats}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(full, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
