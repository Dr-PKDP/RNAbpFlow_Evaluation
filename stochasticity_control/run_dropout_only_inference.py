"""
run_dropout_only_inference.py -- Condition C: dropout ON, generative
initial frame FIXED (identical across all samples for a given target).
Isolates dropout-specific variation conditional on the same generative
starting point, directly answering whether dropout contributes
meaningfully once generative-sampling randomness is held constant.

REQUIRES patch_interpolant_scoped_seed.py to be applied FIRST. Without
it, a fixed seedval would also freeze dropout (both draw from the same
global RNG), silently producing byte-identical output regardless of
dropout state.

HOW THIS WORKS, without modifying inference.py or dataset.py on disk:
inference.py does `from src.data.dataset import RNADataset` at module
level, and Sampler.run_sampling() later calls `RNADataset(...)` by that
name. Python resolves that name from inference.py's own module
namespace at CALL time, not at class-definition time -- so replacing
`inference.RNADataset` with a different class from this script causes
every subsequent sampler.run_sampling() call to use that replacement,
with zero changes to any existing file. This keeps Conditions A and B
completely unaffected; this script only ever runs in its own process.

The replacement dataset (RNADatasetFixedSeed) is a subclass of the
existing, already-validated RNADataset, reusing its load_bp_maps /
__len__ / __getitem__ unchanged -- __init__ is overridden only to
assign one fixed seedval to every sample_id for a target, instead of
the -1 (fully fresh) value Conditions A/B use.

Writes to a NEW output directory only (Predictions_dropout_only_fixedseed/)
-- structurally cannot touch Predictions/ or Predictions_dropout_off/.

Usage (run from backbones/rnabpflow/, same conventions as the other
inference runners):
    python run_dropout_only_inference.py [--n-samples N] [--targets-file FILE]

By default uses the same list.txt as the primary runs. --n-samples lets
you override the sample count for a fast validation pass (e.g. 3)
before committing to the full 20-sample, full-cohort run.
"""
import argparse
import os
import pickle

import torch.nn as nn
import yaml
from omegaconf import DictConfig

import inference
from src.data.dataset import RNADataset as OriginalRNADataset
from src.data import utils as du
from src.data.data_transform import make_atom_mask

FIXED_SEED = 777  # arbitrary but fixed; different targets still get
                   # different noise because tensor shape depends on
                   # each target's own length, even at the same seed


class RNADatasetFixedSeed(OriginalRNADataset):
    """Every sample_id for a given target shares FIXED_SEED, instead of
    the varying-then-disabled seed the primary RNADataset uses. Combined
    with the scoped-reseed patch, this pins only the initial generative
    noise/frame -- dropout draws later in the same call remain genuinely
    random per sample."""

    def __init__(self, samples_cfg, dirpath, pdbID, input_dir):
        self._samples_cfg = samples_cfg
        self.pdbID = pdbID
        self.dirpath = dirpath
        self.input_dir = input_dir
        mapfeat = du.read_pkl(f"{self.dirpath}/{self.pdbID}/map.pkl")
        onehot = mapfeat["onehot"]
        L = onehot.shape[0]
        ss_arr = self.load_bp_maps(f"{self.input_dir}/{self.pdbID}", L)
        all_sample_ids = []
        for sample_id in range(self._samples_cfg.samples_per_sequence):
            seedval = FIXED_SEED  # same value for every sample_id -> same
                                    # generative frame every time for this target
            all_sample_ids.append((sample_id, ss_arr, seedval, onehot, mapfeat))
        self._all_sample_ids = all_sample_ids


def enable_mc_dropout(model: nn.Module) -> int:
    """Identical to the validated function in run_mc_dropout_inference.py."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=None,
                     help="Override samples_per_sequence for all targets "
                          "(e.g. 3, for a fast validation pass)")
    ap.add_argument("--targets-file", default=None,
                     help="Override which list.txt-style file to read "
                          "targets from (defaults to input_dir/list.txt)")
    args = ap.parse_args()

    # Monkey-patch: from this point on, any RNADataset(...) call made
    # inside inference.py's Sampler.run_sampling() resolves to our
    # fixed-seed subclass instead of the original.
    inference.RNADataset = RNADatasetFixedSeed

    CONFIG_FILE_PATH = "configs/inference.yaml"
    with open(CONFIG_FILE_PATH, "r") as f:
        yaml_content = yaml.safe_load(f)

    input_dir = yaml_content["inference"]["input_dir"]
    output_dir = "Predictions_dropout_only_fixedseed"
    yaml_content["inference"]["output_dir"] = output_dir
    cfg = DictConfig(yaml_content)

    sampler = inference.Sampler(cfg)

    n_dropout = enable_mc_dropout(sampler._flow_module)
    print(f"Enabled {n_dropout} dropout modules for Condition C.")
    if n_dropout == 0:
        print("WARNING: zero dropout modules found -- stopping rather than "
              "proceeding on a silently broken assumption.")
        return

    list_file_path = args.targets_file or os.path.join(input_dir, "list.txt")
    with open(list_file_path, "r") as f:
        lines = f.readlines()

    id_list = []
    sample_count_list = []
    for line in lines:
        tokens = line.strip().split()
        id_list.append(tokens[0])
        if len(tokens) == 2:
            sample_count_list.append(int(tokens[1]))

    print(f"Running Condition C (dropout ON, fixed generative frame) on "
          f"{len(id_list)} targets, writing to {output_dir}/")

    for idx, target_id in enumerate(id_list):
        if args.n_samples is not None:
            sampler._samples_cfg.samples_per_sequence = args.n_samples
        elif len(sample_count_list) == len(id_list):
            sampler._samples_cfg.samples_per_sequence = sample_count_list[idx]

        target_dir = os.path.join(output_dir, target_id)
        os.makedirs(target_dir, exist_ok=True)

        atom_dict = make_atom_mask(target_id, input_dir)
        pickle_file_path = os.path.join(target_dir, "map.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(atom_dict, f)

        sampler.run_sampling(target_id)

        os.remove(pickle_file_path)

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(id_list)} targets complete")

    print("Condition C (dropout ON, fixed generative frame) inference complete.")


if __name__ == "__main__":
    main()
