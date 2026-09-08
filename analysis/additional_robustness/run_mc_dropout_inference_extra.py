"""
Generates additional MC-dropout samples per target beyond the primary
run's output, for a convergence check against a larger pooled sample.
Same model-loading and sampling logic as run_mc_dropout_inference.py,
with a configurable list file and output directory.

Usage:
    python run_mc_dropout_inference_extra.py \
        --list-file extra10_list.txt \
        --output-dir Predictions_extra10
"""
import argparse
import os
import pickle

import torch.nn as nn
import yaml
from omegaconf import DictConfig

from inference import Sampler
from src.data.data_transform import make_atom_mask


def enable_mc_dropout(model: nn.Module) -> int:
    """Walks every submodule and sets nn.Dropout instances to train() mode
    specifically, leaving everything else (already in eval() mode from
    Sampler's own construction) untouched. Returns the count found, so we
    can confirm this actually did something rather than silently finding
    zero and producing no real variance at all."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml",
                         help="Same config as the original script -- input_dir is "
                              "read from here and is NOT overridden, since the "
                              "existing input_dir already has everything "
                              "make_atom_mask() needs for these targets.")
    parser.add_argument("--list-file", default=None,
                         help="Path to a list.txt-format file (target_id, optional "
                              "sample count per line). If not given, falls back to "
                              "input_dir/list.txt exactly as the original script did.")
    parser.add_argument("--output-dir", default=None,
                         help="If not given, falls back to the config's output_dir "
                              "exactly as the original script did -- pass this "
                              "explicitly to avoid ever writing into Predictions/.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_content = yaml.safe_load(f)

    input_dir = yaml_content["inference"]["input_dir"]
    if args.output_dir:
        # Override inside yaml_content itself, before building cfg: the
        # Sampler's internal RNADataset reads map.pkl back using its own
        # _output_dir_base, set at Sampler(cfg) construction time from
        # this config value. A local variable override would not affect
        # what the Sampler reads from.
        yaml_content["inference"]["output_dir"] = args.output_dir
    output_dir = yaml_content["inference"]["output_dir"]
    cfg = DictConfig(yaml_content)

    sampler = Sampler(cfg)

    n_dropout = enable_mc_dropout(sampler._flow_module)
    print(f"Enabled {n_dropout} dropout modules for MC-dropout inference.")
    if n_dropout == 0:
        print("WARNING: zero dropout modules found -- stopping rather than "
              "proceeding on a silently broken assumption.")
        return

    list_file_path = args.list_file or os.path.join(input_dir, "list.txt")
    print(f"Reading target list from: {list_file_path}")
    print(f"Writing output to: {output_dir}")
    with open(list_file_path, "r") as f:
        lines = f.readlines()

    id_list = []
    sample_count_list = []
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        id_list.append(tokens[0])
        if len(tokens) == 2:
            sample_count_list.append(int(tokens[1]))

    print(f"{len(id_list)} targets to process.")

    for idx, target_id in enumerate(id_list):
        if len(sample_count_list) == len(id_list):
            sampler._samples_cfg.samples_per_sequence = sample_count_list[idx]

        target_dir = os.path.join(output_dir, target_id)
        os.makedirs(target_dir, exist_ok=True)

        atom_dict = make_atom_mask(target_id, input_dir)
        pickle_file_path = os.path.join(target_dir, "map.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(atom_dict, f)

        print(f"  [{idx+1}/{len(id_list)}] sampling {target_id} "
              f"({sampler._samples_cfg.samples_per_sequence} samples)...")
        sampler.run_sampling(target_id)

        os.remove(pickle_file_path)

    print("Done.")


if __name__ == "__main__":
    main()
