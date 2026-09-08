"""
Runs inference with dropout layers kept active, for Monte Carlo
dropout-based uncertainty estimation. Reuses the standard inference
pipeline's model loading and sampling logic; the addition is switching
dropout modules to train mode after the model is otherwise placed in
eval mode.

Requires per-sample stochasticity in the underlying sampler: if
per-sample seeding is enabled for output reproducibility, disable it
(sentinel value -1, if supported) so repeated calls produce independent
samples.

Usage (run from the inference package's root directory):
    python run_mc_dropout_inference.py
"""
import os
import pickle

import torch.nn as nn
import yaml
from omegaconf import DictConfig

from inference import Sampler
from src.data.data_transform import make_atom_mask


def enable_mc_dropout(model: nn.Module) -> int:
    """Sets nn.Dropout submodules to train() mode, leaving all other
    modules in eval() mode. Returns the count enabled, so a
    misconfigured architecture (zero dropout modules) is detected
    explicitly rather than silently producing no additional variance."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            count += 1
    return count


def main() -> None:
    config_file_path = "configs/inference.yaml"
    with open(config_file_path, "r") as f:
        yaml_content = yaml.safe_load(f)

    input_dir = yaml_content["inference"]["input_dir"]
    output_dir = yaml_content["inference"]["output_dir"]
    cfg = DictConfig(yaml_content)

    sampler = Sampler(cfg)

    n_dropout = enable_mc_dropout(sampler._flow_module)
    print(f"Enabled {n_dropout} dropout modules for MC-dropout inference.")
    if n_dropout == 0:
        print("No dropout modules found in this architecture; MC-dropout "
              "would introduce no additional variance. Aborting.")
        return

    list_file_path = os.path.join(input_dir, "list.txt")
    with open(list_file_path, "r") as f:
        lines = f.readlines()

    id_list = []
    sample_count_list = []
    for line in lines:
        tokens = line.strip().split()
        id_list.append(tokens[0])
        if len(tokens) == 2:
            sample_count_list.append(int(tokens[1]))

    for idx, target_id in enumerate(id_list):
        if len(sample_count_list) == len(id_list):
            sampler._samples_cfg.samples_per_sequence = sample_count_list[idx]

        target_dir = os.path.join(output_dir, target_id)
        os.makedirs(target_dir, exist_ok=True)

        atom_dict = make_atom_mask(target_id, input_dir)
        pickle_file_path = os.path.join(target_dir, "map.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(atom_dict, f)

        sampler.run_sampling(target_id)

        os.remove(pickle_file_path)


if __name__ == "__main__":
    main()
