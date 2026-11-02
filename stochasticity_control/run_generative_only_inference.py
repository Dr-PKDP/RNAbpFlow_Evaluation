"""
run_generative_only_inference.py -- Condition A of the stochasticity-source
control experiment: runs RNAbpFlow inference with dropout left in its
normal eval-mode (OFF) state, so any sample-to-sample variation comes
only from the model's own generative sampling process (the flow-matching
initial noise/frame), not from dropout.

Compare against the existing MC-dropout-enabled predictions (Condition B,
already generated in Predictions/) to isolate how much of the observed
dispersion-error relationship depends on dropout specifically versus the
generative process alone.

Requires the same seedval=-1 patch already applied to
src/data/dataset.py (RNADataset.__init__) for the primary MC-dropout run
-- that patch disables deterministic per-sample seeding regardless of
dropout state, which is exactly what makes this condition meaningful:
without it, every sample_id would produce an identical structure and
there would be no generative dispersion to measure at all.

This script deliberately does NOT call enable_mc_dropout() -- that is
the only difference from run_mc_dropout_inference.py. Every other code
path (Sampler construction, list.txt/config parsing, atom-mask handling)
is identical and already validated.

Writes to a NEW output directory only (Predictions_dropout_off/) --
Predictions/ (the existing MC-dropout condition) is never opened for
writing and cannot be overwritten by this script.

Usage (run from backbones/rnabpflow/, same conventions as
run_mc_dropout_inference.py):
    python run_generative_only_inference.py
"""
import os
import pickle

import yaml
from omegaconf import DictConfig

from inference import Sampler
from src.data.data_transform import make_atom_mask


def main():
    CONFIG_FILE_PATH = "configs/inference.yaml"
    with open(CONFIG_FILE_PATH, "r") as f:
        yaml_content = yaml.safe_load(f)

    input_dir = yaml_content["inference"]["input_dir"]
    # Fixed, distinct output directory -- override whatever the config
    # says, so this can never write into Predictions/ (the dropout-ON
    # condition) even if the config file is reused unmodified.
    output_dir = "Predictions_dropout_off"
    yaml_content["inference"]["output_dir"] = output_dir
    cfg = DictConfig(yaml_content)

    sampler = Sampler(cfg)
    # Deliberately no enable_mc_dropout() call here. Sampler's own
    # construction already leaves the model in eval() mode, so all
    # nn.Dropout submodules stay OFF -- this is the entire point of
    # this condition.
    print("Dropout left in normal eval-mode (OFF). Only the model's "
          "own generative sampling stochasticity will vary across "
          "samples for the same target.")

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

    print(f"Running Condition A (generative-only) on {len(id_list)} targets, "
          f"writing to {output_dir}/")

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

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(id_list)} targets complete")

    print("Condition A (generative-only, dropout OFF) inference complete.")


if __name__ == "__main__":
    main()
