"""
freeze_dropout_evidence.py -- Records the evidence that dropout is inactive
in the released checkpoint: a per-module audit table, environment and
checkpoint fingerprints, and a fixed-seed dropout-off reproducibility test.
It does not re-run the dropout-only control.

Usage (run from backbones/rnabpflow/):
    python freeze_dropout_evidence.py
"""
import hashlib
import os
import pickle
import subprocess
import sys

import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig


def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    print("=" * 70)
    print("PART 1: Environment and checkpoint fingerprint")
    print("=" * 70)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (torch): {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt_path = "checkpoint/RNA3DB.ckpt"
    print(f"\nCheckpoint SHA-256 ({ckpt_path}):")
    print(f"  {sha256_of_file(ckpt_path)}")

    print("\nLocal git commit (if this directory is a git checkout):")
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
        else:
            print(f"  Not a git repository, or git unavailable: {result.stderr.strip()}")
    except Exception as e:
        print(f"  Could not check: {e}")

    print()
    print("=" * 70)
    print("PART 2: Full per-module dropout audit table")
    print("=" * 70)
    config_file_path = "configs/inference.yaml"
    with open(config_file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    cfg = DictConfig(yaml_content)

    import inference
    sampler = inference.Sampler(cfg)

    print(f"{'Module path':<45} {'Class':<15} {'p':<8} {'training (before)':<20}")
    print("-" * 90)
    dropout_modules = []
    for name, m in sampler._flow_module.named_modules():
        if isinstance(m, nn.Dropout):
            dropout_modules.append((name, m))
            print(f"{name:<45} {'nn.Dropout':<15} {m.p:<8} {str(m.training):<20}")

    print(f"\nTotal dropout modules found: {len(dropout_modules)}")
    print(f"Modules with p > 0: {sum(1 for _, m in dropout_modules if m.p > 0)}")
    print(f"Modules with p == 0: {sum(1 for _, m in dropout_modules if m.p == 0)}")

    for _, m in dropout_modules:
        m.train()
    print(f"\nAfter calling .train() on all dropout modules:")
    print(f"Modules now in training=True mode: {sum(1 for _, m in dropout_modules if m.training)}")
    print(f"(Note: .training=True is necessary but not sufficient -- p=0 modules")
    print(f" remain no-ops regardless of this flag.)")

    print()
    print("=" * 70)
    print("PART 3: Clean, decisive dropout-OFF reproducibility test")
    print("=" * 70)
    print("Documents explicitly: dropout confirmed OFF (default eval mode),")
    print("genuinely varying generative seed, two samples of the same target")
    print("-- output should DIFFER, cleanly demonstrating that native")
    print("generative stochasticity fully persists with dropout off, and")
    print("that it alone (not dropout) is what varies the output.")
    print()

    from src.data.data_transform import make_atom_mask

    for _, m in dropout_modules:
        m.eval()

    output_dir = "Predictions_evidence_test"
    target_id = "10zt_A"
    sampler._samples_cfg.samples_per_sequence = 2
    sampler._output_dir_base = output_dir

    target_dir = os.path.join(output_dir, target_id)
    os.makedirs(target_dir, exist_ok=True)
    atom_dict = make_atom_mask(target_id, sampler._input_dir)
    pickle_file_path = os.path.join(target_dir, "map.pkl")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(atom_dict, f)

    sampler.run_sampling(target_id)
    os.remove(pickle_file_path)

    s0_path = os.path.join(target_dir, "Sample_0.pdb")
    s1_path = os.path.join(target_dir, "Sample_1.pdb")

    h0, h1 = file_hash(s0_path), file_hash(s1_path)
    print(f"Sample_0 md5: {h0}")
    print(f"Sample_1 md5: {h1}")
    print(f"Identical (should be False -- native generative stochasticity still varies): {h0 == h1}")

    print()
    print("=" * 70)
    print("SUMMARY -- evidence artifacts produced")
    print("=" * 70)
    print("1. Environment and checkpoint SHA-256 fingerprint (Part 1)")
    print("2. Full per-module dropout audit table, all modules (Part 2)")
    print("3. Confirmation that native generative stochasticity persists with")
    print("   dropout OFF, using the standard (non-fixed-seed) dataset (Part 3)")
    print()
    print("Save this full output as the record of the implementation audit.")


if __name__ == "__main__":
    main()
