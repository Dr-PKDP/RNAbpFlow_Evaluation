# Required Patches to Upstream Code

Two modifications to the upstream RNAbpFlow training/inference code are
required for correct operation of this pipeline. Apply both before
running training or MC-dropout inference.

## 1. Training data loader: missing seed parameter

**File**: `src/data/pdb_na_datamodule_base.py`, `train_dataloader` method.

The training length-batcher is constructed without a seed argument,
causing it to fall back to its class default on every run regardless of
configuration. This means every training run uses identical data
ordering, which defeats the purpose of training multiple independent
model instances for variance estimation.

**Fix**: pass a configurable seed through to the batcher.

```python
# Before:
lb = RNALengthBatcher(
        sampler_cfg=self.data_cfg,
        metadata_csv=self.data_train.csv,
        rank=rank,
        num_replicas=num_replicas
    )

# After:
run_seed = getattr(self.data_cfg, "seed", 123)
lb = RNALengthBatcher(
        sampler_cfg=self.data_cfg,
        metadata_csv=self.data_train.csv,
        rank=rank,
        num_replicas=num_replicas,
        seed=run_seed,
    )
```

Set a distinct seed per training run via the Hydra override
`+data_cfg.seed=<value>`.

## 2. Inference dataset: deterministic per-sample seeding

**File**: `src/data/dataset.py`, `RNADataset.__init__`.

Each generated sample is seeded deterministically as a pure function of
its sample index (`seedval = start_seed * (sample_id + 1) * 10`, with a
hardcoded `start_seed = 42`). This makes every inference run produce
byte-identical output for the same sample index, regardless of when the
run occurs or what randomness sources (including dropout) are active
downstream. This is appropriate for reproducing published example
outputs, but incompatible with any use case requiring genuinely
independent repeated samples — including Monte Carlo dropout, where the
intended additional randomness is otherwise silently suppressed by the
fixed seed.

**Fix**: disable the deterministic seed, using the sampling code's own
existing support for skipping seeding entirely when passed `-1`.

```python
# Before:
seedval = start_seed * (sample_id + 1) * 10

# After:
seedval = -1
```

**Tradeoff**: this removes exact reproducibility of the shipped example
outputs. Apply only when genuine per-call randomness is required (e.g.
MC-dropout inference); revert for reproducing published results from a
fixed checkpoint.
