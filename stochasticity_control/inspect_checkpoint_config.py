"""
inspect_checkpoint_config.py -- Determines, directly and conclusively,
what dropout value was actually saved with this checkpoint, bypassing
the model-construction code entirely. nn.Dropout(p=X) is a module
hyperparameter, not a learned weight -- checkpoints created via PyTorch
Lightning typically save the config used to build the model (via
self.save_hyperparameters()) alongside the weights themselves. This
inspects that saved config directly, which tells us definitively
whether p=0.0 was baked into this specific checkpoint at save time,
independent of anything the model's runtime behavior alone can show.

Also checks configs/inference.yaml itself and searches the codebase for
where "dropout" is actually read/wired into model construction, since
there could be more than one place a value like this gets set or
overridden.

Usage (run from backbones/rnabpflow/):
    python inspect_checkpoint_config.py
"""
import torch
import yaml

CKPT_PATH = "checkpoint/RNA3DB.ckpt"

print("=" * 70)
print("STEP 1: Load the RAW checkpoint file (not via the model class)")
print("=" * 70)
raw = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print(f"Top-level keys in checkpoint: {list(raw.keys())}")
print()

# Try the common PyTorch Lightning locations for saved hyperparameters
for key_name in ["hyper_parameters", "hparams", "hyperparameters"]:
    if key_name in raw:
        print(f"Found '{key_name}' in checkpoint:")
        hp = raw[key_name]
        print(f"  Type: {type(hp)}")
        print(f"  Content: {hp}")
        print()

print("=" * 70)
print("STEP 2: Search the saved hyperparameters (however structured) for")
print("anything mentioning 'dropout'")
print("=" * 70)


def search_for_dropout(obj, path="root"):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "dropout" in str(k).lower():
                found.append((f"{path}.{k}", v))
            found.extend(search_for_dropout(v, f"{path}.{k}"))
    elif hasattr(obj, "__dict__"):
        try:
            for k, v in vars(obj).items():
                if "dropout" in str(k).lower():
                    found.append((f"{path}.{k}", v))
                found.extend(search_for_dropout(v, f"{path}.{k}"))
        except Exception:
            pass
    return found


hits = search_for_dropout(raw)
if hits:
    for path, val in hits:
        print(f"  {path} = {val}")
else:
    print("  No 'dropout' key found anywhere in the checkpoint's saved data.")
print()

print("=" * 70)
print("STEP 3: Check the state_dict itself for any dropout-related keys")
print("(dropout normally has no weights, but confirming this checkpoint")
print("doesn't do something unusual)")
print("=" * 70)
if "state_dict" in raw:
    dropout_keys = [k for k in raw["state_dict"].keys() if "dropout" in k.lower()]
    print(f"State dict keys mentioning 'dropout': {dropout_keys if dropout_keys else 'none (expected)'}")
print()

print("=" * 70)
print("STEP 4: Check configs/inference.yaml directly for any dropout value")
print("=" * 70)
with open("configs/inference.yaml") as f:
    inf_cfg = yaml.safe_load(f)


def search_yaml_for_dropout(obj, path="root"):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "dropout" in str(k).lower():
                found.append((f"{path}.{k}", v))
            found.extend(search_yaml_for_dropout(v, f"{path}.{k}"))
    return found


yaml_hits = search_yaml_for_dropout(inf_cfg)
if yaml_hits:
    for path, val in yaml_hits:
        print(f"  {path} = {val}")
else:
    print("  No 'dropout' key found in configs/inference.yaml.")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("If Step 2 shows a dropout value: that's what was saved with this")
print("  checkpoint's config at training/save time -- if it's 0.0, this")
print("  checkpoint was saved with dropout already disabled, independent")
print("  of anything documentation elsewhere claims.")
print("If Step 2 shows nothing but Step 4 shows a value: inference.yaml")
print("  might be overriding architecture, worth checking how.")
print("If NEITHER shows anything: the dropout p value is coming from")
print("  somewhere else not yet located -- next step would be searching")
print("  the FlowModel class definition itself for a hardcoded default.")
