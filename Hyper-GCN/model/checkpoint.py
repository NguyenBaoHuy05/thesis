"""Checked initialization from official Hyper-GCN and synergy checkpoints."""

from collections import OrderedDict
from collections.abc import Mapping

import torch

from model.training import is_extension_key


def load_model_weights(model, checkpoint, allow_new=False, ignore_weights=()):
    """Allow missing synergy / semantic-hyperedge parameters only when
    ``allow_new`` (fine-tuning from an official checkpoint in train mode)."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a state_dict or contain a state_dict mapping")
    for wrapper in ("state_dict", "model_state_dict"):
        if wrapper in checkpoint:
            checkpoint = checkpoint[wrapper]
            break
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")
    state = OrderedDict()
    ignored = []
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise ValueError("checkpoint keys must be strings")
        key = key[7:] if key.startswith("module.") else key
        if any(pattern in key for pattern in ignore_weights):
            ignored.append(key)
            continue
        if key in state:
            raise ValueError(f"duplicate checkpoint key after stripping module prefix: {key}")
        if not torch.is_tensor(value):
            raise ValueError(f"checkpoint value is not a tensor: {key}")
        state[key] = value
    expected = model.state_dict()
    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    invalid_missing = [
        key for key in missing
        if not any(pattern in key for pattern in ignore_weights)
        and not (allow_new and is_extension_key(key))
    ]
    mismatched = [
        f"{key}: checkpoint {tuple(state[key].shape)} != model {tuple(expected[key].shape)}"
        for key in state.keys() & expected.keys() if state[key].shape != expected[key].shape
    ]
    if unexpected or invalid_missing or mismatched:
        raise ValueError(
            "checkpoint is incompatible; use matching base/large, NTU60/120 and hyper_joints. "
            f"Missing: {invalid_missing}; unexpected: {unexpected}; shapes: {mismatched}"
        )
    model.load_state_dict(state, strict=False)
    return {"loaded": len(state), "initialized": missing, "ignored": ignored}
