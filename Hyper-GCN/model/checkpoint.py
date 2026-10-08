"""Checked initialization from official Hyper-GCN and symbolic checkpoints."""

from collections import OrderedDict
from collections.abc import Mapping

import torch


CSV_DERIVED_BUFFERS = {
    "symbolic_graph.part_incidence", "symbolic_graph.concept_part",
    "symbolic_graph.concept_affinity", "symbolic_graph.support",
    "symbolic_graph.positive_weight",
}


def load_model_weights(model, checkpoint, allow_new_symbolic=False, ignore_weights=()):
    """Allow new symbolic parameters only when explicitly fine-tuning in train mode."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a state_dict or contain a state_dict mapping")
    for wrapper in ("state_dict", "model_state_dict"):
        if wrapper in checkpoint:
            checkpoint = checkpoint[wrapper]
            break
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")
    state = OrderedDict()
    rebuilt, ignored = [], []
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise ValueError("checkpoint keys must be strings")
        key = key[7:] if key.startswith("module.") else key
        if key in CSV_DERIVED_BUFFERS:
            rebuilt.append(key)
            continue
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
    bridge = getattr(model, "symbolic_graph", None)
    invalid_missing = [
        key for key in missing
        if not any(pattern in key for pattern in ignore_weights)
        and not (allow_new_symbolic and key.startswith("symbolic_graph."))
        and not (key == "symbolic_graph.training_class_counts" and bridge is not None
                 and (not bridge.balance_loss or bridge.pos_weight_source == "class"))
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
    return {"loaded": len(state), "initialized": missing, "ignored": ignored, "rebuilt": rebuilt}
