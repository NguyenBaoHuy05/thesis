"""Partial fine-tuning: freeze the backbone up to a layer, keep its BN statistics."""

STEM = ("data_bn", "to_joint_embedding", "pos_embedding")
LAYERS = tuple("l{}".format(i) for i in range(1, 10))
HEAD = ("fc", "synergy")


def is_extension_key(key):
    """Parameters that official Hyper-GCN checkpoints do not have."""
    return key.startswith("synergy.") or ".semantic." in key


def is_head_key(key):
    return key.split(".")[0] in HEAD or ".semantic." in key


def frozen_modules(freeze_until=None):
    """Top-level backbone names frozen by ``freeze_until``.

    None / "none": nothing is frozen. "stem": the input embedding only.
    "lK": the embedding and layers l1..lK; "l9" freezes the whole backbone.
    """
    if freeze_until in (None, "none"):
        return ()
    if freeze_until == "stem":
        return STEM
    if freeze_until not in LAYERS:
        raise ValueError("freeze_until must be none, stem or one of {}".format(LAYERS))
    return STEM + LAYERS[:LAYERS.index(freeze_until) + 1]


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def configure_trainable_parameters(model, freeze_until=None):
    """New parameters (synergy bottleneck, semantic hyperedges) always train.
    In synergy mode the unused classifier ``fc`` is frozen."""
    model_ref = _unwrap(model)
    frozen = frozen_modules(freeze_until)
    synergy = getattr(model_ref, "synergy", None) is not None
    for name, parameter in model_ref.named_parameters():
        top = name.split(".")[0]
        trainable = is_extension_key(name) or top not in frozen
        if synergy and top == "fc":
            trainable = False
        parameter.requires_grad_(trainable)
        if not trainable:
            parameter.grad = None
    trainable = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_ref.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total,
            "frozen_modules": list(frozen)}


def set_training_mode(model, freeze_until=None):
    # requires_grad=False does not freeze running_mean/running_var: frozen
    # modules stay in eval mode while the rest trains.
    model.train()
    model_ref = _unwrap(model)
    for name in frozen_modules(freeze_until):
        module = getattr(model_ref, name)
        if hasattr(module, "eval"):
            module.eval()
