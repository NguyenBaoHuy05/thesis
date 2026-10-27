"""Partial fine-tuning: freeze the backbone up to a layer, keep its BN statistics."""

STEM = ("data_bn", "to_joint_embedding", "pos_embedding")
LAYERS = tuple("l{}".format(i) for i in range(1, 10))
# Top-level modules added on top of Hyper-GCN (e.g. "cg.", model/hypergcn_concept.py; "a3.", model/hypergcn_a3.py):
# absent from official checkpoints, always trainable.
EXTENSION_PREFIXES = ("kg.", "cg.", "a3.", "a3g.")


def is_extension_key(key):
    """Parameters that official Hyper-GCN checkpoints do not have."""
    return key.startswith(EXTENSION_PREFIXES)


def frozen_modules(freeze_until=None):
    """Top-level backbone names frozen by ``freeze_until``.

    None / "none": nothing is frozen. "stem": the input embedding only.
    "lK": the embedding and layers l1..lK; "l9" freezes the whole backbone; "fc" also its classifier.
    """
    if freeze_until in (None, "none"):
        return ()
    if freeze_until == "stem":
        return STEM
    if freeze_until == "fc":
        return STEM + LAYERS + ("fc",)
    if freeze_until not in LAYERS:
        raise ValueError("freeze_until must be none, stem, fc or one of {}".format(LAYERS))
    return STEM + LAYERS[:LAYERS.index(freeze_until) + 1]


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def configure_trainable_parameters(model, freeze_until=None):
    """Extension modules always train; frozen backbone modules do not."""
    model_ref = _unwrap(model)
    frozen = frozen_modules(freeze_until)
    for name, parameter in model_ref.named_parameters():
        trainable = is_extension_key(name) or name.split(".")[0] not in frozen
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
