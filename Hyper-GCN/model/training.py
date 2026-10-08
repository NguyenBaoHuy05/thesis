"""Train a symbolic adapter without changing pretrained layers or their BN state."""


def configure_trainable_parameters(model, symbolic_only=False):
    model_ref = model.module if hasattr(model, "module") else model
    if symbolic_only:
        bridge = getattr(model_ref, "symbolic_graph", None)
        if bridge is None:
            raise ValueError("symbolic_only requires an enabled symbolic_graph branch")
        for parameter in model_ref.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in bridge.parameters():
            parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_ref.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}


def set_training_mode(model, symbolic_only=False):
    if not symbolic_only:
        model.train()
        return
    model_ref = model.module if hasattr(model, "module") else model
    bridge = getattr(model_ref, "symbolic_graph", None)
    if bridge is None:
        raise ValueError("symbolic_only requires an enabled symbolic_graph branch")
    # requires_grad=False does not freeze running_mean/running_var. Keep every
    # pretrained module in eval mode, then enable training only inside the bridge.
    model.eval()
    bridge.train()
