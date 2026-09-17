"""
Backbones Registry and Factory
==============================
Provides factory methods to instantiate SkateFormer backbone adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Type
import torch
import torch.nn as nn

from model.backbones.base_adapter import BaseBackboneAdapter
from model.backbones.skateformer_adapter import SkateFormerAdapter
from model.backbones.hypergcn_adapter import HyperGCNAdapter


BACKBONE_REGISTRY: Dict[str, Type[BaseBackboneAdapter]] = {
    "skateformer": SkateFormerAdapter,
    "hypergcn": HyperGCNAdapter,
}


def register_backbone(name: str, adapter_cls: Type[BaseBackboneAdapter]):
    """Registers a new custom backbone adapter."""
    BACKBONE_REGISTRY[name.lower()] = adapter_cls


def build_backbone(
    backbone_name: str = "skateformer",
    backbone_cfg: Dict[str, Any] = None,
    num_class: int = 60,
    **kwargs,
) -> BaseBackboneAdapter:
    """
    Factory function to construct a backbone adapter by name.

    Args:
        backbone_name: Name of the backbone ('skateformer')
        backbone_cfg: Keyword arguments dict forwarded to the backbone adapter
        num_class: Number of action classes

    Returns:
        Instance of BaseBackboneAdapter
    """
    key = backbone_name.lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Available backbones: {list(BACKBONE_REGISTRY.keys())}"
        )

    cfg = (backbone_cfg or {}).copy()
    cfg.update(kwargs)

    if "num_class" not in cfg and "num_classes" not in cfg:
        cfg["num_classes"] = num_class

    adapter_cls = BACKBONE_REGISTRY[key]
    return adapter_cls(**cfg)
