"""
Backbones Registry and Factory
==============================
Provides factory methods to instantiate and wrap any skeleton action recognition backbone.
"""

from __future__ import annotations

from typing import Any, Dict, Type, Union
import torch
import torch.nn as nn

from model.backbones.base_adapter import BaseBackboneAdapter
from model.backbones.skateformer_adapter import SkateFormerAdapter
from model.backbones.ctrgcn_adapter import CTRGCNAdapter
from model.backbones.stgcn_adapter import STGCNAdapter
from model.backbones.hypergcn_adapter import HyperGCNAdapter
from model.backbones.skgcn_adapter import SKGCNAdapter


BACKBONE_REGISTRY: Dict[str, Type[BaseBackboneAdapter]] = {
    "skateformer": SkateFormerAdapter,
    "ctrgcn": CTRGCNAdapter,
    "ctr_gcn": CTRGCNAdapter,
    "stgcn": STGCNAdapter,
    "st_gcn": STGCNAdapter,
    "baseline": STGCNAdapter,
    "hypergcn": HyperGCNAdapter,
    "hyper_gcn": HyperGCNAdapter,
    "skgcn": SKGCNAdapter,
    "sk_gcn": SKGCNAdapter,
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
        backbone_name: Name of the backbone ('skateformer', 'ctrgcn', 'stgcn', 'hypergcn', 'skgcn')
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

    # Ensure num_class or num_classes is passed properly
    if "num_class" not in cfg and "num_classes" not in cfg:
        if key == "skateformer":
            cfg["num_classes"] = num_class
        else:
            cfg["num_class"] = num_class

    adapter_cls = BACKBONE_REGISTRY[key]
    return adapter_cls(**cfg)
