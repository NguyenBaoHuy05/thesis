"""
Backbones Factory
=================
Provides a factory method to instantiate the SkateFormer backbone adapter,
the sole backbone supported by the Joint Symbolic CBM.
"""

from __future__ import annotations

from typing import Any, Dict

from model.backbones.base_adapter import BaseBackboneAdapter
from model.backbones.skateformer_adapter import SkateFormerAdapter


def build_backbone(
    backbone_cfg: Dict[str, Any] = None,
    num_class: int = 60,
    **kwargs,
) -> SkateFormerAdapter:
    """
    Factory function to construct the SkateFormer backbone adapter.

    Args:
        backbone_cfg: Keyword arguments dict forwarded to the backbone adapter
        num_class: Number of action classes

    Returns:
        Instance of SkateFormerAdapter
    """
    cfg = (backbone_cfg or {}).copy()
    cfg.update(kwargs)

    if "num_class" not in cfg and "num_classes" not in cfg:
        cfg["num_classes"] = num_class

    return SkateFormerAdapter(**cfg)
