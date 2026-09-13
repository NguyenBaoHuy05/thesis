"""
Model package for Skeleton Action Recognition with Joint Symbolic Action Compatibility
"""

from model.SkateFormer import SkateFormer, SkateFormer_
from model.backbones import (
    BaseBackboneAdapter,
    SkateFormerAdapter,
    build_backbone,
    register_backbone,
)

__all__ = [
    "SkateFormer",
    "SkateFormer_",
    "BaseBackboneAdapter",
    "SkateFormerAdapter",
    "build_backbone",
    "register_backbone",
]
