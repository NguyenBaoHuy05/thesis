"""
Model package for Skeleton Action Recognition with Concept Bottlenecks
"""

from model.SkateFormer import SkateFormer, SkateFormer_
from model.backbones import (
    BaseBackboneAdapter,
    SkateFormerAdapter,
    CTRGCNAdapter,
    STGCNAdapter,
    HyperGCNAdapter,
    SKGCNAdapter,
    build_backbone,
    register_backbone,
)
