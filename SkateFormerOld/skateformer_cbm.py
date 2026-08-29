"""
Concept Bottleneck Model (CBM) Core Wrapper
===========================================
Wraps any neural backbone (SkateFormer, CTR-GCN, ST-GCN, Hyper-GCN, SK-GCN) with:
  - Per-concept encoders / MLPs -> concept truth degrees
  - Knowledge-graph buffers (semantic similarity + co-activation)
  - Modular classification heads

This file defines:
  GenericCBM              - Base CBM wrapper for any backbone adapter
  SkateFormerCBM          - Specialized SkateFormer CBM (backward compatible)
  build_skateformer_cbm() - Factory helper
  freeze_backbone()       - Helper to freeze backbone parameters
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import List, Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base_adapter import BaseBackboneAdapter
from model.backbones.skateformer_adapter import SkateFormerAdapter, DEFAULT_SKATEFORMER_CFG
from knowledge_graph import (
    load_concept_matrix,
    build_semantic_similarity_from_concepts,
    build_concept_coactivation_static,
)

CONCEPTS_CSV = os.path.join(os.path.dirname(__file__), "concepts", "ntu_rgbd_60_120.csv")


class GenericCBM(nn.Module):
    """
    Generic Concept Bottleneck Model (CBM) for any backbone.

    Architecture:
    Input (B, C, T, V, M) -> Backbone Adapter -> Feature (B, feat_dim)
        |
        +---> Concept MLPs -> sigmoid -> concept_probs (B, num_concepts)
        |
        +---> Direct Action Classifier (Linear) -> action_logits (B, num_class)
    """

    def __init__(
        self,
        backbone: Union[BaseBackboneAdapter, nn.Module],
        num_class: int = 60,
        num_concepts: int = 74,
        concept_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.backbone_adapter = backbone
        if hasattr(backbone, "backbone"):
            self.backbone = backbone.backbone
        else:
            self.backbone = backbone

        if hasattr(backbone, "feat_dim"):
            self.feat_dim: int = backbone.feat_dim
        else:
            self.feat_dim: int = 192

        # ── Knowledge Graph buffers from CSV ──────────────────────────────
        concept_matrix, concept_names = load_concept_matrix(CONCEPTS_CSV, num_class)
        self.register_buffer("concept_matrix", concept_matrix)   # (num_class, num_concepts)
        self.concept_names: List[str] = concept_names
        self.num_concepts: int = concept_matrix.shape[1]
        self.num_class: int = num_class

        S_KG = build_semantic_similarity_from_concepts(concept_matrix)
        C_KG = build_concept_coactivation_static(concept_matrix)
        self.register_buffer("S_KG_static", S_KG)
        self.register_buffer("C_KG_static", C_KG)

        concept_pos = concept_matrix.float().sum(dim=0)
        concept_neg = float(num_class) - concept_pos
        pos_weight = (concept_neg / concept_pos.clamp_min(1.0)).clamp(1.0, 20.0)
        self.register_buffer("concept_pos_weight", pos_weight)
        self.register_buffer("concept_active_mask", concept_pos > 0, persistent=False)

        # ── Concept head (Head B) ─────────────────────────────────────────
        self.concept_embed_dim = 32
        self.concept_hidden_dim = 64
        self.concept_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feat_dim, self.concept_hidden_dim),
                nn.GELU(),
            ) for _ in range(self.num_concepts)
        ])
        self.concept_truth_heads = nn.ModuleList([
            nn.Linear(self.concept_hidden_dim, 1)
            for _ in range(self.num_concepts)
        ])
        self.concept_embedding_heads = nn.ModuleList([
            nn.Linear(self.concept_hidden_dim, self.concept_embed_dim)
            for _ in range(self.num_concepts)
        ])

        self.concept_loss_weight = concept_loss_weight

    def extract_feat(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extracts pooled feature vector (B, feat_dim) from skeleton input."""
        if hasattr(self.backbone_adapter, "extract_feat"):
            return self.backbone_adapter.extract_feat(x, index_t)
        elif hasattr(self.backbone, "extract_feat"):
            return self.backbone.extract_feat(x)
        else:
            raise NotImplementedError("Backbone must implement extract_feat(x, index_t)")

    def encode_concepts(self, feat: torch.Tensor):
        """Return concept truth logits and per-concept embeddings."""
        logits, embeddings = [], []
        for mlp, truth, emb in zip(
            self.concept_mlps,
            self.concept_truth_heads,
            self.concept_embedding_heads,
        ):
            h = mlp(feat)
            logits.append(truth(h))
            embeddings.append(emb(h))
        return torch.cat(logits, dim=-1), torch.stack(embeddings, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        index_t: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        feat = self.extract_feat(x, index_t)
        concept_logits, concept_embeddings = self.encode_concepts(feat)
        concept_probs = torch.sigmoid(concept_logits)
        if hasattr(self, "fc_action"):
            cls_score = self.fc_action(concept_probs, concept_embeddings)
        else:
            cls_score = concept_probs

        if not return_loss or label is None:
            return cls_score

        concept_gt = self.concept_matrix[label]
        loss_concept = F.binary_cross_entropy_with_logits(
            concept_logits, concept_gt,
            pos_weight=self.concept_pos_weight,
        )

        total_loss = self.concept_loss_weight * loss_concept
        loss_dict = {
            "concept": loss_concept.detach(),
        }
        return total_loss, loss_dict


class SkateFormerCBM(GenericCBM):
    """
    SkateFormer + Concept Bottleneck Model (CBM).
    Preserved for full backward compatibility.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        num_class: int = 60,
        num_concepts: int = 74,
        concept_loss_weight: float = 0.5,
    ):
        adapter = SkateFormerAdapter(**backbone_cfg)
        super().__init__(
            backbone=adapter,
            num_class=num_class,
            num_concepts=num_concepts,
            concept_loss_weight=concept_loss_weight,
        )


def build_skateformer_cbm(
    checkpoint_path: str,
    backbone_cfg: dict,
    num_concepts: int = 74,
    concept_loss_weight: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> SkateFormerCBM:
    """Helper to build SkateFormerCBM and load weights."""
    raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt = OrderedDict((k.replace("module.", ""), v) for k, v in raw.items())

    if "backbone.head.weight" in ckpt:
        num_class = ckpt["backbone.head.weight"].shape[0]
    elif "head.weight" in ckpt:
        num_class = ckpt["head.weight"].shape[0]
    else:
        num_class = backbone_cfg.get("num_classes", 60)

    print(f"[CBM] Auto-detected num_class={num_class}")

    model = SkateFormerCBM(
        backbone_cfg=backbone_cfg,
        num_class=num_class,
        num_concepts=num_concepts,
        concept_loss_weight=concept_loss_weight,
    )

    missing, unexpected = model.backbone_adapter.load_pretrained(checkpoint_path, device="cpu")
    if missing:
        print(f"[CBM][WARN] backbone missing {len(missing)} keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"[CBM][WARN] backbone unexpected {len(unexpected)} keys (first 3: {unexpected[:3]})")

    print(f"[CBM] Pretrained SkateFormer backbone loaded <- {checkpoint_path}")
    return model.to(device)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters in any CBM model."""
    frozen, trainable = 0, 0
    if hasattr(model, "backbone_adapter") and hasattr(model.backbone_adapter, "freeze_backbone"):
        frozen, _ = model.backbone_adapter.freeze_backbone()
    elif hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False
            frozen += p.numel()

    for name, param in model.named_parameters():
        if not name.startswith("backbone"):
            param.requires_grad = True
            trainable += param.numel()

    print(f"[CBM] Frozen backbone {frozen:,} | Trainable (concept+action) {trainable:,}")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all backbone parameters for end-to-end fine-tuning."""
    trainable = 0
    for param in model.parameters():
        param.requires_grad = True
        trainable += param.numel()
    print(f"[CBM] Unfrozen full model | Trainable parameters: {trainable:,}")
