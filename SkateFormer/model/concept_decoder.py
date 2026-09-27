"""
concept_decoder.py
------------------
A lightweight concept decoder that maps backbone feature tokens
[B, T'*V', C] → concept logits [B, 74].

Design:
  - Cross-attention from 74 learnable concept queries to backbone tokens.
  - Feed-forward projection per concept.
  - Sigmoid output (independent binary classification per concept).
  - Supports 3-state labels: positive (1), negative (0), unknown (-1).
    Unknown positions are masked out of the BCE loss.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PositionwiseFF(nn.Module):
    """Standard two-layer position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: queries come from concept embeddings,
    keys/values come from backbone feature tokens.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # [B, num_concepts, d_model]
        kv: torch.Tensor,      # [B, seq_len, feat_dim]  (feat_dim == d_model after projection)
    ) -> torch.Tensor:
        B, Nc, D = query.shape
        _, L, _ = kv.shape

        q = self.q_proj(query).view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, Nc, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nc, D)
        return self.out_proj(out)


class ConceptDecoderLayer(nn.Module):
    """One decoder layer: cross-attention + self-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.use_self_attn = use_self_attn

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        x = queries + self.dropout(self.cross_attn(self.norm1(queries), memory))

        # Optional self-attention among concept queries
        if self.use_self_attn:
            sa_out, _ = self.self_attn(self.norm2(x), self.norm2(x), self.norm2(x))
            x = x + self.dropout(sa_out)

        # FFN
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


# ---------------------------------------------------------------------------
# Main ConceptDecoder
# ---------------------------------------------------------------------------

class ConceptDecoder(nn.Module):
    """
    Concept decoder: maps backbone feature tokens to per-concept probabilities.

    Args:
        backbone_dim:  Channel dimension of backbone output feature map (e.g., 192).
        d_model:       Internal hidden dimension of the decoder (default 256).
        num_concepts:  Number of binary concepts (default 74).
        num_heads:     Number of attention heads (default 8).
        num_layers:    Number of stacked decoder layers (default 2).
        d_ff:          Feed-forward hidden size (default 512).
        dropout:       Dropout rate (default 0.1).
        use_self_attn: Whether to include self-attention among concept queries.
    """

    def __init__(
        self,
        backbone_dim: int = 192,
        d_model: int = 256,
        num_concepts: int = 74,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.d_model = d_model

        # Project backbone features to d_model if needed
        if backbone_dim != d_model:
            self.feat_proj = nn.Sequential(
                nn.Linear(backbone_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.feat_proj = nn.Identity()

        # Learnable concept query embeddings  [num_concepts, d_model]
        self.concept_queries = nn.Parameter(torch.zeros(num_concepts, d_model))
        trunc_normal_(self.concept_queries, std=0.02)

        # Decoder layers
        self.layers = nn.ModuleList([
            ConceptDecoderLayer(d_model, num_heads, d_ff, dropout, use_self_attn)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Per-concept output projection to scalar logit
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, T', V'] feature map from backbone  OR
                      [B, T'*V', C] token sequence.
        Returns:
            concept_logits: [B, num_concepts]  (raw logits, apply sigmoid for probs)
        """
        if features.dim() == 4:
            # [B, C, T', V'] → [B, T'*V', C]
            B, C, T, V = features.shape
            features = features.permute(0, 2, 3, 1).contiguous().view(B, T * V, C)
        elif features.dim() == 2:
            # [B, C] pooled vector → [B, 1, C]
            features = features.unsqueeze(1)

        # Project to d_model
        memory = self.feat_proj(features)   # [B, L, d_model]

        # Expand concept queries for batch
        B = memory.shape[0]
        q = self.concept_queries.unsqueeze(0).expand(B, -1, -1)   # [B, num_concepts, d_model]

        for layer in self.layers:
            q = layer(q, memory)

        q = self.norm(q)                        # [B, num_concepts, d_model]
        logits = self.head(q).squeeze(-1)       # [B, num_concepts]
        return logits

    def get_concept_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning sigmoid probabilities."""
        return torch.sigmoid(self.forward(features))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_bce_loss(
    logits: torch.Tensor,       # [B, num_concepts]
    targets: torch.Tensor,      # [B, num_concepts]  values: 0, 1, or -1 (unknown)
    mask: Optional[torch.Tensor] = None,   # [B, num_concepts] bool; True = include
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary cross-entropy loss that ignores unknown positions.

    If `mask` is None, positions with targets == -1 are automatically excluded.
    If `mask` is provided, only positions where mask == True are included.

    Args:
        logits:    Raw logits from ConceptDecoder.
        targets:   Target values.  Use -1 for unknown/masked.
        mask:      Optional explicit boolean mask.
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        Scalar loss (or per-element tensor if reduction='none').
    """
    if mask is None:
        mask = (targets >= 0)   # True where label is known (0 or 1)

    # Clamp targets to [0, 1] for BCE (unknown positions will be masked anyway)
    targets_clamped = targets.float().clamp(min=0.0)

    bce = F.binary_cross_entropy_with_logits(logits, targets_clamped, reduction="none")
    bce = bce * mask.float()

    if reduction == "mean":
        denom = mask.float().sum().clamp(min=1.0)
        return bce.sum() / denom
    elif reduction == "sum":
        return bce.sum()
    else:
        return bce


def soft_bce_loss(
    logits: torch.Tensor,       # [B, num_concepts]
    soft_targets: torch.Tensor, # [B, num_concepts]  values in [0, 1], -1 = unknown
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Soft BCE loss for refined labels (teacher probabilities rather than hard 0/1).
    Unknown positions (soft_targets == -1) are excluded.
    """
    if mask is None:
        mask = (soft_targets >= 0)
    soft_targets_clamped = soft_targets.float().clamp(min=0.0, max=1.0)
    bce = F.binary_cross_entropy_with_logits(logits, soft_targets_clamped, reduction="none")
    bce = bce * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return bce.sum() / denom
