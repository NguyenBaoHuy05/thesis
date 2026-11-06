"""Group-aware Transformer decoder for concept bottleneck prediction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class GroupAwareConceptDecoder(nn.Module):
    """Decode spatio-temporal backbone tokens into grouped concept logits.

    Each concept group owns a small set of learnable queries.  The queries
    cross-attend to SkateFormer tokens and are pooled into a group feature,
    preserving the anatomical/temporal grouping used by the logic engine.
    """

    def __init__(
        self,
        memory_dim: int,
        concept_groups: Dict[str, List[int]],
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        queries_per_group: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if decoder_dim % num_heads != 0:
            raise ValueError("decoder_dim must be divisible by decoder_heads")

        self.decoder_dim = decoder_dim
        self.group_indices = {name: list(indices) for name, indices in concept_groups.items() if indices}
        self.num_concepts = max((i for indices in self.group_indices.values() for i in indices), default=-1) + 1
        self.memory_projection = nn.Linear(memory_dim, decoder_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.group_queries = nn.ParameterDict()
        self.group_heads = nn.ModuleDict()

        for group_name, indices in concept_groups.items():
            if not indices:
                continue
            self.group_queries[group_name] = nn.Parameter(
                torch.randn(queries_per_group, decoder_dim) * 0.02
            )
            self.group_heads[group_name] = nn.Sequential(
                nn.LayerNorm(decoder_dim),
                nn.Linear(decoder_dim, len(indices)),
            )

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return all concept logits, group logits, and group features."""
        memory = self.memory_projection(tokens)
        batch_size = memory.size(0)
        concept_logits = memory.new_zeros(batch_size, self.num_concepts)
        group_logits: Dict[str, torch.Tensor] = {}
        group_features: Dict[str, torch.Tensor] = {}

        for group_name, head in self.group_heads.items():
            queries = self.group_queries[group_name].unsqueeze(0).expand(batch_size, -1, -1)
            decoded = self.decoder(tgt=queries, memory=memory)
            group_feature = decoded.mean(dim=1)
            logits = head(group_feature)
            indices = self.group_indices[group_name]
            concept_logits[:, indices] = logits
            group_logits[group_name] = logits
            group_features[group_name] = group_feature

        return concept_logits, group_logits, group_features
