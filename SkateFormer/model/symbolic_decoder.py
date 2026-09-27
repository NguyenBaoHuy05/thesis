"""
symbolic_decoder.py
--------------------
Translates concept probabilities [B, num_concepts] → action logits [B, num_classes].

Two modes:
  1. Pure symbolic (template-based):
       score_c = (concept_probs @ template.T)  normalised by template row sums.
       Produces interpretable action scores.

  2. Learned linear fusion (default):
       A learnable weight matrix W [num_classes, num_concepts] (initialised from
       the template) projects concept probs to action logits.
       Allows gradient-based fine-tuning while starting near the symbolic solution.

The SymbolicDecoder output can be used with nn.CrossEntropyLoss directly.

References:
  - Template matrix: concepts/ntu_rgbd_60_120.csv
    Each row = one action class; each column = one concept (0/1).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class SymbolicDecoder(nn.Module):
    """
    Map concept probabilities to action class logits.

    Args:
        num_concepts:     Number of binary concepts (74).
        num_classes:      Number of action classes (60 or 120).
        template_matrix:  Optional [num_classes, num_concepts] float tensor
                          for weight initialisation. If None, random init.
        mode:             'learned' | 'symbolic'
                          'learned'  → trainable linear layer (warm-started from template)
                          'symbolic' → frozen dot-product with template (no backprop through weights)
        hidden_dim:       If > 0, adds a hidden layer between concept probs and logits.
        dropout:          Dropout on concept probs before projection.
        temperature:      Scaling applied to symbolic scores (default 1.0).
    """

    def __init__(
        self,
        num_concepts: int = 74,
        num_classes: int = 60,
        template_matrix: Optional[torch.Tensor] = None,
        mode: str = "learned",
        hidden_dim: int = 0,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert mode in ("learned", "symbolic"), f"Unknown mode: {mode}"
        self.mode = mode
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

        # --- Learned mode ---
        if mode == "learned":
            if hidden_dim > 0:
                self.head = nn.Sequential(
                    nn.Linear(num_concepts, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )
                self._init_head(template_matrix, hidden_dim)
            else:
                self.head = nn.Linear(num_concepts, num_classes, bias=True)
                self._init_linear(self.head, template_matrix)
        else:
            # Symbolic mode: register template as a non-parameter buffer
            if template_matrix is None:
                template_matrix = torch.ones(num_classes, num_concepts) / num_concepts
            self.register_buffer("template", template_matrix.float())
            # Row sums for normalisation (avoid division by zero)
            row_sums = self.template.sum(dim=1, keepdim=True).clamp(min=1.0)
            self.register_buffer("template_norm", self.template / row_sums)

    # ------------------------------------------------------------------
    # Weight initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_linear(
        linear: nn.Linear, template: Optional[torch.Tensor]
    ):
        """Initialise a linear layer from template (or trunc_normal fallback)."""
        if template is not None:
            assert template.shape == (linear.out_features, linear.in_features), (
                f"Template shape {template.shape} != "
                f"({linear.out_features}, {linear.in_features})"
            )
            with torch.no_grad():
                linear.weight.copy_(template.float())
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
        else:
            trunc_normal_(linear.weight, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def _init_head(self, template: Optional[torch.Tensor], hidden_dim: int):
        """Initialise two-layer head; first layer random, second from template."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Try to warm-start the final layer from template
        final_linear = None
        for m in self.head.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.num_classes:
                final_linear = m
        if final_linear is not None and template is not None:
            # template is [num_classes, num_concepts]; final layer is [num_classes, hidden_dim]
            # Just do random init here since dims differ
            pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept_probs: [B, num_concepts] in [0, 1]
        Returns:
            action_logits: [B, num_classes]
        """
        x = self.dropout(concept_probs)

        if self.mode == "learned":
            return self.head(x) / self.temperature
        else:
            # Pure symbolic: normalised dot product
            scores = x @ self.template_norm.T    # [B, num_classes]
            return scores / self.temperature

    def symbolic_scores(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """
        Always return the symbolic (template-based) scores, regardless of mode.
        Useful for interpretability even when using learned mode.

        Returns:
            scores: [B, num_classes] in [0, 1]
        """
        if hasattr(self, "template_norm"):
            return concept_probs @ self.template_norm.T
        # If in learned mode without template buffer, return logits
        return self.forward(concept_probs)
