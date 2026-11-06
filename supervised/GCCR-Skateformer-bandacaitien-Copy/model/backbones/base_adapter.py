"""
Base Backbone Adapter for Human Action Recognition
===================================================
Provides a unified interface for all neural backbones (Transformers, GCNs, Hypergraphs)
so they can seamlessly integrate with the Concept Bottleneck Model (CBM)
and Symbolic Action Compatibility Reasoning framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class BaseBackboneAdapter(nn.Module, ABC):
    """
    Abstract Base Class for Backbone Adapters.

    Every backbone must implement:
      1. extract_feat(x, index_t=None) -> Tensor of shape (B, feat_dim)
      2. feat_dim (property returning the feature embedding dimension)
      3. forward(x, index_t=None) -> Classification logits (B, num_class)
      4. load_pretrained(checkpoint_path, device) -> loads weights & maps keys
      5. get_classifier_weights() -> (weight, bias) tuple from final FC layer
    """

    def __init__(self):
        super().__init__()
        self.backbone: nn.Module = None

    @property
    @abstractmethod
    def feat_dim(self) -> int:
        """Returns the output feature dimension before classification head."""
        pass

    @abstractmethod
    def extract_feat(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract pooled feature vector from raw skeleton input tensor.

        Args:
            x: Input skeleton tensor of shape (B, C, T, V, M)
            index_t: Optional temporal index tensor (used by SkateFormer)

        Returns:
            feat: Pooled feature tensor of shape (B, feat_dim)
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass producing raw action classification logits.

        Args:
            x: Input skeleton tensor of shape (B, C, T, V, M)
            index_t: Optional temporal index tensor

        Returns:
            logits: Action classification logits of shape (B, num_class)
        """
        pass

    @abstractmethod
    def load_pretrained(self, checkpoint_path: str, device: torch.device = torch.device("cpu")) -> Tuple[list, list]:
        """
        Load pretrained weights into the backbone module.
        Handles flat keys, 'module.' prefixes, and sub-module mapping.

        Returns:
            (missing_keys, unexpected_keys)
        """
        pass

    @abstractmethod
    def get_classifier_weights(self) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Extracts (weight, bias) from the backbone's final classification layer
        to warm-start the direct_action classifier in the Joint Symbolic CBM.
        """
        pass

    def freeze_backbone(self) -> Tuple[int, int]:
        """Freezes all parameters in the backbone."""
        frozen, trainable = 0, 0
        for p in self.parameters():
            p.requires_grad = False
            frozen += p.numel()
        return frozen, trainable

    def unfreeze_backbone(self) -> Tuple[int, int]:
        """Unfreezes all parameters in the backbone for fine-tuning."""
        frozen, trainable = 0, 0
        for p in self.parameters():
            p.requires_grad = True
            trainable += p.numel()
        return frozen, trainable
