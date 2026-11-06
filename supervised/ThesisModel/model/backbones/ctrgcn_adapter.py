"""
CTR-GCN Backbone Adapter
========================
Wraps CTR-GCN into the standardized BaseBackboneAdapter interface.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from model.backbones.base_adapter import BaseBackboneAdapter
from model.backbones.ctrgcn import CTRGCNModel


class CTRGCNAdapter(BaseBackboneAdapter):
    """
    Adapter for CTR-GCN Backbone.
    Feature Dimension: 256
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes = kwargs.get("num_class", 60)
        self.backbone = CTRGCNModel(**kwargs)
        self._feat_dim = self.backbone.feat_dim

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def extract_feat(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extracts pooled feature (B, 256). CTR-GCN ignores index_t."""
        return self.backbone.extract_feat(x)

    def forward(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backbone(x)

    def load_pretrained(self, checkpoint_path: str, device: torch.device = torch.device("cpu")) -> Tuple[list, list]:
        raw = torch.load(checkpoint_path, map_location=device)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        state = {k.replace("module.", ""): v for k, v in state.items()}

        if any(k.startswith("backbone.") for k in state):
            state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        return missing, unexpected

    def get_classifier_weights(self) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            return self.backbone.fc.weight, self.backbone.fc.bias
        return None
