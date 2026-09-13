"""
Hyper-GCN Backbone Adapter
===========================
Wraps the Hyper-GCN model (hypergcn/model/hypergcn_base.py) into the
standardized BaseBackboneAdapter interface so it can be used as a drop-in
backbone inside JointSpatialTemporalCBM.

Architecture summary (base variant):
  Input (B, C, T, V, M)
    -> Joint embedding (C -> 128)
    -> 9 x TCN_GCN_unit blocks with hypergraph adjacency
       (3 blocks x 128ch, 3 blocks x 256ch, 3 blocks x 256ch)
    -> Global average pool -> feat (B, 256)
    -> Linear head -> logits (B, num_class)

Feature dimension: 256  (embedding_channels * 2 inside HyperGCNModel)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Load hypergcn_base directly by file path to avoid clashing with the
# project-level `model/` package (both are named "model").
_HYPERGCN_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # model/backbones/
        "..", "..",                                   # project root
        "hypergcn",
    )
)
_GRAPH_DIR = os.path.join(_HYPERGCN_ROOT, "graph")
if _HYPERGCN_ROOT not in sys.path:
    sys.path.insert(0, _HYPERGCN_ROOT)

# hypergcn/graph must be importable by import_class('graph.ntu_rgb_d.Graph')
# Adding _HYPERGCN_ROOT is sufficient (it contains the graph/ package).

def _load_module_from_path(module_name: str, file_path: str):
    """Import a Python file as a module with the given name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_hgcn_base_path = os.path.join(_HYPERGCN_ROOT, "model", "hypergcn_base.py")
_hgcn_mod = _load_module_from_path("hypergcn_base_module", _hgcn_base_path)
HyperGCNModel = _hgcn_mod.Model

from model.backbones.base_adapter import BaseBackboneAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_HYPERGCN_CFG = dict(
    num_class=60,
    num_point=25,
    num_person=2,
    graph="graph.ntu_rgb_d.Graph",
    graph_args={"labeling_mode": "virtual_ensemble"},
    in_channels=3,
    hyper_joints=3,   # number of virtual hyperedge nodes
    drop_out=0,
)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HyperGCNAdapter(BaseBackboneAdapter):
    """
    Adapter for the Hyper-GCN backbone (base variant).

    Feature Dimension: 256  (embedding_channels x 2 inside HyperGCNModel)

    Key quirk: HyperGCNModel.forward() returns a tuple
        (logits, [h_x1, ..., h_x9])
    so both forward() and extract_feat() need to unpack accordingly.
    """

    def __init__(self, **kwargs):
        super().__init__()
        cfg = DEFAULT_HYPERGCN_CFG.copy()

        # Map generic backbone_cfg keys to HyperGCN parameter names
        if "num_classes" in kwargs and "num_class" not in kwargs:
            kwargs["num_class"] = kwargs.pop("num_classes")
        if "num_points" in kwargs and "num_point" not in kwargs:
            kwargs["num_point"] = kwargs.pop("num_points")
        if "num_people" in kwargs and "num_person" not in kwargs:
            kwargs["num_person"] = kwargs.pop("num_people")

        cfg.update(kwargs)

        self.num_classes = cfg["num_class"]
        # HyperGCNModel stores embedding_channels=128, so feat_dim = 128*2 = 256
        self._feat_dim = 256

        self.backbone = HyperGCNModel(
            num_class=cfg["num_class"],
            num_point=cfg["num_point"],
            num_person=cfg["num_person"],
            graph=cfg["graph"],
            graph_args=cfg["graph_args"],
            in_channels=cfg["in_channels"],
            hyper_joints=cfg["hyper_joints"],
            drop_out=cfg["drop_out"],
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    # -----------------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------------

    def extract_feat(
        self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns pooled skeleton feature of shape (B, 256).

        Replicates the feature-extraction portion of HyperGCNModel.forward()
        without the final fc head, so the CBM can attach its own concept
        group heads on top of the same feature vector.
        """
        N, C, T, V, M = x.shape

        # Joint embedding & positional encoding
        h = x.permute(0, 4, 2, 3, 1).contiguous()       # (N, M, T, V, C)
        h = self.backbone.to_joint_embedding(h)           # (N, M, T, V, 128)
        h += self.backbone.pos_embedding[:, :V]
        h = self.backbone.tanh(h)
        h = h.permute(0, 1, 3, 4, 2).contiguous()        # (N, M, V, 128, T)

        # Batch norm over (M*V*128, T)
        h = h.view(N, M * V * self.backbone.embedding_channels, T)
        h = self.backbone.data_bn(h)
        h = h.view(N, M, V, self.backbone.embedding_channels, T)
        h = h.permute(0, 1, 3, 4, 2).contiguous()        # (N, M, 128, T, V)
        h = h.view(N * M, self.backbone.embedding_channels, T, V)

        # 9 residual TCN-GCN blocks (mirrors Model.forward skip connections)
        h, _ = self.backbone.l1(h)
        h1   = h
        h, _ = self.backbone.l2(h)
        h, _ = self.backbone.l3(h + h1)

        h, _ = self.backbone.l4(h)
        h4   = h
        h, _ = self.backbone.l5(h)
        h, _ = self.backbone.l6(h + h4)

        h, _ = self.backbone.l7(h)
        h7   = h
        h, _ = self.backbone.l8(h)
        h, _ = self.backbone.l9(h + h7)

        # Global average pool over (T, V) then pool persons
        c_new = h.size(1)
        h = h.view(N, M, c_new, -1)
        feat = h.mean(3).mean(1)                          # (N, 256)
        feat = self.backbone.drop_out(feat)
        return feat

    def forward(
        self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns classification logits (B, num_class)."""
        logits, _ = self.backbone(x)
        return logits

    def load_pretrained(
        self,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[list, list]:
        """Loads a pretrained Hyper-GCN checkpoint (bare state_dict or wrapped)."""
        raw = torch.load(checkpoint_path, map_location=device)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

        # Strip common key prefixes produced by DataParallel or save wrappers
        for prefix in ("module.", "backbone.", "model."):
            if any(k.startswith(prefix) for k in state):
                state = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state.items()
                }

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        return missing, unexpected

    def get_classifier_weights(
        self,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Returns (weight, bias) from the fc head for warm-starting direct_action."""
        fc = getattr(self.backbone, "fc", None)
        if isinstance(fc, nn.Linear):
            return fc.weight, fc.bias
        return None
