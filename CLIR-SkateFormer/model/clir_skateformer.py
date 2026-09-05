"""CLIR-SkateFormer: Concept Logic Iterative Refinement SkateFormer.

Kiến trúc chính:
  1. SkateFormer Backbone: Trích xuất feature map → concept logits z^0
  2. IterativeLogicReasoner (K=2 loops):
     - Mỗi vòng t: CRLReasoner(z^t) → (r^t, y^t) → ConceptUpdateCell → z^{t+1}
     - Final pass: CRLReasoner(z^K) → y^K (FINAL prediction)
  3. Loss System (Staged Training):
     - Phase 1 (Epoch 1-30):  L_act^K + L_sup (no L_reg)
     - Phase 2 (Epoch 31+):   L_act^K + L_sup + L_reg

Điểm đột phá:
  - Logic Evidence (r^t) từ DNF rules dẫn dắt concept refinement (top-down)
  - ActionHead KHÔNG nhận feature bypass (strict logic interpretability)
  - Binarization chỉ ở LiteralLayer, z^t luôn là logit liên tục
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SkateFormer import SkateFormer_
from .crl import CRLReasoner

# ─── Hằng số mặc định ────────────────────────────────────────────────────────

_N_CONCEPTS_SPATIAL  = 52   # concepts không gian sau khi loại bỏ null concept
_N_CONCEPTS_TEMPORAL = 20   # concepts thời gian sau khi loại bỏ null concept
_N_CONCEPTS          = 72   # = 52 + 20 (khớp với feeder_ntu.py remove_null_concept=True)
_N_ACTIONS           = 120  # NTU RGB+D 120
_LOGIC_DIM           = 512  # = r1(256) + r2a(128) + r2b(128)


# ─────────────────────────────────────────────────────────────────────────────
# ConceptUpdateCell
# ─────────────────────────────────────────────────────────────────────────────

class ConceptUpdateCell(nn.Module):
    """Logic-Guided Concept Update Cell (Top-Down Feedback).

    Thay vì KG tĩnh, dùng Logic Evidence r^t (từ DNF rules) để tinh chỉnh z^t.

    Cơ chế:
        z^t (concept logits) ──┐
                               ├──► Concat [z^t, Proj(r^t)] ──► MLP ──► Δz^t
        r^t (logic evidence) ──┘      (72 + 72 = 144)

        z^{t+1} = z^t + β^t · Δz^t

    Args:
        num_concepts: Số concepts (mặc định 72).
        logic_dim:    Chiều của Logic Evidence r^t (mặc định 512 = R1+R2a+R2b).
        hidden_dim:   Chiều ẩn của MLP (mặc định 256).
        dropout:      Dropout rate (mặc định 0.1).
    """

    def __init__(
        self,
        num_concepts: int = _N_CONCEPTS,
        logic_dim:    int = _LOGIC_DIM,
        hidden_dim:   int = 256,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.logic_dim    = int(logic_dim)

        # Project logic evidence từ 512 → 72
        self.logic_proj = nn.Sequential(
            nn.Linear(logic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_concepts),   # 256 → 72
        )

        # Update MLP: [z^t (72), r_proj (72)] → Δz^t (72)
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * num_concepts, hidden_dim),   # 144 → 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),        # 256 → 72
        )

    def forward(self, z_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        """Tính residual update Δz^t.

        Args:
            z_t: Concept logits hiện tại [B, num_concepts]  (liên tục)
            r_t: Logic Evidence từ CRLReasoner [B, logic_dim]

        Returns:
            delta_z: [B, num_concepts]  (residual, chưa nhân β)
        """
        r_proj = self.logic_proj(r_t)                    # [B, 72]
        cell_input = torch.cat([z_t, r_proj], dim=-1)    # [B, 144]
        return self.update_mlp(cell_input)                # [B, 72]


# ─────────────────────────────────────────────────────────────────────────────
# IterativeLogicReasoner
# ─────────────────────────────────────────────────────────────────────────────

class IterativeLogicReasoner(nn.Module):
    """Vòng lặp suy luận lặp K=2 với Shared CRLReasoner.

    Luồng dữ liệu (K=2):
      z^0 → CRLReasoner → (r^0, y^0) → ConceptUpdateCell → Δz^0 → z^1
      z^1 → CRLReasoner → (r^1, y^1) → ConceptUpdateCell → Δz^1 → z^2
      z^2 → CRLReasoner [FINAL] → y^2  ← FINAL PREDICTION

    Trọng số CRLReasoner và ConceptUpdateCell được chia sẻ qua tất cả vòng.

    Args:
        num_concepts:   Số concepts (72).
        num_actions:    Số lớp hành động (120).
        num_loops:      Số vòng lặp tinh chỉnh K (mặc định 2).
        r1, r2a, r2b:   Kích thước rule banks.
        beta_max:       Giá trị tối đa của β (mặc định 0.5 ở Phase 2).
        beta_init:      Giá trị khởi tạo của β (mặc định 0.1 ở Phase 1).
        literal_mode:   'soft' hoặc 'ste'.
        temperature:    Nhiệt độ STE.
        hidden_dim:     Chiều ẩn của ConceptUpdateCell.
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        num_concepts: int   = _N_CONCEPTS,
        num_actions:  int   = _N_ACTIONS,
        num_loops:    int   = 2,
        r1:           int   = 256,
        r2a:          int   = 128,
        r2b:          int   = 128,
        beta_max:     float = 0.5,
        beta_init:    float = 0.1,
        literal_mode: str   = "ste",
        temperature:  float = 2.0,
        hidden_dim:   int   = 256,
        dropout:      float = 0.1,
        random_feedback: bool = False,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_actions  = int(num_actions)
        self.num_loops    = max(0, int(num_loops))  # Cho phép num_loops=0 (A2: True No-Loop Baseline)
        self.beta_max     = float(beta_max)
        self.random_feedback = bool(random_feedback)
        logic_dim = r1 + r2a + r2b

        # ── Shared CRLReasoner (dùng lại qua tất cả vòng lặp) ──────────────
        self.crl = CRLReasoner(
            n_concepts=num_concepts,
            n_actions=num_actions,
            r1=r1,
            r2a=r2a,
            r2b=r2b,
            literal_mode=literal_mode,
            temperature=temperature,
        )

        # ── Shared ConceptUpdateCell ─────────────────────────────────────────
        self.update_cell = ConceptUpdateCell(
            num_concepts=num_concepts,
            logic_dim=logic_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ── Learnable β per loop ─────────────────────────────────────────────
        # β^t = sigmoid(raw_β^t) × β_max
        # Xử lý an toàn khi num_loops = 0 (A2: True No-Loop) hoặc beta_max <= 0
        if self.num_loops == 0:
            self.raw_beta = nn.Parameter(torch.empty(0, dtype=torch.float32))
        elif self.beta_max <= 0.0:
            raw_init = torch.tensor(-10.0, dtype=torch.float32)
            self.raw_beta = nn.Parameter(
                raw_init.unsqueeze(0).expand(self.num_loops).clone()
            )
        else:
            ratio = min(max(beta_init / self.beta_max, 1e-4), 1.0 - 1e-4)
            raw_init = torch.logit(torch.tensor(ratio, dtype=torch.float32))
            self.raw_beta = nn.Parameter(
                raw_init.unsqueeze(0).expand(self.num_loops).clone()
            )

    # ── Properties để điều chỉnh LiteralLayer ────────────────────────────────
    @property
    def literal_mode(self) -> str:
        return self.crl.literal_mode

    @literal_mode.setter
    def literal_mode(self, value: str):
        self.crl.literal_mode = value

    @property
    def temperature(self) -> float:
        return self.crl.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.crl.temperature = float(value)

    def get_betas(self) -> torch.Tensor:
        """Lấy β^t cho mỗi vòng lặp t ∈ [0, K-1] — shape [K]."""
        if self.num_loops == 0:
            return self.raw_beta
        if self.beta_max <= 0.0:
            return torch.zeros_like(self.raw_beta)
        return torch.sigmoid(self.raw_beta) * self.beta_max

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        z0: torch.Tensor,
        return_intermediates: bool = True,
    ) -> Dict:
        """Thực hiện K vòng lặp suy luận + 1 final pass.

        Args:
            z0:                   Initial concept logits [B, num_concepts]
            return_intermediates: Có lưu các y^t và z^t trung gian không

        Returns:
            dict với keys:
              - y_final:       Final action logits y^K [B, num_actions]
              - z_final:       Final concept logits z^K [B, num_concepts]
              - z_all:         List [z^0, z^1, ..., z^K]
              - y_all:         List [y^0, y^1, ..., y^{K-1}]  (trung gian)
              - betas:         [K] tensor
        """
        betas = self.get_betas()   # [K]
        z_t = z0
        z_all = [z0] if return_intermediates else []
        y_all = []

        # ── K vòng lặp tinh chỉnh ────────────────────────────────────────────
        for t in range(self.num_loops):
            # Suy luận logic: z^t → (r^t, y^t)
            r_t, y_t = self.crl(z_t)
            y_all.append(y_t)

            # A5 Ablation: Random Logic Feedback (thay r_t bằng tensor ngẫu nhiên)
            if self.random_feedback:
                feedback_r = torch.rand_like(r_t)
            else:
                feedback_r = r_t

            # Tính residual: Δz^t
            delta_t = self.update_cell(z_t, feedback_r)

            # Cập nhật concept logits: z^{t+1} = z^t + β^t · Δz^t
            z_t = z_t + betas[t] * delta_t

            if return_intermediates:
                z_all.append(z_t)

        # ── Final pass (không cập nhật thêm) ─────────────────────────────────
        _, y_final = self.crl(z_t)

        return {
            "y_final": y_final,
            "z_final": z_t,
            "z_all":   z_all,
            "y_all":   y_all,   # [y^0, ..., y^{K-1}]
            "betas":   betas,
        }

    def rule_sparsity_loss(self) -> torch.Tensor:
        """Ủy quyền cho CRLReasoner."""
        return self.crl.rule_sparsity_loss()


# ─────────────────────────────────────────────────────────────────────────────
# CLIRSkateFormer
# ─────────────────────────────────────────────────────────────────────────────

class CLIRSkateFormer(nn.Module):
    """CLIR-SkateFormer: Concept Logic Iterative Refinement for Skeleton AR.

    Tích hợp SkateFormer backbone với IterativeLogicReasoner để thực hiện
    tinh chỉnh concept lặp K=2 vòng được dẫn dắt bởi logic evidence.

    Luồng dữ liệu:
        X ∈ R^(B × C × T × V × M)
          → SkateFormer.forward_features_from_input(X)
          → feature_map ∈ R^(B × D × T' × V')
          → spatial_pool_score, temporal_pool_score (attention)
          → z^0 = [z_s (52); z_t (20)] ∈ R^(B × 72)
          → IterativeLogicReasoner (K=2)
          → y^K ∈ R^(B × 120)  [FINAL PREDICTION]

    Args:
        num_classes:         Số classes (120).
        num_concepts:        Tổng số concepts = spatial + temporal (72).
        spatial_concepts:    Số spatial concepts (52).
        temporal_concepts:   Số temporal concepts (20).
        num_people:          Số người trong skeleton (2).
        num_points:          Số keypoints (24).
        num_frames:          Số frames (64).
        backbone_weights:    Path đến SkateFormer pretrained weights.
        concept_names:       Danh sách tên concepts.
        num_loops:           Số vòng lặp K (2).
        r1, r2a, r2b:        Kích thước rule banks (256, 128, 128).
        beta_max:            Giá trị tối đa của β (0.5).
        beta_init:           Khởi tạo của β (0.1).
        literal_mode:        'ste' (mặc định Pure STE) hoặc 'soft'.
        temperature:         Nhiệt độ STE (mặc định 2.0, annealing dần tới 10.0).
        hidden_dim:          Chiều ẩn ConceptUpdateCell (256).
        dropout:             Dropout rate (0.1).
        lambda_sup:          Trọng số L_sup (1.0 cố định).
        lambda_reg:          Trọng số L_reg (mặc định 0.005 cho rule sparsity + loop stability).
        concept_negative_weight: Trọng số negative trong Asymmetric BCE (0.15).
        asymmetric_gamma_neg:    γ- cho Asymmetric BCE (2.0).
        return_aux:          Có trả về thông tin phụ không (True khi train).
    """

    def __init__(
        self,
        num_classes:              int   = _N_ACTIONS,
        num_concepts:             int   = _N_CONCEPTS,
        spatial_concepts:         int   = _N_CONCEPTS_SPATIAL,
        temporal_concepts:        int   = _N_CONCEPTS_TEMPORAL,
        num_people:               int   = 2,
        num_points:               int   = 24,
        num_frames:               int   = 64,
        backbone_weights:         Optional[str]       = None,
        concept_names:            Optional[List[str]] = None,
        # Reasoner
        num_loops:    int   = 2,
        r1:           int   = 256,
        r2a:          int   = 128,
        r2b:          int   = 128,
        beta_max:     float = 0.5,
        beta_init:    float = 0.1,
        literal_mode: str   = "ste",
        temperature:  float = 2.0,
        hidden_dim:   int   = 256,
        dropout:      float = 0.1,
        # Loss weights
        lambda_sup:              float = 1.0,
        lambda_reg:              float = 0.005,
        concept_negative_weight: float = 0.15,
        concept_positive_weight: float = 1.0,
        asymmetric_gamma_neg:    float = 2.0,
        return_aux:              bool  = True,
        # Ablations
        use_feature_bypass:      bool  = False,  # A6: CLIR-Bypass
        random_feedback:         bool  = False,  # A5: Random Logic Feedback
        **kwargs,
    ):
        super().__init__()

        if spatial_concepts + temporal_concepts != num_concepts:
            raise ValueError(
                f"spatial_concepts({spatial_concepts}) + temporal_concepts({temporal_concepts})"
                f" != num_concepts({num_concepts})"
            )

        self.num_classes         = int(num_classes)
        self.num_concepts        = int(num_concepts)
        self.spatial_concepts    = int(spatial_concepts)
        self.temporal_concepts   = int(temporal_concepts)
        self.num_loops           = max(0, int(num_loops))
        self.lambda_sup          = float(lambda_sup)
        self.lambda_reg          = float(lambda_reg)
        self.concept_neg_weight  = float(concept_negative_weight)
        self.concept_pos_weight  = float(concept_positive_weight)
        self.gamma_neg           = float(asymmetric_gamma_neg)
        self.return_aux          = bool(return_aux)
        self.use_feature_bypass  = bool(use_feature_bypass)
        self.accepts_concepts    = True   # flag cho training loop

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = SkateFormer_(
            num_classes=num_classes,
            num_people=num_people,
            num_points=num_points,
            num_frames=num_frames,
            **kwargs,
        )
        self._load_backbone_weights(backbone_weights)
        feature_dim = self.backbone.head.in_features

        # A6: CLIR-Bypass Head (kết nối trực tiếp từ feature map vào ActionHead)
        if self.use_feature_bypass:
            self.bypass_head = nn.Linear(feature_dim, self.num_classes)
        else:
            self.bypass_head = None

        # ── Concept Heads (Spatial + Temporal) ───────────────────────────────
        self.spatial_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)

        self.spatial_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.spatial_concepts),
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.temporal_concepts),
        )

        # concept_names
        if concept_names is None:
            concept_names = [f"concept_{i}" for i in range(self.num_concepts)]
        self.concept_names = list(concept_names)

        # ── IterativeLogicReasoner ────────────────────────────────────────────
        self.reasoner = IterativeLogicReasoner(
            num_concepts=self.num_concepts,
            num_actions=self.num_classes,
            num_loops=self.num_loops,
            r1=r1,
            r2a=r2a,
            r2b=r2b,
            beta_max=beta_max,
            beta_init=beta_init,
            literal_mode=literal_mode,
            temperature=temperature,
            hidden_dim=hidden_dim,
            dropout=dropout,
            random_feedback=random_feedback,
        )

    # ── Backbone weight loading ───────────────────────────────────────────────

    def _load_backbone_weights(self, checkpoint_path: Optional[str]):
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"[Warning] Backbone checkpoint not found: {path}. Initializing randomly.")
            return
        checkpoint = torch.load(str(path), map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError("Unsupported SkateFormer checkpoint format")

        backbone_state = self.backbone.state_dict()
        mapped = OrderedDict()
        for key, value in checkpoint.items():
            clean_key = key.replace("module.", "", 1)
            if clean_key.startswith("backbone."):
                clean_key = clean_key[len("backbone."):]
            if clean_key in backbone_state:
                if backbone_state[clean_key].shape == value.shape:
                    mapped[clean_key] = value
                else:
                    print(
                        f"[Warning] Skipping {clean_key} due to shape mismatch: "
                        f"checkpoint {value.shape} vs model {backbone_state[clean_key].shape}"
                    )
        missing, unexpected = self.backbone.load_state_dict(mapped, strict=False)
        print(
            f"Loaded {len(mapped)} SkateFormer backbone tensors from {path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    # ── Pooling ───────────────────────────────────────────────────────────────

    def _pool_features(
        self, feature_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention pooling dọc theo spatial và temporal dimensions.

        Args:
            feature_map: [B, D, T', V']

        Returns:
            spatial_feat:  [B, D]
            temporal_feat: [B, D]
        """
        # Spatial: pool over T' dimension → [B, D, V']
        spatial_tokens = feature_map.mean(dim=2)                              # [B, D, V']
        spatial_attn   = torch.softmax(self.spatial_pool_score(spatial_tokens), dim=-1)  # [B, 1, V']
        spatial_feat   = (spatial_tokens * spatial_attn).sum(dim=-1)         # [B, D]

        # Temporal: pool over V' dimension → [B, D, T']
        temporal_tokens = feature_map.mean(dim=3)                              # [B, D, T']
        temporal_attn   = torch.softmax(self.temporal_pool_score(temporal_tokens), dim=-1)
        temporal_feat   = (temporal_tokens * temporal_attn).sum(dim=-1)       # [B, D]

        return spatial_feat, temporal_feat

    # ── Loss Functions ────────────────────────────────────────────────────────

    def _asymmetric_bce(
        self,
        logits:    torch.Tensor,
        targets:   torch.Tensor,
        mask:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Asymmetric BCE loss (Mục 8.1 — CLIR plan).

        L_c(z, ĉ) = Σ r_ic [ĉ_ic · softplus(-z_ic)
                             + (1-ĉ_ic) · (1-σ(z_ic))^γ- · softplus(z_ic)] / Σ r_ic

        Với r_ic: positive_weight=1.0, negative_weight=0.15.

        Args:
            logits:  [B, n_concepts]
            targets: [B, n_concepts] ∈ {0, 1}  (soft labels được cho phép)
            mask:    [B, n_concepts] binary mask (None = dùng tất cả)

        Returns:
            scalar loss
        """
        targets = targets.float().clamp(0.0, 1.0)
        probs = torch.sigmoid(logits)

        pos_loss = F.softplus(-logits) * targets
        neg_loss = (
            F.softplus(logits)
            * (1.0 - probs).pow(self.gamma_neg)
            * (1.0 - targets)
        )

        weights = (
            targets * self.concept_pos_weight
            + (1.0 - targets) * self.concept_neg_weight
        )
        if mask is not None:
            weights = weights * mask.float()

        loss = pos_loss + neg_loss
        denom = weights.sum().clamp_min(1.0)
        return (loss * weights).sum() / denom

    def compute_sup_loss(
        self,
        z0:              torch.Tensor,
        z_all:           List[torch.Tensor],
        y_all:           List[torch.Tensor],
        concept_targets: torch.Tensor,
        action_labels:   torch.Tensor,
        concept_mask:    Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tính L_sup = L_c^K + 0.5 · L_c^0 + 0.3 · (1/K) Σ L_act^t.

        Args:
            z0:              Initial concept logits [B, 72]
            z_all:           [z^0, z^1, ..., z^K]
            y_all:           [y^0, ..., y^{K-1}] (intermediate action logits)
            concept_targets: [B, 72]
            action_labels:   [B]
            concept_mask:    [B, 72] optional

        Returns:
            dict với: l_c_K, l_c_0, l_act_aux, l_sup
        """
        z_K = z_all[-1]   # Final concept logits

        # L_c^K: concept loss ở bước cuối
        l_c_K = self._asymmetric_bce(z_K, concept_targets, mask=concept_mask)

        # L_c^0: concept loss sớm để ép backbone học concept chuẩn
        l_c_0 = self._asymmetric_bce(z0, concept_targets, mask=concept_mask)

        # L_act^t: auxiliary action loss trên các bước trung gian
        l_act_aux = z0.new_zeros(())
        if len(y_all) > 0:
            for y_t in y_all:
                l_act_aux = l_act_aux + F.cross_entropy(y_t, action_labels)
            l_act_aux = l_act_aux / len(y_all)

        # L_sup = L_c^K + 0.5 · L_c^0 + 0.3 · L_act_aux
        l_sup = l_c_K + 0.5 * l_c_0 + 0.3 * l_act_aux

        return {
            "l_c_K":     l_c_K,
            "l_c_0":     l_c_0,
            "l_act_aux": l_act_aux,
            "l_sup":     l_sup,
        }

    def compute_reg_loss(
        self,
        z_all: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Tính L_reg = L_sparse + L_stable.

        L_sparse: Ép thưa các trọng số luật
        L_stable: Giữ cho tinh chỉnh hội tụ, không dao động

        Args:
            z_all: [z^0, z^1, ..., z^K]

        Returns:
            dict với: l_sparse, l_stable, l_reg
        """
        # L_sparse: từ CRLReasoner
        l_sparse = self.reasoner.rule_sparsity_loss()

        # L_stable: Σ ||σ(z^{t+1}) - σ(z^t)||_1 / K
        l_stable = z_all[0].new_zeros(())
        K = len(z_all) - 1   # số bước update thực sự
        if K > 0:
            for t in range(K):
                diff = (torch.sigmoid(z_all[t + 1]) - torch.sigmoid(z_all[t])).abs()
                l_stable = l_stable + diff.mean()
            l_stable = l_stable / K

        l_reg = l_sparse + l_stable

        return {
            "l_sparse": l_sparse,
            "l_stable": l_stable,
            "l_reg":    l_reg,
        }

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input:                  torch.Tensor,
        index_t:                Optional[torch.Tensor] = None,
        concept_targets:        Optional[torch.Tensor] = None,
        concept_target_mask:    Optional[torch.Tensor] = None,
        action_labels:          Optional[torch.Tensor] = None,
        return_trace:           bool = False,
    ) -> Union[torch.Tensor, Dict]:
        """Forward pass của CLIRSkateFormer.

        Args:
            input:               [B, C, T, V, M]
            index_t:             Temporal indices (cho SkateFormer)
            concept_targets:     [B, 72] nhãn concept (0/1)
            concept_target_mask: [B, 72] mask nhãn
            action_labels:       [B] nhãn action
            return_trace:        Có trả về intermediate states không

        Returns:
            Khi không train và không có labels: tensor y_final [B, n_actions]
            Khi train hoặc có labels: dict với loss và intermediate states
        """
        # ── Backbone feature extraction ───────────────────────────────────────
        feature_map = self.backbone.forward_features_from_input(input, index_t=index_t)
        # [B, D, T', V']

        spatial_feat, temporal_feat = self._pool_features(feature_map)

        # ── Concept logits z^0 ────────────────────────────────────────────────
        z_s = self.spatial_head(spatial_feat)      # [B, 52]
        z_t = self.temporal_head(temporal_feat)    # [B, 20]
        z0  = torch.cat([z_s, z_t], dim=-1)       # [B, 72]

        # ── Iterative Logic Reasoning ─────────────────────────────────────────
        reasoner_out = self.reasoner(z0, return_intermediates=True)
        y_final = reasoner_out["y_final"]     # [B, 120]
        z_final = reasoner_out["z_final"]     # [B, 72]
        z_all   = reasoner_out["z_all"]       # [z^0, ..., z^K]
        y_all   = list(reasoner_out["y_all"]) # [y^0, ..., y^{K-1}]
        betas   = reasoner_out["betas"]       # [K]

        # A6: CLIR-Bypass connection (thêm trực tiếp từ visual features vào ActionHead)
        if self.use_feature_bypass and self.bypass_head is not None:
            bypass_feat = 0.5 * (spatial_feat + temporal_feat)
            bypass_logits = self.bypass_head(bypass_feat)
            y_final = y_final + bypass_logits
            y_all = [y + bypass_logits for y in y_all]

        # ── Inference mode (không train) ──────────────────────────────────────
        if not self.return_aux and concept_targets is None and not self.training:
            return y_final

        # ── Tính Losses ───────────────────────────────────────────────────────
        output = {
            "action_logits":         y_final,
            "concept_logits":        z_final,
            "initial_concept_logits": z0,
            "spatial_logits":        z_s,
            "temporal_logits":       z_t,
            "all_action_logits":     y_all,
            "betas":                 betas,
        }

        if return_trace:
            output["z_all"] = z_all

        if concept_targets is not None and action_labels is not None:
            targets = concept_targets.float().to(z0.device)

            # L_sup
            sup_losses = self.compute_sup_loss(
                z0=z0,
                z_all=z_all,
                y_all=y_all,
                concept_targets=targets,
                action_labels=action_labels,
                concept_mask=concept_target_mask,
            )
            output.update(sup_losses)

            # L_reg (chỉ khi lambda_reg > 0, tức Phase 2)
            if self.lambda_reg > 0.0:
                reg_losses = self.compute_reg_loss(z_all)
                output.update(reg_losses)
                output["aux_loss"] = (
                    self.lambda_sup * sup_losses["l_sup"]
                    + self.lambda_reg * reg_losses["l_reg"]
                )
            else:
                output["l_reg"] = z0.new_zeros(())
                output["aux_loss"] = self.lambda_sup * sup_losses["l_sup"]

        elif concept_targets is not None:
            # Có concept targets nhưng không có action labels
            targets = concept_targets.float().to(z0.device)
            l_c_K = self._asymmetric_bce(z_final, targets, mask=concept_target_mask)
            l_c_0 = self._asymmetric_bce(z0, targets, mask=concept_target_mask)
            output["l_c_K"] = l_c_K
            output["l_c_0"] = l_c_0
            output["aux_loss"] = l_c_K + 0.5 * l_c_0

        return output

    # ── Convenience methods ───────────────────────────────────────────────────

    def set_phase(self, phase: int, temperature: Optional[float] = None):
        """Chuyển đổi training phase.

        phase=1: Concept warm-up (soft mode, beta_max=0.1, lambda_reg=0)
        phase=2: STE + Rule sparsification (ste mode, beta_max=0.5, lambda_reg=0.005)

        Args:
            phase:       1 hoặc 2.
            temperature: Ghi đè temperature (None = dùng mặc định).
        """
        if phase == 1:
            self.reasoner.literal_mode = "soft"
            if self.reasoner.num_loops > 0 and self.reasoner.beta_max > 0.0:
                self.reasoner.beta_max = 0.1
                with torch.no_grad():
                    # Hiệu chỉnh raw_beta sao cho beta ≈ 0.10 ở Phase 1 (thay vì tụt xuống 0.02)
                    target_ratio = 0.95
                    self.reasoner.raw_beta.fill_(float(torch.logit(torch.tensor(target_ratio))))
            self.lambda_reg = 0.0
            if temperature is not None:
                self.reasoner.temperature = temperature
        elif phase == 2:
            self.reasoner.literal_mode = "ste"
            if self.reasoner.num_loops > 0 and self.reasoner.beta_max > 0.0:
                old_beta_max = self.reasoner.beta_max
                new_beta_max = 0.5
                with torch.no_grad():
                    # Bảo toàn giá trị beta hiện tại từ Phase 1 (≈ 0.10), chuyển trơn sang beta_max = 0.5
                    current_beta = torch.sigmoid(self.reasoner.raw_beta) * old_beta_max
                    new_ratio = (current_beta / new_beta_max).clamp(1e-4, 1.0 - 1e-4)
                    self.reasoner.raw_beta.copy_(torch.logit(new_ratio))
                self.reasoner.beta_max = new_beta_max
            self.lambda_reg = 0.005
            if temperature is not None:
                self.reasoner.temperature = temperature
        else:
            raise ValueError(f"phase phải là 1 hoặc 2, nhận được: {phase}")
        print(
            f"[CLIRSkateFormer] Switched to Phase {phase}: "
            f"mode={self.reasoner.literal_mode!r}, "
            f"beta_max={self.reasoner.beta_max}, "
            f"lambda_reg={self.lambda_reg}, "
            f"temperature={self.reasoner.temperature}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def CLIR_SkateFormer(**kwargs) -> CLIRSkateFormer:
    """Factory function cho CLIRSkateFormer."""
    return CLIRSkateFormer(**kwargs)
