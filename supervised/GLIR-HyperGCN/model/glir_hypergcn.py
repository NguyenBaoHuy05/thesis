"""GLIR-HyperGCN: Grounded Logic Iterative Refinement với backbone HyperGCN.

Nâng cấp trực tiếp từ CLIR-SkateFormer (`d:\\Thesis\\CLIR-SkateFormer\\model\\clir_skateformer.py`)
theo `Architecture/10_CLIR-SkateFormer_v2_Proposal.md`, đổi backbone sang HyperGCN
(`d:\\Thesis\\Hyper-GCN`, bản làm sạch tại `model/hypergcn.py`) và triển khai:

  G1 — BodyPartGroundedConceptHeads: concept không gian được dự đoán RIÊNG cho từng
       nhóm khớp NTU-25 (Head/Hand/Arm/Hip/Leg/Foot), khớp 1-1 với part-grouped L1
       ở tầng logic (`crl/conjunction_layer.py::get_part_groups`) — khắc phục khoảng
       trống G1 (concept head cũ dùng chung 1 vector pooled toàn cục cho mọi concept).
  G2 — LightConceptGrounding: CLIP đóng băng, chỉ học 1 projection nhỏ, text anchor
       encode 1 lần lúc khởi tạo (`crl/light_grounding.py`).
  G3 — set_progress(): lịch annealing liên tục (temperature/beta_max/lambda_reg theo
       progress ∈ [0,1]) thay cho set_phase() rời rạc — set_phase() vẫn giữ lại để
       chạy cấu hình V2 (chưa bật G3); set_progress() dùng cho cấu hình V3.

Vòng lặp logic (ConceptUpdateCell + IterativeLogicReasoner) và hệ loss (Asymmetric
BCE, rule sparsity, loop stability) giữ NGUYÊN từ CLIR-SkateFormer — cơ chế này
backbone-agnostic, không cần thay đổi khi đổi từ SkateFormer sang HyperGCN.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergcn import HyperGCNModel
from .crl import CRLReasoner
from .crl.conjunction_layer import get_part_groups
from .crl.light_grounding import LightConceptGrounding

# ─── Hằng số mặc định ────────────────────────────────────────────────────────

_N_CONCEPTS_SPATIAL  = 52
_N_CONCEPTS_TEMPORAL = 20
_N_CONCEPTS          = 72
_N_ACTIONS           = 120
_LOGIC_DIM           = 512

# NTU 25-joint chuẩn (0-indexed), nhất quán với HyperGraph.inward_ori_index trong
# model/hypergcn.py (1-indexed trong file gốc, đã trừ 1). Phủ đủ 25 khớp, không
# chồng lấn. Thứ tự nhóm PHẢI khớp với get_part_groups() ở conjunction_layer.py
# (head, hand, arm, hip, leg, foot) để concept đầu ra đúng slice.
NTU25_BODY_PART_JOINTS: Dict[str, List[int]] = {
    "head": [2, 3, 20],                       # Neck, Head, SpineShoulder
    "hand": [6, 7, 10, 11, 21, 22, 23, 24],    # WristL/R, HandL/R, HandTipL/R, ThumbL/R
    "arm":  [4, 5, 8, 9],                      # ShoulderL/R, ElbowL/R
    "hip":  [0, 1, 12, 16],                    # SpineBase, SpineMid, HipL, HipR
    "leg":  [13, 17],                          # KneeL, KneeR
    "foot": [14, 15, 18, 19],                  # AnkleL/R, FootL/R
}


# ─────────────────────────────────────────────────────────────────────────────
# ConceptUpdateCell / IterativeLogicReasoner — kế thừa nguyên trạng từ CLIR
# ─────────────────────────────────────────────────────────────────────────────

class ConceptUpdateCell(nn.Module):
    """Logic-Guided Concept Update Cell (Top-Down Feedback). Giống hệt CLIR-SkateFormer —
    backbone-agnostic, chỉ nhận z_t (concept logits) và r_t (logic evidence)."""

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

        self.logic_proj = nn.Sequential(
            nn.Linear(logic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_concepts),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * num_concepts, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, z_t: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        r_proj = self.logic_proj(r_t)
        cell_input = torch.cat([z_t, r_proj], dim=-1)
        return self.update_mlp(cell_input)


class IterativeLogicReasoner(nn.Module):
    """Vòng lặp suy luận lặp K với Shared CRLReasoner. Giống hệt CLIR-SkateFormer."""

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
        literal_mode: str   = "soft",
        temperature:  float = 5.0,
        hidden_dim:   int   = 256,
        dropout:      float = 0.1,
        random_feedback: bool = False,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_actions  = int(num_actions)
        self.num_loops    = max(0, int(num_loops))
        self.beta_max     = float(beta_max)
        self.random_feedback = bool(random_feedback)
        logic_dim = r1 + r2a + r2b

        self.crl = CRLReasoner(
            n_concepts=num_concepts, n_actions=num_actions,
            r1=r1, r2a=r2a, r2b=r2b,
            literal_mode=literal_mode, temperature=temperature,
        )
        self.update_cell = ConceptUpdateCell(
            num_concepts=num_concepts, logic_dim=logic_dim,
            hidden_dim=hidden_dim, dropout=dropout,
        )

        if self.num_loops == 0:
            self.raw_beta = nn.Parameter(torch.empty(0, dtype=torch.float32))
        elif self.beta_max <= 0.0:
            raw_init = torch.tensor(-10.0, dtype=torch.float32)
            self.raw_beta = nn.Parameter(raw_init.unsqueeze(0).expand(self.num_loops).clone())
        else:
            ratio = min(max(beta_init / self.beta_max, 1e-4), 1.0 - 1e-4)
            raw_init = torch.logit(torch.tensor(ratio, dtype=torch.float32))
            self.raw_beta = nn.Parameter(raw_init.unsqueeze(0).expand(self.num_loops).clone())

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
        if self.num_loops == 0:
            return self.raw_beta
        if self.beta_max <= 0.0:
            return torch.zeros_like(self.raw_beta)
        return torch.sigmoid(self.raw_beta) * self.beta_max

    def forward(self, z0: torch.Tensor, return_intermediates: bool = True) -> Dict:
        betas = self.get_betas()
        z_t = z0
        z_all = [z0] if return_intermediates else []
        y_all = []

        for t in range(self.num_loops):
            r_t, y_t = self.crl(z_t)
            y_all.append(y_t)
            feedback_r = torch.rand_like(r_t) if self.random_feedback else r_t
            delta_t = self.update_cell(z_t, feedback_r)
            z_t = z_t + betas[t] * delta_t
            if return_intermediates:
                z_all.append(z_t)

        _, y_final = self.crl(z_t)
        return {"y_final": y_final, "z_final": z_t, "z_all": z_all, "y_all": y_all, "betas": betas}

    def rule_sparsity_loss(self) -> torch.Tensor:
        return self.crl.rule_sparsity_loss()


# ─────────────────────────────────────────────────────────────────────────────
# G1: BodyPartGroundedConceptHeads
# ─────────────────────────────────────────────────────────────────────────────

class BodyPartGroundedConceptHeads(nn.Module):
    """Mỗi nhóm bộ phận pool RIÊNG trên đúng tập khớp NTU-25 của mình rồi mới suy
    ra concept của nhóm đó — khớp 1-1 với part_groups ở tầng logic (get_part_groups)."""

    def __init__(self, feature_dim: int, spatial_concepts: int = _N_CONCEPTS_SPATIAL, dropout: float = 0.1):
        super().__init__()
        all_groups = get_part_groups(spatial_concepts + _N_CONCEPTS_TEMPORAL)
        spatial_groups = [g for g in all_groups if g[0] != "temporal"]

        self.group_order = [g[0] for g in spatial_groups]
        self.concept_slices: Dict[str, Tuple[int, int]] = {g[0]: (g[1], g[2]) for g in spatial_groups}

        missing = set(self.group_order) - set(NTU25_BODY_PART_JOINTS)
        if missing:
            raise ValueError(f"NTU25_BODY_PART_JOINTS thieu nhom: {missing}")

        total = sum(end - start for start, end in self.concept_slices.values())
        if total != spatial_concepts:
            raise ValueError(f"Tong concept theo nhom ({total}) != spatial_concepts ({spatial_concepts})")

        self.pool_scores = nn.ModuleDict({
            g: nn.Conv1d(feature_dim, 1, kernel_size=1) for g in self.group_order
        })
        self.heads = nn.ModuleDict({
            g: nn.Sequential(
                nn.Linear(feature_dim, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, self.concept_slices[g][1] - self.concept_slices[g][0]),
            ) for g in self.group_order
        })

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """feature_map: [B, D, T', V] (V=25, ánh xạ 1:1 khớp NTU chuẩn)."""
        outs = []
        group_feats: Dict[str, torch.Tensor] = {}
        for g in self.group_order:
            joint_idx = NTU25_BODY_PART_JOINTS[g]
            sub = feature_map[:, :, :, joint_idx].mean(dim=2)            # [B, D, |joints_g|] (pool theo T')
            attn = torch.softmax(self.pool_scores[g](sub), dim=-1)       # [B, 1, |joints_g|]
            pooled = (sub * attn).sum(dim=-1)                            # [B, D] — CHỈ từ khớp thuộc nhóm g
            group_feats[g] = pooled
            outs.append(self.heads[g](pooled))
        z_spatial = torch.cat(outs, dim=-1)
        return z_spatial, group_feats


# ─────────────────────────────────────────────────────────────────────────────
# GLIRHyperGCN
# ─────────────────────────────────────────────────────────────────────────────

class GLIRHyperGCN(nn.Module):
    def __init__(
        self,
        num_classes:        int   = _N_ACTIONS,
        num_concepts:        int   = _N_CONCEPTS,
        spatial_concepts:    int   = _N_CONCEPTS_SPATIAL,
        temporal_concepts:   int   = _N_CONCEPTS_TEMPORAL,
        num_person:          int   = 2,
        num_point:           int   = 25,
        in_channels:         int   = 3,
        embedding_channels:  int   = 128,
        backbone_weights:    Optional[str]       = None,
        concept_names:       Optional[List[str]] = None,
        # Reasoner
        num_loops:    int   = 2,
        r1:           int   = 256,
        r2a:          int   = 128,
        r2b:          int   = 128,
        beta_max:     float = 0.5,
        beta_init:    float = 0.1,
        literal_mode: str   = "soft",
        temperature:  float = 5.0,
        hidden_dim:   int   = 256,
        dropout:      float = 0.1,
        # Loss weights
        lambda_sup:              float = 1.0,
        lambda_reg:              float = 0.0,
        lambda_ground:            float = 0.1,
        concept_negative_weight: float = 0.15,
        concept_positive_weight: float = 1.0,
        asymmetric_gamma_neg:    float = 2.0,
        return_aux:              bool  = True,
        # G2 — grounding
        use_grounding:      bool = True,
        clip_backbone:       str = "ViT-B/32",
        # Ablations (kế thừa từ CLIR)
        random_feedback:    bool = False,
        **kwargs,
    ):
        super().__init__()
        if spatial_concepts + temporal_concepts != num_concepts:
            raise ValueError(
                f"spatial_concepts({spatial_concepts}) + temporal_concepts({temporal_concepts})"
                f" != num_concepts({num_concepts})"
            )

        self.num_classes        = int(num_classes)
        self.num_concepts       = int(num_concepts)
        self.spatial_concepts   = int(spatial_concepts)
        self.temporal_concepts  = int(temporal_concepts)
        self.num_loops          = max(0, int(num_loops))
        self.lambda_sup         = float(lambda_sup)
        self.lambda_reg         = float(lambda_reg)
        self.lambda_ground       = float(lambda_ground)
        self.concept_neg_weight = float(concept_negative_weight)
        self.concept_pos_weight = float(concept_positive_weight)
        self.gamma_neg          = float(asymmetric_gamma_neg)
        self.return_aux         = bool(return_aux)
        self.accepts_concepts   = True
        self._progress          = 0.0

        # ── Backbone: HyperGCN ────────────────────────────────────────────────
        self.backbone = HyperGCNModel(
            num_class=num_classes, num_point=num_point, num_person=num_person,
            in_channels=in_channels, embedding_channels=embedding_channels,
            drop_out=dropout,
        )
        self._load_backbone_weights(backbone_weights)
        feature_dim = self.backbone.feat_dim

        # ── G1: Concept Heads theo bộ phận cơ thể ─────────────────────────────
        self.concept_heads = BodyPartGroundedConceptHeads(feature_dim, spatial_concepts, dropout)
        self.temporal_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.temporal_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, temporal_concepts),
        )

        if concept_names is None:
            concept_names = [f"concept_{i}" for i in range(self.num_concepts)]
        self.concept_names = list(concept_names)

        # ── G2: Semantic Grounding (nhẹ, CLIP đóng băng) ──────────────────────
        self.use_grounding = bool(use_grounding)
        if self.use_grounding:
            self.grounding = LightConceptGrounding(
                feature_dim=feature_dim, concept_names=self.concept_names,
                clip_backbone=clip_backbone,
            )
        else:
            self.grounding = None

        # ── IterativeLogicReasoner (kế thừa CLIR nguyên trạng) ────────────────
        self.reasoner = IterativeLogicReasoner(
            num_concepts=self.num_concepts, num_actions=self.num_classes,
            num_loops=self.num_loops, r1=r1, r2a=r2a, r2b=r2b,
            beta_max=beta_max, beta_init=beta_init,
            literal_mode=literal_mode, temperature=temperature,
            hidden_dim=hidden_dim, dropout=dropout,
            random_feedback=random_feedback,
        )

    # ── Backbone weight loading (giống pattern của CLIR) ──────────────────────

    def _load_backbone_weights(self, checkpoint_path: Optional[str]):
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"[Warning] HyperGCN checkpoint khong ton tai: {path}. Khoi tao ngau nhien.")
            return
        checkpoint = torch.load(str(path), map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError("Dinh dang checkpoint HyperGCN khong ho tro")

        backbone_state = self.backbone.state_dict()
        mapped = OrderedDict()
        for key, value in checkpoint.items():
            clean_key = key.replace("module.", "", 1)
            if clean_key.startswith("backbone."):
                clean_key = clean_key[len("backbone."):]
            if clean_key in backbone_state and backbone_state[clean_key].shape == value.shape:
                mapped[clean_key] = value
        missing, unexpected = self.backbone.load_state_dict(mapped, strict=False)
        print(f"Loaded {len(mapped)} HyperGCN tensors tu {path}; missing={len(missing)}, unexpected={len(unexpected)}")

    # ── Pooling temporal (giữ nguyên cơ chế cũ — concept thời gian không thuộc riêng bộ phận nào) ──

    def _pool_temporal(self, feature_map: torch.Tensor) -> torch.Tensor:
        temporal_tokens = feature_map.mean(dim=3)                                    # [B, D, T']
        attn = torch.softmax(self.temporal_pool_score(temporal_tokens), dim=-1)
        return (temporal_tokens * attn).sum(dim=-1)                                  # [B, D]

    # ── G3: lịch huấn luyện ────────────────────────────────────────────────────

    def set_phase(self, phase: int, temperature: Optional[float] = None):
        """Giữ nguyên cơ chế 2-phase rời rạc của CLIR — dùng cho cấu hình V2 (G1+G2, CHƯA G3)."""
        if phase == 1:
            self.reasoner.literal_mode = "soft"
            if self.reasoner.num_loops > 0 and self.reasoner.beta_max > 0.0:
                self.reasoner.beta_max = 0.1
                with torch.no_grad():
                    self.reasoner.raw_beta.fill_(float(torch.logit(torch.tensor(0.95))))
            self.lambda_reg = 0.0
            if temperature is not None:
                self.reasoner.temperature = temperature
        elif phase == 2:
            self.reasoner.literal_mode = "ste"
            if self.reasoner.num_loops > 0 and self.reasoner.beta_max > 0.0:
                old_beta_max = self.reasoner.beta_max
                new_beta_max = 0.5
                with torch.no_grad():
                    current_beta = torch.sigmoid(self.reasoner.raw_beta) * old_beta_max
                    new_ratio = (current_beta / new_beta_max).clamp(1e-4, 1.0 - 1e-4)
                    self.reasoner.raw_beta.copy_(torch.logit(new_ratio))
                self.reasoner.beta_max = new_beta_max
            self.lambda_reg = 0.005
            if temperature is not None:
                self.reasoner.temperature = temperature
        else:
            raise ValueError(f"phase phai la 1 hoac 2, nhan duoc: {phase}")

    def set_progress(self, progress: float):
        """G3 — annealing lien tuc thay set_phase() roi rac. Dung cho cau hinh V3.

        temperature: 1.0 -> 10.0, beta_max: 0.1 -> 0.5, lambda_reg: 0 -> 0.005
        (bat dau tu progress=0.3), khong doi lai optimizer param-group giua chung.
        """
        progress = float(max(0.0, min(1.0, progress)))
        self._progress = progress
        self.reasoner.literal_mode = "ste"
        self.reasoner.temperature = 1.0 + progress * (10.0 - 1.0)
        if self.reasoner.num_loops > 0 and self.reasoner.beta_max is not None:
            self.reasoner.beta_max = 0.1 + progress * (0.5 - 0.1)
            with torch.no_grad():
                self.reasoner.raw_beta.fill_(float(torch.logit(torch.tensor(0.95))))
        self.lambda_reg = 0.0 if progress < 0.3 else 0.005 * (progress - 0.3) / 0.7

    # ── Loss Functions (Asymmetric BCE — giữ nguyên từ CLIR) ──────────────────

    def _asymmetric_bce(self, logits, targets, mask=None) -> torch.Tensor:
        targets = targets.float().clamp(0.0, 1.0)
        probs = torch.sigmoid(logits)
        pos_loss = F.softplus(-logits) * targets
        neg_loss = F.softplus(logits) * (1.0 - probs).pow(self.gamma_neg) * (1.0 - targets)
        weights = targets * self.concept_pos_weight + (1.0 - targets) * self.concept_neg_weight
        if mask is not None:
            weights = weights * mask.float()
        loss = pos_loss + neg_loss
        denom = weights.sum().clamp_min(1.0)
        return (loss * weights).sum() / denom

    def compute_sup_loss(self, z0, z_all, y_all, concept_targets, action_labels, concept_mask=None) -> Dict[str, torch.Tensor]:
        z_K = z_all[-1]
        l_c_K = self._asymmetric_bce(z_K, concept_targets, mask=concept_mask)
        l_c_0 = self._asymmetric_bce(z0, concept_targets, mask=concept_mask)
        l_act_aux = z0.new_zeros(())
        if len(y_all) > 0:
            for y_t in y_all:
                l_act_aux = l_act_aux + F.cross_entropy(y_t, action_labels)
            l_act_aux = l_act_aux / len(y_all)
        l_sup = l_c_K + 0.5 * l_c_0 + 0.3 * l_act_aux
        return {"l_c_K": l_c_K, "l_c_0": l_c_0, "l_act_aux": l_act_aux, "l_sup": l_sup}

    def compute_reg_loss(self, z_all) -> Dict[str, torch.Tensor]:
        l_sparse = self.reasoner.rule_sparsity_loss()
        l_stable = z_all[0].new_zeros(())
        K = len(z_all) - 1
        if K > 0:
            for t in range(K):
                diff = (torch.sigmoid(z_all[t + 1]) - torch.sigmoid(z_all[t])).abs()
                l_stable = l_stable + diff.mean()
            l_stable = l_stable / K
        l_reg = l_sparse + l_stable
        return {"l_sparse": l_sparse, "l_stable": l_stable, "l_reg": l_reg}

    def compute_grounding_loss(self, group_feats: Dict[str, torch.Tensor], concept_targets: torch.Tensor) -> torch.Tensor:
        """G2 — chỉ tính khi use_grounding=True và CLIP tai duoc thanh cong."""
        if self.grounding is None:
            return concept_targets.new_zeros(())
        return self.grounding.compute_loss(group_feats, concept_targets, self.concept_heads.concept_slices)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input:                  torch.Tensor,
        index_t:                Optional[torch.Tensor] = None,   # khong dung (HyperGCN khong can index_t), giu de tuong thich API voi CLIR
        concept_targets:        Optional[torch.Tensor] = None,
        concept_target_mask:    Optional[torch.Tensor] = None,
        action_labels:          Optional[torch.Tensor] = None,
        return_trace:           bool = False,
    ) -> Union[torch.Tensor, Dict]:
        feature_map = self.backbone.extract_feature_map(input)              # [B, D, T', V=25]

        z_spatial, group_feats = self.concept_heads(feature_map)            # G1
        temporal_feat = self._pool_temporal(feature_map)
        z_temporal = self.temporal_head(temporal_feat)
        z0 = torch.cat([z_spatial, z_temporal], dim=-1)                     # [B, 72]

        reasoner_out = self.reasoner(z0, return_intermediates=True)
        y_final = reasoner_out["y_final"]
        z_final = reasoner_out["z_final"]
        z_all   = reasoner_out["z_all"]
        y_all   = list(reasoner_out["y_all"])
        betas   = reasoner_out["betas"]

        if not self.return_aux and concept_targets is None and not self.training:
            return y_final

        output = {
            "action_logits":          y_final,
            "concept_logits":         z_final,
            "initial_concept_logits": z0,
            "spatial_logits":         z_spatial,
            "temporal_logits":        z_temporal,
            "all_action_logits":      y_all,
            "betas":                  betas,
        }
        if return_trace:
            output["z_all"] = z_all

        if concept_targets is not None and action_labels is not None:
            targets = concept_targets.float().to(z0.device)
            sup_losses = self.compute_sup_loss(z0, z_all, y_all, targets, action_labels, concept_target_mask)
            output.update(sup_losses)

            aux_loss = self.lambda_sup * sup_losses["l_sup"]

            if self.lambda_reg > 0.0:
                reg_losses = self.compute_reg_loss(z_all)
                output.update(reg_losses)
                aux_loss = aux_loss + self.lambda_reg * reg_losses["l_reg"]
            else:
                output["l_reg"] = z0.new_zeros(())

            if self.use_grounding and self.lambda_ground > 0.0:
                l_ground = self.compute_grounding_loss(group_feats, targets)
                output["l_ground"] = l_ground
                aux_loss = aux_loss + self.lambda_ground * l_ground

            output["aux_loss"] = aux_loss

        elif concept_targets is not None:
            targets = concept_targets.float().to(z0.device)
            l_c_K = self._asymmetric_bce(z_final, targets, mask=concept_target_mask)
            l_c_0 = self._asymmetric_bce(z0, targets, mask=concept_target_mask)
            output["l_c_K"] = l_c_K
            output["l_c_0"] = l_c_0
            output["aux_loss"] = l_c_K + 0.5 * l_c_0

        return output


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def GLIR_HyperGCN(**kwargs) -> GLIRHyperGCN:
    return GLIRHyperGCN(**kwargs)
