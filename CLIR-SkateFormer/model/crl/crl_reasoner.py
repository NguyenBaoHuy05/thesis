"""CRLReasoner: Concept Reasoning Logic Reasoner.

Module này gói gọn toàn bộ pipeline suy luận logic 1 lượt:
  z^t ∈ R^(B × 72)
    → LiteralLayer → l^t ∈ R^(B × 144)
    → ConjunctionLayer (L1 AND, Part-grouped) → r_and ∈ R^(B × 256)
    → [ConjunctionLayer2 (L2a AND2) ‖ DisjunctionLayer (L2b OR1)]
    → r_and2 ∈ R^(B × 128), r_or1 ∈ R^(B × 128)
    → ActionHead [Linear: 256 → 120] → y^t ∈ R^(B × 120)

    Logic Evidence (cho ConceptUpdateCell):
      r^t = [r_and; r_and2; r_or1] ∈ R^(B × 512)

Trọng số được chia sẻ (shared) giữa các vòng lặp trong IterativeLogicReasoner.

Thiết kế:
  - ActionHead KHÔNG nhận feature bypass từ backbone
    (đảm bảo tính giải thích tuyệt đối — strict logic mode)
  - Logic Evidence kết hợp cả ba tầng: L1 (body-part features),
    L2a (coordinated patterns), L2b (alternative patterns)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .literal_layer import LiteralLayer
from .conjunction_layer import ConjunctionLayer, ConjunctionLayer2
from .disjunction_layer import DisjunctionLayer

# ─── Kích thước mặc định ────────────────────────────────────────────────────
_N_CONCEPTS  = 72   # số concepts (72 sau khi loại 2 null concepts)
_N_ACTIONS   = 120  # số lớp hành động NTU RGB+D 120
_R1          = 256  # L1 AND rules
_R2A         = 128  # L2a AND2 rules
_R2B         = 128  # L2b OR1 rules
_LOGIC_DIM   = _R1 + _R2A + _R2B  # = 512


class CRLReasoner(nn.Module):
    """Concept Reasoning Logic Reasoner (1-pass).

    Được dùng với shared weights trong IterativeLogicReasoner:
    mỗi vòng lặp gọi cùng 1 instance CRLReasoner này.

    Args:
        n_concepts:  Số concepts (mặc định 72).
        n_actions:   Số lớp hành động (mặc định 120).
        r1:          Số L1 AND rules (mặc định 256).
        r2a:         Số L2a AND2 rules (mặc định 128).
        r2b:         Số L2b OR1 rules (mặc định 128).
        literal_mode: 'soft' hoặc 'ste' (có thể thay đổi runtime).
        temperature: Nhiệt độ STE cho LiteralLayer.
    """

    def __init__(
        self,
        n_concepts:   int   = _N_CONCEPTS,
        n_actions:    int   = _N_ACTIONS,
        r1:           int   = _R1,
        r2a:          int   = _R2A,
        r2b:          int   = _R2B,
        literal_mode: str   = "ste",
        temperature:  float = 2.0,
    ):
        super().__init__()
        self.n_concepts  = int(n_concepts)
        self.n_actions   = int(n_actions)
        self.r1  = int(r1)
        self.r2a = int(r2a)
        self.r2b = int(r2b)
        self.logic_dim = r1 + r2a + r2b  # = 512 mặc định

        # ── Các sub-module ──────────────────────────────────────────────────
        self.literal_layer = LiteralLayer(
            n_concepts=n_concepts,
            mode=literal_mode,
            temperature=temperature,
        )

        self.l1_and = ConjunctionLayer(
            n_literals=2 * n_concepts,  # 144
            n_rules=r1,                 # 256
            part_grouped=False,         # REASON: Full-body connections
        )

        self.l2a_and2 = ConjunctionLayer2(
            n_rules_l1=r1,    # 256
            n_rules_l2a=r2a,  # 128
        )

        self.l2b_or1 = DisjunctionLayer(
            n_rules_l1=r1,    # 256
            n_rules_l2b=r2b,  # 128
        )

        # REASON Mechanism 6: ActionHead nhận r_evidence = [r_and1; r_and2; r_or1] (512 chiều)
        # Bao gồm skip-connection trực tiếp từ L1 primitive rules
        self.action_head = nn.Linear(self.logic_dim, n_actions)  # 512 → n_actions

    # ── Properties để điều chỉnh LiteralLayer & Logic Layers ─────────────────
    @property
    def literal_mode(self) -> str:
        return self.literal_layer.mode

    @literal_mode.setter
    def literal_mode(self, value: str):
        self.literal_layer.mode = value

    @property
    def temperature(self) -> float:
        return self.literal_layer.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.literal_layer.temperature = float(value)

    @property
    def grad_graft(self) -> bool:
        return self.l1_and.grad_graft

    @grad_graft.setter
    def grad_graft(self, val: bool):
        self.l1_and.grad_graft = bool(val)
        self.l2a_and2.grad_graft = bool(val)
        self.l2b_or1.grad_graft = bool(val)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        z: torch.Tensor,
        continuous: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Suy luận logic 1 lượt.

        Args:
            z: Concept logits [B, n_concepts] (LUÔN là số thực liên tục)
            continuous: True nếu ép buộc continuous forward, False nếu ép binarized, None = theo layer.grad_graft

        Returns:
            r_evidence: Logic Evidence [B, logic_dim] (= [r_and; r_and2; r_or1], 512 chiều)
            y:          Action logits  [B, n_actions]
        """
        # Bước 1: Tạo literals qua RMSNorm + ScaledSoftsign
        l = self.literal_layer(z)              # [B, 2 * n_concepts]

        # Bước 2: L1 AND rules (Full-body, Łukasiewicz Product)
        r_and = self.l1_and(l, continuous=continuous)           # [B, 256]

        # Bước 3: L2 song song (L2a AND2 ∥ L2b OR1)
        r_and2 = self.l2a_and2(r_and, continuous=continuous)    # [B, 128]
        r_or1  = self.l2b_or1(r_and, continuous=continuous)     # [B, 128]

        # Bước 4: Logic Evidence 512 chiều: [r_and1 (256); r_and2 (128); r_or1 (128)]
        r_evidence = torch.cat([r_and, r_and2, r_or1], dim=-1) # [B, 512]

        # Bước 5: ActionHead nhận toàn bộ 512 chiều (strict logic mode + L1 skip connection)
        y = self.action_head(r_evidence)                        # [B, n_actions]

        return r_evidence, y

    # ── Logic weight clipping & Sparsity helpers ────────────────────────────
    def clip_weights(self):
        """Clip trọng số logic W ∈ [0, 1] sau optimizer step (REASON Mechanism 3)."""
        self.l1_and.clip()
        self.l2a_and2.clip()
        self.l2b_or1.clip()

    def l1_norm(self) -> torch.Tensor:
        """Tổng L1 norm của toàn bộ logic layers (REASON Mechanism 5)."""
        return self.l1_and.l1_norm() + self.l2a_and2.l1_norm() + self.l2b_or1.l1_norm()

    def rule_sparsity_loss(self) -> torch.Tensor:
        """Rule Sparsity Loss (mean weight loss)."""
        return (
            self.l1_and.sparsity_loss()
            + self.l2a_and2.sparsity_loss()
            + self.l2b_or1.sparsity_loss()
        ) / 3.0

    def sparsity_loss(self) -> torch.Tensor:
        """Alias cho rule_sparsity_loss."""
        return self.rule_sparsity_loss()

    def extra_repr(self) -> str:
        return (
            f"n_concepts={self.n_concepts}, n_actions={self.n_actions}, "
            f"r1={self.r1}, r2a={self.r2a}, r2b={self.r2b}, "
            f"logic_dim={self.logic_dim}"
        )

