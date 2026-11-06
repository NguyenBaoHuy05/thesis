"""DisjunctionLayer: Tầng OR rules cho CRL pipeline (L2b).

Module này thực hiện tuyển (disjunction) của các AND rules từ L1:
  r_and ∈ R^(B × 256)  →  r_or1 ∈ R^(B × 128)

Ý nghĩa: Mô tả các hành động có nhiều cách thực hiện khác nhau (alternative patterns).
Ví dụ: (arm_raise ∧ hand_lift) ∨ (hand_grasp ∧ arm_retract)

Công thức Probabilistic OR (De Morgan + log-space):
  r_or1_k = 1 - exp( Σ_j w_jk · log(1 - r_and_j + ε) )
           = 1 - ∏_j (1 - r_and_j + ε)^w_jk

  Trọng số w_jk = sigmoid(θ_jk) ∈ [0, 1]

Lưu ý:
  - Nếu tất cả w ≈ 0: r_or1 ≈ 0 (không kích hoạt)
  - Nếu ít nhất một AND rule j mạnh (r_and_j ≈ 1) và w_jk lớn:
    log(1 - r_and_j) → -∞  → r_or1_k → 1.0
"""

import torch
import torch.nn as nn

_EPS = 1e-6


class DisjunctionLayer(nn.Module):
    """L2b OR1 layer: tuyển của các AND rules (fully-connected).

    Input:  r_and  ∈ R^(B × n_rules_l1)  — L1 conjunction rule activations
    Output: r_or1  ∈ R^(B × n_rules_l2b) — L2b disjunction rule activations

    Công thức:
        r_or1_k = 1 - exp( Σ_j w_jk · log(1 - r_and_j + ε) )

    Trong đó w_jk = sigmoid(θ_jk), với θ_jk là trọng số học được.
    """

    def __init__(
        self,
        n_rules_l1: int = 256,
        n_rules_l2b: int = 128,
        eps: float = _EPS,
    ):
        super().__init__()
        self.n_rules_l1  = int(n_rules_l1)
        self.n_rules_l2b = int(n_rules_l2b)
        self.eps = float(eps)

        # Khởi tạo thưa: sigmoid(-3) ≈ 0.047
        self.weight_logit = nn.Parameter(
            torch.empty(n_rules_l1, n_rules_l2b).fill_(-3.0)
        )

    def _effective_weights(self) -> torch.Tensor:
        """Trọng số thực tế w ∈ [0, 1]."""
        return torch.sigmoid(self.weight_logit)  # [n_rules_l1, n_rules_l2b]

    def forward(self, r_and: torch.Tensor) -> torch.Tensor:
        """Tính disjunction rules.

        Args:
            r_and: L1 conjunction activations [B, n_rules_l1] ∈ (0, 1]

        Returns:
            r_or1: [B, n_rules_l2b] ∈ [0, 1)
        """
        w = self._effective_weights()   # [n_rules_l1, n_rules_l2b]

        # log(1 - r_and + ε): phần bù của AND rule
        log_complement = torch.log((1.0 - r_and).clamp(min=self.eps))  # [B, n_rules_l1]

        # Σ_j w_jk · log(1 - r_and_j): log-space weighted sum
        log_prod = torch.einsum("bi,ij->bj", log_complement, w)   # [B, n_rules_l2b]

        # r_or1 = 1 - exp(log_prod)
        r_or1 = 1.0 - torch.exp(log_prod)   # [B, n_rules_l2b] ∈ [0, 1)
        return r_or1

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty trên trọng số để ép rule thưa."""
        return self._effective_weights().mean()

    def extra_repr(self) -> str:
        return f"n_rules_l1={self.n_rules_l1}, n_rules_l2b={self.n_rules_l2b}"
