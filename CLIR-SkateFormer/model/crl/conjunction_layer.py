"""ConjunctionLayer: Tầng AND rules cho CRL pipeline.

Module này thực hiện hai tầng hội:
  L1: ConjunctionLayer  (AND, part-grouped) — 148 literals → 256 rules
  L2a: ConjunctionLayer2 (AND2, fully-connected) — 256 rules → 128 compound rules

Công thức Log-Space Product (chống vanishing gradient):
  r_j = exp( Σ_i w_ij · log(l_i + ε) )
       = ∏_i (l_i + ε)^w_ij

  Trong đó w_ij = sigmoid(θ_ij) ∈ [0, 1]  (trọng số học được)

Part-Grouped Architecture (L1):
  Mỗi nhóm rule chỉ kết nối với literals của nhóm bộ phận cơ thể tương ứng.
  Group mask là tensor nhị phân cố định (register_buffer).

  Nhóm bộ phận | Concepts (index) | Literals | Rules
  -------------|------------------|----------|------
  Head         |  0 – 5           |  12      |  32
  Hand         |  6 – 16          |  22      |  48
  Arm          | 17 – 27          |  22      |  48
  Hip          | 28 – 35          |  16      |  32
  Leg          | 36 – 44          |  18      |  32
  Foot         | 45 – 51          |  14      |  24
  Temporal     | 53 – 73          |  42      |  40
  Total        |                  | 148      | 256
"""

import torch
import torch.nn as nn

# ─── Hằng số cấu hình Part-Grouped ───────────────────────────────────────────

_DEFAULT_N_CONCEPTS = 72   # 72 concepts sau khi loại bỏ 2 null concepts (52 spatial + 20 temporal)
_DEFAULT_N_LITERALS = 144  # 2 × 72
_N_RULES_L1         = 256  # Σ rules của các nhóm
_N_RULES_L2A        = 128  # AND2 (L2a)

_EPS = 1e-6  # numerical stability cho log


def get_part_groups(n_concepts: int) -> list:
    """Trả về danh sách phân nhóm bộ phận cơ thể phù hợp với số concepts."""
    if n_concepts == 72:
        # Feeder mặc định remove_null_concept=True: 52 spatial + 20 temporal = 72
        return [
            ("head",     0,   6,   32),
            ("hand",     6,  17,   48),
            ("arm",     17,  28,   48),
            ("hip",     28,  36,   32),
            ("leg",     36,  45,   32),
            ("foot",    45,  52,   24),
            ("temporal",52,  72,   40),
        ]
    elif n_concepts == 74:
        # Full ontology chưa loại bỏ null concepts: 53 spatial + 21 temporal = 74
        return [
            ("head",     0,   6,   32),
            ("hand",     6,  17,   48),
            ("arm",     17,  28,   48),
            ("hip",     28,  36,   32),
            ("leg",     36,  45,   32),
            ("foot",    45,  53,   24),
            ("temporal",53,  74,   40),
        ]
    else:
        # Fallback chia đều cho các nhóm
        step = max(1, n_concepts // 7)
        return [
            ("group_0", 0, step, 32),
            ("group_1", step, 2 * step, 48),
            ("group_2", 2 * step, 3 * step, 48),
            ("group_3", 3 * step, 4 * step, 32),
            ("group_4", 4 * step, 5 * step, 32),
            ("group_5", 5 * step, 6 * step, 24),
            ("group_6", 6 * step, n_concepts, 40),
        ]


def _build_part_group_mask(n_literals: int = _DEFAULT_N_LITERALS, n_rules: int = _N_RULES_L1) -> torch.Tensor:
    """Xây dựng group mask nhị phân [n_literals, n_rules].

    Mask[i, j] = 1.0 nếu literal i thuộc nhóm bộ phận của rule j, 0.0 nếu không.
    Literals của concept c: index c (dương) và c + n_concepts (phủ định).
    """
    mask = torch.zeros(n_literals, n_rules)
    rule_start = 0
    n_concepts = n_literals // 2
    groups = get_part_groups(n_concepts)

    for (name, c_start, c_end, n_group_rules) in groups:
        rule_end = rule_start + n_group_rules
        for c in range(c_start, min(c_end, n_concepts)):
            # Literal dương: index c
            mask[c, rule_start:rule_end] = 1.0
            # Literal phủ định: index c + n_concepts
            mask[c + n_concepts, rule_start:rule_end] = 1.0
        rule_start = rule_end

    return mask


class ConjunctionLayer(nn.Module):
    """L1 AND layer với Part-Grouped structure.

    Input:  l ∈ R^(B × 148)   — literals từ LiteralLayer
    Output: r_and ∈ R^(B × 256) — conjunction rule activations

    Công thức:
        r_j = exp( Σ_i mask_ij · w_ij · log(l_i + ε) )

    Trọng số w_ij = sigmoid(θ_ij), và θ_ij bị zero-out ngoài mask
    (hiệu quả = không có kết nối ngoài nhóm).
    """

    def __init__(
        self,
        n_literals: int = _DEFAULT_N_LITERALS,
        n_rules: int = _N_RULES_L1,
        part_grouped: bool = True,
        eps: float = _EPS,
    ):
        super().__init__()
        self.n_literals = int(n_literals)
        self.n_rules = int(n_rules)
        self.part_grouped = bool(part_grouped)
        self.eps = float(eps)

        # Trọng số học được ở không gian logit
        self.weight_logit = nn.Parameter(
            torch.empty(n_literals, n_rules).fill_(-3.0)  # khởi tạo ≈ sigmoid(-3) ≈ 0.047 (thưa)
        )

        if part_grouped:
            mask = _build_part_group_mask(n_literals, n_rules)
            self.register_buffer("group_mask", mask)  # [148, 256]
        else:
            self.group_mask = None

    def _effective_weights(self) -> torch.Tensor:
        """Tính trọng số thực tế (có mask) w ∈ [0, 1]."""
        w = torch.sigmoid(self.weight_logit)  # [n_literals, n_rules]
        if self.group_mask is not None:
            w = w * self.group_mask            # zero-out ngoài nhóm
        return w

    def forward(self, l: torch.Tensor) -> torch.Tensor:
        """Tính conjunction rules.

        Args:
            l: Literals [B, n_literals] ∈ [0, 1]

        Returns:
            r_and: [B, n_rules] ∈ [0, 1]
        """
        w = self._effective_weights()   # [n_literals, n_rules]

        # log(l + ε): [B, n_literals]
        log_l = torch.log(l.clamp(min=self.eps))  # [B, n_literals]

        # Σ_i w_ij · log(l_i): einsum [B, n_lit] × [n_lit, n_rules] → [B, n_rules]
        log_r = torch.einsum("bi,ij->bj", log_l, w)  # [B, n_rules]

        # r_j = exp(log_r_j) ∈ (0, 1]
        return torch.exp(log_r)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty trên trọng số để ép rule thưa."""
        w = self._effective_weights()
        return w.mean()

    def extra_repr(self) -> str:
        return (
            f"n_literals={self.n_literals}, n_rules={self.n_rules}, "
            f"part_grouped={self.part_grouped}"
        )


class ConjunctionLayer2(nn.Module):
    """L2a AND2 layer: hội của các AND rules (fully-connected).

    Mô tả các hành động đòi hỏi nhiều bộ phận phối hợp đồng thời.
    Ví dụ: (arm_raise ∧ hand_lift) ∧ (head_tilt_up ∧ NOT leg_kick)

    Input:  r_and ∈ R^(B × 256)  — L1 conjunction rule activations
    Output: r_and2 ∈ R^(B × 128) — L2a compound conjunction rules

    Công thức:
        r_and2_k = exp( Σ_j w_jk · log(r_and_j + ε) )
    """

    def __init__(
        self,
        n_rules_l1: int = _N_RULES_L1,
        n_rules_l2a: int = _N_RULES_L2A,
        eps: float = _EPS,
    ):
        super().__init__()
        self.n_rules_l1 = int(n_rules_l1)
        self.n_rules_l2a = int(n_rules_l2a)
        self.eps = float(eps)

        # Khởi tạo thưa: sigmoid(-3) ≈ 0.047
        self.weight_logit = nn.Parameter(
            torch.empty(n_rules_l1, n_rules_l2a).fill_(-3.0)
        )

    def _effective_weights(self) -> torch.Tensor:
        return torch.sigmoid(self.weight_logit)  # [n_rules_l1, n_rules_l2a]

    def forward(self, r_and: torch.Tensor) -> torch.Tensor:
        """Tính L2a compound conjunction rules.

        Args:
            r_and: L1 rule activations [B, n_rules_l1] ∈ (0, 1]

        Returns:
            r_and2: [B, n_rules_l2a] ∈ (0, 1]
        """
        w = self._effective_weights()                    # [n_rules_l1, n_rules_l2a]
        log_r = torch.log(r_and.clamp(min=self.eps))    # [B, n_rules_l1]
        log_r2 = torch.einsum("bi,ij->bj", log_r, w)    # [B, n_rules_l2a]
        return torch.exp(log_r2)

    def sparsity_loss(self) -> torch.Tensor:
        return self._effective_weights().mean()

    def extra_repr(self) -> str:
        return f"n_rules_l1={self.n_rules_l1}, n_rules_l2a={self.n_rules_l2a}"
