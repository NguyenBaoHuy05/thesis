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


class Product(torch.autograd.Function):
    """Tensor product function with Łukasiewicz T-norm approximation (từ REASON).

    y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
    Kháng underflow hoàn toàn vì mẫu số không thể triệt tiêu thành 0 như exp(-inf).
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor) -> torch.Tensor:
        # X: [B, input_dim, output_dim]
        X_clamped = X.clamp(min=1e-6)
        log_X = torch.log(X_clamped)
        sum_log_X = torch.sum(log_X, dim=1)  # [B, output_dim]
        y = -1.0 / (-1.0 + sum_log_X)
        ctx.save_for_backward(X_clamped, y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        X_clamped, y = ctx.saved_tensors
        # grad_output: [B, output_dim]
        # grad_input: [B, input_dim, output_dim]
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X_clamped + 1e-10))
        return grad_input


class GradGraft(torch.autograd.Function):
    """Gradient Grafting (STE cho Logic Layers từ REASON).

    Forward:  Dùng binary output (tường minh, phục vụ suy luận logic tượng trưng).
    Backward: Chuyển gradient qua nhánh liên tục để tối ưu hóa trọng số.
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return X

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, grad_output.clone()


class ConjunctionLayer(nn.Module):
    """L1 AND layer (Primitive Full-Body Pose Rules).

    Cải tiến theo 6 cơ chế REASON:
      1. W ∈ [0, 1] trực tiếp (khởi tạo 0.5 * rand, clip sau mỗi optimizer step)
      2. Łukasiewicz Product T-norm: Product.apply(1 - (1 - l) * W)
      3. Full-body connections (mặc định part_grouped=False)
      4. GradGraft: forward binary, backward continuous
      5. L1 Sparsity Penalty: l1_norm() và sparsity_loss()

    Input:  l ∈ R^(B × n_literals)     — literals từ LiteralLayer
    Output: r_and1 ∈ R^(B × n_rules)   — conjunction rule activations
    """

    def __init__(
        self,
        n_literals: int = _DEFAULT_N_LITERALS,
        n_rules: int = _N_RULES_L1,
        part_grouped: bool = False,
        init_range: float = 0.5,
        grad_graft: bool = False,
        eps: float = _EPS,
    ):
        super().__init__()
        self.n_literals = int(n_literals)
        self.n_rules = int(n_rules)
        self.part_grouped = bool(part_grouped)
        self.grad_graft = bool(grad_graft)
        self.eps = float(eps)

        # Trọng số W ∈ [0, 1] trực tiếp (REASON Mechanism 3)
        self.W = nn.Parameter(float(init_range) * torch.rand(self.n_literals, self.n_rules))

        if self.part_grouped:
            mask = _build_part_group_mask(self.n_literals, self.n_rules)
            self.register_buffer("group_mask", mask)
        else:
            self.group_mask = None

    @property
    def weight_logit(self) -> torch.Tensor:
        """Hỗ trợ tương thích ngược cho code cũ truy cập weight_logit."""
        return torch.logit(self.W.data.clamp(1e-4, 1.0 - 1e-4))

    def _effective_weights(self) -> torch.Tensor:
        """Trọng số thực tế W ∈ [0, 1]."""
        w = self.W.clamp(0.0, 1.0)
        if self.group_mask is not None:
            w = w * self.group_mask
        return w

    def clip(self):
        """Clip trọng số vào đoạn [0, 1] sau optimizer step (REASON Mechanism 3)."""
        self.W.data.clamp_(0.0, 1.0)

    def l1_norm(self) -> torch.Tensor:
        """Tổng L1 norm cho sparsity penalty (REASON Mechanism 5)."""
        return torch.sum(self._effective_weights())

    def sparsity_loss(self) -> torch.Tensor:
        """Mean sparsity loss."""
        return self._effective_weights().mean()

    def continuous_forward(self, l: torch.Tensor) -> torch.Tensor:
        """Product T-norm liên tục (Łukasiewicz approximation)."""
        w = self._effective_weights()                       # [n_literals, n_rules]
        X = 1.0 - (1.0 - l).unsqueeze(-1) * w               # [B, n_literals, n_rules]
        return Product.apply(X)                             # [B, n_rules]

    @torch.no_grad()
    def binarized_forward(self, l: torch.Tensor) -> torch.Tensor:
        """Binarized rule execution với W > 0.5."""
        w = self._effective_weights()
        Wb = (w > 0.5).float()
        return torch.prod(1.0 - (1.0 - l).unsqueeze(-1) * Wb, dim=1)

    def forward(self, l: torch.Tensor, continuous: bool = None) -> torch.Tensor:
        """Forward pass.

        Args:
            l: Literals [B, n_literals]
            continuous: Nếu True, dùng continuous forward. Nếu None, dùng not self.grad_graft.
        """
        res_tilde = self.continuous_forward(l)
        use_cont = continuous if continuous is not None else (not self.grad_graft)
        if use_cont:
            return res_tilde
        res_bar = self.binarized_forward(l)
        return GradGraft.apply(res_bar, res_tilde)

    def extra_repr(self) -> str:
        return (
            f"n_literals={self.n_literals}, n_rules={self.n_rules}, "
            f"part_grouped={self.part_grouped}, grad_graft={self.grad_graft}"
        )


class ConjunctionLayer2(nn.Module):
    """L2a AND2 layer: Hội cấp 2 của các AND rules (L1 -> L2a).

    Mô tả các hành động đòi hỏi nhiều bộ phận phối hợp đồng thời.
    Sử dụng Łukasiewicz Product T-norm và W ∈ [0, 1] trực tiếp.

    Input:  r_and ∈ R^(B × n_rules_l1)   — L1 conjunction rule activations
    Output: r_and2 ∈ R^(B × n_rules_l2a) — L2a compound conjunction rules
    """

    def __init__(
        self,
        n_rules_l1: int = _N_RULES_L1,
        n_rules_l2a: int = _N_RULES_L2A,
        init_range: float = 0.5,
        grad_graft: bool = False,
        eps: float = _EPS,
    ):
        super().__init__()
        self.n_rules_l1 = int(n_rules_l1)
        self.n_rules_l2a = int(n_rules_l2a)
        self.grad_graft = bool(grad_graft)
        self.eps = float(eps)

        # Trọng số W ∈ [0, 1] trực tiếp
        self.W = nn.Parameter(float(init_range) * torch.rand(self.n_rules_l1, self.n_rules_l2a))

    @property
    def weight_logit(self) -> torch.Tensor:
        return torch.logit(self.W.data.clamp(1e-4, 1.0 - 1e-4))

    def _effective_weights(self) -> torch.Tensor:
        return self.W.clamp(0.0, 1.0)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)

    def l1_norm(self) -> torch.Tensor:
        return torch.sum(self._effective_weights())

    def sparsity_loss(self) -> torch.Tensor:
        return self._effective_weights().mean()

    def continuous_forward(self, r_and: torch.Tensor) -> torch.Tensor:
        w = self._effective_weights()                       # [n_rules_l1, n_rules_l2a]
        X = 1.0 - (1.0 - r_and).unsqueeze(-1) * w           # [B, n_rules_l1, n_rules_l2a]
        return Product.apply(X)                             # [B, n_rules_l2a]

    @torch.no_grad()
    def binarized_forward(self, r_and: torch.Tensor) -> torch.Tensor:
        Wb = (self._effective_weights() > 0.5).float()
        return torch.prod(1.0 - (1.0 - r_and).unsqueeze(-1) * Wb, dim=1)

    def forward(self, r_and: torch.Tensor, continuous: bool = None) -> torch.Tensor:
        res_tilde = self.continuous_forward(r_and)
        use_cont = continuous if continuous is not None else (not self.grad_graft)
        if use_cont:
            return res_tilde
        res_bar = self.binarized_forward(r_and)
        return GradGraft.apply(res_bar, res_tilde)

    def extra_repr(self) -> str:
        return f"n_rules_l1={self.n_rules_l1}, n_rules_l2a={self.n_rules_l2a}, grad_graft={self.grad_graft}"

