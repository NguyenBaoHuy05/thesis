# LCR-SkateFormer — Kế hoạch triển khai chi tiết (REASON-style Rules)

> **Mục đích:** Ghi lại kế hoạch cụ thể từng thành phần, hướng chạy dữ liệu, công thức loss và vị trí file cho việc nâng cấp `lcr_skateformer.py` sang kiến trúc DNF-CRL với REASON-style rule extraction.
>
> **File nguồn hiện tại:** `d:\Thesis\LCR-SkateFormer\model\lcr_skateformer.py` (869 dòng)
> **Concept ontology:** `d:\Thesis\LCR-SkateFormer\concepts\ntu_rgbd_60_120.csv` (120 action × 74 cột, trong đó 73 cột là concept)

---

## 0. Tổng quan luồng dữ liệu (End-to-End Data Flow)

```
Skeleton Sequence X ∈ R^(B × C × T × V × M)
        │
        ▼
┌──────────────────────────────────────────┐
│  SkateFormer Backbone                    │
│  forward_features_from_input(X)          │
│  → feature_map ∈ R^(B × D × T' × V')    │
└──────────────────────────────────────────┘
        │
        ├─── spatial_pool_score (attn) ──► spatial_feat ∈ R^(B × D)
        └─── temporal_pool_score (attn) ─► temporal_feat ∈ R^(B × D)
                │                                │
                ▼                                ▼
        spatial_head (MLP)             temporal_head (MLP)
        → z_s ∈ R^(B × 52)            → z_t ∈ R^(B × 20)
                │                                │
                └─────────────┬──────────────────┘
                              ▼
                    z⁰ = [z_s; z_t] ∈ R^(B × 72)   ← Initial concept logits
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  IterativeKGReasoner (K=2 loops)                          │
        │                                                           │
        │  Vòng t = 0, 1:                                           │
        │    lᵗ  = LiteralLayer(zᵗ)          ∈ R^(B × 144)         │
        │    r_and, r_and2, r_or1, yᵗ = SharedCRL(lᵗ)             │
        │    rᵗ  = [r_and; r_and2; r_or1]    ∈ R^(B × 512)         │
        │    msg = KGMessage(pᵗ, A_req, A_along, A_excl)            │
        │    Δzᵗ = ConceptUpdateCell(zᵗ, rᵗ, msg)                  │
        │    zᵗ⁺¹ = zᵗ + βᵗ · Δzᵗ                                 │
        │                                                           │
        │  Final step (t = K):                                      │
        │    lᴷ = LiteralLayer(zᴷ)                                 │
        │    _, _, _, yᴷ = SharedCRL(lᴷ)     ← Final output        │
        └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   yᴷ ∈ R^(B × N_actions)  →  Loss + Rule Extraction
```

---

## 1. Cấu trúc thư mục đề xuất

```
LCR-SkateFormer/
├── model/
│   ├── SkateFormer.py              (giữ nguyên — backbone)
│   ├── __init__.py                 (cập nhật export)
│   ├── lcr_skateformer.py          [SỬA] — class chính, thêm KGMessage, dùng DNFReasoner
│   ├── crl/                        [MỚI] — tách module CRL ra riêng
│   │   ├── __init__.py
│   │   ├── literal_layer.py        [MỚI] — LiteralLayer + BinarizeSTE
│   │   ├── conjunction_layer.py    [MỚI] — ConjunctionLayer L1(AND) + L2a(AND2)
│   │   ├── disjunction_layer.py    [MỚI] — DisjunctionLayer L2b(OR1)
│   │   ├── crl_reasoner.py         [MỚI] — CRLReasoner: L1→[AND2 ‖ OR1]→Action
│   │   └── rule_extractor.py       [MỚI] — trích xuất luật sau train
│   └── kg/
│       ├── __init__.py
│       ├── concept_relation.py     [MỚI] — ConceptRelationMatrix (72×72 learnable)
│       └── kg_message.py           [MỚI] — kg_message() pass vào ConceptUpdateCell
├── concepts/
│   ├── ntu_rgbd_60_120.csv         (hiện có — 120 action × 73 concept)
│   └── kg_relations.json           [MỚI] — expert-seed + data-mined relations
└── scripts/
    ├── build_kg_matrices.py        [MỚI] — tính A_require/alongside/excludes từ CSV
    └── extract_rules.py            [MỚI] — chạy rule extractor sau khi train
```

---

## 2. Thành phần 1 — `LiteralLayer` + `BinarizeSTE`

**File:** `model/crl/literal_layer.py`

### Hướng chạy

```
Input:  zᵗ ∈ R^(B × 72)   [concept logits, liên tục]
Output: lᵗ ∈ R^(B × 144)  [72 literal dương + 72 literal phủ định]
```

### Hai mode

**Mode `soft` (Phase 1–2, warm-up):**

```
pᵗ = sigmoid(zᵗ)
lᵗ = [pᵗ,  1 - pᵗ]
```

Gradient truyền thẳng qua sigmoid. Dùng trong epoch 1–40.

**Mode `ste` (Phase 3+, rule extraction):**

```
hard   = (zᵗ > 0).float()             # ∈ {0, 1} — forward binary
soft   = sigmoid(k · zᵗ)              # k = temperature, tăng dần theo scheduler
c_ste  = hard + soft - soft.detach()  # STE trick: forward≈binary, backward=gradient soft
lᵗ     = [c_ste,  1 - c_ste]
```

### Ràng buộc kỹ thuật

> STE **chỉ** áp dụng khi tính `lᵗ` để đưa vào CRL.
> Trạng thái nhớ `zᵗ` truyền sang `zᵗ⁺¹ = zᵗ + β·Δzᵗ` **phải giữ dạng logit liên tục**.
> Không bao giờ binarize `z` trong trạng thái cell.

### Code skeleton

```python
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x, torch.tensor(temperature))
        hard = (x > 0).float()
        soft = torch.sigmoid(temperature * x)
        return hard + soft - soft.detach()

    @staticmethod
    def backward(ctx, grad_output):
        x, temp = ctx.saved_tensors
        soft = torch.sigmoid(temp * x)
        return grad_output * temp * soft * (1 - soft), None


class LiteralLayer(nn.Module):
    def __init__(self, n_concepts: int, init_temperature: float = 1.0):
        super().__init__()
        self.n_concepts  = n_concepts
        self.temperature = init_temperature  # tăng dần qua scheduler bên ngoài
        self.ste_mode    = False             # bật bằng model.set_literal_mode('ste')

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, n_concepts] — logit space
        if self.ste_mode:
            c = BinarizeSTE.apply(z, self.temperature)
        else:
            c = torch.sigmoid(z)
        return torch.cat([c, 1.0 - c], dim=-1)  # [B, 2 * n_concepts]
```

---

## 3. Thành phần 2 — `ConjunctionLayer` (L1 AND Rules)

**File:** `model/crl/conjunction_layer.py`

### Hướng chạy

```
Input:  lᵗ   ∈ R^(B × 144)   [literals từ LiteralLayer]
Output: r_and ∈ R^(B × R1)   [R1=256 conjunction rules, mỗi rule là 1 AND clause]
```

### Công thức — Log-Space Product (chống vanishing gradient)

Mỗi conjunction rule j:

```
r_and_j = exp( Σᵢ  w_and[i,j] · log(l_i + ε) )
```

Tương đương Product T-norm nhưng tính trong log-space để ổn định:

```
r_and_j = Π_i  l_i^(w_and[i,j])
```

Trong đó `w_and[i,j] = sigmoid(θ_and[i,j]) ∈ [0, 1]`.

> **Tại sao không dùng Product T-norm thô?**
> Với R1=256 rules và 144 inputs, tích `Π(1 - w(1-l))` rất dễ underflow khi nhiều `l_i` nhỏ. Log-space tránh hoàn toàn vấn đề này.

### Phân nhóm theo body part (Part-Guided Grouping)

Giới hạn không gian tổ hợp bằng cách mỗi nhóm rule chỉ kết nối literals của nhóm body part tương ứng:

| Nhóm | Literals đầu vào | Số rules |
|---|---|---|
| Head group | head_* × 2 = 12 literals | 32 rules |
| Hand/Arm group | hand_* + arm_* × 2 = 38 literals | 64 rules |
| Hip/Leg/Foot group | hip_* + leg_* + foot_* × 2 = 56 literals | 64 rules |
| Temporal group | temporal_* × 2 = 44 literals | 64 rules |
| Cross-group | toàn bộ 144 literals (thưa) | 32 rules |

Tổng R1 = 256. Group mask là tensor cố định, không học được.

### KG Prior Initialization

```python
# Từ A_require: nếu concept i require concept j → w_and[literal_i, rule_j] cao
for (i, j, conf) in kg_require_edges:
    prior_and[literal_idx(i), rule_idx_in_group(j)] = 0.85 * conf

# Từ A_alongside: cả literal_i và literal_j cùng xuất hiện trong rule
for (i, j, conf) in kg_alongside_edges:
    for rule_k in same_part_group_rules(i, j):
        prior_and[literal_idx(i), rule_k] = 0.70 * conf
        prior_and[literal_idx(j), rule_k] = 0.70 * conf

# Từ A_excludes: literal NOT_i nên xuất hiện khi j active
for (i, j, conf) in kg_excludes_edges:
    prior_and[negated_literal_idx(i), rule_idx_in_group(j)] = 0.90 * conf
```

### Metadata trace

```python
trace_and = {
    "rule_ids":       List[int],        # ID của từng conjunction rule
    "active_literals": List[List[str]], # literal có w > τ_and cho từng rule
    "weights":        Tensor[144, R1],  # ma trận trọng số (detach khi eval)
    "activations":    Tensor[B, R1],    # giá trị kích hoạt
}
```

---

## 4. Thành phần 3 — `ConjunctionLayer2` (L2a AND2) và `DisjunctionLayer` (L2b OR1)

**File:** `model/crl/conjunction_layer.py` (thêm L2a) và `model/crl/disjunction_layer.py` (L2b)

### Kiến trúc tổng quát L2

L1 AND rules feed vào **hai nhánh song song**:

```
r_and ∈ R^(B × R1=256)   ← output từ L1 ConjunctionLayer
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
L2a: ConjunctionLayer2                    L2b: DisjunctionLayer
     (AND2, R2a=128, fully connected)          (OR1, R2b=128, fully connected)
     → r_and2 ∈ R^(B × 128)                   → r_or1  ∈ R^(B × 128)
     "compound requirements"                   "alternative sufficients"
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
              [r_and2 ; r_or1] ∈ R^(B × 256)
                           │
                           ▼
                 ActionHead [Linear: 256 → N_actions]
```

> **Tại sao L2 không phân nhóm body part?**
> L1 đã xử lý body-part locality. L2a và L2b nhận input là abstract AND rules — chúng phải
> **fully connected** để học được cross-body-part compound và alternative patterns.

### 4.1. L2a — `ConjunctionLayer2` (AND của AND rules)

**Ý nghĩa:** Bắt các action cần NHIỀU body-part patterns ĐỒNG THỜI xuất hiện.

```
Ví dụ: R_drink_compound = (arm_raise AND hand_lift) AND (head_tilt_up AND NOT leg_kick)
       → cross-body coordination rule
```

**Công thức (Log-Space Product, giống L1):**

```
r_and2_k = exp( Σⱼ  w_and2[j,k] · log(r_and_j + ε) )
```

Trong đó `w_and2[j,k] = sigmoid(θ_and2[j,k]) ∈ [0, 1]`, fully connected từ R1→R2a.

### 4.2. L2b — `DisjunctionLayer` (OR của AND rules)

**Ý nghĩa:** Bắt các action có thể được nhận ra bởi BẤT KỲ một trong nhiều body-part patterns.

```
Ví dụ: R_drink_alternative = (arm_raise AND hand_lift) OR (hand_grasp AND arm_retract)
       → alternative sufficient condition
```

**Công thức — Probabilistic OR (Log-Space):**

```
r_or1_k = 1 - exp( Σⱼ  w_or[j,k] · log(1 - r_and_j + ε) )
```

Tương đương:

```
r_or1_k = 1 - Π_j  (1 - r_and_j)^(w_or[j,k])
```

### 4.3. ActionHead (Strict-Logic Mode)

Action nhận concat của hai nhánh:

```
y = W_action · [r_and2 ; r_or1] + b_action    [Linear: 256 → N_actions]
```

Không có đường nào từ backbone feature vào đây.

### 4.4. Metadata trace

```python
trace_l2a = {
    "and2_rule_ids":       List[int],
    "active_l1_rules":     List[List[int]],  # L1 AND rules tạo thành AND2 rule này
    "weights":             Tensor[R1, R2a],
    "activations":         Tensor[B, R2a],
}
trace_l2b = {
    "or1_rule_ids":        List[int],
    "active_l1_rules":     List[List[int]],  # L1 AND rules kích hoạt OR1 rule này
    "weights":             Tensor[R1, R2b],
    "activations":         Tensor[B, R2b],
    "action_weights":      Tensor[256, N],   # W_action đầy đủ
}
```

---

## 5. Thành phần 4 — `CRLReasoner` (thay thế `KGPrimedReasoner`)

**File:** `model/crl/crl_reasoner.py`

### Hướng chạy (1 lần gọi = 1 step trong vòng lặp)

```
zᵗ ∈ R^(B × 72)
    │
    ▼
LiteralLayer(zᵗ)              → lᵗ    ∈ R^(B × 144)
    │
    ▼
ConjunctionLayer L1(lᵗ)       → r_and ∈ R^(B × R1=256)     [part-grouped, L1]
    │
    ├──────────────────────────────────────────────────────────────────────────
    │  (r_and feeds into two parallel branches)                               │
    ▼                                                                         ▼
ConjunctionLayer2 L2a(r_and)  → r_and2 ∈ R^(B × R2a=128)  [fully connected, AND2]
                                                                              │
                              DisjunctionLayer L2b(r_and)  → r_or1 ∈ R^(B × R2b=128)  [fully connected, OR1]
    │                                                                         │
    └─────────────────────────────────┬───────────────────────────────────────┘
                                      ▼
                         [r_and2 ; r_or1] ∈ R^(B × 256)
                                      │
                                      ▼
                         ActionHead(256 → N_actions) → yᵗ
```

Logic evidence cho UpdateCell:

```python
rᵗ = torch.cat([r_and, r_and2, r_or1], dim=-1)   # ∈ R^(B × 512)
#               256  +  128   +  128  = 512
```

Cả 3 loại signal được đưa vào ConceptUpdateCell để refinement concept state.

### Bảng thay thế so với code hiện tại

| Hiện tại (`lcr_skateformer.py`) | Thay bằng |
|---|---|
| `KGUnionLayer(2*72, l1_dim=128)` | `ConjunctionLayer L1(144, R1=256)` |
| `KGUnionLayer(2*l1_dim, l2_dim=128)` | `ConjunctionLayer2 L2a(R1, R2a=128)` + `DisjunctionLayer L2b(R1, R2b=128)` |
| `nn.Linear(2*l2_dim, n_actions)` | `ActionHead(R2a+R2b=256, N_actions)` |
| `KGPrimedReasoner` | `CRLReasoner` |
| `logic_dim = 2*l1 + 2*l2 = 512` | `logic_dim = R1 + R2a + R2b = 512` |

---

## 6. Thành phần 5 — `ConceptRelationMatrix` + `KGMessage`

**Files:** `model/kg/concept_relation.py` và `model/kg/kg_message.py`

### 6.1. Xây dựng ma trận từ CSV

**Script:** `scripts/build_kg_matrices.py`

```
Input:  concepts/ntu_rgbd_60_120.csv   [120 × 73 binary: 73 concept columns]
Output: concepts/kg_relations.json     [expert + data-mined edges với confidence]
```

**Ba ma trận concept–concept ∈ R^(72×72):**

**A_require** (có hướng — `i require j` nghĩa là khi i xuất hiện thì j phải xuất hiện):

```
P(j | i) = cooccur(i,j) / (support(i) + ε)
Ứng viên: P(j|i) >= 0.90  AND  support(i,j) >= 5 action
```

**A_alongside** (đối xứng — đồng xuất hiện tự nhiên):

```
Lift(i,j) = P(i,j) / (P(i) · P(j) + ε)
Ứng viên: Lift(i,j) >= 1.5  AND  cooccur(i,j) >= 8 action
```

**A_excludes** (đối xứng — hiếm khi cùng xuất hiện):

```
Ứng viên: cooccur(i,j) = 0
           AND support(i) >= 10  AND support(j) >= 10
```

Không kết luận `excludes` chỉ vì cặp chưa xuất hiện khi support thấp.

**Confidence thay cho nhãn cứng:**

```json
{
  "require": [
    {"from": "hand_lift_to_face", "to": "arm_raise",  "conf": 1.0, "source": "expert"},
    {"from": "leg_step",          "to": "foot_step",  "conf": 0.95, "source": "data"}
  ],
  "alongside": [
    {"a": "arm_raise",  "b": "hand_wave",  "conf": 0.72, "source": "data"},
    {"a": "hand_grasp", "b": "arm_hold",   "conf": 0.68, "source": "data"}
  ],
  "excludes": [
    {"a": "arm_raise",  "b": "hip_sit",    "conf": 0.90, "source": "expert"},
    {"a": "leg_jump",   "b": "leg_static_stand", "conf": 0.85, "source": "data"}
  ]
}
```

### 6.2. Ma trận learnable (soft KG — không cố định)

```python
class ConceptRelationMatrix(nn.Module):
    def __init__(self, n_concepts, A_req_prior, A_along_prior, A_excl_prior):
        super().__init__()
        # Khởi tạo từ logit của prior — học được, bị regularize
        self.theta_require   = nn.Parameter(_logit(A_req_prior.clamp(1e-4, 1-1e-4)))
        self.theta_alongside = nn.Parameter(_logit(A_along_prior.clamp(1e-4, 1-1e-4)))
        self.theta_excludes  = nn.Parameter(_logit(A_excl_prior.clamp(1e-4, 1-1e-4)))
        # Lưu prior để tính L_prior
        self.register_buffer('A_req_prior',   A_req_prior)
        self.register_buffer('A_along_prior', A_along_prior)
        self.register_buffer('A_excl_prior',  A_excl_prior)

    def matrices(self):
        return (
            torch.sigmoid(self.theta_require),    # A_req   ∈ [0,1]^(72×72)
            torch.sigmoid(self.theta_alongside),  # A_along ∈ [0,1]^(72×72)
            torch.sigmoid(self.theta_excludes),   # A_excl  ∈ [0,1]^(72×72)
        )

    def prior_loss(self) -> torch.Tensor:
        A_req, A_along, A_excl = self.matrices()
        return (
            (A_req   - self.A_req_prior).abs().mean() +
            (A_along - self.A_along_prior).abs().mean() +
            (A_excl  - self.A_excl_prior).abs().mean()
        ) / 3.0
```

### 6.3. KGMessage — Message Passing vào UpdateCell

```python
def kg_message(p_t, A_require, A_alongside, A_excludes):
    """
    p_t:        concept probabilities ∈ [0,1]^(B × 72)
    A_require:  ∈ [0,1]^(72 × 72), A_req[i,j] = P(i require j)
    Returns:    kg_msg ∈ R^(B × 72)
    """
    # A_req^T · p: concept j nhận message từ concept i khi i require j
    require_msg   = torch.einsum('ij,bj->bi', A_require.T, p_t)

    # A_along · p: concept i nhận signal từ các concept hay cùng xuất hiện với i
    alongside_msg = torch.einsum('ij,bj->bi', A_alongside, p_t)

    # A_excl · p: ức chế concept i nếu concept xung đột với i đang cao
    excludes_msg  = torch.einsum('ij,bj->bi', A_excludes, p_t)

    return require_msg + alongside_msg - excludes_msg   # ∈ R^(B × 72)
```

---

## 7. Thành phần 6 — `ConceptUpdateCell` (sửa)

**File:** `model/lcr_skateformer.py` (sửa class hiện tại)

### Hướng chạy — THAY ĐỔI so với hiện tại

**Hiện tại:**
```
Δzᵗ = update_mlp([zᵗ, logic_proj(rᵗ)])        # in_dim = 72 + 72 = 144
```

**Đề xuất:**
```
pᵗ     = sigmoid(zᵗ)                                       ∈ R^(B × 72)
A_req, A_along, A_excl = kg_matrix.matrices()
kg_msg = kg_message(pᵗ, A_req, A_along, A_excl)            ∈ R^(B × 72)

# rᵗ = [r_and; r_and2; r_or1]  ∈ R^(B × 512)
r_proj = logic_proj(rᵗ)    [Linear: 512 → 72]              ∈ R^(B × 72)
Δzᵗ    = update_mlp([zᵗ, kg_msg, r_proj])                  in_dim = 3 × 72 = 216
zᵗ⁺¹  = zᵗ + βᵗ · Δzᵗ
```

Bỏ hoàn toàn `action_logits` input (không dùng `use_action_feedback`).

### Code skeleton

```python
class ConceptUpdateCell(nn.Module):
    def __init__(self, num_concepts, logic_dim, kg_matrix_module, hidden_dim=256, dropout=0.1):
        # logic_dim = R1 + R2a + R2b = 256 + 128 + 128 = 512
        super().__init__()
        self.kg_matrix  = kg_matrix_module        # ConceptRelationMatrix
        self.logic_proj = nn.Linear(logic_dim, num_concepts)  # 512 → 72
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * num_concepts, hidden_dim),   # [z(72), kg_msg(72), r_proj(72)] = 216
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, z_t, r_t):
        # r_t = [r_and; r_and2; r_or1] ∈ R^(B × 512)
        p_t    = torch.sigmoid(z_t)
        A_req, A_along, A_excl = self.kg_matrix.matrices()
        kg_msg = kg_message(p_t, A_req, A_along, A_excl)  # ∈ R^(B × 72)
        r_proj = self.logic_proj(r_t)                      # ∈ R^(B × 72)
        return self.update_mlp(torch.cat([z_t, kg_msg, r_proj], dim=-1))
```

---

## 8. Hướng chạy đầy đủ qua IterativeKGReasoner (K=2)

```
z⁰ ∈ R^(B × 72)
 │
 ├──────────────────────────────────────────────── lưu vào list [z⁰]
 ▼
CRLReasoner(z⁰)   [Shared weights]
 ├──► r_and⁰  ∈ R^(B × 256)   (L1 AND rules)
 ├──► r_and2⁰ ∈ R^(B × 128)   (L2a AND2 compound rules)
 ├──► r_or1⁰  ∈ R^(B × 128)   (L2b OR1 alternative rules)
 ├──► r⁰ = [r_and⁰; r_and2⁰; r_or1⁰]  ∈ R^(B × 512)   (logic evidence)
 └──► y⁰ ∈ R^(B × N)          (action logits vòng 0, dùng cho L_aux)
         │
         ▼
ConceptUpdateCell(z⁰, r⁰)   # r⁰ dim = 512
 └──► Δz⁰ ∈ R^(B × 72)
         │
         ▼
z¹ = z⁰ + β · Δz⁰    (β = sigmoid(raw_β) × β_max ∈ [0, 0.5])
 │
 ├──────────────────────────────────────────────── lưu vào list [z⁰, z¹]
 ▼
CRLReasoner(z¹)   [SAME SHARED WEIGHTS]
 ├──► r⁰ = [r_and¹; r_and2¹; r_or1¹]
 └──► y¹                                          (dùng cho L_aux)
         │
         ▼
ConceptUpdateCell(z¹, r¹)   [SAME SHARED WEIGHTS]
 └──► Δz¹
         │
         ▼
z² = z¹ + β · Δz¹
 │
 ├──────────────────────────────────────────────── lưu vào list [z⁰, z¹, z²]
 ▼
CRLReasoner(z²)   [FINAL PASS — không update sau bước này]
 └──► yᴷ = y²  ←──────────────── FINAL ACTION OUTPUT

trace = {
  step_0: {z: z⁰, r_and: r_and⁰, r_and2: r_and2⁰, r_or1: r_or1⁰, y: y⁰, delta_z: Δz⁰},
  step_1: {z: z¹, r_and: r_and¹, r_and2: r_and2¹, r_or1: r_or1¹, y: y¹, delta_z: Δz¹},
  step_2: {z: z², r_and: r_and², r_and2: r_and2², r_or1: r_or1², y: y², delta_z: None},
}
```

---

## 9. Loss — Thu gọn và công thức chi tiết

### 9.0. Tổng quan: 8 → 3 loss groups

Nguyên tắc: nhóm theo **mục đích**, cố định trọng số nội bộ, chỉ để lại **3 lambda cần tune**:

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}}^K + \lambda_{\text{sup}} \cdot \mathcal{L}_{\text{sup}} + \lambda_{\text{kg}} \cdot \mathcal{L}_{\text{kg}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}}$$

| Lambda | Vai trò | Grid search | Default |
|---|---|---|---|
| `λ_sup` | Supervision (concept + intermediate) | `{0.5, 1.0, 2.0}` | `1.0` |
| `λ_kg` | KG ontology awareness | `{0.01, 0.05, 0.1}` | `0.05` |
| `λ_reg` | Regularization (sparsity + stability) | `{0.001, 0.005, 0.01}` | `0.005` |

Grid search tối đa: 3×3×3 = **27 configs** (thay vì $3^7 = 2187$ với 7 lambda riêng).

---

### 9.1. Group 1 — Supervision Loss `L_sup` (Phase 1+)

Gộp tất cả losses cần **nhãn có giám sát** (action + concept labels).

$$\mathcal{L}_{\text{sup}} = \mathcal{L}_c^K + \underbrace{0.5}_{\text{cố định}} \cdot \mathcal{L}_c^0 + \underbrace{0.3}_{\text{cố định}} \cdot \frac{1}{K}\sum_{t=0}^{K-1}\mathcal{L}_{\text{act}}^t$$

- $\mathcal{L}_c^K$: Concept supervision trên zᴷ (quan trọng nhất, hệ số 1.0)
- $\mathcal{L}_c^0$: Concept supervision trên z⁰ (hệ số 0.5 — signal sớm, ít quan trọng hơn)
- $\mathcal{L}_{\text{act}}^t$: Action CE trên intermediate loops (hệ số 0.3 — hỗ trợ các loop không “bỏ cuộc”)

**Concept Loss (Asymmetric BCE):**

$$\mathcal{L}_c(z, \hat{c}) = \frac{\sum_{i,c} r_{ic} \left[ \hat{c}_{ic} \cdot \text{softplus}(-z_{ic}) + (1 - \hat{c}_{ic}) \cdot \left(1 - \sigma(z_{ic})\right)^{\gamma_-} \cdot \text{softplus}(z_{ic}) \right]}{\sum_{i,c} r_{ic}}$$

- $r_{ic}$: trọng số tin cậy nhãn (positive\_weight=1.0, negative\_weight=0.15)
- $\gamma_- = 2.0$: asymmetric focussing cho negative examples

**Vị trí tính:** `LCRSkateFormer.compute_sup_loss()` — gọi trong forward, trả về scalar.

---

### 9.2. Group 2 — KG Loss `L_kg` (Phase 2+)

Gộp tất cả losses liên quan đến **tri thức ontology**.

$$\mathcal{L}_{\text{kg}} = \mathcal{L}_{\text{KG consistency}} + \mathcal{L}_{\text{prior retention}}$$

**KG Consistency (tính trên cả z⁰ và zᴷ):**

$$\mathcal{L}_{\text{require}} = \frac{1}{|\mathcal{R}|}\sum_{(i,j,c_{ij}) \in \mathcal{R}} c_{ij} \cdot \text{ReLU}(p_j - p_i)$$

$$\mathcal{L}_{\text{alongside}} = \frac{1}{|\mathcal{A}|}\sum_{(i,j,c_{ij}) \in \mathcal{A}} c_{ij} \cdot \text{ReLU}(\tau_{\text{along}} - p_i \cdot p_j)$$

$$\mathcal{L}_{\text{excludes}} = \frac{1}{|\mathcal{E}|}\sum_{(i,j,c_{ij}) \in \mathcal{E}} c_{ij} \cdot p_i \cdot p_j$$

$$\mathcal{L}_{\text{KG consistency}} = \frac{1}{2}\left[(\mathcal{L}_{\text{require}} + \mathcal{L}_{\text{along}} + \mathcal{L}_{\text{excl}})\big|_{z^0} + (\cdots)\big|_{z^K}\right]$$

- $c_{ij}$: confidence của relation (expert=1.0, data-mined=0.4–0.7)
- $\tau_{\text{along}} = 0.3$: ngưỡng để loss không trivially zero

**Prior Retention (KG matrix không drift xa expert):**

$$\mathcal{L}_{\text{prior retention}} = \frac{1}{3}\left(\|A_{\text{req}} - A_{\text{req}}^{\text{prior}}\|_1 + \|A_{\text{along}} - A_{\text{along}}^{\text{prior}}\|_1 + \|A_{\text{excl}} - A_{\text{excl}}^{\text{prior}}\|_1\right)$$

**Vị trí tính:** `CRLReasoner.compute_kg_loss()` gọi `ConceptRelationMatrix.prior_loss()` — trả về scalar.

---

### 9.3. Group 3 — Regularizer Loss `L_reg` (Phase 3+)

Gộp tất cả **regularizers không cần nhãn**.

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{sparse}} + \mathcal{L}_{\text{stable}}$$

**Rule Sparsity (khờ thưa, rule dễ đọc):**

$$\mathcal{L}_{\text{sparse}} = \frac{1}{144 \cdot R_1}\sum_{i,j} w^{\text{and1}}_{ij} + \frac{1}{R_1 \cdot R_{2a}}\sum_{j,k} w^{\text{and2}}_{jk} + \frac{1}{R_1 \cdot R_{2b}}\sum_{j,k} w^{\text{or1}}_{jk}$$

- Cả 3 weight matrices (L1 AND, L2a AND2, L2b OR1) đều bị regularize.

**Loop Stability (concept không dao động giữa các vòng):**

$$\mathcal{L}_{\text{stable}} = \frac{1}{K} \sum_{t=0}^{K-1} \left\| \sigma(z^{t+1}) - \sigma(z^t) \right\|_1$$

- Tính trên tất cả K cặp `(z^t, z^{t+1})` — không chỉ đầu và cuối.
- Lưu thêm metric `concept_delta_per_loop[t]` để monitor hội tụ.

**Vị trí tính:** `IterativeKGReasoner.compute_reg_loss()` — trả về scalar.

---

### 9.4. API implementation trong forward()

```python
def forward(self, x, concept_labels=None, action_labels=None, phase=1):
    # ... run backbone, CRL loops ...

    losses = {}
    losses['act'] = F.cross_entropy(y_K, action_labels)   # Primary

    if concept_labels is not None:
        losses['sup'] = self.compute_sup_loss(
            z0, zK, y_intermediate_list, concept_labels, action_labels
        )  # L_c^K + 0.5*L_c^0 + 0.3*mean(L_act^t)

    if phase >= 2:
        losses['kg'] = self.compute_kg_loss(z0, zK)
        # L_KG_consistency + L_prior_retention

    if phase >= 3:
        losses['reg'] = self.compute_reg_loss(z_list)
        # L_sparse + L_stable

    return y_K, losses

# Trong trainer:
total_loss = (
    losses['act']
    + lambda_sup * losses.get('sup', 0)
    + lambda_kg  * losses.get('kg',  0)
    + lambda_reg * losses.get('reg', 0)
)
```


---

## 10. Lịch huấn luyện theo Phase (Staged Training)

### Phase 1 — Concept Warm-up (Epoch 1–20)

```yaml
# Chỉ dùng:
loss_act:    enable   # L_act^K, weight = 1.0 (cố định)
loss_sup:    enable   # L_sup, lambda_sup = 1.0
loss_kg:     disable
loss_reg:    disable

# CRL mode:
literal_mode:  soft
beta_max:      0.1
temperature:   1.0
```

Mục đích: backbone + concept head hội tụ trước khi logic tham gia.

---

### Phase 2 — KG Integration (Epoch 21–40)

```yaml
# Thêm bật:
loss_kg:     enable   # lambda_kg = 0.05  [grid: 0.01 / 0.05 / 0.1]

# CRL mode:
literal_mode:  soft
beta_max:      0.3
temperature:   2.0
```

Mục đích: KG message pass vào UpdateCell, ma trận quan hệ học được.

---

### Phase 3 — STE + Rule Extraction (Epoch 41+)

```yaml
# Thêm bật:
loss_reg:    enable   # lambda_reg = 0.005  [grid: 0.001 / 0.005 / 0.01]

# CRL mode:
literal_mode:  ste
temperature:   5.0 → 10.0   # annealing mỗi 5 epoch
beta_max:      0.5

# Learning rate (parameter groups):
backbone_lr:      1e-5
concept_head_lr:  1e-4
crl_lr:           1e-4
kg_matrix_lr:     5e-4
```

Mục đích: forward xấp xỉ nhị phân, rule trích xuất rõ ràng.

> **[NOTE]** Hyperparameter search chỉ cần chạy ở Phase 2 (`lambda_kg`) và Phase 3 (`lambda_reg`).
> `lambda_sup = 1.0` có thể giữ cố định vì được thiết kế cùng bậc với `L_act^K`.

---

## 11. `RuleExtractor` — Trích xuất luật sau train

**File:** `model/crl/rule_extractor.py`  
**Script CLI:** `scripts/extract_rules.py`

### Quy trình

```
1. Load checkpoint, set model.eval() + literal_mode='ste'

2. Lấy w_and = sigmoid(theta_and)  ∈ [0,1]^(144 × R1)
   Lấy w_or  = sigmoid(theta_or)   ∈ [0,1]^(R1 × R2)
   Lấy w_act = W_action            ∈ R^(R2 × N_actions)

3. Với mỗi conjunction rule j:
   active_lits_j = [concept_names[i//2] if i%2==0 else "NOT "+concept_names[i//2]
                    for i in range(144) if w_and[i,j] >= τ_and]
   → "arm_raise AND hand_lift_to_face AND NOT leg_kick"

4. Với mỗi action class a:
   top_or = argsort(w_act[:, a])[-top_k:]      # top OR rules cho action a
   for k in top_or:
       active_ands = [j for j if w_or[j,k] >= τ_or]
       clause = " AND ".join(active_lits[j] for j in active_ands)
   dnf_rule_a = " OR ".join(clauses)

5. Chạy trên validation set để tính:
   - activation_freq[j]: tỉ lệ sample của class a kích hoạt rule j
   - confidence[j]: mean activation khi predict đúng

6. Sắp xếp theo activation_freq giảm dần
```

### Dạng output mong muốn

```
Action: drink_water [Class 0]

Rule 1 [confidence=0.91, coverage=847/1200 samples]:
  arm_raise AND hand_lift_to_face AND NOT leg_kick AND temporal_direction_motion_upward

Rule 2 [confidence=0.76, coverage=312/1200 samples]:
  hand_grasp AND head_tilt_up AND hand_lift_to_face

Final DNF:
  drink_water ← Rule_1 OR Rule_2
```

### Metrics báo cáo

| Metric | Định nghĩa |
|---|---|
| Rule precision | Tỉ lệ sample kích hoạt rule đúng thuộc class đó |
| Rule coverage | Tỉ lệ sample của class được bao phủ ≥1 rule |
| Rule fidelity | Tỉ lệ sample mà rule predict khớp với model predict |
| Avg literals/rule | Độ phức tạp trung bình của 1 AND clause |
| Avg rules/action | Số clause trung bình trong DNF của 1 action |
| KG violation rate | Tỉ lệ cặp trong rule vi phạm ontology |

---

## 12. Ablation bắt buộc

| ID | Cấu hình | Mục đích |
|---|---|---|
| A0 | SkateFormer gốc | Baseline perception (upper bound perception) |
| A1 | LCR + `KGUnionLayer` cũ | Baseline LCR hiện tại |
| A2 | LCR + `DNFReasoner`, soft | Đo lợi ích DNF so với KGUnion |
| A3 | LCR + `DNFReasoner`, STE | Đo ảnh hưởng binarization |
| A4 | LCR + STE + KG matrix msg | **Mô hình chính đề xuất** |
| A5 | A4 với β=0 | Kiểm tra lợi ích iterative refinement |
| A6 | A4 với K=1 vs K=2 vs K=3 | Tìm số vòng tối ưu |
| A7 | A4 với shared vs unshared CRL | Đo ảnh hưởng shared weights |
| A8 | A4 + neural residual | Đo trade-off accuracy–explainability |
| A9 | A4 với random KG prior | Kiểm tra model có thực sự dùng KG không |

---

## 13. Thứ tự triển khai (4 tuần)

```
TUẦN 1 — Foundation

  [1] scripts/build_kg_matrices.py
      Input:  concepts/ntu_rgbd_60_120.csv
      Output: concepts/kg_relations.json
      Test:   in ra top-10 require/alongside/excludes, kiểm tra bằng mắt

  [2] model/crl/literal_layer.py
      Implement: BinarizeSTE + LiteralLayer
      Unit test:
        - gradient qua STE ≠ 0 khi z gần 0
        - output ∈ [0,1], shape (B, 144)
        - soft mode và ste mode cho output gần giống khi k=10

  [3] model/crl/conjunction_layer.py
      Implement: ConjunctionLayer với part-guided group mask
      Unit test:
        - gradient không vanish với 10 literals (dùng log-space)
        - output ∈ [0,1]^(B, R1)
        - KG prior init: w[arm_raise_literal, rule_j] cao khi rule_j group=hand/arm

TUẦN 2 — CRL Core

  [4] model/crl/disjunction_layer.py
      Implement: DisjunctionLayer + ActionHead
      Unit test:
        - output ∈ [0,1]^(B, R2)
        - gradient flow về đến theta_and

  [5] model/crl/dnf_reasoner.py
      Implement: DNFReasoner = Literal + L1 + L2 + ActionHead
      Thay thế KGPrimedReasoner trong IterativeKGReasoner
      Unit test:
        - logic_dim = R1 + R2 = 384
        - trace metadata đúng định dạng
        - forward shape đúng (B, N_actions)

  [6] model/kg/concept_relation.py + kg_message.py
      Implement: ConceptRelationMatrix + kg_message()
      Unit test:
        - prior_loss = 0 khi learned == prior
        - kg_message shape (B, 72)
        - gradient từ prior_loss về đến theta_require

TUẦN 3 — Integration

  [7] Sửa ConceptUpdateCell (lcr_skateformer.py)
      - Thêm kg_matrix_module làm input
      - Thêm kg_message vào forward
      - in_dim: 144 → 216
      Unit test:
        - gradient flow qua kg_message
        - Δz shape (B, 72)

  [8] Sửa IterativeKGReasoner (lcr_skateformer.py)
      - Dùng DNFReasoner thay KGPrimedReasoner
      - Sửa stability_loss thành per-step formula
      - Lưu toàn bộ [z⁰, z¹, ..., zᴷ] trong trace
      Unit test:
        - stability_loss tính đúng K cặp
        - concept_delta_per_loop[t] giảm dần khi model hội tụ

  [9] Sửa LCRSkateFormer.forward + loss
      - Update tất cả loss computation
      - Thêm phase scheduler (Phase 1/2/3)
      Integration test: forward pass đầy đủ với giả dữ liệu

TUẦN 4 — Rule Extraction + Ablation

  [10] model/crl/rule_extractor.py
       Implement đầy đủ quy trình 6 bước
       Unit test:
         - output string đúng định dạng DNF
         - precision/coverage tính đúng trên 10 mẫu giả

  [11] scripts/extract_rules.py
       CLI: python extract_rules.py --checkpoint best.pth --output rules.json

  [12] Chạy ablation A0, A1, A2, A4, A5
       Ghi kết quả vào experiment log
       Báo cáo Top-1/Top-5, concept F1, KG violation rate, rule fidelity
```

---

## 14. Ghi chú kỹ thuật bắt buộc

> **[CRITICAL]** Không binarize `z` trong trạng thái nhớ cell. STE chỉ áp dụng khi tính `l` → CRL. `z^{t+1}` luôn là continuous logit.

> **[CRITICAL]** Stability loss phải tính trên tất cả cặp `(z^t, z^{t+1})`, không chỉ `(z^0, z^K)`.

> **[IMPORTANT]** Khi báo cáo Explainability, chỉ dùng `LCR-Strict-Logic`. `LCR-Residual` chỉ để đo trade-off, không tuyên bố là fully interpretable.

> **[IMPORTANT]** Báo cáo KG violation rate trước và sau train để chứng minh model học tri thức chứ không quên prior.

> **[NOTE]** `fused_feature` vẫn được tạo (dùng cho `LCR-Residual` ablation) nhưng không nối vào `y_final` trong mô hình chính.
