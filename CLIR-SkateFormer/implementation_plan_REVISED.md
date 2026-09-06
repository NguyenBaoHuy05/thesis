# Kế hoạch Khắc phục CRLReasoner: Adopt Cơ chế REASON + Giữ Kiến trúc Phân cấp CLIR

> [!IMPORTANT]
> **Mục tiêu thực tế (không ảo tưởng):**
> - Khắc phục **3 lỗi toán học + 1 lỗi kiến trúc** đã xác định qua ablation tests thực tế
> - Adopt 6 cơ chế đã kiểm chứng hoạt động từ codebase REASON
> - Giữ **cấu trúc phân cấp 3 tầng của CLIR**: L1=AND only, L2a=AND2, L2b=OR (khác REASON dùng UnionLayer)
> - Loop feedback đi qua **Full L1→L2 (512-dim)** vào ConceptUpdateCell
> - Kỳ vọng accuracy Stage 2 thoát khỏi 1.67% → hội tụ thực tế trên NTU-60

---

## 1. Bằng chứng Thực Nghiệm: Tại sao Code Cũ Sập

### 1.1 Kết quả Ablation Study (3 Seeds × 8 Cấu hình)

| Config | Formula | Init | Head | Val Acc (3 seeds) | Trạng thái |
|:---|:---:|:---:|:---:|:---:|:---|
| **1. All-Old Baseline** | `l^w` | Const | 256 | **1.0 ± 0.2%** | ❌ Đoán ngẫu nhiên |
| **2. Chỉ Fix Head → 512** | `l^w` | Const | 512 | **8.7 ± 5.5%** | ⚠️ Skip-conn giúp đỡ một phần |
| **3. Chỉ Fix Init Normal** | `l^w` | Normal | 256 | **0.1 ± 0.2%** | ❌ Init không đủ |
| **4. Chỉ Fix T-norm** | T-norm | Const | 256 | **0.4 ± 0.0%** | ❌ T-norm không đủ |
| **5. Fix Init + Head** | `l^w` | Normal | 512 | **15.4 ± 4.6%** | ⚠️ Nhích lên |
| **6. Fix T-norm + Head** | T-norm | Const | 512 | **0.3 ± 0.2%** | ❌ Vẫn sập |
| **7. Fix T-norm + Init** | T-norm | Normal | 256 | **0.7 ± 0.5%** | ❌ Vẫn sập |
| **8. Fix All 3** | T-norm | Normal | 512 | **0.6 ± 0.2%** | ❌ Vẫn sập! |

**Phát hiện quan trọng từ Experiment 3 (Zero-out Interpretability):**
```
Baseline (full 512) Val Acc = 0.42%
   r_and1 mean = 0.002   ← L1 gần = 0
   r_and2 mean = 0.0000  ← L2a = 0 tuyệt đối (underflow)
   r_or1  mean = 0.027   ← L2b rất yếu
```
**Chẩn đoán cuối cùng**: Fix formula + init + head là **cần nhưng chưa đủ**. Nguyên nhân gốc rễ là **L2a nhận 512 inputs dense** → tích nhân 512 số hạng ~ $0.75^{512} \approx 0$ ngay cả với T-norm. Giải pháp duy nhất đã được REASON chứng minh: **Sparsity**.

### 1.2 Tại Sao Sparsity Giải Quyết Underflow L2a

```python
# L2a nhận 256 rules của L1 (+ 256 phủ định = 512 literals)
# DENSE W (mọi rules đều tham gia):
product = (1 - 0.076 * 0.85)^512 ≈ 0.75^512 ≈ 10^{-32}  # Underflow!

# SPARSE W (chỉ 4-6 rules tham gia mỗi compound rule):
product = (1 - 0.076 * 0.85)^6 ≈ 0.75^6 ≈ 0.18           # Healthy!
```

**Kết luận**: Khi L1 sparsity penalty + W clipping ép trọng số thưa ra, mỗi compound rule của L2a chỉ gom ~4-6 primitive rules, phép nhân Product T-norm không bao giờ sập số.

---

## 2. Giải pháp: Adopt 6 Cơ chế từ REASON

> [!NOTE]
> **CLIR giữ khác biệt với REASON ở:** L1 của REASON dùng `UnionLayer = ConjunctionLayer || DisjunctionLayer` song song. CLIR giữ L1 = AND only (Primitive Pose Rules), OR nằm riêng tại L2b (Alternative Action Variants). Điều này có ý nghĩa semantic rõ ràng hơn cho luận văn.

### Cơ chế 1: Preprocessing Input — RMSNorm + ScaledSoftsign

**REASON dùng (thay vì sigmoid):**
```python
# Trước khi vào BinarizeLayer/LiteralLayer:
concepts = RMSNorm(n_concepts)(concepts)
concepts = ScaledSoftsign()(concepts)  # = 3x / (1 + |x|), output ∈ (-3, 3)
```

**Lý do**: `ScaledSoftsign` không bão hòa nhanh như `sigmoid`, gradient mượt hơn ở vùng xa 0.

**Ứng dụng vào CLIR**: Áp dụng `RMSNorm + ScaledSoftsign` trên `z` trước khi truyền vào `LiteralLayer`.

---

### Cơ chế 2: Product Formula với Łukasiewicz Approximation

**REASON `Product.forward()`** (Line 218, `components.py`):
```python
# Thay vì: exp(sum(log(X)))
y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
# Gradient: grad_input = grad_output * (y^2 / (X + ε))
```

Đây là **Łukasiewicz T-norm approximation**, kháng underflow hơn `exp(log)` vì mẫu số không thể → 0 như `exp(-inf)`.

**Ứng dụng vào CLIR**:
- AND layer: `Product.apply(1 - W * (1 - l))`
- OR layer: `1 - Product.apply(1 - W * r)`

---

### Cơ chế 3: W Trực tiếp trong [0,1] thay vì sigmoid(logit)

**REASON:**
```python
# Khởi tạo:
self.W = nn.Parameter(INIT_RANGE * torch.rand(input_dim, output_dim))
# INIT_RANGE = 0.5 → W ∈ [0, 0.5] ban đầu
# Sau mỗi optimizer step:
def clip(self):
    self.W.data.clamp_(0.0, 1.0)
```

**Lý do**: Không cần `sigmoid(logit)` — dùng W trực tiếp và clip sau mỗi step. Đơn giản hơn, gradient không bị co lại bởi sigmoid đuôi.

**Ứng dụng vào CLIR**: Thay `self.weight_logit = nn.Parameter(...)` + `sigmoid(weight_logit)` bằng `self.W = nn.Parameter(...)` + `clip_()` sau optimizer step.

---

### Cơ chế 4: GradGraft — Forward Binary, Backward Continuous

**REASON:**
```python
class GradGraft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y):  # X = binarized, Y = continuous
        return X              # Forward: dùng binary (interpretable)
    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()  # Backward: dùng continuous (trainable)

# Trong ConjunctionLayer:
def forward(self, x):
    res_tilde = self.continuous_forward(x)   # Product T-norm liên tục
    res_bar = self.binarized_forward(x)      # Binarize W > 0.5
    return GradGraft.apply(res_bar, res_tilde)  # Forward=binary, Backward=continuous
```

**Lý do**: Inference/Rule extraction dùng binary output → luật tường minh. Training dùng continuous → gradient thông suốt. Đây là STE (Straight-Through Estimator) cho logic.

**Ứng dụng vào CLIR**: Áp dụng `GradGraft` ở output của L1, L2a, L2b.

---

### Cơ chế 5: L1 Sparsity Penalty (Quan trọng nhất!)

**REASON:**
```python
def l1_norm(self):
    return torch.sum(self.con_layer.W) + torch.sum(self.dis_layer.W)
# Trong training loop:
loss = ce_loss + λ * crl.l1_norm()
```

**Tại sao quan trọng nhất**: Đây là cơ chế duy nhất giải quyết underflow L2a. Khi `λ·ΣW` ép W về 0, mỗi rule chỉ giữ lại ~4-6 connections có ý nghĩa → product T-norm chỉ tính ~6 số hạng → không bao giờ underflow.

**Ứng dụng vào CLIR**: Thêm `λ_sparse * sparsity_loss()` vào total loss trong `clir_skateformer.py`. λ bắt đầu nhỏ (1e-5) và tăng dần.

---

### Cơ chế 6: Skip-Connection UnionLayer → LRLayer

**REASON** (Line 555-557):
```python
# Khi building final LR layer:
if self.use_skip and idx >= 3:
    skip_from_layer = self.layer_list[-2]  # UnionLayer1 (L1 equivalent)
    prev_layer_dim += skip_from_layer.output_dim

# Forward:
x = torch.cat((x, skip_from_layer.x_res), dim=1)  # Concat L1 output
```

**Ứng dụng vào CLIR**: ActionHead nhận `r_evidence = [r_and1; r_and2; r_or1]` (512 chiều), trong đó r_and1 là skip-connection từ L1.

---

## 3. Kiến trúc Cuối cùng: CLIR-Reasoner v2

```
Concept Logits z^t ∈ R^(B × 72)
          │
          ▼ RMSNorm + ScaledSoftsign
Literals l = [p, 1-p] ∈ R^(B × 144)   (p = scaled_softsign(rms_norm(z)))
          │
          ▼ L1: ConjunctionLayer (Full-Body AND, W ∈ [0,1], GradGraft)
r_and1 ∈ R^(B × 256)   ← 256 Primitive Full-Body Pose Rules
          │
   ┌──────┴──────┐
   ▼             ▼
L2a: AND2        L2b: OR1
(W ∈ [0,1],     (W ∈ [0,1],
 sparse)          sparse)
r_and2 (B×128)   r_or1 (B×128)
   └──────┬──────┘
          ▼
r_evidence = [r_and1; r_and2; r_or1] ∈ R^(B × 512)
          │
   ┌──────┴──────────────────────┐
   ▼                             ▼
ActionHead                 ConceptUpdateCell
Linear(512 → 60)           logic_proj(512 → 72)
y^t (action logits)        → Δz → z^{t+1}  ← LOOP K=2
```

**Lý do chọn Loop qua Full L1→L2 (512-dim):**
- ConceptUpdateCell cần biết ngữ cảnh action-level (L2a: phối hợp, L2b: biến thể) để tinh chỉnh concept có ý nghĩa
- r_and1 (L1): *"Pose nguyên thủy nào đang xuất hiện?"*
- r_and2 (L2a): *"Tổ hợp tư thế cấp cao nào?"*
- r_or1 (L2b): *"Biến thể hành động nào?"*
- Feedback 3 cấp độ → ConceptUpdateCell điều chỉnh z chính xác hơn so với chỉ dùng L1

**Lý do L1 = AND Only (khác REASON):**
- REASON UnionLayer = AND + OR song song → L1 đã là DNF 2 tầng nén lại
- CLIR tách biệt rõ: L1 = Tư thế, L2b = Biến thể → Giải thích theo cấp độ, rõ ràng hơn cho luận văn
- L1 AND rules: `R12 = head_up ∧ arm_raised ∧ NOT(leg_kick)` (tư thế cụ thể)
- L2b OR rules: `V3 = R12 OR R15 OR R29` (các cách thực hiện "bow" khác nhau)

---

## 4. Kế hoạch Chỉnh sửa Code

### [MODIFY] [model/crl/literal_layer.py](file:///d:/Thesis/CLIR-SkateFormer/model/crl/literal_layer.py)
- Thêm `RMSNorm` và `ScaledSoftsign` trước khi tạo literals
- Giữ hai chế độ: `soft` (training) và `ste` (hard inference)

### [MODIFY] [model/crl/conjunction_layer.py](file:///d:/Thesis/CLIR-SkateFormer/model/crl/conjunction_layer.py)
- **`ConjunctionLayer` (L1) và `ConjunctionLayer2` (L2a):**
  - Đổi từ `weight_logit + sigmoid()` sang `W ∈ [0,1]` trực tiếp với `clip_(0,1)`
  - Đổi công thức sang Łukasiewicz Product T-norm
  - Bỏ `group_mask` chia part cứng ở L1 → Full-body connections
  - Thêm `GradGraft`: forward dùng `W_binary = (W > 0.5).float()`, backward dùng W liên tục
  - Thêm `l1_norm()` method để tính sparsity penalty
  - Khởi tạo: `W = 0.5 * torch.rand(...)` (INIT_RANGE = 0.5, theo REASON)

### [MODIFY] [model/crl/disjunction_layer.py](file:///d:/Thesis/CLIR-SkateFormer/model/crl/disjunction_layer.py)
- **`DisjunctionLayer` (L2b):**
  - Cùng cơ chế: W trực tiếp, clip, Łukasiewicz Product, GradGraft
  - `l1_norm()` method

### [MODIFY] [model/crl/crl_reasoner.py](file:///d:/Thesis/CLIR-SkateFormer/model/crl/crl_reasoner.py)
- Cập nhật `action_head = nn.Linear(r1 + r2a + r2b, n_actions)` (512 → 60)
- Forward trả về `r_evidence = [r_and1, r_and2, r_or1]` đầy đủ 512 chiều
- Thêm `clip_weights()` method để gọi sau optimizer step
- Thêm `sparsity_loss()` = `(l1_and.l1_norm() + l2a_and2.l1_norm() + l2b_or1.l1_norm()) / 3`

### [MODIFY] [model/clir_skateformer.py](file:///d:/Thesis/CLIR-SkateFormer/model/clir_skateformer.py)
- Trong training step, sau `optimizer.step()`:
  ```python
  self.crl.clip_weights()  # Clip W ∈ [0, 1]
  ```
- Thêm `λ_sparse * crl.sparsity_loss()` vào total loss
  - Stage 1 (Concept Warmup): `λ_sparse = 0` (chưa train CRL)
  - Stage 2 đầu (epoch 1-10): `λ_sparse = 1e-5` (thưa dần)
  - Stage 2 sau (epoch 11+): `λ_sparse = 5e-5` (thưa mạnh hơn)

### [MODIFY] [generate_notebook.py](file:///d:/Thesis/CLIR-SkateFormer/generate_notebook.py) & [CLIR_SkateFormer_Colab.ipynb](file:///d:/Thesis/CLIR-SkateFormer/CLIR_SkateFormer_Colab.ipynb)
- Sinh lại notebook với toàn bộ cập nhật

---

## 5. Verification Plan Chi tiết

### 5.1 Local Unit Tests (Trước khi đẩy lên Colab)

**Test 1: Gradient Flow — Tất cả layers đều > 0**
```python
z = torch.randn(4, 72, requires_grad=True)
r_ev, y = crl(z)
y.sum().backward()
assert model.l1_and.W.grad.norm() > 1e-4, "L1 gradient dead!"
assert model.l2a_and2.W.grad.norm() > 1e-4, "L2a gradient dead!"
assert model.l2b_or1.W.grad.norm() > 1e-4, "L2b gradient dead!"
```

**Test 2: Giá trị Activation không bị Underflow/Saturate**
```python
with torch.no_grad():
    assert r_and1.mean() > 0.05, "L1 underflow!"   # Expected ~0.15
    assert r_and2.mean() > 0.01, "L2a underflow!"   # Expected ~0.05+
    assert r_or1.mean() > 0.01, "L2b saturated!"    # Expected ~0.1+
    assert r_or1.std() > 0.01, "L2b constant!"      # Phải có variance
```

**Test 3: W Sparsity Tăng dần qua Training**
- Sau 5 epochs với `λ_sparse = 1e-5`: `mean(W > 0.5) < 0.2` (< 20% weights active)

**Test 4: Kiểm tra Vòng lặp K=2**
- Chạy qua full `CLIRSkateFormer` với `K=2`, xác nhận `z^1 ≠ z^0` và `z^2 ≠ z^1`
- Gradient norm của `raw_beta` > 0 (β đang được học)

### 5.2 Colab Validation (Sau khi local test pass)
- Load Checkpoint Stage 1 (đã đạt 99.4% Concept Acc)
- Chạy Stage 2 với hyperparameters mới
- **Metrics cần theo dõi**:
  - `action_acc@1` trên val set mỗi epoch (mục tiêu: thoát 1.67% sau epoch 5)
  - `r_and1.mean()`, `r_and2.mean()`, `r_or1.mean()` (kiểm tra không sập số)
  - `L1_active_ratio = mean(W_l1 > 0.5)` (theo dõi sparsity)
  - `l2a_grad_norm`, `l2b_grad_norm` (xác nhận không bị dead)
