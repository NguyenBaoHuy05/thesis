# CLIR-SkateFormer — Kế hoạch triển khai chi tiết

> **Tên mô hình:** **CLIR-SkateFormer** — **C**oncept **L**ogic **I**terative **R**efinement for Skeleton Action Recognition

> **Mục đích:** Ghi lại kế hoạch cụ thể từng thành phần, hướng chạy dữ liệu, công thức loss và cấu trúc module cho việc nâng cấp `lcr_skateformer.py`.
>
> **Điểm đột phá kiến trúc:**
> 1. **Tri thức Hành động (Action-Concept Knowledge):** Được mã hóa trực tiếp trong các luật DNF của tầng suy luận logic CRL (L1 AND $\rightarrow$ [L2a AND2 $\parallel$ L2b OR1] $\rightarrow$ ActionHead).
> 2. **Loại bỏ Concept-Concept KG gượng ép:** Không dùng ma trận quan hệ $72 \times 72$ giữa các concept (tránh các quan hệ giả định/spurious correlations do dữ liệu và mâu thuẫn thời gian trong chuỗi video).
> 3. **Logic-Guided Concept Refinement (Top-Down Feedback):** Vòng lặp tinh chỉnh khái niệm ($z^{t+1} = z^t + \beta \Delta z^t$) được dẫn dắt bởi **bằng chứng logic (Logic Evidence $r^t$)** sinh ra từ chính các luật DNF mức cao.
>
> **File nguồn hiện tại:** `d:\Thesis\LCR-SkateFormer\model\lcr_skateformer.py` (869 dòng — đang refactor sang CLIR)  
> **Concept ontology:** `d:\Thesis\LCR-SkateFormer\concepts\ntu_rgbd_60_120.csv` (120 actions $\times$ 72 concepts)

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
        │  IterativeLogicReasoner (K=2 loops)                         │
        │                                                             │
        │  Vòng t = 0, 1:                                             │
        │    lᵗ  = LiteralLayer(zᵗ)          ∈ R^(B × 144)           │
        │    r_and, r_and2, r_or1, yᵗ = SharedCRL(lᵗ)               │
        │    rᵗ  = [r_and; r_and2; r_or1]    ∈ R^(B × 512) (Logic Ev) │
        │    Δzᵗ = ConceptUpdateCell(zᵗ, rᵗ) ∈ R^(B × 72)            │
        │    zᵗ⁺¹ = zᵗ + βᵗ · Δzᵗ                                   │
        │                                                             │
        │  Final step (t = K):                                        │
        │    lᴷ = LiteralLayer(zᴷ)                                   │
        │    _, _, _, yᴷ = SharedCRL(lᴷ)     ← Final Action Logits   │
        └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   yᴷ ∈ R^(B × 120)  →  Loss + Rule Extraction
```

---

## 1. Cấu trúc thư mục tinh gọn

```
CLIR-SkateFormer/          ← Thư mục gốc của mô hình đề xuất
├── model/
│   ├── SkateFormer.py              (giữ nguyên — perception backbone)
│   ├── __init__.py                 (cập nhật export)
│   ├── clir_skateformer.py         [MỚI] — class chính CLIRSkateFormer: tích hợp CRLReasoner & ConceptUpdateCell
│   └── crl/                        [MỚI] — module logic reasoning độc lập
│       ├── __init__.py
│       ├── literal_layer.py        [MỚI] — LiteralLayer + BinarizeSTE (72 -> 144 literals)
│       ├── conjunction_layer.py    [MỚI] — L1 ConjunctionLayer (part-grouped) + L2a AND2
│       ├── disjunction_layer.py    [MỚI] — L2b DisjunctionLayer (OR1)
│       ├── crl_reasoner.py         [MỚI] — SharedCRL: L1 -> [AND2 ‖ OR1] -> ActionHead
│       └── rule_extractor.py       [MỚI] — module trích xuất luật DNF sau huấn luyện
├── concepts/
│   └── ntu_rgbd_60_120.csv         (120 action × 72 concept ontology)
└── scripts/
    └── extract_rules.py            [MỚI] — CLI trích xuất luật và đánh giá fidelity/coverage
```

---

## 2. Thành phần 1 — `LiteralLayer` + `BinarizeSTE`

**File:** `model/crl/literal_layer.py`

### Hướng chạy
```
Input:  zᵗ ∈ R^(B × 72)   [concept logits liên tục]
Output: lᵗ ∈ R^(B × 144)  [72 literal dương + 72 literal phủ định (NOT)]
```

### Hai chế độ (Modes)

1. **Mode `soft` (Phase 1, Warm-up):**
   ```python
   p = torch.sigmoid(z)
   l = torch.cat([p, 1.0 - p], dim=-1)  # [B, 144]
   ```
   Gradient truyền mượt qua hàm sigmoid giúp backbone và concept head ổn định.

2. **Mode `ste` (Phase 2, Rule Sparsification & Extraction):**
   ```python
   hard = (z > 0.0).float()
   soft = torch.sigmoid(temperature * z)
   c_ste = hard + soft - soft.detach()  # Straight-Through Estimator
   l = torch.cat([c_ste, 1.0 - c_ste], dim=-1)
   ```

> **[NGUYÊN TẮC BẮT BUỘC]**  
> * Trạng thái nhớ $z^t$ trong `ConceptUpdateCell` **luôn luôn ở dạng logit liên tục**.  
> * Binarization bằng STE **chỉ áp dụng tạm thời** khi tạo đầu vào $l^t$ truyền vào CRL.

---

## 3. Thành phần 2 — `ConjunctionLayer` (L1 AND Rules)

**File:** `model/crl/conjunction_layer.py`

### Hướng chạy
```
Input:  lᵗ    ∈ R^(B × 144)  [literals từ LiteralLayer]
Output: r_and ∈ R^(B × R1)   [R1 = 256 conjunction rules theo bộ phận cơ thể]
```

### Công thức Log-Space Product (Chống vanishing gradient)
Với mỗi conjunction rule $j$:
$$r\_and_j = \exp\left( \sum_{i} w^{\text{and1}}_{ij} \cdot \log(l_i + \epsilon) \right) = \prod_{i} (l_i)^{w^{\text{and1}}_{ij}}$$
Trong đó $w^{\text{and1}}_{ij} = \text{sigmoid}(\theta^{\text{and1}}_{ij}) \in [0, 1]$.

### Part-Guided Grouping (Phân nhóm theo bộ phận cơ thể)
Giới hạn không gian tổ hợp bằng cách mỗi nhóm rule chỉ kết nối literals của nhóm bộ phận tương ứng:

| Nhóm bộ phận | Literals đầu vào ($2 \times$ concepts) | Số rules phân bổ |
|---|---|---|
| **Head group** | 6 concepts $\times 2 = 12$ literals | 32 rules |
| **Hand group** | 11 concepts $\times 2 = 22$ literals | 48 rules |
| **Arm group** | 11 concepts $\times 2 = 22$ literals | 48 rules |
| **Hip group** | 8 concepts $\times 2 = 16$ literals | 32 rules |
| **Leg group** | 9 concepts $\times 2 = 18$ literals | 32 rules |
| **Foot group** | 7 concepts $\times 2 = 14$ literals | 24 rules |
| **Temporal group** | 20 concepts $\times 2 = 40$ literals | 40 rules |
| **Total R1** | **144 literals** | **256 rules** |

*Group mask là tensor nhị phân cố định, các liên kết ngoài nhóm bị đặt $w=0$.*

---

## 4. Thành phần 3 — Cấu trúc song song L2: `AND2` (L2a) và `OR1` (L2b)

**File:** `model/crl/conjunction_layer.py` (L2a) và `model/crl/disjunction_layer.py` (L2b)

L1 AND rules được đưa đồng thời vào **hai nhánh song song**:

```
r_and ∈ R^(B × 256)   ← atomic body-part rules từ L1
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
L2a: ConjunctionLayer2                    L2b: DisjunctionLayer
     (AND2, R2a=128, fully connected)          (OR1, R2b=128, fully connected)
     → r_and2 ∈ R^(B × 128)                   → r_or1  ∈ R^(B × 128)
     [Compound Coordination Rules]             [Alternative Explanation Rules]
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
              [r_and2 ; r_or1] ∈ R^(B × 256)
                           │
                           ▼
                 ActionHead [Linear: 256 → 120]
```

### 4.1. L2a — `ConjunctionLayer2` (Hội của các AND rules)
* **Ý nghĩa:** Mô tả các hành động đòi hỏi **nhiều bộ phận phối hợp đồng thời**.
  * *Ví dụ:* `(arm_raise ∧ hand_lift) ∧ (head_tilt_up ∧ NOT leg_kick)`.
* **Công thức:**
  $$r\_and2_k = \exp\left( \sum_{j} w^{\text{and2}}_{jk} \cdot \log(r\_and_j + \epsilon) \right)$$

### 4.2. L2b — `DisjunctionLayer` (Tuyển của các AND rules)
* **Ý nghĩa:** Mô tả các hành động có **nhiều cách thực hiện khác nhau (Alternative patterns)**.
  * *Ví dụ:* `(arm_raise ∧ hand_lift) ∨ (hand_grasp ∧ arm_retract)`.
* **Công thức (Probabilistic OR):**
  $$r\_or1_k = 1 - \exp\left( \sum_{j} w^{\text{or1}}_{jk} \cdot \log(1 - r\_and_j + \epsilon) \right)$$

### 4.3. ActionHead (Strict-Logic Mode)
Đầu ra phân loại hành động $y$ hoàn toàn chỉ nhận từ output logic của L2:
$$y = W_{\text{action}} \cdot [r\_and2 \;;\; r\_or1] + b_{\text{action}} \quad [\text{Linear}: 256 \rightarrow 120]$$
*(Không sử dụng feature bypass từ backbone nhằm đảm bảo tính giải thích tuyệt đối).*

---

## 5. Thành phần 4 — `CRLReasoner`

**File:** `model/crl/crl_reasoner.py`

Gói gọn toàn bộ quy trình suy luận 1 lượt:
```python
class CRLReasoner(nn.Module):
    def __init__(self, n_concepts=72, n_actions=120, r1=256, r2a=128, r2b=128):
        super().__init__()
        self.literal_layer = LiteralLayer(n_concepts)
        self.l1_and        = ConjunctionLayer(2 * n_concepts, r1, part_grouped=True)
        self.l2a_and2      = ConjunctionLayer2(r1, r2a)
        self.l2b_or1       = DisjunctionLayer(r1, r2b)
        self.action_head   = nn.Linear(r2a + r2b, n_actions)

    def forward(self, z):
        l = self.literal_layer(z)
        r_and = self.l1_and(l)
        r_and2 = self.l2a_and2(r_and)
        r_or1 = self.l2b_or1(r_and)
        
        # Action logits
        l2_cat = torch.cat([r_and2, r_or1], dim=-1)
        y = self.action_head(l2_cat)
        
        # Logic Evidence cho ConceptUpdateCell (256 + 128 + 128 = 512 dims)
        r_evidence = torch.cat([r_and, r_and2, r_or1], dim=-1)
        
        return r_evidence, y
```

---

## 6. Thành phần 5 — `ConceptUpdateCell` (Logic-Guided Top-Down Feedback)

**File:** `model/clir_skateformer.py`

### Cơ chế hoạt động
Thay vì dựa vào ma trận Concept-Concept KG tĩnh, `ConceptUpdateCell` sử dụng **bằng chứng logic mức cao $r^t$** để điều chỉnh các concept logits mức thấp $z^t$:

```
zᵗ (Concept Logits hiện tại) ───┐
                               ├──► Concat [zᵗ, Proj(rᵗ)] ──► MLP ──► Δzᵗ
rᵗ (Logic Evidence từ DNF)  ───┘       (72 + 72 = 144)
                                                                 │
zᵗ⁺¹ = zᵗ + βᵗ · Δzᵗ ◄───────────────────────────────────────────┘
```

```python
class ConceptUpdateCell(nn.Module):
    def __init__(self, num_concepts=72, logic_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.logic_proj = nn.Sequential(
            nn.Linear(logic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_concepts),  # 512 -> 72
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * num_concepts, hidden_dim),  # [z (72), r_proj (72)] = 144
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, z_t, r_t):
        r_proj = self.logic_proj(r_t)
        cell_input = torch.cat([z_t, r_proj], dim=-1)
        return self.update_mlp(cell_input)
```

* **Hệ số cập nhật $\beta^t$**: Được tham số hóa học được:
  $$\beta^t = \text{sigmoid}(\text{raw\_\beta}^t) \times \beta_{\max}, \quad \text{với } \beta_{\max} = 0.5$$

---

## 7. Chi tiết vòng lặp suy luận lặp (Iterative Loop với K=2)

```
z⁰ ∈ R^(B × 72)  (Trích xuất từ SkateFormer backbone)
 │
 ├──► Lưu vào list [z⁰]
 ▼
CRLReasoner(z⁰)  [Shared Weights]
 ├──► r⁰ = [r_and⁰; r_and2⁰; r_or1⁰] ∈ R^(B × 512)
 └──► y⁰ ∈ R^(B × 120) (Action logits vòng 0, dùng cho L_aux)
         │
         ▼
ConceptUpdateCell(z⁰, r⁰)
 └──► Δz⁰ ∈ R^(B × 72)
         │
         ▼
z¹ = z⁰ + β⁰ · Δz⁰
 │
 ├──► Lưu vào list [z⁰, z¹]
 ▼
CRLReasoner(z¹)  [SAME SHARED WEIGHTS]
 ├──► r¹
 └──► y¹ (Action logits vòng 1, dùng cho L_aux)
         │
         ▼
ConceptUpdateCell(z¹, r¹)
 └──► Δz¹ ∈ R^(B × 72)
         │
         ▼
z² = z¹ + β¹ · Δz¹
 │
 ├──► Lưu vào list [z⁰, z¹, z²]
 ▼
CRLReasoner(z²)  [FINAL PASS — không update thêm]
 └──► yᴷ = y²  ← FINAL ACTION PREDICTION
```

---

## 8. Hệ thống Loss tinh gọn

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}}^K + \lambda_{\text{sup}} \cdot \mathcal{L}_{\text{sup}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}}$$

### 8.1. Thành phần giám sát $\mathcal{L}_{\text{sup}}$ (Phase 1+)
Gộp toàn bộ các loss có nhãn (nhãn action + nhãn concept), trọng số nội bộ cố định:
$$\mathcal{L}_{\text{sup}} = \mathcal{L}_c^K + 0.5 \cdot \mathcal{L}_c^0 + 0.3 \cdot \frac{1}{K}\sum_{t=0}^{K-1}\mathcal{L}_{\text{act}}^t$$

* $\mathcal{L}_c^K$: Asymmetric BCE loss trên concept cuối $z^K$.
* $\mathcal{L}_c^0$: Concept loss sớm trên $z^0$ để ép backbone học biểu diễn concept chuẩn ngay từ đầu.
* $\mathcal{L}_{\text{act}}^t$: Action Cross-Entropy trên các bước trung gian $y^t$ để đảm bảo logic evidence không bị thoái hóa.

**Công thức Asymmetric BCE (Chuẩn ASL Alibaba ICCV 2021 với Margin Clipping):**
$$\mathcal{L}_c(z, \hat{c}) = \frac{\sum_{i,c} r_{ic} \left[ \hat{c}_{ic} \cdot \text{softplus}(-z_{ic}) + (1 - \hat{c}_{ic}) \cdot \max(0, \sigma(z_{ic}) - m)^{\gamma_-} \cdot \text{softplus}(z_{ic}) \right]}{\sum_{i,c} r_{ic}}$$
*(Với $r_{ic}$: positive\_weight=1.0, negative\_weight=0.15, probability margin $m = 0.05$, $\gamma_- = 2.0$).*

### 8.2. Thành phần điều hòa $\mathcal{L}_{\text{reg}}$ (Phase 2+)
Gộp các loss tự điều hòa (không cần nhãn):
$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{sparse}} + \mathcal{L}_{\text{stable}}$$

* **Rule Sparsity Loss:** Ép thưa các trọng số luật để luật ngắn gọn và dễ giải thích:
  $$\mathcal{L}_{\text{sparse}} = \frac{1}{148 \cdot R_1}\sum_{i,j} w^{\text{and1}}_{ij} + \frac{1}{R_1 \cdot R_{2a}}\sum_{j,k} w^{\text{and2}}_{jk} + \frac{1}{R_1 \cdot R_{2b}}\sum_{j,k} w^{\text{or1}}_{jk}$$
* **Loop Stability Loss:** Giữ cho việc tinh chỉnh hội tụ, không bị dao động thất thường giữa các vòng:
  $$\mathcal{L}_{\text{stable}} = \frac{1}{K}\sum_{t=0}^{K-1} \|\sigma(z^{t+1}) - \sigma(z^t)\|_1$$

### 8.3. Bảng Hyperparameters tối ưu
Chỉ còn **duy nhất 1 hyperparameter thực sự cần search** ($\lambda_{\text{reg}}$):

| Hyperparameter | Giá trị đề xuất | Vai trò |
|---|---|---|
| $\lambda_{\text{sup}}$ | `1.0` (Cố định) | Cân bằng giữa nhận diện Action và nhận diện Concept |
| $\lambda_{\text{reg}}$ | `{0.001, 0.005, 0.01}` (Default: `0.005`) | Điều khiển độ thưa của luật và độ ổn định vòng lặp |

---

## 9. Lịch huấn luyện 2 Phase (Staged Training)

### Phase 1 — Concept Warm-up (Epoch 1 – 30)
```yaml
loss_act:    enable   # weight = 1.0
loss_sup:    enable   # lambda_sup = 1.0
loss_reg:    disable  # lambda_reg = 0.0

literal_mode: soft    # Gradient truyền mượt qua sigmoid
beta_max:     0.1     # Cập nhật nhẹ ở các vòng lặp đầu
```
*Mục đích:* Backbone SkateFormer và các concept heads hội tụ vững chắc trước khi áp đặt luật thưa.

---

### Phase 2 — STE & Rule Sparsification (Epoch 31+)
```yaml
loss_act:    enable   # weight = 1.0
loss_sup:    enable   # lambda_sup = 1.0
loss_reg:    enable   # lambda_reg = 0.005

literal_mode: ste     # Bật Straight-Through Estimator
temperature:  5.0 -> 10.0 (Annealing dần)
beta_max:     0.5

# Learning rates:
backbone_lr:     1e-5 (giảm mạnh hoặc freeze)
concept_head_lr: 1e-4
crl_lr:          1e-4
update_cell_lr:  2e-4
```
*Mục đích:* Ép đầu vào logic sang nhị phân $\{0, 1\}$, ép các trọng số luật về thưa để trích xuất DNF rules sắc bén.

---

## 10. `RuleExtractor` — Trích xuất luật DNF sau huấn luyện

**File:** `model/crl/rule_extractor.py`  
**Script:** `scripts/extract_rules.py`

### Quy trình trích xuất
1. Nạp checkpoint tốt nhất, đặt `model.eval()` và `literal_mode='ste'`.
2. Trích xuất các trọng số đã binarize:
   * $w^{\text{and1}} = (\sigma(\theta^{\text{and1}}) \ge \tau_{\text{and}})$.
   * $w^{\text{and2}} = (\sigma(\theta^{\text{and2}}) \ge \tau_{\text{and2}})$.
   * $w^{\text{or1}} = (\sigma(\theta^{\text{or1}}) \ge \tau_{\text{or}})$.
   * $W_{\text{action}} \in \mathbb{R}^{256 \times 120}$.
3. Ghép chuỗi mệnh đề cho từng Action $a$:
   * Lấy top các OR rules có trọng số lớn nhất trong $W_{\text{action}}[:, a]$.
   * Với mỗi OR rule, truy ngược về các AND rules thành phần.
   * Tạo chuỗi DNF dạng:
     $$\text{Action}_a \leftarrow \text{Clause}_1 \lor \text{Clause}_2 \lor \dots$$
4. Đánh giá trên tập Validation để tính:
   * **Rule Precision:** Tỷ lệ mẫu thỏa mãn luật thực sự thuộc về class đó.
   * **Rule Coverage:** Tỷ lệ mẫu của class được bao phủ bởi ít nhất 1 luật.
   * **Model-Rule Fidelity:** Tỷ lệ tương đồng giữa dự đoán của mạng nơ-ron và luật logic tượng trưng.

---

## 11. Danh sách Ablation Studies

| ID | Cấu hình | Mục đích chứng minh |
|---|---|---|
| **A0** | SkateFormer Backbone thuần | Upper-bound của mô hình hộp đen thuần túy |
| **A1** | REASON gốc (1 pass, không có loop) | Baseline mô hình diễn giải DNF đơn lẻ |
| **A2** | CLIR ($\beta = 0$, không có vòng lặp) | Đo lường hiệu quả riêng của kiến trúc DNF L1+L2 (tách khỏi iterative loop) |
| **A3** | CLIR (K=1 vs K=2 vs K=3) | Khảo sát số vòng lặp tối ưu |
| **A4** | **CLIR-SkateFormer (K=2, Logic-Guided Refinement)** | **Mô hình chính — đóng góp của luận văn** |
| **A5** | CLIR + Random Logic Feedback (r^t ngẫu nhiên) | Chứng minh logic evidence mang ngữ nghĩa thực sự, không chỉ là noise |
| **A6** | CLIR-Bypass (Có feature bypass từ backbone vào ActionHead) | Đo lường trade-off giữa Accuracy và Tính giải thích |

---

## 12. Kế hoạch triển khai mã nguồn (3 Tuần)

```
TUẦN 1: Tách và xây dựng Module CRL
  [1] model/crl/literal_layer.py: BinarizeSTE + LiteralLayer (Soft/STE modes)
  [2] model/crl/conjunction_layer.py: L1 ConjunctionLayer (Part-grouped) + L2a AND2
  [3] model/crl/disjunction_layer.py: L2b DisjunctionLayer (OR1)
  [4] model/crl/crl_reasoner.py: Tích hợp L0 + L1 + L2 + ActionHead

TUẦN 2: Tích hợp CLIR và Tinh gọn Loss
  [5] Viết ConceptUpdateCell trong clir_skateformer.py: Nhận r_t (512-dim) -> Δz_t (72-dim)
  [6] Viết IterativeLogicReasoner (vòng lặp K=2) trong clir_skateformer.py
  [7] Cập nhật hàm loss tinh gọn trong CLIRSkateFormer: compute_sup_loss() và compute_reg_loss()
  [8] Test forward/backward pass với batch giả lập

TUẦN 3: Huấn luyện, Rule Extraction và Viết báo cáo
  [9] Viết model/crl/rule_extractor.py và scripts/extract_rules.py
  [10] Chạy huấn luyện Phase 1 (Epoch 1-30) -> Phase 2 (Epoch 31-70)
  [11] Trích xuất luật DNF mẫu cho các hành động NTU và tính các chỉ số Fidelity/Coverage
  [12] Chạy các cấu hình Ablation A0, A1, A2, A4
```
