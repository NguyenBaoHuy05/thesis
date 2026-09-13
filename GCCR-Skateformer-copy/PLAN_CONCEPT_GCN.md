# Kế hoạch Triển khai: Kiến trúc Concept-GCN dựa trên Ma trận $C_{KG}$

- **Ngày lập kế hoạch**: 12/09/2026
- **Mục tiêu**: Chuyển đổi ràng buộc tri thức $C_{KG}$ từ **Hàm mất mát phụ trợ ($\mathcal{L}_c$)** sang **Lớp kiến trúc đồ thị (Concept-GCN Layer)** trực tiếp trong luồng tính toán thuận (Feed-forward).
- **Tập tin liên quan**: 
  - [`joint_symbolic_model.py`](file:///d:/Thesis/GCCR-Skateformer/joint_symbolic_model.py)
  - [`knowledge_graph.py`](file:///d:/Thesis/GCCR-Skateformer/knowledge_graph.py)
  - [`config/ntu/skateformer/ntu_cs_j.yaml`](file:///d:/Thesis/GCCR-Skateformer/config/ntu/skateformer/ntu_cs_j.yaml)

---

## 1. Động lực & Lý do Thiết kế

### Vấn đề hiện tại của $\mathcal{L}_c$:
- Hiện tại, 9 `group_heads` dự đoán concept một cách độc lập từ backbone feature.
- $C_{KG}$ chỉ được dùng làm hàm phạt batch correlation: $\mathcal{L}_c = \text{SmoothL1}(\text{Corr}(P), C_{KG})$.
- **Nhược điểm**: Phụ thuộc vào kích thước batch và phân phối mẫu trong batch. Nếu batch size nhỏ hoặc mẫu ít đa dạng, tương quan thực nghiệm $\text{Corr}(P)$ bị nhiễu.

### Giải pháp Concept-GCN:
- Biến ma trận tri thức tĩnh $C_{KG} \in \mathbb{R}^{73 \times 73}$ thành **Ma trận kề đồ thị (Graph Adjacency Matrix)**.
- Cho phép 73 concept "truyền tin" (Message Passing) cho nhau trực tiếp:
  - Concept cánh tay (`arm_raise`) hỗ trợ kích hoạt concept bàn tay (`hand_touch`).
  - Concept bước chân (`leg_step`) hỗ trợ concept chuyển động tiến tới (`temporal_direction_motion_forward`).
- **Ưu điểm**:
  1. Loại bỏ hoàn toàn sự phụ thuộc vào batch size của $\mathcal{L}_c$.
  2. Ràng buộc giải phẫu cơ thể trở thành **Inductive Bias** gắn chặt vào cấu trúc mạng thay vì chỉ phạt lỏng lẻo ở hàm loss.

---

## 2. Sơ đồ Kiến trúc Đề xuất

```
                         Skeleton Input (B, C, T, V, M)
                                       │
                                       ▼
                         SkateFormer Backbone
                                       │
                                       ▼
                            feat (B, feat_dim=192)
                                       │
                                       ▼
                       9 Concept Group Heads (Linear)
                                       │
                                       ▼
                       Raw Concept Tokens (B, 73, D_c)
                                       │
        C_KG (73, 73) ───►  ┌─────────────────────────┐
       (Adjacency Prior)    │    Concept-GCN Layer    │  <-- Message Passing giữa các Concept
                            └─────────────────────────┘
                                       │
                                       ▼
                          Refined Logits (B, 73)
                                       │
                                       ▼  torch.sigmoid()
                         Group Concept Probs (B, 73)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
      Joint Symbolic Action Scorer                KMeans Rule-Book
          (symbolic_logits)                       (Prototypes Matching)
```

---

## 3. Công thức Toán học

1. **Chuẩn hóa Ma trận kề đồ thị (Symmetric Normalization)**:
   Từ ma trận $C_{KG}$ tĩnh trong [`knowledge_graph.py`](file:///d:/Thesis/GCCR-Skateformer/knowledge_graph.py):
   $$\tilde{A} = \tilde{D}^{-\frac{1}{2}} (C_{KG} + I) \tilde{D}^{-\frac{1}{2}}$$
   với $\tilde{D}_{ii} = \sum_j (C_{KG} + I)_{ij}$.

2. **Cơ chế Truyền tin (Residual Concept-GCN)**:
   Gọi $Z \in \mathbb{R}^{B \times 73}$ là vector logits thô từ 9 group heads ghép lại:
   $$Z_{\text{msg}} = Z \cdot \tilde{A}^T$$
   $$Z_{\text{refined}} = Z + \text{Dropout}\Big(\text{GELU}(Z_{\text{msg}} \cdot W_{\text{gcn}} + b)\Big)$$
   Trong đó:
   - $W_{\text{gcn}} \in \mathbb{R}^{1 \times 1}$ (hoặc một lớp MLP nhỏ biến đổi đặc trưng).
   - Có kết nối tắt **Residual Connection** ($+ Z$) để giữ lại đặc trưng trực tiếp từ xương skeleton.

3. **Xác suất cuối cùng**:
   $$P = \text{Sigmoid}(Z_{\text{refined}})$$

---

## 4. Chi tiết Triển khai Code

### Bước 1: Thêm Module `ConceptGCN` vào `joint_symbolic_model.py`

```python
class ConceptGCN(nn.Module):
    """Truyền tin giữa 73 concept dựa trên ma trận tri thức C_KG."""
    def __init__(self, num_concepts: int = 73, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_concepts))
        self.bias = nn.Parameter(torch.zeros(num_concepts))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # z: (B, 73)
        # adj_norm: (73, 73)
        msg = torch.matmul(z, adj_norm.t())          # Lan truyền theo đồ thị
        refined = self.act(msg * self.weight + self.bias)
        return z + self.drop(refined)                 # Residual connection
```

### Bước 2: Tích hợp vào `JointSpatialTemporalCBM`

Trong hàm `__init__`:
```python
# Chuẩn hóa C_KG làm ma trận kề đối xứng
deg = C_KG.sum(dim=-1).clamp_min(1e-6)
deg_inv_sqrt = deg.pow(-0.5)
A_norm = deg_inv_sqrt[:, None] * C_KG * deg_inv_sqrt[None, :]
self.register_buffer("A_concept", A_norm)

self.concept_gcn = ConceptGCN(num_concepts=self.num_concepts)
```

Trong hàm `encode_group_concepts(feat)`:
```python
def encode_group_concepts(self, feat: torch.Tensor):
    raw_logits = []
    for name in self.group_order:
        raw_logits.append(self.group_heads[name](feat))
    
    # Nối thành full logits: (B, 73)
    z_all = torch.cat(raw_logits, dim=-1)
    
    # Truyền tin qua Concept-GCN
    z_refined = self.concept_gcn(z_all, self.A_concept)
    
    # Tách ngược lại về 9 groups cho Symbolic compatibility
    logits, probs = [], []
    start = 0
    for name in self.group_order:
        n = len(self.concept_groups[name])
        z_group = z_refined[:, start : start + n]
        logits.append(z_group)
        probs.append(torch.sigmoid(z_group))
        start += n
        
    return logits, probs
```

### Bước 3: Cấu hình tắt $\mathcal{L}_c$ trong YAML
Khi $C_{KG}$ đã nằm trong kiến trúc forward, ta có thể đặt trọng số loss về 0 trong file config:
```yaml
kg_c_weight: 0.00  # Đã có Concept-GCN đảm nhiệm, không cần hàm phạt L_c nữa!
```

---

## 5. Kế hoạch Thực nghiệm (Ablation Study)

| Cấu hình | Cơ chế sử dụng $C_{KG}$ | Số epoch hội tụ | Top-1 Action Acc | Rule-book (KMeans) Acc |
| :--- | :---: | :---: | :---: | :---: |
| **M1 (No C_KG)** | Không dùng | 30 | ... | ... |
| **M2 (Baseline GCCR)**| Hàm loss phạt $\mathcal{L}_c$ | 30 | ... | ... |
| **M3 (Đề xuất: Concept-GCN)** | **Lớp kiến trúc Concept-GCN** | 30 | ... | ... |

### Đóng góp bài báo / Luận văn nếu M3 thành công:
1. Đề xuất một kiến trúc mới: **Graph-Conditioned Concept Bottleneck Network (GC-CBM)**.
2. Chứng minh việc biến tri thức ontology thành **kiến trúc nơ-ron (architectural prior)** vượt trội hơn so với việc chỉ dùng nó làm **hàm phạt lỏng lẻo (loss regularizer)**.
