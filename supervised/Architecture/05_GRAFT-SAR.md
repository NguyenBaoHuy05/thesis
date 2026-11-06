# Kiến Trúc Mô Hình GRAFT-SAR (Unified Framework)
> **Tên đầy đủ:** Group-aware Residual Adaptive Fusion with Temperature-Annealed Logic for Skeleton Action Recognition (`GRAFT-SAR`, trước đây là `ThesisModel`)
> **Mã nguồn tương ứng:** `d:\Thesis\GRAFT-SAR\` (`model/graft_sar.py`, `model/concept_decoder.py`, `model/crl/`, `knowledge_graph.py`, `model/backbones/`)

---

## 1. Tổng Quan & Động Lực Thiết Kế

Qua quá trình phân tích các mô hình tiền nhiệm (`SkateFormer`, `ECR-SkateFormer`, `DCLN`, `REASON`), ta nhận thấy một vấn đề kiến trúc thường gặp trong các hệ thống Neurosymbolic:
- **Phụ thuộc vào Logic Bottleneck:** Khi toàn bộ dự đoán phải đi qua tầng concept và logic, nhánh nhận thức và nhánh giải thích bị ghép cứng vào nhau.
- **Hiện tượng nghẽn gradient & Bất ổn định số học:** Các hàm làm tròn cứng (Hard STE), tích logic dài (Product T-norm) hoặc chiến lược huấn luyện ngắt quãng (Staged Training ở Epoch 50) gây sốc tối ưu hóa.

**`GRAFT-SAR`** được thiết kế như một **Khung Làm Việc Neurosymbolic Thống Nhất**, sử dụng Group-aware Concept Decoder, **3 Nhánh Song Song kèm Dung Hợp Thặng Dư** và cơ chế **Soft Temperature-Annealed Logic**.

```
                           Skeleton Coordinates (B, C, T, V, M)
                                            │
                                            ▼
                        [Universal Multi-Backbone Adapter]
                       (SkateFormer / CTR-GCN / HyperGCN)
                     Spatial-Temporal Feature Tokens ∈ ℝ^(B × N_tok × D)
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     │                      │                      │
                     ▼                      ▼                      ▼
               [BRANCH 1]             [BRANCH 2]             [BRANCH 3]
             Direct Neural        Group-Aware Concept       Differentiable
              Classifier          Transformer Decoder        Logic Engine
                   │             (9 Anatomical Groups)             │
                   │                        │                      │
                   │                        ▼                      │
                   │              9 Group Projections              │
                   │             + LoRA CLIP Grounding             │
                   │             + Knowledge Graph                 │
                   │               (S_KG, C_KG)                    │
                   │                        │                      │
                   │                        ▼                      │
                   │              Concept Probabilities            │
                   │                p ∈ [0, 1]^(B × 74) ───────────┤
                   │                                               ▼
                   │                                    Soft Annealed Binarizer
                   │                                     → L1 Conjunctions (∧)
                   │                                     → L2 Disjunctions (∨)
                   ▼                                               ▼
             Neural Logits                                   Logic Logits
               y_neural                                        y_logic
                   │                                               │
                   └──────────────────────┬────────────────────────┘
                                          ▼
                      Residual Fusion: y_final = (1 - α)·y_neural + α·y_logic
                                          │
                                          ▼
                        • Action Classification Logits
                        • Human-Readable First-Order Logic Explanations
```

---

## 2. Chi Tiết Kiến Trúc Kỹ Thuật

### 2.1. Bộ Điều Phối Đa Xương Sống (Universal Multi-Backbone Adapter)

`GRAFT-SAR` tách rời phần trích xuất đặc trưng và phần suy luận logic thông qua lớp trừu tượng `BaseBackboneAdapter`:
- Hỗ trợ chuyển đổi linh hoạt chỉ bằng cấu hình YAML:
  1. `SkateFormerAdapter`: Spatio-Temporal Transformer.
  2. `CTRGCNAdapter`: Channel-Topology Refinement GCN.
  3. `HyperGCNAdapter`: Hypergraph Neural Network.
  4. `STGCNAdapter`: Spatial-Temporal Graph Convolution kinh điển.
- Đầu ra chuẩn hóa: Trích xuất cả Pooled Feature $z \in \mathbb{R}^{B \times D}$ và Tokens $T_{\text{mem}} \in \mathbb{R}^{B \times N_{\text{tok}} \times D}$.

---

### 2.2. Nhánh 1: Phân Loại Nơ-ron Trực Tiếp (Direct Neural Head - $y_{\text{neural}}$)

Nhánh này đóng vai trò mạng nơ-ron nhận thức độc lập:
$$y_{\text{neural}} = \text{Linear}(\text{Dropout}(z)) \in \mathbb{R}^{B \times C_{\text{action}}}$$
- Duy trì một đường dự đoán trực tiếp song song với concept và logic branch, tách biệt nhánh nhận thức khỏi logic bottleneck.

---

### 2.3. Nhánh 2: Bộ Giải Mã Khái Niệm 9 Phân Vùng & Ràng Buộc Đồ Thị Tri Thức

Thay vì dùng ML-Decoder nặng nề, `GRAFT-SAR` thiết kế **Group-Aware Transformer Concept Decoder** phân cấp theo 9 nhóm giải phẫu và động lực học:
1. **9 Nhóm Khái niệm chuyên biệt:**
   - 6 Nhóm giải phẫu: `Head`, `Hand`, `Arm`, `Hip`, `Leg`, `Foot`.
   - 3 Nhóm động lực học thời gian: `Temporal Direction`, `Temporal Sequence`, `Temporal Dynamics`.
2. **Căn chỉnh Ngữ nghĩa với LoRA CLIP (Multimodal Grounding, tùy chọn):**
   - Các đặc trưng nhóm được chiếu sang không gian 512 chiều và căn chỉnh với các text prompt tự nhiên qua `LoRATextEncoder` và `MultiPositiveClipLoss`.
3. **Ma trận quan hệ concept/action (KG-derived priors):**
   - $S_{KG} \in \mathbb{R}^{C \times C}$: Ma trận độ tương đồng ngữ nghĩa giữa các lớp hành động.
   - $C_{KG} \in \mathbb{R}^{K \times K}$: Ma trận đồng kích hoạt giữa các khái niệm.
   - Trong implementation hiện tại, $S_{KG}$ và $C_{KG}$ được dùng cho regularization; $A_{KG}$ là tiện ích độc lập và chưa nối trực tiếp vào forward path của `GRAFT-SAR`.

---

### 2.4. Nhánh 3: Động Cơ Suy Luận Logic Khả Vi Cải Tiến (Differentiable Logic Engine)

Động cơ logic của `GRAFT-SAR` sử dụng temperature annealing để điều khiển relaxation của các literal trong quá trình huấn luyện:
1. **Làm tròn Mềm Luyện Nhiệt Độ (Soft Temperature-Annealed Binarizer):**
   Thay thế Hard STE bằng hàm Sigmoid co dãn tham số nhiệt độ $k$:
   $$b(p) = \sigma(k \cdot (p - 0.5)), \quad k(t) = k_{\text{min}} + (k_{\text{max}} - k_{\text{min}}) \frac{t}{T_{\text{total}}}$$
   - Ở Epoch đầu ($k=3.0$): Hàm mượt, gradient lan truyền dồi dào, tránh tắc nghẽn.
   - Ở Epoch cuối ($k=20.0$): Hàm tiệm cận hàm bước nhảy nhị phân, cho phép trích xuất luật logic hình thức chuẩn xác.
2. **Tầng Hội Logic $L_1$ (Conjunctions - AND Rules):**
   Tổng hợp các biến khẳng định $p$ và phủ định $1-p$ qua ma trận trọng số khả vi $W_{\land} \in [0, 1]^{R \times 2K}$.
3. **Tầng tổng hợp quy tắc $L_2$ (Linear Rule Aggregation):**
   Trong implementation hiện tại, các clause được ánh xạ thành điểm số hành động bằng `LinearLogicClassifier`; đây không phải probabilistic OR thuần túy.

---

### 2.5. Cơ Chế Dung Hợp Thặng Dư Khả Học (Learnable Residual Fusion)

Dự đoán cuối cùng là sự kết hợp thích ứng giữa trực giác nơ-ron (Neural Perception) và lập luận logic (Symbolic Reasoning):
$$y_{\text{final}} = (1 - \alpha) \cdot y_{\text{neural}} + \alpha \cdot y_{\text{logic}}$$
- Tham số $\alpha \in [0, 1]$ được khởi tạo tại $0.5$ và tối ưu hóa tự động qua quá trình huấn luyện bằng Gradient Descent.

---

### 2.6. Hàm Mất Mát Tổng Thể Đa Mục Tiêu (Unified Multi-Task Objective)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{final}} + \lambda_{\text{neu}} \mathcal{L}_{\text{neural}} + \lambda_{\text{log}} \mathcal{L}_{\text{logic}} + \lambda_{\text{con}} \mathcal{L}_{\text{bce}} + \lambda_{\text{clip}} \mathcal{L}_{\text{clip}} + \lambda_{\text{kg}} \mathcal{L}_{\text{kg}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

---

## 3. Ưu Điểm & Nhược Điểm

### 3.1. Ưu Điểm (Pros)
- **Tách biệt nhận thức và suy luận:** Direct Neural Head tồn tại song song với concept và logic branch.
- **Giải thích theo concept và rule:** Cung cấp công cụ `explain.py` để hiển thị concept nổi bật và các rule có trọng số liên quan đến lớp dự đoán:
  $$\text{Drink Water} \iff (\text{arm\_raise} \land \text{hand\_reach\_mouth}) \lor (\text{head\_tilt\_down} \land \text{hand\_hold\_object})$$
- **Temperature annealing:** SoftBinarizeLayer cung cấp relaxation có kiểm soát cho gradient của logic branch.
- **Tính mô-đun hóa:** Backbone được truy cập thông qua adapter, cho phép thay thế các backbone đã đăng ký theo factory/configuration.

### 3.2. Nhược Điểm (Cons)
- **Độ phức tạp trong cân bằng siêu tham số (Hyperparameter Tuning):** Cần điều chỉnh hệ số cân bằng giữa 7 thành phần hàm mất mát ($\lambda_{\text{neu}}, \lambda_{\text{log}}, \lambda_{\text{con}}, \lambda_{\text{clip}}, \lambda_{\text{kg}}, \dots$).
- **Thời gian huấn luyện trên mỗi Epoch tăng nhẹ:** Do phải tính toán đồng thời 3 nhánh (Neural, Concept, Logic) và lan truyền ngược qua LoRA CLIP.
- **Đòi hỏi ma trận Concept chuẩn hóa:** Cần tệp định nghĩa khái niệm chất lượng cao (`concepts/ntu_rgbd_60_120.csv`) để đạt hiệu quả giải thích tối đa.

---

## 4. Kết Luận

`GRAFT-SAR` là framework hợp nhất backbone adapter, group-aware concept decoder, CLIP alignment tùy chọn, KG regularization, differentiable rule engine và residual fusion trong một pipeline thống nhất.
