# Kiến Trúc Mô Hình DCLN
> **Tên đầy đủ:** Dual-stream / Decoupled Concept Logic Network (DCLN)  
> **Mã nguồn tương ứng:** `d:\Thesis\DCLN\` (`model_dcln.py`, `ml_decoder.py`, `crl/`)

---

## 1. Tổng Quan & Động Lực Thiết Kế

Hầu hết các mô hình Concept Bottleneck truyền thống chỉ trích xuất khái niệm ở tầng cuối cùng (Late Bottleneck), khiến các tầng biểu diễn bên trong mạng nơ-ron không nhận được sự hướng dẫn ngữ nghĩa. 

**DCLN (Decoupled Concept Logic Network)** được thiết kế theo kiến trúc **Hai giai đoạn phân rã (Decoupled Two-Stage)** kết hợp giữa:
1. **Khái niệm Mềm (Soft Concepts) & Bộ định tuyến PRN ở giai đoạn sớm (Early Stage):** Sử dụng các mặt nạ không gian - thời gian đóng băng (Spatial/Temporal Banks) để học các khái niệm mềm ngay tại các tầng giữa của mạng CTR-GCN, từ đó sinh ra **Attention Bias** bơm ngược vào các tầng tiếp theo để định hướng sự chú ý.
2. **Khái niệm Cứng (Hard Concepts) & Tầng Suy luận Logic CRL ở giai đoạn muộn (Late Stage):** Sử dụng bộ giải mã đa nhãn **ML-Decoder** để trích xuất các thuộc tính logic tường minh và đưa qua mạng suy luận logic khả vi (**CRL - Concept Reasoning Layers**).

```
                 Skeleton Coordinates X ∈ ℝ^(B × C_in × T × V × M)
                                         │
                                         ▼
                   [Early CTR-GCN Layers: l1 → l7]
                      Intermediate Features F_early
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       │
        [TemporalGraphPRN (Stage 1)]                         │
        - Spatial Concept Bank (53 masks)                    │
        - Temporal Concept Bank (21 masks)                   │
        - Neural Routers (Linear + ReLU)                     │
                     │                                       │
                     ▼                                       │
         Early Soft Concept Logits                           │
          + Spatio-Temporal Attention Bias                   │
                     │                                       │
                     └───────────────────┬───────────────────┘
                                         ▼
                           F_fused = F_early + 0.5 · Bias
                                         │
                                         ▼
                    [Late CTR-GCN Layers: l8 → l10]
                        Mature Features F_mature
                                         │
                                         ▼
                      [ML-Decoder (Multi-Query Attention)]
                     74 Hard Concept Logits (c_late)
                                         │
                                         ▼
                     [Concept Logic Layers (CRL Engine)]
                     - L1 Conjunctions (AND: Product T-norm)
                     - L2 Disjunctions (OR + Skip Connections)
                     - Forced FP32 Execution for Numerical Stability
                                         │
                                         ▼
                           Action Classification Scores
```

---

## 2. Chi Tiết Kiến Trúc Kỹ Thuật

### 2.1. Backbone CTR-GCN Phân Tách 2 Tầng (Decoupled CTR-GCN)

Mạng sử dụng 10 khối TCN-GCN chuẩn của CTR-GCN (Channel-Topology Refinement Graph Convolution):
- **Tầng sớm (Early Backbone - Khối $l_1 \to l_7$):** Xử lý đặc trưng từ kênh ban đầu ($C=3 \to 64 \to 128$). Tại khối $l_7$, đầu ra có kích thước $(N \cdot M, 128, T_{\text{early}}, V)$.
- **Tầng muộn (Late Backbone - Khối $l_8 \to l_{10}$):** Tiếp nhận đặc trưng đã được cộng hưởng Attention Bias và nâng số kênh lên $256 \to 512$, giảm chiều thời gian ($T \to T/4$).

---

### 2.2. Khối Định Tuyến Khái Niệm Sớm (TemporalGraphPRN - Stage 1)

Khối PRN (Prototype Routing Network) duy trì hai ngân hàng khái niệm đóng băng (Frozen Banks):
1. **Spatial Concept Bank (53 khái niệm):** Ma trận mặt nạ không gian nhị phân liên kết với các khớp xương cụ thể (Đầu, Bàn tay, Cánh tay, Hông, Chân, Bàn chân, Tương tác 2 người).
2. **Temporal Concept Bank (21 khái niệm):** Mặt nạ thời gian chia thành 3 phân đoạn chuyển động (Bắt đầu, Giữa, Kết thúc).

Khối định tuyến nơ-ron (Routers) tính toán trọng số kích hoạt và sinh ra bản đồ chú ý:
$$\text{Logits}_{\text{early}} = [\text{Router}_{\text{spa}}(F_{\text{early}}) \,\|\, \text{Router}_{\text{tem}}(F_{\text{early}})]$$
$$\text{Bias}_{\text{attn}} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V \implies F_{\text{fused}} = F_{\text{early}} + 0.5 \cdot \text{Bias}_{\text{attn}}$$

Đặc trưng sau khi bơm bias giúp các khối $l_8 \to l_{10}$ chỉ tập trung trích xuất những vùng khớp và khoảnh khắc thời gian mang thông tin ngữ nghĩa cao nhất.

---

### 2.3. Bộ Giải Mã Khái Niệm Đa Nhãn (ML-Decoder)

Ở giai đoạn muộn, thay vì dùng MLP đơn lẻ, DCLN áp dụng kiến trúc **ML-Decoder** (Multi-Label Decoder):
- Khởi tạo 32-64 truy vấn học được (Learnable Query Embeddings).
- Áp dụng cơ chế Cross-Attention giữa các Query Embeddings và đặc trưng khung xương $F_{\text{mature}}$ để giải mã đồng thời **74 thuộc tính hành động** (53 Spatial + 21 Temporal):
  $$\hat{c}_{\text{late}} = \text{ML-Decoder}(F_{\text{mature}}) \in \mathbb{R}^{B \times 74}$$
- Khả năng mô hình hóa sự đồng xuất hiện (Co-occurrence) giữa các nhãn khái niệm thông qua nhiều query và cross-attention.

---

### 2.4. Tầng Suy Luận Logic Khái Niệm (CRL - Concept Logic Layers)

Vector logits khái niệm $\hat{c}_{\text{late}}$ được chuyển đổi và đưa vào hệ thống suy luận logic khả vi:
1. **Biến đổi Literals:** Ghép cặp biến khẳng định và phủ định $\tilde{c} = [\sigma(\hat{c}), 1 - \sigma(\hat{c})]$.
2. **Khối tích logic $L_1$ (Conjunctions):** $h = \prod (1 - W_1 (1 - \tilde{c}))$.
3. **Khối tổng logic $L_2$ (Disjunctions) kèm Skip-connection:**
   $$\hat{y} = W_2 h + W_{\text{skip}} \tilde{c}$$
4. **Cơ chế ổn định số học (FP32 Guard):** Do phép toán `Product.apply()` tính $\log(X)$ dễ bị tràn số âm ($-\infty/\text{NaN}$) khi chạy ở chế độ FP16 / AMP, DCLN bắt buộc thực thi phân vùng logic trong không gian `float32`.

---

### 2.5. Hệ Thống Hàm Mất Mát Đa Giai Đoạn (Two-Stage Loss)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}}(\hat{y}, y) + \lambda_{\text{early}} \mathcal{L}_{\text{early}}(\text{Logits}_{\text{early}}, c^*) + \lambda_{\text{late}} \mathcal{L}_{\text{late}}(\hat{c}_{\text{late}}, c^*)$$

Trong đó:
- $\mathcal{L}_{\text{early}}$: BCE Loss giám sát các khái niệm mềm ở tầng giữa, định hướng luồng chú ý.
- $\mathcal{L}_{\text{late}}$: BCE Loss giám sát các khái niệm cứng ở tầng cuối trước khi suy luận logic.

---

## 3. Ưu Điểm & Nhược Điểm

### 3.1. Ưu Điểm (Pros)
- **Cơ chế phản hồi chú ý ngữ nghĩa (Semantic Attention Feedback):** Việc trích xuất concept từ tầng sớm và bơm attention bias ngược vào tầng sâu là thiết kế rất sáng tạo, giúp mạng nơ-ron nhận thức có định hướng rõ ràng.
- **Phân rã Soft vs. Hard Concepts:** Tách bạch giữa khái niệm dùng để điều hướng thị giác (Soft) và khái niệm dùng để suy luận logic hình thức (Hard).
- **Mô hình hóa quan hệ đa nhãn mạnh mẽ:** Tích hợp ML-Decoder giúp nắm bắt tương quan giữa 74 concepts tốt hơn MLP tuyến tính thông thường.
- **Tương thích tập dữ liệu phức tạp:** Đã được kiểm thử và hỗ trợ cả tập dữ liệu chuẩn NTU-60/120 và ANUBIS (`anubis_concepts.csv`).

### 3.2. Nhược Điểm (Cons)
- **Liên kết cứng với cấu trúc CTR-GCN (Tight Architectural Coupling):** Can thiệp trực tiếp vào giữa khối $l_7$ và $l_8$ của CTR-GCN khiến DCLN rất khó thay thế bằng các backbone khác như SkateFormer, ST-GCN hay Transformer một cách tổng quát.
- **Tăng chi phí tham số và tính toán:** Việc chạy cả TemporalGraphPRN và ML-Decoder khiến số lượng tham số và thời gian huấn luyện tăng khoảng $30-40\%$ so với CTR-GCN gốc.
- **Độ nhạy cảm số học:** Phép tính Product T-norm trong CRL cần có cơ chế kiểm soát dtype và miền giá trị khi chạy cùng Mixed Precision.
- **Không có căn chỉnh ngôn ngữ tự nhiên:** Hoàn toàn dựa vào nhãn nhị phân trong ma trận concept CSV, chưa khai thác được tri thức ngôn ngữ phong phú từ các mô hình nền tảng như CLIP hoặc LLM.

---

## 4. Kết Luận

DCLN là kiến trúc đưa thuộc tính ngữ nghĩa vào giữa backbone GCN thông qua PRN và Attention Bias, sau đó sử dụng một decoder riêng cho concept ở tầng cuối. Cách phân tách early feedback và late reasoning là nền tảng cho các cơ chế group-aware concept decoding ở những mô hình kế tiếp.
