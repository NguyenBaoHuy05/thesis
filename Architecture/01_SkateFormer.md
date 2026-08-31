# Kiến Trúc SkateFormer Backbone & GCCR-Skateformer
> **Tên đầy đủ:** SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition (ECCV 2024)  
> **Tác giả:** Jeonghyeok Do, Munchurl Kim (KAIST VIC-Lab)  
> **Tên thư mục hiện tại:** **GCCR-Skateformer**  
> **Mã nguồn tương ứng:** `d:\Thesis\GCCR-Skateformer\` và `d:\Thesis\ECR-SkateFormer\`

> **Phạm vi:** `GCCR-Skateformer` là framework tập trung vào SkateFormer backbone kết hợp Concept Bottleneck và grouped concept compatibility. Thành phần chính gồm `JointSpatialTemporalCBM`, `JointSymbolicActionCompatibility`, bộ concept annotation, backbone adapter và công cụ late-fusion. Trong tài liệu này, “SkateFormer” chỉ backbone; “GCCR-Skateformer” chỉ model wrapper và toàn bộ mã nguồn mở rộng.

> **Tên gọi kiến trúc:** `JointSymbolicActionCompatibility` là **Grouped Concept Compatibility Reasoner (GCCR)** ở cấp module. Đây là bộ chấm điểm tương thích giữa xác suất concept theo nhóm và template concept của từng action; nó không phải một model độc lập và cũng không thực hiện phép AND/OR logic tường minh.

---

## 1. Tổng Quan & Động Lực Thiết Kế

Trong nhận dạng hành động người dựa trên chuỗi khung xương (Skeleton-based Human Action Recognition), các mô hình mạng nơ-ron đồ thị không-thời gian (ST-GCN) truyền thống thường bị giới hạn bởi:
1. **Ràng buộc topo đồ thị vật lý cố định:** Khó nắm bắt các mối liên kết ngữ nghĩa giữa các khớp xa nhau (ví dụ: tay cầm chân, hai bàn tay chạm nhau).
2. **Chi phí tính toán của Transformer chuẩn:** Nếu áp dụng Self-Attention toàn cục lên toàn bộ $T$ khung hình và $V$ khớp xương với $M$ người ($N = T \times V \times M$), độ phức tạp là $\mathcal{O}((T \cdot V \cdot M)^2)$, gây bùng nổ bộ nhớ GPU.

**SkateFormer** được thiết kế để xử lý vấn đề này bằng cách phân rã không gian đặc trưng khung xương thành **4 loại phân vùng quan hệ (Partitions)** và áp dụng cơ chế **Skate-Attention**. Phần này chỉ mô tả cấu trúc kiến trúc; hiệu năng thực nghiệm cần được trình bày ở phần thí nghiệm riêng.

```
                  Raw Skeleton Sequence X ∈ ℝ^(B × C_in × T × V × M)
                                      │
                                      ▼
                        Spatio-Temporal Embedding
                    + Partition-Specific Pos-Encoding
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
         Skeletal-Temporal Partitions          Gated Temporal Conv
             (Type I, II, III, IV)                   (G-TCN)
                     │                                 │
                     ▼                                 ▼
             Skate-Attention                    Temporal Dynamics
                     └────────────────┬────────────────┘
                                      ▼
                             Layer Normalization
                                      │
                                      ▼
                       SkateFormer Blocks (Stacked × L)
                                      │
                                      ▼
                          Global Average Pooling
                                      │
                                      ▼
                          Linear Action Classifier
                                      │
                                      ▼
                             Action Logits (ŷ)
```

---

## 2. Chi Tiết Kiến Trúc Kỹ Thuật

### 2.1. Phân Vùng 4 Loại Quan Hệ Không - Thời Gian (Type I - IV Partitions)

Để giảm độ phức tạp tính toán nhưng vẫn bao quát toàn diện các tương tác, SkateFormer định nghĩa 4 loại phân vùng dựa trên hai trục: **Khớp (Joint/Component)** và **Thời gian/Cơ thể (Temporal/Body)**:

| Loại phân vùng | Ký hiệu | Ý nghĩa vật lý / Ngữ nghĩa | Kích thước phân vùng |
| :--- | :--- | :--- | :--- |
| **Type I** | $P_{\text{joint-temp}}$ | **Joint-Temporal:** Chú ý xuyên thời gian của cùng một khớp xương (mô hình hóa quỹ đạo chuyển động của từng khớp). | $(V \cdot M) \times T$ |
| **Type II** | $P_{\text{comp-temp}}$ | **Component-Temporal:** Chú ý xuyên thời gian của từng nhóm chi bộ phận (tay, chân, thân mình). | $(K_{\text{comp}} \cdot M) \times (V / K_{\text{comp}} \cdot T)$ |
| **Type III** | $P_{\text{joint-body}}$ | **Joint-Body:** Chú ý không gian giữa các khớp trong cùng một khung hình thời gian (tư thế tức thời). | $T \times (V \cdot M)$ |
| **Type IV** | $P_{\text{comp-body}}$ | **Component-Body:** Chú ý giữa các nhóm chi qua lại trong cùng một khoảng thời gian (tương tác giữa tay và chân). | $(T / K_{\text{temp}}) \times (K_{\text{temp}} \cdot V \cdot M)$ |

Các phân vùng giúp giới hạn phạm vi chú ý theo từng quan hệ không-thời gian. Độ phức tạp cụ thể phụ thuộc vào cách triển khai kích thước phân vùng và cần được tính theo cấu hình thực tế.

---

### 2.2. Khối Skate-Attention (Skate-Attention Block)

Với mỗi phân vùng $p \in \{\text{Type I, II, III, IV}\}$, biểu diễn đầu vào $X_p$ được chiếu tuyến tính thành Query ($Q$), Key ($K$), Value ($V$):

$$Q = X_p W_Q, \quad K = X_p W_K, \quad V = X_p W_V$$

Ma trận chú ý phân vùng được tính toán kèm theo độ lệch tương đối (Relative Position Bias $B_p$):

$$\text{Attn}_p(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + B_p\right) V$$

Sau đó, các nhánh phân vùng được kết hợp thông qua một ma trận chiếu tuyến tính tổng hợp:

$$\text{SkateAttn}(X) = \text{Concat}(\text{Attn}_{\text{Type I}}, \text{Attn}_{\text{Type II}}, \text{Attn}_{\text{Type III}}, \text{Attn}_{\text{Type IV}}) W_O$$

---

### 2.3. Nhánh Tích Chập Thời Gian Có Cổng (Gated Temporal Convolution - G-TCN)

Bên cạnh Skate-Attention, SkateFormer tích hợp thêm nhánh G-TCN chạy song song để khai thác tính cục bộ và biến thiên vận tốc nhanh:

$$\text{G-TCN}(X) = \phi\left(X * W_f + b_f\right) \odot \sigma\left(X * W_g + b_g\right)$$

- Trong đó $\phi$ là hàm kích hoạt GELU/SiLU, $\sigma$ là hàm Sigmoid đóng vai trò cổng kiểm soát thông tin (Gating Mechanism).
- Phép toán $*$ là tích chập 1D theo chiều thời gian với kernel size thích ứng (ví dụ: $k=3, 5$).

---

### 2.4. Mã Hóa Vị Trí Đa Chiều (Spatio-Temporal Positional Encoding)

Khác với Transformer văn bản chỉ có 1 chiều, SkateFormer sử dụng bộ mã hóa vị trí kết hợp 4 thành phần:
$$PE = PE_{\text{spatial}}(\text{Joint Index}) + PE_{\text{temporal}}(\text{Frame Index}) + PE_{\text{person}}(\text{Person ID}) + PE_{\text{modality}}(\text{Joint/Bone/Vel})$$

Điều này giúp mô hình nhận thức chính xác từng khớp thuộc về ai, ở thời điểm nào, là dạng tọa độ hay xương/vận tốc.

---

## 3. Quy Trình Huấn Luyện & Đa Dòng (Multi-Stream Fusion)

SkateFormer thường được huấn luyện trên 4 dòng dữ liệu độc lập (4 modalities), sau đó ensemble điểm số (Logits fusion):
1. **Joint Stream (J):** Tọa độ tuyệt đối của khớp $(x, y, z)$.
2. **Bone Stream (B):** Vector xương kết nối giữa 2 khớp lân cận $(v_i - v_j)$.
3. **Joint Velocity (JV):** Vận tốc biến thiên tọa độ khớp theo thời gian $(x_{t+1} - x_t)$.
4. **Bone Velocity (BV):** Vận tốc biến thiên của vector xương.

$$y_{\text{ensemble}} = w_J y_J + w_B y_B + w_{JV} y_{JV} + w_{BV} y_{BV}$$

---

## 4. Ưu Điểm & Nhược Điểm

### 4.1. Ưu Điểm (Pros)
- **Mô hình hóa quan hệ linh hoạt:** Cho phép học tương tác giữa các khớp trong những vùng quan hệ được định nghĩa thay vì chỉ phụ thuộc vào cạnh đồ thị vật lý.
- **Phân rã không-thời gian:** Bốn loại phân vùng cung cấp các góc nhìn bổ sung về quỹ đạo khớp, quan hệ bộ phận và tư thế toàn thân.
- **Khả năng mở rộng theo modality:** Có thể dùng làm backbone cho các dòng Joint, Bone, Joint Velocity và Bone Velocity; việc ensemble các dòng là một pipeline bên ngoài backbone.

### 4.2. Nhược Điểm (Cons)
- **Hộp đen hoàn toàn (Complete Black-Box):** Mạng nơ-ron sâu trực tiếp dự đoán phân phối xác suất hành động qua một lớp `nn.Linear` cuối cùng. Không thể giải thích *tại sao* mô hình đưa ra quyết định đó.
- **Không có tính diễn giải ngữ nghĩa (No Semantic Interpretability):** Các trọng số chú ý (Attention weights) trong SkateFormer rất dày đặc và trừu tượng, không thể ánh xạ trực tiếp thành các khái niệm chuyển động của con người (ví dụ: "cúi người", "giơ tay lên", "bước chân trái").
- **Dễ phụ thuộc vào tương quan giả (Spurious Correlations):** Khi gặp dữ liệu ngoài phân phối (OOD) hoặc bị che khuất một phần cơ thể (occlusion), mô hình dễ đưa ra dự đoán sai lệch mà không thể kiểm soát hay can thiệp vào quá trình suy luận.
- **Chi phí suy luận vẫn cao hơn GCN thuần túy:** So với ST-GCN hoặc CTR-GCN cơ bản, SkateFormer có số lượng tham số lớn hơn và thời gian tính toán dài hơn.

---

## 5. Kết Luận & Vai Trò Trong Luận Văn

SkateFormer là **xương sống nơ-ron nhận thức (Perception Backbone)** trong chuỗi nghiên cứu của luận văn. `GCCR-Skateformer` cung cấp wrapper Concept Bottleneck, GCCR compatibility reasoner và công cụ ensemble để kết nối backbone này với các mô hình cải tiến tiếp theo (`ECR-SkateFormer`, `REASON`, `GRAFT-SAR`).
