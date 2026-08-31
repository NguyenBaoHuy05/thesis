# Kiến Trúc Mô Hình REASON
> **Tên đầy đủ:** REASON: Rule-based Explainable Action via Symbolic cONcepts (2026)  
> **Mã nguồn tương ứng:** `d:\Thesis\REASON\` (`model/model.py`, `model/multi_pos_clip.py`, `model/crl/`, `llm_reranker/`, `clip/`, `hgcn/`)

---

## 1. Tổng Quan & Động Lực Thiết Kế

Các mô hình Concept Bottleneck trước đây (như ECR-SkateFormer hay DCLN) dựa hoàn toàn vào nhãn khái niệm nhị phân thô ($0/1$). Nhược điểm của cách tiếp cận này là:
1. Thiếu thông tin ngữ cảnh ngữ nghĩa tự nhiên (ví dụ: mô hình không hiểu "nâng cánh tay" liên quan gì đến từ vựng "uống nước").
2. Biểu diễn của các bộ phận cơ thể dễ bị chồng lấn và trùng lặp thông tin.

**REASON** là khung làm việc **Neurosymbolic Đa Phương Thức (Multimodal Neurosymbolic Framework)** kết hợp:
- **Thị giác khung xương (Skeleton Perception):** HyperGCN / CTR-GCN trích xuất đặc trưng không-thời gian.
- **Ngôn ngữ tự nhiên từ LLM & CLIP (LoRA CLIP Grounding):** Sử dụng các câu mô tả hành vi chi tiết do LLM sinh ra và căn chỉnh với đặc trưng khung xương qua mô hình CLIP được tinh chỉnh bằng LoRA.
- **Tinh chỉnh khái niệm & Trực giao hóa (Concept Refiner & Divergence Loss):** Tinh chỉnh xác suất đồng xuất hiện và triệt tiêu thông tin dư thừa giữa các chi.
- **Suy luận Logic Mệnh đề Bậc nhất (Differentiable First-Order Logic):** Tầng CRL suy diễn ra luật giải thích được.

```
      Skeleton Sequence X                                LLM-Generated Text Prompts
      (B × C × T × V × M)                              ("The right arm raises up...", etc.)
               │                                                          │
               ▼                                                          ▼
     [HyperGCN / CTR-GCN]                                      [LoRA CLIP Text Encoder]
     Spatio-Temporal Features                                   Text Feature Embeddings
        z ∈ ℝ^(B × 512)                                            e_text ∈ ℝ^(N_p × 512)
               │                                                          │
               ├──────────────────────┐                                   │
               │                      ▼                                   │
               │           [Multi-Positive CLIP Loss] ◄───────────────────┘
               │           Aligns Skeleton ↔ Text Space
               │                      │
               ▼                      ▼
       [Divergence Loss]    [ML-Decoder Concept Head]
       Enforces Body-Part     78 Raw Concept Logits
         Orthogonality                │
                                      ▼
                           [Concept Refiner (MLP)]
                          Logit-Space Residual Δc
                                      │
                                      ▼
                           Refined Concept Logits
                           c_final = c_raw + α · Δc
                                      │
                                      ▼
                        [Epoch 1 → 50: Warmup Concept]
                        [Epoch 50+: Activate Logic Layers]
                                      │
                                      ▼
                        [Concept Logic Layers (CRL)]
                        - L1 Conjunctions (AND Gate)
                        - L2 Disjunctions (OR Gate)
                        - KG-Prior Map Initialization
                                      │
                                      ▼
                         Action Classification Logits
                       + Symbolic First-Order Rules
```

---

## 2. Chi Tiết Kiến Trúc Kỹ Thuật

### 2.1. Bộ Mã Hóa Khung Xương (GCN / HyperGCN Encoder)

REASON hỗ trợ kiến trúc **HyperGCN** (Hypergraph Convolutional Network) hoặc CTR-GCN để mô hình hóa các quan hệ phi tuyến và siêu cạnh (Hyperedges) giữa nhiều khớp xương đồng thời. Đầu ra là vector đặc trưng không-thời gian $z \in \mathbb{R}^{B \times 512}$.

---

### 2.2. Căn Chỉnh Ngữ Nghĩa Đa Mẫu Dương (Multi-Positive CLIP Alignment)

REASON tạo ra một không gian ngữ nghĩa chung (Shared Conceptual Space) giữa cử động khung xương và mô tả ngôn ngữ tự nhiên:
1. **LLM Prompt Generation:** Sử dụng Large Language Models (như GPT-4) để tự động sinh ra các đoạn văn bản mô tả từng vi hành vi của từng bộ phận cơ thể (Full-body, Arm, Leg, Posture, Dynamics).
2. **LoRA-Adapted CLIP Text Encoder:** Mô hình CLIP (ViT-B/32) được đóng băng phần lớn và chỉ gắn các ma trận thích ứng cấp thấp (LoRA, $r=4$) vào các tầng chú ý của Text Encoder:
   $$W_{\text{text}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
3. **Multi-Positive CLIP Loss:** Do một hành động có thể kích hoạt nhiều khái niệm cùng lúc, REASON áp dụng hàm mất mát tương phản đa mẫu dương:
   $$\mathcal{L}_{\text{CLIP}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\sum_{p \in \mathcal{P}_i} \exp(\text{sim}(z_i, e_p) / \tau)}{\sum_{j} \exp(\text{sim}(z_i, e_j) / \tau)}$$

---

### 2.3. Khối Tinh Chỉnh Khái Niệm (Concept Refiner)

Để mô hình hóa mối tương quan phức tạp giữa các khái niệm (ví dụ: nếu `arm_raise` và `cup_hold` xuất hiện thì xác suất `hand_to_mouth` tăng lên), REASON thiết kế khối Concept Refiner hoạt động trực tiếp trong không gian Logit:
$$\hat{c}_{\text{refined}} = \hat{c} + \alpha_{\text{ref}} \cdot \text{MLP}(\hat{c})$$
- Hệ số $\alpha_{\text{ref}}$ là tham số học được (khởi tạo bằng $0.0$ để duy trì tính ổn định).
- Không dùng Sigmoid trong phần dư để cho phép mạng học cả việc **tăng cường (excitation)** lẫn **ức chế (suppression)** các khái niệm xung đột.

---

### 2.4. Hàm Mất Mát Trực Giao Hóa Chi Bộ Phận (Divergence Loss)

Để đảm bảo các kênh đặc trưng tương ứng với các bộ phận cơ thể khác nhau (Đầu, Tay, Chân, Thân) không bị trùng lặp thông tin, REASON sử dụng Divergence Loss nhằm tối thiểu hóa độ tương đồng Cosine giữa các biểu diễn:
$$\mathcal{L}_{\text{div}} = \frac{1}{|\mathcal{G}| (|\mathcal{G}| - 1)} \sum_{i \neq j} \text{ReLU}\left(\frac{v_i^T v_j}{\|v_i\|_2 \|v_j\|_2}\right)$$

---

### 2.5. Tầng Suy Luận Logic Phân Kỳ Hai Giai Đoạn (Staged Training Strategy)

Một điểm đặc trưng của REASON là chiến lược huấn luyện ngắt quãng theo giai đoạn (Staged Training):
- **Giai đoạn 1 (Epoch 1 $\to$ 50):** Đóng băng hoặc bỏ qua tầng Logic. Tập trung tối ưu hóa Skeleton Encoder, LoRA CLIP Text Encoder và ML-Decoder để các biểu diễn khái niệm đạt độ tin cậy cao.
- **Giai đoạn 2 (Epoch 51+):** Kích hoạt tầng suy luận logic `ConceptLogicLayers` (CRL), huấn luyện đồng thời cả hệ thống để học các luật hội tuyển DNF/CNF.

---

## 3. Ưu Điểm & Nhược Điểm

### 3.1. Ưu Điểm (Pros)
- **Tích hợp tri thức ngôn ngữ tự nhiên (VLM/LLM Grounding):** Đưa năng lực hiểu biết ngôn ngữ phong phú của CLIP và LLM vào bài toán nhận dạng khung xương, giúp các khái niệm có ý nghĩa ngữ nghĩa thực sự thay vì chỉ là nhãn số khô khan.
- **Khối Concept Refiner thông minh:** Tự động điều chỉnh xác suất khái niệm dựa trên ngữ cảnh đồng xuất hiện trong không gian Logit.
- **Khả năng giải thích bằng Logic Bậc nhất:** Cung cấp các biểu thức logic hình thức tường minh về nguyên nhân dẫn đến dự đoán hành động.
- **Chống dư thừa đặc trưng:** Divergence Loss giúp các biểu diễn bộ phận cơ thể phân hóa rõ rệt và trực giao với nhau.

### 3.2. Nhược Điểm (Cons)
- **Cú sốc tối ưu hóa ở Epoch 50 (Training Discontinuity):** Việc đột ngột kích hoạt tầng Logic ở Epoch 50 gây xung đột gradient mạnh, có thể làm phá vỡ các đặc trưng ổn định đã học được ở giai đoạn 1.
- **Xung đột giữa Divergence Loss và Gom cụm ngữ nghĩa:** Divergence Loss ép các vector biểu diễn trực giao có thể triệt tiêu các đặc trưng vốn có tính phối hợp tự nhiên (ví dụ: tay và chân phối hợp khi đi bộ).
- **Phụ thuộc nặng vào Pretrained Weights của CLIP:** Đòi hỏi phải nạp mô hình CLIP lớn, làm tăng dung lượng lưu trữ và tài nguyên tính toán trong quá trình huấn luyện.
- **Chưa có cơ chế Residual Fusion:** Toàn bộ dự đoán hành động ở giai đoạn 2 đi qua tầng Logic, nên nhánh nhận thức và nhánh giải thích chưa được tách biệt ở đầu ra.

---

## 4. Kết Luận

REASON mở rộng kiến trúc nhận dạng khung xương bằng một nhánh căn chỉnh skeleton–language, Concept Refiner và tầng logic. Các thành phần này tạo tiền đề cho việc thiết kế framework hợp nhất `GRAFT-SAR` với decoder concept phân nhóm và cơ chế fusion tách biệt.
