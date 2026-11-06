# Kiến Trúc Mô Hình ECR-SkateFormer
> **Tên đầy đủ:** Explainable Concept Reasoner SkateFormer (Joint Spatial-Temporal Concept Bottleneck & Differentiable Logic Reasoner)  
> **Mã nguồn tương ứng:** `d:\Thesis\ECR-SkateFormer\` và `d:\Thesis\GCCR-Skateformer\joint_symbolic_model.py`

---

## 1. Tổng Quan & Động Lực Thiết Kế

Mô hình SkateFormer gốc hoạt động như một "hộp đen" (Black-box) ở tầng phân loại cuối. Để kết hợp giữa **Năng lực Thị giác Sâu (Perception)** và **Khả năng Suy luận Ký hiệu (Neuro-Symbolic Reasoning)**, **ECR-SkateFormer** được phát triển nhằm:
1. Chèn một **Lớp thắt cổ chai Khái niệm (Concept Bottleneck Layer)** ngay sau SkateFormer Backbone để chuyển đổi đặc trưng tiềm ẩn thành các thuộc tính ngữ nghĩa con người hiểu được (ví dụ: `head_nod`, `hand_reach`, `arm_raise`, `temporal_fast`).
2. Sử dụng **Công cụ Suy luận Logic Khả vi (Differentiable Logic Reasoner - DCR / KGUnionLayer)** được khởi tạo từ **Đồ thị Tri thức (Knowledge Graph Prior)** để suy diễn hành động từ các khái niệm thay vì dùng lớp phân loại tuyến tính thông thường.

```
                      Skeleton Coordinates X ∈ ℝ^(B × C × T × V × M)
                                            │
                                            ▼
                               [SkateFormer Backbone]
                       Feature Map F ∈ ℝ^(B × C × T × (V·M))
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          [Spatial Attention Pool]                     [Temporal Attention Pool]
          (Gom theo trục thời gian T)                 (Gom theo trục khớp xương V·M)
                     │                                             │
                     ▼                                             ▼
             f_spa ∈ ℝ^(B × C)                             f_tem ∈ ℝ^(B × C)
                     │                                             │
                     ▼                                             ▼
           Spatial Concept Head                          Temporal Concept Head
         (52 / 53 Spatial Concepts)                    (20 / 21 Temporal Concepts)
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                                 Concept Bottleneck Vector
                                    c ∈ [0, 1]^(B × K)
                                            │
                                            ▼
                             Expanded Literals: [c, 1 - c]
                                   (2K chiều logic)
                                            │
                                            ▼
                          [KG-Primed Logic Reasoner (L1)]
                            Conjunctive Rules (AND Gate: ∏)
                           Khởi tạo từ Knowledge Graph Prior
                                            │
                                            ▼
                          [KG-Primed Logic Reasoner (L2)]
                            Disjunctive Rules (OR Gate: 1 - ∏)
                                            │
                                            ▼
                               Action Class Probabilities
                             + Human-Readable Decision Tree
```

---

## 2. Chi Tiết Kiến Trúc Kỹ Thuật

### 2.1. Cơ Chế Dual Attention Pooling (Gom Đặc Trưng Kép)

Thay vì dùng Global Average Pooling làm mất thông tin phân cấp, ECR-SkateFormer tách thành 2 nhánh Attention Pooling chuyên biệt:

1. **Spatial Attention Pool ($f_{spa}$):**
   - Gom chiều thời gian $T$, giữ lại tương quan giữa các khớp để trích xuất tư thế không gian (Spatial Poses & Anatomy).
   $$a_{spa} = \text{Softmax}\left(W_{spa} F\right), \quad f_{spa} = \sum_{t=1}^T a_{spa}^{(t)} \odot F^{(t)}$$
2. **Temporal Attention Pool ($f_{tem}$):**
   - Gom chiều khớp $V \cdot M$, giữ lại biến thiên thời gian để trích xuất động lực học (Motion Dynamics, Tốc độ, Chu kỳ).
   $$a_{tem} = \text{Softmax}\left(W_{tem} F\right), \quad f_{tem} = \sum_{v=1}^{V \cdot M} a_{tem}^{(v)} \odot F^{(v)}$$

---

### 2.2. Khối Thắt Cổ Chai Khái Niệm (Concept Bottleneck Heads)

Từ 2 vector $f_{spa}$ và $f_{tem}$, mô hình đưa qua các mạng MLP chuyên trách để dự đoán xác suất kích hoạt của bộ khái niệm nguyên tử. Ký hiệu $K$ cần được xác định theo phiên bản concept annotation đang sử dụng:
- **Nhánh không gian:** $f_{spa} \xrightarrow{\text{MLP}_{spa}} \hat{c}_{spa}$ (các nhóm Head, Hand, Arm, Hip, Leg, Foot và tương tác hai người).
- **Nhánh thời gian:** $f_{tem} \xrightarrow{\text{MLP}_{tem}} \hat{c}_{tem}$ (các nhóm Direction, Sequence và Dynamics).
- Ghép nối vector khái niệm tổng thể:
  $$c = [\hat{c}_{spa} \,\|\, \hat{c}_{tem}] \in [0, 1]^{72}$$

---

### 2.3. Tầng Suy Luận Logic Khả Vi Khởi Tạo Tri Thức (`KGUnionLayer`)

Để suy luận hành động từ các khái niệm $c$, mô hình mở rộng vector thành tập các biến mệnh đề khẳng định và phủ định (**Literals**):
$$\tilde{c} = [c, 1 - c] \in [0, 1]^{2 \cdot K}$$

Quá trình suy luận diễn ra qua 2 tầng logic chuẩn tắc hội tuyển (Conjunctive & Disjunctive Normal Form):

1. **Tầng 1 - Cổng Hội Logic (AND Conjunction $L_1$):**
   Tính tích logic (Product T-norm) có trọng số:
   $$h_j = \prod_{i=1}^{2K} \left(1 - W_{ij}^{(1)} (1 - \tilde{c}_i)\right)$$
   - Nếu $W_{ij}^{(1)} \to 1$, biến $\tilde{c}_i$ bắt buộc phải thỏa mãn.
   - Nếu $W_{ij}^{(1)} \to 0$, biến $\tilde{c}_i$ bị bỏ qua trong luật logic.

2. **Tầng 2 - Cổng Tuyển Logic (OR Disjunction $L_2$):**
   Tính tổng xác suất độc lập (Probabilistic Sum / De Morgan OR):
   $$\hat{y}_k = 1 - \prod_{j=1}^{H} \left(1 - W_{jk}^{(2)} h_j\right)$$

---

### 2.4. Cơ Chế Khởi Tạo Trọng Số Bằng Đồ Thị Tri Thức (KG-Priming)

Trọng số $W^{(1)}$ không khởi tạo ngẫu nhiên mà được nạp từ **Đồ thị Tri thức Tiên nghiệm (Knowledge Graph Prior Map)** thông qua hàm logit:
$$W = \sigma(\theta), \quad \theta = \text{logit}(P + \epsilon), \quad \epsilon \sim \mathcal{N}(0, 0.02^2)$$

Trong đó ma trận $P$ được gán giá trị theo luật quan hệ bản thể học:
- **Luật đòi hỏi (`require`):** $P_{i,j} = 0.90 \times \text{strength}$ (ví dụ: Hành động *Uống nước* đòi hỏi *Tay nâng lên miệng*).
- **Luật đồng hành (`alongside`):** $P_{i,j} = 0.70 \times \text{strength}$.
- **Luật loại trừ (`excludes`):** $P_{i,j} = 0.01 \times \text{strength}$ (ví dụ: *Ngồi xuống* loại trừ *Nhảy lên*).
- **Nền mặc định:** $P = 0.03$.

---

### 2.5. Hệ Thống Đa Hàm Mất Mát (Multi-Loss Supervision)

Mô hình được huấn luyện đồng thời bởi 4 hàm mục tiêu:
$$\mathcal{L}_{total} = \mathcal{L}_{action}(\hat{y}, y) + \lambda_{con} \mathcal{L}_{concept}(\hat{c}, c^*) + \lambda_{kg} \mathcal{L}_{KG}(W) + \lambda_{part} \mathcal{L}_{part}$$

1. $\mathcal{L}_{action}$: Cross-Entropy giữa dự đoán logic và nhãn hành động thực tế.
2. $\mathcal{L}_{concept}$: Positive-Prioritized BCE loss có dung sai (Slack Loss) để xử lý dữ liệu concept bị mất cân bằng trầm trọng (nhãn 0 chiếm $>90\%$).
3. $\mathcal{L}_{KG}$: Ràng buộc phạt trôi dạt trọng số logic so với Knowledge Graph gốc.
4. $\mathcal{L}_{part}$: Ràng buộc phân tách sự chú ý giữa các nhóm bộ phận cơ thể.

---

## 3. Ưu Điểm & Nhược Điểm

### 3.1. Ưu Điểm (Pros)
- **Tính minh bạch và giải thích được (Explainability):** Mở toang hộp đen của mạng nơ-ron; có thể trích xuất chính xác cây quyết định và luật logic mệnh đề (ví dụ: $\text{Drink Water} \iff \text{Arm\_Raise} \land \text{Hand\_To\_Mouth} \land \neg \text{Leg\_Kick}$).
- **Tích hợp tri thức chuyên gia (Domain Knowledge Integration):** Khởi tạo từ Knowledge Graph giúp mô hình không cần học mò từ đầu, hội tụ nhanh và ổn định hơn.
- **Phân tách không-thời gian tối ưu (Dual Pooling):** Tách biệt rõ ràng giữa tư thế tĩnh (Spatial Pose) và chuyển động động (Temporal Dynamics).
- **Khả năng can thiệp của con người (Human-in-the-loop):** Chuyên gia có thể sửa đổi trọng số logic $W$ hoặc thêm luật mới trực tiếp mà không cần huấn luyện lại toàn bộ backbone.

### 3.2. Nhược Điểm (Cons)
- **Hiện tượng "Đói Logic" (Logic Starvation):** Do nhãn concept rất thưa ($>90\%$ là 0), nếu dùng BCE chuẩn, gradient của nhãn âm sẽ ép tất cả concept về 0, làm triệt tiêu tín hiệu vào tầng logic.
- **Biến mất đạo hàm ở phép nhân T-norm (Gradient Vanishing):** Phép nhân chuỗi dài $\prod$ trong cổng AND dễ khiến gradient tiệm cận 0 khi số lượng concept lớn hoặc chuỗi quy tắc sâu.
- **Phụ thuộc vào Logic Bottleneck:** Việc ép toàn bộ thông tin đi qua vector concept làm backbone và logic branch phụ thuộc chặt vào chất lượng của concept representation.
- **Chi phí gán nhãn ma trận Concept:** Đòi hỏi phải xây dựng ma trận nhãn khái niệm cho toàn bộ 60 hoặc 120 lớp hành động của NTU RGB+D.

---

## 4. Kết Luận & Đóng Góp

ECR-SkateFormer là kiến trúc kết hợp backbone Transformer khung xương với Concept Bottleneck và tầng suy luận logic khả vi. Điểm đóng góp kiến trúc nằm ở việc biến đặc trưng tiềm ẩn thành các concept trung gian trước khi suy luận hành động.
