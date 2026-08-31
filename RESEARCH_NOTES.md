# Tổng Hợp Thảo Luận Nghiên Cứu & Thiết Kế Kiến Trúc
## REASON-SkateFormer & Hierarchical Neuro-Symbolic Reasoning

---

## 1. Bản chất cải tiến của mô hình (REASON-SkateFormer)
So với SkateFormer nguyên bản (chỉ gồm Backbone trích xuất đặc trưng + Linear Classifier dạng Black-box), mô hình cải tiến kết hợp giữa **Perception (Thị giác sâu)** và **Neuro-Symbolic Reasoning (Suy luận Tri thức Ký hiệu)**:

1. **Perception Backbone:** SkateFormer trích xuất bản đồ đặc trưng không-thời gian $F \in \mathbb{R}^{B \times C \times T \times (V \cdot M)}$.
2. **Dual Attention Pooling:** Tách biệt 2 nhánh học chú ý:
   - **Spatial Attention Pool:** Gom theo chiều thời gian để trích xuất tư thế không gian $f_{spa}$.
   - **Temporal Attention Pool:** Gom theo chiều khớp để trích xuất động lực chuyển động $f_{tem}$.
3. **Concept Bottleneck Heads:**
   - Spatial Head ($f_{spa} \to 52$ Spatial Concepts).
   - Temporal Head ($f_{tem} \to 20$ Temporal Concepts).
   - Tổng cộng: **72 Concept ngữ nghĩa** giải thích được.
4. **KG-Primed Differentiable Logic Reasoner (`KGUnionLayer`):**
   - Đưa concept qua biểu diễn Literals $[c, 1 - c]$ ($144$ chiều).
   - Cổng logic hội (AND $\prod$) và tuyển (OR $1 - \prod$) khả vi.
   - 2 tầng suy luận ($L_1 \to L_2$) trước khi đưa ra nhãn hành động cuối cùng.
5. **Hệ thống đa hàm mất mát:**
   $$\mathcal{L}_{total} = \mathcal{L}_{action} + \lambda_{con} \mathcal{L}_{concept} + \lambda_{kg} \mathcal{L}_{KG} + \lambda_{part} \mathcal{L}_{part}$$

---

## 2. Bố cục đề xuất cho bản vẽ Sơ đồ Luận văn / Paper Q1
Nên chia làm **2 đến 3 hình vẽ chuẩn khoa học**:
- **Hình 1 (Kiến trúc tổng thể):** 
  - Giai đoạn 1: Backbone (Màu xanh lam).
  - Giai đoạn 2: Dual Attention Pooling & 72 Concepts (Màu cam).
  - Giai đoạn 3: KG-Primed Reasoner $L_1, L_2$ (Màu xanh lục).
  - Giai đoạn 4: Multi-Loss Supervision (Màu tím/đỏ viền nét đứt).
- **Hình 2 (Chi tiết khối KGUnionLayer & KG-Priming):** Phóng to cơ chế tính toán $W_{AND}, W_{OR}$ và cách ma trận tri thức (Prior) khởi tạo tầng 1.
- **Hình 3 (Case Study giải thích):** Trích xuất cây luật logic trực quan khi nhận diện một hành động cụ thể (vd: *Drink Water*).

---

## 3. Cơ chế Khởi tạo Trọng số (Weight Initialization)
- **Backbone:** Tải Transfer Learning từ pre-trained weights (`SkateFormer_j.pt`), bỏ qua `head.*`.
- **Khối Logic (`KGUnionLayer`):** Trọng số được lưu trong không gian Logit để khi qua Sigmoid luôn $\in [0, 1]$:
  $$W = \sigma(\theta), \quad \theta = \text{logit}(P + \epsilon), \quad \epsilon \sim \mathcal{N}(0, 0.02^2)$$
- **Khởi tạo Ma trận Prior $P$ (Tầng 1):**
  - Nền mặc định: $P = 0.03$.
  - Khái niệm hạt giống (Seed): $P_{i, j} = 0.95$ cho concept tương ứng.
  - Ánh xạ từ Knowledge Graph (`kg_prior_map`):
    - `require` (quan hệ đòi hỏi kéo theo): $P = 0.90 \times \text{strength}$.
    - `alongside` (quan hệ đồng xuất hiện): $P = 0.70 \times \text{strength}$.
    - `excludes` (quan hệ loại trừ): $P = 0.01 \times \text{strength}$.
  - Nửa sau (Negative literals $1 - c$): Giữ nguyên nền $0.03$ để mô hình tự do học logic phủ định.
- **Tầng 2 (L2):** Khởi tạo Identity Prior ($0.90$ trên đường chéo, $0.05$ ngoài đường chéo).

---

## 5. Tính Tĩnh vs. Động của Knowledge Graph (KG)
- **Hiện tại:** Cấu trúc đồ thị (`kg_prior_map`) là **TĨNH (Static / Hand-crafted)**, định nghĩa thủ công trong file YAML.
- **Tính học được (Learnable):** KG tĩnh đóng vai trò "mồi ban đầu" (Priming), còn các trọng số logic $W_{AND}, W_{OR}$ là tham số học (`nn.Parameter`), cho phép mạng tự động điều chỉnh độ mạnh yếu hoặc phát hiện thêm quan hệ mới trong lúc train.
- **Hướng mở rộng:** Có thể nâng cấp thành Adaptive KG Matrix hoặc trích xuất quan hệ tự động qua LLM.

---

## 6. Thiết kế Hidden Nodes & Hierarchical Symbolic Reasoning

### 6.1. Ba nghịch lý chí mạng cần tránh (Phản biện chuyên gia)
1. **Information Leakage:** Tránh dùng các Hidden Node tự do không ràng buộc (Unsupervised Latent Vector) vì gradient sẽ bypass 72 concepts và biến mô hình thành black-box.
2. **Product T-norm Gradient Vanishing:** Tránh nhân tích chuỗi quá dài qua nhiều tầng logic phẳng.
3. **Prior Drift:** Tránh để trọng số logic trôi tự do làm mất cấu trúc tri thức.

### 6.2. Giải pháp: Kiến trúc 4 tầng phân cấp (Hierarchy)
- **Tầng 1 (Atomic Concepts - 72 nodes):** Các vi khái niệm giải phẫu học cơ thể (Head, Hand, Arm, Hip, Leg, Foot, Dynamics).
- **Tầng 2 (Functional Hidden Nodes):** Cụm động lực học chức năng có mặt nạ nhóm (Group-Masked):
  - $H_{Upper}$: Nhóm thao tác chi trên (grasp + raise + lift_to_face).
  - $H_{Lower}$: Nhóm di chuyển và trụ (step + pivot + forward).
  - $H_{Posture}$: Tư thế và góc nhìn (tilt_down + sit/stand).
  - $H_{Dynamics}$: Trạng thái năng lượng và tốc độ (fast + rhythmic).
- **Tầng 3 (Action Schemas):** Kết hợp liên nhóm (Cross-group logic).
- **Tầng 4 (Action Classification):** Dự đoán lớp hành động cuối cùng.

---

## 7. Chiến lược Hàm mất mát Concept ($\mathcal{L}_{concept}$)

### 7.1. Vấn đề cốt lõi
- Tập nhãn Concept có độ thưa cực cao: **$> 90\%$ nhãn là 0 (Negative)**, chỉ $< 10\%$ nhãn là 1 (Positive).
- Sử dụng Standard BCE thông thường sẽ khiến gradient của nhãn 0 áp đảo, ép tất cả concept về gần 0 $\to$ Gây sụp đổ và "bóp nghẹt" tầng suy luận KG (Logic Starvation).
- Nhãn Concept ở mức lớp (Class-level definition) chứa nhiễu do biến thiên tự nhiên giữa các cá nhân (Intra-class variance).

### 7.2. Giải pháp: Positive-Prioritized & Slack Concept Loss
1. **Positive Prioritized Loss:** Đánh trọng số cao cho Positive Concepts (hoặc dùng Asymmetric Loss / Margin Slack Loss).
2. **Dung sai vùng an toàn (Slack Tolerance):** Cho phép concept âm $< 0.15$ và concept dương $> 0.70$ không bị phạt.
3. **Phòng chống False Positives:** Kết hợp phạt **CỰC NẶNG (Hard Penalty)** thông qua $\mathcal{L}_{KG}$ đối với các concept vi phạm luật loại trừ `excludes`.

---

## 8. Đánh giá Tiềm năng Đăng Tạp chí Q1 (Top-tier Checklist)
- **Ý tưởng & Hướng đi:** Rất tiềm năng, đúng trọng tâm xu hướng Neuro-Symbolic AI & XAI trong Thị giác máy tính.
- **Điều kiện cần để đỗ Q1:**
  1. *Benchmark Performance:* Giữ hoặc vượt Accuracy của SkateFormer gốc trên NTU-60, NTU-120.
  2. *Interpretability Evaluation:* Có mục trích xuất luật tường minh (Rule Extraction) và phân tích định tính từng Case Study.
  3. *Toàn diện về Ablation Study:* Bóc tách chi tiết tác dụng của KG-Priming, Dual Pooling, từng thành phần Loss ($\mathcal{L}_{KG}, \mathcal{L}_{part}, \mathcal{L}_{concept}$), và cấu trúc Hierarchical vs Flat.
  4. *Robustness / Generalization:* Đánh giá khả năng chống chịu khi thiếu dữ liệu (Few-shot) hoặc nhiễu che khuất khớp (Occlusion).

---

## Đề xuất chiến lược

Toàn bộ đề xuất cải tiến đã được tách sang [RESEARCH_PROPOSAL.md](D:/Thesis/RESEARCH_PROPOSAL.md).
