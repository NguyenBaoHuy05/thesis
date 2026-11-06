# So Sánh & Phân Tích Toàn Diện Các Mô Hình Trong Luận Văn
> **Tổng hợp so sánh:** SkateFormer Backbone, GCCR-Skateformer, ECR-SkateFormer, DCLN, REASON và GRAFT-SAR  
> **Thư mục lưu trữ:** `d:\Thesis\Architecture\`

---

## 1. Lộ Trình Tiến Hóa Kiến Trúc (Architectural Evolution)

Quá trình nghiên cứu và phát triển trong luận văn đi qua 6 kiến trúc chính, từ backbone nhận thức thuần túy đến framework Neurosymbolic thống nhất:

```
[Giai đoạn 1: Backbone]
SkateFormer ────────► Perception Backbone (Pure Transformer, Black-Box)
                           │
                           ▼
[Giai đoạn 2: Grouped Compatibility]
GCCR-Skateformer ───► JointSpatialTemporalCBM + 9 Group Heads + GCCR Compatibility Reasoner
                      Concept-template matching + Neural/Symbolic Residual Fusion
                           │
                           ▼
[Giai đoạn 3: Early CBM]
ECR-SkateFormer ────► Thêm Concept Bottleneck + KG-Primed DCR
                      Suy luận từ concept sau backbone
                           │
                           ▼
[Giai đoạn 4: Two-Stage Feedback]
DCLN ───────────────► Decoupled PRN (Soft Attention Bias ở tầng giữa) + ML-Decoder (Late Logic)
                           │
                           ▼
[Giai đoạn 5: Multimodal Grounding]
REASON ─────────────► Tích hợp LoRA CLIP + LLM Text Prompts + Concept Refiner
                           │
                           ▼
[Giai đoạn 6: Unified Framework]
GRAFT-SAR ───────────► Group-aware Decoder + 3 Nhánh + Residual Fusion + Temperature-Annealed Logic
                      Hợp nhất các hướng thiết kế của các thế hệ trước
```

---

## 2. Bảng So Sánh Đa Chiều Toàn Diện

### 2.1. Bảng 1: Cấu Trúc Tổng Quan & Khung Làm Việc

| Tiêu chí | SkateFormer | GCCR-Skateformer | ECR-SkateFormer | DCLN | REASON | GRAFT-SAR (Đề xuất) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phân loại kiến trúc** | Perception backbone | Grouped concept compatibility model | Neurosymbolic CBM | Decoupled Two-Stage NeSy | Multimodal Neurosymbolic | Unified Residual Neurosymbolic |
| **Backbone** | SkateFormer Transformer | SkateFormer Transformer | SkateFormer Transformer | CTR-GCN | HyperGCN / CTR-GCN | Backbone adapter |
| **Số nhánh xử lý** | 1 Nhánh tuần tự | 2 Nhánh (Neural + Compatibility) | 1 Nhánh concept–logic | 2 Giai đoạn | Các nhánh perception–language–logic | 3 Nhánh song song |
| **Cơ chế dung hợp (Fusion)** | Classifier | Residual logit fusion | Logic compatibility | Attention feedback + logic | Logic output | Neural–logic residual fusion |
| **Khả năng giải thích** | Không có concept trung gian | Concept theo nhóm + compatibility score | Concept + logic reasoner | Soft/Hard concepts + CRL | Concept + ngữ cảnh ngôn ngữ | Concept + rule + fusion |

---

### 2.2. Bảng 2: Cơ Chế Khái Niệm (Concept Bottleneck) & Căn Chỉnh Ngữ Nghĩa

| Tiêu chí | SkateFormer | GCCR-Skateformer | ECR-SkateFormer | DCLN | REASON | GRAFT-SAR (Đề xuất) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Số lượng concept** | 0 | 74, chia 9 nhóm | 72/phiên bản concept tương ứng | 74 (53 Spatial, 21 Temporal) | 78 concepts | 74 concepts |
| **Cấu trúc phân vùng** | Không | 6 giải phẫu + 3 thời gian | Spatial/Temporal | Spatial/Temporal | Nhóm bộ phận cơ thể | 9 nhóm chuyên biệt |
| **Cơ chế trích xuất** | Không | 9 group heads từ pooled feature | Dual Attention Pooling | ML-Decoder + PRN Banks | ML-Decoder + Refiner | Group-Aware Transformer Decoder |
| **Căn chỉnh ngôn ngữ (CLIP)** | Không | Không | Không | Không | LoRA CLIP | LoRA CLIP tùy chọn |
| **Đồ thị tri thức** | Không | S_KG/C_KG regularization | KG prior | Không | KG/logic tùy cấu hình | S_KG/C_KG regularization |

---

### 2.3. Bảng 3: Động Cơ Logic & Tính Ổn Định Số Học

| Tiêu chí | SkateFormer | GCCR-Skateformer | ECR-SkateFormer | DCLN | REASON | GRAFT-SAR (Đề xuất) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Công cụ logic/compatibility** | Không | GCCR compatibility scorer | `KGUnionLayer` | `ConceptLogicLayers` | `ConceptLogicLayers` | Differentiable Logic Engine |
| **Đầu vào reasoning** | Không | Group concept probabilities + class templates | Concept vector | Early/late concepts | Refined concepts | Concept probabilities |
| **Cơ chế kết hợp** | Không | Weighted group compatibility | L1/L2 logic | CRL + skip/feedback | CRL + language grounding | L1 clauses + linear aggregation |
| **Cơ chế giải thích** | Không | Positive/negative concept theo nhóm | Logic/template explanation | Concept/rule explanation | Concept + language context | Concept/rule explanation |

---

### 2.4. Bảng 4: Tính Mô-đun & Phân Tách Thành Phần

| Đặc tính kiến trúc | SkateFormer | GCCR-Skateformer | ECR-SkateFormer | DCLN | REASON | GRAFT-SAR (Đề xuất) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vị trí concept** | Không có | Sau backbone, 9 group heads | Sau backbone | Tầng giữa và cuối | Tầng cuối + refiner | Decoder phân nhóm |
| **Cơ chế tích hợp tri thức** | Không có | S_KG/C_KG regularization | KG prior/logic | Concept supervision | CLIP/LLM + logic | CLIP tùy chọn + KG regularization |
| **Mức độ coupling backbone** | Backbone chuyên biệt | Wrapper + SkateFormer adapter | Backbone + CBM | Gắn với CTR-GCN | Hỗ trợ HyperGCN/CTR-GCN | Adapter interface |
| **Cơ chế kết hợp đầu ra** | Classifier | Neural–compatibility residual fusion | Logic branch | Attention feedback + logic | Logic branch | Neural–logic residual fusion |

> **Cấu trúc `GCCR-Skateformer`:** Đây là một model/framework độc lập dựa trên SkateFormer backbone. `JointSpatialTemporalCBM` là wrapper model; bên trong, `JointSymbolicActionCompatibility` là module **Grouped Concept Compatibility Reasoner (GCCR)** theo 9 nhóm concept.

---

## 3. Phân Tích SWOT Chi Tiết Từng Mô Hình

### 3.1. SkateFormer
- **Điểm mạnh (S):** Có 4 loại phân vùng Type I–IV để mô hình hóa quan hệ khớp–thời gian.
- **Điểm yếu (W):** Hộp đen $100\%$, không có bất kỳ khả năng giải thích nào.
- **Cơ hội (O):** Làm nền tảng Backbone trích xuất đặc trưng cho các nghiên cứu XAI cấp cao.
- **Thách thức (T):** Khó ứng dụng trong y tế, an ninh, pháp lý vì không giải thích được lý do đưa ra quyết định.

### 3.2. GCCR-Skateformer
- **Điểm mạnh (S):** Tách backbone, 9 group concept heads, compatibility reasoner và neural classifier thành các thành phần riêng.
- **Điểm yếu (W):** Compatibility reasoner dựa trên concept templates và weighted group scores, chưa phải logic AND/OR dạng clause.
- **Cơ hội (O):** Là baseline trung gian để nghiên cứu ảnh hưởng của concept grouping và neural–symbolic fusion.
- **Thách thức (T):** Cần duy trì tính nhất quán giữa concept annotation, group mapping và action templates.

### 3.3. ECR-SkateFormer
- **Điểm mạnh (S):** Đặt nền móng đưa Concept Bottleneck và KG-Priming vào SkateFormer.
- **Điểm yếu (W):** Phụ thuộc vào chất lượng và độ đầy đủ của concept annotation; concept thưa có thể làm yếu tín hiệu logic.
- **Cơ hội (O):** Chứng minh tính khả thi của việc trích xuất cây luật logic từ Transformer.
- **Thách thức (T):** Hiện tượng gradient vanishing ở cổng hội logic dài.

### 3.4. DCLN
- **Điểm mạnh (S):** Cơ chế định tuyến PRN và bơm Attention Bias từ tầng sớm giúp định hướng thị giác tốt.
- **Điểm yếu (W):** Liên kết cứng với CTR-GCN, không tổng quát hóa được cho Transformer; dễ bị lỗi tràn số NaN dưới FP16.
- **Cơ hội (O):** Cung cấp cơ chế phân tách giữa Soft Concepts (thị giác) và Hard Concepts (logic).
- **Thách thức (T):** Chi phí tính toán của ML-Decoder lớn.

### 3.5. REASON
- **Điểm mạnh (S):** Tiên phong đưa căn chỉnh ngữ nghĩa đa phương thức (LoRA CLIP) và LLM Prompts vào skeleton reasoning.
- **Điểm yếu (W):** Chiến lược Staged Training gây sốc tối ưu tại Epoch 50; Divergence Loss triệt tiêu tính phối hợp tự nhiên giữa các chi.
- **Cơ hội (O):** Kết nối thị giác khung xương với không gian biểu diễn ngôn ngữ lớn (VLM/LLM).
- **Thách thức (T):** Tải nặng bộ nhớ và phụ thuộc vào pretrained CLIP weights.

### 3.6. GRAFT-SAR (Mô hình Luận văn Đề xuất)
- **Điểm mạnh (S):** Hợp nhất 3 nhánh song song, residual fusion, temperature annealing và adapter backbone trong cùng một framework.
- **Điểm yếu (W):** Số lượng hàm mất mát cần cân bằng nhiều (cần thiết lập trọng số $\lambda$ hợp lý).
- **Cơ hội (O):** Đạt tiêu chuẩn công bố bài báo tại các hội nghị/tạp chí hàng đầu thế giới (Q1, CVPR, ECCV, ICCV, NeurIPS).
- **Thách thức (T):** Cần đảm bảo chất lượng file ma trận định nghĩa nhãn Concept cho các bộ dữ liệu mới.

---

## 4. Kết Luận & Đóng Góp Khoa Học Của Luận Văn

1. **Tách biệt nhận thức và suy luận:** `GRAFT-SAR` duy trì direct neural branch song song với concept và logic branch nhờ cơ chế Residual Fusion.
2. **Chuẩn hóa Bộ Giải mã Khái niệm Không - Thời Gian (9-Group Anatomical Grounding):** Thay thế các bộ giải mã nặng nề bằng cấu trúc phân cấp giải phẫu học, kết hợp với căn chỉnh ngôn ngữ tự nhiên từ LoRA CLIP.
3. **Đột phá về Giải thuật Tối ưu Logic (Soft Annealed Binarizer):** Cung cấp giải pháp toán học mượt mà để truyền gradient qua các cổng logic mệnh đề bậc nhất mà không bị tắc nghẽn hay tràn số.
4. **Tính mô-đun:** Các thành phần adapter, decoder, CLIP alignment, knowledge graph và logic engine có thể được cấu hình hoặc thay thế độc lập.
