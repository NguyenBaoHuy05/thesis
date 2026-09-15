# Ghi chú: Buổi trao đổi về Tính Giải thích (Interpretability) & Debug HyperGCN

- **Ngày ghi chú**: 15/09/2026
- **Phạm vi**: Đánh giá tính giải thích của `JointSpatialTemporalCBM`, làm rõ ranh giới "symbolic/rule" thật sự,
  và một phiên debug thực tế khiến acc HyperGCN tụt từ 93% (standalone) xuống 79% khi ghép vào joint pipeline.
- **Tập tin liên quan**: [`joint_symbolic_model.py`](joint_symbolic_model.py), [`knowledge_graph.py`](knowledge_graph.py),
  [`eval_rulebook.py`](eval_rulebook.py), [`train_joint.py`](train_joint.py), [`feeders/feeder_ntu.py`](feeders/feeder_ntu.py),
  [`hypergcn/model/hypergcn_base.py`](hypergcn/model/hypergcn_base.py), [`hypergcn/model/hypergcn_large.py`](hypergcn/model/hypergcn_large.py)

---

## 1. Đánh giá kiến trúc hiện tại (Concept Bottleneck + Symbolic)

### Điểm mạnh
- Dự đoán bắt buộc đi qua 9 nhóm / 74 concept có tên (`joint_symbolic_model.py:51-67`) — concept bottleneck thật, không phải embedding ẩn.
- `explain()`/`explain_joint()` (`joint_symbolic_model.py:170-221`) phân rã điểm số theo từng nhóm cơ thể, có positive/negative concept + trọng số nhóm.
- `S_KG`/`C_KG` suy ra bằng Jaccard trực tiếp từ bảng concept CSV (`knowledge_graph.py:136-166`) — mọi entry truy vết được, khác hẳn `NTU60_ACTION_GROUPS` cũ (chính code tự phê bình là "hand-drawn... arbitrary").
- `eval_rulebook.py` đã có sẵn 1 nhánh suy luận thuần dữ liệu (KMeans-RB) để đối chiếu độc lập với nhánh symbolic — hạ tầng so sánh 3 chế độ (neural/symbolic/kmeans) đã tồn tại.

### Điểm yếu / rủi ro
- **Fusion cuối cùng vẫn trộn với nhánh black-box**: `action_logits = (1-α)·neural + α·symbolic`, `fusion_final_alpha=0.50` → chỉ giải thích được ~50% quyết định thật.
- **`ConceptGCN` gây concept leakage**: trộn logits giữa các concept qua `C_KG` trước khi ra xác suất cuối (`joint_symbolic_model.py:224-243`) → xác suất 1 concept không còn thuần phản ánh feature của chính nó.
  - Đã có cờ ablation sẵn: `use_concept_gcn` (yaml, `train_joint.py`, `eval_rulebook.py`).
  - **Nhận định**: lợi ích của `ConceptGCN` (và cả `loss_c`/`kg_c_weight`) phần lớn **dư thừa** — cấu trúc đồng-kích hoạt nó khai thác đã được dạy trực tiếp và mạnh hơn qua BCE (`concept_matrix[label]` quyết định target concept theo class, đã chứa sẵn thông tin đồng-xuất hiện).
  - **Khuyến nghị đã thống nhất**: tắt `use_concept_gcn`, **giữ nguyên** `kg_c_weight` (loss_c vẫn áp trên xác suất concept thô — đúng vai trò "nguyên bản" là loss regularizer, không phải cơ chế trộn cứng ở forward pass). Đây chính xác là cách `C_KG` hoạt động trong [`GCCR-Skateformer-copy`](../GCCR-Skateformer-copy/joint_symbolic_model.py#L362-L371) — không cần copy code, chỉ cần set 2 config này.
  - Nếu muốn giữ `ConceptGCN` mà không hi sinh faithfulness: chuyển thành self-distillation loss (chỉ dùng lúc train, không dùng cho `explain()`/inference) — nhưng cách này thực ra dư thừa so với việc giữ `loss_c` đơn giản, không cần làm thêm.
- **`group_logit` (attention theo lớp, num_class×9)** không tự diễn giải nếu không trực quan hoá riêng; không có ràng buộc sparsity ngoài entropy loss trên concept probs.
- Không có cơ chế can thiệp test-time (concept intervention) kiểu CBM chuẩn.
- Không có metric định lượng cho chất lượng giải thích (concept accuracy vs ground-truth, faithfulness, intervention accuracy).

---

## 2. `explain()` có thực sự là "luật" (rule) không? → KHÔNG, cần gọi đúng tên

Kết luận quan trọng nhất của buổi thảo luận, cần nhớ khi viết luận văn:

- Một **luật đúng nghĩa** (rule-based/symbolic AI) cần: điều kiện rời rạc/kiểm tra được (ngưỡng cố định), áp dụng **nhất quán** qua mọi instance thoả điều kiện, và một cơ chế kết hợp theo **toán tử logic cố định** (AND/OR/NOT, hoặc t-norm trong fuzzy logic).
- `explain()` chỉ là **argmax/argmin trên xác suất liên tục cho một sample cụ thể** — không có ngưỡng, luôn trả về "positive" dù cả nhóm đều mù mờ, và không đảm bảo cùng 1 class → cùng 1 concept "positive" qua các sample khác nhau (chưa kiểm chứng tính ổn định).
- Cơ chế tính điểm bên dưới (`JointSymbolicActionCompatibility.forward`) là weighted average + softmax attention **học bằng gradient** — về bản chất toán học đây là một phép tính khả vi (neural), chỉ khác là vận hành trên biến có tên.
- **Xác suất CÓ THỂ được coi là symbolic** (theo nghĩa probabilistic logic/Markov Logic Networks) — nhưng chỉ khi: (a) gắn với mệnh đề có tên rõ ràng (đạt — 74 concept có tên), VÀ (b) cách kết hợp chúng theo toán tử logic/xác suất có nguyên tắc cố định (KHÔNG đạt — `group_logit` học tự do qua backprop).
- → Gọi đúng: **"concept-based attribution / soft symbolic scorer"**, KHÔNG gọi "rule-based reasoning" hay "symbolic inference" khi bảo vệ luận văn.

### Nguồn để có LUẬT THẬT (nếu cần cho luận văn)
1. **Rule mining trực tiếp trên `concept_matrix` (CSV)** — Apriori/FP-Growth trên bảng nhị phân action×concept có sẵn, cho support/confidence thật, không cần chạy model. Rẻ nhất, độc lập với việc model train tốt hay không.
2. **Decision tree / RIPPER / CN2** trên concept xác suất đã nhị phân hoá (ngưỡng p>0.5) → action label. Mỗi đường đi root→leaf là 1 luật IF-THEN thật.
3. **`KMeansRuleBookClassifier`** (đã có trong `eval_rulebook.py`) thực ra gần symbolic hơn `JointSymbolicActionCompatibility` — là case-based reasoning (nearest-prototype), một nhánh symbolic AI được công nhận.
4. Nếu muốn giữ khả vi mà vẫn "cứng" hơn: thay `group_logit` học tự do bằng toán tử fuzzy logic cố định (t-norm/t-conorm) thay vì attention học — hướng redesign lớn hơn, có thể để future work.

### Không nên bỏ `JointSymbolicActionCompatibility`
Dù không phải "luật", nó vẫn có giá trị thật: (1) bị RÀNG BUỘC bởi template cố định — không thể học shortcut như `direct_action`; (2) đảm bảo faithfulness-by-construction (luôn phân rã được qua `explain()`); (3) là 1 trong 4 điểm trên phổ symbolic→neural đáng so sánh (decision tree cứng / soft scorer / KMeans case-based / neural thuần) — chính phổ này là nội dung thực nghiệm mạnh nhất có thể kể trong luận văn.

---

## 3. Nếu accuracy chỉ ngang baseline thì đóng góp là gì

Đây là tình huống bình thường của dòng nghiên cứu CBM (Koh et al. 2020 cũng không claim vượt SOTA). Vẫn có thể claim:
1. Khả năng giải thích per-concept mà backbone gốc không có ở bất kỳ mức accuracy nào.
2. Công cụ chẩn đoán lỗi (audit sai ở concept nào, không chỉ biết sai).
3. Câu hỏi khoa học độc lập với SOTA: "bao nhiêu % accuracy giải thích được chỉ bằng concept đặt tên rõ ràng" — chính là cái `eval_rulebook.py` đang đo.
4. Đóng góp về framework: adapter backbone-agnostic tái dùng được cho backbone khác.
- **Bắt buộc**: phải có case study định tính thật (vài sample cụ thể, in `explain()`, đối chiếu nhãn/video) — không có thì không trả lời được câu hỏi "vậy dùng để làm gì".

---

## 4. Đề xuất định hướng nếu đăng paper

**Góc**: nghiên cứu thực nghiệm về đánh đổi accuracy–interpretability qua 4 điểm trên phổ symbolic→neural
(decision tree cứng → KMeans case-based → soft symbolic scorer → neural thuần), backbone-agnostic (SkateFormer + HyperGCN).

**Không đăng theo hướng SOTA accuracy.** Venue phù hợp: workshop interpretable/trustworthy AI, hoặc tạp chí HAR — không phải track chính claim SOTA.

Đóng góp cụ thể: (1) concept bottleneck backbone-agnostic với ontology truy vết được; (2) đo lường định lượng cái giá
phải trả khi làm mềm dần ràng buộc symbolic; (3) phân tích per-class — lớp nào đủ giải thích bằng symbolic thuần, lớp
nào cần neural; (4) ablation `ConceptGCN` định lượng leakage vs lợi ích; (5) bài học kỹ thuật về hoán đổi backbone
trong khung concept-bottleneck (mục 5 bên dưới).

---

## 5. Phiên debug HyperGCN: 93% (standalone) → 79% (joint, frozen backbone)

Ba bug độc lập được tìm và sửa, theo đúng thứ tự loại trừ giả thuyết:

### Bug 1 — BatchNorm bị ép eval() không điều kiện khi fine-tune end-to-end
- **Vị trí**: `train_joint.py:302-304` (cũ) — `hasattr(model.backbone, "eval")` luôn `True` với mọi `nn.Module`.
- **Hệ quả**: khi `train_backbone: true` (config `*_e2e.yaml`), backbone vẫn bị ép eval mode → BatchNorm running stats không cập nhật dù trọng số đang bị gradient kéo lệch dần.
- **Vì sao lộ rõ ở HyperGCN chứ không ở SkateFormer**: HyperGCN dùng BatchNorm2d dày đặc (hàng chục lớp qua 9 `TCN_GCN_unit`), SkateFormer chỉ có đúng 1 `BatchNorm2d`, còn lại là `LayerNorm` (không có running stats, không bị ảnh hưởng bởi train/eval mode).
- **Đã sửa**: thêm tham số `train_backbone` vào `train_epoch()`, chỉ `model.backbone.eval()` khi **không** fine-tune end-to-end.
- **Lưu ý**: fix này KHÔNG áp dụng cho config không phải e2e (đa số trường hợp thực tế của user) — backbone đóng băng thì eval() vốn dĩ đúng.

### Bug 2 — NaN định kỳ mỗi epoch do `torch.cdist` chạy fp16 dưới AMP
- **Vị trí**: `hypergcn_base.py`/`hypergcn_large.py`, `HYPERGC.forward()`, nhánh `hyper=True` — `distance_x = torch.cdist(dis_v_x, dis_v_x)` không được ép fp32.
- **Bằng chứng**: log có `[WARN] Non-finite loss` lặp lại mỗi epoch, TẤT CẢ thành phần loss cùng NaN (kể cả `neural_action` — vốn độc lập với nhánh symbolic) → chứng tỏ `feat` (đầu ra backbone) bị NaN, lan ra toàn bộ downstream.
- **Gợi ý từ chính code**: tác giả gốc đã chủ động ép fp32 cho `hyper_norm()`/`a_norm()` trong cùng file (dòng ~109-127) nhưng bỏ sót đúng `cdist`.
- **Đã sửa**: `dis_v_x = dis_v_x.permute(0, 1, 3, 2).contiguous().float()` trước khi gọi `cdist`, ở cả 2 file.

### Bug 3 (nguyên nhân chính, ~11/14 điểm accuracy) — thiếu bước căn giữa khung xương (centering)
- **Phát hiện**: `hypergcn/feeders/feeder_ntu.py` (feeder gốc, tạo checkpoint 93%) có bước:
  ```python
  elif not self.vel:
      trajectory = data_numpy[:, :, 20]
      data_numpy = data_numpy - data_numpy[:, :, 20:21]
      data_numpy[:, :, 20] = trajectory
  ```
  (trừ toạ độ mọi joint cho joint 21 — spine base — giữ lại trajectory riêng của joint 21)
- **`feeders/feeder_ntu.py`** (dùng chung cho joint pipeline, mọi backbone) — nhánh `data_type == 'j'` chỉ `data_numpy.copy()`, **không hề căn giữa**.
- **Hệ quả**: backbone frozen (đã học trên toạ độ tương đối quanh cột sống) nhận toạ độ tuyệt đối (phụ thuộc vị trí người trong khung hình) → distribution shift lớn, đặc biệt hại cho `HYPERGC` vì dùng `torch.cdist` (khoảng cách Euclid tuyệt đối) để dựng hyperedge.
- **Vì sao SkateFormer không bị**: nhiều khả năng checkpoint gốc của SkateFormer cũng được train theo đúng quy ước "không căn giữa" của feeder dùng chung này (feeder có phong cách/API giống code gốc SkateFormer).
- **Đã sửa**: thêm tham số `center=False` vào `Feeder.__init__` (`feeders/feeder_ntu.py`), áp dụng đúng công thức trên khi `True`; bật `center = (backbone == "hypergcn")` trong `train_joint.py` và `eval_rulebook.py` — không đụng đến SkateFormer.

### Kết quả xác nhận (log thực tế)
| Giai đoạn | val (fused) | neural (riêng) |
| :--- | :---: | :---: |
| Trước fix (chỉ đóng băng, chưa sửa gì) | 79.55% | 79.51% |
| Sau fix centering | **90.44%** | **90.43%** |
| Chuẩn standalone (báo cáo của checkpoint gốc) | 93% | — |

- Khoảng cách còn lại (~2.6 điểm) nhiều khả năng do khác biệt protocol crop/eval (`uniform=True` + `p_interval=[0.95]` vs cách script gốc `hypergcn/main.py` đánh giá), hoặc do 93% là số ensemble đa-stream chứ không phải riêng stream `j` — chưa xác nhận, cần đối chiếu thêm nếu muốn khép hoàn toàn khoảng cách này.
- Log thêm cột `neural=` vào mỗi epoch (`train_joint.py`, hàm `evaluate()`) để tách bạch nhánh neural khỏi fused — dùng để chẩn đoán các trường hợp tương tự trong tương lai (nếu accuracy thấp bất thường, so `neural` với `val`: bằng nhau → lỗi ở backbone/feature; khác nhau nhiều → lỗi ở nhánh symbolic).

### Bài học chung (đáng đưa vào phần "lessons learned" nếu đăng paper)
Khi tái sử dụng checkpoint pretrained của 1 backbone bên ngoài (khác repo gốc) trong 1 pipeline mới:
1. Kiểm tra **tiền xử lý dữ liệu** (feeder) khớp với quy ước gốc trước tiên — đây là nguyên nhân lớn nhất, dễ bị bỏ qua nhất vì không hề crash.
2. Kiểm tra **numerical stability dưới AMP** cho các op không chuẩn (`cdist`, `topk`, phép chia/norm tự viết) — không phải mọi op đều được PyTorch tự động ép fp32.
3. Cẩn thận với `.eval()`/`.train()` không điều kiện khi có backbone bị đóng băng có điều kiện (`--train-backbone` dạng cờ bật/tắt).
