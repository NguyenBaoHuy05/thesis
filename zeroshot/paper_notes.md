# Ghi chú paper: hướng cải tiến cho model zero-shot trên Neuron

Ngày đọc: 2026-11-08. Các paper nằm ở `KLTN/paper/`.
Trạng thái: bỏ nhánh KG / NeSy (primitive + rule) vì không hiệu quả, nên các hướng bên dưới dựa trên Neuron, không dùng KG.

## Mốc so sánh ZSL (%)

| Phương pháp | NTU60 55/5 | 48/12 | NTU120 110/10 | 96/24 | Ghi chú |
|---|---|---|---|---|---|
| Neuron (CVPR25, baseline) | 86.9 | 62.7 | 71.5 | 57.1 | |
| SCoPLe (CVPR25) | 84.1 | 53.0 | 74.5 | 52.2 | |
| TDSM (ICCV25) | 86.5 | 56.1 | 74.2 | 65.1 | |
| FS-VAE (ICCV25) | 86.9 | 57.2 | 74.4 | 62.5 | |
| SkeletonContext (2026) | 89.6 | 64.4 | 74.2 | 60.1 | |
| DynaPURLS (2026) | 88.5 | 71.8 | 89.1 | 69.1 | **có TTA** (dùng luồng test) |

- Neuron yếu nhất ở split khó (48/12, 96/24): chỗ dễ có thêm điểm nhất.
- Kết quả có TTA (DynaPURLS, Skeleton-Cache) là transductive: phải so với dòng "+TTA" của mình, báo cáo riêng.
- Lấy từ bảng của SkeletonContext và DynaPURLS (bản pdftotext, bảng bị vỡ cột: nên kiểm tra lại với PDF trước khi đưa vào luận văn).

## Tóm tắt từng paper

| Paper | Ý chính | Dùng được gì |
|---|---|---|
| Skeleton-Cache (NeurIPS) | TTA không train: cache K=8 mẫu tự tin nhất mỗi class (theo entropy). Khoá = global + 4 bộ phận + 3 pha thời gian lấy từ feature map encoder. Trọng số theo class do GPT-4o cho, logit cuối = logit gốc + λ·logit cache. +4-8 điểm ZSL, +4-7 điểm H trên nhiều backbone | Gắn thẳng lên logit Neuron |
| DynaPURLS | Mô tả LLM đa mức (global, 4 bộ phận, 3 pha). Text làm query attend lên các node không-thời gian. Lúc test: học affine (scale, shift) trên text embedding bằng CE trên memory bank cân bằng class (K=16/class, ngưỡng confidence, 10 bước Adam lr 0.01). GZSL: cổng entropy | Tinh chỉnh semantics của Neuron lúc test |
| SCoPLe (CVPR25) | Prompt CLIP hai luồng (giữ attention gốc). "Joint giả" học được (joint-shaped prompts) nối thêm vào joint thật trong cross-attention. Cổng entropy cho GZSL | Prompt cho semantics, cổng GZSL |
| AMSF / Multi-View (TMM25) | 5 loại mô tả (nhãn, giải nghĩa và chuyển động; thủ công và LLM), gộp bằng MHA, Long-CLIP. Căn chỉnh bằng VAE. Cổng seen/unseen = logistic regression học trên validation tách từ class seen. Split easy/hard theo độ giống ngữ nghĩa | Gộp đa mô tả, GZSL trung thực, giao thức đánh giá |
| InfoCPL (TMM) | 100 câu GPT mỗi class (codebook). Top-k attention theo mẫu chọn mô tả. 12 nhánh ensemble với k khác nhau. Attention inversion lúc đầu train | Chọn mô tả theo mẫu |
| SkeletonContext (2026) | LLM sinh ngữ cảnh (môi trường, đồ vật dùng, đồ vật đích). Khôi phục slot [MASK] bằng BERT từ skeleton, che dần theo curriculum. Key-Part Decoupling = bản đồ trọng số joint | Ngữ cảnh đồ vật cho các cặp dễ nhầm |
| FS-VAE (ICCV25) | DCT trên quỹ đạo joint: khuếch đại tần thấp, nén tần cao. Mô tả local + global. Loss alignment có hiệu chỉnh (khoảng cách dạng sigmoid) | Augmentation, loss |
| TDSM (ICCV25) | Diffusion transformer khử nhiễu skeleton feature theo text. Triplet diffusion loss. Suy luận 1 bước ở timestep cố định | Ý triplet / margin loss; phần diffusion quá nặng |
| PCBEAR (supervised, không ZSL) | Concept bottleneck từ pose: phân cụm pose (tĩnh / động) thành concept, lớp phân loại thưa, can thiệp lúc test | Chủ yếu cho phần giải thích |

## Hướng đề xuất (không dùng KG), xếp theo ưu tiên

### 1. Skeleton-Cache trên Neuron: thắng nhanh, không cần train lại
- Khoá cache lấy từ cube (n, 256, 16, 25): global, 4 bộ phận (đầu, thân, tay, chân), 3 pha thời gian.
- Mỗi class unseen giữ K=8 mẫu, thay mẫu theo entropy. logit_cuối = z_neuron + λ_s · Σ_d w_c[d] · o^(d).
- Trọng số w_c: lấy từ LLM như paper. Bản rẻ hơn: trung bình đều, hoặc dò trên validation tách từ seen.
- Chạy trên test cube đã cache với checkpoint hiện có, không cần train lại. Batch 1 theo luồng; cố định seed thứ tự mẫu.

### 2. Tinh chỉnh semantics lúc test kiểu DynaPURLS
- Neuron đã có semantics 3 vòng spatial/temporal (`semantics/ntu/{spatial,temporal}`). Lúc test, học affine (scale, shift) trên embedding semantics của class unseen.
- Loss: CE trên memory bank cân bằng class (K khoảng 8-16, ngưỡng confidence), khoảng 10 bước.
- Theo paper, tăng mạnh nhất ở 48/12 và 96/24, đúng hai split Neuron yếu. Kết hợp được với hướng 1.

### 3. GZSL trung thực
- Hiện γ của calibrated stacking dò trên tập test (giống Neuron), nên kết quả lạc quan.
- Thay bằng: (a) cổng entropy (SCoPLe, DynaPURLS), hoặc (b) chọn γ / cổng trên validation tách một số class seen làm pseudo-unseen (AMSF).

### 4. Làm giàu semantics
- Nhiều mô tả cho mỗi class, gộp bằng trung bình, MHA (AMSF) hoặc top-k theo mẫu (InfoCPL).
- Mô tả theo bộ phận và theo pha (DynaPURLS).
- Slot ngữ cảnh đồ vật (SkeletonContext) cho các cặp như viết / gõ phím, đọc sách / chơi điện thoại.
- Prompt hai luồng cho text encoder (SCoPLe).

### 5. Loss và augmentation
- Margin / triplet giữa class đúng và class âm khó nhất (TDSM, FS-VAE).
- View DCT (FS-VAE) làm view cache thứ hai (`cache_views: 2`). Rủi ro: encoder frozen nên input đã chỉnh tần số sẽ lệch phân phối.
- "Joint giả" học được trong attention (SCoPLe).

### 6. Đánh giá
- Thêm split easy/hard (AMSF) và trung bình 3 split ngẫu nhiên (giao thức SMIE).

Thứ tự gợi ý: 1 → 3 → 2 → 4 → 5.

## Nếu sau này quay lại KG
- Grounding primitive chưa bao giờ được giám sát trực tiếp, chỉ học gián tiếp qua CE của class trên các class seen.
- Cả 71 primitive trong `primitive.json` đều có `definition` hình học chạy được trên toạ độ joint. Có thể sinh nhãn mềm (2, 71, 16) trong `extract()` của `main_nesy.py` (cạnh `second_person`) và thêm loss BCE.
- Đây có thể là lý do nhánh KG chưa hiệu quả: nên thử một lần trước khi kết luận hẳn.
- Cache của Skeleton-Cache có thể dùng vector presence của primitive làm khoá ký hiệu, và lấy trọng số theo class từ luật thay cho LLM.

### KG v2 (2026-11-08)
- File: `primitive_v2.json`, `kg_rules_v2.json`. Sinh bằng `node kg_tools/make_kg_v2.js .`; v1 giữ nguyên.
- Đo độ phân biệt bằng `node kg_tools/kgcmp.js <primitive> <rules>`, chỉ phân tích trên luật, không dùng dữ liệu.
- Lỗi của v1:
  - 7 cặp class có luật giống hệt nhau: writing = cutting paper; tear paper = open bottle; hush = sniff = yawn; cutting nails = fold paper; giving = exchange.
  - 230 cặp "yếu" trên NTU120: không có primitive nào bên này require còn bên kia forbid, và chỉ khác nhau ≤ 2 primitive.
  - Ngay trong các split chính thức có 7 cặp yếu, ví dụ writing ~ typing, thumb up ~ ok sign, drop ~ juggling.
  - Từ vựng thiếu các trục: một tay hay hai tay, giữ yên hay chuyển động, ngắn hay kéo dài, ngửa đầu, xoay cổ tay, chạm ngón cái, hai bên cùng với tay tới nhau.
  - Có primitive bị dùng làm "lấp chỗ trống": `hand.in_front_space` xuất hiện trong 33 luật.
  - `head.nod` không được luật nào dùng, kể cả class "nod head/bow".
- Thay đổi trong v2:
  - Thêm 11 primitive: `hands.both_active`, `hands.one_active`, `hands.near`, `hands.pull_apart`, `hand.static_hold`, `hand.wrist_twist`, `hand.thumb_index_touch`, `head.tilt_back`, `body.brief_motion`, `body.sustained_motion`, `inter.mutual_reach`. Mỗi primitive đều có `definition` hình học.
  - Định nghĩa lại `head.nod` để bao gồm cả cúi chào.
  - Thêm 4 nhóm loại trừ và 6 implication.
  - Sửa khoảng 35 luật.
- Kết quả trên NTU120: cặp giống hệt 7 → 0, cặp yếu 230 → 76. Bốn split chính thức: cặp yếu 7 → 0.
  - Còn 7 cặp chỉ tách được bằng `sequence` (các cặp mặc/cởi). Đây là chủ ý, không phải lỗi.
- Config `config/nesy/*.yaml` đã trỏ sang v2:
  - `work_dir` chuyển sang `nesy_neuron/v2_kg/`, `RUN = "v2_kg"` trong notebook (K đổi từ 71 lên 82 nên không resume được từ checkpoint v1).
  - CLIP text lưu cache mới ở `primitive_text_clip_v2.pt`. Cube cache của encoder dùng lại được.
- Lưu ý: các primitive mới về tay (chạm ngón cái, xoay cổ tay) dựa vào joint đầu ngón và ngón cái của NTU, vốn rất nhiễu. Nếu dùng giám sát từ `definition` thì nên kiểm tra tỷ lệ bật của từng primitive trước.

### Giám sát grounding trực tiếp: đã code (2026-11-08, chưa chạy)
- `Neuron/primitive_labels.py`: 82/82 primitive có detector, dịch từ `definition`.
  - So sánh mềm (sigmoid theo margin); nhãn tính cho từng frame và từng bên trái/phải.
  - Cùng crop với cube nhưng không xoay; pool 64 → 16 bước.
  - `body.staggering` chỉ là xấp xỉ: lắc ngang khi đang di chuyển, không tính hệ số biến thiên của bước chân.
  - `inter.hand_to_other_back` có thêm điều kiện khoảng cách < 0.5L.
- `main_nesy.py`:
  - Sinh và cache nhãn (`*_prim<hash>.npy`); kiểm tra thứ tự mẫu khớp với cube cache.
  - Loss `lambda_prim` dùng BCE với pos_weight lấy từ tỷ lệ nhãn.
  - Log `zsl_oracle`, `prim_auc_unseen`, `prim_auc_seen`, và ghi `prim_auc.json`.
- Model: thêm `ground_detach: True` (loss theo class không train grounding).
- Config: `lambda_prim: 1.0`, `prim_eval: True`.
- Notebook: thêm cell "Bước 0", chạy `primitive_labels.py`: in tỷ lệ bật, độ khớp luật, oracle trên train và test.
- Cách đọc kết quả:
  - `zsl_oracle` thấp → lỗi nằm ở luật hoặc định nghĩa.
  - `prim_auc_unseen` thấp → grounding chưa chuyển được sang class mới.
  - So `zsl_s_sym` với `zsl_oracle` → phần còn hụt là do perception.
- Ablation nên chạy:
  - `ground_detach: False` (grounding được cả CE theo class train).
  - `lambda_prim: 0` (bản cũ).

### Giám sát primitive: chọn nguồn nhãn nào (2026-11-08)
- **CSV của REASON** (`supervised/REASON/concepts/ntu_rgbd_60_120.csv`, 120 class × 74 concept nhị phân): không nên dùng cho zero-shot.
  - Nhãn ở mức class: mọi mẫu trong cùng class đều nhận cùng một vector, nên concept head dễ học thành "nhận diện class" chứ không học concept thật.
  - Số 0 vừa có nghĩa "không xảy ra" vừa có nghĩa "không quan tâm", nên BCE phạt sai cả những concept có thật trong mẫu.
  - Không có thông tin thời gian và bên trái/phải.
  - `writing` (unseen ở 55/5 và 96/24) có vector giống hệt `cutting_nails` (seen ở 96/24): không phân biệt được.
  - `foot_tap` và `seq_cascading` không class nào bật.
  - Trong supervised thì nhãn mức class vẫn ổn, vì class lúc test cũng là class lúc train.
- **KG**, theo thứ tự ưu tiên:
  1. Nhãn theo từng mẫu, từng frame, sinh từ `definition` hình học của primitive. Không phụ thuộc class, nên chuyển được sang class unseen.
  2. Với các primitive chưa code được detector: dùng nhãn class-level từ `kg_rules.json` dạng MIL. `require` → max theo thời gian phải bằng 1; `forbid` → mọi frame bằng 0; primitive không được nhắc tới thì bỏ qua, không tính loss.
- CSV chỉ nên dùng để đối chiếu: concept nào hai nguồn (KG và CSV) đồng ý thì tin hơn.

### Sau khi có nhãn primitive thì làm gì
Không cần align thêm. Nhãn chỉ dùng để sửa grounding (perception); phần suy luận vẫn là rule reasoner hiện có, vốn đã zero-shot vì luật của class unseen viết sẵn từ trước.

0. **Chẩn đoán trước, chưa cần train:**
   - Chạy detector của `definition` trên tập test unseen, đưa thẳng kết quả vào `RuleReasoner`. Được ZSL "oracle", tức trần của KG.
   - Oracle thấp: lỗi nằm ở luật hoặc định nghĩa → sửa KG, giám sát grounding không cứu được.
   - Oracle cao mà model thấp: lỗi nằm ở grounding → bước giám sát này có ích.
1. **Train:** L = L_neuron + λ_prim·BCE(p_logit, y_def) + λ_sym·CE(s_sym) + λ_fuse·CE(z).
   - Nên chặn gradient của CE theo class đi vào grounding (detach), hoặc để trọng số rất nhỏ.
   - Như vậy grounding chỉ học từ nhãn primitive, không học shortcut theo class seen.
   - Có thể warmup một vài epoch chỉ với BCE.
2. **Suy luận:** vẫn là z = τ·s_neu + β·s_sym. Ablation:
   - (a) primitive học được;
   - (b) primitive tính bằng detector (không học);
   - (c) kết hợp: trung bình hoặc chọn theo độ tin cậy của từng detector.
3. **Đo riêng độ chính xác primitive** (so với nhãn từ `definition`) trên class unseen, để chứng minh grounding chuyển được sang class mới. Đây là một kết quả NeSy riêng, đưa được vào luận văn.
