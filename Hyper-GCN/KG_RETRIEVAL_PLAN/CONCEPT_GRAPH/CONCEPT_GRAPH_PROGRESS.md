# Concept–Action Logic Graph: tiến độ

> Plan: [CONCEPT_GRAPH_PLAN.md](CONCEPT_GRAPH_PLAN.md). Sơ đồ: [drawio/architecture_concept_graph.drawio](../../drawio/architecture_concept_graph.drawio).
> Số liệu chép từ log chạy thật; ô chưa có số ghi *chờ*. Dev: **1 seed (0)**.
> Cập nhật lần cuối: 2026-10-20

## Quyết định đã chốt (2026-10-19)

- Nhóm ∨ ở v1: chỉ posture (hip / leg static stand ∨ sit).
- Ngưỡng đạt của P1 như plan §7.
- Cache S = 4 (khoảng 13 GB) được phép, dùng ở E2-S4.
- Baseline REASON: E1b = port lớp logic của REASON lên decoder và cache của mình. REASON đầy đủ để ở P4.
- E1b dùng cùng lr với decoder (1e-3); REASON gốc dùng 1e-4 cho logic trong lịch 110 epoch end-to-end.
- P1: trọng số cạnh α cố định bằng 1 (`learn_alpha: False`), để E1c chỉ khác CL v2 đúng ở ¬(j|i).
  α học được để từ P2.

## P0: code

| File | Trạng thái |
|---|---|
| `model/concept_graph.py`, `tools/build_concept_graph.py` | xong; `tests/test_concept_graph.py` 7/7 qua (local, không cần torch) |
| `model/fuzzy_executor.py`, `model/concept_graph_head.py` | xong; test trong `tests/test_fuzzy_executor.py` (cần torch, *chờ* Colab) |
| `model/reason_logic.py` (port REASON, MIT) | xong; test *chờ* Colab |
| `model/pair_metrics.py` (PCR, per-pair, K5-pair) | xong; test *chờ* Colab |
| `train_concept_logic.py`: `head: cl \| graph \| reason`, chỉ số cặp cho mọi head và cho K0 | xong |
| `config/concept_graph/e1{b,c,d,e}.yaml` | xong (E1a = `config/concept/cl_v2.yaml`) |
| R-GCN theo tầng, nhánh action, head đầy đủ (`model/rgcn.py`, `model/concept_action_graph.py`, `return_tokens`) | xong 2026-10-20; `tests/test_concept_action_graph.py` 7/7 qua (Colab) |

Đồ thị v1 trên v2.4 (`python tools/build_concept_graph.py`):

| | Số lượng |
|---|---|
| Motif ∧ | 25 (114 / 120 lớp có ít nhất 1 motif; 476 ô của M được phủ bằng motif) |
| Cổng ∨ | 2 (posture hip, leg) |
| Cặp dễ nhầm | 58; node ¬(j\|i): 113 (3 hướng có M_j∖M_i rỗng bị bỏ); 66 lớp có ít nhất 1 ¬ |
| Cạnh | r1 56, r2 4, r3 563, r4 226, r5 186, r6 / r7 113, r8 216 |

Ví dụ: ¬(take_off_a_shoe | put_on_a_shoe) = 1 − ∧(motion_upward, seq_hands_first, lower_then_raise).

## Script chạy (Colab)

Ablation và các biến thể đã xoá tạm (2026-10-20, người dùng quyết), lấy lại từ git (commit `56872b5`):
config E1b–E1e, E2, S = 4, ft_e3 / ft_k0, `config/concept/` (CL v1 / v2); code `model/concept_graph_head.py`,
`model/reason_logic.py` (port REASON), CL head + `logic_scores`, `flat_graph` / `random_pair_graph`, cờ `use_rgcn`,
`freeze_node_emb`, `use_r6`. Chỉ còn mô hình chính (và E3 trên cache, `train_concept_logic.py`).

**Ô 1: Drive + code (mỗi phiên)**
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/Hyper-GCN
!git pull
!git log --oneline -1
```

**Ô 2: mô hình chính, một lệnh** (điền 2 đường dẫn `FILL` trong `config/concept_graph/run_refine.yaml` một lần)
```
!python tools/run_refine.py
```
Chạy test → chép dữ liệu → `main.py` với `config/concept_graph/online_e3.yaml`, kết quả ở
`/content/drive/MyDrive/BaoHuy/work_dir/concept_graph/online_e3`. Epoch 0 phải ra Top1 87.63. Dán các dòng `Top1`,
`Top1 of the other outputs`, `Mean loss components` ở mỗi lần eval (epoch 0, 5, …, 30).

## P1: kết quả

Quy tắc đạt (chốt trước): `PCR(tốt nhất của E1c, E1d) ≤ 0.8 × min(PCR E1a, PCR E1b)` **và** Top1 ≥ Top1(E1a) − 0.3
**và** PCR < PCR(E1e). Chênh trong 10% quanh ngưỡng thì chạy thêm seed 1, 2.

Dự đoán ghi trước: E1c giảm PCR 15–30% so với E1a, Top1 ±0.5. E1b có Top1 cao hơn E1a nhưng PCR tương đương.

| | Top1 / Top5 | PCR | Top1 trong / ngoài cặp | sửa / phá vs K0 | K5 hỗ trợ / ngẫu nhiên | K5-pair | train top1 cuối |
|---|---|---|---|---|---|---|---|
| K0 | 87.63 / 97.56 | **3.58** | 85.54 / 90.23 | | | | |
| E1a (CL v2) | 85.60 / **94.98** | 4.18 | 83.59 / 88.12 | 804 / 1836 | **14.49** / 0.00 | **67.30** | 99.80 |
| E1b (logic REASON), lần 1 | sụp | | | | | | 4.45 |
| E1b, lần 2 (công thức REASON + init mặc định) | 83.80 / 91.20 | 6.55 | 81.09 / 87.19 | 600 / 2548 | 20.41 / 0.02 | 43.96 | 99.00 |
| E1c (+¬, ω = 1) | **86.05** / 93.79 | **4.17** | **84.43** / 88.09 | 740 / 1542 | 7.69 / 0.00 | 38.20 | 99.94 |
| E1d (+¬, ω = 0) | 85.83 / 93.59 | **4.17** | 84.34 / 87.69 | 693 / 1610 | 10.19 / 0.00 | 43.19 | 99.91 |
| E1e (¬ ngẫu nhiên) | 86.01 / 94.06 | 4.19 | 84.15 / **88.33** | 699 / 1525 | 8.85 / 0.00 | 59.76 | 99.94 |

PCR chỉ tính trên 28 277 mẫu test có nhãn thuộc ít nhất một cặp.
Cặp nhầm nhiều nhất, giống nhau ở mọi head: make_OK_sign / make_victory_sign khoảng 27%,
put_on_a_shoe / take_off_a_shoe 11–13%, thumb_up / make_victory_sign khoảng 11%,
writing / play_with_phone_tablet khoảng 6%, giving_object / exchange_things khoảng 5–6%.

### Kết luận P1 (2026-10-19, seed 0): **C1 không đạt**

- Ngưỡng: PCR ≤ 0.8 × min(E1a, E1b) ≤ 0.8 × 4.18 = **3.34**. Tốt nhất của E1c / E1d là **4.17**: chỉ giảm 0.01 điểm.
  Ngưỡng không phụ thuộc E1b, vì E1b chỉ có thể làm ngưỡng thấp hơn.
- Cách ngưỡng rất xa (25%, không phải trong vùng 10%), nên **không chạy thêm seed**.
- Đối chứng E1e (¬ trên cặp ngẫu nhiên) có PCR 4.19, gần như bằng E1c. Top1 cũng tăng như E1c (+0.41 so với +0.45).
  Phần Top1 tăng thêm vì vậy không đến từ cặp đúng.
- **Mọi head trên concept đều nhầm cặp nhiều hơn K0** (4.17–4.19 so với 3.58). Top1 trong cặp cũng thấp hơn K0
  khoảng 1–2 điểm. Nút thắt concept làm mất bớt thông tin phân biệt cặp mà backbone có.
- Giải thích: AUC concept là 0.99 ở mọi head. Concept vẫn chỉ là mã của lớp (ECOC), như A1 / A1-G.
  ¬(j|i) là một hàm của cùng vector c, nên khi decoder sai lớp thì các concept phân biệt cũng sai theo (lỗi tương quan).
  Đổi trọng số các literal (ω, ν, cặp đúng hay ngẫu nhiên) không thêm thông tin.
- Ngoài lề: CL v2 hơn CL v1 1.5 điểm Top1 (85.60 so với 84.09). K5 tăng từ 6.6% lên 14.5% nhưng chưa tới 20% như dự đoán.
  K5-pair của E1a là 67%: bật các concept phân biệt thì dự đoán chuyển sang đối tác. Logic phản ứng đúng với concept,
  nhưng concept thì không đúng ở các mẫu khó.

- E1b (logic REASON trên cùng decoder, cùng cache): **thấp hơn CL v2 1.8 điểm Top1 và nhầm cặp nhiều hơn**
  (PCR 6.55 so với 4.18). Cặp drop / thumb_down nhầm tới 57.8%: concept bị nhị phân hoá trước khi vào logic, nên mất
  mức độ của concept, và các cặp chỉ khác nhau 1–2 concept dễ bị gộp. AUC concept 0.974, thấp hơn CL (0.987–0.989).
  K5 20.4% là cao nhất (logic rời rạc nhạy với từng concept). τ dừng ở 0.64.
  Kết luận phụ: AND mềm theo ma trận (CL v2) hơn logic học được kiểu REASON trên cùng concept, cả Top1 lẫn PCR.

**Theo quy tắc của plan: dừng, không làm P2–P3** cho tới khi có hướng đem lại thông tin mới ở các cặp.
E1b vẫn chạy lại để có baseline REASON đầy đủ.

## Đổi mục tiêu (2026-10-19, người dùng quyết)

Diễn giải không còn là ưu tiên; concept không thể đúng ở từng mẫu vì nhãn concept gán theo lớp.
Đồ thị được coi là **nhánh hiệu chỉnh của K0**, không phải nút thắt. Thước đo chính: **Top1 của fusion
K0 + head so với K0 (87.63)**. Mọi lần chạy giờ in thêm:
`fusion K0 + λ head` = log_softmax(K0) + λ·log_softmax(head), λ = `fusion_lambda` (mặc định 1, chốt trước, không dò).

Vì vậy làm tiếp P2 dù P1 không đạt theo quy tắc cũ. Dự đoán ghi trước: motif / ∨ gần như không đổi PCR khi
λ = 0 (vẫn cùng vector c). Chỉ S = 4 có thể đem thêm thông tin (thứ tự thời gian).

## Fusion cho các head P1 (không train lại)

Lệnh: **ô 4** của script.


λ = 1 (chốt trước), 2026-10-19:

| | fusion Top1 / Top5 | sửa / phá vs K0 | PCR fusion | mức trần quét λ (chẩn đoán) |
|---|---|---|---|---|
| K0 | **87.63 / 97.56** | | **3.58** | |
| K0 + E1a | 86.24 / 95.84 | 745 / 1454 | 3.90 | 87.66 (λ 0.02) |
| K0 + E1b | 86.64 / 96.52 | 625 / 1129 | 3.95 | 87.65 (λ 0.02) |
| K0 + E1c | 86.57 / 94.87 | 705 / 1245 | 3.93 | 87.65 (λ 0.02) |
| K0 + E1d | 86.39 / 94.61 | 668 / 1299 | 3.91 | |
| K0 + E1e | 86.56 / 95.12 | 679 / 1224 | 3.89 | |

**Với λ = 1, fusion thấp hơn K0 ở mọi head** (thấp hơn 1.0–1.4 Top1), và phá nhiều hơn sửa.
Lý do có thể: head đạt ~100% trên train và τ khoảng 0.05–0.08, nên log-softmax của head rất nhọn và lấn át K0.
Quét λ trên test (λ ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1}, chẩn đoán):

| λ | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|
| K0 + E1a | **87.66** | 87.61 | 87.44 | 87.15 | 86.91 | 86.62 | 86.24 |
| K0 + E1b | **87.65** | 87.63 | 87.62 | 87.51 | 87.36 | 87.12 | 86.64 |
| K0 + E1c | **87.65** | 87.54 | 87.44 | 87.20 | 87.07 | 86.89 | 86.57 |

### Kết luận fusion (2026-10-19): **không đạt, kể cả ở mức trần**

- Top1 **giảm đều theo λ**. Mức trần là 87.65–87.66 ở λ nhỏ nhất (0.02), tức gần như chỉ còn K0 (+0.02–0.03,
  khoảng 15 mẫu, nằm trong nhiễu). Ngưỡng 87.93 không đạt với cả 3 head.
- Nghĩa là các head concept **không có thông tin nào dùng được mà K0 không có**. Các mẫu head "sửa" được (600–800)
  chỉ là do khác biệt ngẫu nhiên; mọi cách trộn đều phá nhiều hơn sửa.
- Nguyên nhân hợp lý: head học trên **chính map l9 đóng băng** mà fc của K0 dùng. S = 1 trung bình mất thời gian, nên
  head chỉ có thêm cấu trúc theo khớp và người; nhãn concept lại là nhãn lớp. Không có nguồn thông tin mới,
  nên không vượt được K0. Giống kết luận của A1-G: cải thiện phải mang thông tin độc lập với backbone.
- **Theo quyết định ghi trước: không chạy P2 trên S = 1 (ô 5).** Motif / ∨ chỉ gộp lại cùng vector c.
- Còn lại một thử nghiệm có nguồn thông tin mới: **S = 4** (thứ tự thời gian trong token, thứ mà fc của K0 trung bình
  mất). Chỉ chạy E1a-S4 trước. Chỉ chạy E2-S4 nếu mức trần fusion của E1a-S4 ≥ 87.93.

## P2: cấu trúc đồ thị + S = 4

Config ở `config/concept_graph/`:

| | Config | Khác với |
|---|---|---|
| E2 | `e2.yaml` | E1c: thêm 25 motif ∧ + ∨ posture (đồ thị đầy đủ), α cố định |
| E2-α | `e2_alpha.yaml` | E2: α học được (η = 1e-4) |
| E2−OR | `e2_noor.yaml` | E2: bỏ ∨ (26 motif) |
| E1a-S4 | `e1a_s4.yaml` | E1a: cache S = 4 |
| E2-S4 | `e2_s4.yaml` | E2: cache S = 4 |

Cache S = 4: khoảng 12.9 GB train + 10.4 GB test (fp16), ghi vào ổ local của Colab (không cần chép lên Drive).
Train feat được nạp hết lên GPU (khoảng 13 GB trên L4 22 GB).

Lệnh: **ô 5** (S = 1), **ô 6** (tạo cache S = 4), **ô 7** (S = 4).

| | Top1 / Top5 | PCR | fusion Top1 | fusion sửa / phá | K5 | train top1 |
|---|---|---|---|---|---|---|
| E2 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E2-α | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E2−OR | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E1a-S4 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E2-S4 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |

## P3: mô hình đầy đủ (code 2026-10-20, lệnh: ô 7b)

| | Top1 / Top5 | Top1 sym / neural / aux | PCR | fusion Top1 | mức trần fusion | K5 | train top1 |
|---|---|---|---|---|---|---|---|
| K0 | 87.63 / 97.56 | | **3.58** | | | | |
| E1a (S = 1, để so) | 85.60 / 94.98 | | 4.18 | 86.24 | 87.66 (λ 0.02) | 14.49 | 99.80 |
| E3 (S = 1) | 86.38 / 94.63 | 85.90 / 86.51 / 85.58 | 4.13 | 86.63 | 87.59 (λ 0.02) | 10.21 | 99.94 |
| E3-S4 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |

Quy tắc (chốt trước): đồ thị **có ích cho độ chính xác** nếu fusion của E2 (hoặc E2-S4) vượt K0 ≥ 0.3 Top1
**và** vượt fusion của E1a (hoặc E1a-S4, cùng S) ≥ 0.2. Không đạt thì đồ thị không thêm gì so với AND phẳng.

E3 (2026-10-20, seed 0): Top1 trong / ngoài cặp 84.74 / 88.42 (K0 85.54 / 90.23); fusion sửa / phá 647 / 1157,
PCR fusion 4.01; K5-pair 39.26%; AUC concept 0.990; τ cuối 0.0706. Cặp nhầm nhiều nhất vẫn như P1
(make_OK_sign / make_victory_sign 27.7%, put_on / take_off_a_shoe 13.0%, thumb_up / make_victory_sign 11.0%).

### Kết luận E3 trên S = 1 (2026-10-20): **không đạt**

- Fusion 86.63 < K0 87.63 (cần ≥ 87.93). Mức trần quét λ là 87.59, còn **thấp hơn K0**. Vế "hơn fusion E1a ≥ 0.2"
  có đạt (+0.39), nhưng vế chính thì không.
- Đứng một mình, E3 là head tốt nhất tới nay: hơn E1a 0.78 Top1 và hơn E1c 0.33. PCR vẫn như cũ
  (4.13 so với 4.17–4.18), vẫn cao hơn K0 (3.58). Phần tăng Top1 nằm cả trong và ngoài cặp, nên không đến từ ¬(j|i).
- Đường logic tự đứng được (λ = 0: 85.90, chỉ thấp hơn bản đầy đủ 0.48), nên rủi ro "nhánh action lấn át" không xảy ra.
  Ngược lại, R-GCN một mình (86.51) **cao hơn bản kết hợp** (86.38): cộng thêm log s / τ không giúp gì.
- Train top1 99.94%, CE train 0.0015: giống mọi head trước, head khớp hết tập train trên map l9 đóng băng.
  Kết luận fusion của P1 vẫn đúng: thêm cấu trúc đồ thị trên cùng nguồn thông tin không tạo thêm thông tin mới.
- Còn lại S = 4 (ô 6 rồi ô 7b phần S = 4). Ablation của E3 (−RGCN, randE, −r6) không chạy trên S = 1,
  vì bản đầy đủ đã không vượt K0.

## Fine-tune nhẹ: đồ thị E3 làm refinement của K0 (code 2026-10-20; bỏ, thay bằng mô hình chính online)

Người dùng chọn (2026-10-20): refinement + mở l9, có đối chứng K0-ft. Lý do: trên map l9 đóng băng, không head nào
có thông tin K0 chưa có (trần fusion < K0). Fine-tune cho gradient của concept / đồ thị chảy vào l9.

Mô hình (`model/hypergcn_concept.py`, config `config/concept_graph/ft_e3.yaml`):
```
f      = mean(actor, T, V) của l9                  (đúng pooling của K0)
out    = head E3 trên l9 lấy trung bình theo T     (S = 1 như cache; nạp từ $W/e3/head.pt)
r      = Σ_k softmax(out.logits)_k · h_k            (node action sau R-GCN, trọng số theo niềm tin của đồ thị)
f'     = f + W [LN(r) ; c]                          (W khởi tạo 0: epoch 0 = K0)
logits = fc(f')                                     (fc của K0)
L      = CE(fc(f')) + divergence + 1.0 · (CE(head) + 0.5 BCE + aux + KL + margin + η)
```
Công thức (chốt trước, như P6 của A1, chỉ khác là chỉ mở l9): từ `runs-122-120048.pt`, `freeze_until: l8`
(train l9, fc, đồ thị), SGD nesterov lr 0.0025 (lớp W ×10), giảm ×0.1 ở epoch 10, 15 epoch, batch 64,
wd 0.0004, augmentation như train gốc, không nhiễu đặc trưng. Eval ở epoch 0 / 5 / 10 / 15; **báo cáo epoch 15**
(`select_best: False`). Đối chứng **K0-ft** (`ft_k0.yaml`): cùng công thức, không có đồ thị.
Mỗi lần eval in thêm Top1 của `k0` (fc(f) không có Δ, cùng backbone), `head`, `head_sym`, và lưu
`epoch<n>_test_outputs.npz` (mọi logits + nhãn) để tính PCR / fusion sau.

Kiểm tra trước khi tin kết quả: ở epoch 0, ft_e3 phải cho Top1 = 87.63 (bằng K0, vì W = 0) và `head` ≈ 86.38
(E3 trên cache; lệch ≤ 0.1 vì cache lưu fp16). Lệch nhiều thì pooling hoặc head nạp sai: dừng.

**Quy tắc (chốt trước):** người dùng chọn (2026-10-20) chạy ft_e3 trước. Nếu ft_e3 (epoch 15) < K0 + 0.3 = 87.93
thì dừng, không chạy K0-ft. Nếu ≥ 87.93 thì chạy K0-ft: đồ thị có ích khi ft_e3 ≥ K0-ft + 0.3 (fine-tune một mình
từng cho +0.30 ở P6 của A1). Chênh trong ±0.3 thì chạy thêm seed 1 cho cả hai. Phụ: nếu Top1 của ft_e3 ≈ `k0` bên trong nó thì Δ không được dùng.

Dự đoán ghi trước: K0-ft trong khoảng 87.63 ± 0.3. ft_e3 − K0-ft trong khoảng ±0.3 (không đạt): đồ thị vẫn học từ
nhãn concept gán theo lớp, nên gradient của nó gần với gradient của CE.

| | Epoch 0 | Epoch 5 | Epoch 10 | **Epoch 15** | `k0` / `head` / `head_sym` (epoch 15) |
|---|---|---|---|---|---|
| K0-ft | *chờ* (phải ≈ 87.63) | *chờ* | *chờ* | *chờ* | |
| ft_e3 | *chờ* (phải = 87.63) | *chờ* | *chờ* | *chờ* | *chờ* |

## Mô hình chính: chạy backbone mỗi bước, không cache (2026-10-20, lệnh: ô 2)

Người dùng quyết (2026-10-20): **bỏ cache, chạy Hyper-GCN mỗi bước** làm hướng chính. ft_e3 (head.pt từ cache, mở l9)
không chạy nữa. Lý do: trên cache, head khớp hết tập train (một view augment cố định) và chỉ thấy S = 1.

Config `config/concept_graph/online_e3.yaml` (chốt trước):
- backbone **và fc** đóng băng (`freeze_until: fc`). Chỉ đồ thị (`cg.*`) train, nên K0 giữ đúng 87.63 và không cần K0-ft;
- đồ thị E3 học từ đầu (không head.pt), đọc **cả 16 bước** của l9 (`segments: 16`, 800 token / mẫu);
- refinement f' = f + W [LN(r) ; c], W = 0 lúc đầu, logits = fc(f');
- augmentation như train gốc, không nhiễu đặc trưng; Adam 1e-3, wd 0, 30 epoch, giảm ×0.1 ở epoch 20 và 25,
  batch 64; eval mỗi 5 epoch; **báo cáo epoch 30**.

**Quy tắc (chốt trước):** đạt nếu Top1 epoch 30 ≥ K0 + 0.3 = **87.93**. Chênh trong ±0.1 quanh ngưỡng thì chạy seed 1.
Phụ: `head` (đồ thị đứng một mình) so với E3 trên cache (86.38) cho biết augmentation + 16 bước có giúp head không.

Dự đoán ghi trước: `head` cao hơn E3 cache (bớt khớp tập train), nhưng Top1 refinement < 87.93.

| | Epoch 0 | 5 | 10 | 15 | 20 | 25 | **30** | `head` / `head_sym` (30) |
|---|---|---|---|---|---|---|---|---|
| online_e3 | 87.64 (đạt kiểm tra) | 87.29 | 87.31 | 87.28 | *chờ* | *chờ* | *chờ* | *chờ* |
| `head` / `head_sym` | 0.54 / 0.54 | 83.74 / 82.60 | 84.40 / 83.47 | 84.83 / 83.88 | | | | |

Top5 refine: 97.56 → 97.06 (epoch 5–15). `k0` giữ 87.64 ở mọi lần eval (backbone và fc đóng băng đúng).
Train (log): acc 99.5–99.6% suốt; `ce` của refine 0.863 → 0.845 (gần như không giảm); `delta_norm` (‖W‖) tăng
gần tuyến tính 3.5 → 28.5 sau 15 epoch.

### Đọc giữa chừng (epoch 15, 2026-10-20)

- **Refine làm K0 tệ đi**: −0.35 Top1 ở cả 3 lần eval, ổn định, không có xu hướng đi lên. Cần +0.30 nên khó đạt
  87.93 ở epoch 30. W lớn dần mà CE train không giảm: fc của K0 đã đúng 99.6% train, nên Δ chủ yếu học theo
  label smoothing (độ tự tin của logits), không học sửa lỗi; trên test, Δ chỉ làm nhiễu.
- **`head` (đồ thị một mình) chưa hơn E3 trên cache**: 84.83 ở epoch 15 so với 86.38, dù có augmentation và đủ 16
  bước. Head còn tăng (lr chưa giảm, giảm ở epoch 20 và 25), nên chờ epoch 30. Dự đoán "head cao hơn E3 cache"
  hiện đang sai.

## Train theo từng stage: stage 1 = dual decoder (2026-10-20, người dùng quyết)

Người dùng quyết: tách train theo stage, stage 1 là dual decoder train riêng. Lý do: online_e3 (train chung) cho
refine −0.35 so với K0, và head một mình thấp hơn E3 trên cache.

Lệnh (Colab, sau ô 1):
```
!python tools/run_refine.py --model-config config/concept_graph/stage1_decoders.yaml
```
Kết quả ở `/content/drive/MyDrive/BaoHuy/work_dir/concept_graph/stage1_decoders`.

Config `config/concept_graph/stage1_decoders.yaml` (`Stage1Model` trong `model/hypergcn_concept.py`):
- Hyper-GCN (backbone + fc) đóng băng, chạy mỗi bước, đủ 16 bước của l9; chỉ hai decoder train;
- loss = CE (label smoothing 0.1) của decoder action + 0.5 · BCE(c, M[y]); không có đồ thị, không refine;
- Adam 1e-3, 30 epoch, giảm ×0.1 ở epoch 20 và 25, batch 64, eval mỗi 5 epoch, báo cáo epoch 30;
- tên tham số giống head đầy đủ (`cg.head.decoder.*`, `cg.head.action_decoder.*`, `cg.head.aux.*`): checkpoint
  stage 1 dùng để khởi tạo các stage sau.

Mỗi lần eval in: `Top1` = decoder action; `k0` (phải 87.64); `concept_match` = lớp đoán chỉ từ concept c
(log-likelihood của c dưới từng hàng M, không có trọng số lớp học được): cho biết concept tốt tới đâu.

| | Epoch 5 | 10 | 15 | 20 | 25 | **30** |
|---|---|---|---|---|---|---|
| decoder action (Top1) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| `concept_match` | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |

## So sánh lỗi online_e3 với K0 (epoch 15, `tools/compare_outputs.py`, 2026-10-20)

| | Top1 | Trong 6 294 lỗi của K0: cũng sai / sai cùng lớp | chỉ nó đúng | chỉ K0 đúng | oracle với K0 |
|---|---|---|---|---|---|
| head | 84.83 | 86.0% / 61.7% | 880 (1.73%) | 2 312 (4.54%) | 89.37 |
| head_sym | 83.88 | 85.9% / 58.1% | 888 (1.74%) | 2 801 (5.50%) | 89.38 |

head và head_sym: trùng dự đoán 97.3%, khi head sai thì head_sym sai cùng lớp 90.4% (R-GCN gần như không thêm gì
so với bộ thực thi logic). **Kết luận: lỗi của đồ thị tương quan mạnh với K0 (cùng nguồn), xác nhận ECOC.** Phần bù
nhau có (oracle +1.73) nhưng không có bộ chọn nào lấy được (quét fusion < K0).

## Hướng 1: chuyên gia theo cặp dễ nhầm, dẫn hướng bởi ¬(j|i) (đặc tả chốt 2026-10-20)

Câu hỏi: map l9 (cache S = 1, giữ riêng người × khớp) có đủ thông tin để tách các cặp dễ nhầm tốt hơn K0 không, và
mask khớp từ ¬(j|i) có giúp không. Không train lại backbone.

**Bước 0 (`tools/pair_ceiling.py`, numpy, vài giây):** trần = số lỗi của K0 có nhãn ở top-2 **và** (top-1, top-2) là
một trong 58 cặp. Quy tắc: trần < 300 mẫu (0.6%) thì dừng hướng 1.
```
!python tools/pair_ceiling.py /content/drive/MyDrive/BaoHuy/work_dir/concept_graph/online_e3/epoch15_test_outputs.npz
```

**Bước 1 (chỉ khi bước 0 qua, `tools/pair_experts.py`, chưa viết):** mỗi cặp (i, j) hai chuyên gia nhỏ
(token người × khớp → Linear 512→64 → mean có mask → Linear 64→2): A chỉ khớp của concept trong M_j∖M_i ∪ M_i∖M_j,
B mọi khớp. Train trên mẫu train của i, j, nhiễu 1.0, Adam 1e-3, 30 epoch, seed 0. Cổng: test có nhãn ∈ {i, j} và
top-2 K0 = {i, j}. Thay quyết định K0 trên cổng. Đạt nếu Top1 (A) ≥ 87.93; mask có ích nếu A − B ≥ 0.1.
Hạn chế: S = 1 mất thứ tự thời gian (put_on / take_off); nếu chỉ các cặp đó hỏng thì thử lại với S = 4.

| | trần (mẫu) | Top1 nếu sửa hết | Top1 K0 + A | Top1 K0 + B |
|---|---|---|---|---|
| hướng 1 | *chờ* | *chờ* | *chờ* | *chờ* |
