# Ghi chú 2026-11-10: kết quả NeSy-KG trên Flora và bốn hướng cải tiến

Tiếp theo [2026-11-9_flora_nesy_notes.md](2026-11-9_flora_nesy_notes.md). Commit trong ngày: `0a742d1` (sửa import Shift), `16c47f2` (bốn hướng `nesy*.py`). **Chưa commit**: `--feat synse` trong `nesy_common.py` và bản notebook rút gọn.

Mọi run chạy trên Colab **L4**, feature **STAR** (1s-Shift-GCN), ZSL, trừ khi ghi khác. `best` = max qua 20 lần đánh giá trên unseen test (cách Flora báo), `last10` = trung bình 10 lần đánh giá cuối.

## 1. Sửa lỗi

- `ModuleNotFoundError: No module named 'cuda.shift'`: `shiftgcn.py` dùng `sys.path` + `from cuda.shift import Shift`, đụng gói `cuda` của `cuda-python` trên Colab. Sửa: nạp `Shift` theo đường dẫn file bằng `importlib`; `shift.py` chịu được việc thiếu op `shift_cuda` (không dùng khi `use_features=True`).
- Cache embedding primitive trong `trainer.load_prototypes` trước chỉ đặt tên theo `max_length`, sẽ nạp nhầm khi đổi KG. Nay tên gồm tên file KG.

## 2. Đối chiếu với bài báo Flora (arXiv 2511.09388)

- Siêu tham số chung khớp phụ lục C: AdamW lr 1e-4 wd 0.01, batch 64, 1000 + 200 iteration, λ = 0.1, DiT 1 lớp, CLIP ViT-L/14@336, γ = 0.75.
- Bài báo **không ghi** `topk`, `mixed_weights`, `max_length` theo split. Repo chỉ có config STAR cho 55/5; với 48/12, 110/10, 96/24 notebook dùng config SynSE trên feature STAR. Ở 55/5 config STAR và SynSE giống hệt, nên nhiều khả năng tác giả dùng chung theo split (suy luận, không kiểm chứng được).
- Seed: mọi config trong repo `seed: 1`; bài báo không ghi seed, nhiều khả năng mỗi số là một run, lấy `best`.
- Bảng 1 có hai dòng Flora: STAR **85.8 / 61.5 / 80.7 / 64.1**, SynSE (†) **86.3 / 65.3 / 79.6 / 66.4**. Ablation của Flora dùng SynSE.
- Hai điểm lạ ở bảng 1: (a) dòng không có † không bao giờ được tô màu, kể cả khi cao nhất (80.7 ở 110/10), tức tác giả chỉ xếp hạng trong nhóm SynSE mà chú thích không nói; (b) 110/10 là split duy nhất STAR > SynSE.
- Tính tất định: cùng seed, cùng GPU cho ra đúng cùng số (`[lam0]` của nesy1 trùng `compose` seed 1 đến 0.01). `cudnn.benchmark=True`, `deterministic=False`, nên đổi GPU (L4/T4/4090) có thể lệch. **Baseline và phương pháp phải chạy trên cùng loại GPU.**

## 3. Ablation `neighbors` / `compose` (STAR)

48/12, 3 seed (mean ± std):

| kg | best | last |
|---|---|---|
| none | 60.87 ± 1.37 | 58.39 ± 1.20 |
| neighbors | 57.96 ± 0.97 | 54.67 ± 2.07 |
| compose | **61.68 ± 1.49** | **59.40 ± 1.28** |
| both | 56.98 ± 0.88 | 55.28 ± 0.91 |

- `compose` tăng ở cả 3 seed (Δ best +0.76, +1.06, +0.61), nhưng nhỏ so với dao động giữa seed.
- `neighbors` hại, kéo `both` xuống: 4 cặp mặc/cởi có hàng KG giống hệt nên thành láng giềng mạnh. **Bỏ `neighbors`.**

55/5, 3 seed: none best 85.14 / last 83.87; compose 85.28 / 83.50. Gần trần, không khác biệt.

110/10: none best 77.64 / 77.85 / 79.76 (seed 1-3, mean 78.42, thấp hơn bài báo 2.3); compose 79.17 / 79.92 (seed 1-2, Δ +1.53 / +2.07). `last` của none dao động mạnh (70.5–77.6).

96/24, seed 1: none best 67.07 (iter 1200: 54.75), compose 68.50 (58.71). Chênh 12 điểm giữa `best` và lần cuối trong một run: `best` một seed ở split này gần như vô nghĩa, phải xem `last10` và nhiều seed. Việc none (67.07) cao hơn bài báo (64.1) là do nhiễu, không phải lỗi.

## 4. Bốn hướng cải tiến (`zeroshot/Flora/nesy*.py`)

Mỗi hướng một file, dùng chung `nesy_common.py` (sinh config theo Flora, mỗi run một subprocess, bỏ qua run đã xong, tự chạy baseline thiếu, in và lưu `summary.csv`). Notebook chỉ cần `!python {FLORA}/nesyX.py`. Log: `OUT_DIR/<nhãn>/<task>_ZSL_seed<k>/run.log`.

| File | Hướng |
|---|---|
| `nesy1_logic.py` | MLP dự đoán 82 primitive từ feature 256-d (nhãn từ KG trên lớp seen), điểm logic mềm `mean log p(require) + w·mean log(1−p(forbid))`, quyết định `z(−MSE) + λ·z(logic)` |
| `nesy2_hardneg.py` | Phase 2: thay một phần negative ngẫu nhiên bằng lớp seen gần theo KG (softmax sim_KG / τ) |
| `nesy3_primtok.py` | Mỗi primitive `require` là một token (CLIP pooled) nối vào cuối chuỗi ngữ nghĩa của lớp; DiT không có positional embedding nên không sửa kiến trúc |
| `nesy4_seqkg.py` | KG v2seq dựng từ trường `sequence` có sẵn của v2: 29 primitive thứ tự `seq.a>b`, thứ tự ngược vào `forbid`. Số cặp lớp trùng luật 7 → 0 |

Kết quả seed 1, `best` (baseline seed 1: none 62.11 / 77.64, compose 62.87 / 79.17):

| Cấu hình | 48/12 | Δ compose | 110/10 | Δ compose |
|---|---|---|---|---|
| 1 logic (λ = 0.5, nền compose) | 54.54 | −8.33 | 75.08 | −4.09 |
| 2 hardneg (nền compose) | 63.20 | +0.33 | 79.03 | −0.14 |
| 3 primtok (nền none) | 64.81 | +1.94 | 78.54 | −0.63 |
| 3 primtok + KG v2seq (nền none) | 64.87 | +2.00 | 78.51 | −0.66 |
| 4 seqkg (compose với KG v2seq) | 62.75 | −0.12 | 78.98 | −0.19 |
| **3 primtok + compose** | 63.51 | **+0.64** | **79.81** | **+0.64** |

- **Hướng 1 thất bại, có lý do rõ.** Độ chính xác dự đoán primitive 93–95% trên lớp seen nhưng 67% trên lớp unseen (recall `require` 55% / 39%), BCE về 0.0001: đầu primitive học thuộc ánh xạ lớp seen → primitive, vì feature 256-d mã hoá danh tính lớp. `[logic-only]` 30.7% / 46.5%. Khớp với nhận định cũ là không ground được primitive trên feature đã gộp của Flora. Ghi vào luận văn như kết quả âm.
- Hướng 2 và 4 ngang `compose`. Hướng 4 chỉ đụng 1 lớp unseen ở 48/12 và 2 ở 110/10.
- Ban đầu chọn primtok + compose (bị loại sau khi có đủ 4 split, xem bên dưới): duy nhất cao hơn cả none và compose ở cả 48/12 và 110/10; `last10` cao nhất (62.05 / 76.48); ở 48/12 khoảng cách best − last10 chỉ 1.5 (primtok trên none: 2.9).
- primtok + compose trên 55/5 seed 1: 84.94 (none 85.45, compose 85.38); trên 96/24 seed 1: best 68.05, last10 61.14 (none 67.07, compose 68.50).
- **Đánh giá lại sau đủ 4 split (seed 1, `best`):**

  | | 55/5 | 48/12 | 110/10 | 96/24 | trung bình |
  |---|---|---|---|---|---|
  | primtok + compose − none | −0.51 | +1.40 | +2.17 | +0.98 | +1.01 |
  | primtok + compose − compose | −0.44 | +0.64 | +0.64 | −0.45 | +0.10 |

  Phần tăng so với Flora chủ yếu đến từ `compose`; primtok cộng thêm trên compose gần như bằng 0 (hai split dương, hai split âm, cỡ 0.5). Chưa loại được vì mới 1 seed, nhưng chưa thể gọi là cải tiến.
- **Theo `last10` (seed 1), primtok bị loại:**

  | Δ `last10` | 55/5 | 48/12 | 110/10 | 96/24 | trung bình |
  |---|---|---|---|---|---|
  | primtok + compose − compose | −0.68 | +1.02 | −0.21 | −3.07 | −0.74 |
  | compose − none | +0.04 | +1.22 | +2.36 | +2.88 | +1.63 |

  **Bạn chọn đo theo `best`** (như bài báo). Theo `best`, Δ so với none trung bình 4 split: compose +0.91, primtok + compose +1.01, tức primtok chỉ thêm ~0.1, chưa phân định được với 1 seed. Cần 3 seed cho cả hai (`--kg-mode compose --tasks 48 55 110 96 --seeds 1 2 3`).

  Tạm thời: **cấu hình chính là `compose`** (đơn giản hơn, kết quả tương đương). compose − none theo `best` (seed 1): −0.07 / +0.76 / +1.53 / +1.43. Kết luận cho luận văn: KG đưa vào phía ngữ nghĩa (compose) giúp Flora ổn định; các cách dùng KG khác (láng giềng, logic ở bước quyết định, hard negative, token riêng, KG có thứ tự) không cộng thêm được.

## 5. Quyết định về cách báo cáo

- **Chỉ số chính là `best`** (quyết định của bạn, như bài báo). `last10` vẫn được in để tham khảo.
- Hai bảng: (1) so SOTA: chép số đã công bố của mọi phương pháp kể cả Flora, dòng của mình là trung bình N seed, ghi chú rõ; (2) so sánh có kiểm soát: Flora tự chạy lại, + compose, + primtok + compose, cùng seed, cùng GPU, mean ± std và Δ cặp theo seed. Không chỉ so trung bình của mình với số Flora trong bài báo (bài báo một seed `best`; độ lệch tái lập −2.3 ở 110/10 và +3 ở 96/24 sẽ tạo ảo giác thua/thắng).
- Tối thiểu 3 seed, 5 seed cho 96/24.

## 6. Đánh giá tính mới

- Đủ cho khoá luận: tái lập Flora, thiết kế KG, bốn hướng (hai hướng âm có phân tích), một cấu hình tốt nhất nhiều seed.
- Chưa đủ cho hội nghị top hay Q1: làm giàu ngữ nghĩa bằng mô tả chuyển động đã phổ biến (PURLS, STAR, SA-DVAE, InfoCPL, DVTA); mức tăng nhỏ; phần symbolic (logic) thất bại; KG viết tay chỉ cho NTU. Hợp lý: hội nghị khu vực/trong nước hoặc workshop; muốn lên tầm trung cần thêm SynSE, GZSL, phân tích theo lớp, so với mô tả LLM không có KG.
- Việc nâng vị thế đáng làm nhất: chạy trên feature **SynSE** để so với Flora† (65.3 ở 48/12, 79.6 ở 110/10).

## 7. Notebook `flora_nesy.ipynb` hiện tại

Đã xoá các cell Bước 1/2, cell sàng lọc biến thể `compose` (pre / text / llm / β; code vẫn còn trong `trainer.py`, `kg_semantics.py`, `options.py` nhưng **chưa chạy**), cell tổng hợp cũ, và bốn cell `nesy1..4`. Còn: clone, cài đặt, tải feature (nay symlink cả `star_features` và `synse_features`), kiểm tra KG, hai cell primtok (v2seq, compose), và cell SynSE:

```
!python {FLORA}/nesy3_primtok.py --kg-mode compose --feat synse --tasks 48 110 --seeds 1
```

`--feat synse` (trong `nesy_common.py`, chưa commit) dùng `configs/synse/*` và file feature trong config, ghi run và baseline vào `OUT_DIR/synse/`, cột `paper` lấy số Flora†. Chưa kiểm tra archive có thật sự chứa `synse_features` không.

Sau mỗi lần restart runtime phải chạy lại cell tải feature (symlink nằm trong `/content`). Mỗi run: 55/5 ≈ 5.3 phút, 48/12 ≈ 5.7, 110/10 ≈ 10, 96/24 ≈ 21.5 trên L4.

## 8. Việc tiếp theo

1. Xác nhận `compose` so với `none` đủ 3 seed trên 4 split, không chạy phương pháp nào khác (cờ `--baselines-only` mới thêm trong `nesy_common.py`, in bảng compose − none theo seed, mean ± std, lưu `OUT_DIR/baselines_compose_vs_none.csv`):
   `!python {FLORA}/nesy3_primtok.py --baselines-only --tasks 48 55 110 96 --seeds 1 2 3`
   Còn thiếu: 110/10 compose seed 3 (có thể đã có), 96/24 none + compose seed 2-3 (≈ 4 × 21 phút). Cân nhắc seed 4-5 cho 96/24.
2. Commit + push `nesy_common.py` (cột `d_last10_vs_*`, `--feat synse`, `--baselines-only`) và notebook.
3. SynSE: `--feat synse --baselines-only` để xem compose có giúp Flora† không (so với 65.3 / 79.6).
4. Cập nhật `paper_notes.md` (dòng 4 vẫn ghi bỏ nhánh KG / NeSy) và đề cương theo baseline Flora.
