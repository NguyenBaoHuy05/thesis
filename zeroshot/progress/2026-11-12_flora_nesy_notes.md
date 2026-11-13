# Ghi chú 2026-11-12: SeqKG + forbid, COA-Fuse, kết quả 3 seed trên 4 split

Tiếp theo [2026-11-10_flora_nesy_notes.md](2026-11-10_flora_nesy_notes.md). Chạy trên Colab **L4**, ZSL, giao thức của paper (1000 + 200 iteration). `best` = max qua các lần đánh giá trên unseen (cách Flora báo), `last10` = trung bình 10 lần cuối. Mọi số dưới đây là **trung bình 3 seed (1, 2, 3)** trừ khi ghi khác. So với paper Flora trực tiếp (quyết định của người dùng: không chạy lại Flora gốc cho SynSE và GZSL).

Số paper Flora (STAR): 55/5 **85.8**, 48/12 **61.5**, 110/10 **80.7**, 96/24 **64.1**. (SynSE): **86.3 / 65.3 / 79.6 / 66.4**.

## 1. Các file mới (tiền tố = file mà nó cải tiến)

| file | nội dung |
|---|---|
| `nesy4_seqkg.py` | KG v2seq: v2 (82 primitive) + 29 primitive thứ tự `seq.a>b` + 14 forbid thứ tự ngược. Số cặp lớp có luật trùng hệt nhau: 7 → 0 |
| `nesy4_1_forbid.py` | **Signed Compose**: `sem = r + λ(r − f)`, r = trung bình embedding primitive `require`, f = trung bình primitive `forbid` thứ tự, λ = 1, tổng trọng số = 1, không tham số học. `--forbid-order-only 1` (mặc định) |
| `nesy4_1_1_fuse.py` | **COA-Fuse**: thay phép cộng đều bằng 3 bước không tham số: **C**enter (trừ trung bình lớp của số hạng KG), **O**rthogonal (bỏ phần trùng với text của chính lớp, giữ nguyên độ lớn), **A**daptive (β theo lớp từ độ lệch text − KG, cắt ở 3β) |
| `nesy6_gcn.py`, `nesy7_esa.py` | R-GCN và ESA trên đồ thị KG (kết quả âm, mục 4) |

Phương pháp chính (v1): **SeqKG + COA-Fuse**. Đặt tên: C-Fuse = chỉ center, CO-Fuse = center + orth, COA-Fuse = đủ ba bước (các run `c1o0a0`, `c1o1a0`, `c1o1a1`).

## 2. Bảng chính, STAR, `best` (TB 3 seed)

| split | paper | none (chạy lại) | compose v2 | SeqKG + forbid | **SeqKG + COA-Fuse** |
|---|---:|---:|---:|---:|---:|
| 48/12 | 61.5 | 60.87 | 61.68 | 61.55 | **62.39** ± 1.27 |
| 110/10 | 80.7 | 78.42 | 79.70 | 80.14 | **81.10** ± 0.47 |
| 55/5 | 85.8 | 85.14 | 85.28 | 85.16 | 85.24 ± 0.25 |
| 96/24 | 64.1 | 67.20 | 69.13 | 68.65 | 68.16 ± 1.23 |
| **TB 4 split** | | 72.91 | 73.95 | 73.88 | **74.22** |

Δ so với paper (TB 4 split): none −0.12, compose +0.92, SeqKG + forbid +0.84, **COA-Fuse +1.20**. COA-Fuse vượt paper ở 48/12 (+0.89), 110/10 (+0.40), 96/24 (+4.06); thấp hơn ở 55/5 (−0.56).

Số từng seed (s1 / s2 / s3):

| split | compose | SeqKG + forbid | COA-Fuse |
|---|---|---|---|
| 48/12 | 62.87 / 62.17 / 60.01 | 63.96 / 60.71 / 59.98 | 63.75 / 62.20 / 61.23 |
| 110/10 | 79.17 / 79.92 / 80.02 | 79.22 / 80.30 / 80.89 | 80.63 / 81.10 / 81.57 |
| 55/5 | | 84.72 / 85.45 / 85.31 | 85.09 / 85.09 / 85.53 |
| 96/24 | 68.50 / 69.30 / 69.58 | 67.53 / 69.12 / 69.30 | 66.96 / 69.42 / 68.09 |

`last10` Δ vs compose (TB): COA-Fuse 48/12 +0.36, 110/10 +0.58, 55/5 −0.37, 96/24 −1.81; SeqKG + forbid 55/5 −0.66, 96/24 −0.83.

## 3. Kết luận từ bảng

- **Có KG (bất kỳ dạng nào) hơn Flora không KG khoảng 1 điểm** ở TB 4 split: đây là kết luận chắc nhất.
- **SeqKG + forbid ngang compose** (TB −0.08, 5/12 run cao hơn). Lợi thế của nó là biểu diễn: cosine (đã trừ trung bình) giữa các cặp đảo chiều giảm 0.99 → 0.78–0.81, không phải điểm tổng. Cặp đảo chiều chạm lớp unseen: 48/12 (15,16); 110/10 (13,14), (88,89); 55/5 (19,20); 96/24 (15,16), (17,18), (19,20), (86,87).
- **COA-Fuse vs compose:** +0.71 (48/12), +1.40 (110/10), −0.04 (55/5), −0.97 (96/24); TB +0.28, nằm trong nhiễu giữa seed. Chỉ 110/10 có ý nghĩa thống kê sơ bộ (3/3 seed dương, p ≈ 0.01).
- **Center giải thích phần lớn mức tăng ở 110/10:** C-Fuse (chỉ center) 48/12 61.86, 110/10 80.94 (Δ vs compose +0.17 / +1.24). Orth + adaptive thêm +0.54 (48/12) và +0.16 (110/10). Center là phép trừ trung bình, không phụ thuộc KG. **Chưa có đối chứng center không KG** (cần `nesy0_center.py`).
- Adaptive β dồn vào lớp tương tác và chạm (pat on back, touch neck, reach into pocket, handshaking, walking towards/apart, touch pocket), không vào cặp đảo chiều. Cosine cặp đảo chiều của ngữ nghĩa cuối gần như không đổi qua fuse (0.8902 → 0.8898; 0.9527 → 0.9535).
- Số paper cao hơn Flora chạy lại ở 3/4 split (none −0.63 / −2.28 / −0.66), nên một phần mức "hơn paper" là do lần chạy lại. 96/24 hơn paper ~+4 chủ yếu do lần chạy lại (none đã +3.10), không do KG. Split này dao động mạnh (best − last 8–12 điểm).

## 4. Kết quả âm (48/12 seed 1 nếu không ghi khác)

- **ESA (nesy7)** so với compose: sim1 −5.99, sim0 −7.84, sim0.05 + reg1 −0.82 (khi ép gần như bằng 0 thì bằng compose). R-GCN (nesy6) tương tự. Delta học trên KG học thuộc lớp seen (reasoning shortcut), cùng lý do khiến gate không giúp.
- **Text+IDF+gate** và **mix** (0.5 text + 0.5 IDF, không gate): mix 3 seed 61.37 (−0.3 vs compose); gate ≈ compose.
- **forbid mọi loại** (`--forbid-order-only 0`): không ổn định (48/12 −1.79, 110/10 +1.58 nhưng last10 −0.98). Bỏ.
- Từ ngày 10/11: neighbors (hại), logic head, hardneg, primtok, plain SeqKG: không thêm gì.

## 5. SynSE, SeqKG + COA-Fuse, 3 seed (chạy 12/11, chưa có Flora chạy lại)

| split | s1 / s2 / s3 | TB ± std | paper (SynSE) | Δ |
|---|---|---|---:|---:|
| 55/5 | 85.32 / 84.66 / 86.50 | 85.49 ± 0.93 | 86.3 | −0.81 |
| 48/12 | 64.68 / 62.17 / 63.94 | 63.60 ± 1.29 | 65.3 | −1.70 |
| 110/10 | 81.37 / 80.71 / 82.27 | 81.45 ± 0.78 | 79.6 | **+1.85** |
| 96/24 | 66.78 / 66.30 / 65.16 | 66.08 ± 0.85 | 66.4 | −0.32 |

TB 4 split so với paper SynSE: −0.25 (thấp hơn ở 3/4 split). `last10` TB: 82.87 / 61.65 / 79.00 / 60.56. So với run seed 1 có sẵn trong thư mục SynSE: vs none −0.37 / +1.22 / +1.25 / +2.50, vs compose −0.23 / +0.15 / −0.50 / +0.18 (55, 48, 110, 96).

Vượt paper ở 110/10, thấp hơn ở hai split nhỏ (độ lệch chuẩn bằng hoặc lớn hơn phần chênh ở 55/5; 48/12 lệch rõ hơn). Seed 1 của Text+IDF+gate trên SynSE (11/11): 86.06 / 64.31 / 80.24, tức COA-Fuse seed 1 tốt hơn ở 48/12, 110/10 và kém ở 55/5.

## 6. GZSL (từ `Flora_RESULTS.md`, 11/11, seed 1, STAR)

Chỉ có compose trên SeqKG và Text+IDF+gate. ΔH so với paper (compose / gate): 55/5 −0.97 / −0.78, 48/12 **+2.12** / +0.56, 110/10 +0.08 / chưa có. 96/24 chưa có. **Chưa chạy GZSL cho SeqKG + COA-Fuse.** Báo cáo H tại checkpoint tốt nhất theo H; checkpoint cuối của 110/10 chỉ H 60.50 so với 64.78.

## 7. Việc còn lại

1. `nesy0_center.py`: đối chứng center không KG (48/12 và 110/10, 3 seed, không cần baseline).
2. SynSE 96/24 (log chưa thấy); GZSL cho SeqKG + COA-Fuse (STAR, `--eval-class-chunk 16`).
3. Log accuracy và nhầm lẫn theo từng cặp đảo chiều (bằng chứng cho phần "biểu đạt thứ tự"): cần thêm vào `nesy4_1_forbid.py`, ghi log không đổi kết quả.
4. Chưa commit: file ghi chú này và `architecture_seqkg.drawio` (một tab: graph SeqKG chi tiết + pipeline Compose → Fuse → Flora). Code `nesy4_1_forbid.py` và `nesy4_1_1_fuse.py` đã commit (c01e99c, 07e7fb1). Đổi nhãn "Fuse" thành "COA-Fuse" trong sơ đồ nếu muốn.

## 8. Tài liệu đọc

- LogicZSL (Wu et al., CVPR 2025): luật logic từ LLM thành loss mờ trên điểm phân loại ảnh (CZSL); luật chỉ có dạng suy ra, không có phủ định hay thứ tự. Ý có thể mượn: prompt LLM sinh primitive "phân biệt với lớp giống" (KG v3), chưa làm.
- Andolfi và Giunchiglia, NeurIPS 2025 (Prototypical NeSy): reasoning shortcut và neo bằng prototype. Dùng để giải thích vì sao module học trên KG (GCN, ESA) overfit seen còn compose cố định thì không. Mới đọc đến hết phần 6, chưa đọc phụ lục.
