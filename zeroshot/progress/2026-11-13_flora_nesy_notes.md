# Ghi chú 2026-11-13: đối chứng KG xáo trộn, KG-only, chẩn đoán lỗi, ngữ cảnh primitive

Tiếp theo [2026-11-12_flora_nesy_notes.md](2026-11-12_flora_nesy_notes.md). Chạy trên Colab **L4**, ZSL, STAR, giao thức paper (1000 + 200 iteration), `best` = max qua các lần đánh giá trên unseen, `last10` = trung bình 10 lần cuối. Mọi số là trung bình 3 seed (1, 2, 3) trừ khi ghi "seed 1". Flora chỉ lấy số paper (không chạy lại).

Bảng tham chiếu (STAR, `best`, TB 3 seed; đã có từ 12/11):

| | 48/12 | 110/10 |
|---|---:|---:|
| Flora không KG (none) | 60.87 | 78.42 |
| compose v2 | 61.68 | 79.70 |
| SeqKG cộng thô (nesy4_1) | 61.55 | 80.14 |
| C-Fuse (chỉ Center) | 61.86 | 80.94 |
| COA-Fuse (v1) | 62.39 ± 1.27 | 81.10 ± 0.47 |

## 1. Đối chứng quan trọng nhất: KG xáo trộn (`nesy0_shuffle.py`)

**Câu hỏi:** phần tăng điểm của KG đến từ nội dung KG, hay chỉ từ việc cộng thêm một vector (Center, β, độ lớn)?
**Cách đo:** hoán vị các hàng KG giữa các lớp (lớp c nhận luật của lớp π(c), không có điểm cố định; π phụ thuộc seed chạy nên TB 3 seed gồm cả may rủi hoán vị). Giữ nguyên mọi thứ khác: cùng số primitive, cùng phân bố trọng số, cùng Center/Orth/Adaptive/β. Hoán vị áp dụng cho trọng số compose, trọng số forbid thứ tự và độ giống KG của bước Adaptive. Tương quan Pearson giữa độ giống text CLIP và độ giống KG: thật 0.30, xáo −0.00 đến 0.12.

| Cấu hình | 48/12 xáo (s1 / s2 / s3) | 48/12 thật | thật − xáo | 110/10 xáo (s1 / s2 / s3) | 110/10 thật | thật − xáo |
|---|---|---:|---:|---|---:|---:|
| cộng thô (c0 o0 a0) | **57.95** (56.94 / 59.19 / 57.73) | 61.55 | +3.60 | **77.79** (77.50 / 77.10 / 78.77) | 80.14 | +2.35 |
| chỉ Center (c1 o0 a0) | **58.28** (57.55 / 60.16 / 57.13) | 61.86 | +3.58 | **77.77** (77.41 / 77.36 / 78.54) | 80.94 | +3.17 |
| COA-Fuse (c1 o1 a1) | **58.48** (56.88 / 59.37 / 59.19) | 62.39 | +3.91 | **77.96** (77.81 / 76.89 / 79.17) | 81.10 | +3.14 |
| Flora không KG | 60.87 | | | 78.42 | | |

Δ xáo so với none: cộng thô −2.92 / −0.63; Center −2.59 / −0.65; COA-Fuse −2.39 / −0.46 (48/12, 110/10). `last10` xáo: 55.44 / 55.87 / 55.99 (48/12), 74.32 / 74.28 / 74.55 (110/10).

**Kết luận**
1. **Nội dung KG là yếu tố quyết định.** Chênh thật − xáo ≈ +2.4 đến +3.9, khoảng 3 lần độ lệch chuẩn giữa các seed (≈ 1.2–1.4). KG sai làm hại: tụt dưới Flora không KG, tức mô hình thực sự đọc KG.
2. **Nghi vấn Center đã đóng.** Trước đó ta nghi Center là phép dịch không phụ thuộc KG (Center một mình giải thích +1.24 ở 110/10 so với compose). Với KG xáo, Center-only tụt −3.2 so với Center-only KG thật, nên Center chỉ có ích khi nội dung KG đúng.
3. **Orth + Adaptive không giảm được thiệt hại của KG sai** (xáo: COA 58.48 / 77.96 so với Center 58.28 / 77.77). Với KG thật chúng cộng thêm +0.53 (48/12) và +0.16 (110/10) so với Center-only, và Center cộng +0.31 / +0.80 so với cộng thô: mỗi lớp trong gói fuse nằm trong nhiễu. Gói đủ ba bước so với cộng thô: +0.84 / +0.96 (cùng dấu ở hai split, nhưng không giữ ở 96/24 theo kết quả 12/11).
4. **Cách phát biểu cho luận văn:** KG mang thông tin thật về lớp (có đối chứng) và đóng góp khoảng +0.7 đến +1.7 so với không KG khi cộng thô; COA-Fuse đẩy thêm ≈ +1, mức chưa có ý nghĩa thống kê ở từng split riêng lẻ. Nên báo cáo chuỗi cộng dồn cộng thô → Center → COA-Fuse như một ablation trung thực.
5. **Lưu ý:** KG xáo cũng làm sai bước Adaptive (β chọn nhầm lớp), nên mức tụt gồm cả hai tác động. Nên nói "KG sai làm hại hiệu năng", không nói "KG thật cộng đúng +3.9".

Chưa chạy: KG xáo trên 55/5 và 96/24; đối chứng nhiễu Gauss cùng độ lớn (không cần thiết cho luận điểm).

## 2. KG cung cấp ngữ nghĩa thế nào (v1), và KG-only

Chuỗi trong code (`kg_semantics.py`, `nesy4_1_forbid.py`, `nesy4_1_1_fuse.py`):
1. Mỗi primitive (82 gốc + 29 thứ tự) có một câu mô tả; CLIP đóng băng nhúng từng câu riêng, ra (T, 768), cache.
2. Mỗi lớp là một hàng trọng số trên các primitive: Signed Compose `w = (1+λ)·r − λ·f` (r = trung bình đều trên `require`, f = trung bình đều trên forbid thứ tự `seq.*`, λ = 1, tổng = 1); lớp không có forbid thứ tự thì `w = r`.
3. `k_c = Σ_p w[c,p]·emb_p`, lặp 2 lần lên (T, 1536) cho khớp bề rộng của Flora.
4. Hòa vào ngữ nghĩa Flora bằng COA-Fuse: `sem_c = A_c + β_c · k_fused_c`, β = 0.3 (Center → Orthogonal theo `A_c` → Adaptive theo lớp, trần 3β).
5. KG còn cung cấp độ giống lớp (cosine của các hàng W), chỉ dùng ở bước Adaptive. Phía skeleton không nhận gì từ KG.

Ví dụ wear jacket (4 primitive `require`, 1 primitive thứ tự ngược bị forbid): trọng số 2/4 = 0.5 trên mỗi `require` và −1.0 trên primitive thứ tự ngược, tổng 4·0.5 − 1.0 = 1.
**KG-only** (`nesy4_1_1_2_kgonly.py`, 48/12 seed 1, Center + khớp độ lớn, không láng giềng): **41.08** so với Flora paper 61.5 (chance ≈ 8%). KG một mình mang được thông tin phân biệt lớp nhưng thấp hơn text nhiều; hai nguồn bổ sung cho nhau. Biến thể `--kgonly-neighbors 1` và KG thô chưa chạy. Log cosine của file này không hữu ích (cosine giữa các lớp đã Center luôn ≈ −1/(n−1)).

## 3. Chẩn đoán lỗi của DiT trên lớp unseen (`nesy4_1_1_3_diag.py`, seed 1)

Ý tưởng kiểm chứng: vòng lặp chấm lại, lớp nào DiT phân vân thì quay về KG để tăng độ rõ. Tiêu chí đặt trước: (a) top2 − top1 ≥ ~10 điểm, (b) tỉ lệ sai nhóm margin nhỏ nhất / lớn nhất ≥ 2, (c) ≥ 50% mẫu sai có KG nói "khác nhau hơn text" (d = z(text) − z(KG) > 0).

| | 48/12 | 110/10 |
|---|---:|---:|
| top2 − top1 (best) | 14.16 | 13.46 |
| tỉ lệ sai nhóm margin nhỏ nhất / lớn nhất | 2.34 | 23.7 |
| mẫu sai có d > 0 | 46.4% | 34.5% |

(a) và (b) đạt, (c) không đạt ở cả hai split. **Vòng lặp chấm lại theo margin bị bỏ.** Lỗi nằm ở phía skeleton: ví dụ walking towards → falling 59%, punching → wield knife 53%.

**Bổ sung 55/5 (3 seed) và 96/24 (seed 1):**
- 55/5: gần như mọi lỗi là reading ↔ writing (reading → writing 42–53%, writing → reading 15–26% ở `best`); 90–94% mẫu sai có d > 0 (zText 2.58, zKG 1.50).
- 96/24 seed 1 (`best` 66.96): 62–63% mẫu sai có d > 0. Nhầm nhiều nhất: rock-paper-scissors → punching 34% (zKG 2.54 > zText 0.40, tức KG kéo hai lớp lại gần, KG có hại ở đây), typing → writing 72%, blow nose → hand waving 32%, take off a hat/cap → take off glasses 51%.
- Tiêu chí (c) đạt ở 55/5 và 96/24, không đạt ở 48/12 và 110/10. 55/5 chỉ là một cặp nên chưa đủ để làm lại vòng lặp chấm lại. 96/24 seed 2, 3 chưa chạy (đã dừng).

**Khẳng định về thứ tự không kiểm tra được bằng độ chính xác ZSL.** Ở cả 4 split chuẩn, mỗi cặp đảo chiều chỉ có một lớp unseen, lớp đối xứng luôn là seen:

| split | lớp unseen (độ chính xác ở `best`, seed 1) | lớp đối xứng (seen) |
|---|---|---|
| 48/12 | wear a shoe (97.4%) | take off a shoe |
| 110/10 | wear jacket (89.1%), put something into a bag (87.1%) | take off jacket, take something out of a bag |
| 55/5 | put on a hat/cap (95.6%; seed 2, 3: 95.2%, 94.5%) | take off a hat/cap |
| 96/24 | take off a shoe (90.9%), glasses (83.9%), hat/cap (21.2%), bag (57.1%) | wear / put on tương ứng |

ZSL chỉ đoán trong các lớp unseen nên không bao giờ nhầm trong cặp. (Trước đó mình đã nói nhầm là 55/5 và 96/24 có cặp mà cả hai lớp đều unseen; đối chứng λ = 0 vì vậy đã huỷ.) Luận văn chỉ khẳng định được ở mức biểu diễn: cặp lớp có luật trùng 7 → 0, cosine cặp đảo chiều 0.99 → 0.78–0.81. Muốn đo bằng độ chính xác cần split mới có cả hai lớp unseen, mà đặc trưng STAR được tính theo từng split nên phải trích lại đặc trưng.
Chú ý: ma trận MSE của file chẩn đoán trùng điểm của run gốc (63.75 / 80.63), không lệch.

## 4. Ngữ cảnh KHÔNG chứa tên lớp quanh câu primitive (`nesy4_1_1_4_ctx.py`, seed 1, trên COA-Fuse)

Mẫu cố định, giống nhau cho mọi primitive, không có tên lớp: `action` = "In this action, a hand is behind the trunk." và `person` = "A person is performing an action in which …".

| seed 1, `best` (last10) | 48/12 | 110/10 | Δ vs COA-Fuse |
|---|---:|---:|---|
| COA-Fuse (câu trần) | 63.75 | 80.63 | |
| ctx `action` | 64.33 (61.75) | 79.69 (76.12) | +0.58 / −0.94 |
| ctx `person` | 63.78 (61.36) | 78.91 (75.41) | +0.03 / −1.72 |

Tiêu chí đặt trước (≥ +1 ở cả hai split, 3 seed) không đạt; không chạy thêm seed. Lý do (từ log): cosine giữa câu bọc và câu trần là 0.86 (`action`) và 0.74 (`person`) nhưng cấu trúc giữa các lớp gần như không đổi (cosine giữa các lớp sau Center −0.0123 → −0.0108 ở 48/12; −0.0051 → −0.0042 ở 110/10), cosine cặp đảo chiều trên ngữ nghĩa cuối cũng không đổi (0.954; 0.872). Ngữ cảnh chủ yếu dịch mọi primitive theo một hướng chung, mà Center lấy đi. **Đóng nhánh "cải tiến câu primitive" như ablation âm.**
Lưu ý log `person`: 12/111 câu bị cắt ở 48/12 nhưng 0/111 ở 110/10, và cosine trần so với bọc lệch (0.739 so với 0.752); chưa giải thích được (có thể lỗi đếm hoặc hai cache khác nhau). Vì vậy chỉ dựa vào số của `action`.
Lý do không đưa tên lớp vào câu: `A_c` của Flora đã chứa tên lớp, nên `k_c` bị Orthogonal triệt tiêu hoặc chỉ phóng to `A_c`; cặp đảo thứ tự bị kéo giống nhau (đối ngược với lợi thế của forbid thứ tự); primitive mất tính dùng chung giữa seen và unseen. Chưa thử, chỉ lập luận.

## 4b. Chấm lại ứng viên bằng truy hồi + luật KG (`nesy4_1_1_5_retrieval.py`) — kết quả âm

Cơ chế: huấn luyện như COA-Fuse; khi dự đoán, lấy 50 mẫu huấn luyện seen gần nhất (cosine trên đặc trưng 256 chiều), hồ sơ primitive = trung bình có trọng số hàng `require` của lớp các mẫu đó; điểm luật của lớp unseen = TB hồ sơ trên primitive `require` − TB trên primitive `forbid` (chỉ primitive có ở lớp seen); chấm lại top-3 của DiT bằng z(−MSE) + γ·z(luật), γ = 1 cố định.

| `best`, TB 3 seed | 48/12 (s1 / s2 / s3) | 110/10 (s1 / s2 / s3) |
|---|---|---|
| COA-Fuse (không chấm lại) | 62.39 | 81.10 |
| + chấm lại, luật thật | **46.20** (46.40 / 46.49 / 45.70) | **56.44** (56.77 / 57.40 / 55.14) |
| + chấm lại, luật xáo | 42.04 (46.55 / 35.19 / 44.39) | 52.77 (51.85 / 54.08 / 52.37) |

- 48/12 seed 1: điểm DiT gốc trong cùng run đạt 63.75 (trùng COA-Fuse, kiểm tra đúng). Sau chấm lại 44–46% ở mọi lần đánh giá; số mẫu bị làm sai (500–750) gấp 3–5 lần số mẫu được sửa (110–300). Điểm chỉ dùng luật 31.21% (KG-only phía ngữ nghĩa: 41%). Trần top-3 85–89%.
- Lưới γ × top tại lần đánh giá tốt nhất (cơ sở 57.22): γ 0.25 top-2 54.60, top-3 51.50; γ 1 top-3 46.40; γ 2 top-5 37.28. Không ô nào hơn cơ sở → không phải chọn sai γ.
- Độ phủ primitive (primitive `require` của lớp unseen có ở lớp seen): 48/12 0.58, 110/10 0.90, 55/5 0.88, 96/24 0.85 (shake head 0.00). 110/10 phủ tốt mà vẫn tụt −25, nên độ phủ không phải nguyên nhân chính.
- Nguyên nhân: hồ sơ là bản sao luật của lớp seen gần nhất, không phải primitive của mẫu (walking apart → `body.staggering 1.00` → falling; pushing → hồ sơ của một lớp tương tác seen → walking towards). Khi đúng thì rất rõ (brushing hair: `hand.vertical_oscillation 1.00, both_hands.above_head 0.92`; wear a shoe: `hand.at_foot, body.sitting`), nhưng ít hơn nhiều.
- Luật thật hơn luật xáo ~+4 điểm: có tín hiệu thật nhưng yếu hơn DiT rất nhiều. **Không tiếp tục hướng này.**

**Tổng kết phía skeleton:** mọi cách đưa KG vào phía skeleton hoặc bước quyết định đều thất bại (logic head, primitive token, chấm lại theo margin, truy hồi + luật). Chỉ phía ngữ nghĩa có tác dụng (đối chứng KG xáo xác nhận). Diễn giải: đặc trưng skeleton đã gộp thành vector 256 chiều không giữ thông tin khớp/thời gian để tách primitive khỏi nhãn lớp. `nesy0_probe.py` (đã viết, chưa chạy) đo trực tiếp điều này bằng AUC dự đoán primitive trên mẫu unseen; tiêu chí đặt trước: ≥ 0.75 là đặc trưng có primitive, ≤ 0.65 là không.

## 4c. Bảng chính dùng seed 1 (quyết định của người dùng, 13/11)

`main.tex`: hai dòng SeqKG + COA-Fuse đổi sang seed 1 (STAR 85.09 / 63.75 / 80.63 / 66.96; SynSE 85.32 / 64.68 / 81.37 / 66.78), tô lại best/second. Lý do chọn seed 1: cố định từ trước, khớp GZSL seed 1. Trung bình 4 split theo seed — STAR 74.11 / 74.45 / 74.11, SynSE 74.54 / 73.46 / 74.47 (seed 1 / 2 / 3): thứ hạng seed đảo giữa hai loại đặc trưng, chênh lệch là nhiễu. Bảng ablation/KG xáo và bảng β vẫn dùng TB 3 seed.

## 5. Các biến thể ghép text (nhắc lại)

`--kg-compose-src text` (gộp primitive `require` thành một câu mỗi lớp): 48/12 seed 1 là 62.53, thấp hơn compose 62.87 (seed 2, 3 và 110/10 chưa chạy). `mix` (0.5 text + 0.5 IDF, không cổng): 61.37 (3 seed, 48/12) so với compose 61.68. Text+IDF+gate ≈ compose sau 3 seed. Không biến thể nào thắng compose.

## 5b. Độ nhạy β (C-Fuse: Center, không Orth/Adaptive; `nesy4_1_1_fuse.py --fuse-orth 0 --fuse-adaptive 0 --kg-beta <β>`)

| β | 48/12 (s1 / s2 / s3) | 110/10 (s1 / s2 / s3) | TB hai split | 48/12 − none | 110/10 − none |
|---|---|---|---:|---:|---:|
| 0.1 | 61.86 (63.69 / 61.01 / 60.89) | 79.57 (78.94 / 79.31 / 80.47) | 70.72 | +0.99 | +1.16 |
| **0.3** (mặc định, đã có) | 61.86 | 80.94 | **71.40** | +0.99 | +2.52 |
| 0.5 | 60.87 (61.96 / 59.80 / 60.86) | **81.36** (81.55 / 81.41 / 81.12) | 71.12 | 0.00 | **+2.94** |
| 1.0 | 59.27 (58.71 / 57.95 / 61.14) | 79.29 (79.81 / 79.48 / 78.58) | 69.28 | −1.61 | +0.87 |
| none (Flora không KG) | 60.87 | 78.42 | 69.65 | | |

`last10` TB 3 seed (48/12 / 110/10): β 0.1 59.03 / 76.40; 0.5 59.19 / 77.45; 1.0 57.53 / 76.30.

**Đọc kết quả**
- Đường cong hình chữ U ngược ở cả hai split. Vùng tốt là β ∈ [0.3, 0.5]; β = 0.3 (giá trị đặt trước khi quét) cho trung bình hai split cao nhất (71.40), nên không phải giá trị chọn lại theo kết quả.
- 110/10 đỉnh ở β = 0.5 (+2.94 so với none, cả 3 seed ≥ +1.36), 48/12 đỉnh ở β ≤ 0.3 và tụt ở β lớn: β = 1.0 làm hại seed 1 và 2 (−3.40, −3.16 so với none). Cùng chiều với đối chứng KG xáo: KG càng được cộng mạnh, lợi (khi đúng) và hại (khi sai hoặc quá tay) đều lớn hơn.
- Phương pháp nhạy với β ở mức ±1 điểm trong khoảng 0.1 đến 0.5, và hại ở β = 1.0 trên 48/12. Phải nêu điều này trong luận văn, không nên nói "bền với β".
- Lưu ý khi viết: ZSL không có tập validation nên không thể chọn β theo dữ liệu unseen. Báo cáo bảng này như phân tích độ nhạy với β = 0.3 cố định từ trước, không báo cáo "β tốt nhất" như kết quả chính.
- Chưa quét β cho COA-Fuse đầy đủ (β là β gốc, β_c = β·d_c/mean(d), trần 3β).

## 6. Phương án chốt v1 (đề xuất, chờ người dùng)

| Mức | Cấu hình | 48/12 | 110/10 | Ghi chú |
|---|---|---:|---:|---|
| 1 | SeqKG cộng thô (nesy4_1) | 61.55 | 80.14 | gần nhất với "KG cung cấp ngữ nghĩa", điểm thấp nhất |
| 2 | SeqKG + Center | 61.86 | 80.94 | đơn giản, có đối chứng; C-Fuse chưa chạy 3 seed ở 55/5, 96/24 |
| 3 | SeqKG + COA-Fuse | 62.39 | 81.10 | điểm cao nhất, đang ghi trong `main.tex` |

Đề xuất: giữ COA-Fuse làm phương pháp chính và trình bày ba mức như ablation cộng dồn, kèm đối chứng xáo trộn làm bằng chứng KG có tác dụng; nói rõ Orth và Adaptive nằm trong nhiễu.

## 7. Việc còn lại

1. Độ nhạy β: xong cho C-Fuse (mục 5b). Chưa quét cho COA-Fuse đầy đủ.
2. Đối chứng xáo trộn trên 55/5 và 96/24 (để bảng đối chứng đủ 4 split) nếu đưa vào bài.
3. Biến thể `text` còn seed 2, 3 và 110/10; KG-only còn các biến thể.
4. GZSL cho v1 (STAR, `--eval-class-chunk 16`), nhiều bước t khi suy luận (`nesy4_1_1_1_multit.py`, chưa viết).
5. Cập nhật sơ đồ `architecture_seqkg.drawio` (đổi nhãn "Fuse" → "COA-Fuse"), thêm tham khảo LogicZSL (CVPR 2025) và Prototypical NeSy (NeurIPS 2025) vào đề cương.

## 8. File mới của ngày

| file | nội dung | commit |
|---|---|---|
| `nesy4_1_1_2_kgonly.py` | bỏ ngữ nghĩa Flora, chỉ KG | 4cad1db |
| `nesy4_1_1_3_diag.py` | chẩn đoán lỗi DiT (top-k, margin, nhầm lẫn, KG so với text) | 075efae |
| `nesy4_1_1_4_ctx.py` | câu primitive có ngữ cảnh không chứa tên lớp | 44d07e1 |
| `nesy0_shuffle.py` | đối chứng KG xáo trộn | a6fc52b |
| `nesy4_1_1_5_retrieval.py` | truy hồi mẫu seen → hồ sơ primitive → chấm lại top-k bằng luật (âm) | e1abc5e |
| `nesy0_probe.py` | bộ dò primitive trên đặc trưng 256 chiều (AUC trên unseen) | chưa commit |
