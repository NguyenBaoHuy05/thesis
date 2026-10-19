# Concept-logic: tiến độ

> Quyết định 2026-10-19: bỏ vị từ hình học (A1-G, kết luận ở
> [A1G_PROGRESS.md](../A1G_GRAPH_RETRIEVAL_PLAN/A1G_PROGRESS.md)), chuyển sang **lớp = logic trên concept**.
> Số liệu chép từ log chạy thật; ô chưa có số ghi *chờ*.
> Cập nhật lần cuối: 2026-10-19

## Kiến trúc (đã chốt với người dùng)

```
skeleton → Hyper-GCN (đóng băng, l9) → map [M, C, T, V] → trung bình trong S đoạn thời gian (cache)
        → ConceptDecoder: 102 query (một vector học được / concept), cross-attention chỉ vào khớp của nhóm
          concept (head / upper_limb / hip_torso / lower_limb; temporal, participant = mọi khớp)
        → c ∈ (0, 1)^102
        → logic: score_k = [Σ_j M_kj log c_j + ω Σ_j (1 − M_kj) log(1 − c_j)] / (n_pos + ω n_neg)
                 (AND mềm: has → c_j, không có → NOT c_j), logit_k = score_k / τ
```

- Nguồn concept: `annotations/concepts/ntu_action_concept_v2_4.csv` (+ metadata), chép từ
  `SkateFormer/concepts` v2.4: 120 × 102 nhị phân, mỗi lớp 6–14 concept, 120 hàng khác nhau (Hamming nhỏ nhất 1).
  Ma trận dày và thiết kế để tách lớp → ô 0 đọc là **không có** (closed world), trọng số ω.
- Không có tham số riêng theo lớp: lớp chỉ được quyết định qua concept.
- Loss: CE + β BCE(c, M[y]) (β = 0.5).
- Code: `model/concept_logic.py`, `tools/cache_joint_features.py`, `tools/shuffle_concepts.py`,
  `train_concept_logic.py`, `config/concept/cl_full.yaml`, test `tests/test_concept_logic.py` (cần torch, chạy Colab).
- Bước 1 chạy trên cache (S = 1, vài phút / run). Fine-tune qua `main.py` chỉ làm nếu bước 1 đạt.

## Rủi ro đã biết

- **ECOC / học tắt** như A1: concept chỉ có nhãn mức lớp, nên decoder có thể học bất kỳ mã nào tách được lớp.
  Map l9 đã trộn thông tin toàn thân ở mỗi khớp (9 lớp graph / hypergraph), nên mask theo nhóm khớp chỉ
  chặn được một phần. K2 là phép thử quyết định.
- K2: ma trận xáo giữ số concept mỗi lớp và số lớp mỗi concept; ~30% ô 1 vẫn ở chỗ cũ (370–384 / 1225),
  vì các concept phổ biến (`foot_static`, `leg_static_stand`...) phủ gần hết các lớp.

## Dự đoán ghi trước khi chạy (2026-10-19)

- Full: Top1 80–86% (< K0 87.63: nút thắt 102 chiều + logic cố định).
- K2: Top1 trong khoảng ±1 điểm của Full (ECOC). AUC concept so với ma trận thật: Full ≥ 0.85, K2 ≈ 0.6.
- K5: đổi dự đoán khi tắt concept hỗ trợ mạnh nhất > 30%, khi tắt một concept ngẫu nhiên lớp đó không có < 5%.
- Quy tắc kết luận (chốt trước): **KG concept có đóng góp** nếu Full − K2 ≥ 1.0 điểm Top1 **và**
  AUC concept so với ma trận thật của Full − K2 ≥ 0.10. Nếu không: concept chỉ là mã của lớp (như A1).

## Lệnh (Colab)

Ghi cache vào ổ local của Colab (memmap trên Drive chậm), rồi chép sang Drive.

```
python -m pytest tests/test_concept_logic.py -q
python tools/cache_joint_features.py --weights /content/large/ntu120/xsub/joint/runs-122-120048.pt --out-dir /content/jcache
cp /content/jcache/ntu120_xsub_joint_* /content/drive/MyDrive/BaoHuy/kg_cache/
%env J=/content/jcache
%env W=/content/drive/MyDrive/BaoHuy/work_dir/concept
python train_concept_logic.py --config config/concept/cl_full.yaml --cache-dir $J --work-dir $W/cl_full
python train_concept_logic.py --config config/concept/cl_full.yaml --cache-dir $J --work-dir $W/cl_k2_s0 --matrix annotations/concepts/shuffled/ntu_action_concept_v2_4_shuffle_s0.csv
```

## Kết quả

| | Top1 / Top5 | sửa / phá vs K0 | AUC concept (ma trận train / thật) | K5: tắt hỗ trợ / tắt ngẫu nhiên | τ |
|---|---|---|---|---|---|
| K0 | 87.63 / 97.56 | | | | |
| Full | 84.09 / **93.53** | 770 / 2570 | 0.980 / 0.980 | 6.61% / 0.00% | 0.0312 |
| K2 (s0) | **84.27** / 90.55 | 655 / 2364 | 0.967 / 0.462 | 8.69% / 0.00% | 0.0346 |
| K2 (s1) | 84.34 / 90.62 | 629 / 2302 | 0.966 / 0.482 | 9.17% / 0.00% | 0.0346 |
| K2 (s2) | 84.62 / 90.77 | 613 / 2146 | 0.965 / 0.509 | 7.34% / 0.00% | 0.0355 |
| **K2 (TB 3 lần xáo)** | **84.41 / 90.65** | | 0.966 / 0.484 | 8.4% / 0.00% | |

**Full − K2 (TB): Top1 −0.32, Top5 +2.88.** Cả 3 lần xáo đều có Top1 cao hơn Full và Top5 thấp hơn Full khoảng 3 điểm.

Giải thích: K2 xáo kiểu checkerboard **làm các mã lớp cách xa nhau hơn**. Khoảng cách Hamming tới lớp gần nhất
trung bình 8.3–8.4 với ma trận xáo, 4.7 với ma trận thật (thật: 55 cặp lớp cách nhau ≤ 4 concept, xáo: 0–1 cặp).
Theo lý thuyết ECOC, mã cách xa nhau → dễ chọn đúng hạng 1 (Top1 ↑); mã ngẫu nhiên → các lớp đứng gần nhau trong
xếp hạng không liên quan nhau (Top5 ↓). Vì vậy K2 này **lẫn hai yếu tố** (ngữ nghĩa và độ tách mã), và cả Top1 lẫn
Top5 của nó đều không đo sạch đóng góp của ngữ nghĩa.

### K2-hoán vị hàng (đối chứng sạch, 2026-10-19)

`tools/shuffle_concepts.py --mode rows`: mỗi lớp nhận hàng concept của một lớp khác (hoán vị không điểm bất động).
Tập mã và mọi khoảng cách giữa các mã giữ **y hệt** ma trận thật, chỉ đổi lớp nào mang mã nào.
File: `annotations/concepts/shuffled/ntu_action_concept_v2_4_rowperm_s{0,1,2}.csv`.

Dự đoán ghi trước khi chạy:
- Nếu ngữ nghĩa concept có ích: Full > K2-perm ở **cả** Top1 và Top5.
- Quy tắc (chốt trước): ngữ nghĩa có đóng góp nếu Full − TB(K2-perm, 3 seed) ≥ 0.5 Top1 **hoặc** ≥ 1.0 Top5,
  và cả 3 seed K2-perm đều thấp hơn Full.
- Dự đoán của mình: Top1 chênh nhỏ (≤ 0.5), Top5 Full hơn 1–3 điểm.

```
for s in 0 1 2; do
  python train_concept_logic.py --config config/concept/cl_full.yaml --cache-dir $J --work-dir $W/cl_perm_s$s \
      --matrix annotations/concepts/shuffled/ntu_action_concept_v2_4_rowperm_s$s.csv --seed $s
done
```

| | Top1 / Top5 | AUC concept (train / thật) | K5 |
|---|---|---|---|
| K2-perm s0 | *chờ* | *chờ* | *chờ* |
| K2-perm s1 | *chờ* | *chờ* | *chờ* |
| K2-perm s2 | *chờ* | *chờ* | *chờ* |

Lần 1 (2026-10-19, seed 0, S = 1): train top1 100% ở cả hai.

- **Theo quy tắc chốt trước: không đạt.** Full − K2 = −0.18 Top1 (cần ≥ +1.0). Đúng dự đoán ECOC: decoder học
  được ma trận xáo tốt ngang ma trận thật (AUC 0.967 so với chính nó).
- Tiêu chí thứ hai (AUC so với ma trận thật, Full − K2 = 0.52) đạt nhưng **không có thông tin**: K2 học ma trận
  xáo thì AUC so với ma trận thật thấp là đương nhiên. Thiết kế tiêu chí này sai; chỉ tiêu chí Top1 có giá trị.
- AUC concept 0.97–0.98 ở mọi nhóm, cả K2: concept là mã của lớp, không phải bằng chứng độc lập (giống A1).
- **Tín hiệu ngoài quy tắc (thăm dò):** Top5 Full 93.53 so với K2 90.55 (+3.0). Ma trận thật cho lớp giống nhau
  chung concept, nên các lựa chọn hạng 2–5 hợp lý hơn. Cần K2 s1, s2 để chắc chắn đây không phải do một lần xáo.
- K5 yếu (6.6%): với ω = 1, mỗi lớp có ~92 literal NOT và ~10 literal has, mỗi literal chỉ nặng 1/102, nên tắt
  một concept hiếm khi đổi quyết định.
- Top1 thấp hơn K0 3.5 điểm (fixed / broken 770 / 2570): head quá khớp train (1 view, 100%).

## v2: nâng cấp + đối chứng hoán vị hàng chạy cùng lúc (2026-10-19, chốt trước khi chạy)

Hai nâng cấp nhắm vào hai điểm yếu đo được ở lần 1 (`config/concept/cl_v2.yaml`):

| Nâng cấp | Điểm yếu ở lần 1 | Thay đổi |
|---|---|---|
| Logic cân bằng (`balance: True`) | K5 chỉ 6.6%: ~92 literal NOT nhấn chìm ~10 literal has | `score = (mean_has log c + ω mean_not log(1 − c)) / (1 + ω)`: mỗi literal has nặng ~1/10 |
| Nhiễu đặc trưng (`feature_noise: 1.0`) | Train top1 100%, test thấp hơn K0 3.5 | Nhiễu Gauss theo std từng kênh, chỉ khi train |

Mức nhiễu 1.0 chốt trước, không quét theo test. Đối chứng hoán vị hàng dùng **đúng cấu hình v2**.

Dự đoán:
- Full v2: Top1 ≥ Full v1 (84.09); K5 tắt concept hỗ trợ > 20% (v1: 6.6%).
- Quy tắc ngữ nghĩa (như trên): Full v2 − TB(K2-perm v2, 3 seed) ≥ 0.5 Top1 hoặc ≥ 1.0 Top5, cả 3 seed thấp hơn Full.

```
python train_concept_logic.py --config config/concept/cl_v2.yaml --cache-dir $J --work-dir $W/cl_v2
for s in 0 1 2; do
  python train_concept_logic.py --config config/concept/cl_v2.yaml --cache-dir $J --work-dir $W/cl_v2_perm_s$s \
      --matrix annotations/concepts/shuffled/ntu_action_concept_v2_4_rowperm_s$s.csv --seed $s
done
```

| | Top1 / Top5 | sửa / phá vs K0 | AUC concept (train / thật) | K5: tắt hỗ trợ / ngẫu nhiên | train top1 cuối |
|---|---|---|---|---|---|
| Full v2 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| K2-perm v2 s0 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| K2-perm v2 s1 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| K2-perm v2 s2 | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
