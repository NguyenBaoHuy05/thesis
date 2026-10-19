# Concept-logic: tiến độ

> Quyết định 2026-10-19: bỏ vị từ hình học (A1-G, kết luận ở
> [A1G_PROGRESS.md](../A1G_GRAPH_RETRIEVAL_PLAN/A1G_PROGRESS.md)), chuyển sang **lớp = logic trên concept**.
> Số liệu chép từ log chạy thật; ô chưa có số ghi *chờ*.

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
| Full | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| K2 (s0) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
