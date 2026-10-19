# A1-G: tiến độ triển khai

> Kế hoạch: [A1G_GRAPH_RETRIEVAL_PLAN.md](A1G_GRAPH_RETRIEVAL_PLAN.md) (pha G0–G6 ở §8).
> Số liệu A1 dùng làm mốc: [A1_PROGRESS.md](../A1_RETRIEVE_VERIFY_FUSE_PLAN/A1_PROGRESS.md).
> Số liệu chép từ log chạy thật; ô chưa có số ghi *chờ*, không ước lượng.

Cập nhật lần cuối: **2026-10-19**

## Tổng quan

| Pha | Việc | Trạng thái | Còn thiếu |
|---|---|---|---|
| G0 | Code + test | **Đã viết**. Test numpy (vị từ) 10/10 pass trên máy local; test torch chưa chạy (máy local không nạp được torch) | `python -m pytest tests -q` trên Colab |
| G1 | Retrieval đồ thị (G1, G1-0, G1-K2) | G1, G1-0 seed 0 xong: G1 ≈ G1-0 | G1-K2; thêm seed nếu cần |
| G2 | Full-z, K2-z (chẩn đoán) | Chưa chạy | |
| G3 | Cặp khó, mở rộng KG, kiểm tra vị từ | Vị từ đã cache; lọc theo cặp KG xong; `hard_pairs.py --predicates-from` | Mở rộng KG / vị từ cho cặp khó (xem Bàn giao) |
| G4 | Full-pred, K2-pred | Lần 1–4 xong (seed 0): grounding rõ (Full > K2), **Top1 không tăng** | Hướng sửa: xem Bàn giao |
| G5 | K5 can thiệp | Chưa bắt đầu | |
| G6 | Fine-tune (tùy chọn) | Chỉ khi G4 đạt | |

## Quyết định đã chốt

- **Node khởi tạo học từ đầu** (`init: learned`); khởi tạo từ text chỉ là ablation G1-init.
- **Cặp khó chọn từ quan hệ `similar` / `opposite` của KG**, không theo ma trận nhầm trên test; top-25 cặp
  nhầm của K0 chỉ để kiểm tra độ bao phủ.
- **Vị từ: một view tất định cho mỗi mẫu** (transform của test), cả train và test.
- Không có validation: công thức chốt trong config trước khi chạy; test đánh giá một lần.
- **1 seed (seed 0) khi phát triển**, sai thì sửa ngay; thêm seed chỉ khi hai cấu hình chênh < ~0.2 điểm
  (std seed ở A1 ≤ 0.08) hoặc khi chốt số cho luận văn.

## G0: code

Danh sách file: plan §5. Kiểm tra đã làm trên máy local (Python 3.14, numpy, không có torch):

| Kiểm tra | Kết quả |
|---|---|
| `python -m unittest tests.test_predicates` | 10/10 pass (đúng dấu trên khung xương tổng hợp, bất biến với xoay ±0.3 rad, đếm lặp, vị từ hai người) |
| pyflakes trên mọi file mới / sửa | sạch |
| Parse 4 config `config/kg/g*.yaml` | đạt |
| Test torch (`test_graph_encoder.py`, `test_kg_retrieval_graph.py`, test cũ) | *chờ* (Colab) |

Lệnh (Colab):

```
python -m pytest tests -q
```

## G3 (phần đã có): cặp khó từ KG

```
python tools/hard_pairs.py                       # annotations/kg_work/hard_pairs.json
python tools/hard_pairs.py --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache   # + kiểm tra độ bao phủ K0
```

Kết quả trên `annotations/action_kg.json` (2026-10-18, chưa có `--cache-dir`):

| Mục | Giá trị |
|---|---|
| Cặp `similar` / `opposite` | 280 |
| Không có thuộc tính đối lập nào | **46** |
| Không có thuộc tính đối lập nào có vị từ | 85 |
| Thuộc tính KG có vị từ | 37 / 61 |

Cặp `opposite`:

| Cặp | Đối lập | Có vị từ |
|---|---|---|
| put_on_jacket / take_off_jacket | không có | |
| put_on_a_shoe / take_off_a_shoe | không có | |
| put_on_bag / take_off_bag | không có | |
| put_on_a_hat_cap / take_off_a_hat_cap | `single_arm_active` (đáng ngờ: cần duyệt) | có |
| put_object_into_bag / take_object_out_of_bag | `downward_press`, `hand_retracts_to_body` | `hand_retracts_to_body` |
| sit_down / stand_up | `hips_lower`, `hips_rise` | cả hai |
| put_on_glasses / take_off_glasses | `hands_come_together`, `hands_move_apart` | cả hai |
| walking_towards / walking_apart | `distance_decreases`, `distance_increases` | cả hai |
| put_on_headphone / take_off_headphone | `hands_come_together`, `hands_move_apart` | cả hai |

Việc tiếp của G3: chạy `--cache-dir` để biết top-25 cặp nhầm có bao nhiêu nằm trong 280 cặp; thêm thuộc tính
hướng thời gian (`hands_rise_over_time` / `hands_lower_over_time` đã có vị từ) vào KG cho các cặp put_on /
take_off và duyệt tay; tạo lại KG xáo trộn.

## G1: retrieval đồ thị

Dự đoán ghi **trước khi chạy** (2026-10-18):

- G1-0 (node tự do, không cạnh) gần một cosine classifier: Top1 giữa K1 (86.57) và K0 (87.63); Recall@10 ≥ 98.5.
- G1 ≈ G1-0 (chênh ≤ 0.2): có giám sát đầy đủ và thành phần tự do `x_c`, cấu trúc đồ thị khó thêm gì cho
  retrieval. Nếu G1 > G1-0 rõ thì đồ thị có ích ngay ở bước truy xuất.
- G1-K2 ≈ G1 vì cùng lý do.
- `cos(g_i, g_j)` trên các cặp `opposite` cao hơn trung bình mọi cặp ở G1, không ở G1-0.

```
for s in 0 1 2; do
  python train_kg_head.py --config config/kg/g1.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --work-dir work_dir/kg/g1_s$s --seed $s
  python train_kg_head.py --config config/kg/g1_0.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --work-dir work_dir/kg/g1_0_s$s --seed $s
  python train_kg_head.py --config config/kg/g1.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --kg annotations/kg_shuffled/action_kg_shuffle_s$s.json --work-dir work_dir/kg/g1_k2_s$s --seed $s
done
```

| Cấu hình | Top1 | Top5 | Recall@10 | τ cuối |
|---|---|---|---|---|
| K0 | 87.63 | 97.56 | 98.72 | |
| K1 (text) | 86.57 ± 0.02 | 96.81 | 98.32 | 0.030 |
| G1 (s0) | 86.91 | 96.53 | 97.80 | 0.0475 |
| G1-0 (s0) | 86.85 | 96.63 | 97.90 | 0.0496 |
| G1-K2 | *chờ* | *chờ* | *chờ* | *chờ* |

## G2: VERIFY với `a` từ `z` (chẩn đoán)

Dự đoán ghi **trước khi chạy**: Full-z ≤ K2-z (ECOC như A1); `verify_mode: opposed` phá ít mẫu hơn `std`
(số `broken` so với retrieval nhỏ hơn); độ phủ `v ≠ 0` dưới 60%.

```
for s in 0 1 2; do
  python train_kg_head.py --config config/kg/g_full_z.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --work-dir work_dir/kg/g_full_z_s$s --seed $s
  python train_kg_head.py --config config/kg/g_full_z.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --kg annotations/kg_shuffled/action_kg_shuffle_s$s.json --work-dir work_dir/kg/g_k2_z_s$s --seed $s
done
python train_kg_head.py --config config/kg/g_full_z.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
    --head-arg verify_mode=std --work-dir work_dir/kg/g_full_z_vstd_s0 --seed 0
```

| Cấu hình | Top1 | Retrieval trong head | Độ phủ `v ≠ 0` | sửa / phá so với retrieval | sửa / phá so với K0 | λ |
|---|---|---|---|---|---|---|
| Full-z | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| K2-z | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| Full-z, V-std | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |

## G3/G4: vị từ

```
python tools/cache_predicates.py --out-dir /content/drive/MyDrive/BaoHuy/kg_cache
for s in 0 1 2; do
  python train_kg_head.py --config config/kg/g_full_pred.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --work-dir work_dir/kg/g_full_pred_s$s --seed $s
  python train_kg_head.py --config config/kg/g_full_pred.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --kg annotations/kg_shuffled/action_kg_shuffle_s$s.json --work-dir work_dir/kg/g_k2_pred_s$s --seed $s
done
```

Kiểm tra trước khi tin kết quả G4: bảng AUC của `cache_predicates.py` (vị từ `check sign` phải sửa định nghĩa
trước; vị từ `weak` loại khỏi `predicates.names`). Chọn danh sách vị từ **chỉ theo AUC trên train**.

### Lần 1 (2026-10-18, seed 0, 37 vị từ, không lọc, không nhiễu)

| | Full-pred | K2-pred (s0) |
|---|---|---|
| Top1 head / retrieval | 86.89 / 86.86 | 86.81 / 86.81 |
| Recall@10 retrieval | 97.74 | 97.68 |
| AUC thuộc tính (test) | **0.699** | 0.515 |
| λ cuối, τ cuối | 0.943, 0.0463 | 0.836, 0.0465 |
| spread `s` / `λv` | 4.02 / 0.16 | 4.01 / 0.11 |
| đổi dự đoán khi bỏ `λv` | 0.47% | 0.42% |
| sửa / phá so với retrieval; so với K0 | 81 / 67; 532 / 911 | 66 / 65; 516 / 932 |

- **Vị từ đo đúng KG thật**: AUC 0.699 so với 0.515 với KG xáo trộn (≈ ngẫu nhiên). Đây là bằng chứng
  grounding mà A1 không có (AUC của A1 suy từ `z`).
- **VERIFY quá yếu để đổi quyết định**: spread `λv` nhỏ hơn `s` ~25 lần; λ giảm dần khi train vì retrieval đã
  đúng 99.7% trên train (rủi ro #6). Head − retrieval +0.03 (Full) và 0.00 (K2): chưa kết luận được Q3.
- Retrieval đồ thị Recall@10 97.74 < K1 98.32 < K0 98.72: cần G1 / G1-0 để biết R-GCN có làm hại không.
- Độ phủ `v ≠ 0` = 100% (trong 10 ứng viên gần như luôn có một thuộc tính đối lập): chỉ số này không có
  thông tin, đọc `đổi dự đoán khi bỏ λv` thay.

Sửa sau lần 1: `predicates.min_train_auc` (lọc vị từ theo AUC **trên train**, mặc định 0.75);
`--predicates-from` (K2 dùng đúng tập vị từ Full đã chọn); `--feature-noise` (nhiễu trên `z` khi train để
retrieval có lỗi như ở test và λ học được cách dùng `v`).

### Lần 2 (2026-10-19, seed 0, lọc `min_train_auc 0.75` → 11 vị từ, K2 dùng cùng tập)

Bỏ 26 vị từ (AUC train < 0.75). Giữ: hand_on_head_top 0.75, hand_on_own_chest 0.77, hands_down_to_feet 0.79,
elbow_raised_high 0.78, torso_bent_forward 0.86, hips_lower 0.87, hips_rise 0.92, walks_across_floor 0.93,
legs_carry_motion 0.99, head_carries_motion 0.95, distance_decreases 0.83. Cùng 11 vị từ với KG xáo trộn s0:
AUC train 0.20–0.67 (hand_on_own_chest 0.33, head_carries_motion 0.20 ngược dấu).

| | Full-pred | K2-pred | Full-pred, nhiễu 0.5 | K2-pred, nhiễu 0.5 |
|---|---|---|---|---|
| Top1 head / retrieval | 86.87 / 86.84 | 86.83 / 86.81 | 86.89 / 86.89 | 86.90 / 86.96 |
| Recall@10 retrieval | 97.75 | 97.69 | 97.98 | 97.93 |
| AUC thuộc tính (test) | **0.847** | 0.532 | **0.847** | 0.532 |
| train top1 cuối (trên `z` có nhiễu) | 99.77 | 99.74 | 99.64 | 99.62 |
| λ cuối | 0.883 | 0.811 | 0.886 | 0.798 |
| spread `s` / `λv` | 4.03 / 0.16 | 4.02 / 0.14 | 4.23 / 0.16 | 4.23 / 0.14 |
| đổi dự đoán khi bỏ `λv` | 0.28% | 0.46% | 0.26% | 0.41% |
| độ phủ `v ≠ 0` | 74.76% | 80.77% | 74.35% | 80.66% |
| sửa / phá so với retrieval | 48 / 36 (+12) | 82 / 75 (+7) | 38 / 38 (0) | 49 / 81 (−32) |
| sửa / phá so với K0 | 524 / 913 | 524 / 932 | 502 / 877 | 488 / 860 |

- **G1 vs G1-0**: Recall@10 97.80 vs 97.90; Top1 86.91 vs 86.85. Như đã dự đoán: R-GCN không giúp gì, và cũng
  không phải lý do Recall@10 tụt. Bản thân head retrieval (học trên `z` train gần như tách rời hoàn toàn) đã
  thấp hơn K0 (Recall@10 −0.8, Top1 −0.7).
- **Lọc vị từ**: AUC test tăng từ 0.699 lên 0.847 với Full, K2 vẫn ≈ ngẫu nhiên (0.532). Phần grounding vững.
- **VERIFY vẫn không đổi được quyết định**: λ vẫn giảm (không tăng), spread `λv` nhỏ hơn `s` khoảng 25 lần.
  Full và K2 chênh nhau ≤ 0.07 điểm Top1.
- **Nhiễu 0.5 không đủ**: train top1 chỉ giảm từ 99.77 xuống 99.64, retrieval trên train vẫn gần như không có
  lỗi nên λ không có tín hiệu để tăng. Chênh sửa/phá giữa Full (0) và K2 (−32) có đúng hướng nhưng chỉ là
  32/50919 mẫu, 1 seed: chưa kết luận.

### Lần 3: chọn mức nhiễu (quy tắc chốt trước khi chạy, 2026-10-19)

Quét `--feature-noise` ∈ {1, 2, 3} trên Full-pred (seed 0). **Chỉ dùng train top1 của epoch cuối** (tính trên `z`
có nhiễu) để chọn: lấy mức nhỏ nhất có train top1 ≤ 92% (gần mức lỗi Top1 ~87% ở test). Nếu cả ba mức đều
> 92% thì quét tiếp 4, 6. Sau khi chọn, chạy K2-pred với đúng mức đó và cùng tập 11 vị từ. Không đổi mức nhiễu
theo kết quả test.

Kết quả quét (Full-pred, seed 0):

| nhiễu | train top1 cuối | Top1 head / retrieval | Recall@10 | λ cuối | τ cuối | spread `s` / `λv` | đổi khi bỏ `λv` | sửa / phá vs retrieval |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 99.64 | 86.89 / 86.89 | 97.98 | 0.886 | 0.0452 | 4.23 / 0.16 | 0.26% | 38 / 38 |
| 1 | 99.44 | 87.08 / 87.09 | 98.33 | 0.874 | 0.0426 | 4.67 / 0.15 | 0.28% | 43 / 47 |
| 2 | 98.53 | 87.34 / 87.35 | 98.80 | 0.881 | 0.0355 | 5.86 / 0.15 | 0.25% | 35 / 39 |
| 3 | 95.33 | 87.26 / 87.28 | 98.85 | **1.073** | 0.0295 | 7.09 / 0.18 | 0.24% | 35 / 43 |

- Chưa mức nào ≤ 92% → theo quy tắc, quét tiếp 4 và 6.
- Nhiễu 3 là lần đầu λ tăng (> 1). Nhưng τ giảm theo nhiễu nên spread `s` tăng nhanh hơn: tỉ lệ `s`/`λv` từ
  ~26 lên ~39. Head bù nhiễu bằng cách làm `s` sắc hơn, nên `v` không có thêm chỗ để chen vào.
- Phát hiện phụ: nhiễu là regularizer tốt cho retrieval. Recall@10 lên 98.80–98.85 (> K0 98.72, đạt mục tiêu
  ≥ 98), Top1 lên 87.3 (K0 87.63).

Quét tiếp:

| nhiễu | train top1 cuối | Top1 head / retrieval | Recall@10 | λ cuối | τ cuối | spread `s` / `λv` | đổi khi bỏ `λv` | sửa / phá vs retrieval |
|---|---|---|---|---|---|---|---|---|
| 4 | **85.80** | 87.10 / 87.10 | 98.81 | 1.423 | 0.0293 | 6.88 / 0.25 | 0.34% | 55 / 57 |
| 6 | 56.41 | 86.69 / 86.65 | 98.58 | 1.853 | 0.0379 | 4.53 / 0.35 | 0.71% | 121 / 103 |

- **Chọn N = 4** (mức nhỏ nhất có train top1 ≤ 92%). Bước tiếp: K2-pred với nhiễu 4.
- Dự đoán "tỉ lệ `s`/`λv` vẫn 30–40" **sai**: τ ngừng giảm từ nhiễu 3, λ tăng mạnh (1.42 ở 4, 1.85 ở 6), tỉ lệ
  giảm còn ~27 (nhiễu 4) và ~13 (nhiễu 6). Trên train có nhiễu, VERIFY được dùng thật.
- Nhưng ở test, `z` sạch nên `s` vẫn sắc; số dự đoán bị đổi chỉ 0.34% (nhiễu 4) và sửa/phá ≈ hòa. Lỗi ở test
  là lỗi tự tin (margin lớn), khác loại lỗi margin nhỏ do nhiễu tạo ra.

K2-pred, nhiễu 4 (cùng 11 vị từ):

| | Full-pred n4 | K2-pred n4 |
|---|---|---|
| train top1 cuối | 85.80 | 85.52 |
| Top1 head / retrieval | 87.10 / 87.10 | 87.07 / 87.05 |
| Recall@10 | 98.81 | 98.80 |
| λ cuối | **1.423** (tăng) | **0.587** (giảm) |
| spread `s` / `λv` | 6.88 / 0.25 | 6.89 / 0.10 |
| đổi khi bỏ `λv` | 0.34% | 0.25% |
| sửa / phá vs retrieval | 55 / 57 | 49 / 39 |

- **Trên train, KG thật được dùng, KG xáo trộn bị bỏ**: λ 1.42 so với 0.59. Đây là kết quả K2 rõ đầu tiên của VERIFY.
- **Ở test thì không chênh**: Top1 head bằng nhau (87.10 / 87.07), sửa/phá đều ≈ hòa.
- Thêm chẩn đoán `verify_headroom` (train_kg_head.py): trên các nhầm lẫn thật ở test, `v` của lớp đúng có lớn
  hơn `v` của lớp đối thủ không, và so `|λΔv|` với khoảng cách `s`. Chạy lại Full n4 + K2 n4 để đọc.

`verify_headroom` (nhiễu 4, test; đối thủ = top-1 retrieval nếu sai, hạng 2 nếu đúng):

| | Full: errors (n 5961) | Full: correct (n 44350) | K2: errors (n 5987) | K2: correct (n 44323) |
|---|---|---|---|---|
| `v` đúng / sai / hòa | 17.3 / 16.7 / **66.0** | 29.1 / 17.2 / 53.7 | 27.9 / 30.0 / 42.1 | 26.3 / 28.0 / 45.8 |
| trung vị khoảng cách `s` | 4.57 | 15.55 | 4.56 | 15.57 |
| trung vị `|λΔv|` | 0.33 | 0.39 | 0.14 | 0.14 |

- **Trên các lỗi thật ở test, `v` của Full không có thông tin**: đúng 17.3% ≈ sai 16.7%, và 66% là hòa (bằng
  chứng không phân biệt được lớp đúng với lớp đối thủ). Phóng to `λ` hay đổi cách kết hợp sẽ phá ngang bằng sửa.
- `v` của Full có thông tin trên các cặp dễ (đúng 29.1 so với sai 17.2; K2 ≈ 50/50), nhưng ở đó khoảng cách `s`
  đã ~15 nên không cần. Điều này giải thích vì sao λ tăng trên train mà test không đổi.
- Kết luận: nút thắt là **độ phủ của KG + vị từ trên các cặp khó** (G3), không phải λ hay cách kết hợp.
  Bộ lọc AUC toàn cục 0.75 còn bỏ đúng các vị từ dùng cho cặp khó (`hands_come_together` / `hands_move_apart`,
  `hand_retracts_to_body`, `hand_at_mouth`...).
- Thêm: `tie_no_opposed` (hòa vì KG không có thuộc tính đối lập có vị từ cho cặp đó) và
  `hard_pairs.py --predicates-from`.

`hard_pairs.py` (K0 top-25 cặp nhầm ở test, chỉ để kiểm tra độ phủ):

| Tập vị từ | KG pairs không có thuộc tính đối lập có vị từ | top-25 có thuộc tính đối lập có vị từ |
|---|---|---|
| 37 (tất cả) | 85 / 280 | 14 / 25 |
| 11 (lọc AUC toàn cục 0.75) | 239 / 280 | **0 / 25** |

→ Bộ lọc toàn cục xóa toàn bộ bằng chứng cho cặp khó: đúng với 66% hòa trên lỗi ở `verify_headroom`.

### Lần 4: lọc vị từ theo cặp KG (quy tắc chốt trước khi chạy, 2026-10-19)

`--predicate-select pair`: giữ một vị từ nếu, **trên train**, nó đạt AUC ≥ 0.75 bên trong ít nhất một cặp KG
(`similar`/`opposite`) mà nó là thuộc tính đối lập (chỉ mẫu của hai lớp đó, target = lớp có thuộc tính). Cặp lấy
từ KG, không từ nhầm lẫn ở test. Nhiễu 4 (đã chọn ở lần 3). K2 dùng đúng tập vị từ của Full. Thêm một run Full
giữ đủ 37 vị từ (`--min-train-auc none`) để so.

### Lần 4: kết quả (2026-10-19, seed 0, nhiễu 4)

Lọc theo cặp KG: giữ 29 vị từ (bỏ 8), phân biệt được 68 / 280 cặp KG (AUC theo cặp ≥ 0.75).
`hard_pairs.py`: top-25 cặp nhầm của K0 có thuộc tính đối lập gắn vị từ: 12 / 25 (trước là 0 / 25).

| | Full-pair (29) | K2-pair (29) | Full-all (37) |
|---|---|---|---|
| Top1 head / retrieval | 87.06 / 87.08 | 87.06 / 87.06 | 87.05 / 87.06 |
| λ cuối | 1.685 | 0.701 | 1.775 |
| AUC thuộc tính (test) | 0.695 | 0.523 | 0.699 |
| đổi khi bỏ `λv` | 0.81% | 0.29% | 0.75% |
| sửa / phá vs retrieval | 124 / 130 | 52 / 51 | 117 / 124 |
| lỗi: `v` đúng / sai / hòa | 50.9 / 47.9 / 1.3 | 51.9 / 47.9 / 0.3 | 50.7 / 49.1 / 0.2 |
| đúng: `v` đúng / sai / hòa | 57.3 / 41.0 / 1.7 | 51.2 / 48.7 / 0.1 | 57.9 / 41.7 / 0.4 |

- **Độ phủ đã giải quyết** (hòa trên lỗi 66% → 1.3%), nhưng trên lỗi `v` của Full ≈ ngẫu nhiên và ≈ K2.
- `v` chỉ có thông tin trên cặp dễ (57 / 41 so với K2 51 / 49).
- Nguyên nhân: vị từ được giữ vì phân biệt *một vài* cặp, nhưng VERIFY dùng nó trên *mọi* cặp có thuộc tính
  đối lập, kể cả cặp nó không phân biệt được trên train.
- Tiếp: đo trần của VERIFY có cổng theo cặp (`gated_pair_check`): chỉ dùng vị từ phân biệt đúng cặp
  (lớp đúng, đối thủ) trên train (AUC ≥ 0.75, mọi cặp lớp, không chỉ cặp KG).

`gated_pair_check` (cổng theo cặp, AUC theo cặp trên train ≥ 0.75, mọi cặp lớp):

| | Full-pair | K2-pair |
|---|---|---|
| cặp lớp có ≥ 1 vị từ đáng tin | 1914 | 1133 |
| lỗi: phủ / đúng / sai | 12.0% / 49.4 / 49.0 | 5.8% / 52.2 / 47.0 |
| đúng: phủ / đúng / sai | 22.8% / 67.1 / 32.1 | 8.3% / 61.2 / 37.1 |

- **Kể cả vị từ tách đúng cặp đó trên train (AUC ≥ 0.75), trên các mẫu retrieval sai ở test nó vẫn ngẫu nhiên**
  (~720 mẫu, 49.4 / 49.0). Trên mẫu retrieval đúng thì tốt (67 / 32).
- Giải thích: lỗi tương quan. Mẫu mà backbone nhầm là mẫu có chuyển động giống lớp đối thủ, nên số đo hình học
  của nó cũng giống lớp đối thủ. Vị từ là hàm của cùng skeleton mà backbone đã nhìn, nên không thêm thông tin
  độc lập đúng ở chỗ backbone sai.
- Hệ quả: VERIFY hậu kiểm bằng vị từ skeleton không thể nâng Top1 trên K0 với bộ vị từ hiện tại. Nó vẫn có giá trị
  grounding / giải thích (AUC 0.70 vs 0.52, λ 1.69 vs 0.70).
- K2 với cổng cũng đạt 61 / 37 trên mẫu đúng: cổng chọn chiều theo dữ liệu train, nên đo khả năng tổng quát của vị từ
  hơn là tính đúng của KG; khi dùng cổng phải so K2 cẩn thận.

## THEN: thuộc tính thứ tự thời gian (2026-10-19, code xong, chờ chạy)

Mục tiêu: bằng chứng **độc lập** với backbone ở đúng chỗ nó sai (hướng B của Bàn giao). Sơ đồ: drawio trang
"A1-G Logic (AND / NOT)", panel THEN.

| Thành phần | File |
|---|---|
| 5 sự kiện theo frame (`hand_at_head`, `above_head`, `behind`, `chest`, `low`) + `THEN(A, B) = max_t min(max_{t'≤t} A, max_{t''>t} B)` trên chuỗi làm mượt, min-max theo mẫu; 6 vị từ `then_*` (tổng 45 vị từ) | `model/predicates.py` (`EVENTS`, `THEN`), test `tests/test_predicates.py` (12/12 pass local) |
| KG + 6 thuộc tính THEN, 22 cạnh (w = 2) đọc từ trường `phases` của KG; `lacks` chỉ cho cặp đảo chiều mà phases nói rõ thứ tự ngược | `tools/add_then_attributes.py` → `annotations/action_kg_then.json` (validate: 0 lỗi) |
| Đối chứng: K2 (`annotations/kg_then_shuffled/`, 3 seed, validate 0 lỗi) và **KG đảo THEN** (has ↔ lacks của cạnh THEN, phần còn lại giữ nguyên) | `annotations/action_kg_then_reversed.json` |
| Kiểm tra trần trước khi train (numpy, chỉ cần cache) | `tools/then_check.py` |
| Chẩn đoán mới: số lỗi rơi vào cặp `opposite` của KG (K0 / retrieval / head) | `train_kg_head.py` (`opposite_pair_errors`) |

Cặp được phủ: put_on / take_off a_hat_cap, headphone, jacket, bag; put_object_into_bag / take_object_out_of_bag.
put_on / take_off a_shoe: phases không cho thứ tự phân biệt → không có cạnh.

Dự đoán ghi **trước khi chạy**:

- `then_check`: lỗi K0 trong 5 cặp nhỏ (trần Top1 ≤ +0.3); binary accuracy K0 trong cặp ≥ 95%.
- Quy tắc đi tiếp (chốt trước): vote THEN đúng **≥ 60%** trên lỗi K0 trong cặp (và hơn vote vị từ tĩnh) → chạy
  các mô hình. Dưới 60% → dừng, ghi kết luận "lỗi tương quan cả ở chiều thời gian".
- Nếu đi tiếp: Top1 Full-THEN − Full-pair ≤ +0.1 (chỉ 10 lớp); `opposite_pair_errors` của head giảm so với
  Full-pair; KG đảo THEN có nhiều lỗi trong cặp hơn Full-THEN; K2 ≈ Full-pair.

Lệnh (Colab):

```
%env C=/content/drive/MyDrive/BaoHuy/kg_cache
%env W=/content/drive/MyDrive/BaoHuy/work_dir/kg
python -m pytest tests -q
python tools/cache_predicates.py --out-dir $C --name ntu120_xsub_pred_then --kg annotations/action_kg_then.json
python tools/then_check.py --cache-dir $C
# chỉ chạy tiếp nếu qua quy tắc 60%
python train_kg_head.py --config config/kg/g_full_pred.yaml --cache-dir $C --work-dir $W/g_full_pair_n4_v2 --predicate-select pair --feature-noise 4
python train_kg_head.py --config config/kg/g_then_pred.yaml --cache-dir $C --work-dir $W/g_then_n4
python train_kg_head.py --config config/kg/g_then_pred.yaml --cache-dir $C --work-dir $W/g_then_k2_n4  --kg annotations/kg_then_shuffled/action_kg_shuffle_s0.json --predicates-from $W/g_then_n4/kg_metrics.json
python train_kg_head.py --config config/kg/g_then_pred.yaml --cache-dir $C --work-dir $W/g_then_rev_n4 --kg annotations/action_kg_then_reversed.json --predicates-from $W/g_then_n4/kg_metrics.json
```

`then_check`:

| Cặp | n test | lỗi K0 trong cặp | binary K0 | vote THEN: test / trên lỗi K0 | vote tĩnh: test / trên lỗi K0 |
|---|---|---|---|---|---|
| (5 cặp) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |

| | Top1 head / retrieval | λ | lỗi trong cặp opposite (head / retrieval / K0) | sửa / phá vs retrieval |
|---|---|---|---|---|
| Full-pair (chạy lại, baseline) | *chờ* | *chờ* | *chờ* | *chờ* |
| Full-THEN | *chờ* | *chờ* | *chờ* | *chờ* |
| K2-THEN | *chờ* | *chờ* | *chờ* | *chờ* |
| THEN đảo | *chờ* | *chờ* | *chờ* | *chờ* |

## Bàn giao (2026-10-19, cuối session)

### Quyết định

- **Giữ kiến trúc A1-G cũ: RETRIEVE (graph) → Contrastive VERIFY bằng vị từ hình học.** Trong session đã cân
  nhắc bỏ vị từ và chuyển sang "v2: KG làm cấu trúc bộ phân loại" (ghép vector lớp từ hàng KG, kiểu ML-GCN +
  ALE). Đánh giá tính mới: v2 thấp–trung bình (gần ML-GCN, CVPR 2019), A1-G cũ cao hơn → **bỏ v2, quay lại A1-G**.
  Trang drawio v2 đã xóa.
- Không zero-shot. Đánh giá: Top1 so với K0 (87.63), đối chứng K2 (KG xáo trộn), K5 (can thiệp `p ← 1 − p`).
- Một seed (0) khi phát triển.

### Cấu hình hiện tại (lệnh chuẩn)

```
%env C=/content/drive/MyDrive/BaoHuy/kg_cache
%env W=/content/drive/MyDrive/BaoHuy/work_dir/kg
# Full: lọc vị từ theo cặp KG (29 vị từ), nhiễu 4
python train_kg_head.py --config config/kg/g_full_pred.yaml --cache-dir $C --work-dir $W/g_full_pair_n4 --predicate-select pair --feature-noise 4
# K2: cùng tập vị từ
python train_kg_head.py --config config/kg/g_full_pred.yaml --cache-dir $C --work-dir $W/g_k2_pair_n4     --kg annotations/kg_shuffled/action_kg_shuffle_s0.json --predicates-from $W/g_full_pair_n4/kg_metrics.json --feature-noise 4
```

Cờ mới trong `train_kg_head.py` (session này): `--feature-noise`, `--predicates-from`, `--min-train-auc`
(`none` = giữ hết), `--predicate-select global|pair`. Chẩn đoán in ra: `v true vs rival` (`verify_headroom`,
có tách hòa do `no opposed predicate in KG`) và `pair-gated v true vs rival` (`gated_pair_check`, trần của VERIFY
có cổng theo cặp, `pair_reliability` tính trên train). `tools/hard_pairs.py --predicates-from`.
Code đã push (commit của người dùng); file này và drawio chưa commit.

### Kết quả chính (seed 0)

| | Top1 | λ | AUC thuộc tính | lỗi: `v` đúng / sai / hòa | lỗi, có cổng: phủ / đúng / sai |
|---|---|---|---|---|---|
| K0 | 87.63 | | | | |
| Full, 11 vị từ (lọc toàn cục), nhiễu 4 | 87.10 | 1.42 | 0.847 | 17.3 / 16.7 / 66.0 | |
| K2, cùng 11 vị từ | 87.07 | 0.59 | 0.532 | 27.9 / 30.0 / 42.1 | |
| Full, 29 vị từ (lọc theo cặp), nhiễu 4 | 87.06 | 1.69 | 0.695 | 50.9 / 47.9 / 1.3 | 12.0% / 49.4 / 49.0 |
| K2, cùng 29 vị từ | 87.06 | 0.70 | 0.523 | 51.9 / 47.9 / 0.3 | 5.8% / 52.2 / 47.0 |

1. **Grounding đạt**: vị từ khớp KG thật (AUC 0.70–0.85) chứ không khớp KG xáo trộn (~0.52); λ tăng với KG thật,
   giảm với KG xáo trộn. Đây là kết quả K2 dương duy nhất hiện có.
2. **Top1 không tăng**: head ≈ retrieval ≈ 87.06–87.10 < K0. Retrieval đồ thị G1 ≈ G1-0 (R-GCN không giúp).
3. **Nút thắt**: trên các mẫu retrieval sai ở test, bằng chứng vị từ ngẫu nhiên, **kể cả** khi chỉ dùng vị từ tách
   đúng cặp (lớp đúng, đối thủ) trên train (49.4 / 49.0, ~720 mẫu). Trên mẫu retrieval đúng thì tốt (67 / 32).
   Giải thích: lỗi tương quan — mẫu backbone nhầm có chuyển động giống lớp đối thủ, vị từ đo trên cùng skeleton.
   Không phải do λ, chuẩn hóa, k, hay độ phủ (độ phủ đã giải quyết: hòa 66% → 1.3%).
4. Nhiễu `z` khi train: N = 4 chọn theo quy tắc chốt trước (train top1 ≤ 92%); làm retrieval tốt hơn
   (Recall@10 98.80 > K0 98.72) và cho λ tín hiệu học.

### Việc tiếp theo (chưa chốt, cần người dùng chọn)

Muốn VERIFY nâng Top1 thì bằng chứng phải mang thông tin **độc lập** với backbone đúng ở chỗ nó sai:

| Hướng | Nội dung | Ghi chú |
|---|---|---|
| B. Vị từ thứ tự thời gian (THEN) | "A rồi B" (put_on / take_off, ...): pool theo thời gian của backbone có thể làm mất thứ tự | Kiểm tra trần trước: chạy `gated_pair_check` chỉ trên lỗi nhóm put_on / take_off với vị từ mới |
| B'. Khái niệm từ `D:/Thesis/REASON/concepts/ntu_rgbd_60_120.csv` | 120 × 74 khái niệm nhị phân, khớp tên 120/120; nhóm `temporal_direction_*`, `temporal_sequence_*` tách được các cặp put_on / take_off | Chỉ lấy ô 1 → `has` (0 = chưa biết, không phải `lacks`); chỉ khái niệm đo được; ghi nguồn; `writing` = `cutting_nails` (trùng hàng) |
| Node logic AND / OR / NOT / THEN | Lớp = công thức trên literal; AND/NOT chỉ là has/lacks viết lại, thông tin mới chỉ ở THEN (và OR trái/phải) | Cần đường cong vị từ theo frame trong cache (~160 MB fp16) |
| A1-dec (đã có code, chưa chạy) | Decoder thuộc tính trên map l9 trước pool (`config/kg_ft/ft_a1_dec*.yaml`, `main.py`) | Rủi ro học tắt ra lớp + lỗi tương quan; vài giờ GPU/run |
| Cổng theo cặp trong mô hình | Dùng `R[c, c', j]` trong VERIFY | Trần đã đo = ngẫu nhiên trên lỗi → không đáng làm với vị từ hiện tại |
| k nhỏ (`--head-arg k=3`) | | Dự đoán ghi trước: Top1 87.0–87.1, sửa ≈ phá, Full ≈ K2 (trần cặp đã ngẫu nhiên) |

Nếu không hướng nào nâng được Top1: viết A1-G như nghiên cứu phân tích (grounding đạt, K2 tách; hậu kiểm bằng vị từ
không sửa được lỗi vì lỗi tương quan), K5 vẫn đo được.
