# A1-G: tiến độ triển khai

> Kế hoạch: [A1G_GRAPH_RETRIEVAL_PLAN.md](A1G_GRAPH_RETRIEVAL_PLAN.md) (pha G0–G6 ở §8).
> Số liệu A1 dùng làm mốc: [A1_PROGRESS.md](../A1_RETRIEVE_VERIFY_FUSE_PLAN/A1_PROGRESS.md).
> Số liệu chép từ log chạy thật; ô chưa có số ghi *chờ*, không ước lượng.

Cập nhật lần cuối: **2026-10-18**

## Tổng quan

| Pha | Việc | Trạng thái | Còn thiếu |
|---|---|---|---|
| G0 | Code + test | **Đã viết**. Test numpy (vị từ) 10/10 pass trên máy local; test torch chưa chạy (máy local không nạp được torch) | `python -m pytest tests -q` trên Colab |
| G1 | Retrieval đồ thị (G1, G1-0, G1-K2) | G1, G1-0 seed 0 xong: G1 ≈ G1-0 | G1-K2; thêm seed nếu cần |
| G2 | Full-z, K2-z (chẩn đoán) | Chưa chạy | |
| G3 | Cặp khó, mở rộng KG, kiểm tra vị từ | `hard_pairs.py` đã chạy trên KG thật (dưới); vị từ chưa cache | `cache_predicates.py`, duyệt KG |
| G4 | Full-pred, K2-pred | Chưa chạy (cần G3) | |
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
