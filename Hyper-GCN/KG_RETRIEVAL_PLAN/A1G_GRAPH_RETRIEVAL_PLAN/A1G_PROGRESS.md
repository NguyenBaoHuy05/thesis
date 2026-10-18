# A1-G: tiến độ triển khai

> Kế hoạch: [A1G_GRAPH_RETRIEVAL_PLAN.md](A1G_GRAPH_RETRIEVAL_PLAN.md) (pha G0–G6 ở §8).
> Số liệu A1 dùng làm mốc: [A1_PROGRESS.md](../A1_RETRIEVE_VERIFY_FUSE_PLAN/A1_PROGRESS.md).
> Số liệu chép từ log chạy thật; ô chưa có số ghi *chờ*, không ước lượng.

Cập nhật lần cuối: **2026-10-18**

## Tổng quan

| Pha | Việc | Trạng thái | Còn thiếu |
|---|---|---|---|
| G0 | Code + test | **Đã viết**. Test numpy (vị từ) 10/10 pass trên máy local; test torch chưa chạy (máy local không nạp được torch) | `python -m pytest tests -q` trên Colab |
| G1 | Retrieval đồ thị (G1, G1-0, G1-K2) | Chưa chạy | |
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
| G1 | *chờ* | *chờ* | *chờ* | *chờ* |
| G1-0 | *chờ* | *chờ* | *chờ* | *chờ* |
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

*chờ*
