# A3-G: hyperedge do vị từ hình học định nghĩa, bên trong Hyper-GCN

> Trạng thái: **code xong, đã review và sửa, chưa chạy test** (2026-10-27). Test cần torch → chạy trên Colab (§7).
> Gốc: [A3_PLAN.md](../A3_CONCEPT_HYPEREDGE/A3_PLAN.md) (A3 = 87.47 < K0 87.64). Vị từ: [PREDICATES.md](../PAIR_REFEREE/PREDICATES.md).
> Tiến độ: [CONCEPT_GRAPH_PROGRESS.md](../CONCEPT_GRAPH/CONCEPT_GRAPH_PROGRESS.md).
> Dev: 1 seed (0). Seed 1, 2 chỉ khi kết quả sát ngưỡng (±0.1) hoặc khi chốt số.

## 0. Ý tưởng và khác gì A3

A3 đưa tri thức vào topology của l7–l9, nhưng có hai điểm yếu:

| Điểm yếu của A3 | Hậu quả | A3-G sửa thế nào |
|---|---|---|
| Trọng số hyperedge `c = σ(W_c z)` suy từ z của chính K0 (concept_match 84.11 < K0) | G_KG chỉ là "gợi ý lớp K0 đang đoán", sai cùng chỗ với K0 | Trọng số = **vị từ hình học tính trên toạ độ thô**, không qua backbone |
| B theo nhóm thân: 74 hyperedge nhưng ≤ 7 tập khớp khác nhau (ARMS 13 khớp, ...) | Gần như làm trơn trong khối thân, ít cấu trúc | **21 hyperedge nhỏ (3–6 khớp)**, mỗi cái là đúng các khớp của một vị từ, chọn theo tay active |
| G_KG tĩnh theo thời gian | Không nói được "tay chạm miệng ở giữa clip" | G thay đổi **theo frame** |

Tuyên bố cần chứng minh: *topology của GCN được điều khiển bởi vị từ symbolic đo trên mẫu, theo thời gian, cho kết
quả tốt hơn (a) fine-tune cùng công thức không có vị từ, (b) cùng vị từ nhưng của mẫu khác, (c) cùng vị từ đưa vào
làm feature đầu vào.* (c) là ablation bắt buộc cho tính mới (xem §3).

Concept CSV và concept head **không dùng** trong A3-G.

## 1. Kiến trúc

```
x [N, 3, 64, 25, 2] (feeder, joint mode)
 │
 ├─► l1 … l6 (K0, đóng băng) ─► F6 [N·M, 2C, 32, 25]            dùng chung, tính một lần
 │
 └─► GeoPredicates (không học, không gradient)
        hệ trục thân theo từng người (up, right, forward, L)  ← từ A1-G `Frame`
        tay active a / other o theo từng người (quãng đường lớn hơn)
        p[n, e, t] ∈ [0, 1], e = 1..21, t = 1..64               chuẩn hoá bằng q5 / q95 của train (§2)
        avg-pool theo t ─► p³² (cho l7), p¹⁶ (cho l8, l9)

Hyperedge vị từ (theo mẫu, theo frame):
        B[n, e, v] ∈ {0, 1}: tập khớp của hyperedge e (§1.1), phụ thuộc tay active của người n
        H[n, t, v, e] = B[n, e, v] · p[n, e, t]                   hàng virtual joint = 0
        G_geo[n, t] = hyper_norm(H[n, t], softplus(w_e))          [N·M, T, 28, 28], chung cho 8 subset

Lượt 2:  F6 ─► l7′ … l9′ ─► pooling ─► fc ─► logits
         trong mỗi HYPERGC của l7′ … l9′:
           y = Σ_k A_k · d_x_k                                   (gốc: PA + α·G_dyn, theo subset k)
             + β_l · Σ_k G_geo[t] · d_x_k[t]                     (mới: theo frame, einsum 'ntuv,nkctv->nkctu')
         l7′ … l9′ = bản sao l7 … l9 của K0, β_l = 0           ⇒ epoch 0 = K0

L = CE (label smoothing 0.1) + divergence (của Hyper-GCN)        không có loss concept
```

Số frame mà GCN nhìn thấy: l7 nhận T = 32 (sau stride của l4), stride của l7 nằm ở TCN sau GCN; l8, l9 nhận T = 16.
Vì vậy G_geo của l7 dùng p³², của l8 và l9 dùng p¹⁶.

Hai người: Hyper-GCN xử lý từng người riêng (N·M), nên G_geo cũng tính riêng cho từng người. Người không có thật
(toàn 0) → p = 0 → H = 0 → G_geo = 0. Vị từ hai người (nhóm G của PREDICATES.md) **không** làm hyperedge được
vì không có cạnh giữa hai người; để cho Pair Referee.

### 1.1. Bảng hyperedge (chốt trước khi chạy)

Khớp NTU 0-based. Tay active `a` = R: vai 8, khuỷu 9, cổ tay 10, bàn tay 11, đầu ngón 23, ngón cái 24;
`a` = L: 4, 5, 6, 7, 21, 22. Tay `o` là tay còn lại. Mã vị từ theo PREDICATES.md.

| e | Hyperedge | Khớp | Trọng số p | Vị từ |
|---|---|---|---|---|
| 1 | `mouth_a` | cổ tay a, bàn tay a, đầu ngón a, cổ 2, đầu 3 | `near_mouth_a` | A1 |
| 2 | `mouth_o` | cổ tay o, bàn tay o, đầu ngón o, 2, 3 | `near_mouth_o` | A2 |
| 3 | `ear_a` | cổ tay a, bàn tay a, đầu ngón a, 3 | `near_ear_a` | A4 |
| 4 | `above_head_a` | khuỷu a, cổ tay a, bàn tay a, 3 | `above_head_a` | A5 |
| 5 | `chest_a` | cổ tay a, bàn tay a, 1, 20 | `near_chest_a` | A6 |
| 6 | `behind_a` | cổ tay a, bàn tay a, 0, 1 | `hand_behind` | A11 |
| 7 | `hands_close` | 6, 7, 10, 11 | `hands_close` | A9 |
| 8 | `hand_foot` | 7, 11, 14, 15, 18, 19 | `hand_near_foot` | A10 |
| 9 | `thumb_up_a` | cổ tay a, bàn tay a, ngón cái a | `clip(2ê − 1, 0, 1)` của `thumb_up_a` | B1 |
| 10 | `thumb_down_a` | như 9 | `clip(1 − 2ê, 0, 1)` | B1 |
| 11 | `pinch_a` | bàn tay a, đầu ngón a, ngón cái a | `1 − thumb_tip_gap_a` | B3 |
| 12 | `elbow_up_a` | vai a, khuỷu a, cổ tay a | `elbow_raised_a` | C2 |
| 13 | `elbow_up_o` | vai o, khuỷu o, cổ tay o | `elbow_raised_o` | C3 |
| 14 | `head_back` | 20, 2, 3 | `head_back` | D1 |
| 15 | `head_down` | 20, 2, 3 | `head_down` | D2 |
| 16 | `torso_bend` | 0, 1, 20, 2, 12, 16 | `torso_bend` | D4 |
| 17 | `pelvis_low` | 0, 12, 16, 13, 17 | `pelvis_low` | E1 |
| 18 | `foot_raised` | 13, 14, 15, 17, 18, 19 | `foot_raised` | E3 |
| 19 | `rising_a` | vai a, khuỷu a, cổ tay a, bàn tay a | `clip(2ê − 1, 0, 1)` của `rising_a` | F2 |
| 20 | `lowering_a` | như 19 | `clip(1 − 2ê, 0, 1)` | F2 |
| 21 | `fine_a` | cổ tay a, bàn tay a, đầu ngón a, ngón cái a | `fine_motion_a` | F7 |

Vị từ có dấu (B1, F2) tách thành hai hyperedge, vì một hyperedge chỉ mang **độ mạnh**, không mang hướng. Hai
hyperedge cùng tập khớp (14 / 15, 9 / 10, 19 / 20) chỉ khác nhau ở trọng số và w_e riêng. Không dùng làm hyperedge:
vị từ chỉ là độ lớn, không phải quan hệ giữa khớp (F1, F4, F5 tốc độ), C1, D3, D5, E2, F3, F6, A3, A7, A8, nhóm G.

### 1.2. Quyết định thiết kế (chốt trước khi chạy)

| | Chọn | Lý do |
|---|---|---|
| Nguồn trọng số | Vị từ hình học, `no_grad` | Độc lập với backbone; không để CE nắn vị từ |
| Chuẩn hoá | `ê = clip((e − q5) / (q95 − q5), 0, 1)`, q theo vị từ trên train (vị từ có dấu: `0.5 + e / 2q`) | Giữ mức tuyệt đối (A1-G min-max theo mẫu làm mất) |
| Theo thời gian | G_geo theo frame, avg-pool 64 → 32 / 16 | Đúng số frame GCN thấy ở từng tầng |
| Dùng chung subset | Một G_geo cho 8 subset | Như A3; ít tham số |
| Trọng số hyperedge | `w_e` softplus, khởi tạo 1 (21 tham số) | Như A3 |
| Khởi tạo | β = 0, l7′–l9′ = K0 | Epoch 0 = 87.63 (kiểm tra cài đặt) |
| Nạp trọng số | Checkpoint K0 trực tiếp (không cần S1, vì không có concept head) | |
| Bất biến | Vị từ tính trong hệ trục thân, nên không phụ thuộc `random_rot` | Train có augmentation, vị từ vẫn đúng |

## 2. Huấn luyện

| | G0 · Thống kê vị từ (CPU / GPU, không train) | G1 · Fine-tune lượt 2 |
|---|---|---|
| Lệnh | `tools/predicate_stats.py` | `tools/run_a3g.py` |
| Dữ liệu | train, transform của test (tất định) | train có augmentation như A3 |
| Ra | `predicate_stats.npz` (q5, q95 cho 19 vị từ gốc được dùng trong bảng hyperedge, gồm `thumb_tip_gap_a` và `rising_a`), báo cáo AUC theo cặp, kiểm tra bất biến | checkpoint, log |
| Học | — | l7′–l9′, fc, β, w_e (lr ×10 cho β, w_e) |
| Đóng băng | — | l1–l9 lượt 1 (`freeze_until: l9`) |
| Tối ưu | — | **đúng công thức S2 của A3**: SGD nesterov 0.0025, wd 4e-4, 15 epoch, giảm lr ở 10, báo epoch cuối |

Thời gian ước lượng (theo A3 S2: ~8.1 phút train + ~5.4 phút eval mỗi epoch; einsum theo frame thêm một ít):
khoảng 2.5 giờ mỗi run, 5 run ≈ 12.5 giờ GPU.

**K0-ft (quyết định 2026-10-27):** chạy riêng `a3g_off` từ checkpoint K0 (cùng code, cùng công thức, không có G_geo),
không dùng lại `s2_k0ft` của A3. Lý do: checkpoint A3 có khoá `a3.*` khác `a3g.*`, và cùng code path thì so sánh sạch
hơn. Nếu `s2_k0ft` của A3 cũng đã chạy, hai số phải gần nhau (kiểm tra chéo).

## 3. Thí nghiệm và quy tắc đạt (chốt trước)

| Mã | Config | Khác với A3-G | Kiểm tra |
|---|---|---|---|
| K0 | checkpoint | — | mốc 87.63 |
| K0-ft | `config/a3g/a3g_off.yaml` | `mode: off`: không có G_geo | lợi ích chỉ do fine-tune |
| **A3-G** | `config/a3g/a3g.yaml` | — | mô hình chính |
| Shuffle | `a3g_shuffle.yaml` | p của mẫu `n` lấy từ mẫu `(n + 1) mod batch` (cả train và test) | bằng chứng **theo mẫu** có tác dụng không |
| Input | `a3g_input.yaml` | không có G_geo; `F6 += Linear(21 → 2C)(p³²)` cộng đều vào mọi khớp, Linear khởi tạo 0 | **cùng thông tin, khác vị trí**: topology có hơn feature không |
| Random | `a3g_random.yaml` | B thay bằng tập khớp ngẫu nhiên cùng kích thước (seed 0) | tập khớp có ý nghĩa không |

Số phụ mỗi lần eval: `no_geo` (cùng checkpoint, β = 0 lúc test), `beta_l7/l8/l9`, `w_geo` trung bình.

**A3-G đạt nếu** Top1(A3-G) − Top1(K0-ft) ≥ 0.3 **và** Top1(A3-G) − Top1(Shuffle) ≥ 0.3.

**Tuyên bố "topology từ vị từ"** chỉ được viết nếu thêm Top1(A3-G) − Top1(Input) ≥ 0.2. Nếu A3-G đạt mà Input ≈
A3-G, kết luận đổi thành "vị từ hình học mang thông tin bổ sung" (không có điểm mới về vị trí đặt).

Random chỉ dùng để giải thích: Random ≈ A3-G nghĩa là tác dụng nằm ở trọng số theo thời gian, không ở tập khớp.

Chênh trong ±0.1 quanh một ngưỡng → chạy seed 1, 2 cho các run liên quan.

**Dự đoán ghi trước khi chạy** (để so sau):
- A3-G − K0-ft trong khoảng [−0.1, +0.4]; khả năng đạt ngưỡng 0.3 dưới một nửa.
- `no_geo` tụt so với A3-G ít hơn A3 (A3: tụt 2.2 điểm), vì G_geo không mang mã lớp.
- Shuffle < A3-G nếu vị từ được dùng; Shuffle ≈ K0-ft nếu không.
- Input ≈ A3-G ± 0.2 (rủi ro lớn nhất cho tính mới).

Chẩn đoán khi không đạt: β có rời 0; `no_geo` có khác A3-G; số lỗi trên các cặp ở PREDICATES.md §5 trước / sau.

## 4. Điểm dừng trước khi tốn GPU (G0)

Chạy `tools/predicate_stats.py` trước. Dừng, không chạy G1, nếu một trong các điều sau xảy ra:

| Kiểm tra | Ngưỡng dừng |
|---|---|
| `pytest tests/test_a3g.py` | có test lỗi |
| Giá trị vị từ | có NaN / inf |
| Bất biến xoay (±0.3 rad quanh trục dọc, cùng mẫu), trung bình `|Δê|` | ≥ 0.02 |
| AUC trên **train** của 13 cặp ở PREDICATES.md §5: số cặp có ít nhất một hyperedge (trung bình theo 8 đoạn hoặc cả clip) đạt AUC ≥ 0.75 hoặc ≤ 0.25 | < 7 / 13 |

13 cặp = PREDICATES.md §5 bỏ `giving_object / exchange_things` (vị từ hai người không phải hyperedge):
into / out bag, put_on / take_off a_shoe, headphone, hat_cap, thumb_up / down, OK / victory, victory / snap_fingers,
drink_water / yawn, yawn / blow_nose, hush / blow_nose, reading / writing, writing / play_with_phone_tablet,
reading / staple_book (`PAIRS` trong `tools/predicate_stats.py`).

Cách tính (chốt): train và test đều dùng transform của test (`p_interval [0.95]`, không xoay); quantile chỉ trên frame
của người thật; mỗi mẫu lấy max theo người thật của p; 189 cột = 21 hyperedge × (8 đoạn + cả clip); AUC gập
`max(AUC, 1 − AUC)` của cột tốt nhất. Bất biến xoay = trung bình `|Δê|` trên test, cùng thứ tự mẫu, có và không có
`random_rot`. Report ghi `passed`; `run_a3g.py` chỉ chạy G1 khi `passed: true`, kể cả khi chạy lại.

Báo cáo thêm (không phải điểm dừng): cùng AUC đó trên test, tách riêng mẫu K0 sai và K0 đúng (`k0_scores` trong
`run_a3g.yaml`, file score pkl chính thức của K0). Chỉ đọc, không dùng để chỉnh gì.

## 5. Mã nguồn

| File | Nội dung |
|---|---|
| `model/geo_predicates.py` (mới, torch) | `absolute`, `valid_actors`, `BodyFrame`; 19 vị từ gốc theo frame (bao gồm `thumb_tip_gap_a` và `rising_a` mà bảng hyperedge sử dụng); chọn tay active; chuẩn hoá theo `predicate_stats.npz`; `HYPEREDGES` (bảng §1.1); `hyperedge_weights(x) -> p [N·M, 21, 64]`, `incidence(x) -> B [N·M, 21, 25]`. Dùng lại cho Pair Referee sau này |
| `model/hypergcn_large.py` | thêm tham số tuỳ chọn `temporal_adjacency` [N, T, V′, V′] cho `HYPERGC.forward` và `TCN_GCN_unit.forward`: `y += einsum('ntuv,nkctv->nkctu', G, d_x)`. Mặc định None: tầng gốc giữ nguyên từng bit |
| `model/hypergcn_a3g.py` (mới) | `Model` (lượt 2 như A3, không có lượt 1 concept); `mode`: `geo` / `shuffle` / `input` / `random` / `off` (= K0-ft); p, B tính một lần mỗi forward rồi pool theo T của từng tầng; `no_geo` khi eval; `w_geo`, `beta`, `input_proj` dưới tiền tố `a3g.`; dùng lại `hyper_norm`, `PASS2` của A3 |
| `model/training.py` | thêm tiền tố `a3g.` vào nhóm tham số mở rộng (như `a3.`) |
| `tools/predicate_stats.py` (mới) | G0: một lượt qua train (transform test) → `predicate_stats.npz`; AUC theo cặp; bất biến xoay; `--k0-scores` cho báo cáo test |
| `tools/run_a3g.py` (mới) | một lệnh: test → chép dữ liệu → G0 (nếu chưa có stats) → kiểm tra điểm dừng §4 → các run G1; bỏ qua run đã xong |
| `config/a3g/*.yaml` | `a3g` (đầy đủ), `a3g_off`, `a3g_shuffle`, `a3g_input`, `a3g_random` (chỉ đổi `mode`), `run_a3g.yaml` (đường dẫn Drive, thứ tự run) |
| `tests/test_a3g.py` (mới) | §6 |

## 6. Test (`tests/test_a3g.py`, cần torch)

- `temporal_adjacency=None` cho đúng output của tầng gốc; G hằng theo t cho cùng kết quả với `extra_adjacency` của A3.
- Khởi tạo (β = 0, Linear của Input = 0): mọi `mode` cho đúng logits K0; `no_geo` = K0.
- Vị từ trên khung xương tổng hợp đúng dấu: tay chạm miệng → `mouth_a` cao; ngón cái lên / xuống → 9 / 10; ngồi →
  `pelvis_low` cao; tay trái làm động tác → tay active = trái.
- Bất biến xoay ±0.3 rad: `|Δp|` < 1e-4 trên dữ liệu tổng hợp.
- `B`: đúng số khớp mỗi hyperedge; đổi tay active thì khớp tay đổi theo; hàng virtual joint của G_geo = 0.
- Người rỗng → G_geo = 0.
- Shuffle: p của mẫu n bằng p của mẫu n + 1; Random: cùng kích thước, khác B thật, dựng lại giống hệt với cùng seed.
- Gradient: tới l7′–l9′, fc, β, w_e; không tới l1–l9 lượt 1; không có gradient vào vị từ.

## 7. Các bước chạy từ đầu (Colab)

**Ô 1: GPU, Drive, code** (mỗi phiên)
```
from google.colab import drive
drive.mount('/content/drive')
```
```
%cd /content/Hyper-GCN
!git pull
!git log --oneline -1
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Ô 2: kiểm tra đường dẫn** (`config/a3g/run_a3g.yaml`)
```
!ls -la /content/drive/MyDrive/BaoHuy/KLTN_Huy-Nhu/NTU120_CSub.npz
!ls -la /content/drive/MyDrive/BaoHuy/large/ntu120/xsub/joint/runs-122-120048.pt
```

**Ô 3: G0, điểm dừng** (không train; ghi stats vào `<work_dir>/predicate_stats.npz`)
```
!python tools/run_a3g.py --only g0
```
Dán lại: bảng AUC theo cặp, số cặp đạt / 13, bất biến xoay. Không qua §4 thì dừng ở đây.

**Ô 4: toàn bộ G1, một lệnh**
```
!python tools/run_a3g.py
```
Thứ tự: `a3g` → `a3g_off` (K0-ft) → `a3g_shuffle` → `a3g_input` → `a3g_random` (15 epoch mỗi run, đều từ K0).
G1 không chạy nếu report G0 (`<work_dir>/predicate_stats.json`) có `passed: false`. Kết quả ở `/content/drive/MyDrive/BaoHuy/work_dir/a3g/<tên config>/`. Mất kết nối thì chạy lại đúng lệnh: run đã
xong được bỏ qua, run dở chạy lại từ đầu.

Chỉ chạy một phần:
```
!python tools/run_a3g.py --only a3g            # mô hình chính trước
!python tools/run_a3g.py --only a3g_off a3g_shuffle
```

**Kiểm tra cài đặt trước khi đọc kết quả** (`log.txt` của từng run):
- epoch 0 (`Initial evaluation`) của mọi run: Top1 = **87.63** (hoặc 87.64, sai số eval đã biết ở A3). Khác thì dừng.
- `beta_l7`, `beta_l8`, `beta_l9` rời 0 (chỉ có ở `a3g`, `a3g_shuffle`, `a3g_random`).

**Cần dán lại** cho mỗi run: các dòng `Top1`, `Top1 of the other outputs`, `Mean loss components` ở epoch 0, 5, 10,
15, và `best_checkpoint.json`.

## 8. Kết quả

| | Top1 (ep 0 / 5 / 10 / 15) | no_geo | β l7 / l8 / l9 | w_geo |
|---|---|---|---|---|
| K0 | 87.63 | — | — | — |
| K0-ft | *chờ* | — | — | — |
| A3-G | *chờ* | *chờ* | *chờ* | *chờ* |
| Shuffle | *chờ* | *chờ* | *chờ* | *chờ* |
| Input | *chờ* | — | — | — |
| Random | *chờ* | *chờ* | *chờ* | *chờ* |

G0: *chờ*.

## 9. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Vị từ không mang thông tin cho cặp khó | G0 < 7 / 13 cặp | Dừng; ghi kết luận |
| Fine-tune làm hỏng K0 (như A3: epoch 5 tụt 0.47) | Top1 epoch 5 < K0-ft epoch 5 | Không đổi công thức giữa chừng; báo cáo như cũ |
| Mạng bỏ qua G_geo | β ≈ 0, `no_geo` ≈ A3-G | Kết luận: topology đã đủ, thông tin vị từ không cần |
| Input ≈ A3-G | | Bỏ tuyên bố về vị trí đặt (§3) |
| Khớp ngón tay (21–24) nhiễu | AUC B1, B3 thấp ở G0 | Báo cáo; không sửa định nghĩa sau khi xem test |
| Bộ nhớ của einsum theo frame ở l7 | OOM | G_geo [N·M, 32, 28, 28] ≈ 13 MB cho batch 64: không đáng lo; nếu OOM giảm `test_batch_size` |
