# A3: hyperedge do concept định nghĩa, bên trong Hyper-GCN

> Trạng thái: **code xong, chưa chạy** (2026-10-22). Chưa chạy test (cần torch, chạy trên Colab).
> Sơ đồ: [drawio/architecture_a3_concept_hyperedge.drawio](../../drawio/architecture_a3_concept_hyperedge.drawio).
> Ý tưởng gốc: mục A3 trong [KG_RETRIEVAL_PLAN.md](../KG_RETRIEVAL_PLAN.md).
> Dev: 1 seed (0). Chỉ chạy seed 1, 2 khi kết quả sát ngưỡng và khi chốt số cuối.

## 0. Ý tưởng

Mọi head đặt trên đặc trưng đóng băng đều bị giới hạn bởi thông tin của chính đặc trưng đó. A3 đưa tri thức vào
**bên trong** backbone: mỗi concept không gian (ví dụ `hand_touch_head`) là một hyperedge nối các khớp của nó, có
trọng số bằng xác suất concept của chính mẫu đó. Các tầng l7–l9 được chạy lại với hyperedge này cộng vào ma trận kề,
nên tri thức đổi cách đặc trưng được tính, không chỉ đổi lớp ở đầu ra.

Khác REASON: REASON giữ backbone nguyên vẹn và đặt logic concept phía sau. Ở A3 concept quyết định cấu trúc đồ thị
mà backbone dùng.

## 1. Kiến trúc

```
skeleton ─► l1 … l6 (K0, đóng băng) ─► F6                                   tính một lần, dùng cho cả hai lượt

Lượt 1:  F6 ─► l7 … l9 (K0, đóng băng) ─► pooling ─► z ∈ R^512             đúng đặc trưng của K0
         c = σ(W_c z) ∈ (0,1)^102                                           L_concept = BCE(c, M[y])

Hyperedge concept (theo mẫu):
         B ∈ {0,1}^{74×25}: concept không gian → tập khớp                  (concept_joint_mask, theo nhóm metadata)
         H_KG[v, k] = B[k, v] · c_k                                         74 hyperedge, hàng virtual joint = 0
         G_KG = hyper_norm(H_KG, softplus(w_KG))                             đúng hàm HYPERGC.hyper_norm; [N, 1, 28, 28]

Lượt 2:  F6 ─► l7′ … l9′ với A = A_phys + α·G_dyn + β_l·G_KG ─► pooling ─► fc ─► logits
         l7′ … l9′ khởi tạo = l7 … l9 của K0, β_l = 0 (một β cho mỗi tầng)  ⇒ lúc đầu đầu ra = K0

L = CE (label smoothing 0.1) + divergence (của Hyper-GCN) + λ_c · L_concept,   λ_c = 0.5
```

Quyết định thiết kế (chốt trước khi chạy):

| | Chọn | Lý do |
|---|---|---|
| Concept dùng cho hyperedge | 74 concept `scope = spatial` | 25 concept thời gian và 3 concept participant không có tập khớp |
| B | `concept_joint_mask` (nhóm head / upper_limb / hip_torso / lower_limb, thêm đầu / thân theo tên) | có sẵn, không cần chú thích mới; bản theo từng concept để sau |
| c vào G_KG | **detach** | G_KG đọc bằng chứng concept; CE không được nắn c thành mã lớp (đường tắt ECOC) |
| Lượt 1 | K0 đóng băng | nguồn concept ổn định, lượt 2 không kéo lệch được |
| Chạy lại | chỉ l7–l9 | chi phí ~1.3 lần forward; l1–l6 dùng chung |
| Trọng số hyperedge | w_KG theo concept, softplus, khởi tạo 1 | hyperedge của Hyper-GCN cũng có trọng số học được |
| Khởi tạo | β = 0, l7′–l9′ = K0 | epoch 0 phải ra đúng 87.63 (kiểm tra cài đặt) |

## 2. Huấn luyện (2 giai đoạn)

| | S1 · Concept head | S2 · Fine-tune lượt 2 |
|---|---|---|
| Config | `config/a3/s1_concept.yaml` | `config/a3/s2_a3.yaml` (+ 2 đối chứng) |
| Khởi tạo | checkpoint K0 | checkpoint cuối của S1 |
| Học | chỉ `a3.concept` (BCE) | l7′–l9′, fc, β, w_KG, concept head |
| Đóng băng | toàn bộ backbone + fc (`freeze_until: fc`) | l1–l9 của lượt 1 (`freeze_until: l9`) |
| Đầu ra | logits K0 (Top1 = 87.63 mọi epoch) | logits lượt 2 |
| Tối ưu | Adam 1e-3, 10 epoch, giảm lr ở epoch 8 | SGD nesterov 0.0025, wd 4e-4, 15 epoch, giảm ở 10; lr ×10 cho β, w_KG |

S2 dùng đúng công thức fine-tune nhẹ đã dùng cho K0-ft trước đây. Báo cáo epoch cuối (`select_best: False`), không
chọn epoch theo test.

## 3. Thí nghiệm và quy tắc đạt (chốt trước)

| Mã | Config | Khác với A3 |
|---|---|---|
| K0 | (checkpoint) | mốc 87.63 |
| **A3** | `s2_a3.yaml` | — |
| K0-ft | `s2_k0ft.yaml` | `use_kg: False`: cùng lượt 2, cùng công thức, không có G_KG |
| Random | `s2_random.yaml` | `mask: random`: B thay bằng tập khớp ngẫu nhiên cùng kích thước (seed 0) |

Số phụ ghi ở mỗi lần eval (`Top1 of the other outputs`): `no_kg` (cùng checkpoint A3, β = 0 lúc test: mức phụ thuộc
vào G_KG), `concept_match` (lớp suy từ c, không có trọng số lớp: chất lượng concept). Giá trị β_l7, β_l8, β_l9 và
w_KG trung bình nằm trong `Mean loss components` của mỗi epoch.

**A3 đạt nếu** Top1(A3) − Top1(K0-ft) ≥ 0.3 **và** Top1(A3) − Top1(Random) ≥ 0.3.
Chênh nằm trong 0.1 quanh ngưỡng thì chạy thêm seed 1, 2 cho cả ba.

Chẩn đoán khi không đạt: β có rời 0 không; `no_kg` có tụt so với A3 không (nếu không tụt, G_KG không được dùng);
`concept_match` có cao không (c có mang thông tin concept không).

## 4. Mã nguồn

| File | Nội dung |
|---|---|
| `model/hypergcn_large.py` | thêm tham số tùy chọn `extra_adjacency` cho `HYPERGC.forward` và `TCN_GCN_unit.forward` (mặc định None: giữ nguyên tầng gốc) |
| `model/hypergcn_a3.py` (mới) | `Model` (hai lượt, hai giai đoạn, đối chứng), `A3Head` (concept head, B, w_KG, β, l7′–l9′), `hyper_norm`, `random_joint_sets` |
| `model/training.py` | thêm tiền tố `a3.` vào tham số mở rộng (luôn train, không có trong checkpoint chính thức) |
| `config/a3/*.yaml` | S1, S2 (A3, K0-ft, Random), `run_a3.yaml` (đường dẫn Drive) |
| `tools/run_a3.py` (mới) | một lệnh: test → chép dữ liệu → S1 → các S2; bỏ qua run đã xong (có `best_checkpoint.json`) |
| `tests/test_hypergcn_a3.py` (mới) | xem §5 |

Nạp checkpoint: checkpoint chính thức không có `a3.layers.*`, nên `Model.load_state_dict` chép l7–l9 vào đó.
`load_model_weights` vẫn in các khoá này trong `initialized`, nhưng giá trị là của K0. Checkpoint S1 đã có
`a3.layers.*` nên S2 giữ nguyên.

## 5. Test (`tests/test_hypergcn_a3.py`, cần torch)

- `hyper_norm` trùng `HYPERGC.hyper_norm`.
- Cả hai giai đoạn lúc khởi tạo cho đúng logits K0; `no_kg` = K0; 9 hyper output (divergence loss).
- l7′–l9′ được chép từ K0; checkpoint đã có `a3.layers` thì không bị ghi đè.
- G_KG: shape [N, 1, 28, 28], hàng / cột virtual joint = 0; chỉ một concept bật thì G_KG chỉ nằm trên khớp của concept đó.
- Mặt nạ ngẫu nhiên: cùng kích thước hàng, khác B thật, dựng lại giống hệt với cùng seed.
- S2: gradient tới l7′–l9′ (l9′), fc, β (kể cả khi β = 0); không tới l7–l9 lượt 1; CE không tới concept head (detach);
  BCE tới concept head.
- K0-ft: β không có gradient, không có `no_kg`.
- S1: chỉ concept head có gradient.

## 6. Các bước chạy từ đầu (Colab)

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

**Ô 2: kiểm tra đường dẫn** (`config/a3/run_a3.yaml`, lấy từ `script.txt`)
```
!ls -la /content/drive/MyDrive/BaoHuy/KLTN_Huy-Nhu/NTU120_CSub.npz
!ls -la /content/drive/MyDrive/BaoHuy/large/ntu120/xsub/joint/runs-122-120048.pt
```

**Ô 3: toàn bộ A3, một lệnh**
```
!python tools/run_a3.py
```
Thứ tự: test → chép dữ liệu vào `data/ntu120/` → S1 (10 epoch) → `s2_a3` → `s2_k0ft` → `s2_random` (15 epoch mỗi
run). Kết quả ở `/content/drive/MyDrive/BaoHuy/work_dir/a3/<tên config>/`. Mất kết nối thì chạy lại đúng lệnh trên:
run đã xong được bỏ qua, run dở chạy lại từ đầu.

Chỉ chạy một phần (ví dụ khi hết giờ GPU):
```
!python tools/run_a3.py --only s2_a3          # S1 (nếu chưa có) rồi chỉ A3
!python tools/run_a3.py --only s2_k0ft s2_random
```

**Kiểm tra cài đặt trước khi đọc kết quả** (log của từng run, `log.txt` trong thư mục run):
- epoch 0 (`Initial evaluation`) của mọi run: Top1 = **87.63** (K0). Khác thì dừng, báo lại.
- S1: Top1 giữ 87.63 mọi epoch; `concept_bce` giảm; `concept_match` tăng.
- S2 A3: `beta_l7`, `beta_l8`, `beta_l9` rời 0.

**Cần dán lại** cho mỗi run: các dòng `Top1`, `Top1 of the other outputs`, `Mean loss components` ở mỗi lần eval
(epoch 0, 5, 10, 15), và `best_checkpoint.json`.

## 7. Kết quả

| | Top1 epoch cuối | no_kg | concept_match | β_l7 / β_l8 / β_l9 cuối |
|---|---|---|---|---|
| K0 | 87.63 | — | — | — |
| S1 | *chờ* (phải = 87.63) | — | *chờ* | — |
| **A3** | *chờ* | *chờ* | *chờ* | *chờ* |
| K0-ft | *chờ* | — | *chờ* | — |
| Random | *chờ* | *chờ* | *chờ* | *chờ* |

## 8. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| c đến từ cùng backbone: lợi ích chỉ là thiên kiến cấu trúc | A3 ≈ Random | ghi nhận; thử B theo từng concept (chú thích tay) |
| Fine-tune lấn át: K0-ft đã tăng nhiều | A3 ≈ K0-ft > K0 | kết luận lợi ích là do fine-tune, không do tri thức |
| β bị weight decay kéo về 0 | β ≈ 0 suốt, no_kg = A3 | lr ×10 đã đặt; nếu vẫn 0 thì tắt wd cho β (cần sửa `load_optimizer`) |
| G_KG nhiễu khi c sai | Top1 tụt dưới K0 | xem `concept_match`; S1 dài hơn |
| Bộ nhớ: thêm 3 tầng | OOM ở batch 64 | `batch_size: 32` trong các config S2 |
