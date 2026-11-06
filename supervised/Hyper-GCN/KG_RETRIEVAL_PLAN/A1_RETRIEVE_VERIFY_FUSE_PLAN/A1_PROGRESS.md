# A1: tiến độ triển khai

> Kế hoạch: [A1_RETRIEVE_VERIFY_FUSE_PLAN.md](A1_RETRIEVE_VERIFY_FUSE_PLAN.md) (pha P0–P6 ở §7).
> Nhật ký xây KG: [KG_BUILD_LOG.md](../KG_CONSTRUCTION_LLM_PLAN/KG_BUILD_LOG.md).
> Mỗi lần cập nhật sửa bảng tổng quan và thêm số liệu vào mục của pha tương ứng. Số liệu chép từ log
> chạy thật; ô chưa có số ghi *chờ*, không ước lượng.

Cập nhật lần cuối: **2026-10-17**

## Tổng quan

| Pha | Việc | Trạng thái | Còn thiếu |
|---|---|---|---|
| P0 | Cache đặc trưng X-Sub | **Đạt**: Top1 cache 87.63% so với checkpoint gốc 87.64% | Xong |
| P1 | KG từ LLM + duyệt tay | Phần tự động xong, validator sạch | Duyệt tay ≥ 40 lớp (hiện 0/120) |
| P2 | Embedding text + K1 | **Đạt**: K1 mpnet Recall@10 98.32 ± 0.01%, Top1 86.57 ± 0.02% (3 seed); A1-enc: CLIP 86.39, bge 86.56 (1 seed) | Xong. A1-enc chạy lại ở zero-shot (P5) |
| P3 | Head đầy đủ + K2 | **Xong, kết quả âm**: Full 86.29 ± 0.02 < K2 (KG xáo trộn) 86.48 ± 0.08 < K1 86.57 < K0 87.63. KG không giúp Top1 lớp đã thấy | Giả thuyết ECOC **đã kiểm chứng** (`l1` cặp nhầm: thật 8.02 so với xáo trộn ~12.8; Spearman −0.25 so với ≈ 0). Nhánh còn lại của cổng: K6, K5 |
| P4 | Ablation K3, K4, A1-* | Chưa bắt đầu | |
| P5 | K5 can thiệp, K6 zero-shot | Chưa bắt đầu | `evaluate_kg.py` |
| P6 | Fine-tune l8–l9 + head (phương án A) | Đang chạy: **Full-ft 86.62** (epoch 15, +0.30 so với epoch 0) | K0-ft, K2-ft |

Kiến trúc (§1): đã viết đủ `z`, RETRIEVE, `G_q`, đầu thuộc tính `a`, VERIFY (`v_c`), FUSE (`f_c`);
chưa chạy. Loss trong `train_kg_head.py` mới có `CE_all` (bước 4 của P3).

## Quyết định đã chốt

- **Lựa chọn 1 (2026-10-17):** giữ cache l9 đã pool, train head trên cache có regularization, không
  train lại backbone. Lý do và các phương án khác: xem mục *Rủi ro của cache train* bên dưới.
- **Encoder chính: mpnet**; CLIP giữ cho ablation A1-enc; bge chỉ để tham khảo (số liệu P2).
- **Không có tập validation**: công thức train chốt trong config trước khi chạy, test đánh giá một
  lần sau epoch cuối, báo cáo trung bình ± std 3 seed.

## P0: cache đặc trưng

Lệnh (Colab, GPU L4):

```
python tools/cache_features.py --weights /content/large/ntu120/xsub/joint/runs-122-120048.pt \
    --out-dir /content/drive/MyDrive/BaoHuy/kg_cache
```

| Mục | Giá trị |
|---|---|
| Checkpoint | `runs-122-120048.pt` (Hyper-GCN large, NTU120 X-Sub, joint) |
| Nạp checkpoint | `loaded 614, initialized [], ignored []` (khớp hoàn toàn) |
| File | `ntu120_xsub_test.npz`, `ntu120_xsub_train.npz`, `ntu120_xsub_summary.json` trên Drive `BaoHuy/kg_cache` |
| Test | 50 919 mẫu, 1 view; `z` [50919, 512] fp16 (~77 MB) |
| Train | 63 026 mẫu, 5 view `random_rot`; `z` [5, 63026, 512] fp16 (~475 MB) |
| Top1 `fc_logits` train (5 view) | 99.63 / 99.63 / 99.65 / 99.65 / 99.64 % |
| Top1 `fc_logits` test (`P0 check`) | 87.63% / Top5 97.56% (từ `kg_metrics.json` của K1) |
| Top1 gốc của checkpoint | 87.64% (accuracy 0.876392) / Top5 97.56%: `main.py --phase test`, 2026-10-17 |
| **Kết quả P0** | **Đạt**: lệch ≈ 0.01 điểm (< 0.1); Top5 trùng khớp |

`summary.json` hiện chỉ có phần train (lần chạy `--splits train` ghi đè); file test không đổi.

### Rủi ro của cache train

Backbone đã train 140 epoch trên chính tập train: Top1 train ~99.6% (khoảng 230 mẫu sai) so với test
~89–90% (khoảng 5 000 mẫu sai). Hệ quả cần theo dõi:

| Rủi ro | Dấu hiệu | Xử lý trong lựa chọn 1 |
|---|---|---|
| Head không học được cách sửa lỗi backbone | Top1 head ≤ K0 | Ghi nhận; nếu mục tiêu là tăng Top1 thì cân nhắc tách subject (lựa chọn 3) |
| `f` lấn át, λ·v chỉ là trang trí | λ nhỏ, `|λv|` < 20% logit trên test | Bật `γ`, chạy K4 / K4' |
| Quá tự tin | τ kẹp ở 0.01 | Cố định τ hoặc hiệu chỉnh nhiệt độ sau train |
| Không có validation sạch | | Công thức chốt trước, test một lần |

`β·BCE` thuộc tính vẫn có gradient trên mẫu dễ, nên đầu thuộc tính vẫn được học; thứ ít được học là
cách trộn `v` vào quyết định (λ).

Lựa chọn dự phòng: (2) đặc trưng theo khớp `[N, 25, 512]` + decoder thuộc tính có điều kiện KG;
(3) train lại backbone trên ~80% subject, cache 20% còn lại cho head.

## P1: KG

| Mục | Giá trị |
|---|---|
| File | `annotations/action_kg.json`, hash `sha256:fb8c63668bb7…` |
| Quy mô | 120 hành động, 61 thuộc tính, 1 175 cạnh has/lacks; Fleiss' κ 0.805 giữa 3 run LLM |
| `validate_kg.py` | 0 lỗi, 5 cảnh báo (`nod_head_bow`, `shake_head`, `wipe_face` chỉ 2 `has`; `nausea_vomiting`, `support_somebody` không có `lacks`), 0 cặp trùng hồ sơ |
| Duyệt tay | 0/120 lớp; `annotations/kg_review/edges.csv` 1 185 dòng, `human_label` trống |

Việc tiếp:

1. Chọn 40 lớp theo top-25 cặp nhầm của K0 (`fc_logits` test đã cache).
2. Điền `human_label` (ưu tiên priority 1–2), `python tools/kg_review.py import`, chạy lại
   `validate_kg.py`. Giữ bản KG trước duyệt cho ablation A1-kg.
3. Nếu sửa `text` của các cặp gần trùng (xem P2), **chạy lại `build_kg_embeddings.py`**: hash đổi
   thì `model/kg.py` từ chối file embedding cũ.

## P2: embedding text và K1

### Embedding

Script: `tools/build_kg_embeddings.py`, test `tests/test_build_kg_embeddings.py`. Mỗi node là trung
bình chuẩn hóa của các mô tả: hành động = `text` + `paraphrases`, thuộc tính = `text` + `definition`;
`action_text` (chỉ `text`) giữ cho ablation.

| Chỉ số | mpnet | bge-large | CLIP ViT-B/32 |
|---|---|---|---|
| D | 768 | 1024 | 512 |
| Lớp gần nhất là `similar`/`opposite` trong KG | **104/120** | 96/120 | 87/120 |
| Hạng trung bình của lớp opposite (1 = gần nhất) | 1.6 | 1.5 | 1.5 |
| Cosine trung bình (không so sánh được giữa encoder) | 0.615 | 0.784 | 0.914 |
| Số cặp > 0.95 (không so sánh được giữa encoder) | 4 | 25 | 870 |
| File | `kg_emb_mpnet.npz` | `kg_emb_bge.npz` | `kg_emb_clip.npz` |

Nhận xét:

- Thứ tự theo chỉ số dựa trên hạng: mpnet > bge > CLIP. Cosine tuyệt đối khác thang đo giữa encoder.
- Cả ba encoder đặt lớp opposite gần như luôn là lớp gần nhất: văn bản không phân biệt hướng
  chuyển động. Các cặp khó nhất trùng nhau ở mọi encoder (`make_OK_sign / make_victory_sign`,
  `drink_water / yawn`, `put_on_headphone / take_off_headphone`, ...) nên nguyên nhân nằm ở văn bản
  KG, không ở encoder. Chỉ VERIFY (ma trận thuộc tính, 0 cặp trùng) tách được; đây là chỗ so
  sánh Full với K1 có ý nghĩa nhất.
- Cả ba file cùng `kg_hash` `sha256:fb8c…`. Hai file bge và CLIP đang ở Colab, cần chép về
  `annotations/` và commit.

### K1 (chỉ RETRIEVE)

Code: `model/kg.py`, `model/kg_retrieval.py` (`RetrievalScorer`, `topk_metrics`), `train_kg_head.py`,
`config/kg/k1.yaml`, test `tests/test_kg_retrieval.py`. Đã chạy trên Colab (mpnet, 3 seed).

Công thức: `s_c = cos(P z, e_c) / τ`, τ học được trong [0.01, 1] (khởi tạo 0.07), dropout 0.1 trên `z`,
Adam lr 1e-3 cosine, 30 epoch, batch 512, mỗi mẫu lấy ngẫu nhiên 1 trong 5 view.

```
python -m pytest tests/test_kg_retrieval.py -q
for s in 0 1 2; do python train_kg_head.py --config config/kg/k1.yaml \
    --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache --work-dir work_dir/kg/k1_mpnet_s$s --seed $s; done
python train_kg_head.py --config config/kg/k1.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
    --embeddings annotations/kg_emb_clip.npz --work-dir work_dir/kg/k1_clip_s0
```

| Cấu hình | Top1 | Top5 | Recall@10 | Recall@20 | τ cuối | Top1 train (view 0) |
|---|---|---|---|---|---|---|
| K0 (`fc_logits`) | 87.63 | 97.56 | 98.72 | 99.31 | | 99.63 |
| K1 mpnet (seed 0 / 1 / 2) | 86.59 / 86.55 / 86.57 | 96.82 / 96.79 / 96.81 | 98.33 / 98.32 / 98.32 | 99.13 / 99.11 / 99.12 | 0.0298 / 0.0297 / 0.0298 | 99.62 / 99.62 / 99.61 |
| **K1 mpnet (TB ± std)** | **86.57 ± 0.02** | **96.81 ± 0.02** | **98.32 ± 0.01** | **99.12 ± 0.01** | 0.030 | 99.62 |
| K1 CLIP (seed 0) | 86.39 | 96.66 | 98.20 | 99.14 | 0.0195 | 99.46 |
| K1 bge-large (seed 0) | 86.56 | 96.74 | 98.27 | 99.15 | 0.0263 | 99.54 |

Tiêu chí qua P2: Recall@10 ≥ 98% trên test. **K1 mpnet: đạt (98.32%, 2026-10-17).**

Nhận xét:

- K1 thấp hơn K0 1.06 điểm Top1: ràng buộc trọng số lớp qua `Pᵀe_c` (để zero-shot được) tốn khoảng
  1 điểm so với FC tự do. Đây là mốc mà VERIFY + FUSE phải bù.
- Trần Top1 của A1 với k = 10 là 98.32%, cao hơn K0 10.7 điểm: retrieval không phải nút thắt.
- τ hội tụ khoảng 0.030, không chạm cận 0.01: chưa có dấu hiệu quá tự tin cực đoan.
- Train ~99.6% so với test 86.6%: đúng độ chênh dự kiến của cache train (mục P0).
- Std giữa seed rất nhỏ (≤ 0.02): `z` và `e_c` cố định, khác biệt chỉ từ khởi tạo `P` và thứ tự batch.
- A1-enc (lớp đã thấy): mpnet 86.57 ≈ bge 86.56 > CLIP 86.39 Top1. Khi có giám sát, `P` học bù được
  phần lớn khác biệt giữa encoder (chênh ≤ 0.2 điểm), cùng thứ tự với chẩn đoán embedding. CLIP hội
  tụ chậm hơn (loss cuối 0.081 so với 0.030) và cần τ nhỏ hơn (0.0195): không gian CLIP dồn (cosine TB
  0.914) nên phải khuếch đại để tách lớp. Khác biệt encoder dự kiến lớn hơn ở zero-shot (K6), nơi
  không có `P` riêng cho lớp mới; chạy lại A1-enc ở P5. CLIP và bge mới 1 seed.

## P3: head đầy đủ và K2

Chia 5 bước: (1) KG thành ma trận, (2) KG xáo trộn, (3) head đầy đủ, (4) loss + train, (5) test.
Bước 1–3 xong (2026-10-17, Colab): `pytest` 36/36 pass.

| Bước | Code | Test |
|---|---|---|
| 1. KG thành ma trận | `model/kg.py: kg_tensors` → `KGTensors(W, y_attr, known, S_sim, S_opp)` | `tests/test_kg.py` |
| 2. KG xáo trộn | `tools/shuffle_kg.py` → `annotations/kg_shuffled/action_kg_shuffle_s{0,1,2}.json` | `tests/test_shuffle_kg.py` |
| 3. Head đầy đủ | `model/kg_retrieval.py: KGRetrievalHead`, `discriminability`, `verify` | `tests/test_kg_retrieval.py` |

Lệnh cần chạy:

```
python -m pytest tests/test_kg.py tests/test_shuffle_kg.py tests/test_kg_retrieval.py -q
python tools/shuffle_kg.py --seeds 0 1 2
for s in 0 1 2; do python tools/validate_kg.py --kg annotations/kg_shuffled/action_kg_shuffle_s$s.json; done
```

Kết quả KG xáo trộn (`--swaps-per-edge 10`):

| Seed | has dời chỗ | lacks dời chỗ | similar | opposite | `validate_kg` |
|---|---|---|---|---|---|
| 0 | 583/739 (78.9%) | 371/444 (83.6%) | 262/271 | 9/9 | 0 lỗi, 0 cặp trùng |
| 1 | 572/739 (77.4%) | 370/444 (83.3%) | 265/271 | 8/9 | 0 lỗi, 0 cặp trùng |
| 2 | 563/739 (76.2%) | 369/444 (83.1%) | 257/271 | 8/9 | 0 lỗi, 0 cặp trùng |

- Kỳ vọng ban đầu "> 90% dời chỗ" **sai**: giữ bậc mà có thuộc tính phổ biến (`single_occurrence`
  60/120 lớp, `single_arm_active` 49, `raise_hold_lower` 30) thì một cạnh của thuộc tính đó có xác
  suất cao rơi lại đúng chỗ cũ. ~20% cạnh còn nguyên chủ yếu là cạnh của thuộc tính phổ biến, ít
  thông tin; cạnh đặc thù (nơi chứa tri thức) gần như đều bị xáo. Ba seed sát nhau (76–79%) cho
  thấy đã ở trạng thái cân bằng. Kiểm tra tùy chọn: `--swaps-per-edge 50` mà `moved` không đổi thì
  đó là giới hạn của ràng buộc bậc.
- KG xáo trộn có **cùng cảnh báo validator** với KG thật (số has/lacks mỗi lớp được giữ) và cũng
  **0 cặp trùng hồ sơ** → KG sai "đẹp về hình thức" như KG thật; KG thật thắng thì thắng nhờ nội
  dung. Ghi tỉ lệ dời chỗ và lý do vào luận văn (null model giữ bậc, thận trọng).

### Lưu ý thiết kế (đọc trước khi chạy và khi viết luận văn)

**Bước 1, KG thành ma trận**

- `W[c, attr]`: `has` → +w/2, `lacks` → −w/2, không có cạnh → 0, w ∈ {1, 2}; đúng công thức plan.
- Trường `agree` (1.0 / 0.67) **chưa dùng**; có thể thêm thành ablation (nhân vào `W`).
- `similar` **và** `opposite` được đối xứng hóa (một phía khai báo là đủ). Validator chỉ ép `similar`
  đối xứng, nên không đối xứng hóa thì `G_q` phụ thuộc vào phía nào khai báo.
- KG lỗi (thuộc tính lạ, vừa has vừa lacks, w ∉ {1, 2}, quan hệ trỏ tới lớp lạ hoặc chính nó)
  → báo lỗi ngay, không âm thầm bỏ qua.

**Bước 2, KG xáo trộn (K2)**

- Hoán đổi `has` và `lacks` **riêng từng loại**: giữ số has và số lacks của mỗi hành động và mỗi
  thuộc tính (chặt hơn "giữ bậc" của plan); `w`, `agree` đi theo cạnh; `evidence`, `reviewed` bị bỏ.
- Hoán đổi bỏ qua nếu tạo ô vừa has vừa lacks hoặc phá `excludes` → KG xáo trộn qua validator như KG
  thật, so sánh K2 công bằng về hình thức.
- **Xáo cả `opposite`**, không chỉ `similar` như plan: `opposite` cũng vào `G_q` (láng giềng), không
  xáo thì K2 còn giữ một phần tri thức thật.
- **Không cần tạo lại embedding cho K2**: văn bản không đổi nên `e_c` không đổi; `load_embeddings`
  chấp nhận KG xáo trộn qua `shuffled.source_hash` (khác với điều đã nói trước đó là phải chạy lại
  encoder).

**Bước 3, head đầy đủ**

- **Nhiệt độ của π (phương án A, 2026-10-17):** plan dùng `π = softmax(s)`, nhưng `s = cos/τ` với τ ≈
  0.03 (K1) nên π gần one-hot → `std_π` ≈ 0 với mọi thuộc tính → `d` ≈ 0 → **VERIFY tự tắt đúng ở mẫu
  retrieval tự tin, kể cả khi tự tin sai**. Sửa: `π = softmax(cos / τ_π)`, **τ_π = 0.1 cố định**, tách
  khỏi τ của retrieval. τ_π chọn trước, **không tinh chỉnh trên test** (không có validation).
  Ablation: `uniform_pi: True` (π = 1/k, phương án B).
- **`d_attr` = 0 chính xác** khi mọi ứng viên cùng giá trị thuộc tính: ép tường minh. Không ép thì
  làm tròn float32 để lại `d` ≈ 1e-10 (đóng góp vào `v` ≈ 1e-10, nhỏ hơn logit ~11 bậc): **không
  ảnh hưởng kết quả**, chỉ để tính chất đúng chính xác (test so sánh bằng nhau) và bảng giải thích
  không có số rác. Gradient bị ép về 0 cũng đúng toán học (phương sai = 0 với mọi π). Trường hợp
  sai số bị khuếch đại (mọi thuộc tính đều trùng giữa 10 ứng viên) không xảy ra vì validator báo
  0 cặp lớp trùng hồ sơ. `sqrt(var + ε)` giữ gradient hữu hạn tại var = 0.
- **Ép GT vào Cand chỉ khi truyền `labels`** (lúc train): GT thay ứng viên cuối (điểm thấp nhất).
  Test không truyền nhãn → trần Top1 = Recall@10 (98.32% với K1).
- **FUSE:** 1 token query (`U_z z`), 3 loại node (ứng viên, láng giềng, thuộc tính), mặt nạ
  `key_padding_mask` = ngoài `G_q`. Node thuộc tính chứa `a_attr · u_val` → gradient từ quyết định
  cuối cũng chảy về đầu thuộc tính, không chỉ qua BCE.
- **Lời giải thích:** `contributions [N, k, A] = λ · W[c, attr] · d_attr · (2a_attr − 1)`, tổng theo
  thuộc tính đúng bằng `λ·v_c` (có test).
- **Zero-shot:** không có tham số riêng cho từng lớp; `replace_classes(e_c, kg)` thay bảng lớp mà
  không đụng trọng số (có test với 121 lớp).
- Dropout trên `z` áp dụng **một lần**, dùng chung cho retrieval, đầu thuộc tính và FUSE.
- Cờ ablation trong `head_args`: `use_verify` (K3), `use_fuse` (K4), `uniform_d` (A1-d), `uniform_pi`,
  `learn_lambda` + `lambda_init` (K4'), `k` (A1-k).

**Bước 4, loss + train script** (`train_kg_head.py`, `config/kg/a1.yaml`, test `tests/test_train_kg_head.py`)

- Loss: `CE_cand + 0.5·CE_all + 0.5·BCE_masked + 0.1·mean|f| / sg(mean(|s_c| + |λv|))`.
- **γ = 0.1** (plan mặc định γ = 0): theo lựa chọn 1, chặn `f` lấn át ngay từ đầu. Biến thể γ = 0
  chạy làm đối chứng ở P4.
- **Mẫu số của số hạng γ là stop-gradient** (`sg`), khác plan: không chặn thì optimizer có thể giảm
  tỉ số bằng cách **phóng to `s` (hạ τ)** thay vì thu nhỏ `f`, tức đẩy về đúng rủi ro quá tự tin.
  Có `sg` thì số hạng γ là phạt L1 tương đối trên `f`.
- `BCE_masked`: chỉ trên ô KG biết (`known[y]`); ô không biết không có gradient (có test).
- Train top1 của a1 tính trên logit **đã ép GT vào ứng viên** → lạc quan hơn thực tế; chỉ để theo
  dõi hội tụ, không báo cáo.
- `--kg` ghi đè KG (K2); `kg_metrics.json` ghi `kg_hash` của KG đã dùng để tránh nhầm kết quả.
- Batch eval 1024 (FUSE dựng tensor node [N, 181, 256]).
- Metric mới (plan §5) trên test:
  - `head` Top1/Top5 trên logit (−∞ ngoài ứng viên); `retrieval` Recall@1/5/10/20 trên `s`.
  - λ, τ cuối.
  - Mỗi số hạng `s_c`, `λv`, `f` trên ứng viên: `mean_abs` (độ lớn) **và** `spread` (độ lệch chuẩn
    giữa 10 ứng viên). **Spread mới là thứ đổi được quyết định**: `s` có độ lớn ~30 nhưng phần chung
    cho mọi ứng viên không ảnh hưởng argmax.
  - `flip_without_lambda_v`, `flip_without_f`: % mẫu đổi dự đoán khi bỏ số hạng đó.
  - `attribute_auc`: AUC macro của đầu thuộc tính so với `y_attr[nhãn]` trên ô KG biết. **Chỉ để
    chẩn đoán**: nhãn suy ra từ nhãn lớp, không phải bằng chứng grounding (plan §5).

Lệnh (Colab):

```
python -m pytest tests -q
for s in 0 1 2; do python train_kg_head.py --config config/kg/a1.yaml \
    --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache --embeddings /content/drive/MyDrive/BaoHuy/kg_emb_mpnet.npz \
    --work-dir work_dir/kg/a1_s$s --seed $s; done
for s in 0 1 2; do python train_kg_head.py --config config/kg/a1.yaml \
    --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache --embeddings /content/drive/MyDrive/BaoHuy/kg_emb_mpnet.npz \
    --kg annotations/kg_shuffled/action_kg_shuffle_s$s.json --work-dir work_dir/kg/a1_k2_s$s --seed $s; done
```

K2: KG xáo trộn seed s dùng cùng seed train s (3 lần chạy, mỗi lần một KG khác nhau).

| Cấu hình | Top1 | Top5 | Recall@10 (`s`) | λ | spread `s` / `λv` / `f` | đổi dự đoán khi bỏ `λv` / `f` | AUC thuộc tính |
|---|---|---|---|---|---|---|---|
| K0 | 87.63 | 97.56 | | | | | |
| K1 mpnet | 86.57 ± 0.02 | 96.81 | 98.32 | | | | |
| Full (seed 0 / 1 / 2) | 86.30 / 86.31 / 86.27 | 96.69 / 96.64 / 96.65 | 98.27 / 98.25 / 98.22 | 1.160 / 1.163 / 1.160 | 2.86 / 1.96 / 0.01 | 4.58 / 0.10; 4.64 / 0.04; 4.57 / 0.08 | 0.987 |
| **Full (TB ± std)** | **86.29 ± 0.02** | 96.66 | 98.25 | 1.16 | 2.86 / 1.96 / 0.01 | 4.6 / 0.07 | 0.987 |
| K2 (KG s0 / s1 / s2) | 86.53 / 86.52 / 86.39 | 96.97 / 96.88 / 96.95 | 98.26 / 98.27 / 98.27 | 1.192 / 1.189 / 1.176 | 2.85 / 1.96 / 0.00; 2.86 / 1.91 / 0.00; 2.85 / 1.95 / 0.03 | 4.19 / 0.00; 4.21 / 0.00; 4.30 / 0.31 | 0.992 / 0.991 / 0.991 |
| **K2 (TB ± std)** | **86.48 ± 0.08** | **96.93** | 98.27 | 1.19 | 2.85 / 1.94 / 0.01 | 4.2 / 0.10 | 0.991 |

Full A1 (2026-10-17): retrieval bên trong Full có Top1 86.54 (≈ K1 86.57); mean |·| trên ứng viên
`s` 6.9, `λv` 2.07, `f` 0.01; τ cuối 0.0306; train top1 (view 0) 99.61.

Nhận xét Full:

- **VERIFY đủ mạnh nhưng có hại**: spread `λv` 1.96 so với `s` 2.86, đổi dự đoán 4.6% mẫu (~2 300),
  nhưng Top1 giảm 0.25 điểm so với retrieval của chính nó (86.54 → 86.29): số mẫu bị làm sai nhỉnh
  hơn số được sửa. Lo ngại "VERIFY quá yếu" bị loại.
- **Đúng rủi ro của cache train**: trên train `a` gần hoàn hảo nên `v` chỉ đúng lớp → λ tăng lên 1.16;
  trên test `a` suy từ cùng `z` đã sai ở mẫu khó, nên `v` không thêm thông tin độc lập mà vẫn được
  trao trọng số học từ train. AUC thuộc tính 0.987 chỉ cho thấy `a` "đoán ra lớp" tốt (nhãn suy từ
  nhãn lớp), không phải grounding.
- **FUSE chết** (`|f|` ≈ 0.01, đổi dự đoán ≤ 0.1%): do phạt γ hoặc không còn gì để học trên train;
  phân biệt bằng K4 và γ = 0 (P4).
- Cổng quyết định: Full < K0 1.34 điểm (> 1%) → plan yêu cầu kiểm tra Recall@k (98.25, ổn) và `|f|`
  (không lấn át, ngược lại đã chết) → nguyên nhân nằm ở VERIFY. **Chưa kết luận về KG khi chưa có K2.**
- Lần chạy K2 đầu lỗi: `annotations/kg_shuffled/` mất sau khi runtime Colab khởi động lại; tạo lại
  bằng `shuffle_kg.py` (cùng seed cho cùng KG, `moved` khớp lần đầu).

### Cổng quyết định P3 (2026-10-17)

**Full − K2 = −0.19 điểm Top1 (−0.27 Top5): KG xáo trộn tốt hơn KG thật.** Lần chạy tốt nhất của Full
(86.31) dưới lần chạy kém nhất của K2 (86.39) → không phải nhiễu seed. Retrieval bên trong như nhau
(86.54); VERIFY làm mất 0.25 điểm với KG thật, chỉ 0.06 với KG xáo trộn.

Giả thuyết cơ chế (**đã kiểm chứng, xem dưới**): với cách dùng hiện tại, `W[c]` là một **mã lớp** và VERIFY là
bộ giải mã (`a` dự đoán mã từ `z`, `v` đo độ khớp). KG thật cho các lớp hay nhầm (`put_on`/`take_off`,
`drink_water`/`yawn`) mã gần như trùng (khác 1–2 bit), và ở mẫu khó `a` (suy từ cùng `z` đã sai) sai
đúng các bit đó → VERIFY không tách được, còn củng cố lỗi. KG xáo trộn cho mã gần ngẫu nhiên, các lớp
giống nhau vẫn khác nhiều bit → nguyên lý **mã sửa lỗi (ECOC)**: mã ngẫu nhiên tách lớp tốt hơn mã
ngữ nghĩa. Trên lớp đã thấy, **tính ngữ nghĩa của KG là bất lợi**. Lặp lại kiểu kết quả P1 của dự án
cũ (nhóm ngẫu nhiên ≈ nhóm ngữ nghĩa). Kiểm chứng: khoảng cách Hamming của `W` theo top-25 cặp nhầm
của K0, KG thật so với 3 KG xáo trộn (dự đoán: KG thật nhỏ hơn rõ).

Theo plan §7: Full − K2 < 0.5 → **KG không giúp độ chính xác trên lớp đã thấy**. Còn hai nhánh:

- **K6 (zero-shot)**: nơi ngữ nghĩa phải có lợi (mã ngẫu nhiên của lớp chưa thấy không mang thông
  tin). KG thật ≫ xáo trộn → trình bày A1 như mô hình zero-shot / giải thích, không tuyên bố tăng Top1.
- **K5 (can thiệp thuộc tính)**: lời giải thích có trung thực không.
- Cả K5 và K6 KG thật ≈ xáo trộn → dừng A1, xem lại chất lượng KG (A1-kg).

**Kiểm chứng ECOC** (`tools/kg_code_distance.py`, test `tests/test_kg_code_distance.py`). Dự đoán
ghi **trước khi chạy** (2026-10-17):

- (a) Trên top-25 cặp nhầm của K0, khoảng cách `l1` giữa `W[i]` và `W[j]` của KG thật **nhỏ hơn rõ**
  so với 3 KG xáo trộn; KG xáo trộn gần mức trung bình mọi cặp.
- (b) Spearman(số lần nhầm của cặp, `l1`) trên mọi cặp: KG thật **âm**, KG xáo trộn ≈ 0.
- Nếu (a) và (b) không đúng thì giả thuyết ECOC sai và phải tìm cơ chế khác cho Full < K2.

```
python tools/kg_code_distance.py --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
    --out /content/drive/MyDrive/BaoHuy/kg_code_distance.json
```

Kết quả (2026-10-17): top-25 cặp chiếm 1 944 / 6 299 lỗi của K0. **Cả (a) và (b) đúng.**

| KG | `l1` top-25 | `l1` mọi cặp | top / mọi cặp | `differ` top / mọi | `opposed` top / mọi | top `l1` = 0 | Spearman(nhầm, `l1`) |
|---|---|---|---|---|---|---|---|
| **Thật** | **8.02** | 13.35 | **0.60** | 9.36 / 16.37 | **2.00** / 1.13 | 0 | **−0.252** |
| Xáo trộn s0 | 12.66 | 13.43 | 0.94 | 15.36 / 16.44 | 1.00 / 1.13 | 0 | −0.045 |
| Xáo trộn s1 | 13.06 | 13.44 | 0.97 | 15.64 / 16.45 | 1.16 / 1.13 | 0 | −0.034 |
| Xáo trộn s2 | 12.66 | 13.38 | 0.95 | 15.28 / 16.41 | 1.12 / 1.13 | 0 | −0.014 |

- (a) KG thật: mã của cặp hay nhầm gần hơn mức trung bình ~40%; KG xáo trộn ở mức trung bình. Theo
  từng cặp, 20/25 cặp có `l1` KG thật nhỏ hơn cả 3 KG xáo trộn. Chênh nhiều nhất là các cặp giống
  nhau về ngữ nghĩa: `put_object_into_bag / take_object_out_of_bag` 3.5 so với ~14,
  `put_on_a_shoe / take_off_a_shoe` 3.5 so với ~12, `thumb_up / thumb_down` 3.5 so với ~12,
  `giving_object / exchange_things` 4.5 so với ~14. Ngoại lệ: `reading / writing`,
  `reading / staple_book`, `hush / blow_nose`, `yawn / blow_nose`, `make_victory_sign / snap_fingers`.
- (b) Cặp càng hay nhầm thì mã KG thật càng gần (−0.25); KG xáo trộn ≈ 0 (âm nhẹ do giữ bậc của
  thuộc tính phổ biến).
- `opposed`: KG thật có **2.00** thuộc tính đối lập trên top-25 cặp (so với 1.13 mọi cặp và ~1.1 KG
  xáo trộn): KG **có** thuộc tính phân biệt cho cặp khó (S6 cố ý thêm), nhưng chỉ ~2 bit, và `a` suy
  từ cùng `z` không nhận ra được chúng ở mẫu backbone đã nhầm.

Kết luận:

- **Đã chứng minh:** mã thuộc tính KG thật của cặp hay nhầm gần nhau có hệ thống; KG xáo trộn phá vỡ
  cấu trúc đó → giải thích Full < K2.
- **Chưa chứng minh trực tiếp:** các lần VERIFY đổi đúng thành sai tập trung ở các cặp này (cần phân
  tích theo mẫu trên test, tùy chọn).
- Cho luận văn: KG ngữ nghĩa làm mã kiểm chứng cho phân lớp **bất lợi về bản chất**: ngữ nghĩa giống
  thì mã giống, đúng chỗ cần phân biệt. Muốn KG tăng độ chính xác, thuộc tính phải vừa phân biệt cặp
  khó vừa **nhận diện độc lập với `z`** (đặc trưng theo khớp/thời gian: lựa chọn 2, hướng A5).

Quyết định: **không tinh chỉnh VERIFY trên Top1 test** (rò rỉ; cơ chế ECOC cho thấy vấn đề ở bản chất
KG ngữ nghĩa, không ở siêu tham số). Việc tiếp: kiểm chứng Hamming (rẻ), rồi K6.

**Cần theo dõi khi chạy (bước 4 trở đi)**

- **Độ lớn tương đối:** `s` có thang cos/τ ≈ 33 × cos, còn `|v|` ≤ tổng `|W|` của một hàng (~5–10);
  λ khởi tạo 1. Nếu `|λv|` quá nhỏ so với chênh lệch `s` giữa các ứng viên, VERIFY không đổi được
  dự đoán: in tỉ trọng `|s|`, `|λv|`, `|f|` và % mẫu đổi dự đoán khi bỏ `λv`.
- **Bộ nhớ:** FUSE dựng tensor node [N, 181, 256]; batch eval 4096 ≈ 760 MB → giảm batch eval về
  1024 trong train script.
- **Std giữa seed** dự kiến lớn hơn K1 (thêm attention và đầu thuộc tính); cổng quyết định dùng
  trung bình 3 seed.

## P6: fine-tune end-to-end (phương án A, 2026-10-17)

**Vì sao làm trước K5/K6, dù plan chỉ làm P6 khi P3–P5 tích cực:** kết quả âm của P3 và cơ chế ECOC
là kết luận về **`z` đóng băng** (`a` suy từ `z` đã sai ở mẫu khó). Fine-tune cho gradient của BCE
thuộc tính chảy vào backbone, buộc nó mã hóa cả các bit KG coi là phân biệt (ví dụ hướng chuyển
động put_on/take_off). Đây là giả thuyết khác, có tiền lệ (giám sát phụ bằng mô tả văn bản cho
skeleton, kiểu GAP, ICCV 2023). Phương án B (train từ đầu, ~1–2 ngày/lần) để sau, chỉ nếu A có dấu
hiệu tích cực.

Code:

| File | Nội dung |
|---|---|
| `model/hypergcn_kg.py` | `Model` = Hyper-GCN large + `KGRetrievalHead` dưới tiền tố `kg.`; `z` pool **giống hệt cache** (`pooled_z`, `valid_actor_mask` chuyển từ `tools/cache_features.py` sang đây, cache import lại); nạp `head_weights` từ head đã train trên cache, **từ chối** head train với KG khác |
| `model/kg_retrieval.py` | `masked_bce`, `a1_loss` dùng chung cho `train_kg_head.py` và `main.py` |
| `model/training.py` | `EXTENSION_PREFIXES = ("kg.",)` |
| `main.py` | model có `kg` → `a1_loss` + divergence loss; `--kg-loss`; `--select-best False` (báo cáo epoch cuối) |
| `config/kg_ft/ft_k0.yaml`, `ft_a1.yaml`, `ft_a1_k2.yaml` | cùng công thức; chỉ khác `model`, `kg`, `head_weights` |
| `tests/test_hypergcn_kg.py` | logit hữu hạn, ứng viên xếp đầu; head thấy đúng pooling của cache; checkpoint gốc nạp được; gradient tới l9 và head; head của KG khác bị từ chối |

Công thức (chốt trước): từ `runs-122-120048.pt`, `freeze_until: l7` (train l8, l9, head), SGD
nesterov lr 0.0025, giảm ×0.1 ở epoch 10, 15 epoch, batch 64, wd 0.0004, augmentation như train gốc,
eval ở epoch 0 / 5 / 10 / 15, báo cáo epoch 15. lr 0.0025 = 10 × lr cuối của train gốc
(0.025 × 0.01 = 0.00025) và 1/10 lr ban đầu: đủ để đổi model mà không phá cái đã học. Biến thể mở
toàn bộ backbone (`freeze_until: none`) để sau, nếu cần gradient thuộc tính tới các lớp đầu. Loss của Full-ft / K2-ft: `a1_loss` (α 0.5, β 0.5,
γ 0.1) + divergence loss; K0-ft: CE label smoothing 0.1 + divergence loss như train gốc.

Ba lần chạy và câu hỏi trả lời:

| Lần chạy | Head khởi tạo từ | Trả lời |
|---|---|---|
| K0-ft | (không head) | Riêng fine-tune có đổi Top1 không (mốc) |
| Full-ft | `work_dir/kg/a1_s0/head.pt` (KG thật) | Fine-tune + KG thật |
| K2-ft | `work_dir/kg/a1_k2_s0/head.pt` (KG xáo trộn s0) | Đối chứng: tăng nhờ tri thức hay chỉ nhờ có bài toán phụ (ECOC dự đoán mã ngẫu nhiên còn tốt hơn) |

Kiểm tra trước khi tin kết quả: **epoch 0 phải tái tạo số cache**: K0-ft ≈ 87.64, Full-ft ≈ 86.30,
K2-ft ≈ 86.53 (lệch ≤ ~0.05). Lệch nhiều → pooling hoặc head nạp sai, dừng lại.

Lưu ý:

- **Eval loss của model KG vô nghĩa**: logit ngoài ứng viên là −1e4 (để hữu hạn), CE có label
  smoothing của `main.py` cộng phần đó vào → loss ~ hàng trăm. Chỉ đọc Top1/Top5.
- Train acc của model KG tính trên logit đã ép GT → lạc quan.
- Head chuyển từ Adam (cache) sang SGD lr 0.0025: bước cập nhật head nhỏ hơn; weight decay chạm cả
  `log_tau`, `lambda_raw` nhưng chỉ dịch ~0.015 sau 15 epoch, không đáng kể.
- **1 seed mỗi cấu hình** vì chi phí; nhiễu seed của fine-tune chưa biết. Chênh < ~0.3 điểm giữa
  Full-ft và K2-ft → chạy thêm seed trước khi kết luận.
- Rủi ro tín hiệu yếu: checkpoint đã 99.6% train, CE và BCE gần 0; có thể không đổi gì nhiều.

Dự đoán ghi trước khi chạy: K0-ft trong khoảng 87.64 ± 0.3. Câu hỏi then chốt là Full-ft so với
K2-ft; theo ECOC, nếu fine-tune không giúp `a` nhận ra các bit phân biệt thì Full-ft ≤ K2-ft như P3.

Kết quả (Top1 test, %):

| Lần chạy | Epoch 0 | Epoch 5 | Epoch 10 | **Epoch 15** |
|---|---|---|---|---|
| K0-ft | *chờ* (phải ≈ 87.64) | *chờ* | *chờ* | *chờ* |
| Full-ft | **86.32** (kỳ vọng 86.30: đạt) | 85.29 | 83.47 | **86.62** (Top5 96.78; +0.30 so với epoch 0) |
| K2-ft | *chờ* (phải ≈ 86.53) | *chờ* | *chờ* | *chờ* |

- Full-ft epoch 0 lệch 0.02 điểm so với cache → pooling và nạp head đúng. Dòng `Checkpoint load:
  initialized [kg.*]` chỉ nói checkpoint gốc không có `kg.*`; giá trị thật đến từ `head_weights`.
- Full-ft epoch 5: −1.03 điểm, ở giai đoạn lr cao (0.0025 = 10 × lr hội tụ); chưa kết luận, con số
  có ý nghĩa là epoch 15 (sau khi giảm lr ở epoch 10), so với K0-ft cùng epoch.
- Train acc online 98.5–98.9% (so với 99.5% trên cache): mỗi epoch gặp crop/xoay mới, cache chỉ có 5
  view cố định. `divergence` ≈ 0 (model đã hội tụ). Eval loss ~1 000 vô nghĩa (logit −1e4 + label
  smoothing).
- Thời gian trên L4: ~5.5 phút/epoch train, ~3.2 phút/lần eval → ~1 giờ 35 phút/lần chạy.
- Full-ft (2026-10-17): giảm trong giai đoạn lr 0.0025 (thấp nhất 83.47 ở epoch 10), tăng lại sau khi
  giảm lr → 86.62 ở epoch 15 (+0.30). Cao hơn Full trên cache (86.29) và K1 (86.57), vẫn dưới K0
  (87.64). **Chưa quy +0.30 cho KG**: có thể chỉ do train thêm với augmentation (K0-ft trả lời) hoặc
  do có bài toán phụ bất kể ngữ nghĩa (K2-ft trả lời). Train acc cuối 99.61% (đã ép GT).

## Thử lớp khác: l6, l7, l8 (2026-10-17)

Câu hỏi: với **cùng cách pool** (mean(T, V) mỗi actor, max qua actor thật), `z` từ lớp sớm hơn có làm
VERIFY có ích hơn không? Lớp sớm hơn ít "hội tụ về lớp" hơn, nên `a` có thể ít bắt chước quyết
định cuối hơn. Đây là phép thử **độ sâu**, không phải phép thử **giữ chiều thời gian**.

Code: `tools/cache_features.py --layers` (forward hook, một lần chạy backbone cho mọi lớp; test
`tests/test_cache_features.py::EarlierLayerTests`); `train_kg_head.py --cache-name`. Số kênh `z`:
l6 256, l7 512, l8 512. File: `ntu120_xsub_<layer>_{train,test}.npz` (`fc_logits` vẫn từ l9).

Dự đoán ghi **trước khi chạy**:

- K1 Top1 giảm theo độ nông: l8 ≈ l9 (≤ 0.5 điểm), l7 thấp hơn rõ, l6 thấp hơn nhiều (lớp sớm khó
  tách lớp tuyến tính hơn); Recall@10 giảm theo.
- Full − K2 vẫn ≤ 0 ở mọi lớp: pool theo T xóa thứ tự thời gian (put_on / take_off) ở mọi độ sâu,
  nên các bit phân biệt vẫn không đọc được. Nếu Full − K2 > 0 rõ ở một lớp sớm thì dự đoán này sai
  và độ sâu có vai trò.

Lệnh (Colab). **Không chạy cache cùng lúc với fine-tune P6 trên cùng GPU** (chậm cả hai):

```
python tools/cache_features.py --weights /content/large/ntu120/xsub/joint/runs-122-120048.pt \
    --out-dir /content/drive/MyDrive/BaoHuy/kg_cache --layers l6 l7 l8
for L in l6 l7 l8; do
  for cfg in k1 a1; do
    python train_kg_head.py --config config/kg/$cfg.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
        --cache-name ntu120_xsub_$L --embeddings /content/drive/MyDrive/BaoHuy/kg_emb_mpnet.npz \
        --work-dir work_dir/kg/${cfg}_${L}_s0 --seed 0
  done
  python train_kg_head.py --config config/kg/a1.yaml --cache-dir /content/drive/MyDrive/BaoHuy/kg_cache \
      --cache-name ntu120_xsub_$L --embeddings /content/drive/MyDrive/BaoHuy/kg_emb_mpnet.npz \
      --kg annotations/kg_shuffled/action_kg_shuffle_s0.json --work-dir work_dir/kg/a1_k2_${L}_s0 --seed 0
done
```

Dung lượng thêm trên Drive: ~1.5 GB (train l7/l8 ~475 MB mỗi lớp, l6 ~310 MB, cộng test).

| Lớp | K1 Top1 | K1 Recall@10 | Full Top1 | Retrieval trong Full | K2 Top1 | Full − K2 | head − retrieval (Full / K2) | đổi dự đoán khi bỏ `λv` / `f` (Full) | AUC thuộc tính (Full / K2) |
|---|---|---|---|---|---|---|---|---|---|
| l9 (seed 0) | 86.59 | 98.33 | 86.30 | 86.54 | 86.53 | −0.23 | −0.24 / −0.05 | 4.58 / 0.10 | 0.987 / 0.992 |
| l8 | 86.29 | 98.58 | 86.17 | 86.32 | 86.34 | −0.17 | −0.15 / +0.09 | 6.33 / 2.13 | 0.981 / 0.986 |
| l7 | 84.07 | 98.63 | 84.48 | 84.15 | 84.63 | −0.15 | +0.33 / +0.46 | 7.23 / 5.07 | 0.922 / 0.911 |
| l6 | 63.99 | 94.92 | 66.11 | 64.54 | 66.75 | −0.64 | +1.57 / +2.65 | 21.27 / 18.02 | 0.791 / 0.770 |

1 seed mỗi ô (std seed ở l9 ≤ 0.08): chênh < ~0.2 điểm coi là bằng nhau. λ cuối Full / K2: l8 1.45 /
1.58, l7 1.87 / 2.09, l6 1.82 / 2.01.

Kết quả (2026-10-18), **cả hai dự đoán đúng**:

- K1 giảm theo độ nông: l8 ≈ l9 (−0.3), l7 −2.5, l6 −22.6 (Recall@10 94.9%, dưới tiêu chí 98%):
  đặc trưng l6 sau pool (T, V) gần như không tách lớp tuyến tính.
- **Full − K2 ≤ 0 ở mọi lớp** (−0.15 đến −0.64): độ sâu không làm tri thức KG có ích.
- Quan sát mới: head đầy đủ hơn retrieval càng nhiều khi lớp càng nông (l6 +1.6 đến +2.7, l7 +0.3 đến
  +0.5, l8/l9 ≈ 0), FUSE "sống lại" ở lớp nông (bỏ `f` đổi 18% dự đoán ở l6), nhưng **KG xáo trộn lợi
  bằng hoặc hơn KG thật** ở mọi lớp. Phần tăng là do **năng lực mô hình** (đầu thuộc tính +
  attention thêm phi tuyến cho đặc trưng yếu), không do tri thức; mã ngẫu nhiên giúp hơn mã ngữ
  nghĩa đúng như ECOC.

Kết luận: **đóng hướng "cache lớp khác" với cùng cách pool.** Không lớp nào cho Full > K2; độ chính
xác tốt nhất vẫn ở l9 (~86.5 < K0 87.63). Củng cố chẩn đoán: vấn đề là bằng chứng thuộc tính không
độc lập với quyết định và pool theo T xóa thông tin thời gian, không phải độ sâu. Hướng còn lại: giữ
chiều thời gian, hoặc thuộc tính hình học tính từ tọa độ. Dùng được làm ablation trong luận văn:
"head A1 chỉ có ích khi đặc trưng yếu, và khi đó KG ngẫu nhiên cũng có ích như vậy".

## Decoder thuộc tính trên feature map trước pool (lựa chọn 2, 2026-10-18)

**Lý do:** ở P3 và ở các lớp l6–l9, `a = sigmoid(W_a z)` đọc cùng `z` đã pool theo (T, V) với quyết
định lớp: bằng chứng thuộc tính không độc lập, và pool theo T xóa thứ tự thời gian. Decoder cho
mỗi thuộc tính tự tìm bằng chứng trên token (actor, frame, khớp) của map l9 `[N, 2, 512, 16, 25]`.
`z` vẫn giữ cho RETRIEVE và query của FUSE (so với embedding text cần một vector toàn cục).

| File | Nội dung |
|---|---|
| `model/attribute_decoder.py` | `AttributeDecoder`: token = LN(Linear) + embedding thời gian / khớp / actor; query = Linear(embedding text thuộc tính); 2 lớp cross-attention + FFN; thuộc tính không attend lẫn nhau; lớp ra khởi tạo 0 → `a` = 0.5, VERIFY = 0 lúc đầu |
| `model/kg.py` | `attribute_joint_mask`: `body_parts` của KG → khớp NTU; mặt nạ attention; `mirror` mở cả bên kia (KG ghi `hand_R` cho "tay làm động tác") |
| `model/kg_retrieval.py` | `attribute_source: decoder` (mặc định `z`, cache không đổi) |
| `model/hypergcn_kg.py` | truyền map trước pool + `valid` vào head; nạp head cache (bỏ `attribute.*`, decoder mới) |
| `main.py` | `--lr-scale` theo tiền tố tên tham số |
| `config/kg_ft/ft_a1_dec.yaml`, `ft_a1_dec_k2.yaml` | KG thật / KG xáo trộn s0; chỉ khác `kg`, `head_weights`, `work_dir` |
| `tests/test_attribute_decoder.py`, `tests/test_hypergcn_kg.py` | mặt nạ khớp và actor được tôn trọng, thứ tự thời gian có ảnh hưởng, `a` = 0.5 lúc đầu, head cache nạp được, gradient tới decoder; `pytest tests` 81/81 |

Công thức (chốt trước): như P6 nhưng **đóng băng toàn bộ backbone** (`freeze_until: l9`), nên
khác biệt so với head cache chỉ là decoder; lr decoder ×10 (0.025, lr gốc của Hyper-GCN vì train
từ đầu), phần còn lại của head 0.0025; 15 epoch, giảm ×0.1 ở epoch 10; loss `a1_loss` (α 0.5,
β 0.5, γ 0.1); báo cáo epoch 15. Decoder 1.4 M tham số; mỗi thuộc tính thấy trung bình 11.4/25
khớp.

Dự đoán ghi **trước khi chạy**:

- Epoch 0 ≈ head cache bỏ VERIFY (`a` = 0.5): Full ≈ 86.5, K2 ≈ 86.5 (gần retrieval bên trong).
- Nếu decoder đọc được bằng chứng độc lập: `attribute_auc` trên test tăng so với 0.987 **và**
  Full-dec > K2-dec. Nếu Full-dec ≤ K2-dec thì decoder chỉ thêm năng lực (như l6), tri thức KG
  vẫn không giúp; khi đó cần nhãn yếu hình học (khoảng cách tay–miệng, ...) để phá đường tắt
  "nhận lớp → tra mã".
- Rủi ro: train ~99% nên BCE thuộc tính (nhãn suy từ nhãn lớp) vẫn cho phép đường tắt; mặt nạ
  `body_parts` chỉ làm nó khó hơn.

Lệnh (Colab):

```
python -m pytest tests -q
python main.py --config config/kg_ft/ft_a1_dec.yaml
python main.py --config config/kg_ft/ft_a1_dec_k2.yaml
```

| Lần chạy | Epoch 0 | Epoch 5 | Epoch 10 | **Epoch 15** |
|---|---|---|---|---|
| Full-dec | *chờ* | *chờ* | *chờ* | *chờ* |
| K2-dec | *chờ* | *chờ* | *chờ* | *chờ* |

## Sai khác so với kế hoạch

| Kế hoạch | Thực tế | Lý do |
|---|---|---|
| `π = softmax(s_c)` | `π = softmax(cos_c / τ_π)`, τ_π = 0.1 cố định | τ ≈ 0.03 làm π one-hot, VERIFY tự tắt (mục P3) |
| Xáo `has/lacks` giữ bậc, xáo `similar` | Xáo riêng has và lacks; xáo cả `opposite`; tôn trọng `excludes` | Đối chứng chặt hơn; `opposite` cũng là tri thức trong `G_q` |
| K2 cần embedding riêng | Dùng lại embedding KG thật (`source_hash`) | Văn bản không đổi |
| γ = 0, mẫu số `mean(|s| + |λv|)` | γ = 0.1, mẫu số stop-gradient | Lựa chọn 1; tránh giảm phạt bằng cách hạ τ |
| P6 chỉ khi P3–P5 tích cực | P6 (phương án A) làm ngay sau P3 âm, trước K5/K6 | P3 âm là kết luận về `z` đóng băng; fine-tune kiểm tra giả thuyết khác |
| `main.py` chọn checkpoint tốt nhất theo test | `--select-best False`: epoch cuối | Tránh chọn trên test |
| Cache trên Kaggle | Colab, GPU L4 | Môi trường sẵn có |
| Mỗi epoch lấy một view | Mỗi mẫu lấy ngẫu nhiên một view | Trộn augmentation tốt hơn, cùng chi phí |
| Embedding một câu mỗi node | Trung bình các mô tả (prompt ensemble); bản một câu giữ ở `action_text` | Ổn định hơn với cách diễn đạt |
| Encoder mpnet, CLIP | Thêm bge-large để so sánh | Kiểm tra encoder lớn hơn có tốt hơn không: không |

## Chưa commit

`tools/build_kg_embeddings.py`, `model/kg.py`, `model/kg_retrieval.py`, `train_kg_head.py`,
`config/kg/k1.yaml`, `config/kg/a1.yaml`, `tools/shuffle_kg.py`, `tests/test_build_kg_embeddings.py`,
`tests/test_kg_retrieval.py`, `tests/test_kg.py`, `tests/test_shuffle_kg.py`, `tests/test_train_kg_head.py`,
`tools/kg_code_distance.py`, `tests/test_kg_code_distance.py`, `model/hypergcn_kg.py`, `model/training.py`,
`main.py`, `tools/cache_features.py`, `config/kg_ft/*.yaml`, `tests/test_hypergcn_kg.py`,
`annotations/kg_emb_*.npz`, `model/attribute_decoder.py`, `config/kg_ft/ft_a1_dec*.yaml`,
`tests/test_attribute_decoder.py`, file này.
