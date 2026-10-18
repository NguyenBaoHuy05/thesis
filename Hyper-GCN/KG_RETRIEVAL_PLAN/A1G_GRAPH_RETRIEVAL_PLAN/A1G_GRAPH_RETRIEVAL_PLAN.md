# A1-G: `z` truy xuất đồ thị → kiểm chứng bằng vị từ đo được

> Trạng thái: **G0 đã viết code** (2026-10-18), chờ chạy test torch và G1 trên Colab.
> Tiến độ và số liệu: [A1G_PROGRESS.md](A1G_PROGRESS.md).
> Kế thừa: [A1_RETRIEVE_VERIFY_FUSE_PLAN.md](../A1_RETRIEVE_VERIFY_FUSE_PLAN/A1_RETRIEVE_VERIFY_FUSE_PLAN.md),
> số liệu A1: [A1_PROGRESS.md](../A1_RETRIEVE_VERIFY_FUSE_PLAN/A1_PROGRESS.md).
> Sơ đồ: [drawio/architecture_rvf_hgcn.drawio](../../drawio/architecture_rvf_hgcn.drawio), trang **A1-G Graph Retrieval**
> (trang RVF-HGCN cũ giữ nguyên làm hồ sơ A1).
>
> Mục tiêu luận văn: **neuro-symbolic cho nhận dạng hành động trên khung xương, trên các lớp đã thấy.**
> Zero-shot **không** phải mục tiêu: mọi lựa chọn chỉ để phục vụ zero-shot (embedding text cố định, "không có
> tham số riêng cho lớp", `replace_classes`, K6) bị bỏ.

## 0. Vì sao đổi từ A1 sang A1-G

Kết luận từ A1 (số liệu ở A1_PROGRESS.md):

| # | Quan sát ở A1 | Hệ quả cho A1-G |
|---|---|---|
| 1 | RETRIEVE bằng embedding text tốn ~1 điểm (K1 86.57 so với K0 87.63) và đặt lớp `opposite` gần nhất trong không gian text | `z` truy xuất **node của đồ thị**; text chỉ còn là ablation |
| 2 | Cạnh KG chỉ là bảng tra `W`: không ảnh hưởng retrieval, không lan truyền; `similar` và `opposite` bị gộp; FUSE chết nên `similar`/`opposite` gần như không tác động tới đầu ra | Cạnh **tạo ra** vector node qua graph encoder theo từng loại quan hệ; tách `similar` / `opposite` |
| 3 | `a = σ(W_a z)` học bằng nhãn suy từ nhãn lớp → đường tắt "nhận lớp → tra mã"; KG thật thua KG xáo trộn (ECOC, đã kiểm chứng: `l1` cặp nhầm 8.02 so với ~12.8) | Bằng chứng thuộc tính phải **đo từ tọa độ khớp**, độc lập với `z` |
| 4 | `d_attr` coi ô unknown (`W = 0`) là giá trị giữa → đo độ dày chú thích (16% ô được ghi), không đo khác biệt | VERIFY chỉ dùng thuộc tính **đối lập đã biết** giữa các ứng viên |
| 5 | `v_c` không chuẩn hóa theo số cạnh → lớp nhiều cạnh có biên độ lớn hơn | Chuẩn hóa theo số thuộc tính dùng được của lớp |
| 6 | Head học trên đặc trưng train mà backbone đã ~99.6% → λ học tin `v` trong chế độ không có lỗi | Biến thể λ cố định; theo dõi; phương án tách subject (§8) |
| 7 | Cặp khó nhất **không có thuộc tính đối lập** trong KG (xem dưới) | Mở rộng KG cho cặp khó trước khi kỳ vọng VERIFY giúp |

Kiểm tra trên `annotations/action_kg.json` (2026-10-18), thuộc tính có `has` ở lớp này và `lacks` ở lớp kia:

| Cặp | Thuộc tính đối lập |
|---|---|
| `put_on_jacket` / `take_off_jacket` | **không có** (khác nhau chỉ ở ô một bên ghi, một bên unknown) |
| `put_on_a_shoe` / `take_off_a_shoe` | **không có** |
| `drink_water` / `yawn` | **không có** |
| `thumb_up` / `thumb_down` | `hand_upright` |
| `sit_down` / `stand_up` | `hips_lower`, `hips_rise` |
| `reading` / `writing` | 4 thuộc tính |

KG thiếu hẳn thuộc tính **hướng thời gian** (mặc vào so với cởi ra: tay đi từ hông lên vai hay ngược lại).
Không bổ sung thì VERIFY không thể tách đúng những cặp chiếm nhiều lỗi nhất, dù công thức đúng.

## 1. Câu hỏi nghiên cứu

| | Câu hỏi | Kiểm chứng | Ablation |
|---|---|---|---|
| Q1 | Truy xuất bằng đồ thị có giữ độ chính xác của backbone? | Recall@10 ≈ K0, Top1 ≥ K1 (text) | G1, K1, K0 |
| Q2 | Cấu trúc đồ thị có đóng góp vào retrieval? | Graph encoder so với node không có cạnh | G1 vs G1-0 |
| Q3 | Tri thức KG giúp sửa lỗi của neural? | Full-pred > K2-pred và Full-pred > G1 | **K2-pred** |
| Q4 | Bằng chứng phải độc lập với `z`? | Full-pred so với Full-z | Full-z |
| Q5 | Giải thích có trung thực? | Can thiệp vị từ → dự đoán đổi theo luật | K5 |

Không còn: K6 (zero-shot), A1-enc (so encoder text), FUSE như thành phần chính, decoder thuộc tính.

## 2. Kiến trúc

```
skeleton X ─► Hyper-GCN large (ĐÓNG BĂNG, tới l9) ─► mean(T,V), max qua actor hợp lệ ─► z ∈ R^512
             (dùng lại cache P0: ntu120_xsub_{train,test}.npz)

GRAPH ENCODER (không phụ thuộc mẫu; chạy một lần mỗi bước trên 181 node)
  node:     120 hành động + 61 thuộc tính, đặc trưng ban đầu x_i ∈ R^256 HỌC ĐƯỢC
            (ablation: khởi tạo từ embedding text, sau đó vẫn học)
  quan hệ:  has, lacks (hành động → thuộc tính, trọng số w ∈ {1,2}),
            has⁻¹, lacks⁻¹ (thuộc tính → hành động), similar, opposite (hành động ↔ hành động)
  R-GCN, L = 2 lớp:
            h_i^{l+1} = ReLU( W_0^l h_i^l + Σ_r Σ_{j ∈ N_r(i)} (w_ij / c_{i,r}) W_r^l h_j^l ),  c_{i,r} = Σ_j w_ij
  g_c     = x_c + h_c^L                 (residual: thành phần tự do chống sụp lớp gần nhau, §2.2)

RETRIEVE
  s_c     = cos(P z, g_c) / τ           P: Linear 512 → 256, τ học được trong [0.01, 1]
  Cand    = top-k(s), k = 10            (train: ép GT vào Cand như A1)

G_q = Cand ∪ {attr có cạnh has/lacks tới Cand} ∪ láng giềng similar / opposite (giữ 2 loại riêng)

BẰNG CHỨNG THUỘC TÍNH
  G2 (chẩn đoán):  a_attr = σ(W_a z)                    ← tái hiện đường tắt ECOC trên retrieval mới
  G4 (chính):      p_attr = σ((φ_attr(X) − θ_attr) / s_attr)
                   φ_attr: hàm hình học 1 chiều của tọa độ khớp (khoảng cách, góc, vận tốc, thứ tự thời gian)

VERIFY (sửa so với A1)
  O_attr  = 1[ ∃c ∈ Cand: has(c,attr) ] · 1[ ∃c' ∈ Cand: lacks(c',attr) ]     chỉ ô đã biết
  n_c     = Σ_attr O_attr · known[c,attr]
  v_c     = Σ_attr O_attr · W[c,attr] · (2 p_attr − 1) / max(1, n_c)

logit_c = s_c + λ · v_c     c ∈ Cand;   −∞ (−1e4 trong main.py) ngoài Cand;   λ = softplus(λ̃), khởi tạo 1
```

### 2.1 Quyết định thiết kế

- **Retrieval cần recall, VERIFY cần phân biệt; tách hẳn hai việc.** Graph encoder làm mượt theo `similar` là
  *tốt* cho retrieval: cặp gây nhầm cùng vào top-k. Phân biệt là việc của VERIFY trên vị từ đo được. Không thêm
  lực đẩy `opposite` vào retrieval.
- **Vị từ có năng lực thấp có chủ đích.** Mỗi `p_attr` chỉ có 2 tham số (`θ`, `s`) trên một đại lượng hình học 1
  chiều, nên không thể học "nhận ra lớp" kể cả khi giám sát bằng nhãn suy từ lớp. Đây là cách phá đường tắt ở #3.
- **VERIFY thưa là đúng.** Không có thuộc tính đối lập giữa các ứng viên → `v = 0` → quyết định quay về `s`.
  Symbolic chỉ lên tiếng khi KG có luật khẳng định mâu thuẫn; lời giải thích cũng sạch hơn.
- **Bỏ FUSE khỏi mô hình chính.** A1 cho thấy `|f|` ≈ 0.01. Giữ cờ `use_fuse` cho ablation.
- **Phân rã logit chính xác**: `λ v_c` tách theo từng thuộc tính
  `λ · O_attr · W[c,attr] · (2p_attr − 1) / n_c` → lời giải thích trực tiếp, không post-hoc.
- KG nhỏ: mọi thứ là ma trận dày (`[181, 181]` theo từng quan hệ); không cần thư viện đồ thị.

### 2.2 Ba chế độ hỏng của graph retrieval (theo dõi từ G1)

| Chế độ | Dấu hiệu | Xử lý |
|---|---|---|
| Sụp lớp: hàng `W` gần trùng → `g_c` gần trùng → retrieval không tách được, Recall@1 giảm | `cos(g_i, g_j)` trên top-25 cặp nhầm của K0 ≫ trung bình | residual `x_c` (mặc định); nếu vẫn sụp: giảm `L`, dropout cạnh |
| Over-smoothing | `cos` trung bình mọi cặp `g` tăng theo `L` | `L ≤ 2`, so sánh `L = 0, 1, 2` |
| Đồ thị chỉ trang trí | G1 ≈ G1-0 (node tự do, không cạnh) | Ghi nhận trung thực: cấu trúc không giúp retrieval; đóng góp của KG nằm ở VERIFY |

## 3. Vị từ hình học (grounding)

Nguyên tắc: tính từ **tọa độ khớp đã qua tiền xử lý của feeder**, trong hệ quy chiếu thân người (gốc ở
`spine_base`, chuẩn hóa theo chiều dài thân `spine_base–spine_shoulder`), không dùng `z`, không dùng nhãn khi
định nghĩa `φ`.

Ví dụ (chốt danh sách ở G3, tên thuộc tính theo KG; ★ = thuộc tính mới cần thêm vào KG):

| Thuộc tính | `φ` (khớp NTU 0-based) | Cặp khó liên quan |
|---|---|---|
| `hand_at_mouth` | min theo thời gian của `min(‖hand_L − head‖, ‖hand_R − head‖)` (7/11 so với 3), có điều kiện dừng ≥ vài frame | `drink_water` / `yawn`, `hush` / `blow_nose` |
| `both_hands_at_face` | khoảng cách lớn hơn trong hai tay tới đầu, lấy min theo thời gian | `drink_water` / `yawn` |
| `head_tilted_back` | góc `neck → head` so với trục thân, lấy max | `drink_water` / `yawn` |
| `hips_lower` / `hips_rise` | độ cao hông cuối − đầu chuỗi (âm / dương) | `sit_down` / `stand_up` |
| `hand_upright` | hướng `hand → hand_tip` so với trục dọc (21–24 nhiễu, xem §8) | `thumb_up` / `thumb_down` |
| ★ `hands_rise_to_shoulders_over_time` | hệ số góc độ cao tay theo thời gian (đầu → cuối) | `put_on_jacket` / `take_off_jacket` |
| ★ `foot_hand_contact_then_release` | thứ tự: tay gần bàn chân rồi rời ra so với ngược lại | `put_on_a_shoe` / `take_off_a_shoe` |
| `hands_come_together` / `hands_move_apart` | khoảng cách hai tay cuối − đầu | `put_object_into_bag` / `take_object_out_of_bag` |

Cách học `θ`, `s`: BCE trên các ô KG đã biết (nhãn suy từ lớp, chỉ trên tập train), **hoặc** chốt bằng tay từ
phân phối `φ` trên train. Không chạm tập test. Kiểm tra mỗi vị từ trước khi dùng: AUC trên train theo nhãn KG
(chẩn đoán) + xem tay ≥ 10 mẫu mỗi vị từ.

Dữ liệu: `φ` tính **một lần** và lưu cạnh cache (`tools/cache_predicates.py`), cùng thứ tự mẫu với cache đặc
trưng (train script từ chối nếu nhãn lệch). **Mỗi mẫu một view tất định** (transform của test: cắt giữa 0.95,
không xoay) cho cả train lẫn test, thay vì 5 view: các crop ngẫu nhiên của cache đặc trưng phụ thuộc cách chia
worker của DataLoader nên không tái tạo chắc chắn được, và `φ` là bằng chứng về chính video đó, không cần
cùng crop với `z`. Vị từ tính trong hệ quy chiếu thân nên bất biến với xoay; điều này được kiểm bằng unit test
(`tests/test_predicates.py::test_rotation_invariance`), không cần đo trên dữ liệu.

## 4. Thay đổi KG

1. **Danh sách cặp khó lấy từ chính KG** (`tools/hard_pairs.py`): mọi cặp có quan hệ `similar` hoặc `opposite`
   (LLM sinh từ định nghĩa hành động, trước khi thấy lỗi nào của mô hình) →
   `annotations/kg_work/hard_pairs.json`, kèm thuộc tính đối lập, thuộc tính đối lập có vị từ, và thuộc tính
   một phía (một lớp ghi, lớp kia unknown). **Không chọn cặp theo ma trận nhầm trên test** (tránh rò rỉ);
   top-25 cặp nhầm của K0 trên test chỉ dùng để *kiểm tra* độ bao phủ (`--cache-dir`).
   Kết quả trên KG hiện tại (2026-10-18): 280 cặp, **46 cặp không có thuộc tính đối lập nào**, 85 cặp không
   có thuộc tính đối lập nào có vị từ; ba cặp `opposite` put_on/take_off (jacket, shoe, bag) đều không có.
2. **Lấp ô unknown cho cặp khó (thế giới đóng cục bộ)**: với mỗi cặp, điền has/lacks cho các thuộc tính đang
   một bên ghi, một bên unknown (như `both_arms_symmetric`, `hands_above_head` ở cặp jacket).
3. **Thêm thuộc tính hướng thời gian** (★ ở §3) cho các cặp không có thuộc tính đối lập nào.
4. **Duyệt tay** các cạnh của mục 2–3 (`annotations/kg_review/edges.csv`, `human_label`). Đây đồng thời là phần
   duyệt tay ≥ 40 lớp còn nợ ở P1.
5. Chạy lại `tools/validate_kg.py`, `tools/shuffle_kg.py` (KG xáo trộn mới cho K2). Giữ KG trước khi sửa cho
   ablation A1G-kg.
6. Ràng buộc chọn thuộc tính để tránh rò rỉ: thuộc tính mới được chọn theo **cặp của KG và định nghĩa hành
   động** (mục 1), không theo mẫu sai hay ma trận nhầm trên test. Báo cáo trong luận văn bao nhiêu trong top-25
   cặp nhầm thực tế của K0 được nhóm cặp KG bao phủ.

## 5. Mã nguồn (G0 đã viết, 2026-10-18)

| File | Nội dung |
|---|---|
| `model/kg.py` | `RELATIONS` (has, has_inv, lacks, lacks_inv, similar, opposite), `relation_adjacency` → `[6, 181, 181]` chuẩn hóa theo hàng, has/lacks mang trọng số `w`, `similar`/`opposite` tách riêng |
| `model/graph_encoder.py` (mới) | `RGCNEncoder` (node học được, `init: learned` mặc định, `text` cho ablation; residual `g = x + h^L`; `layers: 0` = G1-0); `GraphRetrievalScorer` (`s_c = cos(Pz, g_c)/τ`) |
| `model/kg_retrieval.py` | `KGRetrievalHead`: `retrieval_source: graph` + `graph_args`; `verify_mode: std | masked_std | opposed`; `normalize_verify` (chia `n_c`); `attribute_source: predicate` + `PredicateEvidence` (θ, s > 0 mỗi vị từ; thuộc tính không có vị từ hoặc vị từ không hợp lệ → `a = 0.5`, bị loại khỏi `d`, `n_c`, BCE). Mặc định giữ hành vi A1; `head.pt` của A1 vẫn nạp được |
| `model/predicates.py` (mới, chỉ numpy) | 39 vị từ (37 trùng tên thuộc tính KG + 2 thuộc tính hướng thời gian chưa có trong KG); hệ quy chiếu thân; max qua actor thật; vị từ hai người không hợp lệ khi không có người thứ hai |
| `tools/cache_predicates.py` (mới) | `φ` cho test và train (một view tất định), `ntu120_xsub_pred_{train,test}.npz` + báo cáo AUC theo KG từng vị từ (đánh dấu `check sign` khi AUC < 0.4, `weak` khi < 0.75) |
| `tools/hard_pairs.py` (mới, không cần torch) | cặp `similar`/`opposite` của KG + thuộc tính đối lập / có vị từ / một phía; `--cache-dir` chỉ để kiểm tra độ bao phủ top-25 cặp nhầm của K0 |
| `train_kg_head.py` | head `g1`; khối `predicates:` trong config; embedding text chỉ nạp khi cần; `--head-arg KEY=VALUE` cho ablation; metric mới: `verify_coverage`, `vs_retrieval` / `vs_k0` (sửa / phá), AUC chỉ trên thuộc tính có bằng chứng, θ / s của vị từ |
| `config/kg/g1.yaml`, `g1_0.yaml`, `g_full_z.yaml`, `g_full_pred.yaml` | cùng công thức tối ưu với K1/A1 (Adam 1e-3, cosine, 30 epoch, batch 512) |
| `tests/test_graph_encoder.py`, `tests/test_kg_retrieval_graph.py`, `tests/test_predicates.py` | quan hệ tách loại, cạnh đổi thì `g` đổi, `layers 0` = đồng nhất; `opposed` bỏ qua unknown; chuẩn hóa `n_c`; phân rã `λv`; vị từ không có bằng chứng không vào VERIFY; BCE chỉ trên cột có vị từ; khóa `state_dict` của head A1 không đổi; vị từ đúng dấu trên khung xương tổng hợp và bất biến với xoay |
| `model/hypergcn_kg.py` | chưa đổi: tính `φ` trong forward chỉ cần ở G6 |

Ablation không cần config riêng: `--kg <KG xáo trộn>` (K2), `--head-arg layers=0` (G1-0 từ `g1.yaml`),
`--head-arg verify_mode=std`, `--head-arg normalize_verify=False`, `--head-arg learn_lambda=False`,
`--head-arg graph_args.layers=1`.

## 6. Đánh giá

Trên test X-Sub, mỗi cấu hình 3 seed, báo cáo trung bình ± std:

- **Top1 / Top5** của head; **Recall@1/5/10/20** của `s`.
- **Sửa / phá so với retrieval và so với K0**: số mẫu sai → đúng, đúng → sai. Đây là con số chính của VERIFY.
- **Độ phủ**: % mẫu có `v ≠ 0`; Top1 riêng trên nhóm `v ≠ 0`.
- **Theo cặp khó**: độ chính xác nhị phân trên mẫu thuộc top-25 cặp (trần tăng tối đa: 1 944 lỗi ≈ 3.8 điểm).
- Độ lớn và **spread** của `s`, `λv` trên ứng viên; % đổi dự đoán khi bỏ `λv`; λ, τ cuối.
- Chẩn đoán graph: `cos(g_i, g_j)` top-25 cặp so với mọi cặp; theo `L`.
- Vị từ: AUC theo nhãn KG trên train và test (chẩn đoán, không phải bằng chứng grounding); độ lệch giữa 5 view.
- K5: can thiệp `p_attr ← 1 − p_attr` trên thuộc tính đóng góp lớn nhất → % mẫu đổi dự đoán sang lớp mà luật KG
  dự đoán.

## 7. Ablation

| Mã | Cấu hình | Trả lời |
|---|---|---|
| K0 | backbone + FC (cache) | mốc 87.63 |
| K1 | retrieval text (A1, đã có 86.57) | text so với đồ thị |
| **G1** | retrieval đồ thị, R-GCN 2 lớp, không VERIFY | Q1 |
| G1-0 | node tự do, không cạnh (`L = 0`) | Q2: cấu trúc có giúp retrieval |
| G1-K2 | G1 trên KG xáo trộn | Q2 |
| G1-L | `L = 1, 2, 3` | over-smoothing |
| G1-init | khởi tạo node từ text | học từ đầu so với khởi tạo text |
| Full-z | G1 + VERIFY (`opposed`) với `a = σ(W_a z)` | Q4: tái hiện ECOC |
| **Full-pred** | G1 + VERIFY (`opposed`) với vị từ `p` | Q3, kết quả chính |
| **K2-pred** | Full-pred trên KG xáo trộn | Q3: tri thức hay chỉ năng lực |
| V-std / V-mstd | `verify_mode` std (A1) / masked std | sửa unknown có quan trọng |
| V-nonorm | bỏ chia `n_c` | chuẩn hóa có quan trọng |
| λ-fix | λ = 1 cố định | λ học trên train có hại |
| A1G-kg | KG trước khi sửa §4 | đóng góp của bước mở rộng KG |
| k | 5 / 10 / 20 | |

## 8. Lộ trình và cổng quyết định

Dự đoán của mỗi pha ghi vào A1G_PROGRESS.md **trước khi chạy**. Công thức train chốt trong config trước, không
tinh chỉnh trên test (không có validation).

| Pha | Việc | Tiêu chí / cổng |
|---|---|---|
| G0 | `kg.py` quan hệ, `graph_encoder.py`, `verify_mode`, test CPU | `pytest tests` xanh |
| G1 | G1, G1-0, G1-K2, G1-L trên cache l9 | Recall@10 ≥ 98.5 (K0 98.72); Top1 ≥ K1. G1 ≈ G1-0 → ghi nhận, vẫn tiếp tục |
| G2 | Full-z, K2-z với VERIFY `opposed` | Chẩn đoán. Dự đoán: Full-z ≤ K2-z (ECOC); `opposed` phá ít hơn `std` |
| G3 | `hard_pairs.py`, mở rộng + duyệt KG (§4), định nghĩa và kiểm tra vị từ (§3), `cache_predicates.py` | Mỗi vị từ AUC train ≥ 0.75 và lệch giữa view nhỏ; mọi top-25 cặp có ≥ 1 thuộc tính đối lập |
| G4 | Full-pred, K2-pred, ablation V-*, λ-fix | **Cổng chính**: Full-pred − K2-pred ≥ 0.3 (3 seed, không chồng std) **và** Full-pred > G1 **và** số sửa > số phá |
| G5 | K5 can thiệp vị từ | % đổi dự đoán theo luật ≫ can thiệp thuộc tính ngẫu nhiên |
| G6 | Fine-tune end-to-end (tùy chọn) | Chỉ khi G4 đạt |

Nếu G4 không đạt: kiểm tra theo thứ tự (1) độ phủ `v ≠ 0` có quá thấp không, (2) vị từ có đo đúng không (AUC
test, xem tay mẫu bị phá), (3) λ có quá lớn không (λ-fix). Nếu vị từ đúng mà KG thật ≈ xáo trộn → vấn đề là
nội dung KG, không phải kiến trúc.

## 9. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| Graph retrieval sụp lớp / chỉ trang trí | §2.2; G1-0 là đối chứng bắt buộc |
| λ học trên train (backbone 99.6%) không hợp test | λ-fix; `feature_noise` trên `z`; phương án cuối: train lại backbone trên ~80% subject, head trên 20% còn lại |
| Vị từ nhiễu: khớp tay 21–24 của Kinect rất nhiễu; hai người; che khuất | Ưu tiên khớp ổn định (cổ tay, khuỷu, đầu, hông); vị từ dùng tay/ngón chỉ khi AUC train đạt |
| Độ phủ thấp → tăng nhỏ | Báo cáo theo cặp khó và sửa/phá; trần lý thuyết 3.8 điểm, thực tế dự kiến < 1 |
| Rò rỉ khi chọn thuộc tính theo cặp nhầm trên test | §4 mục 6: kiểm tra độ nhạy với cặp chọn từ train |
| Thêm thuộc tính sau khi xem lỗi → "KG thiết kế theo dataset" | Giữ A1G-kg; mọi thuộc tính mới có định nghĩa hình học công khai và qua K2 |
| Không có validation | Công thức chốt trước, test một lần mỗi cấu hình |

## 10. Những gì giữ, bỏ, tạm dừng từ A1

| Thành phần A1 | A1-G |
|---|---|
| Cache l9, `pooled_z`, `valid_actor_mask` | Giữ |
| KG, `validate_kg`, `shuffle_kg`, `kg_code_distance` | Giữ, mở rộng (§4) |
| Embedding text (`kg_emb_mpnet.npz`) | Chỉ còn cho K1 và G1-init |
| VERIFY `std` | Thay bằng `opposed`; `std` thành ablation |
| FUSE | Ablation |
| Decoder thuộc tính trên map l9 | Tạm dừng (vẫn chung đường tắt nhãn lớp) |
| K6 zero-shot, A1-enc | Bỏ |
| Fine-tune P6 | Dời về G6, chỉ khi G4 đạt |
