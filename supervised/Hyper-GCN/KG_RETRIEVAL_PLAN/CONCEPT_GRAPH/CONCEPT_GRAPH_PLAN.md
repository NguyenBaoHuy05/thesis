# Concept–Action Logic Graph: kế hoạch

> Trạng thái: **P0 (code cho P1) xong** (2026-10-19), chờ chạy P1 trên Colab. Tiến độ: [CONCEPT_GRAPH_PROGRESS.md](CONCEPT_GRAPH_PROGRESS.md).
> Sơ đồ: [drawio/architecture_concept_graph.drawio](../../drawio/architecture_concept_graph.drawio).
> Kế thừa: [CONCEPT_LOGIC_PROGRESS.md](../CONCEPT_LOGIC/CONCEPT_LOGIC_PROGRESS.md) (CL v1, v2 = AND phẳng).
> Đối chiếu chính: **REASON** ([arXiv 2605.07140](https://arxiv.org/abs/2605.07140), IJCAI 2026, code ở `D:\Thesis\REASON`).
>
> Mục tiêu luận văn: neuro-symbolic cho nhận dạng hành động trên khung xương, trên các lớp đã thấy (không zero-shot).
> Dev dùng seed 0; chỉ chạy thêm seed khi kết quả sát ngưỡng và khi chốt số cuối.

## 0. Tính mới cần chứng minh

REASON đã có: backbone GCN (có Hyper-GCN) → concept decoder (BCE theo ma trận M) → các lớp AND/OR học được, phủ
định concept [c; ¬c] → **linear head**. Vì vậy concept bottleneck + AND/OR/¬concept **không mới**. Luận văn claim:

| | Đóng góp | Bằng chứng |
|---|---|---|
| **C1** | **¬(j\|i)**: literal phủ định gắn với cặp dễ nhầm, chỉ dựa trên concept phân biệt M_j∖M_i, nằm trong AND của lớp i | Tỉ lệ nhầm trong cặp (PCR) giảm so với AND phẳng và so với logic kiểu REASON |
| **C2** | Lớp được quyết định **chỉ** bằng thực thi đồ thị logic có cấu trúc (motif ∧ dùng chung, ∨ thay thế, ¬(j\|i)), không có linear head → vết chứng minh trung thực | λ = 0 gần bản đầy đủ; K5 |
| C3 | Đánh giá trung thực: can thiệp K5 qua đồ thị, vết chứng minh | K5, ví dụ định tính |

R-GCN chỉ là chi tiết cài đặt. Nếu ablation không cho thấy nó có ích thì bỏ R-GCN; C1 và C2 vẫn đứng được.

## 1. Câu hỏi nghiên cứu và thứ tự

| | Câu hỏi | Pha | So sánh |
|---|---|---|---|
| Q1 | ¬(j\|i) có giảm nhầm trong cặp không? | **P1** (rẻ, làm trước) | E1c/E1d vs E1a (CL v2), E1b (logic REASON) |
| Q2 | Cấu trúc đồ thị (motif ∧, ∨) có giúp không? | P2 | E2 vs E1c, từng ablation |
| Q3 | Nhánh action + R-GCN theo mẫu có thêm gì không? | P3 | E3 vs E2; E ngẫu nhiên cố định; λ = 0 |
| Q4 | Kết quả cuối, so REASON đầy đủ | P4 | 3 seed, X-Sub + X-Set |

**Không đạt Q1 thì dừng** trước khi làm P2–P3 (C1 là đóng góp chính).

## 2. Dữ liệu và chỉ số chung

- Cache: `tools/cache_joint_features.py` (l9, S = 1, đã có trên Colab: `/content/jcache`, Drive `kg_cache/`).
- Ma trận: `annotations/concepts/ntu_action_concept_v2_4.csv` (120 × 102) + metadata.
- **Cặp dễ nhầm P** (chốt trước, chỉ từ M và tên lớp, không dùng test):
  55 cặp Hamming ≤ 4 ∪ 6 cặp put_on/take_off (jacket, shoe, glasses, hat_cap, headphone, bag) = **58 cặp**
  (3 cặp put_on/take_off đã nằm trong 55). Có hướng: tối đa 116 node ¬(j|i); bỏ node khi M_j∖M_i rỗng (3 trường hợp).
- Chỉ số, mọi lần chạy đều ghi vào `kg_metrics.json`:

| Chỉ số | Định nghĩa |
|---|---|
| Top1 / Top5 | như trước |
| **PCR** (pair confusion rate) | Trên các mẫu test có nhãn y thuộc ít nhất một cặp trong P: tỉ lệ dự đoán là một đối tác j của y |
| PCR theo cặp | Bảng 58 dòng, để vẽ hình |
| sửa / phá vs K0 | như trước |
| AUC concept | macro, so với M[y] |
| K5 | tắt concept hỗ trợ mạnh nhất / tắt concept ngẫu nhiên lớp đó không có |
| K5-pair (mới) | mẫu dự đoán i có đối tác j: bật các concept của M_j∖M_i (c ← 1) → tỉ lệ dự đoán chuyển sang j |

PCR của K0 (fc_logits cache) tính một lần làm mốc.

## 3. P0: hạ tầng (code, chưa chạy thí nghiệm)

| File | Nội dung |
|---|---|
| `model/attribute_decoder.py` | thêm `return_tokens=True` → trả cả `q` sau LN (U [N, A, d]) và xác suất |
| `model/concept_graph.py` | dựng đồ thị (§5): `build_graph(M, names, meta, cfg) → ConceptGraph` (danh sách node, chỉ số cạnh theo quan hệ r1–r8), `save/load` json để đọc tay |
| `tools/build_concept_graph.py` | CLI: ghi `annotations/concepts/graph_v1.json` + báo cáo (số node / cạnh, motif, lớp bị phủ, cặp bỏ) |
| `model/fuzzy_executor.py` | bộ thực thi logic mờ trên `ConceptGraph` (§6.1), có `trace()` cho vết chứng minh |
| `model/rgcn.py` | R-GCN theo tầng (§6.2), basis decomposition |
| `model/reason_logic.py` | port `ConceptLogicLayers` từ `REASON/model/crl/components.py` (MIT, ghi nguồn), cho baseline E1b |
| `model/concept_graph_head.py` | ghép stage 1 + executor (+ R-GCN, nhánh action) + loss |
| `train_concept_logic.py` | thêm `head: cl \| reason \| graph` trong config; thêm PCR, K5-pair; giữ một vòng train |
| `config/concept_graph/*.yaml` | một file / thí nghiệm, siêu tham số chốt trong file |
| tests | `test_concept_graph.py`, `test_fuzzy_executor.py`, `test_rgcn.py`, `test_reason_logic.py` (§9) |

## 4. Stage 1: hai nhánh decoder

- **Nhánh concept**: giữ đúng decoder của CL v2 (102 query one-hot, mask nhóm khớp, 2 lớp, d = 256) → U, c. BCE(c, M[y]).
- **Nhánh action** (chỉ ở P3): `AttributeDecoder` với 120 query, mask toàn khớp, 1 lớp → A [N, 120, d]; head phụ
  ŝ = wᵀa_k, L_aux = CE(ŝ, y) với trọng số nhỏ (0.1). Dropout cả token A (p = 0.3) khi đưa vào đồ thị.
- Nhiễu đặc trưng 1.0 như CL v2 (v1 đạt 100% train top1).

## 5. Stage 2: dựng đồ thị (offline, không train)

Mọi quy tắc chốt ở đây, trước khi chạy; tham số nằm trong `cfg` của `build_graph`.

**Node**: 102 concept, 120 action, cổng ∧ (motif), cổng ∨, node ¬(j|i).

**Motif ∧** (r1 concept ∈ ∧, r4 ∧ → action):
1. Bỏ concept phổ biến (≥ 60/120 lớp: `hip_static_stand`, `leg_static_stand`, `temporal_sequence_seq_hands_first`,
   `temporal_dynamics_duration_brief`) khỏi việc khai thác motif. Chúng vẫn vào lớp qua cạnh thẳng hoặc ∨.
2. Ứng viên: tập 2–3 concept cùng có trong ≥ 3 hàng của M (412 tập cỡ 2, 676 tập cỡ 3 trên v2.4).
3. Chọn tham lam: mỗi lượt lấy ứng viên phủ nhiều cặp (lớp, concept) chưa phủ nhất (ưu tiên tập lớn khi bằng
   nhau). Dừng khi đạt `max_motifs = 40` hoặc ứng viên tốt nhất phủ < 6 ô. Mỗi lớp nhận tối đa 3 motif.
   Motif của lớp phải ⊆ hàng của lớp đó.

**Cổng ∨** (r2 concept ∈ ∨, r5 ∨ → action), danh sách tay, chốt trước:
- `posture`: hip_static_stand ∨ hip_static_sit, leg_static_stand ∨ leg_static_sit.
  Lý do: NTU quay cùng hành động ở cả tư thế đứng và ngồi, trong khi M chỉ ghi một tư thế.
  Áp cho mọi lớp có một trong các concept tư thế tĩnh (88 lớp có hip_static_*). Lớp mà tư thế chính là hành động
  (sit_down, stand_up) dùng concept động (hip_sit, hip_extend), nên không bị ảnh hưởng.
- Không thêm ∨ nào khác ở v1 của đồ thị. Thêm sau phải ghi lý do và chạy ablation.

**Cạnh thẳng** (r3 concept → action): phần của hàng M không thuộc motif hay ∨ nào của lớp.

**¬(j|i)** (r8 concept ∈ M_j∖M_i → ¬(j|i), r7 ¬(j|i) ⊣ action i, r6 action j → ¬(j|i) chỉ cho R-GCN):
một node cho mỗi cặp có hướng (i, j) ∈ P với M_j∖M_i ≠ ∅.

**Kiểm tra bắt buộc (test)**:
- Hợp các đầu vào của mỗi action (motif ∪ ∨ ∪ cạnh thẳng) phủ đúng hàng M của nó, sau khi thay concept tư thế bằng
  nhóm ∨ tương ứng.
- Mọi đầu vào của ¬(j|i) thuộc M_j∖M_i.
- Không có hai action trùng hệt tập literal.

## 6. Stage 3: thực thi theo mẫu

### 6.1 Bộ thực thi logic mờ (không phụ thuộc R-GCN)

Với c ∈ (0, 1)^102, tính trong log, kẹp [1e-6, 1 − 1e-6]:

```
∧_g      = Σ_k α_gk log c_k / Σ_k α_gk                      (trung bình nhân có trọng số; α = softplus, khởi tạo 1)
∨_o      = log(1 − Π_k (1 − c_k))                           (noisy-OR)
¬(j|i)   = log(1 − exp(∧ của c_k, k ∈ M_j∖M_i))
pos_i    = mean có trọng số α của {∧_g, ∨_o, log c (cạnh thẳng)} đi vào i
neg_i    = mean có trọng số α của {¬(j|i)}                (0 nếu i không có đối tác)
glob_i   = mean_{k ∉ M_i} log(1 − c_k)                     (NOT closed-world kiểu CL v2, trọng số ω)
log s_i  = (pos_i + ν·neg_i + ω·glob_i) / (1 + ν·[i có đối tác] + ω)
logit_i  = log s_i / τ                                      (τ học, như CL)
```

Mặc định ν = 1, ω = 1 (giống CL v2 để E1c chỉ khác đúng ở ¬(j|i)). E1d dùng ω = 0.
`trace(n)` trả về, cho dự đoán i: các cổng cùng giá trị, các ¬(j|i) và concept phân biệt nào đang thấp/cao.

### 6.2 R-GCN theo tầng (chỉ ở P3)

- Một R-GCN duy nhất, train end-to-end cùng mọi thứ. Không tiền huấn luyện: embedding node e_v và W_r là tham số.
- Khởi tạo node: concept `[e_j ; c_j·u_j ; c_j]`, action `[e_k ; drop(a_k)]`, cổng / ¬ `[e_v ; 0]`, chiếu về d.
- Ba bước, mỗi bước một lớp R-GCN chỉ cập nhật nhóm node đích (không có vòng):
  - B1: concept → ∧, ∨, ¬ (r1, r2, r8).
  - B2: action⁰ → ¬ (r6).
  - B3: ∧, ∨, concept, ¬ → action (r4, r5, r3, r7).
- Basis B = 4, chuẩn hoá 1/c_{v,r}, self-loop. z_k = wᵀh_k.

### 6.3 Kết hợp và loss

```
logit = log s / τ + λ·z                                     (P2: λ = 0)
L = CE(logit, y) + β·BCE(c, M[y]) + γ·KL(softmax z ‖ softmax(log s/τ)) + μ·margin + η·Σ|α − 1|
margin = mean_{j: ¬(j|y)} max(0, m − log s_y + log s_j)     (m = 0.1)
```

Mặc định (chốt trước): β = 0.5, γ = 0.5, μ = 0.5, η = 1e-4, λ = 0.5. Không quét theo test.
Báo cáo cả λ = 0 khi test (cùng checkpoint), để xem đường logic có tự đứng được không.

## 7. Thí nghiệm

Mọi thí nghiệm: cache S = 1, 30 epoch, seed 0, cấu hình chốt trong `config/concept_graph/`.

### P1: kiểm tra C1 (làm trước, khoảng 6 lần chạy)

| | Head | Khác với E1a |
|---|---|---|
| K0 | fc_logits của backbone | mốc PCR |
| **E1a** | CL v2 (`cl_v2.yaml`, kết quả đang *chờ*) | — |
| **E1b** | decoder CL v2 + `ConceptLogicLayers` của REASON (128/128, use_not, skip, linear head) trên logit(c) | logic học được + linear head |
| **E1c** | CL v2 + ¬(j\|i), ω = 1 | chỉ thêm ¬(j\|i) |
| **E1d** | CL v2 + ¬(j\|i), ω = 0 | ¬ có mục tiêu thay cho NOT closed-world |
| E1e | E1c, ¬(j\|i) lấy từ cặp ngẫu nhiên (cùng số cặp, cùng cỡ M_j∖M_i) | đối chứng: ¬ có cần đúng cặp không |

**Quy tắc đạt (chốt trước):** C1 đạt nếu
`PCR(best of E1c, E1d) ≤ 0.8 × min(PCR E1a, PCR E1b)` **và** Top1 của nó ≥ Top1(E1a) − 0.3
**và** PCR của nó < PCR(E1e).
Nếu chênh nằm trong 10% quanh ngưỡng thì chạy thêm seed 1, 2.

Dự đoán ghi trước: E1c giảm PCR 15–30% so với E1a, Top1 ±0.5. E1b có Top1 cao hơn E1a (linear head) nhưng
PCR tương đương E1a.

### P2: cấu trúc đồ thị (chỉ khi P1 đạt)

| | Thay đổi |
|---|---|
| E2 | đồ thị đầy đủ: motif ∧ + ∨ posture + cạnh thẳng + ¬(j\|i), λ = 0 |
| E2−motif | bỏ motif (mọi concept nối thẳng) = biến thể tốt nhất của P1 |
| E2−OR | bỏ ∨ posture |
| E2-S4 | E2 trên cache S = 4 (concept thời gian có thông tin thứ tự rõ hơn) |

E2-S4 cần cache mới: khoảng 12.8 GB train fp16. Nếu RAM / VRAM Colab không đủ thì giữ memmap trên ổ local.

### P3: nhánh action + R-GCN

| | Thay đổi |
|---|---|
| E3 | E2 + nhánh action + R-GCN theo tầng, λ = 0.5, KL, margin |
| E3 (λ = 0 lúc test) | cùng checkpoint |
| E3−RGCN | bỏ R-GCN, z = ŝ (nhánh action) |
| E3-randE | e_v ngẫu nhiên cố định |
| E3−r6 | bỏ r6 |

Giữ R-GCN chỉ khi E3 > E3−RGCN ≥ 0.3 Top1 hoặc PCR giảm ≥ 10%.

### P4: số cuối

3 seed cho cấu hình thắng, NTU120 X-Sub và X-Set (cache X-Set mới).
REASON đầy đủ (repo gốc, joint) chạy lại trên Colab, nếu tài nguyên cho phép, để có cột so sánh.
Thêm stream bone / ensemble chỉ khi còn thời gian.

## 8. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| ECOC: c chỉ là mã lớp, ¬(j\|i) không thêm gì | E1c ≈ E1a ở PCR; E1e ≈ E1c | Dừng ở P1, viết lại claim |
| S = 1 làm mất thứ tự thời gian | PCR trên cặp put_on/take_off cao ở mọi head | E2-S4; K0 tách các cặp đảo thời gian 97–99.7%, nên thông tin có trong kênh |
| ¬ làm hỏng lớp không thuộc cặp | Top1 ngoài P giảm | báo Top1 trên các lớp ngoài P riêng |
| log s nhỏ, gradient yếu | τ chạm biên, loss đứng | trung bình (không phải tổng) trong log; kẹp; theo dõi τ |
| Nhánh action lấn át (P3) | λ = 0 lúc test tụt mạnh | tăng dropout token A, γ |
| Ma trận v2.4 có thể đến từ pipeline của REASON | — | ghi nguồn, không claim bộ concept |

## 9. Test (chạy trên Colab, cần torch)

- `test_concept_graph`: các kiểm tra bắt buộc ở §5; motif ⊆ hàng; ¬ đúng tập phân biệt; json đọc ghi được.
- `test_fuzzy_executor`: bảng chân trị trên đồ thị nhỏ (c ∈ {0.01, 0.99}): ∧ / ∨ / ¬ / lớp đúng; ω = 0 và ν = 0
  quay về AND chỉ trên has; E1c với P = ∅ cho đúng bằng `logic_scores(balance=True)` của CL v2; gradient tới c và α.
- `test_rgcn`: shape; B1 không đọc action; thay đổi node action không ảnh hưởng đầu ra của B1.
- `test_reason_logic`: forward / backward trên dữ liệu giả, shape [N, 120].
- `test_concept_logic`: giữ nguyên.

## 10. Lệnh (Colab)

```
python -m pytest tests -q
python tools/build_concept_graph.py --out annotations/concepts/graph_v1.json
for e in e1b e1c e1d e1e; do
  python train_concept_logic.py --config config/concept_graph/$e.yaml --cache-dir $J --work-dir $W/$e
done
```

## 11. Cần chốt với người dùng trước khi code

1. Danh sách ∨ ở v1 chỉ có posture (stand ∨ sit). Có thêm nhóm nào không?
2. Ngưỡng đạt của P1 (§7) có đồng ý không?
3. E1b dùng port logic REASON trên cache của mình (công bằng và rẻ). REASON đầy đủ để ở P4. Đồng ý không?
4. E2-S4: có chấp nhận cache khoảng 13 GB không?
