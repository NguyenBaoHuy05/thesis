# Hướng 2 — Semantic Hypergraph: hyperedge có tên làm nút thắt diễn giải

> Trạng thái: **hướng được chọn để thực nghiệm** (có GPU L4).
> Liên quan: [KG_PRIMITIVE_PLAN.md](KG_PRIMITIVE_PLAN.md) (hướng 1, dùng chung bank hyperedge và chú thích LLM).
>
> **Đã triển khai (code P0–P3):** `model/synergies.py` (bank, bank ngẫu nhiên A2, `pair_validity`),
> `model/synergy_bottleneck.py` (pooling, cổng, `SemanticHyperedges`, readout B1/B2, contrib),
> `SemanticHYPERGC` + `synergy_*` trong `model/hypergcn_large.py`, `--freeze-until` /
> `--backbone-lr-scale` trong `main.py`, `model/synergy_metrics.py`, `evaluate_synergy.py`
> (deletion/insertion toàn mạng, ổn định, heatmap), `annotations/class_synergy.json` (bản nháp LLM,
> cần kiểm tra tay), config `config/large/nturgbd120-cross-subject/synergy*.yaml`, test `tests/test_synergy.py`.
> Pipeline concept hình học + logic cũ (B2 ~46%) đã xóa khỏi code.
> Hai điểm lệch so với mục 2.2: (1) `G_sem` dùng `Dv⁻¹ H diag(g) De⁻¹ Hᵀ` với bậc đỉnh **không trọng số**
> thay cho `hyper_norm` — `hyper_norm` chia cho bậc có trọng số nên cổng của khớp chỉ thuộc một synergy
> bị triệt tiêu, còn ở đây g = 0 tắt hẳn synergy; (2) `β` là tham số thường khởi tạo 0, không qua `relu`
> (relu có gradient 0 tại 0 nên β sẽ không bao giờ rời 0).

## 0. Bối cảnh và lý do đổi hướng

- Teacher hình học làm nguồn concept duy nhất có trần thấp: oracle linear (concept đúng 100%)
  chỉ đạt ~46% (4 segment) / 46.7% (8 segment) trên NTU120 X-Sub, so với backbone ~86%.
  Tăng segment không giúp → nút thắt là từ vựng concept, không phải độ phân giải thời gian.
- Hyper-GCN vốn đã suy luận trên **hypergraph**: mỗi `HYPERGC` dựng hyperedge động bằng kNN
  (9 đỉnh) trên embedding khớp, gán trọng số `W = to_W(t_x)` (tanh), chia 8 subset, cộng 3 virtual
  joint. Các hyperedge này **không có tên** và lấy trung bình theo cả chuỗi (`t_x = x.mean(2)`).
- Ý tưởng: thêm vào các lớp cuối một tập **hyperedge ngữ nghĩa có tên** (synergy giữa các bộ phận
  cơ thể), cho chúng **cổng theo từng segment**, và buộc bộ phân loại chỉ nhìn thấy hành động qua
  các synergy này. Lời giải thích = "synergy nào, lúc nào, đóng góp bao nhiêu" — và đó chính là
  trọng số mà mạng thực sự dùng để truyền thông điệp, không phải giải thích hậu kiểm.

## 1. Câu hỏi nghiên cứu và giả thuyết

| | Câu hỏi | Giả thuyết | Kiểm chứng bằng |
|---|---|---|---|
| RQ1 | Nút thắt synergy có giữ được độ chính xác? | Mất ≤ 3–5 điểm so với backbone | Top1 so với B0 |
| RQ2 | Tên synergy có thật sự mang nghĩa? | Nhóm ngữ nghĩa tốt hơn nhóm ngẫu nhiên cùng kích thước, cả về độ chính xác lẫn độ hợp lý | Ablation A2 (random groups) |
| RQ3 | Giải thích có trung thực? | Xóa synergy được nêu tên làm rơi xác suất lớp mạnh hơn xóa synergy ngẫu nhiên | Deletion / insertion |
| RQ4 | Giải thích có hợp lý với tri thức con người? | Synergy quan trọng nhất của mỗi lớp trùng với chú thích (LLM + kiểm tra tay) | Precision@k, nDCG |

## 2. Kiến trúc

```
skeleton [N,3,T,25,M]
  → l1..l6  (pretrained, ĐÓNG BĂNG, giữ hyperedge động gốc)
  → l7..l9  (fine-tune, lr ×0.1) — mỗi HYPERGC:  A = A_phys + α·G_dyn + β·G_sem(s)
                                                     β khởi tạo 0 → ban đầu y hệt pretrained
  → Synergy bottleneck: với mỗi synergy e, segment s:
        f[e,s] = mean_{joint ∈ e, t ∈ s} feature_l9        (C = 512)
        g[e,s] = sigmoid(gate(f[e,s]))                     cổng synergy ∈ [0,1]
  → Readout cộng tính:  logit_c = Σ_{e,s} g[e,s] · ⟨w_c,e , f[e,s]⟩ + b_c
        (biến thể scalar: logit_c = Σ_{e,s} w_c,e,s · g[e,s])
  → max qua actor; synergy hai người tính từ cặp actor (mục 2.3)
```

### 2.1 Bank hyperedge ngữ nghĩa (chỉ số khớp NTU 0-based)

Mỗi synergy là một tập khớp có tên. Bản đầu ~19 synergy một người + 5 synergy hai người:

| # | Tên | Khớp | Hành động gợi ý |
|---|---|---|---|
| 1 | hand_R–head | 10, 11, 23, 24, 2, 3 | uống, ăn, gọi điện, chải tóc |
| 2 | hand_L–head | 6, 7, 21, 22, 2, 3 | như trên, tay trái |
| 3 | bimanual | 6, 7, 21, 22, 10, 11, 23, 24 | vỗ tay, xé giấy, gõ phím |
| 4 | arm_R–torso | 8, 9, 10, 11, 20, 1 | mặc áo, chạm ngực |
| 5 | arm_L–torso | 4, 5, 6, 7, 20, 1 | như trên |
| 6 | hand_R–hip | 10, 11, 16, 0 | lấy đồ trong túi |
| 7 | hand_L–hip | 6, 7, 12, 0 | như trên |
| 8 | hand_R–foot_R | 10, 11, 18, 19 | đi giày, nhặt đồ |
| 9 | hand_L–foot_L | 6, 7, 14, 15 | như trên |
| 10 | legs | 12–19 | nhảy, đá, bước |
| 11 | leg_R–torso | 16, 17, 18, 0, 1 | đá chân phải, đứng một chân |
| 12 | leg_L–torso | 12, 13, 14, 0, 1 | như trên |
| 13 | head–torso | 3, 2, 20, 1 | gật đầu, lắc đầu, cúi chào |
| 14 | spine | 0, 1, 20, 2 | cúi, ngả, nghiêng |
| 15 | shoulder_twist | 4, 8, 20, 1 | xoay người |
| 16 | arm_R–leg_L | 9, 10, 13, 14 | đi bộ (đối chéo) |
| 17 | arm_L–leg_R | 5, 6, 17, 18 | đi bộ (đối chéo) |
| 18 | upper_body | 2, 3, 4–11, 20 | vươn vai, giơ tay |
| 19 | hip–knee–spine | 0, 12, 16, 13, 17, 1, 20 | ngồi xuống, đứng dậy |
| P1 | hand(A)–head(B) | tay A × đầu B | tát, xoa đầu |
| P2 | hand(A)–torso(B) | tay A × thân B | đẩy, đấm, ôm |
| P3 | hand(A)–hand(B) | tay A × tay B | bắt tay, đưa đồ |
| P4 | torso(A)–torso(B) | thân A × thân B | lại gần, đi xa |
| P5 | foot(A)–body(B) | chân A × thân B | đá người |

- Bank được lưu trong `model/synergies.py` (tên, khớp, mô tả ngắn). Có thể chỉnh trước khi train.
- Một **chú thích độ liên quan** `annotations/class_synergy.json` (120 lớp × 24 synergy, nhãn
  {0,1,2}) do LLM đề xuất từ tên lớp + mô tả synergy, kiểm tra tay ~30 lớp. Chỉ dùng **để đánh giá**
  (RQ4) và tùy chọn làm prior yếu — **không** dùng làm nhãn huấn luyện (tránh leakage).

### 2.2 Đưa synergy vào message passing (l7–l9)

- `H_sem` [E, V]: ma trận liên thuộc cố định từ bank (virtual joint không thuộc synergy nào).
- Cổng theo segment: `W_sem[n,e,s] = tanh(gate_e(mean_{t∈s} x[..., e]))`, thay vì một trọng số cho
  cả chuỗi như `to_W` gốc. Từ đó `G_sem(s)` tính bằng `hyper_norm(H_sem, W_sem)` đã có sẵn.
- `A = A_phys + α·G_dyn + β·G_sem(s)`, với `β = relu(param)` khởi tạo 0 → checkpoint pretrained
  nạp vào cho kết quả y hệt ở bước 0; fine-tune dần đưa synergy vào luồng tính toán.
- Hệ quả: đặc trưng l9 của các khớp trong cùng synergy được trộn theo đúng cổng `W_sem` →
  cổng ở nút thắt (2.3) có quan hệ trực tiếp với cách mạng truyền thông điệp.

### 2.3 Nút thắt synergy và readout

- Gộp đặc trưng l9 (T = 16) theo synergy × segment (S = 4): `f[e,s]`.
- Cổng `g[e,s] = sigmoid(MLP_e(LN(f[e,s])))`. Mặc định dùng chung giá trị với `W_sem` ở l9
  (biến thể "tied") để lời giải thích chính là trọng số hypergraph; biến thể "untied" làm ablation.
- Synergy hai người: `f_pair = φ(f_A[e1,s], f_B[e2,s]) + φ(f_B[e1,s], f_A[e2,s])` (đối xứng), chỉ
  khi teacher xác nhận có hai người thật (`pair_validity`, đã có trong code).
- Readout:
  - **B2 — embedding bottleneck (mặc định):** `logit_c = Σ g[e,s]·⟨w_c,e, f[e,s]⟩ + b_c`.
    Đóng góp của từng (synergy, segment) là **chính xác và cộng tính**.
  - **B1 — scalar bottleneck (nghiêm ngặt):** `logit_c = Σ w_c,e,s · g[e,s]`. Diễn giải mạnh nhất,
    kỳ vọng độ chính xác thấp hơn.
- Không có đường tắt từ backbone sang classifier: `fc` gốc bị bỏ khỏi đường dự đoán.

### 2.4 Hàm mất mát

`L = CE(label smoothing 0.1) + λ_g·mean(g) + λ_e·entropy-theo-lớp(contrib) + λ_div·DivergenceLoss`

- `λ_g` (~1e-3): khuyến khích ít synergy mở → giải thích gọn.
- `λ_e` (tùy chọn): đóng góp của mỗi mẫu tập trung vào ít (synergy, segment).
- `DivergenceLoss`: giữ nguyên như Hyper-GCN gốc cho virtual joint.

## 3. Lời giải thích và cách đo

Với mỗi mẫu: `contrib[c,e,s] = g[e,s]·⟨w_c,e, f[e,s]⟩` (B2) — bảng 24 × 4, lấy top-k cho lớp dự đoán.
Ví dụ đầu ra: *"drink_water: hand_R–head ở segment 2–3 (62%), head–torso ở segment 2 (14%)"*.

| Nhóm | Metric | Định nghĩa |
|---|---|---|
| Độ chính xác | Top1 / Top5 | NTU120 X-Sub (chính), X-Set (phụ) |
| Trung thực | Deletion AUC | Tắt lần lượt (synergy, segment) theo thứ tự contrib giảm dần (đặt g = 0 trong cả G_sem và readout), đo P(lớp dự đoán) |
| Trung thực | Insertion AUC | Ngược lại, bắt đầu từ mọi g = 0 |
| Trung thực | Random baseline | Cùng số bước nhưng chọn ngẫu nhiên — chênh lệch với deletion là thước đo chính |
| Gọn | Synergy hiệu dụng / mẫu | Số (e,s) cần để đạt 80% tổng contrib |
| Hợp lý | Precision@3, nDCG | Top-3 synergy trung bình theo lớp so với `class_synergy.json` |
| Ổn định | Jaccard top-k | Giữa 3 seed và giữa mẫu gốc / mẫu có `random_rot` |

## 4. Baseline và ablation

| Mã | Cấu hình | Mục đích |
|---|---|---|
| B0 | Checkpoint Hyper-GCN large joint hiện tại, đo lại | Trần độ chính xác |
| B1 | B0 + fine-tune l7–l9 cùng lịch (không synergy, giữ `fc`) | Tách ảnh hưởng của việc fine-tune |
| B2 | Oracle teacher hình học (đã có: ~46%) | Mốc của hướng cũ |
| **M** | Mô hình đầy đủ (tied, B2-readout, 24 synergy, S = 4) | Kết quả chính |
| A1 | Readout scalar (B1) | Đánh đổi chính xác ↔ diễn giải |
| A2 | **Nhóm ngẫu nhiên** cùng số lượng và kích thước | RQ2 — kiểm soát quan trọng nhất |
| A3 | Bỏ `G_sem` khỏi message passing (chỉ nút thắt) | Vai trò của việc đưa synergy vào l7–l9 |
| A4 | Untied (cổng nút thắt ≠ `W_sem`) | Vai trò của việc buộc giải thích = trọng số hypergraph |
| A5 | Bỏ synergy hai người | Đóng góp của tương tác |
| A6 | S ∈ {1, 2, 4, 8} | Độ phân giải thời gian của giải thích |
| A7 | Fine-tune l4–l9 thay vì l7–l9 | Chi phí GPU ↔ độ chính xác |

## 5. Lộ trình thực nghiệm (L4)

| Pha | Việc | Thời gian ước tính | Tiêu chí đi tiếp |
|---|---|---|---|
| P0 | `model/synergies.py` (bank), `annotations/class_synergy.json` (LLM + kiểm tra 30 lớp), đo lại B0 | 0.5–1 ngày | B0 khớp số đã biết |
| P1 | **Khả thi, backbone đóng băng**: nút thắt synergy + readout B2 trên đặc trưng l9 đóng băng (chưa có `G_sem`), 10 epoch (~4 phút/epoch) | ~1 giờ GPU | Top1 ≥ 75%; nếu < 65% xem lại bank / thử K mode mỗi synergy |
| P2 | `SemanticHYPERGC` cho l7–l9 (β = 0 khởi tạo), fine-tune 30 epoch, lr backbone ×0.1 | ~3–4 giờ GPU | Top1 trong khoảng 3–5 điểm so với B0 |
| P3 | Công cụ đánh giá: contrib, deletion/insertion, precision@k, trực quan hóa (heatmap synergy × segment, skeleton tô màu khớp) | 1–2 ngày | Deletion vượt random rõ rệt |
| P4 | Ablation A1–A7, 3 seed cho M và A2 | ~1–1.5 ngày GPU | — |
| P5 | Tổng hợp bảng, hình, phân tích lỗi theo lớp (`analyze_confusion.py`) | 1–2 ngày | — |

Thứ tự code (tận dụng hạ tầng hiện có):
1. `model/synergies.py` — bank + `incidence(num_joints)`; test: mỗi synergy có ≥ 2 khớp, không trùng tên.
2. `model/synergy_bottleneck.py` — pooling synergy × segment, cổng, readout B1/B2, contrib.
3. Tùy chọn `symbolic_mode: synergy` trong `hypergcn_large.Model`; tái dùng `PartSegmentPooling`
   (tổng quát hóa ma trận gán từ "bộ phận" sang "synergy"), `pair_validity` của teacher,
   `configure_trainable_parameters` (thêm chế độ "fine-tune từ lớp l7").
4. `SemanticHYPERGC(HYPERGC)` — thêm `H_sem`, cổng theo segment, `β`.
5. `main.py`: log contrib top-k, metric mục 3; giữ cơ chế chọn checkpoint theo Top1.

## 6. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Nút thắt quá hẹp | P1 < 65% | Thêm K mode/synergy (2–4), hoặc readout B2 với C lớn hơn |
| Nhóm ngẫu nhiên tốt ngang nhóm ngữ nghĩa (A2) | Top1 và precision@k gần nhau | Vẫn là kết quả trung thực đáng báo cáo; chuyển trọng tâm sang trung thực (RQ3) và thời gian; xem lại bank |
| Một synergy thống trị (vd. upper_body) | Contrib > 50% ở hầu hết lớp | Bỏ synergy quá lớn, tăng `λ_e`, chuẩn hóa theo kích thước |
| Fine-tune làm hỏng đặc trưng | B1 < B0 rõ rệt | Chỉ fine-tune l9, giảm lr, warmup dài hơn |
| Cổng bão hòa (luôn 0 hoặc 1) | Phân bố g dồn hai đầu ngay từ đầu | Nhiệt độ cho sigmoid, giảm `λ_g` |
| "Đóng góp" khác "trung thực" do G_sem lan truyền | Deletion trong readout ≠ deletion toàn mạng | Báo cáo cả hai; biến thể A3 làm đối chứng |

## 7. Đóng góp dự kiến cho luận văn

1. Một Hyper-GCN mà nút thắt quyết định là **hypergraph synergy có tên, cổng theo thời gian**,
   mất ít độ chính xác.
2. Lời giải thích **là** trọng số hypergraph mà mạng dùng (tied), có đo trung thực bằng deletion /
   insertion so với ngẫu nhiên.
3. Phép kiểm soát **nhóm ngữ nghĩa vs nhóm ngẫu nhiên** cho thấy tên synergy có đóng góp thật
   (hoặc không) — một câu hỏi thường bị bỏ qua.
4. Nghiên cứu âm tính có số liệu: concept hình học viết tay làm nút thắt duy nhất bị giới hạn ~46%.
