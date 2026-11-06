# Kế hoạch Neuro-Symbolic cho Hyper-GCN Large

> Phạm vi: chỉ `model/hypergcn_large.py`, NTU120 cross-subject trước. Bản base (`model/hypergcn_base.py`) chỉ còn backbone + FC; bridge symbolic cũ của base đã được gỡ bỏ.

---

## 1. Mục tiêu và định nghĩa

### 1.1. Bài toán

Nhận dạng hành động từ chuỗi skeleton sao cho **mỗi dự đoán đi kèm một lý do đọc được và đúng với cách model thực sự quyết định**, ví dụ:

```text
kicking_something  ←  extend(leg_R)@đoạn 2  ∧  move_forward(foot_R)@đoạn 2  ∧  ¬near(actor1, actor2)
```

### 1.2. Neuro-symbolic hiểu theo nghĩa nào

Theo phân loại Kautz, hệ thống thuộc:

- **Kiểu 3 (Neuro; Symbolic):** mạng nơ-ron chuyển skeleton thành các khái niệm (concept) có tên. Một bộ suy luận logic dùng **chỉ** các concept đó để ra quyết định.
- **Kiểu 5 (Neuro_Symbolic):** tri thức hình học về cơ thể được đưa vào hàm mất mát để ràng buộc ý nghĩa của concept.

Ba điều kiện bắt buộc. Thiếu một điều thì hệ thống không còn là neuro-symbolic thật:

| # | Điều kiện | Nếu thiếu thì sao |
|---|---|---|
| C1 | Concept có **giám sát độc lập với nhãn action** (grounding) | Concept chỉ là tầng ẩn mang tên. Lời giải thích là giả. |
| C2 | Action **chỉ** được tính từ concept, không có đường tắt nào từ feature | Model có thể bỏ qua concept. Lời giải thích không phản ánh quyết định. |
| C3 | Bộ suy luận có dạng **luật đọc được**, thưa và có cấu trúc logic/thời gian | "Giải thích" thành 40 trọng số tuyến tính, không ai đọc được. |

### 1.3. Nguyên tắc

1. **Thủ công tối thiểu, có hệ thống.** Con người chỉ định nghĩa *cách đo* concept từ khớp (hình học). Luật nối concept với action do model tự học.
2. **Mỗi thành phần phải trả lời được câu hỏi "giải quyết vấn đề gì".** Thành phần nào không trả lời được thì không đưa vào.
3. **Báo cáo trung thực trade-off.** Accuracy thấp hơn baseline FC là chấp nhận được và phải được nêu rõ, không thêm đường tắt để che.

---

## 2. Bài học từ phiên bản trước (đã gỡ bỏ)

Phiên bản "structured event graph" trước đây vi phạm cả C1, C2 và C3. Bảng dưới là động lực cho thiết kế mới:

| Vấn đề trong bản cũ | Hậu quả | Thiết kế mới xử lý bằng |
|---|---|---|
| `loss()` của primitive luôn trả 0 | 16 primitive + 6 relation chỉ là tầng ẩn mang tên (vi phạm C1) | Geometric Teacher (§4.2) |
| Pooling `mean(M,T,V)` trước khi dự đoán primitive | Mất thông tin "bộ phận nào, lúc nào, người nào" | Part pooling + temporal segments (§4.3) |
| Relation `near/toward` dự đoán từ feature toàn cục, không có đối số | Không biết "ai gần ai" | PairHead trên cặp actor (§4.5) |
| Action logit = Linear + Linear + logit(conjunction) | Hai nhánh trùng nhau; 110 class không có luật được cộng hằng số +11.5 | Một Logic Reasoner duy nhất (§4.6) |
| YAML prototype viết tay, chỉ phủ 10/120 class | Không mở rộng được, thiên lệch | Luật được học, thưa (§4.6) |
| Event graph với node/edge cố định, không tham gia tính toán | Chỉ để trang trí | Bỏ. Lời giải thích lấy trực tiếp từ clause đang active (§4.7) |
| Lưu H/W/G của 9 layer mỗi forward | Tốn bộ nhớ, chỉ dùng để tính 9 số trung bình | Bỏ |

---

## 3. Kiến trúc tổng thể

```text
                  x : skeleton [N, 3, T=64, V=25, M=2]
                  │
      ┌───────────┴──────────────────────────────┐
      │                                          │ (không gradient)
      ▼                                          ▼
┌──────────────────────┐              ┌───────────────────────────┐
│ ① Hyper-GCN backbone │              │ ② Geometric Teacher       │
│    l1 … l9           │              │   body frame + 5 template │
└──────────┬───────────┘              │   → soft target y         │
           │ F: [N, M, C, T', V]      │     [N, S, K]             │
           ▼                          └────────────┬──────────────┘
┌──────────────────────────────┐                   │
│ ③ Part pooling + Segmenting  │                   │
│   V=25 → P=6 parts           │                   │
│   T'   → S=4 segments        │                   │
└──────────┬───────────────────┘                   │
           │ [N, M, C, S, P]                       │
     ┌─────┴───────────┐                           │
     ▼                 ▼                           │
┌──────────────┐ ┌────────────────┐                │
│ ④ Concept    │ │ ⑤ PairHead     │                │
│   Head       │ │  (actor1,      │                │
│ (per part,   │ │   actor2)      │                │
│  per actor)  │ │                │                │
└──────┬───────┘ └───────┬────────┘                │
       │ c: [N,S,K1]     │ r: [N,S,K2]             │
       └────────┬────────┘                         │
                ▼                                  │
     z = [c ; r] : [N, S, K]  ◄──── BCE ───────────┘   (K ≈ 35–40)
                │
                ▼    (điểm nghẽn: chỉ z đi qua)
┌─────────────────────────────────────────┐
│ ⑥ Logic Reasoner                        │
│   temporal ops → literals               │
│   soft-AND (clause, mask thưa)          │
│   soft-OR  (class)                      │
└──────────┬──────────────────────────────┘
           ▼
   action logits [N, 120]   +   ⑦ explanation / intervention
```

**Luồng gradient:** CE(action) → ⑥ → z → ④⑤ → ①. BCE(z, y) → ④⑤ → ①. Teacher ② không có tham số học.

**Điểm nghẽn (bottleneck) z là ranh giới neural/symbolic.** Mọi thứ phía trên z là perception, mọi thứ phía dưới là reasoning. Không có tensor nào khác đi qua ranh giới này (thỏa C2).

---

## 4. Các thành phần

Mỗi thành phần được trình bày theo bốn mục: **Vai trò**, **Giải quyết vấn đề gì**, **Thiết kế**, **Vì sao không làm cách khác**.

### 4.1. ① Hyper-GCN backbone

**Vai trò.** Trích xuất đặc trưng không gian–thời gian từ skeleton.

**Giải quyết vấn đề gì.** Skeleton thô bị nhiễu (Kinect), lệch góc nhìn và khác nhau giữa người. Vì vậy cần một bộ mã hóa mạnh để concept head có đặc trưng tốt hơn công thức hình học thô. Hyper-GCN đã có checkpoint tốt trên NTU, không cần train lại từ đầu.

**Thiết kế.**
- Giữ nguyên `l1–l9`, dynamic hypergraph và virtual hyper-joints.
- Đưa feature của ba tầng `l3` (joint), `l6` (part), `l9` (action) cho ③. Mặc định chỉ dùng `l9` ([N·M, 512, T/4, 25]). `l6` là ablation.
- Giai đoạn đầu: freeze toàn bộ, BatchNorm ở chế độ eval.

**Vì sao không làm cách khác.**
- *Bỏ backbone, dùng trực tiếp teacher làm concept?* Đây chính là ablation A2. Công thức hình học bị nhiễu và cứng, còn neural head học cách khử nhiễu và tổng quát hóa.
- *Đưa H/W/G của hypergraph vào symbolic?* Không. Ma trận học được không có ngữ nghĩa cố định, nên đưa vào sẽ là đường tắt trá hình.

### 4.2. ② Geometric Teacher

**Vai trò.** Sinh **nhãn mềm cho concept** trực tiếp từ tọa độ khớp, không cần annotation tay.

**Giải quyết vấn đề gì.** Đây là thành phần quan trọng nhất (thỏa C1). NTU chỉ có nhãn action, không có nhãn "tay đi lên". Nếu không có teacher, concept chỉ có thể học từ CE, và khi đó tên concept vô nghĩa. Teacher biến **tri thức hình học về cơ thể người** (thứ kiểm chứng được) thành tín hiệu giám sát.

**Thiết kế.**

*a) Hệ tọa độ cơ thể.* Giải quyết vấn đề lệch góc nhìn và vấn đề kích thước người.

```text
gốc        o = joint 1 (spine mid)
trục lên    u = normalize(j20 − j0)                  (spine shoulder − spine base)
trục ngang  l = normalize(j8 − j4) trực giao hóa với u (vai phải − vai trái)
trục trước  f = u × l
tỉ lệ       s = mean_t |j20 − j0|                     (chiều dài thân)
```

Mọi vị trí và vận tốc được chiếu lên (u, f, l) rồi chia cho s. Nhờ vậy "move_forward" mang cùng nghĩa dù camera đặt ở đâu, và không đổi dưới augmentation `random_rot`.

*b) Năm template vị từ.* Danh sách cụ thể nằm trong config, khoảng 35–40 vị từ:

| Template | Công thức (trên body frame, đã chuẩn hóa) | Ví dụ | Giải quyết |
|---|---|---|---|
| T1 `move(part, dir)` | ⟨v_endpoint, dir⟩, với dir ∈ {u, −u, f, −f} | move_up(hand_R), move_forward(foot_L) | Hướng chuyển động của tay/chân/đầu/thân |
| T2 `extend/flex(limb)` | dấu của dθ/dt tại khuỷu (4-5-6, 8-9-10) và gối (12-13-14, 16-17-18) | extend(arm_R), flex(leg_L) | Duỗi/gập, chẳng hạn đá, đấm, ngồi xuống |
| T3 `near(a, b)` | −‖p_a − p_b‖ / s | near(hand_R, head), near(hand_L, hand_R) | Tư thế tương đối giữa các bộ phận |
| T4 `posture(·)` | góc giữa trục thân và trục dọc camera; độ cao hông | tilted, crouched, lying | Trạng thái toàn thân, chẳng hạn ngã, ngồi |
| T5 `pair(a1, a2)` | khoảng cách spine; d(dist)/dt; min khoảng cách tay–thân người kia | near, approach, separate, contact | Tương tác hai người, chẳng hạn ôm, bắt tay, đi về phía nhau |

*c) Ngưỡng theo phân vị, không tune tay.* Với mỗi vị từ k, ngưỡng q_k là phân vị p_k của giá trị trên tập train (mặc định p = 75 cho vị từ sự kiện), được tính một lần trên một tập con rồi lưu thành buffer. Nhãn mềm theo frame:

```text
y_k(t) = sigmoid( (g_k(t) − q_k) / (0.1 · IQR_k) )
```

*d) Gộp theo đoạn.* Theo mỗi đoạn s, y_k[s] là trung bình của 25% frame có giá trị cao nhất. Cách này bắt được sự kiện ngắn nhưng không bị một frame nhiễu chi phối.

*e) Mặt nạ hợp lệ.* Khi actor 2 không tồn tại (toàn số 0), các vị từ T5 bị mask khỏi loss thay vì gán bằng 0.

*f) Vị trí trong code.* Teacher được tính trong `Model.forward` từ `x` (sau augmentation) và trả ra qua `SymbolicOutput.concept_targets`. Lý do: không cần pipeline tiền xử lý riêng, và target luôn khớp với input đã augment.

**Vì sao không làm cách khác.**
- *Annotation tay?* Quá tốn công cho 114k mẫu.
- *Suy concept từ nhãn action (bảng action→concept)?* Concept khi đó chỉ là mã hóa lại nhãn, gây rò rỉ nhãn và không độc lập (vi phạm C1). Bản base cũ đã gặp đúng vấn đề này.
- *Ngưỡng cố định tay?* Đó là những con số "ma thuật" khó bảo vệ trong luận văn. Phân vị chỉ có một tham số, có ý nghĩa thống kê rõ.

### 4.3. ③ Part pooling + Temporal segments

**Vai trò.** Chuyển feature [N, M, C, T', V] thành [N, M, C, S, P]: 6 bộ phận × 4 đoạn thời gian.

**Giải quyết vấn đề gì.** Concept có **đối số** (bộ phận nào) và **thời điểm** (lúc nào). Pooling toàn bộ V và T như bản cũ xóa mất chính thông tin đó. Ngược lại, giữ nguyên 25 khớp × 16 bước thời gian thì quá chi tiết so với độ phân giải của concept.

**Thiết kế.**

```text
head  : [2, 3]                    torso : [0, 1, 20]
arm_L : [4, 5, 6, 7, 21, 22]      arm_R : [8, 9, 10, 11, 23, 24]
leg_L : [12, 13, 14, 15]          leg_R : [16, 17, 18, 19]
```

Lấy trung bình feature trong mỗi nhóm khớp. T' được chia đều thành S = 4 đoạn (với `l9` và T = 64 thì mỗi đoạn ≈ 4 bước feature ≈ 16 frame).

**Vì sao không làm cách khác.**
- *Attention học được thay cho ánh xạ cố định?* Khi đó "bộ phận" sẽ do model tự định nghĩa và mất ngữ nghĩa. Ánh xạ cố định lấy thẳng từ đặc tả skeleton NTU nên không gây tranh cãi.
- *S = 4:* đủ để biểu diễn trình tự (bắt đầu → kết thúc) mà vẫn rẻ. Số này là siêu tham số, có thể ablation S ∈ {2, 4, 8}.

### 4.4. ④ ConceptHead (vị từ một người)

**Vai trò.** Dự đoán c[N, S, K1]: xác suất các vị từ T1–T4 cho mỗi đoạn.

**Giải quyết vấn đề gì.** Đây là bước **perception**, chuyển đặc trưng liên tục thành mức độ đúng của các ký hiệu rời rạc có tên.

**Thiết kế.**
- Mỗi vị từ chỉ đọc feature của **bộ phận là đối số của nó**. Ví dụ move_up(hand_R) chỉ đọc arm_R, near(hand_R, head) đọc arm_R và head. Đây là **typed head**: cấu trúc đối số được hard-wire vào kiến trúc, nên vị từ không thể "nhìn lén" bộ phận khác để đoán action.
- Mỗi template dùng một MLP nhỏ chung (chia sẻ trọng số giữa trái và phải), kèm embedding bộ phận.
- **Gộp hai actor:** c[s] = max(c_actor1[s], c_actor2[s]) với actor hợp lệ. Lý do: thứ tự actor trong file NTU là tùy ý, và câu hỏi "có ai đang giơ tay không" không phụ thuộc thứ tự.

**Vì sao không làm cách khác.**
- *Một Linear chung từ feature toàn cục ra K concept (như bản hiện tại)?* Concept có thể mã hóa bất kỳ thông tin nào nên dễ rò rỉ nhãn. Typed head giới hạn thông tin mà mỗi concept được thấy.

### 4.5. ⑤ PairHead (vị từ hai người)

**Vai trò.** Dự đoán r[N, S, K2] cho T5: near, approach, separate, contact.

**Giải quyết vấn đề gì.** Khoảng 26/120 class NTU120 là tương tác hai người (ôm, bắt tay, đẩy, đưa đồ…). Các class này không thể phân biệt bằng vị từ một người. Cần quan hệ có **hai đối số thật**.

**Thiết kế.**
- Input đối xứng: [h₁ + h₂ ; |h₁ − h₂|] trên feature torso và arm của hai actor. Phép đối xứng đảm bảo kết quả không đổi khi hoán vị actor.
- Khi chỉ có một actor: r = 0, và loss bị mask (§4.2e).

**Vì sao không làm cách khác.**
- *Gộp vào ConceptHead?* Khi đó mất tính hai đối số và phụ thuộc thứ tự actor.

### 4.6. ⑥ Logic Reasoner

**Vai trò.** Tính action logits **chỉ từ z = [c; r]** bằng các luật logic mềm học được.

**Giải quyết vấn đề gì.** Thỏa C2 và C3: quyết định phải có dạng luật mà người đọc được và kiểm tra được, đồng thời có biểu diễn thời gian. Luật được **học**, nên không cần viết tay 120 prototype.

**Thiết kế: ba tầng.**

*a) Toán tử thời gian → literal.* Mỗi concept z_k[s] sinh ra:

| Literal | Công thức mềm | Ý nghĩa |
|---|---|---|
| `EV(k)` (eventually) | 1 − ∏ₛ (1 − z_k[s]) | xảy ra ở đoạn nào đó |
| `AL(k)` (always) | ∏ₛ z_k[s] | đúng suốt chuỗi (trạng thái) |
| `NEV(k)` (never) | 1 − EV(k) | không hề xảy ra (phủ định) |
| `SEQ(a, b)` *(giai đoạn sau)* | maxₛ₂ ( z_b[s₂] · max_{s₁<s₂} z_a[s₁] ) | a xảy ra trước b |

Tầng này giải quyết vấn đề *thứ tự và thời lượng*. Ví dụ ngồi xuống (flex(leg) rồi crouched) khác với đứng lên (crouched rồi extend(leg)) chỉ ở thứ tự, và hai hành động này là cặp hay bị nhầm trong NTU.

*b) Clause = soft-AND.* Mỗi class có J clause (mặc định J = 3). Mỗi clause j có một mask m_j ∈ [0,1]^L trên L literal (m = sigmoid(θ)):

```text
AND_j = ∏_l ( 1 − m_jl · (1 − lit_l) )        (tính trong log-space)
```

Khi m_jl ≈ 0 thì literal l bị bỏ qua, khi m_jl ≈ 1 thì literal l bắt buộc phải đúng. Mỗi clause là một luật "IF a ∧ b ∧ ¬c".

*c) Class = soft-OR.* Mỗi class được thỏa nếu một trong các clause của nó được thỏa:

```text
P_k = 1 − ∏_{j∈k} (1 − AND_j)
logit_k = α · logit(P_k)                      (α học được, một số vô hướng)
```

OR giải quyết vấn đề **một action có nhiều cách thực hiện**, ví dụ vẫy tay trái hoặc vẫy tay phải.

*d) Ràng buộc thưa.* Loss L1 trên m (§5), tính bằng số literal kỳ vọng mỗi clause, đẩy mỗi clause về ít literal. Khi đánh giá, mask được **nhị phân hóa** (m > 0.5) để có luật rõ ràng, và accuracy được báo cáo với luật đã nhị phân hóa.

*e) Không dùng weight decay cho mask.* Weight decay kéo logit của mask về 0, tức m → 0.5, làm các literal không dùng tới dần thành "nửa active" và phá việc nhị phân hóa. Vì vậy `mask_logits` và `log_scale` nằm trong một param group riêng với `weight_decay = 0`.

**Vì sao không làm cách khác.**
- *Linear trên concept (Concept Bottleneck Model chuẩn)?* Cách này cho phép bù trừ cộng tính, ví dụ nhiều trọng số nhỏ và trọng số âm, nên lời giải thích là một vector 40 số. Đồng thời không biểu diễn được thứ tự thời gian. CBM-linear vẫn được giữ làm **ablation A4** và làm mốc trung gian (chính là code hiện tại).
- *Luật viết tay (YAML)?* Không mở rộng được lên 120 class và đã gây thiên lệch ở bản cũ.
- *Reasoner logic cứng, không khả vi (ProbLog, ASP)?* Không train end-to-end được. Logic mềm cho phép train bằng gradient, rồi nhị phân hóa khi suy luận.

### 4.7. ⑦ Giải thích và can thiệp

**Vai trò.** Xuất lời giải thích và cho phép thử nghiệm phản thực tế (counterfactual).

**Giải quyết vấn đề gì.** Đây là bằng chứng cho C2: lời giải thích phải **chính là** phép tính đã tạo ra logit, không phải một lớp hậu kiểm (post-hoc).

**Thiết kế.**
- `explain(sample)`: lấy class dự đoán k và clause j* = argmax AND_j. Trả về danh sách literal có m > 0.5, giá trị của chúng, và đoạn s nơi concept đạt đỉnh.
- `intervene(z, mask | values)`: gán z_k[s] thành 0, 1 hoặc giá trị khác rồi chạy lại ⑥. Vì logit chỉ phụ thuộc vào z nên phép can thiệp này đo đúng ảnh hưởng nhân quả trong model.

---

## 5. Hàm mất mát

```text
L = CE(action_logits, label)
  + λ_c · BCE_masked(z, y_teacher)          (λ_c ≈ 1.0)
  + λ_s · mean_j Σ_l m_jl                   (λ_s ≈ 1e-3; số literal kỳ vọng mỗi clause)
```

| Thành phần | Giải quyết vấn đề gì |
|---|---|
| CE | Mục tiêu chính: phân loại action |
| BCE với teacher | Giữ **ngữ nghĩa** cho concept (C1). Không có nó, CE sẽ "chiếm dụng" concept để mã hóa class. |
| L1 trên mask | Giữ luật **ngắn và đọc được** (C3). Không có nó, mỗi clause dùng mọi literal. |

Chỉ có ba hạng tử, không thêm loss cho relation, phase hay structure như bản cũ.

---

## 6. Chiến lược huấn luyện

| Giai đoạn | Train | Freeze | Mục đích / vấn đề giải quyết |
|---|---|---|---|
| S0 | (đã có) | – | Baseline backbone + FC, làm mốc accuracy |
| S1 | ④⑤ chỉ với BCE | ①⑥ | Grounding concept **trước** khi CE tác động, tránh concept bị CE "bẻ" nghĩa ngay từ đầu |
| S2 | ⑥ với CE + L1 (z detach) | ①④⑤ | Đo xem concept đã grounding **đủ** để phân loại đến đâu (accuracy "thuần symbolic") |
| S3 | ④⑤⑥ với loss đầy đủ | ① | Cho concept thích nghi nhẹ với nhiệm vụ trong khi BCE giữ nghĩa |
| S4 | toàn bộ, LR backbone nhỏ (×0.1) | – | Tinh chỉnh. So sánh với S3 để đo độ trôi ngữ nghĩa |

Stage được chọn bằng `model_args.symbolic_stage`. S1–S3 bắt buộc `symbolic_only: True`, S4 bắt buộc `symbolic_only: False`; `main.py` báo lỗi nếu hai cờ không khớp. Mỗi stage có một config riêng (`hyper_joint_symbolic.yaml` là S1, `_s2`, `_s3`, `_s4`), và `--weights` trỏ tới checkpoint tốt nhất của stage trước.

Ở S1 reasoner chưa được train, nên checkpoint tốt nhất được chọn theo **concept macro AUC**, không theo accuracy. Teacher được calibrate một lần ở đầu S1, và các ngưỡng được lưu trong checkpoint để các stage sau dùng lại.

---

## 7. Những gì làm thủ công và những gì không

| Thủ công (con người định nghĩa) | Tự động / học |
|---|---|
| Ánh xạ 25 khớp → 6 bộ phận (từ đặc tả NTU) | Ngưỡng concept (phân vị trên dữ liệu) |
| Định nghĩa body frame (§4.2a) | Giá trị concept (ConceptHead / PairHead) |
| 5 template vị từ và danh sách đối số (≈ 40 vị từ) | Luật concept → action (mask trong ⑥) |
| Tham số phân vị p (mặc định 75) | Số literal mỗi luật (nhờ L1) |

**Không cần:** bảng action → concept, prototype YAML cho từng class, ontology relation/event, node/edge type.

Tổng phần thủ công nằm trong một file (`geometric_predicates.py`, ước tính khoảng 150–200 dòng) và một khối config. Trong luận văn, phần này được trình bày như **tri thức miền về giải phẫu chuyển động**, là thành phần "symbolic knowledge" của hệ thống.

---

## 8. Đánh giá

### 8.1. Action
Top-1/Top-5, accuracy theo class, confusion matrix. So sánh với S0. Báo cáo riêng hai nhóm:
- **Nhóm phụ thuộc vật thể** (uống nước, ăn, đánh răng, đọc, viết…): skeleton không có thông tin vật thể nên dự kiến model sẽ yếu ở nhóm này.
- **Nhóm tương tác hai người:** đo đóng góp của ⑤.

### 8.2. Chất lượng concept
- AUC/F1 của z so với teacher trên tập test.
- **Kiểm tra chính teacher:** xem thủ công khoảng 100 mẫu ngẫu nhiên (vẽ skeleton) để ước lượng độ chính xác của vị từ hình học. Pseudo-label không phải ground truth và phải nói rõ điều này.

### 8.3. Độ trung thực của giải thích (faithfulness)
| Phép đo | Cách làm | Chứng minh điều gì |
|---|---|---|
| Deletion | Đặt các literal của clause j* về 0, đo độ giảm của logit | Clause được giải thích thật sự gây ra dự đoán |
| Sufficiency | Chỉ giữ literal của j*, các literal khác = 0.5 | Lời giải thích đủ để tái tạo dự đoán |
| Soft vs hard | Accuracy với z liên tục và với z đã nhị phân hóa | Đo **rò rỉ thông tin** qua xác suất mềm |
| Độ gọn | Số literal trung bình mỗi clause, số clause active | Luật có đọc được không |

---

## 9. Ablation

| ID | Cấu hình | Trả lời câu hỏi |
|---|---|---|
| A1 | Backbone + FC | Giới hạn trên của accuracy |
| A2 | Teacher (không neural) → ⑥ | Riêng hình học giải thích được bao nhiêu? Neural đóng góp thêm bao nhiêu? |
| A3 | Concept **không** grounding (λ_c = 0) → ⑥ | Grounding đổi lấy bao nhiêu accuracy? Đây là cái giá của tính giải thích |
| A4 | Concept grounding → Linear (CBM) | Logic reasoner so với linear: accuracy và độ gọn |
| A5 | ⑥ chỉ có EV (không AL/NEV/SEQ) | Toán tử thời gian có cần không |
| A6 | Không có ⑤ PairHead | Đóng góp của quan hệ hai người |
| A7 | Feature `l6` so với `l9` | Tầng nào tốt hơn cho grounding |
| A8 | S3 so với S4 | Fine-tune backbone có làm trôi nghĩa concept không |

---

## 10. Rủi ro

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Accuracy giảm mạnh so với A1 | Khoảng cách > 10 điểm % | Tăng J, thêm SEQ, thêm template. Báo cáo trade-off, **không** thêm đường tắt |
| Teacher nhiễu | AUC cao nhưng kiểm tra tay thấy sai | Hiệu chỉnh body frame, dùng nhãn mềm. Báo cáo độ chính xác của teacher |
| Rò rỉ qua xác suất mềm | Accuracy hard ≪ soft | Train S3 với straight-through hard concept |
| Gradient tan biến ở soft-AND | Mask không học, clause luôn ≈ 0 | Tính trong log-space, khởi tạo m nhỏ, warm-up với A4 |
| Concept chết hoặc trùng nhau | std ≈ 0, tương quan > 0.95 | Loại hoặc gộp vị từ trong config. `main.py` đã log các chỉ số này |

---

## 11. Cấu trúc code

| File | Nội dung | Trạng thái |
|---|---|---|
| `model/hypergcn_large.py` | Backbone. Khi `symbolic_graph_enabled` thì đưa feature của tầng đã chọn và skeleton thô cho reasoner | ✅ |
| `model/symbolic_output.py` | `action_logits`, concept logits/probabilities/targets/mask, `hard_action_logits`, `faithfulness` | ✅ |
| `model/geometric_predicates.py` | ② khôi phục tọa độ từ feeder, body frame, 5 template, calibration theo phân vị, gộp đoạn, mask | ✅ |
| `model/concept_head.py` | ③ part pooling + segmenting, ④ typed ConceptHead, ⑤ PairHead | ✅ |
| `model/logic_reasoner.py` | ⑥ EV/AL/NEV/SEQ, soft-AND/OR, nhị phân hóa, `rules` / `explain` / `faithfulness`; `LinearReasoner` (A4) | ✅ |
| `model/large_symbolic_reasoner.py` | Kết nối ②–⑥, điều khiển stage, `total_loss`, `intervene`, các ablation A2/A3/A4/A6 | ✅ |
| `model/symbolic_metrics.py` | `grounded_concept_metrics`: AUC/F1 cho từng concept so với teacher | ✅ |
| `main.py` | Calibrate teacher, optimizer theo stage, ghép loss (§5), xuất `symbolic_metrics.json` / `rules.json` / `explanations.json` mỗi epoch | ✅ |
| `config/large/nturgbd120-cross-subject/hyper_joint_symbolic*.yaml` | Một config cho mỗi stage S1–S4 | ✅ |
| `tests/test_large_symbolic.py` | Teacher trên skeleton tổng hợp, typed head, PairHead đối xứng, ngữ nghĩa clause/SEQ, stage | ✅ viết xong, chưa chạy |

**Cách bật các ablation** (§9) qua `model_args`:

| ID | Thiết lập |
|---|---|
| A2 | `symbolic_concept_source: teacher`, stage 2 |
| A3 | `symbolic_concept_weight: 0` |
| A4 | `symbolic_reasoner: linear` |
| A5 | `symbolic_temporal_ops: [ev]` |
| A6 | `symbolic_use_pair: False` |
| A7 | `symbolic_feature_level: l6` |
| SEQ | `symbolic_temporal_ops: [ev, al, nev, seq]` |

---

## 12. Lộ trình

**Phase 0: dọn dẹp** ✅
- [x] Gỡ event graph, rule engine, prototype YAML, relation/phase head, việc thu thập H/W/G.
- [x] Sửa lỗi gọi `loss()` với tham số không tồn tại.
- [x] Reasoner tối giản: concept → linear, không có đường tắt.
- [ ] Xóa file mồ côi: `model/structured_event_graph.py`, `model/primitive_evidence.py`, `model/backbone_evidence.py`, `concepts/ntu_action_prototypes_v1.yaml`.

**Phase 1: Teacher (②)**
- [x] Body frame + 5 template + test đơn vị trên skeleton tổng hợp (ví dụ tay đi lên thì move_up ≈ 1).
- [x] Tính phân vị trên tập con train, lưu vào buffer.
- [ ] Chạy test đơn vị.
- [ ] Kiểm tra tay khoảng 100 mẫu.

**Phase 2: Perception (③④⑤) + S1**
- [x] Part pooling, segmenting, typed head, PairHead.
- [ ] Train S1, báo cáo AUC/F1 cho từng vị từ, loại vị từ chết.

**Phase 3: Reasoner (⑥⑦) + S2/S3**
- [x] EV/AL/NEV, soft-AND/OR, L1, nhị phân hóa.
- [x] `explain` / `intervene` / `faithfulness`.
- [ ] Train S2/S3, chạy A2–A5.

**Phase 4: Mở rộng và tổng kết**
- [x] SEQ (code), S4 (config + optimizer).
- [ ] Chạy S4, A6–A8.
- [ ] Bảng kết quả cuối cùng và phân tích theo nhóm class.

---

## 13. Quyết định đã chốt

| Câu hỏi | Quyết định | Lý do |
|---|---|---|
| Reasoner rule-based, differentiable hay hybrid? | Logic mềm khả vi, nhị phân hóa khi đánh giá | Train được end-to-end mà vẫn cho luật rõ ràng |
| Prototype thủ công hay học? | Học | Mở rộng được lên 120 class, tránh thiên lệch |
| Nguồn giám sát concept? | Teacher hình học, nhãn mềm theo phân vị | Độc lập với nhãn action (C1), không cần annotation |
| Có object/context không? | Không | Skeleton không có thông tin vật thể. Giới hạn này được báo cáo |
| Dataset | NTU120 X-Sub trước | Khó nhất. Các split khác chỉ cần đổi config |
| Backbone | Freeze ở S1–S3, mở một phần ở S4 | Tách đóng góp của symbolic khỏi fine-tune |
| Config cũ | Bỏ hẳn các cờ `symbolic_*` cũ cho bản large | Đã thực hiện ở Phase 0 |
