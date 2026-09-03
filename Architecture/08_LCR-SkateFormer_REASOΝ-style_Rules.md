# LCR-SkateFormer — Hướng cải tiến luật theo kiểu REASON

## Mục tiêu

Mục tiêu của đề xuất là giữ lại ưu điểm **iterative concept refinement** của **LCR-SkateFormer**, nhưng làm cho luật đầu ra có hình thức gần với **REASON**:

```text
concept → literal/NOT literal → conjunction rules → disjunction rules → action
```

Luật cuối cùng có thể đọc được dưới dạng:

```text
Drink Water ←
    (arm_raise AND hand_lift_to_face AND NOT leg_kick)
    OR
    (hand_grasp AND head_tilt_up)
```

## 1. Thay `KGUnionLayer` bằng CRL có forward nhị phân

### 1.1. Thêm `BinarizeSTE`

LCR hiện dùng trực tiếp xác suất concept liên tục. Cần thêm lớp binarization theo STE:

```python
hard = (concept_logits > 0).float()
soft = torch.sigmoid(temperature * concept_logits)
concept_binary = hard + soft - soft.detach()
```

Forward sẽ gần nhị phân để luật dễ trích xuất, còn backward vẫn truyền gradient mềm.

Nên giữ hai chế độ:

- `soft_logic`: dùng cho warm-up và huấn luyện ổn định;
- `ste_logic`: dùng cho giai đoạn reasoning và trích xuất luật.

Không nên bật STE cứng ngay từ epoch đầu.

### 1.2. Tách riêng Conjunction và Disjunction

Thay vì chỉ trả về tensor nối `[AND, OR]`, nên tạo các module rõ ràng:

```text
ConjunctionLayer: tạo các luật AND
DisjunctionLayer: tạo các luật OR
LogicUnionLayer: quản lý hai loại luật
```

Mỗi layer cần lưu:

- input node ID;
- output rule ID;
- trọng số cạnh;
- ngưỡng chọn cạnh;
- tên concept/literal của mỗi node.

## 2. Giữ iterative reasoning của LCR

Mỗi vòng LCR vẫn thực hiện:

```text
z_t
  ↓
literal_t = [concept_t, NOT concept_t]
  ↓
AND/OR rules
  ↓
action evidence y_t
  ↓
Δz_t
  ↓
z_(t+1) = z_t + β Δz_t
```

Điểm khác là mỗi vòng phải lưu rule trace có cấu trúc, không chỉ lưu tensor activation:

```python
trace[t] = {
    "active_literals": ..., 
    "active_and_rules": ...,
    "active_or_rules": ...,
    "action_evidence": ...,
    "delta_concept": ...,
}
```

## 3. Thiết kế logic layer đề xuất

### L0 — Literal layer

Từ concept logits tạo:

```text
c1, c2, ..., cK,
NOT c1, NOT c2, ..., NOT cK
```

### L1 — Conjunction layer

Mỗi output là một conjunction rule:

```text
R1 = c_arm_raise AND c_hand_lift_to_face
R2 = c_head_tilt_up AND NOT c_leg_kick
```

### L2 — Disjunction layer

Ghép các conjunction rule thành action evidence:

```text
Drink Water = R1 OR R2
```

### Action layer

Trong cấu hình strict explainability, action chỉ được phép nhận output từ L2:

```text
action_logits = LogicActionLayer(L2_rules)
```

Không nối trực tiếp `fused_feature` hoặc hidden neural feature vào action output trong cấu hình này.

## 4. Cách trích xuất luật

Sau huấn luyện, dùng ngưỡng để chuyển trọng số mềm thành cạnh logic:

```text
weight >= threshold → giữ literal trong rule
weight < threshold   → loại literal
```

Quy trình:

1. lấy trọng số AND/OR;
2. threshold trọng số;
3. loại rule không kích hoạt hoặc kích hoạt quá thấp;
4. ánh xạ node ID về tên concept;
5. gom các literal thành AND clause;
6. gom các clause theo từng action thành OR;
7. sắp xếp theo activation frequency hoặc contribution.

Kết quả mong muốn:

```text
Action: Drink Water

Rule 1, confidence=0.91:
    arm_raise AND hand_lift_to_face AND NOT leg_kick

Rule 2, confidence=0.76:
    hand_grasp AND head_tilt_up
```

## 5. KG chỉ nên làm prior, không thay thế rule extraction

KG dùng để:

- khởi tạo trọng số rule;
- regularize `require`, `alongside`, `excludes`;
- phạt rule vi phạm ontology;
- hỗ trợ đặt tên và kiểm tra tính hợp lý của rule.

Rule cuối cùng vẫn phải được trích xuất từ trọng số đã học. Cần báo cáo `prior retention` để biết rule giữ lại bao nhiêu tri thức KG.

## 6. Các ràng buộc để explanation trung thành

### 6.1. Không dùng neural bypass trong mô hình chính

Tách hai cấu hình:

- **LCR-Strict-Logic:** action chỉ đi qua L2 logic, dùng để đánh giá explainability;
- **LCR-Residual:** cho phép neural residual, dùng để đánh giá accuracy.

Không dùng kết quả của LCR-Residual để tuyên bố rằng toàn bộ quyết định được tạo bởi luật.

### 6.2. Hạn chế skip connection không có tên

Skip connection từ hidden node có thể làm action phụ thuộc vào thông tin không thể diễn giải. Nếu sử dụng skip, phải:

- đặt tên cho node;
- ghi node đó vào trace;
- đưa nó vào rule extraction;
- có ablation không skip.

Khuyến nghị không dùng skip trong **LCR-Strict-Logic**.

### 6.3. Giới hạn độ dài và độ thưa của rule

Thêm regularization cho:

- số literal trong mỗi conjunction;
- số clause cho mỗi action;
- entropy của trọng số;
- số cạnh logic hoạt động.

Mục tiêu là tạo luật ngắn, thưa và dễ đọc thay vì một conjunction chứa hàng chục concept.

## 7. Loss và lịch huấn luyện

Nên dùng:

```text
L_total = L_action
        + λ_concept L_concept
        + λ_KG L_KG
        + λ_prior L_prior_retention
        + λ_sparse L_rule_sparsity
        + λ_len L_rule_length
        + λ_stable L_loop_stability
```

Lịch huấn luyện đề xuất:

1. warm-up concept head với soft logic;
2. bật KG rule bank với beta nhỏ;
3. chuyển dần sang STE hoặc tăng temperature;
4. fine-tune LCR end-to-end;
5. freeze hoặc giảm learning rate trước khi trích xuất luật.

## 8. Thực nghiệm bắt buộc

| Cấu hình | Mục đích |
|---|---|
| LCR soft logic | Baseline ổn định |
| LCR + STE | Đánh giá forward nhị phân |
| LCR + STE + KG | Đo tác dụng của tri thức |
| LCR-Strict-Logic | Đánh giá explanation trung thành |
| LCR-Residual | Đo trade-off accuracy–explainability |
| LCR không refinement (`β=0`) | Kiểm tra lợi ích của looping |

Metric cần có:

- Top-1/Top-5 action accuracy;
- concept macro-F1 và AUPRC;
- rule precision/coverage;
- rule fidelity;
- KG violation rate;
- counterfactual action drop;
- số literal trung bình mỗi rule;
- số rule trung bình mỗi action;
- accuracy/stability theo từng loop.

## Kết luận

Để tạo luật giống REASON, **LCR-SkateFormer** cần chuyển từ `soft weighted KGUnionLayer` sang một **Iterative CRL** gồm literal binarization, conjunction layer, disjunction layer và rule extraction có tên.

Thay đổi quan trọng nhất là:

1. forward gần nhị phân bằng STE;
2. tách rõ AND và OR;
3. lưu rule trace theo từng loop;
4. cấm neural bypass trong mô hình strict;
5. đánh giá explanation bằng counterfactual fidelity thay vì chỉ hiển thị trọng số.

Như vậy LCR vẫn giữ được ưu điểm suy luận lặp, đồng thời có thể sinh luật dạng:

```text
Action ← (Concept A AND Concept B) OR (Concept C AND NOT Concept D)
```

## 9. Kiến trúc chi tiết: Shared CRL + ConceptUpdateCell + KG Prior

### 9.1. Luồng tổng thể của LCR-SkateFormer

```text
Skeleton sequence X
        ↓
SkateFormer backbone
        ↓
Spatial head + Temporal head
        ↓
Initial concept logits z⁰ ∈ R^(B×72)
        ↓
Shared CRL
        ↓
AND/OR/NOT rule evidence rᵗ và action evidence yᵗ
        ↓
ConceptUpdateCell
        ↓
Δzᵗ
        ↓
zᵗ⁺¹ = zᵗ + βᵗΔzᵗ
        ↺ lặp K vòng
        ↓
Final Shared CRL
        ↓
Final action logits yᴷ
```

LCR vẫn dùng SkateFormer để tạo bằng chứng thị giác ban đầu. CRL chịu trách nhiệm tạo luật, còn `ConceptUpdateCell` cập nhật concept giữa các vòng. KG chỉ định hướng và kiểm soát CRL thông qua prior và regularization.

### 9.2. Shared CRL

Chỉ tạo một CRL dùng chung cho tất cả các vòng:

```text
z⁰ → CRL(shared) → update → z¹
z¹ → CRL(shared) → update → z²
z² → CRL(shared) → action
```

Các vòng dùng chung:

- trọng số conjunction;
- trọng số disjunction;
- mapping concept-to-rule;
- mapping rule-to-action.

Ví dụ cùng một rule `R17 = arm_raise AND hand_lift_to_face` có thể có activation `0.42` ở vòng đầu và `0.75` ở vòng sau. Luật không đổi, chỉ bằng chứng concept được cải thiện.

`shared CRL` nên là mô hình chính. `unshared CRL` chỉ nên dùng làm ablation vì làm tăng tham số và khó phân biệt cải thiện do iterative reasoning hay do tăng capacity.

### 9.3. Cấu trúc bên trong CRL

#### Literal layer

Với trạng thái concept `zᵗ`, tạo:

```text
cᵗ = sigmoid(zᵗ)
lᵗ = [cᵗ, 1 - cᵗ]
```

Với 72 concept, CRL nhận 144 literals, gồm concept dương và concept phủ định.

Để tạo luật gần với REASON, có thể dùng STE:

```python
hard = (z_t > 0).float()
soft = sigmoid(k * z_t)
c_t = hard + soft - soft.detach()
```

Forward gần nhị phân để dễ trích xuất luật; backward vẫn truyền gradient mềm. Nên warm-up với soft logic trước khi bật STE.

#### L1 conjunction layer

L1 tạo các luật AND cấp thấp:

```text
R1 = arm_raise AND hand_lift_to_face
R2 = hand_grasp AND head_tilt_up
R3 = NOT leg_kick AND arm_extend
```

#### L2 disjunction/composition layer

L2 kết hợp các rule cấp thấp:

```text
Q1 = R1 OR R2
Q2 = R1 AND R3
```

Action có thể được giải thích thành:

```text
Drink Water = Q1 OR Q4
```

Mỗi rule cần lưu input node ID, output rule ID, trọng số cạnh và tên literal; không nên chỉ trả về tensor hidden không có metadata.

### 9.4. ConceptUpdateCell

Sau mỗi lần CRL, `ConceptUpdateCell` nhận:

```text
zᵗ ∈ R^(B×72)
rᵗ ∈ R^(B×D_logic)
```

`rᵗ` có thể là phép nối của output L1 và L2. Cell chiếu logic evidence về concept space rồi tạo residual update:

```text
r_projected = Linear(rᵗ)       # về 72 chiều
Δzᵗ = MLP([zᵗ, r_projected])
zᵗ⁺¹ = zᵗ + βᵗΔzᵗ
```

Giới hạn `βᵗ` trong `[0, βmax]` để reasoning không ghi đè hoàn toàn bằng chứng thị giác ban đầu.

Ví dụ:

```text
Vòng 0: arm_raise=0.72, hand_lift_to_face=0.61, hand_grasp=0.32
Vòng 1: hand_grasp=0.48
Vòng 2: hand_grasp=0.57
```

### 9.5. Vai trò của KG prior

KG không thay thế CRL. KG được dùng ở ba vị trí:

```text
KG → khởi tạo trọng số L1
KG → concept consistency loss
KG → prior-retention regularization
```

Ví dụ với quan hệ:

```text
arm_raise require arm_extend
```

trọng số cạnh tương ứng ở L1 được khởi tạo cao hơn. Sau huấn luyện, rule vẫn được học từ dữ liệu nhưng bị regularize để không lệch quá xa KG.

Quan hệ KG có tên chỉ nên áp dụng trực tiếp ở layer mà node còn tương ứng với concept ontology. Hidden node của L2 không nên gán quan hệ concept trực tiếp nếu chưa có mapping ngữ nghĩa.

Các penalty cơ bản:

```text
require(a,b):   max(0, p_b - p_a)
alongside(a,b): max(0, τ - p_a p_b)
excludes(a,b):  p_a p_b
```

Ở đây cần ghi rõ quy ước `b` yêu cầu `a` đối với `require`.

### 9.6. Một vòng iterative hoàn chỉnh

```text
z⁰
 ↓
CRL(z⁰)
 ↓
r⁰, y⁰
 ↓
ConceptUpdateCell(z⁰, r⁰)
 ↓
Δz⁰
 ↓
z¹ = z⁰ + β⁰Δz⁰
```

Vòng tiếp theo dùng lại chính CRL:

```text
z¹ → CRL(shared) → r¹, y¹ → UpdateCell → z²
```

Sau K vòng, chạy CRL lần cuối trên `zᴷ` để tạo `yᴷ`. Reasoning trace nên lưu concept, literal, AND rule, OR rule, action evidence, concept delta và beta của từng vòng.

### 9.7. Strict explainability và cấu hình khuyến nghị

Nên tách hai biến thể:

```text
LCR-Strict-Logic:
  action chỉ nhận output từ CRL/prototype
  use_skip = false
  use_action_feedback = false

LCR-Residual:
  cho phép neural residual
  dùng để đo trade-off accuracy–explainability
```

Cấu hình mô hình chính:

```yaml
num_loops: 2
shared_weights: true
action_mode: prototype
use_skip: false
use_action_feedback: false
kg_init_mode: kg_primed
update_gate_init: 0.1
max_gate: 0.5
```

### 9.8. Loss và lịch huấn luyện

```text
L_total = L_action^K
        + λ_aux L_action^0:K-1
        + λ_concept L_concept^K
        + λ_early L_concept^0
        + λ_KG L_KG
        + λ_prior L_prior_retention
        + λ_sparse L_rule_sparsity
        + λ_stable L_loop_stability
```

Stability loss nên tính giữa từng cặp vòng:

```text
L_stable = mean_t |sigmoid(z^(t+1)) - sigmoid(z^t)|
```

Lịch huấn luyện:

1. warm-up concept head với soft CRL;
2. bật KG prior và beta nhỏ;
3. tăng dần mức binarization/STE;
4. fine-tune end-to-end với learning rate nhỏ cho backbone;
5. freeze hoặc giảm learning rate trước khi trích xuất luật.

### 9.9. Dạng luật đầu ra kỳ vọng

```text
Action: Drink Water

Rule 1:
  arm_raise AND hand_lift_to_face AND NOT leg_kick

Rule 2:
  hand_grasp AND head_tilt_up

Final action:
  Drink Water ← Rule 1 OR Rule 2
```

Kiến trúc này giữ được ba đặc tính của **LCR-SkateFormer**:

1. CRL tạo luật dạng `AND/OR/NOT`;
2. ConceptUpdateCell tạo iterative refinement;
3. KG prior định hướng nhưng không áp đặt toàn bộ luật.

## 10. Xây dựng ma trận quan hệ giữa các concept

### 10.1. Mục tiêu

Để dùng KG thống nhất cho cả `linear` và `prototype`, nên xây dựng ba ma trận **concept–concept** riêng biệt thay vì so sánh trực tiếp action–concept matrix với trọng số hidden của CRL:

```text
A_require    ∈ R^(72×72)
A_alongside  ∈ R^(72×72)
A_excludes   ∈ R^(72×72)
```

Quy ước:

```text
A_require[i,j] = 1
→ concept_i yêu cầu concept_j

A_alongside[i,j] = 1
→ concept_i thường xuất hiện cùng concept_j

A_excludes[i,j] = 1
→ concept_i loại trừ concept_j
```

`A_require` có hướng; `A_alongside` và `A_excludes` thường đối xứng.

### 10.2. Nguồn dữ liệu xây dựng ma trận

Nên kết hợp hai nguồn:

1. **Expert/ontology seed:** các quan hệ chắc chắn về mặt giải phẫu hoặc động tác.
2. **Concept matrix thống kê:** action–concept matrix dùng để tìm quan hệ ứng viên.

Ví dụ expert seed:

```text
hand_lift_to_face require arm_raise
leg_step require foot_step
arm_raise alongside hand_wave
arm_raise excludes hip_sit
```

Concept matrix chỉ cho biết action nào liên quan đến concept nào; nó không tự chứng minh quan hệ `require`. Vì vậy các quan hệ sinh từ thống kê phải được kiểm tra lại.

### 10.3. Các chỉ số thống kê

Với `M[a,i]` là action–concept matrix:

```text
cooccur(i,j) = Σa M[a,i]M[a,j]

P(j|i) = cooccur(i,j) / (Σa M[a,i] + ε)

Lift(i,j) = P(i,j) / (P(i)P(j) + ε)
```

Ứng viên `i require j` có thể thỏa:

```text
P(j|i) ≥ 0.90
support(i,j) đủ lớn
```

Ứng viên `i alongside j` có thể thỏa:

```text
Lift(i,j) ≥ 1.5
cooccur(i,j) ≥ min_support
```

Ứng viên `i excludes j` chỉ nên được tạo khi cả hai concept có support đủ lớn nhưng tỷ lệ đồng xuất hiện rất thấp. Không nên kết luận `excludes` chỉ vì cặp concept chưa xuất hiện trong dữ liệu.

### 10.4. Confidence thay cho nhãn nhị phân tuyệt đối

Nên lưu confidence của quan hệ:

```text
expert-confirmed: 1.0
data-supported:   0.4–0.7
uncertain:         0.1–0.3
```

Ví dụ:

```text
A_require[hand_lift_to_face, arm_raise] = 0.95
A_alongside[arm_raise, hand_wave] = 0.72
A_excludes[arm_raise, hip_sit] = 0.90
```

Các confidence này được dùng làm trọng số của KG loss, không nhất thiết biến thành constraint cứng.

### 10.5. Kết hợp expert và data

Ma trận cuối nên có nguồn quan hệ riêng:

```text
A_final = A_expert ∪ A_data
```

Quan hệ chuyên gia được ưu tiên hơn quan hệ chỉ phát hiện từ thống kê. Nếu hai nguồn mâu thuẫn, cần lưu conflict để kiểm tra ontology thay vì âm thầm ghi đè.

```python
relation_source[i, j] = "expert"
relation_source[i, j] = "data"
relation_source[i, j] = "conflict"
```

### 10.6. Đưa ma trận vào LCR

Ma trận concept–concept nên tác động vào `ConceptUpdateCell`, không so sánh trực tiếp với trọng số concept–hidden-rule:

```text
pᵗ = sigmoid(zᵗ)

kg_messageᵗ =
      A_requireᵀ pᵗ
    + A_alongside pᵗ
    - A_excludes pᵗ

Δzᵗ = MLP([zᵗ, kg_messageᵗ, rᵗ])

zᵗ⁺¹ = zᵗ + βᵗΔzᵗ
```

`rᵗ` là logic evidence từ CRL. Cách này giúp cùng một KG được sử dụng trong cả `linear` và `prototype` action head.

### 10.7. Ma trận learnable và prior retention

Không nên cố định hoàn toàn KG. Dùng ma trận prior để khởi tạo ma trận học được:

```text
A_learned = sigmoid(Θ)
```

và regularize:

```text
L_prior = ||A_learned - A_prior||₁
```

Model có thể điều chỉnh quan hệ khi dữ liệu phản ánh rằng prior ban đầu chưa chính xác, nhưng vẫn bị hạn chế để không quên toàn bộ tri thức.

### 10.8. Các consistency loss

Với `p_i` là xác suất concept:

```text
require(i,j):   max(0, p_i - p_j)
alongside(i,j): max(0, τ - p_i p_j)
excludes(i,j):  p_i p_j
```

Nếu `i require j`, khi `i` có xác suất cao thì `j` cũng cần có xác suất tương ứng. Các penalty nên tính trên `z⁰`, các trạng thái trung gian và `zᴷ`.

### 10.9. Quyết định kiến trúc

Để giảm số lượng configuration, mô hình chính nên dùng:

```text
SkateFormer
  → Initial concepts z⁰
  → Shared CRL
  → Concept–concept KG message
  → ConceptUpdateCell
  → z¹, z², ..., zᴷ
  → Final CRL action head
```

Khuyến nghị:

```yaml
num_loops: 2
shared_weights: true
use_action_feedback: false
use_skip: false
kg_mode: concept_relation_matrix
```

Action–concept matrix vẫn được giữ để đặt tên rule, đánh giá rule fidelity và thực hiện counterfactual test; không dùng nó để trừ trực tiếp với hidden CRL weights.
