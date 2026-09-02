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
