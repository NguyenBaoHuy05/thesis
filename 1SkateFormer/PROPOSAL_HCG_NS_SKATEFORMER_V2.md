# HCG-NS-SkateFormer v2.1

## Hierarchical Concept-Gated Neuro-Symbolic SkateFormer

**Tên viết tắt:** `HCG-NS-SkateFormer`  
**Phiên bản:** v2.1  
**Backbone:** SkateFormer  
**Mục tiêu:** thay thế concept phẳng + relation graph hiện tại bằng một ontology phân cấp, có điều kiện và có kiểm soát độ tin cậy. Model chính v2 giữ temporal concepts như các thuộc tính độc lập nhưng loại bỏ toàn bộ pairwise relation graph.

**Biến thể triển khai:** `HCG-NS-SkateFormer-RF` (*Relation-Free*).

---

## 1. Quyết định kiến trúc

Đây là một phiên bản kiến trúc mới, không phải chỉ là thay đổi một vài trọng số loss.

Repo triển khai hiện tại chỉ chứa pipeline HCG-NS-SkateFormer v2. Các module
Prototype-Graph/relation của v1 đã được loại bỏ; bản sao v1 nằm ngoài repo này.

### Vì sao cần v2?

Kết quả hiện tại cho thấy:

- Action Top-1 đạt khoảng 88.6% trên test.
- Concept exact-match bằng 0.
- Concept sparsity khoảng 0.32, cao hơn nhiều so với mật độ template khoảng 0.135.
- `kicking` và `kicking_something` gần như bị collapse vào một lớp.
- `point_finger` và `point_to_something` cũng bị collapse.
- Hai cặp trên trong concept matrix gần như chỉ khác concept `interaction_two_person`.
- Co-occurrence, order và exclusion hiện tại là các prior pairwise khó xác minh từ nhãn action cấp sequence.

Vấn đề chính không chỉ nằm ở learning rate. Concept ontology hiện tại chưa đủ phân biệt, chưa biểu diễn uncertainty và đang trộn spatial, interaction, state và temporal concepts trong một vector phẳng.

---

## 2. Ý tưởng cốt lõi

Thay vì dự đoán trực tiếp 86 concept độc lập trong ontology v2.1:

```text
SkateFormer → 86 unary concept probabilities → symbolic action score
```

v2 sử dụng cấu trúc phân cấp/factorized:

```text
SkateFormer backbone
        │
        ├── Context/participant head
        │       └── single-person / multi-person / interaction type
        │
        ├── Body-region head
        │       └── head / hand-arm / hip / leg-foot / interaction
        │
        ├── Primitive concept heads
        │       └── grasp / hold / raise / kick / step / point / ...
        │
        ├── Temporal heads
        │       └── direction / sequence / speed / duration
        │
        └── Hierarchical symbolic decoder
                └── action compatibility scores
```

Đây là một **DAG/factorized ontology**, không phải một cây cứng. Một action có thể đồng thời sử dụng nhiều nhánh:

```text
multi-person + upper-body + arm-hold + converging + sustained
```

---

## 3. Các tầng concept v2

### Level 0 — Context

Các concept cấp cao để định hướng không gian tìm kiếm:

- `single_person`
- `multi_person`
- `no_interaction`
- `person_person_interaction`
- `person_object_interaction` nếu skeleton/RGB có đủ bằng chứng

`multi_person` không được đồng nhất với `physical_interaction`. Hai người cùng xuất hiện chưa có nghĩa là họ đang tương tác.

### Level 1 — Body region

- `head`
- `hand_arm`
- `hip_torso`
- `leg_foot`
- `interaction_region`

Tầng này giúp model không phải cạnh tranh trực tiếp giữa toàn bộ 60 action ở bước đầu.

### Level 2 — Observable primitive

Ví dụ:

- `hand_grasp`
- `hand_open`
- `hand_point`
- `hand_lift_to_face`
- `arm_raise`
- `arm_hold`
- `arm_extend`
- `hip_bend`
- `leg_step`
- `leg_kick`
- `leg_lift`
- `foot_balance`

Chỉ giữ concept có thể quan sát hợp lý từ skeleton. Không tạo concept như `holding_object` nếu skeleton không chứa đủ bằng chứng về vật thể.

### Level 3 — Temporal attributes

- `motion_upward/downward/forward/backward`
- `sequence_hands_first/legs_first/body_first`
- `speed_slow/fast`
- `duration_brief/sustained`

Temporal concepts ở tầng này là weak labels và phải có confidence. Không dùng chúng như ground truth theo frame nếu không có annotation temporal.

### Level 4 — Action composition

Action không được đại diện bởi một vector binary phẳng duy nhất. Mỗi action được mô tả bằng các thành phần:

```text
required parent concepts
required primitive concepts
optional temporal concepts
ambiguous/unknown concepts
confidence của từng mapping
```

---

## 4. Cơ chế hierarchical gating

Không được loại action cứng chỉ vì một parent concept có xác suất thấp.

### Gating mềm

Với parent probability `p_parent` và child evidence `p_child`:

```text
p_child_effective = p_child × p_parent
```

hoặc dùng log-score:

```text
action_score = local_score + λ_parent × parent_score + child_score
```

Không dùng:

```python
if p_parent < threshold:
    remove_all_children()
```

vì một lỗi ở parent sẽ làm mất toàn bộ nhánh con.

### Hierarchical consistency

Thêm ràng buộc mềm:

```text
p_child <= p_parent
```

với loss:

```text
L_hierarchy = mean(relu(p_child - p_parent))
```

Ràng buộc này chỉ áp dụng khi quan hệ parent-child đã được xác minh.

---

## 5. Xử lý nhãn concept

Không sử dụng binary complement một cách mù quáng.

Mỗi action-concept mapping v2 có ba trạng thái:

```text
positive  — concept được hỗ trợ
negative  — concept bị loại trừ với độ tin cậy cao
unknown   — không đủ thông tin
```

Loss chỉ áp dụng trên positive/negative đã biết. `unknown` không bị phạt.

Metadata nên có các trường:

```text
concept_id
concept_name
level
parent_id
scope
observable_from_skeleton
label_status
confidence
source
```

---

## 6. Loss v2

```text
L_total = L_action
        + λ_context  L_context
        + λ_primitive L_primitive
        + λ_temporal L_temporal
        + λ_hierarchy L_hierarchy
        + λ_density L_density
        + λ_center L_center
```

### `L_action`

Cross-entropy cho 60 action classes. Luôn giữ loss này để action accuracy không bị hy sinh cho symbolic branch.

### `L_context`

Học các parent concepts như `single_person`, `multi_person`, `interaction_type`.

### `L_primitive`

Weak positive/negative BCE có mask `unknown`. Chỉ phạt negative khi mapping đủ tin cậy.

### `L_hierarchy`

Đảm bảo child evidence không vượt parent evidence một cách vô lý.

### `L_density`

Kiểm soát số lượng concept active theo từng branch, không dùng L1 toàn bộ output vì dễ làm model dự đoán toàn zero.

Một dạng phù hợp:

```text
target_count = số concept positive có confidence cao của action
L_density = SmoothL1(sum(p_concepts), target_count)
```

---

## 7. Temporal concepts không có pairwise relation

Temporal concepts vẫn được giữ lại vì chúng là thuộc tính hữu ích của action sequence:

```text
motion_upward/downward/forward/backward
sequence_hands_first/legs_first/body_first
speed_slow/fast
duration_brief/sustained
```

Chúng được dự đoán như các unary concepts độc lập từ feature sequence:

```text
SkateFormer sequence features → temporal concept heads → action compatibility
```

Model v2 không dùng:

```text
cooccur(A, B)
before(A, B)
exclusion(A, B)
temporal overlap relation
```

Điều này không có nghĩa backbone ngừng học thông tin temporal. SkateFormer vẫn xử lý toàn bộ chuỗi và temporal concept head vẫn có thể dùng temporal pooling/attention. Chỉ có symbolic layer không áp đặt quan hệ giữa hai concept.

`sequence_probs` nếu còn được dùng chỉ là evidence cấp clip cho một concept, không được diễn giải là temporal reasoning hoặc simultaneous reasoning.

---

## 8. Xử lý các hard pair hiện tại

### `kicking` và `kicking_something`

Không được mặc định rằng `interaction_two_person` đủ để phân biệt hai lớp. Cần kiểm tra:

1. Concept mapping có đúng với ý nghĩa action không.
2. Skeleton có quan sát được target/person/object không.
3. Interaction branch có phát hiện được sự khác biệt không.

Nếu khác biệt yêu cầu object/RGB information mà skeleton không chứa, phải đánh dấu cặp này là `modality-limited`, không ép symbolic concept giả.

### `point_finger` và `point_to_something`

Tương tự, cần xác minh khác biệt là:

- hướng cử chỉ;
- target là người/vật;
- hay chỉ là khác biệt annotation của NTU.

Nếu chỉ khác một concept không quan sát được, action residual head hoặc RGB feature là cần thiết cho phân biệt đầy đủ.

---

## 9. Kiến trúc module đề xuất

Các module mới:

```text
model/hierarchical_concept_decoder.py
model/hierarchical_symbolic_decoder.py
model/concept_ontology.py
```

`hierarchical_symbolic_decoder.py` chỉ tính unary concept compatibility và hierarchical gating; không chứa `cooccur_raw`, `order_raw` hoặc `exclusion_raw`.

Các công cụ:

```text
tools/build_concept_hierarchy_v2.py
tools/audit_concept_identifiability.py
tools/audit_action_concept_collisions.py
tools/evaluate_hierarchical_concepts.py
```

Các artifact:

```text
concepts/ntu_concept_hierarchy_v2_1.csv
concepts/ntu_action_concept_v2_1.csv
concepts/ntu_action_concept_v2_1_labels.csv
concepts/ntu_concept_metadata_v2_1.csv
config/train/HCG_NS_SkateFormer_ntu60_cs.yaml
```

Artifact không còn được pipeline v2 sử dụng:

```text
concepts/ntu_order_pairs*.csv
config/train/PrototypeGraph_*.yaml
```

---

## 10. Training schedule v2

### Stage 0 — Ontology audit

- Kiểm tra concept mapping.
- Tìm action rows giống nhau hoặc chỉ khác một concept.
- Đánh dấu concept không quan sát được từ skeleton.

### Stage 1 — Parent grounding

- Freeze SkateFormer backbone.
- Train context/participant heads.
- Learning rate khoảng `1e-4`.
- Không có symbolic relation để bật.

### Stage 2 — Primitive grounding

- Train body-region và primitive concept heads.
- Bật hierarchical consistency.
- Theo dõi parent/child F1 riêng biệt.

### Stage 3 — Symbolic composition

- Bật unary action compatibility.
- Bật soft hierarchical gating.
- Không dùng co-occurrence, order hoặc exclusion relation.

### Stage 4 — Optional adaptation

- Chỉ unfreeze stage cuối của backbone.
- Learning rate nhỏ hơn decoder ít nhất 10 lần.
- Không chọn checkpoint bằng test metric; pipeline dùng checkpoint của epoch cuối.

---

## 11. Đánh giá bắt buộc

Không chỉ báo cáo Top-1.

### Action metrics

- Top-1, Top-5.
- Macro accuracy.
- Confusion matrix.
- Accuracy của hard pairs.

### Hierarchical concept metrics

- Parent F1.
- Child F1.
- Micro/macro F1.
- Exact match, chỉ dùng như metric rất nghiêm ngặt.
- Sparsity so với target density.
- Brier score và ECE.

### Symbolic metrics

- Hierarchy violation rate.
- Unary rule coverage.
- Score delta khi bỏ từng concept branch.
- Prediction change khi can thiệp concept.

### Tiêu chí thành công tối thiểu

HCG-NS-SkateFormer v2 được đánh giá bằng test một lần sau khi train đầy đủ nếu:

1. Test Top-1 được báo cáo cùng protocol với SkateFormer baseline nếu cần so sánh.
2. Hai hard pair collapse được cải thiện rõ rệt.
3. Parent concept có F1 đáng tin cậy.
4. Sparsity giảm về gần target density hơn.
5. Brier/ECE tốt hơn hoặc không xấu hơn đáng kể.
6. Hierarchical concept contribution được chứng minh bằng ablation.

---

## 12. Các run cần so sánh

```text
B0: SkateFormer baseline
B1: HCG-NS-SkateFormer-RF, unary temporal concepts + hierarchy
B2: HCG-NS-SkateFormer-RF không hierarchical gating
B3: HCG-NS-SkateFormer-RF + optional residual action head cho modality-limited pairs
```

Pipeline chính dùng toàn bộ official train split và chỉ đánh giá một lần trên
official test split. Nếu cần tuning hyperparameter, phải tạo một experiment
riêng; không được dùng test để chọn checkpoint.

---

## 13. Kết luận thiết kế

HCG-NS-SkateFormer v2 không cố biến mọi action thành một danh sách binary concept phẳng. Nó học:

```text
context → body/interaction branch → primitive → temporal attribute → action
```

Các unary concept chỉ được dùng khi có evidence hoặc confidence phù hợp. Không tạo pairwise relation giả từ nhãn action cấp sequence. Khi skeleton không đủ thông tin để phân biệt hai action, hệ thống phải biểu diễn uncertainty hoặc dùng modality bổ sung thay vì tạo một symbolic rule không có cơ sở.

**Tên model chính thức được đề xuất:**

> Hierarchical Concept-Gated Neuro-Symbolic SkateFormer — Relation-Free v2 (HCG-NS-SkateFormer-RF)

---

## 14. Trạng thái triển khai

Đã triển khai pipeline v2 độc lập, không có relation graph và không chia validation:

```text
main_hcg_ns.py
model/hcg_ns_model.py
model/hierarchical_concept_decoder.py
model/hierarchical_symbolic_decoder.py
model/concept_ontology.py
```

Artifact NTU60 của v2.1 gồm 86 concept được curate lại, không copy nguyên
mapping cũ. `ntu_action_concept_v2_1.csv` chứa positive support; file
`ntu_action_concept_v2_1_labels.csv` ghi rõ positive/unknown, confidence và
modality limitation cho từng mapping.

```text
concepts/ntu_concept_metadata_v2_1.csv
concepts/ntu_action_concept_v2_1.csv
concepts/ntu_action_concept_v2_1_labels.csv
concepts/ntu_concept_hierarchy_v2_1.csv
config/train/HCG_NS_SkateFormer_ntu60_cs.yaml
```

Các audit/evaluation tool cũng đã có trong `tools/`. Chạy theo thứ tự:

```bash
python tools/build_concept_hierarchy_v2.py --version v2_1
python tools/audit_concept_identifiability.py --concept-csv concepts/ntu_action_concept_v2_1.csv --metadata-csv concepts/ntu_concept_metadata_v2_1.csv --num-concepts 86
python tools/audit_action_concept_collisions.py --concept-csv concepts/ntu_action_concept_v2_1.csv --num-concepts 86
python main_hcg_ns.py --config config/train/HCG_NS_SkateFormer_ntu60_cs.yaml
python tools/evaluate_hierarchical_concepts.py work_dir/hcg_ns/ntu60_cs/test_predictions.npz
```

`main_hcg_ns.py` train toàn bộ train split, lưu `final_model.pt`, sau đó mới
chạy test và sinh `test_metrics.json`, `test_predictions.npz` cùng rulebook.

---


