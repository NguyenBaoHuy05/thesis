# Kế hoạch xây dựng lại Symbolic Layer cho Hyper-GCN

## 1. Mục tiêu và phạm vi

### 1.1. Mục tiêu chính

Xây dựng lại toàn bộ phần symbolic theo hướng:

```text
Skeleton sequence
    -> Hyper-GCN backbone
    -> semantic primitives
    -> spatial/temporal/interaction relations
    -> structured event graph
    -> symbolic action reasoning
    -> action logits
```

Phần symbolic mới phải là một thành phần có cấu trúc, có thể kiểm tra và can thiệp được, thay vì chỉ là một concept auxiliary head gắn vào feature cuối của backbone.

### 1.2. Phạm vi

Giữ lại:

- Hyper-GCN backbone làm bộ trích xuất đặc trưng động học;
- graph NTU 25 joints;
- khả năng xử lý tối đa hai người;
- các checkpoint và config backbone hiện tại nếu còn tương thích;
- các công cụ logging/evaluation có thể tái sử dụng sau khi thay đổi interface.

Loại bỏ:

- `SymbolicHyperConceptBridge` hiện tại;
- concept-to-action table làm trung tâm của toàn bộ suy luận;
- symbolic residual fusion vào feature backbone;
- black-box FC được dùng làm action predictor trong symbolic pipeline;
- cơ chế suy ra concept target trực tiếp từ action support nếu nó không có semantic evidence độc lập;
- kiến trúc `action_decoder="none"`/`"prototype"` hiện tại nếu không còn phù hợp với thiết kế mới.

### 1.3. Nguyên tắc thiết kế

1. **Symbolic representation phải có cấu trúc**: concept không chỉ là vector độc lập mà phải có loại, thuộc tính, scope và quan hệ.
2. **Tách perception khỏi reasoning**: neural encoder phát hiện primitive; symbolic engine kết hợp primitive thành event/action.
3. **Không để action classifier bypass symbolic graph**: action prediction phải phụ thuộc vào structured symbolic representation trong symbolic mode.
4. **Có thể truy vết**: mỗi action prediction phải truy ngược được về primitive, relation, temporal phase và actor.
5. **Có thể can thiệp**: xóa, thay thế hoặc khóa một primitive/relation phải làm thay đổi output theo cách đo được.
6. **Không phụ thuộc cứng vào một ontology duy nhất**: ontology phải là dữ liệu/config có schema rõ ràng.
7. **Giữ neural residual của backbone**: chỉ loại bỏ symbolic residual/fusion cũ, không phá residual giúp backbone ổn định.
8. **Ưu tiên correctness và kiểm chứng** trước khi tối ưu accuracy.

---

## 2. Kiến trúc đề xuất

## 2.1. Tổng quan

```text
Input skeleton [N, C, T, V, M]
        |
        v
Hyper-GCN backbone
        |
        +--> multi-level feature maps
        |      - joint-level
        |      - body-part-level
        |      - action-level
        |
        +--> dynamic graph evidence
        |      - incidence H
        |      - hyperedge weights W
        |      - dynamic adjacency G
        |
        v
Semantic Perception Layer
        |
        +--> pose/state primitives
        +--> motion primitives
        +--> geometric relations
        +--> temporal evidence
        +--> person interaction evidence
        |
        v
+ Structured Event Graph
        |
        +--> nodes: actor, body part, object/context, event
        +--> edges: spatial, temporal, causal/compositional
        +--> attributes: direction, speed, confidence, phase
        |
        v
Symbolic Reasoning Layer
        |
        +--> rule/constraint reasoning
        +--> differentiable relation composition
        +--> action prototype matching
        |
        v
Action logits + explanation trace
```

## 2.2. Backbone evidence interface

Backbone cần cung cấp một output có tên và schema rõ ràng, không trả các tensor trung gian vô danh:

```python
BackboneEvidence(
    final_features,
    joint_features,
    part_features,
    temporal_features,
    static_adjacency,
    dynamic_incidence,
    dynamic_hyperedge_weights,
    dynamic_adjacency,
    layer_metadata,
)
```

### Các tầng đề xuất

| Tầng | Nguồn | Vai trò symbolic |
|---|---|---|
| Joint level | `l1-l3` | pose/state và quan hệ cục bộ |
| Part level | `l4-l6` | chuyển động của tay, chân, đầu, thân |
| Action level | `l7-l9` | phase và composition cấp hành động |
| Dynamic graph | từng `HYPERGC` | bằng chứng quan hệ theo sample |
| Temporal branches | từng TCN block | short/medium/long-term motion |

Backbone hiện tại phải được mở rộng để expose dynamic graph evidence. `hyper_joint` tĩnh không được coi là dynamic explanation.

---

## 3. Semantic ontology mới

## 3.1. Không dùng action-to-concept table làm trung tâm

File dạng:

```text
action_class -> concept support
```

chỉ được dùng làm weak prior hoặc compatibility data, không phải nguồn semantic duy nhất.

Semantic ontology mới phải mô tả các primitive độc lập hơn với action label.

## 3.2. Các nhóm primitive

### A. Actor và body part

```text
actor_1
actor_2
head
torso
left_arm
right_arm
left_hand
right_hand
left_leg
right_leg
left_foot
right_foot
```

### B. Pose/state

```text
hand_near_head
knee_bent
torso_leaning_forward
arms_crossed
body_upright
```

Mỗi state nên có:

- actor;
- body part;
- thời gian hoặc temporal interval;
- confidence;
- evidence joints.

### C. Motion primitive

```text
move_up
move_down
move_forward
move_backward
move_toward
move_away
rotate
extend
flex
raise
lower
stop
repeat
```

Motion primitive nên được dự đoán từ trajectory/feature chứ không gán trực tiếp từ action label.

### D. Geometric relation

```text
above
below
left_of
right_of
near
far
overlaps
aligned_with
```

Các relation hình học phải chuẩn hóa theo torso/body scale, tránh phụ thuộc tuyệt đối vào camera hoặc vị trí người trong frame.

### E. Temporal relation

```text
before
after
during
starts_with
ends_with
repeated
overlaps_in_time
```

Temporal relation nên được suy ra từ các interval/phase, không chỉ từ bốn bin cố định.

### F. Interaction relation

```text
actor_1_near_actor_2
actor_1_moves_toward_actor_2
actor_1_moves_away_from_actor_2
hands_contact
actors_move_together
actors_move_apart
```

Đây là nhóm quan trọng cho các hành động như hugging, shaking hands, giving object, following và exchange things.

## 3.3. Schema ontology

Mỗi primitive/relation nên có schema tương tự:

```yaml
name: hand_moves_up
kind: motion
arity: 1
arguments:
  - actor
  - body_part
allowed_scopes:
  - temporal
evidence:
  joints: [7, 11, 21, 22, 23, 24]
  feature_levels: [joint, part]
parameters:
  direction: vector
  speed: scalar
  confidence: scalar
```

Không hard-code toàn bộ ontology vào Python. Python chỉ kiểm tra schema và thực thi operator.

---

## 4. Các module code mới

## 4.1. `model/backbone_evidence.py`

Định nghĩa dataclass/type cho output của backbone:

- feature maps;
- graph evidence;
- temporal metadata;
- actor mask;
- layer names.

Mục tiêu là tránh coupling symbolic layer với vị trí tensor nội bộ của `Model.forward`.

## 4.2. `model/hypergcn_evidence.py`

Các thay đổi chính:

1. Mở rộng `HYPERGC.forward()` để trả:
   - static adjacency;
   - dynamic incidence `H`;
   - hyperedge weight `W`;
   - dynamic adjacency `G`;
   - normalized adjacency.
2. Giữ backward-compatible mode nếu classifier cũ vẫn cần.
3. Gắn tên cho từng layer/hypergraph.
4. Không coi `hyper_joint` là activation.

## 4.3. `model/semantic_perception.py`

Chịu trách nhiệm chuyển backbone evidence thành primitive logits:

- `PoseStateHead`;
- `MotionPrimitiveHead`;
- `SpatialRelationHead`;
- `TemporalEvidenceHead`;
- `InteractionHead`.

Mỗi head phải trả:

```python
PrimitiveEvidence(
    logits,
    probabilities,
    arguments,
    support_mask,
    temporal_interval,
    source_features,
)
```

## 4.4. `model/structured_event_graph.py`

Xây graph có cấu trúc:

```text
nodes:
  actor
  body_part
  state
  motion
  event

edges:
  part_of
  connected_to
  above/below/near
  before/after
  causes/composes
  interacts_with
```

Module này không được tự ý biến mọi feature thành dense fully-connected graph. Chỉ tạo các relation được schema cho phép.

## 4.5. `model/symbolic_reasoner.py`

Reasoner gồm ba nhóm:

### Rule layer

Các rule đơn giản và kiểm tra được:

```text
move_up(hand) + near(hand, head) -> hand_lift_to_head
extend(leg) + move_forward(foot) -> leg_kick_phase
near(actor_1, actor_2) + arms_around -> hugging_evidence
```

### Differentiable composition layer

Cho phép học soft composition nhưng giữ nguyên semantics:

- conjunction;
- disjunction;
- temporal composition;
- relation confidence;
- contradiction penalty.

### Action prototype layer

Action prototype phải mô tả event graph, không chỉ là vector 102 concept:

```yaml
action: kicking
required:
  - motion: extend
  - body_part: leg
  - direction: forward
optional:
  - speed: fast
forbidden:
  - both_feet_stationary
```

## 4.6. `model/symbolic_output.py`

Output symbolic phải gồm:

```python
SymbolicOutput(
    action_logits,
    primitive_probabilities,
    event_graph,
    rule_activations,
    explanation_trace,
    intervention_handles,
)
```

`action_logits` là output duy nhất dùng trong symbolic mode. Không chạy thêm FC bypass.

---

## 5. Loss và training strategy

## 5.1. Tách các mục tiêu

Loss đề xuất:

```text
L_total =
    L_action
  + lambda_primitive * L_primitive
  + lambda_relation * L_relation
  + lambda_temporal * L_temporal
  + lambda_interaction * L_interaction
  + lambda_structure * L_structure
  + lambda_consistency * L_consistency
  + lambda_sparsity * L_sparsity
```

Không nên dùng một `symbolic_graph_weight` duy nhất cho toàn bộ symbolic supervision.

## 5.2. Khi thiếu annotation primitive

NTU action labels không đủ để supervise đầy đủ primitive. Cần ba nguồn:

1. **Pseudo-label hình học**
   - vị trí tương đối;
   - velocity/acceleration;
   - body-part distance;
   - actor distance.
2. **Weak labels từ action**
   - chỉ dùng như prior;
   - không coi là ground truth tuyệt đối.
3. **Self-supervised temporal objectives**
   - order prediction;
   - masked motion reconstruction;
   - phase consistency;
   - augmentation consistency.

## 5.3. Training stages

### Stage 0: kiểm tra backbone

- đảm bảo output backbone không đổi khi symbolic tắt;
- kiểm tra shape và device;
- serialize dynamic graph evidence;
- benchmark memory/time.

### Stage 1: primitive perception

- freeze backbone trước;
- train pose/motion/relation heads;
- kiểm tra primitive metrics và calibration;
- chưa train action reasoner phức tạp.

### Stage 2: structured event graph

- bật graph construction;
- train temporal/interaction composition;
- kiểm tra graph validity và sparsity.

### Stage 3: symbolic action reasoning

- train action prototype/rule composition;
- cấm bypass FC;
- đánh giá action và explanation đồng thời.

### Stage 4: controlled fine-tuning

- chỉ mở một phần backbone;
- giữ ontology constraints;
- so sánh với frozen-backbone baseline.

---

## 6. Config mới

Không tiếp tục mở rộng config cũ bằng các cờ `symbolic_*` rời rạc. Tạo schema config mới, ví dụ:

```yaml
symbolic:
  enabled: true
  mode: structured_reasoning
  ontology: concepts/ntu_semantic_ontology_v1.yaml
  action_prototypes: concepts/ntu_action_prototypes_v1.yaml
  expose_backbone_evidence: true
  use_dynamic_hypergraph: true
  feature_levels: [joint, part, action]
  temporal_reasoning: interval_graph
  interaction_reasoning: true
  action_bypass: false
  freeze_backbone: true
  losses:
    action: 1.0
    primitive: 0.2
    relation: 0.2
    temporal: 0.1
    interaction: 0.2
    structure: 0.01
    consistency: 0.1
```

Các config cũ trong `config/symbolic/` và `config/symbolic_large/` phải được đánh dấu legacy hoặc chuyển sang schema mới, không âm thầm thay đổi semantics.

---

## 7. Dữ liệu và annotation cần chuẩn bị

## 7.1. Ontology files

Tạo:

- `concepts/ntu_semantic_ontology_v1.yaml`
- `concepts/ntu_motion_primitives_v1.yaml`
- `concepts/ntu_action_prototypes_v1.yaml`
- `concepts/ntu_joint_semantics_v1.yaml`

## 7.2. Derived evidence

Tạo pipeline sinh pseudo-label:

- normalized joint position;
- velocity;
- acceleration;
- body-part angle;
- pairwise distance;
- actor-to-actor distance;
- temporal phase candidates.

Pseudo-label phải lưu source và confidence, không ghi đè thành ground truth.

## 7.3. Annotation confidence

Mỗi label cần phân biệt:

```text
ground_truth
human_annotation
geometric_pseudolabel
action_weak_label
model_pseudo_label
```

Loss phải có khả năng weighting theo nguồn.

---

## 8. Evaluation plan

## 8.1. Action performance

- Top-1/Top-5 accuracy;
- macro-F1;
- per-class accuracy;
- confusion matrix;
- NTU cross-subject/cross-view/cross-set.

## 8.2. Primitive quality

- precision/recall/F1;
- calibration;
- temporal IoU cho interval;
- relation accuracy;
- interaction detection F1.

## 8.3. Explanation faithfulness

Phải có ít nhất:

1. **Concept/primitive deletion**
   - xóa primitive quan trọng;
   - đo giảm action logit/accuracy.
2. **Counterfactual intervention**
   - đổi `move_up` thành `move_down`;
   - đổi actor;
   - đổi temporal order.
3. **Sufficiency**
   - chỉ giữ evidence được giải thích;
   - action prediction còn đúng đến mức nào.
4. **Completeness**
   - symbolic graph có bao phủ đủ evidence cần cho action không.
5. **Consistency**
   - cùng semantic event dưới augmentation vẫn cho reasoning trace tương tự.
6. **Robustness**
   - missing person;
   - joint noise;
   - temporal crop;
   - viewpoint variation.

## 8.4. Anti-bypass checks

Automated tests phải xác nhận:

- symbolic mode không gọi `fc`;
- action logits phụ thuộc vào event graph;
- xóa toàn bộ primitive làm output suy giảm;
- không có feature residual bypass symbolic reasoner;
- explanation trace không được sinh sau khi action prediction đã hoàn tất theo kiểu post-hoc.

---

## 9. Lộ trình triển khai

### Phase 1: baseline và interface

- [ ] Chụp baseline accuracy của base/large.
- [ ] Tách output backbone thành `BackboneEvidence`.
- [ ] Expose `H`, `W`, `G` ở mỗi `HYPERGC`.
- [ ] Viết shape/device/serialization tests.
- [ ] Đảm bảo symbolic tắt không thay đổi output backbone.

### Phase 2: ontology và preprocessing

- [ ] Định nghĩa ontology schema.
- [ ] Tạo joint/body-part semantics.
- [ ] Viết motion geometry extractor.
- [ ] Tạo pseudo-label với confidence.
- [ ] Viết validator cho ontology/prototype.

### Phase 3: primitive perception

- [ ] Implement pose/state heads.
- [ ] Implement motion heads.
- [ ] Implement spatial relation heads.
- [ ] Implement interaction heads.
- [ ] Viết primitive metrics.

### Phase 4: event graph

- [ ] Implement typed nodes/edges.
- [ ] Implement temporal interval handling.
- [ ] Implement graph validity constraints.
- [ ] Implement graph logging/export.

### Phase 5: symbolic reasoner

- [ ] Implement rule operators.
- [ ] Implement differentiable composition.
- [ ] Implement action event prototypes.
- [ ] Cấm FC bypass.
- [ ] Viết intervention tests.

### Phase 6: training và đánh giá

- [ ] Train frozen-backbone symbolic model.
- [ ] So sánh với FC baseline.
- [ ] Chạy ablation từng nhóm primitive.
- [ ] Chạy faithfulness/counterfactual evaluation.
- [ ] Chỉ fine-tune backbone sau khi symbolic pipeline ổn định.

---

## 10. Ablation bắt buộc

| Thí nghiệm | Mục đích |
|---|---|
| Backbone + FC | baseline accuracy |
| Backbone + primitive heads | đo chất lượng semantic detection |
| Không dynamic hypergraph | đo đóng góp của `G` |
| Không temporal reasoning | đo đóng góp temporal graph |
| Không interaction reasoning | đo đóng góp multi-person |
| Không action prototype constraints | đo mức độ cần cấu trúc |
| Symbolic reasoner + FC bypass | kiểm tra symbolic dependency |
| Fine-tune backbone vs frozen | đo semantic drift |
| Ontology cũ vs ontology mới | kiểm tra lợi ích của thiết kế mới |
| Primitive deletion/counterfactual | đo faithfulness |

---

## 11. Rủi ro và cách kiểm soát

### Rủi ro 1: ontology quá lớn

Giải pháp:

- bắt đầu với primitive core;
- giới hạn số loại relation;
- thêm ontology theo ablation;
- dùng validator chống duplicate.

### Rủi ro 2: pseudo-label hình học không chính xác

Giải pháp:

- lưu confidence;
- không coi pseudo-label là ground truth;
- đánh giá riêng quality của pseudo-label;
- dùng consistency/self-supervision.

### Rủi ro 3: symbolic graph trở thành neural black box mới

Giải pháp:

- typed edges;
- giới hạn operator;
- log từng rule activation;
- kiểm tra counterfactual;
- cấm dense unconstrained relation.

### Rủi ro 4: action accuracy giảm mạnh

Giải pháp:

- dùng curriculum;
- giữ backbone frozen ở giai đoạn đầu;
- tăng dần action prototype complexity;
- báo cáo trade-off accuracy/faithfulness thay vì che giấu.

### Rủi ro 5: dynamic graph quá tốn bộ nhớ

Giải pháp:

- chỉ lưu top-k edges;
- tùy chọn detach graph evidence khi không train reasoner;
- log theo sample subset;
- benchmark trước khi bật toàn bộ layer.

---

## 12. Tiêu chí hoàn thành

Phần symbolic rebuild chỉ được coi là hoàn thành khi:

1. Có ontology và schema độc lập với action-to-concept CSV cũ.
2. Backbone expose được evidence có tên và có test.
3. Có ít nhất pose, motion, temporal và interaction primitives.
4. Action output đi qua structured event graph/reasoner.
5. Không có FC hoặc residual feature bypass trong symbolic mode.
6. Có explanation trace cho từng prediction.
7. Có counterfactual/intervention evaluation.
8. Có ablation chứng minh đóng góp của từng thành phần.
9. Có so sánh rõ với backbone + FC baseline.
10. Các giới hạn của pseudo-label và ontology được báo cáo minh bạch.

---

## 13. Quyết định cần chốt trước khi code

Các quyết định sau cần được thống nhất trước Phase 2:

1. Symbolic reasoner là rule-based hoàn toàn, differentiable, hay hybrid?
2. Có dùng action prototype thủ công hay học prototype từ event graph?
3. Mức độ phụ thuộc vào pseudo-label hình học cho primitive?
4. Có hỗ trợ object/context hay chỉ skeleton/person interaction?
5. Có yêu cầu symbolic output cho cả NTU60 và NTU120 ngay từ đầu không?
6. Giữ một config tương thích legacy hay chuyển hẳn sang schema mới?
7. Backbone được freeze hoàn toàn hay cho phép fine-tune ở phase cuối?

Khuyến nghị ban đầu:

```text
hybrid reasoner
+ frozen backbone
+ geometry-derived primitive supervision
+ typed event graph
+ differentiable soft rules
+ action prototype không dùng FC bypass
```

---

## 14. Trạng thái triển khai hiện tại

Phạm vi triển khai được giới hạn ở **Hyper-GCN large**:

- [x] Tạo `LargeSymbolicReasoner` độc lập với `SymbolicHyperConceptBridge` cũ.
- [x] Expose static/dynamic graph evidence từ `model.hypergcn_large.HYPERGC`.
- [x] Giữ evidence của `l1-l3`, `l4-l6`, `l7-l9` làm joint/part/action levels.
- [x] Tạo primitive và relation heads có schema tên rõ ràng.
- [x] Tạo structured event evidence và semantic action composition.
- [x] Loại bỏ FC bypass khi `symbolic_graph_enabled: True` ở large.
- [x] Thêm config large cho structured symbolic mode.
- [x] Thêm ontology YAML v1 và focused unit tests.
- [x] Bổ sung geometry-derived pseudo-labels độc lập khỏi action-name weak support.
- [x] Hoàn thiện typed event graph với node/edge schema và trace can thiệp.
- [x] Bổ sung rule composition, counterfactual intervention và confidence-weighted metrics.
- [ ] Chạy benchmark trên NTU120 sau khi môi trường có đủ dependency và dữ liệu.

`model.hypergcn_base` và các config base không thuộc phạm vi rebuild này; symbolic bridge cũ chỉ được giữ để tránh thay đổi ngoài scope.
