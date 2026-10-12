# Hyper-GCN V2 — Grounded Concept Hypergraph

**Trạng thái:** Core V2 đã triển khai; dataset experiments/ablations chưa chạy  
**Ngày:** 2026-10-08  
**Phạm vi:** V2A được đổi tên thành **V2**. Không triển khai V2B/typed semantic concept-action matrix.

## Trạng thái triển khai

Core V2 đã được triển khai ngày 2026-10-08:

- [x] Validated concept-to-joint masks cho 102 concepts.
- [x] Hard-grounded encoder trên `[N,M,C,T,V]`.
- [x] Bốn temporal bins, concept-conditioned attention và person padding mask.
- [x] Explicit pair tokens cho participant/interaction concepts.
- [x] Concept-token hypergraph message passing `102 × 22`.
- [x] IDF-weighted asymmetric Tversky decoder; Bernoulli/cosine ablations vẫn có cùng API.
- [x] Frozen backbone/BN và exclusive concept bottleneck.
- [x] Grounding/hypergraph JSON, static audit JSON/CSV và concept diagnostics.
- [x] CPU unit tests và full-model forward/backward smoke test.

Chưa thực hiện vì cần dataset/checkpoint/GPU experiment:

- [ ] Train V2 trên NTU120.
- [ ] Chạy V1/V2 và D0/D1/D2 ablations cùng seed/budget.
- [ ] Chạy raw-input/feature-map occlusion trên evaluation set.
- [ ] Tổng hợp faithfulness và action accuracy report.

Kiểm tra kỹ thuật mới nhất (không chạy NTU/GPU training):

- `python -m unittest discover -s tests -q`: **26 tests, OK**.
- CPU forward theo config V2: action logits `[2,120]`, concept logits `[2,102]`.
- CPU backward symbolic-only: grounder có gradient, backbone không có gradient.
- `git diff --check` và `py_compile`: **OK**.

## 1. Mục tiêu và phạm vi

V2 nối trực tiếp biểu diễn spatiotemporal joint của backbone Hyper-GCN với từng concept bằng body mapping rõ ràng. Mỗi concept chỉ lấy bằng chứng từ đúng joint/body part và person scope, sau đó mới truyền qua learnable concept hypergraph V1 và IDF-weighted asymmetric Tversky action decoder.

Mục tiêu:

- Tận dụng trực tiếp output joint-level của Hyper-GCN.
- Ground concept vào bộ phận cơ thể quan sát được.
- Giữ 102 concept là các symbol có danh tính ổn định và can thiệp được.
- Giữ learnable concept hypergraph `102 × 22` của V1.
- Giữ concept-action matrix `120 × 102` cố định nhưng thay balanced Bernoulli matching bằng asymmetric Tversky matching.
- Không tạo đường phân loại action đi vòng qua concept bottleneck.
- Tương thích official pretrained Hyper-GCN checkpoint.

Ngoài phạm vi V2:

- Text encoder hoặc semantic embedding.
- Typed matrix `required/supportive/contradictory/unknown`.
- Hypergraph riêng cho từng sample.
- Fine-tune toàn bộ backbone ngay từ đầu.
- Thay dynamic hypergraph bên trong các block Hyper-GCN.

## 2. Kiến trúc mục tiêu

```text
Skeleton [N, C_in, T, V, M]
        ↓
Pretrained Hyper-GCN: l1 ... l9
        ↓  (giữ dynamic skeleton hypergraphs gốc)
Backbone map [N, M, C_b, T', V]
        ↓
Hard concept–joint grounding mask [K, V]
        ↓
Concept-specific temporal/person evidence
        ↓
Grounded concept tokens [N, K, D_c]
        ↓
Learnable concept hypergraph H [K, E]
        ↓
Concept logits [N, K]
        ↓
Fixed concept-action matrix [A, K]
        ↓ IDF-weighted asymmetric Tversky
        ↓
Action logits [N, A]
```

Kích thước NTU120 hiện tại:

- `K = 102` concept nodes.
- `E = 22` concept hyperedges.
- `A = 120` action classes.
- `V = 25` physical joints.
- `M = 2` persons.
- `C_b = 256` channels sau `l9`.
- `T'` là temporal resolution sau `l4` và `l7` stride 2.

## 3. V2 giữ tính symbolic như thế nào?

### 3.1. Symbol identity cố định

- Giữ nguyên concept ID, tên và thứ tự của 102 concept.
- Concept query `c` luôn tương ứng đúng concept `c` trong metadata.
- Không tạo latent slot vô danh hoặc cho phép hoán vị symbol.

### 3.2. Explicit grounding schema

- Mỗi concept có body-part mapping rõ ràng.
- Body mapping được chuyển thành hard joint mask.
- Concept không được đọc joint ngoài mask.
- Whole-body và interaction concept phải có scope tường minh.

### 3.3. Explicit ontology

- Giữ 22 hyperedge có tên thuộc body, group và scope.
- Incidence prior có thể đọc và xuất ra.
- Learnable incidence là residual quanh prior, không phải dense attention vô danh.

### 3.4. Exclusive concept bottleneck

- Action logits chỉ sinh từ concept logits.
- Original `self.fc` chỉ tồn tại để load official checkpoint, không được gọi trong prototype mode.
- Không thêm residual `global_feature -> action_logits`.

### 3.5. Explicit concept-action knowledge

- V2 giữ fixed concept-action matrix hiện tại; không học dense classifier mới.
- Balanced Bernoulli được thay bằng asymmetric Tversky để zero chỉ bị phạt mềm thay vì bị xem là hard negative.
- Matrix, matched/missing/extra evidence vẫn audit và can thiệp được.

### 3.6. Intervention và audit

- Ép concept ON/OFF và đo action score.
- Occlude body part và đo concept tương ứng.
- Xuất temporal/person grounding summary.
- Xuất learned hypergraph mỗi epoch.

## 4. Hiện trạng code (đã triển khai)

`Model.forward()` đã lấy output sau `l9` trước global pooling. Feature hiện được reshape:

```python
[N * M, C_b, T', V] -> [N, M, C_b, T', V]
```

rồi truyền trực tiếp vào `SymbolicHyperConceptBridge`. Vì vậy V2 không cần hook hoặc chạy backbone lần hai; bridge giữ trục person/joint tường minh, áp hard joint mask trước attention và không có global action residual.

Dynamic skeleton hypergraph trong `HYPERGC` vẫn được dùng nguyên vẹn. Concept hypergraph V2 là tầng symbolic phía trên, không thay thế nó.

## 5. Quyết định thiết kế

### 5.1. Input contract mới

```python
backbone_map: Tensor[N, M, C_b, T_prime, V]
```

Không dùng `[N, C_b, T_prime, M * V]` làm representation chính nữa. Việc này giữ ranh giới hai người, cho phép mask đúng joint và audit person contribution.

### 5.2. Hard grounding trước, attention mềm sau

```text
hard joint mask -> shared projection -> temporal/person attention
```

Không dùng attention toàn skeleton rồi chỉ thêm grounding loss mềm, vì mô hình có thể học shortcut.

### 5.3. Không global residual cho body-specific concept

- Body-specific concept chỉ đọc joint được cấp phép.
- Whole-body concept có mask toàn bộ physical joints.
- Interaction concept dùng hai người qua explicit pair representation.
- Không cộng global pooled backbone vector vào mọi concept token.

### 5.4. Giữ 102 concept, thay matching rule

- Không prune concept đồng thời với thay đổi grounding.
- Giữ `symbolic_action_mode: prototype` và fixed concept-action matrix.
- Thay balanced Bernoulli bằng IDF-weighted asymmetric Tversky.
- Sau khi V2 ổn định mới audit dead/redundant concepts.

## 6. Grounded concept encoder

### 6.1. Concept–joint mask

Tạo buffer:

```python
concept_joint_mask: BoolTensor[K, V]
```

Quy tắc:

- Mỗi concept có ít nhất một joint.
- Chỉ dùng 25 physical joints; virtual hyper-joints không xuất hiện sau `HYPERGC`.
- Multi-part concept dùng hợp các joint tương ứng.
- Whole-body concept dùng toàn bộ 25 joints.
- Joint index phải lấy từ NTU graph layout trong repository.

Audit metadata:

```python
concept_part_names: list[list[str]]
concept_joint_indices: list[list[int]]
```

### 6.2. Concept-specific joint pooling

V2-0 dùng normalized hard-masked mean:

```text
F: [N, M, C_b, T', V]
G: [K, V]
Z: [N, M, K, T', C_b]
```

```math
Z_{n,m,k,t,:} =
  \frac{\sum_v G_{k,v}F_{n,m,:,t,v}}{\sum_v G_{k,v}}
```

Masked mean được chọn trước learned joint attention vì dễ kiểm chứng, ít parameter và không cho phép shortcut. Nếu sau này thêm within-mask attention, hard mask vẫn phải áp trước softmax.

### 6.3. Shared channel projection

Chiếu `C_b -> D_c`:

```python
Linear(C_b, concept_dim)
LayerNorm(concept_dim)
GELU()
Dropout(...)
```

Khởi tạo:

```yaml
symbolic_concept_dim: 64
symbolic_grounding_dropout: 0.1
```

Shared projection hạn chế mỗi concept trở thành một classifier riêng quá mạnh.

### 6.4. Temporal grounding

V2-1 giữ thứ tự thời gian bằng 4 bin:

1. Adaptive average pool `T'` thành 4 bin có thứ tự.
2. Cộng learned temporal-position embedding.
3. Concept query attention qua 4 bin.
4. Log attention entropy và top bin.

```yaml
symbolic_temporal_bins: 4
```

### 6.5. Person handling

Không average `M` trước khi xét scope:

- `single-person`: concept query attention theo person rồi weighted sum.
- `interaction`: pair token từ mean, absolute difference và elementwise product của hai person tokens.
- `global`: aggregate các valid persons.

Tên scope phải map từ giá trị thực tế trong metadata sau audit, không hard-code theo giả định. Person padding phải bị mask trước softmax.

Output:

```python
concept_tokens: Tensor[N, K, D_c]
grounding_aux: dict  # temporal weights, person weights, valid-person mask
```

## 7. Concept hypergraph trong V2

Concept hypergraph nhận token thay vì pooled feature chung:

```python
concept_tokens: Tensor[N, 102, D_c]
incidence: Tensor[102, 22]
```

Giữ node-to-edge-to-node propagation của V1:

```math
E = D_e^{-1}H^T Z
```

```math
Z' = Z + \tanh(g_h)D_v^{-1}HW_eE
```

Concept logits được sinh riêng từ từng token sau message passing:

```math
l_c = w_c^T Z'_c + b_c
```

Giữ từ V1:

- 22 named hyperedges.
- Prior membership `0.95` và non-prior `0.01`.
- Learnable incidence residual và edge weights.
- Residual hypergraph gate.
- Anchor, sparsity, binarization, coverage và edge-weight regularization.
- JSON hypergraph snapshot.

Chưa làm trong V2:

- Không tạo `H(x)` theo sample.
- Không tăng số hyperedge.
- Không dùng action label/matrix để tạo concept hyperedge.

## 8. Dự đoán action trong V2

Action được dự đoán **sau concept hypergraph**, chỉ từ 102 concept logits. Không dùng original blackbox `self.fc`.

Đường forward đầy đủ:

```text
grounded concept tokens [N, 102, D_c]
        ↓ concept hypergraph
refined concept tokens [N, 102, D_c]
        ↓ concept heads
concept logits z [N, 102]
        ↓ sigmoid
concept probabilities p [N, 102]
        ↓ fixed matrix M [120, 102]
        ↓ IDF-weighted asymmetric Tversky
action logits s [N, 120]
        ↓ argmax / cross-entropy
predicted action
```

### 8.1. Lý do thay balanced Bernoulli

Balanced Bernoulli dùng cả `log(p_c)` cho ô `1` và `log(1-p_c)` cho ô `0`. Trong matrix hiện tại, nhiều ô `0` chỉ có nghĩa là concept chưa được gán, không chắc chắn là concept mâu thuẫn. Vì vậy decoder cũ có thể tạo negative supervision sai và ép concept profile trở thành mã lớp.

V2 giữ matrix binary để bảo toàn phạm vi, nhưng thay matching rule bằng asymmetric Tversky. Zero vẫn tạo `extra evidence` penalty nhỏ, không còn là hard negative ngang với concept mâu thuẫn đã biết.

### 8.2. IDF concept weights

Concept xuất hiện ở quá nhiều action có ít sức phân biệt hơn. Với `freq_c` là số action chứa concept `c`:

```math
w_c = \operatorname{clip}\left(
\log\frac{A+1}{freq_c+1},
w_{min}, w_{max}
\right)
```

Khởi tạo:

```yaml
symbolic_action_idf_weighting: true
symbolic_action_idf_min: 0.5
symbolic_action_idf_max: 2.0
```

IDF được tính một lần từ fixed matrix, đăng ký như buffer và export trong audit. Không học IDF từ validation/test distribution.

### 8.3. Asymmetric Tversky score

Với action `a`, định nghĩa:

```math
I_a = \sum_c w_c M_{a,c}p_c
```

```math
Missing_a = \sum_c w_c M_{a,c}(1-p_c)
```

```math
Extra_a = \sum_c w_c(1-M_{a,c})p_c
```

Tversky similarity:

```math
T_a = \frac{I_a}
{I_a + \alpha Missing_a + \beta Extra_a + \epsilon}
```

Trong đó:

- `I_a`: concept evidence khớp prototype.
- `Missing_a`: concept trong prototype nhưng chưa được phát hiện.
- `Extra_a`: concept được phát hiện nhưng không có trong prototype.
- `alpha`: mức phạt missing evidence.
- `beta`: mức phạt extra/unknown evidence.

Thiết lập đầu:

```yaml
symbolic_action_decoder: tversky
symbolic_action_missing_weight: 1.0   # alpha
symbolic_action_extra_weight: 0.15    # beta
symbolic_action_temperature: 0.15
```

`beta < alpha` là quyết định quan trọng: thiếu concept đã biết bị phạt mạnh, trong khi concept nằm ở ô `0` chỉ bị phạt nhẹ.

### 8.4. Chuyển similarity thành action logits

Không đưa trực tiếp score `[0,1]` vào cross-entropy. Chuyển thành log-odds và temperature scaling:

```math
s_a = \frac{1}{\tau_a}
\log\frac{T_a+\epsilon}{1-T_a+\epsilon}
```

Kết quả `s` là action logits `[N,120]`; training dùng cross-entropy với ground-truth action label. Concept loss và hypergraph regularization là auxiliary losses, còn quyết định action cuối cùng đến từ prototype matching trên concept probabilities.

### 8.5. Implementation contract

- `SymbolicHyperConceptBridge` sinh `concept_logits` và gọi/cho phép gọi Tversky prototype decoder.
- `Model.forward()` ở `symbolic_action_mode: prototype` phải trả action logits này.
- `self.fc(global_feature)` tuyệt đối không được gọi trong mode trên.
- Không concat global backbone feature vào concept/action logits.
- Intervention trên một concept phải thay action logits đúng theo cột tương ứng của `M`.
- `M`, IDF và Tversky constants phải là buffers/config, không phải free classifier parameters.
- Clamp `p` và `T` bằng `epsilon` để forward/backward không có `log(0)` hoặc NaN.

### 8.6. Decoder ablation

Để tách đóng góp của grounding và matching rule, chạy ba decoder trên cùng V2 grounder:

| ID | Decoder | Zero handling | Trainable classifier parameters |
|---|---|---|---:|
| D0 | Balanced Bernoulli | Hard negative | 0 |
| D1 | IDF-weighted cosine | Phạt gián tiếp qua normalization | 0 |
| D2 | IDF-weighted asymmetric Tversky | Soft extra penalty `beta=0.15` | 0 |

`D2` là decoder chính của V2. `D0` chỉ giữ làm baseline tương thích V1; `D1` là baseline đơn giản để kiểm tra Tversky có thực sự cần thiết.

## 9. Loss, checkpoint và training

### 9.1. Loss

Giữ cấu trúc V1:

```math
L = L_{action} + \lambda_cL_{concept} + \lambda_hL_{hypergraph}
```

Concept loss vẫn dùng fixed teacher matrix hiện tại. Khởi tạo comparison:

```yaml
symbolic_graph_weight: giữ như V1
symbolic_hypergraph_reg_weight: 0.01
```

Hard mask đã đảm bảo zero leakage ở grounder nên chưa thêm grounding loss. Temporal/person entropy regularization mặc định `0.0`; chỉ bật nếu log cho thấy collapse.

### 9.2. Checkpoint nguồn

V2 khởi tạo từ official Hyper-GCN checkpoint, không phụ thuộc V1 symbolic checkpoint:

- Grounded head có representation contract mới.
- Tránh mang shortcut của V1 concept head.
- Backbone parameter names phải giữ nguyên để load official weights.

### 9.3. Frozen backbone

Khi `symbolic_only: true`:

- Backbone parameters có `requires_grad=False`.
- `data_bn`, input embedding, positional embedding và `l1 ... l9` giữ eval mode.
- Grounder và concept hypergraph ở train mode.
- Original `self.fc` frozen và không gọi.

Phải có helper/override để `model.train()` không bật lại backbone BatchNorm. `requires_grad=False` một mình không ngăn BN running statistics thay đổi.

### 9.4. Optimizer

Optimizer chỉ nhận:

- Grounded concept encoder.
- Concept query/bias.
- Hypergraph incidence residual.
- Hyperedge weights và hypergraph gate.
- Concept token/logit projections.

Backbone và original `fc` không nằm trong optimizer param groups.

## 10. Kế hoạch thay đổi theo file

### 10.1. `model/concept_schema.py`

- [ ] Chuẩn hóa body-part names/aliases.
- [ ] Đọc một hoặc nhiều body part cho mỗi concept.
- [ ] Validate body mapping và scope.
- [ ] Xây `concept_joint_mask [K,V]` từ NTU layout.
- [ ] Fail-fast với empty mask hoặc index ngoài `[0,V-1]`.
- [ ] Xuất audit `concept -> parts -> joints -> scope`.
- [ ] Giữ nguyên concept ID/thứ tự.

### 10.2. `model/grounded_concept_encoder.py` — file mới

- [ ] Nhận `[N,M,C_b,T',V]`.
- [ ] Áp hard concept-joint mask.
- [ ] Pool joint theo concept.
- [ ] Project `C_b -> D_c`.
- [ ] Pool thành temporal bins và thêm position embedding.
- [ ] Concept-conditioned temporal attention.
- [ ] Scope-aware person aggregation.
- [ ] Mask person padding.
- [ ] Trả `[N,K,D_c]` và `grounding_aux`.
- [ ] Xuất grounding snapshot JSON-safe.
- [ ] Không import hoặc sử dụng action label.

### 10.3. `model/symbolic_hyperconcept.py`

- [ ] Tích hợp `GroundedConceptEncoder`.
- [ ] Đổi bridge input sang `[N,M,C_b,T',V]`.
- [ ] Cho concept hypergraph xử lý `[N,K,D_c]`.
- [ ] Sinh concept logit riêng từ concept token.
- [ ] Giữ fixed concept-action matrix nhưng thay matching function bằng IDF-weighted Tversky.
- [ ] Implement Bernoulli/cosine chỉ như decoder ablation modes.
- [ ] Giữ V1 hypergraph parameters/regularization.
- [ ] Mở rộng audit với time/person/joint grounding.
- [ ] Cập nhật `trainable_parameters()`.

### 10.4. `model/hypergcn_base.py`

- [ ] Sau `l9`, reshape thành `[N,M,C_b,T',V]`.
- [ ] Truyền tensor 5 chiều trực tiếp vào bridge.
- [ ] Không flatten `M*V` trước grounding.
- [ ] Giữ blackbox mode cũ hoạt động nếu prototype tắt.
- [ ] Không đổi backbone state-dict keys.
- [ ] Validate `M`, `V`, `C_b`.
- [ ] Prototype mode chỉ trả action logits từ concepts.

### 10.5. `model/training.py`

- [ ] Freeze backbone parameters.
- [ ] Giữ backbone BN ở eval trong symbolic-only training.
- [ ] Chỉ optimizer V2 symbolic parameters.
- [ ] Log trainable/frozen count theo module.
- [ ] Giữ checkpoint-loading diagnostics.

### 10.6. `main.py`

- [ ] Log grounding mode và token dimensions.
- [ ] Log temporal/person attention statistics.
- [ ] Log dead concept/token norms.
- [ ] Export `epochN_test_grounding.json`.
- [ ] Giữ `epochN_test_hypergraph.json`.
- [ ] Thêm body-part occlusion evaluation tùy chọn.
- [ ] Giữ initial evaluation trước train.
- [ ] Báo delta so với initial V2 và V1.

### 10.7. Config V2

Theo yêu cầu triển khai, active config hiện tại được chuyển trực tiếp sang V2:

```text
config/base/nturgbd120-cross-subject/hyper_joint_symbolic.yaml
```

Các lựa chọn bắt buộc như hard joint mask, scope-aware person handling và không
có global residual được enforce trong code, không đưa thành YAML flags thừa.

```yaml
symbolic_concept_dim: 64
symbolic_temporal_bins: 4
symbolic_grounding_dropout: 0.1

symbolic_hypergraph_mode: learnable_v1
symbolic_hypergraph_reg_weight: 0.01
symbolic_action_mode: prototype
symbolic_action_decoder: tversky
symbolic_action_missing_weight: 1.0
symbolic_action_extra_weight: 0.15
symbolic_action_temperature: 0.15
symbolic_action_idf_weighting: true
symbolic_action_idf_min: 0.5
symbolic_action_idf_max: 2.0
```

Comparison đầu giữ training giống V1:

```yaml
optimizer: Adam
base_lr: 0.0003
warm_up_epoch: 1
step: [12, 17]
num_epoch: 20
```

Nếu validation vẫn tăng ở epoch 20 mới tạo experiment 30 epoch; không sửa kết quả comparison 20 epoch.

## 11. Test bắt buộc

### 11.1. Schema và shape

- [ ] Đúng 102 concepts và 25 physical joints.
- [ ] Mỗi concept có non-empty valid joint mask.
- [ ] Body aliases normalize deterministically.
- [ ] Input `[N,M,256,T',25]` trả tokens `[N,102,64]`.
- [ ] Concept logits `[N,102]`, action logits `[N,120]`.
- [ ] Temporal weights có 4 bins.

### 11.2. Hard grounding

Với synthetic backbone map và body-specific concept:

- [ ] Thay đổi joint ngoài mask không đổi pre-hypergraph concept token trong eval mode.
- [ ] Thay đổi joint trong mask làm token đổi.
- [ ] Attention ngoài mask bằng chính xác 0.
- [ ] Whole-body concept phản ứng với nhiều body parts.

Test trực tiếp trên synthetic feature map để cô lập grounder; raw skeleton đã bị Hyper-GCN message passing trộn thông tin giữa joints.

### 11.3. Person handling

- [ ] Padding person không nhận attention.
- [ ] Single-person concept không bị padding noise ảnh hưởng.
- [ ] Interaction concept thay đổi khi relation hai người thay đổi.
- [ ] Không flatten person/joint trước grounder.

### 11.4. Gradient và freeze

- [ ] Action loss truyền gradient tới grounder.
- [ ] Gradient tới incidence, edge weights và gate.
- [ ] Backbone không có gradient.
- [ ] Original `fc` không gradient và không được gọi.
- [ ] Không NaN trong masked attention.
- [ ] `model.train()` không bật backbone BN training.
- [ ] Một optimizer step không đổi backbone weights hoặc BN buffers.

### 11.5. Checkpoint và bottleneck

- [ ] Official checkpoint load được.
- [ ] V2 params được khởi tạo mới và báo rõ.
- [ ] V1 symbolic checkpoint không silently load sai shape.
- [ ] V2 resume phục hồi optimizer/epoch.
- [ ] Monkeypatch original `fc` để raise; prototype forward vẫn chạy.
- [ ] Concept intervention đổi action score theo prototype matrix.
- [ ] Không có direct backbone-to-action module mới.

### 11.6. Tversky decoder

- [ ] IDF được tính chỉ từ training ontology/fixed matrix.
- [ ] IDF nằm trong `[0.5, 2.0]` và được register như buffer.
- [ ] Perfect concept prototype xếp action tương ứng cao hơn prototype không khớp.
- [ ] Tăng matched evidence không làm score giảm.
- [ ] Tăng missing evidence không làm score tăng.
- [ ] Extra evidence bị phạt yếu hơn missing evidence khi `beta < alpha`.
- [ ] All-zero/all-one concept probabilities vẫn cho finite logits và gradients.
- [ ] Decoder không có trainable classifier parameters.
- [ ] Bernoulli, cosine và Tversky đều trả logits `[N,120]` để chạy ablation cùng interface.

## 12. Logging và artifacts

Mỗi epoch ghi:

- Classification: Top-1/Top-5, loss, best epoch, delta so với initial và V1.
- Concepts: mAP, micro/macro F1, activation frequency, variance, dead concepts, correlated pairs.
- Grounding: temporal/person entropy, top bin/person, token norm, occlusion effects.
- Hypergraph: gate, structure loss, prior/non-prior means, changed memberships, edge weights.

Artifacts:

```text
epochN_test_grounding.json
epochN_test_hypergraph.json
best_grounding.json
best_hypergraph.json
concept_body_joint_audit.csv
```

Không export attention cho toàn bộ sample mặc định; chỉ aggregate và các sample đại diện được cấu hình.

## 13. Faithfulness evaluation

### 13.1. Body-part occlusion

Với concept `c` có body mask `P_c`:

1. Chạy sample nguyên bản, lấy `p_c`.
2. Occlude body part liên quan, lấy `p_c_related`.
3. Occlude part không liên quan có kích thước tương đương, lấy `p_c_unrelated`.
4. So sánh:

```math
\Delta_{related} = p_c - p_{c,related}
```

```math
\Delta_{unrelated} = p_c - p_{c,unrelated}
```

Grounding tốt khi `Delta_related > Delta_unrelated` trên aggregate dataset.

Báo hai mức occlusion:

- Raw-input occlusion: faithfulness end-to-end, có ảnh hưởng message passing của backbone.
- Feature-map occlusion sau `l9`: cô lập symbolic grounder.

### 13.2. Concept intervention

- Force concept ON/OFF trước Tversky prototype decoder.
- Đo action rank và logit delta.
- Hướng thay đổi phải khớp fixed concept-action matrix.

### 13.3. Completeness

- Action prediction chỉ từ concept logits.
- Không có global backbone residual.
- Báo accuracy khi dùng full concepts và top-k concepts.

## 14. Ablation tối thiểu

| ID | Concept evidence | Temporal | Person-aware | Concept hypergraph | Decoder |
|---|---|---|---|---|---|
| V1 | Bridge hiện tại | Hiện tại | Flatten `M×V` | Learnable V1 | Bernoulli prototype |
| V2-0 | Hard joint masked mean | Global mean | Mean valid persons | Learnable V1 | Tversky prototype |
| V2-1 | Hard joint masked mean | 4 bins + attention | Scope-aware | Learnable V1 | Tversky prototype |
| V2-2 | V2-1 | 4 bins + attention | Scope-aware | Tắt propagation | Tversky prototype |

So sánh kiến trúc chính: `V1` với `V2-1`. Vì hai hàng này đồng thời khác grounder và decoder, phải báo thêm decoder ablation D0/D1/D2 ở mục 8.6 trên cùng V2-1 grounder. `V2-0` kiểm tra attention có cần thiết không; `V2-2` tách đóng góp grounding khỏi concept hypergraph. Không đưa V2B vào ablation.

## 15. Tiêu chí nghiệm thu

### 15.1. Bắt buộc kỹ thuật

- [ ] Toàn bộ unit tests pass.
- [ ] Official checkpoint load thành công.
- [ ] Backbone weights và BN buffers không đổi khi symbolic-only.
- [ ] Original FC không tham gia forward.
- [ ] Không có backbone-to-action bypass.
- [ ] Mỗi concept có non-empty hard joint mask.
- [ ] Attention ngoài mask bằng 0.
- [ ] Grounder và hypergraph parameters nhận gradient.
- [ ] Grounding/hypergraph snapshots hợp lệ.

### 15.2. Mục tiêu thực nghiệm

- V2 Top-1 không thấp hơn V1 quá 1 percentage point với cùng checkpoint, seed, optimizer và 20 epochs.
- Relevant-part occlusion effect lớn hơn unrelated-part effect.
- Concept intervention consistency gần tuyệt đối với fixed matrix và Tversky decoder.
- Không tăng mạnh dead concepts/correlation so với V1.
- Hypergraph gate không kẹt ở 0 suốt training.

Accuracy không phải tiêu chí duy nhất. Nếu Top-1 tương đương nhưng grounding và faithfulness tốt hơn rõ rệt, V2 vẫn đạt mục tiêu neuro-symbolic.

## 16. Thứ tự triển khai

### Milestone 0 — Lock V1 baseline

- [ ] Sao lưu config V1.
- [ ] Ghi checkpoint, seed và metrics V1.
- [ ] Chạy và ghi unit-test baseline.

### Milestone 1 — Schema và masks

- [ ] Audit body-part/scope values.
- [ ] Xác minh NTU25 joint indices.
- [ ] Tạo part-to-joint mapping.
- [ ] Tạo `concept_joint_mask`.
- [ ] Thêm schema tests và audit CSV.

### Milestone 2 — Grounder tối giản V2-0

- [ ] Thêm `grounded_concept_encoder.py`.
- [ ] Hard masked joint mean.
- [ ] Global temporal mean.
- [ ] Valid-person mean.
- [ ] Nối bridge và model.
- [ ] Pass shape, mask, gradient tests.
- [ ] Smoke forward/backward CPU.

### Milestone 3 — Full grounder V2-1

- [ ] Thêm 4 temporal bins.
- [ ] Concept-conditioned temporal attention.
- [ ] Scope-aware person aggregation.
- [ ] Grounding snapshot.
- [ ] Pass person/padding tests.

### Milestone 4 — Training safety

- [ ] Khóa backbone BN eval mode.
- [ ] Kiểm tra optimizer param groups.
- [ ] Test official checkpoint loading.
- [ ] Test V2 resume checkpoint.
- [ ] So sánh backbone state trước/sau optimizer step.

### Milestone 5 — Logging và evaluation

- [ ] Grounding metrics trong console/TensorBoard.
- [ ] Export JSON/CSV artifacts.
- [ ] Feature-map occlusion.
- [ ] Raw-skeleton occlusion tùy chọn.
- [ ] Concept intervention report.

### Milestone 6 — Ablation

- [ ] Chạy V1 cùng seed/budget.
- [ ] Chạy V2-0.
- [ ] Chạy V2-1.
- [ ] Chạy V2-2 tắt hypergraph propagation.
- [ ] Tổng hợp accuracy, grounding và faithfulness.

## 17. Rủi ro và biện pháp

### Body mapping sai joint index

- Lấy index trực tiếp từ NTU graph layout.
- Test các joint đại diện trái/phải/head/torso.
- Xuất audit CSV trước training.

### Joint feature đã trộn qua Hyper-GCN

Output joint sau `l9` đã nhận message từ joint khác. Hard mask bảo đảm routing theo output location, nhưng không chứng minh feature hoàn toàn độc lập khỏi toàn thân.

- Mô tả đúng giới hạn trong luận văn.
- Báo cả feature-map và raw-input occlusion.
- Không thêm global residual shortcut.

### Person padding được chọn

- Tạo valid-person mask từ raw skeleton energy/motion.
- Mask trước softmax.
- Test sample một người.

### Concept query trở thành class code

- Shared channel projection.
- Hard body mask.
- Fixed concept IDs/matrix và decoder không có free classifier weights.
- Theo dõi redundancy, activation variance và intervention.

### Temporal attention collapse

- Log entropy.
- Không bật entropy regularization mặc định.
- Chỉ thêm regularizer khi quan sát collapse thực tế.

### Accuracy giảm khi bỏ global shortcut

- Implement V2-0 trước.
- So sánh per-class để tìm mapping sai.
- Sửa ontology/body mapping trước khi tăng capacity.
- Không tự động thêm global residual vì làm yếu lập luận symbolic.

## 18. Definition of Done

V2 hoàn thành khi:

1. Action prediction đi qua grounded concept tokens và concept hypergraph.
2. Mỗi concept có explicit, validated body/joint mask.
3. Person/joint axes không bị flatten trước grounding.
4. Official pretrained backbone được dùng nguyên vẹn và frozen đúng.
5. Original blackbox FC không được gọi.
6. Fixed concept-action matrix được giữ; decoder dùng IDF-weighted asymmetric Tversky và không có free classifier weights.
7. Tests chứng minh hard grounding, gradient flow, BN freeze và bottleneck exclusivity.
8. Training xuất classification, concept, grounding và hypergraph metrics.
9. Có ablation V1/V2-0/V2-1/V2-2 cùng điều kiện.
10. Có body-part occlusion và concept intervention report, không chỉ Top-1.

## 19. Quyết định phạm vi cuối cùng

V2 tập trung vào:

```text
Hyper-GCN skeleton representation
        -> explicit body grounding
        -> named concepts
        -> learnable concept hypergraph
        -> fixed symbolic matrix + Tversky action decoder
```

Typed semantic concept-action matrix là hướng độc lập, không thuộc V2. Thiết kế này giúp thí nghiệm trả lời rõ một câu hỏi: **grounding concept trực tiếp vào skeleton hypergraph có làm mô hình faithful và hiệu quả hơn V1 hay không?**
