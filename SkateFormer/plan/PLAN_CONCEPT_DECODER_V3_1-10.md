# Kế hoạch nâng cấp Concept Decoder V3 và loại bỏ Parent Gate

## 1. Quyết định đã khóa

Kế hoạch này được chốt trước khi chạy các thí nghiệm V3 để tránh thay đổi thiết kế theo từng lỗi trên official test set.

1. Xóa hoàn toàn multiplicative parent gate khỏi model, API, evaluator và YAML.
2. Giữ parent head như một nhánh auxiliary có supervision.
3. Giữ `parent_weight` trong symbolic decoder để parent vẫn có thể đóng góp trực tiếp vào action score.
4. Giữ hierarchy loss như một soft regularizer; không dùng parent probability để sửa concept probability ở inference.
5. Không dùng `concept_priority.csv` sinh từ official test để chọn concept hoặc thiết kế head riêng cho từng concept.
6. Nâng cấp concept decoder theo hai thay đổi tổng quát áp dụng đồng đều cho toàn bộ concept:
   - adaptive multiple-instance temporal pooling;
   - per-concept multi-scale fusion.

## 2. Cơ sở thực nghiệm

Oracle diagnosis trên NTU120-CSub cho checkpoint hiện tại:

| Chỉ số | Kết quả |
|---|---:|
| Predicted Top-1 | 84.0040% |
| Oracle concept Top-1 | 100.0000% |
| Oracle parent Top-1 | 84.0943% |
| Oracle concept + parent Top-1 | 100.0000% |
| Concept-only Top-1 | 83.9471% |
| Concept margin share | 95.50% |
| Parent margin share | 1.18% |

Top-k intervention tăng Top-1 lên 90.10% khi sửa một concept sai nhất, 95.43% khi sửa ba concept và 97.53% khi sửa năm concept. Vì vậy bottleneck chính là chất lượng concept evidence, không phải năng lực biểu diễn của symbolic decoder.

## 3. Phạm vi

### Trong phạm vi

- Xóa parent gate và các cấu hình liên quan.
- Giữ output concept độc lập với parent ở inference.
- Thay temporal noisy-OR bằng một pooling MIL thích nghi, có thể bật/tắt bằng config để làm ablation.
- Nâng global scale fusion hiện tại thành per-concept scale fusion.
- Đồng bộ training, evaluator, intervention evaluator, artifact schema và tài liệu.
- Thêm regression test để chứng minh việc xóa gate với checkpoint `gate_floor=1.0` không đổi action score.

### Ngoài phạm vi

- Không thay ontology 102 concept.
- Không thêm concept dựa trên confusion matrix của test.
- Không thay symbolic rule support hoặc learned symbolic weights.
- Không thêm relation graph hoặc pairwise concept reasoning.
- Không xóa parent branch trong V3; `w/o parent` là một ablation riêng.
- Không dùng validation để chọn checkpoint; vẫn dùng final epoch như protocol hiện tại.

## 4. Thay đổi A — xóa Parent Gate

### Semantics mới

Hiện tại:

```text
raw_sequence_probs = temporal_pool(time_probs)
gate = gate_floor + (1 - gate_floor) * parent_for_child
sequence_probs = raw_sequence_probs * gate
```

Sau thay đổi:

```text
sequence_probs = temporal_pool(time_logits)
parent_probs = parent_head(features)
action_scores = symbolic(sequence_probs, parent_probs)
```

Parent chỉ còn ba vai trò độc lập:

1. `parent_loss`: auxiliary supervision;
2. `hierarchy_loss(sequence_probs, parent_probs)`: soft consistency regularizer;
3. `parent_weight * parent_score`: đóng góp trực tiếp trong symbolic action score.

### API cần xóa

- Tham số `gate_floor` khỏi:
  - `HierarchicalConceptDecoder`;
  - `HCGNSSkateFormer`;
  - `main_hcg_ns.py`;
  - `eval_hcg_concepts.py`;
  - `eval_concept_intervention.py`.
- Output `soft_gate` khỏi concept decoder và model output.
- Nhánh `effective_sequence_probs = raw_sequence_probs * gate`.
- Mọi khóa `gate_floor` trong config train/ablation.

### Dữ liệu chẩn đoán sau khi xóa

- Chỉ giữ `sequence_probs` làm concept probability chính thức.
- `parent_for_child` vẫn giữ vì hierarchy loss và hierarchy metric cần nó.
- `hierarchy_violation_rate` được tính trực tiếp từ:

```text
sequence_probs - parent_for_child
```

- Artifact mới không ghi `raw_concept_probs` vì không còn khái niệm pre-gate/post-gate.
- Tool đọc artifact cũ vẫn chấp nhận `raw_concept_probs` để bảo toàn backward compatibility.

### Tương thích checkpoint

`gate_floor` không nằm trong `state_dict`, vì vậy checkpoint V2 với `gate_floor=1.0` có thể dùng làm regression reference. Việc xóa gate phải cho action scores giống checkpoint hiện tại trong legacy pooling mode, sai số tối đa mục tiêu `1e-5`.

## 5. Thay đổi B — Adaptive MIL Temporal Pooling

### Vấn đề của noisy-OR hiện tại

```text
P(concept) = 1 - product_t(1 - P_t(concept))
```

Noisy-OR tăng theo số timestep và dễ bão hòa khi có nhiều xác suất nhỏ. Nó cũng dùng cùng một giả định temporal cho concept xuất hiện ngắn như `throw` và concept kéo dài như `standing`.

### Thiết kế đề xuất

Thêm module `AdaptiveTemporalMILPool` trong `model/concept_prototypes.py`. Với concept `k`:

```text
mean_k = mean_t(logit_tk)
lse_k  = [logsumexp_t(beta_k * logit_tk) - log(T)] / beta_k
mix_k  = sigmoid(mix_raw_k)
sequence_logit_k = mix_k * lse_k + (1 - mix_k) * mean_k + bias_k
sequence_prob_k  = sigmoid(sequence_logit_k)
```

Trong đó:

- `lse_k` là evidence kiểu max mềm, phù hợp concept xuất hiện ngắn;
- `mean_k` là evidence theo duration, phù hợp concept kéo dài;
- `mix_k` được học riêng cho mỗi concept nhưng cùng một công thức chung;
- `beta_k` dương và bị chặn trong miền ổn định;
- `bias_k` hiệu chỉnh prevalence của concept.

Không tạo head đặc biệt dựa trên test. Mọi concept đều có cùng cấu trúc và cùng số tham số.

### Chế độ cấu hình

```yaml
prototype_args:
  temporal_pooling: adaptive_mil   # legacy_noisy_or | adaptive_mil
```

Giữ `legacy_noisy_or` để tái tạo baseline và làm ablation. V3 dùng `adaptive_mil`.

### Outputs temporal

- `time_logits` và `time_probs` vẫn giữ để grounding/visualization.
- `expected_time`, `peak_time`, `duration`, `rising_activity`, `falling_activity` vẫn được tính như hiện tại.
- Chỉ cách tạo clip-level `sequence_probs` thay đổi.

## 6. Thay đổi C — Per-Concept Multi-Scale Fusion

### Vấn đề hiện tại

Multi-scale decoder hiện dùng một vector `scale_logits[L]` chung cho toàn bộ concept. Điều này buộc concept chuyển động nhanh, posture, interaction và object-context dùng cùng tỷ lệ feature stage.

### Thiết kế mới

Đổi scale logits thành:

```text
scale_logits[L, K]
scale_weights = softmax(scale_logits, dim=0)
fused_time_logits[b,t,k] = sum_l scale_weights[l,k] * logits[l,b,t,k]
```

Trong đó `L` là số backbone levels và `K` là số concept. Trọng số khởi tạo đều để không ưu tiên scale nào trước training.

### Cấu hình

```yaml
prototype_args:
  multi_scale: true
  pyramid_levels: [2, 3]
  scale_fusion: per_concept
```

Không thay backbone và không unfreeze thêm stage trong thí nghiệm đầu tiên, nhằm cô lập đóng góp của decoder.

## 7. Loss và protocol training

Trong vòng V3 đầu tiên, giữ nguyên toàn bộ loss weights, số epoch, optimizer và learning rate để phép so sánh chỉ phản ánh decoder:

- action cross-entropy;
- weak concept loss;
- parent loss;
- hierarchy loss trên `sequence_probs`;
- density loss;
- center grounding + separation;
- symbolic sparsity loss.

Không thêm focal loss/asymmetric loss trong cùng vòng triển khai. Nếu Adaptive MIL + multi-scale chưa đủ, thay đổi loss sẽ là vòng ablation V3.1 riêng.

Stage schedule vẫn là:

```text
Stage 1: parent grounding
Stage 2: unary symbolic composition
Stage 3: selective backbone adaptation (hiện bằng 0 epoch)
```

## 8. Ma trận ablation được khóa trước

Các run dùng cùng seed, full official training split, final epoch và official test split. Không chọn checkpoint bằng test.

| Run | Parent gate | Temporal pooling | Scale fusion | Mục đích |
|---|---|---|---|---|
| V2-R | Không | legacy noisy-OR | single final scale | Regression baseline sau khi xóa code gate |
| V3-A | Không | adaptive MIL | single final scale | Đo riêng temporal pooling |
| V3-B | Không | legacy noisy-OR | per-concept multi-scale | Đo riêng scale fusion |
| V3-C | Không | adaptive MIL | per-concept multi-scale | Mô hình V3 đầy đủ |
| V3-D | Không | adaptive MIL | per-concept multi-scale, parent disabled | Ablation auxiliary parent toàn phần |

`V3-D` chỉ chạy sau khi V3-C hoàn tất; không dùng kết quả test để thay đổi V3-C.

Nếu đủ tài nguyên, chạy ba seed và báo cáo mean ± standard deviation. Nếu chỉ chạy một seed, ghi rõ hạn chế và không diễn giải chênh lệch dưới khoảng 0.1 điểm phần trăm như một cải thiện chắc chắn.

## 9. Metrics bắt buộc

### Action

- Top-1, Top-5;
- per-class accuracy;
- confusion matrix.

### Concept

- micro-F1, macro-F1;
- exact match;
- precision/recall/F1 theo concept;
- Brier score và ECE;
- known-negative violation rate.

### Hierarchy

- parent micro/macro-F1;
- hierarchy violation rate/mean trên `sequence_probs`;
- không báo metric pre-gate/post-gate vì gate đã bị xóa.

### Faithfulness

- Oracle concept/parent/both;
- score decomposition;
- shuffle test;
- top-k intervention curve.

Official test per-concept ranking chỉ dùng cho error analysis sau khi thiết kế đã khóa, không dùng để thêm head riêng hoặc điều chỉnh loss theo concept.

## 10. Danh sách file dự kiến thay đổi

### Model

- `model/concept_prototypes.py`
  - thêm `AdaptiveTemporalMILPool`;
  - hỗ trợ `legacy_noisy_or` và `adaptive_mil`;
  - đổi optional multi-scale fusion thành per-concept.
- `model/hierarchical_concept_decoder.py`
  - xóa toàn bộ gate;
  - trả trực tiếp `sequence_probs` từ temporal pool;
  - giữ parent outputs và scope fusion.
- `model/hcg_ns_model.py`
  - xóa `gate_floor` và `soft_gate`;
  - hierarchy loss dùng `sequence_probs`.

### Training/evaluation

- `main_hcg_ns.py`
  - xóa đọc `gate_floor`;
  - cập nhật dump/metric schema.
- `eval_hcg_concepts.py`
  - xóa đọc `gate_floor`;
  - dùng schema concept probability duy nhất.
- `eval_concept_intervention.py`
  - xóa đọc `gate_floor`;
  - giữ exact score recomputation guard.
- `tools/evaluate_hierarchical_concepts.py`
  - đọc schema mới;
  - fallback cho artifact V2 cũ.

### Config

- Các YAML runtime cũ dưới `config/train/new/` và
  `config/train/ablation_ntu60_cs_v23/` đã được loại bỏ sau khi chốt V3-E.
- Ma trận V3.1 tập trung vào NTU120-CSub được đặt tại
  `config/train/ablation_ntu120_csub_v31/`:
  - tạo config riêng cho V2-R, V3-A, V3-B, V3-C, V3-D, V3-E;
  - dùng ontology 120 action/102 concept v2.4 và pretrained NTU120-CSub;
  - dùng `work_dir` riêng, không ghi đè checkpoint cũ.

## 11. Kiểm thử

### Unit tests

1. Adaptive pool nhận `[B,T,K]`, trả `[B,K]`, không NaN/Inf.
2. Adaptive pool bất biến với hoán vị timestep khi không có temporal mask.
3. Gradient đi tới time logits, `beta`, `mix` và `bias`.
4. Per-concept scale weights có shape `[L,K]` và tổng theo `L` bằng 1.
5. `parent_enabled=false` không tạo parent contribution nhưng concept path vẫn chạy.
6. Hierarchy loss nhận trực tiếp `sequence_probs`.

### Regression tests

1. Với legacy pooling và checkpoint `gate_floor=1.0`, action score trước/sau refactor sai khác không quá `1e-5`.
2. Oracle evaluator có `recompute_max_abs_diff < 1e-3`.
3. Rule replay không đổi argmax so với model output.
4. `rg` không còn `gate_floor` hoặc `soft_gate` trong runtime Python/config mới.

### Integration smoke test

1. Parse từng config mới.
2. Khởi tạo model.
3. Load pretrained SkateFormer backbone.
4. Chạy một batch forward, loss, backward và optimizer step.
5. Kiểm tra mọi loss hữu hạn.
6. Chạy evaluator và intervention evaluator trên checkpoint smoke-test.

## 12. Tiêu chí hoàn tất

Implementation được xem là hoàn tất khi:

- parent gate đã biến mất khỏi runtime/API/config;
- regression baseline tái tạo checkpoint `gate_floor=1.0` trong tolerance;
- năm cấu hình V2-R/V3-A/V3-B/V3-C/V3-D parse và chạy độc lập;
- evaluator và rule replay nhất quán;
- V3 không sử dụng bất kỳ concept-specific decision nào lấy từ official test;
- kết quả được báo cáo đầy đủ, kể cả khi V3 không vượt baseline.

## 13. Thứ tự triển khai

1. Viết regression test cho checkpoint gate-disabled hiện tại.
2. Xóa parent gate và cập nhật artifact schema.
3. Chạy regression + smoke test V2-R.
4. Thêm Adaptive MIL pooling và unit tests.
5. Thêm per-concept multi-scale fusion và unit tests.
6. Tạo năm config ablation với `work_dir` riêng.
7. Chạy V2-R, V3-A, V3-B, V3-C theo ma trận đã khóa.
8. Chạy V3-D nếu đủ tài nguyên.
9. Chạy evaluator, Oracle intervention và tổng hợp bảng kết quả.

