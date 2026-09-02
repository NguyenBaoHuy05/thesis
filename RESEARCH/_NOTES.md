# Research Notes: ECR + KG-guided Looping Thinking

## Quyết định hướng nghiên cứu

Hướng chính được chọn là **ECR-SkateFormer + KG-guided Looping Concept Reasoning**. Không tiếp tục dùng feedback attention trong phiên bản đầu tiên. Feedback attention được giữ ở mục future work.

Giả thuyết nghiên cứu:

> Sau khi SkateFormer trích xuất đặc trưng, concept ban đầu có thể được tinh chỉnh qua nhiều vòng suy luận dựa trên Knowledge Graph; các vòng sau sẽ tạo action evidence nhất quán hơn mà không cần đưa thông tin ngược trở lại backbone.

Base code được chọn:

```text
ECR-SkateFormer/model/reason_skateformer.py
ECR-SkateFormer/model/SkateFormer.py
```

Không chọn DCLN làm base vì PRN/feedback của DCLN gắn chặt với CTR-GCN. Không chọn REASON làm base vì REASON đồng thời có HyperGCN/CTR-GCN, CLIP LoRA, ML-Decoder và staged training, khiến khó xác định tác động riêng của looping reasoning. Ta chỉ mượn concept/KG/logic đã có trong ECR và có thể thêm CLIP sau.

## Looping thinking là gì?

Looping thinking không phải gọi một classifier nhiều lần với cùng input. Mỗi vòng phải đọc trạng thái concept của vòng trước, tạo bằng chứng logic mới, cập nhật trạng thái concept và sinh action evidence mới.

```text
Skeleton X
   ↓
SkateFormer backbone
   ↓
Initial concept logits z⁰ / probabilities c⁰
   ↓
┌────────────────────────────────────────────┐
│ Vòng t:                                    │
│  1. Read    cᵗ                             │
│  2. Reason  [cᵗ, 1-cᵗ] → KG clauses rᵗ     │
│  3. Predict rᵗ → action evidence yᵗ         │
│  4. Refine  [cᵗ, rᵗ, yᵗ] → cᵗ⁺¹             │
└────────────────────────────────────────────┘
   ↓ lặp K vòng
Final concept cᴷ + final action yᴷ + reasoning trace
```

Ký hiệu:

- `K`: số vòng lặp, bắt đầu bằng `2`;
- `c⁰`: concept nhận diện trực tiếp từ feature của SkateFormer;
- `cᵗ`: concept state ở vòng `t`;
- `rᵗ`: logic/knowledge evidence ở vòng `t`;
- `yᵗ`: action logits hoặc action evidence ở vòng `t`.

Điểm quan trọng: loop chỉ hoạt động ở **concept/reasoning space**, không feedback về `feature_map`, không sửa attention của SkateFormer trong phiên bản chính. Nhờ vậy ta tách được hai câu hỏi: backbone nhìn thấy gì, và reasoning sửa/kiểm tra nhận thức ban đầu ra sao.

## Một vòng lặp chi tiết

### Bước 0: Initial perception

Giữ ECR hiện tại:

```python
feature_map = backbone.forward_features_from_input(x, index_t=index_t)
spatial_feat, temporal_feat = self._pool_features(feature_map)
z_spa = self.spatial_head(spatial_feat)
z_tem = self.temporal_head(temporal_feat)
z0 = torch.cat([z_spa, z_tem], dim=-1)  # [B, K]
c0 = torch.sigmoid(z0)                # [B, K]
```

`feature_map` có dạng `[B, C, T', N']`; hai pooling tạo feature `[B, C]`; `K` là tổng spatial và temporal concepts, hiện thường là `72` trong ECR.

`c0` phải được xem là **nhận thức ban đầu**, chưa phải kết luận cuối cùng. Dùng concept target ở đây để giữ tín hiệu perception và phát hiện lỗi do backbone/concept head.

### Bước 1: Literal construction

Tạo cả concept dương và phủ định:

\[
\ell^t = [c^t, 1-c^t] \in [0,1]^{2K}.
\]

Trong training không threshold về 0/1. Threshold chỉ dùng lúc xuất rule/explanation. Với concept `hand_near_face = 0.62`, literal dương là `0.62`, literal phủ định là `0.38`.

### Bước 2: KG reasoning

Đưa literals vào reasoner hiện tại:

```python
hidden_1_t = first_rule_bank(literals_t)
hidden_2_t = second_rule_bank(hidden_1_t)
y_t = action_head(hidden_2_t)
```

Rule bank cần giữ được:

- `require`: concept B cần concept A;
- `alongside`: hai concept thường cùng xuất hiện;
- `excludes`: concept này chống lại concept kia;
- `prior_retention`: tránh learned rules quên KG prior;
- `sparsity`: tránh mọi concept nối dày vào mọi rule.

Ở bước đầu, tái sử dụng `KGPrimedReasoner` và sửa nó để trả `hidden_1`, `hidden_2`, `action_logits`, `clauses`/`literals`. Không cần viết lại logic engine.

### Bước 3: Action evidence

Mỗi vòng tạo ra `yᵗ`. Có hai mode:

1. `action_mode="linear"`: dùng làm baseline kỹ thuật, dễ ổn định nhưng action head có thể kém giải thích.
2. `action_mode="prototype"`: action score dựa trên concept schema của từng action, phù hợp phiên bản giải thích chính.

Nên làm linear mode trước để xác nhận loop chạy đúng. Sau đó thêm prototype mode như ablation/phiên bản chính nếu accuracy và fidelity chấp nhận được.

### Bước 4: Concept refinement

Logic evidence được chiếu về concept space rồi cập nhật concept logits:

\[
u^t = \operatorname{MLP}_{logic}(r^t)
\]

\[
\Delta z^t = \operatorname{MLP}_{update}([z^t,u^t,e_y^t])
\]

\[
z^{t+1}=z^t+\beta\Delta z^t,
\qquad c^{t+1}=\sigma(z^{t+1}).
\]

Trong phiên bản tối thiểu có thể bỏ `e_yᵗ` và chỉ dùng `[zᵗ, uᵗ]`. Sau đó mới thêm action evidence với trọng số nhỏ để tránh action head chi phối ngược concept quá sớm.

Nên cập nhật ở **logit space**, không phải probability space, để tránh giá trị gần 0/1 bị kẹt gradient. `β` là residual update gate, khởi tạo `0.1` hoặc `0`, giới hạn trong `[0, 0.5]`.

Không nối trực tiếp `feature_map` vào `update_mlp`. Nếu nối, loop có thể bỏ qua concept và trở thành một classifier ẩn; khi đó explanation không còn đáng tin.
Không nối trực tiếp `feature_map` vào `update_mlp`. Nếu nối, loop có thể bỏ qua concept và trở thành một classifier ẩn; khi đó explanation không còn đáng tin.

## Module cần triển khai

Tạo class mới trong `ECR-SkateFormer/model/reason_skateformer.py`:

```python
class IterativeKGReasoner(nn.Module):
    def __init__(self, base_reasoner, num_concepts, num_actions,
                 num_loops=2, shared_weights=True,
                 update_gate_init=0.1):
        ...

    def forward(self, concept_logits, return_trace=False):
        ...
```

Cấu trúc:

```text
IterativeKGReasoner
 ├── base_reasoner: KGPrimedReasoner
 ├── logic_projection: hidden evidence → state dimension
 ├── update_mlp: [zᵗ, logic_state] → Δzᵗ
 ├── beta/update gate
 └── optional action projection
```

Pseudocode tối thiểu:

```python
z_t = concept_logits
trace = []

for step in range(num_loops):
    y_t, logic_t = base_reasoner(z_t, return_trace=True)
    r_t = project_logic_state(logic_t)
    delta_t = update_mlp(torch.cat([z_t, r_t], dim=-1))
    z_next = z_t + beta * delta_t

    if return_trace:
        trace.append({
            "step": step,
            "concept_logits": z_t,
            "concept_probs": torch.sigmoid(z_t),
            "action_logits": y_t,
            "logic": logic_t,
            "delta": delta_t,
        })
    z_t = z_next

y_final, final_logic = base_reasoner(z_t, return_trace=True)
return y_final, torch.sigmoid(z_t), trace
```

Thực tế `KGPrimedReasoner` hiện trả tuple khác nhau tùy `return_trace`; cần chuẩn hóa interface trước. `logic_projection` không nên phụ thuộc shape cứng của `hidden_1`/`hidden_2`; concat hai state rồi dùng `Linear` hoặc pooling theo feature dimension.

Khi `return_trace=False`, không giữ toàn bộ tensor graph của các vòng để tiết kiệm memory. Khi explain/evaluation mới trả trace đầy đủ. Không detach `z_t`, `r_t` hoặc `delta_t` trong forward training; chỉ detach bản sao dùng để log.

## Shared weights và số vòng

Phiên bản chính dùng **shared weights**:

```text
Reasoner_0 = Reasoner_1 = ... = Reasoner_K
UpdateMLP_0 = UpdateMLP_1 = ... = UpdateMLP_K
```

Shared weights giúp diễn giải đây là một quy trình suy luận lặp, giảm tham số và tránh việc mô hình chỉ học một mạng sâu có tầng 1/tầng 2 khác nhau. `shared_weights=False` chỉ dùng làm ablation.

Số vòng cần kiểm tra:

- `K=1`: tương đương ECR baseline;
- `K=2`: cấu hình đề xuất đầu tiên;
- `K=3`: kiểm tra vòng refinement thêm;
- `K>3`: chưa nên làm khi chưa chứng minh hội tụ.

Không thêm adaptive halting ngay từ đầu. Halting network là future work vì làm khó ổn định và khó so sánh.

## Ý nghĩa kỳ vọng của từng vòng

### Vòng 0 – Perception

Trả lời: “Skeleton trực tiếp cho thấy những concept nào?” Ví dụ:

```text
hand_near_face = 0.62
arm_raised    = 0.58
fast_motion   = 0.31
```

### Vòng 1 – Local KG refinement

Dùng `require` và `alongside` để kiểm tra concept liên quan, đồng thời dùng `excludes` để giảm concept mâu thuẫn:

```text
arm_raised + hand_near_face
→ tăng evidence của upper-body interaction
fast_motion thấp
→ giảm evidence của các action cần dynamics nhanh
```

### Vòng 2 – Action-consistent refinement

Dùng action evidence để kiểm tra concept schema của các action cạnh tranh, rồi tính lại action score. Ví dụ action candidate `drink_water` cần nhóm concept tay–mặt và tư thế phù hợp; nếu các concept bắt buộc không nhất quán, score phải giảm.

Nếu chưa dùng action prototype, vòng 2 chỉ nên dùng action evidence với trọng số nhỏ. Nếu muốn explanation mạnh hơn, chuyển sang `action_mode="prototype"` để action score dựa trên concept schema thay vì linear action head tự do.

## Loss cho looping reasoning

Loss đề xuất:

\[
\mathcal{L} =
\mathcal{L}_{action}^{K}
+ \lambda_{aux}\sum_{t=1}^{K-1} w_t\mathcal{L}_{action}^{t}
+ \lambda_{concept}\mathcal{L}_{concept}^{K}
+ \lambda_{early}\mathcal{L}_{concept}^{0}
+ \lambda_{KG}\mathcal{L}_{KG}
+ \lambda_{stable}\mathcal{L}_{stable}.
\]

Khuyến nghị:

- action loss cuối có trọng số lớn nhất;
- auxiliary action loss vòng trước nhỏ hơn, khoảng `0.2–0.3`;
- concept loss ở `c0` để perception không mất tín hiệu;
- concept loss cuối để refinement không làm concept sai;
- KG loss áp dụng ở mỗi vòng hoặc lấy trung bình;
- stability loss nhỏ để tránh dao động:

\[
\mathcal{L}_{stable} =
\frac{1}{K-1}\sum_t \|c^{t+1}-c^t\|_1.
\]

Không ép `cᵗ⁺¹ = cᵗ`; loop phải được phép sửa concept. Delta penalty chỉ nên nhỏ, nếu quá mạnh model sẽ học identity mapping.

Giữ asymmetric/confidence-aware concept loss của ECR vì concept target là class-level và thường mất cân bằng. Không quay lại standard BCE thuần túy nếu concept dương quá thưa.

## Training schedule

### Phase 1 – ECR baseline

Chạy ECR hiện tại với một lần suy luận. Lưu top-1/top-5, concept macro/micro-F1, calibration, KG violation, số tham số và memory.

### Phase 2 – Loop warm-up

- khởi tạo `beta=0` hoặc `0.1`;
- train initial concept head và base reasoner;
- bật `K=2` nhưng update gate gần 0 trong vài epoch đầu;
- bảo đảm model mới bắt đầu gần baseline.

### Phase 3 – Iterative refinement

- anneal `beta` lên khoảng `0.2–0.5`;
- bật auxiliary action loss ở vòng 1;
- giữ prior-retention loss để rule không drift quá sớm;
- log `||delta_t||`, concept entropy và action confidence sau từng vòng.

### Phase 4 – End-to-end fine-tune

- backbone learning rate nhỏ;
- reasoner/update MLP learning rate lớn hơn;
- chạy ít nhất 3 random seeds;
- so sánh `K=1`, `K=2`, `K=3` trên cùng split và checkpoint.

## Ablation bắt buộc

| ID | Cấu hình | Mục đích |
|---|---|---|
| L0 | Original SkateFormer | perception baseline |
| L1 | ECR, một lần suy luận | ECR baseline |
| L2 | ECR + 2 loops, gate cố định | loop cơ bản |
| L3 | ECR + 2 loops, shared weights | phiên bản chính |
| L4 | ECR + 3 loops, shared weights | lợi ích vòng thứ ba |
| L5 | 2 loops, independent weights | kiểm tra shared structure |
| L6 | 2 loops, không KG | tách tác dụng KG |
| L7 | 2 loops, không concept refinement | chứng minh refinement cần thiết |
| L8 | 2 loops, không auxiliary loss | vai trò deep supervision |
| L9 | 2 loops + action prototype | prototype-constrained reasoning |
| L10 | 2 loops + CLIP alignment | semantic grounding giai đoạn sau |

Metric phải có action accuracy ở từng vòng, concept F1/AUPRC/calibration, `||cᵗ⁺¹-cᵗ||`, KG violation theo vòng, rule sparsity, prior drift, explanation fidelity, seed stability, latency và memory.

## Chống “loop giả”

Trước khi gọi đây là looping reasoning, cần kiểm tra:

1. `z_t` của vòng sau thật sự khác vòng trước trên mẫu khó.
2. Tắt `update_mlp` làm kết quả trở về gần L1.
3. Shuffle/zero KG evidence làm refinement thay đổi theo dự kiến.
4. Shared-weight loop vẫn hoạt động tốt.
5. Xóa concept quan trọng ở `c0` làm thay đổi các vòng sau và action.
6. Delta giữa các vòng ổn định, không dao động hoặc tăng vô hạn.
7. Phiên bản giải thích chính không nối linear action head trực tiếp từ feature để bypass loop.

## Thứ tự code

1. Giữ ECR hiện tại và chạy L0/L1.
2. Sửa `KGPrimedReasoner.forward()` để `return_trace=True` trả hidden logic evidence ổn định.
3. Tạo `IterativeKGReasoner` trong `reason_skateformer.py`.
4. Implement `K=2`, shared `update_mlp`, residual logit update.
5. Đặt `IterativeKGReasoner` sau `z0` trong `ReasonSkateFormer`; không sửa SkateFormer attention.
6. Thêm config: `num_loops`, `update_gate_init`, `shared_weights`.
7. Thêm `loop_trace`, chỉ giữ đầy đủ ở eval/explain mode.
8. Thêm auxiliary action loss, KG loss mỗi vòng và stability loss nhỏ.
9. Chạy L2–L8 trước prototype và CLIP.
10. Viết evaluator concept deletion, KG violation theo vòng và occlusion.

Feedback attention chỉ quay lại như future work sau khi looping reasoning đã có kết quả. Nếu sau này kết hợp, cần đánh giá nó như một nhánh perception refinement riêng.

## Cập nhật kết luận về base

Base code vẫn là **`ECR-SkateFormer/model/reason_skateformer.py`**. Phần cần mở rộng chính là `KGPrimedReasoner`; không dùng `DCLN/model_dcln.py` hay `REASON/model/model.py` làm nền chính.

```text
ReasonSkateFormer
    ├── SkateFormer backbone
    ├── spatial/temporal concept heads
    └── IterativeKGReasoner
          ├── shared KG rule reasoner
          ├── shared concept update cell
          └── loop trace
```
