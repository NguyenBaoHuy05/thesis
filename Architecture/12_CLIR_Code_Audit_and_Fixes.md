# Báo Cáo Rà Soát Mã Nguồn & Danh Sách Lỗi Cần Sửa (CLIR-SkateFormer)

> **Mục đích:** Ghi nhận toàn diện các lỗi logic, lỗi tính toán toán học, xung đột cấu hình và các điểm bất hợp lý được phát hiện trong codebase sau các đợt cập nhật (đặc biệt là sau commit tích hợp 3-Stage Training `7c49643` và kết quả thử nghiệm `0d35efd6`).
> **Đối tượng:** Toàn bộ hệ thống `CLIR-SkateFormer`, `LCR-SkateFormer` và các file cấu hình liên quan.

---

## Bảng Tổng Hợp Mức Độ Nghiêm Trọng

| STT | Phân Loại | Vị Trí File | Tóm Tắt Lỗi | Mức Độ |
|---|---|---|---|---|
| **1** | **Logic & Tối Ưu** | `CLIR-SkateFormer/main.py:879-884, 650-664` | Scheduler bị truyền `global_step` làm LR sập về `min_lr` (1e-6) ở Stage 3 | 🚨 **CRITICAL** |
| **2** | **Logic Đánh Giá** | `CLIR-SkateFormer/main.py:1108` | `best_model.pt` không bao giờ được lưu trong Stage 3 | 🚨 **CRITICAL** |
| **3** | **Hiệu Năng & Graph** | `CLIR-SkateFormer/main.py:365-382, 586-630` | Bỏ sót cập nhật `requires_grad` khi bắt đầu thẳng từ Stage 2 hoặc Resume | ⚠️ **MEDIUM** |
| **4** | **Toán Học & Loss** | `CLIR-SkateFormer/model/clir_skateformer.py:553-569` | Bình phương nhãn vô tình (`targets^2`) & sai lệch ASL gốc (ICCV 2021) | ⚠️ **MEDIUM** |
| **5** | **Xung Đột Cấu Hình** | `CLIR-SkateFormer/main.py:506-513` | Cấu hình `phase1_epochs` cũ phá vỡ freeze backbone ở Stage 2 mới | ⚠️ **MEDIUM** |
| **6** | **Thuật Toán Baseline** | `LCR-SkateFormer/model/lcr_skateformer.py:752-754` | Số mũ bị ngược `(1 - p)^γ` làm mất tính năng down-weight easy negative | ⚠️ **MEDIUM** |
| **7** | **Cấu Hình Môi Trường** | `CLIR-SkateFormer/config/train/ntu60_cs/CLIR_SkateFormer_j.yaml` | Hardcoded đường dẫn tuyệt đối máy cá nhân (`D:\KLTN_Huy\...`) | ℹ️ **MINOR** |

---

## Chi Tiết Từng Lỗi & Giải Pháp Khắc Phục

### 1. [CRITICAL] LR Scheduler bị kẹt ở `min_lr` (1e-6) khiến Stage 3 không được huấn luyện

#### Vị trí:
- [CLIR-SkateFormer/main.py:879-884](file:///d:/Thesis/CLIR-SkateFormer/main.py#L879-L884)
- [CLIR-SkateFormer/main.py:650-664](file:///d:/Thesis/CLIR-SkateFormer/main.py#L650-L664)

#### Mô tả lỗi:
Khi hệ thống kích hoạt cơ chế 3-Stage Training (Stage 1: Concept Warm-up Epoch 1–5, Stage 2: Logic Reasoning Epoch 6–50, Stage 3: End-to-End Fine-tuning Epoch 51–70):
1. Mỗi khi chuyển sang Stage 2 hoặc Stage 3, hàm `_maybe_switch_stage()` tạo mới một `CosineLRScheduler` với độ dài đúng bằng số bước còn lại:
   $$\text{num\_steps} = \text{remaining\_epochs} \times n\_iter \quad (\text{Stage 3 có 20 epoch } \implies \text{num\_steps} = 20 \times n\_iter)$$
2. Tuy nhiên, trong vòng lặp huấn luyện từng batch tại dòng 879:
   ```python
   if is_stage1 or self.arg.phase1_epochs <= 0 or self.current_training_phase == 1:
       self.lr_scheduler.step_update(self.global_step)
   else:
       self.lr_scheduler.step_update(self.phase_step)
       self.phase_step += 1
   self.global_step += 1
   ```
3. Khi cấu hình mặc định là Pure STE (`phase1_epochs: 0`), điều kiện `self.arg.phase1_epochs <= 0` **luôn luôn bằng True**. Do đó:
   - Code **luôn luôn truyền `self.global_step`** vào `lr_scheduler.step_update()`.
   - `self.global_step` là biến đếm tích lũy toàn cục từ epoch 1. Tại epoch 51 (bắt đầu Stage 3), `self.global_step` đã đạt khoảng **$50 \times n\_iter$ bước**.
   - Scheduler mới của Stage 3 chỉ được khởi tạo cho **$20 \times n\_iter$ bước**!
   - Vì bước truyền vào ($50 \times n\_iter$) lớn hơn rất nhiều so với `t_initial` ($20 \times n\_iter$), `CosineLRScheduler` lập tức coi như chu kỳ đã kết thúc và **kẹp cứng toàn bộ learning rate ở `min_lr` ($10^{-6}$)** ngay từ batch đầu tiên!

#### Hậu quả:
Suốt từ Epoch 51 đến 70, Backbone, Concept Heads, CRL Reasoner và UpdateCell đều chạy với $LR = 10^{-6}$, giai đoạn End-to-End Fine-tuning coi như bị tê liệt.

#### Giải pháp khắc phục:
Quản lý biến đếm bước riêng cho từng Stage (ví dụ `self.stage_step`), reset về 0 mỗi khi chuyển Stage:
```python
# Trong _maybe_switch_stage():
self.stage_step = 0

# Trong train loop:
self.lr_scheduler.step_update(self.stage_step)
self.stage_step += 1
self.global_step += 1
```

---

### 2. [CRITICAL] `best_model.pt` không bao giờ được lưu trong Stage 3

#### Vị trí:
- [CLIR-SkateFormer/main.py:1108](file:///d:/Thesis/CLIR-SkateFormer/main.py#L1108)

#### Mô tả lỗi:
Khi thêm Stage 3 vào pipeline, điều kiện lưu trọng số tốt nhất trong hàm `eval()` bị sót kiểm tra:
```python
# Chỉ lưu best_model.pt ở Stage 2 khi Action Classifier & Reasoner hoạt động thực sự
if (getattr(self, "current_stage", 2) == 2 or getattr(self.arg, "concept_warmup_epochs", 0) <= 0) and top1_acc > self.best_acc:
    self.best_acc = top1_acc
    self.best_acc_epoch = epoch + 1
    best_path = os.path.join(self.arg.work_dir, "best_model.pt")
    torch.save(..., best_path)
```
- Khi ở Stage 3 (`self.current_stage == 3`), điều kiện `current_stage == 2` là `False`.
- Nếu `concept_warmup_epochs > 0` (mặc định là 5), vế sau cũng là `False`.
- `(False or False) and ...` $\implies$ luôn `False`.

#### Hậu quả:
Stage 3 là giai đoạn mở khóa toàn bộ mô hình để đạt độ chính xác cao nhất (đỉnh acc của luận văn), nhưng nếu mô hình đạt Top-1 Acc kỷ lục ở Stage 3 (ví dụ 92.5%), checkpoint `best_model.pt` **sẽ không bao giờ được lưu**. Checkpoint tốt nhất được ghi nhận vẫn chỉ là checkpoint cũ của Stage 2.

#### Giải pháp khắc phục:
Sửa điều kiện tại dòng 1108:
```python
if (getattr(self, "current_stage", 2) in (2, 3) or getattr(self.arg, "concept_warmup_epochs", 0) <= 0) and top1_acc > self.best_acc:
```

---

### 3. [MEDIUM] Bỏ sót `requires_grad = False` cho Backbone & Concept Heads ở Stage 2

> **Nguyên tắc thiết kế cốt lõi:**
> Ở **Stage 2** (Logic Reasoning — thường gọi là Phase 2), mô hình **CHỈ ĐƯỢC TRAIN RIÊNG CRL Reasoner, ActionHead và UpdateCell**, **HOÀN TOÀN KHÔNG TRAIN CHO BACKBONE VÀ CONCEPT HEADS**.

#### Vị trí:
- [CLIR-SkateFormer/main.py:365-382](file:///d:/Thesis/CLIR-SkateFormer/main.py#L365-L382)
- [CLIR-SkateFormer/main.py:531-536](file:///d:/Thesis/CLIR-SkateFormer/main.py#L531-L536)
- [CLIR-SkateFormer/main.py:586-635](file:///d:/Thesis/CLIR-SkateFormer/main.py#L586-L635)

#### Mô tả lỗi:
1. **Lỗi bỏ sót `requires_grad = False` khi bắt đầu thẳng từ Stage 2 hoặc Resume checkpoint:**
   - Trong `Processor.__init__`, nếu khởi động với `start_epoch >= concept_warmup_epochs` (ví dụ `start_epoch: 5` hoặc resume từ checkpoint), `self.current_stage = 2`.
   - Đến đầu epoch huấn luyện, hàm `_maybe_switch_stage(epoch)` kiểm tra `if target_stage != self.current_stage:`. Vì cả hai đều bằng 2 nên khối đóng băng tham số bên trong `if` **bị bỏ qua**.
   - Hậu quả: Backbone giữ nguyên trạng thái mặc định của PyTorch (`requires_grad = True`). Dù LR là 0.0, PyTorch vẫn tính backward gradient qua toàn bộ backbone SkateFormer khổng lồ $\implies$ gây tốn 2x–3x VRAM và làm chậm 40–50% tốc độ huấn luyện.
2. **Lỗi đưa param LR = 0.0 vào Optimizer:**
   - Trong `_get_param_groups()` cho Stage 2 (dòng 531–536):
     ```python
     return [
         {"params": backbone_params, "lr": stage2_backbone_lr, "name": "backbone"},
         {"params": head_params,     "lr": stage2_concept_heads_lr, "name": "concept_heads"},
         {"params": crl_params,      "lr": crl_lr, "name": "crl"},
         {"params": uc_params,       "lr": update_cell_lr, "name": "update_cell"},
     ]
     ```
   - Khi `stage2_backbone_lr == 0.0` và `stage2_concept_heads_lr == 0.0`, `backbone_params` và `head_params` **vẫn bị đưa vào danh sách tối ưu hóa của AdamW**, khiến AdamW khởi tạo bộ nhớ tracking trạng thái (exp_avg, exp_avg_sq) cho hàng triệu tham số không cần thiết.

#### Giải pháp chuẩn hóa:
1. **Đóng băng cứng 100% Backbone & Concept Heads:**
   Tạo hàm phụ trợ `self._apply_stage_freeze(stage)` và gọi ngay trong `__init__` cũng như trong `_maybe_switch_stage(epoch)`:
   - Khi `stage == 2`: Set toàn bộ `p.requires_grad = False` cho Backbone, Pool Score và Concept Heads.
   - Chỉ set `p.requires_grad = True` cho Reasoner, CRL, ActionHead và UpdateCell.
2. **Loại bỏ param LR = 0 khỏi Optimizer:**
   Trong `_get_param_groups()` của Stage 2, **loại bỏ hoàn toàn `backbone_params` và `head_params`**, chỉ trả về các nhóm thực sự cần huấn luyện:
   ```python
   # Stage 2: CHỈ train riêng CRL (Reasoner, ActionHead, bypass) và UpdateCell
   return [
       {"params": crl_params, "lr": crl_lr, "name": "crl"},
       {"params": uc_params,  "lr": update_cell_lr, "name": "update_cell"},
   ]
   ```

---

### 4. [MEDIUM] Lỗi bình phương nhãn vô tình (`targets^2`) & sai lệch ASL gốc (ICCV 2021)

#### Vị trí:
- [CLIR-SkateFormer/model/clir_skateformer.py:553-569](file:///d:/Thesis/CLIR-SkateFormer/model/clir_skateformer.py#L553-L569)
- [Architecture/09_CLIR-SkateFormer_Implementation_Plan.md:322-325](file:///d:/Thesis/Architecture/09_CLIR-SkateFormer_Implementation_Plan.md#L322-L325)

#### Mô tả lỗi & Vấn đề toán học:
1. **Lỗi bình phương nhãn trong code:**
   ```python
   pos_loss = F.softplus(-logits) * targets
   neg_loss = F.softplus(logits) * p_m.pow(self.gamma_neg) * (1.0 - targets)

   weights = (
       targets * self.concept_pos_weight
       + (1.0 - targets) * self.concept_neg_weight
   )
   loss = pos_loss + neg_loss
   return (loss * weights).sum() / denom
   ```
   - `pos_loss` đã nhân `targets`. Khi nhân tiếp `loss * weights`, phần positive bị nhân thêm `targets * pos_weight` $\implies$ thành $\text{targets}^2 \cdot w_+$.
   - Với nhãn hard $\{0, 1\}$, $1^2 = 1, 0^2 = 0$ thì không đổi. Nhưng nếu sử dụng **soft labels, label smoothing hoặc confidence score** $\in (0, 1)$, nhãn sẽ bị bóp méo bình phương vô cớ ($0.8 \to 0.64$).
2. **Sai lệch so với Asymmetric Loss (ASL) gốc (Ridnik et al., ICCV 2021):**
   - ASL gốc dùng: $L_- = (p_m)^{\gamma_-} \log(1 - p_m)$ với $p_m = \max(p - m, 0)$. Tức là $p_m$ nằm cả trong số mũ lẫn trong hàm $\log$.
   - Bản trong code hiện tại chỉ lấy $(p_m)^{\gamma_-}$ nhân với $\text{softplus}(z) = -\log(1 - p)$, đồng thời nhân thêm trọng số tĩnh $r_{ic}$ (vốn thừa hưởng từ code LCR/ECR cũ).
   - Trong báo cáo / luận văn, không nên ghi đây là "ASL chuẩn ICCV 2021" mà cần gọi chính xác là **"Modified Asymmetric BCE with Margin Down-weighting"** hoặc chuẩn hóa lại công thức.

#### Giải pháp khắc phục:
Chuẩn hóa lại code không bị nhân 2 lần nhãn:
```python
pos_loss = self.concept_pos_weight * targets * F.softplus(-logits)
neg_loss = (
    self.concept_neg_weight
    * (1.0 - targets)
    * p_m.pow(self.gamma_neg)
    * F.softplus(logits)
)
if mask is not None:
    mask_f = mask.float()
    pos_loss = pos_loss * mask_f
    neg_loss = neg_loss * mask_f

loss = pos_loss + neg_loss
weights = targets * self.concept_pos_weight + (1.0 - targets) * self.concept_neg_weight
if mask is not None:
    weights = weights * mask_f
return loss.sum() / weights.sum().clamp_min(1.0)
```

---

### 5. [MEDIUM] Xung đột cấu hình `phase1_epochs` cũ phá vỡ freeze backbone của Stage 2 mới

#### Vị trí:
- [CLIR-SkateFormer/main.py:506-513](file:///d:/Thesis/CLIR-SkateFormer/main.py#L506-L513)

#### Mô tả lỗi:
```python
is_staged_phase1 = (
    self.current_training_phase == 1
    and not getattr(self, "is_pure_soft", False)
    and 0 < self.arg.phase1_epochs < self.arg.num_epoch
)
if is_staged_phase1:
    return [{"params": self.model.parameters(), "lr": base_lr}]
```
- Nếu cấu hình chạy chế độ 2-Phase Staged Training cũ (`phase1_epochs > 0`, ví dụ 30) kết hợp với `concept_warmup_epochs: 5`:
- Ở Epoch 6 (bắt đầu Stage 2), `current_stage == 2` nhưng `current_training_phase == 1` (vì epoch < 30).
- Khối lệnh trên sẽ gộp **toàn bộ tham số mô hình vào một param group duy nhất với `base_lr`**, bỏ qua hoàn toàn `stage2_backbone_lr: 0.0`.

#### Hậu quả:
Backbone bị huấn luyện tự do ngay từ Epoch 6 với LR lớn, làm hỏng hoàn toàn chiến lược 3-Stage Training.

#### Giải pháp khắc phục:
Chỉ cho phép `is_staged_phase1` kích hoạt khi không sử dụng cơ chế multi-stage (hoặc ưu tiên `current_stage` trước `current_training_phase`).

---

### 6. [MEDIUM] Lỗi ngược số mũ trong `LCR-SkateFormer` làm mất tác dụng down-weight

#### Vị trí:
- [LCR-SkateFormer/model/lcr_skateformer.py:752-754](file:///d:/Thesis/LCR-SkateFormer/model/lcr_skateformer.py#L752-L754)

#### Mô tả lỗi:
```python
negative_loss = (
    F.softplus(logits)
    * (1.0 - probabilities).pow(self.asymmetric_gamma_neg)
    * (1.0 - targets)
)
```
- Với mẫu âm ($y=0$), mẫu âm dễ (easy negative) là mẫu có xác suất dự đoán $p \to 0$.
- Khi $p \to 0 \implies (1 - p) \to 1 \implies (1 - p)^\gamma \to 1$.
- Hệ số phạt của mẫu âm dễ vẫn xấp xỉ 1, hoàn toàn **không bị triệt tiêu/down-weight** như nguyên lý của Focal Loss / ASL.
- Đúng ra phải là $p^\gamma$ hoặc $(p_m)^\gamma$ (để khi $p \to 0$ thì hệ số $\to 0$).

#### Hậu quả:
Nếu bạn chạy lại mô hình `LCR-SkateFormer` làm baseline đối chứng cho luận văn, kết quả của LCR sẽ bị suy giảm do lỗi này.

#### Giải pháp khắc phục:
Sửa `(1.0 - probabilities).pow(...)` thành `probabilities.pow(...)` hoặc áp dụng cơ chế margin clipping như CLIR.

---

### 7. [MINOR] Hardcoded đường dẫn tuyệt đối máy cá nhân trong file cấu hình mới

#### Vị trí:
- [CLIR-SkateFormer/config/train/ntu60_cs/CLIR_SkateFormer_j.yaml:7-8, 43](file:///d:/Thesis/CLIR-SkateFormer/config/train/ntu60_cs/CLIR_SkateFormer_j.yaml#L7-L8)

#### Mô tả lỗi:
File config vừa commit chứa các đường dẫn tuyệt đối tới máy cá nhân `D:\KLTN_Huy\...`:
```yaml
train_feeder_args:
  data_path: D:\KLTN_Huy\thesis\data\ntu\NTU60_CS.npz
  concepts_csv: D:\KLTN_Huy\thesis\CLIR-SkateFormer\concepts\ntu_rgbd_60_120.csv
...
model_args:
  backbone_weights: D:\KLTN_Huy\thesis\pretrained\ntu60_CSub\SkateFormer_j.pt
```

#### Hậu quả:
Khi chạy trên workspace `d:\Thesis` hoặc đưa lên Google Colab / Linux Cluster, code sẽ báo lỗi `FileNotFoundError` ngay lập tức.

#### Giải pháp khắc phục:
Đổi về đường dẫn tương đối hoặc khớp với workspace hiện tại `d:\Thesis`.
