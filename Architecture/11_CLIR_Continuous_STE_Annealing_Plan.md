# KẾ HOẠCH NÂNG CẤP: CONTINUOUS STE ANNEALING CHO CLIR-SKATEFORMER

> **Tài liệu:** `Architecture/11_CLIR_Continuous_STE_Annealing_Plan.md`  
> **Mục tiêu:** Bỏ chế độ `soft` riêng biệt, chuyển sang chạy **100% STE với cơ chế làm mượt liên tục (Continuous Annealing)**, loại bỏ cú sốc chuyển pha và hiện tượng triệt tiêu luật ("Dead Rules") tại Epoch 31.

---

## 1. Bối cảnh & Vấn đề Thực nghiệm

### 1.1 Hiện tượng quan sát được trong lần chạy trước
* **Kết quả tại Epoch 53/70:** `Loss = 3.6477`, `Acc = 23.10%`.
* Tốc độ học bị tắc nghẽn, độ chính xác tăng rất chậm suốt từ Epoch 31 đến Epoch 53.

### 1.2 Nguyên nhân gốc rễ (4 cú sốc cùng lúc ở Epoch 31)
1. **Nhảy vọt chế độ:** Đột ngột chuyển từ `soft` (số thực mượt) sang `ste` (nhị phân cứng $\{0, 1\}$). Một literal đang có giá trị $0.48$ lập tức bị làm tròn về $0$. Trong phép toán hội AND ($\prod_i l_i^{w_i}$), chỉ cần một literal bằng $0$ là toàn bộ luật bằng $0$ ($\log(0 + \epsilon) = -13.8$). Hơn 80% luật AND bị triệt tiêu ("Dead Conjunctions").
2. **Siết ngưỡng lọc:** $\beta_{\max}$ tăng vọt từ $0.1$ lên $0.5$ (nghiêm ngặt gấp 5 lần).
3. **Bật phạt thưa đột ngột:** $\lambda_{\text{reg}}$ nhảy từ $0$ lên $0.005$.
4. **Reset toàn bộ Optimizer:** Code gọi `_init_optimizer()`, xóa sạch toàn bộ momentum ($m_t, v_t$) của AdamW và dập Learning Rate của Backbone xuống 10 lần ngay tại lúc mô hình đang cần gradient để thích nghi nhất.

---

## 2. Mục tiêu Cốt lõi (Core Objectives)

1. **Bỏ chế độ `soft` riêng biệt:** Thống nhất mô hình chạy trên **1 kiến trúc suy luận duy nhất (STE)** từ Epoch 1 đến Epoch 70.
2. **Loại bỏ "Cú sốc chuyển pha" (Zero Discretization Shock):** Thay thế việc đổi pha gián đoạn bằng **hàm biến thiên liên tục (Continuous Annealing)** theo tiến trình $progress = \frac{epoch}{num\_epoch - 1} \in [0.0, 1.0]$.
3. **Không bao giờ Reset Optimizer:** Duy trì 1 Optimizer AdamW và 1 LR Scheduler duy nhất xuyên suốt toàn bộ quá trình huấn luyện, bảo toàn momentum và quá trình giảm tốc độ học (Cosine Decay) mượt mà.
4. **Bảo toàn 100% Tính Diễn Giải:** Đến cuối quá trình huấn luyện, mô hình đạt tính nhị phân $\{0, 1\}$ tuyệt đối, sẵn sàng để trích xuất luật DNF chuẩn toán học với Fidelity cao.

---

## 3. Yêu cầu Kỹ thuật Cần Đạt (Technical Requirements)

### 3.1 Quy luật Annealing Tham số (Theo tiến độ $p \in [0.0, 1.0]$)

| Tham số | Đầu huấn luyện ($p=0.0$) | Cuối huấn luyện ($p=1.0$) | Công thức biến thiên | Ý nghĩa kỹ thuật |
| :--- | :---: | :---: | :--- | :--- |
| **Nhiệt độ STE ($\tau$)** | $1.0$ | $10.0$ | $\tau(p) = 1.0 + p \times 9.0$ | $\tau=1.0$ giữ độ dốc sigmoid thoải, gradient lan truyền mượt; $\tau=10.0$ ép nhát cắt nhị phân sắc nét. |
| **Hệ số STE ($\alpha$)** | $0.0$ | $1.0$ | $\alpha(p) = \min(1.0, \max(0.0, \frac{p - 0.1}{0.8}))$ | Forward kết hợp mượt $l = (1-\alpha)p_{\text{soft}} + \alpha p_{\text{ste}}$, cho phép luật thích nghi không bị dead rule. |
| **Ngưỡng luật ($\beta_{\max}$)** | $0.1$ | $0.5$ | $\beta_{\max}(p) = 0.1 + p \times 0.4$ | Khởi đầu lỏng để nhiều luật được khám phá, về sau siết chặt để luật cô đọng. |
| **Phạt thưa ($\lambda_{\text{reg}}$)** | $0.0$ | $0.005$ | $p < 0.3 \implies 0.0$<br>$p \ge 0.3 \implies 0.005 \times \frac{p - 0.3}{0.7}$ | 30% epoch đầu tập trung học đặc trưng concept, 70% sau mới phạt thưa luật. |

### 3.2 Chiến lược Tối ưu hóa (Optimizer Strategy)
* **Khởi tạo Per-group LR ngay từ Epoch 0:**
  * Backbone (SkateFormer): `base_lr * 0.1` (bảo vệ trọng số tiền huấn luyện).
  * Concept Heads & CRL Reasoner: `base_lr * 1.0` (học mạnh các tầng logic mới).
  * Update Cell & Gating: `base_lr * 1.0`.
* **Scheduler duy nhất:** `CosineLRScheduler` với 5 epoch warmup, decay dần về `min_lr = 1e-5` ở epoch 70. **Không tạo lại optimizer ở bất kỳ epoch nào.**

---

## 4. Tiêu chí Nghiệm thu & Chỉ số Đo lường (Acceptance Criteria & KPIs)

| Tiêu chí | Phiên bản cũ (Epoch 53) | Mục tiêu Phiên bản mới (Continuous STE) |
| :--- | :---: | :---: |
| **Độ sụt giảm Acc giữa chừng** | Tụt thảm hại ở Ep 31 (Shock drop) | **Độ dốc Acc tăng đều, không có gãy gập** |
| **Độ chính xác Huấn luyện (Train Acc)** | 23.10% | **$\ge 40.0\%$** |
| **Độ chính xác Đánh giá (Val Acc)** | $\approx 20\% - 22\%$ | **$\ge 35.0\% - 42.0\%$** |
| **Tỉ lệ luật hoạt động (Active Rules)** | $< 20\%$ (phần lớn rule ra 0) | **$\ge 60\%$** luật AND duy trì gradient |
| **Fidelity khi Trích xuất luật DNF** | Không ổn định nếu train soft | **$\ge 90\%$** tương đồng giữa luật và model |
| **Tính toàn vẹn số học** | Cảnh báo NumPy Deprecation | **Triệt tiêu toàn bộ cảnh báo deprecated và NaN/Inf** |

---

## 5. Lộ trình Triển khai Chi tiết (Implementation Roadmap)

### Bước 1: Nâng cấp Tầng Literal (`model/crl/literal_layer.py`)
- Bổ sung tham số `ste_weight` (hệ số $\alpha$) vào `LiteralLayer`.
- Trong chế độ STE:
  $$\text{Forward: } l = (1 - \alpha) \cdot \sigma(\tau \cdot z) + \alpha \cdot \text{BinarizeSTE}(z, \tau)$$
  $$\text{Backward: } \frac{\partial l}{\partial z} = \frac{\partial \sigma(\tau \cdot z)}{\partial z}$$
- Khi $\alpha = 0$: tương đương hàm mượt hoàn toàn.
- Khi $\alpha = 1$: tương đương nhị phân cứng 100%.

### Bước 2: Nâng cấp Reasoner & CLIR Model (`model/crl/crl_reasoner.py`, `model/clir_skateformer.py`)
- Hiện thực hóa hàm `set_progress(progress: float)`:
  - Tự động tính và cập nhật $\tau$, $\alpha$, $\beta_{\max}$, $\lambda_{\text{reg}}$ theo đúng công thức ở Mục 3.1.
  - Bảo lưu `set_phase(phase)` để tương thích ngược.

### Bước 3: Cập nhật Vòng lặp Huấn luyện (`generate_notebook.py`)
- Định cấu hình `TRAIN_CONFIG` mới:
  - Loại bỏ các thiết lập giật pha `phase1_epochs`.
  - Bổ sung dải nhiệt độ: `ste_temp_start = 1.0`, `ste_temp_end = 10.0`.
  - Bổ sung dải ngưỡng: `beta_start = 0.1`, `beta_end = 0.5`.
- Sửa đổi `CLIRTrainer`:
  - Khởi tạo Optimizer per-group 1 lần duy nhất trong `__init__`.
  - Thay thế `check_phase_switch` bằng `on_epoch_start(epoch)` gọi `self.model.set_progress(progress)`.
  - Vá lỗi NumPy deprecation trong `feeders/tools.py` (`np.random.rand(1)` trả về array thay vì float).

### Bước 4: Kiểm thử & Xác thực Cục bộ
- Cập nhật `verify_model.py` để chạy forward/backward kiểm tra gradient qua các mốc $progress \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$.
- Xác nhận không có lỗi chia 0 hoặc chết gradient.

### Bước 5: Tái sinh File Notebook (`CLIR_SkateFormer_Colab.ipynb`)
- Chạy `python generate_notebook.py` để ghi đè toàn bộ notebook bằng phiên bản Continuous STE mới nhất.
