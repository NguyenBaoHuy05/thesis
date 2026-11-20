# Đề xuất hướng nghiên cứu Bài 2: Sub-Action Concept Mapping cho ZSAR (tận dụng feature 256-D sẵn có)

Ngày tạo: 2026-11-19  
Bối cảnh: Bài 1 (**PRIZM**) can thiệp vào phía **ngữ nghĩa lớp** (KG + ORE + PPC + COA-Fuse) và giữ nguyên mạng ánh xạ của **Flora**. Bài 2 đi theo hướng song song, tập trung cải tiến phía **Mapping** (từ đặc trưng xương 256-D trích xuất sẵn của STAR / SynSE / ST-GCN suy ra trực tiếp **concept hành động nhỏ** thay vì ánh xạ hộp đen thẳng sang lớp).

---

## Phần 1: Tổng hợp 4 hướng Mapping dùng lại feature 256-D (Hướng 1, 2, 3, 5)

> **Tiền đề thực nghiệm từ PRIZM (`nesy0_probe.py`):**  
> Đặc trưng Shift-GCN 256-D đã gộp **vẫn chứa thông tin chuyển giao tốt về các concept hành động nhỏ** (MLP probe đạt Unseen Concept AUC `0.858–0.882` so với luật hoán vị `0.535–0.820`). Tuy nhiên, phân loại lớp *chỉ* từ concept nhị phân rời rạc bị tụt điểm (`25.8%–72.9%`) do nhiều hành động tinh vi dùng chung phần lớn concept (ví dụ *reading* vs. *writing*). Vì vậy, phương pháp mapping mới cần vừa suy ra được concept nhỏ, vừa bảo toàn được độ phân biệt tinh vi.

### 1. Disentangled Concept–Residual Mapping *(Ưu tiên triển khai trước)*
- **Ý tưởng cốt lõi:** Tách không gian tiềm ẩn $\mathbf{z} \in \mathbb{R}^D$ của mạng mapping thành hai không gian con trực giao:
  - **Nhánh Concept ($\mathbf{z}_c \in \mathbb{R}^{D_c}$):** ánh xạ đặc trưng xương sang không gian các hành động nhỏ (motion primitives), được giám sát trực tiếp bằng nhãn concept và/hoặc căn chỉnh với CLIP embedding của từng primitive.
  - **Nhánh Residual ($\mathbf{z}_r \in \mathbb{R}^{D_r}$):** lưu giữ thông tin chuyển động tinh tế mà từ điển concept chưa mô tả hết, có ràng buộc trực giao với $\mathbf{z}_c$ để tránh rò rỉ thông tin và hạn chế *reasoning shortcuts*.
- **Suy luận:**
  - Từ $\mathbf{z}_c$ → suy ra danh sách/xác suất các **concept hành động nhỏ** đang diễn ra trong mẫu xương unseen.
  - Từ $[\mathbf{z}_c, \mathbf{z}_r]$ → đưa qua Flow Matching (DiT của Flora) để phân loại **lớp unseen** với đầy đủ thông tin tinh vi.
- **Ưu / Nhược trên 256-D:**
  - *Ưu:* Khớp tự nhiên với vector 256-D và Phase 1 VAE của Flora; rủi ro kỹ thuật thấp nhất; giải quyết trực tiếp bài toán *accuracy–interpretability trade-off* của Concept Bottleneck Models (CBM) trong ZSL.
  - *Nhược:* Cần thiết kế kỹ ràng buộc trực giao và cân bằng trọng số loss để nhánh $\mathbf{z}_r$ không "lấn át" nhánh $\mathbf{z}_c$.

### 2. Compositional Flow Matching trên không gian Concept (Concept-Factorized DiT)
- **Ý tưởng cốt lõi:** Thay vì để Phase 2 DiT của Flora học một trường vận tốc nguyên khối $v_\theta(\mathbf{z}_t, t \mid c)$ đi thẳng từ class embedding $\mathbf{z}_0^c$ đến skeleton latent $\mathbf{z}_1$, mô hình học **trường vận tốc ở mức concept nhỏ** $v_\theta(\mathbf{z}_t, t \mid p)$ và tổng hợp trường vận tốc của hành động phức hợp bằng phép chồng chập (superposition):
  $$v_\theta(\mathbf{z}_t, t \mid c) = \sum_{p \in \mathcal{R}_c} w_{cp}\, v_\theta(\mathbf{z}_t, t \mid p) - \lambda \sum_{q \in \mathcal{F}_c^o} \tilde{w}_{cq}\, v_\theta(\mathbf{z}_t, t \mid q)$$
- **Suy luận:**
  - Với mẫu xương unseen $\mathbf{z}_1$, tính sai số vận tốc $e(\mathbf{z}_1, p)$ cho từng concept đơn lẻ $p$ → các concept có sai số thấp chính là hành động nhỏ đang xảy ra.
  - Tổng hợp trường vận tốc theo cấu trúc concept của từng lớp unseen $c$ để phân loại lớp.
- **Ưu / Nhược trên 256-D:**
  - *Ưu:* Tính mới (novelty) rất cao (kết nối *Compositional Generative Models / Flow Matching* với ZSAR); tận dụng trực tiếp feature 256-D và khung Flora.
  - *Nhược:* DiT gốc của Flora rất nhẹ (1 layer, 200 iterations) nên khi chuyển sang tổ hợp nhiều trường vận tốc sẽ tốn công tinh chỉnh kiến trúc, tốc độ tính toán và độ ổn định hội tụ hơn Hướng 1.

### 3. Geometric / Entailment Concept Mapping (Hyperbolic hoặc Box Embeddings)
- **Ý tưởng cốt lõi:** Mô hình hóa quan hệ **bao hàm phân cấp (part–whole / entailment)** giữa hành động lớn (class) và hành động nhỏ (concept) trong không gian **Hyperbolic (Entailment Cones)** hoặc **Box Embeddings**:
  - Mỗi concept nhỏ là một vùng rộng (hộp/nón lớn); mỗi lớp hành động nằm tại **phần giao** của các vùng concept tạo nên nó (và tách rời với các vùng concept bị cấm).
  - Đặc trưng xương 256-D được ánh xạ vào một điểm (hoặc hộp nhỏ) trong không gian này.
- **Suy luận:** Kiểm tra điểm biểu diễn xương rơi vào những vùng concept nào → suy ra tập concept nhỏ → gán cho lớp unseen tương ứng với phần giao sát nhất.
- **Ưu / Nhược trên 256-D:**
  - *Ưu:* Cơ sở toán học đẹp, biểu diễn tự nhiên quan hệ nhiều-nhiều giữa concept và class cũng như ràng buộc phủ định (`forbid`).
  - *Nhược:* Tối ưu hình học Hyperbolic/Box dễ gặp vấn đề gradient biến mất/bùng nổ và khó ghép trơn tru với VAE Gaussian + Flow Matching của Flora (thường phải thay hẳn nhánh phân loại).

### 5. Sparse Concept Dictionary Mapping (Mã hóa thưa trên từ điển Concept)
- **Ý tưởng cốt lõi:** Xem tập $P$ concept (được mã hóa bởi CLIP và chiếu sang không gian chung) như một **từ điển nguyên tử (concept dictionary)** $\mathbf{D} \in \mathbb{R}^{P \times d}$. Mạng mapping biểu diễn đặc trưng xương $\mathbf{f} \in \mathbb{R}^{256}$ dưới dạng **tổ hợp tuyến tính thưa, không âm** (sparse coding / L1-regularized attention hoặc một số bước unrolled ISTA):
  $$\mathbf{z}(\mathbf{f}) \approx \sum_{p=1}^{P} \alpha_p(\mathbf{f})\,\mathbf{d}_p + \mathbf{r}(\mathbf{f}), \quad \alpha_p(\mathbf{f}) \ge 0,\ \|\boldsymbol{\alpha}\|_1 \text{ nhỏ}$$
- **Suy luận:** Các hệ số $\alpha_p(\mathbf{f}) > 0$ chỉ trực tiếp **những hành động nhỏ nào đang kích hoạt và mức độ đóng góp của từng động tác**; vector $\boldsymbol{\alpha}(\mathbf{f})$ (hoặc tái tạo $\mathbf{z}(\mathbf{f})$) dùng để đối chiếu với lớp unseen.
- **Ưu / Nhược trên 256-D:**
  - *Ưu:* Rất gọn, minh bạch, chạy trực tiếp trên 256-D, tự nhiên chọn lọc chỉ vài concept thật sự xảy ra thay vì kích hoạt dàn trải.
  - *Nhược:* Nếu chỉ dùng tái tạo từ từ điển thuần túy thì vẫn gặp nút cổ chai tinh vi (cần thêm thành phần dư $\mathbf{r}(\mathbf{f})$ — khi đó sẽ hội tụ rất gần với Hướng 1).

### Bảng so sánh nhanh trên feature 256-D

| Hướng | Điểm chạm vào Flora | Khả năng suy ra concept nhỏ | Độ khó triển khai / rủi ro | Tính mới học thuật |
| :--- | :--- | :--- | :--- | :--- |
| **1. Concept–Residual Disentangled** | Phase 1 (Encoder + Losses) | Trực tiếp (từ $\mathbf{z}_c$) | Thấp (rất ổn định) | Cao (CBM + Residual trực giao cho ZSAR) |
| **2. Compositional Flow Matching** | Phase 2 (DiT velocity field) | Trực tiếp (qua velocity error theo concept) | Trung bình – Cao | Rất cao (Compositional FM) |
| **3. Hyperbolic / Box Entailment** | Phase 1 + Classifier | Trực tiếp (quan hệ bao hàm vùng) | Cao (khó tối ưu ổn định) | Cao |
| **5. Sparse Concept Dictionary** | Phase 1 (Encoder) | Trực tiếp (hệ số thưa $\alpha_p$) | Thấp – Trung bình | Khá cao (có thể gộp với Hướng 1) |

---

## Phần 2: Phân tích chi tiết các yếu tố cần để triển khai Hướng 1 trước

Mục tiêu: kế thừa trực tiếp từ `Flora/models/fm.py` (`MappingNet`) và `Flora/trainer.py` (`Trainer`) trong thư mục dự án mới mà **không phá vỡ baseline Flora**.

### 2.1. Dữ liệu & Tín hiệu giám sát Concept (Inputs & Targets)
1. **Đặc trưng xương sẵn có (không cần trích lại):**
   - `STAR` (1s-Shift-GCN, 256-D), `SynSE` (4s-Shift-GCN, 256-D), và `ST-GCN` cho random splits (NTU-60, NTU-120, PKU-MMD I).
   - Lưu ý trong `Flora/models/fm.py` (dòng 328): Flora lặp vector `(B, 256)` thành `(B, L+1, 256)` với `L+1 = 76` token để khớp kích thước với text. Với nhánh concept, nên tính dự đoán concept từ biểu diễn đã gộp theo token (ví dụ `mu_ske_c.mean(dim=1)`) để tránh tính dư thừa 76 lần.
2. **Ma trận nhãn Concept từ KG sẵn có:**
   - Tận dụng `kg/primitive_v2.json` + `kg/kg_rules_v2.json` ($P = 82$ primitives; có thể mở rộng lên $P = 111$ với `v2seq` khi thử nghiệm thêm thứ tự).
   - Với mỗi lớp seen $y \in \mathcal{Y}^s$, tạo:
     - **Target nhị phân** $\mathbf{t}_y \in \{0, 1\}^P$: $t_{y,p} = 1$ nếu $p \in \mathcal{R}_y$ (`require`), $t_{y,p} = 0$ nếu không (hoặc gắn trọng số cao hơn cho $p \in \mathcal{F}_y$ để nhấn mạnh `forbid`).
     - **Mask độ phủ tập seen** $\mathbf{m}_{\text{seen}} \in \{0, 1\}^P$: chỉ tính supervised loss trên các primitive có ít nhất 1 lớp seen kích hoạt trong split hiện tại (tránh ép các primitive chỉ xuất hiện ở tập unseen về 0!).
     - **CLIP embedding của từ điển primitive** $\mathbf{E}_{\mathcal{P}} \in \mathbb{R}^{P \times 768}$ (lấy pooled token cuối đã chuẩn hóa L2 từ cache CLIP).

### 2.2. Kiến trúc mạng Mapping (Architecture)
Giữ tổng chiều tiềm ẩn `latent_dim = D = 768` để **tương thích 100% với Decoder và Phase 2 DiT của Flora**, tách $D = D_c + D_r$ (gợi ý khởi đầu: $D_c = 256, D_r = 512$ hoặc $384 + 384$):

1. **Motion Encoder tách nhánh ($E_v$):**
   - Backbone chung (MLP trên 256-D) hoặc 2 nhánh riêng chiếu ra:
     - $(\mu_{\text{ske}, c}, \log\sigma^2_{\text{ske}, c}) \in \mathbb{R}^{B \times (L+1) \times D_c}$
     - $(\mu_{\text{ske}, r}, \log\sigma^2_{\text{ske}, r}) \in \mathbb{R}^{B \times (L+1) \times D_r}$
   - Ghép lại thành $\mu_{\text{ske}} = [\mu_{\text{ske}, c}, \mu_{\text{ske}, r}] \in \mathbb{R}^{B \times (L+1) \times D}$ và reparameterize thành $\mathbf{z}_1 = [\mathbf{z}_{1,c}, \mathbf{z}_{1,r}]$.
2. **Semantic Encoder tách tương ứng ($E_t$):**
   - Xuất $(\mu_{\text{text}, c}, \mu_{\text{text}, r})$ với cùng kích thước $(D_c, D_r)$ để căn chỉnh từng nhánh với xương:
     - $\mu_{\text{text}, c} \leftrightarrow \mu_{\text{ske}, c}$ (căn chỉnh không gian Concept).
     - $\mu_{\text{text}, r} \leftrightarrow \mu_{\text{ske}, r}$ (căn chỉnh không gian Residual).
3. **Concept Prediction Head ($g_\phi$) — Điểm then chốt để chuyển giao sang Unseen:**
   - **Không dùng** một lớp `nn.Linear(D_c, P)` độc lập thuần túy vì các primitive hiếm ở tập seen sẽ không học được trọng số tốt.
   - **Nên dùng CLIP-Anchored Concept Head (Semantic Prototype Readout):**
     - Chiếu $\bar{\mu}_c = \text{mean\_token}(\mu_c) \in \mathbb{R}^{B \times D_c}$ qua một lớp tuyến tính $W_c: \mathbb{R}^{D_c} \to \mathbb{R}^{768}$ (hoặc MLP nhẹ + LayerNorm).
     - Tính logit cho từng primitive $p$ bằng độ tương đồng cosine với CLIP embedding cố định $\mathbf{e}_p$ của primitive đó:
       $$\hat{p}_k(\mathbf{x}) = \sigma\!\big(\tau \cdot \cos(W_c \bar{\mu}_{\text{ske}, c},\; \mathbf{e}_k) + b_k\big)$$
     - Nhờ neo vào $\mathbf{e}_k$ cố định của CLIP, ngay cả những concept ít xuất hiện ở tập seen vẫn giữ đúng cấu trúc ngữ nghĩa khi chuyển sang tập unseen!
4. **Decoders & Phase 2 DiT:**
   - Giữ nguyên `motion_decoder(D -> 256)`, `semantic_decoder(D -> 1536)` và `velocity_predictor` (DiT nhận `in_channels = D`) trên $[\mathbf{z}_c, \mathbf{z}_r]$.

### 2.3. Hệ thống hàm mất mát (Phase 1 Losses)
Tổng hàm mất mát Phase 1:
$$\mathcal{L}_{\text{Phase1}} = \mathcal{L}_{\text{Flora}} + \lambda_c\, \mathcal{L}_{\text{concept}} + \lambda_{\text{orth}}\, \mathcal{L}_{\text{orth}}$$

1. **$\mathcal{L}_{\text{Flora}}$ (giữ nguyên của Flora):**
   - `loss_recon` (tái tạo 4 chiều `s2s, t2t, s2t, t2s`) + `lambda_align * (loss_mu + loss_log_var)`.
2. **$\mathcal{L}_{\text{concept}}$ (giám sát nhánh Concept cho cả 2 modality):**
   - Áp dụng Weighted/Masked Binary Cross-Entropy (có `pos_weight` để xử lý mất cân bằng vì trung bình mỗi lớp chỉ bật `3.1 / 82` primitives):
     $$\mathcal{L}_{\text{concept}} = \text{BCE}_{\text{masked}}\big(\hat{\mathbf{p}}(\mu_{\text{ske}, c}),\; \mathbf{t}_y\big) + \alpha_t\, \text{BCE}_{\text{masked}}\big(\hat{\mathbf{p}}(\mu_{\text{text}, c}),\; \mathbf{t}_y\big)$$
   - Việc giám sát *cả* $\mu_{\text{ske}, c}$ và $\mu_{\text{text}, c}$ qua **cùng một head $g_\phi$ dùng chung** buộc hai modality phải gặp nhau đúng tại tọa độ các hành động nhỏ.
3. **$\mathcal{L}_{\text{orth}}$ (ràng buộc tách biệt Concept – Residual):**
   - Chuẩn hóa theo batch (zero-mean, unit-variance theo từng chiều) hai ma trận $\bar{\mathbf{Z}}_c \in \mathbb{R}^{B \times D_c}$ và $\bar{\mathbf{Z}}_r \in \mathbb{R}^{B \times D_r}$, sau đó tối thiểu hóa ma trận tương quan chéo:
     $$\mathcal{L}_{\text{orth}} = \frac{1}{D_c D_r}\left\| \frac{1}{B}\, \bar{\mathbf{Z}}_c^\top \bar{\mathbf{Z}}_r \right\|_F^2$$
   - Tác dụng: triệt tiêu tương quan tuyến tính giữa nhánh Concept và nhánh Residual; đảm bảo nhánh Concept học trọn vẹn thông tin primitive còn nhánh Residual chỉ giữ phần bổ sung trực giao.
   - *(Lưu ý kỹ thuật: dùng `batch_size >= 64` như mặc định của Flora để ước lượng tương quan chéo ổn định.)*

### 2.4. Bộ chỉ số đánh giá (Dual Evaluation & Verification)
Để chứng minh bài 2 thành công ở cả **khả năng suy ra concept** lẫn **độ chính xác nhận diện lớp**, cần log đồng thời 4 nhóm chỉ số:
1. **Unseen Concept Inference (đầu ra chính mới):**
   - **Unseen Concept Macro-AUC** và **mAP / F1** của $\hat{\mathbf{p}}(\mu_{\text{ske}, c})$ trên các mẫu test *unseen* (chỉ tính trên các primitive có xuất hiện ở tập unseen).
   - Mốc đối chiếu: MLP probe độc lập từ `nesy0_probe.py` (`~0.86–0.88` AUC). Mục tiêu là nhánh $\mathbf{z}_c$ đạt tương đương hoặc cao hơn probe độc lập, đồng thời hòa hợp với không gian phân loại ZSL.
2. **Disentanglement / Leakage Check (chứng minh tách không gian thật sự):**
   - Sau khi train xong Phase 1, đóng băng encoder và train một linear probe kiểm tra trên $\mu_{\text{ske}, r}$ để dự đoán primitive: kỳ vọng **AUC trên $\mathbf{z}_c$ cao** còn **AUC trên $\mathbf{z}_r$ giảm mạnh** so với khi không có $\mathcal{L}_{\text{orth}}$.
3. **ZSL & GZSL Classification (đầu ra lớp):**
   - Top-1 ZSL Acc và GZSL ($\mathcal{S}, \mathcal{U}, \mathcal{H}$) trên NTU-60 (`55/5`, `48/12`), NTU-120 (`110/10`, `96/24`), Low-shot (`1%`, `10%`), và Random splits (bao gồm PKU-MMD I).
   - **Thử nghiệm cộng hưởng (Bonus):** So sánh khi chạy Hướng 1 trên nền text thuần của Flora vs. khi ghép chung với ngữ nghĩa KG của PRIZM!
4. **Qualitative Concept Explanation (Trực quan hóa giải thích):**
   - Với các mẫu xương unseen, xuất Top-5 concept nhỏ được dự đoán từ $\mathbf{z}_c$ kèm xác suất, so sánh với tập `require`/`forbid` của lớp đúng và lớp nhầm lẫn.

### 2.5. Lộ trình triển khai gọn & Ablation tối thiểu
1. **Tạo thư mục bài mới** (ví dụ `zeroshot/concept_map/` hoặc tên bài bạn chọn) với cấu trúc gọn:
   - `model_cr.py`: kế thừa `MappingNet`, cài đặt `ConceptResidualMappingNet`.
   - `trainer_cr.py`: kế thừa `Trainer`, nạp ma trận concept `t_y`, mask `m_seen`, cache CLIP `E_P`, thêm log `Unseen Concept AUC` song song với `ZSL/GZSL Acc`.
   - `run_cr.py`: CLI chạy nhanh trên `48/12` và `110/10` trước.
2. **Thử nghiệm kiểm chứng nhanh (Sanity check trên STAR `48/12` và `110/10`):**
   - Mốc 0: Flora gốc (`lambda_c = 0, lambda_orth = 0`).
   - Mốc 1: Chỉ nhánh Concept (`D_r = 0`, Concept Bottleneck thuần) → xác nhận có suy ra concept tốt nhưng tụt ZSL Acc.
   - Mốc 2: Concept + Residual (`lambda_c > 0, lambda_orth = 0`).
   - Mốc 3: Concept + Residual + Orthogonality (`lambda_c > 0, lambda_orth > 0`) → so sánh cả ZSL Acc, Unseen Concept AUC và Residual Leakage AUC.
   - Mốc 4: So sánh `Linear Concept Head` vs. `CLIP-Anchored Concept Head`.
   - Quét nhẹ siêu tham số: tỷ lệ tách chiều $(D_c, D_r) \in \{(256, 512), (384, 384)\}$, $\lambda_c \in \{0.05, 0.1, 0.2\}$, $\lambda_{\text{orth}} \in \{0.05, 0.1, 0.5\}$.
