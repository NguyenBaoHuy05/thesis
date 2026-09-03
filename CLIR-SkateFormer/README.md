# CLIR-SkateFormer

> **Concept Logic Iterative Refinement for Skeleton-Based Action Recognition**

CLIR-SkateFormer kết hợp backbone nhận thức không-thời gian **SkateFormer** với động cơ suy luận logic tượng trưng lặp (**Iterative Logic Reasoner - CRL**) theo cấu trúc DNF (Disjunctive Normal Form) phân cấp.

---

## 🌟 Đột phá kiến trúc

1. **Mã hóa tri thức Action-Concept trực tiếp bằng luật DNF:**
   - **L1 (Atomic AND Rules):** Part-guided grouping theo 7 nhóm bộ phận cơ thể (Head, Hand, Arm, Hip, Leg, Foot, Temporal), 144 literals → 256 conjunction rules.
   - **L2a (AND2 - Compound Coordination):** Hội của các AND rules (128 rules) mô tả các hành động đòi hỏi phối hợp đa bộ phận đồng thời.
   - **L2b (OR1 - Alternative Patterns):** Tuyển xác suất của các AND rules (128 rules) mô tả các phương thức thực hiện hành động thay thế.
   - **ActionHead (Strict Logic):** Tuyến tính trực tiếp từ $[r_{\text{and2}}; r_{\text{or1}}]$ (256 dims) $\rightarrow$ 120 classes, **không có feature bypass** từ backbone để đảm bảo tính giải thích tuyệt đối.

2. **Loại bỏ Concept-Concept KG gượng ép:**
   - Không dùng ma trận quan hệ tĩnh $72 \times 72$ giữa các concept (loại bỏ spurious correlations do dữ liệu và mâu thuẫn thời gian trong video).

3. **Logic-Guided Concept Refinement (Top-Down Feedback):**
   - Vòng lặp tinh chỉnh ($K=2$ loops):
     $$z^{t+1} = z^t + \beta^t \cdot \Delta z^t$$
   - $\Delta z^t$ được sinh ra từ **Logic Evidence** $r^t = [r_{\text{and}}; r_{\text{and2}}; r_{\text{or1}}] \in \mathbb{R}^{512}$ phản ánh đánh giá logic mức cao xuống không gian concept logits liên tục.

4. **Huấn luyện 2 Phase (Staged Training):**
   - **Phase 1 (Epoch 1–30):** Warm-up với chế độ `soft` (Sigmoid liên tục, $\beta_{\max}=0.1, \lambda_{\text{reg}}=0$) giúp backbone và concept heads hội tụ vững.
   - **Phase 2 (Epoch 31+):** STE (Straight-Through Estimator, $\beta_{\max}=0.5$) với lịch annealing temperature $5.0 \rightarrow 10.0$ và $\lambda_{\text{reg}}=0.005$ để ép luật thưa và trích xuất mệnh đề DNF sắc bén.

---

## 📂 Cấu trúc thư mục

```
CLIR-SkateFormer/
├── model/
│   ├── SkateFormer.py          # Perception backbone (partition & reverse attention)
│   ├── __init__.py             # Exports kiến trúc CLIR
│   ├── clir_skateformer.py     # Class chính CLIRSkateFormer, IterativeLogicReasoner, ConceptUpdateCell
│   └── crl/                    # Tầng suy luận logic độc lập
│       ├── __init__.py
│       ├── literal_layer.py    # LiteralLayer & BinarizeSTE (72 -> 144 literals)
│       ├── conjunction_layer.py# L1 AND (part-grouped) + L2a AND2
│       ├── disjunction_layer.py# L2b OR1 (probabilistic OR)
│       ├── crl_reasoner.py     # SharedCRL pipeline (1-pass)
│       └── rule_extractor.py   # Trích xuất luật DNF & tính Fidelity/Coverage
├── concepts/
│   └── ntu_rgbd_60_120.csv     # 120 actions × 72 concepts ontology
├── config/
│   └── train/
│       ├── ntu120_cset/CLIR_SkateFormer_j.yaml
│       ├── ntu120_csub/CLIR_SkateFormer_j.yaml
│       ├── ntu60_cs/CLIR_SkateFormer_j.yaml
│       └── ntu60_cv/CLIR_SkateFormer_j.yaml
├── feeders/                    # Feeder dataset NTU RGB+D
├── torchlight/                 # IO and Argument parsers
├── scripts/
│   └── extract_rules.py        # CLI trích xuất luật & kiểm tra fidelity
├── main.py                     # Huấn luyện và đánh giá với 2-phase auto switch
└── verify_model.py             # Test suite chẩn đoán và kiểm tra tính toàn vẹn
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Kiểm tra tính toàn vẹn (Verification Suite)

Chạy test suite chẩn đoán toàn bộ 7 bài test (tất cả các module, forward/backward pass, gradient flow và trích xuất luật):

```bash
python verify_model.py
```

### 2. Huấn luyện mô hình (Training)

Chạy huấn luyện trên benchmark NTU RGB+D 120 Cross-Setup:

```bash
python main.py --config config/train/ntu120_cset/CLIR_SkateFormer_j.yaml
```

*Quá trình huấn luyện sẽ tự động kích hoạt Phase 1 ở Epoch 1–30 và chuyển sang Phase 2 (bật STE, cập nhật per-group learning rate, kích hoạt $\mathcal{L}_{\text{reg}}$) từ Epoch 31.*

### 3. Trích xuất luật logic DNF (Rule Extraction)

Sau khi huấn luyện xong, chạy script trích xuất để sinh ra file luật DNF dạng JSON và văn bản dễ đọc:

```bash
python scripts/extract_rules.py \
    --checkpoint work_dir/ntu120_cset/CLIR_SkateFormer_j/best_model.pt \
    --config config/train/ntu120_cset/CLIR_SkateFormer_j.yaml \
    --concept-csv concepts/ntu_rgbd_60_120.csv \
    --output-dir work_dir/rules \
    --tau-and 0.5 --tau-and2 0.5 --tau-or 0.5 \
    --eval-fidelity
```
