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

4. **Huấn luyện 100% Pure STE (Straight-Through Estimator) với Continuous Temperature Annealing:**
   - **Forward Pass:** Ngay từ Epoch 1, toàn bộ literals $l^t$ được binarize cứng về $\{0, 1\}$ qua `BinarizeSTE`, đảm bảo tính trung thực tuyệt đối giữa mô hình mạng nơ-ron và động cơ luật tượng trưng.
   - **Continuous Temperature Annealing ($2.0 \rightarrow 10.0$):**
     - Ở các epoch đầu ($T = 2.0$): Vùng gradient hoạt động rộng $\big[-\frac{3}{T}, \frac{3}{T}\big] = [-1.5, 1.5]$ chống bão hòa sớm, giúp gradient truyền mượt về backbone SkateFormer mà không bị triệt tiêu (giải quyết triệt để cold-start problem khi bỏ warm-up).
     - Ở các epoch cuối ($T \rightarrow 10.0$): Hàm surrogate tiệm cận bước nhảy Heaviside lý tưởng, triệt tiêu chênh lệch giữa training và exact DNF rule evaluation.
   - **Tối ưu hóa liên tục 1 chu kỳ:** Áp dụng per-group learning rate ngay từ Epoch 0 (backbone scale 0.1, CRL scale 1.0, UpdateCell scale 2.0) với Cosine LR Scheduler liên tục, không bị gián đoạn hay reset giữa chừng.
   - **Tương thích ngược 2-Phase:** Hỗ trợ cơ chế 2-phase cũ (Phase 1 Soft $\rightarrow$ Phase 2 STE) khi cấu hình `phase1_epochs > 0`.

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

*Mặc định mô hình chạy ở chế độ 100% Pure STE ngay từ Epoch 1 (`phase1_epochs: 0`), tự động anneal temperature từ 2.0 đến 10.0, áp dụng per-group learning rate và regularizer $\mathcal{L}_{\text{reg}}$ liên tục suốt toàn bộ số epoch.*

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
