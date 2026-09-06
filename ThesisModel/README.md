# Unified Neurosymbolic Framework for Skeleton-Based Action Recognition (`ThesisModel`)

Bộ khung Neurosymbolic hoàn chỉnh kết hợp ưu điểm của **First-Order Logic vi phân (Differentiable FOL Reasoning)**, **Semantic Concept Grounding với LoRA CLIP**, **Knowledge Graph Ontological Priors ($S_{KG}, C_{KG}, A_{KG}$)** và **Spatio-Temporal Backbones (SkateFormer Transformer & GCNs)**.

---

## 🌟 Kiến trúc Tổng quan (Architecture Breakdown)

Hệ thống được thiết kế theo cấu trúc 3 nhánh song song với cơ chế hợp nhất thặng dư (Residual Neural-Symbolic Fusion):

```
                       Skeleton Coordinates (B, C, T, V, M)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
        [SkateFormer / CTR-GCN]                       [9 Concept Groups]
        Spatial-Temporal Features                       (Head, Hand, Arm,
               z ∈ ℝ^(B × D)                            Hip, Leg, Foot, ...)
                │           │                                 │
   ┌────────────┴───┐       └──────────────┬──────────────────┘
   ▼                ▼                      ▼
[Branch 1]      [Branch 2]             [Branch 3]
Direct Neural   9 Concept Heads        Differentiable Logic Engine
Classifier      + LoRA CLIP Grounding  (Soft Temperature Binarizer
y_neural        + Knowledge Graph      → L1 Conjunctions (∧)
                (C_KG, S_KG)           → L2 Disjunctions (∨) / Classifier)
                p ∈ [0, 1]^(B × N_c)   y_logic
   │                                       │
   └──────────────────────┬────────────────┘
                          ▼
             Residual Fusion: y_final = (1 - α) · y_neural + α · y_logic
                          ▼
            Action Class Logits + Symbolic Rule Explanation
```

### Các Cải tiến Cốt lõi:
1. **Loại bỏ Thắt cổ chai & Thành phần Rác**:
   - Thay thế Hard STE Binarizer bằng **Soft Temperature-Annealed Binarizer** ($\sigma(k \cdot x)$) giúp lan truyền gradient mượt mà từ Epoch 1.
   - Loại bỏ MLDecoder cồng kềnh, phân cụm trực tiếp 9 nhóm giải phẫu và động lực học.
   - Loại bỏ DivergenceLoss gây xung đột gradient; thay thế bằng ma trận tiên nghiệm Knowledge Graph $C_{KG}$ (concept co-occurrence) và $S_{KG}$ (action semantic similarity).
2. **Universal Multi-Backbone Adapter**: Hỗ trợ chuyển đổi tức thì giữa `SkateFormer` (Spatio-Temporal Transformer), `CTR-GCN`, `ST-GCN`, `HyperGCN`, `SK-GCN`.
3. **Symbolic Explainability**: Trích xuất trực tiếp công thức logic toán học $\text{Action} \iff (C_1 \land C_2) \lor (C_3 \land \neg C_4)$ cho từng mẫu dự đoán.

---

## 📁 Cấu trúc Thư mục

```
d:\Thesis\ThesisModel\
├── concepts\
│   ├── ntu_rgbd_60_120.csv           # Nhãn 74 concepts chuẩn hóa cho 60/120 classes
│   └── concept_utils.py              # Parsing, 9-group mapping, CLIP prompt builder
├── feeders\
│   ├── feeder_ntu.py                 # Multi-modal Dataloader (Skeleton + Concepts + Prompts)
│   └── tools.py                      # Augmentations (Crop/Resize, Bone, Velocity, Rot)
├── knowledge_graph.py                # S_KG, C_KG, A_KG ontological knowledge matrices
├── model\
│   ├── backbones\                    # Universal Backbone Adapters
│   │   ├── base_adapter.py           # Abstract Base Adapter
│   │   ├── skateformer_adapter.py    # Transformer Backbone Adapter
│   │   ├── ctrgcn_adapter.py         # CTR-GCN Adapter
│   │   ├── stgcn_adapter.py          # ST-GCN Adapter
│   │   ├── hypergcn_adapter.py       # HyperGCN Adapter
│   │   ├── graph_utils.py            # Graph topology & adjacency
│   │   └── __init__.py               # build_backbone factory
│   ├── SkateFormer.py                # SkateFormer Model with Type I-IV partitions
│   ├── crl\                          # Differentiable Logic Engine
│   │   ├── components.py             # Soft Binarizer, L1 Conjunction, L2 Disjunction, Rules
│   │   └── __init__.py
│   ├── loralib\                      # LoRA layers & utilities
│   ├── clip\                         # CLIP text encoder & tokenizer
│   ├── clip_alignment.py             # MultiPositiveClipLoss & LoRA Text Encoder
│   ├── thesis_model.py               # Core Unified Neurosymbolic Model
│   └── __init__.py
├── config\
│   ├── ntu60_skateformer_logic.yaml  # Config huấn luyện SkateFormer Backbone
│   └── ntu60_ctrgcn_logic.yaml       # Config huấn luyện CTR-GCN Backbone
├── train.py                          # Unified End-to-End Trainer
├── eval.py                           # Benchmark Evaluation & Metric Reporting
├── explain.py                        # Trích xuất luật logic tượng trưng & giải thích
├── test_model.py                     # Unit & Smoke Test Suite
└── README.md
```

---

## 🚀 Hướng dẫn Chạy trên Google Colab / Kaggle GPU

### 1. Cài đặt Môi trường
```bash
pip install timm ftfy regex pyyaml pandas tqdm
```

### 2. Huấn luyện Mô hình (End-to-End Training)

#### A. Với Backbone SkateFormer Transformer (Độ chính xác cao nhất):
```bash
python train.py --config config/ntu60_skateformer_logic.yaml --work-dir ./work_dir/skateformer_run
```

#### B. Với Backbone CTR-GCN (Huấn luyện nhanh, bộ nhớ thấp):
```bash
python train.py --config config/ntu60_ctrgcn_logic.yaml --work-dir ./work_dir/ctrgcn_run
```

### 3. Đánh giá Benchmark & Báo cáo Chỉ số
```bash
python eval.py --config config/ntu60_skateformer_logic.yaml --checkpoint ./work_dir/skateformer_run/best_model.pt
```
*Kết quả xuất ra bao gồm: Top-1/Top-5 Acc (Final, Neural, Logic), Concept Micro-F1, Precision, Recall, Exact Match Ratio.*

### 4. Trích xuất Giải thích Logic (Symbolic Rule Explainability)
```bash
python explain.py --config config/ntu60_skateformer_logic.yaml --checkpoint ./work_dir/skateformer_run/best_model.pt --num-samples 10
```

---

## 🧪 Tự kiểm tra Mã nguồn (Self-Test)
Để kiểm tra tính toàn vẹn của mô hình:
```bash
python test_model.py
```
