# Implementation Plan - Unified Neurosymbolic Framework (`ThesisModel`)

Xây dựng bộ khung hoàn chỉnh **`ThesisModel`** tại [d:\Thesis\ThesisModel](file:///d:/Thesis/ThesisModel) kết hợp tinh hoa của **REASON** (First-Order Logic vi phân, LoRA CLIP Grounding, Rule Extraction) và **SkateFormer** (Spatio-Temporal Transformer, 9 Nhóm Concept giải phẫu, Knowledge Graph Regularization, Neural-Symbolic Residual Fusion), đồng thời loại bỏ các thành phần rác/thắt cổ chai (MLDecoder, ConceptRefiner, DivergenceLoss, Hard STE Binarizer).

---

## User Review Required

> [!IMPORTANT]
> Toàn bộ mã nguồn mới sẽ được tổ chức độc lập, module hóa chuẩn tại thư mục [d:\Thesis\ThesisModel](file:///d:/Thesis/ThesisModel), không làm ảnh hưởng hay sửa đổi trực tiếp vào code cũ của [REASON](file:///d:/Thesis/REASON) và [New folder\SkateFormer](file:///d:/Thesis/New%20folder/SkateFormer).

---

## Proposed Changes

### Component Architecture Breakdown in `d:\Thesis\ThesisModel`

```
d:\Thesis\ThesisModel\
├── concepts\
│   ├── ntu_rgbd_60_120.csv           # Concept annotations (78/100 concepts)
│   └── concept_utils.py              # Parsing, group mapping, and prompt builders
├── feeders\
│   ├── feeder_ntu.py                 # Multi-modal Dataloader (Skeleton + Concepts + Prompts)
│   └── tools.py                      # Data augmentation and preprocessing
├── knowledge_graph.py                # S_KG, C_KG, A_KG ontological knowledge matrices
├── model\
│   ├── backbones\                    # Universal Multi-Backbone Adapter System
│   │   ├── base_adapter.py           # Base abstract class for skeleton backbones
│   │   ├── skateformer_adapter.py    # Transformer backbone adapter
│   │   ├── ctrgcn_adapter.py         # CTR-GCN backbone adapter
│   │   ├── hypergcn_adapter.py       # HyperGCN backbone adapter
│   │   ├── stgcn_adapter.py          # ST-GCN backbone adapter
│   │   ├── graph_utils.py            # Graph adjacency & topology helpers
│   │   └── __init__.py               # build_backbone factory
│   ├── SkateFormer.py                # SkateFormer Spatio-Temporal Transformer
│   ├── crl\                          # Differentiable First-Order Logic Engine
│   │   ├── components.py             # Soft-Binarizer, L1 Conjunction, L2 Disjunction, Rule Extraction
│   │   └── __init__.py
│   ├── loralib\                      # Low-Rank Adaptation utilities for text encoder
│   ├── clip_alignment.py             # LoRA TextCLIP & MultiPositiveClipLoss
│   ├── thesis_model.py               # Core Unified Neurosymbolic Model
│   └── __init__.py
├── config\
│   ├── ntu60_skateformer_logic.yaml  # Config with SkateFormer backbone
│   └── ntu60_ctrgcn_logic.yaml       # Config with CTR-GCN backbone
├── train.py                          # Unified End-to-End Trainer with Temperature Annealing
├── eval.py                           # Benchmark Evaluation & Metric Reporting
├── explain.py                        # Explicit Logic Rule Extraction & Case Studies
└── README.md                         # Architecture documentation & execution guide
```

---

### Step-by-Step Implementation

#### 1. Foundation & Concepts ([NEW])
* **[NEW] `concepts/ntu_rgbd_60_120.csv`**: Copy và chuẩn hóa file nhãn concept từ dataset NTU.
* **[NEW] `concepts/concept_utils.py`**: Định nghĩa `GROUP_SPECS` (9 nhóm giải phẫu và động lực học), hàm phân cụm `build_concept_groups()`, và sinh prompt văn bản cho CLIP alignment.
* **[NEW] `knowledge_graph.py`**: Tích hợp các ma trận tiên nghiệm $S_{KG}$ (Action semantic similarity), $C_{KG}$ (Concept co-occurrence correlation), và $A_{KG}$ (Joint anatomical functional connectivity).

#### 2. Data Pipeline ([NEW])
* **[NEW] `feeders/feeder_ntu.py` & `feeders/tools.py`**: Bộ nạp dữ liệu thống nhất hỗ trợ đồng thời:
  * Skeleton coordinates & bone/joint modalities $(N, C, T, V, M)$.
  * Multi-label ground truth vector cho 9 nhóm concept.
  * Batch text prompts cho nhánh CLIP grounding.

#### 3. Backbone Adapters & SkateFormer ([NEW])
* **[NEW] `model/backbones/base_adapter.py`**: Chuẩn hóa interface trích xuất đặc trưng `extract_feat()`, `get_feat_dim()`.
* **[NEW] `model/SkateFormer.py` & `model/backbones/skateformer_adapter.py`**: Backbone Transformer mạnh mẽ với partition type I-IV.
* **[NEW] `model/backbones/ctrgcn_adapter.py`, `hypergcn_adapter.py`, `stgcn_adapter.py`**: Hỗ trợ linh hoạt chuyển đổi giữa các kiến trúc GCN.

#### 4. Logic Reasoning & Semantic Grounding ([NEW])
* **[NEW] `model/crl/components.py`**:
  * Thay thế Hard STE Binarizer bằng **Soft Temperature-Annealed Binarizer** ($k$-scaling theo epoch).
  * Tầng $L_1$ (Conjunctions $\land$) học mệnh đề cục bộ theo 9 nhóm giải phẫu.
  * Tầng $L_2$ (Disjunctions $\lor$) tổng hợp mệnh đề cấp hành động.
  * Module trích xuất luật tự động `extract_rules()` và cắt tỉa `prune_rules()`.
* **[NEW] `model/loralib/` & `model/clip_alignment.py`**: Module LoRA TextCLIP và `MultiPositiveClipLoss` để neo đặc trưng skeleton vào không gian ngôn ngữ.

#### 5. Unified Model Architecture ([NEW])
* **[NEW] `model/thesis_model.py` (`ThesisNeurosymbolicModel`)**:
  * Nhánh 1: Direct Neural Classifier ($y_{\text{neural}}$).
  * Nhánh 2: 9-Group Concept Heads $\to$ MultiPositiveClipLoss + $C_{KG}, S_{KG}$ Regularization.
  * Nhánh 3: Hierarchical Logic Layers $\to$ ($y_{\text{logic}}$).
  * Fusion: $y_{\text{final}} = (1-\alpha) \cdot y_{\text{neural}} + \alpha \cdot y_{\text{logic}}$.
  * Trả về đồng thời Loss, Accuracy, Concept F1, và giải thích suy luận `explain()`.

#### 6. Training, Evaluation & Explanation Scripts ([NEW])
* **[NEW] `train.py`**: Huấn luyện End-to-End từ Epoch 1 (không cần hard switch ở epoch 50), hỗ trợ learning rate scheduling và temperature annealing.
* **[NEW] `eval.py`**: Đánh giá toàn diện Top-1/Top-5 Acc, Concept F1, Exact Match.
* **[NEW] `explain.py`**: Xuất cây suy luận logic toán học ($A \land B \implies \text{Action}$) cho từng mẫu dữ liệu kiểm thử.
* **[NEW] `config/*.yaml`**: Cấu hình mẫu cho các thực nghiệm.

---

## Verification Plan

### Automated Tests
1. **Model Forward/Backward Smoke Test**:
   * Viết script test nhanh `test_model.py` để verify forward pass trên dummy batch `(B=4, C=3, T=64, V=25, M=2)`:
     * Kiểm tra output shapes của `y_final`, `y_neural`, `y_logic`, `concept_probs`.
     * Kiểm tra tính hợp lệ của tất cả các thành phần loss (`action_loss`, `concept_bce`, `clip_loss`, `kg_loss`, `logic_reg_loss`).
     * Kiểm tra backward pass và cập nhật gradient trơn tru.
2. **Logic Rule Extraction Verification**:
   * Chạy hàm `explain()` trên dummy input và kiểm tra xem text output có sinh ra công thức logic chuẩn $\land, \lor, \neg$ hay không.
3. **Multi-Backbone Switch Test**:
   * Kiểm tra xem model có thể khởi tạo và forward mượt mà với cả 2 backbone `skateformer` và `ctrgcn`.
