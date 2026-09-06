"""Generator for CLIR_SkateFormer_Colab.ipynb.

Creates an all-in-one Google Colab & Local Jupyter Notebook for CLIR-SkateFormer:
- Default Benchmark: NTU RGB+D 60 Cross-Subject (60CSub / NTU60_CS, 60 classes, 72 concepts)
- 100% Self-contained: All models, feeders, tools, loss functions, rule extractor, and trainers embedded directly
- Zero external local module import requirements
- Concept ontology auto-creation (72 concepts)
- Quick dummy smoke test (60 classes)
- Complete end-to-end dataset loading, 2-phase staged training loop, evaluation, and checkpoint saving
- Interactive post-hoc DNF rule extraction, visualization, and Fidelity/Precision/Coverage evaluation
- Alternative CLI commands for NTU60 Cross-Subject
"""

import json
from pathlib import Path


def build_notebook():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")],
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")],
        })

    source_root = Path(__file__).resolve().parent

    tools_source = (source_root / "feeders" / "tools.py").read_text(encoding="utf-8")
    feeder_ntu_source = (source_root / "feeders" / "feeder_ntu.py").read_text(encoding="utf-8")
    skateformer_source = (source_root / "model" / "SkateFormer.py").read_text(encoding="utf-8")
    literal_source = (source_root / "model" / "crl" / "literal_layer.py").read_text(encoding="utf-8")
    conj_source = (source_root / "model" / "crl" / "conjunction_layer.py").read_text(encoding="utf-8")
    disj_source = (source_root / "model" / "crl" / "disjunction_layer.py").read_text(encoding="utf-8")
    crl_source = (source_root / "model" / "crl" / "crl_reasoner.py").read_text(encoding="utf-8")
    rule_ext_source = (source_root / "model" / "crl" / "rule_extractor.py").read_text(encoding="utf-8")
    clir_source = (source_root / "model" / "clir_skateformer.py").read_text(encoding="utf-8")
    concepts_csv = (source_root / "concepts" / "ntu_rgbd_60_120.csv").read_text(encoding="utf-8")

    # Clean imports for notebook scope
    import re
    feeder_import_replacement = """import sys, types
if '.' not in sys.path:
    sys.path.insert(0, '.')
try:
    from feeders import tools
except Exception:
    if 'tools' not in globals() or tools is None:
        tools = sys.modules.get('tools', sys.modules.get('feeders.tools', None))
"""
    feeder_clean = feeder_ntu_source.replace("from feeders import tools", feeder_import_replacement)
    getitem_guard = """    def __getitem__(self, index):
        global tools
        if 'tools' not in globals() or tools is None:
            import sys
            try:
                from feeders import tools
            except Exception:
                tools = sys.modules.get('tools', sys.modules.get('feeders.tools', None))
"""
    feeder_clean = feeder_clean.replace("    def __getitem__(self, index):", getitem_guard)
    crl_clean = re.sub(r"^from \.[^\n]+", "# relative import in notebook scope", crl_source, flags=re.MULTILINE)
    rule_ext_clean = re.sub(r"^from \.[^\n]+", "# relative import in notebook scope", rule_ext_source, flags=re.MULTILINE)
    clir_clean = re.sub(r"^from \.[^\n]+", "# relative import in notebook scope", clir_source, flags=re.MULTILINE)

    # Header
    add_md(r"""# 🚀 CLIR-SkateFormer: All-in-One Google Colab & Local Notebook (NTU60 Cross-Subject / 60CSub)
> **Mô hình**: **CLIR-SkateFormer (Concept Logic Iterative Refinement for Skeleton Action Recognition)**  
> **Benchmark mặc định**: **NTU RGB+D 60 Cross-Subject (60CSub / NTU60_CS)**  
> **Số lớp hành động**: **60 classes** (A1–A60)  
> **Số khái niệm**: **72 concepts** (52 spatial + 20 temporal)  
> **Điểm đột phá**:
> 1. Mã hóa tri thức Action-Concept trực tiếp bằng luật DNF phân cấp (L1 part-grouped AND → L2a AND2 ∥ L2b OR1 → Strict ActionHead).
> 2. Loại bỏ hoàn toàn ma trận tương quan Concept-Concept tĩnh $72 \times 72$.
> 3. Tinh chỉnh khái niệm vòng lặp ($K=2$) dẫn dắt bởi **Logic Evidence** mức cao ($r^t \in \mathbb{R}^{512}$).
> 4. Huấn luyện 100% Pure STE (Straight-Through Estimator) với Continuous Temperature Annealing ($T_{\text{start}}=2.0 \rightarrow T_{\text{end}}=10.0$).

---

### 📌 Cấu trúc Toàn diện Notebook:
1. **Block 1**: Môi trường, Dependencies & Mount Google Drive
2. **Block 2**: Concept Ontology CSV (60/120 actions $\times$ 72 concepts) & NTU Feeder
3. **Block 3**: Perception Backbone SkateFormer
4. **Block 4**: Tầng Logic CRL (LiteralLayer, ConjunctionLayer, DisjunctionLayer, CRLReasoner)
5. **Block 5**: ConceptUpdateCell, IterativeLogicReasoner & CLIRSkateFormer
6. **Block 6**: RuleExtractor (Post-hoc DNF Rule Extraction & Fidelity Engine)
7. **Block 7**: Kiểm thử nhanh tính toàn vẹn (Quick Smoke Test cho 60CSub)
8. **Block 8**: Thiết lập Cấu hình & DataLoaders (Mặc định NTU60 Cross-Subject)
9. **Block 9**: Vòng lặp Huấn luyện 100% Pure STE (hoặc 2-Phase fallback) & Lưu Checkpoint
10. **Block 10**: Trích xuất & Trực quan hóa Luật DNF (Precision, Coverage, Fidelity)
11. **Block 11**: Huấn luyện qua dòng lệnh CLI (Alternative CLI Execution cho NTU60 CS)
""")

    # Block 1
    add_md("## 📦 Block 1: Cài đặt Dependencies & Môi trường")
    add_code("""# Cài đặt thư viện cần thiết
!pip install timm tensorboardX pyyaml tqdm -q

import os
import sys
import time
import json
import torch
import numpy as np

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Mount Google Drive nếu chạy trên Google Colab
if 'google.colab' in str(get_ipython()):
    from google.colab import drive
    drive.mount('/content/drive')
""")

    # Block 2
    add_md("## 📑 Block 2: Concept Ontology & Data Feeder")
    add_code(f"""# Lưu concept ontology CSV vào workspace (72 concepts chuẩn hóa)
os.makedirs("concepts", exist_ok=True)
with open("concepts/ntu_rgbd_60_120.csv", "w", encoding="utf-8") as f:
    f.write('''{concepts_csv}''')
print("Concept CSV written: concepts/ntu_rgbd_60_120.csv (72 concepts)")
""")
    add_code(f"""# Data Augmentation Tools
import os
import sys
import types

# 1. Lưu feeders/tools.py vào workspace để DataLoader workers có thể import feeders.tools bình thường
os.makedirs("feeders", exist_ok=True)
with open("feeders/__init__.py", "w", encoding="utf-8") as f:
    f.write("")
with open("feeders/tools.py", "w", encoding="utf-8") as f:
    f.write({repr(tools_source)})

if "." not in sys.path:
    sys.path.insert(0, ".")

# 2. Định nghĩa hàm augmentation trong notebook scope
{tools_source}

# 3. Đăng ký module 'tools' vào sys.modules và globals để cả gọi trực tiếp lẫn qua tools.<func> đều hoạt động 100%
tools = types.ModuleType("tools")
for _k, _v in list(globals().items()):
    if not _k.startswith("_"):
        setattr(tools, _k, _v)
sys.modules["tools"] = tools
sys.modules["feeders.tools"] = tools

feeders_mod = types.ModuleType("feeders")
setattr(feeders_mod, "tools", tools)
sys.modules["feeders"] = feeders_mod
print("Data augmentation tools ready and feeders/tools.py written to disk!")
""")
    add_code(f"""# NTU RGB+D Feeder
{feeder_clean}
""")

    # Block 3
    add_md("## 🧠 Block 3: SkateFormer Perception Backbone")
    add_code(f"""# SkateFormer Backbone Architecture
{skateformer_source}
""")

    # Block 4
    add_md("## ⚡ Block 4: Tầng Logic CRL")
    add_code(f"""# 1. LiteralLayer & BinarizeSTE
{literal_source}
""")
    add_code(f"""# 2. ConjunctionLayer (L1 Part-Grouped & L2a AND2)
{conj_source}
""")
    add_code(f"""# 3. DisjunctionLayer (L2b OR1)
{disj_source}
""")
    add_code(f"""# 4. CRLReasoner (1-Pass Shared Reasoner)
{crl_clean}
""")

    # Block 5
    add_md("## 🔄 Block 5: ConceptUpdateCell, IterativeLogicReasoner & CLIRSkateFormer")
    add_code(f"""# CLIR Main Components
{clir_clean}
""")

    # Block 6
    add_md("## 📜 Block 6: RuleExtractor (DNF Rules & Fidelity Evaluation)")
    add_code(f"""# RuleExtractor
{rule_ext_clean}
""")

    # Block 7
    add_md("## 🧪 Block 7: Kiểm thử nhanh tính toàn vẹn (Quick Smoke Test cho 60CSub)")
    add_code("""# Quick Model Verification Test cho NTU60 Cross-Subject (60 classes)
model_smoke = CLIR_SkateFormer(
    num_classes=60,
    num_concepts=72,
    spatial_concepts=52,
    temporal_concepts=20,
    num_loops=2,
    lambda_sup=1.0,
    lambda_reg=0.005,
)
print("CLIR-SkateFormer (60CSub) initialized successfully!")
print(f"Total Parameters: {sum(p.numel() for p in model_smoke.parameters() if p.requires_grad):,}")

# Test Dummy Forward với 60 action classes
x_dummy = torch.randn(2, 3, 64, 24, 2)
c_targets = torch.randint(0, 2, (2, 72)).float()
a_labels = torch.randint(0, 60, (2,)).long()

# Pure STE Default Check (Epoch 1 setup)
out_default = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
print(f"Default Pure STE Loss: {out_default['aux_loss'].item():.4f}, l_reg: {out_default['l_reg'].item():.4f}, beta={model_smoke.reasoner.get_betas().detach().numpy()}")
assert out_default["action_logits"].shape == (2, 60), "Action logits must have shape [2, 60] for 60CSub"

# Phase 1 Fallback Check
model_smoke.set_phase(1)
out1 = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
print(f"Phase 1 Fallback Loss: {out1['aux_loss'].item():.4f}, beta={model_smoke.reasoner.get_betas().detach().numpy()}")
assert out1["action_logits"].shape == (2, 60), "Action logits must have shape [2, 60] for 60CSub"

# Phase 2 Check
model_smoke.set_phase(2, temperature=6.0)
out2 = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
print(f"Phase 2 Loss: {out2['aux_loss'].item():.4f}, l_reg: {out2['l_reg'].item():.4f}, beta={model_smoke.reasoner.get_betas().detach().numpy()}")
assert out2["action_logits"].shape == (2, 60), "Action logits must have shape [2, 60] for 60CSub"
print("Smoke Test for 60CSub Passed!")
""")

    # Block 8
    add_md(r"""## ⚙️ Block 8: Cấu hình Huấn luyện & Chuẩn bị DataLoaders (NTU60 Cross-Subject)
Mặc định được thiết lập cho **NTU60 Cross-Subject (60CSub)**.

> 💡 **TÙY CHỌN CHẾ ĐỘ HUẤN LUYỆN ĐỂ SO SÁNH**:
> Dưới đây có 2 block cấu hình sẵn sàng. Bạn chỉ cần **comment / uncomment** 1 trong 2 block để so sánh trực tiếp:
> 1. **Block 1**: 🚀 `100% Pure STE Annealing` (Đề xuất cho luận văn - Binarize cứng $\{0, 1\}$, Temperature Annealing $2.0 \rightarrow 10.0$, $\mathcal{L}_{\text{reg}}=0.005$).
> 2. **Block 2**: 🌿 `Pure Soft Baseline` (Để so sánh ablation - Sigmoid liên tục, Không dùng STE, Không binarize, $\mathcal{L}_{\text{reg}}=0.0$).
""")
    add_code("""# Cấu hình Benchmark (Mặc định NTU60 Cross-Subject / 60CSub)
BENCHMARK = "ntu60_cs"  # @param ["ntu60_cs", "ntu60_cv", "ntu120_csub", "ntu120_cset"]

DATASET_CONFIGS = {
    "ntu60_cs": {
        "num_classes": 60,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz",
        "backbone_weights": "/content/ntu60_CS/SkateFormer_j.pt",
        "work_dir": "./work_dir/ntu60_cs/CLIR_SkateFormer_j",
    },
    "ntu60_cv": {
        "num_classes": 60,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz",
        "backbone_weights": "/content/ntu60_CV/SkateFormer_j.pt",
        "work_dir": "./work_dir/ntu60_cv/CLIR_SkateFormer_j",
    },
    "ntu120_csub": {
        "num_classes": 120,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSub_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSub_64f.npz",
        "backbone_weights": "/content/ntu120_CSub/SkateFormer_j.pt",
        "work_dir": "./work_dir/ntu120_csub/CLIR_SkateFormer_j",
    },
    "ntu120_cset": {
        "num_classes": 120,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSet_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSet_64f.npz",
        "backbone_weights": "/content/ntu120_CSet/SkateFormer_j.pt",
        "work_dir": "./work_dir/ntu120_cset/CLIR_SkateFormer_j",
    },
}

selected = DATASET_CONFIGS[BENCHMARK]

# ==============================================================================
# 🎯 LỰA CHỌN CHẾ ĐỘ HUẤN LUYỆN: COMMENT / UNCOMMENT 1 TRONG 2 BLOCK DƯỚI ĐÂY
# ==============================================================================

# --- [BLOCK 1] 🚀 CHẾ ĐỘ 100% PURE STE ANNEALING (MẶC ĐỊNH - ĐỀ XUẤT CHO LUẬN VĂN) ---
# Đặc điểm: Literals nhị phân {0, 1} từ Epoch 1, T anneal 2.0 -> 10.0, L_reg = 0.005
TRAINING_MODE = {
    "mode_name": "Pure_STE_Annealing",
    "literal_mode": "ste",
    "phase1_epochs": 0,             # 0 = Bắt đầu STE ngay từ Epoch 1
    "ste_temperature_start": 2.0,   # T_start = 2.0 chống bão hòa / chết gradient
    "ste_temperature_end": 10.0,    # T_end = 10.0 tiệm cận hàm Heaviside, tối đa Rule Fidelity
    "lambda_reg": 0.005,            # Bật rule sparsity & loop stability
    "beta_max": 0.5,                # Biên độ feedback tinh chỉnh logic K=2
}

# --- [BLOCK 2] 🌿 CHẾ ĐỘ CHỈ SOFT KHÔNG (PURE SOFT BASELINE ĐỂ SO SÁNH ABLATION) ---
# (Để chạy so sánh: Uncomment block dưới và comment block 1 ở trên)
# TRAINING_MODE = {
#     "mode_name": "Pure_Soft",
#     "literal_mode": "soft",
#     "phase1_epochs": 999,           # Chạy Soft toàn bộ quá trình (không bao giờ chuyển STE)
#     "ste_temperature_start": 1.0,   # Không dùng STE
#     "ste_temperature_end": 1.0,
#     "lambda_reg": 0.0,              # Tắt rule sparsity regularization
#     "beta_max": 0.1,                # Giữ beta nhỏ ổn định cho continuous sigmoid
# }

TRAIN_CONFIG = {
    "benchmark": BENCHMARK,
    "num_classes": selected["num_classes"],
    "num_concepts": 72,
    "spatial_concepts": 52,
    "temporal_concepts": 20,
    
    # Đường dẫn dữ liệu
    "train_data": selected["train_data"],
    "test_data": selected["test_data"],
    "backbone_weights": selected["backbone_weights"],
    "work_dir": selected["work_dir"] if TRAINING_MODE.get("mode_name") != "Pure_Soft" else selected["work_dir"] + "_PureSoft",
    
    # Tham số huấn luyện chung
    "batch_size": 32,
    "test_batch_size": 32,
    "num_epoch": 70,
    "base_lr": 1e-4,
    "min_lr": 1e-6,
    "warmup_lr": 1e-6,
    "warm_up_epoch": 2,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    
    # 🎯 Two-Stage Concept-to-Logic Training (Giải pháp B):
    # Stage 1: Huấn luyện riêng Concept Head (L_c^0) với Reasoner đóng băng
    # Stage 2: Mở khóa Reasoner và ActionHead học End-to-End Logic Reasoning
    "concept_warmup_epochs": 5,   # Khuyên dùng 5 (hoặc 3). Đặt 0 nếu muốn tắt Stage 1
    
    # Học theo nhóm (áp dụng cho cả 2 mode để so sánh công bằng)
    "backbone_lr_scale": 0.1,
    "crl_lr_scale": 1.0,
    "update_cell_lr_scale": 2.0,
    "lambda_sup": 1.0,
    
    # Vòng lặp CLIR
    "num_loops": 2,
    "beta_init": 0.1,
}

# Hợp nhất cấu hình chế độ được chọn
TRAIN_CONFIG.update(TRAINING_MODE)

os.makedirs(TRAIN_CONFIG["work_dir"], exist_ok=True)
print("=" * 75)
print(f"Active Benchmark:     {BENCHMARK.upper()} ({TRAIN_CONFIG['num_classes']} action classes)")
print(f"Active Training Mode: [{TRAINING_MODE.get('mode_name')}]")
if TRAINING_MODE.get('mode_name') == 'Pure_Soft':
    print("                      🌿 PURE SOFT BASELINE (Continuous Sigmoid, No STE, lambda_reg=0.0)")
else:
    print(f"                      🚀 100% PURE STE ANNEALING (T: {TRAIN_CONFIG['ste_temperature_start']} -> {TRAIN_CONFIG['ste_temperature_end']}, lambda_reg={TRAIN_CONFIG['lambda_reg']})")
print(f"Work Directory:       {TRAIN_CONFIG['work_dir']}")
print("=" * 75)
""")

    add_code("""# Khởi tạo DataLoader nếu file dữ liệu tồn tại
from torch.utils.data import DataLoader

if os.path.exists(TRAIN_CONFIG["train_data"]):
    print(f"Loading {TRAIN_CONFIG['benchmark'].upper()} dataset from disk...")
    train_dataset = Feeder(
        data_path=TRAIN_CONFIG["train_data"],
        concepts_csv="concepts/ntu_rgbd_60_120.csv",
        split="train",
        window_size=64,
        thres=64,
        uniform=True,
        partition=True,
        remove_null_concept=True,
        p_interval=[0.5, 1],
        aug_method="a123489",
        intra_p=0.5,
        inter_p=0.2,
    )
    test_dataset = Feeder(
        data_path=TRAIN_CONFIG["test_data"],
        concepts_csv="concepts/ntu_rgbd_60_120.csv",
        split="test",
        window_size=64,
        thres=64,
        uniform=True,
        partition=True,
        remove_null_concept=True,
        p_interval=[0.95],
    )
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG["test_batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    print(f"DataLoaders ready! Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
else:
    print(f"Warning: {TRAIN_CONFIG['train_data']} not found. DataLoaders skipped.")
    train_loader, test_loader = None, None
""")

    # Block 9
    add_md(r"""## 🏋️ Block 9: Vòng lặp Huấn luyện 100% Pure STE & Lưu Checkpoint
- **Pure STE Training (Mặc định)**: Literals nhị phân $\{0, 1\}$ ở forward pass ngay từ Epoch 1, backward nhận gradient qua surrogate sigmoid.
- **Continuous Temperature Annealing**: $T$ tăng tuyến tính từ $2.0 \rightarrow 10.0$ xuyên suốt toàn bộ số epoch (vừa chống chết gradient, vừa tối đa hóa Rule Fidelity).
- **Per-group LRs**: Backbone SkateFormer (scale 0.1), CRL Modules (scale 1.0), UpdateCell (scale 2.0).
- **Scheduler**: 1 chu kỳ Cosine Annealing liên tục kèm 10 epoch warm-up.
- **Fallback 2-Phase**: Nếu cấu hình `phase1_epochs > 0`, hệ thống tự động chạy chế độ 2-Phase (Warm-up Phase 1 $\rightarrow$ STE Phase 2).
""")
    add_code("""from tqdm import tqdm
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler

class CLIRTrainer:
    def __init__(self, cfg, train_loader, test_loader):
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Khởi tạo mô hình CLIR cho số classes của benchmark
        self.model = CLIR_SkateFormer(
            num_classes=cfg["num_classes"],
            num_concepts=cfg["num_concepts"],
            spatial_concepts=cfg["spatial_concepts"],
            temporal_concepts=cfg["temporal_concepts"],
            backbone_weights=cfg["backbone_weights"] if os.path.exists(cfg["backbone_weights"]) else None,
            num_loops=cfg["num_loops"],
            beta_max=cfg["beta_max"],
            beta_init=cfg["beta_init"],
            lambda_sup=cfg["lambda_sup"],
            lambda_reg=cfg["lambda_reg"],
        ).to(self.device)
        
        self.is_pure_soft = (
            self.cfg.get("literal_mode") == "soft"
            or self.cfg.get("phase1_epochs", 0) >= self.cfg["num_epoch"]
        )

        if self.is_pure_soft:
            self.current_phase = 1
            self.model.set_phase(1)
            self.model.lambda_reg = self.cfg.get("lambda_reg", 0.0)
            if "beta_max" in self.cfg:
                self.model.reasoner.beta_max = self.cfg["beta_max"]
        elif self.cfg.get("phase1_epochs", 0) <= 0:
            self.current_phase = 2
            self.model.set_phase(2, temperature=self.cfg.get("ste_temperature_start", 2.0))
            self.model.lambda_reg = self.cfg.get("lambda_reg", 0.005)
            if "beta_max" in self.cfg:
                self.model.reasoner.beta_max = self.cfg["beta_max"]
        else:
            self.current_phase = 1
            self.model.set_phase(1)

        self.best_acc = 0.0
        self.global_step = 0
        self.phase_step = 0
        self.start_epoch = 0

        # 🎯 Quản lý Two-Stage Concept-to-Logic Training (Giải pháp B):
        # Stage 1: Huấn luyện riêng Concept Head (L_c^0) với Reasoner đóng băng
        # Stage 2: Mở khóa Reasoner và ActionHead học End-to-End Logic Reasoning
        self.current_stage = 1 if self.cfg.get("concept_warmup_epochs", 0) > 0 else 2
        self._set_stage(self.current_stage, verbose=False)

    def _set_stage(self, stage: int, verbose: bool = True):
        self.current_stage = stage
        if stage == 1:
            if verbose:
                print(f"\n{'='*70}")
                print(f"[STAGE 1 ACTIVATED] 🎯 CONCEPT WARM-UP (Epoch 1 → {self.cfg.get('concept_warmup_epochs', 0)})")
                print("  • Mục tiêu: Dạy Concept Heads & Backbone nhận diện 72 concepts chuẩn xác.")
                print("  • Đóng băng (Freeze): Tạm khóa CRL Reasoner & ActionHead để chống nhiễu gradient.")
                print("  • Hàm mất mát: Asymmetric BCE Loss trên 72 concepts (L_c^0).")
                print(f"{'='*70}\n")
            for p in self.model.reasoner.parameters():
                p.requires_grad = False
            self._init_optimizer()
        else:
            if verbose:
                print(f"\n{'='*70}")
                print(f"[STAGE 2 ACTIVATED] 🚀 END-TO-END LOGIC REASONING (Epoch {self.cfg.get('concept_warmup_epochs', 0) + 1} → {self.cfg['num_epoch']})")
                print("  • Mục tiêu: Mở khóa CRL Reasoner & ActionHead để học các luật DNF và phân loại hành động.")
                print("  • Dữ liệu vào: Concept logits z^0 đã được định hình sạch sẽ từ Stage 1.")
                print("  • Hàm mất mát: Action Cross-Entropy + Auxiliary Concept Loss.")
                print(f"{'='*70}\n")
            for p in self.model.reasoner.parameters():
                p.requires_grad = True
            self._init_optimizer()

    def _init_optimizer(self):
        base_lr = self.cfg["base_lr"]

        if self.current_stage == 1:
            # Stage 1: Chỉ tối ưu hóa Backbone và Concept Heads
            bb_scale = self.cfg.get("backbone_lr_scale", 0.1)
            backbone_params = list(self.model.backbone.parameters()) + \
                              list(self.model.spatial_pool_score.parameters()) + \
                              list(self.model.temporal_pool_score.parameters())
            head_params = list(self.model.spatial_head.parameters()) + \
                          list(self.model.temporal_head.parameters())
            params = [
                {"params": backbone_params, "lr": base_lr * bb_scale, "name": "backbone"},
                {"params": head_params, "lr": base_lr, "name": "concept_heads"},
            ]
            self.optimizer = optim.AdamW(params, lr=base_lr, weight_decay=self.cfg["weight_decay"])

            if self.train_loader is not None:
                n_iter = len(self.train_loader)
                warmup_epochs = max(1, self.cfg.get("concept_warmup_epochs", 1))
                total_steps = warmup_epochs * n_iter
                self.scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=total_steps, lr_min=self.cfg["min_lr"],
                    warmup_t=0, cycle_limit=1, t_in_epochs=False,
                )
            return

        # Stage 2: Mở toàn bộ mô hình
        is_staged_p1 = (
            self.current_phase == 1
            and not getattr(self, "is_pure_soft", False)
            and 0 < self.cfg.get("phase1_epochs", 0) < self.cfg["num_epoch"]
        )
        if is_staged_p1:
            params = [{"params": self.model.parameters(), "lr": base_lr}]
        else:
            bb_scale = self.cfg["backbone_lr_scale"]
            crl_scale = self.cfg["crl_lr_scale"]
            uc_scale = self.cfg["update_cell_lr_scale"]

            backbone_params = list(self.model.backbone.parameters()) + \
                              list(self.model.spatial_pool_score.parameters()) + \
                              list(self.model.temporal_pool_score.parameters()) + \
                              list(self.model.spatial_head.parameters()) + \
                              list(self.model.temporal_head.parameters())
            crl_params = list(self.model.reasoner.crl.parameters())
            if getattr(self.model, "bypass_head", None) is not None:
                crl_params += list(self.model.bypass_head.parameters())
            uc_params = list(self.model.reasoner.update_cell.parameters()) + [self.model.reasoner.raw_beta]

            params = [
                {"params": backbone_params, "lr": base_lr * bb_scale, "name": "backbone"},
                {"params": crl_params, "lr": base_lr * crl_scale, "name": "crl"},
                {"params": uc_params, "lr": base_lr * uc_scale, "name": "update_cell"},
            ]

        self.optimizer = optim.AdamW(params, lr=base_lr, weight_decay=self.cfg["weight_decay"])

        if self.train_loader is not None:
            n_iter = len(self.train_loader)
            stage1_epochs = self.cfg.get("concept_warmup_epochs", 0)
            remaining_epochs = max(1, self.cfg["num_epoch"] - stage1_epochs)
            total_steps = remaining_epochs * n_iter
            warmup_steps = min(int(self.cfg.get("warm_up_epoch", 2) * n_iter), int(total_steps * 0.1))
            self.scheduler = CosineLRScheduler(
                self.optimizer, t_initial=total_steps, lr_min=self.cfg["min_lr"],
                warmup_lr_init=self.cfg["warmup_lr"], warmup_t=warmup_steps, cycle_limit=1, t_in_epochs=False,
            )

    def check_stage_and_phase(self, epoch):
        # 1. Kiểm tra chuyển đổi Stage (Stage 1 -> Stage 2)
        warmup_ep = self.cfg.get("concept_warmup_epochs", 0)
        target_stage = 1 if (warmup_ep > 0 and epoch < warmup_ep) else 2
        if target_stage != self.current_stage:
            self._set_stage(target_stage, verbose=True)
            self.phase_step = 0

        # 2. Kiểm tra nhiệt độ STE / Phase switch khi ở Stage 2
        if self.current_stage == 2:
            self.check_phase_switch(epoch)

    def check_phase_switch(self, epoch):
        if getattr(self, "is_pure_soft", False):
            return

        stage1_ep = self.cfg.get("concept_warmup_epochs", 0)
        effective_epoch = max(0, epoch - stage1_ep)
        effective_total = max(1, self.cfg["num_epoch"] - stage1_ep - 1)

        if self.cfg.get("phase1_epochs", 0) <= 0:
            # Pure STE continuous annealing từ 2.0 → 10.0 trong Stage 2
            progress = min(1.0, max(0.0, effective_epoch / effective_total))
            temp = self.cfg["ste_temperature_start"] + progress * (self.cfg["ste_temperature_end"] - self.cfg["ste_temperature_start"])
            self.model.reasoner.temperature = temp
            return

        new_phase = 1 if epoch < self.cfg["phase1_epochs"] else 2
        if new_phase != self.current_phase:
            print(f"\\n{'='*60}\\n[PHASE SWITCH] Switching from Phase {self.current_phase} to Phase {new_phase} at Epoch {epoch+1}\\n{'='*60}")
            self.current_phase = new_phase
            self.model.set_phase(new_phase)
            self._init_optimizer()
            self.phase_step = 0

        if self.current_phase == 2:
            rem = max(1, self.cfg["num_epoch"] - self.cfg["phase1_epochs"])
            progress = (epoch - self.cfg["phase1_epochs"]) / rem
            temp = self.cfg["ste_temperature_start"] + progress * (self.cfg["ste_temperature_end"] - self.cfg["ste_temperature_start"])
            self.model.reasoner.temperature = temp

    def train_epoch(self, epoch):
        self.model.train()
        self.check_stage_and_phase(epoch)

        is_stage1 = (self.current_stage == 1)
        total_loss, correct, total = 0.0, 0, 0

        if is_stage1:
            mode_tag = f"Stage 1: Concept-Warmup ({epoch+1}/{self.cfg.get('concept_warmup_epochs', 0)})"
        elif getattr(self, "is_pure_soft", False):
            mode_tag = "Stage 2: Pure-Soft"
        elif self.current_phase == 2:
            mode_tag = f"Stage 2: Pure-STE T={self.model.reasoner.temperature:.2f}"
        else:
            mode_tag = f"Stage 2: Phase-{self.current_phase}"

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['num_epoch']} ({mode_tag})")
        for batch in pbar:
            if len(batch) == 5:
                data, index_t, label, _, c_targets = batch
            else:
                data, index_t, label, _ = batch[:4]
                c_targets = None

            data = data.float().to(self.device)
            index_t = index_t.float().to(self.device)
            label = label.long().to(self.device)
            c_targets = c_targets.float().to(self.device) if c_targets is not None else None

            # Step scheduler
            if self.current_stage == 1:
                self.scheduler.step_update(self.global_step)
            else:
                self.scheduler.step_update(self.phase_step)
                self.phase_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            if is_stage1:
                # Stage 1: Huấn luyện riêng Concept Head bằng Asymmetric BCE Loss
                out = self.model(data, index_t=index_t, concept_targets=c_targets)
                loss = out.get("l_c_0", None)
                if loss is None:
                    loss = self.model._asymmetric_bce(out["initial_concept_logits"], c_targets)

                loss.backward()
                if self.cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.optimizer.step()

                total_loss += loss.item()
                c_pred = (out["initial_concept_logits"] > 0.0).float()
                c_acc = (c_pred == c_targets).float().mean().item() if c_targets is not None else 0.0
                total += 1
                correct += c_acc
                pbar.set_postfix({"c_loss": f"{loss.item():.4f}", "c_acc": f"{correct/total*100:.1f}%"})
            else:
                # Stage 2: Huấn luyện End-to-End đầy đủ (Action Cross-Entropy + Logic Aux Loss)
                out = self.model(data, index_t=index_t, concept_targets=c_targets, action_labels=label)
                loss = torch.nn.functional.cross_entropy(out["action_logits"], label)
                if "aux_loss" in out:
                    loss += out["aux_loss"]

                loss.backward()
                if self.cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                self.optimizer.step()
                if hasattr(self.model, "clip_weights"):
                    self.model.clip_weights()

                total_loss += loss.item()
                pred = out["action_logits"].argmax(dim=-1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.2f}%"})

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()
        correct, total = 0, 0
        for batch in tqdm(self.test_loader, desc="Evaluation"):
            if len(batch) >= 4:
                data, index_t, label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].long().to(self.device)
            out = self.model(data, index_t=index_t)
            logits = out["action_logits"] if isinstance(out, dict) else out
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        acc = correct / total if total > 0 else 0.0
        return acc

    def _build_checkpoint(self, epoch):
        # Gói đầy đủ state để resume, khớp optimizer/phase hiện tại (Phase 1 hoặc 2)
        return {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "phase_step": self.phase_step,
            "current_stage": self.current_stage,
            "current_phase": self.current_phase,
            "best_acc": self.best_acc,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint_path, resume_training=True):
        \"\"\"Nạp checkpoint vào mô hình và khôi phục trạng thái huấn luyện nếu cần.\"\"\"
        if not os.path.exists(checkpoint_path):
            print(f"[CLIRTrainer] ⚠️ Không tìm thấy checkpoint tại: {checkpoint_path}")
            return False

        print(f"[CLIRTrainer] 🔄 Đang nạp checkpoint từ: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Trích xuất model state_dict (hỗ trợ checkpoint đầy đủ hoặc pure state_dict)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
        print(f"[CLIRTrainer] ✓ Đã nạp weights thành công. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        if resume_training and isinstance(ckpt, dict) and "epoch" in ckpt:
            self.start_epoch = ckpt.get("epoch", 0)
            self.global_step = ckpt.get("global_step", 0)
            self.phase_step = ckpt.get("phase_step", 0)
            self.best_acc = ckpt.get("best_acc", 0.0)

            # Khôi phục đúng Stage theo start_epoch
            warmup_ep = self.cfg.get("concept_warmup_epochs", 0)
            target_stage = 1 if (warmup_ep > 0 and self.start_epoch < warmup_ep) else 2
            self._set_stage(target_stage, verbose=True)

            if self.current_stage == 2:
                self.current_phase = ckpt.get("current_phase", self.current_phase)
                if hasattr(self.model, "set_phase"):
                    self.model.set_phase(self.current_phase)
                self.check_phase_switch(self.start_epoch)

            # Nạp trạng thái optimizer
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                    print("[CLIRTrainer] ✓ Đã khôi phục trạng thái Optimizer.")
                except Exception as e:
                    print(f"[CLIRTrainer] ⚠️ Không thể khôi phục optimizer state ({e}). Sẽ dùng optimizer mới khởi tạo.")

            print(f"★ [CLIRTrainer] Khôi phục phiên huấn luyện: Bắt đầu từ Epoch {self.start_epoch + 1}/{self.cfg['num_epoch']} (Stage {self.current_stage}, Phase {self.current_phase}, Best Acc: {self.best_acc*100:.2f}%)")
        else:
            print("★ [CLIRTrainer] Đã nạp weights (không khôi phục optimizer/epoch metadata).")

        return True

    def run(self):
        start_ep = getattr(self, "start_epoch", 0)
        total_ep = self.cfg["num_epoch"]
        if start_ep >= total_ep:
            print(f"[CLIRTrainer] Quá trình huấn luyện đã hoàn tất: start_epoch ({start_ep}) >= num_epoch ({total_ep}).")
            return

        print(f"Starting training from Epoch {start_ep + 1} to {total_ep} on {self.cfg['benchmark'].upper()}...")
        for epoch in range(start_ep, total_ep):
            train_loss, train_acc = self.train_epoch(epoch)

            # Lưu checkpoint mới nhất mỗi epoch (đầy đủ optimizer, khớp stage & phase hiện tại)
            latest_path = os.path.join(self.cfg["work_dir"], "checkpoint_latest.pt")
            torch.save(self._build_checkpoint(epoch), latest_path)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.cfg["num_epoch"]:
                val_acc = self.eval_epoch()
                stage_tag = "Stage 1 (Concept-Warmup)" if self.current_stage == 1 else "Stage 2 (Logic)"
                metric_name = "Mean Concept Acc" if self.current_stage == 1 else "Train Acc"
                print(f"[Epoch {epoch+1} - {stage_tag}] Train Loss: {train_loss:.4f}, {metric_name}: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

                # Save best checkpoint (chỉ lưu best ở Stage 2 khi đã có suy luận action)
                if (self.current_stage == 2 or self.cfg.get("concept_warmup_epochs", 0) <= 0) and val_acc > self.best_acc:
                    self.best_acc = val_acc
                    save_path = os.path.join(self.cfg["work_dir"], "best_model.pt")
                    torch.save(self.model.state_dict(), save_path)
                    best_ckpt_path = os.path.join(self.cfg["work_dir"], "best_checkpoint.pt")
                    torch.save(self._build_checkpoint(epoch), best_ckpt_path)
                    print(f"★ Saved new best model to {save_path} (Acc: {self.best_acc*100:.2f}%)")

print("CLIRTrainer defined successfully!")
""")

    add_code("""# Chạy huấn luyện (Bỏ comment để bắt đầu huấn luyện nếu đã có dữ liệu)
# Tùy chọn Resume Checkpoint:
# - Nếu muốn tiếp tục huấn luyện từ lần chạy trước khi Colab bị ngắt, trỏ đến checkpoint_latest.pt
# - Nếu muốn huấn luyện mới hoàn toàn từ Epoch 1, đặt RESUME_CHECKPOINT = None
# RESUME_CHECKPOINT = os.path.join(TRAIN_CONFIG["work_dir"], "checkpoint_latest.pt")
# RESUME_CHECKPOINT = None

# if train_loader is not None and test_loader is not None:
#     trainer = CLIRTrainer(TRAIN_CONFIG, train_loader, test_loader)
#     if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
#         trainer.load_checkpoint(RESUME_CHECKPOINT, resume_training=True)
#     trainer.run()
""")

    # Block 10
    add_md(r"""## 📜 Block 10: Trích xuất Luật Logic DNF & Đánh giá Tính giải thích (Rule Extraction)
Sau khi có checkpoint đã huấn luyện (hoặc dùng `model_smoke` để thử nghiệm), trích xuất các mệnh đề DNF tường minh cho 60 hành động NTU60 và đánh giá **Rule Coverage**, **Rule Precision**, **Model-Rule Fidelity**.
""")
    add_code("""# Đọc danh sách tên concept và action (khớp với số lớp của benchmark)
concept_names, action_names = [], []
with open("concepts/ntu_rgbd_60_120.csv", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
    concept_names = [h.strip() for h in lines[0].split(",")[1:]]
    action_names = [l.split(",")[0].strip() for l in lines[1:]]

# Lọc đúng 60 actions cho NTU60 Cross-Subject
action_names = action_names[:TRAIN_CONFIG["num_classes"]]
print(f"Loaded {len(concept_names)} concepts and {len(action_names)} actions ({TRAIN_CONFIG['benchmark'].upper()}).")

# Khởi tạo mô hình & nạp trọng số đã huấn luyện (nếu có best_model.pt hoặc best_checkpoint.pt)
best_model_path = os.path.join(TRAIN_CONFIG["work_dir"], "best_model.pt")
best_ckpt_path = os.path.join(TRAIN_CONFIG["work_dir"], "best_checkpoint.pt")

if os.path.exists(best_model_path) or os.path.exists(best_ckpt_path):
    load_path = best_model_path if os.path.exists(best_model_path) else best_ckpt_path
    print(f"★ Nạp mô hình đã huấn luyện từ: {load_path}")
    eval_model = CLIR_SkateFormer(
        num_classes=TRAIN_CONFIG["num_classes"],
        num_concepts=TRAIN_CONFIG["num_concepts"],
        spatial_concepts=TRAIN_CONFIG["spatial_concepts"],
        temporal_concepts=TRAIN_CONFIG["temporal_concepts"],
        num_loops=TRAIN_CONFIG["num_loops"],
        beta_max=TRAIN_CONFIG["beta_max"],
        beta_init=TRAIN_CONFIG["beta_init"],
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_data = torch.load(load_path, map_location="cpu")
    w = ckpt_data.get("model", ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data
    eval_model.load_state_dict({k.replace("module.", ""): v for k, v in w.items()}, strict=False)
    eval_model.eval()
    model_for_extract = eval_model.reasoner.crl
    model_for_eval = eval_model
    print("★ Đã nạp thành công trọng số thực tế để trích xuất luật DNF!")
else:
    print("ℹ Chưa tìm thấy checkpoint huấn luyện thực tế. Sử dụng model_smoke để minh họa luồng trích xuất luật.")
    model_for_extract = model_smoke.reasoner.crl
    model_for_eval = model_smoke

# Khởi tạo RuleExtractor
extractor = RuleExtractor(
    reasoner=model_for_extract,
    concept_names=concept_names,
    action_names=action_names,
    top_k_l2=3,
    top_k_l1=3,
    top_k_literals=4,
)

# Trích xuất luật DNF cho tất cả actions
os.makedirs("rules_output", exist_ok=True)
all_rules = extractor.extract_all_rules()

print(f"Successfully extracted DNF rules for all {len(all_rules)} actions!\\n")
print("--- Ví dụ 3 luật DNF trích xuất mẫu ---")
for act_idx in [0, 1, 2]:
    text_rule = RuleExtractor.format_rule_as_dnf(all_rules[act_idx])
    print(text_rule)
    print("-" * 50)

# Lưu vào file
RuleExtractor.save_rules(all_rules, "rules_output")
print("Saved rules to: rules_output/dnf_rules.json and rules_output/dnf_rules.txt")
""")

    add_code("""# Đánh giá Fidelity, Coverage, Precision trên tập kiểm tra thực tế
if test_loader is not None:
    print("Evaluating DNF rule metrics on test set...")
    metrics = extractor.evaluate_fidelity(model_for_eval, test_loader, device="cuda" if torch.cuda.is_available() else "cpu", n_batches=30)
    print("\\n" + "="*50)
    print("🎯 DNF RULE EXPLAINABILITY EVALUATION")
    print("="*50)
    print(f"  • Rule Coverage:       {metrics['coverage']*100:.2f}%  (Tỷ lệ mẫu có ít nhất 1 luật kích hoạt)")
    print(f"  • Rule Precision:      {metrics['precision']*100:.2f}%  (Độ chính xác khi luật DNF đưa ra dự đoán)")
    print(f"  • Model-Rule Fidelity: {metrics['fidelity']*100:.2f}%  (Độ tương đồng giữa luật và mạng nơ-ron)")
    print("="*50)
else:
    print("Test loader not active. DNF evaluation on dummy concept logits:")
    z_eval = torch.randn(8, 72)
    preds, scores, cov = extractor.evaluate_dnf(z_eval)
    print(f"Evaluated on 8 samples. Covered: {cov.sum().item()}/8")
""")

    # Block 11
    add_md(r"""## 💻 Block 11: Huấn luyện & Trích xuất qua Dòng lệnh CLI (NTU60 Cross-Subject)
Nếu bạn muốn huấn luyện theo script chuẩn hóa qua terminal hoặc cell Colab:
""")
    add_code("""# 1. Huấn luyện qua dòng lệnh trên NTU RGB+D 60 Cross-Subject (60CSub)
!python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j.yaml

# 2. Tiếp tục huấn luyện (Resume) từ checkpoint nếu bị gián đoạn:
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j.yaml \\
#     --weights work_dir/ntu60_cs/CLIR_SkateFormer_j/checkpoint_latest.pt \\
#     --start-epoch 35

# 3. Chỉ đánh giá (Test/Eval) checkpoint tốt nhất:
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j.yaml \\
#     --weights work_dir/ntu60_cs/CLIR_SkateFormer_j/best_model.pt \\
#     --phase test

# 4. Huấn luyện các biến thể Ablation trên NTU60 CS
# Ablation A2: True No-Loop (beta=0, num_loops=0)
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j_A2_NoLoop.yaml

# Ablation A5: Random Logic Feedback
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j_A5_RandomFeedback.yaml

# Ablation A6: CLIR-Bypass (Feature bypass branch)
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j_A6_Bypass.yaml

# 5. Trích xuất luật DNF sau huấn luyện từ file checkpoint
# !python scripts/extract_rules.py \\
#     --checkpoint work_dir/ntu60_cs/CLIR_SkateFormer_j/best_model.pt \\
#     --config config/train/ntu60_cs/CLIR_SkateFormer_j.yaml \\
#     --output-dir work_dir/rules_ntu60_cs \\
#     --eval-fidelity
""")

    # Save notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out_path = source_root / "CLIR_SkateFormer_Colab.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Generated Colab Notebook at: {out_path}")


if __name__ == "__main__":
    build_notebook()
