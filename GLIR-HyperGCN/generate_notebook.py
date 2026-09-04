"""Generator for GLIR_HyperGCN_Colab.ipynb.

Viet theo dung khuon mau cua CLIR-SkateFormer/generate_notebook.py:
- Tat ca model/feeder/logic layer duoc doc truc tiep tu file nguon roi nhung
  (embed) thanh text vao tung cell code — notebook chay duoc doc lap, khong can
  clone lai repo.
- Cac import tuong doi (`from .xxx import yyy`) bi thay bang comment, vi trong
  notebook moi thu da nam chung 1 scope (cac class da duoc dinh nghia o cell truoc).
- Ho tro CA 2 cau hinh V2 (set_phase roi rac) va V3 (set_progress lien tuc) —
  chon qua bien VARIANT trong Block 9, giong tinh than train.py --variant v2|v3.
- Them Block rieng cho G2 (LightConceptGrounding, dung CLIP cai qua pip thay vi
  vendor lai file nhi phan bpe vocab — clip.load() du gi cung can internet de tai
  pretrained weights nen khong the "offline hoan toan" duoc).
"""

import json
import re
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

    tools_source       = (source_root / "feeders" / "tools.py").read_text(encoding="utf-8")
    feeder_ntu_source  = (source_root / "feeders" / "feeder_ntu.py").read_text(encoding="utf-8")
    hypergcn_source    = (source_root / "model" / "hypergcn.py").read_text(encoding="utf-8")
    literal_source     = (source_root / "model" / "crl" / "literal_layer.py").read_text(encoding="utf-8")
    conj_source        = (source_root / "model" / "crl" / "conjunction_layer.py").read_text(encoding="utf-8")
    disj_source        = (source_root / "model" / "crl" / "disjunction_layer.py").read_text(encoding="utf-8")
    crl_source         = (source_root / "model" / "crl" / "crl_reasoner.py").read_text(encoding="utf-8")
    grounding_source   = (source_root / "model" / "crl" / "light_grounding.py").read_text(encoding="utf-8")
    rule_ext_source    = (source_root / "model" / "crl" / "rule_extractor.py").read_text(encoding="utf-8")
    glir_source        = (source_root / "model" / "glir_hypergcn.py").read_text(encoding="utf-8")
    concepts_csv       = (source_root / "concepts" / "ntu_rgbd_60_120.csv").read_text(encoding="utf-8")

    # ── Don dep import cho scope phang cua notebook ───────────────────────────
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

    crl_clean       = re.sub(r"^from \.[^\n]+", "# relative import in notebook scope", crl_source, flags=re.MULTILINE)
    rule_ext_clean  = re.sub(r"^from \.[^\n]+", "# relative import in notebook scope", rule_ext_source, flags=re.MULTILINE)
    glir_clean      = re.sub(r"^from \.[^\n]+", "# relative import in notebook scope", glir_source, flags=re.MULTILINE)
    # G2 dung ban CLIP cai qua pip (Block 1: pip install git+CLIP) thay vi vendor
    # lai package model/clip (co file nhi phan bpe vocab, kho embed gon trong notebook).
    grounding_clean = grounding_source.replace(
        "from model.clip import clip  # vendored package trong model/clip/ (khong phai pip package 'clip')",
        "import clip  # cai qua pip trong Block 1 (git+https://github.com/openai/CLIP.git)",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────
    add_md(r"""# 🚀 GLIR-HyperGCN: All-in-One Google Colab & Local Notebook (NTU60 Cross-Subject / 60CSub)
> **Mô hình**: **GLIR-HyperGCN (Grounded Logic Iterative Refinement với backbone HyperGCN)**
> **Nâng cấp từ**: `CLIR-SkateFormer` — xem `Architecture/10_CLIR-SkateFormer_v2_Proposal.md`
> **Benchmark mặc định**: **NTU RGB+D 60 Cross-Subject (60CSub / NTU60_CS)**
> **Số lớp hành động**: **60 classes** (A1–A60)
> **Số khái niệm**: **72 concepts** (52 spatial + 20 temporal), backbone **HyperGCN (NTU-25 chuẩn)**
> **3 khoảng trống đã vá so với CLIR-SkateFormer gốc**:
> 1. **G1 — BodyPartGroundedConceptHeads**: concept không gian được dự đoán RIÊNG theo từng nhóm khớp NTU-25 (Head/Hand/Arm/Hip/Leg/Foot), khớp 1-1 với part-grouped L1 ở tầng logic (trước đây dùng chung 1 vector pooled toàn cục cho mọi concept).
> 2. **G2 — LightConceptGrounding**: CLIP đóng băng (không LoRA), text anchor encode 1 lần, chỉ học 1 projection nhỏ.
> 3. **G3 — set_progress()**: annealing liên tục (temperature/beta_max/lambda_reg) thay `set_phase()` đột ngột — chọn qua biến `VARIANT` (`v2` = set_phase cũ, `v3` = set_progress mới).

---

### 📌 Cấu trúc Toàn diện Notebook:
1. **Block 1**: Môi trường, Dependencies (torch, CLIP) & Mount Google Drive
2. **Block 2**: Concept Ontology CSV & NTU Feeder
3. **Block 3**: Perception Backbone HyperGCN
4. **Block 4**: Tầng Logic CRL (LiteralLayer, ConjunctionLayer, DisjunctionLayer, CRLReasoner)
5. **Block 5**: G2 — LightConceptGrounding (CLIP đóng băng)
6. **Block 6**: ConceptUpdateCell, IterativeLogicReasoner, BodyPartGroundedConceptHeads & GLIRHyperGCN
7. **Block 7**: RuleExtractor (Post-hoc DNF Rule Extraction & Fidelity Engine)
8. **Block 8**: Kiểm thử nhanh tính toàn vẹn (Quick Smoke Test — cả V2 và V3)
9. **Block 9**: Thiết lập Cấu hình, chọn VARIANT (v2/v3) & DataLoaders
10. **Block 10**: Vòng lặp Huấn luyện (GLIRTrainer) & Lưu Checkpoint
11. **Block 11**: Trích xuất & Đánh giá Luật DNF (Precision, Coverage, Fidelity)
12. **Block 12**: Huấn luyện & Trích xuất qua Dòng lệnh CLI (train.py / scripts/extract_rules.py)
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 1
    # ─────────────────────────────────────────────────────────────────────────
    add_md("## 📦 Block 1: Cài đặt Dependencies & Môi trường")
    add_code("""# Cài đặt thư viện cần thiết
!pip install timm tensorboardX pyyaml tqdm ftfy regex -q
!pip install git+https://github.com/openai/CLIP.git -q

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

    # ─────────────────────────────────────────────────────────────────────────
    # Block 2
    # ─────────────────────────────────────────────────────────────────────────
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
    add_code(f"""# NTU RGB+D Feeder (giống hệt CLIR-SkateFormer, khong doi khi doi backbone)
{feeder_clean}
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 3
    # ─────────────────────────────────────────────────────────────────────────
    add_md("## 🧠 Block 3: HyperGCN Perception Backbone\n"
           "Backbone NTU-25 chuẩn — `extract_feature_map()` trả về `[N, D, T', V=25]` "
           "(V không bị downsample qua backbone), dùng cho G1 ở Block 6.")
    add_code(f"""# HyperGCN Backbone Architecture
{hypergcn_source}
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 4
    # ─────────────────────────────────────────────────────────────────────────
    add_md("## ⚡ Block 4: Tầng Logic CRL (kế thừa nguyên trạng từ CLIR-SkateFormer)")
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

    # ─────────────────────────────────────────────────────────────────────────
    # Block 5 — G2
    # ─────────────────────────────────────────────────────────────────────────
    add_md("## 🌐 Block 5: G2 — LightConceptGrounding (CLIP đóng băng)\n"
           "CLIP **không** LoRA fine-tune — chỉ encode text anchor 1 lần lúc khởi tạo model, "
           "học 1 projection nhỏ. Nếu không có internet/không tải được CLIP checkpoint, "
           "module tự tắt (`available=False`, `L_ground=0`) — model vẫn chạy bình thường.")
    add_code(f"""# LightConceptGrounding (G2)
{grounding_clean}
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 6
    # ─────────────────────────────────────────────────────────────────────────
    add_md("## 🔄 Block 6: ConceptUpdateCell, IterativeLogicReasoner, BodyPartGroundedConceptHeads (G1) & GLIRHyperGCN")
    add_code(f"""# GLIR-HyperGCN Main Components
{glir_clean}
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 7
    # ─────────────────────────────────────────────────────────────────────────
    add_md("## 📜 Block 7: RuleExtractor (DNF Rules & Fidelity Evaluation — kế thừa nguyên trạng từ CLIR)")
    add_code(f"""# RuleExtractor
{rule_ext_clean}
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 8
    # ─────────────────────────────────────────────────────────────────────────
    add_md("## 🧪 Block 8: Kiểm thử nhanh tính toàn vẹn (Quick Smoke Test — V2 và V3, 60CSub)")
    add_code("""# Quick Model Verification Test cho NTU60 Cross-Subject (60 classes)
model_smoke = GLIRHyperGCN(
    num_classes=60,
    num_concepts=72,
    spatial_concepts=52,
    temporal_concepts=20,
    num_point=25,           # NTU-25 chuan (KHONG phai 24 nhu SkateFormer)
    embedding_channels=128,
    num_loops=2,
    lambda_sup=1.0,
    lambda_reg=0.0,
    use_grounding=True,
)
print("GLIR-HyperGCN (60CSub) initialized successfully!")
print(f"Total Parameters: {sum(p.numel() for p in model_smoke.parameters() if p.requires_grad):,}")
print(f"Grounding available (CLIP loaded): {model_smoke.grounding.available if model_smoke.grounding else False}")

# Test Dummy Forward — V=25 khop, khong phai V=24 nhu SkateFormer
x_dummy = torch.randn(2, 3, 64, 25, 2)
c_targets = torch.randint(0, 2, (2, 72)).float()
a_labels = torch.randint(0, 60, (2,)).long()

# V2 — set_phase() roi rac
model_smoke.set_phase(1)
out1 = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
print(f"[V2] Phase 1 Loss: {out1['aux_loss'].item():.4f}, beta={model_smoke.reasoner.get_betas().detach().numpy()}")
model_smoke.set_phase(2, temperature=6.0)
out2 = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
print(f"[V2] Phase 2 Loss: {out2['aux_loss'].item():.4f}, l_reg: {out2['l_reg'].item():.4f}")
assert out2["action_logits"].shape == (2, 60)

# V3 — set_progress() lien tuc
for progress in (0.0, 0.5, 1.0):
    model_smoke.set_progress(progress)
    out3 = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
    print(f"[V3] progress={progress:.1f}: temperature={model_smoke.reasoner.temperature:.2f}, "
          f"beta_max={model_smoke.reasoner.beta_max:.2f}, loss={out3['aux_loss'].item():.4f}")
assert out3["action_logits"].shape == (2, 60)
print("Smoke Test (V2 + V3) for 60CSub Passed!")
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 9
    # ─────────────────────────────────────────────────────────────────────────
    add_md(r"""## ⚙️ Block 9: Cấu hình Huấn luyện, chọn VARIANT (V2/V3) & Chuẩn bị DataLoaders
Mặc định **NTU60 Cross-Subject (60CSub)**. `VARIANT` quyết định lịch huấn luyện dùng ở Block 10:
`v2` = `set_phase()` rời rạc (giống CLIR gốc, dùng làm baseline so sánh); `v3` = `set_progress()` liên tục (đề xuất chính).
""")
    add_code("""# Cấu hình Benchmark & Variant
BENCHMARK = "ntu60_cs"  # @param ["ntu60_cs", "ntu60_cv", "ntu120_csub", "ntu120_cset"]
VARIANT = "v3"           # @param ["v2", "v3"]

DATASET_CONFIGS = {
    "ntu60_cs": {
        "num_classes": 60,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz",
        "backbone_weights": "/content/ntu60_CS/HyperGCN_j.pt",
        "work_dir": "./work_dir/ntu60_cs/glir_hypergcn",
    },
    "ntu60_cv": {
        "num_classes": 60,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz",
        "backbone_weights": "/content/ntu60_CV/HyperGCN_j.pt",
        "work_dir": "./work_dir/ntu60_cv/glir_hypergcn",
    },
    "ntu120_csub": {
        "num_classes": 120,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSub_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSub_64f.npz",
        "backbone_weights": "/content/ntu120_CSub/HyperGCN_j.pt",
        "work_dir": "./work_dir/ntu120_csub/glir_hypergcn",
    },
    "ntu120_cset": {
        "num_classes": 120,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSet_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSet_64f.npz",
        "backbone_weights": "/content/ntu120_CSet/HyperGCN_j.pt",
        "work_dir": "./work_dir/ntu120_cset/glir_hypergcn",
    },
}

selected = DATASET_CONFIGS[BENCHMARK]
work_dir = selected["work_dir"] + f"_{VARIANT}"

TRAIN_CONFIG = {
    "benchmark": BENCHMARK,
    "variant": VARIANT,
    "num_classes": selected["num_classes"],
    "num_concepts": 72,
    "spatial_concepts": 52,
    "temporal_concepts": 20,
    "num_point": 25,
    "embedding_channels": 128,

    "train_data": selected["train_data"],
    "test_data": selected["test_data"],
    "backbone_weights": selected["backbone_weights"],
    "work_dir": work_dir,

    "batch_size": 16,
    "test_batch_size": 16,
    "num_epoch": 70,
    "phase1_epochs": 30,     # chi dung cho VARIANT == "v2"
    "base_lr": 1e-4,
    "min_lr": 1e-6,
    "warmup_lr": 1e-6,
    "warm_up_epoch": 10,
    "weight_decay": 0.05,
    "grad_clip": 1.0,

    "lambda_sup": 1.0,
    "lambda_reg": 0.005,
    "lambda_ground": 0.1,

    "num_loops": 2,
    "beta_init": 0.1,
    "beta_max": 0.5,
    "use_grounding": True,
}

os.makedirs(TRAIN_CONFIG["work_dir"], exist_ok=True)
print(f"Active Benchmark: {BENCHMARK.upper()} ({TRAIN_CONFIG['num_classes']} action classes) — VARIANT={VARIANT.upper()}")
for k, v in TRAIN_CONFIG.items():
    print(f"  {k}: {v}")
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

    # ─────────────────────────────────────────────────────────────────────────
    # Block 10
    # ─────────────────────────────────────────────────────────────────────────
    add_md(r"""## 🏋️ Block 10: Vòng lặp Huấn luyện (GLIRTrainer — hỗ trợ cả V2 và V3) & Lưu Checkpoint
- **V2** (`set_phase`): Phase 1 (Epoch 1–30, `soft`, `beta_max=0.1`) → Phase 2 (Epoch 31+, `ste`, `beta_max=0.5`, `lambda_reg=0.005`) — **đột ngột**.
- **V3** (`set_progress`): nội suy liên tục `temperature: 1→10`, `beta_max: 0.1→0.5`, `lambda_reg: 0→0.005` theo `epoch/num_epoch` — **không có bước nhảy**.
- `checkpoint_latest.pt` lưu mỗi epoch (đầy đủ optimizer để resume); `best_checkpoint.pt` lưu đầy đủ khi đạt best acc, `best_model.pt` chỉ lưu weights (dùng cho RuleExtractor/inference).
""")
    add_code("""from tqdm import tqdm
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler

class GLIRTrainer:
    def __init__(self, cfg, train_loader, test_loader):
        self.cfg = cfg
        self.variant = cfg["variant"]
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GLIRHyperGCN(
            num_classes=cfg["num_classes"],
            num_concepts=cfg["num_concepts"],
            spatial_concepts=cfg["spatial_concepts"],
            temporal_concepts=cfg["temporal_concepts"],
            num_point=cfg["num_point"],
            embedding_channels=cfg["embedding_channels"],
            backbone_weights=cfg["backbone_weights"] if os.path.exists(cfg["backbone_weights"]) else None,
            num_loops=cfg["num_loops"],
            beta_max=cfg["beta_max"],
            beta_init=cfg["beta_init"],
            lambda_sup=cfg["lambda_sup"],
            lambda_reg=cfg["lambda_reg"] if self.variant == "v3" else 0.0,
            lambda_ground=cfg["lambda_ground"],
            use_grounding=cfg["use_grounding"],
        ).to(self.device)

        self.best_acc = 0.0
        self.global_step = 0

        if self.variant == "v2":
            self.model.set_phase(1)
        else:
            self.model.set_progress(0.0)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg["base_lr"], weight_decay=cfg["weight_decay"])
        if self.train_loader is not None:
            n_iter = len(self.train_loader)
            total_steps = cfg["num_epoch"] * n_iter
            warmup_steps = cfg["warm_up_epoch"] * n_iter
            self.scheduler = CosineLRScheduler(
                self.optimizer, t_initial=total_steps, lr_min=cfg["min_lr"],
                warmup_lr_init=cfg["warmup_lr"], warmup_t=warmup_steps, cycle_limit=1, t_in_epochs=False,
            )

    def _set_epoch_schedule(self, epoch):
        num_epoch = self.cfg["num_epoch"]
        if self.variant == "v2":
            new_phase = 1 if epoch < self.cfg["phase1_epochs"] else 2
            want_mode = "soft" if new_phase == 1 else "ste"
            if self.model.reasoner.literal_mode != want_mode:
                print(f"\\n{'='*60}\\n[V2][PHASE SWITCH] -> Phase {new_phase} tai epoch {epoch+1} (dot ngot)\\n{'='*60}")
                self.model.set_phase(new_phase)
        else:
            progress = epoch / max(1, num_epoch - 1)
            self.model.set_progress(progress)

    def train_epoch(self, epoch):
        self.model.train()
        self._set_epoch_schedule(epoch)
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"[{self.variant.upper()}] Epoch {epoch+1}/{self.cfg['num_epoch']}")
        for batch in pbar:
            if len(batch) == 5:
                data, _index_t, label, _, c_targets = batch
            else:
                data, _index_t, label, _ = batch[:4]
                c_targets = None

            data = data.float().to(self.device)
            label = label.long().to(self.device)
            c_targets = c_targets.float().to(self.device) if c_targets is not None else None

            self.scheduler.step_update(self.global_step)
            self.global_step += 1

            self.optimizer.zero_grad()
            out = self.model(data, concept_targets=c_targets, action_labels=label)
            loss = torch.nn.functional.cross_entropy(out["action_logits"], label)
            if "aux_loss" in out:
                loss += out["aux_loss"]
            loss.backward()
            if self.cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.optimizer.step()

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
            data, label = batch[0].float().to(self.device), batch[2].long().to(self.device)
            out = self.model(data)
            logits = out["action_logits"] if isinstance(out, dict) else out
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        return correct / total if total > 0 else 0.0

    def _build_checkpoint(self, epoch):
        return {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "variant": self.variant,
            "best_acc": self.best_acc,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def run(self):
        print(f"Starting full training ({self.variant.upper()}) for {self.cfg['num_epoch']} epochs on {self.cfg['benchmark'].upper()}...")
        for epoch in range(self.cfg["num_epoch"]):
            train_loss, train_acc = self.train_epoch(epoch)

            latest_path = os.path.join(self.cfg["work_dir"], "checkpoint_latest.pt")
            torch.save(self._build_checkpoint(epoch), latest_path)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.cfg["num_epoch"]:
                val_acc = self.eval_epoch()
                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    save_path = os.path.join(self.cfg["work_dir"], "best_model.pt")
                    torch.save(self.model.state_dict(), save_path)
                    best_ckpt_path = os.path.join(self.cfg["work_dir"], "best_checkpoint.pt")
                    torch.save(self._build_checkpoint(epoch), best_ckpt_path)
                    print(f"★ Saved new best model to {save_path} (Acc: {self.best_acc*100:.2f}%)")

print("GLIRTrainer defined successfully!")
""")

    add_code("""# Chạy huấn luyện (Bỏ comment để bắt đầu huấn luyện nếu đã có dữ liệu)
# if train_loader is not None and test_loader is not None:
#     trainer = GLIRTrainer(TRAIN_CONFIG, train_loader, test_loader)
#     trainer.run()
""")

    # ─────────────────────────────────────────────────────────────────────────
    # Block 11
    # ─────────────────────────────────────────────────────────────────────────
    add_md(r"""## 📜 Block 11: Trích xuất Luật Logic DNF & Đánh giá Tính giải thích (Rule Extraction)
Sau khi có checkpoint đã huấn luyện (hoặc dùng `model_smoke` để thử nghiệm), trích xuất các mệnh đề DNF tường minh cho 60 hành động NTU60 và đánh giá **Rule Coverage**, **Rule Precision**, **Model-Rule Fidelity**.
""")
    add_code("""# Đọc danh sách tên concept và action (khớp với số lớp của benchmark)
concept_names, action_names = [], []
with open("concepts/ntu_rgbd_60_120.csv", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
    concept_names = [h.strip() for h in lines[0].split(",")[1:]]
    action_names = [l.split(",")[0].strip() for l in lines[1:]]

action_names = action_names[:TRAIN_CONFIG["num_classes"]]
print(f"Loaded {len(concept_names)} concepts and {len(action_names)} actions ({TRAIN_CONFIG['benchmark'].upper()}).")

# Dùng model_smoke (Block 8) hoặc nạp lại từ best_model.pt sau khi train
model_for_extract = model_smoke.reasoner.crl
model_smoke.reasoner.literal_mode = "ste"  # bat buoc STE de trich xuat luat

extractor = RuleExtractor(
    reasoner=model_for_extract,
    concept_names=concept_names,
    action_names=action_names,
    top_k_l2=3,
    top_k_l1=3,
    top_k_literals=4,
)

os.makedirs("rules_output", exist_ok=True)
all_rules = extractor.extract_all_rules()

print(f"Successfully extracted DNF rules for all {len(all_rules)} actions!\\n")
print("--- Ví dụ 3 luật DNF trích xuất mẫu ---")
for act_idx in [0, 1, 2]:
    text_rule = RuleExtractor.format_rule_as_dnf(all_rules[act_idx])
    print(text_rule)
    print("-" * 50)

RuleExtractor.save_rules(all_rules, "rules_output")
print("Saved rules to: rules_output/dnf_rules.json and rules_output/dnf_rules.txt")
""")

    add_code("""# Đánh giá Fidelity, Coverage, Precision trên tập kiểm tra thực tế
if test_loader is not None:
    print("Evaluating DNF rule metrics on test set...")
    metrics = extractor.evaluate_fidelity(model_smoke, test_loader, device="cuda" if torch.cuda.is_available() else "cpu", n_batches=30)
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

    # ─────────────────────────────────────────────────────────────────────────
    # Block 12
    # ─────────────────────────────────────────────────────────────────────────
    add_md(r"""## 💻 Block 12: Huấn luyện & Trích xuất qua Dòng lệnh CLI (NTU60 Cross-Subject)
Nếu bạn muốn huấn luyện theo script chuẩn hóa qua terminal hoặc cell Colab (cần clone repo `GLIR-HyperGCN` đầy đủ, không chỉ notebook này):
""")
    add_code("""# 1. Huấn luyện qua dòng lệnh — VARIANT v2 (set_phase rời rạc, baseline) hoặc v3 (set_progress liên tục, đề xuất)
!python train.py --variant v3 --config config/ntu60_cs/glir_hypergcn.yaml
# !python train.py --variant v2 --config config/ntu60_cs/glir_hypergcn.yaml

# 2. Trích xuất luật DNF sau huấn luyện
# !python scripts/extract_rules.py \\
#     --checkpoint work_dir/ntu60_cs/glir_hypergcn_v3/best_model.pt \\
#     --config config/ntu60_cs/glir_hypergcn.yaml \\
#     --output-dir work_dir/rules_ntu60_cs \\
#     --eval-fidelity
""")

    # ── Save notebook ──────────────────────────────────────────────────────────
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

    out_path = source_root / "GLIR_HyperGCN_Colab.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Generated Colab Notebook at: {out_path}")


if __name__ == "__main__":
    build_notebook()
