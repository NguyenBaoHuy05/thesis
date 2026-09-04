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
> 4. Huấn luyện Staged 2-Phase (Phase 1 Soft Sigmoid Warm-up $\rightarrow$ Phase 2 STE & Rule Sparsification).

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
9. **Block 9**: Vòng lặp Huấn luyện Staged 2-Phase & Lưu Checkpoint
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

# Phase 1
model_smoke.set_phase(1)
out1 = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
print(f"Phase 1 Loss: {out1['aux_loss'].item():.4f}, beta={model_smoke.reasoner.get_betas().detach().numpy()}")
assert out1["action_logits"].shape == (2, 60), "Action logits must have shape [2, 60] for 60CSub"

# Phase 2
model_smoke.set_phase(2, temperature=6.0)
out2 = model_smoke(x_dummy, concept_targets=c_targets, action_labels=a_labels)
print(f"Phase 2 Loss: {out2['aux_loss'].item():.4f}, l_reg: {out2['l_reg'].item():.4f}, beta={model_smoke.reasoner.get_betas().detach().numpy()}")
assert out2["action_logits"].shape == (2, 60), "Action logits must have shape [2, 60] for 60CSub"
print("Smoke Test for 60CSub Passed!")
""")

    # Block 8
    add_md(r"""## ⚙️ Block 8: Cấu hình Huấn luyện & Chuẩn bị DataLoaders (NTU60 Cross-Subject)
Mặc định được thiết lập cho **NTU60 Cross-Subject (60CSub)**. Bạn cũng có thể chọn benchmark khác qua menu dropdown.
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

TRAIN_CONFIG = {
    "benchmark": BENCHMARK,
    "num_classes": selected["num_classes"],
    "num_concepts": 72,
    "spatial_concepts": 52,
    "temporal_concepts": 20,
    
    # Đường dẫn dữ liệu (thay đổi theo thư mục Google Drive của bạn)
    "train_data": selected["train_data"],
    "test_data": selected["test_data"],
    "backbone_weights": selected["backbone_weights"],
    "work_dir": selected["work_dir"],
    
    # Tham số huấn luyện
    "batch_size": 32,
    "test_batch_size": 32,
    "num_epoch": 70,
    "phase1_epochs": 30,
    "base_lr": 1e-4,
    "min_lr": 1e-6,
    "warmup_lr": 1e-6,
    "warm_up_epoch": 10,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    
    # Học theo nhóm ở Phase 2
    "backbone_lr_scale": 0.1,
    "crl_lr_scale": 1.0,
    "update_cell_lr_scale": 2.0,
    "lambda_sup": 1.0,
    "lambda_reg": 0.005,
    "ste_temperature_start": 5.0,
    "ste_temperature_end": 10.0,
    
    # Vòng lặp CLIR
    "num_loops": 2,
    "beta_init": 0.1,
    "beta_max": 0.5,
}

os.makedirs(TRAIN_CONFIG["work_dir"], exist_ok=True)
print(f"Active Benchmark: {BENCHMARK.upper()} ({TRAIN_CONFIG['num_classes']} action classes)")
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

    # Block 9
    add_md(r"""## 🏋️ Block 9: Vòng lặp Huấn luyện Staged 2-Phase & Lưu Checkpoint
- **Phase 1 (Epoch 1–30)**: Concept Warm-up (`soft mode`, $\beta_{\max}=0.1$, $\lambda_{\text{reg}}=0$).
- **Phase 2 (Epoch 31–70)**: STE Binarization (`ste mode`, $\beta_{\max}=0.5$, $\lambda_{\text{reg}}=0.005$, temperature $5.0 \rightarrow 10.0$).
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
        
        self.current_phase = 1
        self.model.set_phase(1)
        self.best_acc = 0.0
        self.global_step = 0
        self.phase_step = 0
        
        self._init_optimizer()

    def _init_optimizer(self):
        base_lr = self.cfg["base_lr"]
        if self.current_phase == 1:
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
        
        # Scheduler
        if self.train_loader is not None:
            n_iter = len(self.train_loader)
            if self.current_phase == 1:
                total_steps = self.cfg["num_epoch"] * n_iter
                warmup_steps = self.cfg["warm_up_epoch"] * n_iter
                self.scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=total_steps, lr_min=self.cfg["min_lr"],
                    warmup_lr_init=self.cfg["warmup_lr"], warmup_t=warmup_steps, cycle_limit=1, t_in_epochs=False,
                )
            else:
                remaining_epochs = max(1, self.cfg["num_epoch"] - self.cfg["phase1_epochs"])
                total_steps = remaining_epochs * n_iter
                self.scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=total_steps, lr_min=self.cfg["min_lr"],
                    warmup_t=0, cycle_limit=1, t_in_epochs=False,
                )

    def check_phase_switch(self, epoch):
        new_phase = 1 if epoch < self.cfg["phase1_epochs"] else 2
        if new_phase != self.current_phase:
            print(f"\\n{'='*60}\\n[PHASE SWITCH] Switching from Phase {self.current_phase} to Phase {new_phase} at Epoch {epoch+1}\\n{'='*60}")
            self.current_phase = new_phase
            self.model.set_phase(new_phase)
            self._init_optimizer()
            self.phase_step = 0
            
        if self.current_phase == 2:
            # Linear temperature annealing
            progress = (epoch - self.cfg["phase1_epochs"]) / max(1, self.cfg["num_epoch"] - self.cfg["phase1_epochs"])
            temp = self.cfg["ste_temperature_start"] + progress * (self.cfg["ste_temperature_end"] - self.cfg["ste_temperature_start"])
            self.model.reasoner.temperature = temp

    def train_epoch(self, epoch):
        self.model.train()
        self.check_phase_switch(epoch)
        total_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['num_epoch']} (P{self.current_phase})")
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
            if self.current_phase == 1:
                self.scheduler.step_update(self.global_step)
            else:
                self.scheduler.step_update(self.phase_step)
            self.global_step += 1
            self.phase_step += 1
            
            self.optimizer.zero_grad()
            out = self.model(data, index_t=index_t, concept_targets=c_targets, action_labels=label)
            
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
            "current_phase": self.current_phase,
            "best_acc": self.best_acc,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def run(self):
        print(f"Starting full training for {self.cfg['num_epoch']} epochs on {self.cfg['benchmark'].upper()}...")
        for epoch in range(self.cfg["num_epoch"]):
            train_loss, train_acc = self.train_epoch(epoch)

            # Lưu checkpoint mới nhất mỗi epoch (đầy đủ optimizer, khớp phase hiện tại)
            latest_path = os.path.join(self.cfg["work_dir"], "checkpoint_latest.pt")
            torch.save(self._build_checkpoint(epoch), latest_path)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.cfg["num_epoch"]:
                val_acc = self.eval_epoch()
                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

                # Save best checkpoint
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    save_path = os.path.join(self.cfg["work_dir"], "best_model.pt")
                    torch.save(self.model.state_dict(), save_path)
                    best_ckpt_path = os.path.join(self.cfg["work_dir"], "best_checkpoint.pt")
                    torch.save(self._build_checkpoint(epoch), best_ckpt_path)
                    print(f"★ Saved new best model to {save_path} (Acc: {self.best_acc*100:.2f}%)")

print("CLIRTrainer defined successfully!")
""")

    add_code("""# Chạy huấn luyện (Bỏ comment để bắt đầu huấn luyện nếu đã có dữ liệu)
# if train_loader is not None and test_loader is not None:
#     trainer = CLIRTrainer(TRAIN_CONFIG, train_loader, test_loader)
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

# Khởi tạo RuleExtractor
# Sử dụng model_smoke hoặc nạp từ best_model.pt
model_for_extract = model_smoke.reasoner.crl

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

    # Block 11
    add_md(r"""## 💻 Block 11: Huấn luyện & Trích xuất qua Dòng lệnh CLI (NTU60 Cross-Subject)
Nếu bạn muốn huấn luyện theo script chuẩn hóa qua terminal hoặc cell Colab:
""")
    add_code("""# 1. Huấn luyện qua dòng lệnh trên NTU RGB+D 60 Cross-Subject (60CSub)
!python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j.yaml

# 2. Huấn luyện các biến thể Ablation trên NTU60 CS
# Ablation A2: True No-Loop (beta=0, num_loops=0)
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j_A2_NoLoop.yaml

# Ablation A5: Random Logic Feedback
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j_A5_RandomFeedback.yaml

# Ablation A6: CLIR-Bypass (Feature bypass branch)
# !python main.py --config config/train/ntu60_cs/CLIR_SkateFormer_j_A6_Bypass.yaml

# 3. Trích xuất luật DNF sau huấn luyện
# !python scripts/extract_rules.py \
#     --checkpoint work_dir/ntu60_cs/CLIR_SkateFormer_j/best_model.pt \
#     --config config/train/ntu60_cs/CLIR_SkateFormer_j.yaml \
#     --output-dir work_dir/rules_ntu60_cs \
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
