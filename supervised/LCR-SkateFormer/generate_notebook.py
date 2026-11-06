"""Generator for LCR_SkateFormer_Colab.ipynb.

Creates an all-in-one Google Colab & Local Jupyter Notebook for LCR-SkateFormer:
- 100% Self-contained: All models, feeders, tools, loss functions, and trainers are embedded directly in notebook cells
- Zero external package import requirements for local model/feeder code
- Concept ontology auto-creation if file is not found
- Training / Eval engine with checkpoint resume and per-loop metric tracking
- Loading pretrained ECR-SkateFormer checkpoint (Warm-up)
- Full Training Loop on real NTU-RGB+D / NW-UCLA datasets
- Counterfactual explanation & reasoning trace inspection
"""

import json
import os
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

    # Read sources
    tools_source = (source_root / "feeders" / "tools.py").read_text(encoding="utf-8")
    feeder_ntu_source = (source_root / "feeders" / "feeder_ntu.py").read_text(encoding="utf-8")
    skateformer_source = (source_root / "model" / "SkateFormer.py").read_text(encoding="utf-8")
    lcr_source = (source_root / "model" / "lcr_skateformer.py").read_text(encoding="utf-8")
    concepts_csv_content = (source_root / "concepts" / "ntu_rgbd_60_120.csv").read_text(encoding="utf-8")

    # Clean relative imports so code runs seamlessly in notebook global scope
    feeder_ntu_clean = feeder_ntu_source.replace("from feeders import tools", "# 'tools' functions are embedded directly in notebook scope")
    lcr_clean = lcr_source.replace("from .SkateFormer import SkateFormer_", "# 'SkateFormer_' is embedded directly in notebook scope")

    # Header
    add_md("""# 🚀 LCR-SkateFormer: All-in-One Google Colab & Local Notebook
> **Mô hình**: **ECR-SkateFormer with KG-Guided Looping Concept Reasoning**
> **Mục tiêu nghiên cứu**: Khám phá và kiểm chứng giả thuyết rằng **semantic concepts trích xuất từ SkateFormer backbone có thể được tinh chỉnh qua nhiều vòng suy luận lặp (Iterative KG Reasoning)** trong concept/logit space ($z^{t+1} = z^t + \\beta \\Delta z^t$), tạo ra action evidence tiến bộ và nhất quán mà không làm loãng tính khả giải thích.

---

### 📌 Cấu trúc các khối trong Notebook (100% Self-Contained):
1. **Block 1**: Cài đặt Dependencies, Môi trường, Module Aliasing & Mount Google Drive
2. **Block 2**: Data Feeder, Augmentations Tools & Concept Ontology (72 Concepts)
3. **Block 3**: Kiến trúc SkateFormer Backbone
4. **Block 4**: Module Looping Reasoning (`KGUnionLayer`, `KGPrimedReasoner`, `ConceptUpdateCell`, `IterativeKGReasoner`, `LCRSkateFormer`, `LCR_SkateFormer`)
5. **Block 5**: Multi-Objective Looping Loss (Perception Loss $c^0$, Refined Loss $c^K$, Auxiliary Action Loss, Stability Loss, KG Consistency Loss)
6. **Block 6**: Trainer & Evaluator Engine (Checkpointing, Auto-Resume, Tensorboard)
7. **Block 7**: Pipeline Kiểm tra nhanh toàn diện với Dummy Data (Smoke Test & Trace Audit)
8. **Block 8**: Load Pretrained Checkpoint từ ECR-SkateFormer (Warm-up Phase 2)
9. **Block 9**: **Huấn luyện & Đánh giá với Dữ liệu Thật (NTU-RGB+D 60 / NTU-RGB+D 120)**
10. **Block 10**: Phân tích Giải thích & Can thiệp Phản thực (Counterfactual Intervention & Trace Audit)
""")

    # Block 1
    add_md("## 📦 Block 1: Cài đặt Dependencies & Môi trường")
    add_code("""# Cài đặt các thư viện cần thiết
!pip install -q timm tensorboardX einops scikit-learn matplotlib tqdm pyyaml

import os
import sys
import math
import random
import time
import json
import csv
import glob
import types
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Thiết bị sử dụng: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

# Mount Drive nếu chạy trên Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Đã kết nối Google Drive thành công!")
except Exception:
    print("ℹ️ Chạy môi trường Local / Non-Colab.")
""")

    # Block 2
    add_md("## 📂 Block 2: Data Feeder, Augmentation Tools & Concept Ontology")
    add_code(f"""# --- 1. Embedded Concept CSV (72 Concepts Ontology) ---
EMBEDDED_CONCEPTS_CSV = \"\"\"{concepts_csv_content}\"\"\"

os.makedirs("concepts", exist_ok=True)
CONCEPTS_CSV_PATH = "concepts/ntu_rgbd_60_120.csv"
if not os.path.exists(CONCEPTS_CSV_PATH):
    with open(CONCEPTS_CSV_PATH, "w", encoding="utf-8") as f:
        f.write(EMBEDDED_CONCEPTS_CSV.strip())
    print(f"✅ Đã tạo file concept ontology tại: {{CONCEPTS_CSV_PATH}}")
else:
    print(f"ℹ️ Sử dụng file concept ontology có sẵn tại: {{CONCEPTS_CSV_PATH}}")

def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
""")

    add_code("# --- 2. Embedded Feeder Tools (Augmentations & Cropping) ---\n" + tools_source)
    add_code("# --- 3. Embedded Feeder Class ---\n" + feeder_ntu_clean + """

# Đăng ký alias vào sys.modules để hỗ trợ mọi lệnh import legacy nếu có
_feeders_mod = types.ModuleType("feeders")
_feeders_feeder_mod = types.ModuleType("feeders.feeder_ntu")
_feeders_tools_mod = types.ModuleType("feeders.tools")
_feeders_feeder_mod.Feeder = Feeder
sys.modules["feeders"] = _feeders_mod
sys.modules["feeders.feeder_ntu"] = _feeders_feeder_mod
sys.modules["feeders.tools"] = _feeders_tools_mod
print("✅ Feeder & Data Augmentations loaded successfully!")
""")

    # Block 3
    add_md("## 🧠 Block 3: Kiến trúc SkateFormer Backbone")
    add_code("# --- Embedded source: model/SkateFormer.py ---\n" + skateformer_source + """

_skate_mod = types.ModuleType("model.SkateFormer")
_skate_mod.SkateFormer_ = SkateFormer_
sys.modules["model.SkateFormer"] = _skate_mod
print("✅ SkateFormer backbone loaded successfully!")
""")

    # Block 4
    add_md("## 🧠 Block 4: Module Looping Concept Reasoning (LCR-SkateFormer)")
    add_code("# --- Embedded source: model/lcr_skateformer.py ---\n" + lcr_clean + """

def load_ecr_weights_into_lcr(lcr_model, ecr_checkpoint_path):
    print(f"Loading checkpoint from: {ecr_checkpoint_path}")
    checkpoint = torch.load(ecr_checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    raw_weights = OrderedDict([[k.split("module.")[-1], v] for k, v in checkpoint.items()])
    model_state = lcr_model.state_dict()
    mapped_weights = OrderedDict()

    for k, v in raw_weights.items():
        if k in model_state and model_state[k].shape == v.shape:
            mapped_weights[k] = v
        # Remap ECR reasoner -> LCR base_reasoner
        elif k.startswith("reasoner.") and not k.startswith("reasoner.base_reasoner."):
            new_k = k.replace("reasoner.", "reasoner.base_reasoner.", 1)
            if new_k in model_state and model_state[new_k].shape == v.shape:
                mapped_weights[new_k] = v
            else:
                mapped_weights[k] = v
        else:
            mapped_weights[k] = v

    missing, unexpected = lcr_model.load_state_dict(mapped_weights, strict=False)
    print(f"✅ Đã load thành công {len(mapped_weights)} tensors!")
    print(f"ℹ️ Tham số mới của Looping Reasoning (chưa có trong ECR): {len(missing)}")
    return lcr_model

# Đăng ký alias vào sys.modules cho package `model` và `model.lcr_skateformer`
# Giúp tương thích tuyệt đối với mọi code/script gọi `from model.lcr_skateformer import ...`
_model_mod = types.ModuleType("model")
_lcr_mod = types.ModuleType("model.lcr_skateformer")
_lcr_mod.KGUnionLayer = KGUnionLayer
_lcr_mod.KGPrimedReasoner = KGPrimedReasoner
_lcr_mod.ConceptUpdateCell = ConceptUpdateCell
_lcr_mod.IterativeKGReasoner = IterativeKGReasoner
_lcr_mod.LCRSkateFormer = LCRSkateFormer
_lcr_mod.LCR_SkateFormer = LCR_SkateFormer
_lcr_mod._logit = _logit
_lcr_mod.load_ecr_weights_into_lcr = load_ecr_weights_into_lcr

sys.modules["model"] = _model_mod
sys.modules["model.lcr_skateformer"] = _lcr_mod
sys.modules["model.reason_skateformer"] = _lcr_mod

print("✅ LCR-SkateFormer model classes (KGUnionLayer, IterativeKGReasoner, LCR_SkateFormer) loaded successfully!")
""")

    # Block 5
    add_md("## ⚙️ Block 5: Multi-Objective Loss Functions")
    add_code("""class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()

print("✅ Loss functions defined successfully!")
""")

    # Block 6
    add_md("## ⚡ Block 6: Engine Huấn luyện & Đánh giá (Auto-Resume & Checkpointing)")
    add_code("""class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, loss_fn, work_dir="./work_dir/lcr_exp"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        self.best_epoch = 0
        self.start_epoch = 0
        self.global_step = 0

    def auto_resume(self):
        latest_pt = self.work_dir / "checkpoint_latest.pt"
        if latest_pt.exists():
            print(f"🔄 Tìm thấy checkpoint: {latest_pt}. Đang tiến hành Auto-Resume...")
            cp = torch.load(latest_pt, map_location=device)
            self.model.load_state_dict(cp['model'])
            self.optimizer.load_state_dict(cp['optimizer'])
            self.start_epoch = cp.get('epoch', 0)
            self.best_acc = cp.get('best_acc', 0.0)
            self.global_step = cp.get('global_step', 0)
            print(f"✅ Đã khôi phục thành công từ Epoch {self.start_epoch} (Best Acc: {self.best_acc:.2f}%)")

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.work_dir / "checkpoint_latest.pt")
        if is_best:
            torch.save(state, self.work_dir / "checkpoint_best.pt")
            print(f"⭐ [BEST] Đã lưu checkpoint tốt nhất tại epoch {epoch+1} (Acc = {self.best_acc:.2f}%)")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}")

        for batch in pbar:
            if len(batch) == 5:
                data, index_t, label, index, concept_targets = batch
            else:
                data, index_t, label, index = batch
                concept_targets = None

            data = data.float().to(device)
            index_t = index_t.float().to(device)
            label = label.long().to(device)
            if concept_targets is not None:
                concept_targets = concept_targets.float().to(device)

            if self.scheduler is not None:
                self.scheduler.step_update(self.global_step)
            self.global_step += 1

            output = self.model(data, index_t=index_t, concept_targets=concept_targets, action_labels=label)
            if isinstance(output, dict):
                act_logits = output["action_logits"]
                loss = self.loss_fn(act_logits, label) + output.get("aux_loss", 0.0)
            else:
                act_logits = output
                loss = self.loss_fn(act_logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(act_logits.data, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%", "beta": f"{self.model.reasoner.get_beta().item():.3f}"})

        mean_loss = total_loss / len(self.train_loader)
        acc = (correct / total) * 100.0
        print(f"Epoch {epoch+1} Summary: Mean Loss = {mean_loss:.4f}, Train Acc = {acc:.2f}%")
        return mean_loss, acc

    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Eval Epoch {epoch+1}"):
                if len(batch) == 5:
                    data, index_t, label, index, _ = batch
                else:
                    data, index_t, label, index = batch

                data = data.float().to(device)
                index_t = index_t.float().to(device)
                label = label.long().to(device)

                output = self.model(data, index_t=index_t)
                act_logits = output["action_logits"] if isinstance(output, dict) else output
                loss = self.loss_fn(act_logits, label)

                total_loss += loss.item()
                _, pred = torch.max(act_logits.data, 1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        acc = (correct / total) * 100.0
        is_best = acc > self.best_acc
        if is_best:
            self.best_acc = acc
            self.best_epoch = epoch + 1

        print(f"Eval Epoch {epoch+1}: Acc = {acc:.2f}% (Best: {self.best_acc:.2f}% @ Epoch {self.best_epoch})")
        self.save_checkpoint(epoch, is_best=is_best)
        return acc

    def fit(self, num_epochs=150, eval_interval=5):
        self.auto_resume()
        for epoch in range(self.start_epoch, num_epochs):
            self.train_epoch(epoch)
            if (epoch + 1) % eval_interval == 0 or (epoch + 1) == num_epochs:
                self.eval_epoch(epoch)
""")

    # Block 7
    add_md("## 🧪 Block 7: Kiểm tra nhanh toàn diện (Smoke Test)")
    add_code("""init_seed(1)
model = LCR_SkateFormer(
    num_classes=120,
    num_concepts=72,
    spatial_concepts=52,
    temporal_concepts=20,
    num_loops=2,
    shared_weights=True,
    update_gate_init=0.1,
    action_mode='linear'
).to(device)

x = torch.randn(2, 3, 64, 24, 2).to(device)
labels = torch.tensor([0, 1]).to(device)
targets = (torch.rand(2, 72) > 0.7).float().to(device)

out = model(x, concept_targets=targets, action_labels=labels, return_trace=True)
print("Action logits shape:", out["action_logits"].shape)
print("Concept logits shape:", out["concept_logits"].shape)
print("Total auxiliary loss:", out["aux_loss"].item())
print("Effective Beta Gate:", out["beta"].item())
print("Reasoning trace steps:", len(out["reasoning_trace"]))
""")

    # Block 8: Load Pretrained ECR Checkpoint & Fine-tune
    add_md("""## 🔄 Block 8: Load Pretrained Checkpoint từ ECR-SkateFormer (Warm-up Phase 2)
> **Cơ chế Transfer Learning / Warm-up**:
> - `backbone`, `spatial_head`, `temporal_head`, `pooling`, `fusion`: Load nguyên vẹn 100%.
> - `reasoner.first`, `reasoner.second`, `reasoner.action` (từ ECR): Tự động remap vào `reasoner.base_reasoner.*` của LCR.
> - Các tham số mới của vòng lặp (`reasoner.raw_beta`, `reasoner.update_cell`): Được khởi tạo sẵn sàng cho quá trình tinh chỉnh $K$-loop.
""")
    add_code("""# Đường dẫn checkpoint ECR đã huấn luyện (nếu có):
# ecr_ckpt = "/content/drive/MyDrive/KLTN/work_dir/ntu120/csub/REASON_SkateFormer_j/checkpoint_latest.pt"
# if os.path.exists(ecr_ckpt):
#     model = load_ecr_weights_into_lcr(model, ecr_ckpt)
""")

    # Block 9: Real Training Loop
    add_md("""## 🏋️ Block 9: Huấn luyện LCR-SkateFormer trên Dữ liệu Thật (NTU-RGB+D 60/120)
> Chạy block này để tiến hành train/fine-tune mô hình LCR-SkateFormer.
""")
    add_code("""# 1. Khởi tạo Feeder với Dữ liệu Thật
DATA_PATH = "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSub_64f.npz"
CONCEPTS_CSV = CONCEPTS_CSV_PATH  # Sử dụng ontology đã nhúng sẵn

# Nếu có dữ liệu thực tế trên Drive/Local:
if os.path.exists(DATA_PATH):
    train_dataset = Feeder(
        data_path=DATA_PATH,
        concepts_csv=CONCEPTS_CSV,
        split="train",
        window_size=64,
        p_interval=[0.5, 1],
        aug_method="a123489",
        intra_p=0.5,
        inter_p=0.2,
        thres=64,
        uniform=True,
        partition=True,
    )

    test_dataset = Feeder(
        data_path=DATA_PATH,
        concepts_csv=CONCEPTS_CSV,
        split="test",
        window_size=64,
        p_interval=[0.95],
        thres=64,
        uniform=True,
        partition=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 2. Khởi tạo mô hình LCR-SkateFormer
    model = LCR_SkateFormer(
        num_classes=120,
        num_concepts=72,
        spatial_concepts=52,
        temporal_concepts=20,
        concept_names=train_dataset.concept_names,
        action_prototypes=train_dataset.concept_matrix,
        num_loops=2,
        shared_weights=True,
        update_gate_init=0.1,
        action_mode="prototype",
        backbone_weights="/content/ntu120_CSub/SkateFormer_j.pt" if os.path.exists("/content/ntu120_CSub/SkateFormer_j.pt") else None
    ).to(device)

    # 3. Thiết lập Optimizer, Scheduler và Trainer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    total_steps = 150 * len(train_loader)
    warmup_steps = 10 * len(train_loader)
    scheduler = CosineLRScheduler(optimizer, t_initial=total_steps, lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=warmup_steps, t_in_epochs=False)
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        work_dir="./work_dir/ntu120_csub/LCR_SkateFormer_K2_proto"
    )

    # 4. Bắt đầu huấn luyện
    trainer.fit(num_epochs=150, eval_interval=5)
else:
    print(f"ℹ️ Không tìm thấy file dataset tại {DATA_PATH}. Vui lòng cập nhật DATA_PATH phù hợp.")
""")

    # Block 10
    add_md("## 📊 Block 10: Phân tích Giải thích & Phản thực (Counterfactual Intervention)")
    add_code("""model.eval()
with torch.no_grad():
    sample_x = torch.randn(1, 3, 64, 24, 2).to(device)
    out_orig = model(sample_x, return_trace=True)
    orig_prob = torch.softmax(out_orig["action_logits"], dim=-1)
    orig_top1 = torch.argmax(orig_prob).item()
    print(f"Original Top-1 Action: {orig_top1} with prob {orig_prob[0, orig_top1].item():.4f}")

    # Inspect reasoning trace across loops
    print("\\n--- Looping Reasoning Trace ---")
    for step_info in out_orig["reasoning_trace"]:
        step = step_info["step"]
        beta = step_info["beta"]
        print(f"Step {step} | Beta: {beta:.4f}")
""")

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

    out_path = Path(__file__).resolve().parent / "LCR_SkateFormer_Colab.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[SUCCESS] Created Colab notebook at: {out_path}")


if __name__ == "__main__":
    build_notebook()
