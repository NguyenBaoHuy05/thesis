"""CLIR-SkateFormer Training and Evaluation Processor.

Hỗ trợ:
- Pure STE Training (mặc định) với Continuous Temperature Annealing (2.0 → 10.0)
- 2-Phase Staged Training fallback (Phase 1 Soft Warm-up, Phase 2 STE) khi cấu hình phase1_epochs > 0
- Per-group learning rates: backbone (scale 0.1), crl (scale 1.0), update_cell (scale 2.0)
- Deep supervision: aux action loss trên intermediate y^t
- Concept supervision: Asymmetric BCE loss
- Rule Sparsity + Loop Stability regularization
- Reasoning trace logging (save_trace)
"""

from __future__ import print_function

import argparse
import csv
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs): pass
            def add_scalar(self, *args, **kwargs): pass
            def close(self): pass

from tqdm import tqdm

# Import local torchlight helper
_LOCAL_TORCHLIGHT = Path(__file__).resolve().parent / "torchlight"
if _LOCAL_TORCHLIGHT.is_dir() and str(_LOCAL_TORCHLIGHT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_TORCHLIGHT))

try:
    from torchlight import DictAction
except (ImportError, ModuleNotFoundError):
    class DictAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super().__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            input_dict = eval(f"dict({values})")
            output_dict = getattr(namespace, self.dest)
            output_dict.update(input_dict)
            setattr(namespace, self.dest, output_dict)

from timm.scheduler.cosine_lr import CosineLRScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition(".")
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            f"Class {class_str} cannot be found "
            f"({traceback.format_exception(*sys.exc_info())})"
        )


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported boolean value encountered.")


def compute_confusion_matrix(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(
        description="CLIR-SkateFormer: Concept Logic Iterative Refinement SkateFormer"
    )
    parser.add_argument("--work-dir", default="./work_dir", help="folder for storing results")
    parser.add_argument("--model_saved_name", default="")
    parser.add_argument("--config", default="./config", help="path to configuration file")

    # processor
    parser.add_argument("--phase", default="train", help="must be train or test")
    parser.add_argument("--save-score", type=str2bool, default=False)
    parser.add_argument("--save-trace", type=str2bool, default=False)

    # visualize and debug
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--save-epoch", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--print-log", type=str2bool, default=True)
    parser.add_argument("--show-topk", type=int, default=[1, 5], nargs="+")

    # feeder
    parser.add_argument("--feeder", default="feeders.feeder_ntu.Feeder")
    parser.add_argument("--num-worker", type=int, default=4)
    parser.add_argument("--train-feeder-args", action=DictAction, default=dict())
    parser.add_argument("--test-feeder-args", action=DictAction, default=dict())

    # model
    parser.add_argument("--model", default="model.clir_skateformer.CLIR_SkateFormer")
    parser.add_argument("--model-args", action=DictAction, default=dict())
    parser.add_argument("--weights", default=None)
    parser.add_argument("--ignore-weights", type=str, default=[], nargs="+")

    # optimization
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-lr", type=float, default=1e-6)
    parser.add_argument("--warmup_prefix", type=bool, default=False)
    parser.add_argument("--warm_up_epoch", type=int, default=10)
    parser.add_argument("--grad-clip", type=bool, default=True)
    parser.add_argument("--grad-max", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=[0], nargs="+")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--nesterov", type=str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=70)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--loss-type", type=str, default="LSCE")

    # ── CLIR-specific ────────────────────────────────────────────────────────
    parser.add_argument(
        "--phase1-epochs", type=int, default=0,
        help="Number of epochs for Phase 1 (0 = Pure STE from start, >0 = 2-phase staged training)"
    )
    parser.add_argument(
        "--backbone-lr-scale", type=float, default=0.1,
        help="Learning rate scale for backbone in Phase 2 / Pure STE (0 = freeze)"
    )
    parser.add_argument(
        "--crl-lr-scale", type=float, default=1.0,
        help="Learning rate scale for CRL modules"
    )
    parser.add_argument(
        "--update-cell-lr-scale", type=float, default=2.0,
        help="Learning rate scale for ConceptUpdateCell"
    )
    parser.add_argument(
        "--ste-temperature-start", type=float, default=2.0,
        help="Starting STE temperature (default 2.0)"
    )
    parser.add_argument(
        "--ste-temperature-end", type=float, default=10.0,
        help="Final STE temperature at end of training (linear annealing to 10.0)"
    )
    parser.add_argument(
        "--lambda-reg", type=float, default=0.005,
        help="Regularization weight lambda_reg for rule sparsity and stability"
    )
    parser.add_argument(
        "--concept-warmup-epochs", type=int, default=0,
        help="Number of epochs for Stage 1 Concept Warm-up (0 = disabled, >0 = freeze reasoner and train concept heads only)"
    )
    parser.add_argument(
        "--concept-threshold", type=float, default=0.5,
        help="Sigmoid threshold used for concept Precision/Recall/F1 logging"
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────

class Processor:
    """CLIR-SkateFormer Training / Evaluation Processor."""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == "train":
            if not arg.train_feeder_args.get("debug", False):
                arg.model_saved_name = os.path.join(arg.work_dir, "runs")
                if os.path.isdir(arg.model_saved_name):
                    print(f"Log dir {arg.model_saved_name} already exists. Appending runs.")
                self.train_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "train"), "train"
                )
                self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "val"), "val"
                )
            else:
                self.train_writer = self.val_writer = SummaryWriter(
                    os.path.join(arg.work_dir, "debug"), "debug"
                )

        self.global_step = 0
        self.best_acc = 0.0
        self.best_acc_epoch = 0

        # Nhận diện chế độ huấn luyện: Pure Soft vs Pure STE vs 2-Phase Staged
        self.is_pure_soft = (
            self.arg.model_args.get("literal_mode", "") == "soft"
            or self.arg.phase1_epochs >= self.arg.num_epoch
        )
        if self.is_pure_soft:
            self.current_training_phase = 1
        elif self.arg.phase1_epochs <= 0:
            self.current_training_phase = 2
        else:
            self.current_training_phase = 1

        self.use_cuda = (
            torch.cuda.is_available()
            and len(self.arg.device) > 0
            and self.arg.device[0] >= 0
        )
        self.output_device = self.arg.device[0] if self.use_cuda else "cpu"

        self.load_data()

        # Tự động load concept_names và concept_matrix từ dataset
        concept_dataset = self.data_loader.get("train", self.data_loader.get("test")).dataset
        self.phase_step = 0
        self.load_model()
        # Khởi tạo phase mô hình: Pure Soft, Pure STE, hoặc 2-Phase Warm-up
        m = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(m, "set_phase"):
            if self.is_pure_soft:
                m.set_phase(1)
                m.lambda_reg = 0.0
                self.print_log("[Training Mode] 🌿 PURE SOFT BASELINE (Continuous Sigmoid, No STE, lambda_reg=0.0)")
            elif self.arg.phase1_epochs <= 0:
                m.set_phase(2, temperature=self.arg.ste_temperature_start)
                m.lambda_reg = self.arg.lambda_reg
                self.print_log(
                    f"[Training Mode] 🚀 100% PURE STE ANNEALING (T: {self.arg.ste_temperature_start} → {self.arg.ste_temperature_end}, "
                    f"lambda_reg: {self.arg.lambda_reg})"
                )
            else:
                m.set_phase(1)
                self.print_log(
                    f"[Training Mode] 🔄 2-PHASE STAGED TRAINING (Phase 1: 1–{self.arg.phase1_epochs} soft, "
                    f"Phase 2: {self.arg.phase1_epochs + 1}–{self.arg.num_epoch} STE)"
                )

        if self.use_cuda:
            self.model = self.model.cuda(self.output_device)
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device,
                )

        self.current_stage = 1 if getattr(self.arg, "concept_warmup_epochs", 0) > 0 and self.arg.start_epoch < self.arg.concept_warmup_epochs else 2
        if self.current_stage == 1:
            if hasattr(m, "reasoner"):
                for p in m.reasoner.parameters():
                    p.requires_grad = False
            if hasattr(m, "action_head"):
                for p in m.action_head.parameters():
                    p.requires_grad = False
            if getattr(m, "bypass_head", None) is not None:
                for p in m.bypass_head.parameters():
                    p.requires_grad = False
            self.print_log(f"[Training Stage] 🎯 STAGE 1: Concept Warm-up (Epoch {self.arg.start_epoch + 1} → {self.arg.concept_warmup_epochs}). Reasoner & ActionHead frozen.")

        if self.arg.phase == "train":
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader["train"]))

        if self.arg.loss_type == "LSCE":
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        else:
            self.loss = nn.CrossEntropyLoss()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
            )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
        )

    def load_model(self):
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args)

        if self.arg.weights:
            self.print_log(f"Loading weights from {self.arg.weights}")
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "rb") as f:
                    checkpoint = pickle.load(f)
            else:
                checkpoint = torch.load(self.arg.weights, map_location="cpu")

            if isinstance(checkpoint, dict):
                for key in ("state_dict", "model"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        checkpoint = checkpoint[key]
                        break

            raw_weights = OrderedDict(
                [[k.split("module.")[-1], v] for k, v in checkpoint.items()]
            )

            for w in self.arg.ignore_weights:
                for k in list(raw_weights.keys()):
                    if w in k:
                        self.print_log(f"Ignoring weight: {k}")
                        del raw_weights[k]

            model_state = self.model.state_dict()
            mapped_weights = OrderedDict()
            for k, v in raw_weights.items():
                if k in model_state and model_state[k].shape == v.shape:
                    mapped_weights[k] = v
                else:
                    mapped_weights[k] = v

            missing, unexpected = self.model.load_state_dict(mapped_weights, strict=False)
            self.print_log(
                f"Loaded {len(mapped_weights)} tensors. "
                f"Missing={len(missing)}, Unexpected={len(unexpected)}"
            )

    def _get_param_groups(self):
        """Tạo parameter groups với per-group learning rates.

        Phase 1: Tất cả dùng base_lr
        Phase 2: backbone dùng backbone_lr_scale * base_lr (hoặc freeze),
                 CRL dùng crl_lr_scale * base_lr,
                 update_cell dùng update_cell_lr_scale * base_lr
        """
        # Lấy model thực (unwrap DataParallel nếu có)
        m = self.model.module if hasattr(self.model, "module") else self.model

        base_lr = self.arg.base_lr
        bb_scale = self.arg.backbone_lr_scale
        crl_scale = self.arg.crl_lr_scale
        uc_scale = self.arg.update_cell_lr_scale

        if getattr(self, "current_stage", 2) == 1:
            # Stage 1: Chỉ tối ưu hóa Backbone và Concept Heads
            backbone_params = list(m.backbone.parameters()) + \
                              list(m.spatial_pool_score.parameters()) + \
                              list(m.temporal_pool_score.parameters())
            head_params = list(m.spatial_head.parameters()) + \
                          list(m.temporal_head.parameters())
            return [
                {"params": backbone_params, "lr": base_lr * bb_scale, "name": "backbone"},
                {"params": head_params, "lr": base_lr, "name": "concept_heads"},
            ]

        # Staged training ở Phase 1 (warm-up): single group
        is_staged_phase1 = (
            self.current_training_phase == 1
            and not getattr(self, "is_pure_soft", False)
            and 0 < self.arg.phase1_epochs < self.arg.num_epoch
        )
        if is_staged_phase1:
            # Phase 1 của Staged Training: single group, tất cả trainable
            return [{"params": self.model.parameters(), "lr": base_lr}]
        else:
            # Pure STE, Pure Soft, hoặc Phase 2: per-group LRs (công bằng cho ablation so sánh)
            backbone_params = list(m.backbone.parameters()) + \
                              list(m.spatial_pool_score.parameters()) + \
                              list(m.temporal_pool_score.parameters()) + \
                              list(m.spatial_head.parameters()) + \
                              list(m.temporal_head.parameters())

            crl_params = list(m.reasoner.crl.parameters())
            # A6 Ablation: Thêm bypass_head vào nhóm crl để được tối ưu ở Phase 2
            if getattr(m, "bypass_head", None) is not None:
                crl_params += list(m.bypass_head.parameters())

            uc_params  = list(m.reasoner.update_cell.parameters()) + \
                         [m.reasoner.raw_beta]

            return [
                {"params": backbone_params, "lr": base_lr * bb_scale,  "name": "backbone"},
                {"params": crl_params,      "lr": base_lr * crl_scale, "name": "crl"},
                {"params": uc_params,       "lr": base_lr * uc_scale,  "name": "update_cell"},
            ]

    def load_optimizer(self):
        param_groups = self._get_param_groups()
        if self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                param_groups,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                param_groups,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

    def load_scheduler(self, n_iter_per_epoch):
        if getattr(self, "current_stage", 2) == 1:
            warmup_epochs = max(1, self.arg.concept_warmup_epochs - self.arg.start_epoch)
            num_steps = int(warmup_epochs * n_iter_per_epoch)
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=num_steps,
                lr_min=self.arg.min_lr,
                warmup_t=0,
                cycle_limit=1,
                t_in_epochs=False,
            )
        else:
            stage1_ep = getattr(self.arg, "concept_warmup_epochs", 0)
            effective_start = max(0, self.arg.start_epoch - stage1_ep)
            remaining_epochs = max(1, self.arg.num_epoch - stage1_ep - effective_start)
            num_steps = int(remaining_epochs * n_iter_per_epoch)
            warmup_steps = int(min(self.arg.warm_up_epoch, remaining_epochs * 0.1) * n_iter_per_epoch) if self.arg.start_epoch <= stage1_ep else 0
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
                lr_min=self.arg.min_lr,
                warmup_lr_init=self.arg.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=self.arg.warmup_prefix,
            )

    def _maybe_switch_stage(self, epoch: int):
        """Kiểm tra và chuyển đổi từ Stage 1 (Concept Warm-up) sang Stage 2 (Logic Reasoning)."""
        warmup_ep = getattr(self.arg, "concept_warmup_epochs", 0)
        target_stage = 1 if (warmup_ep > 0 and epoch < warmup_ep) else 2

        if target_stage != self.current_stage:
            self.current_stage = target_stage
            m = self.model.module if hasattr(self.model, "module") else self.model

            if target_stage == 2:
                # Mở khóa toàn bộ mô hình cho Stage 2
                if hasattr(m, "reasoner"):
                    for p in m.reasoner.parameters():
                        p.requires_grad = True
                if hasattr(m, "action_head"):
                    for p in m.action_head.parameters():
                        p.requires_grad = True
                if getattr(m, "bypass_head", None) is not None:
                    for p in m.bypass_head.parameters():
                        p.requires_grad = True

                self.print_log(
                    f"\n{'='*60}\n"
                    f"[STAGE SWITCH] 🔓 Chuyển sang STAGE 2: Logic Reasoning tại Epoch {epoch + 1}.\n"
                    f"Mở khóa CRL Reasoner & ActionHead để học luật DNF từ các concept đã ổn định.\n"
                    f"{'='*60}"
                )

                # Tái khởi tạo optimizer và scheduler cho số epochs còn lại
                self.load_optimizer()
                remaining_epochs = max(1, self.arg.num_epoch - epoch)
                n_iter = len(self.data_loader["train"])
                num_steps = int(remaining_epochs * n_iter)
                warmup_steps = int(min(self.arg.warm_up_epoch, remaining_epochs * 0.1) * n_iter)
                self.lr_scheduler = CosineLRScheduler(
                    self.optimizer,
                    t_initial=num_steps,
                    lr_min=self.arg.min_lr,
                    warmup_lr_init=self.arg.warmup_lr,
                    warmup_t=warmup_steps,
                    cycle_limit=1,
                    t_in_epochs=False,
                )
                self.phase_step = 0

    def _maybe_switch_phase(self, epoch: int):
        """Kiểm tra và chuyển phase nếu cần."""
        if getattr(self, "is_pure_soft", False):
            # Chế độ Pure Soft: chạy cố định Phase 1 suốt toàn bộ số epoch
            return

        if self.arg.phase1_epochs <= 0:
            # Chế độ Pure STE: chạy cố định Phase 2 từ Epoch 1, không switch/rebuild
            return

        m = self.model.module if hasattr(self.model, "module") else self.model

        if epoch < self.arg.phase1_epochs:
            new_phase = 1
        else:
            new_phase = 2

        if new_phase != self.current_training_phase:
            self.current_training_phase = new_phase
            if hasattr(m, "set_phase"):
                m.set_phase(new_phase)
                # Cập nhật lambda_reg từ arg
                if new_phase == 2:
                    m.lambda_reg = self.arg.lambda_reg

            # Rebuild optimizer với param groups mới
            self.load_optimizer()
            # Rebuild CosineLRScheduler cho Phase 2 với số epoch còn lại
            remaining_epochs = max(1, self.arg.num_epoch - epoch)
            n_iter = len(self.data_loader["train"])
            num_steps = int(remaining_epochs * n_iter)
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=num_steps,
                lr_min=self.arg.min_lr,
                warmup_t=0,
                cycle_limit=1,
                t_in_epochs=False,
            )
            self.phase_step = 0
            self.print_log(
                f"[Phase Switch] → Phase {new_phase} at epoch {epoch + 1}. "
                f"Optimizer rebuilt with per-group LRs and CosineLRScheduler initialized for remaining {remaining_epochs} epochs."
            )

    def _get_ste_temperature(self, epoch: int) -> float:
        """Tính STE temperature theo lịch annealing tuyến tính."""
        stage1_ep = getattr(self.arg, "concept_warmup_epochs", 0)
        effective_epoch = max(0, epoch - stage1_ep)
        effective_total = max(1, self.arg.num_epoch - stage1_ep - 1)

        if self.arg.phase1_epochs <= 0:
            # Pure STE: Annealing tuyến tính mượt mà từ T_start → T_end trên toàn bộ Stage 2
            progress = min(1.0, max(0.0, effective_epoch / effective_total))
            return (
                self.arg.ste_temperature_start
                + progress * (self.arg.ste_temperature_end - self.arg.ste_temperature_start)
            )

        # 2-Phase fallback
        if epoch < self.arg.phase1_epochs:
            return self.arg.ste_temperature_start
        progress = (epoch - self.arg.phase1_epochs) / max(
            1, self.arg.num_epoch - self.arg.phase1_epochs
        )
        return (
            self.arg.ste_temperature_start
            + progress * (self.arg.ste_temperature_end - self.arg.ste_temperature_start)
        )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, "config.yaml"), "w") as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_log(self, msg, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = f"[ {localtime} ] {msg}"
        print(msg)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, "log.txt"), "a", encoding="utf-8") as f:
                print(msg, file=f)

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log(f"--- Training Epoch: {epoch + 1}/{self.arg.num_epoch} ---")

        # ── Kiểm tra stage switch (Stage 1 -> Stage 2) ───────────────────────
        self._maybe_switch_stage(epoch)

        is_stage1 = (getattr(self, "current_stage", 2) == 1)

        # ── Cập nhật trạng thái từng epoch ──────────────────────────────────
        if is_stage1:
            self.print_log(
                f"[Stage 1: Concept Warm-up] Epoch {epoch + 1}/{self.arg.concept_warmup_epochs} - "
                f"Tối ưu hóa Concept Heads với Asymmetric BCE Loss (Reasoner & ActionHead frozen)"
            )
        else:
            # Kiểm tra phase switch trong Stage 2
            self._maybe_switch_phase(epoch)
            if self.current_training_phase == 2:
                temp = self._get_ste_temperature(epoch)
                m = self.model.module if hasattr(self.model, "module") else self.model
                if hasattr(m, "reasoner"):
                    m.reasoner.temperature = temp
                self.print_log(f"[Stage 2: Pure STE] - STE Temperature: {temp:.3f}, lambda_reg: {getattr(m, 'lambda_reg', 0.0)}")
            else:
                m = self.model.module if hasattr(self.model, "module") else self.model
                mode_desc = "Pure Soft" if getattr(self, "is_pure_soft", False) else "Phase 1 Warm-up"
                self.print_log(f"[Stage 2: {mode_desc}] - Mode: 'soft' (Continuous Sigmoid), lambda_reg: {getattr(m, 'lambda_reg', 0.0)}")

        # In các tham số thực tế đang được dùng ở đầu mỗi epoch để đối chiếu
        # với file YAML/notebook.
        m = self.model.module if hasattr(self.model, "module") else self.model
        reasoner = getattr(m, "reasoner", None)
        temperature = getattr(reasoner, "temperature", "n/a")
        literal_mode = getattr(reasoner, "literal_mode", "n/a")
        beta_text = "n/a"
        if reasoner is not None and hasattr(reasoner, "get_betas"):
            try:
                beta_values = reasoner.get_betas().detach().cpu().tolist()
                beta_text = ",".join(f"{b:.4f}" for b in beta_values)
            except Exception:
                beta_text = "unavailable"
        lr_text = ", ".join(
            f"{group.get('name', i)}={group['lr']:.3e}"
            for i, group in enumerate(self.optimizer.param_groups)
        )
        self.print_log(
            f"[Epoch Params] batch={self.arg.batch_size}, workers={self.arg.num_worker}, "
            f"batches={len(self.data_loader['train'])}, stage={self.current_stage}, "
            f"phase={self.current_training_phase}, mode={literal_mode}, T={temperature}, "
            f"lambda_reg={getattr(m, 'lambda_reg', 'n/a')}, beta=[{beta_text}], "
            f"concept_threshold={getattr(self.arg, 'concept_threshold', 0.5):.2f}, "
            f"LR=[{lr_text}]"
        )

        loader = self.data_loader["train"]
        loss_values, metric_values = [], []
        loss_component_values = {}

        def track_loss(name, value):
            """Lưu các loss scalar để in trung bình ở cuối epoch."""
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    return
                value = value.detach().item()
            elif not isinstance(value, (int, float)):
                return
            loss_component_values.setdefault(name, []).append(float(value))

        metric_name = "c_f1" if is_stage1 else "acc"
        concept_threshold = float(getattr(self.arg, "concept_threshold", 0.5))
        if not 0.0 <= concept_threshold <= 1.0:
            raise ValueError(f"concept_threshold must be in [0, 1], got {concept_threshold}")
        desc_stage = "Stage 1: Concept" if is_stage1 else "Stage 2: Action"
        process = tqdm(loader, desc=f"Epoch {epoch + 1} ({desc_stage})")

        # Stage 1 concept metrics are accumulated globally over the epoch.
        # Macro metrics are reported over concepts with at least one positive
        # target, avoiding the distortion from concepts absent in the split.
        if is_stage1:
            concept_tp = None
            concept_fp = None
            concept_fn = None

        def concept_prf(tp, fp, fn):
            precision = np.divide(
                tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
            )
            recall = np.divide(
                tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
            )
            f1 = np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) != 0,
            )
            supported = (tp + fn) > 0
            if np.any(supported):
                return (
                    float(precision[supported].mean()),
                    float(recall[supported].mean()),
                    float(f1[supported].mean()),
                    int(supported.sum()),
                )
            return 0.0, 0.0, 0.0, 0

        for batch_idx, batch in enumerate(process):
            if len(batch) == 5:
                data, index_t, label, index, concept_targets = batch
            else:
                data, index_t, label, index = batch
                concept_targets = None

            if is_stage1 or self.arg.phase1_epochs <= 0 or self.current_training_phase == 1:
                self.lr_scheduler.step_update(self.global_step)
            else:
                self.lr_scheduler.step_update(self.phase_step)
                self.phase_step += 1
            self.global_step += 1

            if self.use_cuda:
                data    = data.float().cuda(self.output_device)
                index_t = index_t.float().cuda(self.output_device)
                label   = label.long().cuda(self.output_device)
                if concept_targets is not None:
                    concept_targets = concept_targets.float().cuda(self.output_device)

            # Forward
            output = self.model(
                data,
                index_t=index_t,
                concept_targets=concept_targets,
                action_labels=label,
                return_trace=False,
            )

            if is_stage1:
                # Stage 1: Tối ưu l_c_0 (Asymmetric BCE Loss trên Concept Heads)
                if isinstance(output, dict) and "l_c_0" in output and output["l_c_0"] is not None and not (isinstance(output["l_c_0"], (int, float)) and output["l_c_0"] == 0.0):
                    total_loss = output["l_c_0"]
                else:
                    action_logits = output["action_logits"] if isinstance(output, dict) else output
                    total_loss = self.loss(action_logits, label)

                # Model hiện trả về initial concept logits với key
                # "initial_concept_logits" (không phải "z_0").
                if isinstance(output, dict) and concept_targets is not None:
                    z_0 = output.get("initial_concept_logits", output.get("z_0"))
                else:
                    z_0 = None

                if z_0 is not None:
                    pred_c = (torch.sigmoid(z_0) >= concept_threshold)
                    target_c = (concept_targets >= 0.5)
                    if concept_tp is None:
                        n_concepts = target_c.shape[1]
                        concept_tp = np.zeros(n_concepts, dtype=np.float64)
                        concept_fp = np.zeros(n_concepts, dtype=np.float64)
                        concept_fn = np.zeros(n_concepts, dtype=np.float64)
                    pred_np = pred_c.detach().cpu().numpy()
                    target_np = target_c.detach().cpu().numpy()
                    concept_tp += np.sum(pred_np & target_np, axis=0)
                    concept_fp += np.sum(pred_np & ~target_np, axis=0)
                    concept_fn += np.sum(~pred_np & target_np, axis=0)
                    c_precision, c_recall, c_f1, c_supported = concept_prf(
                        concept_tp, concept_fp, concept_fn
                    )
                    step_metric = c_f1
                else:
                    c_precision = c_recall = c_f1 = 0.0
                    c_supported = 0
                    step_metric = 0.0

                process.set_postfix({
                    "c_loss": f"{total_loss.item():.4f}",
                    "c_f1": f"{c_f1 * 100:.1f}%",
                    "c_p": f"{c_precision * 100:.1f}%",
                    "c_r": f"{c_recall * 100:.1f}%",
                })
            else:
                # Stage 2: Tối ưu toàn bộ mô hình (Action CE + Aux Loss)
                if isinstance(output, dict):
                    action_logits = output["action_logits"]
                    main_loss = self.loss(action_logits, label)
                    aux_loss  = output.get("aux_loss", 0.0)
                    total_loss = main_loss + aux_loss
                    track_loss("loss_main", main_loss)
                    track_loss("aux_loss", aux_loss)
                else:
                    action_logits = output
                    total_loss = self.loss(action_logits, label)
                    aux_loss = 0.0

                _, pred_label = torch.max(action_logits.data, 1)
                step_metric = torch.mean((pred_label == label.data).float()).item()
                process.set_postfix({"loss": f"{total_loss.item():.4f}", "acc": f"{step_metric * 100:.1f}%"})

            track_loss("loss_total", total_loss)
            if isinstance(output, dict):
                for component_name in (
                    "l_c_0", "l_c_K", "l_act_aux", "l_sup",
                    "l_sparse", "l_stable", "l_reg",
                ):
                    if component_name in output:
                        track_loss(component_name, output[component_name])

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()
            actual_model = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(actual_model, "clip_weights"):
                actual_model.clip_weights()

            loss_values.append(total_loss.item())
            metric_values.append(step_metric)

            self.train_writer.add_scalar(f"train/{metric_name}", step_metric, self.global_step)
            self.train_writer.add_scalar(f"train/{'c_loss' if is_stage1 else 'loss'}", total_loss.item(), self.global_step)
            if is_stage1:
                self.train_writer.add_scalar("train/concept_precision", c_precision, self.global_step)
                self.train_writer.add_scalar("train/concept_recall", c_recall, self.global_step)

            # Log loss components
            if isinstance(output, dict) and batch_idx % self.arg.log_interval == 0:
                if "l_sup" in output and isinstance(output["l_sup"], torch.Tensor):
                    self.train_writer.add_scalar(
                        "l_sup", output["l_sup"].item(), self.global_step
                    )
                if "l_reg" in output and isinstance(output["l_reg"], torch.Tensor):
                    self.train_writer.add_scalar(
                        "l_reg", output["l_reg"].item(), self.global_step
                    )

        mean_loss = np.mean(loss_values)
        mean_metric = np.mean(metric_values) * 100.0
        cur_lr    = self.optimizer.param_groups[0]["lr"]
        loss_details = " | ".join(
            f"{name}={np.mean(values):.6f}"
            for name, values in loss_component_values.items()
        )

        if is_stage1:
            if concept_tp is not None:
                concept_precision, concept_recall, concept_f1, concept_supported = concept_prf(
                    concept_tp, concept_fp, concept_fn
                )
            else:
                concept_precision = concept_recall = concept_f1 = 0.0
                concept_supported = 0
            self.print_log(
                f"\t[Stage 1: Concept Warm-up] Loss: {mean_loss:.4f} | "
                f"Concept F1 (macro): {concept_f1 * 100:.2f}% | "
                f"Precision: {concept_precision * 100:.2f}% | "
                f"Recall: {concept_recall * 100:.2f}% | "
                f"Supported: {concept_supported} | LR: {cur_lr:.6f} | "
                f"Loss details: {loss_details}"
            )
        else:
            self.print_log(
                f"\t[Stage 2: Logic Reasoning] Phase={self.current_training_phase} | "
                f"Loss: {mean_loss:.4f} | Action Acc: {mean_metric:.2f}% | LR: {cur_lr:.6f} | "
                f"Loss details: {loss_details}"
            )

        # Checkpoint
        m = self.model.module if hasattr(self.model, "module") else self.model
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split("module.")[-1], v.cpu()] for k, v in state_dict.items()])
        checkpoint = {
            "epoch":          epoch + 1,
            "global_step":    self.global_step,
            "stage":          self.current_stage,
            "model":          weights,
            "optimizer":      self.optimizer.state_dict(),
            "best_acc":       self.best_acc,
            "best_acc_epoch": self.best_acc_epoch,
            "training_phase": self.current_training_phase,
        }
        torch.save(checkpoint, os.path.join(self.arg.work_dir, "checkpoint_latest.pt"))

        if save_model:
            torch.save(
                weights,
                os.path.join(self.arg.work_dir, f"epoch_{epoch+1}_{metric_name}_{mean_metric:.2f}.pt"),
            )

    def eval(self, epoch, save_score=False, save_trace=False):
        self.model.eval()
        self.print_log(f"--- Evaluating Epoch: {epoch + 1} ---")
        loader = self.data_loader["test"]

        loss_values, score_fragments, label_list, pred_list = [], [], [], []
        trace_samples = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Eval")):
                if len(batch) == 5:
                    data, index_t, label, index, _ = batch
                else:
                    data, index_t, label, index = batch

                label_list.append(label.numpy())
                if self.use_cuda:
                    data    = data.float().cuda(self.output_device)
                    index_t = index_t.float().cuda(self.output_device)
                    label   = label.long().cuda(self.output_device)

                output = self.model(data, index_t=index_t, return_trace=save_trace)
                if isinstance(output, dict):
                    action_logits = output["action_logits"]
                    if save_trace and batch_idx == 0:
                        trace_samples.append({
                            "z_all": [z.cpu() for z in output.get("z_all", [])],
                            "betas": output.get("betas", None),
                        })
                else:
                    action_logits = output

                loss = self.loss(action_logits, label)
                loss_values.append(loss.item())
                score_fragments.append(action_logits.cpu().numpy())
                _, preds = torch.max(action_logits, 1)
                pred_list.append(preds.cpu().numpy())

        scores = np.concatenate(score_fragments)
        labels = np.concatenate(label_list)
        preds  = np.concatenate(pred_list)

        top1_acc = np.mean(preds == labels) * 100.0
        top5_preds = np.argsort(scores, axis=1)[:, -5:]
        top5_acc = np.mean([labels[i] in top5_preds[i] for i in range(len(labels))]) * 100.0

        # Chỉ lưu best_model.pt ở Stage 2 khi Action Classifier & Reasoner hoạt động thực sự
        if (getattr(self, "current_stage", 2) == 2 or getattr(self.arg, "concept_warmup_epochs", 0) <= 0) and top1_acc > self.best_acc:
            self.best_acc = top1_acc
            self.best_acc_epoch = epoch + 1
            # Save best checkpoint
            best_path = os.path.join(self.arg.work_dir, "best_model.pt")
            torch.save(
                OrderedDict([[k.split("module.")[-1], v.cpu()]
                             for k, v in self.model.state_dict().items()]),
                best_path,
            )
            self.print_log(f"\t[New Best] Saved to {best_path}")

        self.print_log(
            f"\tEval Loss: {np.mean(loss_values):.4f} | "
            f"Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}% | "
            f"Best: {self.best_acc:.2f}% @ Epoch {self.best_acc_epoch}"
        )

        if self.arg.phase == "train":
            self.val_writer.add_scalar("top1_acc", top1_acc, self.global_step)
            self.val_writer.add_scalar("top5_acc", top5_acc, self.global_step)

        if save_score:
            with open(os.path.join(self.arg.work_dir, f"epoch{epoch+1}_score.pkl"), "wb") as f:
                pickle.dump(scores, f)

        if save_trace and trace_samples:
            with open(os.path.join(self.arg.work_dir, f"epoch{epoch+1}_trace.pkl"), "wb") as f:
                pickle.dump(trace_samples, f)

        # Confusion matrix
        cm = compute_confusion_matrix(labels, preds)
        diag = np.diag(cm)
        sums = np.sum(cm, axis=1)
        per_class_acc = np.divide(
            diag, sums, out=np.zeros_like(diag, dtype=float), where=sums != 0
        )
        with open(
            os.path.join(self.arg.work_dir, f"epoch{epoch+1}_per_class_acc.csv"),
            "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(per_class_acc)
            writer.writerows(cm)

    def start(self):
        self.print_log(f"Parameters: {self.arg}")
        if self.arg.phase == "train":
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (
                    (epoch + 1) >= self.arg.save_epoch
                    and ((epoch + 1) % self.arg.save_interval == 0)
                )
                self.train(epoch, save_model=save_model)
                if (epoch + 1) % self.arg.eval_interval == 0:
                    self.eval(
                        epoch,
                        save_score=self.arg.save_score,
                        save_trace=self.arg.save_trace,
                    )
        elif self.arg.phase == "test":
            self.eval(epoch=0, save_score=self.arg.save_score, save_trace=self.arg.save_trace)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None and os.path.exists(p.config):
        with open(p.config, "r") as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f"Unknown Key in Config: {k}")
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
