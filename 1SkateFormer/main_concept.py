"""
main_concept.py
---------------
Training and evaluation entry point for the concept decoder pipeline.

Supports phases:
  - train_concept    Train the ConceptDecoder + SymbolicDecoder on top of
                       frozen SkateFormer backbone.
  - test_concept     Evaluate concept predictions and action accuracy.
  - extract_features Extract and cache backbone features (for fast OOF).
  - build_template   Build template pseudo-labels from CSV.

Usage:
    python main_concept.py --config config/final/ntu60_cs.yaml
    python main_concept.py --config config/final/ntu60_cs.yaml --phase test_concept \\
        --weights work_dir/final_pipeline/ntu60_cs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from model.concept_decoder import masked_bce_loss, soft_bce_loss
from model.concept_model import ConceptModel
from feeders.feeder_ntu_concept import ConceptFeeder
from tools.build_template_labels import get_template_matrix, build_template_labels


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def init_seed(seed: int):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Unsupported value encountered.")


def import_class(name: str):
    """Dynamic class import by dotted path string."""
    mod_str, _, class_str = name.rpartition(".")
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError as e:
        raise ImportError(f"Cannot import {class_str} from {mod_str}: {e}")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def concept_metrics(
    probs: np.ndarray,       # [N, C]
    targets: np.ndarray,     # [N, C] values: 0, 1, or -1 (unknown)
    threshold: float = 0.5,
) -> dict:
    """Compute per-concept and aggregate metrics, ignoring unknown positions."""
    mask = targets >= 0
    preds = (probs >= threshold).astype(np.int32)

    # Flatten masked positions
    flat_targets = targets[mask].astype(np.int32)
    flat_preds   = preds[mask]

    macro_f1  = f1_score(flat_targets, flat_preds, average="macro",  zero_division=0)
    micro_f1  = f1_score(flat_targets, flat_preds, average="micro",  zero_division=0)
    precision = precision_score(flat_targets, flat_preds, average="micro", zero_division=0)
    recall    = recall_score(flat_targets, flat_preds, average="micro",    zero_division=0)
    coverage  = mask.mean()
    entropy   = float((-probs * np.log(probs.clip(1e-7)) - (1 - probs) * np.log((1 - probs).clip(1e-7))).mean())

    return {
        "macro_f1":  float(macro_f1),
        "micro_f1":  float(micro_f1),
        "precision": float(precision),
        "recall":    float(recall),
        "coverage":  float(coverage),
        "entropy":   float(entropy),
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description="SkateFormer Concept Decoder Pipeline")

    parser.add_argument("--work_dir", default="./work_dir/final_pipeline/default")
    parser.add_argument("--config", default=None)
    parser.add_argument("--phase", default="train_concept",
                        choices=["train_concept", "test_concept",
                                 "extract_features", "build_template"])

    # Seed
    parser.add_argument("--seed", type=int, default=1)

    # Feeder
    parser.add_argument("--feeder", default="feeders.feeder_ntu_concept.ConceptFeeder")
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--train_feeder_args", default=dict(), type=dict)
    parser.add_argument("--test_feeder_args",  default=dict(), type=dict)

    # Model
    parser.add_argument("--num_classes",  type=int, default=60)
    parser.add_argument("--num_concepts", type=int, default=74)
    parser.add_argument("--backbone_ckpt", default=None,
                        help="Path to pretrained SkateFormer checkpoint")
    parser.add_argument("--backbone_dim",  type=int, default=192)
    parser.add_argument("--backbone_kwargs", default=dict(), type=dict)
    parser.add_argument("--decoder_kwargs",  default=dict(), type=dict)
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)
    parser.add_argument("--use_symbolic",    type=str2bool, default=True)
    parser.add_argument("--symbolic_mode",   default="learned",
                        choices=["learned", "symbolic"])
    parser.add_argument("--weights",         default=None,
                        help="Decoder checkpoint to load (for fine-tuning or eval)")
    parser.add_argument("--ignore_weights",  default=[], nargs="+")

    # Loss
    parser.add_argument("--lambda_action",  type=float, default=1.0,
                        help="Weight for action CE loss")
    parser.add_argument("--lambda_concept", type=float, default=1.0,
                        help="Weight for concept BCE loss")
    parser.add_argument("--loss_type", default="masked_bce",
                        choices=["masked_bce", "soft_bce"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Optimiser
    parser.add_argument("--optimizer",     default="AdamW")
    parser.add_argument("--base_lr",       type=float, default=1e-3)
    parser.add_argument("--min_lr",        type=float, default=1e-5)
    parser.add_argument("--warmup_lr",     type=float, default=1e-7)
    parser.add_argument("--warm_up_epoch", type=int,   default=5)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--grad_clip",     type=str2bool, default=True)
    parser.add_argument("--grad_max",      type=float,    default=1.0)
    parser.add_argument("--warmup_prefix", type=str2bool, default=False)

    # Training schedule
    parser.add_argument("--device",          type=int, default=[0], nargs="+")
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_epoch",       type=int, default=20)
    parser.add_argument("--start_epoch",     type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--save_epoch",      type=int, default=0)
    parser.add_argument("--eval_interval",   type=int, default=1)
    parser.add_argument("--log_interval",    type=int, default=50)
    parser.add_argument("--print_log",       type=str2bool, default=True)
    parser.add_argument("--save_score",      type=str2bool, default=False)

    # Protocol
    parser.add_argument("--protocol",        default="ntu60_cs")
    parser.add_argument("--concept_csv",
                        default="./concepts/ntu_rgbd_60_120.csv")

    return parser


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ConceptProcessor:
    def __init__(self, arg):
        self.arg = arg
        self.best_acc = 0.0
        self.best_acc_epoch = 0
        self.no_improve_count = 0
        self.global_step = 0

        os.makedirs(arg.work_dir, exist_ok=True)
        self._save_config()
        self._setup_device()
        self._load_template()
        self._load_model()
        self._load_data()
        if arg.phase == "train_concept":
            self._load_optimizer()

    # ---- Setup ----

    def _setup_device(self):
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        torch.cuda.set_device(self.output_device)

    def _save_config(self):
        with open(os.path.join(self.arg.work_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(self.arg), f)

    def _load_template(self):
        """Build template matrix and save it."""
        self.template_matrix = get_template_matrix(
            num_classes=self.arg.num_classes,
            csv_path=self.arg.concept_csv,
        )
        np.save(
            os.path.join(self.arg.work_dir, "template_matrix.npy"),
            self.template_matrix,
        )
        self.log(f"Template matrix loaded: shape {self.template_matrix.shape}")

    def _load_model(self):
        template_tensor = torch.from_numpy(self.template_matrix)

        dk = dict(self.arg.decoder_kwargs)
        dk["mode"] = self.arg.symbolic_mode

        self.model = ConceptModel(
            backbone_ckpt=self.arg.backbone_ckpt,
            num_classes=self.arg.num_classes,
            num_concepts=self.arg.num_concepts,
            backbone_dim=self.arg.backbone_dim,
            backbone_kwargs=dict(self.arg.backbone_kwargs),
            decoder_kwargs=dk,
            freeze_backbone=self.arg.freeze_backbone,
            use_symbolic=self.arg.use_symbolic,
            template_matrix=template_tensor,
        ).cuda(self.output_device)

        # Load decoder weights if provided
        if self.arg.weights is not None:
            self._load_weights(self.arg.weights)

        self.log(f"Trainable params: {self.model.count_trainable():,}")
        self.log(f"Total params:     {self.model.count_total():,}")

        # DataParallel
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=self.output_device,
            )

    def _load_weights(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Weights not found: {path}")
        raw = torch.load(path, map_location="cpu")
        if "state_dict" in raw:
            raw = raw["state_dict"]
        weights = OrderedDict([(k.replace("module.", ""), v) for k, v in raw.items()])
        state = self.model.state_dict()
        # Only load matching keys
        filtered = {k: v for k, v in weights.items() if k in state and v.shape == state[k].shape}
        state.update(filtered)
        self.model.load_state_dict(state)
        self.log(f"Loaded decoder weights from {path} ({len(filtered)} keys)")

    def _load_data(self):
        arg = self.arg
        FeedClass = import_class(arg.feeder)
        self.data_loader: dict = {}

        if arg.phase == "train_concept":
            train_ds = FeedClass(**arg.train_feeder_args)
            self.data_loader["train"] = DataLoader(
                train_ds,
                batch_size=arg.batch_size,
                shuffle=True,
                num_workers=arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
                pin_memory=True,
            )

        test_ds = FeedClass(**arg.test_feeder_args)
        self.data_loader["test"] = DataLoader(
            test_ds,
            batch_size=arg.test_batch_size,
            shuffle=False,
            num_workers=arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
            pin_memory=True,
        )

    def _load_optimizer(self):
        params = self.model.module.decoder_parameters() if hasattr(self.model, "module") \
                 else self.model.decoder_parameters()

        if self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(params, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(params, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.arg.optimizer}")

        n_iter = len(self.data_loader["train"])
        num_steps   = int(self.arg.num_epoch * n_iter)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter)

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
            lr_min=self.arg.min_lr,
            warmup_lr_init=self.arg.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=self.arg.warmup_prefix,
        )

    # ---- Logging ----

    def log(self, msg: str, print_time: bool = True):
        if print_time:
            ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
            msg = f"{ts} {msg}"
        print(msg)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, "log.txt"), "a") as f:
                print(msg, file=f)

    # ---- Training ----

    def _compute_loss(
        self,
        concept_logits: torch.Tensor,
        concept_targets: torch.Tensor,
        concept_mask: torch.Tensor,
        action_logits: Optional[torch.Tensor],
        action_labels: torch.Tensor,
    ) -> tuple:
        # Concept loss
        if self.arg.loss_type == "soft_bce":
            loss_c = soft_bce_loss(concept_logits, concept_targets, concept_mask)
        else:
            loss_c = masked_bce_loss(concept_logits, concept_targets, concept_mask)

        # Action loss
        loss_a = torch.tensor(0.0, device=concept_logits.device)
        if action_logits is not None:
            loss_a = F.cross_entropy(action_logits, action_labels)

        total = self.arg.lambda_concept * loss_c + self.arg.lambda_action * loss_a
        return total, loss_c, loss_a

    def train(self, epoch: int):
        model = self.model
        model.train()
        loader = self.data_loader["train"]

        loss_vals, concept_loss_vals, action_loss_vals, acc_vals = [], [], [], []
        process = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.arg.num_epoch}")

        for batch_idx, batch in enumerate(process):
            self.scheduler.step_update(self.global_step)
            self.global_step += 1

            data, index_t, action_label, concept_target, concept_mask, _ = batch

            data           = data.float().cuda(self.output_device)
            index_t        = index_t.float().cuda(self.output_device)
            action_label   = action_label.long().cuda(self.output_device)
            concept_target = concept_target.float().cuda(self.output_device)
            concept_mask   = concept_mask.bool().cuda(self.output_device)

            concept_logits, action_logits = model(data, index_t)
            total, loss_c, loss_a = self._compute_loss(
                concept_logits, concept_target, concept_mask,
                action_logits, action_label
            )

            self.optimizer.zero_grad()
            total.backward()
            if self.arg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            loss_vals.append(total.item())
            concept_loss_vals.append(loss_c.item())
            action_loss_vals.append(loss_a.item())

            if action_logits is not None:
                acc = (action_logits.argmax(1) == action_label).float().mean().item()
                acc_vals.append(acc)

            if batch_idx % self.arg.log_interval == 0:
                process.set_postfix({
                    "loss": f"{total.item():.4f}",
                    "c_loss": f"{loss_c.item():.4f}",
                    "a_loss": f"{loss_a.item():.4f}",
                })

        self.log(
            f"  Epoch {epoch + 1} | Loss={np.mean(loss_vals):.4f} "
            f"| C={np.mean(concept_loss_vals):.4f} | A={np.mean(action_loss_vals):.4f} "
            f"| Acc={np.mean(acc_vals) * 100:.2f}% | LR={self.optimizer.param_groups[0]['lr']:.2e}"
        )

    # ---- Evaluation ----

    def eval(self, epoch: int, save_preds: bool = False) -> float:
        model = self.model
        model.eval()

        all_concept_probs = []
        all_concept_targets = []
        all_concept_masks = []
        all_action_scores = []
        all_action_labels = []
        all_sample_indices = []

        with torch.no_grad():
            for batch in tqdm(self.data_loader["test"], desc="Eval"):
                data, index_t, action_label, concept_target, concept_mask, sample_idx = batch

                data           = data.float().cuda(self.output_device)
                index_t        = index_t.float().cuda(self.output_device)
                concept_target = concept_target.float().cuda(self.output_device)
                concept_mask   = concept_mask.bool().cuda(self.output_device)

                concept_logits, action_logits = model(data, index_t)
                concept_probs = torch.sigmoid(concept_logits)

                all_concept_probs.append(concept_probs.cpu().numpy())
                all_concept_targets.append(concept_target.cpu().numpy())
                all_concept_masks.append(concept_mask.cpu().numpy())
                if action_logits is not None:
                    all_action_scores.append(action_logits.cpu().numpy())
                all_action_labels.append(action_label.numpy())
                all_sample_indices.append(sample_idx.numpy())

        all_concept_probs   = np.concatenate(all_concept_probs)
        all_concept_targets = np.concatenate(all_concept_targets)
        all_concept_masks   = np.concatenate(all_concept_masks)
        all_action_labels   = np.concatenate(all_action_labels)
        all_sample_indices  = np.concatenate(all_sample_indices)

        # Action accuracy
        action_acc = 0.0
        if all_action_scores:
            all_action_scores = np.concatenate(all_action_scores)
            pred_labels = all_action_scores.argmax(1)
            action_acc = (pred_labels == all_action_labels).mean()
            top5_acc = self.data_loader["test"].dataset.top_k(all_action_scores, 5)
            self.log(f"  [Eval] Top-1: {action_acc * 100:.2f}% | Top-5: {top5_acc * 100:.2f}%")

        # Concept metrics
        cmets = concept_metrics(all_concept_probs, all_concept_targets)
        self.log(
            f"  [Concept] macro-F1={cmets['macro_f1']:.4f} micro-F1={cmets['micro_f1']:.4f} "
            f"P={cmets['precision']:.4f} R={cmets['recall']:.4f} "
            f"coverage={cmets['coverage']:.4f} entropy={cmets['entropy']:.4f}"
        )

        if save_preds:
            out = {
                "concept_probs":   all_concept_probs,
                "concept_targets": all_concept_targets,
                "concept_masks":   all_concept_masks,
                "action_scores":   all_action_scores if all_action_scores else np.array([]),
                "action_labels":   all_action_labels,
                "sample_indices":  all_sample_indices,
                "concept_metrics": cmets,
                "action_top1":     float(action_acc),
            }
            save_path = os.path.join(self.arg.work_dir, f"eval_epoch{epoch + 1}.npz")
            np.savez_compressed(save_path, **out)
            with open(save_path.replace(".npz", "_metrics.json"), "w") as f:
                json.dump({**cmets, "action_top1": float(action_acc)}, f, indent=2)
            self.log(f"  Saved eval predictions to {save_path}")

        return float(action_acc)

    def save_checkpoint(self, epoch: int, tag: str = ""):
        ckpt_dir = os.path.join(self.arg.work_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        model = self.model.module if hasattr(self.model, "module") else self.model
        state = OrderedDict([(k, v.cpu()) for k, v in model.state_dict().items()])

        name = f"epoch{epoch + 1}{tag}.pt"
        path = os.path.join(ckpt_dir, name)
        torch.save(state, path)
        self.log(f"  Checkpoint saved: {path}")
        return path

    # ---- Main loop ----

    def start(self):
        arg = self.arg

        if arg.phase == "build_template":
            self._phase_build_template()
            return

        if arg.phase in ("test_concept",):
            self.eval(epoch=0, save_preds=True)
            return

        # --- train_concept ---
        self.log(f"Starting training for {arg.num_epoch} epochs")
        best_ckpt = None

        for epoch in range(arg.start_epoch, arg.num_epoch):
            self.train(epoch)

            if (epoch + 1) % arg.eval_interval == 0 or epoch == arg.num_epoch - 1:
                save_preds = (epoch == arg.num_epoch - 1) or arg.save_score
                acc = self.eval(epoch, save_preds=save_preds)

                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_acc_epoch = epoch + 1
                    self.no_improve_count = 0
                    best_ckpt = self.save_checkpoint(epoch, tag="_best")
                else:
                    self.no_improve_count += 1

            if (epoch + 1) >= arg.save_epoch:
                self.save_checkpoint(epoch)

            if (
                arg.early_stop_patience > 0
                and self.no_improve_count >= arg.early_stop_patience
            ):
                self.log(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {arg.early_stop_patience} evals)"
                )
                break

        self.log(
            f"\nTraining done. Best Top-1: {self.best_acc * 100:.2f}% "
            f"at epoch {self.best_acc_epoch}"
        )
        if best_ckpt:
            self.log(f"Best checkpoint: {best_ckpt}")

        # Final eval with best checkpoint
        if best_ckpt is not None and os.path.isfile(best_ckpt):
            self._load_weights(best_ckpt)
            self.log("=== Final evaluation with best checkpoint ===")
            self.eval(epoch=self.best_acc_epoch - 1, save_preds=True)

    def _phase_build_template(self):
        arg = self.arg
        out_dir = os.path.join(arg.work_dir, "pseudo_labels")
        for split in ("train", "test"):
            feeder_args = arg.train_feeder_args if split == "train" else arg.test_feeder_args
            data_path = feeder_args.get("data_path")
            if data_path:
                build_template_labels(
                    data_path=data_path,
                    split=split,
                    num_classes=arg.num_classes,
                    out_dir=out_dir,
                    protocol=arg.protocol,
                    csv_path=arg.concept_csv,
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = get_parser()
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    proc = ConceptProcessor(arg)
    proc.start()


if __name__ == "__main__":
    main()
