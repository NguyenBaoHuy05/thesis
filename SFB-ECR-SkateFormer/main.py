"""Main Execution and Training Script for SFB-ECR-SkateFormer.

Supports:
- End-to-end multi-task training with dynamic lambda feedback annealing
- Automatic checkpoint management (checkpoint_latest.pt, checkpoint_best.pt) and Auto-Resume
- Multi-metric evaluation (Action Top-1/5 Acc, Concept Macro/Micro F1, mAP, KG Violation Rate)
- Occlusion Robustness Evaluation (Joint group & temporal masking)
- Counterfactual Concept Intervention Analysis
"""

import argparse
import inspect
import os
import sys
import time
import json
import csv
import glob
import random
import pickle
import traceback
from collections import OrderedDict
from pathlib import Path

# Configure utf-8 stdout
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler

# Add local torchlight and project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_LOCAL_TORCHLIGHT = _PROJECT_ROOT / 'torchlight'
if _LOCAL_TORCHLIGHT.is_dir() and str(_LOCAL_TORCHLIGHT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_TORCHLIGHT))

try:
    from torchlight import DictAction, import_class, str2bool
except ImportError:
    def import_class(name):
        mod_str, _sep, class_str = name.rpartition('.')
        __import__(mod_str)
        return getattr(sys.modules[mod_str], class_str)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ('yes', 'true', 't', 'y', '1')

    class DictAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            input_dict = eval(f'dict({values})')
            output_dict = getattr(namespace, self.dest, {}) or {}
            output_dict.update(input_dict)
            setattr(namespace, self.dest, output_dict)

from feeders.occlusion import mask_joint_groups, mask_temporal_frames, OcclusionDatasetWrapper, PARTITIONED_GROUPS


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class Processor:
    """Processor handles Training, Validation, Checkpointing and Diagnostic Evaluation."""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and arg.device else 'cpu')
        
        # IO & Logging
        os.makedirs(self.arg.work_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.arg.work_dir, 'log.txt'), 'a', encoding='utf-8')
        self.tb_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'tb_logs')) if arg.phase == 'train' else None

        self.print_log(f"⚡ Device: {self.device}")
        self.print_log(f"📂 Work Directory: {self.arg.work_dir}")

        self.load_model()
        self.load_optimizer()
        self.load_data()

        self.best_acc = 0.0
        self.best_epoch = 0
        self.start_epoch = 0

        # Auto-Resume if checkpoint_latest.pt exists
        if self.arg.auto_resume:
            self.auto_resume_checkpoint()

    def print_log(self, str_content):
        t = time.asctime(time.localtime(time.time()))
        msg = f"[{t}] {str_content}"
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def save_arg(self):
        os.makedirs(self.arg.work_dir, exist_ok=True)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(vars(self.arg), f, default_flow_style=False)

    def load_model(self):
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).to(self.device)
        self.print_log(f"✅ Loaded Model: {self.arg.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f"📊 Model Parameters: Total = {total_params:,} | Trainable = {trainable_params:,}")

        if self.arg.weights:
            self.print_log(f"📥 Loading weights from {self.arg.weights}...")
            state = torch.load(self.arg.weights, map_location='cpu')
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
            clean_state = OrderedDict([[k.replace('module.', ''), v] for k, v in state.items()])
            missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
            self.print_log(f"Weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )

        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.arg.num_epoch,
            lr_min=self.arg.min_lr,
            warmup_lr_init=self.arg.warmup_lr,
            warmup_t=self.arg.warm_up_epoch,
            warmup_prefix=self.arg.warmup_prefix,
            cycle_limit=1,
            t_in_epochs=True
        )

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if self.arg.phase == 'train':
            self.train_dataset = Feeder(**self.arg.train_feeder_args)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=True,
                drop_last=True
            )
            self.print_log(f"📦 Train Dataset: {len(self.train_dataset)} samples ({len(self.train_loader)} batches)")

        self.test_dataset = Feeder(**self.arg.test_feeder_args)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            pin_memory=True
        )
        self.print_log(f"📦 Test Dataset: {len(self.test_dataset)} samples ({len(self.test_loader)} batches)")

    def auto_resume_checkpoint(self):
        latest_path = os.path.join(self.arg.work_dir, 'checkpoint_latest.pt')
        if os.path.exists(latest_path):
            self.print_log(f"🔄 Found latest checkpoint: {latest_path}. Auto-resuming...")
            cp = torch.load(latest_path, map_location=self.device)
            self.model.load_state_dict(cp['model_state'])
            self.optimizer.load_state_dict(cp['optimizer_state'])
            if 'lr_scheduler_state' in cp and cp['lr_scheduler_state'] is not None:
                try:
                    self.lr_scheduler.load_state_dict(cp['lr_scheduler_state'])
                except Exception:
                    pass
            self.start_epoch = cp.get('epoch', 0) + 1
            self.best_acc = cp.get('best_acc', 0.0)
            self.best_epoch = cp.get('best_epoch', 0)
            self.print_log(f"✅ Resumed from Epoch {self.start_epoch} (Best Top-1: {self.best_acc:.2f}% at Epoch {self.best_epoch})")

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict() if hasattr(self.lr_scheduler, 'state_dict') else None,
            'config': vars(self.arg)
        }
        latest_file = os.path.join(self.arg.work_dir, 'checkpoint_latest.pt')
        torch.save(state, latest_file)

        if is_best:
            best_file = os.path.join(self.arg.work_dir, 'checkpoint_best.pt')
            torch.save(state, best_file)
            self.print_log(f"⭐ Saved new BEST checkpoint to {best_file}")

    def adjust_feedback_lambda(self, epoch):
        """Anneal lambda_fb from 0 up to target over warmup_epochs."""
        target_lambda = self.arg.model_args.get('feedback_lambda', 0.1)
        warmup_epochs = 10
        if epoch < warmup_epochs:
            current_lambda = target_lambda * float(epoch + 1) / float(warmup_epochs)
        else:
            current_lambda = target_lambda
        self.model.set_feedback_lambda(current_lambda)
        return current_lambda

    def train_epoch(self, epoch):
        self.model.train()
        self.lr_scheduler.step(epoch)
        current_lr = self.optimizer.param_groups[0]['lr']
        current_lambda_fb = self.adjust_feedback_lambda(epoch)

        self.print_log(f"\n--- Epoch {epoch}/{self.arg.num_epoch} | LR: {current_lr:.6f} | Lambda_fb: {current_lambda_fb:.4f} ---")

        total_loss_accum = 0.0
        action_loss_accum = 0.0
        concept_loss_accum = 0.0
        correct_action = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
        for batch_idx, batch in enumerate(pbar):
            data = batch[0].float().to(self.device)
            index_t = batch[1].float().to(self.device) if len(batch) > 1 else None
            labels = batch[2].long().to(self.device)
            concepts = batch[4].float().to(self.device) if len(batch) >= 5 else None

            self.optimizer.zero_grad()
            outputs = self.model(data, index_t=index_t, return_aux=True)
            action_logits, aux = outputs

            loss, loss_dict = self.model.compute_loss(outputs, labels, concepts)
            loss.backward()

            if self.arg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.arg.grad_max)

            self.optimizer.step()

            # Stats
            batch_size = data.size(0)
            total_samples += batch_size
            total_loss_accum += loss.item() * batch_size
            action_loss_accum += loss_dict["action_loss"] * batch_size
            concept_loss_accum += loss_dict["late_concept_loss"] * batch_size

            preds = action_logits.argmax(dim=-1)
            correct_action += (preds == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct_action / total_samples:.2f}%",
                'c_loss': f"{loss_dict['late_concept_loss']:.3f}"
            })

        train_acc = 100.0 * correct_action / total_samples
        avg_loss = total_loss_accum / total_samples
        self.print_log(f"Train Epoch {epoch} Result: Loss = {avg_loss:.4f} | Action Acc = {train_acc:.2f}%")

        if self.tb_writer:
            self.tb_writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
            self.tb_writer.add_scalar('Train/Action_Acc', train_acc, epoch)
            self.tb_writer.add_scalar('Train/LR', current_lr, epoch)
            self.tb_writer.add_scalar('Train/Lambda_FB', current_lambda_fb, epoch)

    def eval(self, epoch=0, save_score=False):
        self.model.eval()
        self.print_log(f"🔍 Evaluating on Test Set...")

        total_samples = 0
        correct_top1 = 0
        correct_top5 = 0
        
        all_preds = []
        all_targets = []
        all_concept_preds = []
        all_concept_targets = []
        all_early_concept_preds = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Eval"):
                data = batch[0].float().to(self.device)
                index_t = batch[1].float().to(self.device) if len(batch) > 1 else None
                labels = batch[2].long().to(self.device)
                concepts = batch[4].float().to(self.device) if len(batch) >= 5 else None

                outputs = self.model(data, index_t=index_t, return_aux=True)
                action_logits, aux = outputs

                batch_size = data.size(0)
                total_samples += batch_size

                # Top-1 & Top-5 Action Acc
                _, top1_idx = action_logits.topk(1, dim=-1)
                _, top5_idx = action_logits.topk(min(5, action_logits.size(-1)), dim=-1)

                correct_top1 += (top1_idx.squeeze(-1) == labels).sum().item()
                correct_top5 += (top5_idx == labels.unsqueeze(-1)).any(dim=-1).sum().item()

                all_preds.extend(top1_idx.squeeze(-1).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                if concepts is not None and "concept_logits" in aux and aux["concept_logits"] is not None:
                    c_preds = torch.sigmoid(aux["concept_logits"]).cpu().numpy()
                    all_concept_preds.append(c_preds)
                    all_concept_targets.append(concepts.cpu().numpy())
                if concepts is not None and "early_concept_logits" in aux and aux["early_concept_logits"] is not None:
                    c_early_preds = torch.sigmoid(aux["early_concept_logits"]).cpu().numpy()
                    all_early_concept_preds.append(c_early_preds)

        top1_acc = 100.0 * correct_top1 / total_samples
        top5_acc = 100.0 * correct_top5 / total_samples

        self.print_log(f"🎯 Test Result (Epoch {epoch}): Top-1 Acc = {top1_acc:.2f}% | Top-5 Acc = {top5_acc:.2f}%")

        # Concept Metrics
        if all_concept_preds and all_concept_targets:
            c_preds_mat = np.vstack(all_concept_preds)
            c_targets_mat = np.vstack(all_concept_targets)
            bin_preds = (c_preds_mat >= 0.5).astype(np.float32)
            macro_f1 = f1_score(c_targets_mat, bin_preds, average='macro', zero_division=0) * 100.0
            micro_f1 = f1_score(c_targets_mat, bin_preds, average='micro', zero_division=0) * 100.0
            self.print_log(f"🧠 Final Concept F1: Macro = {macro_f1:.2f}% | Micro = {micro_f1:.2f}%")

            if all_early_concept_preds:
                c_early_mat = np.vstack(all_early_concept_preds)
                bin_early = (c_early_mat >= 0.5).astype(np.float32)
                early_macro_f1 = f1_score(c_targets_mat, bin_early, average='macro', zero_division=0) * 100.0
                self.print_log(f"🌱 Early Concept F1: Macro = {early_macro_f1:.2f}%")

            if self.tb_writer:
                self.tb_writer.add_scalar('Eval/Concept_Macro_F1', macro_f1, epoch)

        if self.tb_writer:
            self.tb_writer.add_scalar('Eval/Top1_Acc', top1_acc, epoch)
            self.tb_writer.add_scalar('Eval/Top5_Acc', top5_acc, epoch)

        is_best = top1_acc > self.best_acc
        if is_best:
            self.best_acc = top1_acc
            self.best_epoch = epoch
            self.print_log(f"🏆 NEW BEST ACCURACY: {self.best_acc:.2f}% at Epoch {self.best_epoch}")

        return top1_acc, top5_acc

    def eval_robustness_occlusion(self):
        """Evaluate model performance across various joint occlusions and temporal masking."""
        self.model.eval()
        self.print_log("\n" + "=" * 60)
        self.print_log("🛡️ STARTING OCCLUSION & ROBUSTNESS EVALUATION")
        self.print_log("=" * 60)

        groups_to_test = ['none', 'arms', 'legs', 'trunk', 'head_torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg']
        results = {}

        for group in groups_to_test:
            correct = 0
            total = 0
            for batch in tqdm(self.test_loader, desc=f"Occlusion: {group}"):
                data = batch[0].float()
                index_t = batch[1].float().to(self.device) if len(batch) > 1 else None
                labels = batch[2].long().to(self.device)

                masked_data = mask_joint_groups(data, group_name=group).to(self.device)
                with torch.no_grad():
                    action_logits, _ = self.model(masked_data, index_t=index_t, return_aux=True)
                    preds = action_logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += data.size(0)

            acc = 100.0 * correct / total
            results[f"Joint_{group}"] = acc
            self.print_log(f"  - Joint Occlusion [{group.upper()}]: Top-1 Acc = {acc:.2f}%")

        # Temporal Masking
        temporal_ratios = [0.2, 0.4, 0.6]
        for ratio in temporal_ratios:
            correct = 0
            total = 0
            for batch in tqdm(self.test_loader, desc=f"Temporal Mask: {int(ratio*100)}%"):
                data = batch[0].float()
                index_t = batch[1].float().to(self.device) if len(batch) > 1 else None
                labels = batch[2].long().to(self.device)

                masked_data = mask_temporal_frames(data, mask_ratio=ratio, mode='continuous').to(self.device)
                with torch.no_grad():
                    action_logits, _ = self.model(masked_data, index_t=index_t, return_aux=True)
                    preds = action_logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += data.size(0)

            acc = 100.0 * correct / total
            results[f"Temporal_{int(ratio*100)}%"] = acc
            self.print_log(f"  - Temporal Masking [{int(ratio*100)}%]: Top-1 Acc = {acc:.2f}%")

        report_file = os.path.join(self.arg.work_dir, "robustness_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        self.print_log(f"📁 Robustness report saved to {report_file}")
        return results

    def eval_counterfactual_intervention(self, num_samples=100):
        """Intervene on concept predictions (c_i -> 0) and measure action evidence drops."""
        self.model.eval()
        self.print_log("\n" + "=" * 60)
        self.print_log("🔬 STARTING COUNTERFACTUAL CONCEPT INTERVENTION ANALYSIS")
        self.print_log("=" * 60)

        processed = 0
        drops = []

        with torch.no_grad():
            for batch in self.test_loader:
                data = batch[0].float().to(self.device)
                index_t = batch[1].float().to(self.device) if len(batch) > 1 else None
                labels = batch[2].long().to(self.device)
                concepts = batch[4].float().to(self.device) if len(batch) >= 5 else None

                if concepts is None:
                    continue

                action_logits, aux = self.model(data, index_t=index_t, return_aux=True)
                orig_probs = F.softmax(action_logits, dim=-1)
                concept_logits = aux["concept_logits"]

                # For each sample, identify its top active ground truth concept and suppress it
                for i in range(data.size(0)):
                    target_label = labels[i].item()
                    gt_concept_indices = torch.nonzero(concepts[i] > 0.5).squeeze(-1)
                    if len(gt_concept_indices) == 0:
                        continue

                    # Suppress the first active concept
                    suppressed_logits = concept_logits[i:i+1].clone()
                    target_c_idx = gt_concept_indices[0].item()
                    suppressed_logits[0, target_c_idx] = -10.0  # force sigmoid to ~0

                    intervened_logits = self.model.reasoner(suppressed_logits)
                    intervened_probs = F.softmax(intervened_logits, dim=-1)

                    drop = orig_probs[i, target_label].item() - intervened_probs[0, target_label].item()
                    drops.append(drop)
                    processed += 1
                    if processed >= num_samples:
                        break
                if processed >= num_samples:
                    break

        avg_drop = float(np.mean(drops)) if drops else 0.0
        self.print_log(f"📊 Evaluated {len(drops)} interventions. Average Action Probability Drop: {avg_drop:.4f}")
        return avg_drop

    def start(self):
        if self.arg.phase == 'train':
            self.print_log("🚀 Starting Training Loop...")
            for epoch in range(self.start_epoch, self.arg.num_epoch):
                self.train_epoch(epoch)
                top1, top5 = self.eval(epoch=epoch)
                self.save_checkpoint(epoch, is_best=(top1 == self.best_acc))
            self.print_log(f"\n🎉 Training Finished! Best Top-1 Acc: {self.best_acc:.2f}% (Epoch {self.best_epoch})")

        elif self.arg.phase == 'test':
            self.eval(epoch=0)

        elif self.arg.phase == 'test_occlusion':
            self.eval_robustness_occlusion()

        elif self.arg.phase == 'test_counterfactual':
            self.eval_counterfactual_intervention()


def get_parser():
    parser = argparse.ArgumentParser(description='SFB-ECR-SkateFormer Runner')
    parser.add_argument('--config', default='./config/train/ntu_cs/SFB_ECR_SkateFormer_j.yaml', help='path to config file')
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'test_occlusion', 'test_counterfactual'])
    parser.add_argument('--work_dir', default='./work_dir/ntu/cs/SFB_ECR_SkateFormer_j/')
    parser.add_argument('--weights', default=None, help='path to model checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True, help='auto resume from checkpoint_latest.pt')
    parser.add_argument('--feeder', default='feeders.feeder_ntu.Feeder')
    parser.add_argument('--train_feeder_args', action=DictAction, default={})
    parser.add_argument('--test_feeder_args', action=DictAction, default={})
    parser.add_argument('--model', default='model.reason_skateformer.SFB_ECR_SkateFormer')
    parser.add_argument('--model_args', action=DictAction, default={})
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.000001)
    parser.add_argument('--warmup_lr', type=float, default=0.000001)
    parser.add_argument('--warm_up_epoch', type=int, default=2)
    parser.add_argument('--warmup_prefix', type=str2bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--grad_clip', type=str2bool, default=True)
    parser.add_argument('--grad_max', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--device', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=1)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'Unknown Key: {k}')
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
