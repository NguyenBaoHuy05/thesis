"""LCR-SkateFormer Training and Evaluation Processor.

Supports:
- End-to-end training of SkateFormer with IterativeKGReasoner
- Multi-loop deep supervision with auxiliary action losses
- Early perception anchoring and refined concept supervision
- Knowledge Graph consistency, stability, and rule regularization
- Step-by-step reasoning trace logging and counterfactual audit
"""

from __future__ import print_function

import argparse
import csv
import glob
import inspect
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

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

# LR Scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler


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
            f"Class {class_str} cannot be found ({traceback.format_exception(*sys.exc_info())})"
        )


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported boolean value encountered.")


def compute_confusion_matrix(y_true, y_pred, num_classes=None):
    """Numpy-based confusion matrix with zero sklearn dependency."""
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


def get_parser():
    parser = argparse.ArgumentParser(
        description="LCR-SkateFormer: Looping Concept Reasoning SkateFormer"
    )
    parser.add_argument("--work-dir", default="./work_dir", help="folder for storing results")
    parser.add_argument("--model_saved_name", default="")
    parser.add_argument("--config", default="./config", help="path to configuration file")

    # processor
    parser.add_argument("--phase", default="train", help="must be train or test")
    parser.add_argument("--save-score", type=str2bool, default=False, help="store classification score")
    parser.add_argument("--save-trace", type=str2bool, default=False, help="store step reasoning traces")

    # visualize and debug
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log-interval", type=int, default=100, help="interval for printing messages")
    parser.add_argument("--save-interval", type=int, default=1, help="interval for storing models")
    parser.add_argument("--save-epoch", type=int, default=30, help="start epoch to save model")
    parser.add_argument("--eval-interval", type=int, default=5, help="interval for evaluating models")
    parser.add_argument("--print-log", type=str2bool, default=True, help="print logging")
    parser.add_argument("--show-topk", type=int, default=[1, 5], nargs="+", help="Top K accuracies to show")

    # feeder
    parser.add_argument("--feeder", default="feeders.feeder_ntu.Feeder", help="feeder class")
    parser.add_argument("--num-worker", type=int, default=4, help="workers for data loader")
    parser.add_argument("--train-feeder-args", action=DictAction, default=dict(), help="training feeder args")
    parser.add_argument("--test-feeder-args", action=DictAction, default=dict(), help="test feeder args")

    # model
    parser.add_argument("--model", default="model.lcr_skateformer.LCR_SkateFormer", help="model class")
    parser.add_argument("--model-args", action=DictAction, default=dict(), help="model args")
    parser.add_argument("--weights", default=None, help="weights for model initialization")
    parser.add_argument("--ignore-weights", type=str, default=[], nargs="+", help="weights to ignore")

    # optimization
    parser.add_argument("--base-lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--min-lr", type=float, default=0.000001, help="minimum learning rate")
    parser.add_argument("--warmup-lr", type=float, default=0.000001, help="warmup learning rate")
    parser.add_argument("--warmup_prefix", type=bool, default=False)
    parser.add_argument("--warm_up_epoch", type=int, default=10)
    parser.add_argument("--grad-clip", type=bool, default=True)
    parser.add_argument("--grad-max", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=[0], nargs="+", help="GPU device indices")
    parser.add_argument("--optimizer", default="AdamW", help="optimizer type")
    parser.add_argument("--lr-scheduler", default="cosine", help="scheduler type")
    parser.add_argument("--nesterov", type=str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=32, help="test batch size")
    parser.add_argument("--start-epoch", type=int, default=0, help="start training epoch")
    parser.add_argument("--num-epoch", type=int, default=150, help="total training epochs")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--loss-type", type=str, default="LSCE", help="CE or LSCE")
    return parser


class Processor:
    """Main training, validation, and explanation processor for LCR-SkateFormer."""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == "train":
            if not arg.train_feeder_args.get("debug", False):
                arg.model_saved_name = os.path.join(arg.work_dir, "runs")
                if os.path.isdir(arg.model_saved_name):
                    print(f"Log dir {arg.model_saved_name} already exists. Appending runs.")
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, "train"), "train")
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, "val"), "val")
            else:
                self.train_writer = self.val_writer = SummaryWriter(
                    os.path.join(arg.work_dir, "debug"), "debug"
                )

        self.global_step = 0
        self.checkpoint = None
        self.best_acc = 0.0
        self.best_acc_epoch = 0

        # Device setup
        self.use_cuda = torch.cuda.is_available() and len(self.arg.device) > 0 and self.arg.device[0] >= 0
        self.output_device = self.arg.device[0] if self.use_cuda else "cpu"

        self.load_data()

        # Automatic alignment of concept names and action prototypes from dataset CSV
        concept_dataset = self.data_loader.get("train", self.data_loader.get("test")).dataset
        if getattr(concept_dataset, "concept_names", None) is not None:
            self.arg.model_args["concept_names"] = list(concept_dataset.concept_names)
        if getattr(concept_dataset, "concept_matrix", None) is not None:
            self.arg.model_args["action_prototypes"] = concept_dataset.concept_matrix.tolist()

        self.load_model()
        if self.use_cuda:
            self.model = self.model.cuda(self.output_device)
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model, device_ids=self.arg.device, output_device=self.output_device
                )

        if self.arg.phase == "train":
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader["train"]))

        # Loss function
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

            # Filter ignored weights
            for w in self.arg.ignore_weights:
                for k in list(raw_weights.keys()):
                    if w in k:
                        self.print_log(f"Ignoring weight: {k}")
                        del raw_weights[k]

            # Intelligent Remap: Adapt ECR-SkateFormer keys to LCR-SkateFormer
            model_state = self.model.state_dict()
            mapped_weights = OrderedDict()
            for k, v in raw_weights.items():
                if k in model_state and model_state[k].shape == v.shape:
                    mapped_weights[k] = v
                # Remap ECR reasoner -> LCR reasoner.base_reasoner
                elif k.startswith("reasoner.") and not k.startswith("reasoner.base_reasoner."):
                    new_k = k.replace("reasoner.", "reasoner.base_reasoner.", 1)
                    if new_k in model_state and model_state[new_k].shape == v.shape:
                        mapped_weights[new_k] = v
                    else:
                        mapped_weights[k] = v
                else:
                    mapped_weights[k] = v

            missing, unexpected = self.model.load_state_dict(mapped_weights, strict=False)
            self.print_log(
                f"Successfully loaded {len(mapped_weights)} tensors. "
                f"Missing (new/uninitialized parameters): {len(missing)}, Unexpected: {len(unexpected)}"
            )

    def load_optimizer(self):
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

    def load_scheduler(self, n_iter_per_epoch):
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)

        if self.arg.lr_scheduler == "cosine":
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
        else:
            raise ValueError(f"Unsupported scheduler: {self.arg.lr_scheduler}")

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
        loader = self.data_loader["train"]

        loss_values, acc_values, aux_loss_values = [], [], []
        process = tqdm(loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(process):
            if len(batch) == 5:
                data, index_t, label, index, concept_targets = batch
            else:
                data, index_t, label, index = batch
                concept_targets = None

            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1

            if self.use_cuda:
                data = data.float().cuda(self.output_device)
                index_t = index_t.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                if concept_targets is not None:
                    concept_targets = concept_targets.float().cuda(self.output_device)

            # Forward pass
            output = self.model(
                data,
                index_t=index_t,
                concept_targets=concept_targets,
                action_labels=label,
                return_trace=False,
            )

            if isinstance(output, dict):
                action_logits = output["action_logits"]
                main_loss = self.loss(action_logits, label)
                aux_loss = output.get("aux_loss", 0.0)
                total_loss = main_loss + aux_loss
            else:
                action_logits = output
                total_loss = self.loss(action_logits, label)
                aux_loss = 0.0

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            loss_values.append(total_loss.item())
            if isinstance(aux_loss, torch.Tensor):
                aux_loss_values.append(aux_loss.item())

            _, pred_label = torch.max(action_logits.data, 1)
            acc = torch.mean((pred_label == label.data).float()).item()
            acc_values.append(acc)

            self.train_writer.add_scalar("acc", acc, self.global_step)
            self.train_writer.add_scalar("loss", total_loss.item(), self.global_step)

        mean_loss = np.mean(loss_values)
        mean_acc = np.mean(acc_values) * 100.0
        cur_lr = self.optimizer.param_groups[0]["lr"]

        self.print_log(f"\tMean Train Loss: {mean_loss:.4f} | Mean Train Acc: {mean_acc:.2f}% | LR: {cur_lr:.6f}")

        # Save latest checkpoint
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split("module.")[-1], v.cpu()] for k, v in state_dict.items()])
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "model": weights,
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
            "best_acc_epoch": self.best_acc_epoch,
        }
        torch.save(checkpoint, os.path.join(self.arg.work_dir, "checkpoint_latest.pt"))

        if save_model:
            torch.save(
                weights,
                os.path.join(self.arg.work_dir, f"epoch_{epoch+1}_acc_{mean_acc:.2f}.pt"),
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
                    data = data.float().cuda(self.output_device)
                    index_t = index_t.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                output = self.model(data, index_t=index_t, return_trace=save_trace)
                if isinstance(output, dict):
                    action_logits = output["action_logits"]
                    if save_trace and batch_idx == 0:
                        trace_samples.append(output["reasoning_trace"])
                else:
                    action_logits = output

                loss = self.loss(action_logits, label)
                loss_values.append(loss.item())
                score_fragments.append(action_logits.cpu().numpy())
                _, preds = torch.max(action_logits, 1)
                pred_list.append(preds.cpu().numpy())

        scores = np.concatenate(score_fragments)
        labels = np.concatenate(label_list)
        preds = np.concatenate(pred_list)

        top1_acc = np.mean(preds == labels) * 100.0
        # Top-5 accuracy
        top5_preds = np.argsort(scores, axis=1)[:, -5:]
        top5_acc = np.mean([labels[i] in top5_preds[i] for i in range(len(labels))]) * 100.0

        if top1_acc > self.best_acc:
            self.best_acc = top1_acc
            self.best_acc_epoch = epoch + 1

        self.print_log(f"\tEval Loss: {np.mean(loss_values):.4f} | Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}% (Best: {self.best_acc:.2f}% @ Epoch {self.best_acc_epoch})")

        if self.arg.phase == "train":
            self.val_writer.add_scalar("top1_acc", top1_acc, self.global_step)
            self.val_writer.add_scalar("top5_acc", top5_acc, self.global_step)

        if save_score:
            with open(os.path.join(self.arg.work_dir, f"epoch{epoch+1}_score.pkl"), "wb") as f:
                pickle.dump(scores, f)

        if save_trace and len(trace_samples) > 0:
            with open(os.path.join(self.arg.work_dir, f"epoch{epoch+1}_trace.pkl"), "wb") as f:
                pickle.dump(trace_samples, f)

        # Confusion matrix & Per-class Accuracy
        cm = compute_confusion_matrix(labels, preds)
        diag = np.diag(cm)
        sums = np.sum(cm, axis=1)
        per_class_acc = np.divide(diag, sums, out=np.zeros_like(diag, dtype=float), where=sums != 0)
        with open(os.path.join(self.arg.work_dir, f"epoch{epoch+1}_per_class_acc.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(per_class_acc)
            writer.writerows(cm)

    def start(self):
        self.print_log(f"Parameters: {self.arg}")
        if self.arg.phase == "train":
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (epoch + 1) >= self.arg.save_epoch and ((epoch + 1) % self.arg.save_interval == 0)
                self.train(epoch, save_model=save_model)
                if (epoch + 1) % self.arg.eval_interval == 0:
                    self.eval(epoch, save_score=self.arg.save_score, save_trace=self.arg.save_trace)
        elif self.arg.phase == "test":
            self.eval(epoch=0, save_score=self.arg.save_score, save_trace=self.arg.save_trace)


if __name__ == "__main__":
    parser = get_parser()
    # Read config file if specified
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
