"""Train CQFormer. Resumes from <work_dir>/last.pt; the test split is only read at the very end.

    python -u train.py --config config/ntu60_xsub.yaml --data_path $D --work_dir $W
    python -u train.py --config config/ntu60_xsub.yaml --data_path $D --work_dir /content/smoke --smoke
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from feeders.feeder_ntu import Feeder
from model.cqformer import CQFormer
from model.joint_sets import load_concept_matrix


def get_parser():
    p = argparse.ArgumentParser(description="Train CQFormer")
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_epoch", type=int, default=None, help="override the config")
    p.add_argument("--batch_size", type=int, default=None, help="override the config")
    p.add_argument("--smoke", action="store_true", help="one short epoch: measure time and memory, then stop")
    p.add_argument("--device", default="cuda")
    return p


def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg["base_lr"])


def lr_at(step, total, warmup, cfg):
    if step < warmup:
        return cfg["warmup_lr"] + (cfg["base_lr"] - cfg["warmup_lr"]) * step / max(1, warmup)
    q = (step - warmup) / max(1, total - warmup)
    return cfg["min_lr"] + (cfg["base_lr"] - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * q))


def compute_loss(out, y, model, cfg):
    ce = F.cross_entropy(out["logits"], y, label_smoothing=cfg["label_smoothing"])
    named = F.cross_entropy(out["logits_named"], y, label_smoothing=cfg["label_smoothing"])
    concept = F.binary_cross_entropy_with_logits(out["concept_logits"], model.S[y])
    l1 = model.W.abs().mean() + model.motif.weight.abs().mean()
    total = ce + cfg["named_weight"] * named + cfg["concept_weight"] * concept + cfg["l1_weight"] * l1
    return total, {"ce": ce.item(), "named": named.item(), "concept": concept.item()}


@torch.no_grad()
def evaluate(model, loader, device, amp, return_scores=False):
    model.eval()
    hits, count, named_hits, scores, labels = 0, 0, 0, [], []
    for x, index_t, y, _ in loader:
        x, index_t, y = x.to(device), index_t.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x, index_t)
        hits += (out["logits"].argmax(1) == y).sum().item()
        named_hits += (out["logits_named"].argmax(1) == y).sum().item()
        count += y.numel()
        if return_scores:
            scores.append(out["logits"].float().cpu().numpy())
            labels.append(y.cpu().numpy())
    res = {"top1": hits / count, "top1_named": named_hits / count}
    return (res, np.concatenate(scores), np.concatenate(labels)) if return_scores else res


def save_atomic(state, path):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    arg = get_parser().parse_args()
    with open(arg.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["train"]
    if arg.num_epoch is not None:
        tcfg["num_epoch"] = arg.num_epoch
    if arg.batch_size is not None:
        tcfg["batch_size"] = arg.batch_size

    os.makedirs(arg.work_dir, exist_ok=True)
    log_path = os.path.join(arg.work_dir, "log.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(arg.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    fcfg = dict(cfg["feeder"])
    val_fraction = fcfg.pop("val_fraction")
    test_p = fcfg.pop("test_p_interval")
    train_set = Feeder(arg.data_path, split="train", **fcfg)
    val_set = train_set.holdout(val_fraction, seed=1, p_interval=test_p)
    log("clips: train {} val {} | bf16 {}".format(len(train_set), len(val_set), amp))

    S = load_concept_matrix(cfg["concepts_csv"], cfg["num_classes"])
    model = CQFormer(cfg["num_classes"], S, **cfg["model"]).to(device)
    log("parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    loader_args = dict(num_workers=tcfg["num_worker"], pin_memory=True, worker_init_fn=seed_worker,
                       persistent_workers=tcfg["num_worker"] > 0)
    train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, batch_size=tcfg["test_batch_size"], **loader_args)

    optimizer = build_optimizer(model, tcfg)
    total = tcfg["num_epoch"] * len(train_loader)
    warmup = tcfg["warm_up_epoch"] * len(train_loader)
    start_epoch, best_val = 0, 0.0
    last_path, best_path = os.path.join(arg.work_dir, "last.pt"), os.path.join(arg.work_dir, "best.pt")
    if os.path.exists(last_path) and not arg.smoke:
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best_val"]
        log("resumed at epoch {}, best val {:.4f}".format(start_epoch, best_val))

    for epoch in range(start_epoch, 1 if arg.smoke else tcfg["num_epoch"]):
        model.train()
        t0 = time.time()
        sums, hits, count = {}, 0, 0
        for it, (x, index_t, y, _) in enumerate(train_loader):
            if arg.smoke and it == tcfg["smoke_iters"]:
                break
            if arg.smoke and it == 5:   # the first iterations include worker start-up
                torch.cuda.reset_peak_memory_stats()
                t_smoke = time.time()
            step = epoch * len(train_loader) + it
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total, warmup, tcfg)
            x, index_t, y = x.to(device, non_blocking=True), index_t.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x, index_t)
                loss, parts = compute_loss(out, y, model, tcfg)
            if not torch.isfinite(loss):
                raise RuntimeError("loss is {} at epoch {} iter {}".format(loss.item(), epoch, it))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_max"])
            optimizer.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            hits += (out["logits"].argmax(1) == y).sum().item()
            count += y.numel()
            if not arg.smoke and (it + 1) % tcfg["log_every"] == 0:   # one live line, not kept in log.txt
                per_iter = (time.time() - t0) / (it + 1)
                print("  epoch {} iter {}/{} | ce {:.3f} | train {:.4f} | {:.2f} s/iter | {:.1f} min left in epoch".format(
                    epoch + 1, it + 1, len(train_loader), sums["ce"] / (it + 1), hits / count, per_iter,
                    per_iter * (len(train_loader) - it - 1) / 60), end="\r", flush=True)

        if arg.smoke:
            per_iter = (time.time() - t_smoke) / (tcfg["smoke_iters"] - 5)
            epoch_min = per_iter * len(train_loader) / 60
            log("smoke: {:.3f} s/iter | {:.1f} min/epoch | {:.1f} h per 100 epochs | peak {:.1f} GB | loss {:.3f}".format(
                per_iter, epoch_min, epoch_min * 100 / 60, torch.cuda.max_memory_allocated() / 2 ** 30, loss.item()))
            log("val: {}".format(evaluate(model, val_loader, device, amp)))
            return

        n = len(train_loader)
        val = evaluate(model, val_loader, device, amp)
        log("epoch {}/{} | lr {:.2e} | ce {:.3f} named {:.3f} concept {:.3f} | train {:.4f} | val {:.4f} named {:.4f} | {:.0f}s".format(
            epoch + 1, tcfg["num_epoch"], optimizer.param_groups[0]["lr"], sums["ce"] / n, sums["named"] / n,
            sums["concept"] / n, hits / count, val["top1"], val["top1_named"], time.time() - t0))
        if val["top1"] > best_val:
            best_val = val["top1"]
            save_atomic({"model": model.state_dict(), "epoch": epoch, "val": val}, best_path)
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "best_val": best_val}, last_path)

    # the one and only read of the test split
    test_set = Feeder(arg.data_path, split="test", p_interval=test_p, window_size=fcfg["window_size"],
                      thres=fcfg["thres"], uniform=fcfg["uniform"])
    test_loader = DataLoader(test_set, batch_size=tcfg["test_batch_size"], num_workers=tcfg["num_worker"], pin_memory=True)
    results = {"best_val": best_val}
    for name, path in (("best", best_path), ("last", last_path)):
        model.load_state_dict(torch.load(path, map_location=device)["model"])
        res, scores, labels = evaluate(model, test_loader, device, amp, return_scores=True)
        results[name] = res
        np.savez(os.path.join(arg.work_dir, "scores_{}.npz".format(name)), scores=scores, labels=labels)
        log("TEST {}: top-1 {:.4f} | named path only {:.4f}".format(name, res["top1"], res["top1_named"]))
    with open(os.path.join(arg.work_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
