"""Train the Causal Concept Program (pipeline v4, docs/architecture_v4.html).

Stages (run in order; each starts from the best VAL checkpoint of the previous one):
  1  concepts : thresholds tau_c / temperatures + latent head, MIL loss vs the CSV
                (HyperGCN frozen). Preceded by calibration (signal mean / std, tau init).
  2  program  : logic weights, CE on logits_logic + L1 sparsity + small concept loss
                (HyperGCN frozen, concept params at lr * 0.1).
  3  joint    : fused = logic + alpha * net, CE + gate penalty on alpha + sparsity +
                concept loss + counterfactual consistency L_cf (HyperGCN fine-tuned
                unless --freeze_backbone; L_cf needs a trainable backbone).

Split = v3 split: VAL = stratified --val_frac of the NTU train split (seed), TRAIN-FIT =
the rest; TEST is never touched here (scripts/eval_ccp.py).

Outputs in --out_dir: ccp_stage{1,2,3}.pt (best on VAL), history.json, rules_stage{2,3}.json.

Example (Colab):
    python scripts/train_ccp.py --hypergcn_root $H --checkpoint $CK --data_path $D \
        --out_dir $W/ccp --stages 1 2 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sip.backbone import build_feeder, load_hypergcn, split_train_rows  # noqa: E402
from sip.ccp import CCP, backbone_pass, calibrate, export_rules  # noqa: E402
from sip.concept_bank import build_bank  # noqa: E402
from sip.concept_map import GCCR_CONCEPT_CSV  # noqa: E402
from sip.counterfactual import editable_geo, generate  # noqa: E402
from sip.geo_torch import geo_signals  # noqa: E402


def get_parser():
    p = argparse.ArgumentParser(description="Train the v4 Causal Concept Program")
    p.add_argument("--hypergcn_root", required=True)
    p.add_argument("--checkpoint", required=True, help="trained HyperGCN joint checkpoint")
    p.add_argument("--data_path", required=True, help="NTU60_CS.npz")
    p.add_argument("--config", default=None)
    p.add_argument("--concept_csv", default=GCCR_CONCEPT_CSV)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_class", type=int, default=60)
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--init_ckpt", default=None, help="start from this ccp checkpoint (skips calibration)")
    p.add_argument("--no_extras", action="store_true", help="drop the x_* discovery concepts")
    p.add_argument("--no_before", action="store_true", help="drop the before(a, b) literals")
    p.add_argument("--train_crop", action="store_true",
                   help="random temporal crop p_interval [0.5, 1] on TRAIN-FIT (no rotation)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_worker", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--calib_batches", type=int, default=60)
    p.add_argument("--epochs1", type=int, default=6)
    p.add_argument("--epochs2", type=int, default=15)
    p.add_argument("--epochs3", type=int, default=5)
    p.add_argument("--lr1", type=float, default=3e-3)
    p.add_argument("--lr2", type=float, default=1e-2)
    p.add_argument("--lr3", type=float, default=1e-3)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--freeze_backbone", action="store_true", help="stage 3: keep HyperGCN frozen")
    p.add_argument("--lam_concept2", type=float, default=0.2, help="concept loss weight in stages 2-3")
    p.add_argument("--lam_sparse", type=float, default=1e-3)
    p.add_argument("--lam_gate", type=float, default=0.1)
    p.add_argument("--lam_cf", type=float, default=0.5)
    p.add_argument("--cf_frac", type=float, default=0.25, help="fraction of each batch given a counterfactual")
    p.add_argument("--cf_steps", type=int, default=10)
    p.add_argument("--cf_min_w", type=float, default=0.5, help="only literals with w+ >= this are edited")
    p.add_argument("--cf_margin", type=float, default=1.0)
    p.add_argument("--max_train_batches", type=int, default=None, help="debug: batches per epoch")
    p.add_argument("--max_val_batches", type=int, default=None)
    p.add_argument("--log_every", type=int, default=50)
    return p


def make_loaders(arg):
    train_set = build_feeder(arg.hypergcn_root, arg.data_path, "train", config=arg.config)
    labels = np.asarray(train_set.label).astype(np.int64)
    val_index, fit_rows = split_train_rows(labels, arg.val_frac, arg.seed)
    fit_set = train_set
    if arg.train_crop:
        fit_set = build_feeder(arg.hypergcn_root, arg.data_path, "train", config=arg.config)
        fit_set.p_interval = [0.5, 1]
    fit = torch.utils.data.Subset(fit_set, fit_rows.tolist())
    val = torch.utils.data.Subset(train_set, val_index.tolist())
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=arg.batch_size, shuffle=sh,
                                                    num_workers=arg.num_worker, drop_last=sh,
                                                    pin_memory=True)
    return mk(fit, True), mk(val, False)


def save(path, ccp, model, stage, epoch, metrics, arg, c6, backbone_trained):
    torch.save(dict(stage=stage, epoch=epoch, val=metrics, args=vars(arg),
                    ccp=ccp.state_dict(),
                    backbone=model.state_dict() if backbone_trained else None,
                    meta=dict(c6=c6, use_before=not arg.no_before, extras=not arg.no_extras,
                              concept_names=ccp.concept_names, num_class=arg.num_class,
                              concept_csv=arg.concept_csv)), path)


def cf_loss(model, ccp, x, y, out, arg, editable):
    """Counterfactual consistency: removing a concept that the program of the true class
    relies on must lower the network's logit of that class by >= cf_margin."""
    n = int(round(arg.cf_frac * x.shape[0]))
    zero = x.new_zeros(())
    if n == 0:
        return zero, {}
    gi = ccp.concepts.geo_index
    with torch.no_grad():
        wp = ccp.logic.weights()[0]
        ys = y[:n]
        cand = (wp[ys][:, gi] >= arg.cf_min_w) & editable[None] & (out["z"][:n, gi].detach() >= 0.6)
        rows = torch.nonzero(cand.any(1)).flatten()
    if rows.numel() == 0:
        return zero, dict(cf_n=0)
    gpos = torch.multinomial(cand[rows].float(), 1).flatten()
    gen = generate(ccp, x[rows], gpos, steps=arg.cf_steps)
    ok = gen["success"]
    if not ok.any():
        return zero, dict(cf_n=0, cf_success=0.0)
    rows, ys = rows[ok], y[rows][ok]
    logits_x = out["logits_net"][rows].detach()
    logits_cf, _ = backbone_pass(model, gen["x_cf"][ok])
    ar = torch.arange(len(rows), device=x.device)
    drop = logits_x[ar, ys] - logits_cf[ar, ys]
    return F.relu(arg.cf_margin - drop).mean(), dict(
        cf_n=int(len(rows)), cf_success=float(ok.float().mean()), cf_drop=float(drop.mean()),
        cf_edit=float(gen["edit"][ok].mean()))


def forward(model, ccp, x, backbone_grad):
    with torch.set_grad_enabled(backbone_grad and torch.is_grad_enabled()):
        logits_net, feat6 = backbone_pass(model, x)
    out = ccp(x, feat6, logits_net, geo=geo_signals(x))
    out["logits_net"] = logits_net
    return out


@torch.no_grad()
def evaluate(model, ccp, loader, arg, stage):
    ccp.eval()
    n = 0
    acc = dict(logic=0, fused=0, net=0)
    closs = 0.0
    for b, (data, label, _) in enumerate(loader):
        if arg.max_val_batches is not None and b >= arg.max_val_batches:
            break
        x = data.float().to(arg.device)
        y = label.long().to(arg.device)
        out = forward(model, ccp, x, False)
        for k, key in (("logic", "logic"), ("fused", "fused"), ("net", "logits_net")):
            acc[k] += int((out[key].argmax(1) == y).sum())
        closs += float(ccp.concept_loss(out["logit_z"], y)) * len(y)
        n += len(y)
    m = {k: v / max(n, 1) for k, v in acc.items()}
    m["concept_loss"] = closs / max(n, 1)
    m["alpha"] = float(ccp.alpha())
    return m


def run_stage(stage, model, ccp, fit_loader, val_loader, arg, c6, history):
    backbone_trained = stage == 3 and not arg.freeze_backbone
    for p in model.parameters():
        p.requires_grad_(backbone_trained)
    model.eval()   # BN statistics stay those of the trained HyperGCN

    concept_params = list(ccp.concepts.parameters())
    logic_params = list(ccp.logic.parameters())
    if stage == 1:
        groups = [dict(params=concept_params, lr=arg.lr1)]
        epochs, lr = arg.epochs1, arg.lr1
    elif stage == 2:
        groups = [dict(params=logic_params, lr=arg.lr2), dict(params=concept_params, lr=arg.lr2 * 0.1)]
        epochs, lr = arg.epochs2, arg.lr2
    else:
        groups = [dict(params=logic_params + [ccp.a_gate], lr=arg.lr3),
                  dict(params=concept_params, lr=arg.lr3 * 0.1)]
        if backbone_trained:
            groups.append(dict(params=list(model.parameters()), lr=arg.backbone_lr))
        epochs, lr = arg.epochs3, arg.lr3
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    n_batch = len(fit_loader) if arg.max_train_batches is None else min(len(fit_loader), arg.max_train_batches)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * n_batch))
    editable = editable_geo(ccp)
    if stage == 3 and not backbone_trained and arg.lam_cf > 0:
        print("[stage 3] backbone frozen -> L_cf has nothing to train and is disabled")

    key = {1: "concept_loss", 2: "logic", 3: "fused"}[stage]
    best, best_path = None, os.path.join(arg.out_dir, "ccp_stage{}.pt".format(stage))
    for ep in range(epochs):
        ccp.train()
        t0 = time.time()
        run = dict(loss=0.0, n=0, correct=0)
        for b, (data, label, _) in enumerate(fit_loader):
            if b >= n_batch:
                break
            x = data.float().to(arg.device)
            y = label.long().to(arg.device)
            out = forward(model, ccp, x, backbone_trained)
            lc = ccp.concept_loss(out["logit_z"], y)
            extra = {}
            if stage == 1:
                loss = lc
                pred = out["logic"]
            elif stage == 2:
                loss = F.cross_entropy(out["logic"], y) + arg.lam_sparse * ccp.logic.sparsity() \
                    + arg.lam_concept2 * lc
                pred = out["logic"]
            else:
                loss = F.cross_entropy(out["fused"], y) + arg.lam_gate * out["alpha"] \
                    + arg.lam_sparse * ccp.logic.sparsity() + arg.lam_concept2 * lc
                if backbone_trained and arg.lam_cf > 0:
                    lcf, extra = cf_loss(model, ccp, x, y, out, arg, editable)
                    loss = loss + arg.lam_cf * lcf
                pred = out["fused"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run["loss"] += float(loss) * len(y)
            run["n"] += len(y)
            run["correct"] += int((pred.argmax(1) == y).sum())
            if (b + 1) % arg.log_every == 0 or b + 1 == n_batch:
                print("[stage {} ep {} {}/{}] loss {:.4f} acc {:.4f} alpha {:.3f} {}  {:.0f}s".format(
                    stage, ep + 1, b + 1, n_batch, run["loss"] / run["n"], run["correct"] / run["n"],
                    float(out["alpha"]), " ".join("{}={:.3g}".format(k, v) for k, v in extra.items()),
                    time.time() - t0))
        m = evaluate(model, ccp, val_loader, arg, stage)
        m.update(stage=stage, epoch=ep + 1, train_loss=run["loss"] / max(run["n"], 1),
                 train_acc=run["correct"] / max(run["n"], 1))
        history.append(m)
        print("[stage {} ep {}] VAL logic={:.4f} fused={:.4f} net={:.4f} concept_loss={:.4f} alpha={:.3f}".format(
            stage, ep + 1, m["logic"], m["fused"], m["net"], m["concept_loss"], m["alpha"]))
        score = -m[key] if key == "concept_loss" else m[key]
        if best is None or score > best:
            best = score
            save(best_path, ccp, model, stage, ep + 1, m, arg, c6, backbone_trained)
            print("  saved best -> {}".format(best_path))
        with open(os.path.join(arg.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
    return best_path


def load_into(path, ccp, model, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck["meta"]["concept_names"] == ccp.concept_names, "concept bank changed since this checkpoint"
    ccp.load_state_dict(ck["ccp"])
    if ck.get("backbone") is not None:
        model.load_state_dict(ck["backbone"])
    print("[load] {} (stage {}, epoch {}, VAL {})".format(path, ck["stage"], ck["epoch"], ck["val"]))


def main():
    arg = get_parser().parse_args()
    os.makedirs(arg.out_dir, exist_ok=True)
    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)

    model = load_hypergcn(arg.hypergcn_root, arg.checkpoint, num_class=arg.num_class, device=arg.device)
    fit_loader, val_loader = make_loaders(arg)
    bank = build_bank(arg.concept_csv, num_class=arg.num_class, extras=not arg.no_extras)
    print("[bank]", bank.summary())

    with torch.no_grad():
        data, _, _ = next(iter(val_loader))
        _, feat6 = backbone_pass(model, data[:2].float().to(arg.device))
    c6 = int(feat6.shape[1])
    ccp = CCP(bank, c6, use_before=not arg.no_before).to(arg.device)
    print("[ccp] c6={} before-pairs={} params={}".format(
        c6, ccp.pairs.shape[0], sum(p.numel() for p in ccp.parameters())))

    if arg.init_ckpt:
        load_into(arg.init_ckpt, ccp, model, arg.device)
    else:
        calibrate(ccp, fit_loader, arg.device, n_batches=arg.calib_batches)

    history = []
    for stage in sorted(arg.stages):
        print("=" * 20, "stage", stage, "=" * 20)
        path = run_stage(stage, model, ccp, fit_loader, val_loader, arg, c6, history)
        load_into(path, ccp, model, arg.device)
        if stage >= 2:
            rules = export_rules(ccp)
            with open(os.path.join(arg.out_dir, "rules_stage{}.json".format(stage)), "w", encoding="utf-8") as f:
                json.dump(rules, f, indent=1, ensure_ascii=False)
            n_lit = [len(r["literals"]) for r in rules]
            print("[rules] stage {}: mean literals/class {:.1f}".format(stage, float(np.mean(n_lit))))
    print("[done]", arg.out_dir)


if __name__ == "__main__":
    main()
