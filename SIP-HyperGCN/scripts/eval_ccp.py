"""Evaluate a trained Causal Concept Program on TEST (pipeline v4).

Report sections (<out_dir>/report.md, plus json / csv files):
  1. accuracy: original HyperGCN, fine-tuned HyperGCN (if stage 3 trained it), logic only, fused
  2. concept quality: AUROC of every concept's clip-level truth value z_c against the CSV
     row of the true class (class-level labels; extras have no label)
  3. program: one rule per class (literals with weight >= --rule_min_w, thresholds in
     signal units), hard coverage / precision on TEST
  4. knowledge revision: CSV literals confirmed / removed, new literals (incl. x_* extras)
  5. counterfactual faithfulness: for each class and each of its strongest editable geo
     literals, TEST clips of that class where the concept holds are edited so that the
     concept is switched off (sip/counterfactual.py); the drop of the class logit is
     compared with a random edit of the same size on the same joints / frames
     (paired effect, bootstrap 95% CI). Measured on the ORIGINAL HyperGCN (does the
     trained network depend on the concept?) and on the fused model.
  6. robustness (--robust): Gaussian joint noise, joint dropout, frame dropout.

Example (Colab):
    python scripts/eval_ccp.py --hypergcn_root $H --checkpoint $CK --data_path $D \
        --ccp_ckpt $W/ccp/ccp_stage3.pt --out_dir $W/ccp_eval --robust
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sip.backbone import build_feeder, load_hypergcn  # noqa: E402
from sip.ccp import CCP, backbone_pass, export_rules  # noqa: E402
from sip.concept_bank import build_bank  # noqa: E402
from sip.counterfactual import editable_geo, generate, random_control  # noqa: E402
from sip.geo_torch import geo_signals  # noqa: E402


def get_parser():
    p = argparse.ArgumentParser(description="Evaluate the v4 Causal Concept Program on TEST")
    p.add_argument("--hypergcn_root", required=True)
    p.add_argument("--checkpoint", required=True, help="ORIGINAL HyperGCN joint checkpoint")
    p.add_argument("--data_path", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--ccp_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--concept_csv", default=None, help="default: the one stored in the checkpoint")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_worker", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--rule_min_w", type=float, default=0.3)
    p.add_argument("--cf_max_literals", type=int, default=4, help="per class, strongest editable literals")
    p.add_argument("--cf_per_literal", type=int, default=32, help="TEST clips per (class, literal)")
    p.add_argument("--cf_steps", type=int, default=30)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--robust", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


# --------------------------------------------------------------------------- helpers
def auroc(s, y):
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_ci(v, n_boot, rng):
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stack_clips(dataset, rows):
    xs = [torch.as_tensor(np.asarray(dataset[int(i)][0])) for i in rows]
    return torch.stack(xs).float()


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else "{:.{}f}".format(x, d)


class Models:
    def __init__(self, net0, net_ft, ccp):
        self.net0, self.net_ft, self.ccp = net0, net_ft, ccp

    @torch.no_grad()
    def run(self, x):
        l0, _ = backbone_pass(self.net0, x)
        lft, feat6 = backbone_pass(self.net_ft, x)
        out = self.ccp(x, feat6, lft, geo=geo_signals(x))
        return dict(net0=l0, net_ft=lft, logic=out["logic"], fused=out["fused"], z=out["z"],
                    before=out["before"])


# --------------------------------------------------------------------------- sections
def collect(models, loader, device):
    keys = ("net0", "net_ft", "logic", "fused")
    res = {k: [] for k in keys}
    res.update(z=[], before=[], y=[])
    t0 = time.time()
    for b, (data, label, _) in enumerate(loader):
        o = models.run(data.float().to(device))
        for k in keys:
            res[k].append(o[k].float().cpu().numpy())
        res["z"].append(o["z"].float().cpu().numpy())
        res["before"].append(o["before"].half().cpu().numpy())
        res["y"].append(np.asarray(label).astype(np.int64))
        if (b + 1) % 50 == 0:
            print("[collect] batch {}/{} {:.0f}s".format(b + 1, len(loader), time.time() - t0))
    return {k: np.concatenate(v) for k, v in res.items()}


def rule_truth(rule, z, before):
    sat = np.ones(len(z), dtype=bool)
    for lit in rule["literals"]:
        if lit["kind"] == "pos":
            sat &= z[:, lit["index"]] >= 0.5
        elif lit["kind"] == "neg":
            sat &= z[:, lit["index"]] < 0.5
        else:
            sat &= before[:, lit["index"]].astype(np.float32) >= 0.5
    return sat


def counterfactual_section(models, dataset, res, rules, ccp, arg, rng):
    device = arg.device
    gi = ccp.concepts.geo_index.tolist()
    gpos_of = {c: g for g, c in enumerate(gi)}
    editable = editable_geo(ccp).tolist()
    y, z = res["y"], res["z"]
    records = []
    t0 = time.time()
    for rule in rules:
        k = rule["class_idx"]
        lits = [l for l in rule["literals"] if l["kind"] == "pos" and l["index"] in gpos_of
                and editable[gpos_of[l["index"]]]][:arg.cf_max_literals]
        for lit in lits:
            c = lit["index"]
            rows = np.flatnonzero((y == k) & (z[:, c] >= 0.5))
            if len(rows) == 0:
                records.append(dict(class_idx=k, class_name=rule["class_name"], concept=lit["concept"],
                                    in_csv=lit["in_csv"], weight=lit["weight"], n=0))
                continue
            rows = rng.permutation(rows)[:arg.cf_per_literal]
            x = stack_clips(dataset, rows).to(device)
            gpos = torch.full((len(rows),), gpos_of[c], dtype=torch.long, device=device)
            gen = generate(ccp, x, gpos, steps=arg.cf_steps)
            x_rand = random_control(x, gen["delta"], gen["mask"])
            o, ocf, orn = models.run(x), models.run(gen["x_cf"]), models.run(x_rand)
            ok = gen["success"].cpu().numpy()
            rec = dict(class_idx=k, class_name=rule["class_name"], concept=lit["concept"],
                       in_csv=lit["in_csv"], weight=lit["weight"], n=int(len(rows)),
                       success=float(ok.mean()), edit=float(gen["edit"].mean()))
            for key in ("net0", "fused"):
                lx = o[key][:, k].cpu().numpy()
                d_cf = (lx - ocf[key][:, k].cpu().numpy())[ok]
                d_rn = (lx - orn[key][:, k].cpu().numpy())[ok]
                eff = d_cf - d_rn
                m, lo, hi = boot_ci(eff, arg.n_boot, rng)
                flip_cf = float((ocf[key].argmax(1).cpu().numpy() != k)[ok].mean()) if ok.any() else float("nan")
                flip_rn = float((orn[key].argmax(1).cpu().numpy() != k)[ok].mean()) if ok.any() else float("nan")
                rec[key] = dict(drop_cf=float(d_cf.mean()) if ok.any() else float("nan"),
                                drop_rand=float(d_rn.mean()) if ok.any() else float("nan"),
                                effect=m, ci_lo=lo, ci_hi=hi, causal=bool(lo > 0),
                                flip_cf=flip_cf, flip_rand=flip_rn)
            records.append(rec)
        print("[cf] class {} done ({} literals) {:.0f}s".format(k, len(lits), time.time() - t0))
    return records


def perturb(x, kind, level, gen):
    nz = (x.abs().sum(1, keepdim=True) > 0).float()          # (N,1,T,V,M) non-zero joints
    if kind == "noise":
        return x + level * torch.randn(x.shape, device=x.device, generator=gen) * nz
    if kind == "joint_drop":
        N, C, T, V, M = x.shape
        keep = (torch.rand(N, 1, 1, V, 1, device=x.device, generator=gen) >= level).float()
        keep[:, :, :, 20] = 1.0
        return x * keep
    if kind == "frame_drop":
        N, C, T, V, M = x.shape
        keep = (torch.rand(N, 1, T, 1, 1, device=x.device, generator=gen) >= level).float()
        return x * keep
    raise ValueError(kind)


@torch.no_grad()
def robustness_section(models, loader, device, seed):
    settings = [("clean", 0.0), ("noise", 0.01), ("noise", 0.03), ("joint_drop", 0.2),
                ("frame_drop", 0.3)]
    out = []
    for kind, level in settings:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        corr = dict(net0=0, net_ft=0, logic=0, fused=0)
        n = 0
        for data, label, _ in loader:
            x = data.float().to(device)
            if kind != "clean":
                x = perturb(x, kind, level, gen)
            o = models.run(x)
            y = label.long().to(device)
            for key in corr:
                corr[key] += int((o[key].argmax(1) == y).sum())
            n += len(y)
        row = dict(kind=kind, level=level, **{k: v / n for k, v in corr.items()})
        print("[robust]", row)
        out.append(row)
    return out


# --------------------------------------------------------------------------- main
def main():
    arg = get_parser().parse_args()
    os.makedirs(arg.out_dir, exist_ok=True)
    rng = np.random.RandomState(arg.seed)
    torch.manual_seed(arg.seed)
    device = arg.device

    ck = torch.load(arg.ccp_ckpt, map_location=device, weights_only=False)
    meta = ck["meta"]
    csv_path = arg.concept_csv or meta["concept_csv"]
    bank = build_bank(csv_path, num_class=meta["num_class"], extras=meta["extras"])
    assert bank.names == meta["concept_names"], "concept bank differs from the checkpoint"
    ccp = CCP(bank, meta["c6"], use_before=meta["use_before"]).to(device)
    ccp.load_state_dict(ck["ccp"])
    ccp.eval()
    net0 = load_hypergcn(arg.hypergcn_root, arg.checkpoint, num_class=meta["num_class"], device=device)
    net_ft = net0
    if ck.get("backbone") is not None:
        net_ft = load_hypergcn(arg.hypergcn_root, arg.checkpoint, num_class=meta["num_class"], device=device)
        net_ft.load_state_dict(ck["backbone"])
        net_ft.eval()
    models = Models(net0, net_ft, ccp)
    print("[eval] ccp stage {} epoch {}; backbone fine-tuned: {}".format(
        ck["stage"], ck["epoch"], ck.get("backbone") is not None))

    dataset = build_feeder(arg.hypergcn_root, arg.data_path, "test", config=arg.config)
    if arg.max_samples is not None and arg.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(arg.max_samples)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=arg.batch_size, shuffle=False,
                                         num_workers=arg.num_worker)

    # 1. accuracy
    res = collect(models, loader, device)
    y = res["y"]
    acc = {k: float((res[k].argmax(1) == y).mean()) for k in ("net0", "net_ft", "logic", "fused")}
    acc["alpha"] = float(ccp.alpha())
    print("[acc]", acc)

    # 2. concept quality
    names = ccp.concept_names
    csv_m = ccp.csv_matrix.cpu().numpy()
    concept_rows = []
    for c, spec in enumerate(ccp.specs):
        lab = csv_m[y, c] if spec.in_csv else None
        concept_rows.append(dict(concept=names[c], kind=spec.kind, in_csv=spec.in_csv,
                                 signal=spec.signal, csv_classes=int(csv_m[:, c].sum()),
                                 auc=auroc(res["z"][:, c], lab) if lab is not None else float("nan"),
                                 true_rate=float((res["z"][:, c] >= 0.5).mean())))

    # 3. program
    rules = export_rules(ccp, min_w=arg.rule_min_w)
    for r in rules:
        k = r["class_idx"]
        if r["literals"]:
            sat = rule_truth(r, res["z"], res["before"])
            r["coverage"] = float(sat[y == k].mean()) if (y == k).any() else float("nan")
            r["precision"] = float((y[sat] == k).mean()) if sat.any() else float("nan")
        else:
            r["coverage"] = r["precision"] = float("nan")

    # 4. knowledge revision
    wp, wn, _ = (w.detach().cpu().numpy() for w in ccp.logic.weights())
    thr = arg.rule_min_w
    sup = np.array([s.in_csv for s in ccp.specs])
    confirmed = [(k, c) for k in range(len(rules)) for c in np.flatnonzero((csv_m[k] > 0) & (wp[k] >= thr))]
    removed = [(k, c) for k in range(len(rules)) for c in np.flatnonzero((csv_m[k] > 0) & (wp[k] < thr))]
    new_pos = [(k, c) for k in range(len(rules)) for c in np.flatnonzero((csv_m[k] == 0) & (wp[k] >= thr))]
    new_neg = [(k, c) for k in range(len(rules)) for c in np.flatnonzero(wn[k] >= thr)]
    cls = ccp.class_names
    revision = dict(
        confirmed=[[cls[k], names[c]] for k, c in confirmed],
        removed=[[cls[k], names[c], float(wp[k, c])] for k, c in removed],
        new_positive=[[cls[k], names[c], float(wp[k, c]), bool(sup[c])] for k, c in new_pos],
        new_negative=[[cls[k], names[c], float(wn[k, c])] for k, c in new_neg])

    # 5. counterfactual faithfulness
    cf = counterfactual_section(models, dataset, res, rules, ccp, arg, rng)

    # 6. robustness
    robust = robustness_section(models, loader, device, arg.seed) if arg.robust else []

    # ---- write
    with open(os.path.join(arg.out_dir, "rules.json"), "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=1, ensure_ascii=False)
    with open(os.path.join(arg.out_dir, "revision.json"), "w", encoding="utf-8") as f:
        json.dump(revision, f, indent=1, ensure_ascii=False)
    with open(os.path.join(arg.out_dir, "counterfactual.json"), "w", encoding="utf-8") as f:
        json.dump(cf, f, indent=1, ensure_ascii=False)
    with open(os.path.join(arg.out_dir, "metrics.json"), "w") as f:
        json.dump(dict(accuracy=acc, robustness=robust, concepts=concept_rows), f, indent=1)
    write_report(arg, acc, concept_rows, rules, revision, cf, robust)
    print("[done] {}/report.md".format(arg.out_dir))


def write_report(arg, acc, concept_rows, rules, revision, cf, robust):
    L = ["# Causal Concept Program on TEST", ""]
    L += ["## 1. Accuracy", "", "| model | top-1 |", "|---|---|",
          "| HyperGCN (original) | {} |".format(fmt(acc["net0"], 4)),
          "| HyperGCN (fine-tuned in stage 3) | {} |".format(fmt(acc["net_ft"], 4)),
          "| logic only | {} |".format(fmt(acc["logic"], 4)),
          "| fused (logic + {:.3f} * net) | {} |".format(acc["alpha"], fmt(acc["fused"], 4)), ""]

    geo = [r["auc"] for r in concept_rows if r["kind"] == "geo" and r["in_csv"] and not np.isnan(r["auc"])]
    lat = [r["auc"] for r in concept_rows if r["kind"] == "latent" and not np.isnan(r["auc"])]
    L += ["## 2. Concept quality (AUROC vs CSV row of the true class)", "",
          "- geo concepts: mean AUROC {} over {}".format(fmt(float(np.mean(geo)) if geo else float("nan")), len(geo)),
          "- latent concepts: mean AUROC {} over {}".format(fmt(float(np.mean(lat)) if lat else float("nan")), len(lat)),
          "", "| concept | kind | signal | #classes in CSV | AUROC | true rate |", "|---|---|---|---|---|---|"]
    for r in sorted(concept_rows, key=lambda r: (np.isnan(r["auc"]), -np.nan_to_num(r["auc"]))):
        L.append("| {} | {} | {} | {} | {} | {} |".format(r["concept"], r["kind"], r["signal"] or "-",
                                                          r["csv_classes"], fmt(r["auc"]), fmt(r["true_rate"])))
    L.append("")

    n_lit = [len(r["literals"]) for r in rules]
    in_csv = [l["in_csv"] for r in rules for l in r["literals"] if l["kind"] == "pos"]
    L += ["## 3. Program (weight >= {})".format(arg.rule_min_w), "",
          "- literals per class: mean {:.1f}, min {}, max {}".format(float(np.mean(n_lit)), min(n_lit), max(n_lit)),
          "- positive literals from the CSV: {}/{}".format(sum(in_csv), len(in_csv)), "",
          "| k | class | rule | TEST coverage | TEST precision |", "|---|---|---|---|---|"]
    for r in rules:
        txt = " ∧ ".join("{} [w={:.2f}]".format(l["text"], l["weight"]) for l in r["literals"]) or "(empty)"
        L.append("| {} | {} | {} | {} | {} |".format(r["class_idx"], r["class_name"], txt,
                                                     fmt(r["coverage"]), fmt(r["precision"])))
    L.append("")

    L += ["## 4. Knowledge revision", "",
          "- CSV literals confirmed: {}".format(len(revision["confirmed"])),
          "- CSV literals removed: {}".format(len(revision["removed"])),
          "- new positive literals: {} (of which x_* extras: {})".format(
              len(revision["new_positive"]), sum(1 for r in revision["new_positive"] if not r[3])),
          "- new negative literals: {}".format(len(revision["new_negative"])), "",
          "New positive literals:", ""]
    L += ["- {}: {} (w={:.2f})".format(a, c, w) for a, c, w, _ in revision["new_positive"]] or ["- none"]
    L.append("")

    done = [r for r in cf if r.get("n", 0) > 0 and "net0" in r]
    for key, title in (("net0", "original HyperGCN"), ("fused", "fused model")):
        causal = [r for r in done if r[key]["causal"]]
        L += ["## 5{}. Counterfactual faithfulness: {}".format("a" if key == "net0" else "b", title), "",
              "- (class, literal) tested: {}; causal (95% CI of effect > 0): {}".format(len(done), len(causal)),
              "- effect = drop of the class logit under the counterfactual minus under a random edit of the same size", ""]
        L += ["| class | literal | CSV | n | success | edit (L) | drop cf | drop rand | effect [95% CI] | flip cf / rand | causal |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in done:
            e = r[key]
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} [{}, {}] | {} / {} | {} |".format(
                r["class_name"], r["concept"], "Y" if r["in_csv"] else "n", r["n"], fmt(r["success"], 2),
                fmt(r["edit"]), fmt(e["drop_cf"]), fmt(e["drop_rand"]), fmt(e["effect"]), fmt(e["ci_lo"]),
                fmt(e["ci_hi"]), fmt(e["flip_cf"], 2), fmt(e["flip_rand"], 2), "Y" if e["causal"] else "n"))
        L.append("")

    if robust:
        L += ["## 6. Robustness", "", "| perturbation | level | HyperGCN original | HyperGCN ft | logic | fused |",
              "|---|---|---|---|---|---|"]
        for r in robust:
            L.append("| {} | {} | {} | {} | {} | {} |".format(r["kind"], r["level"], fmt(r["net0"], 4),
                                                              fmt(r["net_ft"], 4), fmt(r["logic"], 4), fmt(r["fused"], 4)))
        L.append("")
    with open(os.path.join(arg.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
