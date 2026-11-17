"""Generate 3 compact PNG figures for PRIZM:
  1. fig_reversed_pairs.png   : Dumbbell plot separating the 7 reversed-order pairs (KG + ORE vs KG + ORE + PPC).
  2. fig_beta_curve.png       : Sensitivity curve over KG weight beta (from Table tab:beta).
  3. fig_qualitative_cases.png: Top-3 flow-velocity MSE cases on 48/12 (No KG wrong -> PRIZM right).

Usage (Colab):
  !python {FLORA}/plot_extra.py                 # all 3
  !python {FLORA}/plot_extra.py --only reversed # just #1
  !python {FLORA}/plot_extra.py --only beta     # just #2
  !python {FLORA}/plot_extra.py --only qual     # just #3 (uses cached qual_48_s1.npz after first run)
"""
import argparse
import json
import logging
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import kg_semantics as KS
import nesy_common as C
import utils.utils as U
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_forbid import signed_compose_weights, order_only_pairs
from nesy4_1_1_fuse import FuseTrainer
from trainer import Trainer


plt.rcParams.update({
    'font.size': 9.5,
    'axes.titlesize': 10,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _centered_cos(x):
    z = x[:, -1, :].float()
    z = F.normalize(z - z.mean(0, keepdim=True), dim=-1)
    return (z @ z.t()).cpu().numpy()


# ------------------------------------------------------------------ 1) Reversed-order dumbbell
def plot_reversed_pairs(fig_dir):
    build_seq_kg(verbose=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_cls = 120
    kg = KS.PrimitiveKG(DST_PRIMS, DST_RULES, n_cls, 0.5)
    rules = json.load(open(DST_RULES, encoding='utf-8'))['rules']
    cls_name = {r['id'] - 1: r.get('class', str(r['id'])) for r in rules}

    prim_cache = os.path.join(fig_dir, 'cache_v2seq_prim_clip_L75.pt')
    if os.path.exists(prim_cache):
        prim_emb = torch.load(prim_cache, map_location=device)
    else:
        from transformers import CLIPTokenizer, CLIPTextModel
        clip_id = 'openai/clip-vit-large-patch14-336'
        tok = CLIPTokenizer.from_pretrained(clip_id)
        enc = CLIPTextModel.from_pretrained(clip_id).to(device).eval()
        prim_emb = KS.embed_texts(tok, enc, kg.prim_text, 75, device, prim_cache)
        del tok, enc
        if device == 'cuda':
            torch.cuda.empty_cache()

    w_plain = torch.from_numpy(kg.compose_weights()).to(device, prim_emb.dtype)
    w_ppc = torch.from_numpy(signed_compose_weights(kg, 1.0, order_only=True)).to(device, prim_emb.dtype)
    c_plain = _centered_cos(torch.einsum('np,ptd->ntd', w_plain, prim_emb))
    c_ppc = _centered_cos(torch.einsum('np,ptd->ntd', w_ppc, prim_emb))

    pairs = order_only_pairs(DST_RULES, n_cls)
    labels = [f'{cls_name[i]}  ↔  {cls_name[j]}' for i, j in pairs]
    v_plain = np.array([c_plain[i, j] for i, j in pairs])
    v_ppc = np.array([c_ppc[i, j] for i, j in pairs])

    order = np.argsort(v_ppc)
    labels = [labels[k] for k in order]
    v_plain, v_ppc = v_plain[order], v_ppc[order]

    y = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k in range(len(pairs)):
        ax.plot([v_ppc[k], v_plain[k]], [y[k], y[k]], color='#cbd5e1', linewidth=2.4, zorder=1)
        ax.text(v_ppc[k] - 0.012, y[k], f'{v_ppc[k]:.2f}', va='center', ha='right',
                fontsize=8.5, color='#1d4ed8', fontweight='bold')
        ax.text(v_plain[k] + 0.012, y[k], f'{v_plain[k]:.2f}', va='center', ha='left',
                fontsize=8.5, color='#64748b')

    ax.scatter(v_plain, y, color='#94a3b8', s=55, zorder=3, label=f'KG + ORE (mean {v_plain.mean():.2f})')
    ax.scatter(v_ppc, y, color='#2563eb', s=55, zorder=4, label=f'KG + ORE + PPC (mean {v_ppc.mean():.2f})')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.62, 1.06)
    ax.set_xlabel('Centered Cosine Similarity (lower = more distinct)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'fig_reversed_pairs.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved {out}')


# ------------------------------------------------------------------ 2) Beta sensitivity curve (Delta vs No KG)
def plot_beta_curve(fig_dir):
    betas = np.array([0.1, 0.3, 0.5, 1.0])
    d_48 = np.array([63.7, 63.7, 62.0, 58.7]) - 62.1
    d_110 = np.array([78.9, 80.2, 81.6, 79.8]) - 77.6

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.axhspan(-4.5, 0.0, color='#fee2e2', alpha=0.35, zorder=0)
    ax.axhline(0.0, color='#475569', linestyle='--', linewidth=1.2, zorder=1, label='No KG baseline (0.0%)')

    ax.plot(betas, d_110, 's-', color='#dc2626', linewidth=2.0, label='NTU-120 110/10', zorder=3)
    ax.plot(betas, d_48, 'o-', color='#2563eb', linewidth=2.0, label='NTU-60 48/12', zorder=3)

    ax.axvline(0.3, color='#059669', linestyle='--', linewidth=1.3, zorder=2, label=r'Fixed $\beta = 0.3$')

    ax.text(0.5, d_110[2] + 0.25, f'+{d_110[2]:.1f}%', ha='center', fontsize=8, color='#dc2626', fontweight='bold')
    ax.text(0.3, d_48[1] + 0.25, f'+{d_48[1]:.1f}%', ha='center', fontsize=8, color='#2563eb', fontweight='bold')
    ax.text(1.0, d_48[3] - 0.55, f'{d_48[3]:.1f}%', ha='center', fontsize=8, color='#2563eb', fontweight='bold')

    ax.set_xticks(betas)
    ax.set_ylim(-4.3, 4.9)
    ax.set_xlabel(r'KG Weight $\beta$ (C-Fuse)')
    ax.set_ylabel(r'Gain over No KG ($\Delta$ ZSL Acc, %)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()
    out = os.path.join(fig_dir, 'fig_beta_curve.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved {out}')


# ------------------------------------------------------------------ 3) Qualitative cases on 48/12
def _make_mse_trainer(base_cls):
    """Exact same forward pass & RNG stream as Flora's predict_zsl, returning MSE matrix."""
    class MseTrainer(base_cls):
        def test_zsl(self, nb_iter):
            prev = self.best_accuracy[0]
            m = self.mapping_model
            m.eval()
            sem = self.current_attuned_semantics()
            total, correct, mses, ys = 0, 0, [], []
            with torch.no_grad():
                for _, (gt, lab, _, _, _) in enumerate(self.zsl_loader):
                    gt = gt.to(self.device, dtype=torch.float32, non_blocking=True)
                    labels = self.zsl_target_lookup[lab.to(self.device, non_blocking=True)]
                    B = gt.size(0)
                    sk = gt.unsqueeze(1).expand(-1, m.args.max_length + 1, -1)
                    z1_ske, _, _ = m.motion_encoder(sk)
                    z0_text, _, _ = m.semantic_encoder(sem[self.unseen_classes])
                    timestep = torch.ones(B, device=sk.device) * m.args.step_ratio
                    chunk = m.args.eval_class_chunk or len(self.unseen_classes)
                    err_chunks = []
                    for s in range(0, len(self.unseen_classes), chunk):
                        z0c = z0_text[s:s + chunk]
                        nc = z0c.shape[0]
                        z0b = z0c.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nc, *z0c.shape[1:])
                        z1b = z1_ske.unsqueeze(1).expand(-1, nc, -1, -1).reshape(B * nc, *z1_ske.shape[1:])
                        tb = timestep.unsqueeze(1).expand(-1, nc).reshape(-1)
                        zt = m.psi(tb, z0b, z1b)
                        pv = m.velocity_predictor(zt, tb)
                        tv = m.Dt_psi(tb, z0b, z1b)
                        err_chunks.append(F.mse_loss(pv, tv, reduction='none').mean((1, 2)).view(B, nc))
                    err = torch.cat(err_chunks, dim=1)
                    preds = torch.argmin(err, dim=1)
                    correct += (preds == labels).sum().item()
                    total += B
                    mses.append(err.float().cpu().numpy())
                    ys.append(labels.cpu().numpy())
            acc = 100.0 * correct / total
            self.best_accuracy[0] = max(self.best_accuracy[0], acc)
            if acc >= prev:
                self.best_qual = dict(acc=acc, mse=np.concatenate(mses), y=np.concatenate(ys))
            self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')
            m.train()
    return MseTrainer


def _run_single_qual(mode, out_npz, fig_dir):
    C.set_feat('star')
    build_seq_kg(verbose=False)
    logging.getLogger('Exp').handlers.clear()
    task = 'ntu60_xsub_seen48_unseen12'
    run_dir = os.path.join(fig_dir, '_tmp_qual_48', mode)
    os.makedirs(run_dir, exist_ok=True)
    if mode == 'none':
        cls, flora_ov, extra = Trainer, dict(kg_mode='none', setting='ZSL'), {}
    else:
        cls = FuseTrainer
        flora_ov = dict(kg_mode='compose', kg_beta=0.3, kg_prims=DST_PRIMS, kg_rules=DST_RULES,
                        kg_compose_src='avg', kg_compose_at='post', kg_compose_half='both', setting='ZSL')
        extra = dict(forbid_lam=1.0, forbid_order_only=1, fuse_center=1, fuse_orth=1,
                     fuse_adaptive=1, fuse_clip=3.0)
    args = C.build_args(task, 1, run_dir, flora_ov, extra)
    U.init_seed(1)
    tr = _make_mse_trainer(cls)(args)
    tr.train()
    np.savez(out_npz, **tr.best_qual)
    print(f'[qual] {mode}: best ZSL {tr.best_accuracy[0]:.2f}%', flush=True)


def plot_qualitative_cases(fig_dir, rerun=False):
    cache = os.path.join(fig_dir, 'qual_48_s1.npz')
    if rerun or not os.path.exists(cache):
        parts = {}
        for mode in ('none', 'prizm'):
            part_npz = os.path.join(fig_dir, f'_tmp_qual_48_{mode}.npz')
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            '--only', 'qual', '--out-dir', fig_dir,
                            '--single-mode', mode, '--single-out', part_npz], check=True)
            d = np.load(part_npz)
            parts[mode] = {k: d[k] for k in ('acc', 'mse', 'y')}
        np.savez(cache,
                 none_acc=parts['none']['acc'], none_mse=parts['none']['mse'], y=parts['none']['y'],
                 prizm_acc=parts['prizm']['acc'], prizm_mse=parts['prizm']['mse'])

    d = np.load(cache)
    y = d['y']
    mse_no, mse_pr = d['none_mse'], d['prizm_mse']
    yp_no, yp_pr = mse_no.argmin(1), mse_pr.argmin(1)

    _, unseen, *_ = U.task_definition('ntu60_xsub_seen48_unseen12')
    rules = json.load(open('kg/kg_rules_v2.json', encoding='utf-8'))['rules']
    by_id = {r['id'] - 1: r for r in rules}
    unseen_names = [by_id[int(u)]['class'] for u in unseen]

    # Samples flipped from wrong (No KG) -> right (PRIZM); pick 3 distinct classes
    flipped = np.where((yp_no != y) & (yp_pr == y))[0]
    margin_gain = (mse_no[flipped, y[flipped]] - mse_no[flipped, yp_no[flipped]]) - \
                  (mse_pr[flipped, y[flipped]] - mse_pr[flipped].min(1))
    # Prioritize iconic classes seen in t-SNE (falling, wear a shoe) + strongest flip
    chosen, seen_cls = [], set()
    for pref in ('falling', 'wear a shoe'):
        cands = [i for i in flipped if unseen_names[y[i]] == pref]
        if cands:
            best_i = max(cands, key=lambda i: (mse_pr[i][np.argsort(mse_pr[i])[1]] - mse_pr[i, y[i]]))
            chosen.append(best_i)
            seen_cls.add(y[best_i])
    for idx in flipped[np.argsort(-margin_gain)]:
        if len(chosen) == 3:
            break
        if y[idx] not in seen_cls:
            chosen.append(int(idx))
            seen_cls.add(y[idx])

    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11.4, 2.3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    fig.subplots_adjust(wspace=0.68, hspace=0.68)

    for r_idx, sample_i in enumerate(chosen):
        true_k = int(y[sample_i])
        true_cls = unseen_names[true_k]
        req_str = ', '.join(by_id[int(unseen[true_k])]['require'][:3])
        for c_idx, (title, mse_row) in enumerate((('No KG', mse_no[sample_i]), ('PRIZM', mse_pr[sample_i]))):
            ax = axes[r_idx, c_idx]
            top3 = np.argsort(mse_row)[:3][::-1]   # reversed so rank #1 is on top
            vals = mse_row[top3]
            lbls = [f'✓ {unseen_names[k]}' if k == true_k else unseen_names[k] for k in top3]
            cols = ['#2563eb' if k == true_k else ('#ef4444' if k == top3[-1] else '#cbd5e1') for k in top3]
            yy = np.arange(3)
            ax.barh(yy, vals, color=cols, height=0.56)
            for yv, v in zip(yy, vals):
                ax.text(v + vals.max() * 0.03, yv, f'{v:.3f}', va='center', fontsize=8)
            ax.set_yticks(yy)
            ax.set_yticklabels(lbls, fontsize=8.5)
            ax.set_xlim(0, vals.max() * 1.28)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if r_idx == 0:
                ax.text(0.5, 1.30, f'{title}  (Top-3 Velocity MSE, lower = closer)',
                        transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold', color='#0f172a')
            if c_idx == 0:
                ax.text(0.0, 1.07, f'Case {r_idx + 1} — True: "{true_cls}"   [KG require: {req_str}]',
                        transform=ax.transAxes, fontsize=8.8, fontweight='bold', color='#1e293b')
            ax.grid(axis='x', linestyle='--', alpha=0.3)

    out = os.path.join(fig_dir, 'fig_qualitative_cases.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved {out}')


def main():
    default_out = os.path.join(C.DRIVE_OUT, 'figures') if os.path.isdir(os.path.dirname(C.DRIVE_OUT)) \
        else os.path.join(HERE, 'output', 'figures')
    ap = argparse.ArgumentParser(description='Generate extra PRIZM PNG figures')
    ap.add_argument('--only', choices=['all', 'reversed', 'beta', 'qual'], default='all')
    ap.add_argument('--out-dir', default=default_out)
    ap.add_argument('--rerun', action='store_true')
    ap.add_argument('--single-mode', choices=['none', 'prizm'], help=argparse.SUPPRESS)
    ap.add_argument('--single-out', help=argparse.SUPPRESS)
    opts = ap.parse_args()
    os.makedirs(opts.out_dir, exist_ok=True)

    if opts.single_mode:
        _run_single_qual(opts.single_mode, opts.single_out, opts.out_dir)
        return
    if opts.only in ('all', 'reversed'):
        plot_reversed_pairs(opts.out_dir)
    if opts.only in ('all', 'beta'):
        plot_beta_curve(opts.out_dir)
    if opts.only in ('all', 'qual'):
        plot_qualitative_cases(opts.out_dir, rerun=opts.rerun)


if __name__ == '__main__':
    main()
