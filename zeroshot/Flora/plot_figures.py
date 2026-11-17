"""Generate clean visualization figures for PRIZM (saved as PDF and 300-DPI PNG).

Figures produced in <out-dir> (default: <DRIVE_OUT>/figures):
  1. fig_reversed_pairs.{pdf,png} : Dumbbell plot of centered cosine similarity for the
                                    7 reversed-order action pairs (KG vs. + ORE + PPC).
  2. fig_perclass_110.{pdf,png}   : Per-class ZSL accuracy on NTU-120 110/10 (No KG vs. PRIZM).
  3. fig_cm_110.{pdf,png}         : 10x10 normalized confusion matrix of PRIZM on NTU-120 110/10.

Usage (from Colab /flora_nesy.ipynb):
  !python {FLORA}/plot_figures.py
  !python {FLORA}/plot_figures.py --only reversed
  !python {FLORA}/plot_figures.py --only 110
"""
import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _centered_cos(x):
    z = x[:, -1, :].float()
    z = F.normalize(z - z.mean(0, keepdim=True), dim=-1)
    return (z @ z.t()).cpu().numpy()


def plot_reversed_pairs(fig_dir, device='cuda'):
    """Minimalist dumbbell plot: Plain KG vs. + ORE + PPC across the 7 reversed-order pairs."""
    build_seq_kg(verbose=False)
    n_cls = 120
    kg = KS.PrimitiveKG(DST_PRIMS, DST_RULES, n_cls, 0.5)
    names = pd.read_csv('semantics/ntu120.csv')['label'].tolist()

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
    labels = [f'{names[i]}  ↔  {names[j]}' for i, j in pairs]
    v_plain = np.array([c_plain[i, j] for i, j in pairs])
    v_ppc = np.array([c_ppc[i, j] for i, j in pairs])

    order = np.argsort(v_ppc)
    labels = [labels[k] for k in order]
    v_plain, v_ppc = v_plain[order], v_ppc[order]

    y = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for k in range(len(pairs)):
        ax.plot([v_ppc[k], v_plain[k]], [y[k], y[k]], color='#cbd5e1', linewidth=2.4, zorder=1)
        ax.text(v_ppc[k] - 0.012, y[k], f'{v_ppc[k]:.2f}', va='center', ha='right',
                fontsize=8.5, color='#1d4ed8', fontweight='bold')
        ax.text(v_plain[k] + 0.012, y[k], f'{v_plain[k]:.2f}', va='center', ha='left',
                fontsize=8.5, color='#64748b')

    ax.scatter(v_plain, y, color='#94a3b8', s=55, zorder=3, label=f'KG (mean {v_plain.mean():.2f})')
    ax.scatter(v_ppc, y, color='#2563eb', s=55, zorder=4, label=f'+ ORE + PPC (mean {v_ppc.mean():.2f})')

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.66, 1.06)
    ax.set_xlabel('Centered Cosine Similarity (lower = more distinct)')
    ax.legend(loc='lower left', frameon=True)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(fig_dir, f'fig_reversed_pairs.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved {os.path.join(fig_dir, "fig_reversed_pairs.pdf")}')


def _make_recording_trainer(base_cls):
    class RecTrainer(base_cls):
        def test_zsl(self, nb_iter):
            prev_best = self.best_accuracy[0]
            self.mapping_model.eval()
            total, correct, yt, yp = 0, 0, [], []
            for _, (gt_motion, label_idx, _, _, _) in enumerate(self.zsl_loader):
                with torch.no_grad():
                    gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
                    label_idx = label_idx.to(self.device, non_blocking=True)
                    labels = self.zsl_target_lookup[label_idx]
                    preds = self.mapping_model.predict_zsl(
                        gt_motion, self.unseen_classes, self.current_attuned_semantics(), labels)['predictions_idx']
                    yt.append(labels.cpu().numpy())
                    yp.append(preds.cpu().numpy())
                    correct += (preds == labels).sum().item()
                    total += gt_motion.size(0)
            acc = 100.0 * correct / total
            self.best_accuracy[0] = max(self.best_accuracy[0], acc)
            if acc >= prev_best:
                self.best_preds = (np.concatenate(yt), np.concatenate(yp))
            self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')
            self.mapping_model.train()
    return RecTrainer


def plot_110(fig_dir, rerun=False):
    """Per-class accuracy bar chart and single clean confusion matrix on NTU-120 110/10."""
    C.set_feat('star')
    task = 'ntu120_xsub_seen110_unseen10'
    _, unseen, *_ = U.task_definition(task)
    names = pd.read_csv('semantics/ntu120.csv')['label'].tolist()
    unseen_names = [names[int(u)] for u in unseen]
    cache_npz = os.path.join(fig_dir, 'preds_110_s1.npz')

    if rerun or not os.path.exists(cache_npz):
        build_seq_kg(verbose=False)

        def run_one(cls, run_dir, flora_ov, extra):
            logging.getLogger('Exp').handlers.clear()
            os.makedirs(run_dir, exist_ok=True)
            args = C.build_args(task, 1, run_dir, flora_ov, extra)
            U.init_seed(1)
            tr = _make_recording_trainer(cls)(args)
            tr.train()
            return tr.best_accuracy[0], tr.best_preds

        tmp = os.path.join(fig_dir, '_tmp_110')
        print('[plot] running No KG on 110/10 (seed 1)...', flush=True)
        acc_no, (yt_no, yp_no) = run_one(Trainer, os.path.join(tmp, 'none'), dict(kg_mode='none', setting='ZSL'), {})
        print('[plot] running PRIZM on 110/10 (seed 1)...', flush=True)
        acc_pr, (yt_pr, yp_pr) = run_one(
            FuseTrainer, os.path.join(tmp, 'prizm'),
            dict(kg_mode='compose', kg_beta=0.3, kg_prims=DST_PRIMS, kg_rules=DST_RULES,
                 kg_compose_src='avg', kg_compose_at='post', kg_compose_half='both', setting='ZSL'),
            dict(forbid_lam=1.0, forbid_order_only=1, fuse_center=1, fuse_orth=1, fuse_adaptive=1, fuse_clip=3.0))
        np.savez(cache_npz, acc_no=acc_no, yt_no=yt_no, yp_no=yp_no, acc_pr=acc_pr, yt_pr=yt_pr, yp_pr=yp_pr)
        print(f'[plot] 110/10 ZSL — No KG: {acc_no:.2f}%, PRIZM: {acc_pr:.2f}%', flush=True)

    dat = np.load(cache_npz)
    yt_no, yp_no = dat['yt_no'], dat['yp_no']
    yt_pr, yp_pr = dat['yt_pr'], dat['yp_pr']
    nc = len(unseen)

    pc_no = np.array([100.0 * (yp_no[yt_no == k] == k).mean() for k in range(nc)])
    pc_pr = np.array([100.0 * (yp_pr[yt_pr == k] == k).mean() for k in range(nc)])

    # ---- 1) Clean vertical grouped bar chart of per-class accuracy
    x = np.arange(nc)
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(x - w / 2, pc_no, width=w, color='#cbd5e1', label=f'No KG ({float(dat["acc_no"]):.1f}%)')
    ax.bar(x + w / 2, pc_pr, width=w, color='#2563eb', label=f'PRIZM ({float(dat["acc_pr"]):.1f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(unseen_names, rotation=30, ha='right')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'fig_perclass_110.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved {os.path.join(fig_dir, "fig_perclass_110.pdf")}')

    # ---- 2) Single clean 10x10 Confusion Matrix of PRIZM
    cm = np.zeros((nc, nc), dtype=np.float64)
    for t, p in zip(yt_pr, yp_pr):
        cm[t, p] += 1
    cm = 100.0 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    ax.set_xticklabels(unseen_names, rotation=35, ha='right', fontsize=8.5)
    ax.set_yticklabels(unseen_names, fontsize=8.5)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    for i in range(nc):
        for j in range(nc):
            if cm[i, j] >= 1.0:
                ax.text(j, i, f'{cm[i, j]:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > 55 else '#0f172a')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Accuracy (%)')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'fig_cm_110.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved {os.path.join(fig_dir, "fig_cm_110.pdf")}')


def main():
    default_out = os.path.join(C.DRIVE_OUT, 'figures') if os.path.isdir(os.path.dirname(C.DRIVE_OUT)) \
        else os.path.join(HERE, 'output', 'figures')
    ap = argparse.ArgumentParser(description='Generate PRIZM figures')
    ap.add_argument('--out-dir', default=default_out, help='directory to save PDF/PNG figures')
    ap.add_argument('--only', choices=['all', 'reversed', '110'], default='all')
    ap.add_argument('--rerun', action='store_true', help='re-train 110/10 even if cached predictions exist')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.only in ('all', 'reversed'):
        plot_reversed_pairs(args.out_dir, device=device)
    if args.only in ('all', '110'):
        plot_110(args.out_dir, rerun=args.rerun)


if __name__ == '__main__':
    main()
