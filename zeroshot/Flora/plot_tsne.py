"""t-SNE of the shared latent space on the unseen test set: No KG vs. PRIZM (one compact PNG).

Runs Flora (No KG) and PRIZM (SeqKG + COA-Fuse) in isolated subprocesses so training
and CUDA RNG match standard Flora/PRIZM runs identically. Because motion_encoder and
semantic_encoder are frozen during Phase 2, latent means (mu) are extracted once after
training finishes without perturbing evaluation RNG.

Dots = unseen test samples (skeleton latent mu), stars = unseen class semantics (mu0).
Title reports the exact ZSL accuracy and cosine silhouette score in the original latent space.

Usage (Colab):
  !python {FLORA}/plot_tsne.py --task 48          # NTU-60 48/12 (default)
  !python {FLORA}/plot_tsne.py --task 48 --rerun  # retrain and refresh cached latents
Output: <out-dir>/tsne_<task>.png  (cache: <out-dir>/tsne_<task>_latents.npz)
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

import nesy_common as C
import utils.utils as U
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_1_fuse import FuseTrainer
from trainer import Trainer

TASKS = {'55': 'ntu60_xsub_seen55_unseen5', '48': 'ntu60_xsub_seen48_unseen12',
         '110': 'ntu120_xsub_seen110_unseen10', '96': 'ntu120_xsub_seen96_unseen24'}
SPLIT_NAME = {'55': 'NTU-60 55/5', '48': 'NTU-60 48/12', '110': 'NTU-120 110/10', '96': 'NTU-120 96/24'}


def run_single_extract(task, mode, out_npz, fig_dir):
    """Train one model in a clean process and extract frozen Phase-1 latent means."""
    C.set_feat('star')
    build_seq_kg(verbose=False)
    logging.getLogger('Exp').handlers.clear()
    run_dir = os.path.join(fig_dir, f'_tmp_tsne_{task}', mode)
    os.makedirs(run_dir, exist_ok=True)

    if mode == 'none':
        cls, flora_ov, extra = Trainer, dict(kg_mode='none', setting='ZSL'), {}
    else:
        cls = FuseTrainer
        flora_ov = dict(kg_mode='compose', kg_beta=0.3, kg_prims=DST_PRIMS, kg_rules=DST_RULES,
                        kg_compose_src='avg', kg_compose_at='post', kg_compose_half='both', setting='ZSL')
        extra = dict(forbid_lam=1.0, forbid_order_only=1, fuse_center=1, fuse_orth=1,
                     fuse_adaptive=1, fuse_clip=3.0)

    args = C.build_args(TASKS[task], 1, run_dir, flora_ov, extra)
    U.init_seed(1)
    tr = cls(args)
    tr.train()

    m = tr.mapping_model
    m.eval()
    sem = tr.current_attuned_semantics()
    zs, ys = [], []
    with torch.no_grad():
        for gt, lab, _, _, _ in tr.zsl_loader:
            gt = gt.to(tr.device, dtype=torch.float32, non_blocking=True)
            labels = tr.zsl_target_lookup[lab.to(tr.device, non_blocking=True)]
            sk = gt.unsqueeze(1).expand(-1, tr.args.max_length + 1, -1)
            _, mu, _ = m.motion_encoder(sk)
            zs.append(mu[:, 0].float().cpu().numpy())
            ys.append(labels.cpu().numpy())
        _, mu0, _ = m.semantic_encoder(sem[tr.unseen_classes])

    np.savez(out_npz, acc=tr.best_accuracy[0], z=np.concatenate(zs), y=np.concatenate(ys),
             anchors=mu0.mean(1).float().cpu().numpy())
    print(f'[tsne] {mode}: best ZSL {tr.best_accuracy[0]:.2f}%', flush=True)


def train_both(task, fig_dir):
    parts = {}
    for mode in ('none', 'prizm'):
        part_npz = os.path.join(fig_dir, f'_tmp_tsne_{task}_{mode}.npz')
        cmd = [sys.executable, os.path.abspath(__file__),
               '--task', task, '--out-dir', fig_dir, '--single-mode', mode, '--single-out', part_npz]
        subprocess.run(cmd, check=True)
        d = np.load(part_npz)
        parts[mode] = {k: d[k] for k in ('acc', 'z', 'y', 'anchors')}
    return parts['none'], parts['prizm']


def embed(lat, max_per_class, seed=0):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    rng = np.random.RandomState(seed)
    z, y, a = lat['z'], lat['y'], lat['anchors']
    keep = np.concatenate([rng.permutation(np.where(y == k)[0])[:max_per_class] for k in np.unique(y)])
    z, y = z[keep], y[keep]
    sil = silhouette_score(z, y, metric='cosine')
    x = np.concatenate([z, a])
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    e = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto',
             random_state=seed).fit_transform(x)
    return e[:len(z)], y, e[len(z):], sil


def main():
    default_out = os.path.join(C.DRIVE_OUT, 'figures') if os.path.isdir(os.path.dirname(C.DRIVE_OUT)) \
        else os.path.join(HERE, 'output', 'figures')
    ap = argparse.ArgumentParser(description='t-SNE of unseen latents: No KG vs PRIZM')
    ap.add_argument('--task', choices=list(TASKS), default='48')
    ap.add_argument('--out-dir', default=default_out)
    ap.add_argument('--max-per-class', type=int, default=250)
    ap.add_argument('--rerun', action='store_true')
    ap.add_argument('--single-mode', choices=['none', 'prizm'], help=argparse.SUPPRESS)
    ap.add_argument('--single-out', help=argparse.SUPPRESS)
    opts = ap.parse_args()
    os.makedirs(opts.out_dir, exist_ok=True)

    if opts.single_mode:
        run_single_extract(opts.task, opts.single_mode, opts.single_out, opts.out_dir)
        return

    cache = os.path.join(opts.out_dir, f'tsne_{opts.task}_latents.npz')
    if opts.rerun or not os.path.exists(cache):
        none, prizm = train_both(opts.task, opts.out_dir)
        np.savez(cache, **{f'none_{k}': v for k, v in none.items()}, **{f'prizm_{k}': v for k, v in prizm.items()})
    d = np.load(cache)
    lat = {m: {k: d[f'{m}_{k}'] for k in ('acc', 'z', 'y', 'anchors')} for m in ('none', 'prizm')}

    _, unseen, *_ = U.task_definition(TASKS[opts.task])
    rules = json.load(open('kg/kg_rules_v2.json', encoding='utf-8'))['rules']
    cls_name = {r['id'] - 1: r.get('class', str(r['id'])) for r in rules}
    names = [cls_name[int(u)] for u in unseen]
    nc = len(unseen)
    cmap = plt.get_cmap('tab10' if nc <= 10 else 'tab20')
    colors = [cmap(k % cmap.N) for k in range(nc)]

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, (m, title) in zip(axes, (('none', 'No KG'), ('prizm', 'PRIZM'))):
        e, y, ea, sil = embed(lat[m], opts.max_per_class)
        for k in range(nc):
            p = e[y == k]
            ax.scatter(p[:, 0], p[:, 1], s=5, color=colors[k], alpha=0.45, linewidths=0)
        for k in range(nc):
            ax.scatter(ea[k, 0], ea[k, 1], marker='*', s=175, color=colors[k],
                       edgecolors='black', linewidths=0.8, zorder=5 + k * 0.01)
            # Label key classes or index so overlapping stars remain traceable
            if names[k] in ('falling', 'wear a shoe'):
                ax.annotate(names[k], (ea[k, 0], ea[k, 1]), xytext=(5, 4),
                            textcoords='offset points', fontsize=7.5, fontweight='bold',
                            color='#0f172a', zorder=20)
        ax.set_title(f'{title}  (acc {float(lat[m]["acc"]):.1f}%, silhouette {sil:.2f})')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#cbd5e1')

    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=6, color=colors[k], label=names[k])
               for k in range(nc)]
    handles.append(plt.Line2D([], [], marker='*', linestyle='', markersize=10, color='white',
                              markeredgecolor='black', label='class semantics'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 5 if nc > 5 else 6),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13 if nc > 5 else 0.07, 1, 1))
    out = os.path.join(opts.out_dir, f'tsne_{opts.task}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[tsne] {SPLIT_NAME[opts.task]} saved {out}')


if __name__ == '__main__':
    main()
