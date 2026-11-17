"""t-SNE of the shared latent space on the unseen test set: No KG vs. PRIZM (one compact PNG).

For each model, the script trains Flora (No KG) and PRIZM (SeqKG + COA-Fuse) on one split,
and at the checkpoint with the best ZSL accuracy it stores
  - the skeleton latent (mean of the motion encoder) of every unseen test sample, and
  - the semantic latent (mean of the semantic encoder, averaged over tokens) of every unseen class.
Both are embedded jointly with t-SNE. Dots = test samples, stars = class semantics.
The panel title reports the ZSL accuracy and the silhouette score of the skeleton latents
(cosine, computed in the original latent space, not in 2-D).

Usage (Colab):
  !python {FLORA}/plot_tsne.py --task 48          # NTU-60 48/12 (default)
  !python {FLORA}/plot_tsne.py --task 55          # NTU-60 55/5
  !python {FLORA}/plot_tsne.py --task 48 --rerun  # retrain instead of using the cached latents
Output: <out-dir>/tsne_<task>.png  (cache: <out-dir>/tsne_<task>_latents.npz)
"""
import argparse
import json
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
import torch

import nesy_common as C
import utils.utils as U
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_1_fuse import FuseTrainer
from trainer import Trainer

TASKS = {'55': 'ntu60_xsub_seen55_unseen5', '48': 'ntu60_xsub_seen48_unseen12',
         '110': 'ntu120_xsub_seen110_unseen10', '96': 'ntu120_xsub_seen96_unseen24'}
SPLIT_NAME = {'55': 'NTU-60 55/5', '48': 'NTU-60 48/12', '110': 'NTU-120 110/10', '96': 'NTU-120 96/24'}


def recording(base_cls):
    """Same ZSL evaluation as Flora, plus the latents at the best checkpoint."""
    class Rec(base_cls):
        def test_zsl(self, nb_iter):
            prev = self.best_accuracy[0]
            m = self.mapping_model
            m.eval()
            sem = self.current_attuned_semantics()
            tot, cor, zs, ys = 0, 0, [], []
            with torch.no_grad():
                for gt, lab, _, _, _ in self.zsl_loader:
                    gt = gt.to(self.device, dtype=torch.float32, non_blocking=True)
                    labels = self.zsl_target_lookup[lab.to(self.device, non_blocking=True)]
                    pred = m.predict_zsl(gt, self.unseen_classes, sem, labels)['predictions_idx']
                    cor += (pred == labels).sum().item()
                    tot += gt.size(0)
                    sk = gt.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
                    _, mu, _ = m.motion_encoder(sk)          # tokens are identical copies
                    zs.append(mu[:, 0].float().cpu().numpy())
                    ys.append(labels.cpu().numpy())
                _, mu0, _ = m.semantic_encoder(sem[self.unseen_classes])
            acc = 100.0 * cor / tot
            self.best_accuracy[0] = max(self.best_accuracy[0], acc)
            if acc >= prev:
                self.best_latents = dict(acc=acc, z=np.concatenate(zs), y=np.concatenate(ys),
                                         anchors=mu0.mean(1).float().cpu().numpy())
            self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')
            m.train()
    return Rec


def train_and_record(task, fig_dir):
    C.set_feat('star')
    build_seq_kg(verbose=False)

    def run(cls, name, flora_ov, extra):
        logging.getLogger('Exp').handlers.clear()
        run_dir = os.path.join(fig_dir, f'_tmp_tsne_{task}', name)
        os.makedirs(run_dir, exist_ok=True)
        args = C.build_args(TASKS[task], 1, run_dir, flora_ov, extra)
        U.init_seed(1)
        tr = recording(cls)(args)
        tr.train()
        print(f'[tsne] {name}: best ZSL {tr.best_accuracy[0]:.2f}%', flush=True)
        return tr.best_latents

    none = run(Trainer, 'none', dict(kg_mode='none', setting='ZSL'), {})
    prizm = run(FuseTrainer, 'prizm',
                dict(kg_mode='compose', kg_beta=0.3, kg_prims=DST_PRIMS, kg_rules=DST_RULES,
                     kg_compose_src='avg', kg_compose_at='post', kg_compose_half='both', setting='ZSL'),
                dict(forbid_lam=1.0, forbid_order_only=1, fuse_center=1, fuse_orth=1,
                     fuse_adaptive=1, fuse_clip=3.0))
    return none, prizm


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
    opts = ap.parse_args()
    os.makedirs(opts.out_dir, exist_ok=True)

    cache = os.path.join(opts.out_dir, f'tsne_{opts.task}_latents.npz')
    if opts.rerun or not os.path.exists(cache):
        none, prizm = train_and_record(opts.task, opts.out_dir)
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
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3))
    for ax, (m, title) in zip(axes, (('none', 'No KG'), ('prizm', 'PRIZM'))):
        e, y, ea, sil = embed(lat[m], opts.max_per_class)
        for k in range(nc):
            p = e[y == k]
            ax.scatter(p[:, 0], p[:, 1], s=5, color=colors[k], alpha=0.55, linewidths=0)
        for k in range(nc):
            ax.scatter(ea[k, 0], ea[k, 1], marker='*', s=230, color=colors[k],
                       edgecolors='black', linewidths=0.9, zorder=5)
        ax.set_title(f'{title}  (acc {float(lat[m]["acc"]):.1f}%, silhouette {sil:.2f})')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#cbd5e1')

    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=6, color=colors[k], label=names[k])
               for k in range(nc)]
    handles.append(plt.Line2D([], [], marker='*', linestyle='', markersize=11, color='white',
                              markeredgecolor='black', label='class semantics'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 5 if nc > 5 else 6),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13 if nc > 5 else 0.07, 1, 1))
    out = os.path.join(opts.out_dir, f'tsne_{opts.task}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[tsne] {SPLIT_NAME[opts.task]} saved {out}')


if __name__ == '__main__':
    main()
