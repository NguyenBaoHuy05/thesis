"""Skeleton-Cache test-time adaptation (NeurIPS) on top of a trained Neuron++ model. Training-free, transductive.

    python tta_cache.py --config config/plus/ntu60_xsub_48_12.yaml --ckpt <work_dir>/base/<split>/last.pt \
                        --work-dir <out_dir> --cache-device cuda

Reads the frozen encoder's TEST cubes (the same cache main_plus.py builds), scores them with the checkpoint, then
streams the test samples one by one in a random order (fixed seeds), as Skeleton-Cache does with batch size 1:

  key      8 descriptors of the cube (n, 256, 16, 25): global, 4 body regions, 3 temporal thirds (paper eq. 3-4)
  value    the predicted class (one-hot), confidence = entropy of softmax(scale * z)          (eq. 5)
  update   per class a block of K=8 entries; when full, a more confident sample replaces the least confident   (eq. 5)
  retrieve a_{j,i}(d) = exp(-beta (1 - cos(q_d, k_{j,i,d})));  o(d) = sum_i a_{j,i}(d)   (one-hot values, eq. 6-8)
  fuse     s_j = sum_d w[d] o_j(d) with w uniform (the paper's LLM weights are not used);   logits = scale z + lam s  (eq. 11-12)

order `paper`: the sample is inserted into the cache before it retrieves, so it can match itself (paper's algorithm).
order `no_self`: it retrieves first, then is inserted (ablation: removes the self-match).
The base scores are the cosines (n, C) of Neuron++ (zsl_s_neu); `scale` turns them into logits for the entropy and the
final fusion (the cosines alone give an almost uniform softmax). scale / lam / beta are NOT tuned here beyond the
small grid you pass: a grid read on the test set is optimistic, so report the default row, not the best one.

ZSL : unseen test samples only, over the unseen classes (cache of |Yu| x K entries).
GZSL: all test samples over all classes; the stream uses the base scores calibrated with the base gamma
      (found on test, as Neuron); the final scores get their own gamma sweep (same protocol on both sides).
Differences from the paper: uniform descriptor weights (no GPT-4o prior); joint groups below are 0-based anatomical
groups of the NTU skeleton (the paper's list mixes index bases).
"""
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import main_plus
from main_plus import Processor, best_gamma, get_parser, init_seed

# NTU 25-joint skeleton, 0-based. head: neck, head, spine-shoulder; torso: spine base, spine mid, shoulders, hips;
# arms: both arms and hands; legs: both legs and the spine base.
REGIONS = dict(head=[2, 3, 20], torso=[0, 1, 4, 8, 12, 16, 20],
               arms=[4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24], legs=[0, 12, 13, 14, 15, 16, 17, 18, 19])
N_DESC = 1 + len(REGIONS) + 3


def descriptors(cube):
    """(n, 256, T, V) -> (n, 8, 256) unit vectors: global, 4 regions, 3 temporal thirds."""
    n, c, t, v = cube.shape
    g = cube.mean((2, 3))[:, None]
    sp = torch.stack([cube[..., j].mean((2, 3)) for j in REGIONS.values()], dim=1)
    cuts = [int(round(i * t / 3)) for i in range(4)]
    tm = torch.stack([cube[:, :, a:b].mean((2, 3)) for a, b in zip(cuts[:-1], cuts[1:])], dim=1)
    return F.normalize(torch.cat([g, sp, tm], dim=1), dim=-1)


class TTAProcessor(Processor):
    """Processor that only needs the test cubes: the train cache is not loaded."""

    def build_cache(self, part, N, view=0):
        return None if part == 'train' else super().build_cache(part, N, view)


@torch.no_grad()
def stream(zb, D, K, beta, lam, scale, order, perm):
    """zb (n, C) base scores (already restricted / calibrated), D (n, 8, 256) descriptors, perm: stream order.
    Returns the fused logits (n, C) in the original sample order."""
    dev = zb.device
    n, C = zb.shape
    keys = torch.zeros(C, K, N_DESC, D.shape[-1], device=dev)
    ent = torch.full((C, K), float('inf'), device=dev)
    cnt = [0] * C
    w = torch.full((N_DESC,), 1.0 / N_DESC, device=dev)
    out = torch.empty_like(zb)

    def insert(c, q, h):
        if cnt[c] < K:
            keys[c, cnt[c]] = q
            ent[c, cnt[c]] = h
            cnt[c] += 1
        else:
            m = int(ent[c].argmax())
            if h < ent[c, m]:
                keys[c, m] = q
                ent[c, m] = h

    for i in perm.tolist():
        z, q = zb[i], D[i]
        logit = scale * z
        p = logit.softmax(0)
        h = -(p * p.clamp_min(1e-12).log()).sum()
        c = int(p.argmax())
        if order == 'paper':
            insert(c, q, h)
        valid = torch.isfinite(ent)                                           # (C, K) filled slots
        cos = torch.einsum('ckdn,dn->ckd', keys, q)                           # (C, K, 8)
        a = torch.exp(-beta * (1 - cos)) * valid[..., None]
        s = (a.sum(1) * w).sum(-1)                                            # (C,)
        out[i] = logit + lam * s
        if order == 'no_self':
            insert(c, q, h)
    return out


def acc(pred, y):
    return float((pred == y).float().mean())


def main():
    parser = get_parser()
    parser.add_argument('--ckpt', required=True, help='trained model (last.pt / best.pt of main_plus.py)')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--lam', type=float, nargs='+', default=[5.0])
    parser.add_argument('--scale', type=float, nargs='+', default=[50.0])
    parser.add_argument('--orders', nargs='+', default=['paper', 'no_self'], choices=['paper', 'no_self'])
    parser.add_argument('--n-seeds', type=int, default=5)
    p = parser.parse_args()
    with open(p.config) as f:
        cfg = yaml.safe_load(f)
    bad = [k for k in cfg if k not in vars(p)]
    assert not bad, f'unknown config keys: {bad}'
    parser.set_defaults(**cfg)
    arg = parser.parse_args()
    init_seed(arg.seed)

    P = TTAProcessor(arg)
    ck = torch.load(arg.ckpt, map_location=P.dev, weights_only=False)
    P.model.load_state_dict(ck['model'])
    P.model.eval()
    P.log(f'checkpoint {arg.ckpt} (epoch {ck.get("epoch")})')

    # ---- test scores and descriptors, once
    zs, Ds, ys = [], [], []
    with torch.no_grad():
        for inp, y, is_cube in P.batches('test', shuffle=False):
            assert is_cube, 'tta_cache.py needs feature_mode: cache'
            out = P.forward(inp, True, P.cls_all)
            zs.append(P.model.s_neu(out).float())
            Ds.append(descriptors(inp))
            ys.append(y)
    Z, D, y = torch.cat(zs), torch.cat(Ds), torch.cat(ys)
    P.log(f'test: {len(y)} samples, scores {tuple(Z.shape)}, descriptors {tuple(D.shape)}')

    uns = P.unseen_t
    u_mask = torch.isin(y, uns)
    seen_col = torch.isin(P.cls_all, P.cls_seen).float()

    # ---- ZSL: unseen samples, unseen classes
    Zu, Du, yu = Z[u_mask][:, uns], D[u_mask], y[u_mask]
    yu_idx = (yu[:, None] == uns[None]).float().argmax(1)
    base_zsl = acc(Zu.argmax(1), yu_idx)
    # ---- GZSL base
    g0, S0, U0, H0 = best_gamma(Z, y, P.cls_all, P.cls_seen)
    Zc = Z - g0 * seen_col
    P.log(f'base   ZSL {base_zsl:.4f} | GZSL S {S0:.4f} U {U0:.4f} H {H0:.4f} (gamma {g0:.3f})')

    rows = [dict(method='base', order='-', scale=None, lam=None, zsl=base_zsl, zsl_std=0.0, S=S0, U=U0, H=H0, H_std=0.0)]
    for order, scale, lam in itertools.product(arg.orders, arg.scale, arg.lam):
        zsl, S, U, H = [], [], [], []
        t0 = time.time()
        for seed in range(arg.n_seeds):
            g = torch.Generator().manual_seed(1000 + seed)
            fu = stream(Zu, Du, arg.K, arg.beta, lam, scale, order, torch.randperm(len(yu), generator=g))
            zsl.append(acc(fu.argmax(1), yu_idx))
            fa = stream(Zc, D, arg.K, arg.beta, lam, scale, order, torch.randperm(len(y), generator=g))
            _, s_, u_, h_ = best_gamma(fa, y, P.cls_all, P.cls_seen)
            S.append(s_); U.append(u_); H.append(h_)
        r = dict(method='skeleton-cache', order=order, scale=scale, lam=lam, zsl=float(np.mean(zsl)),
                 zsl_std=float(np.std(zsl)), S=float(np.mean(S)), U=float(np.mean(U)), H=float(np.mean(H)),
                 H_std=float(np.std(H)))
        rows.append(r)
        P.log(f'cache  order {order:7s} scale {scale:g} lam {lam:g} beta {arg.beta:g} K {arg.K}: ZSL {r["zsl"]:.4f} '
              f'+- {r["zsl_std"]:.4f} | H {r["H"]:.4f} +- {r["H_std"]:.4f} (S {r["S"]:.4f} U {r["U"]:.4f}) '
              f'[{time.time() - t0:.0f}s, {arg.n_seeds} seeds]')
    json.dump(dict(ckpt=arg.ckpt, split=arg.split, K=arg.K, beta=arg.beta, rows=rows),
              open(f'{arg.work_dir}/tta_cache.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
