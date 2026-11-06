"""Step 1 of direction 1 (KG_RETRIEVAL_PLAN/CONCEPT_GRAPH/CONCEPT_GRAPH_PROGRESS.md): one small expert per confusable
pair on the S = 1 cache, applied where K0 hesitates between the two classes of the pair.

  python tools/pair_experts.py --cache-dir /content/jcache --out <folder> [--cache-name ntu120_xsub_joint]
      [--epochs 30] [--seed 0] [--device cuda]

Per pair (i, j) with a non-empty gate, two experts with the same recipe:
    A (knowledge): reads only the joints of the concepts that tell i and j apart, (M_j minus M_i) or (M_i minus M_j),
                   through their body-group masks (model/concept_logic.py: concept_joint_mask);
    B (control):   reads every joint.
    expert: token (actor, joint) of the cached l9 map -> Linear 512*S -> 64, LayerNorm, GELU -> mean over the real
            actors and the allowed joints -> Linear 64 -> 2.
Train: train samples of i and j, Gaussian feature noise (1.0 x channel std), Adam 1e-3, `epochs`, batch 128, seed 0.
Gate (test): samples whose K0 top-2 is {i, j} (no label used); the expert picks i or j there, K0 decides elsewhere.
Binary accuracy is reported on the gate samples whose label is i or j.
Pre-registered: pass if Top1 (K0 + A) >= 87.93; the mask helps if Top1 (K0 + A) - Top1 (K0 + B) >= 0.1.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from model.concept_graph import confusable_pairs, read_matrix
from model.concept_logic import concept_joint_mask, load_concepts
from model.concept_schema import NTU_ACTION_NAMES
from train_concept_logic import channel_std, load_split

PASS_TOP1, MASK_GAIN = 0.8793, 0.001


class PairExpert(nn.Module):
    def __init__(self, in_dim, joints, d=64):
        super().__init__()
        self.register_buffer('joints', joints.float())                     # [V]
        self.token = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU())
        self.out = nn.Linear(d, 2)

    def forward(self, x, valid):
        """x [n, M, C, S, V], valid [n, M] -> [n, 2]."""
        n, M, C, S, V = x.shape
        h = self.token(x.permute(0, 1, 4, 2, 3).reshape(n, M, V, C * S))   # [n, M, V, d]
        w = valid.float()[:, :, None] * self.joints[None, None, :]          # [n, M, V]
        h = (h * w[..., None]).sum((1, 2)) / w.sum((1, 2)).clamp_min(1)[:, None]
        return self.out(h)


def train_expert(joints, x, valid, y, std, args, device):
    torch.manual_seed(args.seed)
    model = PairExpert(x.shape[2] * x.shape[3], joints).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(y), device=device).split(128):
            xb = x[idx].float()
            xb = xb + args.noise * std * torch.randn_like(xb)
            loss = loss_fn(model(xb, valid[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-name', default='ntu120_xsub_joint')
    parser.add_argument('--matrix', default='annotations/concepts/ntu_action_concept_v2_4.csv')
    parser.add_argument('--metadata', default='annotations/concepts/ntu_concept_metadata_v2_4.csv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--noise', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.cache_dir, args.out = os.path.abspath(args.cache_dir), os.path.abspath(args.out)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    M, names, meta = load_concepts(args.matrix, args.metadata, NTU_ACTION_NAMES)
    joint_mask = concept_joint_mask(names, meta)                             # [K, V]
    actions, _, rows = read_matrix(args.matrix)
    pairs = confusable_pairs(actions, rows)

    feat_tr, valid_tr, _, y_tr = load_split(args.cache_dir, args.cache_name, 'train', device)
    feat_te, valid_te, fc_te, y_te = load_split(args.cache_dir, args.cache_name, 'test')
    valid_tr, y_tr = valid_tr.to(device), y_tr.to(device)
    std = channel_std(feat_tr, valid_tr)[None, None, :, None, None]
    top2 = fc_te.topk(2, 1).indices
    k0_pred = top2[:, 0].clone()
    pred = {'A': k0_pred.clone(), 'B': k0_pred.clone()}
    print(f'train {tuple(feat_tr.shape)}, test {tuple(feat_te.shape)}, K0 Top1 '
          f'{100 * (k0_pred == y_te).float().mean():.2f}%, {len(pairs)} pairs')

    report = []
    for i, j in pairs:
        gate = ((top2[:, 0] == i) & (top2[:, 1] == j)) | ((top2[:, 0] == j) & (top2[:, 1] == i))
        if not gate.any():
            continue
        diff = (M[j].bool() & ~M[i].bool()) | (M[i].bool() & ~M[j].bool())
        joints_a = joint_mask[diff].any(0) if diff.any() else torch.ones(joint_mask.shape[1], dtype=torch.bool)
        sel = (y_tr == i) | (y_tr == j)
        x, v, y = feat_tr[sel], valid_tr[sel], (y_tr[sel] == j).long()
        g_idx = gate.nonzero().squeeze(1)
        xg = torch.from_numpy(np.array(feat_te[g_idx.numpy()])).to(device).float()
        vg = valid_te[g_idx].to(device)
        in_pair = (y_te[g_idx] == i) | (y_te[g_idx] == j)
        row = dict(pair=[actions[i], actions[j]], gate=int(gate.sum()), gate_in_pair=int(in_pair.sum()),
                   train=int(sel.sum()), joints_a=int(joints_a.sum()),
                   k0_binary=float((k0_pred[g_idx] == y_te[g_idx])[in_pair].float().mean()))
        for name, joints in (('A', joints_a), ('B', torch.ones_like(joints_a))):
            model = train_expert(joints.to(device), x, v, y, std, args, device)
            with torch.no_grad():
                choice = model(xg, vg).argmax(1).cpu()
            pred[name][g_idx] = torch.where(choice == 1, torch.tensor(j), torch.tensor(i))
            row[f'{name}_binary'] = float((pred[name][g_idx] == y_te[g_idx])[in_pair].float().mean())
        report.append(row)
        print(f"{row['gate']:5d} gate ({row['joints_a']:2d} joints)  K0 {100 * row['k0_binary']:5.1f}%  "
              f"A {100 * row['A_binary']:5.1f}%  B {100 * row['B_binary']:5.1f}%  {actions[i]} / {actions[j]}",
              flush=True)

    top1 = {name: float((p == y_te).float().mean()) for name, p in (('K0', k0_pred), ('K0+A', pred['A']),
                                                                      ('K0+B', pred['B']))}
    fixed_broken = {name: dict(fixed=int(((pred[name] == y_te) & (k0_pred != y_te)).sum()),
                               broken=int(((pred[name] != y_te) & (k0_pred == y_te)).sum())) for name in pred}
    verdict = dict(pass_=top1['K0+A'] >= PASS_TOP1, mask_helps=top1['K0+A'] - top1['K0+B'] >= MASK_GAIN)
    with open(os.path.join(args.out, 'pair_experts.json'), 'w', encoding='utf-8') as handle:
        json.dump(dict(args=vars(args), top1=top1, fixed_broken=fixed_broken, verdict=verdict, pairs=report),
                  handle, indent=2)
    print('\n' + ', '.join(f'{k} {100 * v:.2f}%' for k, v in top1.items()))
    print(', '.join(f"{k}: fixed {v['fixed']} / broken {v['broken']}" for k, v in fixed_broken.items()))
    print(f"pass (K0+A >= 87.93): {verdict['pass_']}; knowledge mask helps (A - B >= 0.1): {verdict['mask_helps']}")
