"""Train / evaluate the concept-logic head (model/concept_logic.py) on cached l9 joint features.

  python train_concept_logic.py --config config/concept/cl_full.yaml [--cache-dir <folder>]
      [--work-dir <folder>] [--matrix <csv>] [--seed 0] [--device cuda]

Reads <cache_dir>/<cache_name>_{train,test}_feat.npy + .npz (tools/cache_joint_features.py). K2 = the
same config with --matrix annotations/concepts/shuffled/ntu_action_concept_v2_4_shuffle_s<seed>.csv.
No validation split (as train_kg_head.py): the recipe is fixed in the config, test is scored once.

Reported (kg_metrics.json in work_dir):
- head Top1/Top5 against K0 (cached fc_logits), samples fixed / broken against K0;
- concept AUC (macro over concepts) of c against the class row of the matrix used for training and
  against the real matrix (--real-matrix, default the config's): a shuffled matrix that still reaches a
  high AUC against itself but ~0.5 against the real one is fitting a code, not the concepts;
- K5 on the test split: zero the concept that supports the predicted class most (has, largest c) and
  count changed predictions, against zeroing a random concept the predicted class does not have.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from model.concept_logic import ConceptLogicHead, concept_joint_mask, concept_loss, load_concepts
from model.concept_schema import NTU_ACTION_NAMES
from model.kg_retrieval import topk_metrics

EVAL_BATCH = 512


def load_split(cache_dir, name, split, device=None, chunk=4096):
    """feat [N, M, C, S, V] float16 (on `device` if given, else a CPU memmap), valid, fc_logits, label."""
    feat = np.load(os.path.join(cache_dir, f'{name}_{split}_feat.npy'), mmap_mode='r')
    with np.load(os.path.join(cache_dir, f'{name}_{split}.npz')) as data:
        rest = {key: torch.from_numpy(data[key]) for key in ('valid', 'fc_logits', 'label')}
    if device is not None:
        out = torch.empty(feat.shape, dtype=torch.float16, device=device)
        for i in range(0, len(feat), chunk):
            out[i:i + chunk] = torch.from_numpy(np.array(feat[i:i + chunk])).to(device)
        feat = out
    return feat, rest['valid'].bool(), rest['fc_logits'].float(), rest['label'].long()


def batches(feat, idx, device):
    block = feat[idx] if torch.is_tensor(feat) else torch.from_numpy(np.array(feat[idx.numpy()]))
    return block.to(device).float()


def channel_std(feat, valid, chunk=2048):
    """Per-channel std of the train features over real actors, segments and joints: [C]."""
    total = torch.zeros(feat.shape[2], dtype=torch.float64, device=feat.device)
    total_sq, count = torch.zeros_like(total), 0
    for i in range(0, len(feat), chunk):
        x = feat[i:i + chunk].float()                                          # [n, M, C, S, V]
        w = valid[i:i + chunk].to(x.device)[:, :, None, None, None].float()
        total += (x * w).sum((0, 1, 3, 4)).double()
        total_sq += (x * x * w).sum((0, 1, 3, 4)).double()
        count += w.sum().item() * x.shape[3] * x.shape[4]
    mean = total / count
    return (total_sq / count - mean ** 2).clamp_min(0).sqrt().float()


def concept_auc(c, target):
    """Macro AUC over concepts (columns with both classes present); c, target [N, K]."""
    aucs = []
    for j in range(c.shape[1]):
        t = target[:, j]
        n_pos = int(t.sum())
        n_neg = len(t) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = c[:, j].argsort().argsort().double() + 1
        aucs.append(((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())
    return float(np.mean(aucs)), aucs


@torch.no_grad()
def predict(head, feat, valid, device):
    head.eval()
    cs, logits = [], []
    for i in range(0, len(valid), EVAL_BATCH):
        idx = torch.arange(i, min(i + EVAL_BATCH, len(valid)))
        out = head(batches(feat, idx, device), valid[idx].to(device))
        cs.append(out['c'].cpu())
        logits.append(out['logits'].cpu())
    return torch.cat(cs), torch.cat(logits)


@torch.no_grad()
def intervention(head, c, generator):
    """K5: change rate of the prediction when the strongest supporting concept of the predicted class
    is set to 0, against setting a random concept outside its has-set to 0."""
    M = head.M.cpu()
    pred = head.logits(c.to(head.M.device)).argmax(1).cpu()
    rows = torch.arange(len(c))
    has = M[pred].bool()
    support = torch.where(has, c, torch.full_like(c, -1)).argmax(1)
    weights = (~has).float()
    other = torch.multinomial(weights, 1, generator=generator).squeeze(1)
    out = {}
    for name, j in (('drop_support', support), ('drop_random_absent', other)):
        c2 = c.clone()
        c2[rows, j] = 0.0
        out[name] = (head.logits(c2.to(head.M.device)).argmax(1).cpu() != pred).float().mean().item()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--work-dir')
    parser.add_argument('--matrix', help='overrides matrix (K2: a shuffled matrix)')
    parser.add_argument('--real-matrix', help='matrix the concept AUC is also reported against (default: config matrix)')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--feature-noise', type=float, help='overrides feature_noise')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    real_matrix = args.real_matrix or config['matrix']
    for key in ('cache_dir', 'work_dir', 'matrix', 'seed', 'feature_noise'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    os.makedirs(config['work_dir'], exist_ok=True)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    device = torch.device(args.device)

    M, names, meta = load_concepts(config['matrix'], config['metadata'], NTU_ACTION_NAMES)
    M_real, real_names, _ = load_concepts(real_matrix, config['metadata'], NTU_ACTION_NAMES)
    if real_names != names:
        raise ValueError('real and training matrices have different concept columns')
    joint_mask = concept_joint_mask(names, meta)

    feat_train, valid_train, _, y_train = load_split(config['cache_dir'], config['cache_name'], 'train', device)
    feat_test, valid_test, fc_test, y_test = load_split(config['cache_dir'], config['cache_name'], 'test')
    N, n_person, C, S, V = feat_train.shape
    print(f'train feat {tuple(feat_train.shape)}, test feat {tuple(feat_test.shape)}, '
          f'{len(names)} concepts from {config["matrix"]}')

    head_args = dict(config.get('head_args', {}))
    decoder_args = dict(dict(num_person=n_person, max_frames=S), **head_args.pop('decoder_args', {}))
    head = ConceptLogicHead(C, M, joint_mask, decoder_args=decoder_args, **head_args).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
    steps = config['epochs'] * ((N + config['batch_size'] - 1) // config['batch_size'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    M_dev, y_train_dev, valid_train_dev = head.M, y_train.to(device), valid_train.to(device)
    # Gaussian noise on the train features, in units of each channel's std (train only, as feature_noise
    # in train_kg_head.py): the head reaches 100% train top1 on clean cached features.
    noise = config.get('feature_noise', 0.0)
    std = channel_std(feat_train, valid_train_dev)[None, None, :, None, None] if noise else None

    history = []
    for epoch in range(config['epochs']):
        head.train()
        order = torch.randperm(N, device=device)
        sums, correct = dict(loss=0.0, ce=0.0, bce=0.0), 0
        for start in range(0, N, config['batch_size']):
            idx = order[start:start + config['batch_size']]
            x = feat_train[idx].float()
            if noise:
                x = x + noise * std * torch.randn_like(x)
            out = head(x, valid_train_dev[idx])
            loss, terms = concept_loss(out, y_train_dev[idx], M_dev, beta=config['loss']['beta'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            sums['loss'] += loss.item() * len(idx)
            for key, value in terms.items():
                sums[key] += value * len(idx)
            correct += (out['logits'].argmax(1) == y_train_dev[idx]).sum().item()
        row = dict(epoch=epoch + 1, **{k: v / N for k, v in sums.items()}, train_top1=correct / N,
                   tau=head.tau.item())
        history.append(row)
        print(f"epoch {epoch + 1}/{config['epochs']}: loss {row['loss']:.4f} (ce {row['ce']:.4f}, "
              f"bce {row['bce']:.4f}), train top1 {100 * row['train_top1']:.2f}%, tau {row['tau']:.4f}")

    c_test, logits = predict(head, feat_test, valid_test, device)
    pred, k0 = logits.argmax(1), fc_test.argmax(1)
    auc_own, aucs_own = concept_auc(c_test, M[y_test])
    auc_real, _ = concept_auc(c_test, M_real[y_test])
    groups = {}
    for name, a in zip(names, aucs_own):
        groups.setdefault(meta[name]['group'], []).append(a)
    metrics = dict(
        config=config, real_matrix=real_matrix,
        head=topk_metrics(logits, y_test, ks=(1, 5)), k0_fc_logits=topk_metrics(fc_test, y_test, ks=(1, 5)),
        vs_k0=dict(fixed=int(((pred == y_test) & (k0 != y_test)).sum()),
                   broken=int(((pred != y_test) & (k0 == y_test)).sum())),
        concept_auc=auc_own, concept_auc_vs_real=auc_real,
        concept_auc_by_group={g: float(np.mean(v)) for g, v in groups.items()},
        concepts_auc_below_0_6=[n for n, a in zip(names, aucs_own) if a < 0.6],
        k5=intervention(head, c_test, torch.Generator().manual_seed(config['seed'])),
        tau=head.tau.item(),
    )
    torch.save(head.state_dict(), os.path.join(config['work_dir'], 'head.pt'))
    np.save(os.path.join(config['work_dir'], 'test_concepts.npy'), c_test.numpy().astype(np.float16))
    with open(os.path.join(config['work_dir'], 'history.json'), 'w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
    with open(os.path.join(config['work_dir'], 'kg_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    for key in ('head', 'k0_fc_logits'):
        print(f'{key:>13}: ' + ', '.join(f'{k} {100 * v:.2f}%' for k, v in metrics[key].items()))
    print(f"fixed / broken vs K0: {metrics['vs_k0']['fixed']} / {metrics['vs_k0']['broken']}")
    print(f"concept AUC vs training matrix {auc_own:.3f}, vs real matrix {auc_real:.3f}; by group: "
          + ', '.join(f'{g} {a:.3f}' for g, a in metrics['concept_auc_by_group'].items()))
    print(f"{len(metrics['concepts_auc_below_0_6'])} concepts with AUC < 0.6: "
          + ', '.join(metrics['concepts_auc_below_0_6']))
    print(f"K5: prediction changes when the strongest supporting concept is zeroed "
          f"{100 * metrics['k5']['drop_support']:.2f}%, a random absent concept "
          f"{100 * metrics['k5']['drop_random_absent']:.2f}%")
