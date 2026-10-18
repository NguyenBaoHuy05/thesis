"""Cache the geometric predicates phi (model/predicates.py) for the A1-G VERIFY step (A1-G plan S3).

  python tools/cache_predicates.py --out-dir <folder> [--name ntu120_xsub_pred]
      [--config config/large/nturgbd120-cross-subject/hyper_joint.yaml] [--splits test train]
      [--kg annotations/action_kg.json] [--batch-size 256] [--num-worker 4]

Writes <folder>/<name>_{test,train}.npz with phi [N, P] float32, valid [N, P] bool, names [P],
label [N], index [N], and <name>_summary.json with a per-predicate report (AUC against the KG,
valid fraction, mean / std).

Both splits use the deterministic test transform (`test_feeder_args`: centre crop p_interval 0.95,
no random_rot), the train split only switching `split`. The feature cache has 5 random views of the
train split; phi is one fixed value per sample, so it does not have to reproduce those random crops
(a crop seed depends on the DataLoader worker layout). Rows follow the dataset order, as in
tools/cache_features.py, and train_kg_head.py refuses a predicate cache whose labels differ.

The AUC report is a diagnostic, not evidence of grounding (targets come from class labels): for
each predicate that is a KG attribute, AUC of phi against has (1) / lacks (0) of the sample's class,
on cells the KG knows. AUC well below 0.5 points at a sign error in the predicate definition (e.g.
the forward axis); the plan's G3 gate asks for AUC >= 0.75 on train before a predicate is used.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from model.predicates import PREDICATES, compute


def import_class(name):
    mod_str, _sep, class_str = name.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def auc(score, target):
    """Rank AUC (ties averaged); nan if one class is missing."""
    n_pos = int(target.sum())
    n_neg = len(target) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    sorted_score = score[order]
    start = 0
    for end in range(1, len(score) + 1):
        if end == len(score) or sorted_score[end] != sorted_score[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    return float((ranks[target == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def kg_targets(kg_json):
    """{attribute: (has [C] bool, known [C] bool)} in label order."""
    actions = sorted(kg_json['actions'].values(), key=lambda a: a['id'])
    out = {}
    for attr in kg_json['attributes']:
        has = np.array([attr in a['has'] for a in actions])
        lacks = np.array([attr in a['lacks'] for a in actions])
        out[attr] = (has, has | lacks)
    return out


def report(phi, valid, labels, names, kg_json):
    targets = kg_targets(kg_json)
    rows = {}
    for j, name in enumerate(names):
        ok = valid[:, j]
        row = dict(valid_fraction=float(ok.mean()),
                   mean=float(phi[ok, j].mean()) if ok.any() else None,
                   std=float(phi[ok, j].std()) if ok.any() else None,
                   in_kg=name in targets)
        if name in targets:
            has, known = targets[name]
            cell = ok & known[labels]
            row['auc'] = auc(phi[cell, j], has[labels[cell]].astype(np.int64))
            row['cells'] = int(cell.sum())
        rows[name] = row
    return rows


def run(dataset, batch_size, num_worker):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    phis, valids, labels, indices = [], [], [], []
    for data, label, index in loader:
        phi, valid = compute(data.numpy(), PREDICATES)
        phis.append(phi)
        valids.append(valid)
        labels.append(label.numpy())
        indices.append(np.asarray(index))
        print(f'\r{sum(len(l) for l in labels)}/{len(dataset)}', end='', flush=True)
    print()
    return np.concatenate(phis), np.concatenate(valids), np.concatenate(labels), np.concatenate(indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--name', default='ntu120_xsub_pred')
    parser.add_argument('--config', default='config/large/nturgbd120-cross-subject/hyper_joint.yaml')
    parser.add_argument('--splits', nargs='+', choices=['test', 'train'], default=['test', 'train'])
    parser.add_argument('--kg', default='annotations/action_kg.json')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-worker', type=int, default=4)
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.abspath(args.out_dir)
    config_path = os.path.abspath(args.config if os.path.isfile(args.config) else os.path.join(repo, args.config))
    kg_path = os.path.abspath(args.kg if os.path.isfile(args.kg) else os.path.join(repo, args.kg))
    os.chdir(repo)                       # data_path in the config is relative to the repo root
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    with open(kg_path, encoding='utf-8') as handle:
        kg_json = json.load(handle)

    Feeder = import_class(config['feeder'])
    feeder_args = dict(config['test_feeder_args'])
    if feeder_args.get('bone') or feeder_args.get('vel'):
        raise ValueError('predicates need the joint modality (bone: False, vel: False)')
    summary = dict(config=config_path, kg=kg_path, kg_hash=kg_json['hash'], feeder_args=feeder_args,
                   predicates=list(PREDICATES), splits={})
    for split in sorted(args.splits, key=['test', 'train'].index):
        dataset = Feeder(**dict(feeder_args, split=split))
        phi, valid, labels, index = run(dataset, args.batch_size, args.num_worker)
        path = os.path.join(out_dir, f'{args.name}_{split}.npz')
        np.savez(path, phi=phi, valid=valid, names=np.array(PREDICATES), label=labels.astype(np.int64),
                 index=index.astype(np.int64))
        rows = report(phi, valid, labels, PREDICATES, kg_json)
        summary['splits'][split] = dict(path=path, samples=int(len(labels)), predicates=rows)
        print(f'Saved {path}: phi {phi.shape}')
        for name, row in rows.items():
            flag = '' if not row['in_kg'] else ('  <- check sign' if row['auc'] < 0.4 else
                                                ('  <- weak' if row['auc'] < 0.75 else ''))
            auc_text = f"AUC {row['auc']:.3f} on {row['cells']} cells" if row['in_kg'] else 'not in KG'
            print(f"  {name:>28}: {auc_text}, valid {100 * row['valid_fraction']:.1f}%{flag}")
    with open(os.path.join(out_dir, f'{args.name}_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
