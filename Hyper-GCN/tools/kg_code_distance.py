"""Test the ECOC hypothesis of A1_PROGRESS.md (P3 gate): VERIFY reads the rows W[c] as class
codes, and the real KG gives the classes K0 confuses nearly identical codes, while a shuffled
KG gives them distant ones.

  python tools/kg_code_distance.py --cache-dir <folder> [--cache-name ntu120_xsub]
      [--kg annotations/action_kg.json] [--shuffled annotations/kg_shuffled/*.json]
      [--top 25] [--out kg_code_distance.json]

Confusion comes from the cached K0 fc_logits on the test split: pair count = C[i, j] + C[j, i].
Distances between the signed rows of W (has +w/2, lacks -w/2, unknown 0, as in model/kg.py):

  l1       sum |W[i] - W[j]|: what VERIFY's v_i - v_j is built from
  differ   number of attributes with W[i] != W[j] (Hamming)
  opposed  number of attributes that one class has and the other lacks

Reported per KG: the mean of each distance over the top confused pairs and over all pairs, how
many top pairs have l1 = 0 (VERIFY cannot separate them at all), and the Spearman correlation
between pair confusion and l1 over all pairs. Prediction before running: real KG smaller on
the top pairs and negative correlation; shuffled KGs close to the all-pair mean and ~0.
"""

import argparse
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.kg import action_names, kg_tensors, load_kg


def confusion_pairs(cache_dir, name):
    with np.load(os.path.join(cache_dir, f'{name}_test.npz')) as data:
        fc_logits, labels = data['fc_logits'], data['label']
    num_class = fc_logits.shape[1]
    predicted = fc_logits.argmax(1)
    matrix = np.zeros((num_class, num_class), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    pairs = list(itertools.combinations(range(num_class), 2))
    counts = np.array([matrix[i, j] + matrix[j, i] for i, j in pairs])
    return pairs, counts, float((predicted == labels).mean())


def distances(W, pairs):
    i, j = np.array(pairs).T
    diff = W[i] - W[j]
    return dict(l1=np.abs(diff).sum(1), differ=(diff != 0).sum(1),
                opposed=((W[i] * W[j]) < 0).sum(1))


def average_ranks(x):
    """Ranks 1..n with ties sharing their mean rank (for Spearman with many zero counts)."""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    sorted_x = x[order]
    start = 0
    for end in range(1, len(x) + 1):
        if end == len(x) or sorted_x[end] != sorted_x[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    return ranks


def spearman(x, y):
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def summarize(kg_json, pairs, counts, top):
    W = kg_tensors(kg_json).W.numpy()
    d = distances(W, pairs)
    order = np.argsort(-counts, kind='mergesort')[:top]
    return dict(
        top={k: float(v[order].mean()) for k, v in d.items()},
        all={k: float(v.mean()) for k, v in d.items()},
        top_l1_zero=int((d['l1'][order] == 0).sum()),
        spearman_confusion_l1=spearman(counts, d['l1']),
        per_pair=[dict(pair=[int(pairs[p][0]), int(pairs[p][1])], confusions=int(counts[p]),
                       **{k: float(v[p]) for k, v in d.items()}) for p in order],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--cache-name', default='ntu120_xsub')
    parser.add_argument('--kg', default=os.path.join(repo, 'annotations', 'action_kg.json'))
    parser.add_argument('--shuffled', default=os.path.join(repo, 'annotations', 'kg_shuffled', '*.json'),
                        help='glob of shuffled KGs')
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--out', help='optional JSON report')
    args = parser.parse_args()

    pairs, counts, top1 = confusion_pairs(args.cache_dir, args.cache_name)
    real = load_kg(args.kg)
    names = action_names(real)
    reports = {'real': summarize(real, pairs, counts, args.top)}
    for path in sorted(glob.glob(args.shuffled)):
        kg = load_kg(path)
        if kg.get('shuffled', {}).get('source_hash') != real['hash']:
            raise ValueError(f'{path} is not a shuffle of {args.kg}')
        reports[f"shuffle_s{kg['shuffled']['seed']}"] = summarize(kg, pairs, counts, args.top)

    print(f'K0 test top1 {100 * top1:.2f}%; top {args.top} of {len(pairs)} class pairs by confusion '
          f'({int(counts[np.argsort(-counts)[:args.top]].sum())} of {int(counts.sum())} errors)')
    print(f"{'KG':>12} | {'top l1':>7} {'all l1':>7} | {'top differ':>10} {'all differ':>10} | "
          f"{'top opposed':>11} {'all opposed':>11} | {'top l1=0':>8} | {'spearman':>8}")
    for name, r in reports.items():
        print(f"{name:>12} | {r['top']['l1']:7.2f} {r['all']['l1']:7.2f} | {r['top']['differ']:10.2f} "
              f"{r['all']['differ']:10.2f} | {r['top']['opposed']:11.2f} {r['all']['opposed']:11.2f} | "
              f"{r['top_l1_zero']:8d} | {r['spearman_confusion_l1']:8.3f}")

    print(f'\ntop {args.top} confused pairs, l1 distance (real KG vs each shuffle):')
    shuffles = [name for name in reports if name != 'real']
    for rank, row in enumerate(reports['real']['per_pair']):
        i, j = row['pair']
        others = ' '.join(f"{reports[s]['per_pair'][rank]['l1']:5.1f}" for s in shuffles)
        print(f"  {row['confusions']:4d}  {names[i]:>28} / {names[j]:<28} real {row['l1']:5.1f} | {others}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(dict(names=names, k0_top1=top1, reports=reports), handle, indent=2)
        print(f'Saved {args.out}')
