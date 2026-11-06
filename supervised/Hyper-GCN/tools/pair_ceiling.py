"""Step 0 of the pair-expert test (direction 1, KG_RETRIEVAL_PLAN/CONCEPT_GRAPH/CONCEPT_GRAPH_PROGRESS.md): how many
errors of K0 a perfect expert on the confusable pairs could fix. Numpy only, no training.

  python tools/pair_ceiling.py <npz> [--key k0] [--matrix annotations/concepts/ntu_action_concept_v2_4.csv]

<npz>: main.py's epoch<n>_test_outputs.npz (K0 under `k0`) or the cache's ntu120_xsub_joint_test.npz (`fc_logits`).
Pairs: the 58 confusable pairs of model/concept_graph.py (from M and the class names only, never the test split).

Gate of a pair (i, j): test samples whose K0 top-2 is {i, j}. On the gate an expert decides i or j; it can fix the
samples whose label is K0's top-2 (the ceiling). Pre-registered: ceiling < 300 samples (0.6%) -> stop direction 1.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.concept_graph import confusable_pairs, read_matrix

STOP_BELOW = 300


def pct(x, n):
    return f'{100 * x / n:.2f}% ({x})'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--key', help='K0 logits in the npz (default: k0, else fc_logits)')
    parser.add_argument('--matrix', default='annotations/concepts/ntu_action_concept_v2_4.csv')
    parser.add_argument('--top', type=int, default=15, help='pairs listed')
    args = parser.parse_args()
    data = np.load(args.path)
    key = args.key or ('k0' if 'k0' in data.files else 'fc_logits')
    logits, label = data[key], data['label']
    n = len(label)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matrix = args.matrix if os.path.isfile(args.matrix) else os.path.join(repo, args.matrix)
    actions, _, rows = read_matrix(matrix)
    pairs = confusable_pairs(actions, rows)
    pair_set = {frozenset(p) for p in pairs}

    order = np.argsort(-logits, 1)
    top1, top2 = order[:, 0], order[:, 1]
    wrong = top1 != label
    in_top2 = wrong & (top2 == label)
    in_top5 = wrong & (order[:, :5] == label[:, None]).any(1)
    gated = np.array([frozenset((a, b)) in pair_set for a, b in zip(top1, top2)])
    fixable = in_top2 & gated

    print(f'{key}: {n} test samples, Top1 {pct((~wrong).sum(), n)}, {len(pairs)} confusable pairs')
    print(f'errors of K0                              {pct(wrong.sum(), n)}')
    print(f'  label in K0 top-2                       {pct(in_top2.sum(), n)}   (any top-2 reranker)')
    print(f'  label in K0 top-5                       {pct(in_top5.sum(), n)}')
    print(f'  label in top-2 AND top-2 is a pair      {pct(fixable.sum(), n)}   <- ceiling of direction 1')
    print(f'    = {100 * fixable.sum() / max(wrong.sum(), 1):.1f}% of the errors; Top1 if all fixed '
          f'{100 * ((~wrong).sum() + fixable.sum()) / n:.2f}%')
    print(f'gate (top-2 is a pair)                    {pct(gated.sum(), n)}; K0 right on it '
          f'{100 * (gated & ~wrong).sum() / max(gated.sum(), 1):.2f}%')
    if 'head' in data.files:
        head_right = data['head'].argmax(1) == label
        print(f'"only head right" samples inside the fixable set: {(fixable & head_right).sum()} '
              f'of {(wrong & head_right).sum()}')

    rows_out = []
    for i, j in pairs:
        g = gated & np.isin(top1, (i, j)) & np.isin(top2, (i, j))
        if g.any():
            rows_out.append((int((fixable & g).sum()), int(g.sum()), int((g & ~wrong).sum()), actions[i], actions[j]))
    rows_out.sort(reverse=True)
    print(f'\npairs by fixable samples (top {args.top}): fixable / gate, K0 binary accuracy on the gate')
    for fix, size, right, a, b in rows_out[:args.top]:
        print(f'  {fix:5d} / {size:5d}  K0 {100 * right / size:5.1f}%  {a} / {b}')
    print('\n' + ('STOP direction 1' if fixable.sum() < STOP_BELOW else 'continue: train the pair experts')
          + f' (ceiling {fixable.sum()} vs {STOP_BELOW})')
