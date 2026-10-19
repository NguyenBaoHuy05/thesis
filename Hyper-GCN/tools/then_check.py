"""Ceiling check for the THEN attributes, before training anything (A1-G, THEN extension).

  python tools/then_check.py --cache-dir <folder> [--cache-name ntu120_xsub] [--pred-name ntu120_xsub_pred_then]
      [--kg annotations/action_kg_then.json]

numpy only. For every KG `opposite` pair on which some THEN attribute is opposed (one action has it, the
other lacks it), on the test split:

- K0 (cached fc_logits): samples of the pair, errors inside the pair (true a, predicted b or the
  reverse), and the binary accuracy argmax(logit_a, logit_b). Binary accuracy near 100% means z already
  knows the order and THEN has nothing left to fix.
- A vote of the opposed predicates, fitted on the TRAIN samples of the pair only:
      vote = sum_j sign(W[a, j]) (phi_j - mid_j) / std_j,   mid_j = mean of the two train class means
  predicts a if vote > 0. Reported for the THEN attributes and, for comparison, for the static
  (non-THEN) predicate attributes opposed in the same pair: accuracy on all test samples of the pair
  and on the K0 errors inside the pair. THEN right well above 50% on those errors is the go signal
  for the model runs; around 50% means order evidence fails where the backbone fails, as the static
  predicates did (gated_pair_check).
"""

import argparse
import json
import os

import numpy as np


def load(cache_dir, name, split):
    with np.load(os.path.join(cache_dir, f'{name}_{split}.npz')) as data:
        return {key: data[key] for key in data.files}


def auc(score, target):
    n_pos, n_neg = int(target.sum()), int((1 - target).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = score.argsort().argsort() + 1.0
    return float((ranks[target == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def kg_matrix(kg, names):
    """W [C, P]: +1 has, -1 lacks, 0 unknown, for predicate names that are KG attributes (else column 0)."""
    actions = sorted(kg['actions'].values(), key=lambda a: a['id'])
    W = np.zeros((len(actions), len(names)))
    for c, action in enumerate(actions):
        for j, name in enumerate(names):
            W[c, j] = 1 if name in action['has'] else -1 if name in action['lacks'] else 0
    return W


def vote(phi, columns, sign, train_a, train_b):
    """Signed z-score vote for class a over the given predicate columns, centred between the class means."""
    total = np.zeros(len(phi))
    for j, s in zip(columns, sign):
        mid = (train_a[:, j].mean() + train_b[:, j].mean()) / 2
        std = np.concatenate([train_a[:, j], train_b[:, j]]).std() + 1e-8
        total += s * (phi[:, j] - mid) / std
    return total


def rate(right, mask):
    n = int(mask.sum())
    return f'{100 * right[mask].mean():5.1f}% (n {n})' if n else '   -    (n 0)'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--cache-name', default='ntu120_xsub')
    parser.add_argument('--pred-name', default='ntu120_xsub_pred_then')
    parser.add_argument('--kg', default='annotations/action_kg_then.json')
    args = parser.parse_args()

    with open(args.kg, encoding='utf-8') as handle:
        kg = json.load(handle)
    feat = load(args.cache_dir, args.cache_name, 'test')
    pred_train, pred_test = load(args.cache_dir, args.pred_name, 'train'), load(args.cache_dir, args.pred_name, 'test')
    names = [str(n) for n in pred_test['names']]
    if not any(n.startswith('then_') for n in names):
        raise ValueError('the predicate cache has no THEN predicate: rerun tools/cache_predicates.py')
    fc = feat['fc_logits'][0] if feat['fc_logits'].ndim == 3 else feat['fc_logits']
    y_test, y_train = pred_test['label'], pred_train['label']
    if not np.array_equal(feat['label'], y_test):
        raise ValueError('predicate cache test rows do not match the feature cache')
    k0 = fc.argmax(1)
    W = kg_matrix(kg, names)
    action_names = [n for n, _ in sorted(kg['actions'].items(), key=lambda item: item[1]['id'])]
    index = {n: i for i, n in enumerate(action_names)}
    in_kg = np.array([n in kg['attributes'] for n in names])
    is_then = np.array([n.startswith('then_') for n in names])

    pairs = sorted({tuple(sorted((index[a], index[o]))) for a, act in kg['actions'].items() for o in act['opposite']})
    totals = {'then': [0, 0], 'static': [0, 0], 'k0_err': 0}
    print(f'{"pair":>48} | K0: n, err in pair, binary acc | vote: all test / on K0 errors in pair')
    for a, b in pairs:
        opposed = (W[a] * W[b] < 0) & in_kg
        if not (opposed & is_then).any():
            continue
        tr_a, tr_b = pred_train['phi'][y_train == a], pred_train['phi'][y_train == b]
        test = (y_test == a) | (y_test == b)
        err = test & (((y_test == a) & (k0 == b)) | ((y_test == b) & (k0 == a)))
        binary = np.where(fc[:, a] > fc[:, b], a, b)
        print(f'{action_names[a] + " / " + action_names[b]:>48} | n {int(test.sum())}, err {int(err.sum())}, '
              f'binary {100 * (binary[test] == y_test[test]).mean():.1f}%')
        totals['k0_err'] += int(err.sum())
        for kind, cols in (('then', np.flatnonzero(opposed & is_then)), ('static', np.flatnonzero(opposed & ~is_then))):
            if len(cols) == 0:
                print(f'{"":>48} | {kind:>6}: no opposed predicate')
                continue
            v = vote(pred_test['phi'], cols, W[a, cols], tr_a, tr_b)
            right = np.where(v > 0, a, b) == y_test
            v_train = vote(np.concatenate([tr_a, tr_b]), cols, W[a, cols], tr_a, tr_b)
            train_acc = (np.where(v_train > 0, a, b) == np.r_[np.full(len(tr_a), a), np.full(len(tr_b), b)]).mean()
            totals[kind][0] += int(right[err].sum())
            totals[kind][1] += int(err.sum())
            aucs = ', '.join(f'{names[j]} {auc(pred_test["phi"][test, j], (y_test[test] == (a if W[a, j] > 0 else b)).astype(float)):.2f}'
                             for j in cols)
            print(f'{"":>48} | {kind:>6}: train {100 * train_acc:5.1f}%, test {rate(right, test)}, '
                  f'K0 errors {rate(right, err)}  [test AUC: {aucs}]')
    print(f'\nK0 errors inside these pairs: {totals["k0_err"]} of {len(y_test)} test samples '
          f'(upper bound on the Top1 gain: +{100 * totals["k0_err"] / len(y_test):.2f})')
    for kind in ('then', 'static'):
        r, n = totals[kind]
        print(f'{kind:>6} vote right on K0 errors inside the pairs: {r} / {n}' + (f' ({100 * r / n:.1f}%)' if n else ''))
