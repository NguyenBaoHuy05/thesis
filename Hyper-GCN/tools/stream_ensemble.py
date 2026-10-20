"""Step 0 of direction 2 (concept-level fusion of the Hyper-GCN streams): what the four streams give together at the
score level, the bar a concept-level fusion has to beat. Numpy only, reads the test caches of
tools/cache_joint_features.py (fc_logits of each stream, same sample order).

  python tools/stream_ensemble.py --cache-dir /content/jcache
      [--names j=ntu120_xsub_joint b=ntu120_xsub_bone jm=ntu120_xsub_joint_motion bm=ntu120_xsub_bone_motion]

Reports: Top1 per stream; score ensembles (equal weights, and ensemble.py's alpha for large / ntu120/xsub
[1, 1.5, 0.7, 0.3], chosen by the Hyper-GCN authors); oracle (any stream right); of the errors of the joint
stream (K0), how many another stream gets right; and the same on the 634 pair-ceiling samples of direction 1.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.concept_graph import confusable_pairs, read_matrix

ORDER = ('j', 'b', 'jm', 'bm')                          # ensemble.py: joint, bone, joint motion, bone motion
ALPHA = (1, 1.5, 0.7, 0.3)                              # ensemble.py: ensemble_alpha['large']['ntu120/xsub']
DEFAULT = dict(j='ntu120_xsub_joint', b='ntu120_xsub_bone', jm='ntu120_xsub_joint_motion',
               bm='ntu120_xsub_bone_motion')


def pct(x, n):
    return f'{100 * x / n:6.2f}% ({x})'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--names', nargs='+', help='stream=cache_name, default: the four ntu120 xsub caches')
    args = parser.parse_args()
    names = dict(DEFAULT, **dict(s.split('=', 1) for s in (args.names or [])))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    logits, label, index = {}, None, None
    for s in ORDER:
        with np.load(os.path.join(args.cache_dir, f'{names[s]}_test.npz')) as data:
            logits[s] = data['fc_logits'].astype(np.float64)
            if label is None:
                label, index = data['label'], data['index']
            elif not (np.array_equal(label, data['label']) and np.array_equal(index, data['index'])):
                sys.exit(f'{names[s]}: test samples are not in the same order as {names["j"]}')
    n = len(label)
    right = {s: logits[s].argmax(1) == label for s in ORDER}
    for s in ORDER:
        print(f'{s:>3} {names[s]:<28} Top1 {pct(right[s].sum(), n)}')

    def ensemble(weights, streams=ORDER):
        return sum(w * logits[s] for w, s in zip(weights, streams)).argmax(1) == label
    alpha = ALPHA
    print('\nscore ensembles')
    print(f'  j + b, equal                {pct(ensemble((1, 1), ("j", "b")).sum(), n)}')
    print(f'  4 streams, equal            {pct(ensemble((1, 1, 1, 1)).sum(), n)}')
    best = ensemble(alpha)
    print(f'  4 streams, alpha {alpha}  {pct(best.sum(), n)}   <- bar of direction 2')
    oracle = np.any([right[s] for s in ORDER], 0)
    print(f'  oracle (any stream right)   {pct(oracle.sum(), n)}')

    wrong_j = ~right['j']
    print(f'\nof the {wrong_j.sum()} errors of the joint stream (K0):')
    for s in ORDER[1:]:
        print(f'  {s:>3} right {pct((wrong_j & right[s]).sum(), n)}')
    print(f'  any other stream right   {pct((wrong_j & oracle).sum(), n)}')
    print(f'  4-stream ensemble right  {pct((wrong_j & best).sum(), n)}; ensemble breaks {(right["j"] & ~best).sum()}')

    actions, _, rows = read_matrix(os.path.join(repo, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv'))
    pair_set = {frozenset(p) for p in confusable_pairs(actions, rows)}
    order = np.argsort(-logits['j'], 1)
    ceiling = wrong_j & (order[:, 1] == label) & np.array([frozenset(p) in pair_set for p in order[:, :2]])
    print(f'\non the {ceiling.sum()} pair-ceiling samples of direction 1 (K0 wrong, label 2nd, top-2 a pair):')
    for s in ORDER[1:]:
        print(f'  {s:>3} right {(ceiling & right[s]).sum()}')
    print(f'  any other stream right {(ceiling & oracle).sum()}, 4-stream ensemble right {(ceiling & best).sum()}')

    print('\nerror overlap between streams (both wrong / errors of the first)')
    for a, b in itertools.combinations(ORDER, 2):
        both = (~right[a] & ~right[b]).sum()
        print(f'  {a:>2} / {b:<2}: {100 * both / max((~right[a]).sum(), 1):.1f}%')
