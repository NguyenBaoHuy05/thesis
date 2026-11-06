"""Compare the predictions saved by main.py at test time (epoch<n>_test_outputs.npz: output, label and the
eval_outputs of the model, e.g. k0 / concept_match for Stage1Model).

  python tools/compare_outputs.py <work_dir>/epoch30_test_outputs.npz [--keys output concept_match k0]

Per output: Top1. Per pair (a, b): agree (same prediction), both wrong, both wrong with the same wrong class,
only a right, only b right, and the oracle (either right). If concept_match is wrong on the same samples and with
the same wrong class as the action decoder (output), the concepts are only a code of the class.
"""

import argparse
import itertools

import numpy as np


def pct(x, n):
    return f'{100 * x / n:6.2f}% ({x})'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--keys', nargs='+', help='outputs to compare (default: every array but label)')
    args = parser.parse_args()
    data = np.load(args.path)
    label = data['label']
    keys = args.keys or [k for k in data.files if k != 'label']
    pred = {k: data[k].argmax(1) for k in keys}
    right = {k: pred[k] == label for k in keys}
    n = len(label)

    print(f'{n} test samples')
    for k in keys:
        print(f'  {k:>14}: Top1 {pct(right[k].sum(), n)}')
    for a, b in itertools.combinations(keys, 2):
        both_wrong = ~right[a] & ~right[b]
        print(f'\n{a} vs {b}')
        print(f'  agree (same prediction)        {pct((pred[a] == pred[b]).sum(), n)}')
        print(f'  both wrong                     {pct(both_wrong.sum(), n)}')
        print(f'    same wrong class             {pct((both_wrong & (pred[a] == pred[b])).sum(), n)}')
        print(f'  only {a} right {"":<{max(0, 16 - len(a))}}{pct((right[a] & ~right[b]).sum(), n)}')
        print(f'  only {b} right {"":<{max(0, 16 - len(b))}}{pct((right[b] & ~right[a]).sum(), n)}')
        print(f'  oracle (either right)          {pct((right[a] | right[b]).sum(), n)}')
        wrong_a = (~right[a]).sum()
        if wrong_a:
            print(f'  of the {wrong_a} errors of {a}: {b} also wrong {100 * both_wrong.sum() / wrong_a:.1f}%, '
                  f'with the same class {100 * (both_wrong & (pred[a] == pred[b])).sum() / wrong_a:.1f}%')
