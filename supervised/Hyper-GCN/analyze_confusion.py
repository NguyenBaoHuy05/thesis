#!/usr/bin/env python
"""Per-class accuracy and most confused class pairs from a result file.

main.py writes ``<best checkpoint>_right.txt`` after training: one line per
test sample, ``predicted,true``. The confused pairs show which actions the
model cannot tell apart, e.g. where the KG needs a discriminating attribute.

Usage:
  python analyze_confusion.py RESULT [--compare OTHER_RESULT] [--top 25] [--out report.json]

  --compare  a second result file on the same test set (for example the
             backbone), to list the classes the first run loses most on.
"""

import argparse
import json

import numpy as np

from model.concept_schema import NTU_ACTION_NAMES


def load_results(path):
    data = np.loadtxt(path, delimiter=',', dtype=np.int64, ndmin=2)
    return data[:, 0], data[:, 1]


def confusion_matrix(predicted, true, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def per_class_accuracy(matrix):
    return np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)


def name(k):
    return NTU_ACTION_NAMES[k] if k < len(NTU_ACTION_NAMES) else str(k)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('result')
    parser.add_argument('--compare', default=None)
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    predicted, true = load_results(args.result)
    num_classes = int(max(predicted.max(), true.max())) + 1
    matrix = confusion_matrix(predicted, true, num_classes)
    accuracy = per_class_accuracy(matrix)
    report = dict(result=args.result, samples=int(len(true)),
                  top1=float((predicted == true).mean()))
    print('{}: {} samples, top1 = {:.2%}'.format(args.result, len(true), report['top1']))

    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    worst = np.argsort(accuracy)[:args.top]
    print('\nWorst {} classes (accuracy, most frequent wrong prediction):'.format(args.top))
    report['worst_classes'] = []
    for k in worst:
        confused = int(errors[k].argmax())
        share = errors[k, confused] / max(matrix[k].sum(), 1)
        print('  {:3d} {:28s} {:6.1%}  -> {:28s} {:5.1%}'.format(
            k, name(k), accuracy[k], name(confused), share))
        report['worst_classes'].append(dict(
            class_id=int(k), name=name(k), accuracy=float(accuracy[k]),
            confused_with=name(confused), confused_share=float(share)))

    # Symmetric pair counts: a mistaken for b plus b mistaken for a.
    pairs = errors + errors.T
    upper = np.triu_indices(num_classes, k=1)
    order = np.argsort(pairs[upper])[::-1][:args.top]
    print('\nMost confused pairs (errors both ways):')
    report['confused_pairs'] = []
    for index in order:
        a, b = int(upper[0][index]), int(upper[1][index])
        count = int(pairs[a, b])
        if count == 0:
            break
        print('  {:5d}  {:28s} <-> {:28s} ({} / {})'.format(
            count, name(a), name(b), int(errors[a, b]), int(errors[b, a])))
        report['confused_pairs'].append(dict(
            a=name(a), b=name(b), errors=count,
            a_as_b=int(errors[a, b]), b_as_a=int(errors[b, a])))

    if args.compare:
        other_predicted, other_true = load_results(args.compare)
        if not np.array_equal(other_true, true):
            raise ValueError('--compare must be a result file on the same test samples')
        other_accuracy = per_class_accuracy(confusion_matrix(other_predicted, other_true, num_classes))
        delta = accuracy - other_accuracy
        print('\nLargest losses vs {} (top1 {:.2%}):'.format(
            args.compare, float((other_predicted == other_true).mean())))
        report['losses_vs_compare'] = []
        for k in np.argsort(delta)[:args.top]:
            print('  {:3d} {:28s} {:6.1%} vs {:6.1%}  ({:+.1%})'.format(
                k, name(k), accuracy[k], other_accuracy[k], delta[k]))
            report['losses_vs_compare'].append(dict(
                class_id=int(k), name=name(k), accuracy=float(accuracy[k]),
                compare_accuracy=float(other_accuracy[k]), delta=float(delta[k])))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        print('\nReport written to', args.out)


if __name__ == '__main__':
    main()
