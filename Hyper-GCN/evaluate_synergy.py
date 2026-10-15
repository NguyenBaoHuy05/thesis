#!/usr/bin/env python
"""Explanation evaluation of a trained synergy model (SEMANTIC_HYPERGRAPH_PLAN.md, section 3).

- Top1 and readout-level metrics (as logged by main.py).
- Whole-network deletion / insertion: (synergy, segment) gates are set to 0
  (deletion) or opened (insertion) in G_sem of l7-l9 *and* in the readout, in
  the order of their contribution to the predicted class, versus a random order.
- Plausibility (precision@k, nDCG) against annotations/class_synergy.json.
- Stability: top-k Jaccard under a random rotation of the input, and against
  another run (--compare, e.g. another seed).
- Heatmaps synergy x segment per class and joint-importance skeletons (--plot-classes).

Usage:
  python evaluate_synergy.py --config WORK_DIR/config.yaml --weights BEST.pt \
      [--steps 0 1 2 4 8 16 32 64] [--max-batches 20] [--rotation 0.3] \
      [--compare OTHER/synergy_eval.json] [--plot-classes drink_water kicking] [--out-dir DIR]
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from main import import_class
from model.checkpoint import load_model_weights
from model.concept_schema import NTU_ACTION_NAMES
from model.synergies import NUM_JOINTS, synergy_bank
from model.synergy_metrics import (
    SynergyAccumulator, jaccard, load_annotations, plausibility, predicted_probability, topk_sets,
)


def rotation_matrices(angles):
    """[N,3] angles around x, y, z -> rotation matrices [N,3,3]."""
    cos, sin = angles.cos(), angles.sin()
    one, zero = torch.ones_like(cos[:, 0]), torch.zeros_like(cos[:, 0])
    rx = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], 1)
    ry = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], 1)
    rz = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], 1)
    return rz.view(-1, 3, 3) @ ry.view(-1, 3, 3) @ rx.view(-1, 3, 3)


def rotate(data, theta, generator):
    """Rotate joint-mode input [N,3,T,V,M]. Rotation is linear, so it commutes
    with the feeder's subtraction of the reference joint."""
    angles = (torch.rand(data.shape[0], 3, generator=generator) * 2 - 1) * theta
    R = rotation_matrices(angles).to(data)
    return torch.einsum('nij,njtvm->nitvm', R, data[:, :3])


def gate_masks(order, count, shape, keep_first):
    """Masks [N,S,E] with the first ``count`` entries of ``order`` [N,E*S] off
    (deletion) or the only ones on (insertion)."""
    N, S, E = shape
    flat = torch.zeros(N, E * S, device=order.device) if keep_first else torch.ones(N, E * S, device=order.device)
    if count:
        flat.scatter_(1, order[:, :count], 1.0 if keep_first else 0.0)
    return flat.view(N, E, S).transpose(1, 2)


def topk_entries(contributions, predicted, k):
    rows = torch.arange(len(predicted), device=predicted.device)
    order = contributions[rows, predicted].flatten(1).argsort(dim=1, descending=True)[:, :k]
    return [set(row) for row in order.tolist()]


def joint_importance(profile_row, singles):
    """Spread each single synergy's absolute contribution over its joints: [25]."""
    importance = np.zeros(NUM_JOINTS)
    for value, synergy in zip(profile_row, singles):
        importance[list(synergy.joints)] += abs(value) / len(synergy.joints)
    return importance / max(importance.max(), 1e-8)


# Standing template pose (x, y) and NTU bones, for joint-importance plots.
TEMPLATE = {
    0: (0, 0), 1: (0, .3), 2: (0, .55), 3: (0, .65), 20: (0, .5),
    4: (.2, .5), 5: (.25, .25), 6: (.25, .05), 7: (.25, 0), 21: (.25, -.05), 22: (.29, 0),
    8: (-.2, .5), 9: (-.25, .25), 10: (-.25, .05), 11: (-.25, 0), 23: (-.25, -.05), 24: (-.29, 0),
    12: (.1, 0), 13: (.1, -.45), 14: (.1, -.9), 15: (.15, -.95),
    16: (-.1, 0), 17: (-.1, -.45), 18: (-.1, -.9), 19: (-.15, -.95),
}
BONES = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (6, 22),
         (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (10, 24), (0, 12), (12, 13), (13, 14),
         (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]


def plot_classes(profile, names, singles, class_names, classes, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in classes:
        if name not in class_names:
            print('unknown class', name)
            continue
        c = class_names.index(name)
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw=dict(width_ratios=[2, 1]))
        bound = max(np.abs(profile[c]).max(), 1e-8)
        image = left.imshow(profile[c], cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        left.set_yticks(range(len(names)), names, fontsize=8)
        left.set_xticks(range(profile.shape[2]), ['seg {}'.format(s + 1) for s in range(profile.shape[2])])
        left.set_title('{}: mean contribution'.format(name))
        fig.colorbar(image, ax=left)
        importance = joint_importance(profile[c].sum(-1)[:len(singles)], singles)
        for a, b in BONES:
            right.plot(*zip(TEMPLATE[a], TEMPLATE[b]), color='0.7', zorder=1)
        xy = np.array([TEMPLATE[j] for j in range(NUM_JOINTS)])
        right.scatter(xy[:, 0], xy[:, 1], c=importance, cmap='Reds', vmin=0, vmax=1,
                      s=60 + 240 * importance, edgecolors='k', zorder=2)
        right.set_aspect('equal')
        right.axis('off')
        right.set_title('joint importance (facing the viewer)')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'heatmap_{}.png'.format(name)), dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='limit the test batches (whole-network deletion runs many forwards)')
    parser.add_argument('--steps', type=int, nargs='+', default=[0, 1, 2, 4, 8, 16, 32, 64],
                        help='numbers of (synergy, segment) gates removed / inserted')
    parser.add_argument('--rotation', type=float, default=0.3, help='0 disables the rotation test')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--annotations', default='annotations/class_synergy.json')
    parser.add_argument('--compare', default=None, help='synergy_eval.json of another run')
    parser.add_argument('--plot-classes', nargs='*', default=[])
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.weights)), 'synergy_eval')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = import_class(config['model'])(**config['model_args'])
    load_model_weights(model, torch.load(args.weights, map_location='cpu'))
    model = model.to(device).eval()
    bottleneck = model.synergy
    if bottleneck is None:
        raise ValueError('the model has no synergy bottleneck (synergy_enabled: False)')
    class_names = NTU_ACTION_NAMES[:bottleneck.num_classes]
    singles, _ = synergy_bank(config['model_args'].get('synergy_bank_kind', 'semantic'),
                              config['model_args'].get('synergy_seed', 0))
    E, S = bottleneck.num_synergies, bottleneck.segments
    steps = sorted({min(step, E * S) for step in args.steps})

    feeder = import_class(config['feeder'])(**config['test_feeder_args'])
    loader = torch.utils.data.DataLoader(feeder, batch_size=args.batch_size, shuffle=False,
                                         num_workers=config.get('num_worker', 4))
    accumulator = SynergyAccumulator(bottleneck, class_names)
    rotated_accumulator = SynergyAccumulator(bottleneck, class_names) if args.rotation > 0 else None
    generator = torch.Generator().manual_seed(0)
    curves = {name: np.zeros(len(steps)) for name in
              ('deletion', 'insertion', 'random_deletion', 'random_insertion')}
    correct = samples = 0
    sample_jaccard = []

    with torch.no_grad():
        for batch, (data, label, _) in enumerate(loader):
            if args.max_batches is not None and batch >= args.max_batches:
                break
            data, label = data.float().to(device), label.long().to(device)
            output = model(data)
            accumulator.update(output, label)
            predicted = output.action_logits.argmax(dim=1)
            correct += int((predicted == label).sum())
            samples += len(label)

            rows = torch.arange(len(predicted), device=device)
            ranked = output.contributions[rows, predicted].flatten(1).argsort(dim=1, descending=True)
            shuffled = torch.rand(ranked.shape, generator=generator).argsort(dim=1).to(device)
            for prefix, order in (('', ranked), ('random_', shuffled)):
                for kind, keep_first in (('deletion', False), ('insertion', True)):
                    for i, step in enumerate(steps):
                        mask = gate_masks(order, step, (len(label), S, E), keep_first)
                        logits = model(data, gate_mask=mask).action_logits
                        curves[prefix + kind][i] += float(
                            predicted_probability(logits[..., None], predicted).sum())

            if rotated_accumulator is not None:
                rotated = model(rotate(data, args.rotation, generator))
                rotated_accumulator.update(rotated, label)
                original = topk_entries(output.contributions, predicted, 5)
                turned = topk_entries(rotated.contributions, predicted, 5)
                sample_jaccard += [len(a & b) / len(a | b) for a, b in zip(original, turned)]
            print('batch {}: top1 so far {:.2%}'.format(batch + 1, correct / samples))

    profile = accumulator.profile()
    class_profile = profile.sum(-1)
    curves = {name: (values / samples).tolist() for name, values in curves.items()}
    whole = {'steps': steps, 'curves': curves}
    for name, values in curves.items():
        whole[name + '_auc'] = float(np.mean(values))
    whole['deletion_gap'] = whole['random_deletion_auc'] - whole['deletion_auc']
    whole['insertion_gap'] = whole['insertion_auc'] - whole['random_insertion_auc']

    annotations = None
    if args.annotations and os.path.exists(args.annotations):
        annotations = load_annotations(args.annotations, class_names, bottleneck.names)
    report = dict(weights=os.path.abspath(args.weights), samples=samples, top1=correct / samples,
                  whole_network=whole,
                  readout=accumulator.summary(annotations, args.top_k),
                  synergy_names=list(bottleneck.names),
                  class_profile=class_profile.tolist())
    if annotations is not None:
        report['plausibility'] = plausibility(class_profile, annotations, args.top_k)
    if rotated_accumulator is not None:
        report['rotation_stability'] = dict(
            theta=args.rotation,
            sample_top5_jaccard=float(np.mean(sample_jaccard)),
            class_topk_jaccard=jaccard(topk_sets(class_profile, args.top_k),
                                       topk_sets(rotated_accumulator.profile().sum(-1), args.top_k)),
        )
    if args.compare:
        with open(args.compare, encoding='utf-8') as handle:
            other = json.load(handle)
        if other['synergy_names'] != report['synergy_names']:
            raise ValueError('--compare must use the same synergy bank')
        report['compare'] = dict(path=args.compare, class_topk_jaccard=jaccard(
            topk_sets(class_profile, args.top_k), topk_sets(np.array(other['class_profile']), args.top_k)))

    with open(os.path.join(out_dir, 'synergy_eval.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    np.save(os.path.join(out_dir, 'synergy_profile.npy'), profile)
    if args.plot_classes:
        plot_classes(profile, list(bottleneck.names), singles, list(class_names), args.plot_classes, out_dir)
    print(json.dumps({key: report[key] for key in ('top1',)}, indent=2))
    print('whole network: deletion AUC {:.4f} (random {:.4f}), insertion AUC {:.4f} (random {:.4f})'.format(
        whole['deletion_auc'], whole['random_deletion_auc'], whole['insertion_auc'], whole['random_insertion_auc']))
    print('report written to', out_dir)


if __name__ == '__main__':
    main()
