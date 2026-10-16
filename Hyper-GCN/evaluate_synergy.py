#!/usr/bin/env python
"""Evaluation of the synergy bottleneck (SEMANTIC_HYPERGRAPH_PLAN.md, section 3).

- Top1, readout deletion / insertion vs random order, effective synergies,
  plausibility (precision@k, nDCG against annotations/class_synergy.json).
- Stability: top-k Jaccard under a random input rotation, and against another
  run with the same bank (--compare, e.g. another seed).
- Locality of the backbone features per layer (swap test): for synergy e,
  replace the joints *outside* e (or *inside* e) by those of another sample
  and measure how much the pooled feature f[e] changes.
      locality = change_inside / (change_inside + change_outside)
  1 means f[e] depends only on its own joints; a low value means f[e] is a
  whole-body feature, so the synergy name does not say where the evidence is.
  --locality-only needs just the official backbone weights (no training).
- Heatmaps synergy x segment per class (--plot-classes).

Usage:
  python evaluate_synergy.py --config WORK_DIR/config.yaml --weights BEST.pt \
      [--max-batches 40] [--compare OTHER/synergy_eval/synergy_eval.json] [--plot-classes drink_water]
  python evaluate_synergy.py --config config/large/nturgbd120-cross-subject/synergy.yaml \
      --weights OFFICIAL_JOINT.pt --locality-only --locality-layers l3 l6 l8 l9
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
from model.synergy_bottleneck import SynergyPooling
from model.synergy_metrics import SynergyAccumulator, jaccard, load_annotations, plausibility, topk_sets


def rotation_matrices(angles):
    """[N,3] angles around x, y, z -> rotation matrices [N,3,3]."""
    cos, sin = angles.cos(), angles.sin()
    one, zero = torch.ones_like(cos[:, 0]), torch.zeros_like(cos[:, 0])
    rx = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], 1)
    ry = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], 1)
    rz = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], 1)
    return rz.view(-1, 3, 3) @ ry.view(-1, 3, 3) @ rx.view(-1, 3, 3)


def rotate(data, theta, generator):
    """Rotate joint-mode input [N,3,T,V,M]; rotation commutes with the feeder's
    subtraction of the reference joint."""
    angles = (torch.rand(data.shape[0], 3, generator=generator) * 2 - 1) * theta
    return torch.einsum('nij,njtvm->nitvm', rotation_matrices(angles).to(data), data[:, :3])


def topk_entries(contributions, predicted, k):
    rows = torch.arange(len(predicted), device=predicted.device)
    order = contributions[rows, predicted].flatten(1).argsort(dim=1, descending=True)[:, :k]
    return [set(row) for row in order.tolist()]


@torch.no_grad()
def locality(model, data, synergies, layer, segments):
    """Swap-test locality of every synergy at ``layer`` for the first actor: [E]."""
    pooling = SynergyPooling([s.joints for s in synergies], segments).to(data.device)
    other = data.roll(1, dims=0)  # joints of another sample
    reference = pooling(model.features(data, layer)[0][:, 0])  # N,S,E,C
    inside_change, outside_change = [], []
    for e, synergy in enumerate(synergies):
        inside = torch.zeros(NUM_JOINTS, dtype=torch.bool, device=data.device)
        inside[list(synergy.joints)] = True
        changes = []
        for swapped in (inside, ~inside):
            x = torch.where(swapped[None, None, None, :, None], other, data)
            f = pooling(model.features(x, layer)[0][:, 0])[:, :, e]
            changes.append(((f - reference[:, :, e]).norm(dim=-1)
                            / reference[:, :, e].norm(dim=-1).clamp_min(1e-6)).mean())
        inside_change.append(changes[0])
        outside_change.append(changes[1])
    inside_change, outside_change = torch.stack(inside_change), torch.stack(outside_change)
    return (inside_change / (inside_change + outside_change).clamp_min(1e-8)).cpu().numpy()


def plot_classes(profile, names, class_names, classes, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in classes:
        if name not in class_names:
            print('unknown class', name)
            continue
        c = class_names.index(name)
        fig, ax = plt.subplots(figsize=(7, 6))
        bound = max(np.abs(profile[c]).max(), 1e-8)
        image = ax.imshow(profile[c], cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xticks(range(profile.shape[2]), ['seg {}'.format(s + 1) for s in range(profile.shape[2])])
        ax.set_title('{}: mean contribution'.format(name))
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'heatmap_{}.png'.format(name)), dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--rotation', type=float, default=0.3, help='0 disables the rotation test')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--annotations', default='annotations/class_synergy.json')
    parser.add_argument('--compare', default=None, help='synergy_eval.json of another run')
    parser.add_argument('--plot-classes', nargs='*', default=[])
    parser.add_argument('--locality-only', action='store_true',
                        help='only the locality test; the weights may be a backbone-only checkpoint')
    parser.add_argument('--locality-layers', nargs='*', default=['l3', 'l6', 'l8', 'l9'])
    parser.add_argument('--locality-batches', type=int, default=4)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.weights)), 'synergy_eval')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model_args = dict(config['model_args'])
    model_args.setdefault('synergy_enabled', True)
    model = import_class(config['model'])(**model_args)
    load_model_weights(model, torch.load(args.weights, map_location='cpu'), allow_new=args.locality_only)
    model = model.to(device).eval()
    synergies = synergy_bank(model_args.get('synergy_bank_kind', 'semantic'), model_args.get('synergy_seed', 0))
    segments = model_args.get('synergy_segments', 4)

    feeder = import_class(config['feeder'])(**config['test_feeder_args'])
    loader = torch.utils.data.DataLoader(feeder, batch_size=args.batch_size, shuffle=False,
                                         num_workers=config.get('num_worker', 4))
    report = dict(weights=os.path.abspath(args.weights), synergy_names=[s.name for s in synergies])

    # Locality of the backbone features, layer by layer.
    scores = {layer: [] for layer in args.locality_layers}
    with torch.no_grad():
        for batch, (data, _, _) in enumerate(loader):
            if batch >= args.locality_batches:
                break
            data = data.float().to(device)
            for layer in args.locality_layers:
                scores[layer].append(locality(model, data, synergies, layer, segments))
    report['locality'] = {
        layer: dict(mean=float(np.mean(values)),
                    per_synergy=dict(zip(report['synergy_names'], np.mean(values, axis=0).tolist())))
        for layer, values in scores.items()
    }
    for layer, value in report['locality'].items():
        print('locality {}: {:.3f}'.format(layer, value['mean']))

    if not args.locality_only:
        bottleneck = model.synergy
        class_names = NTU_ACTION_NAMES[:bottleneck.num_classes]
        accumulator = SynergyAccumulator(bottleneck, class_names)
        rotated_accumulator = SynergyAccumulator(bottleneck, class_names) if args.rotation > 0 else None
        generator = torch.Generator().manual_seed(0)
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
                if rotated_accumulator is not None:
                    rotated = model(rotate(data, args.rotation, generator))
                    rotated_accumulator.update(rotated, label)
                    sample_jaccard += [len(a & b) / len(a | b) for a, b in zip(
                        topk_entries(output.contributions, predicted, 5),
                        topk_entries(rotated.contributions, predicted, 5))]

        profile = accumulator.profile()
        class_profile = profile.sum(-1)
        annotations = None
        if args.annotations and os.path.exists(args.annotations):
            annotations = load_annotations(args.annotations, class_names, bottleneck.names)
        report.update(samples=samples, top1=correct / samples,
                      readout=accumulator.summary(annotations, args.top_k),
                      class_profile=class_profile.tolist())
        if annotations is not None:
            report['plausibility'] = plausibility(class_profile, annotations, args.top_k)
        if rotated_accumulator is not None:
            report['rotation_stability'] = dict(
                theta=args.rotation, sample_top5_jaccard=float(np.mean(sample_jaccard)),
                class_topk_jaccard=jaccard(topk_sets(class_profile, args.top_k),
                                           topk_sets(rotated_accumulator.profile().sum(-1), args.top_k)))
        if args.compare:
            with open(args.compare, encoding='utf-8') as handle:
                other = json.load(handle)
            report['compare'] = dict(path=args.compare, class_topk_jaccard=jaccard(
                topk_sets(class_profile, args.top_k), topk_sets(np.array(other['class_profile']), args.top_k)))
        np.save(os.path.join(out_dir, 'synergy_profile.npy'), profile)
        if args.plot_classes:
            plot_classes(profile, list(bottleneck.names), list(class_names), args.plot_classes, out_dir)
        print('top1 {:.4f}'.format(report['top1']))

    with open(os.path.join(out_dir, 'synergy_eval.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print('report written to', out_dir)


if __name__ == '__main__':
    main()
