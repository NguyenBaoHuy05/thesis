"""A3-G G0 (plan §4): train quantiles of the predicates and the stop checks, before any GPU training.

  python tools/predicate_stats.py --data data/ntu120/NTU120_CSub.npz --output <work_dir>/predicate_stats.npz
                                  [--k0-scores <K0 test score pkl>]

1. train, test transform (p_interval 0.95, no rotation): q5 / q95 of every predicate over the frames of real actors
   -> predicate_stats.npz (with the names, so a stale file is refused by GeoPredicates);
2. hyperedge weights p with those stats; per sample the max over real actors; features = mean over 8 segments and over
   the clip (21 x 9). AUC per hard pair (PAIRS) on train: best column, folded max(AUC, 1 - AUC);
3. rotation: the test split twice, without and with random_rot, in the same order: mean |delta e_hat|;
4. report (<output>.json) with "passed"; G1 (tools/run_a3g.py) runs only if passed is true.
Report only: the same AUC on test, and (--k0-scores) on test split into samples K0 gets right / wrong.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feeders.feeder_ntu import Feeder                                      # noqa: E402
from model.concept_schema import NTU_ACTION_NAMES                          # noqa: E402
from model.geo_predicates import HYPEREDGE_NAMES, PREDICATE_NAMES, GeoPredicates, save_stats   # noqa: E402

# PREDICATES.md §5 without giving_object / exchange_things (two-person predicates are not hyperedges).
PAIRS = (
    ('put_object_into_bag', 'take_object_out_of_bag'),
    ('put_on_a_shoe', 'take_off_a_shoe'),
    ('put_on_headphone', 'take_off_headphone'),
    ('put_on_a_hat_cap', 'take_off_a_hat_cap'),
    ('thumb_up', 'thumb_down'),
    ('make_OK_sign', 'make_victory_sign'),
    ('make_victory_sign', 'snap_fingers'),
    ('drink_water', 'yawn'),
    ('yawn', 'blow_nose'),
    ('hush', 'blow_nose'),
    ('reading', 'writing'),
    ('writing', 'play_with_phone_tablet'),
    ('reading', 'staple_book'),
)
SEGMENTS = 8
AUC_PASS = 0.75          # folded AUC: max(AUC, 1 - AUC) >= 0.75  <=>  AUC >= 0.75 or <= 0.25
PAIRS_PASS = 7
ROTATION_MAX = 0.02
COLUMNS = [f'{e}@{s}' for e in HYPEREDGE_NAMES for s in range(SEGMENTS)] + [f'{e}@clip' for e in HYPEREDGE_NAMES]


def loader(args, split, random_rot):
    dataset = Feeder(data_path=args.data, split=split, window_size=64, p_interval=[0.95], random_rot=random_rot,
                     vel=False, bone=False)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


def features(geo, raw, valid):
    """raw [N, M, K, T] -> [N, 21 * 9]: hyperedge weights, max over real actors, 8 segment means + clip mean."""
    p = geo.weights(geo.normalize(raw), valid)                                      # [N, M, E, T]
    p = p.masked_fill(~valid[:, :, None, None], -1).amax(1).clamp_min(0)           # every sample has a real actor 0
    segments = torch.stack([s.mean(-1) for s in p.chunk(SEGMENTS, dim=-1)], -1)     # [N, E, 8]
    return torch.cat([segments.flatten(1), p.mean(-1)], 1).numpy()


def auc(values, positive):
    """Mann-Whitney AUC of every column of values [n, d] for a boolean target [n]: [d]."""
    ranks = rankdata(values, axis=0)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum(0) - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)


def pair_auc(feats, labels, subset=None):
    out = []
    for first, second in PAIRS:
        a, b = NTU_ACTION_NAMES.index(first), NTU_ACTION_NAMES.index(second)
        mask = np.isin(labels, [a, b]) if subset is None else subset & np.isin(labels, [a, b])
        target = labels[mask] == a
        if target.all() or not target.any():
            out.append(dict(pair=[first, second], n=int(mask.sum()), auc=None, column=None))
            continue
        values = auc(feats[mask], target)
        folded = np.maximum(values, 1 - values)
        best = int(np.nanargmax(folded))
        out.append(dict(pair=[first, second], n=int(mask.sum()), auc=round(float(folded[best]), 4),
                        column=COLUMNS[best]))
    return out


def k0_correct(path, labels):
    """Official score pkl (main.py: {sample name: score}, in dataset order) -> [N] bool, K0 right."""
    with open(path, 'rb') as handle:
        scores = pickle.load(handle)
    scores = np.stack(list(scores.values())) if isinstance(scores, dict) else np.asarray(scores)
    if scores.shape != (len(labels), len(NTU_ACTION_NAMES)):
        raise SystemExit(f'{path}: expected [{len(labels)}, {len(NTU_ACTION_NAMES)}] scores, got {scores.shape}')
    return scores.argmax(1) == labels


def main(args):
    # 1. train quantiles over the frames of real actors
    plain = GeoPredicates()
    raws, valids, labels = [], [], []
    for data, label, _ in loader(args, 'train', random_rot=False):
        raw, valid, _ = plain.raw(data)
        raws.append(raw)
        valids.append(valid)
        labels.append(label.numpy())
    raw, valid, labels = torch.cat(raws), torch.cat(valids), np.concatenate(labels)
    frames = raw.permute(0, 1, 3, 2)[valid].reshape(-1, len(PREDICATE_NAMES)).numpy()      # [frames, K]
    finite = bool(np.isfinite(frames).all())
    q5, q95 = np.quantile(frames, [0.05, 0.95], axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_stats(args.output, q5, q95)
    geo = GeoPredicates(args.output)

    # 2. hard pairs on train
    train_auc = pair_auc(features(geo, raw, valid), labels)
    del raw, raws

    # 3. rotation invariance and test features (same order: shuffle = False, deterministic crop)
    delta, count, test_feats, test_labels = 0.0, 0, [], []
    for (data, label, _), (rotated, _, _) in zip(loader(args, 'test', False), loader(args, 'test', True)):
        raw, valid, _ = geo.raw(data)
        raw_rot, _, _ = geo.raw(rotated)
        e, e_rot = geo.normalize(raw), geo.normalize(raw_rot)
        delta += (e - e_rot).abs()[valid].sum().item()
        count += e[valid].numel()
        test_feats.append(features(geo, raw, valid))
        test_labels.append(label.numpy())
    rotation = delta / max(count, 1)
    test_feats, test_labels = np.concatenate(test_feats), np.concatenate(test_labels)

    passing = sum(p['auc'] is not None and p['auc'] >= AUC_PASS for p in train_auc)
    report = dict(
        predicates=list(PREDICATE_NAMES), q5=q5.tolist(), q95=q95.tolist(), finite=finite,
        rotation_delta=rotation, train_pair_auc=train_auc, pairs_passing=f'{passing} / {len(PAIRS)}',
        test_pair_auc=pair_auc(test_feats, test_labels))
    if args.k0_scores:
        right = k0_correct(args.k0_scores, test_labels)
        report['k0'] = dict(right=int(right.sum()), wrong=int((~right).sum()),
                            pair_auc_k0_right=pair_auc(test_feats, test_labels, right),
                            pair_auc_k0_wrong=pair_auc(test_feats, test_labels, ~right))
    failed = [reason for reason, bad in (
        ('NaN / inf in the predicates', not finite),
        (f'rotation delta {rotation:.4f} >= {ROTATION_MAX}', rotation >= ROTATION_MAX),
        (f'{passing} / {len(PAIRS)} pairs with folded AUC >= {AUC_PASS} (need {PAIRS_PASS})', passing < PAIRS_PASS),
    ) if bad]
    report.update(passed=not failed, failed=failed)
    with open(os.path.splitext(args.output)[0] + '.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit('G0 stop: ' + '; '.join(failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--k0-scores', help='official K0 test score pkl (report only)')
    main(parser.parse_args())
