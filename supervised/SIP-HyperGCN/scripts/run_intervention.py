"""Step 3 (v3, tier 2 "what"): LEACE-erase each continuous MEASURE at each layer and
continue the forward pass (docs/PIPELINE_V3.md).

Anti-leakage split:
  * VAL      = stratified (per class) `--val_frac` subset of the rows of train_feats.npz,
               drawn with a fixed RNG seeded by `--seed` -> identical `val_index` in every run.
  * TRAIN-FIT = train rows minus VAL. LEACE erasers, Z standardisation statistics and the
               random controls' mean are fitted on TRAIN-FIT only.
  * --eval_split train -> the forward is run on the VAL rows of the NTU train split.
    --eval_split test  -> the forward is run on the test split (all rows or --max_samples).

Units: every measure m of sip.measures.MEASURE_NAMES (or the `--measures` subset), plus
R = --n_random random controls ("__random_0__" .. "__random_{R-1}__", last on the U axis).
  * measure m at layer l: LEACE fitted on TRAIN-FIT pooled `feat_l{l}` (N, C) against the
    standardised continuous multi-column Z = meas[:, measure_slices()[m]]
    (z-score with TRAIN-FIT mean / std; constant columns dropped; all constant -> skipped).
    The eraser has rank <= #columns of m (unit_rank).
  * random control r at layer l: orthogonal projection onto a random subspace of rank =
    median LEACE rank of the fitted measures at that layer (random_rank), same TRAIN-FIT mean.
  * the (affine) eraser is applied to the output of layer l along channels at every
    (person, t, v) position (=> it also erases the information from the pooled mean);
    the residual copy of the layer output is edited as well (sip/backbone.py).

Efficiency: the evaluation set is traversed ONCE. Per batch the clean state after each
layer l is kept and shared by all U units; only layers l+1..l9 (+ head) are recomputed per
unit, i.e. 1 clean pass + L*U partial passes per batch.

Output: <out_dir>/<out_name> (default intervention_<eval_split>.npz). Keys:
  layers (L,), unit_names (U,), n_random, labels (N,), base_logits (N,K), base_pred (N,),
  correct (L,U,N) bool, pred_label (L,U,N) int16, acc_per_class (L,U,K),
  base_acc_per_class (K,), class_count (K,), skipped (L,U), eval_split, eval_index (N,),
  val_index (N_val,), val_frac, seed, base_final_feat (N,C),
  final_feat (L,U,N,C) f16 (absent with --no_final_feat),
  unit_rank (L,U) int (erased rank; 0 if skipped), random_rank (L,), cross_cov_rel (L,U)
  (TRAIN-FIT ||Cov(erased X, Z)|| / ||Cov(X, Z)||, NaN for randoms / skipped).
`eval_index` = row indices into <eval_split>_feats.npz of the evaluated samples.

Example (Colab):
    python scripts/run_intervention.py --eval_split train \
        --hypergcn_root ../GCCR-Skateformer-TreeRule/hypergcn \
        --checkpoint /content/drive/MyDrive/hypergcn_joint.pt \
        --data_path /content/data/NTU60_CS.npz \
        --train_feats work_dir/sip/train_feats.npz --out_dir work_dir/sip
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sip.backbone import (load_hypergcn, NUM_LAYERS, split_train_rows,  # noqa: E402
                          build_eval_dataset, intervention_sweep, sweep_outputs,
                          print_sweep_summary)
from sip.measures import MEASURE_NAMES, COLUMNS, measure_slices  # noqa: E402
from sip.leace import fit_leace, apply_leace_torch, random_eraser, verify_erasure  # noqa: E402


def random_name(r):
    return '__random_{}__'.format(r)


def get_parser():
    parser = argparse.ArgumentParser(description='Measure (LEACE) intervention on HyperGCN')
    parser.add_argument('--hypergcn_root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True, help='NTU60_CS.npz')
    parser.add_argument('--train_feats', required=True, help='train_feats.npz from extract_features.py')
    parser.add_argument('--out_dir', default='./work_dir/sip')
    parser.add_argument('--out_name', default=None,
                        help='default intervention_val.npz (eval_split train) / intervention_test.npz')
    parser.add_argument('--eval_split', default='test', choices=['train', 'test'],
                        help='train = run on the VAL rows of the train split; test = run on the test split')
    parser.add_argument('--val_frac', type=float, default=0.2, help='stratified VAL fraction of train_feats rows')
    parser.add_argument('--seed', type=int, default=0, help='seed of the VAL split and of the random controls')
    parser.add_argument('--n_random', type=int, default=5, help='number R of random subspace controls')
    parser.add_argument('--no_final_feat', action='store_true', help='do not store final_feat (saves RAM/disk)')
    parser.add_argument('--config', default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 6, 9])
    parser.add_argument('--measures', nargs='+', default=None,
                        help='subset of sip.measures.MEASURE_NAMES (default all, in MEASURE_NAMES order)')
    parser.add_argument('--num_class', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='cap on evaluated samples (test: first rows; train: first VAL rows)')
    parser.add_argument('--device', default='cuda')
    return parser


def default_out_name(eval_split):
    return 'intervention_{}.npz'.format('val' if eval_split == 'train' else 'test')


def standardise(Z, std_tol=1e-8):
    """z-score the columns of Z (TRAIN-FIT stats); drop (near-)constant columns.
    Returns (Zs (N, k), kept column mask)."""
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(0)
    sd = Z.std(0)
    keep = sd > std_tol * np.maximum(1.0, np.abs(mu))
    return (Z[:, keep] - mu[keep]) / sd[keep], keep


def fit_erasers(train, fit_rows, layers, measures, n_random, seed):
    """erasers[li][u] (dict or None) over the U = M + R axis, fitted on TRAIN-FIT rows."""
    slices = measure_slices()
    meas = np.asarray(train['meas'][fit_rows], dtype=np.float64)
    assert np.isfinite(meas).all(), 'non-finite values in train meas'
    Zs = {}
    for m in measures:
        z, keep = standardise(meas[:, slices[m]])
        if not keep.all():
            cols = COLUMNS[slices[m]]
            print('  {}: constant on TRAIN-FIT, dropped columns {}'.format(
                m, [c for c, k in zip(cols, keep) if not k]))
        Zs[m] = z if z.shape[1] > 0 else None

    L, M = len(layers), len(measures)
    U = M + n_random
    erasers = []
    skipped = np.zeros((L, U), dtype=bool)
    unit_rank = np.zeros((L, U), dtype=np.int64)
    cross_cov_rel = np.full((L, U), np.nan)
    random_rank = np.zeros(L, dtype=np.int64)
    for li, l in enumerate(layers):
        X = np.asarray(train['feat_l{}'.format(l)][fit_rows], dtype=np.float64)
        row = []
        for mi, m in enumerate(measures):
            if Zs[m] is None:
                print('  [l{}] {}: all columns constant on TRAIN-FIT -> skipped'.format(l, m))
                row.append(None)
                skipped[li, mi] = True
                continue
            e = fit_leace(X, Zs[m])
            if e['rank'] == 0:
                print('  [l{}] {}: zero cross-covariance -> skipped'.format(l, m))
                row.append(None)
                skipped[li, mi] = True
                continue
            row.append(e)
            unit_rank[li, mi] = e['rank']
            cross_cov_rel[li, mi] = verify_erasure(X, Zs[m], e)['cross_cov_rel']
        fitted = unit_rank[li, :M][~skipped[li, :M]]
        rr = int(round(float(np.median(fitted)))) if fitted.size else 1
        rr = max(rr, 1)
        random_rank[li] = rr
        for r in range(n_random):
            # independent stream from the VAL split RNG; identical across runs for a seed
            row.append(random_eraser(X, rank=rr, seed=seed * 1000003 + 7919 * (l + 1) + r))
            unit_rank[li, M + r] = rr
        erasers.append(row)
        print('[fit] layer l{}: C={} n_fit={} fitted {} measure erasers (ranks {}), '
              '{} random rank {}; max cross_cov_rel={:.2e}'.format(
                  l, X.shape[1], X.shape[0], int((~skipped[li, :M]).sum()),
                  unit_rank[li, :M].tolist(), n_random, rr,
                  np.nanmax(cross_cov_rel[li]) if np.isfinite(cross_cov_rel[li]).any() else float('nan')))
    return erasers, skipped, unit_rank, random_rank, cross_cov_rel


def main():
    arg = get_parser().parse_args()
    os.makedirs(arg.out_dir, exist_ok=True)
    layers = sorted(set(arg.layers))
    assert all(1 <= l <= NUM_LAYERS for l in layers), 'measure layers must be in 1..9'
    assert arg.n_random >= 0
    out_name = arg.out_name or default_out_name(arg.eval_split)

    # ---- VAL / TRAIN-FIT split on the rows of train_feats.npz
    train = np.load(arg.train_feats)
    cols = [str(c) for c in train['meas_cols']]
    assert cols == list(COLUMNS), 'train_feats meas_cols != sip.measures.COLUMNS (re-run extract_features.py)'
    train_labels = np.asarray(train['labels']).astype(np.int64)
    train_feeder_index = np.asarray(train['index']).astype(np.int64)
    val_index, fit_rows = split_train_rows(train_labels, arg.val_frac, arg.seed)

    if arg.measures is None:
        measures = list(MEASURE_NAMES)
    else:
        unknown = [m for m in arg.measures if m not in MEASURE_NAMES]
        assert not unknown, 'unknown measures {} (choices: {})'.format(unknown, list(MEASURE_NAMES))
        measures = list(dict.fromkeys(arg.measures))
    unit_names = measures + [random_name(r) for r in range(arg.n_random)]
    erasers, skipped, unit_rank, random_rank, cross_cov_rel = fit_erasers(
        train, fit_rows, layers, measures, arg.n_random, arg.seed)
    del train

    # ---- model + evaluation data
    model = load_hypergcn(arg.hypergcn_root, arg.checkpoint, num_class=arg.num_class, device=arg.device)
    dataset, eval_index, expected_feeder, expected_labels = build_eval_dataset(
        arg.hypergcn_root, arg.data_path, arg.eval_split, arg.config, val_index,
        train_feeder_index, train_labels, arg.max_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=arg.batch_size, shuffle=False,
                                         num_workers=arg.num_worker, drop_last=False)

    # intervention functions (default arg binds the eraser, avoids late-binding bug)
    fns = [[None if e is None else (lambda h, e=e: apply_leace_torch(h, e, channel_dim=1))
            for e in row] for row in erasers]
    res = intervention_sweep(model, loader, layers, fns, arg.num_class,
                             store_final=not arg.no_final_feat, device=arg.device, tag='measures')

    out = sweep_outputs(res, arg.num_class, layers, unit_names, skipped, arg.eval_split,
                        eval_index, val_index, arg.n_random, arg.val_frac, arg.seed,
                        expected_feeder, expected_labels)
    out['unit_rank'] = unit_rank
    out['random_rank'] = random_rank
    out['cross_cov_rel'] = cross_cov_rel
    path = os.path.join(arg.out_dir, out_name)
    np.savez(path, **out)
    print('saved {}'.format(path))
    print_sweep_summary(out)


if __name__ == '__main__':
    main()
