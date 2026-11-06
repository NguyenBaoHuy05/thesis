"""Step 2 (v3, tier 1 "where"): mean-ablate each body REGION at each layer and continue
the forward pass (docs/PIPELINE_V3.md, docs/architecture_v3.html section 4-5).

Split: identical to run_intervention.py (VAL = stratified `--val_frac` of train_feats rows,
seed `--seed`; TRAIN-FIT = the rest). Ablation means come from TRAIN-FIT only.

Units: sip.regions.REGION_NAMES, then R = --n_random random joint-set controls
("__random_0__" .. "__random_{R-1}__", last on the U axis).
  * region r != other_person: the joints REGIONS[r] of BOTH person slots are replaced by the
    TRAIN-FIT mean (ablate_torch(..., person=None)).
  * other_person: all 25 joints of person slot 1 (ablate_torch(..., person=1)).
  * random controls: sip.regions.random_joint_sets(R, sizes, seed') with sizes = the joint
    counts of the body regions (REGION_NAMES without other_person) = [2, 3, 6, 6, 4, 4],
    cycled (R = 5 -> sizes 2, 3, 6, 6, 4); each random set is
    ablated on both person slots, like the body regions. `other_person` is EXCLUDED from
    the random sizing: a 25-joint random set would be the whole body and is not a
    comparable control. The random sets are stored in unit_joint_mask / unit_person.

Layers (default 0 3 6 9). Layer 0 = output of the joint embedding + data_bn, before l1
(sip/backbone.py, intervene[0]); the ablation at layer l edits the layer output and its
residual copy.

Pass 1 (means): over `--n_mean_samples` (default 4000) TRAIN-FIT rows, stratified per
class (seed), run the clean forward up to max(layers) and accumulate, per layer, the
per-person-slot mean over T of the layer output -> mean_l (M, C, V). Slot 0 is averaged over
all samples; slot m >= 1 only over samples where that person is present (raw feeder input
of the slot non-zero; sip.backbone.person_present). If no sample has a 2nd person, slot 1
falls back to the all-sample mean and other_person is marked skipped.
Absent person slots are never ablated (ablate_torch(..., present=person_present(x))): for a
single-person sample, body regions are ablated in slot 0 only and other_person is a no-op.

Pass 2 (evaluation): VAL (--eval_split train) or TEST, one clean pass per batch keeping the
state after each layer, then L*U partial passes (layers l+1..9 + head) with ablate_torch.

Output: <out_dir>/<out_name> (default region_intervention_val.npz / region_intervention_test.npz).
Keys (same as intervention_*.npz): layers (L,), unit_names (U,), n_random, labels (N,),
base_logits (N,K), base_pred (N,), correct (L,U,N) bool, pred_label (L,U,N) int16,
acc_per_class (L,U,K), base_acc_per_class (K,), class_count (K,), skipped (L,U), eval_split,
eval_index (N,), val_index (N_val,), val_frac, seed, base_final_feat (N,C),
final_feat (L,U,N,C) f16 (absent with --no_final_feat); plus
unit_joint_mask (U, 25) bool, unit_person (U,) int (-1 = both slots), mean_l{l} (M, C, V) f32
per layer, mean_count (M,) samples used per person slot, mean_index (n,) TRAIN-FIT rows of
train_feats.npz used for the means.

Example (Colab):
    python scripts/run_region_intervention.py --eval_split train \
        --hypergcn_root ../GCCR-Skateformer-TreeRule/hypergcn \
        --checkpoint /content/drive/MyDrive/hypergcn_joint.pt \
        --data_path /content/data/NTU60_CS.npz \
        --train_feats work_dir/sip/train_feats.npz --out_dir work_dir/sip
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sip.backbone import (load_hypergcn, build_feeder, init_state, run_layers,  # noqa: E402
                          person_joint_mean, person_present, NUM_LAYERS, DEFAULT_MODEL_ARGS,
                          stratified_val_index, split_train_rows, build_eval_dataset,
                          intervention_sweep, sweep_outputs, print_sweep_summary)
from sip.regions import (REGION_NAMES, REGIONS, region_ablation_args,  # noqa: E402
                         random_joint_sets, ablate_torch)

OTHER = 'other_person'
NUM_JOINTS = 25


def random_name(r):
    return '__random_{}__'.format(r)


def get_parser():
    parser = argparse.ArgumentParser(description='Region mean-ablation intervention on HyperGCN')
    parser.add_argument('--hypergcn_root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True, help='NTU60_CS.npz')
    parser.add_argument('--train_feats', required=True,
                        help='train_feats.npz from extract_features.py (labels / index / VAL split)')
    parser.add_argument('--out_dir', default='./work_dir/sip')
    parser.add_argument('--out_name', default=None,
                        help='default region_intervention_val.npz (eval_split train) / _test.npz')
    parser.add_argument('--eval_split', default='test', choices=['train', 'test'],
                        help='train = run on the VAL rows of the train split; test = run on the test split')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0, help='seed of the VAL split, mean subset and random sets')
    parser.add_argument('--n_random', type=int, default=5, help='number R of random joint-set controls')
    parser.add_argument('--n_mean_samples', type=int, default=4000,
                        help='TRAIN-FIT samples (stratified) used to estimate the ablation means')
    parser.add_argument('--no_final_feat', action='store_true')
    parser.add_argument('--config', default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 3, 6, 9])
    parser.add_argument('--num_class', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='cap on evaluated samples (test: first rows; train: first VAL rows)')
    parser.add_argument('--device', default='cuda')
    return parser


def default_out_name(eval_split):
    return 'region_intervention_{}.npz'.format('val' if eval_split == 'train' else 'test')


def mean_subset(fit_rows, train_labels, n_mean, seed):
    """Stratified (per class) subset of TRAIN-FIT rows of size ~n_mean (all if n_mean >= |fit|)."""
    if n_mean is None or n_mean <= 0 or n_mean >= len(fit_rows):
        return np.asarray(fit_rows, dtype=np.int64)
    frac = float(n_mean) / len(fit_rows)
    sub = stratified_val_index(train_labels[fit_rows], frac, seed + 7777)
    return np.asarray(fit_rows, dtype=np.int64)[sub]


@torch.no_grad()
def estimate_means(model, loader, layers, num_person, device):
    """Per layer: (M, C, V) TRAIN-FIT mean over samples and T, per person slot.
    Slot 0: all samples. Slot m >= 1: only samples where person m is present."""
    max_l = max(layers)
    sums = {l: None for l in layers}       # (M, C, V) float64
    sums_all = {l: None for l in layers}   # unweighted, for the fallback
    count = torch.zeros(num_person, dtype=torch.float64, device=device)
    n_total = 0
    t0 = time.time()
    for b, (data, label, index) in enumerate(loader):
        x = data.float().to(device)
        n = x.size(0)
        pres = person_present(x).double()                   # (n, M)
        pres[:, 0] = 1.0                                    # slot 0: every sample
        count += pres.sum(0)
        n_total += n
        state = init_state(model, x)
        for l in layers:
            state = run_layers(model, state, l)
            pm = person_joint_mean(state['x'], state['N'], state['M']).double()   # (n, M, C, V)
            s_w = (pm * pres[:, :, None, None]).sum(0)
            s_a = pm.sum(0)
            sums[l] = s_w if sums[l] is None else sums[l] + s_w
            sums_all[l] = s_a if sums_all[l] is None else sums_all[l] + s_a
        del state
        if (b + 1) % 10 == 0 or b + 1 == len(loader):
            print('[means] batch {}/{} (up to l{})  elapsed {:.0f}s'.format(
                b + 1, len(loader), max_l, time.time() - t0))
    count_np = count.cpu().numpy()
    means = {}
    for l in layers:
        m = torch.empty_like(sums[l])
        for p in range(num_person):
            if count_np[p] > 0:
                m[p] = sums[l][p] / count_np[p]
            else:
                print('warning: person slot {} never present in the mean subset -> all-sample mean'.format(p))
                m[p] = sums_all[l][p] / max(n_total, 1)
        means[l] = m.float()
    return means, count_np, n_total


def build_units(n_random, seed):
    """unit_names, joints list, person (None / int) per unit; random sets sized like the
    body regions (other_person excluded from sizing)."""
    names, joints, persons = [], [], []
    for r in REGION_NAMES:
        js, p = region_ablation_args(r)   # other_person -> (all 25 joints, person 1)
        names.append(r)
        joints.append(sorted(int(j) for j in js))
        persons.append(p)
    sizes = [len(REGIONS[r]) for r in REGION_NAMES if r != OTHER]
    if n_random > 0:
        rsets = random_joint_sets(n_random, sizes, seed)
        assert len(rsets) == n_random, 'random_joint_sets returned {} sets, expected {}'.format(
            len(rsets), n_random)
        for r, js in enumerate(rsets):
            names.append(random_name(r))
            joints.append(sorted(int(j) for j in js))
            persons.append(None)
    return names, joints, persons


def main():
    arg = get_parser().parse_args()
    os.makedirs(arg.out_dir, exist_ok=True)
    device = arg.device
    layers = sorted(set(arg.layers))
    assert all(0 <= l <= NUM_LAYERS for l in layers), 'region layers must be in 0..9'
    assert arg.n_random >= 0
    out_name = arg.out_name or default_out_name(arg.eval_split)
    num_person = DEFAULT_MODEL_ARGS['num_person']

    # ---- VAL / TRAIN-FIT split (identical to run_intervention.py)
    train = np.load(arg.train_feats)
    train_labels = np.asarray(train['labels']).astype(np.int64)
    train_feeder_index = np.asarray(train['index']).astype(np.int64)
    del train
    val_index, fit_rows = split_train_rows(train_labels, arg.val_frac, arg.seed)

    # ---- units
    unit_names, unit_joints, unit_persons = build_units(arg.n_random, arg.seed + 1000003)
    U, L = len(unit_names), len(layers)
    unit_joint_mask = np.zeros((U, NUM_JOINTS), dtype=bool)
    for u, js in enumerate(unit_joints):
        unit_joint_mask[u, js] = True
    unit_person = np.array([-1 if p is None else p for p in unit_persons], dtype=np.int64)
    for u in range(U):
        print('  unit {:16s} person={:2d} joints={}'.format(unit_names[u], unit_person[u], unit_joints[u]))

    model = load_hypergcn(arg.hypergcn_root, arg.checkpoint, num_class=arg.num_class, device=device)

    # ---- pass 1: TRAIN-FIT means (M, C, V) per layer
    mean_index = mean_subset(fit_rows, train_labels, arg.n_mean_samples, arg.seed)
    train_ds = build_feeder(arg.hypergcn_root, arg.data_path, 'train', config=arg.config)
    mean_ds = torch.utils.data.Subset(train_ds, train_feeder_index[mean_index].tolist())
    mean_loader = torch.utils.data.DataLoader(mean_ds, batch_size=arg.batch_size, shuffle=False,
                                              num_workers=arg.num_worker, drop_last=False)
    print('[means] {} TRAIN-FIT samples (stratified, requested {})'.format(len(mean_index), arg.n_mean_samples))
    means, mean_count, n_mean = estimate_means(model, mean_loader, layers, num_person, device)
    del train_ds, mean_ds, mean_loader
    print('[means] person-slot counts {} / {}'.format(mean_count.tolist(), n_mean))

    skipped = np.zeros((L, U), dtype=bool)
    if num_person > 1 and mean_count[1] == 0:
        skipped[:, unit_names.index(OTHER)] = True

    # ---- pass 2: evaluation
    dataset, eval_index, expected_feeder, expected_labels = build_eval_dataset(
        arg.hypergcn_root, arg.data_path, arg.eval_split, arg.config, val_index,
        train_feeder_index, train_labels, arg.max_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=arg.batch_size, shuffle=False,
                                         num_workers=arg.num_worker, drop_last=False)

    # absent person slots (single-person samples) are never ablated: the mask is
    # refreshed for every batch by on_batch and read by the ablation closures
    ctx = {'present': None}

    def on_batch(x):
        ctx['present'] = person_present(x)

    fns = []
    for li, l in enumerate(layers):
        row = []
        for u in range(U):
            if skipped[li, u]:
                row.append(None)
                continue
            row.append(lambda h, js=unit_joints[u], mu=means[l], p=unit_persons[u]:
                       ablate_torch(h, js, mu, num_person=num_person, person=p,
                                    present=ctx['present']))
        fns.append(row)
    res = intervention_sweep(model, loader, layers, fns, arg.num_class,
                             store_final=not arg.no_final_feat, device=device, tag='regions',
                             on_batch=on_batch)

    out = sweep_outputs(res, arg.num_class, layers, unit_names, skipped, arg.eval_split,
                        eval_index, val_index, arg.n_random, arg.val_frac, arg.seed,
                        expected_feeder, expected_labels)
    out['unit_joint_mask'] = unit_joint_mask
    out['unit_person'] = unit_person
    for l in layers:
        out['mean_l{}'.format(l)] = means[l].cpu().numpy().astype(np.float32)
    out['mean_count'] = mean_count.astype(np.int64)
    out['mean_index'] = mean_index.astype(np.int64)
    out['n_mean_samples'] = np.array(len(mean_index), dtype=np.int64)
    path = os.path.join(arg.out_dir, out_name)
    np.savez(path, **out)
    print('saved {}'.format(path))
    print_sweep_summary(out)


if __name__ == '__main__':
    main()
