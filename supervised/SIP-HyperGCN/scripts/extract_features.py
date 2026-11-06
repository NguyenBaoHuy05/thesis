"""Step 1 (v3): dump per-layer HyperGCN features + continuous measures.

Output: <out_dir>/<split>_feats.npz (docs/PIPELINE_V3.md):
  feat_l1..feat_l9 (N, C) f32, featj_l3/6/9 (N, C, 25) f16, logits (N, K), labels (N,),
  index (N,) feeder indices, meas (N, D) f32 (sip.measures.compute_measures on the
  feeder input), meas_cols (D,) str (= sip.measures.COLUMNS).

Example (Colab):
    python scripts/extract_features.py \
        --hypergcn_root ../GCCR-Skateformer-TreeRule/hypergcn \
        --checkpoint /content/drive/MyDrive/hypergcn_joint.pt \
        --data_path /content/data/NTU60_CS.npz --split test --out_dir work_dir/sip
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sip.backbone import load_hypergcn, build_feeder, forward_tapped, check_equivalence  # noqa: E402
from sip.measures import compute_measures, COLUMNS  # noqa: E402

JOINT_LAYERS = (3, 6, 9)


def get_parser():
    parser = argparse.ArgumentParser(description='Extract HyperGCN layer features + continuous measures')
    parser.add_argument('--hypergcn_root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_path', required=True, help='NTU60_CS.npz')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--out_dir', default='./work_dir/sip')
    parser.add_argument('--config', default=None, help='hyper_joint.yaml (default: inside hypergcn_root)')
    parser.add_argument('--num_class', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    return parser


def main():
    arg = get_parser().parse_args()
    os.makedirs(arg.out_dir, exist_ok=True)
    device = arg.device

    model = load_hypergcn(arg.hypergcn_root, arg.checkpoint, num_class=arg.num_class, device=device)
    dataset = build_feeder(arg.hypergcn_root, arg.data_path, arg.split, config=arg.config)
    if arg.max_samples is not None and arg.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(arg.max_samples)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=arg.batch_size, shuffle=False,
                                         num_workers=arg.num_worker, drop_last=False)

    feats = {k: [] for k in range(1, 10)}
    jfeats = {k: [] for k in JOINT_LAYERS}
    logits_all, labels_all, index_all, meas_all = [], [], [], []

    n_batch = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for b, (data, label, index) in enumerate(loader):
            data_np = data.numpy().astype(np.float32)
            x = data.float().to(device)
            if b == 0:
                check_equivalence(model, x)
            logits, f, fj = forward_tapped(model, x, joint_layers=JOINT_LAYERS)

            for k in range(1, 10):
                feats[k].append(f[k].float().cpu().numpy())
            for k in JOINT_LAYERS:
                jfeats[k].append(fj[k].float().cpu().numpy().astype(np.float16))
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(np.asarray(label).astype(np.int64))
            index_all.append(np.asarray(index).astype(np.int64))
            meas_all.append(np.asarray(compute_measures(data_np, input_format='feeder'), dtype=np.float32))

            if (b + 1) % 20 == 0 or b + 1 == n_batch:
                el = time.time() - t0
                eta = el / (b + 1) * (n_batch - b - 1)
                print('[{}] batch {}/{}  elapsed {:.0f}s  ETA {:.0f}s'.format(arg.split, b + 1, n_batch, el, eta))

    out = {}
    for k in range(1, 10):
        out['feat_l{}'.format(k)] = np.concatenate(feats[k]).astype(np.float32)
    for k in JOINT_LAYERS:
        out['featj_l{}'.format(k)] = np.concatenate(jfeats[k]).astype(np.float16)
    out['logits'] = np.concatenate(logits_all).astype(np.float32)
    out['labels'] = np.concatenate(labels_all)
    out['index'] = np.concatenate(index_all)
    out['meas'] = np.concatenate(meas_all).astype(np.float32)
    out['meas_cols'] = np.array(list(COLUMNS), dtype=str)
    assert out['meas'].shape == (len(out['labels']), len(out['meas_cols'])),         'meas shape {} != (N, len(COLUMNS))'.format(out['meas'].shape)
    n_bad = int((~np.isfinite(out['meas'])).sum())
    assert n_bad == 0, 'compute_measures returned {} non-finite values'.format(n_bad)

    acc = (out['logits'].argmax(1) == out['labels']).mean()
    path = os.path.join(arg.out_dir, '{}_feats.npz'.format(arg.split))
    np.savez(path, **out)
    print('saved {}  N={}  top1={:.4f}'.format(path, len(out['labels']), acc))
    for k, v in out.items():
        print('  {:12s} {} {}'.format(k, v.shape, v.dtype))


if __name__ == '__main__':
    main()
