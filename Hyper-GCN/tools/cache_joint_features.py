"""Cache the frozen l9 map per actor and joint for the concept-logic head (model/concept_logic.py).

  python tools/cache_joint_features.py --weights <checkpoint.pt> --out-dir <folder>
      [--config config/large/nturgbd120-cross-subject/hyper_joint.yaml] [--name ntu120_xsub_joint]
      [--segments 1] [--splits test train] [--batch-size 128] [--num-worker 4] [--device cuda]

Writes per split
    <name>_<split>_feat.npy   [N, M, C, S, V] float16, l9 output averaged inside S equal time segments
                              (written through a memmap: the NTU120 train array already takes ~11 GB of RAM)
    <name>_<split>.npz        valid [N, M] bool (real actors, model/hypergcn_kg.py: valid_actor_mask),
                              fc_logits [N, classes], label [N], index [N] as tools/cache_features.py
One pass per split. The train split uses train_feeder_args (random_rot) once, seed 0: the head sees
one augmented view, unlike the 5 views of the z cache. Size: N x 2 x 512 x S x 25 x 2 bytes, about
3.2 GB (train) and 2.6 GB (test) for S = 1.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from model.hypergcn_kg import valid_actor_mask
from tools.cache_features import build_loader, fc_logits_from_features, import_class, load_frozen_model


def segment_mean(features, segments):
    """[N, M, C, T, V] -> [N, M, C, S, V]: mean inside S equal (adaptive) time segments."""
    N, M, C, T, V = features.shape
    x = features.permute(0, 1, 2, 4, 3).reshape(N * M * C, V, T)
    x = F.adaptive_avg_pool1d(x, segments)
    return x.reshape(N, M, C, V, segments).permute(0, 1, 2, 4, 3)


@torch.no_grad()
def run_pass(model, loader, device, segments, feat_path):
    feat, start = None, 0
    valids, logits, labels, names = [], [], [], []
    for data, label, index in tqdm(loader, ncols=40):
        data = data.float().to(device)
        features, _ = model.features(data, until='l9')                      # [N, M, C, T, V]
        block = segment_mean(features, segments).half().cpu().numpy()
        if feat is None:
            feat = np.lib.format.open_memmap(feat_path, mode='w+', dtype=np.float16,
                                             shape=(len(loader.dataset),) + block.shape[1:])
        feat[start:start + len(block)] = block
        start += len(block)
        valids.append(valid_actor_mask(data).cpu().numpy())
        logits.append(fc_logits_from_features(model, features).float().cpu().numpy())
        labels.append(label.numpy())
        names.extend(index.tolist() if torch.is_tensor(index) else list(index))
    feat.flush()
    shape = feat.shape
    del feat
    return shape, dict(valid=np.concatenate(valids),
                fc_logits=np.concatenate(logits), label=np.concatenate(labels).astype(np.int64),
                index=np.array(names, dtype=np.int64))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--config', default='config/large/nturgbd120-cross-subject/hyper_joint.yaml')
    parser.add_argument('--name', default='ntu120_xsub_joint')
    parser.add_argument('--segments', type=int, default=1, help='time segments kept per sample')
    parser.add_argument('--splits', nargs='+', choices=['test', 'train'], default=['test', 'train'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights, out_dir = os.path.abspath(args.weights), os.path.abspath(args.out_dir)
    config_path = os.path.abspath(args.config if os.path.isfile(args.config) else os.path.join(repo, args.config))
    os.chdir(repo)
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    model, report = load_frozen_model(config, weights, args.device)
    print(f'Checkpoint load: {report}')
    Feeder = import_class(config['feeder'])

    summary = dict(config=config_path, weights=weights, segments=args.segments, splits={})
    for split in sorted(args.splits, key=['test', 'train'].index):
        dataset = Feeder(**config['train_feeder_args' if split == 'train' else 'test_feeder_args'])
        feat_path = os.path.join(out_dir, f'{args.name}_{split}_feat.npy')
        shape, arrays = run_pass(model, build_loader(dataset, args.batch_size, args.num_worker, seed=0),
                                 args.device, args.segments, feat_path)
        path = os.path.join(out_dir, f'{args.name}_{split}.npz')
        np.savez(path, **arrays)
        top1 = float((arrays['fc_logits'].argmax(1) == arrays['label']).mean())
        summary['splits'][split] = dict(path=path, feat=feat_path, feat_shape=list(shape),
                                        samples=int(len(arrays['label'])), fc_logits_top1=top1)
        print(f"Saved {feat_path}: feat {shape}; {path}: fc_logits top1 {100 * top1:.2f}%")
        del arrays
    with open(os.path.join(out_dir, f'{args.name}_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
