"""Cache frozen Hyper-GCN features for the KG retrieval head (A1 plan, P0,
KG_RETRIEVAL_PLAN/A1_RETRIEVE_VERIFY_FUSE_PLAN/A1_RETRIEVE_VERIFY_FUSE_PLAN.md S1.1/S1.2).

  python tools/cache_features.py --weights <checkpoint.pt> --out-dir <folder>
      [--config config/large/nturgbd120-cross-subject/hyper_joint.yaml] [--name ntu120_xsub]
      [--splits test train] [--views 5] [--layers l9] [--batch-size 128] [--num-worker 4] [--device cuda]

Writes <folder>/<name>_test.npz, <folder>/<name>_train.npz and <folder>/<name>_summary.json
(paths, sample counts, fc_logits Top1 per view). The test split runs first and is printed as the
P0 check. `--layers l6 l7 l8` caches the pooled output of those layers from the same pass, as
<name>_<layer>_{test,train}.npz (same fc_logits, label, index; `z` has that layer's channels:
256 for l4-l6, 512 for l7-l9) and <name>_l6_l7_l8_summary.json; l9 keeps the original names.

Backbone frozen, eval mode, one `features(x, until='l9')` pass per batch (not a second full
forward): `fc_logits` is recomputed from that same pass by replicating `Model.forward`'s pooling
(mean over T,V then over actors) + `fc`, so it must match the checkpoint's own output
(the P0 pass criterion: Top1 of cached fc_logits within 0.1% of the original). `z` is pooled
as the A1 plan specifies: mean(T,V) per actor, then channel-wise max over the valid (real)
actors. The NTU preprocessing (data/ntu*/seq_transformation.py, align_frames) fills the second
actor of a single-person sample with a *copy* of the first, not zeros; `valid_actor_mask`
drops such copies (and all-zero actors), so single-person z comes from the one real actor.
Each .npz also has `label` and `index`
(the dataset position of each row, matching `main.py`'s `*_wrong.txt`/`*_right.txt` convention,
not the `"train_0"`-style strings in `dataset.sample_name`).

The train split with `--views K>1` repeats the feeder K times with different seeds
(`train_feeder_args` has `random_rot`) and stacks a leading view axis in `z`/`fc_logits`; the head
samples one view per epoch at train time. The test split is always a single deterministic pass.
"""

import argparse
import json
import os
import pickle
import random
import sys

# `python tools/cache_features.py` puts tools/ on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from model.checkpoint import load_model_weights
# Shared with the end-to-end KG model, so cached and fine-tuned z are pooled identically.
from model.hypergcn_kg import pooled_z, valid_actor_mask  # noqa: F401 (re-exported for tests)


def import_class(name):
    mod_str, _sep, class_str = name.rpartition('.')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def fc_logits_from_features(model, features):
    """Replicate `Model.forward`'s pooling (mean over flattened T*V, then over actors) + `fc`,
    so this equals `model(x)[0]` (up to float summation order) without a second backbone pass."""
    pooled = features.flatten(3).mean(3).mean(1)
    return model.fc(model.drop_out(pooled))


@torch.no_grad()
def run_pass(model, loader, device, layers=('l9',)):
    """One backbone pass per batch. z of an earlier layer comes from a forward hook on that layer
    ([N*M, C, T, V] output, before any residual is added for the next layer), pooled like l9.
    Returns ({layer: z}, fc_logits, labels, names)."""
    captured = {}
    hooks = [getattr(model, layer).register_forward_hook(
        lambda module, inputs, output, layer=layer: captured.__setitem__(layer, output[0]))
        for layer in layers if layer != 'l9']
    zs, logits, labels, names = {layer: [] for layer in layers}, [], [], []
    try:
        for data, label, index in tqdm(loader, ncols=40):
            data = data.float().to(device)
            features, _ = model.features(data, until='l9')
            valid = valid_actor_mask(data)
            N, M = data.shape[0], data.shape[-1]
            for layer in layers:
                out = features if layer == 'l9' else captured[layer].view(N, M, *captured[layer].shape[1:])
                zs[layer].append(pooled_z(out, valid).cpu())
            logits.append(fc_logits_from_features(model, features).cpu())
            labels.append(label)
            names.extend(index.tolist() if torch.is_tensor(index) else list(index))
    finally:
        for hook in hooks:
            hook.remove()
    return ({layer: torch.cat(z).numpy() for layer, z in zs.items()}, torch.cat(logits).numpy(),
            torch.cat(labels).numpy(), names)


def load_frozen_model(config, weights_path, device):
    Model = import_class(config['model'])
    model = Model(**config['model_args']).to(device).eval()
    if weights_path.endswith('.pkl'):
        with open(weights_path, 'rb') as handle:
            weights = pickle.load(handle)
    else:
        weights = torch.load(weights_path, map_location='cpu')
    report = load_model_weights(model, weights)
    return model, report


def build_loader(dataset, batch_size, num_worker, seed):
    # Workers draw their torch/numpy/random seeds from the main torch RNG at iteration start.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_worker, drop_last=False)


def cache(model, config, split, views, batch_size, num_worker, device, layers=('l9',)):
    """Returns {layer: arrays of one split's .npz} and the Top1 of fc_logits per view."""
    if split == 'test' and views != 1:
        raise ValueError('the test split is cached with a single view')
    # One dataset for all views: the NTU120 train array is ~11 GB, rebuilding it per view
    # would hold two copies at once. Augmentation happens in __getitem__, so views still differ.
    Feeder = import_class(config['feeder'])
    dataset = Feeder(**config['train_feeder_args' if split == 'train' else 'test_feeder_args'])

    z_views, logit_views, label_ref, names_ref, top1s = {layer: [] for layer in layers}, [], None, None, []
    for view in range(views):
        loader = build_loader(dataset, batch_size, num_worker, seed=view)
        zs, fc_logits, labels, names = run_pass(model, loader, device, layers)
        if label_ref is None:
            label_ref, names_ref = labels, names
        elif not np.array_equal(label_ref, labels):
            raise RuntimeError('label order changed between views; feeder must not shuffle')
        for layer in layers:
            z_views[layer].append(zs[layer])
        logit_views.append(fc_logits)
        top1 = float((fc_logits.argmax(1) == labels).mean())
        top1s.append(top1)
        shapes = ', '.join(f'z_{layer} {zs[layer].shape}' for layer in layers)
        print(f'{split} view {view}: {shapes}, fc_logits {fc_logits.shape}, '
              f'fc_logits top1 {100 * top1:.2f}%')

    logit_out = logit_views[0] if views == 1 else np.stack(logit_views)
    shared = dict(fc_logits=logit_out.astype(np.float32), label=label_ref.astype(np.int64),
                  index=np.array(names_ref, dtype=np.int64))
    arrays = {layer: dict(z=(z[0] if views == 1 else np.stack(z)).astype(np.float16), **shared)
              for layer, z in z_views.items()}
    return arrays, top1s


def cache_path(out_dir, name, layer, split):
    """l9 keeps the original <name>_<split>.npz; an earlier layer is <name>_<layer>_<split>.npz."""
    return os.path.join(out_dir, f'{name}_{split}.npz' if layer == 'l9' else f'{name}_{layer}_{split}.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out-dir', required=True, help='folder for <name>_{test,train}.npz')
    parser.add_argument('--config', default='config/large/nturgbd120-cross-subject/hyper_joint.yaml')
    parser.add_argument('--name', default='ntu120_xsub', help='file name prefix')
    parser.add_argument('--splits', nargs='+', choices=['test', 'train'], default=['test', 'train'])
    parser.add_argument('--views', type=int, default=5, help='random_rot passes of the train split')
    parser.add_argument('--layers', nargs='+', default=['l9'], choices=[f'l{i}' for i in range(1, 10)],
                        help='layers whose pooled output to cache, all from one backbone pass')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Arguments are relative to the caller's folder; data_path inside the config is relative
    # to the repo root, as for main.py, so work from there.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights = os.path.abspath(args.weights)
    out_dir = os.path.abspath(args.out_dir)
    config_path = args.config if os.path.isfile(args.config) else os.path.join(repo, args.config)
    config_path = os.path.abspath(config_path)
    os.chdir(repo)
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    model, report = load_frozen_model(config, weights, args.device)
    print(f'Checkpoint load: {report}')

    summary = dict(config=config_path, weights=weights, views=args.views, layers=args.layers, splits={})
    # Test first: it is short and carries the P0 pass criterion.
    for split in sorted(args.splits, key=['test', 'train'].index):
        arrays, top1s = cache(model, config, split, args.views if split == 'train' else 1,
                              args.batch_size, args.num_worker, args.device, args.layers)
        paths = {}
        for layer, layer_arrays in arrays.items():
            paths[layer] = cache_path(out_dir, args.name, layer, split)
            np.savez(paths[layer], **layer_arrays)
            print(f"Saved {paths[layer]}: z {layer_arrays['z'].shape} ({layer_arrays['z'].dtype}), "
                  f"fc_logits {layer_arrays['fc_logits'].shape}, label {layer_arrays['label'].shape}")
        first = next(iter(arrays.values()))
        summary['splits'][split] = dict(paths=paths, samples=int(first['label'].shape[0]),
                                        fc_logits_top1=top1s)
        del arrays, first

    summary_name = args.name if args.layers == ['l9'] else f"{args.name}_{'_'.join(args.layers)}"
    with open(os.path.join(out_dir, f'{summary_name}_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    if 'test' in summary['splits']:
        print(f"P0 check: test fc_logits top1 {100 * summary['splits']['test']['fc_logits_top1'][0]:.2f}% "
              '(must be within 0.1 points of the checkpoint\'s own Top1)')
