"""Feature extraction with the frozen Neuron Shift-GCN encoder.

For every sample this gives
  pooled (256,)          : mean over time, joints and persons (the usual pooled feature)
  cube   (P=8, T=16, 256): last-layer features averaged over persons, joints grouped into body parts.
The cube is cached as fp16 .npy files (memory-mapped while writing), so RAM stays small.
Requires the nesy folder on sys.path (imports model.*, feeders.*, graph.*).
"""
import glob
import os
import time

import numpy as np
import torch

from feeders import tools
from model.shiftgcn_match import Model

# NTU joints, 0-based. Hand tips / thumbs go with the hand, wrists with the arm.
PARTS = {
    'torso':  [0, 1, 20],
    'head':   [2, 3],
    'l_arm':  [4, 5, 6],
    'r_arm':  [8, 9, 10],
    'l_hand': [7, 21, 22],
    'r_hand': [11, 23, 24],
    'l_leg':  [12, 13, 14, 15],
    'r_leg':  [16, 17, 18, 19],
}
PART_NAMES = list(PARTS)


def weight_path(weights_dir, task):
    ds, prot, s, u = task.split('_')
    hits = glob.glob(f'{weights_dir}/**/{ds}_ShiftGCN_joint_{prot}_{s}_{u}-*.pt', recursive=True)
    assert len(hits) == 1, (task, hits)
    return hits[0]


def load_encoder(weights_dir, task, device='cuda'):
    sd = torch.load(weight_path(weights_dir, task), map_location='cpu')
    sd = {k.split('module.')[-1]: v for k, v in sd.items()}
    m = Model(num_class=sd['fc.weight'].shape[0], num_point=25, num_person=2,
              graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def preprocess(x, p_interval=(0.95,), window=64):
    """x: (T, 150) CTR-GCN format -> (C, T, V, M); center crop 95% as in Shift-GCN pretraining test."""
    T = x.shape[0]
    d = x.reshape(T, 2, 25, 3).transpose(3, 0, 2, 1)
    valid = int(np.sum(d.sum(0).sum(-1).sum(-1) != 0))
    return tools.valid_crop_resize(d, valid, list(p_interval), window)


@torch.no_grad()
def extract_to(model, X, idx, pooled_out, cube_out, bs=256, device='cuda'):
    """Runs X[idx] through the encoder; writes pooled_out[i] (256,) and cube_out[i] (P, T, 256)."""
    part_idx = [torch.tensor(j, device=device) for j in PARTS.values()]
    for i in range(0, len(idx), bs):
        xb = np.stack([preprocess(X[j]) for j in idx[i:i + bs]])
        _, f, cube = model(torch.from_numpy(xb).float().to(device))      # cube: (n, 256, 16, 25)
        parts = torch.stack([cube.index_select(3, j).mean(3) for j in part_idx], 1)   # (n, P, 256, 16)
        pooled_out[i:i + len(xb)] = f.cpu().numpy()
        cube_out[i:i + len(xb)] = parts.permute(0, 1, 3, 2).half().cpu().numpy()     # (n, P, 16, 256)


def cache_paths(cache_dir, task, split):
    base = f'{cache_dir}/{task}_{split}'
    return dict(pooled=f'{base}_pooled.npy', cube=f'{base}_cube.npy', label=f'{base}_label.npy')


def extract_split(skel, weights_dir, task, unseen, cache_dir, log=print):
    """skel: dict with x_train, y_train, x_test, y_test (CTR-GCN npz loaded in RAM).
    Writes train (seen classes only, original order) and test (all samples) caches. Skips if present."""
    os.makedirs(cache_dir, exist_ok=True)
    todo = [s for s in ('train', 'test') if not os.path.exists(cache_paths(cache_dir, task, s)['label'])]
    if not todo:
        log(f'[{task}] cache có sẵn')
        return
    enc = load_encoder(weights_dir, task)
    for split in todo:
        X, y = skel[f'x_{split}'], np.argmax(skel[f'y_{split}'], 1)
        idx = np.where(~np.isin(y, unseen))[0] if split == 'train' else np.arange(len(y))
        p = cache_paths(cache_dir, task, split)
        pooled = np.lib.format.open_memmap(p['pooled'], mode='w+', dtype=np.float32, shape=(len(idx), 256))
        cube = np.lib.format.open_memmap(p['cube'], mode='w+', dtype=np.float16, shape=(len(idx), len(PARTS), 16, 256))
        t0 = time.time()
        extract_to(enc, X, idx, pooled, cube)
        pooled.flush()
        cube.flush()
        del pooled, cube
        np.save(p['label'], y[idx])          # written last: marks the cache as complete
        log(f'[{task}] {split}: {len(idx)} mẫu, {time.time() - t0:.0f}s')


def load_cache(cache_dir, task, kind='cube', device='cuda', chunk=8192):
    """Returns dict Xtr, ytr, Xte, yte as tensors on device (cube stays fp16)."""
    out = {}
    for split, k in (('train', 'tr'), ('test', 'te')):
        p = cache_paths(cache_dir, task, split)
        arr = np.load(p[kind], mmap_mode='r')
        t = torch.empty(arr.shape, dtype=torch.float16 if kind == 'cube' else torch.float32, device=device)
        for i in range(0, len(arr), chunk):           # copy in chunks: no full copy in RAM
            t[i:i + chunk] = torch.from_numpy(np.ascontiguousarray(arr[i:i + chunk])).to(device)
        out[f'X{k}'] = t
        out[f'y{k}'] = torch.as_tensor(np.load(p['label']), device=device)
    return out
