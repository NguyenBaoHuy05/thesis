"""Feature extraction with the frozen Neuron Shift-GCN encoder, directly into GPU memory.

For every sample this gives
  pooled (256,)          : mean over time, joints and persons (the usual pooled feature)
  cube   (P=8, T=16, 256): last-layer features averaged over persons, joints grouped into body parts.
Preprocessing (center crop 95% of the valid frames + bilinear resize to 64 frames, as in Shift-GCN
pretraining test) runs batched on the GPU; gpu_preprocess reproduces feeders/tools.valid_crop_resize.
No disk cache: one NTU120 split takes a few minutes. Requires the nesy folder on sys.path.
"""
import glob
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
    """CPU reference (one sample). x: (T, 150) CTR-GCN format -> (C, T, V, M)."""
    T = x.shape[0]
    d = x.reshape(T, 2, 25, 3).transpose(3, 0, 2, 1)
    valid = int(np.sum(d.sum(0).sum(-1).sum(-1) != 0))
    return tools.valid_crop_resize(d, valid, list(p_interval), window)


def gpu_preprocess(xb, p=0.95, window=64):
    """Batched version of preprocess. xb: (B, T, 150) float tensor on GPU -> (B, 3, window, 25, 2).
    Same as tools.valid_crop_resize with p_interval=[p]: valid frames are the leading frames whose
    coordinate sum is non-zero; crop bias = int((1-p) * valid / 2) on both ends; then
    F.interpolate(bilinear, align_corners=False) along time, which samples
    src = (j + 0.5) * L / window - 0.5 (clamped at 0) between frames floor(src) and floor(src)+1."""
    B = xb.shape[0]
    valid = (xb.sum(-1) != 0).sum(1)                                            # (B,)
    bias = torch.floor((1 - p) * valid.double() / 2).long()
    L = valid - 2 * bias                                                        # cropped length
    j = torch.arange(window, device=xb.device, dtype=torch.float32)
    src = ((j[None] + 0.5) * (L.float()[:, None] / window) - 0.5).clamp(min=0)  # (B, window)
    i0 = torch.minimum(src.floor().long(), (L - 1)[:, None])
    lam = (src - i0.float())[..., None]
    i1 = torch.minimum(i0 + 1, (L - 1)[:, None])
    g = lambda i: torch.gather(xb, 1, (bias[:, None] + i)[..., None].expand(-1, -1, xb.shape[-1]))
    out = (1 - lam) * g(i0) + lam * g(i1)                                       # (B, window, 150)
    return out.view(B, window, 2, 25, 3).permute(0, 4, 1, 3, 2).contiguous()   # (B, C, T, V, M)


@torch.no_grad()
def encode(model, X, idx, bs=512, device='cuda'):
    """X: numpy (N, T, 150); idx: sample indices. Returns pooled (n, 256) fp32 and cube (n, P, 16, 256)
    fp16, both on device."""
    part_idx = [torch.tensor(j, device=device) for j in PARTS.values()]
    pooled = torch.empty(len(idx), 256, device=device)
    cube = torch.empty(len(idx), len(PARTS), 16, 256, dtype=torch.float16, device=device)
    for i in range(0, len(idx), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[idx[i:i + bs]])).to(device).float()
        _, f, c = model(gpu_preprocess(xb))                                     # c: (n, 256, 16, 25)
        parts = torch.stack([c.index_select(3, j).mean(3) for j in part_idx], 1)   # (n, P, 256, 16)
        pooled[i:i + len(xb)] = f
        cube[i:i + len(xb)] = parts.permute(0, 1, 3, 2).half()
    return pooled, cube


def encode_task(skel, weights_dir, task, unseen, kind='cube', log=print):
    """skel: dict with x_train, y_train (one-hot), x_test, y_test (CTR-GCN npz in RAM).
    Returns dict Xtr, ytr (seen classes only, original order), Xte, yte (all test samples) on GPU.
    kind: 'cube' (P, 16, 256) fp16 | 'pooled' (256,) | 'both' (adds Ptr / Pte = pooled)."""
    enc = load_encoder(weights_dir, task)
    out, t0 = {}, time.time()
    for split, k in (('train', 'tr'), ('test', 'te')):
        y = np.argmax(skel[f'y_{split}'], 1)
        idx = np.where(~np.isin(y, unseen))[0] if split == 'train' else np.arange(len(y))
        pooled, cube = encode(enc, skel[f'x_{split}'], idx)
        out[f'X{k}'] = pooled if kind == 'pooled' else cube
        if kind == 'both':
            out[f'P{k}'] = pooled
        out[f'y{k}'] = torch.as_tensor(y[idx], device='cuda')
        del pooled, cube
    del enc
    torch.cuda.empty_cache()
    log(f'[{task}] encoded: train {len(out["ytr"])}, test {len(out["yte"])}, {time.time() - t0:.0f}s')
    return out
