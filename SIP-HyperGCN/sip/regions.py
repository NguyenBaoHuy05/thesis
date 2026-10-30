"""Body regions of the NTU-25 skeleton and mean-ablation of their joints (tier 1, "where").

Joint indices are 0-based NTU RGB+D (Kinect v2) order:
0 spine base, 1 spine mid, 2 neck, 3 head, 4-7 L shoulder/elbow/wrist/hand,
8-11 R shoulder/elbow/wrist/hand, 12-15 L hip/knee/ankle/foot, 16-19 R hip/knee/ankle/foot,
20 spine shoulder, 21 L hand tip, 22 L thumb, 23 R hand tip, 24 R thumb.

==============  ==========================  =====================================
region          joints                      person slot ablated
==============  ==========================  =====================================
head            2, 3                        all persons
torso           0, 1, 20                    all persons
left_arm        4, 5, 6, 7, 21, 22          all persons
right_arm       8, 9, 10, 11, 23, 24        all persons
left_leg        12, 13, 14, 15              all persons
right_leg       16, 17, 18, 19              all persons
other_person    0..24 (all 25)              person slot 1 only
==============  ==========================  =====================================

The six body regions partition the 25 joints. ``other_person`` ablates the whole second
skeleton (person axis M), see ``region_ablation_args``.

Layout of a HyperGCN hidden state
---------------------------------
``Model.forward`` (hypergcn_base.py / sip.backbone.embed) builds the layer input as
``x.view(N, M, C, T, V)`` -> ``x.view(N * M, C, T, V)``, so row ``i`` of a layer output
``h`` of shape (N*M, C, T, V) is sample ``i // M``, person slot ``i % M``. Rows of one
sample are consecutive; ``h.view(N, M, C, T, V)`` recovers the person axis.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "REGION_NAMES",
    "REGIONS",
    "REGION_PERSON",
    "NUM_JOINTS",
    "region_joint_mask",
    "region_ablation_args",
    "random_joint_sets",
    "ablate_torch",
]

NUM_JOINTS = 25

REGION_NAMES: list[str] = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg", "other_person"]

REGIONS: dict[str, list[int]] = {
    "head": [2, 3],
    "torso": [0, 1, 20],
    "left_arm": [4, 5, 6, 7, 21, 22],
    "right_arm": [8, 9, 10, 11, 23, 24],
    "left_leg": [12, 13, 14, 15],
    "right_leg": [16, 17, 18, 19],
    "other_person": list(range(NUM_JOINTS)),
}

# person slot to ablate for each region (None = every person slot)
REGION_PERSON: dict[str, int | None] = {r: None for r in REGION_NAMES}
REGION_PERSON["other_person"] = 1

assert list(REGIONS) == REGION_NAMES
assert sorted(j for r in REGION_NAMES[:6] for j in REGIONS[r]) == list(range(NUM_JOINTS))


def region_joint_mask(name: str) -> np.ndarray:
    """(25,) bool mask of the region's joints (``other_person`` -> all True)."""
    if name not in REGIONS:
        raise KeyError(f"unknown region {name!r}; expected one of {REGION_NAMES}")
    m = np.zeros(NUM_JOINTS, dtype=bool)
    m[REGIONS[name]] = True
    return m


def region_ablation_args(name: str) -> tuple[list[int], int | None]:
    """(joints, person) to pass to ``ablate_torch`` for region ``name``."""
    if name not in REGIONS:
        raise KeyError(f"unknown region {name!r}; expected one of {REGION_NAMES}")
    return list(REGIONS[name]), REGION_PERSON[name]


def random_joint_sets(n: int, sizes: list[int], seed: int) -> list[list[int]]:
    """Random control joint sets (sampled without replacement from the 25 joints).

    Returns ``n`` sorted lists; set ``i`` has size ``sizes[i % len(sizes)]`` (clipped to
    [1, 25]). Pass e.g. ``sizes=[len(REGIONS[r]) for r in body regions]`` to get controls
    of matched sizes. Deterministic given ``seed`` (numpy default_rng).
    """
    if n <= 0:
        return []
    if not sizes:
        raise ValueError("sizes must be non-empty")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = int(min(max(int(sizes[i % len(sizes)]), 1), NUM_JOINTS))
        out.append(sorted(int(j) for j in rng.choice(NUM_JOINTS, size=k, replace=False)))
    return out


def ablate_torch(h, joints, mean_mcv, num_person: int = 2, person: int | None = None,
                 present=None):
    """Mean-ablate joints of a HyperGCN hidden state.

    Args:
        h: torch tensor (N*M, C, T, V); row i = sample i // M, person slot i % M.
        joints: iterable of joint indices along V to ablate.
        mean_mcv: torch tensor or numpy array (M, C, V): TRAIN-FIT mean of this layer's
            output over samples and T, per person slot.
        num_person: M.
        person: None -> ablate the joints in every person slot; int -> only that slot
            (``other_person`` uses person=1 and all 25 joints).
        present: optional (N, M) bool (torch or numpy). Person slots that are absent in
            the input are left untouched, so ablating a region never inserts a "mean
            person" into the empty slot of a single-person sample.
    Returns:
        A new tensor (same shape/dtype/device as h) where h[..., joints] of the selected
        person slot(s) is replaced by mean_mcv[slot, :, joints] broadcast over T.
        ``h`` itself is not modified.
    """
    import torch  # lazy: the rest of this module is numpy-only

    if h.dim() != 4:
        raise ValueError(f"expected h of shape (N*M, C, T, V), got {tuple(h.shape)}")
    NM, C, T, V = h.shape
    M = int(num_person)
    if NM % M != 0:
        raise ValueError(f"first dim {NM} not divisible by num_person={M}")
    N = NM // M
    mean = torch.as_tensor(mean_mcv).to(device=h.device, dtype=h.dtype)
    if tuple(mean.shape) != (M, C, V):
        raise ValueError(f"mean_mcv must have shape {(M, C, V)}, got {tuple(mean.shape)}")
    joints = [int(j) for j in joints]
    out = h.clone(memory_format=torch.contiguous_format)
    if not joints:
        return out
    if person is not None and not (0 <= int(person) < M):
        raise ValueError(f"person={person} out of range for M={M}")
    idx = torch.as_tensor(joints, dtype=torch.long, device=h.device)
    J = idx.numel()
    hv = out.view(N, M, C, T, V)
    fill = mean.index_select(2, idx)  # (M, C, J)
    if person is None:
        hv[..., idx] = fill[None, :, :, None, :].expand(N, M, C, T, J)
    else:
        p = int(person)
        # hv[:, p] is a view (N, C, T, V); indexing it avoids mixing an int with a
        # tensor index, whose result-dimension order differs between numpy and torch
        hv[:, p][..., idx] = fill[p][None, :, None, :].expand(N, C, T, J)
    if present is not None:
        keep = torch.as_tensor(present, device=h.device).bool()
        if tuple(keep.shape) != (N, M):
            raise ValueError(f"present must have shape {(N, M)}, got {tuple(keep.shape)}")
        out = torch.where(keep.reshape(NM, 1, 1, 1), out, h)
    return out
