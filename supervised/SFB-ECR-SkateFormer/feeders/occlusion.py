"""Occlusion and Robustness Evaluation Utilities.

Supports:
1. Joint group occlusion (Arms, Legs, Torso, Head).
2. Temporal frame masking (continuous block masking or random dropout).
3. PyTorch Dataset wrapper for evaluation under occlusion.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# NTU-24 (Partitioned Joint ordering: 6 groups of 4 joints)
# Partitioned indices:
# Right Arm (0..3): [7, 8, 22, 23] - 1
# Left Arm (4..7): [11, 12, 24, 25] - 1
# Right Leg (8..11): [13, 14, 15, 16] - 1
# Left Leg (12..15): [17, 18, 19, 20] - 1
# Head & High Torso (16..19): [5, 9, 6, 10] - 1
# Waist & Low Torso (20..23): [2, 3, 1, 4] - 1

PARTITIONED_GROUPS = {
    'none': [],
    'right_arm': list(range(0, 4)),
    'left_arm': list(range(4, 8)),
    'arms': list(range(0, 8)),
    'right_leg': list(range(8, 12)),
    'left_leg': list(range(12, 16)),
    'legs': list(range(8, 16)),
    'head_torso': list(range(16, 20)),
    'waist_torso': list(range(20, 24)),
    'trunk': list(range(16, 24)),
}

def mask_joint_groups(data, group_name='arms', fill_value=0.0):
    """
    Mask specific joint groups in skeleton data.
    data: shape [C, T, V, M] or [B, C, T, V, M] (numpy or torch tensor)
    group_name: one of 'right_arm', 'left_arm', 'arms', 'right_leg', 'left_leg', 'legs', 'trunk', 'head_torso'
    """
    if group_name not in PARTITIONED_GROUPS or group_name == 'none':
        return data

    indices = PARTITIONED_GROUPS[group_name]
    
    if isinstance(data, np.ndarray):
        masked = data.copy()
        if masked.ndim == 4:  # [C, T, V, M]
            masked[:, :, indices, :] = fill_value
        elif masked.ndim == 5:  # [B, C, T, V, M]
            masked[:, :, :, indices, :] = fill_value
        return masked
    elif torch.is_tensor(data):
        masked = data.clone()
        if masked.ndim == 4:
            masked[:, :, indices, :] = fill_value
        elif masked.ndim == 5:
            masked[:, :, :, indices, :] = fill_value
        return masked
    return data

def mask_temporal_frames(data, mask_ratio=0.3, mode='continuous', fill_value=0.0):
    """
    Mask temporal frames in skeleton sequence.
    data: shape [C, T, V, M] or [B, C, T, V, M] (numpy or torch tensor)
    mask_ratio: fraction of frames to mask out (e.g. 0.2, 0.4, 0.6)
    mode: 'continuous' (consecutive block) or 'random' (random frame dropout)
    """
    if mask_ratio <= 0.0:
        return data

    is_torch = torch.is_tensor(data)
    T = data.shape[1] if data.ndim == 4 else data.shape[2]
    num_mask = max(1, int(round(T * mask_ratio)))

    if is_torch:
        masked = data.clone()
    else:
        masked = data.copy()

    if mode == 'continuous':
        start_t = np.random.randint(0, max(1, T - num_mask + 1))
        end_t = min(T, start_t + num_mask)
        if masked.ndim == 4:
            masked[:, start_t:end_t, :, :] = fill_value
        elif masked.ndim == 5:
            masked[:, :, start_t:end_t, :, :] = fill_value
    elif mode == 'random':
        mask_indices = np.random.choice(T, size=num_mask, replace=False)
        if masked.ndim == 4:
            masked[:, mask_indices, :, :] = fill_value
        elif masked.ndim == 5:
            masked[:, :, mask_indices, :, :] = fill_value

    return masked


class OcclusionDatasetWrapper(Dataset):
    """Wraps a skeleton dataset to apply joint or temporal occlusions during evaluation."""
    def __init__(self, base_dataset, joint_occlusion='none', temporal_mask_ratio=0.0, temporal_mask_mode='continuous'):
        self.base_dataset = base_dataset
        self.joint_occlusion = joint_occlusion
        self.temporal_mask_ratio = float(temporal_mask_ratio)
        self.temporal_mask_mode = temporal_mask_mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        # item is (data_numpy, index_t, label, index, [concepts])
        data_numpy = item[0]
        
        # Apply joint occlusion
        if self.joint_occlusion != 'none':
            data_numpy = mask_joint_groups(data_numpy, self.joint_occlusion)

        # Apply temporal masking
        if self.temporal_mask_ratio > 0.0:
            data_numpy = mask_temporal_frames(data_numpy, self.temporal_mask_ratio, self.temporal_mask_mode)

        if len(item) == 5:
            return data_numpy, item[1], item[2], item[3], item[4]
        return data_numpy, item[1], item[2], item[3]
