"""
Data Preprocessing and Augmentation Tools for Skeleton Action Recognition
========================================================================
Supports:
- Valid Crop and Resize (linear interpolation / bilinear)
- Temporal & Spatial augmentations (rotation, shear, scaling, flipping)
- Modality conversions (Joint, Bone, Velocity, Bone-Velocity)
"""

import math
import random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# NTU 25-Joint Bone Connections (1-indexed converted to 0-indexed)
NTU_PAIRS = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 8), (21, 21), (23, 8), (24, 12), (25, 12)
)

NTU_PAIRS_0 = tuple((v1 - 1, v2 - 1) for v1, v2 in NTU_PAIRS)

TRANSFORM_ORDER_NTU = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]


def valid_crop_resize(
    data_numpy: np.ndarray,
    valid_frame_num: int,
    p_interval: Tuple[float, ...],
    window: int,
    thres: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops valid temporal segments and resizes to target window length.
    Input: data_numpy of shape (C, T, V, M)
    Returns: (resized_data (C, window, V, M), index_t (window,))
    """
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num if valid_frame_num > 0 else T
    valid_size = max(end - begin, 1)

    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        cropped_length = int(valid_size * p)
        c_b = begin + bias
        c_e = min(begin + bias + cropped_length, end)
        data = data_numpy[:, c_b:c_e, :, :]
    else:
        p = np.random.rand(1)[0] * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = int(np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size))
        bias = np.random.randint(0, max(valid_size - cropped_length + 1, 1))
        c_b = begin + bias
        c_e = begin + bias + cropped_length
        data = data_numpy[:, c_b:c_e, :, :]

    if data.shape[1] == 0:
        data = data_numpy[:, :window, :, :]
        c_b, c_e = 0, min(T, window)

    curr_len = data.shape[1]
    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_data = tensor_data.permute(2, 3, 0, 1).contiguous().view(V * M, C, curr_len)
    if curr_len != window:
        tensor_data = F.interpolate(tensor_data, size=window, mode="linear", align_corners=False)
    out_data = tensor_data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()

    index_t = torch.arange(start=c_b, end=c_e, dtype=torch.float32)
    if len(index_t) != window:
        index_t = F.interpolate(index_t[None, None, :], size=window, mode="linear", align_corners=False).squeeze()
    index_t = 2.0 * index_t / max(valid_size, 1) - 1.0

    return out_data, index_t.numpy()


def to_bone(data_numpy: np.ndarray, pairs: Tuple[Tuple[int, int], ...] = NTU_PAIRS_0) -> np.ndarray:
    """
    Converts joint coordinates (C, T, V, M) into bone vectors:
    bone[j] = joint[j] - joint[parent(j)].
    """
    C, T, V, M = data_numpy.shape
    bone_data = np.zeros_like(data_numpy)
    for v1, v2 in pairs:
        bone_data[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
    return bone_data


def to_velocity(data_numpy: np.ndarray) -> np.ndarray:
    """
    Computes temporal difference: vel[:, t] = data[:, t+1] - data[:, t].
    """
    C, T, V, M = data_numpy.shape
    vel_data = np.zeros_like(data_numpy)
    vel_data[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]
    vel_data[:, -1, :, :] = vel_data[:, -2, :, :] if T > 1 else 0.0
    return vel_data


def auto_padding(data_numpy: np.ndarray, size: int, random_pad: bool = False) -> np.ndarray:
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        pad_data = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
        pad_data[:, begin : begin + T, :, :] = data_numpy
        return pad_data
    return data_numpy


def random_choose(data_numpy: np.ndarray, size: int, auto_pad: bool = True) -> np.ndarray:
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    if T < size:
        return auto_padding(data_numpy, size, random_pad=True) if auto_pad else data_numpy
    begin = random.randint(0, T - size)
    return data_numpy[:, begin : begin + size, :, :]


def random_rot(data_numpy: np.ndarray, theta: float = 0.3) -> np.ndarray:
    """Random 3D spatial rotation around X, Y, Z axes."""
    C, T, V, M = data_numpy.shape
    if C < 3:
        return data_numpy

    thetas = np.random.uniform(-theta, theta, size=3)
    cx, cy, cz = np.cos(thetas)
    sx, sy, sz = np.sin(thetas)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx

    reshaped = data_numpy[:3].reshape(3, -1)
    rotated = R @ reshaped
    out = data_numpy.copy()
    out[:3] = rotated.reshape(3, T, V, M)
    return out


def random_move(
    data_numpy: np.ndarray,
    angle_candidate=[-10.0, -5.0, 0.0, 5.0, 10.0],
    scale_candidate=[0.9, 1.0, 1.1],
    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
    move_time_candidate=[1],
) -> np.ndarray:
    """Random non-linear spatial translation & perturbation."""
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, max(1, T * 1.0 / move_time)).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    dx = np.random.choice(transform_candidate, num_node)
    dy = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    b = np.zeros(T)
    for i in range(num_node - 1):
        a[node[i] : node[i + 1]] = np.linspace(dx[i], dx[i + 1], node[i + 1] - node[i])
        b[node[i] : node[i + 1]] = np.linspace(dy[i], dy[i + 1], node[i + 1] - node[i])

    out = data_numpy.copy()
    for p in range(M):
        out[0, :, :, p] += a[:, None]
        out[1, :, :, p] += b[:, None]
    return out


def spatial_flip(data_numpy: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Spatially flip skeleton joints left-to-right based on NTU layout."""
    if random.random() < p:
        order = TRANSFORM_ORDER_NTU
        out = data_numpy.copy()
        out = out[:, :, order, :]
        out[0] = -out[0]  # Invert X coordinate
        return out
    return data_numpy
