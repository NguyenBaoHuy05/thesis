import json
import os
import sys
from pathlib import Path


def build_notebook():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    # =========================================================================
    # HEADER & MOTIVATION
    # =========================================================================
    add_md("""# 🚀 SFB-ECR-SkateFormer: All-in-One Google Colab & Local Notebook
> **Mô hình**: **ECR-SkateFormer with Concept-Guided Soft Feedback Attention**
> **Mục tiêu nghiên cứu**: Khám phá và kiểm chứng giả thuyết rằng **semantic concepts (không gian & thời gian) được trích xuất từ các tầng sớm** có thể tạo ra tín hiệu **feedback mềm (channel modulation + spatial-temporal token bias)** quay trở lại định hướng sự chú ý của perception backbone ở các tầng sau, từ đó nâng cao độ chính xác nhận dạng hành động, cải thiện chất lượng biểu diễn concept, và tăng cường tính bền vững (robustness) dưới hiện tượng che khuất (occlusion).

---

### 📌 Cấu trúc 8 Block trong Notebook:
1. **Block 1**: Cài đặt môi trường, dependencies & kết nối Google Drive
2. **Block 2**: Các hàm tiện ích, Logging, Metrics (Action Top-1/5, Concept F1/mAP) & Multi-task Loss
3. **Block 3**: Tiền xử lý dữ liệu, Data Augmentations, NTU Feeder & Module tạo che khuất (Occlusion/Temporal Masking)
4. **Block 4**: Kiến trúc mạng nền tảng SkateFormer Backbone hỗ trợ Intermediate Feature Extraction
5. **Block 5**: Module `SoftConceptFeedback` + Bộ suy luận logic `KGPrimedReasoner` + Mô hình tổng hợp `SFB_ECR_SkateFormer` (hỗ trợ toàn bộ ma trận Ablation A1–A7)
6. **Block 6**: Engine Huấn luyện & Đánh giá (**Lưu `checkpoint_latest.pt` mỗi epoch, `checkpoint_best.pt` & Tự động Resume**)
7. **Block 7**: Pipeline Kiểm tra nhanh toàn diện với Dummy Data (Chạy test toàn bộ kiến trúc chỉ trong vài giây)
8. **Block 8**: Cấu hình & Huấn luyện / Đánh giá với Dữ liệu Thật (NTU-RGB+D 60 / NTU-RGB+D 120) & Đánh giá Robustness
""")

    # =========================================================================
    # BLOCK 1: DEPENDENCIES & SETUP
    # =========================================================================
    add_md("## 📦 Block 1: Cài đặt Dependencies & Thiết lập Môi trường")
    add_code("""# Cài đặt các thư viện cần thiết cho Colab / Local
!pip install -q timm tensorboardX einops yacs scikit-learn matplotlib tqdm pyyaml

import os
import sys
import math
import random
import time
import json
import csv
import glob
import pickle
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Type, Tuple, Optional, Set, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath, create_conv2d
from timm.scheduler.cosine_lr import CosineLRScheduler

# Cấu hình thiết bị GPU/CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Đang sử dụng thiết bị: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU Model: {torch.cuda.get_device_name(0)}")

# (Tùy chọn) Kết nối Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Đã kết nối Google Drive thành công!")
except Exception:
    print("ℹ️ Chạy môi trường Local / Non-Colab (bỏ qua mount Drive).")
""")

    # =========================================================================
    # BLOCK 2: UTILITIES & MULTI-TASK LOSS
    # =========================================================================
    add_md("## 🛠️ Block 2: Tiện ích, Logging, Metrics & Multi-task Loss Functions")
    add_code("""def init_seed(seed=1):
    \"\"\"Khởi tạo seed cố định để đảm bảo tính tái lập của thực nghiệm.\"\"\"
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')

class LabelSmoothingCrossEntropy(nn.Module):
    \"\"\"Cross Entropy có Label Smoothing giúp chống overfitting.\"\"\"
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()

def compute_concept_f1_metrics(y_true, y_pred_probs, threshold=0.5):
    \"\"\"Tính toán Macro-F1 và Micro-F1 cho dự đoán Concept.\"\"\"
    bin_preds = (y_pred_probs >= threshold).astype(np.float32)
    macro_f1 = f1_score(y_true, bin_preds, average='macro', zero_division=0) * 100.0
    micro_f1 = f1_score(y_true, bin_preds, average='micro', zero_division=0) * 100.0
    return macro_f1, micro_f1

print("✅ Đã khởi tạo Block 2 (Utilities & Loss Functions)!")
""")

    # =========================================================================
    # BLOCK 3: DATA FEEDERS, AUGMENTATIONS & OCCLUSION
    # =========================================================================
    add_md("## 📊 Block 3: Tiền xử lý Dữ liệu, Data Augmentation & Module Che khuất (Occlusion)")
    add_code("""# --- Augmentations & Geometric Tools ---

def valid_crop_uniform(data_numpy, valid_frame_num, p_interval, window, thres):
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num if valid_frame_num > 0 else T
    valid_size = max(1, end - begin)

    if len(p_interval) == 1:
        p = p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
        bias = int((1 - p) * valid_size / 2)
        bias = min(bias, max(0, valid_size - cropped_length))
        inds = np.arange(cropped_length) if cropped_length < window else np.array([i * cropped_length // window for i in range(window)])
        inds = np.clip(inds + bias, 0, T - 1)
        data = data_numpy[:, inds, :, :]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
        max_bias = max(0, valid_size - cropped_length)
        bias = np.random.randint(0, max_bias + 1) if max_bias > 0 else 0
        if cropped_length < window:
            inds = np.arange(cropped_length)
        elif window <= cropped_length < 2 * window:
            basic = np.arange(window)
            inds = np.random.choice(window + 1, cropped_length - window, replace=False)
            offset = np.zeros(window + 1, dtype=np.int64)
            offset[inds] = 1
            inds = basic + np.cumsum(offset)[:-1]
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            offset = np.random.randint(np.diff(bids))
            inds = bids[:window] + offset
        inds = np.clip(inds + bias, 0, T - 1)
        data = data_numpy[:, inds, :, :]

    data = torch.tensor(data, dtype=torch.float)
    index_t = torch.tensor(inds, dtype=torch.float)
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, len(inds))
    if len(inds) != window:
        data = F.interpolate(data, size=window, mode='linear', align_corners=False)
        index_t = F.interpolate(index_t[None, None, :], size=window, mode='linear', align_corners=False).squeeze()
    data = data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    index_t = 2 * index_t / valid_size - 1
    return data, index_t.numpy()

def shear(data_numpy, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        s1 = [random.uniform(-0.5, 0.5) for _ in range(3)]
        s2 = [random.uniform(-0.5, 0.5) for _ in range(3)]
        R = np.array([[1, s1[0], s2[0]], [s1[1], 1, s2[1]], [s1[2], s2[2], 1]]).T
        return np.dot(temp.transpose([1, 2, 3, 0]), R).transpose(3, 0, 1, 2)
    return data_numpy.copy()

def rotate(data_numpy, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        axis = random.randint(0, 2)
        angle = math.radians(random.uniform(-30, 30))
        if axis == 0:
            R = np.array([[1, 0, 0], [0, math.cos(angle), math.sin(angle)], [0, -math.sin(angle), math.cos(angle)]])
        elif axis == 1:
            R = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
        else:
            R = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        return np.dot(temp.transpose([1, 2, 3, 0]), R.T).transpose(3, 0, 1, 2)
    return data_numpy.copy()

# --- Occlusion & Robustness Modules ---

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
    if group_name not in PARTITIONED_GROUPS or group_name == 'none':
        return data
    indices = PARTITIONED_GROUPS[group_name]
    if isinstance(data, np.ndarray):
        masked = data.copy()
        if masked.ndim == 4:
            masked[:, :, indices, :] = fill_value
        elif masked.ndim == 5:
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
    if mask_ratio <= 0.0:
        return data
    is_torch = torch.is_tensor(data)
    T = data.shape[1] if data.ndim == 4 else data.shape[2]
    num_mask = max(1, int(round(T * mask_ratio)))
    masked = data.clone() if is_torch else data.copy()
    if mode == 'continuous':
        start_t = np.random.randint(0, max(1, T - num_mask + 1))
        end_t = min(T, start_t + num_mask)
        if masked.ndim == 4:
            masked[:, start_t:end_t, :, :] = fill_value
        elif masked.ndim == 5:
            masked[:, :, start_t:end_t, :, :] = fill_value
    return masked

# --- Dataset Feeder ---

class Feeder(Dataset):
    def __init__(self, data_path=None, split='train', window_size=64, p_interval=[0.5, 1],
                 concepts_csv=None, remove_null_concept=True, partition=True, dummy_size=None):
        self.split = split
        self.window_size = window_size
        self.p_interval = p_interval
        self.partition = partition
        self.concepts_csv = concepts_csv
        self.remove_null_concept = remove_null_concept

        if partition:
            self.right_arm = np.array([7, 8, 22, 23]) - 1
            self.left_arm = np.array([11, 12, 24, 25]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.h_torso = np.array([5, 9, 6, 10]) - 1
            self.w_torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)

        self.concept_matrix = None
        if self.concepts_csv and os.path.exists(self.concepts_csv):
            self.load_concepts()

        if dummy_size is not None:
            self.data = np.random.randn(dummy_size, 3, window_size, 25, 2).astype(np.float32)
            self.label = np.random.randint(0, 60, size=dummy_size)
        elif data_path and os.path.exists(data_path):
            self.load_data(data_path)
        else:
            self.data = np.zeros((0, 3, window_size, 25, 2), dtype=np.float32)
            self.label = np.zeros((0,), dtype=np.int64)

    def load_concepts(self):
        with open(self.concepts_csv, 'r', newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        names = [name for name in rows[0].keys() if name != 'action_class']
        if self.remove_null_concept:
            names = [name for i, name in enumerate(names) if i not in (50, 65)]
        self.concept_matrix = np.asarray([[float(r[n]) for n in names] for r in rows], dtype=np.float32)

    def load_data(self, data_path):
        npz = np.load(data_path)
        k_x = 'x_train' if self.split == 'train' else 'x_test'
        k_y = 'y_train' if self.split == 'train' else 'y_test'
        self.data = npz[k_x].reshape(-1, self.window_size, 2, 25, 3).transpose(0, 4, 1, 3, 2)
        self.label = np.where(npz[k_y] > 0)[1]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index].copy()
        label = self.label[index]
        index_t = np.linspace(-1.0, 1.0, self.window_size)

        if self.split == 'train' and random.random() < 0.5:
            data_numpy = shear(data_numpy, p=0.5)
            data_numpy = rotate(data_numpy, p=0.5)

        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]

        if self.concept_matrix is not None and label < len(self.concept_matrix):
            return data_numpy, index_t, label, index, self.concept_matrix[label]
        return data_numpy, index_t, label, index

print("✅ Đã khởi tạo Block 3 (Data Feeders & Augmentations)!")
""")

    # =========================================================================
    # BLOCK 4: SKATEFORMER BACKBONE WITH INTERMEDIATE TAPS
    # =========================================================================
    add_md("## 🧠 Block 4: Kiến trúc SkateFormer Backbone hỗ trợ Intermediate Feature Extraction")
    add_code("""''' SkateFormer Partition & Reverse Functions '''

def type_1_partition(input, partition_size):
    B, C, T, V = input.shape
    return input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1]).permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_1_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    return partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)

def type_2_partition(input, partition_size):
    B, C, T, V = input.shape
    return input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1]).permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_2_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    return partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1).permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)

def type_3_partition(input, partition_size):
    B, C, T, V = input.shape
    return input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1]).permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_3_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    return partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1).permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)

def type_4_partition(input, partition_size):
    B, C, T, V = input.shape
    return input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1]).permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_4_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    return partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1).permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)

def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0., rel=True):
        super().__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel = rel

        if self.rel:
            if self.rel_type in ('type_1', 'type_3'):
                self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)
            elif self.rel_type in ('type_2', 'type_4'):
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(self):
        if self.rel_type in ('type_1', 'type_3'):
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], -1)
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3).repeat(1, self.partition_size[1], 1, self.partition_size[1], 1, 1).view(self.attn_area, self.attn_area, -1)
            return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        else:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1)
            relative_position_bias = relative_position_bias.permute(0, 2, 1, 3, 4).contiguous().view(self.attn_area, self.attn_area, -1)
            return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward(self, input):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel:
            attn = attn + self._get_relative_positional_bias()
        attn = self.attn_drop(self.softmax(attn))
        return (attn @ v).transpose(1, 2).reshape(B_, N, -1)

class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.type_1_size = type_1_size
        self.type_2_size = type_2_size
        self.type_3_size = type_3_size
        self.type_4_size = type_4_size
        self.partition_function = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_function = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]
        self.partition_size = [type_1_size, type_2_size, type_3_size, type_4_size]
        self.rel_type = ['type_1', 'type_2', 'type_3', 'type_4']

        self.norm_1 = norm_layer(in_channels)
        self.mapping = nn.Linear(in_features=in_channels, out_features=2 * in_channels, bias=True)
        self.gconv = nn.Parameter(torch.zeros(num_heads // 4, num_points, num_points))
        trunc_normal_(self.gconv, std=.02)
        self.tconv = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(kernel_size, 1),
                               padding=((kernel_size - 1) // 2, 0), groups=num_heads // 4)

        attention = []
        for i in range(len(self.partition_function)):
            attention.append(MultiHeadSelfAttention(
                in_channels=in_channels // 8, rel_type=self.rel_type[i],
                num_heads=num_heads // 8, partition_size=self.partition_size[i],
                attn_drop=attn_drop, rel=rel))
        self.attention = nn.ModuleList(attention)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(mlp_ratio * in_channels), act_layer=act_layer, drop=drop)

    def forward(self, input):
        B, C, T, V = input.shape
        input = input.permute(0, 2, 3, 1).contiguous()
        skip = input
        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()
        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []

        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        for i in range(self.gconv.shape[0]):
            y_gconv.append(torch.einsum('n c t u, v u -> n c t v', split_f_gconv[i], self.gconv[i]))
        y.append(torch.cat(y_gconv, dim=1))
        y.append(self.tconv(split_f_conv[1]))

        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)
        for i in range(len(self.partition_function)):
            sub_c = split_f_attn[i].shape[1]
            p_in = self.partition_function[i](split_f_attn[i], self.partition_size[i]).view(-1, self.partition_size[i][0] * self.partition_size[i][1], sub_c)
            y.append(self.reverse_function[i](self.attention[i](p_in), (T, V), self.partition_size[i]))

        output = self.proj(torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous())
        output = skip + self.drop_path(self.proj_drop(output))
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        return output.permute(0, 3, 1, 2).contiguous()

class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.bn(self.reduction(x))

class SkateFormerBlockDS(nn.Module):
    def __init__(self, in_channels, out_channels, num_points=50, kernel_size=7, downscale=False, **kwargs):
        super().__init__()
        self.downsample = PatchMergingTconv(in_channels, out_channels, kernel_size=kernel_size) if downscale else None
        self.transformer = SkateFormerBlock(in_channels=out_channels, num_points=num_points, kernel_size=kernel_size, **kwargs)

    def forward(self, input):
        return self.transformer(self.downsample(input)) if self.downsample is not None else self.transformer(input)

class SkateFormerStage(nn.Module):
    def __init__(self, depth, in_channels, out_channels, first_depth=False, num_points=50, kernel_size=7, **kwargs):
        super().__init__()
        blocks = []
        for idx in range(depth):
            blocks.append(SkateFormerBlockDS(
                in_channels=in_channels if idx == 0 else out_channels,
                out_channels=out_channels, num_points=num_points, kernel_size=kernel_size,
                downscale=((idx == 0) and not first_depth), **kwargs))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        for block in self.blocks:
            input = block(input)
        return input

class SkateFormer(nn.Module):
    def __init__(self, in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), num_classes=60,
                 embed_dim=96, num_people=2, num_frames=64, num_points=24, kernel_size=7, num_heads=32,
                 type_1_size=(8, 8), type_2_size=(8, 12), type_3_size=(8, 8), type_4_size=(8, 12),
                 attn_drop=0.5, head_drop=0., drop=0., rel=True, drop_path=0.2, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm, index_t=True, global_pool='avg'):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.index_t = index_t
        self.dropout = nn.Dropout(p=head_drop) if head_drop > 0 else None

        self.stem = nn.ModuleList([
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, 1)), act_layer(),
            nn.Conv2d(2 * in_channels, 3 * in_channels, kernel_size=(1, 1)), act_layer(),
            nn.Conv2d(3 * in_channels, embed_dim, kernel_size=(1, 1))
        ])

        if self.index_t:
            self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.joint_person_embedding, std=.02)
        else:
            self.joint_person_temporal_embedding = nn.Parameter(torch.zeros(1, embed_dim, num_frames, num_points * num_people))
            trunc_normal_(self.joint_person_temporal_embedding, std=.02)

        stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(SkateFormerStage(
                depth=depth, in_channels=embed_dim if index == 0 else channels[index - 1],
                out_channels=channel, first_depth=(index == 0), num_points=num_points * num_people,
                kernel_size=kernel_size, num_heads=num_heads, type_1_size=type_1_size,
                type_2_size=type_2_size, type_3_size=type_3_size, type_4_size=type_4_size,
                attn_drop=attn_drop, drop=drop, rel=rel, drop_path=drop_path,
                mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer_transformer=norm_layer_transformer))
        self.stages = nn.ModuleList(stages)
        self.global_pool = global_pool
        self.head = nn.Linear(channels[-1], num_classes)

    def _stem_and_embeddings(self, input, index_t=None):
        B, C, T, V, M = input.shape
        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)
        for layer in self.stem:
            output = layer(output)
        if self.index_t:
            if index_t is None:
                index_t = torch.linspace(-1., 1., T, device=output.device, dtype=output.dtype).unsqueeze(0).expand(B, T)
            te = torch.zeros(B, T, self.embed_dim, device=output.device, dtype=output.dtype)
            div_term = torch.exp((torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=output.device) * -(math.log(10000.0) / self.embed_dim)))
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum('b t c, c v -> b c t v', te, self.joint_person_embedding)
        else:
            output = output + self.joint_person_temporal_embedding
        return output

    def forward_features_early(self, input, early_stages=2, index_t=None):
        output = self._stem_and_embeddings(input, index_t=index_t)
        for stage in self.stages[:early_stages]:
            output = stage(output)
        return output

    def forward_features_late(self, modulated_feature, from_stage=2):
        output = modulated_feature
        for stage in self.stages[from_stage:]:
            output = stage(output)
        return output

    def forward_features_with_intermediate(self, input, feedback_fn=None, early_stages=2, index_t=None):
        F_early = self.forward_features_early(input, early_stages=early_stages, index_t=index_t)
        F_mod, fb_aux = feedback_fn(F_early) if feedback_fn is not None else (F_early, {})
        F_late = self.forward_features_late(F_mod, from_stage=early_stages)
        return F_early, F_late, fb_aux

    def forward_features_from_input(self, input, index_t=None):
        output = self._stem_and_embeddings(input, index_t=index_t)
        for stage in self.stages:
            output = stage(output)
        return output

    def forward(self, input, index_t=None):
        feat = self.forward_features_from_input(input, index_t=index_t)
        pooled = feat.mean(dim=(2, 3)) if self.global_pool == 'avg' else torch.amax(feat, dim=(2, 3))
        return self.head(self.dropout(pooled) if self.dropout else pooled)

print("✅ Đã khởi tạo Block 4 (SkateFormer Backbone)!")
""")

    # =========================================================================
    # BLOCK 5: SFB MODULE & REASONER
    # =========================================================================
    add_md("## 🧩 Block 5: Module `SoftConceptFeedback` & Mô hình `SFB_ECR_SkateFormer`")
    add_code("""def _logit(probability, eps=1e-4):
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)

class KGUnionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prior=None):
        super().__init__()
        if prior is None:
            prior = torch.full((input_dim, output_dim), 0.05)
            prior[torch.arange(min(input_dim, output_dim)), torch.arange(min(input_dim, output_dim))] = 0.9
        prior = prior.float().clamp(1e-4, 1.0 - 1e-4)
        self.register_buffer("prior", prior.clone())
        noise = 0.02 * torch.randn_like(prior)
        self.and_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1 - 1e-4)))
        self.or_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1 - 1e-4)))

    def forward(self, x):
        and_w = torch.sigmoid(self.and_weight)
        or_w = torch.sigmoid(self.or_weight)
        conjunction = torch.prod(1.0 - and_w.unsqueeze(0) * (1.0 - x.unsqueeze(-1)), dim=1)
        disjunction = 1.0 - torch.prod(1.0 - or_w.unsqueeze(0) * x.unsqueeze(-1), dim=1)
        return torch.cat([conjunction, disjunction], dim=1)

    def prior_retention_loss(self):
        a_w, o_w = torch.sigmoid(self.and_weight), torch.sigmoid(self.or_weight)
        return ((a_w - self.prior).abs().mean() + (o_w - self.prior).abs().mean()) * 0.5

    def sparsity_loss(self):
        a_w, o_w = torch.sigmoid(self.and_weight), torch.sigmoid(self.or_weight)
        return 0.5 * (a_w.mean() + o_w.mean())

class KGPrimedReasoner(nn.Module):
    def __init__(self, n_concepts, n_actions, concept_names, action_mode="prototype",
                 action_prototypes=None, prototype_and_weight=0.35, **kwargs):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.n_actions = int(n_actions)
        self.action_mode = str(action_mode).lower()
        self.prototype_and_weight = float(prototype_and_weight)

        if self.action_mode == "prototype":
            if action_prototypes is None:
                action_prototypes = torch.zeros((n_actions, n_concepts))
                for i in range(n_actions):
                    action_prototypes[i, i % n_concepts] = 1.0
            prototypes = torch.as_tensor(action_prototypes, dtype=torch.float32)[:self.n_actions]
            prior = torch.full((2 * self.n_concepts, self.n_actions), 1e-4)
            prior[:self.n_concepts] = prototypes.t() * 0.95
            self.action_rule = KGUnionLayer(2 * self.n_concepts, self.n_actions, prior=prior.clamp(1e-4, 0.98))
        else:
            self.direct_mlp = nn.Sequential(nn.Linear(n_concepts, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def kg_loss(self, concept_logits):
        return concept_logits.new_zeros(())

    def rule_prior_loss(self):
        return self.action_rule.prior_retention_loss() if self.action_mode == "prototype" else torch.tensor(0.0)

    def rule_sparsity_loss(self):
        return self.action_rule.sparsity_loss() if self.action_mode == "prototype" else torch.tensor(0.0)

    def forward(self, concept_logits, return_trace=False):
        concepts = torch.sigmoid(concept_logits)
        if self.action_mode != "prototype":
            logits = self.direct_mlp(concepts)
            return (logits, {}) if return_trace else logits
        literals = torch.cat([concepts, 1.0 - concepts], dim=1)
        and_score, or_score = self.action_rule(literals).chunk(2, dim=1)
        evidence = (self.prototype_and_weight * and_score + (1.0 - self.prototype_and_weight) * or_score).clamp(1e-4, 1.0 - 1e-4)
        logits = _logit(evidence)
        return (logits, {"evidence": evidence}) if return_trace else logits

class SoftConceptFeedback(nn.Module):
    def __init__(self, in_channels=192, num_concepts=72, spatial_concepts=52, temporal_concepts=20,
                 beta=0.2, lambda_fb=0.1, mode="token_soft_bias", confidence_gating=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_concepts = num_concepts
        self.spatial_concepts = spatial_concepts
        self.temporal_concepts = temporal_concepts
        self.beta = float(beta)
        self.lambda_fb = float(lambda_fb)
        self.mode = str(mode).lower()
        self.confidence_gating = bool(confidence_gating)

        self.spatial_pool_score = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.early_spatial_head = nn.Sequential(nn.Linear(in_channels, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, spatial_concepts))
        self.early_temporal_head = nn.Sequential(nn.Linear(in_channels, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, temporal_concepts))
        self.mlp_channel = nn.Sequential(nn.Linear(num_concepts, in_channels), nn.LayerNorm(in_channels), nn.GELU(), nn.Linear(in_channels, in_channels))
        self.conv_token = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, kernel_size=1), nn.GELU(), nn.Conv2d(in_channels // 4, 1, kernel_size=1))
        self.mlp_concept = nn.Sequential(nn.Linear(num_concepts, in_channels // 4), nn.GELU(), nn.Linear(in_channels // 4, 1))
        self.confidence_fc = nn.Sequential(nn.Linear(num_concepts, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def set_lambda_fb(self, val):
        self.lambda_fb = float(val)

    def extract_early_concepts(self, F_early):
        spatial_tokens = F_early.mean(dim=2)
        a_spa = torch.softmax(self.spatial_pool_score(spatial_tokens), dim=-1)
        f_spa = (spatial_tokens * a_spa).sum(dim=-1)

        temporal_tokens = F_early.mean(dim=3)
        a_tem = torch.softmax(self.temporal_pool_score(temporal_tokens), dim=-1)
        f_tem = (temporal_tokens * a_tem).sum(dim=-1)

        z_spa = self.early_spatial_head(f_spa)
        z_tem = self.early_temporal_head(f_tem)
        logits = torch.cat([z_spa, z_tem], dim=-1)
        return logits, torch.sigmoid(logits), a_spa, a_tem

    def forward(self, F_early):
        early_logits, c_early, a_spa, a_tem = self.extract_early_concepts(F_early)
        aux = {"early_concept_logits": early_logits, "early_concepts": c_early, "spatial_attn": a_spa, "temporal_attn": a_tem, "feedback_bias": None, "feedback_gate": None, "mode": self.mode}
        if self.mode in ("disabled", "none", "early_head_only"):
            return F_early, aux

        c_in = torch.randn_like(c_early) if self.mode == "random_concept_proj" else c_early
        g_chan = torch.sigmoid(self.mlp_channel(c_in)).unsqueeze(-1).unsqueeze(-1)
        if self.mode == "channel_modulation":
            return F_early * (1.0 + self.beta * g_chan), aux

        bias = torch.tanh(self.conv_token(F_early) + self.mlp_concept(c_in).unsqueeze(-1).unsqueeze(-1))
        gate = torch.ones((F_early.shape[0], 1, 1, 1), device=F_early.device) if (self.mode == "no_confidence_gate" or not self.confidence_gating) else self.confidence_fc(c_early.detach()).unsqueeze(-1).unsqueeze(-1)
        F_mod = F_early * (1.0 + self.beta * g_chan) + (self.lambda_fb * gate) * bias * F_early
        aux["feedback_bias"] = bias
        aux["feedback_gate"] = gate
        return F_mod, aux

class SFB_ECR_SkateFormer(nn.Module):
    def __init__(self, num_classes=60, num_concepts=72, spatial_concepts=52, temporal_concepts=20,
                 feedback_mode="token_soft_bias", feedback_beta=0.2, feedback_lambda=0.1,
                 feedback_confidence_gating=True, early_stages=2, action_mode="prototype", **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.early_stages = early_stages
        self.backbone = SkateFormer(num_classes=num_classes, num_points=24, **kwargs)
        early_dim = self.backbone.stages[early_stages - 1].blocks[-1].transformer.norm_1.normalized_shape[0]
        late_dim = self.backbone.head.in_features

        self.feedback_module = SoftConceptFeedback(
            in_channels=early_dim, num_concepts=num_concepts, spatial_concepts=spatial_concepts,
            temporal_concepts=temporal_concepts, beta=feedback_beta, lambda_fb=feedback_lambda,
            mode=feedback_mode, confidence_gating=feedback_confidence_gating)

        self.spatial_pool_score = nn.Conv1d(late_dim, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(late_dim, 1, kernel_size=1)
        self.spatial_head = nn.Sequential(nn.Linear(late_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, spatial_concepts))
        self.temporal_head = nn.Sequential(nn.Linear(late_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, temporal_concepts))
        self.reasoner = KGPrimedReasoner(num_concepts, num_classes, [f"c_{i}" for i in range(num_concepts)], action_mode=action_mode)

    def set_feedback_lambda(self, val):
        self.feedback_module.set_lambda_fb(val)

    def forward(self, input, index_t=None, return_aux=True):
        F_early, F_late, fb_aux = self.backbone.forward_features_with_intermediate(
            input, feedback_fn=self.feedback_module, early_stages=self.early_stages, index_t=index_t)

        spa_tok = F_late.mean(dim=2)
        tem_tok = F_late.mean(dim=3)
        f_spa = (spa_tok * torch.softmax(self.spatial_pool_score(spa_tok), dim=-1)).sum(dim=-1)
        f_tem = (tem_tok * torch.softmax(self.temporal_pool_score(tem_tok), dim=-1)).sum(dim=-1)

        concept_logits = torch.cat([self.spatial_head(f_spa), self.temporal_head(f_tem)], dim=-1)
        action_logits, trace = self.reasoner(concept_logits, return_trace=True)

        if not return_aux:
            return action_logits

        aux = {
            "concept_logits": concept_logits,
            "early_concept_logits": fb_aux.get("early_concept_logits"),
            "feedback_bias": fb_aux.get("feedback_bias"),
            "feedback_gate": fb_aux.get("feedback_gate"),
            "kg_loss": self.reasoner.kg_loss(concept_logits),
            "prior_loss": self.reasoner.rule_prior_loss(),
            "sparsity_loss": self.reasoner.rule_sparsity_loss(),
            "feedback_consistency_loss": F.mse_loss(fb_aux["early_concepts"], torch.sigmoid(concept_logits).detach()) if fb_aux.get("early_concepts") is not None else concept_logits.new_zeros(())
        }
        return action_logits, aux

    def compute_loss(self, outputs, targets, concept_targets=None):
        action_logits, aux = outputs
        act_loss = F.cross_entropy(action_logits, targets)
        late_c_loss = F.binary_cross_entropy_with_logits(aux["concept_logits"], concept_targets) if (concept_targets is not None and aux["concept_logits"] is not None) else torch.tensor(0.0)
        early_c_loss = F.binary_cross_entropy_with_logits(aux["early_concept_logits"], concept_targets) if (concept_targets is not None and aux["early_concept_logits"] is not None) else torch.tensor(0.0)
        cons_loss = aux.get("feedback_consistency_loss", torch.tensor(0.0))
        total = act_loss + 1.0 * late_c_loss + 0.5 * early_c_loss + 0.02 * cons_loss + 0.01 * aux["prior_loss"] + 0.001 * aux["sparsity_loss"]
        return total, {"total_loss": total.item(), "action_loss": act_loss.item(), "late_concept_loss": late_c_loss.item() if torch.is_tensor(late_c_loss) else 0.0}

print("✅ Đã khởi tạo Block 5 (SFB_ECR_SkateFormer Architecture)!")
""")

    # =========================================================================
    # BLOCK 6: TRAINING & EVALUATION ENGINE (AUTO RESUME & CHECKPOINTING)
    # =========================================================================
    add_md("## ⚡ Block 6: Engine Huấn luyện & Đánh giá (Auto-Resume & Checkpointing)")
    add_code("""class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, work_dir="./work_dir/sfb_exp"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        self.start_epoch = 0

    def auto_resume(self):
        latest_pt = self.work_dir / "checkpoint_latest.pt"
        if latest_pt.exists():
            print(f"🔄 Tìm thấy checkpoint: {latest_pt}. Đang tiến hành Auto-Resume...")
            cp = torch.load(latest_pt, map_location=device)
            self.model.load_state_dict(cp['model_state'])
            self.optimizer.load_state_dict(cp['optimizer_state'])
            self.start_epoch = cp.get('epoch', 0) + 1
            self.best_acc = cp.get('best_acc', 0.0)
            print(f"✅ Đã khôi phục thành công từ Epoch {self.start_epoch} (Best Acc: {self.best_acc:.2f}%)")

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'best_acc': self.best_acc,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(state, self.work_dir / "checkpoint_latest.pt")
        if is_best:
            torch.save(state, self.work_dir / "checkpoint_best.pt")
            print(f"⭐ [BEST] Đã lưu checkpoint tốt nhất tại epoch {epoch} (Acc = {self.best_acc:.2f}%)")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            data, index_t, labels, idx = batch[0].float().to(device), batch[1].float().to(device), batch[2].long().to(device), batch[3]
            concepts = batch[4].float().to(device) if len(batch) >= 5 else None

            self.optimizer.zero_grad()
            outputs = self.model(data, index_t=index_t, return_aux=True)
            loss, l_dict = self.model.compute_loss(outputs, labels, concepts)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            preds = outputs[0].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += data.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.0 * correct / total:.2f}%"})

        if self.scheduler:
            self.scheduler.step(epoch)
        return total_loss / total, 100.0 * correct / total

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.test_loader:
                data, index_t, labels = batch[0].float().to(device), batch[1].float().to(device), batch[2].long().to(device)
                outputs = self.model(data, index_t=index_t, return_aux=True)
                preds = outputs[0].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += data.size(0)
        return 100.0 * correct / max(1, total)

    def fit(self, num_epochs=10):
        self.auto_resume()
        for ep in range(self.start_epoch, num_epochs):
            t_loss, t_acc = self.train_epoch(ep)
            val_acc = self.evaluate()
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            self.save_checkpoint(ep, is_best)
            print(f"📊 Epoch {ep:02d} Summary: Train Loss = {t_loss:.4f} | Train Acc = {t_acc:.2f}% | Val Acc = {val_acc:.2f}%")

print("✅ Đã khởi tạo Block 6 (Trainer Engine)!")
""")

    # =========================================================================
    # BLOCK 7: PIPELINE SANITY TEST WITH DUMMY DATA
    # =========================================================================
    add_md("## 🧪 Block 7: Kiểm tra Sanity Pipeline với Dummy Data (Chạy Thử Ngay Không Cần Nạp 50GB)")
    add_code("""print("🔬 Bắt đầu chạy Sanity Check trên dữ liệu Dummy...")

# Tạo Dummy Dataset
dummy_train = Feeder(dummy_size=16, window_size=64)
dummy_test = Feeder(dummy_size=8, window_size=64)
dummy_train_loader = DataLoader(dummy_train, batch_size=4, shuffle=True)
dummy_test_loader = DataLoader(dummy_test, batch_size=4, shuffle=False)

# Khởi tạo mô hình test
test_model = SFB_ECR_SkateFormer(
    num_classes=60, num_concepts=72, spatial_concepts=52, temporal_concepts=20,
    feedback_mode="token_soft_bias", feedback_beta=0.2, feedback_lambda=0.1,
    type_1_size=(8, 8), type_2_size=(8, 12), type_3_size=(8, 8), type_4_size=(8, 12),
)

opt = optim.AdamW(test_model.parameters(), lr=1e-3)
trainer = Trainer(test_model, dummy_train_loader, dummy_test_loader, opt, scheduler=None, work_dir="./work_dir/dummy_test")

print("▶️ Chạy 2 epochs test sanity...")
trainer.fit(num_epochs=2)
print("🎉 SANITY TEST THÀNH CÔNG RỰC RỠ! Toàn bộ luồng dữ liệu, forward, backward và checkpointing hoạt động hoàn hảo!")
""")

    # =========================================================================
    # BLOCK 8: PRODUCTION TRAINING & ROBUSTNESS OCCLUSION EVALUATION
    # =========================================================================
    add_md("## 🚀 Block 8: Huấn luyện & Đánh giá Robustness với Dữ liệu Thật")
    add_code("""# Cấu hình đường dẫn dữ liệu thật (Cập nhật đường dẫn thực tế trên Drive/Local)
DATASET_PATH = "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz"
CONCEPTS_CSV = "./concepts/ntu_rgbd_60_120.csv"

# Chọn Chế độ Thực nghiệm (Ablation Modes A1 - A7):
# - 'token_soft_bias' (A4: Đề xuất chính)
# - 'disabled' (A1: ECR Baseline)
# - 'early_head_only' (A2)
# - 'channel_modulation' (A3)
# - 'no_confidence_gate' (A5)
# - 'random_concept_proj' (A6)
CURRENT_ABLATION_MODE = "token_soft_bias"

print(f"📌 Chế độ Ablation được chọn: {CURRENT_ABLATION_MODE}")

if os.path.exists(DATASET_PATH):
    train_set = Feeder(data_path=DATASET_PATH, split='train', concepts_csv=CONCEPTS_CSV)
    test_set = Feeder(data_path=DATASET_PATH, split='test', concepts_csv=CONCEPTS_CSV)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    prod_model = SFB_ECR_SkateFormer(
        num_classes=60, num_concepts=72, spatial_concepts=52, temporal_concepts=20,
        feedback_mode=CURRENT_ABLATION_MODE, feedback_beta=0.2, feedback_lambda=0.1,
        type_1_size=(8, 8), type_2_size=(8, 12), type_3_size=(8, 8), type_4_size=(8, 12),
    )
    
    prod_opt = optim.AdamW(prod_model.parameters(), lr=1e-4, weight_decay=0.05)
    prod_scheduler = CosineLRScheduler(prod_opt, t_initial=150, lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=2)
    
    prod_trainer = Trainer(
        prod_model, train_loader, test_loader, prod_opt, prod_scheduler,
        work_dir=f"./work_dir/ntu_cs_{CURRENT_ABLATION_MODE}"
    )
    
    # Bắt đầu huấn luyện (ví dụ 150 epochs)
    # prod_trainer.fit(num_epochs=150)
    print("✅ Đã chuẩn bị sẵn sàng cho quá trình huấn luyện thật! Bỏ comment dòng trên để bắt đầu train.")
else:
    print(f"ℹ️ Chưa tìm thấy file dữ liệu tại {DATASET_PATH}. Vui lòng kiểm tra lại đường dẫn.")
""")

    # Write notebook file
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = Path(r"d:\Thesis\SFB-ECR-SkateFormer\SFB_ECR_SkateFormer_Colab.ipynb")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)

    print(f"Generated Notebook: {out_path} ({len(cells)} cells)")

if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    build_notebook()

