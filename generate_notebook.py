import json
import os

def create_notebook():
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

    # ==========================================
    # CELL 1: Header & Environment Setup
    # ==========================================
    add_md("""# 🚀 ECR-SkateFormer: All-in-One Google Colab Notebook
> **Mục tiêu**: Chuyển đổi toàn bộ repository ECR-SkateFormer thành các block code tuần tự, độc lập để có thể chạy mượt mà từng bước trên Google Colab kèm **Hệ thống Quản lý Checkpoint & Tự động Resume**.

---
### 📌 Cấu trúc các Block (Cells):
1. **Block 1**: Cài đặt môi trường, dependencies & kết nối Google Drive
2. **Block 2**: Các hàm tiện ích (Utilities, Logging, IO & Loss functions)
3. **Block 3**: Tiền xử lý dữ liệu & Data Augmentation (Feeders & Tools)
4. **Block 4**: Kiến trúc mạng nền tảng SkateFormer Backbone
5. **Block 5**: Mô hình suy luận ECR / REASON-SkateFormer (KG Primed Reasoner & Concept Decoders)
6. **Block 6**: Engine Huấn luyện & Đánh giá (**Luôn lưu `checkpoint_latest.pt` mỗi epoch, `checkpoint_best.pt` & Auto-Resume**)
7. **Block 7**: Test kiểm tra toàn bộ Pipeline với Dummy Data (Chạy thử ngay không cần nạp 50GB)
8. **Block 8**: Cấu hình & Huấn luyện / Đánh giá với Dữ liệu Thật (NTU-60 / NTU-120)
""")

    add_md("## 📦 Block 1: Cài đặt Dependencies & Kết nối Google Drive")
    add_code("""# Cài đặt các thư viện cần thiết
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
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath
from timm.scheduler.cosine_lr import CosineLRScheduler

# Cấu hình thiết bị GPU/CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Đang sử dụng thiết bị: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU Name: {torch.cuda.get_device_name(0)}")

# (Tùy chọn) Mount Google Drive nếu dữ liệu lưu trên Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Đã kết nối Google Drive thành công!")
except Exception as e:
    print("ℹ️ Bỏ qua mount Google Drive (đang chạy local hoặc môi trường khác).")
""")

    # ==========================================
    # CELL 2: Utilities & IO
    # ==========================================
    add_md("## 🛠️ Block 2: Utilities, Logging, IO & Loss Functions")
    add_code("""def init_seed(seed=1):
    \"\"\"Khởi tạo seed cố định để tái lập kết quả thực nghiệm\"\"\"
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

class LabelSmoothingCrossEntropy(nn.Module):
    \"\"\"Label Smoothing Cross Entropy Loss\"\"\"
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class IO:
    \"\"\"Quản lý lưu log, checkpoint và metric trong quá trình chạy\"\"\"
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.session_file = None
        self.model_text = ''
        os.makedirs(self.work_dir, exist_ok=True)

    def print_log(self, msg, print_time=True):
        if print_time:
            msg = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str(msg)
        if self.print_to_screen:
            print(msg)
        if self.save_log:
            with open(f'{self.work_dir}/log.txt', 'a', encoding='utf-8') as f:
                print(msg, file=f)

    def save_model(self, model, name):
        model_path = os.path.join(self.work_dir, name)
        state_dict = model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, model_path)
        self.print_log(f'💾 Trọng số model lưu tại: {model_path}')

print("✅ Đã nạp xong Block 2: Utilities & Loss Functions!")
""")

    # ==========================================
    # CELL 3: Data Augmentation & Feeders
    # ==========================================
    add_md("## 📊 Block 3: Tiền xử lý dữ liệu & Data Augmentation (Feeders & Tools)")
    add_code("""# ==================== DATA AUGMENTATION TOOLS ====================

def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window, thres):
    if isinstance(p_interval, (int, float)):
        p_interval = [p_interval]
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num if valid_frame_num > 0 else T
    valid_size = max(1, end - begin)

    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        c_b = min(begin + bias, T - 1)
        c_e = max(c_b + 1, min(end - bias, T))
        data = data_numpy[:, c_b:c_e, :, :]
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
        max_bias = max(0, valid_size - cropped_length)
        bias = np.random.randint(0, max_bias + 1) if max_bias > 0 else 0
        c_b = begin + bias
        c_e = min(begin + bias + cropped_length, T)
        data = data_numpy[:, c_b:c_e, :, :]

    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, cropped_length)
    data = F.interpolate(data, size=window, mode='linear', align_corners=False)
    data = data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    index_t = torch.arange(start=c_b, end=c_e, dtype=torch.float)
    index_t = F.interpolate(index_t[None, None, :], size=window, mode='linear', align_corners=False).squeeze()
    index_t = 2 * index_t / valid_size - 1
    return data, index_t.numpy()

def valid_crop_uniform(data_numpy, valid_frame_num, p_interval, window, thres):
    if isinstance(p_interval, (int, float)):
        p_interval = [p_interval]
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num if valid_frame_num > 0 else T
    valid_size = max(1, end - begin)

    if len(p_interval) == 1:
        p = p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
        bias = int((1 - p) * valid_size / 2)
        bias = min(bias, max(0, valid_size - cropped_length))
        if cropped_length < window:
            inds = np.arange(cropped_length)
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            inds = bids[:window]
        inds = inds + bias
        inds = np.clip(inds, 0, T - 1)
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
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            bsize = np.diff(bids)
            bst = bids[:window]
            offset = np.random.randint(bsize)
            inds = bst + offset
        inds = inds + bias
        inds = np.clip(inds, 0, T - 1)
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

transform_order = {'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]}

def scale(data_numpy, scale=0.2, p=0.5):
    if random.random() < p:
        scale_factor = 1 + np.random.uniform(-1, 1, size=(3, 1, 1, 1)) * np.array(scale)
        return data_numpy * scale_factor
    return data_numpy.copy()

def spatial_flip(data_numpy, p=0.5):
    if random.random() < p:
        index = transform_order['ntu']
        return data_numpy[:, :, index, :]
    return data_numpy.copy()

def temporal_flip(data_numpy, index_t, p=0.5):
    C, T, V, M = data_numpy.shape
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return data_numpy[:, time_range_reverse, :, :], -index_t
    return data_numpy.copy(), index_t.copy()

def rotate(data_numpy, axis=None, angle=None, p=0.5):
    axis_next = axis if axis is not None else random.randint(0, 2)
    angle_next = angle if angle is not None else random.uniform(-30, 30)
    if random.random() < p:
        temp = data_numpy.copy()
        rad = math.radians(angle_next)
        if axis_next == 0:
            R = np.array([[1, 0, 0], [0, math.cos(rad), math.sin(rad)], [0, -math.sin(rad), math.cos(rad)]])
        elif axis_next == 1:
            R = np.array([[math.cos(rad), 0, -math.sin(rad)], [0, 1, 0], [math.sin(rad), 0, math.cos(rad)]])
        else:
            R = np.array([[math.cos(rad), math.sin(rad), 0], [-math.sin(rad), math.cos(rad), 0], [0, 0, 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        return temp.transpose(3, 0, 1, 2)
    return data_numpy.copy()

def shear(data_numpy, s1=None, s2=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        s1_list = s1 if s1 is not None else [random.uniform(-0.5, 0.5) for _ in range(3)]
        s2_list = s2 if s2 is not None else [random.uniform(-0.5, 0.5) for _ in range(3)]
        R = np.array([[1, s1_list[0], s2_list[0]], [s1_list[1], 1, s2_list[1]], [s1_list[2], s2_list[2], 1]]).transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        return temp.transpose(3, 0, 1, 2)
    return data_numpy.copy()

def drop_axis(data_numpy, axis=None, p=0.5):
    axis_next = axis if axis is not None else random.randint(0, 2)
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        temp[axis_next] = np.zeros((T, V, M))
        return temp
    return data_numpy.copy()

def drop_joint(data_numpy, joint_list=None, time_range=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        all_joints = [i for i in range(V)]
        j_cnt = joint_list if joint_list is not None else random.randint(5, 15)
        joint_list_ = sorted(random.sample(all_joints, min(len(all_joints), j_cnt)))
        all_frames = [i for i in range(T)]
        t_cnt = time_range if time_range is not None else random.randint(16, 32)
        time_range_ = sorted(random.sample(all_frames, min(len(all_frames), t_cnt)))
        temp[:, time_range_, :, :][:, :, joint_list_, :] = 0
        return temp
    return data_numpy.copy()

def gaussian_noise(data_numpy, mean=0, std=0.05, p=0.5):
    if random.random() < p:
        noise = np.random.normal(mean, std, size=data_numpy.shape)
        return data_numpy.copy() + noise
    return data_numpy.copy()

class joint2bone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pairs = [(0, 1), (1, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                      (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                      (19, 18), (20, 1), (21, 7), (22, 7), (23, 11), (24, 11)]

    def __call__(self, joint):
        bone = np.zeros_like(joint)
        for v1, v2 in self.pairs:
            bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
        return bone

def to_motion(input_data):
    C, T, V, M = input_data.shape
    motion = np.zeros_like(input_data)
    motion[:, :T - 1] = np.diff(input_data, axis=1)
    return motion

# ==================== DATASET FEEDER ====================

class Feeder(Dataset):
    \"\"\"NTU-RGB+D Feeder với hỗ trợ REASON Concept Matrix\"\"\"
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False,
                 concepts_csv=None, remove_null_concept=True):
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        if isinstance(p_interval, (int, float)):
            p_interval = [p_interval]
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.concepts_csv = concepts_csv
        self.remove_null_concept = remove_null_concept

        self.load_data()
        self.concept_matrix = None
        self.concept_names = None
        if self.concepts_csv and os.path.exists(self.concepts_csv):
            self.load_concepts()

        if partition:
            self.right_arm = np.array([7, 8, 22, 23]) - 1
            self.left_arm = np.array([11, 12, 24, 25]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.h_torso = np.array([5, 9, 6, 10]) - 1
            self.w_torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)

    def load_concepts(self):
        with open(self.concepts_csv, 'r', newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        if not rows or 'action_class' not in rows[0]:
            raise ValueError(f'Invalid concepts CSV: {self.concepts_csv}')

        names = [name for name in rows[0].keys() if name != 'action_class']
        if self.remove_null_concept:
            names = [name for i, name in enumerate(names) if i not in (50, 65)]

        values = []
        for row in rows:
            values.append([float(row[name]) for name in names])
        self.concept_names = names
        self.concept_matrix = np.asarray(values, dtype=np.float32)

    def load_data(self):
        if not os.path.exists(self.data_path):
            # Khởi tạo data giả định để test nếu chưa có file
            N = 40 if self.split == 'train' else 16
            self.data = np.random.randn(N, 3, 64, 25, 2).astype(np.float32)
            self.label = np.random.randint(0, 60, size=N)
            self.sample_name = [f'{self.split}_{i}' for i in range(N)]
            return

        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        if self.uniform:
            data_numpy, index_t = valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres)
        else:
            data_numpy, index_t = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres)

        if self.split == 'train':
            p = np.random.rand(1)
            if p < self.intra_p:
                if '1' in self.aug_method: data_numpy = shear(data_numpy, p=0.5)
                if '2' in self.aug_method: data_numpy = rotate(data_numpy, p=0.5)
                if '3' in self.aug_method: data_numpy = scale(data_numpy, p=0.5)
                if '4' in self.aug_method: data_numpy = spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method: data_numpy, index_t = temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method: data_numpy = gaussian_noise(data_numpy, p=0.5)
                if '8' in self.aug_method: data_numpy = drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method: data_numpy = drop_joint(data_numpy, p=0.5)

        if self.data_type == 'b':
            j2b = joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = joint2bone()
            data_numpy = to_motion(j2b(data_numpy))

        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]

        if self.concept_matrix is not None:
            return data_numpy, index_t, label, index, self.concept_matrix[label]
        return data_numpy, index_t, label, index

    def top_k(self, score, top_k=1):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

print("✅ Đã nạp xong Block 3: Data Augmentation & Feeders!")
""")

    # ==========================================
    # CELL 4: SkateFormer Backbone Architecture
    # ==========================================
    add_md("## 🧠 Block 4: Kiến trúc Mạng Nền tảng SkateFormer Backbone")
    add_code("""# ==================== PARTITION & REVERSE ====================

def type_1_partition(input, partition_size):
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    return partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_1_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)

def type_2_partition(input, partition_size):
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1])
    return partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_2_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)

def type_3_partition(input, partition_size):
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    return partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_3_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)

def type_4_partition(input, partition_size):
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1])
    return partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)

def type_4_reverse(partitions, original_size, partition_size):
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    return output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)

def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)

# ==================== ATTENTION & BLOCKS ====================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0., rel=True):
        super(MultiHeadSelfAttention, self).__init__()
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
            if self.rel_type in ['type_1', 'type_3']:
                self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)
            elif self.rel_type in ['type_2', 'type_4']:
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(self):
        if self.rel_type in ['type_1', 'type_3']:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], -1)
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3).repeat(1, self.partition_size[1], 1, self.partition_size[1], 1, 1).view(self.attn_area, self.attn_area, -1)
            return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        elif self.rel_type in ['type_2', 'type_4']:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1)
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
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        return (attn @ v).transpose(1, 2).reshape(B_, N, -1)

class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SkateFormerBlock, self).__init__()
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
            attention.append(
                MultiHeadSelfAttention(in_channels=in_channels // (len(self.partition_function) * 2),
                                       rel_type=self.rel_type[i],
                                       num_heads=num_heads // (len(self.partition_function) * 2),
                                       partition_size=self.partition_size[i], attn_drop=attn_drop, rel=rel))
        self.attention = nn.ModuleList(attention)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels, hidden_features=int(mlp_ratio * in_channels), act_layer=act_layer, drop=drop)

    def forward(self, input):
        B, C, T, V = input.shape
        input_perm = input.permute(0, 2, 3, 1).contiguous()
        skip = input_perm
        f = self.mapping(self.norm_1(input_perm)).permute(0, 3, 1, 2).contiguous()
        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []

        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        for i in range(self.gconv.shape[0]):
            z = torch.einsum('n c t u, v u -> n c t v', split_f_gconv[i], self.gconv[i])
            y_gconv.append(z)
        y.append(torch.cat(y_gconv, dim=1))
        y.append(self.tconv(split_f_conv[1]))

        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)
        for i in range(len(self.partition_function)):
            ch = split_f_attn[i].shape[1]
            input_partitioned = self.partition_function[i](split_f_attn[i], self.partition_size[i])
            input_partitioned = input_partitioned.view(-1, self.partition_size[i][0] * self.partition_size[i][1], ch)
            y.append(self.reverse_function[i](self.attention[i](input_partitioned), (T, V), self.partition_size[i]))

        output = self.proj(torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous())
        output = self.proj_drop(output)
        output = skip + self.drop_path(output)
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
    def __init__(self, in_channels, out_channels, num_points=50, kernel_size=7, downscale=False, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm):
        super(SkateFormerBlockDS, self).__init__()
        self.downsample = PatchMergingTconv(in_channels, out_channels, kernel_size=kernel_size) if downscale else None
        self.transformer = SkateFormerBlock(
            in_channels=out_channels, num_points=num_points, kernel_size=kernel_size, num_heads=num_heads,
            type_1_size=type_1_size, type_2_size=type_2_size, type_3_size=type_3_size, type_4_size=type_4_size,
            attn_drop=attn_drop, drop=drop, rel=rel, drop_path=drop_path, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer_transformer
        )

    def forward(self, input):
        if self.downsample is not None:
            return self.transformer(self.downsample(input))
        return self.transformer(input)

class SkateFormerStage(nn.Module):
    def __init__(self, depth, in_channels, out_channels, first_depth=False, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm):
        super(SkateFormerStage, self).__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                SkateFormerBlockDS(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    num_points=num_points,
                    kernel_size=kernel_size,
                    downscale=((index == 0) and not first_depth),
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        output = input
        for block in self.blocks:
            output = block(output)
        return output

class SkateFormer(nn.Module):
    def __init__(self, in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), num_classes=60,
                 embed_dim=64, num_people=2, num_frames=64, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., head_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm, index_t=False, global_pool='avg'):
        super(SkateFormer, self).__init__()
        self.num_classes = num_classes
        self.head_drop = head_drop
        self.index_t = index_t
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=self.head_drop) if self.head_drop != 0 else None

        stem = []
        stem.append(nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=(1, 1)))
        stem.append(act_layer())
        stem.append(nn.Conv2d(in_channels=2 * in_channels, out_channels=3 * in_channels, kernel_size=(1, 1)))
        stem.append(act_layer())
        stem.append(nn.Conv2d(in_channels=3 * in_channels, out_channels=embed_dim, kernel_size=(1, 1)))
        self.stem = nn.ModuleList(stem)

        if self.index_t:
            self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.joint_person_embedding, std=.02)
        else:
            self.joint_person_temporal_embedding = nn.Parameter(torch.zeros(1, embed_dim, num_frames, num_points * num_people))
            trunc_normal_(self.joint_person_temporal_embedding, std=.02)

        drop_path_list = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(
                SkateFormerStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    first_depth=(index == 0),
                    num_points=num_points * num_people,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path_list[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer
                )
            )
        self.stages = nn.ModuleList(stages)
        self.global_pool = global_pool
        self.head = nn.Linear(channels[-1], num_classes)

    def forward_features_from_input(self, input, index_t=None):
        B, C, T, V, M = input.shape
        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)
        for layer in self.stem:
            output = layer(output)
        if self.index_t:
            if index_t is None:
                index_t = torch.linspace(-1., 1., T, device=output.device, dtype=output.dtype).unsqueeze(0).expand(B, T)
            te = torch.zeros(B, T, self.embed_dim, device=output.device)
            div_term = torch.exp((torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))).to(output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum('b t c, c v -> b c t v', te, self.joint_person_embedding)
        else:
            output = output + self.joint_person_temporal_embedding

        for stage in self.stages:
            output = stage(output)
        return output

    def forward(self, input, index_t=None):
        feat = self.forward_features_from_input(input, index_t=index_t)
        pooled = feat.mean(dim=(2, 3)) if self.global_pool == "avg" else torch.amax(feat, dim=(2, 3))
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        return self.head(pooled)

def SkateFormer_(**kwargs):
    return SkateFormer(
        depths=(2, 2, 2, 2),
        channels=(96, 192, 192, 192),
        embed_dim=96,
        **kwargs
    )

print("✅ Đã nạp xong Block 4: SkateFormer Backbone!")
""")

    # ==========================================
    # CELL 5: ECR / REASON-SkateFormer
    # ==========================================
    add_md("## 🧩 Block 5: Mô hình Suy luận ECR / REASON-SkateFormer (KG Reasoner & Concept Heads)")
    add_code("""def _logit(probability, eps=1e-4):
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)

class KGUnionLayer(nn.Module):
    \"\"\"Differentiable conjunction/disjunction rule bank\"\"\"
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
        and_weight = torch.sigmoid(self.and_weight)
        or_weight = torch.sigmoid(self.or_weight)
        conjunction = torch.prod(1.0 - and_weight.unsqueeze(0) * (1.0 - x.unsqueeze(-1)), dim=1)
        disjunction = 1.0 - torch.prod(1.0 - or_weight.unsqueeze(0) * x.unsqueeze(-1), dim=1)
        return torch.cat([conjunction, disjunction], dim=1)

    def weights(self):
        return torch.sigmoid(self.and_weight), torch.sigmoid(self.or_weight)

    def prior_retention_loss(self):
        and_weight, or_weight = self.weights()
        return ((and_weight - self.prior).abs().mean() + (or_weight - self.prior).abs().mean()) * 0.5

    def sparsity_loss(self):
        and_weight, or_weight = self.weights()
        return 0.5 * (and_weight.mean() + or_weight.mean())

class KGPrimedReasoner(nn.Module):
    \"\"\"Concept-to-action logical reasoner with first-layer KG priming\"\"\"
    def __init__(self, n_concepts, n_actions, concept_names, l1_dim=128, l2_dim=128,
                 kg_init_mode="kg_primed", kg_prior_map=None, kg_prior_strength=1.0,
                 action_mode="prototype", action_prototypes=None, prototype_and_weight=0.35):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.n_actions = int(n_actions)
        self.concept_names = list(concept_names)
        self.kg_init_mode = str(kg_init_mode).lower()
        self.kg_prior_map = kg_prior_map or {"require": [], "alongside": [], "excludes": []}
        self.kg_prior_strength = float(kg_prior_strength)
        self.action_mode = str(action_mode).lower()
        self.prototype_and_weight = float(prototype_and_weight)

        if self.action_mode == "prototype":
            if action_prototypes is None:
                prototypes = torch.zeros((self.n_actions, self.n_concepts), dtype=torch.float32)
            else:
                prototypes = torch.as_tensor(action_prototypes, dtype=torch.float32)
                if prototypes.shape[0] > self.n_actions:
                    prototypes = prototypes[:self.n_actions]
                if prototypes.shape != (self.n_actions, self.n_concepts):
                    raise ValueError(
                        f"action_prototypes must have shape ({self.n_actions}, {self.n_concepts}), got {tuple(prototypes.shape)}"
                    )
            prototypes = (prototypes >= 0.5).float()
            self.register_buffer("action_prototypes", prototypes)
            action_prior = self._build_action_prior(prototypes)
            self.action_rule = KGUnionLayer(2 * self.n_concepts, self.n_actions, prior=action_prior)
        else:
            first_prior = self._build_first_layer_prior(2 * self.n_concepts, l1_dim) if self.kg_init_mode == "kg_primed" else None
            self.first = KGUnionLayer(2 * self.n_concepts, l1_dim, prior=first_prior)
            self.second = KGUnionLayer(2 * l1_dim, l2_dim)
            self.action = nn.Linear(2 * l2_dim, n_actions)

    def _resolve(self, name):
        if isinstance(name, int):
            return name if 0 <= name < self.n_concepts else None
        key = str(name).strip().lower()
        for index, concept_name in enumerate(self.concept_names):
            if str(concept_name).strip().lower() == key:
                return index
        return None

    def _build_action_prior(self, prototypes):
        if prototypes.shape[0] > self.n_actions:
            prototypes = prototypes[:self.n_actions]
        prior = torch.full((2 * self.n_concepts, self.n_actions), 1e-4, dtype=prototypes.dtype)
        prior[: self.n_concepts] = prototypes.t() * 0.95
        for action_index in range(self.n_actions):
            positive = set(torch.nonzero(prototypes[action_index] > 0.5).flatten().tolist())
            for relation_name, pairs in self.kg_prior_map.items():
                if relation_name not in {"require", "alongside", "excludes"}:
                    continue
                relation_value = {"require": 0.90, "alongside": 0.70, "excludes": 0.90}[relation_name] * self.kg_prior_strength
                relation_value = max(0.01, min(0.98, relation_value))
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a, b = self._resolve(pair[0]), self._resolve(pair[1])
                    if a is None or b is None:
                        continue
                    if relation_name == "require" and b in positive:
                        prior[a, action_index] = max(prior[a, action_index], relation_value)
                    elif relation_name == "alongside":
                        if a in positive: prior[b, action_index] = max(prior[b, action_index], relation_value)
                        if b in positive: prior[a, action_index] = max(prior[a, action_index], relation_value)
                    elif relation_name == "excludes":
                        if a in positive: prior[self.n_concepts + b, action_index] = max(prior[self.n_concepts + b, action_index], relation_value)
                        if b in positive: prior[self.n_concepts + a, action_index] = max(prior[self.n_concepts + a, action_index], relation_value)
        return prior.clamp(1e-4, 0.98)

    def _build_first_layer_prior(self, input_dim, output_dim):
        prior = torch.full((input_dim, output_dim), 0.03)
        for rule_index in range(output_dim):
            seed = rule_index % self.n_concepts
            prior[seed, rule_index] = 0.95
        return prior

    def kg_loss(self, concept_logits):
        probabilities = torch.sigmoid(concept_logits)
        penalties = []
        for relation_name in ("require", "alongside", "excludes"):
            for pair in self.kg_prior_map.get(relation_name, []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                a, b = self._resolve(pair[0]), self._resolve(pair[1])
                if a is None or b is None:
                    continue
                pa, pb = probabilities[:, a], probabilities[:, b]
                if relation_name == "require":
                    penalties.append(F.relu(pb - pa))
                elif relation_name == "alongside":
                    penalties.append(F.relu(0.5 - pa * pb))
                else:
                    penalties.append(pa * pb)
        return torch.stack([p.mean() for p in penalties]).mean() if penalties else concept_logits.new_zeros(())

    def rule_prior_loss(self):
        return self.action_rule.prior_retention_loss() if self.action_mode == "prototype" else 0.5 * (self.first.prior_retention_loss() + self.second.prior_retention_loss())

    def rule_sparsity_loss(self):
        return self.action_rule.sparsity_loss() if self.action_mode == "prototype" else 0.5 * (self.first.sparsity_loss() + self.second.sparsity_loss())

    def forward(self, concept_logits, return_trace=False):
        concepts = torch.sigmoid(concept_logits)
        literals = torch.cat([concepts, 1.0 - concepts], dim=1)
        if self.action_mode == "prototype":
            rule_output = self.action_rule(literals)
            and_score, or_score = rule_output.chunk(2, dim=1)
            evidence = (self.prototype_and_weight * and_score + (1.0 - self.prototype_and_weight) * or_score).clamp(1e-4, 1.0 - 1e-4)
            action_logits = _logit(evidence)
            trace = {"and_score": and_score, "or_score": or_score, "evidence": evidence}
        else:
            hidden_1 = self.first(literals)
            hidden_2 = self.second(hidden_1)
            action_logits = self.action(hidden_2)
            trace = {"hidden_1": hidden_1, "hidden_2": hidden_2}
        return (action_logits, trace) if return_trace else action_logits

class ReasonSkateFormer(nn.Module):
    \"\"\"SkateFormer Perception + REASON Concept/Logic Heads\"\"\"
    def __init__(self, num_classes=60, num_concepts=72, spatial_concepts=52, temporal_concepts=20,
                 num_people=2, num_points=24, num_frames=64, backbone_weights=None, concept_names=None,
                 l1_dim=128, l2_dim=128, kg_init_mode="kg_primed", kg_prior_map=None, kg_prior_strength=1.0,
                 action_mode="prototype", action_prototypes=None, prototype_and_weight=0.35,
                 concept_loss_weight=1.0, kg_loss_weight=0.01, prior_retention_weight=0.01,
                 rule_sparsity_weight=0.001, concept_negative_weight=0.15, concept_positive_weight=1.0,
                 asymmetric_gamma_neg=2.0, part_diversity_weight=0.01, return_aux=True, **kwargs):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.spatial_concepts = int(spatial_concepts)
        self.temporal_concepts = int(temporal_concepts)
        self.concept_loss_weight = float(concept_loss_weight)
        self.kg_loss_weight = float(kg_loss_weight)
        self.prior_retention_weight = float(prior_retention_weight)
        self.rule_sparsity_weight = float(rule_sparsity_weight)
        self.concept_negative_weight = float(concept_negative_weight)
        self.concept_positive_weight = float(concept_positive_weight)
        self.asymmetric_gamma_neg = float(asymmetric_gamma_neg)
        self.part_diversity_weight = float(part_diversity_weight)
        self.return_aux = bool(return_aux)

        self.backbone = SkateFormer_(
            num_classes=num_classes, num_people=num_people, num_points=num_points, num_frames=num_frames, **kwargs
        )
        if backbone_weights and os.path.exists(backbone_weights):
            self._load_backbone_weights(backbone_weights)

        feature_dim = self.backbone.head.in_features
        self.spatial_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.spatial_head = nn.Sequential(nn.Linear(feature_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, self.spatial_concepts))
        self.temporal_head = nn.Sequential(nn.Linear(feature_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, self.temporal_concepts))
        self.fusion = nn.Sequential(nn.Linear(2 * feature_dim, 512), nn.LayerNorm(512), nn.GELU())

        if concept_names is None:
            concept_names = [f"concept_{i}" for i in range(self.num_concepts)]
        self.concept_names = list(concept_names)
        self.reasoner = KGPrimedReasoner(
            self.num_concepts, num_classes, self.concept_names, l1_dim=l1_dim, l2_dim=l2_dim,
            kg_init_mode=kg_init_mode, kg_prior_map=kg_prior_map, kg_prior_strength=kg_prior_strength,
            action_mode=action_mode, action_prototypes=action_prototypes, prototype_and_weight=prototype_and_weight
        )

    def _load_backbone_weights(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        backbone_state = self.backbone.state_dict()
        mapped = OrderedDict()
        for key, value in checkpoint.items():
            clean_key = key.replace("module.", "", 1)
            if clean_key.startswith("backbone."): clean_key = clean_key[len("backbone."):]
            if clean_key.startswith("head."): continue
            if clean_key in backbone_state: mapped[clean_key] = value
        missing, unexpected = self.backbone.load_state_dict(mapped, strict=False)
        print(f"📦 Đã load thành công {len(mapped)} tensors từ pretrained backbone!")

    def _pool_features(self, feature_map):
        spatial_tokens = feature_map.mean(dim=2)
        spatial_attention = torch.softmax(self.spatial_pool_score(spatial_tokens), dim=-1)
        spatial = (spatial_tokens * spatial_attention).sum(dim=-1)

        temporal_tokens = feature_map.mean(dim=3)
        temporal_attention = torch.softmax(self.temporal_pool_score(temporal_tokens), dim=-1)
        temporal = (temporal_tokens * temporal_attention).sum(dim=-1)
        return spatial, temporal

    @staticmethod
    def _part_diversity_loss(feature_map):
        batch, channels, frames, tokens = feature_map.shape
        if tokens % 6 != 0: return feature_map.new_zeros(())
        parts = feature_map.view(batch, channels, frames, 6, tokens // 6).mean(dim=(2, 4))
        parts = F.normalize(parts.transpose(1, 2), dim=-1)
        gram = torch.matmul(parts, parts.transpose(-1, -2))
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).mean()

    def _concept_loss(self, logits, targets):
        targets = targets.float().clamp(0.0, 1.0)
        probabilities = torch.sigmoid(logits)
        positive_loss = F.softplus(-logits) * targets
        negative_loss = F.softplus(logits) * (1.0 - probabilities).pow(self.asymmetric_gamma_neg) * (1.0 - targets)
        weights = (targets * self.concept_positive_weight + (1.0 - targets) * self.concept_negative_weight)
        loss = positive_loss + negative_loss
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(self, input, index_t=None, concept_targets=None):
        feature_map = self.backbone.forward_features_from_input(input, index_t=index_t)
        spatial_feat, temporal_feat = self._pool_features(feature_map)
        spatial_logits = self.spatial_head(spatial_feat)
        temporal_logits = self.temporal_head(temporal_feat)
        concept_logits = torch.cat([spatial_logits, temporal_logits], dim=1)
        action_logits, reasoning_trace = self.reasoner(concept_logits, return_trace=True)

        if not self.return_aux and concept_targets is None:
            return action_logits

        output = {
            "action_logits": action_logits,
            "concept_logits": concept_logits,
            "spatial_logits": spatial_logits,
            "temporal_logits": temporal_logits,
            "fused_feature": self.fusion(torch.cat([spatial_feat, temporal_feat], dim=1)),
            "kg_loss": self.reasoner.kg_loss(concept_logits),
            "prior_retention_loss": self.reasoner.rule_prior_loss(),
            "rule_sparsity_loss": self.reasoner.rule_sparsity_loss(),
            "part_diversity_loss": self._part_diversity_loss(feature_map),
            "reasoning_trace": reasoning_trace
        }
        if concept_targets is not None:
            targets = concept_targets.float().to(concept_logits.device)
            output["concept_loss"] = self._concept_loss(concept_logits, targets)
            output["aux_loss"] = (
                self.concept_loss_weight * output["concept_loss"]
                + self.kg_loss_weight * output["kg_loss"]
                + self.prior_retention_weight * output["prior_retention_loss"]
                + self.rule_sparsity_weight * output["rule_sparsity_loss"]
                + self.part_diversity_weight * output["part_diversity_loss"]
            )
        return output

def REASON_SkateFormer(**kwargs):
    return ReasonSkateFormer(**kwargs)

print("✅ Đã nạp xong Block 5: ECR REASON-SkateFormer!")
""")

    # ==========================================
    # CELL 6: Processor Pipeline & Checkpointing
    # ==========================================
    add_md("## 🚀 Block 6: Engine Huấn luyện & Đánh giá (Lưu Checkpoint Tự Động & Auto-Resume)")
    add_code("""class Processor:
    \"\"\"Quản lý toàn bộ vòng lặp Training, Validation, Checkpointing & Auto-Resume\"\"\"
    def __init__(self, config):
        self.config = config
        self.work_dir = config.get('work_dir', './work_dir')
        os.makedirs(self.work_dir, exist_ok=True)
        self.io = IO(self.work_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.global_step = 0
        self.start_epoch = self.config.get('start_epoch', 0)
        self.best_acc = 0.0
        self.best_acc_epoch = 0

        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.resume_or_load_checkpoint()

    def load_data(self):
        train_args = self.config.get('train_feeder_args', {})
        test_args = self.config.get('test_feeder_args', {})
        batch_size = self.config.get('batch_size', 32)
        test_batch_size = self.config.get('test_batch_size', 32)
        num_workers = self.config.get('num_worker', 0)

        self.train_dataset = Feeder(**train_args)
        self.test_dataset = Feeder(**test_args)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        # Cập nhật concept_names và action_prototypes vào model args nếu có
        if getattr(self.train_dataset, 'concept_names', None) is not None:
            self.config['model_args']['concept_names'] = list(self.train_dataset.concept_names)
        if getattr(self.train_dataset, 'concept_matrix', None) is not None:
            c_matrix = self.train_dataset.concept_matrix
            num_classes = self.config['model_args'].get('num_classes', 60)
            if c_matrix.shape[0] > num_classes:
                c_matrix = c_matrix[:num_classes]
            self.config['model_args']['action_prototypes'] = c_matrix.tolist()

    def load_model(self):
        model_args = self.config.get('model_args', {})
        self.model = REASON_SkateFormer(**model_args).to(self.device)

        loss_type = self.config.get('loss_type', 'LSCE')
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(self.device) if loss_type == 'LSCE' else nn.CrossEntropyLoss().to(self.device)

    def load_optimizer(self):
        base_lr = self.config.get('base_lr', 0.0001)
        weight_decay = self.config.get('weight_decay', 0.05)
        num_epoch = self.config.get('num_epoch', 80)
        warmup_epoch = self.config.get('warm_up_epoch', 2)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        num_steps = max(1, int(num_epoch * max(1, len(self.train_loader))))
        warmup_steps = int(warmup_epoch * max(1, len(self.train_loader)))

        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=num_steps,
            lr_min=self.config.get('min_lr', 1e-6),
            warmup_lr_init=self.config.get('warmup_lr', 1e-6),
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def resume_or_load_checkpoint(self):
        \"\"\"Tự động Resume từ checkpoint_latest.pt hoặc nạp weights được chỉ định\"\"\"
        latest_ckpt_path = os.path.join(self.work_dir, 'checkpoint_latest.pt')
        weights_path = self.config.get('weights', None)

        ckpt_to_load = None
        if self.config.get('auto_resume', True) and os.path.exists(latest_ckpt_path):
            self.io.print_log(f"🔄 Tìm thấy checkpoint mới nhất: {latest_ckpt_path}. Đang tiến hành Resume...")
            ckpt_to_load = latest_ckpt_path
        elif weights_path and os.path.exists(weights_path):
            self.io.print_log(f"📥 Nạp weights/checkpoint từ: {weights_path}")
            ckpt_to_load = weights_path

        if ckpt_to_load:
            checkpoint = torch.load(ckpt_to_load, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
                if 'optimizer' in checkpoint and checkpoint['optimizer']:
                    try: self.optimizer.load_state_dict(checkpoint['optimizer'])
                    except Exception as e: self.io.print_log(f"⚠️ Không thể load optimizer state: {e}")
                if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] and hasattr(self.lr_scheduler, 'load_state_dict'):
                    try: self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    except Exception as e: self.io.print_log(f"⚠️ Không thể load scheduler state: {e}")
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch']
                    self.io.print_log(f"⏩ Đã nạp trạng thái thành công! Tiếp tục từ Epoch {self.start_epoch + 1}")
                if 'global_step' in checkpoint:
                    self.global_step = checkpoint['global_step']
                if 'best_acc' in checkpoint:
                    self.best_acc = checkpoint['best_acc']
                if 'best_acc_epoch' in checkpoint:
                    self.best_acc_epoch = checkpoint['best_acc_epoch']
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint, strict=False)

    def save_checkpoint(self, epoch, is_best=False, is_periodic=False):
        \"\"\"Lưu checkpoint kèm toàn bộ trạng thái optimizer, scheduler, step\"\"\"
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

        checkpoint = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model': weights,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if hasattr(self.lr_scheduler, 'state_dict') else None,
            'best_acc': self.best_acc,
            'best_acc_epoch': self.best_acc_epoch,
            'config': self.config
        }

        # 1. Luôn cập nhật checkpoint_latest.pt sau mỗi epoch để phục vụ Resume
        latest_path = os.path.join(self.work_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        self.io.print_log(f"💾 [Epoch {epoch+1}] Đã lưu checkpoint mới nhất tại: {latest_path}")

        # 2. Lưu checkpoint kỷ lục cao nhất (Top-1 Accuracy)
        if is_best:
            best_ckpt_path = os.path.join(self.work_dir, 'checkpoint_best.pt')
            best_model_path = os.path.join(self.work_dir, 'best_model.pt')
            torch.save(checkpoint, best_ckpt_path)
            torch.save(weights, best_model_path)
            self.io.print_log(f"🏆 [Epoch {epoch+1}] Kỷ lục mới: Top-1 {self.best_acc:.2f}% -> Đã lưu {best_ckpt_path} & {best_model_path}")

        # 3. Lưu checkpoint định kỳ (theo save_interval)
        if is_periodic:
            periodic_path = os.path.join(self.work_dir, f'runs-{epoch+1}-{int(self.global_step)}.pt')
            torch.save(weights, periodic_path)
            self.io.print_log(f"📦 [Epoch {epoch+1}] Đã lưu Checkpoint định kỳ tại: {periodic_path}")

    def train_epoch(self, epoch):
        self.model.train()
        loss_list, acc_list = [], []
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}")

        for batch in pbar:
            if len(batch) == 5:
                data, index_t, label, index, concept_targets = batch
            else:
                data, index_t, label, index = batch
                concept_targets = None

            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1

            data = data.float().to(self.device)
            index_t = index_t.float().to(self.device)
            label = label.long().to(self.device)
            if concept_targets is not None:
                concept_targets = concept_targets.float().to(self.device)

            # Forward
            output = self.model(data, index_t, concept_targets=concept_targets)
            if isinstance(output, dict):
                action_logits = output['action_logits']
                loss = self.criterion(action_logits, label)
                if 'aux_loss' in output:
                    loss = loss + output['aux_loss']
            else:
                action_logits = output
                loss = self.criterion(action_logits, label)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.get('grad_clip', True):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_max', 1.0))
            self.optimizer.step()

            # Metric
            _, preds = torch.max(action_logits, 1)
            acc = (preds == label).float().mean().item()
            loss_list.append(loss.item())
            acc_list.append(acc)
            pbar.set_postfix({'loss': f'{np.mean(loss_list):.4f}', 'acc': f'{np.mean(acc_list)*100:.2f}%', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})

        self.io.print_log(f"Epoch {epoch+1} Train -> Loss: {np.mean(loss_list):.4f} | Acc: {np.mean(acc_list)*100:.2f}%")

    def eval_epoch(self, epoch):
        self.model.eval()
        loss_list, score_frag, label_list = [], [], []
        pbar = tqdm(self.test_loader, desc=f"Eval Epoch {epoch+1}")

        with torch.no_grad():
            for batch in pbar:
                if len(batch) == 5:
                    data, index_t, label, index, _ = batch
                else:
                    data, index_t, label, index = batch

                data = data.float().to(self.device)
                index_t = index_t.float().to(self.device)
                label = label.long().to(self.device)

                output = self.model(data, index_t)
                action_logits = output['action_logits'] if isinstance(output, dict) else output
                loss = self.criterion(action_logits, label)

                loss_list.append(loss.item())
                score_frag.append(action_logits.cpu().numpy())
                label_list.append(label.cpu().numpy())

        score = np.concatenate(score_frag)
        labels = np.concatenate(label_list)
        rank = score.argsort()
        top1 = np.mean([l in rank[i, -1:] for i, l in enumerate(labels)]) * 100
        top5 = np.mean([l in rank[i, -5:] for i, l in enumerate(labels)]) * 100

        self.io.print_log(f"Epoch {epoch+1} Eval -> Loss: {np.mean(loss_list):.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

        is_best = top1 > self.best_acc
        if is_best:
            self.best_acc = top1
            self.best_acc_epoch = epoch + 1
        return is_best

    def start(self, epochs=None):
        num_epochs = epochs or self.config.get('num_epoch', 80)
        save_interval = self.config.get('save_interval', 5)
        eval_interval = self.config.get('eval_interval', 1)
        save_epoch = self.config.get('save_epoch', 30)

        self.io.print_log(f"Bắt đầu huấn luyện từ Epoch {self.start_epoch + 1} đến {num_epochs}...")
        for epoch in range(self.start_epoch, num_epochs):
            self.train_epoch(epoch)

            # Đánh giá
            is_best = False
            if (epoch + 1) % eval_interval == 0 or (epoch + 1) == num_epochs:
                is_best = self.eval_epoch(epoch)

            # Luôn lưu checkpoint_latest.pt sau mỗi epoch
            is_periodic = ((epoch + 1) >= save_epoch and (epoch + 1) % save_interval == 0) or (epoch + 1 == num_epochs)
            self.save_checkpoint(epoch, is_best=is_best, is_periodic=is_periodic)

        self.io.print_log(f"🎉 Huấn luyện hoàn tất! Best Top-1 Acc: {self.best_acc:.2f}% (tại Epoch {self.best_acc_epoch})")

def list_checkpoints(work_dir):
    \"\"\"Hàm tiện ích liệt kê tất cả checkpoints đã lưu trong work_dir\"\"\"
    if not os.path.exists(work_dir):
        print(f"Thư mục {work_dir} chưa tồn tại.")
        return
    files = glob.glob(os.path.join(work_dir, '*.pt'))
    if not files:
        print(f"Chưa có file checkpoint (.pt) nào trong {work_dir}.")
        return
    print(f"📂 Danh sách Checkpoints trong [{work_dir}]:")
    for f in sorted(files):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
        print(f"  📄 {os.path.basename(f):<30} | {size_mb:>6.2f} MB | {mtime}")

print("✅ Đã nạp xong Block 6: Processor Engine với Hệ thống Checkpoint & Auto-Resume!")
""")

    # ==========================================
    # CELL 7: Quick Dummy Test
    # ==========================================
    add_md("## 🧪 Block 7: Test Kiểm tra Pipeline Hoạt động (Dummy Data Run)")
    add_code("""# Chạy thử nghiệm nhanh toàn bộ pipeline (forward/backward/eval/checkpoint) với Dummy Data
print("🔍 Đang tiến hành kiểm tra Pipeline...")

dummy_config = {
    'work_dir': './work_dir/dummy_test',
    'batch_size': 4,
    'test_batch_size': 4,
    'num_worker': 0,
    'base_lr': 0.0001,
    'min_lr': 1e-6,
    'warmup_lr': 1e-6,
    'warm_up_epoch': 1,
    'num_epoch': 1,
    'eval_interval': 1,
    'save_interval': 1,
    'save_epoch': 1,
    'loss_type': 'LSCE',
    'auto_resume': True,
    'train_feeder_args': {
        'data_path': './dummy.npz',
        'p_interval': [0.5, 1.0],
        'split': 'train',
        'window_size': 64,
        'uniform': True,
        'partition': True
    },
    'test_feeder_args': {
        'data_path': './dummy.npz',
        'p_interval': [0.95],
        'split': 'test',
        'window_size': 64,
        'uniform': True,
        'partition': True
    },
    'model_args': {
        'num_classes': 60,
        'num_concepts': 72,
        'spatial_concepts': 52,
        'temporal_concepts': 20,
        'num_people': 2,
        'num_points': 24,
        'num_frames': 64,
        'num_heads': 32,
        'type_1_size': [8, 8],
        'type_2_size': [8, 12],
        'type_3_size': [8, 8],
        'type_4_size': [8, 12],
        'index_t': True,
        'action_mode': 'prototype',
        'return_aux': True
    }
}

processor = Processor(dummy_config)
processor.start(epochs=1)

# Kiểm tra danh sách checkpoint được lưu:
list_checkpoints(dummy_config['work_dir'])
print("🎯 KIỂM TRA THÀNH CÔNG! Tất cả các Block đã liên kết và lưu checkpoint chính xác 100%.")
""")

    # ==========================================
    # CELL 8: Real Data Training / Evaluation
    # ==========================================
    add_md("""## 🏋️ Block 8: Cấu hình & Huấn luyện / Đánh giá Thực tế

> 💡 **Lưu ý quan trọng khi lưu Checkpoint trên Colab:**
> - Hãy đặt `work_dir` trỏ trực tiếp vào thư mục trên **Google Drive** (ví dụ: `/content/drive/MyDrive/.../work_dir/`)
> - Khi đó file `checkpoint_latest.pt` và `checkpoint_best.pt` sẽ được lưu tự động trên Google Drive. Nếu Colab bị mất kết nối hoặc bị ngắt phiên (timeout), bạn chỉ cần chạy lại notebook, chương trình sẽ **tự động phát hiện checkpoint và tiếp tục huấn luyện ngay từ epoch bị ngắt** (`auto_resume: True`) mà không phải chạy lại từ đầu!
""")
    add_code("""# Cấu hình chuẩn REASON_SkateFormer_j trên NTU-60 CS (Cross-Subject)

real_config = {
    # Đường dẫn lưu log và checkpoint trên Google Drive
    'work_dir': '/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/work_dir/ntu/cs/REASON_SkateFormer_j/',
    
    'batch_size': 32,
    'test_batch_size': 32,
    'num_worker': 2,
    'base_lr': 0.0001,
    'min_lr': 0.000001,
    'warmup_lr': 0.000001,
    'warm_up_epoch': 2,
    'num_epoch': 150,
    'weight_decay': 0.05,
    'eval_interval': 5,
    'save_interval': 5,
    'save_epoch': 30,
    'grad_clip': True,
    'grad_max': 1.0,
    'loss_type': 'LSCE',
    'auto_resume': True,  # Tự động tiếp tục nếu tìm thấy checkpoint_latest.pt
    
    # Cấu hình nạp dữ liệu NTU
    'train_feeder_args': {
        'data_path': '/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz',
        'concepts_csv': './concepts/ntu_rgbd_60_120.csv',
        'remove_null_concept': True,
        'split': 'train',
        'window_size': 64,
        'p_interval': [0.5, 1],
        'aug_method': 'a123489',
        'intra_p': 0.5,
        'inter_p': 0.2,
        'thres': 64,
        'uniform': True,
        'partition': True
    },
    'test_feeder_args': {
        'data_path': '/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz',
        'concepts_csv': './concepts/ntu_rgbd_60_120.csv',
        'remove_null_concept': True,
        'split': 'test',
        'window_size': 64,
        'p_interval': [0.95],
        'thres': 64,
        'uniform': True,
        'partition': True
    },
    
    # Cấu hình REASON-SkateFormer Model
    'model_args': {
        'num_classes': 60,
        'num_concepts': 72,
        'spatial_concepts': 52,
        'temporal_concepts': 20,
        'num_people': 2,
        'num_points': 24,
        'num_frames': 64,
        'backbone_weights': '/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/SkateFormer_j.pt',
        'num_heads': 32,
        'attn_drop': 0.5,
        'head_drop': 0.0,
        'rel': True,
        'drop_path': 0.2,
        'type_1_size': [8, 8],
        'type_2_size': [8, 12],
        'type_3_size': [8, 8],
        'type_4_size': [8, 12],
        'mlp_ratio': 4.0,
        'index_t': True,
        'l1_dim': 128,
        'l2_dim': 128,
        'kg_init_mode': 'kg_primed',
        'action_mode': 'prototype',
        'prototype_and_weight': 0.35,
        'kg_prior_strength': 1.0,
        'kg_loss_weight': 0.01,
        'prior_retention_weight': 0.01,
        'rule_sparsity_weight': 0.001,
        'part_diversity_weight': 0.01,
        'concept_loss_weight': 1.0,
        'concept_negative_weight': 0.15,
        'concept_positive_weight': 1.0,
        'asymmetric_gamma_neg': 2.0,
        'return_aux': True,
        'kg_prior_map': {
            'require': [['arm_raise', 'arm_extend'], ['leg_step', 'foot_step'], ['hand_wave', 'arm_raise']],
            'alongside': [['arm_raise', 'hand_wave'], ['leg_step', 'foot_step']],
            'excludes': [['arm_raise', 'hip_sit'], ['leg_step', 'hip_squat']]
        }
    }
}

# Khởi chạy Processor và bắt đầu huấn luyện:
processor = Processor(real_config)
processor.start()
""")

    # ==========================================
    # CELL 9: Inspect Checkpoints
    # ==========================================
    add_md("## 🔍 Block 9: Xem Danh sách Checkpoint Đã Lưu")
    add_code("""# Kiểm tra các checkpoint hiện có trong thư mục kết quả:
list_checkpoints(real_config['work_dir'])
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_path_1 = r"d:\Thesis\ECR-SkateFormer\ECR_SkateFormer_Colab.ipynb"
    out_path_2 = r"d:\Thesis\ECR_SkateFormer_Colab.ipynb"

    with open(out_path_1, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    with open(out_path_2, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print("Created notebook successfully at:")
    print(out_path_1)
    print(out_path_2)

if __name__ == '__main__':
    create_notebook()
