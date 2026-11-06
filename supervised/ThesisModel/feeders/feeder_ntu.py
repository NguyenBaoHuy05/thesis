"""
Multi-Modal Dataset Feeder for NTU RGB+D
=======================================
Integrates skeleton joint/bone coordinates, 9-group anatomical & temporal concept vectors,
and natural language CLIP prompts into a high-throughput PyTorch Dataset.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from concepts.concept_utils import (
    build_concept_groups,
    generate_concept_prompts,
    load_concept_annotations,
)
from feeders import tools


def custom_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Collate function packaging skeleton tensor, labels, concept dicts, and CLIP prompts.
    """
    data_list = []
    labels_list = []
    concept_full_list = []
    index_t_list = []
    
    # Check what items exist in batch
    for item in batch:
        data_list.append(torch.from_numpy(item[0]) if isinstance(item[0], np.ndarray) else item[0])
        labels_list.append(item[1])
        concept_full_list.append(torch.from_numpy(item[2]) if isinstance(item[2], np.ndarray) else item[2])
        if len(item) > 3 and item[3] is not None:
            index_t_list.append(torch.from_numpy(item[3]) if isinstance(item[3], np.ndarray) else item[3])

    data_tensor = torch.stack(data_list, dim=0).float()
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    concept_full_tensor = torch.stack(concept_full_list, dim=0).float()

    index_t_tensor = torch.stack(index_t_list, dim=0).float() if index_t_list else None

    # Group concept vectors
    first_item_dict = batch[0][4] if len(batch[0]) > 4 else {}
    concept_dict = {}
    for group_name in first_item_dict.keys():
        concept_dict[group_name] = torch.stack([
            torch.from_numpy(b[4][group_name]) if isinstance(b[4][group_name], np.ndarray) else b[4][group_name]
            for b in batch
        ], dim=0).float()

    prompts_by_group = batch[0][5] if len(batch[0]) > 5 else {}

    return data_tensor, labels_tensor, concept_full_tensor, concept_dict, index_t_tensor, prompts_by_group


class NTUFeeder(Dataset):
    """
    Unified Dataset Feeder for NTU RGB+D 60/120 Action Recognition.
    """

    def __init__(
        self,
        data_path: str,
        label_path: Optional[str] = None,
        split: str = "train",
        data_type: str = "j",
        p_interval: Tuple[float, ...] = (0.5, 1.0),
        window_size: int = 64,
        bone: bool = False,
        vel: bool = False,
        random_rot: bool = False,
        random_move: bool = False,
        spatial_flip: bool = False,
        partition: bool = False,
        concepts_csv: Optional[str] = "concepts/ntu_rgbd_60_120.csv",
        num_classes: int = 60,
        debug: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type.lower()
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.random_rot = random_rot
        self.random_move = random_move
        self.spatial_flip = spatial_flip
        self.partition = partition
        self.num_classes = num_classes
        self.debug = debug

        # SkateFormer follows the original implementation and removes NTU
        # joint 21, using a reordered 24-joint layout. This is required for
        # pretrained SkateFormer checkpoints whose graph has 48 nodes
        # (24 joints x 2 people).
        if self.partition:
            right_arm = np.array([7, 8, 22, 23]) - 1
            left_arm = np.array([11, 12, 24, 25]) - 1
            right_leg = np.array([13, 14, 15, 16]) - 1
            left_leg = np.array([17, 18, 19, 20]) - 1
            h_torso = np.array([5, 9, 6, 10]) - 1
            w_torso = np.array([2, 3, 1, 4]) - 1
            self.partition_indices = np.concatenate(
                (right_arm, left_arm, right_leg, left_leg, h_torso, w_torso), axis=-1
            )
        else:
            self.partition_indices = None

        # Load concept annotations and groups
        self.has_concepts = False
        if concepts_csv and os.path.exists(concepts_csv):
            self.concept_matrix, self.concept_names, self.class_names = load_concept_annotations(
                concepts_csv, num_classes=num_classes
            )
            self.concept_groups = build_concept_groups(self.concept_names)
            
            # Pre-generate prompts per group
            self.prompts_by_group = {}
            for g_name, indices in self.concept_groups.items():
                g_concepts = [self.concept_names[i] for i in indices]
                self.prompts_by_group[g_name] = generate_concept_prompts(g_concepts)
            
            self.has_concepts = True
        else:
            self.concept_matrix = None
            self.concept_names = []
            self.concept_groups = {}
            self.prompts_by_group = {}

        self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_path):
            if self.debug:
                # Generate dummy dataset for testing
                N = 20 if self.split == "train" else 8
                self.data = np.random.randn(N, 3, 64, 25, 2).astype(np.float32)
                self.label = np.random.randint(0, self.num_classes, size=(N,))
                return
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        if self.data_path.endswith(".npz"):
            npz = np.load(self.data_path)
            if self.split == "train":
                self.data = npz["x_train"]
                raw_y = npz["y_train"]
            else:
                self.data = npz["x_test"]
                raw_y = npz["y_test"]

            if raw_y.ndim > 1:
                self.label = np.where(raw_y > 0)[1] if raw_y.shape[1] > 1 else raw_y.flatten()
            else:
                self.label = raw_y
        elif self.data_path.endswith(".npy"):
            self.data = np.load(self.data_path)
            if self.label_path and os.path.exists(self.label_path):
                self.label = np.load(self.label_path)
            else:
                self.label = np.zeros((len(self.data),), dtype=np.int64)
        else:
            raise ValueError(f"Unsupported file format for data_path: {self.data_path}")

        # Standardize skeleton data to (N, C, T, V, M) with C=3
        if self.data.ndim == 3:
            N, T, D = self.data.shape
            if D == 150:
                self.data = self.data.reshape(N, T, 2, 25, 3).transpose(0, 4, 1, 3, 2)
            elif D == 75:
                self.data = self.data.reshape(N, T, 1, 25, 3).transpose(0, 4, 1, 3, 2)
            elif D == 60:
                self.data = self.data.reshape(N, T, 1, 20, 3).transpose(0, 4, 1, 3, 2)
            elif D == 120:
                self.data = self.data.reshape(N, T, 2, 20, 3).transpose(0, 4, 1, 3, 2)
            else:
                M = 2 if (D % 50 == 0) else 1
                V = D // (3 * M)
                self.data = self.data.reshape(N, T, M, V, 3).transpose(0, 4, 1, 3, 2)
        elif self.data.ndim == 4:
            if self.data.shape[1] == 3:
                self.data = self.data[:, :, :, :, None]
            elif self.data.shape[3] == 3:
                self.data = self.data.transpose(0, 3, 1, 2)[:, :, :, :, None]
        elif self.data.ndim == 5:
            if self.data.shape[3] == 3 and self.data.shape[1] != 3:
                self.data = self.data.transpose(0, 3, 1, 2, 4)
            elif self.data.shape[4] == 3 and self.data.shape[1] != 3:
                self.data = self.data.transpose(0, 4, 1, 3, 2)

        if self.partition:
            if self.data.shape[3] != 25:
                raise ValueError(
                    "SkateFormer partition expects 25-joint NTU input before "
                    f"partitioning, got V={self.data.shape[3]}.")
            if self.data.shape[1] != 3:
                raise ValueError(
                    "SkateFormer pretrained Joint/Bone/Motion input must have "
                    f"C=3, but the dataset has C={self.data.shape[1]}. "
                    "Use an un-concatenated modality export."
                )
        self.num_points = 24 if self.partition else int(self.data.shape[3])

        if self.debug:
            self.data = self.data[:16]
            self.label = self.label[:16]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        data_numpy = self.data[index].copy()  # (C, T, V, M)
        label = int(self.label[index])

        # Valid frame calculation
        valid_frame = (data_numpy != 0).sum(axis=(0, 2, 3)) > 0
        valid_frame_num = int(valid_frame.sum())
        if valid_frame_num == 0:
            valid_frame_num = data_numpy.shape[1]

        # Temporal crop & resize
        if self.split == "train":
            data_numpy, index_t = tools.valid_crop_resize(
                data_numpy, valid_frame_num, self.p_interval, self.window_size
            )
            if self.random_rot:
                data_numpy = tools.random_rot(data_numpy)
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            if self.spatial_flip:
                data_numpy = tools.spatial_flip(data_numpy)
        else:
            data_numpy, index_t = tools.valid_crop_resize(
                data_numpy, valid_frame_num, (1.0,), self.window_size
            )

        # Modality conversion. Match the original SkateFormer feeder:
        # j=joint, b=bone, jm=joint motion, bm=bone motion. Each modality
        # remains C=3; modalities are not concatenated into C=6 here.
        if self.data_type == "b" or (self.data_type == "j" and self.bone):
            data_numpy = tools.to_bone(data_numpy)
        if self.data_type in ("jm", "bm") or (self.data_type == "j" and self.vel):
            data_numpy = tools.to_velocity(data_numpy)

        # Partition only after modality conversion, matching the original
        # SkateFormer feeder (bone edges use the original 25-joint layout).
        if self.partition:
            data_numpy = data_numpy[:, :, self.partition_indices, :]

        # Concept vector ground truth
        if self.has_concepts:
            concept_full = self.concept_matrix[label].numpy().copy()
            concept_by_group = {}
            for g_name, indices in self.concept_groups.items():
                concept_by_group[g_name] = concept_full[indices]
        else:
            concept_full = np.zeros((1,), dtype=np.float32)
            concept_by_group = {}

        return data_numpy, label, concept_full, index_t, concept_by_group, self.prompts_by_group
