"""
Symbolic Action Knowledge Graph Module
======================================
Provides ontological priors, class-class semantic similarity matrices (S_KG),
concept-concept co-activation correlation matrices (C_KG), and joint functional
dependency graphs (A_KG) for skeleton-based action recognition.
"""

from typing import Dict, List, Tuple
import csv
import numpy as np
import torch


# 25 Joints Functional Groupings (NTU 25-joint layout)
JOINT_GROUPS: Dict[str, List[int]] = {
    "Head_Neck": [2, 3, 20],  # Head, Neck, Spine_Neck
    "Torso": [0, 1, 12, 16],  # Spine_Base, Spine_Mid, Hip_Left, Hip_Right
    "Left_Arm": [4, 5, 6, 7, 21, 22],  # Shoulder, Elbow, Wrist, Hand, HandTip, Thumb
    "Right_Arm": [8, 9, 10, 11, 23, 24],  # Shoulder, Elbow, Wrist, Hand, HandTip, Thumb
    "Left_Leg": [12, 13, 14, 15],  # Hip, Knee, Ankle, Foot
    "Right_Leg": [16, 17, 18, 19],   # Hip, Knee, Ankle, Foot
}


def build_joint_semantic_dependency_matrix(num_joints: int = 25, num_subsets: int = 3) -> torch.Tensor:
    """
    Constructs a (3, 25, 25) Joint Functional Dependency Prior Matrix A_KG.
    Encodes anatomical functional couplings (e.g. Hand-to-Head, Hand-to-Hand).
    """
    A_KG = np.zeros((num_subsets, num_joints, num_joints), dtype=np.float32)

    # Subset 0: Intra-group functional connections (Self & Same Body Part)
    for group_name, joints in JOINT_GROUPS.items():
        for j1 in joints:
            for j2 in joints:
                A_KG[0, j1, j2] = 1.0

    # Subset 1: Hand-to-Head / Upper-body coordination
    for h in JOINT_GROUPS["Head_Neck"]:
        for arm in JOINT_GROUPS["Left_Arm"] + JOINT_GROUPS["Right_Arm"]:
            A_KG[1, h, arm] = 0.8
            A_KG[1, arm, h] = 0.8

    # Subset 2: Bilateral coordination (Left Arm <-> Right Arm, Left Leg <-> Right Leg)
    for l_arm, r_arm in zip(JOINT_GROUPS["Left_Arm"], JOINT_GROUPS["Right_Arm"]):
        A_KG[2, l_arm, r_arm] = 0.7
        A_KG[2, r_arm, l_arm] = 0.7
    for l_leg, r_leg in zip(JOINT_GROUPS["Left_Leg"], JOINT_GROUPS["Right_Leg"]):
        A_KG[2, l_leg, r_leg] = 0.7
        A_KG[2, r_leg, l_leg] = 0.7

    # Normalize along rows
    for k in range(num_subsets):
        row_sum = A_KG[k].sum(axis=-1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        A_KG[k] = A_KG[k] / row_sum

    return torch.from_numpy(A_KG).float()


def build_semantic_similarity_from_concepts(concept_matrix: torch.Tensor) -> torch.Tensor:
    """
    Class-class Jaccard similarity over the (num_class, num_concepts) concept matrix:
    S_KG[i, j] = |concepts shared by i and j| / |concepts used by either|.
    """
    M = concept_matrix.float()
    intersection = M @ M.t()
    row_sum = M.sum(dim=1, keepdim=True)
    union = row_sum + row_sum.t() - intersection
    S = intersection / union.clamp(min=1.0)
    S.fill_diagonal_(1.0)
    return S


def build_concept_coactivation_static(concept_matrix: torch.Tensor) -> torch.Tensor:
    """
    Concept-concept Jaccard co-occurrence, computed over which action classes
    use each concept: C_KG[i, j] = |classes using both concept i and j| /
    |classes using either|.
    """
    Mt = concept_matrix.float().t()  # (num_concepts, num_class)
    intersection = Mt @ Mt.t()
    row_sum = Mt.sum(dim=1, keepdim=True)
    union = row_sum + row_sum.t() - intersection
    C = intersection / union.clamp(min=1.0)
    C.fill_diagonal_(1.0)
    return C
