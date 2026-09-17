"""
Symbolic Action Knowledge Graph Module (SK-GCN)
=================================================
Author: Antigravity AI & Thesis Research Team
Description:
    Provides domain-specific ontology priors, semantic similarity matrices,
    and joint functional dependency graphs for NTU-RGB+D 60 dataset.
    Used by SK-GCN to constrain neural graph topologies and guide
    inter-class affinitive contrastive learning.
"""

from typing import Dict, List, Tuple
import csv
import numpy as np
import torch


# ─── NTU-RGB+D 60 Semantic Action Ontology Groupings ─────────────────────────

NTU60_ACTION_GROUPS: Dict[str, List[int]] = {
    "UpperBody_Daily": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 34, 35],
    "LowerBody_Locomotion": [26, 27, 28, 29, 42],
    "Object_Interaction": [10, 11, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39],
    "Health_Medical": [43, 44, 45, 46, 47, 48, 49],
    "Interactive_Social": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "Head_Neck": [15, 16, 17, 18, 19, 20, 21, 40, 41]
}

# 25 Joints Functional Groupings (NTU 25-joint layout)
JOINT_GROUPS: Dict[str, List[int]] = {
    "Head_Neck": [2, 3, 20],  # Head, Neck, Spine_Neck
    "Torso": [0, 1, 12, 16],  # Spine_Base, Spine_Mid, Hip_Left, Hip_Right
    "Left_Arm": [4, 5, 6, 7, 21, 22],  # Shoulder, Elbow, Wrist, Hand, HandTip, Thumb
    "Right_Arm": [8, 9, 10, 11, 23, 24],  # Shoulder, Elbow, Wrist, Hand, HandTip, Thumb
    "Left_Leg": [12, 13, 14, 15],  # Hip, Knee, Ankle, Foot
    "Right_Leg": [16, 17, 18, 19]   # Hip, Knee, Ankle, Foot
}


def build_semantic_similarity_matrix(num_classes: int = 60) -> torch.Tensor:
    """
    Constructs a normalized (60, 60) Semantic Similarity Matrix S_KG from Ontology.
    High similarity indicates actions sharing body parts or functional intent.
    """
    S_KG = np.eye(num_classes, dtype=np.float32)

    for group_name, action_ids in NTU60_ACTION_GROUPS.items():
        for i in action_ids:
            for j in action_ids:
                if i < num_classes and j < num_classes:
                    S_KG[i, j] += 0.5

    norm = np.linalg.norm(S_KG, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    S_KG = S_KG / norm
    S_KG = (S_KG + S_KG.T) / 2.0  # Symmetric

    return torch.from_numpy(S_KG).float()


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


# ─── Meaningful KG: grounded in named motion-concept annotations ─────────────
#
# NTU60_ACTION_GROUPS above is a hand-drawn clustering — useful as a joint-level
# anatomical prior, but too coarse/arbitrary to justify class-class "semantic
# similarity" claims. The functions below instead derive S_KG and a concept-
# concept co-occurrence graph (C_KG) from an explicit, named motion-concept
# annotation table (one row per action class, one column per concept, e.g.
# `hand_grasp`, `arm_raise`, `temporal_dynamics_speed_fast`). Every entry in
# the resulting matrices is traceable back to a concrete shared/absent concept
# set, not to an invented grouping.

def load_concept_matrix(csv_path: str, num_class: int) -> Tuple[torch.Tensor, List[str]]:
    """
    Load a (num_class, num_concepts) binary concept-annotation matrix.

    Expects a CSV with header `action_class,<concept_1>,<concept_2>,...` and
    one row per action class, in canonical NTU action-ID order (row 0 = A001,
    row 59 = A060, ...). Only the first `num_class` rows are used, so the same
    file backs both NTU-60 and NTU-120 configs.
    """
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        concept_names = header[1:]
        rows = list(reader)

    if len(rows) < num_class:
        raise ValueError(
            f"Concept CSV at {csv_path} has only {len(rows)} classes, "
            f"but num_class={num_class} was requested."
        )

    matrix = np.array(
        [[float(v) for v in row[1:]] for row in rows[:num_class]],
        dtype=np.float32,
    )
    return torch.from_numpy(matrix), concept_names


def build_semantic_similarity_from_concepts(concept_matrix: torch.Tensor) -> torch.Tensor:
    """
    Class-class Jaccard similarity over the (num_class, num_concepts) binary
    concept matrix: S_KG[i, j] = |concepts shared by i and j| / |concepts used
    by either|. Meaningful by construction — every entry is explainable by
    the concrete set of shared named concepts, unlike an ad-hoc grouping.
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
    |classes using either|. This is the static (symbolic) half of the
    Concept_Coactivation signal — the dynamic (empirical) half is tracked at
    train time from the concept head's own prediction errors.
    """
    Mt = concept_matrix.float().t()  # (num_concepts, num_class)
    intersection = Mt @ Mt.t()
    row_sum = Mt.sum(dim=1, keepdim=True)
    union = row_sum + row_sum.t() - intersection
    C = intersection / union.clamp(min=1.0)
    C.fill_diagonal_(1.0)
    return C
