"""
Concept Utilities and Semantic Grouping Specifications
======================================================
Unified concept management for Neurosymbolic Action Recognition.
Supports partitioning concepts into 9 anatomical & temporal regions
and building natural language prompts for CLIP grounding.
"""

import os
import csv
from typing import Dict, List, Tuple
import numpy as np
import torch

GROUP_SPECS = [
    ("head", ("head_",)),
    ("hand", ("hand_", "interaction_two_person")),
    ("arm", ("arm_",)),
    ("hip", ("hip_",)),
    ("leg", ("leg_",)),
    ("foot", ("foot_",)),
    ("temporal_direction", ("temporal_direction_",)),
    ("temporal_sequence", ("temporal_sequence_",)),
    ("temporal_dynamics", ("temporal_dynamics_",)),
]


def load_concept_annotations(
    csv_path: str,
    num_classes: int = 60,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Loads concept annotations from CSV.
    
    Returns:
        concept_matrix: Tensor of shape (num_classes, num_concepts) with binary/weighted values.
        concept_names: List of all concept names.
        class_names: List of all action class names.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Concept CSV not found at: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    # First column is class_name / action_id
    concept_names = [col.strip() for col in header[1:]]
    
    class_names = []
    matrix_rows = []
    
    for row in rows[1:num_classes + 1]:
        if not row:
            continue
        class_names.append(row[0].strip())
        vals = [float(v.strip()) if v.strip() else 0.0 for v in row[1:]]
        matrix_rows.append(vals)

    concept_matrix = torch.tensor(matrix_rows, dtype=torch.float32)
    return concept_matrix, concept_names, class_names


def build_concept_groups(concept_names: List[str]) -> Dict[str, List[int]]:
    """
    Partitions concept names into 9 semantically grouped anatomical & temporal regions.
    """
    groups = {name: [] for name, _ in GROUP_SPECS}
    used = set()
    
    for name, prefixes in GROUP_SPECS:
        for idx, concept in enumerate(concept_names):
            if idx in used:
                continue
            if any(concept.startswith(prefix) or concept == prefix for prefix in prefixes):
                groups[name].append(idx)
                used.add(idx)

    missing = [i for i in range(len(concept_names)) if i not in used]
    if missing:
        # Assign any remaining unassigned concepts to temporal_dynamics or general group
        for idx in missing:
            groups["temporal_dynamics"].append(idx)

    return groups


def generate_concept_prompts(concept_names: List[str]) -> List[str]:
    """
    Generates standardized descriptive text prompts for CLIP text encoder.
    Example: "a person performing action involving head nod"
    """
    prompts = []
    for c in concept_names:
        clean_name = c.replace("_", " ").strip()
        prompts.append(f"a human skeleton motion performing {clean_name}")
    return prompts
