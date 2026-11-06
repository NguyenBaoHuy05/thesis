"""Mapping between continuous measures / body regions and the class-level concept CSV (v3).

The concept CSV (``GCCR-Skateformer-TreeRule/concepts/ntu_rgbd_60_120.csv``, 74 binary
concept columns, 120 action rows, first 60 rows = NTU-60 in label order A1..A60) is used
ONLY for the final human-knowledge check of induced rules; it never drives measurement,
intervention or rule induction.

``MEASURE_TO_CONCEPTS[m]["<"]`` lists the concepts implied when measure ``m`` is LOW
(rule literal ``m:stat < theta``), ``[">"]`` those implied when it is HIGH. Only real
column names of the GCCR CSV are used (checked by ``check_concept_names``).
"""
from __future__ import annotations

import csv
import os

import numpy as np

from .measures import MEASURE_NAMES
from .regions import REGION_NAMES

__all__ = [
    "GCCR_CONCEPT_CSV",
    "MEASURE_TO_CONCEPTS",
    "REGION_TO_CONCEPTS",
    "load_concept_csv",
    "check_concept_names",
    "class_level_measure_prior",
    "class_level_region_prior",
]

_THESIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GCCR_CONCEPT_CSV = os.path.join(_THESIS_ROOT, "GCCR-Skateformer-TreeRule", "concepts", "ntu_rgbd_60_120.csv")

_SIT = ["hip_sit", "hip_static_sit", "leg_static_sit", "leg_squat", "hip_squat"]

MEASURE_TO_CONCEPTS: dict[str, dict[str, list[str]]] = {
    "hand_head_dist": {"<": ["hand_lift_to_face"], ">": []},
    "hand_chest_dist": {"<": ["arm_hold", "arm_cross"], ">": ["arm_extend"]},
    "hands_dist": {"<": ["hand_clap", "temporal_direction_motion_converging"],
                   ">": ["temporal_direction_motion_diverging", "arm_lateral"]},
    "hand_height_head": {"<": [], ">": ["arm_overhead"]},
    "hand_height_shoulder": {"<": [], ">": ["arm_raise", "arm_overhead"]},
    "hand_floor": {"<": ["hip_bend"], ">": []},
    "arm_reach_fwd": {"<": [], ">": ["arm_extend", "hand_point"]},
    "elbow_angle": {"<": ["arm_hold"], ">": ["arm_extend"]},
    "wrist_speed": {"<": ["temporal_dynamics_speed_slow"],
                    ">": ["temporal_dynamics_speed_fast", "arm_swing", "hand_wave"]},
    "hand_symmetry": {"<": ["arm_alternating"], ">": ["temporal_sequence_seq_simultaneous", "hand_clap"]},
    "torso_pitch": {"<": [], ">": ["hip_bend"]},
    "hip_height": {"<": list(_SIT), ">": ["hip_static_stand", "leg_static_stand"]},
    "knee_angle": {"<": list(_SIT), ">": ["leg_static_stand"]},
    "foot_height_diff": {"<": [], ">": ["leg_kick", "foot_kick", "leg_lift", "leg_balance"]},
    "leg_alternation": {"<": ["leg_static_stand", "foot_static"],
                        ">": ["leg_step", "foot_step", "leg_alternating"]},
    "body_disp_h": {"<": ["foot_static"],
                    ">": ["leg_step", "foot_step", "temporal_direction_motion_forward",
                          "temporal_direction_motion_backward"]},
    "body_disp_v": {"<": ["hip_sit", "temporal_direction_motion_downward"],
                    ">": ["hip_extend", "temporal_direction_motion_upward"]},
    "person_dist": {"<": ["interaction_two_person"], ">": []},
    "contact_dist": {"<": ["interaction_two_person", "hand_touch"], ">": []},
    "second_person": {"<": [], ">": ["interaction_two_person"]},
}
assert list(MEASURE_TO_CONCEPTS) == MEASURE_NAMES, set(MEASURE_TO_CONCEPTS) ^ set(MEASURE_NAMES)

_ARM_CONCEPTS = [
    "hand_grasp", "hand_release", "hand_wave", "hand_point", "hand_clap", "hand_manipulate",
    "hand_fist", "hand_open", "hand_touch", "hand_lift_to_face", "hand_circular",
    "arm_raise", "arm_lower", "arm_extend", "arm_retract", "arm_circular", "arm_swing",
    "arm_cross", "arm_hold", "arm_alternating", "arm_overhead", "arm_lateral",
]
_LEG_CONCEPTS = [
    "leg_step", "leg_kick", "leg_jump", "leg_squat", "leg_lift", "leg_alternating", "leg_balance",
    "foot_step", "foot_kick", "foot_jump", "foot_pivot", "foot_tap", "foot_slide",
]

# the CSV has no laterality: both arms (legs) map to the same concepts
REGION_TO_CONCEPTS: dict[str, list[str]] = {
    "head": ["head_nod", "head_shake", "head_tilt_down", "head_tilt_up", "head_rotate"],
    "torso": ["hip_bend", "hip_extend", "hip_sit", "hip_squat", "hip_rotate", "hip_sway"],
    "left_arm": list(_ARM_CONCEPTS),
    "right_arm": list(_ARM_CONCEPTS),
    "left_leg": list(_LEG_CONCEPTS),
    "right_leg": list(_LEG_CONCEPTS),
    "other_person": ["interaction_two_person"],
}
assert list(REGION_TO_CONCEPTS) == REGION_NAMES


def load_concept_csv(path: str = GCCR_CONCEPT_CSV):
    """Returns (action_names: list[str], concept_names: list[str], matrix (A, C) float32).

    All concept columns are kept in file order (no null-column removal).
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    if header[0] != "action_class":
        raise ValueError(f"{path}: first column must be action_class, got {header[0]!r}")
    concept_names = header[1:]
    action_names, mat = [], []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        action_names.append(r[0].strip())
        mat.append([float(v) for v in r[1:1 + len(concept_names)]])
    return action_names, concept_names, np.asarray(mat, dtype=np.float32)


def check_concept_names(csv_path: str = GCCR_CONCEPT_CSV) -> list[str]:
    """Names used in the maps that are NOT columns of the CSV (should be empty)."""
    _, concept_names, _ = load_concept_csv(csv_path)
    cols = set(concept_names)
    used = {c for d in MEASURE_TO_CONCEPTS.values() for lst in d.values() for c in lst}
    used |= {c for lst in REGION_TO_CONCEPTS.values() for c in lst}
    return sorted(used - cols)


def _any_concept(mat: np.ndarray, col: dict, names: list[str]) -> np.ndarray:
    idx = [col[c] for c in names if c in col]
    if not idx:
        return np.zeros(mat.shape[0], dtype=np.int8)
    return (mat[:, idx] > 0.5).any(axis=1).astype(np.int8)


def class_level_measure_prior(csv_path: str = GCCR_CONCEPT_CSV, num_class: int = 60) -> dict:
    """measure -> {"<": (K,) int8, ">": (K,) int8}.

    Entry [dir][k] = 1 if class k (first ``num_class`` CSV rows, NTU label order) has any
    concept that MEASURE_TO_CONCEPTS lists for that measure and direction. Empty / absent
    mappings give all-zero vectors.
    """
    _, concept_names, mat = load_concept_csv(csv_path)
    mat = mat[:num_class]
    col = {c: i for i, c in enumerate(concept_names)}
    return {m: {d: _any_concept(mat, col, MEASURE_TO_CONCEPTS[m][d]) for d in ("<", ">")}
            for m in MEASURE_NAMES}


def class_level_region_prior(csv_path: str = GCCR_CONCEPT_CSV, num_class: int = 60) -> np.ndarray:
    """(K, R) int8, columns in REGION_NAMES order: 1 if class k has any concept of region r."""
    _, concept_names, mat = load_concept_csv(csv_path)
    mat = mat[:num_class]
    col = {c: i for i, c in enumerate(concept_names)}
    return np.stack([_any_concept(mat, col, REGION_TO_CONCEPTS[r]) for r in REGION_NAMES], axis=1)
