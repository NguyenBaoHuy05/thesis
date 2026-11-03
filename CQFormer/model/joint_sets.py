"""Concept -> joint assignment: each concept only ever observes the joints of its set.

Joint indices are 0-based NTU RGB+D (Kinect v2) order:
0 spine base, 1 spine mid, 2 neck, 3 head, 4-7 L shoulder/elbow/wrist/hand,
8-11 R shoulder/elbow/wrist/hand, 12-15 L hip/knee/ankle/foot, 16-19 R hip/knee/ankle/foot,
20 spine shoulder, 21 L hand tip, 22 L thumb, 23 R hand tip, 24 R thumb.

The 21 ``temporal_*`` CSV columns are not concepts here: order and rhythm are left to the
temporal motif layer of the model.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

NUM_JOINTS = 25

# same pairs as feeders.tools.joint2bone; joint 1 (spine mid) is the root
PARENT = {0: 1, 1: 1, 2: 20, 3: 2, 4: 20, 5: 4, 6: 5, 7: 6, 8: 20, 9: 8, 10: 9, 11: 10,
          12: 0, 13: 12, 14: 13, 15: 14, 16: 0, 17: 16, 18: 17, 19: 18, 20: 1,
          21: 7, 22: 7, 23: 11, 24: 11}

_HEAD = [2, 3, 20]
_HANDS = [6, 7, 21, 22, 10, 11, 23, 24]
_ARMS = [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24]
_LEGS = [12, 13, 14, 15, 16, 17, 18, 19]
_FEET = [14, 15, 18, 19]
_HIP = [0, 1, 12, 16, 20]
_HIP_KNEE = _HIP + [13, 17]


@dataclass(frozen=True)
class JointSet:
    name: str
    joints: tuple[int, ...]
    concepts: tuple[str, ...]
    two_person: bool = False   # True: both skeletons are seen together, as one token grid


JOINT_SETS: list[JointSet] = [
    JointSet("head", tuple(_HEAD), ("head_nod", "head_shake", "head_tilt_down", "head_tilt_up",
                                    "head_rotate", "head_static")),
    JointSet("hands", tuple(_HANDS), ("hand_grasp", "hand_release", "hand_wave", "hand_point",
                                      "hand_clap", "hand_manipulate", "hand_fist", "hand_open",
                                      "hand_touch", "hand_circular")),
    JointSet("hands_head", tuple(_HANDS + _HEAD), ("hand_lift_to_face",)),
    JointSet("arms", tuple(_ARMS), ("arm_raise", "arm_lower", "arm_extend", "arm_retract",
                                    "arm_circular", "arm_swing", "arm_cross", "arm_hold",
                                    "arm_alternating", "arm_lateral")),
    JointSet("arms_head", tuple(_ARMS + _HEAD), ("arm_overhead",)),
    JointSet("hip", tuple(_HIP), ("hip_bend", "hip_extend", "hip_rotate", "hip_static_stand",
                                  "hip_sway")),
    JointSet("hip_knee", tuple(_HIP_KNEE), ("hip_sit", "hip_squat", "hip_static_sit")),
    JointSet("legs", tuple(_LEGS), ("leg_step", "leg_kick", "leg_jump", "leg_squat", "leg_lift",
                                    "leg_static_stand", "leg_static_sit", "leg_alternating",
                                    "leg_balance", "foot_jump")),
    JointSet("feet", tuple(_FEET), ("foot_step", "foot_kick", "foot_pivot", "foot_static",
                                    "foot_tap", "foot_slide")),
    JointSet("interaction", tuple(range(NUM_JOINTS)), ("interaction_two_person",), two_person=True),
]


def concept_names(sets: list[JointSet] = JOINT_SETS) -> list[str]:
    return [c for s in sets for c in s.concepts]


def is_state_concept(name: str) -> bool:
    """Concepts that must hold across the clip: their temporal pooling is close to min, not max."""
    return "static" in name or name == "arm_hold"


def load_concept_matrix(csv_path: str, num_classes: int, sets: list[JointSet] = JOINT_SETS) -> np.ndarray:
    """(num_classes, num_concepts) binary matrix, columns in the order of ``concept_names``."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    columns = [c for c in rows[0] if c != "action_class"]
    names = concept_names(sets)
    expected = {c for c in columns if not c.startswith("temporal_")}
    if expected != set(names) or len(names) != len(set(names)):
        raise ValueError("JOINT_SETS and {} disagree: missing {}, unknown {}".format(
            csv_path, sorted(expected - set(names)), sorted(set(names) - expected)))
    if len(rows) < num_classes:
        raise ValueError("{} has {} classes, need {}".format(csv_path, len(rows), num_classes))
    return np.array([[float(r[c]) for c in names] for r in rows[:num_classes]], dtype=np.float32)
