"""Named body-part synergies on the NTU RGB+D skeleton.

Each synergy is a set of 0-based NTU joints with a name; the synergy bottleneck
pools backbone features over these sets. A random bank with the same group
sizes is the control for the names. See SEMANTIC_HYPERGRAPH_PLAN.md, section 2.1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

NUM_JOINTS = 25

HAND_L = (6, 7, 21, 22)
HAND_R = (10, 11, 23, 24)
HEAD = (2, 3)


@dataclass(frozen=True)
class Synergy:
    name: str
    joints: tuple[int, ...]
    description: str = ""


SINGLE_SYNERGIES = (
    Synergy("hand_R-head", HAND_R + HEAD, "drink, eat, phone call, brush hair"),
    Synergy("hand_L-head", HAND_L + HEAD, "as above, left hand"),
    Synergy("bimanual", HAND_L + HAND_R, "clap, tear paper, type"),
    Synergy("arm_R-torso", (8, 9, 10, 11, 20, 1), "put on jacket, touch chest"),
    Synergy("arm_L-torso", (4, 5, 6, 7, 20, 1), "as above, left arm"),
    Synergy("hand_R-hip", (10, 11, 16, 0), "reach into pocket"),
    Synergy("hand_L-hip", (6, 7, 12, 0), "as above, left hand"),
    Synergy("hand_R-foot_R", (10, 11, 18, 19), "put on shoe, pick up"),
    Synergy("hand_L-foot_L", (6, 7, 14, 15), "as above, left side"),
    Synergy("legs", tuple(range(12, 20)), "jump, kick, step"),
    Synergy("leg_R-torso", (16, 17, 18, 0, 1), "kick with right leg, stand on one leg"),
    Synergy("leg_L-torso", (12, 13, 14, 0, 1), "as above, left leg"),
    Synergy("head-torso", (3, 2, 20, 1), "nod, shake head, bow"),
    Synergy("spine", (0, 1, 20, 2), "bend, lean, tilt"),
    Synergy("shoulder_twist", (4, 8, 20, 1), "twist the upper body"),
    Synergy("arm_R-leg_L", (9, 10, 13, 14), "walking (contralateral)"),
    Synergy("arm_L-leg_R", (5, 6, 17, 18), "walking (contralateral)"),
    Synergy("upper_body", (2, 3) + tuple(range(4, 12)) + (20,), "stretch, raise arms"),
    Synergy("hip-knee-spine", (0, 12, 16, 13, 17, 1, 20), "sit down, stand up"),
)


def random_bank(seed=0):
    """Control: distinct random joint groups with the sizes of the semantic bank."""
    rng = random.Random(seed)
    used, bank = set(), []
    for i, synergy in enumerate(SINGLE_SYNERGIES):
        while True:
            group = tuple(sorted(rng.sample(range(NUM_JOINTS), len(synergy.joints))))
            if group not in used:
                break
        used.add(group)
        bank.append(Synergy(f"random_{i:02d}", group))
    return tuple(bank)


def synergy_bank(kind="semantic", seed=0):
    if kind == "semantic":
        return SINGLE_SYNERGIES
    if kind == "random":
        return random_bank(seed)
    raise ValueError("synergy bank must be 'semantic' or 'random'")


def validate_bank(synergies):
    names = [s.name for s in synergies]
    if len(set(names)) != len(names):
        raise ValueError("synergy names must be unique")
    for s in synergies:
        if len(s.joints) < 2 or len(set(s.joints)) != len(s.joints):
            raise ValueError(f"synergy {s.name} needs at least two distinct joints")
        if any(not 0 <= j < NUM_JOINTS for j in s.joints):
            raise ValueError(f"synergy {s.name} has a joint outside 0..{NUM_JOINTS - 1}")


def incidence(groups, num_joints=NUM_JOINTS):
    """Binary incidence matrix [num_joints, E] of joint groups (tuples of joints)."""
    matrix = torch.zeros(num_joints, len(groups))
    for e, joints in enumerate(groups):
        matrix[list(joints), e] = 1.0
    return matrix
