"""Named semantic hyperedges (body-part synergies) on the NTU RGB+D skeleton.

Each synergy is a set of 0-based NTU joints with a name. Single-person
synergies become hyperedges inside the last HYPERGC layers and nodes of the
synergy bottleneck; two-person synergies pair a joint group of one actor with a
joint group of the other. See SEMANTIC_HYPERGRAPH_PLAN.md, section 2.1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

NUM_JOINTS = 25
SPINE_BASE, SPINE_SHOULDER = 0, 20

HAND_L = (6, 7, 21, 22)
HAND_R = (10, 11, 23, 24)
HEAD = (2, 3)
TORSO = (0, 1, 20)
FEET = (14, 15, 18, 19)
BODY = (0, 1, 2, 3, 20, 4, 8, 12, 16)

# A second skeleton closer than this (mean joint distance / torso length) is a
# copy of the first actor, not a second person.
DUPLICATE_DISTANCE = 0.1


@dataclass(frozen=True)
class Synergy:
    name: str
    joints: tuple[int, ...]
    description: str = ""


@dataclass(frozen=True)
class PairSynergy:
    """Joint group ``joints_a`` of one actor with ``joints_b`` of the other."""

    name: str
    joints_a: tuple[int, ...]
    joints_b: tuple[int, ...]
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

PAIR_SYNERGIES = (
    PairSynergy("hand(A)-head(B)", HAND_L + HAND_R, HEAD, "slap, pat on head"),
    PairSynergy("hand(A)-torso(B)", HAND_L + HAND_R, TORSO, "push, punch, hug"),
    PairSynergy("hand(A)-hand(B)", HAND_L + HAND_R, HAND_L + HAND_R, "shake hands, give object"),
    PairSynergy("torso(A)-torso(B)", TORSO, TORSO, "walk towards, walk apart"),
    PairSynergy("foot(A)-body(B)", FEET, BODY, "kick somebody"),
)


def _random_group(rng, size, used):
    while True:
        group = tuple(sorted(rng.sample(range(NUM_JOINTS), size)))
        if group not in used:
            used.add(group)
            return group


def random_bank(seed=0):
    """Ablation A2: random joint groups with the sizes of the semantic bank."""
    rng = random.Random(seed)
    used = set()
    singles = tuple(
        Synergy(f"random_{i:02d}", _random_group(rng, len(s.joints), used))
        for i, s in enumerate(SINGLE_SYNERGIES)
    )
    pairs = tuple(
        PairSynergy(f"random_pair_{i}", tuple(sorted(rng.sample(range(NUM_JOINTS), len(p.joints_a)))),
                    tuple(sorted(rng.sample(range(NUM_JOINTS), len(p.joints_b)))))
        for i, p in enumerate(PAIR_SYNERGIES)
    )
    return singles, pairs


def synergy_bank(kind="semantic", seed=0, use_pair=True):
    """(single synergies, pair synergies) of the requested bank."""
    if kind == "semantic":
        singles, pairs = SINGLE_SYNERGIES, PAIR_SYNERGIES
    elif kind == "random":
        singles, pairs = random_bank(seed)
    else:
        raise ValueError("synergy bank must be 'semantic' or 'random'")
    return singles, (pairs if use_pair else ())


def validate_bank(singles, pairs=()):
    names = [s.name for s in singles] + [p.name for p in pairs]
    if len(set(names)) != len(names):
        raise ValueError("synergy names must be unique")
    groups = [s.joints for s in singles] + [g for p in pairs for g in (p.joints_a, p.joints_b)]
    for group in groups:
        if len(set(group)) != len(group) or not group:
            raise ValueError(f"joint group {group} is empty or repeats a joint")
        if any(not 0 <= j < NUM_JOINTS for j in group):
            raise ValueError(f"joint group {group} has an index outside 0..{NUM_JOINTS - 1}")
    if any(len(s.joints) < 2 for s in singles):
        raise ValueError("every single-person synergy needs at least two joints")


def incidence(groups, num_joints=NUM_JOINTS):
    """Binary incidence matrix [num_joints, E] of joint groups (tuples of joints).

    Rows past NUM_JOINTS (virtual joints) belong to no group.
    """
    matrix = torch.zeros(num_joints, len(groups))
    for e, joints in enumerate(groups):
        matrix[list(joints), e] = 1.0
    return matrix


def pair_validity(x, reference_joint=SPINE_SHOULDER, duplicate_distance=DUPLICATE_DISTANCE):
    """[N] bool: the joint-mode input [N,C,T,V,M] holds two distinct actors.

    Some preprocessed NTU files fill the second actor slot of single-person
    samples with (a copy of) the first actor; such samples do not count.
    """
    N, _, T, V, M = x.shape
    if M < 2:
        return torch.zeros(N, dtype=torch.bool, device=x.device)
    positions = x[:, :3, :, :, :2].permute(0, 4, 2, 3, 1)  # N,2,T,V,3
    if reference_joint is not None:
        # The NTU feeder keeps the reference joint's trajectory and stores the
        # other joints relative to it.
        others = torch.ones(V, 1, device=x.device, dtype=x.dtype)
        others[reference_joint] = 0
        positions = positions + positions[..., reference_joint:reference_joint + 1, :] * others
    frame_valid = x[:, :3, :, :, :2].abs().sum(dim=(1, 3)) > 0  # N,T,2
    both = frame_valid[..., 0] & frame_valid[..., 1]
    frames = both.sum(-1).clamp_min(1)
    torso = (positions[..., SPINE_SHOULDER, :] - positions[..., SPINE_BASE, :]).norm(dim=-1)  # N,2,T
    scale = ((torso * both[:, None]).sum(-1) / frames[:, None]).mean(-1).clamp_min(1e-3)
    gap = (positions[:, 0] - positions[:, 1]).norm(dim=-1).mean(-1)  # N,T
    gap = (gap * both).sum(-1) / frames / scale
    return both.any(-1) & (gap > duplicate_distance)
