"""Geometry-derived supervision for symbolic motion primitives.

The targets in this module are deliberately soft.  They are evidence from the
input skeleton, not action-label replacements, so callers can use the returned
confidence to mask or weight the auxiliary loss.
"""

from __future__ import annotations

import torch

from model.concept_schema import BODY_PART_JOINTS
from model.large_symbolic_reasoner import PRIMITIVE_NAMES, RELATION_NAMES


def _part_mean(skeleton, joints):
    return skeleton[..., list(joints), :, :].mean(dim=-3)


def geometry_pseudo_labels(skeleton, eps=1e-6):
    """Return soft primitive/relation targets and confidence from ``[N,C,T,V,M]``.

    Coordinates are centered and scaled by the torso before motion statistics
    are computed.  This removes the dominant translation and body-size effects.
    """
    if skeleton.ndim != 5:
        raise ValueError("skeleton must have shape [N,C,T,V,M]")
    if skeleton.shape[3] != 25:
        raise ValueError("geometry pseudo-labels require NTU's 25 joints")
    if skeleton.shape[2] < 2:
        raise ValueError("geometry pseudo-labels require at least two frames")

    # [N,T,V,M,C], with person and time kept explicit.
    pose = skeleton.permute(0, 2, 3, 4, 1).float()
    torso = _part_mean(pose, BODY_PART_JOINTS["torso"])
    scale = (pose[..., 3, :] - torso).norm(dim=-1).mean(dim=(1, 2, 3))
    centered = (pose - torso.unsqueeze(-3)) / scale.clamp_min(eps).view(-1, 1, 1, 1, 1)
    velocity = centered[:, 1:] - centered[:, :-1]
    speed = velocity.norm(dim=-1).mean(dim=(1, 2, 3))

    parts = {
        "upper_body_motion": ("left_arm", "right_arm", "left_hand", "right_hand"),
        "lower_body_motion": ("left_leg", "right_leg", "left_foot", "right_foot"),
        "hand_motion": ("left_hand", "right_hand"),
        "foot_motion": ("left_foot", "right_foot"),
        "head_motion": ("head",),
        "torso_motion": ("torso",),
    }
    motion = {}
    for name, part_names in parts.items():
        joints = tuple(joint for part in part_names for joint in BODY_PART_JOINTS[part])
        motion[name] = velocity[..., list(joints), :].norm(dim=-1).mean(dim=(1, 2, 3))
    vertical = velocity[..., 1].mean(dim=(1, 2, 3))
    forward = velocity[..., 2].mean(dim=(1, 2, 3))
    turning = velocity[..., 0].mean(dim=(1, 2, 3))
    hand_distance = (
        centered[..., BODY_PART_JOINTS["left_hand"], :, :].mean(dim=-3)
        - centered[..., BODY_PART_JOINTS["right_hand"], :, :].mean(dim=-3)
    ).norm(dim=-1).mean(dim=(1, 2))
    person_distance = (
        torso[:, :, 0] - torso[:, :, 1]
    ).norm(dim=-1).mean(dim=1) if pose.shape[3] >= 2 else torch.zeros_like(speed)

    def positive(value):
        return torch.sigmoid(8.0 * (value - 0.05))

    values = {
        name: positive(value) for name, value in motion.items()
    }
    values.update({
        "move_up": positive(vertical),
        "move_down": positive(-vertical),
        "move_forward": positive(forward),
        "move_backward": positive(-forward),
        "rotate": positive(turning.abs()),
        "extend": positive(motion["upper_body_motion"] + vertical.abs()),
        "flex": positive(motion["torso_motion"] + vertical.abs()),
        "repeat": positive(speed),
        "person_interaction": positive(-person_distance + 0.8),
        "body_contact": positive(-hand_distance + 0.35),
    })
    primitive = torch.stack([values[name] for name in PRIMITIVE_NAMES], dim=1)
    relation = torch.stack([
        positive(-person_distance + 0.8),
        positive(person_distance - 1.2),
        positive(forward),
        positive(-forward),
        positive(speed),
        positive(speed),
    ], dim=1)
    confidence = torch.sigmoid(8.0 * (speed - 0.03)).unsqueeze(1).expand_as(primitive)
    relation_confidence = confidence[:, :len(RELATION_NAMES)]
    return {
        "primitive_targets": primitive.clamp(0.0, 1.0),
        "relation_targets": relation.clamp(0.0, 1.0),
        "primitive_confidence": confidence,
        "relation_confidence": relation_confidence,
    }
