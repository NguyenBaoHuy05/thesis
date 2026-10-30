"""Concept bank of pipeline v4: every column of the concepts CSV becomes a concept.

* ``geo`` concept: grounded in ONE per-frame signal of sip/geo_torch.py, a direction and
  a temporal quantifier. Its truth value is
      z_c = sigmoid((q_c - tau_c) / T_c),
      u_t = dir_c * (s_t - mu_s) / sd_s          (s = signal, standardised on TRAIN-FIT)
      q_c = soft-max_t u_t   ("exists")  or  mean_t u_t  ("mostly")
  with ONE learned threshold tau_c shared by every class. Back in signal units the
  literal reads "exists t: s_t > mu + sd * tau" (dir +1) or "< mu - sd * tau" (dir -1).
* ``latent`` concept: not measurable from 25 joints (hand shape, rhythm, sequencing, ...);
  learned from HyperGCN layer-6 features (sip/ccp.py), supervised by the CSV only.
* extra concepts (name ``x_*``, not in the CSV): geo concepts without CSV supervision,
  available to the logic program for knowledge DISCOVERY.

``joints`` = joints the counterfactual generator may move to switch the concept off
(person slot of the main actor; joint 20 is never edited because every other joint is
stored relative to it). Empty = not editable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sip.concept_map import GCCR_CONCEPT_CSV, load_concept_csv
from sip.geo_torch import SIGNAL_INDEX

__all__ = ["ConceptSpec", "ConceptBank", "GEO_MAP", "EXTRA_CONCEPTS", "build_bank"]

HEAD_J = [2, 3]
TORSO_J = [0, 1]
ARM_J = [4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24]
LEG_J = [12, 13, 14, 15, 16, 17, 18, 19]
BODY_J = TORSO_J + HEAD_J + ARM_J + LEG_J

# concept -> (signal, direction, quantifier, editable joints)
GEO_MAP: dict[str, tuple[str, int, str, list[int]]] = {
    "head_nod": ("head_pitch_speed", +1, "exists", HEAD_J),
    "head_shake": ("head_lat_speed", +1, "exists", HEAD_J),
    "head_tilt_down": ("head_pitch", +1, "exists", HEAD_J),
    "head_tilt_up": ("head_pitch", -1, "exists", HEAD_J),
    "head_static": ("head_speed", -1, "mostly", HEAD_J),
    "hand_wave": ("raised_hand_speed", +1, "exists", ARM_J),
    "hand_point": ("hand_reach_fwd", +1, "exists", ARM_J),
    "hand_clap": ("hands_dist", -1, "exists", ARM_J),
    "hand_touch": ("hand_body", -1, "exists", ARM_J),
    "hand_lift_to_face": ("hand_head", -1, "exists", ARM_J),
    "arm_raise": ("hand_above_shoulder", +1, "exists", ARM_J),
    "arm_lower": ("hand_vy_min", -1, "exists", ARM_J),
    "arm_extend": ("elbow_max", +1, "exists", ARM_J),
    "arm_retract": ("hand_chest_v", -1, "exists", ARM_J),
    "arm_swing": ("wrist_speed", +1, "exists", ARM_J),
    "arm_cross": ("arms_crossed", +1, "exists", ARM_J),
    "arm_hold": ("elbow_min", -1, "mostly", ARM_J),
    "arm_alternating": ("arm_sym", -1, "mostly", ARM_J),
    "arm_overhead": ("hand_above_head", +1, "exists", ARM_J),
    "arm_lateral": ("hand_lateral", +1, "exists", ARM_J),
    "hip_bend": ("torso_pitch", +1, "exists", TORSO_J),
    "hip_extend": ("hip_vy", +1, "exists", TORSO_J + LEG_J),
    "hip_sit": ("hip_height", -1, "exists", TORSO_J + LEG_J),
    "hip_squat": ("knee_min", -1, "exists", LEG_J),
    "hip_rotate": ("shoulder_yaw_speed", +1, "exists", [4, 8]),
    "hip_static_stand": ("hip_height", +1, "mostly", TORSO_J + LEG_J),
    "hip_static_sit": ("hip_height", -1, "mostly", TORSO_J + LEG_J),
    "hip_sway": ("hip_lat_speed", +1, "exists", TORSO_J),
    "leg_step": ("ankle_speed", +1, "exists", LEG_J),
    "leg_kick": ("foot_height_diff", +1, "exists", LEG_J),
    "leg_jump": ("feet_lift", +1, "exists", LEG_J),
    "leg_squat": ("knee_min", -1, "exists", LEG_J),
    "leg_lift": ("foot_height_diff", +1, "exists", LEG_J),
    "leg_static_stand": ("hip_height", +1, "mostly", TORSO_J + LEG_J),
    "leg_static_sit": ("hip_height", -1, "mostly", TORSO_J + LEG_J),
    "leg_alternating": ("leg_alt_speed", +1, "mostly", LEG_J),
    "leg_balance": ("foot_height_diff", +1, "mostly", LEG_J),
    "foot_step": ("ankle_speed", +1, "exists", LEG_J),
    "foot_kick": ("foot_height_diff", +1, "exists", LEG_J),
    "foot_jump": ("feet_lift", +1, "exists", LEG_J),
    "foot_static": ("ankle_speed", -1, "mostly", LEG_J),
    "foot_tap": ("foot_vy", +1, "exists", LEG_J),
    "interaction_two_person": ("second_present", +1, "mostly", []),
    "temporal_direction_motion_upward": ("hand_vy_max", +1, "exists", ARM_J),
    "temporal_direction_motion_downward": ("hand_vy_min", -1, "exists", ARM_J),
    "temporal_direction_motion_forward": ("hand_fwd_vmax", +1, "exists", ARM_J),
    "temporal_direction_motion_backward": ("hand_fwd_vmin", -1, "exists", ARM_J),
    "temporal_direction_motion_converging": ("hands_dist_v", -1, "exists", ARM_J),
    "temporal_direction_motion_diverging": ("hands_dist_v", +1, "exists", ARM_J),
    "temporal_dynamics_speed_slow": ("body_speed", -1, "mostly", BODY_J),
    "temporal_dynamics_speed_fast": ("body_speed", +1, "mostly", BODY_J),
}

# extra geo concepts (no CSV column): discovery vocabulary for two-person actions / gait
EXTRA_CONCEPTS: dict[str, tuple[str, int, str, list[int]]] = {
    "x_contact_partner": ("contact_dist", -1, "exists", ARM_J),
    "x_close_partner": ("person_dist", -1, "mostly", []),
    "x_approach_partner": ("person_dist_v", -1, "exists", []),
    "x_leave_partner": ("person_dist_v", +1, "exists", []),
    "x_walk_forward": ("body_fwd_v", +1, "mostly", []),
    "x_walk_backward": ("body_fwd_v", -1, "mostly", []),
}


@dataclass
class ConceptSpec:
    name: str
    kind: str                      # "geo" | "latent"
    signal: str | None = None
    direction: int = 1
    quant: str = "exists"          # "exists" | "mostly"
    joints: list[int] = field(default_factory=list)
    in_csv: bool = True


@dataclass
class ConceptBank:
    specs: list[ConceptSpec]
    class_names: list[str]
    csv_matrix: np.ndarray         # (K, n_concepts) 0/1, zero rows for extras
    supervised: np.ndarray         # (n_concepts,) bool, True for CSV columns

    @property
    def names(self):
        return [s.name for s in self.specs]

    @property
    def geo_index(self):
        return np.array([i for i, s in enumerate(self.specs) if s.kind == "geo"], dtype=np.int64)

    @property
    def latent_index(self):
        return np.array([i for i, s in enumerate(self.specs) if s.kind == "latent"], dtype=np.int64)

    def geo_tables(self):
        """Arrays over the geo concepts (in geo_index order): signal index, direction,
        mostly flag."""
        gi = self.geo_index
        sig = np.array([SIGNAL_INDEX[self.specs[i].signal] for i in gi], dtype=np.int64)
        dirs = np.array([self.specs[i].direction for i in gi], dtype=np.float32)
        mostly = np.array([self.specs[i].quant == "mostly" for i in gi], dtype=bool)
        return sig, dirs, mostly

    def summary(self):
        n_geo = len(self.geo_index)
        n_lat = len(self.latent_index)
        n_x = sum(1 for s in self.specs if not s.in_csv)
        return "concepts={} (geo={} incl. {} extra, latent={})".format(len(self.specs), n_geo, n_x, n_lat)


def build_bank(csv_path: str = GCCR_CONCEPT_CSV, num_class: int = 60, extras: bool = True) -> ConceptBank:
    action_names, concept_names, mat = load_concept_csv(csv_path)
    unknown = sorted(set(GEO_MAP) - set(concept_names))
    assert not unknown, "GEO_MAP names not in the CSV: {}".format(unknown)
    specs = []
    for c in concept_names:
        if c in GEO_MAP:
            sig, d, q, js = GEO_MAP[c]
            specs.append(ConceptSpec(c, "geo", sig, d, q, list(js)))
        else:
            specs.append(ConceptSpec(c, "latent"))
    if extras:
        for c, (sig, d, q, js) in EXTRA_CONCEPTS.items():
            specs.append(ConceptSpec(c, "geo", sig, d, q, list(js), in_csv=False))
    for s in specs:
        assert s.kind != "geo" or s.signal in SIGNAL_INDEX, s
        assert 20 not in s.joints, s
    K, C = num_class, len(specs)
    csv_m = np.zeros((K, C), dtype=np.float32)
    csv_m[:, :len(concept_names)] = (mat[:num_class] > 0.5).astype(np.float32)
    supervised = np.array([s.in_csv for s in specs], dtype=bool)
    return ConceptBank(specs, list(action_names[:num_class]), csv_m, supervised)
