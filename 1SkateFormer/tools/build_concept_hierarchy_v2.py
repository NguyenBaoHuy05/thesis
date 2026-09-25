"""Build the HCG-NS ontology while preserving the NTU concept source.

The 74-column ``ntu_rgbd_60_120.csv`` matrix is the authoritative mapping for
the original spatial, temporal and ``interaction_two_person`` concepts. The
v2.1 additions are layered on top as separate, weak clip-level cues. v2.2
removes derived context aliases and keeps the source interaction concept;
unknown means that the action description does not provide reliable skeleton
evidence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPATIAL_CONCEPTS = [
    "head_nod", "head_shake", "head_tilt_down", "head_tilt_up",
    "head_rotate", "head_static", "hand_grasp", "hand_release",
    "hand_wave", "hand_point", "hand_clap", "hand_manipulate",
    "hand_fist", "hand_open", "hand_touch", "hand_lift_to_face",
    "hand_circular", "arm_raise", "arm_lower", "arm_extend",
    "arm_retract", "arm_circular", "arm_swing", "arm_cross", "arm_hold",
    "arm_alternating", "arm_overhead", "arm_lateral", "hip_bend",
    "hip_extend", "hip_sit", "hip_squat", "hip_rotate",
    "hip_static_stand", "hip_static_sit", "hip_sway", "leg_step",
    "leg_kick", "leg_jump", "leg_squat", "leg_lift", "leg_static_stand",
    "leg_static_sit", "leg_alternating", "leg_balance", "foot_step",
    "foot_kick", "foot_jump", "foot_pivot", "foot_static", "foot_tap",
    "foot_slide",
]

TEMPORAL_CONCEPTS = [
    "temporal_direction_motion_upward",
    "temporal_direction_motion_downward",
    "temporal_direction_motion_forward",
    "temporal_direction_motion_backward",
    "temporal_direction_motion_converging",
    "temporal_direction_motion_diverging",
    "temporal_direction_motion_reversible",
    "temporal_sequence_seq_hands_first",
    "temporal_sequence_seq_legs_first",
    "temporal_sequence_seq_body_first",
    "temporal_sequence_seq_simultaneous",
    "temporal_sequence_seq_alternating",
    "temporal_sequence_seq_cascading",
    "temporal_dynamics_speed_slow",
    "temporal_dynamics_speed_fast",
    "temporal_dynamics_speed_accelerating",
    "temporal_dynamics_speed_decelerating",
    "temporal_dynamics_rhythm_regular",
    "temporal_dynamics_rhythm_irregular",
    "temporal_dynamics_duration_brief",
    "temporal_dynamics_duration_sustained",
]

CONTEXT_CONCEPTS = [
    "context_multi_person",
    "context_person_person_interaction",
    "context_target_person",
    "context_target_object",
    "context_single_person",
]

BASE_CONCEPTS = SPATIAL_CONCEPTS + CONTEXT_CONCEPTS[:1] + TEMPORAL_CONCEPTS + CONTEXT_CONCEPTS[1:]

# v2.2 keeps the authoritative source concept instead of creating four
# derived context aliases from the same interaction_two_person column.
V22_BASE_CONCEPTS = (
    SPATIAL_CONCEPTS + ["interaction_two_person"] + TEMPORAL_CONCEPTS
    + ["context_target_object"]
)

# v2.1 additions are still unary and skeleton-observable.  They are appended
# instead of inserted so all v2 concept ids remain stable for comparison.
V21_EXTRA_CONCEPTS = [
    "hand_contact_mouth_region",
    "hand_contact_head_region",
    "hand_contact_chest_region",
    "hand_contact_neck_region",
    "hand_contact_back_region",
    "hand_fine_repetitive_motion",
    "hand_bimanual_hold",
    "hand_repetitive_tapping",
]

V21_NTU120_EXTRA_CONCEPTS = [
    "hand_repetitive_pressing",
    "hand_repetitive_slicing",
    "hand_repetitive_pull_apart",
    "hand_repetitive_opening",
    "hand_contact_face_region",
    "hand_contact_nose_region",
    "arm_cross_chest",
    "hand_fold_motion",
    "hand_rotational_manipulation",
]

NTU120_DROPPED_CONCEPTS = {
    "foot_tap",
    "temporal_sequence_seq_cascading",
    "context_target_object",
}

# v2.3 is the cleaned shared 60+120 ontology.  The first three concepts are
# never positive for any of the 120 actions (context_target_object only ever
# carried "unknown" states, which are ignored by the losses).  foot_static is
# positive for 98/120 actions, so it is near-constant and carries no
# discriminative information.  NTU120-only concepts (e.g. hip_squat,
# hand_repetitive_slicing) are kept so NTU60 and NTU120 share concept ids.
V23_DROPPED_CONCEPTS = NTU120_DROPPED_CONCEPTS | {"foot_static"}

# Versions built on the v2.2 base (source interaction concept, no aliases).
V22_FAMILY = {"v2_2", "v2_2_ntu120", "v2_3"}
# Versions that include the NTU120 action rows and NTU120 extra support.
NTU120_VERSIONS = {"v2_1_ntu120", "v2_2", "v2_2_ntu120", "v2_3"}

CONCEPTS = BASE_CONCEPTS
CONCEPT_ID = {name: index for index, name in enumerate(CONCEPTS)}


def _set(*names: str) -> set[str]:
    return set(names)


# Conservative primitive evidence.  These are semantic supports, not frame
# annotations; every omitted concept remains unknown.
PRIMITIVE_SUPPORT: dict[str, set[str]] = {
    "drink_water": _set("hand_grasp", "hand_lift_to_face", "arm_raise", "arm_hold"),
    "eat_meal": _set("hand_grasp", "hand_lift_to_face", "arm_raise"),
    "brush_teeth": _set(
        "hand_grasp", "hand_lift_to_face", "arm_raise", "arm_alternating",
        "hand_contact_mouth_region",
    ),
    "brush_hair": _set(
        "hand_grasp", "hand_lift_to_face", "arm_raise", "arm_alternating",
        "hand_contact_head_region",
    ),
    "drop": _set("hand_release", "hand_open", "arm_lower"),
    "pick_up": _set("hand_grasp", "hip_bend", "arm_lower"),
    "throw": _set("hand_release", "arm_extend", "arm_swing", "arm_raise"),
    "sit_down": _set("hip_bend", "hip_sit", "hip_squat", "leg_squat"),
    "stand_up": _set("hip_extend", "hip_static_stand", "leg_static_stand"),
    "clapping": _set("hand_clap", "arm_extend"),
    "reading": _set("hand_manipulate", "arm_hold", "hand_bimanual_hold"),
    "writing": _set(
        "hand_manipulate", "arm_hold", "hand_fine_repetitive_motion"
    ),
    "tear_up_paper": _set("hand_grasp", "hand_release", "hand_manipulate", "arm_extend"),
    "put_on_jacket": _set("hand_grasp", "hand_release", "hand_manipulate", "arm_raise", "arm_extend"),
    "take_off_jacket": _set("hand_grasp", "hand_release", "hand_manipulate", "arm_lower", "arm_extend"),
    "put_on_a_shoe": _set("hand_grasp", "hand_manipulate", "hip_bend"),
    "take_off_a_shoe": _set("hand_grasp", "hand_manipulate", "hip_bend"),
    "put_on_glasses": _set("hand_grasp", "hand_lift_to_face", "hand_touch", "arm_raise", "hand_contact_face_region"),
    "take_off_glasses": _set("hand_grasp", "hand_lift_to_face", "hand_touch", "arm_raise", "hand_contact_face_region"),
    "put_on_a_hat_cap": _set("hand_grasp", "hand_lift_to_face", "arm_raise"),
    "take_off_a_hat_cap": _set("hand_grasp", "hand_lift_to_face", "arm_raise"),
    "cheer_up": _set("arm_raise", "arm_overhead", "hand_open"),
    "hand_waving": _set("hand_wave", "arm_swing", "arm_lateral"),
    "kicking_something": _set("leg_kick", "foot_kick", "leg_balance"),
    "reach_into_pocket": _set("hand_grasp", "hand_touch", "hand_manipulate", "arm_retract"),
    "hopping": _set("leg_jump", "foot_jump", "leg_balance", "leg_alternating"),
    "jump_up": _set("leg_jump", "foot_jump", "arm_raise"),
    "phone_call": _set("hand_grasp", "hand_lift_to_face", "hand_touch", "arm_raise"),
    "play_with_phone_tablet": _set("hand_manipulate", "hand_touch", "arm_hold"),
    "type_on_a_keyboard": _set(
        "hand_manipulate", "arm_hold", "arm_alternating",
        "hand_fine_repetitive_motion", "hand_repetitive_tapping",
        "hand_repetitive_pressing",
    ),
    "point_to_something": _set("hand_point", "arm_extend"),
    "taking_a_selfie": _set("hand_grasp", "arm_extend", "arm_hold"),
    "check_time_(from_watch)": _set("hand_lift_to_face", "hand_touch", "arm_hold"),
    "rub_two_hands": _set(
        "hand_touch", "hand_manipulate", "arm_alternating",
        "hand_fine_repetitive_motion",
    ),
    "nod_head_bow": _set("head_nod", "head_tilt_down", "head_tilt_up"),
    "shake_head": _set("head_shake", "head_rotate"),
    "wipe_face": _set("hand_touch", "hand_lift_to_face", "arm_swing"),
    "salute": _set("hand_lift_to_face", "hand_open", "arm_raise"),
    "put_palms_together": _set("hand_touch", "hand_open", "arm_extend"),
    "cross_hands_in_front": _set("arm_cross", "arm_hold", "arm_cross_chest"),
    "sneeze_cough": _set("hand_lift_to_face"),
    "staggering": _set("leg_alternating", "leg_balance", "foot_step", "foot_slide", "hip_sway"),
    "falling_down": _set("hip_bend", "leg_balance"),
    "headache": _set(
        "hand_touch", "hand_lift_to_face", "hand_contact_head_region"
    ),
    "chest_pain": _set(
        "hand_touch", "hand_lift_to_face", "hand_contact_chest_region"
    ),
    "back_pain": _set(
        "hand_touch", "arm_retract", "hand_contact_back_region"
    ),
    "neck_pain": _set(
        "hand_touch", "hand_lift_to_face", "hand_contact_neck_region"
    ),
    "nausea_vomiting": _set("hand_lift_to_face", "hip_bend"),
    "fan_self": _set("hand_open", "arm_swing", "arm_lateral"),
    "punch_slap": _set("hand_fist", "arm_extend", "arm_swing"),
    "kicking": _set("leg_kick", "foot_kick", "leg_balance"),
    "pushing": _set("hand_open", "arm_extend", "arm_hold"),
    "pat_on_back": _set(
        "hand_open", "arm_extend", "arm_hold",
        "hand_contact_back_region", "hand_repetitive_tapping",
    ),
    "point_finger": _set("hand_point", "arm_extend"),
    "hugging": _set("arm_extend", "arm_hold", "arm_cross", "hand_contact_chest_region"),
    "giving_object": _set("hand_grasp", "hand_release", "arm_extend"),
    "touch_pocket": _set("hand_touch", "hand_manipulate", "arm_extend"),
    "shaking_hands": _set("hand_grasp", "arm_extend", "arm_alternating"),
    "walking_towards": _set("leg_step", "foot_step", "leg_alternating"),
    "walking_apart": _set("leg_step", "foot_step", "leg_alternating"),
}

TEMPORAL_SUPPORT: dict[str, set[str]] = {
    "drop": _set("temporal_direction_motion_downward", "temporal_dynamics_duration_brief"),
    "pick_up": _set("temporal_direction_motion_upward"),
    "throw": _set("temporal_direction_motion_forward", "temporal_dynamics_duration_brief"),
    "sit_down": _set("temporal_direction_motion_downward"),
    "stand_up": _set("temporal_direction_motion_upward"),
    "put_on_jacket": _set("temporal_direction_motion_upward"),
    "take_off_jacket": _set("temporal_direction_motion_downward"),
    "put_on_a_shoe": _set("temporal_direction_motion_downward"),
    "take_off_a_shoe": _set("temporal_direction_motion_upward"),
    "put_on_glasses": _set("temporal_direction_motion_upward"),
    "take_off_glasses": _set("temporal_direction_motion_downward"),
    "put_on_a_hat_cap": _set("temporal_direction_motion_upward"),
    "take_off_a_hat_cap": _set("temporal_direction_motion_downward"),
    "hopping": _set("temporal_direction_motion_upward", "temporal_sequence_seq_legs_first"),
    "jump_up": _set("temporal_direction_motion_upward", "temporal_sequence_seq_legs_first"),
    "falling_down": _set("temporal_direction_motion_downward", "temporal_dynamics_duration_brief"),
    "walking_towards": _set(
        "temporal_direction_motion_forward",
        "temporal_direction_motion_converging",
        "temporal_sequence_seq_legs_first",
        "temporal_dynamics_duration_sustained",
    ),
    "walking_apart": _set(
        "temporal_direction_motion_backward",
        "temporal_direction_motion_diverging",
        "temporal_sequence_seq_legs_first",
        "temporal_dynamics_duration_sustained",
    ),
    "hand_waving": _set("temporal_sequence_seq_alternating"),
    "rub_two_hands": _set("temporal_sequence_seq_alternating"),
    "type_on_a_keyboard": _set("temporal_sequence_seq_alternating"),
}

# Conservative extensions for NTU RGB+D 120 classes (action ids 60-119).
# These are generic skeleton-observable primitives, not action-name concepts.
NTU120_EXTRA_SUPPORT: dict[str, set[str]] = {
    "tear_up_paper": _set("hand_repetitive_pull_apart"),
    "wipe_face": _set("hand_contact_face_region"),
    "cheer_up": _set("arm_swing"),
    "taking_a_selfie": _set("hand_lift_to_face"),
    "put_on_headphone": _set(
        "hand_grasp", "hand_lift_to_face", "hand_touch", "arm_raise",
        "hand_contact_head_region",
    ),
    "take_off_headphone": _set(
        "hand_grasp", "hand_lift_to_face", "hand_touch", "arm_lower",
        "hand_contact_head_region",
    ),
    "shoot_at_basket": _set("hand_grasp", "arm_extend", "arm_raise", "arm_hold"),
    "bounce_ball": _set("hand_open", "hand_manipulate", "arm_raise", "arm_lower"),
    "tennis_bat_swing": _set("hand_grasp", "arm_swing", "arm_extend", "hip_rotate"),
    "juggle_table_tennis_ball": _set(
        "hand_manipulate", "arm_alternating", "hand_fine_repetitive_motion"
    ),
    "hush": _set("hand_lift_to_face", "hand_touch", "hand_contact_mouth_region"),
    "flick_hair": _set(
        "hand_lift_to_face", "hand_touch", "arm_swing", "hand_contact_head_region"
    ),
    "thumb_up": _set("hand_fist", "arm_raise"),
    "thumb_down": _set("hand_fist", "arm_lower"),
    "make_OK_sign": _set("hand_manipulate", "hand_open"),
    "make_victory_sign": _set("hand_open", "arm_raise"),
    "staple_book": _set(
        "hand_manipulate", "arm_hold", "hand_fine_repetitive_motion",
        "hand_repetitive_pressing",
    ),
    "counting_money": _set(
        "hand_manipulate", "arm_hold", "arm_alternating", "hand_fine_repetitive_motion"
    ),
    "cutting_nails": _set(
        "hand_manipulate", "hand_lift_to_face", "hand_fine_repetitive_motion",
        "hand_repetitive_slicing",
    ),
    "cutting_paper": _set(
        "hand_manipulate", "arm_hold", "hand_fine_repetitive_motion",
        "hand_repetitive_slicing",
    ),
    "snap_fingers": _set("hand_open", "hand_fine_repetitive_motion"),
    "open_bottle": _set(
        "hand_grasp", "hand_manipulate", "hand_circular",
        "hand_repetitive_opening", "hand_rotational_manipulation",
    ),
    "sniff_smell": _set("hand_lift_to_face", "arm_raise"),
    "squat_down": _set("hip_bend", "leg_squat"),
    "toss_a_coin": _set("hand_release", "arm_extend"),
    "fold_paper": _set(
        "hand_manipulate", "hand_bimanual_hold", "hand_fine_repetitive_motion",
        "hand_fold_motion",
    ),
    "ball_up_paper": _set(
        "hand_grasp", "hand_manipulate", "hand_fine_repetitive_motion",
        "hand_fold_motion",
    ),
    "play_magic_cube": _set(
        "hand_manipulate", "hand_bimanual_hold", "hand_fine_repetitive_motion",
        "hand_rotational_manipulation",
    ),
    "apply_cream_on_face": _set(
        "hand_manipulate", "hand_touch", "hand_lift_to_face",
        "hand_contact_face_region",
    ),
    "apply_cream_on_hand": _set("hand_manipulate", "hand_touch"),
    "put_on_bag": _set("hand_grasp", "arm_hold", "arm_raise"),
    "take_off_bag": _set("hand_grasp", "arm_hold", "arm_lower"),
    "put_object_into_bag": _set("hand_grasp", "hand_release", "hand_manipulate", "arm_lower"),
    "take_object_out_of_bag": _set("hand_grasp", "hand_manipulate", "arm_raise"),
    "open_a_box": _set(
        "hand_grasp", "hand_release", "hand_manipulate", "arm_extend",
        "hand_repetitive_opening", "hand_repetitive_pull_apart",
    ),
    "move_heavy_objects": _set("hand_grasp", "arm_extend", "arm_swing", "hip_bend"),
    "shake_fist": _set("hand_fist", "arm_swing", "arm_alternating"),
    "throw_up_cap_hat": _set("hand_release", "arm_raise", "arm_extend"),
    "capitulate": _set("hand_open", "arm_raise", "arm_overhead", "arm_hold"),
    "cross_arms": _set("arm_cross", "arm_hold", "arm_cross_chest"),
    "arm_circles": _set("arm_circular", "arm_swing"),
    "arm_swings": _set("arm_swing", "arm_alternating"),
    "run_on_the_spot": _set("leg_step", "foot_step", "leg_alternating", "leg_balance"),
    "butt_kicks": _set("leg_kick", "leg_alternating", "leg_balance"),
    "cross_toe_touch": _set("hip_bend", "arm_extend", "leg_balance"),
    "side_kick": _set("leg_kick", "foot_kick", "leg_balance"),
    "yawn": _set("head_tilt_down", "head_tilt_up"),
    "stretch_oneself": _set("arm_raise", "arm_overhead", "arm_extend", "hip_extend"),
    "blow_nose": _set(
        "hand_lift_to_face", "hand_touch", "hand_contact_face_region",
        "hand_contact_nose_region",
    ),
    "hit_with_object": _set("hand_grasp", "arm_extend", "arm_swing"),
    "wield_knife": _set("hand_grasp", "hand_manipulate", "arm_swing"),
    "knock_over": _set("hand_open", "arm_extend", "arm_swing"),
    "grab_stuff": _set("hand_grasp", "arm_extend"),
    "shoot_with_gun": _set("hand_grasp", "arm_extend", "arm_hold"),
    "step_on_foot": _set("leg_step", "foot_step", "leg_balance"),
    "high-five": _set("hand_open", "arm_extend"),
    "cheers_and_drink": _set("hand_grasp", "hand_lift_to_face", "arm_raise"),
    "carry_object": _set("hand_grasp", "arm_hold"),
    "take_a_photo": _set(
        "hand_grasp", "arm_extend", "arm_hold", "hand_bimanual_hold"
    ),
    "follow": _set("leg_step", "foot_step", "leg_alternating"),
    "whisper": _set("hand_lift_to_face", "hand_touch"),
    "exchange_things": _set("hand_grasp", "hand_release", "arm_extend"),
    "support_somebody": _set("arm_extend", "arm_hold"),
    "rock-paper-scissors": _set("hand_open", "hand_fist", "hand_manipulate", "arm_extend"),
}

NTU120_EXTRA_TEMPORAL_SUPPORT: dict[str, set[str]] = {
    "bounce_ball": _set("temporal_direction_motion_reversible", "temporal_sequence_seq_alternating"),
    "tennis_bat_swing": _set("temporal_direction_motion_forward"),
    "juggle_table_tennis_ball": _set("temporal_sequence_seq_alternating"),
    "squat_down": _set("temporal_direction_motion_downward"),
    "toss_a_coin": _set("temporal_direction_motion_forward", "temporal_dynamics_duration_brief"),
    "throw_up_cap_hat": _set("temporal_direction_motion_upward"),
    "run_on_the_spot": _set("temporal_dynamics_speed_fast", "temporal_dynamics_duration_sustained"),
    "butt_kicks": _set("temporal_sequence_seq_alternating"),
    "side_kick": _set("temporal_direction_motion_forward"),
    "stretch_oneself": _set("temporal_dynamics_duration_sustained"),
    "arm_swings": _set("temporal_sequence_seq_alternating"),
    "shake_fist": _set("temporal_sequence_seq_alternating"),
    "rock-paper-scissors": _set("temporal_sequence_seq_alternating"),
    "exchange_things": _set("temporal_sequence_seq_alternating"),
}

OBJECT_UNKNOWN_ACTIONS = {
    "drop", "pick_up", "throw", "tear_up_paper", "put_on_jacket",
    "take_off_jacket", "put_on_a_shoe", "take_off_a_shoe", "put_on_glasses",
    "take_off_glasses", "put_on_a_hat_cap", "take_off_a_hat_cap",
    "kicking_something", "reach_into_pocket", "phone_call",
    "play_with_phone_tablet", "type_on_a_keyboard", "point_to_something",
    "taking_a_selfie", "check_time_(from_watch)", "giving_object", "touch_pocket",
    "put_on_headphone", "take_off_headphone", "shoot_at_basket", "bounce_ball",
    "tennis_bat_swing", "juggle_table_tennis_ball", "staple_book", "counting_money",
    "cutting_nails", "cutting_paper", "open_bottle", "toss_a_coin", "fold_paper",
    "ball_up_paper", "play_magic_cube", "apply_cream_on_face", "apply_cream_on_hand",
    "put_on_bag", "take_off_bag", "put_object_into_bag", "take_object_out_of_bag",
    "open_a_box", "move_heavy_objects", "throw_up_cap_hat", "hit_with_object",
    "wield_knife", "grab_stuff", "shoot_with_gun", "carry_object", "take_a_photo",
}

INTERACTION_ACTIONS = {
    "punch_slap", "kicking", "pushing", "pat_on_back", "point_finger", "hugging",
    "giving_object", "touch_pocket", "shaking_hands", "walking_towards", "walking_apart",
    "hit_with_object", "step_on_foot", "high-five", "cheers_and_drink", "follow",
    "whisper", "exchange_things", "support_somebody", "rock-paper-scissors",
}


def _parent_and_level(name: str) -> tuple[str, int, str, str, str]:
    if name in CONTEXT_CONCEPTS or name == "interaction_two_person":
        observable = "no" if name == "context_target_object" else "yes"
        return "context_participant", 0, "context", "context", observable
    if name in TEMPORAL_CONCEPTS:
        if name.startswith("temporal_direction"):
            parent = "temporal_direction"
        elif name.startswith("temporal_sequence"):
            parent = "temporal_sequence"
        else:
            parent = "temporal_dynamics"
        return parent, 3, "temporal_attribute", "temporal", "partial"
    if name.startswith("head_"):
        parent = "region_head"
    elif name.startswith("hand_") or name.startswith("arm_"):
        parent = "region_upper_limb"
    elif name.startswith("hip_"):
        parent = "region_hip_torso"
    else:
        parent = "region_lower_limb"
    return parent, 2, "spatial_primitive", "spatial", "yes"


def _confidence(name: str) -> tuple[str, str]:
    if name == "context_target_object":
        return "0.20", "unknown_only"
    if name == "interaction_two_person":
        return "0.90", "weak_positive"
    if name in V21_EXTRA_CONCEPTS or name in V21_NTU120_EXTRA_CONCEPTS:
        return "0.60", "weak_positive"
    if name.startswith("temporal_"):
        return "0.60", "weak_positive"
    if name.startswith("context_"):
        return "0.90", "weak_positive"
    return "0.80", "weak_positive"


def _source(name: str, dataset: str = "NTU60") -> str:
    if name in V21_EXTRA_CONCEPTS or name in V21_NTU120_EXTRA_CONCEPTS:
        return f"curated_{dataset}_skeleton_semantics"
    return f"curated_{dataset}_action_semantics"


def _legacy_source(name: str, dataset: str) -> str:
    """Return provenance for concepts copied from ntu_rgbd_60_120.csv."""
    if name in V21_EXTRA_CONCEPTS or name in V21_NTU120_EXTRA_CONCEPTS:
        return _source(name, dataset)
    return "ntu_rgbd_60_120.csv:legacy_concept_mapping"


def _notes(name: str) -> str:
    if name in V21_EXTRA_CONCEPTS or name in V21_NTU120_EXTRA_CONCEPTS:
        return "weak clip-level support; skeleton-observable unary cue; not frame annotation"
    return "weak clip-level support; not frame annotation"


def _action_mapping(
    action_name: str, include_ntu120: bool = False,
) -> tuple[set[str], set[str], set[str]]:
    positives = set(PRIMITIVE_SUPPORT.get(action_name, set()))
    positives.update(TEMPORAL_SUPPORT.get(action_name, set()))
    if include_ntu120:
        positives.update(NTU120_EXTRA_SUPPORT.get(action_name, set()))
        positives.update(NTU120_EXTRA_TEMPORAL_SUPPORT.get(action_name, set()))
    negatives = set()
    if action_name in INTERACTION_ACTIONS:
        positives.update({
            "context_multi_person",
            "context_person_person_interaction",
            "context_target_person",
        })
        negatives.add("context_single_person")
    else:
        positives.add("context_single_person")
    unknown = set()
    if action_name in OBJECT_UNKNOWN_ACTIONS:
        unknown.add("context_target_object")
    return positives, negatives, unknown


def build(action_path: Path, output_dir: Path, version: str = "v2") -> None:
    with action_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_reader = csv.DictReader(handle)
        source_rows = list(source_reader)
        source_fields = list(source_reader.fieldnames or [])
    if not source_rows or not source_fields:
        raise ValueError("source action CSV is empty")
    # v2.2 is the shared 60+120 ontology: one matrix contains all 120
    # actions, while the NTU60 loader consumes its first 60 action rows.
    action_count = 120 if version in NTU120_VERSIONS else 60
    source_rows = source_rows[:action_count]
    if len(source_rows) != action_count:
        raise ValueError(f"HCG-NS mapping requires exactly {action_count} action rows")
    if source_fields[0] != "action_class":
        raise ValueError("source action CSV must start with an action_class column")
    action_names = [row["action_class"] for row in source_rows]
    if len(set(action_names)) != len(action_names):
        raise ValueError("source action CSV contains duplicate action names")
    required_source_concepts = set(SPATIAL_CONCEPTS + TEMPORAL_CONCEPTS)
    required_source_concepts.add("interaction_two_person")
    missing_source_concepts = sorted(required_source_concepts - set(source_fields))
    if missing_source_concepts:
        raise ValueError(
            "source action CSV is missing authoritative concept columns: "
            + ", ".join(missing_source_concepts)
        )
    source_by_action = {row["action_class"]: row for row in source_rows}

    if version not in {"v2", "v2_1", "v2_1_ntu120", "v2_2", "v2_2_ntu120", "v2_3"}:
        raise ValueError(f"unsupported ontology version: {version}")
    base_concepts = V22_BASE_CONCEPTS if version in V22_FAMILY else BASE_CONCEPTS
    concepts = list(base_concepts)
    if version == "v2_1":
        concepts.extend(V21_EXTRA_CONCEPTS)
    elif version == "v2_2":
        concepts.extend(V21_EXTRA_CONCEPTS)
        concepts.extend(V21_NTU120_EXTRA_CONCEPTS)
    elif version in {"v2_1_ntu120", "v2_2_ntu120"}:
        concepts.extend(V21_EXTRA_CONCEPTS)
        concepts.extend(V21_NTU120_EXTRA_CONCEPTS)
        concepts = [name for name in concepts if name not in NTU120_DROPPED_CONCEPTS]
    elif version == "v2_3":
        concepts.extend(V21_EXTRA_CONCEPTS)
        concepts.extend(V21_NTU120_EXTRA_CONCEPTS)
        concepts = [name for name in concepts if name not in V23_DROPPED_CONCEPTS]
    concept_id = {name: index for index, name in enumerate(concepts)}
    suffix = version
    dataset = (
        "NTU60_120" if version in {"v2_2", "v2_3"}
        else ("NTU120" if version.endswith("_ntu120") else "NTU60")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"ntu_concept_metadata_{suffix}.csv"
    metadata_fields = [
        "concept_id", "concept_name", "scope", "group", "parent_name", "level",
        "node_type", "label_status", "observable_from_skeleton", "confidence", "source",
    ]
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields)
        writer.writeheader()
        for concept_index, concept_name in enumerate(concepts):
            parent, level, node_type, scope, observable = _parent_and_level(concept_name)
            confidence, label_status = _confidence(concept_name)
            group = parent.removeprefix("region_").removeprefix("context_")
            writer.writerow({
                "concept_id": concept_index,
                "concept_name": concept_name,
                "scope": scope,
                "group": group,
                "parent_name": parent,
                "level": level,
                "node_type": node_type,
                "label_status": label_status,
                "observable_from_skeleton": observable,
                "confidence": confidence,
                "source": _source(concept_name, dataset),
            })

    matrix_path = output_dir / f"ntu_action_concept_{suffix}.csv"
    label_path = output_dir / f"ntu_action_concept_{suffix}_labels.csv"
    matrix_header = ["action_class", *concepts]
    labels_fields = [
        "action_id", "action_class", "concept_id", "concept_name", "state",
        "confidence", "source", "notes",
    ]
    with matrix_path.open("w", encoding="utf-8", newline="") as matrix_handle, \
            label_path.open("w", encoding="utf-8", newline="") as label_handle:
        matrix_writer = csv.writer(matrix_handle)
        label_writer = csv.DictWriter(label_handle, fieldnames=labels_fields)
        matrix_writer.writerow(matrix_header)
        label_writer.writeheader()
        for action_id, action_name in enumerate(action_names):
            source_row = source_by_action[action_name]
            interaction = source_row["interaction_two_person"].strip() == "1"
            positives = set()
            for concept_name in base_concepts:
                if concept_name == "interaction_two_person":
                    is_positive = interaction
                elif concept_name == "context_multi_person":
                    is_positive = interaction
                elif concept_name in {
                    "context_person_person_interaction", "context_target_person",
                }:
                    is_positive = interaction
                elif concept_name == "context_single_person":
                    is_positive = not interaction
                elif concept_name == "context_target_object":
                    is_positive = False
                else:
                    is_positive = source_row[concept_name].strip() == "1"
                if is_positive:
                    positives.add(concept_name)

            mapped_positives, _, unknown = _action_mapping(
                action_name,
                include_ntu120=version in NTU120_VERSIONS,
            )
            extra_concepts = set(concepts) - set(base_concepts)
            positives.update(mapped_positives & extra_concepts)
            # Source columns of dropped concepts (e.g. foot_static) must not
            # leak into the label file.
            positives &= set(concepts)
            negatives = (
                {"context_single_person"} if interaction and version not in V22_FAMILY
                else set()
            )
            matrix_writer.writerow([
                action_name,
                *[1 if concept in positives else 0 for concept in concepts],
            ])
            for concept_name in sorted(positives):
                confidence, _ = _confidence(concept_name)
                label_writer.writerow({
                    "action_id": action_id,
                    "action_class": action_name,
                    "concept_id": concept_id[concept_name],
                    "concept_name": concept_name,
                    "state": "positive",
                    "confidence": confidence,
                    "source": _legacy_source(concept_name, dataset),
                    "notes": _notes(concept_name),
                })
            for concept_name in sorted(unknown & set(concepts)):
                label_writer.writerow({
                    "action_id": action_id,
                    "action_class": action_name,
                    "concept_id": concept_id[concept_name],
                    "concept_name": concept_name,
                    "state": "unknown",
                    "confidence": "0.20",
                    "source": "modality_limit_skeleton_only",
                    "notes": "action semantics suggest an object, but skeleton cannot verify it",
                })
            for concept_name in sorted(negatives):
                label_writer.writerow({
                    "action_id": action_id,
                    "action_class": action_name,
                    "concept_id": concept_id[concept_name],
                    "concept_name": concept_name,
                    "state": "negative",
                    "confidence": "0.90",
                    "source": "ntu_rgbd_60_120.csv:interaction_two_person",
                    "notes": "source interaction label is incompatible with a single-person context",
                })

    hierarchy_path = output_dir / f"ntu_concept_hierarchy_{suffix}.csv"
    hierarchy_fields = [
        "parent_name", "parent_level", "child_id", "child_name", "child_level",
        "relation", "confidence", "source",
    ]
    with hierarchy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hierarchy_fields)
        writer.writeheader()
        for child_id, child_name in enumerate(concepts):
            parent, level, _, _, _ = _parent_and_level(child_name)
            confidence, _ = _confidence(child_name)
            writer.writerow({
                "parent_name": parent,
                "parent_level": max(level - 1, 0),
                "child_id": child_id,
                "child_name": child_name,
                "child_level": level,
                "relation": "is_a",
                "confidence": confidence,
                "source": f"curated_{dataset}_ontology_{'v2_3' if version == 'v2_3' else 'v2_2'}"
                if version in V22_FAMILY
                else (
                    f"curated_{dataset}_ontology_v2_1"
                    if version in {"v2_1", "v2_1_ntu120"}
                    else "curated_NTU60_ontology"
                ),
            })
    print(f"Wrote {metadata_path}")
    print(f"Wrote {matrix_path}")
    print(f"Wrote {label_path}")
    print(f"Wrote {hierarchy_path}")
    print(f"{len(action_names)} actions, {len(concepts)} concepts, positive/negative/unknown mapping")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="concepts/ntu_rgbd_60_120.csv")
    parser.add_argument("--output-dir", default="concepts")
    parser.add_argument(
        "--version",
        choices=("v2", "v2_1", "v2_1_ntu120", "v2_2", "v2_2_ntu120", "v2_3"),
        default="v2",
    )
    args = parser.parse_args()
    build(Path(args.actions), Path(args.output_dir), version=args.version)


if __name__ == "__main__":
    main()
