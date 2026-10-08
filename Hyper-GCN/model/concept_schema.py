"""Validation and anatomical mapping for the NTU action-concept tables."""

import csv
from pathlib import Path


# Zero-based NTU label order, using the spellings in the v2.4 concept table.
# Keep this contract independent of the CSV so swapped action rows fail early.
NTU_ACTION_NAMES = (
    "drink_water", "eat_meal", "brush_teeth", "brush_hair", "drop", "pick_up",
    "throw", "sit_down", "stand_up", "clapping", "reading", "writing",
    "tear_up_paper", "put_on_jacket", "take_off_jacket", "put_on_a_shoe",
    "take_off_a_shoe", "put_on_glasses", "take_off_glasses", "put_on_a_hat_cap",
    "take_off_a_hat_cap", "cheer_up", "hand_waving", "kicking_something",
    "reach_into_pocket", "hopping", "jump_up", "phone_call",
    "play_with_phone_tablet", "type_on_a_keyboard", "point_to_something",
    "taking_a_selfie", "check_time_(from_watch)", "rub_two_hands",
    "nod_head_bow", "shake_head", "wipe_face", "salute", "put_palms_together",
    "cross_hands_in_front", "sneeze_cough", "staggering", "falling_down",
    "headache", "chest_pain", "back_pain", "neck_pain", "nausea_vomiting",
    "fan_self", "punch_slap", "kicking", "pushing", "pat_on_back",
    "point_finger", "hugging", "giving_object", "touch_pocket", "shaking_hands",
    "walking_towards", "walking_apart", "put_on_headphone", "take_off_headphone",
    "shoot_at_basket", "bounce_ball", "tennis_bat_swing",
    "juggle_table_tennis_ball", "hush", "flick_hair", "thumb_up", "thumb_down",
    "make_OK_sign", "make_victory_sign", "staple_book", "counting_money",
    "cutting_nails", "cutting_paper", "snap_fingers", "open_bottle", "sniff_smell",
    "squat_down", "toss_a_coin", "fold_paper", "ball_up_paper", "play_magic_cube",
    "apply_cream_on_face", "apply_cream_on_hand", "put_on_bag", "take_off_bag",
    "put_object_into_bag", "take_object_out_of_bag", "open_a_box",
    "move_heavy_objects", "shake_fist", "throw_up_cap_hat", "capitulate",
    "cross_arms", "arm_circles", "arm_swings", "run_on_the_spot", "butt_kicks",
    "cross_toe_touch", "side_kick", "yawn", "stretch_oneself", "blow_nose",
    "hit_with_object", "wield_knife", "knock_over", "grab_stuff", "shoot_with_gun",
    "step_on_foot", "high-five", "cheers_and_drink", "carry_object",
    "take_a_photo", "follow", "whisper", "exchange_things", "support_somebody",
    "rock-paper-scissors",
)

BODY_PART_JOINTS = {
    "head": (3,), "torso": (0, 1, 2, 20),
    "left_arm": (4, 5, 6), "right_arm": (8, 9, 10),
    "left_hand": (7, 21, 22), "right_hand": (11, 23, 24),
    "left_leg": (12, 13, 14), "right_leg": (16, 17, 18),
    "left_foot": (15,), "right_foot": (19,), "full_body": tuple(range(25)),
}
PART_NAMES = tuple(BODY_PART_JOINTS)

CONCEPT_PART_OVERRIDES = {
    "hand_lift_to_face": ("left_hand", "right_hand", "head"),
    "hand_contact_nose_region": ("left_hand", "right_hand", "head"),
    "lower_limb_kick_targeted": (
        "left_leg", "right_leg", "left_foot", "right_foot",
    ),
    "upper_limb_hands_crossed_front": (
        "left_arm", "right_arm", "left_hand", "right_hand", "torso",
    ),
}


def concept_parts(name, scope, group):
    """Return explicit parts; do not silently map unknown spatial concepts globally."""
    name, scope, group = (value.strip().lower() for value in (name, scope, group))
    if scope in {"temporal", "context"}:
        return ("full_body",)
    if scope != "spatial":
        raise ValueError(f"unknown scope {scope!r} for concept {name!r}")
    if name in CONCEPT_PART_OVERRIDES:
        return CONCEPT_PART_OVERRIDES[name]
    if name.startswith("head_"):
        return ("head",)
    if name.startswith("hand_"):
        parts = ["left_hand", "right_hand"]
        if any(token in name for token in (
            "head_region", "face_region", "mouth_region", "nose_region",
        )):
            parts.append("head")
        if any(token in name for token in ("chest_region", "neck_region", "back_region")):
            parts.append("torso")
        return tuple(parts)
    if name.startswith("arm_"):
        parts = ["left_arm", "right_arm"]
        if "cross" in name or "chest" in name:
            parts.append("torso")
        return tuple(parts)
    if name.startswith("upper_limb_"):
        parts = ["left_arm", "right_arm", "left_hand", "right_hand"]
        if "head_region" in name:
            parts.append("head")
        return tuple(parts)
    if name.startswith("hip_"):
        return ("torso", "left_leg", "right_leg")
    if name.startswith("leg_"):
        return ("left_leg", "right_leg")
    if name.startswith("foot_"):
        return ("left_foot", "right_foot")
    raise ValueError(f"no anatomical mapping for spatial concept {name!r} ({group})")


def load_concept_tables(concept_csv, metadata_csv, num_classes, num_concepts):
    """Read and validate all rows, including unused NTU120 rows in an NTU60 run."""
    if num_classes not in (60, 120):
        raise ValueError("NTU concept tables require num_classes=60 or 120")
    if num_concepts < 1:
        raise ValueError("num_concepts must be positive")
    with Path(concept_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or not rows[0] or rows[0][0].strip() != "action_class":
        raise ValueError("concept CSV must start with action_class")
    names = [value.strip() for value in rows[0][1:]]
    if len(names) != num_concepts or any(not name for name in names):
        raise ValueError(f"concept CSV must have {num_concepts} nonempty concept columns")
    if len(set(names)) != len(names):
        raise ValueError("concept CSV contains duplicate concept names")
    action_rows = rows[1:]
    if len(action_rows) not in (60, 120) or len(action_rows) < num_classes:
        raise ValueError("concept CSV must contain 60 or 120 action rows matching num_classes")
    values = []
    for index, row in enumerate(action_rows):
        if len(row) != num_concepts + 1:
            raise ValueError(f"concept CSV row {index + 2} has an incorrect column count")
        if row[0].strip() != NTU_ACTION_NAMES[index]:
            raise ValueError(
                f"NTU label {index} must be {NTU_ACTION_NAMES[index]!r}, got {row[0]!r}"
            )
        try:
            vector = [float(value) for value in row[1:]]
        except ValueError as error:
            raise ValueError(f"nonnumeric concept value at CSV row {index + 2}") from error
        if any(value not in (0.0, 1.0) for value in vector):
            raise ValueError(f"concept values must be binary at CSV row {index + 2}")
        if not any(vector):
            raise ValueError(f"action {row[0]!r} has no positive concepts")
        values.append(vector)
    with Path(metadata_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"concept_id", "concept_name", "scope", "group"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"metadata CSV requires columns {sorted(required)}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("metadata CSV contains duplicate column names")
        metadata = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in metadata):
        raise ValueError("metadata CSV contains an incorrect column count")
    try:
        metadata.sort(key=lambda row: int(row["concept_id"]))
        ids = [int(row["concept_id"]) for row in metadata]
    except (ValueError, TypeError) as error:
        raise ValueError("metadata concept_id must be an integer") from error
    if ids != list(range(num_concepts)):
        raise ValueError("metadata concept_id must be unique and contiguous from 0")
    for row in metadata:
        if any(row.get(key) is None for key in required):
            raise ValueError("metadata CSV contains an incomplete row")
        for key in required:
            row[key] = row[key].strip()
        if any(not row[key] for key in required):
            raise ValueError("metadata CSV requires nonempty ID, name, scope and group")
        row["scope"] = row["scope"].lower()
        row["group"] = row["group"].lower()
        concept_parts(row["concept_name"], row["scope"], row["group"])
    if names != [row["concept_name"] for row in metadata]:
        raise ValueError("concept CSV and metadata concept order do not match")
    return values[:num_classes], metadata
