"""Build and audit Multi-Axis Semantic-Kinematic Concept Annotations for all 120 NTU classes.

Follows the 3-Stage Prompting Protocol documented in prompt_templates.md:
  - Stage 1: 72 concepts across 6 orthogonal axes (concept_bank_v1.json).
  - Stage 2: Per-class multi-axis annotations for NTU-120 (classes 1..60 = NTU-60).
  - Stage 3: Automated discriminative verification & coverage audit.
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.path.join(HERE, 'concept_bank_v1.json')
OUT_PATH = os.path.join(HERE, 'ntu120_concept_annotations.json')

# 1-indexed class_id -> (class_name, list_of_concept_ids)
NTU120_ANNOTATIONS = {
    1: ("Drink water", [
        "topo.unilateral_hand", "topo.head_neck_primary", "zone.mouth_nose_face",
        "dyn.lift_raise_up", "dyn.pull_retract_in", "dyn.head_tilt_look",
        "inter.single_object_hold", "temp.reach_and_retract", "intent.eating_drinking",
    ]),
    2: ("Eat meal/snack", [
        "topo.unilateral_hand", "zone.mouth_nose_face", "zone.front_workspace",
        "dyn.lift_raise_up", "dyn.rhythmic_oscillation", "dyn.pull_retract_in",
        "inter.single_object_hold", "temp.cyclic_repetitive", "intent.eating_drinking",
    ]),
    3: ("Brushing teeth", [
        "topo.unilateral_hand", "zone.mouth_nose_face",
        "dyn.rhythmic_oscillation", "dyn.fine_wrist_finger",
        "inter.single_object_hold", "inter.self_contact_touch",
        "temp.cyclic_repetitive", "intent.personal_grooming",
    ]),
    4: ("Brushing hair", [
        "topo.unilateral_hand", "zone.head_hair_ear", "zone.overhead_high",
        "dyn.rhythmic_oscillation", "dyn.lower_descend_drop",
        "inter.single_object_hold", "inter.self_contact_touch",
        "temp.cyclic_repetitive", "intent.personal_grooming",
    ]),
    5: ("Drop", [
        "topo.unilateral_hand", "zone.front_workspace", "zone.floor_foot_low",
        "dyn.lower_descend_drop", "inter.object_release_launch",
        "temp.single_impulsive", "intent.object_container",
    ]),
    6: ("Pickup", [
        "topo.unilateral_hand", "topo.torso_bend_lean", "topo.lower_body_legs",
        "zone.floor_foot_low", "dyn.lower_descend_drop", "dyn.lift_raise_up",
        "inter.single_object_hold", "temp.reach_and_retract", "intent.object_container",
    ]),
    7: ("Throw", [
        "topo.unilateral_hand", "topo.torso_twist", "zone.overhead_high", "zone.arm_reach_forward",
        "dyn.ballistic_strike", "dyn.push_extend_out",
        "inter.single_object_hold", "inter.object_release_launch",
        "temp.single_impulsive", "intent.sport_ball_play",
    ]),
    8: ("Sitting down", [
        "topo.lower_body_legs", "topo.torso_bend_lean", "topo.whole_body_coord",
        "zone.floor_foot_low", "dyn.lower_descend_drop",
        "temp.state_transition", "intent.posture_change",
    ]),
    9: ("Standing up (from sitting position)", [
        "topo.lower_body_legs", "topo.whole_body_coord",
        "zone.floor_foot_low", "dyn.lift_raise_up",
        "temp.state_transition", "intent.posture_change",
    ]),
    10: ("Clapping", [
        "topo.bilateral_symmetric", "zone.front_workspace",
        "dyn.rhythmic_oscillation", "dyn.together_closing", "dyn.apart_opening",
        "inter.hand_to_hand_contact", "temp.cyclic_repetitive", "intent.greeting_polite",
    ]),
    11: ("Reading", [
        "topo.bilateral_symmetric", "topo.head_neck_primary", "zone.front_workspace",
        "dyn.static_posture_hold", "dyn.head_tilt_look",
        "inter.bimanual_object_work", "temp.sustained_stationary", "intent.paper_craft_office",
    ]),
    12: ("Writing", [
        "topo.bilateral_asymmetric", "topo.head_neck_primary", "zone.front_workspace",
        "dyn.fine_wrist_finger", "dyn.head_tilt_look",
        "inter.single_object_hold", "temp.continuous_flow", "intent.paper_craft_office",
    ]),
    13: ("Tear up paper", [
        "topo.bilateral_symmetric", "zone.front_workspace",
        "dyn.apart_opening", "dyn.fine_wrist_finger", "dyn.ballistic_strike",
        "inter.bimanual_object_work", "temp.single_impulsive", "intent.paper_craft_office",
    ]),
    14: ("Wear jacket", [
        "topo.bilateral_symmetric", "topo.torso_twist",
        "zone.back_shoulder_rear", "zone.chest_front_torso",
        "dyn.pull_retract_in", "dyn.push_extend_out",
        "inter.garment_wear_put", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    15: ("Take off jacket", [
        "topo.bilateral_symmetric", "topo.torso_twist",
        "zone.back_shoulder_rear", "zone.side_lateral_space",
        "dyn.apart_opening", "dyn.lower_descend_drop",
        "inter.garment_remove_off", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    16: ("Wear a shoe", [
        "topo.bilateral_symmetric", "topo.torso_bend_lean", "topo.lower_body_legs",
        "zone.floor_foot_low", "dyn.lower_descend_drop", "dyn.pull_retract_in",
        "inter.garment_wear_put", "inter.self_contact_touch",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    17: ("Take off a shoe", [
        "topo.bilateral_symmetric", "topo.torso_bend_lean", "topo.lower_body_legs",
        "zone.floor_foot_low", "dyn.lower_descend_drop", "dyn.push_extend_out",
        "inter.garment_remove_off", "inter.self_contact_touch",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    18: ("Wear on glasses", [
        "topo.bilateral_symmetric", "zone.mouth_nose_face", "zone.head_hair_ear",
        "dyn.lift_raise_up", "dyn.pull_retract_in",
        "inter.garment_wear_put", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    19: ("Take off glasses", [
        "topo.bilateral_symmetric", "zone.mouth_nose_face", "zone.head_hair_ear",
        "dyn.push_extend_out", "dyn.lower_descend_drop",
        "inter.garment_remove_off", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    20: ("Put on a hat/cap", [
        "topo.bilateral_symmetric", "zone.head_hair_ear", "zone.overhead_high",
        "dyn.lift_raise_up", "dyn.lower_descend_drop", "dyn.pull_retract_in",
        "inter.garment_wear_put", "temp.state_transition", "intent.dressing_accessories",
    ]),
    21: ("Take off a hat/cap", [
        "topo.bilateral_symmetric", "zone.head_hair_ear", "zone.overhead_high",
        "dyn.lift_raise_up", "dyn.push_extend_out", "dyn.lower_descend_drop",
        "inter.garment_remove_off", "temp.state_transition", "intent.dressing_accessories",
    ]),
    22: ("Cheer up", [
        "topo.bilateral_symmetric", "zone.overhead_high", "zone.side_lateral_space",
        "dyn.lift_raise_up", "dyn.rhythmic_oscillation",
        "inter.symbolic_hand_sign", "temp.cyclic_repetitive", "intent.greeting_polite",
    ]),
    23: ("Hand waving", [
        "topo.unilateral_hand", "zone.head_hair_ear", "zone.overhead_high", "zone.side_lateral_space",
        "dyn.rhythmic_oscillation", "dyn.lift_raise_up",
        "inter.symbolic_hand_sign", "temp.cyclic_repetitive", "intent.greeting_polite",
    ]),
    24: ("Kicking something", [
        "topo.lower_body_legs", "zone.floor_foot_low", "zone.arm_reach_forward",
        "dyn.ballistic_strike", "dyn.push_extend_out",
        "temp.single_impulsive", "intent.fitness_exercise",
    ]),
    25: ("Reach into pocket", [
        "topo.unilateral_hand", "zone.waist_hip_pocket",
        "dyn.lower_descend_drop", "dyn.pull_retract_in",
        "inter.self_contact_touch", "temp.reach_and_retract", "intent.object_container",
    ]),
    26: ("Hopping (one foot jumping)", [
        "topo.lower_body_legs", "topo.whole_body_coord", "zone.floor_foot_low",
        "dyn.airborne_jump", "dyn.rhythmic_oscillation",
        "temp.cyclic_repetitive", "intent.fitness_exercise",
    ]),
    27: ("Jumping up", [
        "topo.lower_body_legs", "topo.whole_body_coord", "zone.floor_foot_low", "zone.overhead_high",
        "dyn.airborne_jump", "dyn.lift_raise_up",
        "temp.single_impulsive", "intent.fitness_exercise",
    ]),
    28: ("Make a phone call/Answer phone", [
        "topo.unilateral_hand", "zone.head_hair_ear",
        "dyn.lift_raise_up", "dyn.static_posture_hold",
        "inter.device_screen_key", "inter.single_object_hold",
        "temp.sustained_stationary", "intent.digital_tech_use",
    ]),
    29: ("Playing with phone/tablet", [
        "topo.bilateral_asymmetric", "topo.head_neck_primary", "zone.front_workspace",
        "dyn.fine_wrist_finger", "dyn.head_tilt_look",
        "inter.device_screen_key", "inter.bimanual_object_work",
        "temp.sustained_stationary", "intent.digital_tech_use",
    ]),
    30: ("Typing on a keyboard", [
        "topo.bilateral_symmetric", "zone.front_workspace", "zone.waist_hip_pocket",
        "dyn.fine_wrist_finger", "dyn.rhythmic_oscillation",
        "inter.device_screen_key", "temp.cyclic_repetitive", "intent.digital_tech_use",
    ]),
    31: ("Pointing to something with finger", [
        "topo.unilateral_hand", "zone.arm_reach_forward",
        "dyn.push_extend_out", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.reach_and_retract", "intent.expressive_gesture",
    ]),
    32: ("Taking a selfie", [
        "topo.unilateral_hand", "zone.arm_reach_forward", "zone.overhead_high",
        "dyn.lift_raise_up", "dyn.push_extend_out", "dyn.static_posture_hold",
        "inter.device_screen_key", "inter.single_object_hold",
        "temp.sustained_stationary", "intent.digital_tech_use",
    ]),
    33: ("Checking time (from watch)", [
        "topo.unilateral_hand", "topo.head_neck_primary", "zone.front_workspace",
        "dyn.lift_raise_up", "dyn.circular_twisting", "dyn.head_tilt_look",
        "temp.reach_and_retract", "intent.object_container",
    ]),
    34: ("Rubbing two hands together", [
        "topo.bilateral_symmetric", "zone.front_workspace",
        "dyn.rhythmic_oscillation", "dyn.fine_wrist_finger",
        "inter.hand_to_hand_contact", "temp.cyclic_repetitive", "intent.micro_fidget_play",
    ]),
    35: ("Nod head/bow", [
        "topo.head_neck_primary", "topo.torso_bend_lean", "zone.chest_front_torso",
        "dyn.lower_descend_drop", "dyn.head_tilt_look",
        "temp.reach_and_retract", "intent.greeting_polite",
    ]),
    36: ("Shake head", [
        "topo.head_neck_primary", "zone.head_hair_ear",
        "dyn.rhythmic_oscillation", "dyn.circular_twisting",
        "temp.cyclic_repetitive", "intent.expressive_gesture",
    ]),
    37: ("Wiping face", [
        "topo.unilateral_hand", "zone.mouth_nose_face",
        "dyn.rhythmic_oscillation", "dyn.circular_twisting",
        "inter.self_contact_touch", "temp.cyclic_repetitive", "intent.personal_grooming",
    ]),
    38: ("Salute", [
        "topo.unilateral_hand", "zone.head_hair_ear", "zone.side_lateral_space",
        "dyn.lift_raise_up", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.sustained_stationary", "intent.greeting_polite",
    ]),
    39: ("Putting the palms together", [
        "topo.bilateral_symmetric", "zone.chest_front_torso", "zone.front_workspace",
        "dyn.together_closing", "dyn.static_posture_hold",
        "inter.hand_to_hand_contact", "inter.symbolic_hand_sign",
        "temp.sustained_stationary", "intent.greeting_polite",
    ]),
    40: ("Crossing hands in front (saying stop)", [
        "topo.bilateral_symmetric", "zone.front_workspace", "zone.chest_front_torso",
        "dyn.together_closing", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    41: ("Sneeze/cough", [
        "topo.unilateral_hand", "topo.head_neck_primary", "topo.torso_bend_lean",
        "zone.mouth_nose_face", "zone.neck_throat", "dyn.pull_retract_in", "dyn.ballistic_strike",
        "inter.self_contact_touch", "temp.single_impulsive", "intent.health_pain_reflex",
    ]),
    42: ("Staggering", [
        "topo.whole_body_coord", "topo.lower_body_legs", "zone.side_lateral_space",
        "dyn.instability_sway", "dyn.locomotion_walk",
        "temp.continuous_flow", "intent.health_pain_reflex",
    ]),
    43: ("Falling", [
        "topo.whole_body_coord", "topo.torso_bend_lean", "zone.floor_foot_low",
        "dyn.instability_sway", "dyn.lower_descend_drop", "dyn.ballistic_strike",
        "temp.state_transition", "intent.health_pain_reflex",
    ]),
    44: ("Touching head (headache)", [
        "topo.bilateral_symmetric", "topo.head_neck_primary", "zone.head_hair_ear",
        "dyn.lift_raise_up", "dyn.static_posture_hold",
        "inter.self_contact_touch", "temp.sustained_stationary", "intent.health_pain_reflex",
    ]),
    45: ("Touching chest (stomachache/heart pain)", [
        "topo.bilateral_symmetric", "topo.torso_bend_lean", "zone.chest_front_torso",
        "dyn.pull_retract_in", "dyn.static_posture_hold",
        "inter.self_contact_touch", "temp.sustained_stationary", "intent.health_pain_reflex",
    ]),
    46: ("Touching back (backache)", [
        "topo.bilateral_symmetric", "topo.torso_bend_lean", "zone.back_shoulder_rear",
        "dyn.pull_retract_in", "dyn.static_posture_hold",
        "inter.self_contact_touch", "temp.sustained_stationary", "intent.health_pain_reflex",
    ]),
    47: ("Touching neck (neckache)", [
        "topo.unilateral_hand", "topo.head_neck_primary", "zone.neck_throat",
        "dyn.pull_retract_in", "dyn.rhythmic_oscillation",
        "inter.self_contact_touch", "temp.sustained_stationary", "intent.health_pain_reflex",
    ]),
    48: ("Nausea or vomiting", [
        "topo.torso_bend_lean", "topo.head_neck_primary", "zone.mouth_nose_face", "zone.chest_front_torso",
        "dyn.lower_descend_drop", "dyn.head_tilt_look",
        "inter.self_contact_touch", "temp.cyclic_repetitive", "intent.health_pain_reflex",
    ]),
    49: ("Using a fan (with hand or paper)", [
        "topo.unilateral_hand", "zone.mouth_nose_face", "zone.chest_front_torso",
        "dyn.rhythmic_oscillation", "dyn.fine_wrist_finger",
        "inter.single_object_hold", "temp.cyclic_repetitive", "intent.micro_fidget_play",
    ]),
    50: ("Punching/Slapping other person", [
        "topo.two_person_directed", "topo.unilateral_hand", "topo.torso_twist",
        "zone.interpersonal_gap", "zone.arm_reach_forward",
        "dyn.ballistic_strike", "dyn.push_extend_out",
        "inter.interpersonal_aggressive", "temp.single_impulsive", "intent.conflict_aggression",
    ]),
    51: ("Kicking other person", [
        "topo.two_person_directed", "topo.lower_body_legs",
        "zone.interpersonal_gap", "zone.floor_foot_low",
        "dyn.ballistic_strike", "dyn.push_extend_out",
        "inter.interpersonal_aggressive", "temp.single_impulsive", "intent.conflict_aggression",
    ]),
    52: ("Pushing other person", [
        "topo.two_person_directed", "topo.bilateral_symmetric",
        "zone.interpersonal_gap", "zone.arm_reach_forward",
        "dyn.push_extend_out", "dyn.ballistic_strike",
        "inter.interpersonal_aggressive", "temp.single_impulsive", "intent.conflict_aggression",
    ]),
    53: ("Patting on back of other person", [
        "topo.two_person_directed", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.back_shoulder_rear",
        "dyn.rhythmic_oscillation", "dyn.push_extend_out",
        "inter.interpersonal_friendly", "temp.cyclic_repetitive", "intent.social_play_support",
    ]),
    54: ("Pointing finger at the other person", [
        "topo.two_person_directed", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.arm_reach_forward",
        "dyn.push_extend_out", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.reach_and_retract", "intent.expressive_gesture",
    ]),
    55: ("Hugging other person", [
        "topo.two_person_mutual", "topo.bilateral_symmetric",
        "zone.interpersonal_gap", "zone.chest_front_torso",
        "dyn.apart_opening", "dyn.together_closing", "dyn.pull_retract_in",
        "inter.interpersonal_friendly", "temp.sustained_stationary", "intent.greeting_polite",
    ]),
    56: ("Giving something to other person", [
        "topo.two_person_directed", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.arm_reach_forward",
        "dyn.push_extend_out", "inter.interpersonal_exchange", "inter.object_release_launch",
        "temp.reach_and_retract", "intent.object_container",
    ]),
    57: ("Touching other person's pocket", [
        "topo.two_person_directed", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.waist_hip_pocket",
        "dyn.push_extend_out", "dyn.pull_retract_in",
        "inter.interpersonal_aggressive", "temp.reach_and_retract", "intent.conflict_aggression",
    ]),
    58: ("Handshaking", [
        "topo.two_person_mutual", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.front_workspace",
        "dyn.push_extend_out", "dyn.rhythmic_oscillation",
        "inter.interpersonal_friendly", "inter.hand_to_hand_contact",
        "temp.cyclic_repetitive", "intent.greeting_polite",
    ]),
    59: ("Walking towards each other", [
        "topo.two_person_mutual", "topo.lower_body_legs", "topo.whole_body_coord",
        "zone.interpersonal_gap", "dyn.locomotion_walk", "dyn.together_closing",
        "inter.interpersonal_proxemic", "temp.continuous_flow", "intent.spatial_navigation",
    ]),
    60: ("Walking apart from each other", [
        "topo.two_person_mutual", "topo.lower_body_legs", "topo.whole_body_coord",
        "zone.interpersonal_gap", "dyn.locomotion_walk", "dyn.apart_opening",
        "inter.interpersonal_proxemic", "temp.continuous_flow", "intent.spatial_navigation",
    ]),
    61: ("Putting on headphones", [
        "topo.bilateral_symmetric", "zone.head_hair_ear", "zone.side_lateral_space",
        "dyn.lift_raise_up", "dyn.together_closing", "dyn.pull_retract_in",
        "inter.garment_wear_put", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    62: ("Taking off headphones", [
        "topo.bilateral_symmetric", "zone.head_hair_ear", "zone.side_lateral_space",
        "dyn.apart_opening", "dyn.lower_descend_drop",
        "inter.garment_remove_off", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    63: ("Shooting at the basket", [
        "topo.bilateral_symmetric", "topo.whole_body_coord",
        "zone.overhead_high", "zone.arm_reach_forward",
        "dyn.lift_raise_up", "dyn.ballistic_strike", "dyn.push_extend_out",
        "inter.bimanual_object_work", "inter.object_release_launch",
        "temp.single_impulsive", "intent.sport_ball_play",
    ]),
    64: ("Bouncing a ball", [
        "topo.unilateral_hand", "zone.waist_hip_pocket", "zone.floor_foot_low",
        "dyn.rhythmic_oscillation", "dyn.lower_descend_drop",
        "inter.single_object_hold", "temp.cyclic_repetitive", "intent.sport_ball_play",
    ]),
    65: ("Tennis bat swing", [
        "topo.unilateral_hand", "topo.torso_twist", "topo.whole_body_coord",
        "zone.side_lateral_space", "zone.arm_reach_forward",
        "dyn.ballistic_strike", "dyn.circular_twisting",
        "inter.single_object_hold", "temp.single_impulsive", "intent.sport_ball_play",
    ]),
    66: ("Juggling table tennis balls", [
        "topo.unilateral_hand", "zone.front_workspace",
        "dyn.rhythmic_oscillation", "dyn.fine_wrist_finger", "dyn.lift_raise_up",
        "inter.single_object_hold", "temp.cyclic_repetitive", "intent.sport_ball_play",
    ]),
    67: ("Hushing", [
        "topo.unilateral_hand", "zone.mouth_nose_face", "zone.neck_throat",
        "dyn.lift_raise_up", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "inter.self_contact_touch",
        "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    68: ("Flicking hair", [
        "topo.unilateral_hand", "topo.head_neck_primary", "zone.head_hair_ear",
        "dyn.ballistic_strike", "dyn.push_extend_out",
        "inter.self_contact_touch", "temp.single_impulsive", "intent.personal_grooming",
    ]),
    69: ("Thumbs up", [
        "topo.unilateral_hand", "zone.front_workspace",
        "dyn.lift_raise_up", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    70: ("Thumbs down", [
        "topo.unilateral_hand", "zone.front_workspace",
        "dyn.lower_descend_drop", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    71: ("Making OK sign", [
        "topo.unilateral_hand", "zone.front_workspace",
        "dyn.fine_wrist_finger", "dyn.together_closing", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    72: ("Making victory sign", [
        "topo.unilateral_hand", "zone.front_workspace", "zone.head_hair_ear",
        "dyn.fine_wrist_finger", "dyn.apart_opening", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    73: ("Stapling book", [
        "topo.bilateral_asymmetric", "zone.front_workspace",
        "dyn.together_closing", "dyn.fine_wrist_finger",
        "inter.bimanual_object_work", "temp.single_impulsive", "intent.paper_craft_office",
    ]),
    74: ("Counting money", [
        "topo.bilateral_asymmetric", "zone.front_workspace",
        "dyn.fine_wrist_finger", "dyn.rhythmic_oscillation",
        "inter.bimanual_object_work", "temp.cyclic_repetitive", "intent.paper_craft_office",
    ]),
    75: ("Cutting nails", [
        "topo.bilateral_asymmetric", "topo.head_neck_primary", "zone.front_workspace",
        "dyn.fine_wrist_finger", "dyn.head_tilt_look",
        "inter.hand_to_hand_contact", "inter.self_contact_touch",
        "temp.cyclic_repetitive", "intent.personal_grooming",
    ]),
    76: ("Cutting paper (using scissors)", [
        "topo.bilateral_asymmetric", "zone.front_workspace",
        "dyn.fine_wrist_finger", "dyn.push_extend_out", "dyn.rhythmic_oscillation",
        "inter.bimanual_object_work", "temp.continuous_flow", "intent.paper_craft_office",
    ]),
    77: ("Snapping fingers", [
        "topo.unilateral_hand", "zone.front_workspace",
        "dyn.fine_wrist_finger", "dyn.ballistic_strike",
        "inter.symbolic_hand_sign", "temp.single_impulsive", "intent.micro_fidget_play",
    ]),
    78: ("Opening bottle", [
        "topo.bilateral_asymmetric", "zone.front_workspace",
        "dyn.circular_twisting", "dyn.fine_wrist_finger",
        "inter.bimanual_object_work", "temp.state_transition", "intent.object_container",
    ]),
    79: ("Sniffing", [
        "topo.unilateral_hand", "topo.head_neck_primary", "zone.mouth_nose_face",
        "dyn.pull_retract_in", "dyn.head_tilt_look", "dyn.static_posture_hold",
        "inter.single_object_hold", "temp.sustained_stationary", "intent.health_pain_reflex",
    ]),
    80: ("Squatting down", [
        "topo.lower_body_legs", "topo.torso_bend_lean", "topo.whole_body_coord",
        "zone.floor_foot_low", "dyn.lower_descend_drop", "dyn.static_posture_hold",
        "temp.state_transition", "intent.posture_change",
    ]),
    81: ("Tossing a coin", [
        "topo.unilateral_hand", "zone.front_workspace", "zone.overhead_high",
        "dyn.fine_wrist_finger", "dyn.lift_raise_up", "dyn.ballistic_strike",
        "inter.object_release_launch", "temp.single_impulsive", "intent.micro_fidget_play",
    ]),
    82: ("Folding paper", [
        "topo.bilateral_symmetric", "zone.front_workspace",
        "dyn.together_closing", "dyn.fine_wrist_finger",
        "inter.bimanual_object_work", "temp.continuous_flow", "intent.paper_craft_office",
    ]),
    83: ("Balling up paper", [
        "topo.bilateral_symmetric", "zone.front_workspace",
        "dyn.together_closing", "dyn.circular_twisting", "dyn.fine_wrist_finger",
        "inter.bimanual_object_work", "inter.hand_to_hand_contact",
        "temp.cyclic_repetitive", "intent.paper_craft_office",
    ]),
    84: ("Playing with a magic cube", [
        "topo.bilateral_asymmetric", "zone.front_workspace",
        "dyn.circular_twisting", "dyn.fine_wrist_finger", "dyn.rhythmic_oscillation",
        "inter.bimanual_object_work", "temp.cyclic_repetitive", "intent.micro_fidget_play",
    ]),
    85: ("Applying cream on face", [
        "topo.bilateral_asymmetric", "zone.mouth_nose_face", "zone.front_workspace",
        "dyn.circular_twisting", "dyn.rhythmic_oscillation",
        "inter.self_contact_touch", "temp.cyclic_repetitive", "intent.personal_grooming",
    ]),
    86: ("Applying cream on hand back", [
        "topo.bilateral_asymmetric", "zone.front_workspace",
        "dyn.circular_twisting", "dyn.rhythmic_oscillation",
        "inter.hand_to_hand_contact", "inter.self_contact_touch",
        "temp.cyclic_repetitive", "intent.personal_grooming",
    ]),
    87: ("Putting on a bag", [
        "topo.bilateral_symmetric", "topo.torso_twist",
        "zone.back_shoulder_rear", "zone.overhead_high",
        "dyn.lift_raise_up", "dyn.pull_retract_in",
        "inter.garment_wear_put", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    88: ("Taking off a bag", [
        "topo.bilateral_symmetric", "topo.torso_twist",
        "zone.back_shoulder_rear", "zone.waist_hip_pocket",
        "dyn.lower_descend_drop", "dyn.push_extend_out",
        "inter.garment_remove_off", "inter.bimanual_object_work",
        "temp.state_transition", "intent.dressing_accessories",
    ]),
    89: ("Putting something into a bag", [
        "topo.bilateral_asymmetric", "zone.front_workspace", "zone.waist_hip_pocket",
        "dyn.lower_descend_drop", "dyn.together_closing",
        "inter.bimanual_object_work", "inter.object_release_launch",
        "temp.state_transition", "intent.object_container",
    ]),
    90: ("Taking something out of a bag", [
        "topo.bilateral_asymmetric", "zone.front_workspace", "zone.waist_hip_pocket",
        "dyn.lift_raise_up", "dyn.apart_opening", "dyn.pull_retract_in",
        "inter.bimanual_object_work", "temp.state_transition", "intent.object_container",
    ]),
    91: ("Opening a box", [
        "topo.bilateral_symmetric", "zone.front_workspace",
        "dyn.apart_opening", "dyn.lift_raise_up",
        "inter.bimanual_object_work", "temp.state_transition", "intent.object_container",
    ]),
    92: ("Moving heavy objects", [
        "topo.bilateral_symmetric", "topo.torso_bend_lean", "topo.lower_body_legs", "topo.whole_body_coord",
        "zone.floor_foot_low", "dyn.lift_raise_up", "dyn.locomotion_walk",
        "inter.bimanual_object_work", "temp.continuous_flow", "intent.spatial_navigation",
    ]),
    93: ("Shaking fist", [
        "topo.unilateral_hand", "zone.head_hair_ear", "zone.arm_reach_forward",
        "dyn.rhythmic_oscillation", "dyn.ballistic_strike",
        "inter.symbolic_hand_sign", "temp.cyclic_repetitive", "intent.expressive_gesture",
    ]),
    94: ("Throwing up a cap/hat", [
        "topo.unilateral_hand", "zone.overhead_high",
        "dyn.lift_raise_up", "dyn.ballistic_strike",
        "inter.single_object_hold", "inter.object_release_launch",
        "temp.single_impulsive", "intent.greeting_polite",
    ]),
    95: ("Raising hands up", [
        "topo.bilateral_symmetric", "zone.overhead_high", "zone.side_lateral_space",
        "dyn.lift_raise_up", "dyn.static_posture_hold",
        "inter.symbolic_hand_sign", "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    96: ("Crossing arms", [
        "topo.bilateral_symmetric", "zone.chest_front_torso",
        "dyn.together_closing", "dyn.pull_retract_in", "dyn.static_posture_hold",
        "inter.self_contact_touch", "temp.sustained_stationary", "intent.expressive_gesture",
    ]),
    97: ("Making arm circles", [
        "topo.bilateral_symmetric", "zone.side_lateral_space", "zone.overhead_high",
        "dyn.circular_twisting", "dyn.rhythmic_oscillation",
        "temp.cyclic_repetitive", "intent.fitness_exercise",
    ]),
    98: ("Swinging arms", [
        "topo.bilateral_symmetric", "topo.torso_twist", "zone.side_lateral_space",
        "dyn.rhythmic_oscillation", "dyn.apart_opening", "dyn.together_closing",
        "temp.cyclic_repetitive", "intent.fitness_exercise",
    ]),
    99: ("Running on the spot", [
        "topo.whole_body_coord", "topo.lower_body_legs", "zone.floor_foot_low",
        "dyn.rhythmic_oscillation", "dyn.lift_raise_up",
        "temp.cyclic_repetitive", "intent.fitness_exercise",
    ]),
    100: ("Butt kicks (kick backward)", [
        "topo.lower_body_legs", "zone.back_shoulder_rear", "zone.floor_foot_low",
        "dyn.rhythmic_oscillation", "dyn.pull_retract_in",
        "temp.cyclic_repetitive", "intent.fitness_exercise",
    ]),
    101: ("Cross toe touch", [
        "topo.whole_body_coord", "topo.torso_bend_lean", "topo.torso_twist",
        "zone.floor_foot_low", "dyn.lower_descend_drop", "dyn.rhythmic_oscillation",
        "inter.self_contact_touch", "temp.cyclic_repetitive", "intent.fitness_exercise",
    ]),
    102: ("Side kick", [
        "topo.lower_body_legs", "topo.whole_body_coord", "zone.side_lateral_space",
        "dyn.ballistic_strike", "dyn.lift_raise_up",
        "temp.single_impulsive", "intent.fitness_exercise",
    ]),
    103: ("Yawning", [
        "topo.unilateral_hand", "topo.head_neck_primary", "topo.torso_bend_lean",
        "zone.mouth_nose_face", "dyn.lift_raise_up", "dyn.static_posture_hold",
        "inter.self_contact_touch", "temp.sustained_stationary", "intent.health_pain_reflex",
    ]),
    104: ("Stretching oneself", [
        "topo.bilateral_symmetric", "topo.torso_bend_lean", "topo.whole_body_coord",
        "zone.overhead_high", "zone.side_lateral_space",
        "dyn.apart_opening", "dyn.lift_raise_up", "dyn.static_posture_hold",
        "temp.sustained_stationary", "intent.posture_change",
    ]),
    105: ("Blowing nose", [
        "topo.unilateral_hand", "topo.head_neck_primary", "topo.torso_bend_lean",
        "zone.mouth_nose_face", "dyn.pull_retract_in", "dyn.fine_wrist_finger",
        "inter.self_contact_touch", "inter.single_object_hold",
        "temp.reach_and_retract", "intent.health_pain_reflex",
    ]),
    106: ("Hitting other person with something", [
        "topo.two_person_directed", "topo.bilateral_symmetric", "topo.torso_twist",
        "zone.interpersonal_gap", "zone.overhead_high",
        "dyn.ballistic_strike", "dyn.circular_twisting",
        "inter.interpersonal_aggressive", "inter.bimanual_object_work",
        "temp.single_impulsive", "intent.conflict_aggression",
    ]),
    107: ("Wielding knife towards other person", [
        "topo.two_person_directed", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.arm_reach_forward",
        "dyn.rhythmic_oscillation", "dyn.push_extend_out",
        "inter.interpersonal_aggressive", "inter.single_object_hold",
        "temp.cyclic_repetitive", "intent.conflict_aggression",
    ]),
    108: ("Knocking over other person (hit with body)", [
        "topo.two_person_directed", "topo.whole_body_coord", "topo.torso_twist",
        "zone.interpersonal_gap", "zone.side_lateral_space",
        "dyn.ballistic_strike", "dyn.push_extend_out",
        "inter.interpersonal_aggressive", "temp.single_impulsive", "intent.conflict_aggression",
    ]),
    109: ("Grabbing other person's stuff", [
        "topo.two_person_directed", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.arm_reach_forward",
        "dyn.ballistic_strike", "dyn.pull_retract_in",
        "inter.interpersonal_aggressive", "inter.interpersonal_exchange",
        "temp.reach_and_retract", "intent.conflict_aggression",
    ]),
    110: ("Shooting at other person with a gun", [
        "topo.two_person_directed", "topo.bilateral_symmetric",
        "zone.interpersonal_gap", "zone.arm_reach_forward",
        "dyn.push_extend_out", "dyn.static_posture_hold", "dyn.fine_wrist_finger",
        "inter.interpersonal_aggressive", "inter.bimanual_object_work",
        "temp.sustained_stationary", "intent.conflict_aggression",
    ]),
    111: ("Stepping on foot", [
        "topo.two_person_directed", "topo.lower_body_legs",
        "zone.interpersonal_gap", "zone.floor_foot_low",
        "dyn.lift_raise_up", "dyn.lower_descend_drop", "dyn.ballistic_strike",
        "inter.interpersonal_aggressive", "temp.single_impulsive", "intent.conflict_aggression",
    ]),
    112: ("High-five", [
        "topo.two_person_mutual", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.overhead_high",
        "dyn.lift_raise_up", "dyn.ballistic_strike", "dyn.together_closing",
        "inter.interpersonal_friendly", "inter.hand_to_hand_contact",
        "temp.single_impulsive", "intent.social_play_support",
    ]),
    113: ("Cheers and drink", [
        "topo.two_person_mutual", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.mouth_nose_face",
        "dyn.push_extend_out", "dyn.pull_retract_in", "dyn.lift_raise_up",
        "inter.interpersonal_friendly", "inter.single_object_hold",
        "temp.reach_and_retract", "intent.social_play_support", "intent.eating_drinking",
    ]),
    114: ("Carrying something with other person", [
        "topo.two_person_mutual", "topo.bilateral_symmetric", "topo.whole_body_coord",
        "zone.interpersonal_gap", "zone.floor_foot_low",
        "dyn.lift_raise_up", "dyn.locomotion_walk",
        "inter.interpersonal_exchange", "inter.bimanual_object_work",
        "temp.continuous_flow", "intent.spatial_navigation",
    ]),
    115: ("Taking a photo of other person", [
        "topo.two_person_directed", "topo.bilateral_symmetric",
        "zone.interpersonal_gap", "zone.front_workspace", "zone.head_hair_ear",
        "dyn.lift_raise_up", "dyn.static_posture_hold", "dyn.fine_wrist_finger",
        "inter.device_screen_key", "inter.bimanual_object_work",
        "temp.sustained_stationary", "intent.digital_tech_use",
    ]),
    116: ("Following other person", [
        "topo.two_person_directed", "topo.lower_body_legs", "topo.whole_body_coord",
        "zone.interpersonal_gap", "dyn.locomotion_walk",
        "inter.interpersonal_proxemic", "temp.continuous_flow", "intent.spatial_navigation",
    ]),
    117: ("Whispering in other person's ear", [
        "topo.two_person_mutual", "topo.torso_bend_lean", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.head_hair_ear", "zone.mouth_nose_face",
        "dyn.together_closing", "dyn.static_posture_hold",
        "inter.interpersonal_proxemic", "temp.sustained_stationary", "intent.social_play_support",
    ]),
    118: ("Exchanging things with other person", [
        "topo.two_person_mutual", "topo.bilateral_asymmetric",
        "zone.interpersonal_gap", "zone.front_workspace",
        "dyn.push_extend_out", "dyn.pull_retract_in",
        "inter.interpersonal_exchange", "temp.reach_and_retract", "intent.object_container",
    ]),
    119: ("Supporting somebody with hand", [
        "topo.two_person_mutual", "topo.whole_body_coord", "topo.lower_body_legs",
        "zone.interpersonal_gap", "zone.side_lateral_space",
        "dyn.locomotion_walk", "inter.interpersonal_friendly",
        "temp.continuous_flow", "intent.social_play_support", "intent.spatial_navigation",
    ]),
    120: ("Finger-guessing game (rock-paper-scissors)", [
        "topo.two_person_mutual", "topo.unilateral_hand",
        "zone.interpersonal_gap", "zone.front_workspace",
        "dyn.rhythmic_oscillation", "dyn.fine_wrist_finger",
        "inter.symbolic_hand_sign", "temp.cyclic_repetitive", "intent.social_play_support",
    ]),
}


def main():
    bank = json.load(open(BANK_PATH, encoding='utf-8'))
    valid_ids = {c['id']: c['axis'] for c in bank}
    assert len(NTU120_ANNOTATIONS) == 120, f'Expected 120 classes, got {len(NTU120_ANNOTATIONS)}'

    records = []
    usage = Counter()
    sigs = {}
    dups = []
    for cid in range(1, 121):
        name, cids = NTU120_ANNOTATIONS[cid]
        cids_unique = list(dict.fromkeys(cids))
        bad = [c for c in cids_unique if c not in valid_ids]
        assert not bad, f'Class {cid} ({name}) has unknown concepts: {bad}'
        for c in cids_unique:
            usage[c] += 1
        sig = frozenset(cids_unique)
        if sig in sigs:
            dups.append((sigs[sig], (cid, name)))
        else:
            sigs[sig] = (cid, name)
        records.append({'class_id': cid, 'class_name': name, 'concepts': cids_unique})

    # Audit 1: Exact duplicates
    assert not dups, f'Found duplicate concept signatures: {dups}'

    # Audit 2: Orphan concepts
    orphans = [c['id'] for c in bank if usage[c['id']] < 2]
    print(f'Total concepts: {len(bank)} | Total classes: {len(records)}')
    print(f'Avg concepts/class: {sum(len(r["concepts"]) for r in records)/120:.2f} '
          f'(min={min(len(r["concepts"]) for r in records)}, max={max(len(r["concepts"]) for r in records)})')
    print(f'Exact duplicate pairs: {len(dups)} | Concepts with df < 2: {orphans}')

    # Audit 3: High-overlap pairs (Jaccard >= 0.80)
    high_ov = []
    for i in range(120):
        si = set(records[i]['concepts'])
        for j in range(i + 1, 120):
            sj = set(records[j]['concepts'])
            jac = len(si & sj) / len(si | sj)
            if jac >= 0.80:
                high_ov.append((jac, records[i]['class_id'], records[i]['class_name'],
                                records[j]['class_id'], records[j]['class_name']))
    high_ov.sort(reverse=True)
    print(f'Pairs with Jaccard >= 0.80: {len(high_ov)}')
    for jac, i, ni, j, nj in high_ov[:10]:
        print(f'  J={jac:.2f}: ({i}) {ni}  <->  ({j}) {nj}')

    # Audit 4: Seen/Unseen concept coverage on standard ZSL splits
    unseen_48 = {4, 6, 10, 12, 18, 22, 24, 30, 34, 41, 49, 55}  # 1-indexed approximate check
    unseen_110 = {5, 14, 38, 41, 61, 69, 77, 86, 102, 114}
    for label, total_c, u_set in [('NTU-60 (48/12)', 60, unseen_48), ('NTU-120 (110/10)', 120, unseen_110)]:
        seen_c = set()
        unseen_c = set()
        for r in records[:total_c]:
            if r['class_id'] in u_set:
                unseen_c.update(r['concepts'])
            else:
                seen_c.update(r['concepts'])
        cov = len(unseen_c & seen_c)
        print(f'{label}: unseen concepts covered by seen = {cov}/{len(unseen_c)} '
              f'(unseen-only = {len(unseen_c - seen_c)})')

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
