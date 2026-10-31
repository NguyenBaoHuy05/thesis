# Causal Concept Program on TEST

## 1. Accuracy

| model | top-1 |
|---|---|
| HyperGCN (original) | 0.9108 |
| HyperGCN (fine-tuned in stage 3) | 0.9101 |
| logic only | 0.7339 |
| fused (logic + 0.348 * net) | 0.9098 |

## 2. Concept quality (AUROC vs CSV row of the true class)

- geo concepts: mean AUROC 0.694 over 49
- latent concepts: mean AUROC 0.949 over 21

| concept | kind | signal | #classes in CSV | AUROC | true rate |
|---|---|---|---|---|---|
| foot_slide | latent | - | 1 | 1.000 | 0.016 |
| temporal_sequence_seq_legs_first | latent | - | 5 | 0.991 | 0.080 |
| hand_fist | latent | - | 1 | 0.990 | 0.010 |
| head_rotate | latent | - | 1 | 0.986 | 0.003 |
| temporal_sequence_seq_alternating | latent | - | 3 | 0.982 | 0.034 |
| temporal_dynamics_speed_decelerating | latent | - | 2 | 0.982 | 0.029 |
| temporal_sequence_seq_simultaneous | latent | - | 9 | 0.975 | 0.115 |
| foot_pivot | latent | - | 3 | 0.970 | 0.055 |
| temporal_direction_motion_reversible | latent | - | 8 | 0.970 | 0.078 |
| interaction_two_person | geo | second_present | 11 | 0.964 | 0.179 |
| hand_circular | latent | - | 2 | 0.957 | 0.002 |
| hand_manipulate | latent | - | 4 | 0.957 | 0.061 |
| hand_release | latent | - | 3 | 0.951 | 0.024 |
| temporal_sequence_seq_body_first | latent | - | 10 | 0.951 | 0.130 |
| temporal_dynamics_rhythm_irregular | latent | - | 5 | 0.935 | 0.053 |
| temporal_dynamics_rhythm_regular | latent | - | 15 | 0.929 | 0.169 |
| temporal_dynamics_speed_accelerating | latent | - | 5 | 0.928 | 0.045 |
| temporal_sequence_seq_hands_first | latent | - | 33 | 0.914 | 0.515 |
| hand_lift_to_face | geo | hand_head | 13 | 0.912 | 0.216 |
| temporal_dynamics_duration_sustained | latent | - | 24 | 0.910 | 0.320 |
| hand_grasp | latent | - | 23 | 0.906 | 0.241 |
| leg_alternating | geo | leg_alt_speed | 4 | 0.902 | 0.021 |
| arm_overhead | geo | hand_above_head | 4 | 0.891 | 0.031 |
| leg_step | geo | ankle_speed | 2 | 0.883 | 0.162 |
| hand_open | latent | - | 5 | 0.881 | 0.050 |
| leg_squat | geo | knee_min | 3 | 0.877 | 0.052 |
| foot_step | geo | ankle_speed | 3 | 0.876 | 0.173 |
| temporal_dynamics_duration_brief | latent | - | 34 | 0.865 | 0.497 |
| hip_sit | geo | hip_height | 1 | 0.864 | 0.039 |
| leg_lift | geo | foot_height_diff | 2 | 0.844 | 0.098 |
| hip_bend | geo | torso_pitch | 9 | 0.839 | 0.123 |
| head_nod | geo | head_pitch_speed | 1 | 0.834 | 0.109 |
| hip_sway | geo | hip_lat_speed | 3 | 0.830 | 0.151 |
| leg_kick | geo | foot_height_diff | 2 | 0.816 | 0.092 |
| foot_kick | geo | foot_height_diff | 2 | 0.816 | 0.092 |
| arm_raise | geo | hand_above_shoulder | 21 | 0.808 | 0.396 |
| hand_clap | geo | hands_dist | 1 | 0.793 | 0.000 |
| foot_static | geo | ankle_speed | 48 | 0.776 | 0.975 |
| hip_extend | geo | hip_vy | 3 | 0.775 | 0.116 |
| leg_balance | geo | foot_height_diff | 5 | 0.756 | 0.044 |
| arm_hold | geo | elbow_min | 12 | 0.746 | 0.063 |
| arm_cross | geo | arms_crossed | 2 | 0.738 | 0.014 |
| hand_point | geo | hand_reach_fwd | 2 | 0.733 | 0.010 |
| hand_wave | geo | raised_hand_speed | 2 | 0.721 | 0.210 |
| arm_extend | geo | elbow_max | 14 | 0.709 | 0.003 |
| leg_jump | geo | feet_lift | 3 | 0.694 | 0.056 |
| foot_jump | geo | feet_lift | 3 | 0.694 | 0.056 |
| temporal_direction_motion_backward | geo | hand_fwd_vmin | 3 | 0.657 | 0.189 |
| arm_lateral | geo | hand_lateral | 5 | 0.653 | 0.067 |
| temporal_direction_motion_diverging | geo | hands_dist_v | 2 | 0.651 | 0.146 |
| leg_static_stand | geo | hip_height | 42 | 0.630 | 0.923 |
| temporal_direction_motion_downward | geo | hand_vy_min | 15 | 0.629 | 0.258 |
| hip_rotate | geo | shoulder_yaw_speed | 5 | 0.621 | 0.185 |
| leg_static_sit | geo | hip_height | 6 | 0.614 | 0.047 |
| hip_static_stand | geo | hip_height | 35 | 0.606 | 0.836 |
| head_static | geo | head_speed | 2 | 0.598 | 0.000 |
| head_tilt_down | geo | head_pitch | 12 | 0.594 | 0.075 |
| head_tilt_up | geo | head_pitch | 5 | 0.582 | 0.055 |
| arm_lower | geo | hand_vy_min | 7 | 0.571 | 0.213 |
| hip_static_sit | geo | hip_height | 5 | 0.571 | 0.042 |
| temporal_direction_motion_upward | geo | hand_vy_max | 18 | 0.570 | 0.280 |
| arm_swing | geo | wrist_speed | 5 | 0.568 | 0.199 |
| temporal_dynamics_speed_fast | geo | body_speed | 10 | 0.545 | 0.033 |
| temporal_direction_motion_converging | geo | hands_dist_v | 5 | 0.540 | 0.166 |
| temporal_direction_motion_forward | geo | hand_fwd_vmax | 14 | 0.532 | 0.251 |
| arm_alternating | geo | arm_sym | 6 | 0.524 | 0.001 |
| arm_retract | geo | hand_chest_v | 6 | 0.479 | 0.170 |
| hand_touch | geo | hand_body | 9 | 0.471 | 0.000 |
| head_shake | geo | head_lat_speed | 1 | 0.365 | 0.195 |
| temporal_dynamics_speed_slow | geo | body_speed | 8 | 0.331 | 0.000 |
| arm_circular | latent | - | 0 | - | 0.001 |
| hip_squat | geo | knee_min | 0 | - | 0.018 |
| foot_tap | geo | foot_vy | 0 | - | 0.094 |
| temporal_sequence_seq_cascading | latent | - | 0 | - | 0.003 |
| x_contact_partner | geo | contact_dist | 0 | - | 0.185 |
| x_close_partner | geo | person_dist | 0 | - | 0.181 |
| x_approach_partner | geo | person_dist_v | 0 | - | 0.187 |
| x_leave_partner | geo | person_dist_v | 0 | - | 0.186 |
| x_walk_forward | geo | body_fwd_v | 0 | - | 0.858 |
| x_walk_backward | geo | body_fwd_v | 0 | - | 0.143 |

## 3. Program (weight >= 0.3)

- literals per class: mean 15.1, min 9, max 21
- positive literals from the CSV: 551/593

| k | class | rule | TEST coverage | TEST precision |
|---|---|---|---|---|
| 0 | drink_water | latent:hand_grasp [w=2.23] ∧ latent:temporal_sequence_seq_hands_first [w=2.20] ∧ NOT latent:hand_release [w=1.87] ∧ NOT latent:hand_open [w=1.60] ∧ latent:temporal_dynamics_duration_brief [w=1.09] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.85] ∧ latent:hand_circular [w=0.71] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.57] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.48] ∧ max_t[head_pitch] < -47.4 deg  (head_tilt_up) [w=0.46] ∧ NOT max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.45] ∧ NOT latent:temporal_direction_motion_reversible [w=0.45] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.41] ∧ NOT max_t[arms_crossed] > 0.676 L  (arm_cross) [w=0.37] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.36] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.35] ∧ NOT latent:hand_manipulate [w=0.34] ∧ NOT max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.33] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.31] | 0.000 | - |
| 1 | eat_meal | NOT latent:temporal_sequence_seq_simultaneous [w=1.83] ∧ NOT latent:temporal_dynamics_duration_sustained [w=1.20] ∧ NOT latent:hand_circular [w=1.09] ∧ latent:hand_manipulate [w=0.89] ∧ latent:hand_grasp [w=0.85] ∧ latent:temporal_dynamics_rhythm_regular [w=0.83] ∧ latent:temporal_sequence_seq_hands_first [w=0.72] ∧ NOT latent:hand_open [w=0.69] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.66] ∧ NOT max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.54] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.44] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.44] ∧ NOT max_t[elbow_max] > 174 deg  (arm_extend) [w=0.41] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.39] | 0.000 | - |
| 2 | brush_teeth | NOT latent:temporal_sequence_seq_simultaneous [w=2.24] ∧ latent:temporal_sequence_seq_hands_first [w=2.17] ∧ latent:hand_circular [w=1.67] ∧ latent:temporal_dynamics_rhythm_regular [w=1.03] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.68] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.68] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.62] ∧ NOT latent:foot_pivot [w=0.61] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.59] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.57] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.51] ∧ latent:hand_grasp [w=0.51] ∧ latent:temporal_dynamics_duration_sustained [w=0.50] ∧ NOT latent:hand_grasp [w=0.47] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.46] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.45] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.31] | 0.000 | - |
| 3 | brush_hair | NOT latent:hand_circular [w=2.34] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=1.87] ∧ NOT latent:hand_release [w=1.53] ∧ latent:temporal_dynamics_rhythm_regular [w=1.50] ∧ latent:hand_grasp [w=1.30] ∧ latent:temporal_sequence_seq_hands_first [w=1.15] ∧ latent:temporal_dynamics_duration_sustained [w=1.05] ∧ NOT latent:temporal_sequence_seq_body_first [w=1.00] ∧ NOT latent:hand_manipulate [w=0.97] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.75] ∧ latent:hand_open [w=0.59] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.55] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.55] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.45] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.45] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.43] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.41] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.40] ∧ NOT latent:temporal_dynamics_duration_sustained [w=0.39] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.34] | 0.000 | - |
| 4 | drop | NOT latent:temporal_dynamics_rhythm_regular [w=1.47] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.16] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=1.14] ∧ latent:hand_release [w=1.10] ∧ latent:temporal_dynamics_duration_brief [w=0.90] ∧ latent:temporal_dynamics_speed_accelerating [w=0.86] ∧ latent:temporal_sequence_seq_hands_first [w=0.74] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.73] ∧ NOT latent:temporal_dynamics_speed_accelerating [w=0.51] ∧ max_t[hand_vy_min] < -0.282 L/frame  (arm_lower) [w=0.42] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.41] ∧ NOT latent:head_rotate [w=0.37] ∧ NOT max_t[arms_crossed] > 0.676 L  (arm_cross) [w=0.36] ∧ NOT max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=0.31] | 0.000 | - |
| 5 | pick_up | latent:temporal_sequence_seq_body_first [w=1.87] ∧ latent:hand_grasp [w=1.58] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.13] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=1.01] ∧ latent:temporal_dynamics_duration_brief [w=0.94] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.93] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.64] ∧ NOT latent:temporal_direction_motion_reversible [w=0.64] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.63] ∧ max_t[knee_min] < 47.2 deg  (leg_squat) [w=0.42] | 0.000 | - |
| 6 | throw | latent:temporal_dynamics_speed_accelerating [w=1.33] ∧ latent:foot_pivot [w=1.18] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.06] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.82] ∧ latent:hand_release [w=0.77] ∧ latent:temporal_dynamics_duration_brief [w=0.75] ∧ max_t[shoulder_yaw_speed] > 14.4 deg/frame  (hip_rotate) [w=0.54] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.51] ∧ latent:temporal_sequence_seq_body_first [w=0.50] ∧ NOT latent:temporal_sequence_seq_hands_first [w=0.46] | 0.000 | - |
| 7 | sit_down | latent:temporal_dynamics_speed_decelerating [w=2.32] ∧ NOT latent:foot_pivot [w=1.62] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=1.10] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.98] ∧ latent:temporal_sequence_seq_body_first [w=0.90] ∧ NOT latent:foot_slide [w=0.89] ∧ latent:temporal_dynamics_duration_brief [w=0.84] ∧ mean_t[body_fwd_v] < -0.00115 L/frame  (x_walk_backward) [w=0.67] ∧ NOT mean_t[body_fwd_v] > -0.00115 L/frame  (x_walk_forward) [w=0.67] ∧ max_t[knee_min] < 47.2 deg  (leg_squat) [w=0.53] ∧ NOT mean_t[hip_height] < 0.585 L  (hip_static_sit) [w=0.38] ∧ NOT mean_t[hip_height] < 0.607 L  (leg_static_sit) [w=0.35] ∧ max_t[hip_height] < 0.389 L  (hip_sit) [w=0.35] ∧ max_t[hand_vy_min] < -0.282 L/frame  (arm_lower) [w=0.35] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.35] | 0.000 | - |
| 8 | stand_up | latent:temporal_dynamics_speed_accelerating [w=2.25] ∧ latent:temporal_sequence_seq_legs_first [w=1.51] ∧ latent:temporal_dynamics_duration_brief [w=1.00] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.89] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.68] ∧ max_t[hip_vy] > 0.144 L/frame  (hip_extend) [w=0.62] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.50] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.50] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.44] | 0.000 | - |
| 9 | clapping | NOT latent:hand_open [w=3.22] ∧ NOT latent:temporal_sequence_seq_body_first [w=1.75] ∧ NOT latent:temporal_sequence_seq_legs_first [w=1.34] ∧ latent:temporal_sequence_seq_simultaneous [w=1.34] ∧ latent:temporal_dynamics_duration_sustained [w=0.95] ∧ NOT latent:head_rotate [w=0.94] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.79] ∧ NOT latent:hand_grasp [w=0.78] ∧ latent:temporal_dynamics_rhythm_regular [w=0.72] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.70] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.66] ∧ max_t[hands_dist_v] < -0.372 L/frame  (temporal_direction_motion_converging) [w=0.60] ∧ NOT mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.56] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.41] ∧ NOT latent:temporal_sequence_seq_cascading [w=0.41] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.37] ∧ mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.31] ∧ NOT max_t[hands_dist_v] > 0.382 L/frame  (temporal_direction_motion_diverging) [w=0.30] | 0.000 | - |
| 10 | reading | latent:hand_grasp [w=2.34] ∧ latent:temporal_dynamics_duration_sustained [w=1.81] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.36] ∧ latent:hand_manipulate [w=1.08] ∧ latent:temporal_sequence_seq_hands_first [w=0.91] ∧ NOT latent:head_rotate [w=0.78] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.74] ∧ latent:temporal_dynamics_rhythm_irregular [w=0.70] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.64] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.61] ∧ NOT latent:temporal_direction_motion_reversible [w=0.54] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.46] ∧ mean_t[hip_height] < 0.607 L  (leg_static_sit) [w=0.42] ∧ mean_t[hip_height] < 0.585 L  (hip_static_sit) [w=0.42] ∧ NOT latent:hand_circular [w=0.35] | 0.000 | - |
| 11 | writing | latent:temporal_dynamics_rhythm_irregular [w=1.45] ∧ latent:temporal_dynamics_duration_sustained [w=1.27] ∧ latent:hand_manipulate [w=1.26] ∧ latent:hand_grasp [w=1.13] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=0.94] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.89] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.75] ∧ NOT latent:head_rotate [w=0.74] ∧ NOT latent:temporal_dynamics_speed_accelerating [w=0.71] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.68] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.53] ∧ latent:temporal_sequence_seq_hands_first [w=0.52] ∧ mean_t[hip_height] < 0.607 L  (leg_static_sit) [w=0.47] ∧ mean_t[hip_height] < 0.585 L  (hip_static_sit) [w=0.47] | 0.000 | - |
| 12 | tear_up_paper | NOT max_t[elbow_max] > 174 deg  (arm_extend) [w=1.97] ∧ latent:hand_grasp [w=1.95] ∧ NOT latent:temporal_sequence_seq_body_first [w=1.70] ∧ latent:temporal_sequence_seq_simultaneous [w=0.91] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.90] ∧ latent:temporal_dynamics_speed_decelerating [w=0.65] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.63] ∧ max_t[hands_dist_v] > 0.382 L/frame  (temporal_direction_motion_diverging) [w=0.58] ∧ mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.45] ∧ latent:temporal_dynamics_duration_brief [w=0.44] ∧ max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.43] ∧ NOT latent:temporal_direction_motion_reversible [w=0.39] ∧ NOT max_t[head_pitch_speed] > 32.5 deg/frame  (head_nod) [w=0.35] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.35] ∧ NOT max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.33] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.30] | 0.000 | - |
| 13 | put_on_jacket | latent:temporal_dynamics_duration_sustained [w=4.35] ∧ latent:hand_grasp [w=2.11] ∧ NOT latent:temporal_sequence_seq_body_first [w=1.27] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.04] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.94] ∧ latent:temporal_direction_motion_reversible [w=0.88] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.84] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.79] ∧ latent:temporal_sequence_seq_hands_first [w=0.52] ∧ max_t[hand_fwd_vmin] < -0.295 L/frame  (temporal_direction_motion_backward) [w=0.46] ∧ max_t[hand_above_head] > 0.326 L  (arm_overhead) [w=0.44] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.41] ∧ NOT max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.37] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.32] ∧ max_t[shoulder_yaw_speed] > 14.4 deg/frame  (hip_rotate) [w=0.31] | 0.000 | - |
| 14 | take_off_jacket | NOT latent:hand_circular [w=3.40] ∧ NOT latent:head_rotate [w=3.01] ∧ latent:temporal_direction_motion_reversible [w=2.04] ∧ latent:temporal_dynamics_duration_sustained [w=1.66] ∧ latent:hand_grasp [w=1.53] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=1.45] ∧ NOT latent:temporal_sequence_seq_hands_first [w=1.07] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.96] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.93] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.83] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.80] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.45] ∧ latent:temporal_sequence_seq_hands_first [w=0.41] ∧ NOT latent:temporal_dynamics_duration_sustained [w=0.39] ∧ NOT latent:hand_manipulate [w=0.32] | 0.000 | - |
| 15 | put_on_a_shoe | NOT latent:foot_pivot [w=2.09] ∧ latent:temporal_direction_motion_reversible [w=2.05] ∧ latent:hand_grasp [w=1.48] ∧ latent:temporal_sequence_seq_body_first [w=0.95] ∧ NOT latent:hand_open [w=0.74] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.63] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.60] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.59] ∧ NOT latent:temporal_dynamics_duration_sustained [w=0.55] ∧ mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.52] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.48] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.46] ∧ max_t[foot_height_diff] > 0.568 L  (leg_lift) [w=0.37] ∧ NOT mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.32] | 0.000 | - |
| 16 | take_off_a_shoe | NOT latent:temporal_sequence_seq_simultaneous [w=2.62] ∧ latent:hand_grasp [w=1.79] ∧ NOT latent:temporal_sequence_seq_hands_first [w=1.54] ∧ NOT latent:temporal_dynamics_duration_sustained [w=1.06] ∧ latent:temporal_direction_motion_reversible [w=1.03] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.93] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.75] ∧ NOT latent:temporal_dynamics_speed_accelerating [w=0.66] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.66] ∧ NOT latent:hand_manipulate [w=0.63] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.61] ∧ NOT latent:hand_release [w=0.58] ∧ latent:temporal_sequence_seq_body_first [w=0.57] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.56] ∧ max_t[foot_height_diff] > 0.568 L  (leg_lift) [w=0.44] ∧ latent:temporal_sequence_seq_hands_first [w=0.43] ∧ mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.34] | 0.000 | - |
| 17 | put_on_glasses | NOT latent:temporal_dynamics_speed_accelerating [w=5.17] ∧ NOT latent:foot_pivot [w=3.05] ∧ latent:temporal_direction_motion_reversible [w=1.32] ∧ latent:temporal_sequence_seq_hands_first [w=1.18] ∧ latent:hand_grasp [w=0.89] ∧ latent:temporal_dynamics_duration_brief [w=0.78] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.72] ∧ NOT latent:head_rotate [w=0.70] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.59] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.57] ∧ mean_t[head_speed] < -0.0195 L/frame  (head_static) [w=0.56] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.53] ∧ latent:hand_circular [w=0.44] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.44] ∧ NOT max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=0.37] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.36] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.34] ∧ max_t[hand_vy_min] < -0.282 L/frame  (arm_lower) [w=0.32] ∧ NOT latent:hand_open [w=0.31] | 0.000 | - |
| 18 | take_off_glasses | NOT latent:hand_circular [w=1.36] ∧ latent:temporal_dynamics_duration_brief [w=1.27] ∧ latent:temporal_sequence_seq_hands_first [w=1.13] ∧ latent:temporal_direction_motion_reversible [w=1.01] ∧ NOT latent:temporal_direction_motion_reversible [w=0.98] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=0.94] ∧ mean_t[head_speed] < -0.0195 L/frame  (head_static) [w=0.79] ∧ NOT latent:foot_pivot [w=0.68] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.54] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.52] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.51] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.51] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.49] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.48] ∧ latent:hand_grasp [w=0.47] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.42] ∧ before(arm_raise, hand_grasp) [w=0.32] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.31] | 0.000 | - |
| 19 | put_on_a_hat_cap | NOT latent:temporal_sequence_seq_body_first [w=2.78] ∧ NOT latent:hand_circular [w=2.43] ∧ latent:temporal_direction_motion_reversible [w=1.74] ∧ latent:temporal_sequence_seq_hands_first [w=1.33] ∧ latent:temporal_dynamics_duration_brief [w=1.33] ∧ NOT latent:foot_pivot [w=1.25] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.05] ∧ latent:temporal_dynamics_speed_accelerating [w=1.04] ∧ NOT latent:temporal_sequence_seq_cascading [w=0.92] ∧ latent:hand_grasp [w=0.89] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.81] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.77] ∧ NOT latent:arm_circular [w=0.37] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.36] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.32] | 0.088 | 1.000 |
| 20 | take_off_a_hat_cap | latent:temporal_direction_motion_reversible [w=2.14] ∧ latent:foot_pivot [w=1.36] ∧ latent:temporal_dynamics_duration_brief [w=1.12] ∧ latent:temporal_sequence_seq_hands_first [w=1.11] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=1.05] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.86] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.81] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.72] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.47] ∧ latent:hand_grasp [w=0.46] ∧ max_t[hand_above_head] > 0.326 L  (arm_overhead) [w=0.40] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.37] ∧ max_t[head_pitch] < -47.4 deg  (head_tilt_up) [w=0.31] | 0.000 | - |
| 21 | cheer_up | latent:hand_open [w=2.42] ∧ NOT latent:temporal_sequence_seq_body_first [w=2.05] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.91] ∧ latent:temporal_sequence_seq_simultaneous [w=0.90] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.89] ∧ max_t[hand_above_head] > 0.326 L  (arm_overhead) [w=0.81] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.75] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.64] ∧ NOT max_t[head_pitch] < -47.4 deg  (head_tilt_up) [w=0.52] ∧ latent:temporal_dynamics_duration_brief [w=0.48] ∧ NOT max_t[hip_vy] > 0.144 L/frame  (hip_extend) [w=0.42] ∧ max_t[feet_lift] > 0.532 L  (leg_jump) [w=0.32] ∧ max_t[feet_lift] > 0.532 L  (foot_jump) [w=0.32] ∧ max_t[head_pitch] < -47.4 deg  (head_tilt_up) [w=0.30] | 0.000 | - |
| 22 | hand_waving | max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=1.88] ∧ NOT latent:hand_grasp [w=1.68] ∧ latent:temporal_dynamics_rhythm_regular [w=1.56] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.30] ∧ latent:temporal_sequence_seq_hands_first [w=1.23] ∧ latent:foot_pivot [w=1.01] ∧ NOT max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.97] ∧ latent:temporal_dynamics_duration_sustained [w=0.87] ∧ NOT latent:temporal_direction_motion_reversible [w=0.47] ∧ NOT max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.46] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.46] ∧ NOT latent:head_rotate [w=0.43] ∧ max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.41] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.38] ∧ max_t[raised_hand_speed] > 0.249 L/frame  (hand_wave) [w=0.34] | 0.000 | - |
| 23 | kicking_something | NOT latent:temporal_dynamics_speed_accelerating [w=1.81] ∧ latent:temporal_sequence_seq_legs_first [w=1.61] ∧ NOT latent:temporal_sequence_seq_body_first [w=1.26] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.96] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.94] ∧ latent:temporal_dynamics_duration_brief [w=0.85] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.73] ∧ max_t[shoulder_yaw_speed] > 14.4 deg/frame  (hip_rotate) [w=0.64] ∧ NOT mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.57] ∧ max_t[foot_height_diff] > 0.59 L  (leg_kick) [w=0.49] ∧ max_t[foot_height_diff] > 0.59 L  (foot_kick) [w=0.49] ∧ max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.45] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.35] ∧ mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.34] | 0.000 | - |
| 24 | reach_into_pocket | NOT max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=4.45] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=1.82] ∧ NOT latent:hand_manipulate [w=1.37] ∧ NOT latent:temporal_sequence_seq_body_first [w=1.27] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.18] ∧ latent:hand_grasp [w=1.14] ∧ NOT latent:hand_circular [w=1.08] ∧ NOT latent:temporal_sequence_seq_hands_first [w=1.03] ∧ latent:temporal_sequence_seq_hands_first [w=0.98] ∧ NOT latent:foot_pivot [w=0.84] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.82] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.70] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=0.64] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.64] ∧ NOT latent:hand_grasp [w=0.52] ∧ max_t[hand_chest_v] < -0.217 L/frame  (arm_retract) [w=0.45] ∧ latent:temporal_dynamics_duration_brief [w=0.42] ∧ NOT latent:temporal_sequence_seq_legs_first [w=0.33] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.32] | 0.000 | - |
| 25 | hopping | latent:temporal_dynamics_rhythm_regular [w=2.50] ∧ latent:temporal_sequence_seq_legs_first [w=2.35] ∧ latent:temporal_dynamics_duration_sustained [w=0.70] ∧ NOT latent:temporal_sequence_seq_alternating [w=0.60] ∧ max_t[wrist_speed] > 0.524 L/frame  (arm_swing) [w=0.43] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.42] ∧ mean_t[leg_alt_speed] > 0.117 L/frame  (leg_alternating) [w=0.42] ∧ NOT max_t[hip_vy] > 0.144 L/frame  (hip_extend) [w=0.31] ∧ max_t[hip_vy] > 0.144 L/frame  (hip_extend) [w=0.31] | 0.000 | - |
| 26 | jump_up | latent:temporal_sequence_seq_simultaneous [w=2.22] ∧ latent:temporal_dynamics_duration_brief [w=2.12] ∧ NOT latent:hand_circular [w=1.92] ∧ NOT latent:temporal_sequence_seq_hands_first [w=1.59] ∧ NOT max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=1.16] ∧ max_t[hip_vy] > 0.144 L/frame  (hip_extend) [w=0.78] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.72] ∧ NOT latent:temporal_sequence_seq_cascading [w=0.62] ∧ NOT max_t[knee_min] < 29.9 deg  (hip_squat) [w=0.53] ∧ latent:temporal_sequence_seq_legs_first [w=0.47] ∧ max_t[feet_lift] > 0.532 L  (leg_jump) [w=0.47] ∧ max_t[feet_lift] > 0.532 L  (foot_jump) [w=0.47] ∧ NOT max_t[knee_min] < 47.2 deg  (leg_squat) [w=0.43] ∧ NOT max_t[hip_height] < 0.389 L  (hip_sit) [w=0.39] ∧ NOT max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.35] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.30] | 0.000 | - |
| 27 | phone_call | NOT latent:temporal_sequence_seq_simultaneous [w=1.21] ∧ latent:temporal_sequence_seq_hands_first [w=1.17] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.86] ∧ NOT latent:temporal_dynamics_duration_sustained [w=0.69] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.67] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.65] ∧ NOT latent:temporal_direction_motion_reversible [w=0.64] ∧ NOT max_t[hand_above_head] > 0.326 L  (arm_overhead) [w=0.63] ∧ latent:hand_manipulate [w=0.62] ∧ NOT latent:foot_pivot [w=0.57] ∧ NOT latent:hand_circular [w=0.49] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.49] ∧ latent:temporal_dynamics_duration_sustained [w=0.48] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.48] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.45] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.45] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.42] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.39] ∧ latent:head_rotate [w=0.39] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.37] ∧ latent:hand_grasp [w=0.36] | 0.000 | - |
| 28 | play_with_phone_tablet | NOT latent:temporal_sequence_seq_alternating [w=2.30] ∧ latent:hand_manipulate [w=2.06] ∧ latent:temporal_dynamics_rhythm_irregular [w=1.72] ∧ NOT latent:head_rotate [w=0.98] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.89] ∧ NOT max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.73] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.67] ∧ latent:temporal_sequence_seq_hands_first [w=0.62] ∧ NOT latent:temporal_sequence_seq_hands_first [w=0.55] ∧ latent:temporal_dynamics_duration_sustained [w=0.53] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.49] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.47] ∧ NOT max_t[knee_min] < 47.2 deg  (leg_squat) [w=0.44] ∧ latent:hand_grasp [w=0.35] ∧ mean_t[hip_height] < 0.585 L  (hip_static_sit) [w=0.33] ∧ mean_t[hip_height] < 0.607 L  (leg_static_sit) [w=0.33] | 0.000 | - |
| 29 | type_on_a_keyboard | latent:temporal_sequence_seq_alternating [w=1.48] ∧ latent:hand_manipulate [w=1.40] ∧ latent:temporal_dynamics_duration_sustained [w=1.22] ∧ latent:temporal_dynamics_rhythm_irregular [w=1.03] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.98] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.88] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.79] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.71] ∧ mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.65] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.55] ∧ NOT max_t[elbow_max] > 174 deg  (arm_extend) [w=0.41] ∧ mean_t[hip_height] < 0.607 L  (leg_static_sit) [w=0.35] ∧ mean_t[hip_height] < 0.585 L  (hip_static_sit) [w=0.35] | 0.000 | - |
| 30 | point_to_something | latent:temporal_dynamics_duration_brief [w=1.92] ∧ latent:temporal_sequence_seq_hands_first [w=1.72] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=1.05] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.98] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.68] ∧ NOT latent:arm_circular [w=0.67] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.63] ∧ NOT latent:temporal_sequence_seq_cascading [w=0.61] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.54] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.49] ∧ latent:foot_pivot [w=0.46] ∧ NOT latent:temporal_direction_motion_reversible [w=0.40] ∧ NOT max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.38] ∧ max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=0.37] ∧ NOT max_t[hand_fwd_vmin] < -0.295 L/frame  (temporal_direction_motion_backward) [w=0.32] | 0.000 | - |
| 31 | taking_a_selfie | NOT latent:temporal_direction_motion_reversible [w=2.68] ∧ latent:temporal_sequence_seq_hands_first [w=2.65] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=1.90] ∧ latent:temporal_dynamics_duration_brief [w=1.24] ∧ NOT max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=1.18] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.05] ∧ latent:hand_grasp [w=0.97] ∧ NOT latent:hand_release [w=0.92] ∧ NOT latent:hand_open [w=0.68] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.64] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.60] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.58] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.55] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.53] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.49] ∧ max_t[head_pitch] < -47.4 deg  (head_tilt_up) [w=0.45] | 0.004 | 1.000 |
| 32 | check_time_(from_watch) | NOT latent:temporal_dynamics_rhythm_regular [w=3.44] ∧ NOT latent:temporal_sequence_seq_body_first [w=2.41] ∧ NOT latent:foot_pivot [w=1.59] ∧ latent:temporal_dynamics_duration_brief [w=1.30] ∧ NOT latent:hand_open [w=1.16] ∧ latent:temporal_sequence_seq_hands_first [w=1.05] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.88] ∧ NOT latent:temporal_dynamics_speed_accelerating [w=0.81] ∧ NOT max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.81] ∧ latent:hand_circular [w=0.63] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.60] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.57] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.56] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.56] ∧ NOT latent:head_rotate [w=0.54] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.52] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.39] ∧ max_t[hand_chest_v] < -0.217 L/frame  (arm_retract) [w=0.36] ∧ latent:temporal_sequence_seq_simultaneous [w=0.35] | 0.000 | - |
| 33 | rub_two_hands | latent:hand_circular [w=1.77] ∧ latent:temporal_sequence_seq_simultaneous [w=1.17] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.80] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.79] ∧ latent:temporal_dynamics_rhythm_regular [w=0.77] ∧ latent:temporal_dynamics_duration_sustained [w=0.66] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.65] ∧ max_t[hands_dist_v] < -0.372 L/frame  (temporal_direction_motion_converging) [w=0.59] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.52] ∧ NOT latent:temporal_sequence_seq_hands_first [w=0.49] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.49] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.46] | 0.000 | - |
| 34 | nod_head_bow | NOT latent:foot_slide [w=5.20] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=3.62] ∧ NOT latent:hand_open [w=2.60] ∧ latent:temporal_sequence_seq_body_first [w=1.54] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=1.39] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=0.99] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.93] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.91] ∧ latent:temporal_dynamics_duration_brief [w=0.77] ∧ NOT latent:temporal_direction_motion_reversible [w=0.70] ∧ latent:temporal_dynamics_rhythm_regular [w=0.70] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.58] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.43] ∧ max_t[head_pitch_speed] > 32.5 deg/frame  (head_nod) [w=0.43] ∧ max_t[hand_chest_v] < -0.217 L/frame  (arm_retract) [w=0.41] ∧ NOT mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.36] ∧ NOT latent:temporal_sequence_seq_legs_first [w=0.33] | 0.000 | - |
| 35 | shake_head | NOT max_t[hands_dist] < -0.173 L  (hand_clap) [w=2.99] ∧ latent:temporal_dynamics_rhythm_regular [w=1.85] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=1.20] ∧ latent:temporal_dynamics_duration_brief [w=0.94] ∧ NOT latent:hand_grasp [w=0.93] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.91] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.78] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.78] ∧ latent:temporal_dynamics_rhythm_irregular [w=0.77] ∧ max_t[head_lat_speed] > 0.0818 L/frame  (head_shake) [w=0.71] ∧ NOT latent:temporal_sequence_seq_hands_first [w=0.59] ∧ latent:temporal_sequence_seq_hands_first [w=0.57] ∧ NOT max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.56] ∧ NOT latent:temporal_sequence_seq_cascading [w=0.36] | 0.000 | - |
| 36 | wipe_face | NOT latent:foot_pivot [w=1.88] ∧ latent:temporal_dynamics_rhythm_regular [w=1.26] ∧ latent:hand_open [w=1.13] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.89] ∧ latent:temporal_sequence_seq_hands_first [w=0.76] ∧ NOT latent:temporal_sequence_seq_hands_first [w=0.73] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.72] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.71] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.68] ∧ latent:hand_grasp [w=0.66] ∧ latent:temporal_dynamics_rhythm_irregular [w=0.59] ∧ latent:temporal_dynamics_duration_brief [w=0.51] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.48] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.36] ∧ latent:temporal_sequence_seq_body_first [w=0.32] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.31] | 0.000 | - |
| 37 | salute | NOT latent:temporal_sequence_seq_body_first [w=2.33] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=2.21] ∧ latent:temporal_sequence_seq_hands_first [w=1.80] ∧ latent:temporal_dynamics_duration_brief [w=1.60] ∧ latent:hand_open [w=1.48] ∧ NOT latent:head_rotate [w=1.41] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=1.26] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.24] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.75] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.69] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.63] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.53] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.48] ∧ NOT max_t[arms_crossed] > 0.676 L  (arm_cross) [w=0.30] | 0.011 | 0.333 |
| 38 | put_palms_together | NOT latent:hand_release [w=4.13] ∧ latent:temporal_sequence_seq_simultaneous [w=1.54] ∧ NOT latent:foot_pivot [w=1.19] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=1.12] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.08] ∧ latent:temporal_dynamics_duration_brief [w=1.06] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=1.05] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.93] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.63] ∧ max_t[hands_dist_v] < -0.372 L/frame  (temporal_direction_motion_converging) [w=0.61] ∧ NOT latent:hand_manipulate [w=0.60] ∧ NOT latent:temporal_dynamics_speed_accelerating [w=0.45] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.33] ∧ NOT max_t[head_lat_speed] > 0.0818 L/frame  (head_shake) [w=0.30] | 0.000 | - |
| 39 | cross_hands_in_front | NOT latent:temporal_dynamics_speed_decelerating [w=3.00] ∧ latent:temporal_dynamics_duration_brief [w=2.56] ∧ latent:temporal_sequence_seq_simultaneous [w=2.14] ∧ NOT latent:hand_manipulate [w=1.54] ∧ NOT latent:hand_release [w=1.35] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=1.23] ∧ NOT max_t[hands_dist] < -0.173 L  (hand_clap) [w=1.13] ∧ max_t[arms_crossed] > 0.676 L  (arm_cross) [w=0.96] ∧ latent:hand_grasp [w=0.75] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.58] ∧ max_t[hand_above_head] > 0.326 L  (arm_overhead) [w=0.54] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.52] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.50] ∧ NOT max_t[shoulder_yaw_speed] > 14.4 deg/frame  (hip_rotate) [w=0.48] ∧ NOT max_t[hip_lat_speed] > 0.207 L/frame  (hip_sway) [w=0.37] | 0.000 | - |
| 40 | sneeze_cough | latent:temporal_sequence_seq_body_first [w=1.53] ∧ latent:temporal_dynamics_duration_brief [w=1.17] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.09] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.78] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.74] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.72] ∧ NOT latent:hand_grasp [w=0.68] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=0.68] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.56] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.47] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.44] ∧ NOT max_t[hand_above_head] > 0.326 L  (arm_overhead) [w=0.37] ∧ latent:temporal_direction_motion_reversible [w=0.36] ∧ NOT max_t[arms_crossed] > 0.676 L  (arm_cross) [w=0.34] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.33] | 0.000 | - |
| 41 | staggering | latent:temporal_dynamics_rhythm_irregular [w=2.29] ∧ NOT latent:temporal_sequence_seq_hands_first [w=1.27] ∧ latent:temporal_sequence_seq_legs_first [w=1.23] ∧ latent:temporal_dynamics_duration_sustained [w=0.97] ∧ mean_t[leg_alt_speed] > 0.117 L/frame  (leg_alternating) [w=0.95] ∧ mean_t[body_speed] < -0.0425 L/frame  (temporal_dynamics_speed_slow) [w=0.85] ∧ max_t[hip_lat_speed] > 0.207 L/frame  (hip_sway) [w=0.73] ∧ max_t[ankle_speed] > 0.321 L/frame  (foot_step) [w=0.48] ∧ max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.40] ∧ NOT mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.38] ∧ NOT latent:hand_open [w=0.37] ∧ mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.36] | 0.000 | - |
| 42 | falling_down | latent:foot_slide [w=2.27] ∧ latent:temporal_dynamics_speed_accelerating [w=1.52] ∧ latent:temporal_sequence_seq_body_first [w=1.14] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.84] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.84] ∧ latent:temporal_dynamics_duration_brief [w=0.68] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.59] ∧ max_t[knee_min] < 47.2 deg  (leg_squat) [w=0.40] ∧ NOT latent:hand_grasp [w=0.35] | 0.000 | - |
| 43 | headache | NOT latent:hand_release [w=3.56] ∧ NOT latent:temporal_dynamics_speed_accelerating [w=2.41] ∧ NOT latent:foot_pivot [w=2.11] ∧ latent:temporal_sequence_seq_hands_first [w=1.36] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.99] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.96] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.86] ∧ NOT latent:hand_manipulate [w=0.84] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.78] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.73] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.70] ∧ max_t[hand_vy_max] > 0.267 L/frame  (temporal_direction_motion_upward) [w=0.70] ∧ NOT latent:temporal_sequence_seq_cascading [w=0.64] ∧ NOT latent:arm_circular [w=0.63] ∧ NOT latent:temporal_dynamics_duration_sustained [w=0.58] ∧ NOT latent:temporal_direction_motion_reversible [w=0.56] ∧ NOT mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.54] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.47] ∧ NOT max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.45] ∧ latent:temporal_dynamics_duration_sustained [w=0.40] | 0.000 | - |
| 44 | chest_pain | latent:temporal_dynamics_duration_sustained [w=1.36] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.97] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.92] ∧ max_t[hand_chest_v] < -0.217 L/frame  (arm_retract) [w=0.78] ∧ latent:temporal_sequence_seq_hands_first [w=0.76] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.76] ∧ NOT latent:temporal_sequence_seq_hands_first [w=0.76] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.67] ∧ NOT latent:hand_open [w=0.66] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.61] ∧ latent:arm_circular [w=0.55] ∧ NOT max_t[hand_chest_v] < -0.217 L/frame  (arm_retract) [w=0.55] ∧ NOT latent:hand_grasp [w=0.54] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.44] ∧ latent:temporal_sequence_seq_body_first [w=0.43] ∧ NOT latent:hand_manipulate [w=0.36] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=0.32] | 0.000 | - |
| 45 | back_pain | NOT latent:hand_manipulate [w=1.82] ∧ NOT max_t[hands_dist] < -0.173 L  (hand_clap) [w=1.66] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.65] ∧ latent:temporal_dynamics_duration_sustained [w=1.62] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.51] ∧ NOT latent:hand_circular [w=1.50] ∧ NOT max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=1.45] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=1.38] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=1.29] ∧ latent:temporal_sequence_seq_hands_first [w=1.19] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.99] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.79] ∧ NOT latent:temporal_sequence_seq_cascading [w=0.77] ∧ NOT max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.68] ∧ NOT latent:arm_circular [w=0.62] ∧ NOT max_t[hand_fwd_vmin] < -0.295 L/frame  (temporal_direction_motion_backward) [w=0.53] ∧ NOT latent:foot_pivot [w=0.48] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.44] ∧ NOT latent:hand_grasp [w=0.35] ∧ max_t[hand_chest_v] < -0.217 L/frame  (arm_retract) [w=0.31] | 0.000 | - |
| 46 | neck_pain | latent:head_rotate [w=2.11] ∧ NOT latent:hand_manipulate [w=1.61] ∧ latent:temporal_sequence_seq_hands_first [w=1.51] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.87] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.87] ∧ latent:temporal_dynamics_duration_sustained [w=0.79] ∧ latent:temporal_dynamics_rhythm_irregular [w=0.76] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.69] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.44] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.35] | 0.018 | 1.000 |
| 47 | nausea_vomiting | latent:temporal_sequence_seq_body_first [w=3.13] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=1.08] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.06] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.99] ∧ latent:temporal_dynamics_duration_brief [w=0.63] ∧ max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.56] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.54] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.52] ∧ latent:temporal_dynamics_duration_sustained [w=0.44] ∧ max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=0.44] ∧ max_t[head_pitch] > 58.9 deg  (head_tilt_down) [w=0.40] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.38] ∧ NOT latent:temporal_dynamics_duration_brief [w=0.33] | 0.000 | - |
| 48 | fan_self | NOT latent:temporal_dynamics_rhythm_irregular [w=2.79] ∧ NOT latent:head_rotate [w=2.30] ∧ latent:temporal_dynamics_duration_sustained [w=2.11] ∧ NOT latent:temporal_sequence_seq_body_first [w=2.03] ∧ latent:temporal_dynamics_rhythm_regular [w=1.56] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=1.55] ∧ NOT latent:foot_pivot [w=1.39] ∧ NOT latent:temporal_sequence_seq_cascading [w=1.14] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.99] ∧ latent:temporal_sequence_seq_hands_first [w=0.91] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.81] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.81] ∧ max_t[raised_hand_speed] > 0.249 L/frame  (hand_wave) [w=0.59] ∧ NOT latent:temporal_sequence_seq_legs_first [w=0.55] ∧ NOT max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=0.51] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=0.43] ∧ NOT latent:arm_circular [w=0.43] ∧ mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.40] ∧ NOT latent:temporal_sequence_seq_alternating [w=0.37] | 0.000 | - |
| 49 | punch_slap | latent:hand_fist [w=1.64] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.22] ∧ latent:temporal_dynamics_duration_brief [w=0.89] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.86] ∧ NOT max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.82] ∧ latent:temporal_dynamics_speed_accelerating [w=0.78] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=0.64] ∧ latent:foot_pivot [w=0.61] ∧ NOT latent:temporal_direction_motion_reversible [w=0.58] ∧ latent:temporal_sequence_seq_body_first [w=0.50] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.50] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.49] ∧ max_t[wrist_speed] > 0.524 L/frame  (arm_swing) [w=0.36] | 0.000 | - |
| 50 | kicking | latent:temporal_dynamics_duration_brief [w=1.60] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=1.57] ∧ latent:temporal_sequence_seq_legs_first [w=1.56] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=1.39] ∧ NOT latent:hand_grasp [w=1.15] ∧ NOT max_t[hand_reach_fwd] > 0.964 L  (hand_point) [w=0.80] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=0.68] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.59] ∧ mean_t[body_speed] > 0.11 L/frame  (temporal_dynamics_speed_fast) [w=0.56] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=0.48] ∧ NOT max_t[knee_min] < 47.2 deg  (leg_squat) [w=0.47] ∧ max_t[shoulder_yaw_speed] > 14.4 deg/frame  (hip_rotate) [w=0.45] ∧ NOT max_t[knee_min] < 29.9 deg  (hip_squat) [w=0.45] ∧ NOT mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.41] ∧ mean_t[foot_height_diff] > 0.404 L  (leg_balance) [w=0.38] ∧ max_t[hand_lateral] > 0.678 L  (arm_lateral) [w=0.35] | 0.000 | - |
| 51 | pushing | latent:foot_pivot [w=1.58] ∧ latent:temporal_dynamics_speed_decelerating [w=1.50] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=1.10] ∧ NOT latent:hand_fist [w=1.06] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.84] ∧ latent:hand_open [w=0.79] ∧ latent:temporal_dynamics_duration_brief [w=0.76] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.72] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=0.52] ∧ max_t[shoulder_yaw_speed] > 14.4 deg/frame  (hip_rotate) [w=0.42] ∧ latent:temporal_sequence_seq_body_first [w=0.40] | 0.000 | - |
| 52 | pat_on_back | NOT latent:temporal_dynamics_rhythm_irregular [w=2.70] ∧ latent:temporal_sequence_seq_hands_first [w=1.58] ∧ latent:hand_open [w=1.50] ∧ NOT max_t[hand_head] < 0.486 L  (hand_lift_to_face) [w=1.25] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.09] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.91] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=0.90] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.90] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.88] ∧ NOT latent:temporal_direction_motion_reversible [w=0.71] ∧ latent:temporal_dynamics_duration_sustained [w=0.69] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.63] ∧ NOT latent:temporal_sequence_seq_alternating [w=0.61] ∧ latent:temporal_dynamics_rhythm_regular [w=0.55] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.55] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.53] ∧ mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.52] ∧ NOT latent:hand_grasp [w=0.48] | 0.000 | - |
| 53 | point_finger | latent:temporal_dynamics_duration_brief [w=2.71] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=1.87] ∧ NOT latent:temporal_sequence_seq_cascading [w=1.61] ∧ latent:temporal_sequence_seq_hands_first [w=1.45] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=1.31] ∧ NOT latent:temporal_sequence_seq_legs_first [w=1.28] ∧ NOT latent:temporal_direction_motion_reversible [w=1.28] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.21] ∧ NOT latent:arm_circular [w=1.05] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.98] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.98] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.85] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=0.66] ∧ NOT max_t[hand_fwd_vmin] < -0.295 L/frame  (temporal_direction_motion_backward) [w=0.47] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.42] ∧ max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.37] ∧ latent:hand_open [w=0.32] ∧ NOT latent:hand_grasp [w=0.31] ∧ NOT latent:head_rotate [w=0.30] | 0.000 | - |
| 54 | hugging | latent:temporal_dynamics_duration_sustained [w=2.66] ∧ NOT latent:hand_manipulate [w=2.51] ∧ latent:temporal_sequence_seq_simultaneous [w=1.51] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=1.25] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=1.17] ∧ NOT latent:hand_circular [w=1.09] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=1.04] ∧ NOT mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.89] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.82] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.82] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.76] ∧ NOT max_t[arms_crossed] > 0.676 L  (arm_cross) [w=0.67] ∧ max_t[hands_dist_v] < -0.372 L/frame  (temporal_direction_motion_converging) [w=0.65] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.41] ∧ NOT max_t[hands_dist] < -0.173 L  (hand_clap) [w=0.38] ∧ mean_t[elbow_min] < 54.3 deg  (arm_hold) [w=0.34] | 0.000 | - |
| 55 | giving_object | latent:hand_release [w=1.87] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=1.39] ∧ NOT latent:hand_manipulate [w=1.26] ∧ latent:hand_grasp [w=1.18] ∧ latent:temporal_dynamics_duration_brief [w=1.03] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.94] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.67] ∧ latent:temporal_sequence_seq_hands_first [w=0.67] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.62] ∧ NOT max_t[torso_pitch] > 40.6 deg  (hip_bend) [w=0.57] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.52] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=0.52] ∧ NOT max_t[hands_dist] < -0.173 L  (hand_clap) [w=0.51] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=0.49] | 0.000 | - |
| 56 | touch_pocket | NOT latent:temporal_dynamics_rhythm_irregular [w=4.31] ∧ NOT latent:temporal_sequence_seq_simultaneous [w=1.36] ∧ NOT latent:temporal_dynamics_duration_brief [w=1.23] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.96] ∧ latent:temporal_sequence_seq_hands_first [w=0.95] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.93] ∧ NOT latent:temporal_dynamics_rhythm_regular [w=0.93] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=0.86] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=0.67] ∧ max_t[hand_body] < 0.00488 L  (hand_touch) [w=0.58] ∧ latent:temporal_dynamics_duration_brief [w=0.56] ∧ latent:hand_grasp [w=0.54] ∧ NOT latent:hand_manipulate [w=0.45] ∧ max_t[hand_vy_min] < -0.241 L/frame  (temporal_direction_motion_downward) [w=0.40] ∧ max_t[hand_vy_min] < -0.282 L/frame  (arm_lower) [w=0.40] ∧ NOT max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.37] ∧ NOT mean_t[head_speed] < -0.0195 L/frame  (head_static) [w=0.31] | 0.000 | - |
| 57 | shaking_hands | latent:hand_grasp [w=1.80] ∧ latent:temporal_dynamics_duration_brief [w=1.67] ∧ mean_t[ankle_speed] < 0.148 L/frame  (foot_static) [w=1.23] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=1.20] ∧ latent:temporal_dynamics_rhythm_regular [w=1.17] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=1.10] ∧ max_t[elbow_max] > 174 deg  (arm_extend) [w=1.00] ∧ mean_t[arm_sym] < -0.36 cos  (arm_alternating) [w=0.86] ∧ latent:temporal_sequence_seq_simultaneous [w=0.86] ∧ mean_t[hip_height] > 0.891 L  (hip_static_stand) [w=0.73] ∧ mean_t[hip_height] > 0.704 L  (leg_static_stand) [w=0.71] ∧ max_t[hand_fwd_vmax] > 0.267 L/frame  (temporal_direction_motion_forward) [w=0.50] ∧ NOT max_t[hand_above_shoulder] > 0.0724 L  (arm_raise) [w=0.32] | 0.000 | - |
| 58 | walking_towards | latent:temporal_sequence_seq_alternating [w=3.36] ∧ NOT latent:temporal_dynamics_speed_decelerating [w=2.59] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=0.93] ∧ latent:temporal_dynamics_rhythm_regular [w=0.92] ∧ mean_t[leg_alt_speed] > 0.117 L/frame  (leg_alternating) [w=0.73] ∧ max_t[hip_lat_speed] > 0.207 L/frame  (hip_sway) [w=0.69] ∧ NOT latent:foot_pivot [w=0.53] ∧ NOT latent:temporal_sequence_seq_body_first [w=0.52] ∧ NOT max_t[person_dist_v] > -0.115 L/frame  (x_leave_partner) [w=0.43] ∧ max_t[wrist_speed] > 0.524 L/frame  (arm_swing) [w=0.36] ∧ latent:temporal_dynamics_duration_sustained [w=0.32] ∧ max_t[ankle_speed] > 0.321 L/frame  (foot_step) [w=0.31] ∧ max_t[ankle_speed] > 0.34 L/frame  (leg_step) [w=0.30] | 0.000 | - |
| 59 | walking_apart | latent:temporal_sequence_seq_alternating [w=1.93] ∧ latent:temporal_dynamics_duration_sustained [w=1.38] ∧ mean_t[second_present] > 0.502 frac  (interaction_two_person) [w=1.11] ∧ mean_t[leg_alt_speed] > 0.117 L/frame  (leg_alternating) [w=0.82] ∧ NOT latent:temporal_sequence_seq_hands_first [w=0.64] ∧ NOT latent:hand_grasp [w=0.58] ∧ latent:temporal_dynamics_rhythm_regular [w=0.55] ∧ max_t[wrist_speed] > 0.524 L/frame  (arm_swing) [w=0.52] ∧ max_t[hip_lat_speed] > 0.207 L/frame  (hip_sway) [w=0.47] ∧ latent:foot_pivot [w=0.43] ∧ max_t[ankle_speed] > 0.34 L/frame  (leg_step) [w=0.42] ∧ max_t[ankle_speed] > 0.321 L/frame  (foot_step) [w=0.42] ∧ NOT max_t[person_dist_v] < 0.122 L/frame  (x_approach_partner) [w=0.35] ∧ NOT latent:temporal_dynamics_rhythm_irregular [w=0.33] | 0.000 | - |

## 4. Knowledge revision

- CSV literals confirmed: 551
- CSV literals removed: 47
- new positive literals: 42 (of which x_* extras: 1)
- new negative literals: 311

New positive literals:

- drink_water: hand_circular (w=0.71)
- brush_hair: hand_open (w=0.59)
- sit_down: x_walk_backward (w=0.67)
- stand_up: hip_bend (w=0.44)
- reading: hand_manipulate (w=1.08)
- reading: temporal_dynamics_rhythm_irregular (w=0.70)
- tear_up_paper: temporal_dynamics_speed_decelerating (w=0.65)
- put_on_jacket: hip_rotate (w=0.31)
- take_off_a_shoe: temporal_sequence_seq_body_first (w=0.57)
- put_on_glasses: hand_circular (w=0.44)
- put_on_glasses: arm_lower (w=0.32)
- put_on_glasses: temporal_direction_motion_downward (w=0.34)
- take_off_glasses: temporal_direction_motion_upward (w=0.49)
- put_on_a_hat_cap: temporal_dynamics_speed_accelerating (w=1.04)
- take_off_a_hat_cap: foot_pivot (w=1.36)
- hand_waving: foot_pivot (w=1.01)
- jump_up: temporal_sequence_seq_legs_first (w=0.47)
- phone_call: head_rotate (w=0.39)
- phone_call: hand_manipulate (w=0.62)
- phone_call: arm_extend (w=0.39)
- point_to_something: arm_raise (w=1.05)
- point_to_something: foot_pivot (w=0.46)
- check_time_(from_watch): hand_lift_to_face (w=0.52)
- check_time_(from_watch): hand_circular (w=0.63)
- check_time_(from_watch): temporal_sequence_seq_simultaneous (w=0.35)
- nod_head_bow: arm_retract (w=0.41)
- shake_head: temporal_dynamics_rhythm_irregular (w=0.77)
- wipe_face: hand_grasp (w=0.66)
- wipe_face: temporal_sequence_seq_body_first (w=0.32)
- wipe_face: temporal_dynamics_rhythm_irregular (w=0.59)
- cross_hands_in_front: arm_raise (w=1.23)
- cross_hands_in_front: arm_overhead (w=0.54)
- sneeze_cough: temporal_direction_motion_reversible (w=0.36)
- chest_pain: arm_circular (w=0.55)
- chest_pain: hip_bend (w=0.61)
- chest_pain: temporal_sequence_seq_body_first (w=0.43)
- nausea_vomiting: temporal_dynamics_duration_sustained (w=0.44)
- point_finger: hand_open (w=0.32)
- point_finger: arm_raise (w=0.37)
- hugging: arm_extend (w=0.41)
- touch_pocket: hand_grasp (w=0.54)
- walking_apart: foot_pivot (w=0.43)

## 5a. Counterfactual faithfulness: original HyperGCN

- (class, literal) tested: 210; causal (95% CI of effect > 0): 8
- effect = drop of the class logit under the counterfactual minus under a random edit of the same size

| class | literal | CSV | n | success | edit (L) | drop cf | drop rand | effect [95% CI] | flip cf / rand | causal |
|---|---|---|---|---|---|---|---|---|---|---|
| drink_water | hand_lift_to_face | Y | 32 | 1.00 | 1.261 | 4.333 | 5.408 | -1.075 [-1.499, -0.638] | 0.97 / 1.00 | n |
| drink_water | arm_raise | Y | 32 | 1.00 | 0.993 | 2.548 | 5.292 | -2.743 [-3.431, -2.097] | 0.59 / 1.00 | n |
| drink_water | head_tilt_up | Y | 1 | 0.00 | 0.341 | - | - | - [-, -] | - / - | n |
| eat_meal | hand_lift_to_face | Y | 32 | 1.00 | 1.152 | 2.716 | 4.139 | -1.424 [-1.933, -0.944] | 0.84 / 1.00 | n |
| eat_meal | arm_raise | Y | 32 | 1.00 | 1.026 | 1.181 | 3.854 | -2.674 [-3.647, -1.839] | 0.72 / 1.00 | n |
| eat_meal | foot_static | Y | 32 | 1.00 | 0.547 | 0.032 | 1.319 | -1.287 [-1.670, -0.944] | 0.34 / 0.41 | n |
| eat_meal | head_tilt_down | Y | 1 | 1.00 | 0.334 | 0.642 | 0.607 | 0.035 [0.035, 0.035] | 1.00 / 1.00 | Y |
| brush_teeth | foot_static | Y | 32 | 1.00 | 0.548 | 0.116 | 1.161 | -1.045 [-1.313, -0.785] | 0.22 / 0.28 | n |
| brush_teeth | hip_static_stand | Y | 32 | 1.00 | 1.443 | 0.010 | 2.996 | -2.986 [-3.418, -2.544] | 0.09 / 0.25 | n |
| brush_teeth | leg_static_stand | Y | 32 | 1.00 | 1.509 | -0.134 | 3.433 | -3.567 [-4.023, -3.071] | 0.09 / 0.41 | n |
| brush_hair | foot_static | Y | 32 | 1.00 | 0.546 | -0.029 | 0.430 | -0.459 [-0.754, -0.173] | 0.06 / 0.03 | n |
| brush_hair | head_tilt_down | Y | 9 | 0.89 | 0.227 | 0.573 | 0.692 | -0.118 [-1.033, 0.724] | 0.25 / 0.00 | n |
| brush_hair | arm_raise | Y | 32 | 1.00 | 1.022 | 0.977 | 4.133 | -3.155 [-3.742, -2.564] | 0.19 / 0.88 | n |
| brush_hair | hand_lift_to_face | Y | 32 | 1.00 | 1.352 | 3.578 | 5.364 | -1.786 [-2.262, -1.318] | 0.78 / 1.00 | n |
| drop | leg_static_stand | Y | 32 | 1.00 | 1.414 | 0.329 | 2.790 | -2.461 [-3.043, -1.877] | 0.12 / 0.31 | n |
| drop | hip_static_stand | Y | 32 | 1.00 | 1.394 | 0.326 | 2.758 | -2.432 [-2.949, -1.902] | 0.25 / 0.44 | n |
| drop | foot_static | Y | 32 | 1.00 | 0.548 | 0.219 | 1.124 | -0.905 [-1.290, -0.510] | 0.22 / 0.22 | n |
| drop | arm_lower | Y | 12 | 0.92 | 0.184 | 1.283 | 2.474 | -1.191 [-1.735, -0.629] | 0.45 / 0.55 | n |
| pick_up | foot_static | Y | 32 | 1.00 | 0.514 | -0.017 | 0.070 | -0.087 [-0.273, 0.091] | 0.06 / 0.03 | n |
| pick_up | hip_bend | Y | 32 | 1.00 | 0.514 | -0.009 | 0.062 | -0.071 [-0.187, 0.039] | 0.00 / 0.00 | n |
| throw | leg_static_stand | Y | 32 | 1.00 | 1.428 | 0.123 | 1.379 | -1.256 [-1.587, -0.939] | 0.19 / 0.28 | n |
| throw | arm_extend | Y | 2 | 1.00 | 0.591 | 0.178 | 1.987 | -1.809 [-3.340, -0.278] | 0.00 / 0.50 | n |
| throw | hip_rotate | Y | 32 | 0.91 | 0.123 | -0.036 | 0.148 | -0.184 [-0.296, -0.080] | 0.17 / 0.21 | n |
| throw | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.465 | 0.701 | 1.273 | -0.572 [-0.861, -0.305] | 0.19 / 0.31 | n |
| sit_down | hip_bend | Y | 32 | 0.97 | 0.477 | -0.062 | -0.155 | 0.093 [-0.006, 0.192] | 0.00 / 0.00 | n |
| sit_down | foot_static | Y | 32 | 1.00 | 0.527 | -0.007 | 0.092 | -0.099 [-0.199, -0.003] | 0.03 / 0.03 | n |
| sit_down | leg_squat | Y | 32 | 1.00 | 0.745 | 0.120 | 0.563 | -0.443 [-0.692, -0.220] | 0.12 / 0.16 | n |
| sit_down | hip_sit | Y | 32 | 1.00 | 0.738 | 0.069 | 0.691 | -0.622 [-0.900, -0.374] | 0.16 / 0.16 | n |
| stand_up | foot_static | Y | 32 | 1.00 | 0.554 | -0.062 | 0.022 | -0.084 [-0.200, 0.025] | 0.00 / 0.00 | n |
| stand_up | temporal_direction_motion_upward | Y | 32 | 1.00 | 0.193 | 0.016 | -0.012 | 0.028 [-0.071, 0.127] | 0.03 / 0.06 | n |
| stand_up | hip_extend | Y | 32 | 0.97 | 0.175 | -0.062 | 0.022 | -0.084 [-0.150, -0.019] | 0.03 / 0.03 | n |
| stand_up | arm_raise | Y | 30 | 0.97 | 0.984 | 0.073 | 1.189 | -1.117 [-1.619, -0.666] | 0.10 / 0.31 | n |
| clapping | foot_static | Y | 32 | 1.00 | 0.542 | -0.120 | -0.217 | 0.097 [-0.137, 0.338] | 0.28 / 0.12 | n |
| clapping | arm_hold | Y | 28 | 1.00 | 0.869 | 0.672 | 3.628 | -2.955 [-3.602, -2.369] | 0.29 / 0.93 | n |
| clapping | temporal_direction_motion_converging | Y | 32 | 1.00 | 0.323 | 0.407 | 1.037 | -0.630 [-0.992, -0.301] | 0.28 / 0.25 | n |
| reading | foot_static | Y | 32 | 1.00 | 0.545 | -0.077 | 0.283 | -0.360 [-0.662, -0.089] | 0.34 / 0.28 | n |
| reading | head_tilt_down | Y | 3 | 0.67 | 0.220 | -0.362 | 0.445 | -0.807 [-0.942, -0.672] | 0.50 / 0.50 | n |
| reading | arm_hold | Y | 20 | 1.00 | 0.913 | 0.124 | 2.199 | -2.075 [-3.087, -1.158] | 0.55 / 0.75 | n |
| writing | arm_hold | Y | 32 | 1.00 | 0.852 | 0.555 | 4.610 | -4.055 [-4.500, -3.683] | 0.47 / 1.00 | n |
| writing | head_tilt_down | Y | 4 | 0.75 | 0.280 | 0.809 | 1.591 | -0.783 [-2.180, 1.168] | 1.00 / 1.00 | n |
| writing | foot_static | Y | 32 | 1.00 | 0.545 | -0.063 | 0.545 | -0.608 [-0.873, -0.355] | 0.53 / 0.53 | n |
| tear_up_paper | foot_static | Y | 32 | 1.00 | 0.545 | 0.001 | 0.312 | -0.312 [-0.661, -0.019] | 0.12 / 0.12 | n |
| tear_up_paper | temporal_direction_motion_diverging | Y | 32 | 1.00 | 0.292 | 0.408 | 0.538 | -0.130 [-0.486, 0.277] | 0.06 / 0.06 | n |
| put_on_jacket | foot_static | Y | 32 | 1.00 | 0.515 | -0.006 | 0.133 | -0.140 [-0.235, -0.062] | 0.00 / 0.00 | n |
| put_on_jacket | leg_static_stand | Y | 32 | 1.00 | 1.412 | -0.072 | 0.295 | -0.367 [-0.525, -0.233] | 0.00 / 0.00 | n |
| put_on_jacket | hip_static_stand | Y | 32 | 1.00 | 1.468 | -0.135 | 0.222 | -0.357 [-0.536, -0.164] | 0.03 / 0.03 | n |
| take_off_jacket | foot_static | Y | 32 | 1.00 | 0.513 | -0.050 | 0.283 | -0.333 [-0.461, -0.215] | 0.00 / 0.00 | n |
| take_off_jacket | hip_static_stand | Y | 32 | 1.00 | 1.393 | -0.039 | 0.724 | -0.763 [-1.066, -0.503] | 0.00 / 0.06 | n |
| take_off_jacket | leg_static_stand | Y | 32 | 1.00 | 1.407 | 0.060 | 0.809 | -0.749 [-1.012, -0.487] | 0.00 / 0.06 | n |
| put_on_a_shoe | hip_bend | Y | 32 | 0.97 | 0.580 | -0.003 | 0.036 | -0.039 [-0.183, 0.095] | 0.10 / 0.16 | n |
| put_on_a_shoe | leg_balance | Y | 32 | 1.00 | 0.551 | -0.060 | 0.759 | -0.819 [-1.171, -0.489] | 0.25 / 0.38 | n |
| take_off_a_shoe | foot_static | Y | 32 | 1.00 | 0.519 | 1.214 | 0.390 | 0.824 [0.507, 1.145] | 0.31 / 0.22 | Y |
| take_off_a_shoe | hip_bend | Y | 32 | 0.94 | 0.583 | 0.047 | -0.152 | 0.199 [-0.008, 0.426] | 0.33 / 0.27 | n |
| put_on_glasses | foot_static | Y | 32 | 1.00 | 0.553 | 0.095 | 0.567 | -0.473 [-0.846, -0.136] | 0.06 / 0.12 | n |
| put_on_glasses | hip_static_stand | Y | 32 | 1.00 | 1.567 | -0.164 | 2.038 | -2.202 [-2.579, -1.810] | 0.06 / 0.25 | n |
| put_on_glasses | hand_lift_to_face | Y | 32 | 1.00 | 1.153 | 2.603 | 5.652 | -3.049 [-3.493, -2.554] | 0.62 / 1.00 | n |
| take_off_glasses | hip_static_stand | Y | 32 | 1.00 | 1.544 | -0.002 | 2.586 | -2.588 [-2.916, -2.267] | 0.00 / 0.12 | n |
| take_off_glasses | hand_lift_to_face | Y | 32 | 1.00 | 1.129 | 2.642 | 5.580 | -2.938 [-3.351, -2.535] | 0.50 / 1.00 | n |
| take_off_glasses | arm_raise | Y | 32 | 1.00 | 0.905 | 1.772 | 5.859 | -4.087 [-4.746, -3.460] | 0.28 / 1.00 | n |
| put_on_a_hat_cap | foot_static | Y | 32 | 1.00 | 0.552 | 0.084 | 0.666 | -0.582 [-0.820, -0.346] | 0.03 / 0.09 | n |
| put_on_a_hat_cap | leg_static_stand | Y | 32 | 1.00 | 1.490 | 0.049 | 1.897 | -1.847 [-2.257, -1.436] | 0.03 / 0.25 | n |
| put_on_a_hat_cap | hip_static_stand | Y | 32 | 1.00 | 1.466 | -0.100 | 1.833 | -1.933 [-2.473, -1.438] | 0.03 / 0.25 | n |
| put_on_a_hat_cap | arm_raise | Y | 32 | 1.00 | 0.891 | 0.380 | 3.794 | -3.414 [-3.902, -2.962] | 0.03 / 0.72 | n |
| take_off_a_hat_cap | leg_static_stand | Y | 32 | 1.00 | 1.505 | -0.173 | 1.747 | -1.919 [-2.315, -1.529] | 0.00 / 0.16 | n |
| take_off_a_hat_cap | hip_static_stand | Y | 32 | 1.00 | 1.540 | -0.151 | 1.716 | -1.868 [-2.302, -1.471] | 0.09 / 0.19 | n |
| take_off_a_hat_cap | foot_static | Y | 32 | 1.00 | 0.545 | -0.085 | 0.078 | -0.163 [-0.389, 0.029] | 0.00 / 0.03 | n |
| take_off_a_hat_cap | arm_overhead | Y | 20 | 0.95 | 0.617 | 0.221 | 2.602 | -2.380 [-3.197, -1.674] | 0.05 / 0.42 | n |
| cheer_up | arm_raise | Y | 32 | 0.81 | 0.916 | 0.633 | 2.878 | -2.245 [-2.755, -1.776] | 0.12 / 0.65 | n |
| cheer_up | temporal_direction_motion_upward | Y | 32 | 0.94 | 0.228 | 0.114 | 0.106 | 0.009 [-0.167, 0.155] | 0.03 / 0.07 | n |
| cheer_up | arm_overhead | Y | 32 | 0.94 | 0.756 | 0.414 | 1.271 | -0.857 [-1.112, -0.602] | 0.10 / 0.20 | n |
| cheer_up | temporal_dynamics_speed_fast | Y | 10 | 1.00 | 0.947 | 1.173 | 2.695 | -1.522 [-2.839, -0.219] | 0.30 / 0.70 | n |
| hand_waving | arm_raise | Y | 32 | 1.00 | 1.015 | 0.745 | 5.197 | -4.451 [-4.823, -4.072] | 0.06 / 1.00 | n |
| hand_waving | foot_static | Y | 32 | 1.00 | 0.556 | -0.106 | -0.011 | -0.095 [-0.349, 0.138] | 0.09 / 0.12 | n |
| hand_waving | hip_static_stand | Y | 32 | 1.00 | 1.470 | -0.142 | 0.526 | -0.668 [-1.101, -0.263] | 0.03 / 0.12 | n |
| hand_waving | arm_lateral | Y | 32 | 1.00 | 0.520 | -0.062 | 1.138 | -1.200 [-1.632, -0.796] | 0.09 / 0.19 | n |
| kicking_something | hip_rotate | Y | 18 | 0.83 | 0.120 | 0.019 | 0.092 | -0.073 [-0.222, 0.059] | 0.27 / 0.27 | n |
| kicking_something | leg_kick | Y | 32 | 1.00 | 0.354 | -0.027 | -0.213 | 0.186 [0.074, 0.306] | 0.06 / 0.03 | Y |
| kicking_something | foot_kick | Y | 32 | 0.97 | 0.367 | 0.054 | -0.090 | 0.145 [0.003, 0.284] | 0.00 / 0.00 | Y |
| reach_into_pocket | foot_static | Y | 32 | 1.00 | 0.551 | -0.030 | 2.128 | -2.158 [-2.512, -1.872] | 0.06 / 0.19 | n |
| reach_into_pocket | hip_static_stand | Y | 32 | 1.00 | 1.439 | 0.305 | 4.190 | -3.885 [-4.407, -3.303] | 0.28 / 0.91 | n |
| reach_into_pocket | leg_static_stand | Y | 32 | 1.00 | 1.454 | 0.026 | 4.327 | -4.302 [-4.687, -3.913] | 0.12 / 0.84 | n |
| reach_into_pocket | arm_retract | Y | 21 | 1.00 | 0.276 | -0.149 | 0.503 | -0.652 [-1.200, -0.163] | 0.33 / 0.52 | n |
| hopping | arm_swing | Y | 22 | 1.00 | 0.267 | -0.093 | 0.019 | -0.112 [-0.253, 0.022] | 0.00 / 0.00 | n |
| hopping | temporal_direction_motion_upward | Y | 22 | 1.00 | 0.210 | -0.036 | 0.059 | -0.095 [-0.265, 0.052] | 0.00 / 0.00 | n |
| hopping | leg_alternating | Y | 2 | 1.00 | 0.239 | -0.075 | 0.040 | -0.115 [-0.220, -0.011] | 0.00 / 0.00 | n |
| hopping | hip_extend | Y | 32 | 1.00 | 0.183 | -0.023 | -0.018 | -0.005 [-0.090, 0.088] | 0.03 / 0.03 | n |
| jump_up | hip_extend | Y | 32 | 1.00 | 0.231 | 0.008 | 0.057 | -0.049 [-0.100, 0.005] | 0.00 / 0.00 | n |
| jump_up | temporal_dynamics_speed_fast | Y | 1 | 1.00 | 1.221 | 0.925 | 3.661 | -2.736 [-2.736, -2.736] | 0.00 / 1.00 | n |
| jump_up | leg_jump | Y | 32 | 0.72 | 0.383 | 0.007 | 0.333 | -0.326 [-0.458, -0.217] | 0.00 / 0.00 | n |
| jump_up | foot_jump | Y | 32 | 0.81 | 0.371 | 0.048 | 0.270 | -0.223 [-0.307, -0.142] | 0.00 / 0.00 | n |
| phone_call | hand_lift_to_face | Y | 32 | 1.00 | 1.098 | 3.789 | 5.938 | -2.150 [-2.629, -1.698] | 0.88 / 1.00 | n |
| phone_call | foot_static | Y | 32 | 1.00 | 0.537 | 0.044 | 0.583 | -0.539 [-0.848, -0.250] | 0.06 / 0.12 | n |
| phone_call | hip_static_stand | Y | 32 | 1.00 | 1.404 | 0.136 | 2.192 | -2.056 [-2.570, -1.631] | 0.12 / 0.38 | n |
| phone_call | arm_hold | Y | 32 | 1.00 | 0.838 | 2.403 | 5.652 | -3.248 [-4.069, -2.501] | 0.62 / 1.00 | n |
| play_with_phone_tablet | arm_hold | Y | 26 | 1.00 | 0.912 | 1.834 | 6.728 | -4.894 [-5.431, -4.319] | 0.69 / 1.00 | n |
| play_with_phone_tablet | head_tilt_down | Y | 1 | 1.00 | 0.183 | 0.213 | 0.255 | -0.042 [-0.042, -0.042] | 0.00 / 0.00 | n |
| play_with_phone_tablet | foot_static | Y | 32 | 1.00 | 0.544 | -0.071 | 1.089 | -1.161 [-1.540, -0.827] | 0.25 / 0.41 | n |
| type_on_a_keyboard | foot_static | Y | 32 | 1.00 | 0.545 | 0.475 | 2.193 | -1.718 [-2.207, -1.257] | 0.22 / 0.50 | n |
| type_on_a_keyboard | temporal_dynamics_speed_fast | Y | 7 | 1.00 | 1.149 | 2.173 | 3.397 | -1.224 [-2.728, -0.007] | 0.86 / 1.00 | n |
| type_on_a_keyboard | arm_hold | Y | 29 | 1.00 | 0.976 | 1.152 | 4.012 | -2.860 [-3.330, -2.411] | 0.86 / 1.00 | n |
| point_to_something | arm_raise | n | 32 | 0.97 | 0.977 | 1.653 | 5.816 | -4.163 [-4.838, -3.536] | 0.65 / 1.00 | n |
| point_to_something | foot_static | Y | 32 | 1.00 | 0.559 | 0.483 | 0.802 | -0.319 [-0.598, -0.057] | 0.28 / 0.25 | n |
| point_to_something | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.442 | 1.980 | 4.331 | -2.352 [-2.780, -1.884] | 0.69 / 0.94 | n |
| point_to_something | hip_static_stand | Y | 32 | 1.00 | 1.492 | -0.019 | 1.923 | -1.943 [-2.464, -1.411] | 0.12 / 0.28 | n |
| taking_a_selfie | arm_raise | Y | 32 | 1.00 | 0.981 | 0.391 | 6.057 | -5.666 [-6.166, -5.103] | 0.19 / 1.00 | n |
| taking_a_selfie | foot_static | Y | 32 | 1.00 | 0.560 | -0.093 | 0.297 | -0.390 [-0.715, -0.069] | 0.19 / 0.19 | n |
| taking_a_selfie | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.464 | 0.622 | 4.249 | -3.628 [-4.157, -3.052] | 0.19 / 0.81 | n |
| taking_a_selfie | hip_static_stand | Y | 32 | 1.00 | 1.556 | -0.030 | 2.168 | -2.198 [-2.657, -1.775] | 0.09 / 0.19 | n |
| check_time_(from_watch) | arm_raise | Y | 32 | 1.00 | 1.027 | -0.244 | 5.982 | -6.226 [-6.847, -5.531] | 0.16 / 1.00 | n |
| check_time_(from_watch) | hip_static_stand | Y | 32 | 1.00 | 1.467 | -0.008 | 1.988 | -1.996 [-2.482, -1.543] | 0.09 / 0.19 | n |
| check_time_(from_watch) | foot_static | Y | 32 | 1.00 | 0.568 | 0.053 | 0.274 | -0.221 [-0.480, 0.024] | 0.06 / 0.09 | n |
| check_time_(from_watch) | leg_static_stand | Y | 32 | 1.00 | 1.437 | -0.129 | 2.559 | -2.688 [-3.121, -2.247] | 0.03 / 0.09 | n |
| rub_two_hands | foot_static | Y | 32 | 1.00 | 0.567 | 0.008 | -0.022 | 0.030 [-0.150, 0.198] | 0.09 / 0.09 | n |
| rub_two_hands | arm_hold | Y | 23 | 1.00 | 0.991 | 0.748 | 4.001 | -3.253 [-4.032, -2.438] | 0.39 / 0.91 | n |
| rub_two_hands | temporal_direction_motion_converging | Y | 32 | 1.00 | 0.317 | 0.538 | 1.169 | -0.631 [-1.146, -0.089] | 0.31 / 0.38 | n |
| rub_two_hands | hip_static_stand | Y | 32 | 1.00 | 1.435 | 0.079 | 1.465 | -1.387 [-1.753, -1.037] | 0.12 / 0.19 | n |
| nod_head_bow | hip_bend | Y | 32 | 0.97 | 0.495 | 0.042 | -0.028 | 0.069 [-0.069, 0.175] | 0.00 / 0.03 | n |
| nod_head_bow | leg_static_stand | Y | 32 | 1.00 | 1.443 | 0.278 | 2.766 | -2.489 [-2.856, -2.153] | 0.06 / 0.31 | n |
| nod_head_bow | foot_static | Y | 32 | 1.00 | 0.526 | 0.166 | 0.849 | -0.683 [-0.920, -0.463] | 0.00 / 0.03 | n |
| nod_head_bow | temporal_direction_motion_downward | Y | 32 | 1.00 | 0.192 | 0.107 | 0.289 | -0.182 [-0.360, -0.021] | 0.09 / 0.09 | n |
| shake_head | hip_static_stand | Y | 32 | 1.00 | 1.427 | -0.424 | 5.249 | -5.673 [-6.147, -5.160] | 0.03 / 0.94 | n |
| shake_head | foot_static | Y | 32 | 1.00 | 0.551 | -0.248 | 2.138 | -2.385 [-2.806, -1.958] | 0.06 / 0.16 | n |
| shake_head | leg_static_stand | Y | 32 | 1.00 | 1.435 | -0.365 | 5.207 | -5.572 [-6.002, -5.020] | 0.06 / 0.94 | n |
| shake_head | head_shake | Y | 22 | 0.23 | 0.073 | 1.052 | 0.153 | 0.899 [-0.514, 2.933] | 0.40 / 0.00 | n |
| wipe_face | temporal_direction_motion_upward | Y | 32 | 1.00 | 0.212 | 0.345 | 0.310 | 0.035 [-0.286, 0.397] | 0.09 / 0.09 | n |
| wipe_face | hip_static_stand | Y | 32 | 1.00 | 1.476 | 0.249 | 3.435 | -3.185 [-3.800, -2.599] | 0.12 / 0.59 | n |
| wipe_face | leg_static_stand | Y | 32 | 1.00 | 1.436 | 0.332 | 3.610 | -3.278 [-3.773, -2.765] | 0.16 / 0.59 | n |
| wipe_face | hand_lift_to_face | Y | 32 | 1.00 | 1.356 | 3.060 | 6.379 | -3.319 [-3.970, -2.680] | 0.62 / 1.00 | n |
| salute | hip_static_stand | Y | 32 | 1.00 | 1.420 | 0.052 | 3.071 | -3.020 [-3.440, -2.585] | 0.03 / 0.38 | n |
| salute | leg_static_stand | Y | 32 | 1.00 | 1.425 | 0.027 | 3.158 | -3.131 [-3.576, -2.704] | 0.06 / 0.44 | n |
| salute | foot_static | Y | 32 | 1.00 | 0.557 | -0.023 | 1.010 | -1.033 [-1.274, -0.794] | 0.03 / 0.06 | n |
| salute | temporal_direction_motion_upward | Y | 31 | 1.00 | 0.218 | 0.958 | 3.130 | -2.171 [-2.732, -1.632] | 0.32 / 0.71 | n |
| put_palms_together | hip_static_stand | Y | 32 | 1.00 | 1.460 | -0.192 | 1.769 | -1.961 [-2.413, -1.497] | 0.03 / 0.06 | n |
| put_palms_together | leg_static_stand | Y | 32 | 1.00 | 1.416 | -0.064 | 1.978 | -2.042 [-2.505, -1.573] | 0.03 / 0.12 | n |
| put_palms_together | arm_hold | Y | 32 | 1.00 | 0.862 | 4.637 | 6.469 | -1.832 [-2.192, -1.454] | 0.94 / 1.00 | n |
| cross_hands_in_front | arm_raise | n | 32 | 1.00 | 0.778 | 1.226 | 4.158 | -2.932 [-3.393, -2.481] | 0.31 / 1.00 | n |
| cross_hands_in_front | hip_static_stand | Y | 32 | 1.00 | 1.437 | 0.118 | 2.024 | -1.907 [-2.140, -1.670] | 0.09 / 0.12 | n |
| cross_hands_in_front | arm_overhead | n | 2 | 1.00 | 0.841 | 0.684 | 1.164 | -0.480 [-1.735, 0.774] | 1.00 / 1.00 | n |
| sneeze_cough | leg_static_stand | Y | 32 | 1.00 | 1.463 | 0.166 | 4.143 | -3.978 [-4.327, -3.569] | 0.16 / 0.66 | n |
| sneeze_cough | foot_static | Y | 32 | 1.00 | 0.553 | 0.126 | 1.758 | -1.632 [-2.037, -1.259] | 0.22 / 0.28 | n |
| sneeze_cough | temporal_dynamics_speed_fast | Y | 2 | 1.00 | 0.945 | -0.264 | 1.105 | -1.369 [-2.506, -0.232] | 1.00 / 1.00 | n |
| sneeze_cough | hand_lift_to_face | Y | 32 | 1.00 | 1.221 | 1.800 | 5.407 | -3.607 [-4.258, -2.998] | 0.41 / 1.00 | n |
| staggering | leg_alternating | Y | 20 | 1.00 | 0.242 | 0.035 | 0.098 | -0.063 [-0.138, -0.001] | 0.00 / 0.00 | n |
| staggering | hip_sway | Y | 32 | 0.94 | 0.199 | 0.030 | 0.027 | 0.003 [-0.051, 0.056] | 0.00 / 0.00 | n |
| staggering | foot_step | Y | 32 | 0.97 | 0.271 | -0.012 | 0.047 | -0.060 [-0.167, 0.055] | 0.03 / 0.00 | n |
| falling_down | temporal_direction_motion_downward | Y | 32 | 1.00 | 0.212 | -0.069 | -0.112 | 0.044 [-0.088, 0.171] | 0.06 / 0.03 | n |
| falling_down | hip_bend | Y | 32 | 0.97 | 0.487 | 0.013 | -0.057 | 0.070 [-0.035, 0.183] | 0.03 / 0.03 | n |
| falling_down | leg_squat | Y | 32 | 1.00 | 0.698 | -0.040 | -0.047 | 0.007 [-0.189, 0.173] | 0.06 / 0.06 | n |
| headache | hip_static_stand | Y | 32 | 1.00 | 1.475 | 0.107 | 2.680 | -2.572 [-2.891, -2.237] | 0.22 / 0.47 | n |
| headache | leg_static_stand | Y | 32 | 1.00 | 1.468 | 0.167 | 2.804 | -2.637 [-3.065, -2.182] | 0.19 / 0.47 | n |
| headache | hand_lift_to_face | Y | 32 | 1.00 | 1.329 | 4.338 | 6.529 | -2.191 [-2.732, -1.626] | 0.84 / 1.00 | n |
| headache | arm_raise | Y | 32 | 1.00 | 0.977 | 3.062 | 6.669 | -3.607 [-4.282, -2.974] | 0.62 / 1.00 | n |
| chest_pain | foot_static | Y | 32 | 1.00 | 0.543 | 0.043 | 1.186 | -1.143 [-1.507, -0.851] | 0.03 / 0.06 | n |
| chest_pain | arm_retract | Y | 24 | 1.00 | 0.287 | 0.796 | 1.769 | -0.973 [-1.402, -0.569] | 0.29 / 0.50 | n |
| chest_pain | hip_static_stand | Y | 32 | 1.00 | 1.425 | 0.042 | 3.307 | -3.265 [-3.732, -2.840] | 0.09 / 0.44 | n |
| back_pain | leg_static_stand | Y | 32 | 1.00 | 1.439 | 0.114 | 3.343 | -3.229 [-3.708, -2.781] | 0.06 / 0.44 | n |
| back_pain | foot_static | Y | 32 | 1.00 | 0.553 | -0.077 | 1.114 | -1.191 [-1.499, -0.866] | 0.12 / 0.12 | n |
| back_pain | hip_bend | Y | 16 | 0.94 | 0.564 | 0.090 | 0.018 | 0.072 [-0.188, 0.338] | 0.47 / 0.40 | n |
| neck_pain | hip_static_stand | Y | 32 | 1.00 | 1.416 | 0.071 | 3.335 | -3.264 [-3.759, -2.843] | 0.16 / 0.47 | n |
| neck_pain | leg_static_stand | Y | 32 | 1.00 | 1.497 | 0.170 | 3.456 | -3.285 [-3.687, -2.913] | 0.28 / 0.47 | n |
| neck_pain | foot_static | Y | 32 | 1.00 | 0.547 | -0.008 | 1.429 | -1.437 [-1.713, -1.157] | 0.09 / 0.16 | n |
| neck_pain | arm_raise | Y | 32 | 1.00 | 0.970 | 1.316 | 5.245 | -3.930 [-4.360, -3.487] | 0.16 / 1.00 | n |
| nausea_vomiting | temporal_dynamics_speed_fast | Y | 32 | 1.00 | 0.882 | -0.229 | 0.530 | -0.759 [-1.300, -0.260] | 0.03 / 0.09 | n |
| nausea_vomiting | foot_static | Y | 32 | 1.00 | 0.536 | -0.012 | 0.313 | -0.324 [-0.654, -0.014] | 0.31 / 0.25 | n |
| nausea_vomiting | leg_static_stand | Y | 32 | 1.00 | 1.457 | 0.114 | 0.674 | -0.560 [-0.913, -0.218] | 0.19 / 0.09 | n |
| nausea_vomiting | hip_bend | Y | 32 | 0.91 | 0.576 | 0.008 | -0.257 | 0.265 [0.100, 0.451] | 0.14 / 0.10 | Y |
| fan_self | temporal_dynamics_speed_fast | Y | 6 | 1.00 | 1.008 | 2.262 | 1.941 | 0.321 [-1.715, 1.886] | 0.67 / 0.33 | n |
| fan_self | foot_static | Y | 32 | 1.00 | 0.544 | 0.007 | 0.162 | -0.155 [-0.318, 0.012] | 0.09 / 0.06 | n |
| fan_self | leg_static_stand | Y | 32 | 1.00 | 1.463 | -0.054 | 0.608 | -0.662 [-0.910, -0.432] | 0.09 / 0.06 | n |
| fan_self | hip_static_stand | Y | 32 | 1.00 | 1.458 | -0.120 | 0.828 | -0.949 [-1.240, -0.647] | 0.09 / 0.06 | n |
| punch_slap | leg_static_stand | Y | 32 | 1.00 | 1.380 | 0.208 | 2.467 | -2.259 [-2.577, -1.917] | 0.12 / 0.28 | n |
| punch_slap | arm_extend | Y | 1 | 1.00 | 0.605 | 0.369 | 1.800 | -1.431 [-1.431, -1.431] | 0.00 / 0.00 | n |
| punch_slap | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.444 | 0.596 | 1.130 | -0.534 [-0.927, -0.192] | 0.25 / 0.28 | n |
| punch_slap | arm_swing | Y | 32 | 1.00 | 0.282 | -0.110 | 0.302 | -0.412 [-0.637, -0.194] | 0.03 / 0.09 | n |
| kicking | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.443 | 0.490 | 0.960 | -0.470 [-0.728, -0.229] | 0.28 / 0.38 | n |
| kicking | temporal_dynamics_speed_fast | Y | 1 | 1.00 | 0.933 | 0.890 | 0.977 | -0.086 [-0.086, -0.086] | 1.00 / 1.00 | n |
| kicking | hip_rotate | Y | 32 | 0.88 | 0.114 | 0.082 | 0.115 | -0.033 [-0.186, 0.110] | 0.11 / 0.07 | n |
| kicking | leg_balance | Y | 25 | 0.96 | 0.512 | 0.172 | 0.734 | -0.562 [-0.786, -0.379] | 0.00 / 0.04 | n |
| pushing | leg_static_stand | Y | 32 | 1.00 | 1.366 | 0.621 | 3.037 | -2.416 [-2.749, -2.103] | 0.06 / 0.38 | n |
| pushing | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.443 | -0.048 | 0.320 | -0.368 [-0.678, -0.095] | 0.06 / 0.06 | n |
| pushing | hip_rotate | Y | 32 | 0.81 | 0.116 | -0.041 | -0.031 | -0.010 [-0.136, 0.106] | 0.00 / 0.04 | n |
| pat_on_back | foot_static | Y | 32 | 1.00 | 0.547 | -0.151 | 0.326 | -0.478 [-0.623, -0.332] | 0.03 / 0.03 | n |
| pat_on_back | leg_static_stand | Y | 32 | 1.00 | 1.407 | 0.107 | 0.499 | -0.392 [-0.654, -0.146] | 0.06 / 0.06 | n |
| pat_on_back | arm_extend | Y | 2 | 1.00 | 0.477 | 0.122 | 0.714 | -0.592 [-1.109, -0.075] | 0.00 / 1.00 | n |
| pat_on_back | hip_static_stand | Y | 32 | 1.00 | 1.355 | 0.144 | 0.435 | -0.290 [-0.550, -0.026] | 0.06 / 0.00 | n |
| point_finger | foot_static | Y | 32 | 1.00 | 0.556 | 0.346 | 0.679 | -0.334 [-0.489, -0.190] | 0.06 / 0.06 | n |
| point_finger | hip_static_stand | Y | 32 | 1.00 | 1.387 | 0.078 | 1.350 | -1.273 [-1.474, -1.061] | 0.03 / 0.06 | n |
| point_finger | leg_static_stand | Y | 32 | 1.00 | 1.398 | 0.195 | 1.154 | -0.959 [-1.125, -0.784] | 0.09 / 0.09 | n |
| point_finger | arm_extend | Y | 4 | 1.00 | 0.540 | 0.508 | 0.865 | -0.358 [-0.744, 0.029] | 0.00 / 0.25 | n |
| hugging | foot_static | Y | 32 | 1.00 | 0.537 | 0.121 | 0.546 | -0.425 [-0.597, -0.260] | 0.00 / 0.00 | n |
| hugging | hip_static_stand | Y | 32 | 1.00 | 1.301 | -0.167 | 1.532 | -1.699 [-1.911, -1.502] | 0.03 / 0.09 | n |
| hugging | leg_static_stand | Y | 32 | 1.00 | 1.345 | -0.044 | 1.863 | -1.906 [-2.173, -1.651] | 0.03 / 0.19 | n |
| hugging | temporal_direction_motion_converging | Y | 32 | 1.00 | 0.334 | -0.197 | 0.001 | -0.198 [-0.446, 0.039] | 0.00 / 0.03 | n |
| giving_object | foot_static | Y | 32 | 1.00 | 0.535 | -0.113 | 0.340 | -0.453 [-0.643, -0.269] | 0.03 / 0.09 | n |
| giving_object | hip_static_stand | Y | 32 | 1.00 | 1.378 | -0.249 | 1.581 | -1.830 [-2.079, -1.582] | 0.06 / 0.16 | n |
| giving_object | leg_static_stand | Y | 32 | 1.00 | 1.349 | -0.253 | 1.580 | -1.834 [-2.127, -1.576] | 0.00 / 0.09 | n |
| giving_object | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.436 | -0.636 | 0.986 | -1.622 [-2.035, -1.235] | 0.03 / 0.38 | n |
| touch_pocket | hip_static_stand | Y | 32 | 1.00 | 1.379 | 0.486 | 2.332 | -1.846 [-2.216, -1.512] | 0.09 / 0.25 | n |
| touch_pocket | leg_static_stand | Y | 32 | 1.00 | 1.387 | 0.358 | 2.098 | -1.740 [-2.131, -1.399] | 0.03 / 0.19 | n |
| touch_pocket | foot_static | Y | 32 | 1.00 | 0.528 | 0.053 | 0.597 | -0.544 [-0.841, -0.299] | 0.12 / 0.12 | n |
| shaking_hands | foot_static | Y | 32 | 1.00 | 0.536 | -0.028 | 0.559 | -0.588 [-0.758, -0.426] | 0.03 / 0.03 | n |
| shaking_hands | hip_static_stand | Y | 32 | 1.00 | 1.353 | -0.014 | 1.929 | -1.943 [-2.218, -1.662] | 0.00 / 0.12 | n |
| walking_towards | leg_alternating | Y | 32 | 1.00 | 0.254 | 0.080 | 0.123 | -0.043 [-0.094, 0.011] | 0.00 / 0.00 | n |
| walking_towards | hip_sway | Y | 32 | 0.91 | 0.174 | 0.019 | 0.017 | 0.002 [-0.036, 0.047] | 0.03 / 0.03 | n |
| walking_towards | arm_swing | Y | 32 | 1.00 | 0.248 | 0.029 | 0.192 | -0.162 [-0.250, -0.066] | 0.00 / 0.00 | n |
| walking_towards | foot_step | Y | 32 | 0.97 | 0.264 | 0.217 | 0.212 | 0.005 [-0.112, 0.146] | 0.00 / 0.00 | n |
| walking_apart | leg_alternating | Y | 32 | 1.00 | 0.251 | 0.069 | -0.118 | 0.187 [0.122, 0.262] | 0.00 / 0.00 | Y |
| walking_apart | arm_swing | Y | 32 | 1.00 | 0.268 | 0.053 | -0.230 | 0.284 [0.181, 0.391] | 0.12 / 0.06 | Y |
| walking_apart | hip_sway | Y | 32 | 0.81 | 0.175 | 0.015 | 0.011 | 0.003 [-0.045, 0.053] | 0.00 / 0.00 | n |
| walking_apart | leg_step | Y | 32 | 0.91 | 0.251 | 0.129 | -0.106 | 0.235 [0.141, 0.331] | 0.03 / 0.03 | Y |

## 5b. Counterfactual faithfulness: fused model

- (class, literal) tested: 210; causal (95% CI of effect > 0): 74
- effect = drop of the class logit under the counterfactual minus under a random edit of the same size

| class | literal | CSV | n | success | edit (L) | drop cf | drop rand | effect [95% CI] | flip cf / rand | causal |
|---|---|---|---|---|---|---|---|---|---|---|
| drink_water | hand_lift_to_face | Y | 32 | 1.00 | 1.261 | 17.816 | 25.793 | -7.977 [-9.810, -6.234] | 0.94 / 1.00 | n |
| drink_water | arm_raise | Y | 32 | 1.00 | 0.993 | 12.234 | 23.498 | -11.264 [-13.463, -9.294] | 0.50 / 1.00 | n |
| drink_water | head_tilt_up | Y | 1 | 0.00 | 0.341 | - | - | - [-, -] | - / - | n |
| eat_meal | hand_lift_to_face | Y | 32 | 1.00 | 1.152 | 10.722 | 16.936 | -6.214 [-7.598, -4.740] | 0.88 / 1.00 | n |
| eat_meal | arm_raise | Y | 32 | 1.00 | 1.026 | 7.162 | 15.627 | -8.465 [-10.752, -6.530] | 0.69 / 1.00 | n |
| eat_meal | foot_static | Y | 32 | 1.00 | 0.547 | 1.107 | 5.722 | -4.615 [-5.613, -3.580] | 1.00 / 0.41 | n |
| eat_meal | head_tilt_down | Y | 1 | 1.00 | 0.334 | 3.128 | 6.400 | -3.272 [-3.272, -3.272] | 1.00 / 1.00 | n |
| brush_teeth | foot_static | Y | 32 | 1.00 | 0.548 | 20.197 | 5.101 | 15.097 [14.113, 15.976] | 1.00 / 0.22 | Y |
| brush_teeth | hip_static_stand | Y | 32 | 1.00 | 1.443 | 1.388 | 14.549 | -13.161 [-14.378, -11.975] | 0.09 / 0.28 | n |
| brush_teeth | leg_static_stand | Y | 32 | 1.00 | 1.509 | 1.659 | 15.687 | -14.029 [-15.318, -12.606] | 0.06 / 0.31 | n |
| brush_hair | foot_static | Y | 32 | 1.00 | 0.546 | 16.473 | 1.779 | 14.693 [13.917, 15.481] | 0.97 / 0.03 | Y |
| brush_hair | head_tilt_down | Y | 9 | 0.89 | 0.227 | 3.725 | 5.709 | -1.985 [-4.513, 0.814] | 0.38 / 0.12 | n |
| brush_hair | arm_raise | Y | 32 | 1.00 | 1.022 | 7.812 | 9.983 | -2.171 [-3.900, -0.495] | 0.22 / 0.56 | n |
| brush_hair | hand_lift_to_face | Y | 32 | 1.00 | 1.352 | 14.509 | 13.642 | 0.867 [-0.644, 2.322] | 0.75 / 0.88 | n |
| drop | leg_static_stand | Y | 32 | 1.00 | 1.414 | 4.006 | 11.258 | -7.252 [-8.500, -5.897] | 0.22 / 0.41 | n |
| drop | hip_static_stand | Y | 32 | 1.00 | 1.394 | 4.038 | 11.165 | -7.127 [-8.421, -5.952] | 0.25 / 0.50 | n |
| drop | foot_static | Y | 32 | 1.00 | 0.548 | 21.469 | 6.694 | 14.775 [13.799, 15.661] | 1.00 / 0.25 | Y |
| drop | arm_lower | Y | 12 | 0.92 | 0.184 | 4.466 | 7.613 | -3.147 [-4.002, -2.199] | 0.36 / 0.55 | n |
| pick_up | foot_static | Y | 32 | 1.00 | 0.514 | 13.292 | 1.776 | 11.516 [10.869, 12.111] | 0.97 / 0.12 | Y |
| pick_up | hip_bend | Y | 32 | 1.00 | 0.514 | 2.579 | 1.082 | 1.497 [0.988, 1.941] | 0.03 / 0.03 | Y |
| throw | leg_static_stand | Y | 32 | 1.00 | 1.428 | 2.380 | 10.397 | -8.016 [-9.037, -7.073] | 0.19 / 0.25 | n |
| throw | arm_extend | Y | 2 | 1.00 | 0.591 | 8.444 | 11.278 | -2.833 [-5.126, -0.540] | 0.00 / 0.50 | n |
| throw | hip_rotate | Y | 32 | 0.91 | 0.123 | 1.554 | 1.240 | 0.313 [-0.189, 0.826] | 0.14 / 0.21 | n |
| throw | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.465 | 9.361 | 7.396 | 1.966 [0.974, 3.001] | 0.28 / 0.38 | Y |
| sit_down | hip_bend | Y | 32 | 0.97 | 0.477 | 2.938 | 0.088 | 2.850 [2.539, 3.167] | 0.00 / 0.00 | Y |
| sit_down | foot_static | Y | 32 | 1.00 | 0.527 | 18.395 | 2.239 | 16.156 [15.200, 17.032] | 0.97 / 0.03 | Y |
| sit_down | leg_squat | Y | 32 | 1.00 | 0.745 | 2.943 | 4.839 | -1.896 [-2.956, -0.919] | 0.12 / 0.16 | n |
| sit_down | hip_sit | Y | 32 | 1.00 | 0.738 | 2.883 | 5.499 | -2.616 [-3.407, -1.892] | 0.12 / 0.19 | n |
| stand_up | foot_static | Y | 32 | 1.00 | 0.554 | 23.733 | 1.995 | 21.738 [20.669, 22.771] | 1.00 / 0.00 | Y |
| stand_up | temporal_direction_motion_upward | Y | 32 | 1.00 | 0.193 | 1.365 | -0.129 | 1.494 [1.206, 1.795] | 0.00 / 0.03 | Y |
| stand_up | hip_extend | Y | 32 | 0.97 | 0.175 | 1.133 | 0.192 | 0.941 [0.559, 1.239] | 0.03 / 0.00 | Y |
| stand_up | arm_raise | Y | 30 | 0.97 | 0.984 | 1.057 | 5.885 | -4.827 [-6.167, -3.508] | 0.03 / 0.38 | n |
| clapping | foot_static | Y | 32 | 1.00 | 0.542 | 17.749 | 1.590 | 16.159 [15.288, 17.008] | 1.00 / 0.12 | Y |
| clapping | arm_hold | Y | 28 | 1.00 | 0.869 | 5.640 | 10.997 | -5.357 [-7.206, -3.656] | 0.39 / 0.82 | n |
| clapping | temporal_direction_motion_converging | Y | 32 | 1.00 | 0.323 | 3.143 | 2.341 | 0.803 [-0.467, 2.004] | 0.19 / 0.19 | n |
| reading | foot_static | Y | 32 | 1.00 | 0.545 | 12.206 | 1.906 | 10.300 [9.587, 11.067] | 1.00 / 0.34 | Y |
| reading | head_tilt_down | Y | 3 | 0.67 | 0.220 | 2.327 | 4.278 | -1.951 [-3.675, -0.227] | 0.50 / 0.50 | n |
| reading | arm_hold | Y | 20 | 1.00 | 0.913 | 0.918 | 0.870 | 0.048 [-2.650, 2.511] | 0.50 / 0.50 | n |
| writing | arm_hold | Y | 32 | 1.00 | 0.852 | 3.384 | 11.903 | -8.519 [-10.055, -6.910] | 0.47 / 1.00 | n |
| writing | head_tilt_down | Y | 4 | 0.75 | 0.280 | 4.680 | 4.595 | 0.085 [-1.638, 3.197] | 1.00 / 1.00 | n |
| writing | foot_static | Y | 32 | 1.00 | 0.545 | 0.249 | 2.676 | -2.426 [-3.243, -1.611] | 1.00 / 0.75 | n |
| tear_up_paper | foot_static | Y | 32 | 1.00 | 0.545 | 11.218 | 2.610 | 8.608 [7.854, 9.309] | 0.97 / 0.09 | Y |
| tear_up_paper | temporal_direction_motion_diverging | Y | 32 | 1.00 | 0.292 | 3.446 | 0.709 | 2.737 [1.523, 3.936] | 0.06 / 0.03 | Y |
| put_on_jacket | foot_static | Y | 32 | 1.00 | 0.515 | 16.476 | 1.711 | 14.765 [14.263, 15.257] | 0.94 / 0.06 | Y |
| put_on_jacket | leg_static_stand | Y | 32 | 1.00 | 1.412 | 4.190 | 5.180 | -0.990 [-1.777, -0.149] | 0.00 / 0.00 | n |
| put_on_jacket | hip_static_stand | Y | 32 | 1.00 | 1.468 | 3.856 | 5.187 | -1.331 [-1.922, -0.706] | 0.03 / 0.03 | n |
| take_off_jacket | foot_static | Y | 32 | 1.00 | 0.513 | 21.375 | 2.954 | 18.421 [17.501, 19.304] | 0.97 / 0.00 | Y |
| take_off_jacket | hip_static_stand | Y | 32 | 1.00 | 1.393 | 2.125 | 5.723 | -3.598 [-4.561, -2.626] | 0.00 / 0.00 | n |
| take_off_jacket | leg_static_stand | Y | 32 | 1.00 | 1.407 | 2.748 | 5.731 | -2.983 [-3.727, -2.237] | 0.03 / 0.06 | n |
| put_on_a_shoe | hip_bend | Y | 32 | 0.97 | 0.580 | 2.254 | 0.576 | 1.678 [1.045, 2.260] | 0.06 / 0.06 | Y |
| put_on_a_shoe | leg_balance | Y | 32 | 1.00 | 0.551 | 0.843 | 1.054 | -0.211 [-0.944, 0.511] | 0.28 / 0.31 | n |
| take_off_a_shoe | foot_static | Y | 32 | 1.00 | 0.519 | 20.529 | 0.828 | 19.702 [18.551, 20.793] | 1.00 / 0.19 | Y |
| take_off_a_shoe | hip_bend | Y | 32 | 0.94 | 0.583 | 2.289 | 0.388 | 1.901 [1.152, 2.628] | 0.40 / 0.33 | Y |
| put_on_glasses | foot_static | Y | 32 | 1.00 | 0.553 | 19.699 | 6.066 | 13.633 [12.644, 14.608] | 1.00 / 0.12 | Y |
| put_on_glasses | hip_static_stand | Y | 32 | 1.00 | 1.567 | 1.063 | 14.544 | -13.481 [-14.732, -12.246] | 0.12 / 0.25 | n |
| put_on_glasses | hand_lift_to_face | Y | 32 | 1.00 | 1.153 | 14.813 | 22.707 | -7.894 [-8.962, -6.687] | 0.56 / 1.00 | n |
| take_off_glasses | hip_static_stand | Y | 32 | 1.00 | 1.544 | 1.534 | 13.163 | -11.629 [-12.585, -10.601] | 0.00 / 0.03 | n |
| take_off_glasses | hand_lift_to_face | Y | 32 | 1.00 | 1.129 | 14.545 | 20.647 | -6.102 [-7.462, -4.780] | 0.44 / 1.00 | n |
| take_off_glasses | arm_raise | Y | 32 | 1.00 | 0.905 | 12.065 | 20.098 | -8.034 [-9.999, -6.298] | 0.31 / 1.00 | n |
| put_on_a_hat_cap | foot_static | Y | 32 | 1.00 | 0.552 | 21.189 | 4.266 | 16.924 [16.099, 17.723] | 1.00 / 0.00 | Y |
| put_on_a_hat_cap | leg_static_stand | Y | 32 | 1.00 | 1.490 | 2.979 | 12.122 | -9.142 [-10.005, -8.195] | 0.03 / 0.16 | n |
| put_on_a_hat_cap | hip_static_stand | Y | 32 | 1.00 | 1.466 | 2.524 | 13.388 | -10.865 [-12.030, -9.698] | 0.03 / 0.12 | n |
| put_on_a_hat_cap | arm_raise | Y | 32 | 1.00 | 0.891 | 7.019 | 12.197 | -5.178 [-6.350, -3.946] | 0.03 / 0.62 | n |
| take_off_a_hat_cap | leg_static_stand | Y | 32 | 1.00 | 1.505 | 2.779 | 12.976 | -10.196 [-10.935, -9.487] | 0.06 / 0.22 | n |
| take_off_a_hat_cap | hip_static_stand | Y | 32 | 1.00 | 1.540 | 3.008 | 12.825 | -9.817 [-10.848, -8.672] | 0.09 / 0.25 | n |
| take_off_a_hat_cap | foot_static | Y | 32 | 1.00 | 0.545 | 15.238 | 3.531 | 11.707 [11.069, 12.391] | 0.91 / 0.03 | Y |
| take_off_a_hat_cap | arm_overhead | Y | 20 | 0.95 | 0.617 | 3.410 | 10.305 | -6.896 [-8.267, -5.372] | 0.05 / 0.79 | n |
| cheer_up | arm_raise | Y | 32 | 0.81 | 0.916 | 4.884 | 8.082 | -3.199 [-4.682, -1.882] | 0.19 / 0.69 | n |
| cheer_up | temporal_direction_motion_upward | Y | 32 | 0.94 | 0.228 | 2.419 | 0.841 | 1.578 [1.145, 2.075] | 0.10 / 0.07 | Y |
| cheer_up | arm_overhead | Y | 32 | 0.94 | 0.756 | 2.836 | 4.436 | -1.599 [-2.787, -0.577] | 0.17 / 0.23 | n |
| cheer_up | temporal_dynamics_speed_fast | Y | 10 | 1.00 | 0.947 | 2.529 | 11.030 | -8.501 [-13.718, -4.417] | 0.20 / 0.80 | n |
| hand_waving | arm_raise | Y | 32 | 1.00 | 1.015 | 8.691 | 23.722 | -15.031 [-16.628, -13.488] | 0.19 / 0.97 | n |
| hand_waving | foot_static | Y | 32 | 1.00 | 0.556 | 24.770 | 3.542 | 21.228 [20.208, 22.175] | 1.00 / 0.19 | Y |
| hand_waving | hip_static_stand | Y | 32 | 1.00 | 1.470 | 0.663 | 10.846 | -10.183 [-11.184, -9.167] | 0.03 / 0.22 | n |
| hand_waving | arm_lateral | Y | 32 | 1.00 | 0.520 | 1.423 | 5.765 | -4.341 [-5.866, -2.917] | 0.09 / 0.22 | n |
| kicking_something | hip_rotate | Y | 18 | 0.83 | 0.120 | 0.787 | 0.083 | 0.704 [0.222, 1.163] | 0.27 / 0.20 | Y |
| kicking_something | leg_kick | Y | 32 | 1.00 | 0.354 | 3.199 | 0.561 | 2.638 [2.248, 3.028] | 0.00 / 0.00 | Y |
| kicking_something | foot_kick | Y | 32 | 0.97 | 0.367 | 3.377 | 0.750 | 2.627 [2.203, 2.994] | 0.00 / 0.00 | Y |
| reach_into_pocket | foot_static | Y | 32 | 1.00 | 0.551 | 29.054 | 6.936 | 22.118 [21.250, 22.844] | 1.00 / 0.09 | Y |
| reach_into_pocket | hip_static_stand | Y | 32 | 1.00 | 1.439 | 3.275 | 14.666 | -11.391 [-13.182, -9.598] | 0.19 / 0.62 | n |
| reach_into_pocket | leg_static_stand | Y | 32 | 1.00 | 1.454 | 2.758 | 13.701 | -10.943 [-12.204, -9.725] | 0.12 / 0.56 | n |
| reach_into_pocket | arm_retract | Y | 21 | 1.00 | 0.276 | 0.422 | 0.343 | 0.078 [-1.089, 1.281] | 0.43 / 0.38 | n |
| hopping | arm_swing | Y | 22 | 1.00 | 0.267 | 0.374 | -0.444 | 0.818 [0.382, 1.227] | 0.00 / 0.00 | Y |
| hopping | temporal_direction_motion_upward | Y | 22 | 1.00 | 0.210 | 0.250 | -0.486 | 0.736 [0.222, 1.224] | 0.00 / 0.00 | Y |
| hopping | leg_alternating | Y | 2 | 1.00 | 0.239 | 0.290 | -0.163 | 0.453 [-0.077, 0.983] | 0.00 / 0.00 | n |
| hopping | hip_extend | Y | 32 | 1.00 | 0.183 | 0.352 | -0.029 | 0.380 [0.107, 0.680] | 0.03 / 0.03 | Y |
| jump_up | hip_extend | Y | 32 | 1.00 | 0.231 | 1.985 | 0.614 | 1.372 [1.143, 1.588] | 0.00 / 0.00 | Y |
| jump_up | temporal_dynamics_speed_fast | Y | 1 | 1.00 | 1.221 | 2.765 | 13.937 | -11.172 [-11.172, -11.172] | 0.00 / 0.00 | n |
| jump_up | leg_jump | Y | 32 | 0.72 | 0.383 | 1.629 | 1.223 | 0.406 [-0.014, 0.797] | 0.00 / 0.00 | n |
| jump_up | foot_jump | Y | 32 | 0.81 | 0.371 | 1.632 | 1.138 | 0.494 [0.123, 0.873] | 0.00 / 0.00 | Y |
| phone_call | hand_lift_to_face | Y | 32 | 1.00 | 1.098 | 15.450 | 20.658 | -5.209 [-6.471, -3.867] | 0.91 / 1.00 | n |
| phone_call | foot_static | Y | 32 | 1.00 | 0.537 | 11.139 | 3.331 | 7.808 [6.788, 8.767] | 1.00 / 0.06 | Y |
| phone_call | hip_static_stand | Y | 32 | 1.00 | 1.404 | 1.010 | 9.794 | -8.784 [-10.207, -7.139] | 0.12 / 0.28 | n |
| phone_call | arm_hold | Y | 32 | 1.00 | 0.838 | 11.370 | 18.104 | -6.734 [-8.541, -4.953] | 0.66 / 1.00 | n |
| play_with_phone_tablet | arm_hold | Y | 26 | 1.00 | 0.912 | 8.499 | 24.134 | -15.635 [-16.950, -14.051] | 0.69 / 1.00 | n |
| play_with_phone_tablet | head_tilt_down | Y | 1 | 1.00 | 0.183 | 3.086 | 5.578 | -2.492 [-2.492, -2.492] | 0.00 / 0.00 | n |
| play_with_phone_tablet | foot_static | Y | 32 | 1.00 | 0.544 | -41.298 | -1.911 | -39.387 [-41.788, -37.097] | 0.00 / 0.16 | n |
| type_on_a_keyboard | foot_static | Y | 32 | 1.00 | 0.545 | -7.592 | 4.087 | -11.679 [-13.354, -9.967] | 0.97 / 0.53 | n |
| type_on_a_keyboard | temporal_dynamics_speed_fast | Y | 7 | 1.00 | 1.149 | 2.632 | 10.567 | -7.935 [-11.703, -5.118] | 0.86 / 1.00 | n |
| type_on_a_keyboard | arm_hold | Y | 29 | 1.00 | 0.976 | 4.725 | 11.794 | -7.069 [-8.662, -5.349] | 0.90 / 1.00 | n |
| point_to_something | arm_raise | n | 32 | 0.97 | 0.977 | 8.416 | 21.344 | -12.927 [-14.648, -11.036] | 0.77 / 1.00 | n |
| point_to_something | foot_static | Y | 32 | 1.00 | 0.559 | 21.856 | 6.211 | 15.645 [14.785, 16.440] | 0.97 / 0.56 | Y |
| point_to_something | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.442 | 12.032 | 15.984 | -3.952 [-5.102, -2.828] | 0.94 / 0.97 | n |
| point_to_something | hip_static_stand | Y | 32 | 1.00 | 1.492 | 0.889 | 11.641 | -10.752 [-11.869, -9.502] | 0.09 / 0.66 | n |
| taking_a_selfie | arm_raise | Y | 32 | 1.00 | 0.981 | 7.646 | 20.855 | -13.209 [-14.690, -11.612] | 0.16 / 1.00 | n |
| taking_a_selfie | foot_static | Y | 32 | 1.00 | 0.560 | 14.082 | 3.865 | 10.217 [9.400, 11.048] | 1.00 / 0.12 | Y |
| taking_a_selfie | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.464 | 6.179 | 12.478 | -6.299 [-7.626, -4.965] | 0.38 / 0.75 | n |
| taking_a_selfie | hip_static_stand | Y | 32 | 1.00 | 1.556 | 2.077 | 11.791 | -9.714 [-10.774, -8.621] | 0.06 / 0.19 | n |
| check_time_(from_watch) | arm_raise | Y | 32 | 1.00 | 1.027 | 5.334 | 25.284 | -19.950 [-21.825, -17.825] | 0.22 / 1.00 | n |
| check_time_(from_watch) | hip_static_stand | Y | 32 | 1.00 | 1.467 | 0.861 | 10.564 | -9.703 [-10.840, -8.395] | 0.06 / 0.19 | n |
| check_time_(from_watch) | foot_static | Y | 32 | 1.00 | 0.568 | 16.063 | 4.846 | 11.217 [10.457, 11.957] | 1.00 / 0.03 | Y |
| check_time_(from_watch) | leg_static_stand | Y | 32 | 1.00 | 1.437 | 0.675 | 11.599 | -10.924 [-12.031, -9.824] | 0.03 / 0.31 | n |
| rub_two_hands | foot_static | Y | 32 | 1.00 | 0.567 | 12.091 | 2.982 | 9.109 [8.347, 9.870] | 1.00 / 0.22 | Y |
| rub_two_hands | arm_hold | Y | 23 | 1.00 | 0.991 | 5.754 | 17.122 | -11.368 [-13.700, -8.812] | 0.39 / 0.96 | n |
| rub_two_hands | temporal_direction_motion_converging | Y | 32 | 1.00 | 0.317 | 4.505 | 5.477 | -0.972 [-2.198, 0.344] | 0.38 / 0.44 | n |
| rub_two_hands | hip_static_stand | Y | 32 | 1.00 | 1.435 | 0.139 | 9.571 | -9.432 [-10.653, -8.313] | 0.12 / 0.44 | n |
| nod_head_bow | hip_bend | Y | 32 | 0.97 | 0.495 | 3.641 | 0.649 | 2.991 [2.680, 3.315] | 0.03 / 0.03 | Y |
| nod_head_bow | leg_static_stand | Y | 32 | 1.00 | 1.443 | 6.336 | 12.757 | -6.421 [-7.237, -5.497] | 0.06 / 0.31 | n |
| nod_head_bow | foot_static | Y | 32 | 1.00 | 0.526 | 21.900 | 6.622 | 15.278 [14.658, 15.996] | 1.00 / 0.06 | Y |
| nod_head_bow | temporal_direction_motion_downward | Y | 32 | 1.00 | 0.192 | 2.499 | 2.867 | -0.368 [-0.715, -0.046] | 0.09 / 0.12 | n |
| shake_head | hip_static_stand | Y | 32 | 1.00 | 1.427 | 0.051 | 15.750 | -15.699 [-17.110, -14.187] | 0.06 / 0.84 | n |
| shake_head | foot_static | Y | 32 | 1.00 | 0.551 | 18.228 | 6.133 | 12.095 [11.445, 12.699] | 1.00 / 0.16 | Y |
| shake_head | leg_static_stand | Y | 32 | 1.00 | 1.435 | 0.590 | 16.421 | -15.830 [-17.314, -14.344] | 0.09 / 0.94 | n |
| shake_head | head_shake | Y | 22 | 0.23 | 0.073 | 5.634 | 1.809 | 3.825 [-0.163, 7.715] | 0.00 / 0.00 | n |
| wipe_face | temporal_direction_motion_upward | Y | 32 | 1.00 | 0.212 | 3.405 | 0.332 | 3.073 [1.754, 4.437] | 0.16 / 0.16 | Y |
| wipe_face | hip_static_stand | Y | 32 | 1.00 | 1.476 | 2.976 | 15.004 | -12.028 [-13.354, -10.529] | 0.09 / 0.53 | n |
| wipe_face | leg_static_stand | Y | 32 | 1.00 | 1.436 | 3.114 | 14.420 | -11.306 [-12.769, -9.966] | 0.19 / 0.53 | n |
| wipe_face | hand_lift_to_face | Y | 32 | 1.00 | 1.356 | 14.927 | 20.841 | -5.915 [-7.712, -3.975] | 0.78 / 1.00 | n |
| salute | hip_static_stand | Y | 32 | 1.00 | 1.420 | 3.463 | 15.247 | -11.784 [-12.823, -10.765] | 0.09 / 0.44 | n |
| salute | leg_static_stand | Y | 32 | 1.00 | 1.425 | 3.633 | 15.180 | -11.547 [-12.673, -10.378] | 0.09 / 0.72 | n |
| salute | foot_static | Y | 32 | 1.00 | 0.557 | 17.658 | 8.150 | 9.508 [8.714, 10.254] | 1.00 / 0.16 | Y |
| salute | temporal_direction_motion_upward | Y | 31 | 1.00 | 0.218 | 5.719 | 11.982 | -6.263 [-7.810, -4.872] | 0.45 / 0.77 | n |
| put_palms_together | hip_static_stand | Y | 32 | 1.00 | 1.460 | 1.719 | 9.747 | -8.028 [-9.042, -7.121] | 0.03 / 0.19 | n |
| put_palms_together | leg_static_stand | Y | 32 | 1.00 | 1.416 | 1.871 | 10.780 | -8.909 [-10.125, -7.624] | 0.09 / 0.28 | n |
| put_palms_together | arm_hold | Y | 32 | 1.00 | 0.862 | 18.177 | 24.024 | -5.847 [-6.724, -5.008] | 1.00 / 1.00 | n |
| cross_hands_in_front | arm_raise | n | 32 | 1.00 | 0.778 | 8.878 | 17.383 | -8.505 [-9.961, -7.061] | 0.41 / 0.91 | n |
| cross_hands_in_front | hip_static_stand | Y | 32 | 1.00 | 1.437 | 2.044 | 13.373 | -11.330 [-12.402, -10.337] | 0.03 / 0.19 | n |
| cross_hands_in_front | arm_overhead | n | 2 | 1.00 | 0.841 | 3.269 | 6.794 | -3.526 [-9.692, 2.641] | 1.00 / 1.00 | n |
| sneeze_cough | leg_static_stand | Y | 32 | 1.00 | 1.463 | 2.125 | 16.934 | -14.808 [-15.732, -13.863] | 0.12 / 0.91 | n |
| sneeze_cough | foot_static | Y | 32 | 1.00 | 0.553 | 23.559 | 9.085 | 14.473 [13.692, 15.339] | 1.00 / 0.50 | Y |
| sneeze_cough | temporal_dynamics_speed_fast | Y | 2 | 1.00 | 0.945 | -0.437 | 8.425 | -8.862 [-10.456, -7.267] | 1.00 / 1.00 | n |
| sneeze_cough | hand_lift_to_face | Y | 32 | 1.00 | 1.221 | 11.089 | 21.708 | -10.619 [-12.314, -9.000] | 0.53 / 1.00 | n |
| staggering | leg_alternating | Y | 20 | 1.00 | 0.242 | 1.460 | -0.394 | 1.855 [1.518, 2.124] | 0.00 / 0.00 | Y |
| staggering | hip_sway | Y | 32 | 0.94 | 0.199 | 1.366 | -0.071 | 1.437 [1.209, 1.650] | 0.00 / 0.00 | Y |
| staggering | foot_step | Y | 32 | 0.97 | 0.271 | 2.050 | -0.788 | 2.837 [2.429, 3.241] | 0.06 / 0.00 | Y |
| falling_down | temporal_direction_motion_downward | Y | 32 | 1.00 | 0.212 | 1.937 | 1.209 | 0.728 [0.220, 1.215] | 0.03 / 0.03 | Y |
| falling_down | hip_bend | Y | 32 | 0.97 | 0.487 | 1.855 | 0.288 | 1.567 [1.200, 1.905] | 0.03 / 0.03 | Y |
| falling_down | leg_squat | Y | 32 | 1.00 | 0.698 | 2.334 | 3.298 | -0.964 [-1.755, -0.127] | 0.06 / 0.06 | n |
| headache | hip_static_stand | Y | 32 | 1.00 | 1.475 | 3.150 | 10.407 | -7.257 [-8.035, -6.527] | 0.19 / 0.31 | n |
| headache | leg_static_stand | Y | 32 | 1.00 | 1.468 | 3.084 | 10.722 | -7.639 [-8.665, -6.635] | 0.25 / 0.31 | n |
| headache | hand_lift_to_face | Y | 32 | 1.00 | 1.329 | 15.787 | 19.614 | -3.827 [-5.100, -2.647] | 0.91 / 1.00 | n |
| headache | arm_raise | Y | 32 | 1.00 | 0.977 | 13.307 | 18.657 | -5.350 [-7.114, -3.840] | 0.72 / 1.00 | n |
| chest_pain | foot_static | Y | 32 | 1.00 | 0.543 | 23.060 | 8.917 | 14.143 [13.301, 14.974] | 1.00 / 0.06 | Y |
| chest_pain | arm_retract | Y | 24 | 1.00 | 0.287 | 4.145 | 5.503 | -1.358 [-2.440, -0.227] | 0.29 / 0.50 | n |
| chest_pain | hip_static_stand | Y | 32 | 1.00 | 1.425 | 2.580 | 15.220 | -12.640 [-13.651, -11.559] | 0.09 / 0.38 | n |
| back_pain | leg_static_stand | Y | 32 | 1.00 | 1.439 | 3.854 | 14.118 | -10.265 [-11.490, -9.013] | 0.06 / 0.22 | n |
| back_pain | foot_static | Y | 32 | 1.00 | 0.553 | 19.604 | 6.232 | 13.372 [12.792, 13.978] | 1.00 / 0.09 | Y |
| back_pain | hip_bend | Y | 16 | 0.94 | 0.564 | 1.344 | 1.241 | 0.103 [-0.673, 0.871] | 0.40 / 0.47 | n |
| neck_pain | hip_static_stand | Y | 32 | 1.00 | 1.416 | 1.771 | 14.902 | -13.131 [-14.539, -11.642] | 0.12 / 0.34 | n |
| neck_pain | leg_static_stand | Y | 32 | 1.00 | 1.497 | 2.698 | 15.767 | -13.069 [-14.558, -11.472] | 0.28 / 0.47 | n |
| neck_pain | foot_static | Y | 32 | 1.00 | 0.547 | 16.021 | 7.798 | 8.223 [7.674, 8.735] | 1.00 / 0.22 | Y |
| neck_pain | arm_raise | Y | 32 | 1.00 | 0.970 | 10.217 | 20.979 | -10.762 [-12.755, -8.863] | 0.28 / 1.00 | n |
| nausea_vomiting | temporal_dynamics_speed_fast | Y | 32 | 1.00 | 0.882 | 0.540 | 7.376 | -6.836 [-8.083, -5.642] | 0.03 / 0.25 | n |
| nausea_vomiting | foot_static | Y | 32 | 1.00 | 0.536 | 17.397 | 4.424 | 12.972 [12.356, 13.595] | 1.00 / 0.22 | Y |
| nausea_vomiting | leg_static_stand | Y | 32 | 1.00 | 1.457 | 3.355 | 9.432 | -6.077 [-6.782, -5.320] | 0.09 / 0.19 | n |
| nausea_vomiting | hip_bend | Y | 32 | 0.91 | 0.576 | 2.216 | 0.148 | 2.067 [1.340, 2.979] | 0.07 / 0.07 | Y |
| fan_self | temporal_dynamics_speed_fast | Y | 6 | 1.00 | 1.008 | 16.431 | 13.254 | 3.177 [-3.576, 8.737] | 0.83 / 0.50 | n |
| fan_self | foot_static | Y | 32 | 1.00 | 0.544 | 15.265 | 1.444 | 13.821 [13.177, 14.473] | 1.00 / 0.06 | Y |
| fan_self | leg_static_stand | Y | 32 | 1.00 | 1.463 | 2.991 | 4.894 | -1.903 [-2.666, -1.189] | 0.06 / 0.03 | n |
| fan_self | hip_static_stand | Y | 32 | 1.00 | 1.458 | 2.977 | 5.714 | -2.736 [-3.670, -1.848] | 0.06 / 0.00 | n |
| punch_slap | leg_static_stand | Y | 32 | 1.00 | 1.380 | 2.622 | 10.683 | -8.061 [-9.007, -7.119] | 0.09 / 0.19 | n |
| punch_slap | arm_extend | Y | 1 | 1.00 | 0.605 | 7.365 | 7.394 | -0.029 [-0.029, -0.029] | 0.00 / 0.00 | n |
| punch_slap | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.444 | 7.770 | 7.160 | 0.610 [-0.472, 1.629] | 0.19 / 0.19 | n |
| punch_slap | arm_swing | Y | 32 | 1.00 | 0.282 | 1.861 | 3.684 | -1.823 [-2.434, -1.223] | 0.03 / 0.12 | n |
| kicking | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.443 | 6.507 | 3.455 | 3.052 [2.553, 3.507] | 0.12 / 0.12 | Y |
| kicking | temporal_dynamics_speed_fast | Y | 1 | 1.00 | 0.933 | 3.058 | 0.579 | 2.479 [2.479, 2.479] | 1.00 / 1.00 | Y |
| kicking | hip_rotate | Y | 32 | 0.88 | 0.114 | 0.978 | 0.622 | 0.356 [-0.156, 0.803] | 0.04 / 0.04 | n |
| kicking | leg_balance | Y | 25 | 0.96 | 0.512 | 2.112 | 4.175 | -2.063 [-2.895, -1.368] | 0.00 / 0.00 | n |
| pushing | leg_static_stand | Y | 32 | 1.00 | 1.366 | 3.430 | 10.600 | -7.170 [-8.015, -6.339] | 0.03 / 0.19 | n |
| pushing | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.443 | 5.671 | 3.442 | 2.229 [1.511, 3.016] | 0.09 / 0.09 | Y |
| pushing | hip_rotate | Y | 32 | 0.81 | 0.116 | 0.730 | 0.459 | 0.271 [-0.282, 0.742] | 0.04 / 0.04 | n |
| pat_on_back | foot_static | Y | 32 | 1.00 | 0.547 | 18.254 | 3.185 | 15.068 [14.618, 15.498] | 1.00 / 0.03 | Y |
| pat_on_back | leg_static_stand | Y | 32 | 1.00 | 1.407 | 3.476 | 5.822 | -2.345 [-2.988, -1.696] | 0.06 / 0.06 | n |
| pat_on_back | arm_extend | Y | 2 | 1.00 | 0.477 | 3.878 | 0.403 | 3.475 [1.352, 5.598] | 0.50 / 0.00 | Y |
| pat_on_back | hip_static_stand | Y | 32 | 1.00 | 1.355 | 3.290 | 5.375 | -2.086 [-2.797, -1.290] | 0.06 / 0.06 | n |
| point_finger | foot_static | Y | 32 | 1.00 | 0.556 | 19.355 | 5.003 | 14.352 [13.857, 14.829] | 1.00 / 0.12 | Y |
| point_finger | hip_static_stand | Y | 32 | 1.00 | 1.387 | 2.757 | 7.878 | -5.121 [-5.895, -4.409] | 0.09 / 0.19 | n |
| point_finger | leg_static_stand | Y | 32 | 1.00 | 1.398 | 2.971 | 7.999 | -5.027 [-5.490, -4.544] | 0.22 / 0.16 | n |
| point_finger | arm_extend | Y | 4 | 1.00 | 0.540 | 3.818 | 4.478 | -0.659 [-2.219, 0.313] | 0.00 / 0.25 | n |
| hugging | foot_static | Y | 32 | 1.00 | 0.537 | 16.117 | 5.111 | 11.006 [10.580, 11.478] | 1.00 / 0.00 | Y |
| hugging | hip_static_stand | Y | 32 | 1.00 | 1.301 | 3.384 | 9.600 | -6.217 [-6.782, -5.639] | 0.03 / 0.06 | n |
| hugging | leg_static_stand | Y | 32 | 1.00 | 1.345 | 3.340 | 9.539 | -6.200 [-6.890, -5.478] | 0.06 / 0.06 | n |
| hugging | temporal_direction_motion_converging | Y | 32 | 1.00 | 0.334 | 1.492 | 1.655 | -0.162 [-0.671, 0.358] | 0.00 / 0.00 | n |
| giving_object | foot_static | Y | 32 | 1.00 | 0.535 | 11.733 | 3.620 | 8.113 [7.474, 8.726] | 0.97 / 0.09 | Y |
| giving_object | hip_static_stand | Y | 32 | 1.00 | 1.378 | 1.564 | 9.030 | -7.466 [-8.369, -6.663] | 0.06 / 0.12 | n |
| giving_object | leg_static_stand | Y | 32 | 1.00 | 1.349 | 1.917 | 8.718 | -6.800 [-7.545, -6.125] | 0.03 / 0.03 | n |
| giving_object | temporal_direction_motion_forward | Y | 32 | 1.00 | 0.436 | 4.452 | 4.859 | -0.406 [-1.074, 0.307] | 0.03 / 0.06 | n |
| touch_pocket | hip_static_stand | Y | 32 | 1.00 | 1.379 | 4.096 | 8.716 | -4.620 [-5.627, -3.488] | 0.06 / 0.19 | n |
| touch_pocket | leg_static_stand | Y | 32 | 1.00 | 1.387 | 4.400 | 8.716 | -4.317 [-5.221, -3.342] | 0.03 / 0.12 | n |
| touch_pocket | foot_static | Y | 32 | 1.00 | 0.528 | 7.687 | 4.267 | 3.420 [2.941, 3.833] | 0.97 / 0.09 | Y |
| shaking_hands | foot_static | Y | 32 | 1.00 | 0.536 | 19.539 | 5.300 | 14.239 [13.804, 14.725] | 1.00 / 0.03 | Y |
| shaking_hands | hip_static_stand | Y | 32 | 1.00 | 1.353 | 2.912 | 11.257 | -8.345 [-9.105, -7.550] | 0.00 / 0.06 | n |
| walking_towards | leg_alternating | Y | 32 | 1.00 | 0.254 | 1.552 | 0.064 | 1.489 [1.301, 1.649] | 0.00 / 0.00 | Y |
| walking_towards | hip_sway | Y | 32 | 0.91 | 0.174 | 1.696 | 0.141 | 1.555 [1.384, 1.727] | 0.00 / 0.00 | Y |
| walking_towards | arm_swing | Y | 32 | 1.00 | 0.248 | 0.712 | 0.201 | 0.512 [0.287, 0.736] | 0.00 / 0.00 | Y |
| walking_towards | foot_step | Y | 32 | 0.97 | 0.264 | 2.635 | 0.184 | 2.451 [2.096, 2.822] | 0.00 / 0.00 | Y |
| walking_apart | leg_alternating | Y | 32 | 1.00 | 0.251 | 1.664 | -0.314 | 1.978 [1.776, 2.209] | 0.00 / 0.00 | Y |
| walking_apart | arm_swing | Y | 32 | 1.00 | 0.268 | 1.528 | -0.749 | 2.277 [1.920, 2.662] | 0.06 / 0.00 | Y |
| walking_apart | hip_sway | Y | 32 | 0.81 | 0.175 | 0.987 | 0.209 | 0.778 [0.679, 0.878] | 0.00 / 0.00 | Y |
| walking_apart | leg_step | Y | 32 | 0.91 | 0.251 | 2.936 | -0.201 | 3.137 [2.775, 3.470] | 0.00 / 0.00 | Y |

## 6. Robustness

| perturbation | level | HyperGCN original | HyperGCN ft | logic | fused |
|---|---|---|---|---|---|
| clean | 0.0 | 0.9108 | 0.9101 | 0.7339 | 0.9098 |
| noise | 0.01 | 0.8260 | 0.7652 | 0.4950 | 0.7696 |
| noise | 0.03 | 0.5974 | 0.1630 | 0.1965 | 0.2890 |
| joint_drop | 0.2 | 0.8524 | 0.8436 | 0.4682 | 0.8028 |
| frame_drop | 0.3 | 0.2433 | 0.2359 | 0.1621 | 0.2614 |
