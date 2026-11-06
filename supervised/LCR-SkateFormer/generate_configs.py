"""Configuration Generator for LCR-SkateFormer.

Generates YAML configuration files for NTU RGB+D 60, NTU RGB+D 120, and NW-UCLA datasets,
supporting multiple modalities (joint, bone, motion) and ablation studies (K=1, 2, 3, linear/prototype, shared/unshared).
"""

import os
from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"

DATASET_CONFIGS = {
    "ntu60_cs": {
        "num_classes": 60,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CS_64f.npz",
        "backbone_weights": "/content/ntu60_CS/SkateFormer_j.pt",
    },
    "ntu60_cv": {
        "num_classes": 60,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz",
        "backbone_weights": "/content/ntu60_CV/SkateFormer_j.pt",
    },
    "ntu120_csub": {
        "num_classes": 120,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSub_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSub_64f.npz",
        "backbone_weights": "/content/ntu120_CSub/SkateFormer_j.pt",
    },
    "ntu120_cset": {
        "num_classes": 120,
        "train_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSet_64f.npz",
        "test_data": "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU120_CSet_64f.npz",
        "backbone_weights": "/content/ntu120_CSet/SkateFormer_j.pt",
    },
}

KG_PRIOR_MAP = {
    "require": [
        ["arm_raise", "arm_extend"],
        ["leg_step", "foot_step"],
        ["hand_wave", "arm_raise"],
    ],
    "alongside": [
        ["arm_raise", "hand_wave"],
        ["leg_step", "foot_step"],
    ],
    "excludes": [
        ["arm_raise", "hip_sit"],
        ["leg_step", "hip_squat"],
    ],
}


def build_config(
    split_name,
    info,
    num_loops=2,
    shared_weights=True,
    action_mode="prototype",
    modality="j",
):
    work_dir = f"./work_dir/{split_name}/LCR_SkateFormer_{modality}_K{num_loops}_{action_mode}"
    config = {
        "seed": 1,
        "num_worker": 4,
        "work_dir": work_dir,
        "phase": "train",
        "feeder": "feeders.feeder_ntu.Feeder",
        "train_feeder_args": {
            "data_path": info["train_data"],
            "concepts_csv": "./concepts/ntu_rgbd_60_120.csv",
            "remove_null_concept": True,
            "split": "train",
            "debug": False,
            "window_size": 64,
            "p_interval": [0.5, 1],
            "aug_method": "a123489",
            "intra_p": 0.5,
            "inter_p": 0.2,
            "thres": 64,
            "uniform": True,
            "partition": True,
        },
        "test_feeder_args": {
            "data_path": info["test_data"],
            "concepts_csv": "./concepts/ntu_rgbd_60_120.csv",
            "remove_null_concept": True,
            "split": "test",
            "window_size": 64,
            "p_interval": [0.95],
            "thres": 64,
            "uniform": True,
            "partition": True,
            "debug": False,
        },
        "model": "model.lcr_skateformer.LCR_SkateFormer",
        "model_args": {
            "num_classes": info["num_classes"],
            "num_concepts": 72,
            "spatial_concepts": 52,
            "temporal_concepts": 20,
            "num_people": 2,
            "num_points": 24,
            "num_frames": 64,
            "backbone_weights": info["backbone_weights"],
            "num_heads": 32,
            "attn_drop": 0.5,
            "head_drop": 0.0,
            "rel": True,
            "drop_path": 0.2,
            "type_1_size": [8, 8],
            "type_2_size": [8, 12],
            "type_3_size": [8, 8],
            "type_4_size": [8, 12],
            "mlp_ratio": 4.0,
            "index_t": True,
            # Looping Concept Reasoning Parameters
            "num_loops": num_loops,
            "shared_weights": shared_weights,
            "update_gate_init": 0.1,
            "max_gate": 0.5,
            "use_action_feedback": False,
            "l1_dim": 128,
            "l2_dim": 128,
            "kg_init_mode": "kg_primed",
            "action_mode": action_mode,
            "prototype_and_weight": 0.35,
            "kg_prior_strength": 1.0,
            # Loss weights
            "concept_loss_weight": 1.0,
            "early_concept_loss_weight": 0.5,
            "aux_action_loss_weight": 0.3,
            "stability_loss_weight": 0.05,
            "kg_loss_weight": 0.01,
            "prior_retention_weight": 0.01,
            "rule_sparsity_weight": 0.001,
            "part_diversity_weight": 0.01,
            "concept_negative_weight": 0.15,
            "concept_positive_weight": 1.0,
            "asymmetric_gamma_neg": 2.0,
            "return_aux": True,
            "kg_prior_map": KG_PRIOR_MAP,
        },
        "optimizer": "AdamW",
        "weight_decay": 0.05,
        "lr_scheduler": "cosine",
        "base_lr": 0.0001,
        "min_lr": 0.000001,
        "warmup_lr": 0.000001,
        "warmup_prefix": False,
        "warm_up_epoch": 10,
        "device": [0],
        "batch_size": 32,
        "test_batch_size": 32,
        "num_epoch": 150,
        "nesterov": True,
        "grad_clip": True,
        "grad_max": 1.0,
        "loss_type": "LSCE",
    }
    return config


def generate_all_configs():
    for split_name, info in DATASET_CONFIGS.items():
        train_dir = CONFIG_DIR / "train" / split_name
        test_dir = CONFIG_DIR / "test" / split_name
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Standard K=2 prototype (Proposed Main Model)
        main_cfg = build_config(split_name, info, num_loops=2, action_mode="prototype")
        with open(train_dir / "LCR_SkateFormer_j_K2_proto.yaml", "w") as f:
            yaml.dump(main_cfg, f, default_flow_style=False)

        # K=2 Linear baseline
        linear_cfg = build_config(split_name, info, num_loops=2, action_mode="linear")
        with open(train_dir / "LCR_SkateFormer_j_K2_linear.yaml", "w") as f:
            yaml.dump(linear_cfg, f, default_flow_style=False)

        # Ablation K=1 (Single Step / ECR baseline)
        k1_cfg = build_config(split_name, info, num_loops=1, action_mode="prototype")
        with open(train_dir / "LCR_SkateFormer_j_K1_proto.yaml", "w") as f:
            yaml.dump(k1_cfg, f, default_flow_style=False)

        # Ablation K=3
        k3_cfg = build_config(split_name, info, num_loops=3, action_mode="prototype")
        with open(train_dir / "LCR_SkateFormer_j_K3_proto.yaml", "w") as f:
            yaml.dump(k3_cfg, f, default_flow_style=False)

        # Ablation K=2 Unshared weights
        unshared_cfg = build_config(split_name, info, num_loops=2, shared_weights=False, action_mode="prototype")
        with open(train_dir / "LCR_SkateFormer_j_K2_unshared.yaml", "w") as f:
            yaml.dump(unshared_cfg, f, default_flow_style=False)

        # Test config (eval phase)
        test_cfg = dict(main_cfg)
        test_cfg["phase"] = "test"
        test_cfg["save_score"] = True
        test_cfg["save_trace"] = True
        with open(test_dir / "test_LCR_SkateFormer_j.yaml", "w") as f:
            yaml.dump(test_cfg, f, default_flow_style=False)

    print(f"[SUCCESS] All LCR-SkateFormer configs generated in: {CONFIG_DIR}")


if __name__ == "__main__":
    generate_all_configs()
