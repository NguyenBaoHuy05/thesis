"""Configuration Generator for CLIR-SkateFormer.

Generates YAML configuration files for NTU RGB+D 60 and NTU RGB+D 120 benchmarks,
supporting standard runs and all ablation studies from Section 11:
- A2: CLIR (beta=0, no refinement loop)
- A3: Loop variations (K=1, K=2, K=3)
- A4: Main CLIR-SkateFormer (K=2, logic-guided refinement)
- A5: CLIR + Random Logic Feedback (random_feedback=True)
- A6: CLIR-Bypass (use_feature_bypass=True)
"""

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


def build_config(
    split_name: str,
    info: dict,
    num_loops: int = 2,
    beta_init: float = 0.1,
    beta_max: float = 0.5,
    ablation_tag: str = "",
    random_feedback: bool = False,
    use_feature_bypass: bool = False,
) -> dict:
    num_classes = info["num_classes"]
    train_data = info["train_data"]
    test_data = info["test_data"]
    weights = info["backbone_weights"]

    tag = f"_{ablation_tag}" if ablation_tag else f"_K{num_loops}"
    work_dir = f"./work_dir/{split_name}/CLIR_SkateFormer_j{tag}"

    return {
        "phase": "train",
        "seed": 1,
        "work_dir": work_dir,
        "feeder": "feeders.feeder_ntu.Feeder",
        "num_worker": 4,
        "train_feeder_args": {
            "data_path": train_data,
            "concepts_csv": "./concepts/ntu_rgbd_60_120.csv",
            "split": "train",
            "window_size": 64,
            "thres": 64,
            "uniform": True,
            "partition": True,
            "remove_null_concept": True,
            "p_interval": [0.5, 1],
            "aug_method": "a123489",
            "intra_p": 0.5,
            "inter_p": 0.2,
            "debug": False,
        },
        "test_feeder_args": {
            "data_path": test_data,
            "concepts_csv": "./concepts/ntu_rgbd_60_120.csv",
            "split": "test",
            "window_size": 64,
            "thres": 64,
            "uniform": True,
            "partition": True,
            "remove_null_concept": True,
            "p_interval": [0.95],
            "debug": False,
        },
        "model": "model.clir_skateformer.CLIR_SkateFormer",
        "model_args": {
            "num_classes": num_classes,
            "num_concepts": 72,
            "spatial_concepts": 52,
            "temporal_concepts": 20,
            "num_people": 2,
            "num_points": 24,
            "num_frames": 64,
            "backbone_weights": weights,
            "num_loops": num_loops,
            "r1": 256,
            "r2a": 128,
            "r2b": 128,
            "beta_max": beta_max,
            "beta_init": beta_init,
            "literal_mode": "soft",
            "temperature": 5.0,
            "hidden_dim": 256,
            "dropout": 0.1,
            "lambda_sup": 1.0,
            "lambda_reg": 0.0,
            "concept_negative_weight": 0.15,
            "concept_positive_weight": 1.0,
            "asymmetric_gamma_neg": 2.0,
            "return_aux": True,
            "random_feedback": random_feedback,
            "use_feature_bypass": use_feature_bypass,
            "num_heads": 32,
            "mlp_ratio": 4.0,
            "drop_path": 0.2,
            "attn_drop": 0.5,
            "head_drop": 0.0,
            "rel": True,
            "index_t": True,
            "type_1_size": [8, 8],
            "type_2_size": [8, 12],
            "type_3_size": [8, 8],
            "type_4_size": [8, 12],
        },
        "optimizer": "AdamW",
        "base_lr": 0.0001,
        "min_lr": 0.000001,
        "warmup_lr": 0.000001,
        "warm_up_epoch": 10,
        "warmup_prefix": False,
        "weight_decay": 0.05,
        "nesterov": True,
        "lr_scheduler": "cosine",
        "grad_clip": True,
        "grad_max": 1.0,
        "batch_size": 32,
        "test_batch_size": 32,
        "num_epoch": 70,
        "start_epoch": 0,
        "loss_type": "LSCE",
        "phase1_epochs": 30,
        "backbone_lr_scale": 0.1,
        "crl_lr_scale": 1.0,
        "update_cell_lr_scale": 2.0,
        "ste_temperature_start": 5.0,
        "ste_temperature_end": 10.0,
        "lambda_reg": 0.005,
        "log_interval": 100,
        "save_interval": 5,
        "save_epoch": 30,
        "eval_interval": 5,
        "print_log": True,
        "show_topk": [1, 5],
        "save_score": False,
        "save_trace": False,
        "device": [0],
    }


def main():
    for split_name, info in DATASET_CONFIGS.items():
        out_dir = CONFIG_DIR / "train" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Standard K=2 (Main Model A4)
        cfg_k2 = build_config(split_name, info, num_loops=2, ablation_tag="A4_K2")
        with open(out_dir / "CLIR_SkateFormer_j_K2.yaml", "w") as f:
            yaml.dump(cfg_k2, f, sort_keys=False)

        # Standard default alias
        with open(out_dir / "CLIR_SkateFormer_j.yaml", "w") as f:
            yaml.dump(cfg_k2, f, sort_keys=False)

        # Ablation A3: K=1
        cfg_k1 = build_config(split_name, info, num_loops=1, ablation_tag="A3_K1")
        with open(out_dir / "CLIR_SkateFormer_j_K1.yaml", "w") as f:
            yaml.dump(cfg_k1, f, sort_keys=False)

        # Ablation A3: K=3
        cfg_k3 = build_config(split_name, info, num_loops=3, ablation_tag="A3_K3")
        with open(out_dir / "CLIR_SkateFormer_j_K3.yaml", "w") as f:
            yaml.dump(cfg_k3, f, sort_keys=False)

        # Ablation A2: True No-Loop (num_loops=0, beta=0)
        cfg_a2 = build_config(split_name, info, num_loops=0, beta_init=0.0, beta_max=0.0, ablation_tag="A2_NoLoop")
        with open(out_dir / "CLIR_SkateFormer_j_A2_NoLoop.yaml", "w") as f:
            yaml.dump(cfg_a2, f, sort_keys=False)

        # Ablation A5: Random Logic Feedback
        cfg_a5 = build_config(split_name, info, num_loops=2, random_feedback=True, ablation_tag="A5_RandomFeedback")
        with open(out_dir / "CLIR_SkateFormer_j_A5_RandomFeedback.yaml", "w") as f:
            yaml.dump(cfg_a5, f, sort_keys=False)

        # Ablation A6: CLIR-Bypass (Feature bypass)
        cfg_a6 = build_config(split_name, info, num_loops=2, use_feature_bypass=True, ablation_tag="A6_Bypass")
        with open(out_dir / "CLIR_SkateFormer_j_A6_Bypass.yaml", "w") as f:
            yaml.dump(cfg_a6, f, sort_keys=False)

    print(f"Generated all configurations under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
