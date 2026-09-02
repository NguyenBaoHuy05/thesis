import yaml
from pathlib import Path

base_yaml_path = Path(r"d:\Thesis\SFB-ECR-SkateFormer\config\train\ntu_cs\SFB_ECR_SkateFormer_j.yaml")
with open(base_yaml_path, 'r', encoding='utf-8') as f:
    base_cfg = yaml.safe_load(f)

configs = {
    # A1: ECR Baseline (no feedback)
    "ablation_A1_no_feedback.yaml": {
        "work_dir": "./work_dir/ntu/cs/ablation_A1_no_feedback/",
        "model_args": {
            "feedback_mode": "disabled",
            "early_concept_loss_weight": 0.0,
            "feedback_consistency_weight": 0.0,
        }
    },
    # A2: ECR + Early Concept Head (Auxiliary Loss only, no feedback)
    "ablation_A2_early_head_only.yaml": {
        "work_dir": "./work_dir/ntu/cs/ablation_A2_early_head_only/",
        "model_args": {
            "feedback_mode": "early_head_only",
            "early_concept_loss_weight": 0.5,
            "feedback_consistency_weight": 0.02,
        }
    },
    # A3: ECR + Channel Modulation Only
    "ablation_A3_channel_mod.yaml": {
        "work_dir": "./work_dir/ntu/cs/ablation_A3_channel_mod/",
        "model_args": {
            "feedback_mode": "channel_modulation",
            "feedback_beta": 0.2,
            "feedback_lambda": 0.0,
            "early_concept_loss_weight": 0.5,
        }
    },
    # A4: Proposed Full Model (Token & Feature Soft Bias)
    "ablation_A4_soft_bias.yaml": {
        "work_dir": "./work_dir/ntu/cs/ablation_A4_soft_bias/",
        "model_args": {
            "feedback_mode": "token_soft_bias",
            "feedback_beta": 0.2,
            "feedback_lambda": 0.1,
            "feedback_confidence_gating": True,
            "early_concept_loss_weight": 0.5,
            "feedback_consistency_weight": 0.02,
        }
    },
    # A5: A4 without Confidence Gate
    "ablation_A5_no_gate.yaml": {
        "work_dir": "./work_dir/ntu/cs/ablation_A5_no_gate/",
        "model_args": {
            "feedback_mode": "no_confidence_gate",
            "feedback_beta": 0.2,
            "feedback_lambda": 0.1,
            "feedback_confidence_gating": False,
            "early_concept_loss_weight": 0.5,
            "feedback_consistency_weight": 0.02,
        }
    },
    # A6: A4 with Random Concept Projection (Testing Semantic Causality)
    "ablation_A6_random_proj.yaml": {
        "work_dir": "./work_dir/ntu/cs/ablation_A6_random_proj/",
        "model_args": {
            "feedback_mode": "random_concept_proj",
            "feedback_beta": 0.2,
            "feedback_lambda": 0.1,
            "early_concept_loss_weight": 0.5,
            "feedback_consistency_weight": 0.02,
        }
    },
    # A7: A4 without Symbolic KG Reasoner (Direct MLP action classification)
    "ablation_A7_no_kg.yaml": {
        "work_dir": "./work_dir/ntu/cs/ablation_A7_no_kg/",
        "model_args": {
            "feedback_mode": "token_soft_bias",
            "action_mode": "direct_mlp",
            "kg_loss_weight": 0.0,
            "prior_retention_weight": 0.0,
            "rule_sparsity_weight": 0.0,
        }
    },
    # Bone Modality
    "SFB_ECR_SkateFormer_b.yaml": {
        "work_dir": "./work_dir/ntu/cs/SFB_ECR_SkateFormer_b/",
        "train_feeder_args": {"data_type": "b"},
        "test_feeder_args": {"data_type": "b"},
    }
}

target_dir = Path(r"d:\Thesis\SFB-ECR-SkateFormer\config\train\ntu_cs")
for filename, override in configs.items():
    cfg = yaml.safe_load(open(base_yaml_path, 'r', encoding='utf-8'))
    for k, v in override.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    out_file = target_dir / filename
    with open(out_file, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Generated config: {out_file}")

# Also generate ntu_cv config
cv_dir = Path(r"d:\Thesis\SFB-ECR-SkateFormer\config\train\ntu_cv")
cv_dir.mkdir(parents=True, exist_ok=True)
cv_cfg = yaml.safe_load(open(base_yaml_path, 'r', encoding='utf-8'))
cv_cfg["work_dir"] = "./work_dir/ntu/cv/SFB_ECR_SkateFormer_j/"
cv_cfg["train_feeder_args"]["data_path"] = "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz"
cv_cfg["test_feeder_args"]["data_path"] = "/content/drive/MyDrive/KLTN/KLTN_Huy-Nhu/NTU60_CV_64f.npz"
with open(cv_dir / "SFB_ECR_SkateFormer_j.yaml", 'w', encoding='utf-8') as f:
    yaml.dump(cv_cfg, f, default_flow_style=False, sort_keys=False)
print("Generated NTU-CV config.")
