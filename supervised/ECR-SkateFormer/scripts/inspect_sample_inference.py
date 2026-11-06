"""
Inspect Sample Inference & Concept Activations for ECR-SkateFormer
==================================================================
Load 2-3 mẫu dữ liệu kiểm thử thực tế từ dataset NTU-60:
1. Dự đoán hành động và đối chiếu với nhãn thực tế (Ground Truth vs Prediction).
2. Trích xuất xác suất các concept được kích hoạt thực tế trên mẫu video đó: p(concept | skeleton).
3. Đối chiếu với trọng số luật w của hành động để xem các concept nào thực sự dẫn dắt quyết định.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feeders.feeder_ntu import Feeder
from model.reason_skateformer import ReasonSkateFormer


def main():
    checkpoint_path = str(PROJECT_ROOT / "work_dir/ntu/cs/REASON_SkateFormer_j/best_model.pt")
    concepts_csv = str(PROJECT_ROOT / "concepts/ntu_rgbd_60_120.csv")

    # Ưu tiên load sampled data nếu có để chạy tức thì trong 1 giây
    sample_data_path = str(PROJECT_ROOT.parent / "data/ntu/NTU60_CS_sampled64.npz")
    if not os.path.isfile(sample_data_path):
        sample_data_path = str(PROJECT_ROOT.parent / "data/ntu/NTU60_CS.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    print(f"Data source  : {sample_data_path}")

    # 1. Khởi tạo Feeder
    print("Loading test dataset ...")
    test_ds = Feeder(
        data_path=sample_data_path,
        split="test",
        window_size=64,
        p_interval=[0.95],
        thres=64,
        uniform=True,
        partition=True,
        concepts_csv=concepts_csv,
        remove_null_concept=True,
        concept_num_classes=60,
    )
    concept_names = test_ds.concept_names
    action_prototypes = test_ds.concept_matrix
    print(f"Test dataset size: {len(test_ds)} samples")

    # 2. Khởi tạo Model từ YAML config & Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    import yaml
    config_path = str(PROJECT_ROOT / "config/train/ntu_cs/REASON_SkateFormer_j.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_args = cfg.get("model_args", {})
    model_args["action_prototypes"] = action_prototypes
    model_args["concept_names"] = concept_names
    if "backbone_weights" in model_args:
        del model_args["backbone_weights"]

    model = ReasonSkateFormer(**model_args).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Lấy trọng số rule của model
    w_and = torch.sigmoid(model.reasoner.action_rule.and_weight).detach().cpu().numpy()  # [144, 60]
    w_or  = torch.sigmoid(model.reasoner.action_rule.or_weight).detach().cpu().numpy()   # [144, 60]

    # Danh sách tên 60 action
    action_names = []
    with open(concepts_csv, "r", encoding="utf-8-sig") as f:
        import csv
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if len(action_names) < 60 and r:
                action_names.append(r[0].strip())

    # 3. Chọn 3 mẫu thuộc các hành động khác nhau để kiểm tra
    target_classes = [0, 9, 8]  # [0: drink_water, 9: clapping, 8: stand_up]
    selected_samples = []

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        data, index_t, label, index = item[0], item[1], item[2], item[3]
        if label in target_classes and label not in [s[2] for s in selected_samples]:
            selected_samples.append((data, index_t, label, index, idx))
            if len(selected_samples) == len(target_classes):
                break

    print("\n" + "=" * 90)
    print("              KẾT QUẢ SUY DIỄN & KÍCH HOẠT CONCEPT TRÊN 3 MẪU THỰC TẾ")
    print("=" * 90)

    for s_idx, (data_np, index_t_np, true_label, sample_id, dataset_idx) in enumerate(selected_samples):
        act_true_name = action_names[true_label].upper()

        data_t = torch.from_numpy(data_np).unsqueeze(0).float().to(device)
        index_t_t = torch.from_numpy(index_t_np).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(data_t, index_t_t)
            action_logits = output["action_logits"]
            concept_logits = output["concept_logits"]
            concept_probs = torch.sigmoid(concept_logits)[0].cpu().numpy()  # [72]

            pred_label = action_logits.argmax(dim=-1).item()
            act_pred_name = action_names[pred_label].upper()
            pred_prob = torch.softmax(action_logits, dim=-1)[0, pred_label].item()

        print(f"\n>> MẪU #{s_idx+1} [Dataset Index: {dataset_idx} | Sample ID: {sample_id}]")
        print(f"   * Nhãn thực tế (Ground Truth) : [{true_label:02d}] {act_true_name}")
        print(f"   * Mô hình dự đoán (Prediction): [{pred_label:02d}] {act_pred_name} (Độ tin cậy: {pred_prob*100:.2f}%)")
        status = "CHÍNH XÁC (CORRECT)" if pred_label == true_label else "SAI (WRONG)"
        print(f"   * Đánh giá                    : {status}")
        print("-" * 90)

        # Phân tách 52 Spatial Concepts và 20 Temporal Concepts
        spatial_probs = concept_probs[:52]
        temporal_probs = concept_probs[52:]

        print(f"   [A] TOP SPATIAL CONCEPTS (Khớp nối, tư thế giải phẫu cơ thể - 52 concepts):")
        print(f"   {'No.':<4} | {'Tên Concept':<35} | {'Xác suất p(C)':<15} | {'Trọng số w(AND)':<16} | {'Đánh giá vai trò'}")
        print("   " + "-" * 88)
        sorted_sp = np.argsort(spatial_probs)[::-1]
        for rank, ci in enumerate(sorted_sp[:5]):
            prob = spatial_probs[ci]
            c_name = concept_names[ci]
            w_val = w_and[ci, pred_label]
            role = "🔥 QUYẾT ĐỊNH" if (prob >= 0.5 and w_val >= 0.90) else ("Bối cảnh phụ" if prob >= 0.5 else "Ít kích hoạt")
            print(f"   {rank+1:<4} | {c_name:<35} | {prob*100:>6.2f}%         | {w_val:>8.4f}         | {role}")

        print(f"\n   [B] TOP TEMPORAL CONCEPTS (Động học, hướng & nhịp điệu thời gian - 20 concepts):")
        print(f"   {'No.':<4} | {'Tên Concept':<35} | {'Xác suất p(C)':<15} | {'Trọng số w(AND)':<16} | {'Đánh giá vai trò'}")
        print("   " + "-" * 88)
        sorted_tp = np.argsort(temporal_probs)[::-1]
        for rank, ci in enumerate(sorted_tp[:4]):
            global_ci = 52 + ci
            prob = temporal_probs[ci]
            c_name = concept_names[global_ci]
            w_val = w_and[global_ci, pred_label]
            role = "🔥 QUYẾT ĐỊNH" if (prob >= 0.5 and w_val >= 0.90) else ("Bối cảnh phụ" if prob >= 0.5 else "Ít kích hoạt")
            print(f"   {rank+1:<4} | {c_name:<35} | {prob*100:>6.2f}%         | {w_val:>8.4f}         | {role}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
