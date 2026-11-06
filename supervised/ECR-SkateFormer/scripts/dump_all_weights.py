"""
Dump and Analyze All Concept Weights for ECR-SkateFormer
=======================================================
Xuất toàn bộ ma trận trọng số w (cả 72 concepts khẳng định và 72 phủ định)
cho tất cả 60 hành động, phân tích phân phối trọng số và in ra màn hình.
"""

import csv
import os
import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_rules import load_concepts_and_actions


def main():
    checkpoint_path = str(PROJECT_ROOT / "work_dir/ntu/cs/REASON_SkateFormer_j/best_model.pt")
    config_path = str(PROJECT_ROOT / "config/train/ntu_cs/REASON_SkateFormer_j.yaml")
    concepts_csv = str(PROJECT_ROOT / "concepts/ntu_rgbd_60_120.csv")
    output_dir = str(PROJECT_ROOT / "work_dir/rules")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading concepts from: {concepts_csv}")
    concept_names, action_names, prototypes = load_concepts_and_actions(concepts_csv, num_classes=60)
    num_concepts = len(concept_names)  # 72

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    and_weight = state_dict["reasoner.action_rule.and_weight"]  # [144, 60]
    or_weight = state_dict["reasoner.action_rule.or_weight"]    # [144, 60]

    w_and = torch.sigmoid(and_weight).numpy()
    w_or = torch.sigmoid(or_weight).numpy()

    literal_names = [c for c in concept_names] + [f"NOT_{c}" for c in concept_names]

    # ── 1. Xuất toàn bộ ma trận ra file CSV ──────────────────────────────────
    csv_and_path = os.path.join(output_dir, "matrix_all_and_weights.csv")
    with open(csv_and_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Literal"] + action_names)
        for i, lit in enumerate(literal_names):
            writer.writerow([lit] + [f"{w_and[i, a]:.5f}" for a in range(len(action_names))])
    print(f"-> Saved Full AND Matrix CSV: {csv_and_path}")

    csv_or_path = os.path.join(output_dir, "matrix_all_or_weights.csv")
    with open(csv_or_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Literal"] + action_names)
        for i, lit in enumerate(literal_names):
            writer.writerow([lit] + [f"{w_or[i, a]:.5f}" for a in range(len(action_names))])
    print(f"-> Saved Full OR Matrix CSV : {csv_or_path}")

    # ── 2. Thống kê phân bố toàn bộ ma trận ───────────────────────────────────
    print("\n" + "=" * 75)
    print("       THỐNG KÊ PHÂN BỐ TRỌNG SỐ TOÀN MÔ HÌNH (144 literals x 60 actions)")
    print("=" * 75)
    for name, mat in [("AND Weights", w_and), ("OR Weights", w_or)]:
        total = mat.size
        c_extreme_high = np.sum(mat >= 0.97)
        c_prior_range  = np.sum((mat >= 0.90) & (mat < 0.97))
        c_mid_range    = np.sum((mat >= 0.30) & (mat < 0.90))
        c_low_range    = np.sum((mat >= 0.01) & (mat < 0.30))
        c_dead_zero    = np.sum(mat < 0.01)

        print(f"[{name}] Tổng số liên kết: {total}")
        print(f"  * Cực lớn (w >= 0.97)           : {c_extreme_high:5d} ({c_extreme_high/total*100:5.2f}%)  <- Các luật cốt lõi")
        print(f"  * Vùng Prior KG (0.90 <= w < 0.97): {c_prior_range:5d} ({c_prior_range/total*100:5.2f}%)  <- Duy trì từ Ontology")
        print(f"  * Vùng trung gian (0.30 <= w < 0.90): {c_mid_range:5d} ({c_mid_range/total*100:5.2f}%)  <- Đang học/chuyển tiếp")
        print(f"  * Trọng số nhỏ (0.01 <= w < 0.30) : {c_low_range:5d} ({c_low_range/total*100:5.2f}%)")
        print(f"  * Triệt tiêu (w < 0.01)           : {c_dead_zero:5d} ({c_dead_zero/total*100:5.2f}%)  <- Đã bị loại bỏ")
        print()

    # ── 3. Xuất chi tiết toàn bộ 72 concepts cho 1 hành động mẫu ra text file ─
    txt_detail_path = os.path.join(output_dir, "action00_drink_water_all_72_weights.txt")
    with open(txt_detail_path, "w", encoding="utf-8") as f:
        f.write("CHI TIẾT TRỌNG SỐ TOÀN BỘ 72 CONCEPTS CHO ACTION [00]: DRINK_WATER\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'No.':<4} | {'Concept Name':<38} | {'AND (w)':<10} | {'OR (w)':<10} | {'Prior KG':<8}\n")
        f.write("-" * 85 + "\n")

        proto_0 = prototypes[0].numpy()
        # Sắp xếp theo AND weight giảm dần
        sorted_pos = np.argsort(w_and[:num_concepts, 0])[::-1]
        for rank, ci in enumerate(sorted_pos):
            c_name = concept_names[ci]
            and_val = w_and[ci, 0]
            or_val = w_or[ci, 0]
            has_prior = "YES (1)" if proto_0[ci] > 0.5 else "no (0)"
            f.write(f"{rank+1:<4} | {c_name:<38} | {and_val:<10.5f} | {or_val:<10.5f} | {has_prior:<8}\n")

        f.write("\n" + "=" * 85 + "\n")
        f.write("PHẦN PHỦ ĐỊNH - NEGATION (NOT Concepts):\n")
        f.write("-" * 85 + "\n")
        sorted_neg = np.argsort(w_and[num_concepts:, 0])[::-1]
        for rank, ci in enumerate(sorted_neg):
            c_name = f"NOT_{concept_names[ci]}"
            and_val = w_and[num_concepts + ci, 0]
            or_val = w_or[num_concepts + ci, 0]
            if and_val > 0.10 or or_val > 0.10:  # Chỉ in các phủ định có kích hoạt
                f.write(f"{rank+1:<4} | {c_name:<38} | {and_val:<10.5f} | {or_val:<10.5f}\n")

    print(f"-> Saved Action 00 full ranking to: {txt_detail_path}")


if __name__ == "__main__":
    main()
