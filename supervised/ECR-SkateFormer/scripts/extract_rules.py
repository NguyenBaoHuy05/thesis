"""
Rule Extraction Script for ECR-SkateFormer (REASON-SkateFormer)
==============================================================
Trích xuất luật logic tượng trưng (Symbolic Rules) từ mô hình ECR-SkateFormer đã huấn luyện:
  1. Trích xuất Conjunction (AND Rules) & Disjunction (OR Rules) cho từng hành động.
  2. Phân tách rõ Literal khẳng định (+concept) và phủ định (-NOT concept).
  3. So sánh với Prototype ban đầu (KG Prior) để chỉ ra các concept được củng cố (retained) hoặc học mới (learned).
  4. Xuất kết quả ra định dạng JSON (cho phân tích) và TXT / Markdown (dễ đọc, dùng cho KLTN).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_concepts_and_actions(csv_path: str, num_classes: int = 60) -> Tuple[List[str], List[str], torch.Tensor]:
    """Đọc file CSV tri thức concepts ontology, lọc ra 72 concepts chuẩn và danh sách actions."""
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if num_classes is not None and len(rows) > num_classes:
        rows = rows[:num_classes]

    action_names = [row["action_class"] for row in rows]
    all_concept_names = [k for k in rows[0].keys() if k != "action_class"]

    # REASON lọc bỏ index 50 (foot_tap) và 65 (temporal_sequence_seq_cascading) để còn đúng 72 concepts
    if len(all_concept_names) > 72:
        concept_names = [name for i, name in enumerate(all_concept_names) if i not in (50, 65)]
    else:
        concept_names = all_concept_names

    prototypes = []
    for row in rows:
        prototypes.append([float(row[c]) for c in concept_names])

    prototypes_tensor = torch.tensor(prototypes, dtype=torch.float32)
    return concept_names, action_names, prototypes_tensor


def extract_prototype_rules(
    and_weight: torch.Tensor,
    or_weight: torch.Tensor,
    concept_names: List[str],
    action_names: List[str],
    prototypes: torch.Tensor = None,
    tau_and: float = 0.40,
    tau_or: float = 0.40,
    top_k: int = 8,
) -> Dict:
    """Trích xuất luật khi action_mode == 'prototype'."""
    num_literals, num_actions = and_weight.shape
    num_concepts = len(concept_names)

    # Sigmoid bounded weights: [0, 1]
    w_and = torch.sigmoid(and_weight).cpu()
    w_or = torch.sigmoid(or_weight).cpu()

    rules_by_action = []

    for action_idx in range(num_actions):
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action_{action_idx}"

        # ── 1. AND Conjunction ──────────────────────────────
        and_scores = w_and[:, action_idx]  # Shape (2 * num_concepts,)
        # Lấy top-k hoặc các literal vượt ngưỡng tau_and
        sorted_and_indices = torch.argsort(and_scores, descending=True)
        and_literals = []
        for idx in sorted_and_indices:
            score = and_scores[idx].item()
            if len(and_literals) >= top_k and score < tau_and:
                break
            if score < 0.15:  # Bỏ qua các trọng số quá nhỏ
                break

            idx_val = idx.item()
            is_neg = (idx_val >= num_concepts)
            c_idx = idx_val - num_concepts if is_neg else idx_val
            c_name = concept_names[c_idx] if c_idx < len(concept_names) else f"concept_{c_idx}"

            and_literals.append({
                "concept": c_name,
                "is_negated": is_neg,
                "weight": round(score, 4),
                "text": f"NOT({c_name})" if is_neg else c_name
            })
            if len(and_literals) >= top_k:
                break

        # ── 2. OR Disjunction ───────────────────────────────
        or_scores = w_or[:, action_idx]
        sorted_or_indices = torch.argsort(or_scores, descending=True)
        or_literals = []
        for idx in sorted_or_indices:
            score = or_scores[idx].item()
            if len(or_literals) >= top_k and score < tau_or:
                break
            if score < 0.15:
                break

            idx_val = idx.item()
            is_neg = (idx_val >= num_concepts)
            c_idx = idx_val - num_concepts if is_neg else idx_val
            c_name = concept_names[c_idx] if c_idx < len(concept_names) else f"concept_{c_idx}"

            or_literals.append({
                "concept": c_name,
                "is_negated": is_neg,
                "weight": round(score, 4),
                "text": f"NOT({c_name})" if is_neg else c_name
            })
            if len(or_literals) >= top_k:
                break

        # ── 3. So sánh với Prototype Prior ─────────────────
        prior_positive = []
        if prototypes is not None and action_idx < len(prototypes):
            prior_positive = [
                concept_names[ci] for ci, val in enumerate(prototypes[action_idx]) if val > 0.5
            ]

        rules_by_action.append({
            "action_index": action_idx,
            "action_name": action_name,
            "prior_prototype_concepts": prior_positive,
            "and_rule": and_literals,
            "or_rule": or_literals,
        })

    return {"mode": "prototype", "total_actions": num_actions, "rules": rules_by_action}


def extract_linear_rules(
    reasoner,
    concept_names: List[str],
    action_names: List[str],
    top_k_l2: int = 5,
    top_k_l1: int = 5,
    top_k_literals: int = 6,
    tau_act: float = 0.05,
) -> Dict:
    """Trích xuất luật DNF 2 tầng khi action_mode == 'linear'."""
    w_first_and, w_first_or = reasoner.first.weights()
    w_sec_and, w_sec_or = reasoner.second.weights()
    w_act = reasoner.action.weight.data.cpu()  # [n_actions, 2 * l2_dim]

    num_concepts = len(concept_names)
    num_actions = len(action_names)
    l1_dim = w_first_and.shape[1]
    l2_dim = w_sec_and.shape[1]

    rules_by_action = []

    for act_idx in range(num_actions):
        act_name = action_names[act_idx]
        act_weights = w_act[act_idx]  # Shape [2 * l2_dim]

        # Lọc các L2 nodes có trọng số dương mạnh
        pos_l2_indices = torch.where(act_weights > tau_act)[0]
        if len(pos_l2_indices) == 0:
            pos_l2_indices = torch.argsort(act_weights, descending=True)[:top_k_l2]
        else:
            sorted_idx = torch.argsort(act_weights[pos_l2_indices], descending=True)
            pos_l2_indices = pos_l2_indices[sorted_idx][:top_k_l2]

        clauses = []
        for l2_idx in pos_l2_indices:
            l2_val = l2_idx.item()
            l2_weight = act_weights[l2_val].item()
            is_or_clause = (l2_val >= l2_dim)
            l2_node = l2_val - l2_dim if is_or_clause else l2_val

            # Nhánh L2: xem xét các L1 nodes kết nối vào
            l2_weights_in = (w_sec_or if is_or_clause else w_sec_and)[:, l2_node].cpu()
            top_l1 = torch.argsort(l2_weights_in, descending=True)[:top_k_l1]

            l1_subclauses = []
            for l1_idx in top_l1:
                l1_val = l1_idx.item()
                is_l1_or = (l1_val >= l1_dim)
                l1_node = l1_val - l1_dim if is_l1_or else l1_val

                # Lấy literals từ input concepts
                l1_weights_in = (w_first_or if is_l1_or else w_first_and)[:, l1_node].cpu()
                top_lit = torch.argsort(l1_weights_in, descending=True)[:top_k_literals]

                literals = []
                for lit_idx in top_lit:
                    lit_val = lit_idx.item()
                    is_neg = (lit_val >= num_concepts)
                    c_idx = lit_val - num_concepts if is_neg else lit_val
                    c_name = concept_names[c_idx] if c_idx < num_concepts else f"c_{c_idx}"
                    literals.append(f"NOT({c_name})" if is_neg else c_name)

                l1_subclauses.append({
                    "l1_type": "OR1" if is_l1_or else "AND1",
                    "literals": literals,
                })

            clauses.append({
                "l2_type": "L2_OR" if is_or_clause else "L2_AND",
                "weight": round(l2_weight, 4),
                "l1_subclauses": l1_subclauses,
            })

        rules_by_action.append({
            "action_index": act_idx,
            "action_name": act_name,
            "clauses": clauses,
        })

    return {"mode": "linear", "total_actions": num_actions, "rules": rules_by_action}


def format_human_readable(rule_data: Dict) -> str:
    """Tạo văn bản Markdown / TXT dễ đọc cho từng action rule."""
    lines = [
        "=" * 80,
        "          ECR-SKATEFORMER: EXTRACTED SYMBOLIC LOGICAL RULES",
        "=" * 80,
        f"Model Logic Mode: {rule_data.get('mode', 'prototype').upper()}",
        f"Total Actions   : {rule_data.get('total_actions', 60)}",
        "=" * 80,
        "",
    ]

    mode = rule_data.get("mode", "prototype")
    for r in rule_data.get("rules", []):
        act_idx = r["action_index"]
        act_name = r["action_name"]

        lines.append(f"Action [{act_idx:02d}]: {act_name.upper()}")
        lines.append("-" * 60)

        if mode == "prototype":
            prior = r.get("prior_prototype_concepts", [])
            lines.append(f"  [Prior Concepts] : {', '.join(prior) if prior else '(none)'}")

            and_lits = [f"{item['text']} (w={item['weight']})" for item in r.get("and_rule", [])]
            lines.append(f"  [AND Conjunction]: {' ∧ '.join(and_lits) if and_lits else '(none)'}")

            or_lits = [f"{item['text']} (w={item['weight']})" for item in r.get("or_rule", [])]
            lines.append(f"  [OR Disjunction] : {' ∨ '.join(or_lits) if or_lits else '(none)'}")

            # Logical rule summary: Lấy các positive literals hàng đầu VÀ các NOT literals nếu có
            top_pos = [item["text"] for item in r.get("and_rule", []) if not item["is_negated"]][:3]
            top_neg = [item["text"] for item in r.get("and_rule", []) if item["is_negated"]][:2]
            and_expr = " ∧ ".join(top_pos + top_neg)

            top_or_pos = [item["text"] for item in r.get("or_rule", []) if not item["is_negated"]][:3]
            top_or_neg = [item["text"] for item in r.get("or_rule", []) if item["is_negated"]][:1]
            or_expr = " ∨ ".join(top_or_pos + top_or_neg)

            lines.append(f"  -> Formal Rule  : IF ({and_expr}) OR ANY_OF({or_expr}) THEN {act_name}")

        else:
            clauses = r.get("clauses", [])
            for ci, clause in enumerate(clauses):
                lines.append(f"  Clause {ci+1} [{clause['l2_type']} w={clause['weight']}]:")
                for sub in clause.get("l1_subclauses", []):
                    lines.append(f"    - ({' ∧ '.join(sub['literals'][:4])})")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract Symbolic Rules from ECR-SkateFormer Checkpoint")
    parser.add_argument("--checkpoint", default="work_dir/ntu/cs/REASON_SkateFormer_j/best_model.pt",
                        help="Path to best_model.pt or checkpoint_best.pt")
    parser.add_argument("--config", default="config/train/ntu_cs/REASON_SkateFormer_j.yaml",
                        help="Path to config YAML")
    parser.add_argument("--concepts-csv", default="concepts/ntu_rgbd_60_120.csv",
                        help="Path to concepts CSV")
    parser.add_argument("--output-dir", default="work_dir/rules",
                        help="Directory to save extracted rules")
    parser.add_argument("--tau-and", type=float, default=0.35, help="Threshold for AND rule literals")
    parser.add_argument("--tau-or", type=float, default=0.35, help="Threshold for OR rule literals")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k literals per rule")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load config
    cfg = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    model_args = cfg.get("model_args", {})
    num_classes = model_args.get("num_classes", 60)

    # 2. Load concepts and actions
    csv_path = args.concepts_csv
    if not os.path.isabs(csv_path):
        csv_path = str(PROJECT_ROOT / csv_path)

    print(f"Loading concepts ontology from: {csv_path} ...")
    concept_names, action_names, prototypes = load_concepts_and_actions(csv_path, num_classes=num_classes)
    print(f"  -> Concepts: {len(concept_names)} | Action Classes: {len(action_names)}")

    # 3. Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = str(PROJECT_ROOT / ckpt_path)

    print(f"Loading model checkpoint from: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 4. Determine reasoner mode & extract rules
    if "reasoner.action_rule.and_weight" in state_dict:
        print("Detected Reasoner Mode: PROTOTYPE (action_rule KGUnionLayer)")
        and_w = state_dict["reasoner.action_rule.and_weight"]
        or_w = state_dict["reasoner.action_rule.or_weight"]

        rule_data = extract_prototype_rules(
            and_weight=and_w,
            or_weight=or_w,
            concept_names=concept_names,
            action_names=action_names,
            prototypes=prototypes,
            tau_and=args.tau_and,
            tau_or=args.tau_or,
            top_k=args.top_k,
        )

    elif "reasoner.first.and_weight" in state_dict:
        print("Detected Reasoner Mode: LINEAR (Hierarchical 2-layer CRL)")
        from model.reason_skateformer import ReasonSkateFormer
        model = ReasonSkateFormer(
            num_classes=num_classes,
            num_concepts=len(concept_names),
            concept_names=concept_names,
            **{k: v for k, v in model_args.items() if k not in ["num_classes", "num_concepts", "concept_names"]}
        )
        model.load_state_dict(state_dict, strict=False)
        rule_data = extract_linear_rules(
            reasoner=model.reasoner,
            concept_names=concept_names,
            action_names=action_names,
        )
    else:
        raise KeyError("Could not find recognizable reasoner weights ('action_rule' or 'first') in state_dict!")

    # 5. Save outputs
    json_path = os.path.join(args.output_dir, "ecr_rules.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rule_data, f, ensure_ascii=False, indent=2)
    print(f"  [JSON Saved] -> {json_path}")

    txt_path = os.path.join(args.output_dir, "ecr_rules_readable.txt")
    readable_text = format_human_readable(rule_data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(readable_text)
    print(f"  [TXT Saved]  -> {txt_path}")

    print("\n" + "=" * 80)
    print("  TRÍCH XUẤT LUẬT THÀNH CÔNG! DƯỚI ĐÂY LÀ VÍ DỤ 3 ACTION ĐẦU TIÊN:")
    print("=" * 80)
    sample_preview = "\n".join(readable_text.split("\n")[:38])
    print(sample_preview)
    print("=" * 80)


if __name__ == "__main__":
    main()
