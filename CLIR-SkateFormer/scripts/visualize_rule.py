"""Render one extracted DNF clause as a concept -> AND -> action diagram."""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", required=True, help="Path to dnf_rules.json")
    parser.add_argument("--action", type=int, default=0)
    parser.add_argument("--clause", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    rule = rules[args.action]
    clause = rule["clauses"][args.clause]
    literals = clause["l1_components"][0]["literals"]
    rule_idx = clause.get("l1_rule_idx", clause.get("l2_rule_idx", "?"))
    layer_name = clause["type"].upper()

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        6.5, 7.55,
        f"DNF rule visualization — {rule['action_name']} | Clause {args.clause + 1}",
        ha="center", va="center", fontsize=17, fontweight="bold",
    )
    ax.text(
        6.5, 7.08,
        f"{layer_name} rule index: {rule_idx}    |    raw weight: {clause['raw_weight']:.4f}",
        ha="center", va="center", fontsize=11, color="#555555",
    )

    if clause["type"].lower() == "or1":
        ax.set_xlim(0, 16)
        ax.text(6.5, 6.55, "Alternative patterns: at least one AND branch must hold",
                ha="center", va="center", fontsize=11, color="#555555")
        branch_ys = [5.2, 3.75, 2.3]
        for branch_idx, (component, y) in enumerate(zip(clause["l1_components"], branch_ys), 1):
            literals_text = " ∧ ".join(
                name if positive else f"NOT {name}"
                for name, positive in component["literals"]
            )
            branch_box = FancyBboxPatch(
                (0.35, y - 0.48), 5.0, 0.96,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.8, edgecolor="#7b61a8", facecolor="#eee8fa",
            )
            ax.add_patch(branch_box)
            ax.text(2.85, y, f"Branch {branch_idx}:\n{textwrap.fill(literals_text, width=48)}",
                    ha="center", va="center", fontsize=9.2)
            ax.add_patch(FancyArrowPatch(
                (5.35, y), (6.35, 3.75), arrowstyle="->",
                mutation_scale=13, linewidth=1.2, color="#777777",
                connectionstyle="arc3,rad=0.0",
            ))

        or_box = FancyBboxPatch(
            (6.35, 3.05), 1.8, 1.4,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            linewidth=2.2, edgecolor="#9b59b6", facecolor="#f3e5f5",
        )
        ax.add_patch(or_box)
        ax.text(7.25, 3.75, "OR\n(any branch\ncan hold)",
                ha="center", va="center", fontsize=12, fontweight="bold")
        ax.add_patch(FancyArrowPatch(
            (8.15, 3.75), (10.0, 3.75), arrowstyle="->",
            mutation_scale=15, linewidth=1.8, color="#2c3e50",
        ))
        action_box = FancyBboxPatch(
            (10.0, 3.0), 5.0, 1.5,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            linewidth=2.4, edgecolor="#1f5f99", facecolor="#dceeff",
        )
        ax.add_patch(action_box)
        ax.text(12.5, 3.75, f"ACTION\n{rule['action_name']}",
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="#174a78")
        ax.text(2.85, 0.55, "Each branch is an AND rule; purple = alternative OR branches",
                ha="center", va="center", fontsize=10, color="#555555")
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved visualization to: {args.output}")
        return

    ys = [5.8, 4.45, 3.1, 1.75]
    for (name, positive), y in zip(literals, ys):
        label = name if positive else f"NOT {name}"
        face = "#d9f2df" if positive else "#fde0df"
        edge = "#2e8b57" if positive else "#c0392b"
        box = FancyBboxPatch(
            (0.45, y - 0.42), 3.65, 0.84,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.8, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(box)
        ax.text(
            2.275, y, textwrap.fill(label, width=27),
            ha="center", va="center", fontsize=10,
        )
        ax.add_patch(FancyArrowPatch(
            (4.1, y), (5.65, 3.75), arrowstyle="->",
            mutation_scale=13, linewidth=1.2, color="#777777",
            connectionstyle="arc3,rad=0.0",
        ))

    and_box = FancyBboxPatch(
        (5.65, 3.05), 2.1, 1.4,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=2.2, edgecolor="#2c3e50", facecolor="#e8eef7",
    )
    ax.add_patch(and_box)
    ax.text(6.7, 3.75, "AND\n(all literals\nmust hold)",
            ha="center", va="center", fontsize=12, fontweight="bold")

    ax.add_patch(FancyArrowPatch(
        (7.75, 3.75), (9.05, 3.75), arrowstyle="->",
        mutation_scale=15, linewidth=1.8, color="#2c3e50",
    ))
    action_box = FancyBboxPatch(
        (9.05, 3.0), 3.35, 1.5,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=2.4, edgecolor="#1f5f99", facecolor="#dceeff",
    )
    ax.add_patch(action_box)
    ax.text(10.725, 3.75, f"ACTION\n{rule['action_name']}",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#174a78")

    ax.text(2.25, 0.55, "green = positive literal    |    red = negated literal",
            ha="center", va="center", fontsize=10, color="#555555")
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
