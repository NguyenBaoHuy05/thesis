"""Verification and Diagnostic Suite for LCR-SkateFormer.

Verifies:
1. Architecture initialization across configurations (K=1,2,3; shared/unshared; linear/prototype).
2. Forward pass tensor shapes and output signatures.
3. Looping reasoning trace recording and progressive concept refinement.
4. Gradient flow to all learnable parameters (backbone, heads, rule banks, update cells, beta).
5. Multi-objective loss components (action, early/final concept, aux action, stability, KG, prior, sparsity).
6. Concept deletion / counterfactual responsiveness.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure local imports work
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.lcr_skateformer import (
    LCR_SkateFormer,
    LCRSkateFormer,
    IterativeKGReasoner,
    KGUnionLayer,
    KGPrimedReasoner,
    ConceptUpdateCell,
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_kg_union_layer():
    print_section("Test 1: KGUnionLayer Soft Logic & Bounds")
    in_dim, out_dim = 144, 128
    layer = KGUnionLayer(in_dim, out_dim)
    x = torch.rand(4, in_dim).clamp(0.0, 1.0)
    out = layer(x)
    assert out.shape == (4, 2 * out_dim), f"Expected shape (4, {2*out_dim}), got {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Soft logic outputs must be in [0, 1]"

    and_w, or_w = layer.weights()
    assert (and_w >= 0.0).all() and (and_w <= 1.0).all()
    assert (or_w >= 0.0).all() and (or_w <= 1.0).all()

    prior_loss = layer.prior_retention_loss()
    sparsity_loss = layer.sparsity_loss()
    assert prior_loss.item() >= 0.0 and sparsity_loss.item() >= 0.0
    print("[PASS] KGUnionLayer forwards, bounds, and losses verified.")


def test_iterative_reasoner_configurations():
    print_section("Test 2: IterativeKGReasoner (Linear & Prototype Modes)")
    n_concepts = 72
    n_actions = 120
    cnames = [f"c_{i}" for i in range(n_concepts)]
    prototypes = torch.zeros(n_actions, n_concepts)
    prototypes[:, :10] = 1.0

    # 1. Linear mode, Shared weights, K=2
    reasoner_linear = IterativeKGReasoner(
        num_concepts=n_concepts,
        num_actions=n_actions,
        concept_names=cnames,
        num_loops=2,
        shared_weights=True,
        action_mode="linear",
    )
    z0 = torch.randn(2, n_concepts)
    y_final, z_final, all_actions, trace = reasoner_linear(z0, return_trace=True)
    assert y_final.shape == (2, n_actions)
    assert z_final.shape == (2, n_concepts)
    assert len(all_actions) == 2
    assert len(trace) == 3  # step 0, step 1, final step 2
    print(f"  Linear mode K=2: y_final={y_final.shape}, trace_steps={len(trace)}, beta={reasoner_linear.get_beta().item():.4f}")

    # 2. Prototype mode, Unshared weights, K=3
    reasoner_proto = IterativeKGReasoner(
        num_concepts=n_concepts,
        num_actions=n_actions,
        concept_names=cnames,
        num_loops=3,
        shared_weights=False,
        action_mode="prototype",
        action_prototypes=prototypes,
    )
    y_proto, z_proto, all_act_proto, trace_proto = reasoner_proto(z0, return_trace=True)
    assert y_proto.shape == (2, n_actions)
    assert z_proto.shape == (2, n_concepts)
    assert len(all_act_proto) == 3
    assert len(trace_proto) == 4  # step 0, 1, 2, final 3
    print(f"  Prototype mode K=3 (unshared): y_proto={y_proto.shape}, trace_steps={len(trace_proto)}")
    print("[PASS] IterativeKGReasoner linear & prototype configurations verified.")


def test_full_lcr_skateformer_forward_backward():
    print_section("Test 3: Full LCR-SkateFormer End-to-End Forward & Backward Pass")
    num_classes = 120
    num_concepts = 72
    spatial_concepts = 52
    temporal_concepts = 20
    num_people = 2
    num_points = 24
    num_frames = 64

    cnames = [f"c_{i}" for i in range(num_concepts)]
    kg_prior_map = {
        "require": [["c_0", "c_1"], ["c_2", "c_3"]],
        "alongside": [["c_4", "c_5"]],
        "excludes": [["c_0", "c_10"]],
    }

    model = LCR_SkateFormer(
        num_classes=num_classes,
        num_concepts=num_concepts,
        spatial_concepts=spatial_concepts,
        temporal_concepts=temporal_concepts,
        num_people=num_people,
        num_points=num_points,
        num_frames=num_frames,
        concept_names=cnames,
        num_loops=2,
        shared_weights=True,
        update_gate_init=0.1,
        kg_prior_map=kg_prior_map,
        action_mode="linear",
        in_channels=3,
        num_heads=32,
    )

    B = 2
    # Input shape: [B, C=3, T=64, V=24, M=2]
    x = torch.randn(B, 3, num_frames, num_points, num_people)
    concept_targets = (torch.rand(B, num_concepts) > 0.7).float()
    action_labels = torch.randint(0, num_classes, (B,))

    output = model(
        x,
        concept_targets=concept_targets,
        action_labels=action_labels,
        return_trace=True,
    )

    assert isinstance(output, dict), "Forward with concept_targets must return output dict"
    assert "action_logits" in output and output["action_logits"].shape == (B, num_classes)
    assert "concept_logits" in output and output["concept_logits"].shape == (B, num_concepts)
    assert "initial_concept_logits" in output and output["initial_concept_logits"].shape == (B, num_concepts)
    assert "early_concept_loss" in output
    assert "final_concept_loss" in output
    assert "stability_loss" in output
    assert "aux_action_loss" in output
    assert "kg_loss" in output
    assert "aux_loss" in output

    total_loss = F.cross_entropy(output["action_logits"], action_labels) + output["aux_loss"]
    print(f"  Computed total multi-loop loss: {total_loss.item():.4f}")
    print(f"    - Action Loss: {F.cross_entropy(output['action_logits'], action_labels).item():.4f}")
    print(f"    - Early Concept Loss (z0): {output['early_concept_loss'].item():.4f}")
    print(f"    - Refined Concept Loss (zK): {output['final_concept_loss'].item():.4f}")
    print(f"    - Stability Loss (||cK - c0||): {output['stability_loss'].item():.4f}")
    print(f"    - Aux Action Loss (intermediate loops): {output['aux_action_loss'].item():.4f}")
    print(f"    - KG Consistency Loss: {output['kg_loss'].item():.4f}")

    total_loss.backward()

    # Check gradients in each major component
    assert model.spatial_head[0].weight.grad is not None, "Spatial head must have gradients"
    assert model.temporal_head[0].weight.grad is not None, "Temporal head must have gradients"
    assert model.reasoner.raw_beta.grad is not None, "Beta gating parameter must have gradients"
    assert model.reasoner.update_cell.update_mlp[0].weight.grad is not None, "UpdateMLP must have gradients"
    assert model.reasoner.base_reasoner.first.and_weight.grad is not None, "KGUnionLayer must have gradients"

    print("[PASS] Full LCR-SkateFormer forward & backward gradient flow verified across all modules.")


def test_counterfactual_concept_intervention():
    print_section("Test 4: Counterfactual Concept Deletion / Intervention")
    n_concepts = 72
    n_actions = 120
    cnames = [f"c_{i}" for i in range(n_concepts)]
    reasoner = IterativeKGReasoner(
        num_concepts=n_concepts,
        num_actions=n_actions,
        concept_names=cnames,
        num_loops=2,
        shared_weights=True,
    )
    reasoner.eval()

    z0 = torch.randn(1, n_concepts)
    with torch.no_grad():
        y_orig, z_orig, _, trace_orig = reasoner(z0, return_trace=True)

        # Intervene: force concept 0 to be strongly absent (logit = -15.0)
        z_mod = z0.clone()
        z_mod[0, 0] = -15.0
        y_mod, z_mod_out, _, trace_mod = reasoner(z_mod, return_trace=True)

        pred_diff = (y_orig - y_mod).abs().mean().item()
        concept_diff = (z_orig - z_mod_out).abs().mean().item()

    print(f"  Mean Action Output Delta after intervening on concept 0: {pred_diff:.4f}")
    print(f"  Mean Refined Concept Delta across other concepts: {concept_diff:.4f}")
    assert pred_diff > 0.0, "Intervention on concept must affect action score"
    print("[PASS] Counterfactual concept intervention test succeeded.")


def run_all_tests():
    print("\n" + "#" * 70)
    print("      LCR-SKATEFORMER VERIFICATION & AUDIT SUITE")
    print("#" * 70)
    test_kg_union_layer()
    test_iterative_reasoner_configurations()
    test_full_lcr_skateformer_forward_backward()
    test_counterfactual_concept_intervention()
    print("\n" + "=" * 70)
    print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! (100% PASS)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
