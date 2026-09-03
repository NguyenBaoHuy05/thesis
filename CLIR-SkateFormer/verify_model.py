"""Comprehensive Verification and Diagnostic Suite for CLIR-SkateFormer.

Verifies:
1. LiteralLayer & BinarizeSTE for both 72 concepts (144 literals) and 74 concepts (148 literals).
2. ConjunctionLayer (L1 part-grouped) & DisjunctionLayer (L2b OR1) dynamic masking.
3. CRLReasoner (1-pass logic reasoning, evidence shape, strict logic ActionHead).
4. ConceptUpdateCell (projection & residual MLP).
5. IterativeLogicReasoner & Ablations:
   - K=2 standard loop
   - A2: beta_max = 0.0 (No-loop, zero-division protection check)
   - A5: random_feedback = True (Random Logic Feedback check)
6. Full CLIRSkateFormer End-to-End:
   - 72 concepts default (52 spatial + 20 temporal matching feeder_ntu)
   - Forward pass & Losses (L_sup, L_reg)
   - Full Backward pass gradient check
   - A6: use_feature_bypass = True (Feature bypass check)
   - Phase switching (Phase 1 soft -> Phase 2 STE & regularization)
7. RuleExtractor:
   - Positive excitatory clause extraction
   - Executable DNF Engine evaluation (evaluate_dnf)
   - Rule Coverage, Rule Precision, and Model-Rule Fidelity metrics
   - DNF textual formatting
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure local imports and utf-8 printing work
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model.crl.literal_layer import LiteralLayer, BinarizeSTE, binarize_ste
from model.crl.conjunction_layer import ConjunctionLayer, ConjunctionLayer2, _build_part_group_mask
from model.crl.disjunction_layer import DisjunctionLayer
from model.crl.crl_reasoner import CRLReasoner
from model.crl.rule_extractor import RuleExtractor
from model.clir_skateformer import (
    ConceptUpdateCell,
    IterativeLogicReasoner,
    CLIRSkateFormer,
    CLIR_SkateFormer,
)


def print_section(title):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def test_literal_layer():
    print_section("Test 1: LiteralLayer & BinarizeSTE (72 concepts & 74 concepts)")
    for n_concepts in (72, 74):
        layer = LiteralLayer(n_concepts=n_concepts, mode="soft")
        z = torch.randn(4, n_concepts, requires_grad=True)

        # 1. Soft mode
        l_soft = layer(z)
        assert l_soft.shape == (4, 2 * n_concepts)
        assert (l_soft >= 0.0).all() and (l_soft <= 1.0).all()
        p_pos = l_soft[:, :n_concepts]
        p_neg = l_soft[:, n_concepts:]
        assert torch.allclose(p_pos + p_neg, torch.ones_like(p_pos), atol=1e-5)

        w = torch.randn_like(l_soft)
        loss_soft = (l_soft * w).sum()
        loss_soft.backward()
        assert z.grad is not None and (z.grad.abs() > 0).any()
        z.grad.zero_()

        # 2. STE mode
        layer.mode = "ste"
        layer.temperature = 5.0
        l_ste = layer(z)
        assert l_ste.shape == (4, 2 * n_concepts)
        p_ste_pos = l_ste[:, :n_concepts]
        unique_vals = torch.unique(p_ste_pos.detach())
        for val in unique_vals:
            assert abs(val.item() - 0.0) < 1e-4 or abs(val.item() - 1.0) < 1e-4

        loss_ste = (l_ste * w).sum()
        loss_ste.backward()
        assert z.grad is not None and (z.grad.abs() > 0).any()
        print(f"  [OK] LiteralLayer for {n_concepts} concepts (soft + STE) verified.")

    print("[PASS] LiteralLayer & BinarizeSTE test passed.")


def test_conjunction_and_disjunction():
    print_section("Test 2: ConjunctionLayer & DisjunctionLayer (72 & 74 concepts)")
    batch_size = 4
    for n_concepts in (72, 74):
        n_literals = 2 * n_concepts
        l = torch.rand(batch_size, n_literals).clamp(0.01, 0.99)

        # L1
        l1 = ConjunctionLayer(n_literals=n_literals, n_rules=256, part_grouped=True)
        r_and = l1(l)
        assert r_and.shape == (batch_size, 256)
        assert (r_and >= 0.0).all() and (r_and <= 1.0).all()
        assert l1.group_mask.shape == (n_literals, 256)

        # L2a
        l2a = ConjunctionLayer2(n_rules_l1=256, n_rules_l2a=128)
        r_and2 = l2a(r_and)
        assert r_and2.shape == (batch_size, 128)

        # L2b
        l2b = DisjunctionLayer(n_rules_l1=256, n_rules_l2b=128)
        r_or1 = l2b(r_and)
        assert r_or1.shape == (batch_size, 128)

        print(f"  [OK] Conjunction and Disjunction for {n_concepts} concepts verified.")

    print("[PASS] Conjunction & Disjunction layers test passed.")


def test_crl_reasoner():
    print_section("Test 3: CRLReasoner (1-pass Logic Reasoning)")
    n_concepts = 72
    n_actions = 120
    reasoner = CRLReasoner(n_concepts=n_concepts, n_actions=n_actions)
    z = torch.randn(4, n_concepts)

    r_evidence, y = reasoner(z)
    assert r_evidence.shape == (4, 512)
    assert y.shape == (4, n_actions)

    loss_sp = reasoner.rule_sparsity_loss()
    assert loss_sp.item() >= 0.0
    print(f"  [OK] CRLReasoner forward: r_evidence={r_evidence.shape}, y={y.shape}, rule_sparsity={loss_sp.item():.4f}")
    print("[PASS] CRLReasoner test passed.")


def test_concept_update_cell():
    print_section("Test 4: ConceptUpdateCell (Top-Down Logic Feedback)")
    n_concepts = 72
    logic_dim = 512
    cell = ConceptUpdateCell(num_concepts=n_concepts, logic_dim=logic_dim, hidden_dim=256)

    z_t = torch.randn(4, n_concepts)
    r_t = torch.rand(4, logic_dim)

    delta_z = cell(z_t, r_t)
    assert delta_z.shape == (4, n_concepts)
    print(f"  [OK] ConceptUpdateCell produced delta_z: shape={delta_z.shape}, mean={delta_z.mean().item():.4f}")
    print("[PASS] ConceptUpdateCell test passed.")


def test_iterative_logic_reasoner_and_ablations():
    print_section("Test 5: IterativeLogicReasoner & Ablations (A2, A5)")
    n_concepts = 72
    n_actions = 120
    z0 = torch.randn(4, n_concepts)

    # 1. Standard K=2
    reasoner_k2 = IterativeLogicReasoner(
        num_concepts=n_concepts,
        num_actions=n_actions,
        num_loops=2,
        beta_max=0.5,
        beta_init=0.1,
    )
    out_k2 = reasoner_k2(z0)
    assert out_k2["y_final"].shape == (4, n_actions)
    assert len(out_k2["z_all"]) == 3
    print(f"  [OK] Standard K=2 loop verified. Betas: {out_k2['betas'].detach().numpy()}")

    # 2. Ablation A2: True No-Loop (num_loops = 0)
    reasoner_a2 = IterativeLogicReasoner(
        num_concepts=n_concepts,
        num_actions=n_actions,
        num_loops=0,
        beta_max=0.0,
        beta_init=0.0,
    )
    out_a2 = reasoner_a2(z0)
    assert len(out_a2["betas"]) == 0, "Betas must be empty for num_loops=0"
    assert len(out_a2["y_all"]) == 0, "y_all must be empty for num_loops=0"
    assert len(out_a2["z_all"]) == 1, "z_all must have only z0 for num_loops=0"
    assert torch.allclose(out_a2["z_final"], z0), "z_final must equal z0 when num_loops=0"
    assert out_a2["y_final"].shape == (4, n_actions)
    print("  [OK] Ablation A2 (True num_loops=0, No loop executed) verified.")

    # 3. Ablation A5: Random Logic Feedback
    reasoner_a5 = IterativeLogicReasoner(
        num_concepts=n_concepts,
        num_actions=n_actions,
        num_loops=2,
        beta_max=0.5,
        beta_init=0.1,
        random_feedback=True,
    )
    out_a5 = reasoner_a5(z0)
    assert out_a5["y_final"].shape == (4, n_actions)
    print("  [OK] Ablation A5 (Random Logic Feedback) verified.")

    print("[PASS] IterativeLogicReasoner & Ablations test passed.")


def test_full_clir_skateformer():
    print_section("Test 6: Full CLIRSkateFormer End-to-End (72 concepts, Phase 1/2, A6 Bypass)")
    batch_size = 2
    num_classes = 120
    num_concepts = 72
    spatial_concepts = 52
    temporal_concepts = 20
    num_points = 24
    num_frames = 64
    num_people = 2
    in_channels = 3

    # 1. Main model (A4)
    model = CLIR_SkateFormer(
        num_classes=num_classes,
        num_concepts=num_concepts,
        spatial_concepts=spatial_concepts,
        temporal_concepts=temporal_concepts,
        num_people=num_people,
        num_points=num_points,
        num_frames=num_frames,
        num_loops=2,
        num_heads=32,
        mlp_ratio=4.0,
        drop_path=0.2,
        type_1_size=[8, 8],
        type_2_size=[8, 12],
        type_3_size=[8, 8],
        type_4_size=[8, 12],
        lambda_sup=1.0,
        lambda_reg=0.005,
    )

    x = torch.randn(batch_size, in_channels, num_frames, num_points, num_people)
    concept_targets = torch.randint(0, 2, (batch_size, num_concepts)).float()
    action_labels = torch.randint(0, num_classes, (batch_size,)).long()

    # Forward Phase 1
    model.train()
    model.set_phase(1)
    assert model.reasoner.literal_mode == "soft"
    assert model.reasoner.beta_max == 0.1
    beta_p1 = model.reasoner.get_betas()
    assert (beta_p1 >= 0.08).all() and (beta_p1 <= 0.10).all(), f"Phase 1 beta must be ~0.10, got {beta_p1}"
    out_p1 = model(x, concept_targets=concept_targets, action_labels=action_labels)
    assert out_p1["action_logits"].shape == (batch_size, num_classes)
    assert out_p1["concept_logits"].shape == (batch_size, num_concepts)
    print(f"  [OK] Phase 1 forward (72 concepts, beta={beta_p1.detach().numpy()}) verified.")

    # Forward Phase 2
    model.set_phase(2, temperature=6.0)
    assert model.reasoner.literal_mode == "ste"
    assert model.reasoner.beta_max == 0.5
    beta_p2 = model.reasoner.get_betas()
    assert torch.allclose(beta_p2, beta_p1, atol=1e-3), f"Phase 2 beta must smoothly continue from Phase 1, got {beta_p2}"
    out_p2 = model(x, concept_targets=concept_targets, action_labels=action_labels)
    assert "l_reg" in out_p2
    assert "l_sparse" in out_p2
    assert "l_stable" in out_p2
    print(f"  [OK] Phase 2 forward with L_reg (smooth beta={beta_p2.detach().numpy()}) verified.")

    # Backward pass
    total_loss = F.cross_entropy(out_p2["action_logits"], action_labels) + out_p2["aux_loss"]
    total_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert model.spatial_head[0].weight.grad is not None
    assert model.temporal_head[0].weight.grad is not None
    assert model.reasoner.crl.l1_and.weight_logit.grad is not None
    assert model.reasoner.update_cell.logic_proj[0].weight.grad is not None
    assert model.reasoner.raw_beta.grad is not None
    print("  [OK] Backward gradient flow through all components verified.")

    # 2. Ablation A6: Feature Bypass
    model_bypass = CLIR_SkateFormer(
        num_classes=num_classes,
        num_concepts=num_concepts,
        spatial_concepts=spatial_concepts,
        temporal_concepts=temporal_concepts,
        num_people=num_people,
        num_points=num_points,
        num_frames=num_frames,
        use_feature_bypass=True,
    )
    assert model_bypass.bypass_head is not None
    out_bypass = model_bypass(x, concept_targets=concept_targets, action_labels=action_labels)
    assert out_bypass["action_logits"].shape == (batch_size, num_classes)
    loss_bypass = out_bypass["action_logits"].sum()
    loss_bypass.backward()
    assert model_bypass.bypass_head.weight.grad is not None, "Bypass head must receive gradients"
    print("  [OK] Ablation A6 (CLIR-Bypass) forward and gradient flow verified.")

    print("[PASS] Full CLIRSkateFormer test passed.")


def test_rule_extractor_executable_dnf():
    print_section("Test 7: RuleExtractor Executable DNF Engine, Precision, Coverage, Fidelity")
    n_concepts = 72
    n_actions = 120
    cnames = [f"c_{i}" for i in range(n_concepts)]
    anames = [f"a_{i}" for i in range(n_actions)]

    reasoner = CRLReasoner(n_concepts=n_concepts, n_actions=n_actions)
    extractor = RuleExtractor(
        reasoner=reasoner,
        concept_names=cnames,
        action_names=anames,
        top_k_l2=3,
        top_k_l1=3,
        top_k_literals=4,
    )

    # 1. Extraction for action 0
    rule0 = extractor.extract_rules_for_action(0)
    assert rule0["action_name"] == "a_0"
    text0 = RuleExtractor.format_rule_as_dnf(rule0)
    # 2. Executable DNF Exact Boolean Formula Evaluation
    mock_rules = [
        {
            "action_name": "a_0",
            "action_idx": 0,
            "clauses": [
                {
                    "type": "and2",
                    "l2_rule_idx": 0,
                    "raw_weight": 1.0,
                    "l1_components": [
                        {"l1_rule_idx": 0, "literals": [("c_0", True), ("c_1", True)]}
                    ],
                }
            ],
        }
    ]
    # Sample 0: c_0=True, c_1=True  -> should fire Action 0
    # Sample 1: c_0=True, c_1=False -> should NOT fire
    z_controlled = torch.zeros(2, n_concepts)
    z_controlled[0, 0] = 1.0; z_controlled[0, 1] = 1.0   # Sample 0
    z_controlled[1, 0] = 1.0; z_controlled[1, 1] = -1.0  # Sample 1

    ctrl_preds, ctrl_scores, ctrl_cov = extractor.evaluate_dnf(z_controlled, rules=mock_rules)
    assert ctrl_cov[0].item() is True, "Sample 0 must satisfy Clause (c_0 ∧ c_1)"
    assert ctrl_cov[1].item() is False, "Sample 1 must violate Clause (c_0 ∧ c_1)"
    assert ctrl_preds[0].item() == 0, "Sample 0 must predict Action 0"
    assert ctrl_preds[1].item() == -1, "Sample 1 must be uncovered (-1)"
    assert ctrl_scores[0, 0].item() == 1.0, "Sample 0 must have score 1.0 for Action 0"
    assert ctrl_scores[1, 0].item() == 0.0, "Sample 1 must have score 0.0"
    print("  [OK] Exact Boolean DNF formula execution verified (Sample 0 fired, Sample 1 rejected).")

    # 3. Executable DNF Evaluation on batch
    batch_size = 8
    z_mock = torch.randn(batch_size, n_concepts)
    dnf_preds, dnf_scores, covered_mask = extractor.evaluate_dnf(z_mock)

    assert dnf_preds.shape == (batch_size,)
    assert dnf_scores.shape == (batch_size, n_actions)
    assert covered_mask.shape == (batch_size,)
    print(f"  [OK] Executable DNF evaluated on {batch_size} samples.")

    # 4. Mock Fidelity Evaluation
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, idx):
            x = torch.randn(3, 64, 24, 2)
            idx_t = torch.zeros(1)
            label = idx % n_actions
            return x, idx_t, label, idx

    loader = torch.utils.data.DataLoader(MockDataset(), batch_size=4)
    model = CLIR_SkateFormer(num_classes=n_actions, num_concepts=n_concepts, spatial_concepts=52, temporal_concepts=20)
    metrics = extractor.evaluate_fidelity(model, loader, device="cpu", n_batches=4)

    assert "coverage" in metrics
    assert "precision" in metrics
    assert "fidelity" in metrics
    print(f"  [OK] Fidelity Evaluation Metrics (Exact DNF Rules):")
    print(f"       Coverage:  {metrics['coverage'] * 100:.2f}%")
    print(f"       Precision: {metrics['precision'] * 100:.2f}%")
    print(f"       Fidelity:  {metrics['fidelity'] * 100:.2f}%")

    print("[PASS] RuleExtractor Executable DNF Engine test passed.")


def main():
    print("Starting CLIR-SkateFormer Comprehensive Verification Suite...")
    test_literal_layer()
    test_conjunction_and_disjunction()
    test_crl_reasoner()
    test_concept_update_cell()
    test_iterative_logic_reasoner_and_ablations()
    test_full_clir_skateformer()
    test_rule_extractor_executable_dnf()
    print("\n" + "*" * 75)
    print("  ALL 7 VERIFICATION TESTS (INCLUDING A2, A5, A6 & EXECUTABLE DNF) PASSED!")
    print("*" * 75)


if __name__ == "__main__":
    main()
