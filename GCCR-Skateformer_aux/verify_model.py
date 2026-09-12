"""
Verification script for Unified JointSpatialTemporalCBM with SkateFormer & AuxKGBranch
=====================================================================================
Validates:
  1. Model initialization & architecture:
     - 74 concepts, 9 anatomical/temporal groups.
     - AuxKGBranch (9 x KGUnionLayer with L_small=16 -> 288d -> num_class).
     - Factory build_joint_model.
  2. AuxKGBranch standalone output shape:
     - group_probs -> aux_logits shape (B, num_class).
  3. Inference forward pass & zero symbolic overhead:
     - Pure neural path: direct_action(feat) -> (B, num_class).
     - Aux branch NOT invoked during inference.
  4. Loss computation with AuxKGBranch regularizer:
     - Returns action, concept, kg, and aux_kg losses.
     - Verified removal of legacy competing logit fusion keys.
  5. Multi-task gradient flow:
     - Aux branch gradients flow back to concept group heads.
  6. Symbolic Explainability module (explain_joint):
     - 9 group explanations per action class.
  7. Freezing / Unfreezing backbone for 2-stage training.
"""

import sys
import os
from unittest.mock import patch
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_symbolic_model import (
    JointSpatialTemporalCBM,
    AuxKGBranch,
    KGUnionLayer,
    build_joint_model,
    freeze_backbone,
    unfreeze_backbone,
)


def verify_all():
    print("=" * 70)
    print("Verifying Unified Neuro-Symbolic SkateFormer CBM (AuxKGBranch)...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Using device: {device}")

    # 1. Model Instantiation & Architecture
    print("\n[2] Initializing JointSpatialTemporalCBM with AuxKGBranch...")
    model = JointSpatialTemporalCBM(
        backbone="skateformer",
        num_class=60,
        concept_loss_weight=0.5,
        concept_negative_weight=0.1,
        kg_s_weight=0.05,
        kg_c_weight=0.05,
        aux_kg_weight=0.05,
    ).to(device)

    print(f"    - Number of classes: {model.num_class}")
    print(f"    - Number of concepts: {model.num_concepts}")
    print(f"    - Number of concept groups: {len(model.group_order)}")
    print(f"    - Feature dimension: {model.feat_dim}")
    print(f"    - Aux KG weight: {model.aux_kg_weight}")
    print("    - Concept groups:", list(model.concept_groups.keys()))

    assert isinstance(model.aux_kg, AuxKGBranch), "model.aux_kg must be an AuxKGBranch instance!"
    assert len(model.aux_kg.layers) == 9, f"AuxKGBranch must contain 9 group layers, got {len(model.aux_kg.layers)}"
    assert model.aux_kg.aux_head.in_features == 288, f"AuxKGBranch aux_head must have 288 input features, got {model.aux_kg.aux_head.in_features}"
    print("    - AuxKGBranch initialized correctly with 9 KGUnionLayers (total_dim=288 -> num_class=60).")

    # Verify factory build_joint_model
    factory_model = build_joint_model(
        backbone_name="skateformer",
        num_class=60,
        aux_kg_weight=0.05,
        device=device,
    )
    assert isinstance(factory_model.aux_kg, AuxKGBranch), "Factory model must have AuxKGBranch!"
    print("    - build_joint_model factory constructed model successfully.")

    # 2. Test AuxKGBranch standalone forward pass
    print("\n[3] Testing AuxKGBranch standalone forward pass...")
    B = 2
    dummy_group_probs = [
        torch.rand(B, len(model.concept_groups[name]), device=device)
        for name in model.group_order
    ]
    aux_logits = model.aux_kg(dummy_group_probs)
    print(f"    - AuxKGBranch standalone output shape: {aux_logits.shape} (Expected: ({B}, 60))")
    assert aux_logits.shape == (B, 60), f"Expected aux_logits shape ({B}, 60), got {aux_logits.shape}"

    # 3. Dummy Input Forward Pass & Inference (Zero Symbolic Overhead)
    C_in, T, V, M = 3, 64, 24, 2
    x = torch.randn(B, C_in, T, V, M, device=device)
    labels = torch.tensor([0, 5], device=device)

    print(f"\n[4] Running Inference Forward pass with dummy input shape {x.shape}...")
    with patch.object(model.aux_kg, "forward", wraps=model.aux_kg.forward) as mock_aux:
        logits = model(x, return_loss=False)
        print(f"    - Inference output logits shape: {logits.shape} (Expected: ({B}, 60))")
        assert logits.shape == (B, 60), f"Expected shape ({B}, 60), got {logits.shape}"
        assert not mock_aux.called, "AuxKGBranch was called during inference! Must have zero overhead at inference."
        print("    - Confirmed: AuxKGBranch NOT invoked at inference (zero symbolic overhead).")

    # 4. Loss Computation & Backward Pass
    print("\n[5] Running Training Forward with Loss computation & Backward pass...")
    total_loss, loss_dict = model(x, label=labels, return_loss=True)
    print(f"    - Total Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"      * {k}: {v.item():.4f}")

    assert "action" in loss_dict, "loss_dict must contain 'action'"
    assert "concept" in loss_dict, "loss_dict must contain 'concept'"
    assert "kg" in loss_dict, "loss_dict must contain 'kg'"
    assert "aux_kg" in loss_dict, "loss_dict must contain 'aux_kg'"
    assert "symbolic" not in loss_dict, "Legacy 'symbolic' key must be removed"
    assert "neural_action" not in loss_dict, "Legacy 'neural_action' key must be removed"
    assert "fusion_alpha" not in loss_dict, "Legacy 'fusion_alpha' key must be removed"

    model.zero_grad()
    total_loss.backward()
    print("    - Backward pass successful! Checking gradients...")

    group_heads_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.group_heads.parameters())
    aux_head_grad = model.aux_kg.aux_head.weight.grad is not None and model.aux_kg.aux_head.weight.grad.norm() > 0
    direct_action_grad = model.direct_action.weight.grad is not None and model.direct_action.weight.grad.norm() > 0

    print(f"    - Direct action head has gradients: {direct_action_grad}")
    print(f"    - Aux KG head has gradients: {aux_head_grad}")
    print(f"    - Concept group heads have gradients: {group_heads_grad}")

    assert direct_action_grad, "direct_action head should receive gradients!"
    assert aux_head_grad, "AuxKGBranch head should receive gradients!"
    assert group_heads_grad, "Concept group heads should receive gradients!"

    # 5. Test Gradient Flow directly from AuxKGBranch into Concept Heads
    print("\n[6] Testing specific gradient flow from AuxKGBranch to Concept Heads...")
    model.zero_grad()
    feat = model.extract_feat(x)
    _, g_probs = model.encode_group_concepts(feat)
    aux_out = model.aux_kg(g_probs)
    aux_ce_loss = F.cross_entropy(aux_out, labels)
    aux_ce_loss.backward()

    aux_flowed = any(p.grad is not None and p.grad.norm() > 0 for p in model.group_heads.parameters())
    print(f"    - Gradient flows from AuxKGBranch directly to group_heads: {aux_flowed}")
    assert aux_flowed, "AuxKGBranch must pass multi-task gradients back to group_heads!"

    # 6. Symbolic Explainability
    print("\n[7] Testing Symbolic Explainability (explain_joint)...")
    explanation = model.explain_joint(x[0:1], action_idx=0)
    print(f"    - Generated {len(explanation)} group explanations for sample 0 (Action 0):")
    for exp in explanation:
        print(f"      * Group '{exp['group']}': Positive = {exp['positive']} (score={exp['positive_score']}) | Negative = {exp['negative']}")
    assert len(explanation) == 9, f"Expected 9 group explanations, got {len(explanation)}"

    # 7. Freezing & Unfreezing Backbone
    print("\n[8] Testing Freezing & Unfreezing Backbone...")
    frozen, _ = model.freeze_backbone()
    print(f"    - Frozen {frozen} backbone parameters.")
    unfrozen, _ = model.unfreeze_backbone()
    print(f"    - Unfrozen {unfrozen} backbone parameters.")

    print("\n" + "=" * 70)
    print("ALL AUX-KG VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    verify_all()
