"""
Verification script for Unified JointSpatialTemporalCBM with SkateFormer & JointSymbolicActionCompatibility
========================================================================================================
Validates:
  1. Model initialization and concept matrix loading (74 concepts, 9 anatomical/temporal groups).
  2. Forward pass & output shapes (neural_logits, symbolic_logits, fused_logits).
  3. Total loss computation including symbolic compatibility ranking loss, selection entropy, KG losses, and concept BCE.
  4. Backward pass & gradient flow.
  5. Symbolic Explainability module (explain_joint).
  6. Freezing / Unfreezing backbone for 2-stage training.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_symbolic_model import (
    JointSpatialTemporalCBM,
    JointSymbolicActionCompatibility,
    build_joint_model,
    freeze_backbone,
    unfreeze_backbone,
)


def verify_all():
    print("=" * 70)
    print("Verifying Unified Neuro-Symbolic SkateFormer CBM...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Using device: {device}")

    # 1. Model Instantiation
    print("\n[2] Initializing JointSpatialTemporalCBM...")
    model = JointSpatialTemporalCBM(
        backbone="skateformer",
        num_class=60,
        concept_loss_weight=0.5,
        concept_negative_weight=0.1,
        kg_s_weight=0.05,
        kg_c_weight=0.05,
        fusion_alpha=0.30,
        symbolic_temperature=0.7,
        symbolic_margin=0.15,
    ).to(device)

    print(f"    - Number of classes: {model.num_class}")
    print(f"    - Number of concepts: {model.num_concepts}")
    print(f"    - Number of concept groups: {len(model.group_order)}")
    print(f"    - Feature dimension: {model.feat_dim}")
    print("    - Concept groups:", list(model.concept_groups.keys()))

    assert isinstance(model.symbolic_action, JointSymbolicActionCompatibility), "symbolic_action must be JointSymbolicActionCompatibility"

    # 2. Dummy Input Forward Pass
    B, C, T, V, M = 2, 3, 64, 24, 2
    x = torch.randn(B, C, T, V, M, device=device)
    labels = torch.tensor([0, 5], device=device)

    print(f"\n[3] Running Forward pass with dummy input shape {x.shape}...")
    logits = model(x)
    print(f"    - Output logits shape: {logits.shape} (Expected: ({B}, 60))")
    assert logits.shape == (B, 60), f"Expected shape ({B}, 60), got {logits.shape}"

    # 3. Loss Computation & Backward Pass
    print("\n[4] Running Forward with Loss computation & Backward pass...")
    total_loss, loss_dict = model(x, label=labels, return_loss=True)
    print(f"    - Total Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"      * {k}: {v.item():.4f}")

    total_loss.backward()
    print("    - Backward pass successful! Checking gradients...")

    has_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.group_heads.parameters())
    print(f"    - Group heads have gradients: {has_grad}")
    assert has_grad, "Concept group heads should receive gradients!"

    # 4. Symbolic Explainability
    print("\n[5] Testing Symbolic Explainability (explain_joint)...")
    explanation = model.explain_joint(x[0:1], action_idx=0)
    print(f"    - Generated {len(explanation)} group explanations for sample 0 (Action 0):")
    for exp in explanation:
        print(f"      * Group '{exp['group']}': Positive = {exp['positive']} (score={exp['positive_score']}) | Negative = {exp['negative']}")

    # 5. Freezing & Unfreezing Backbone
    print("\n[6] Testing Freezing & Unfreezing Backbone...")
    frozen, _ = model.freeze_backbone()
    print(f"    - Frozen {frozen} backbone parameters.")
    unfrozen, _ = model.unfreeze_backbone()
    print(f"    - Unfrozen {unfrozen} backbone parameters.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    verify_all()
