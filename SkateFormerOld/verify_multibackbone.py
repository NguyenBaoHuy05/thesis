"""
Verification Script for Multi-Backbone Architecture
===================================================
Tests instantiation, forward pass, feature extraction, and Joint Symbolic CBM
across all registered backbones (SkateFormer, CTR-GCN, ST-GCN, Hyper-GCN, SK-GCN).
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.backbones import build_backbone, BACKBONE_REGISTRY
from joint_symbolic_model import build_joint_model, JointSpatialTemporalCBM


def test_backbones():
    print("=" * 70)
    print("  TEST 1: BACKBONE ADAPTERS FORWARD & FEATURE EXTRACTION")
    print("=" * 70)

    B, C, T, V, M = 2, 3, 64, 25, 2
    x = torch.randn(B, C, T, V, M)
    index_t = torch.arange(T).unsqueeze(0).expand(B, -1)
    x_skate = torch.randn(B, C, T, 24, M)

    for name in ["skateformer", "ctrgcn", "stgcn", "hypergcn", "skgcn"]:
        print(f"\n[Testing Backbone: {name.upper()}]")
        input_tensor = x_skate if name == "skateformer" else x
        
        adapter = build_backbone(name, num_class=60)
        feat = adapter.extract_feat(input_tensor, index_t if name == "skateformer" else None)
        logits = adapter(input_tensor, index_t if name == "skateformer" else None)
        
        print(f"  feat_dim    : {adapter.feat_dim}")
        print(f"  feat shape  : {tuple(feat.shape)} (Expected: ({B}, {adapter.feat_dim}))")
        print(f"  logits shape: {tuple(logits.shape)} (Expected: ({B}, 60))")
        
        assert feat.shape == (B, adapter.feat_dim), f"Feature shape mismatch for {name}"
        assert logits.shape == (B, 60), f"Logits shape mismatch for {name}"
        print(f"  -> PASS: {name} adapter verified successfully!")


def test_joint_symbolic_models():
    print("\n" + "=" * 70)
    print("  TEST 2: JOINT SYMBOLIC CBM WITH MULTIPLE BACKBONES")
    print("=" * 70)

    B, C, T, V, M = 2, 3, 64, 25, 2
    x = torch.randn(B, C, T, V, M)
    labels = torch.tensor([0, 5], dtype=torch.long)
    index_t = torch.arange(T).unsqueeze(0).expand(B, -1)
    x_skate = torch.randn(B, C, T, 24, M)

    for name in ["skateformer", "ctrgcn", "stgcn", "hypergcn", "skgcn"]:
        print(f"\n[Testing Joint Symbolic Model with {name.upper()}]")
        input_tensor = x_skate if name == "skateformer" else x
        
        model = build_joint_model(
            checkpoint_path=None,
            backbone_name=name,
            num_class=60,
            device=torch.device("cpu"),
        )
        
        # 1. Forward pass (eval mode)
        model.eval()
        logits = model(input_tensor, index_t if name == "skateformer" else None)
        print(f"  Eval logits : {tuple(logits.shape)}")
        assert logits.shape == (B, 60), f"Logits shape mismatch for {name}"

        # 2. Forward pass with loss (train mode)
        model.train()
        total_loss, loss_dict = model(
            input_tensor,
            index_t=index_t if name == "skateformer" else None,
            label=labels,
            return_loss=True,
        )
        print(f"  Total loss  : {total_loss.item():.4f}")
        print(f"  Loss items  : action={loss_dict['action']:.4f}, symbolic={loss_dict['symbolic']:.4f}, "
              f"concept={loss_dict['concept']:.4f}, kg={loss_dict['kg']:.4f}")
        assert torch.isfinite(total_loss), f"Loss is non-finite for {name}"

        # 3. Backward pass
        total_loss.backward()
        print(f"  Backward pass completed successfully.")

        # 4. Symbolic Explanations
        model.eval()
        exps = model.explain_joint(input_tensor[:1], index_t[:1] if name == "skateformer" else None, action_idx=0)
        print(f"  Explanations: Generated {len(exps)} group explanations (e.g. Group '{exps[0]['group']}' -> +{exps[0]['positive']})")
        assert len(exps) == 9, f"Expected 9 concept group explanations, got {len(exps)}"
        print(f"  -> PASS: Joint Symbolic CBM with {name} verified successfully!")


def main():
    print("\nStarting Multi-Backbone Verification Suite...\n")
    test_backbones()
    test_joint_symbolic_models()
    print("\n" + "=" * 70)
    print("  ALL MULTI-BACKBONE VERIFICATION TESTS PASSED SUCCESSFULLY! (10/10)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
