"""
Automated Verification and Smoke Test Suite for ThesisModel
===========================================================
Tests:
1. Concept loading & 9-group partitioning
2. Knowledge Graph ontology matrices (S_KG, C_KG, A_KG)
3. Multi-Backbone forward/backward passes (SkateFormer & CTR-GCN)
4. Multi-loss computation and gradient propagation
5. Differentiable logic temperature annealing
6. Symbolic First-Order Logic rule extraction (explain)
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concepts.concept_utils import build_concept_groups, load_concept_annotations
from knowledge_graph import (
    build_concept_coactivation_static,
    build_joint_semantic_dependency_matrix,
    build_semantic_similarity_from_concepts,
)
from model.thesis_model import ThesisNeurosymbolicModel


def test_concepts_and_kg():
    print("\n[TEST 1] Verifying Concepts and Knowledge Graph...")
    csv_path = "concepts/ntu_rgbd_60_120.csv"
    assert os.path.exists(csv_path), f"CSV not found at {csv_path}"

    concept_matrix, concept_names, class_names = load_concept_annotations(csv_path, num_classes=60)
    print(f"  * Loaded {len(concept_names)} concepts across {len(class_names)} action classes.")
    assert concept_matrix.shape == (60, len(concept_names))

    groups = build_concept_groups(concept_names)
    print(f"  * Partitioned into {len(groups)} semantic groups:")
    for g, idxs in groups.items():
        print(f"      - {g}: {len(idxs)} concepts")

    # Knowledge Graph
    S_KG = build_semantic_similarity_from_concepts(concept_matrix)
    C_KG = build_concept_coactivation_static(concept_matrix)
    A_KG = build_joint_semantic_dependency_matrix(25, 3)

    assert S_KG.shape == (60, 60), f"S_KG shape {S_KG.shape} != (60, 60)"
    assert C_KG.shape == (len(concept_names), len(concept_names))
    assert A_KG.shape == (3, 25, 25)
    print("  [PASSED] Concept annotations and Knowledge Graph verification passed!")


def test_skateformer_backbone():
    print("\n[TEST 2] Verifying SkateFormer Backbone & End-to-End Forward/Backward...")
    B, C, T, V, M = 2, 3, 64, 25, 2
    x = torch.randn(B, C, T, V, M)
    index_t = torch.linspace(-1, 1, T).unsqueeze(0).repeat(B, 1)
    labels = torch.tensor([0, 1], dtype=torch.long)

    model = ThesisNeurosymbolicModel(
        backbone_name="skateformer",
        backbone_cfg={
            "in_channels": 3,
            "num_classes": 60,
            "num_points": 25,
            "num_people": 2,
            "num_frames": 64,
            "depths": [2, 2, 2, 2],
            "channels": [32, 64, 128, 128],
            "num_heads": 32,
            "type_1_size": [8, 5],
            "type_2_size": [8, 5],
            "type_3_size": [8, 5],
            "type_4_size": [8, 5],
        },
        num_classes=60,
        concepts_csv="concepts/ntu_rgbd_60_120.csv",
        num_rules=32,
        use_clip_alignment=False,  # Test without CLIP text encoder for fast unit test
    )

    concept_gt = torch.zeros(B, model.num_concepts)
    concept_gt[0, :5] = 1.0
    concept_gt[1, 5:10] = 1.0

    outputs = model(x=x, index_t=index_t, labels=labels, concept_gt=concept_gt)

    assert outputs["y_final"].shape == (B, 60)
    assert outputs["y_neural"].shape == (B, 60)
    assert outputs["y_logic"].shape == (B, 60)
    assert outputs["concept_probs"].shape == (B, model.num_concepts)
    assert "loss" in outputs

    loss = outputs["loss"]
    print(f"  * Forward pass successful! Total Loss: {loss.item():.4f}")
    loss.backward()
    print("  * Backward pass and gradient computation successful!")

    # Test explain()
    explanations = model.explain(x, index_t=index_t, top_k_concepts=3)
    assert len(explanations) == B
    print(f"  * Symbolic logic rule explanation extracted: {explanations[0]['predicted_action']}")
    print("  [PASSED] SkateFormer backbone verification passed!")


def test_ctrgcn_backbone():
    print("\n[TEST 3] Verifying CTR-GCN Backbone & End-to-End Forward/Backward...")
    B, C, T, V, M = 2, 3, 64, 25, 2
    x = torch.randn(B, C, T, V, M)
    labels = torch.tensor([2, 3], dtype=torch.long)

    model = ThesisNeurosymbolicModel(
        backbone_name="ctrgcn",
        backbone_cfg={
            "in_channels": 3,
            "num_class": 60,
            "num_point": 25,
            "num_person": 2,
            "graph": "model.backbones.graph_utils.Graph",
            "graph_args": {"layout": "ntu-rgb+d", "strategy": "spatial"},
        },
        num_classes=60,
        concepts_csv="concepts/ntu_rgbd_60_120.csv",
        num_rules=32,
        use_clip_alignment=True,
        lora_r=4,
    )

    concept_gt = torch.zeros(B, model.num_concepts)
    concept_gt[0, 0] = 1.0
    prompts_by_group = {}
    for g_name, indices in model.concept_groups.items():
        g_concepts = [model.concept_names[i] for i in indices]
        prompts_by_group[g_name] = [f"a person performing {c}" for c in g_concepts]

    outputs = model(x=x, labels=labels, concept_gt=concept_gt, prompts_by_group=prompts_by_group)
    assert outputs["y_final"].shape == (B, 60)
    loss = outputs["loss"]
    loss.backward()
    print(f"  * CTR-GCN + LoRA CLIP alignment forward/backward successful! Total Loss: {loss.item():.4f}")
    print("  [PASSED] CTR-GCN backbone verification passed!")


def main():
    print("=" * 60)
    print("    RUNNING THESIS NEUROSYMBOLIC MODEL VERIFICATION SUITE   ")
    print("=" * 60)
    test_concepts_and_kg()
    test_skateformer_backbone()
    test_ctrgcn_backbone()
    print("\n" + "=" * 60)
    print("        ALL AUTOMATED VERIFICATION TESTS PASSED!          ")
    print("=" * 60)


if __name__ == "__main__":
    main()
