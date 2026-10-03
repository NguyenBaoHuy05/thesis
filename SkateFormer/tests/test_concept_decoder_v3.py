"""Focused unit tests for the V3 concept decoder changes."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from main_hcg_ns import load_config
from model.concept_prototypes import AdaptiveTemporalMILPool, ConceptPrototypeDecoder
from model.hierarchical_concept_decoder import HierarchicalConceptDecoder
from model.hierarchical_symbolic_decoder import HierarchicalSymbolicDecoder


class ConceptDecoderV3Tests(unittest.TestCase):
    def test_adaptive_pool_shape_finite_and_permutation_invariant(self) -> None:
        pool = AdaptiveTemporalMILPool(num_concepts=5)
        logits = torch.randn(3, 11, 5, requires_grad=True)
        probabilities = pool(logits)
        self.assertEqual(tuple(probabilities.shape), (3, 5))
        self.assertTrue(torch.isfinite(probabilities).all())
        permuted = pool(logits[:, torch.randperm(logits.shape[1])])
        self.assertTrue(torch.allclose(probabilities, permuted, atol=1e-6))

    def test_adaptive_pool_gradients_reach_all_parameters(self) -> None:
        pool = AdaptiveTemporalMILPool(num_concepts=4)
        logits = torch.randn(2, 7, 4, requires_grad=True)
        pool(logits).sum().backward()
        self.assertIsNotNone(logits.grad)
        for parameter in (pool.beta_raw, pool.mix_raw, pool.bias):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_per_concept_scale_weights(self) -> None:
        decoder = ConceptPrototypeDecoder(
            feature_dim=8,
            num_concepts=5,
            multi_scale=True,
            pyramid_dims=(8, 8),
            pyramid_levels=(0, 1),
            temporal_pooling="adaptive_mil",
            scale_fusion="per_concept",
        )
        weights = decoder.scale_weights
        self.assertIsNotNone(weights)
        self.assertEqual(tuple(weights.shape), (2, 5))
        self.assertTrue(torch.allclose(weights.sum(dim=0), torch.ones(5)))

        output = decoder([
            torch.randn(2, 8, 6, 4),
            torch.randn(2, 8, 3, 4),
        ])
        self.assertEqual(tuple(output["time_logits"].shape), (2, 6, 5))
        output["sequence_probs"].sum().backward()
        self.assertIsNotNone(decoder.scale_logits.grad)
        self.assertTrue(torch.isfinite(decoder.scale_logits.grad).all())

    def test_parent_disabled_keeps_concept_path_without_gate_output(self) -> None:
        metadata = [
            {
                "concept_id": "0",
                "concept_name": "concept_a",
                "scope": "spatial",
                "parent_name": "parent_a",
                "level": "2",
            },
            {
                "concept_id": "1",
                "concept_name": "concept_b",
                "scope": "temporal",
                "parent_name": "parent_b",
                "level": "2",
            },
        ]
        decoder = HierarchicalConceptDecoder(
            feature_dim=8,
            metadata=metadata,
            num_concepts=2,
            num_people=1,
            num_points=2,
            interaction_reasoning=False,
            parent_enabled=False,
            temporal_pooling="adaptive_mil",
        )
        output = decoder(torch.randn(2, 8, 6, 2))
        self.assertTrue(torch.equal(output["parent_probs"], torch.zeros_like(output["parent_probs"])))
        self.assertEqual(tuple(output["sequence_probs"].shape), (2, 2))
        self.assertNotIn("soft_gate", output)
        self.assertNotIn("raw_sequence_probs", output)

    def test_legacy_floor_one_regression_is_identity(self) -> None:
        """A V2 checkpoint configured with floor=1 had an identity gate."""
        metadata = [
            {
                "concept_id": "0",
                "concept_name": "concept_a",
                "scope": "spatial",
                "parent_name": "parent_a",
                "level": "2",
            },
            {
                "concept_id": "1",
                "concept_name": "concept_b",
                "scope": "temporal",
                "parent_name": "parent_b",
                "level": "2",
            },
        ]
        decoder = HierarchicalConceptDecoder(
            feature_dim=8,
            metadata=metadata,
            num_concepts=2,
            num_people=1,
            num_points=2,
            interaction_reasoning=False,
            parent_enabled=True,
            temporal_pooling="legacy_noisy_or",
        )
        output = decoder(torch.randn(2, 8, 6, 2))
        old_raw = decoder.base._temporal_statistics(
            output["time_probs"], output["time_logits"]
        )["sequence_probs"]
        old_floor_one = old_raw * (
            1.0 + 0.0 * output["parent_for_child"]
        )
        self.assertTrue(torch.allclose(output["sequence_probs"], old_floor_one, atol=1e-7))

        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
        )
        sequence = torch.rand(2, 102)
        parents = torch.rand(2, symbolic.num_parents)
        legacy_sequence = sequence * (
            1.0 + 0.0 * parents[:, symbolic.concept_to_parent]
        )
        self.assertTrue(torch.allclose(
            symbolic(sequence, parents), symbolic(legacy_sequence, parents), atol=1e-7
        ))

    def test_hierarchy_loss_accepts_official_sequence_probs(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
        )
        sequence_probs = torch.full((2, 102), 0.25)
        parent_probs = torch.full((2, symbolic.num_parents), 0.5)
        loss = symbolic.hierarchy_loss(sequence_probs, parent_probs)
        self.assertTrue(torch.isfinite(loss))

    def test_parent_graph_regularizes_only_violating_concepts(self) -> None:
        metadata = [
            {
                "concept_id": "0", "concept_name": "concept_a",
                "scope": "spatial", "parent_name": "parent_a", "level": "2",
            },
            {
                "concept_id": "1", "concept_name": "concept_b",
                "scope": "spatial", "parent_name": "parent_a", "level": "2",
            },
            {
                "concept_id": "2", "concept_name": "concept_c",
                "scope": "temporal", "parent_name": "parent_b", "level": "2",
            },
        ]
        decoder = HierarchicalConceptDecoder(
            feature_dim=8,
            metadata=metadata,
            num_concepts=3,
            num_people=1,
            num_points=2,
            interaction_reasoning=False,
        )
        concepts = torch.tensor([[0.8, 0.2, 0.4]], requires_grad=True)
        parents = torch.tensor([[0.5, 0.6]], requires_grad=True)
        loss = decoder.parent_graph_loss(concepts, parents)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertGreater(float(concepts.grad[0, 0]), 0.0)
        self.assertEqual(float(concepts.grad[0, 1]), 0.0)
        self.assertEqual(float(concepts.grad[0, 2]), 0.0)
        self.assertIsNone(parents.grad)

    def test_parent_graph_is_zero_when_parent_disabled(self) -> None:
        metadata = [{
            "concept_id": "0", "concept_name": "concept_a", "scope": "spatial",
            "parent_name": "parent_a", "level": "2",
        }]
        decoder = HierarchicalConceptDecoder(
            feature_dim=8,
            metadata=metadata,
            num_concepts=1,
            num_people=1,
            num_points=2,
            interaction_reasoning=False,
            parent_enabled=False,
        )
        loss = decoder.parent_graph_loss(torch.tensor([[0.9]]), torch.tensor([[0.1]]))
        self.assertEqual(float(loss), 0.0)

    def test_parent_disabled_removes_symbolic_parent_contribution(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
            parent_weight=0.35,
            parent_enabled=False,
        )
        sequence = torch.rand(3, 102)
        low_parent = torch.zeros(3, symbolic.num_parents)
        high_parent = torch.ones_like(low_parent)
        self.assertTrue(torch.equal(
            symbolic(sequence, low_parent), symbolic(sequence, high_parent)
        ))

    def test_symbolic_decoder_has_no_class_bias(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
        )
        self.assertNotIn("bias", dict(symbolic.named_parameters()))
        self.assertNotIn("bias", symbolic.export_rulebook()["dense_weights"])

    def test_logit_mode_maps_concept_and_parent_to_same_domain(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
            concept_score_mode="logit",
        )
        probabilities = torch.tensor([[0.2, 0.5, 0.8]])
        expected = torch.logit(probabilities)
        self.assertTrue(torch.allclose(symbolic._score_evidence(probabilities), expected))

    def test_ntu120_csub_ablation_configs_resolve(self) -> None:
        config_dir = Path("config/train/ablation_ntu120_csub_v31")
        expected = {
            "V2_R.yaml": ("legacy_noisy_or", False, "shared", True),
            "V3_A.yaml": ("adaptive_mil", False, "shared", True),
            "V3_B.yaml": ("legacy_noisy_or", True, "per_concept", True),
            "V3_C.yaml": ("adaptive_mil", True, "per_concept", True),
            "V3_D.yaml": ("adaptive_mil", True, "per_concept", False),
            "V3_E.yaml": ("adaptive_mil", True, "per_concept", True),
        }
        for name, settings in expected.items():
            with self.subTest(config=name):
                config = load_config(config_dir / name)
                prototype = config["prototype_args"]
                self.assertEqual(config["num_classes"], 120)
                self.assertEqual(config["num_concepts"], 102)
                self.assertEqual(prototype["temporal_pooling"], settings[0])
                self.assertEqual(prototype["multi_scale"], settings[1])
                self.assertEqual(prototype["scale_fusion"], settings[2])
                self.assertEqual(prototype["parent_enabled"], settings[3])
                self.assertIn("ntu120_csub_v31", config["work_dir"])
                if name == "V3_E.yaml":
                    self.assertEqual(config["stage1_loss_weights"]["graph"], 0.0)
                    self.assertEqual(config["stage1_loss_weights"]["hierarchy"], 0.50)
                    self.assertEqual(config["stage2_loss_weights"]["graph"], 0.10)
                    self.assertEqual(config["stage2_loss_weights"]["hierarchy"], 0.0)


if __name__ == "__main__":
    unittest.main()
