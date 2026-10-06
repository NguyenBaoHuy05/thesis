"""Focused unit tests for the V3 concept decoder changes."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from main_hcg_ns import (
    interpolate_loss_weights,
    load_config,
    scheduled_learning_rate,
)
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

    def test_margin_grounding_uses_positive_and_hard_negative_margins(self) -> None:
        decoder = ConceptPrototypeDecoder(
            feature_dim=2,
            num_concepts=3,
            center_loss_mode="margin_hard_negative",
            center_positive_margin=0.35,
            center_negative_margin=0.10,
            center_negative_weight=0.5,
            center_negative_topk=1,
        )
        cosine = torch.tensor([[[[0.20, 0.30, -0.10]]]], requires_grad=True)
        support = torch.tensor([[1.0, 0.0, 0.0]])
        loss = decoder.center_grounding_loss(cosine, torch.tensor([0]), support)

        # Positive hinge: .35 - .20 = .15. Hard negative hinge:
        # .5 * (.30 - .10) = .10.
        self.assertTrue(torch.allclose(loss, torch.tensor(0.25)))
        loss.backward()
        self.assertLess(float(cosine.grad[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(cosine.grad[0, 0, 0, 1]), 0.0)
        self.assertEqual(float(cosine.grad[0, 0, 0, 2]), 0.0)

    def test_pairwise_gap_grounding_uses_topk_negative_mean(self) -> None:
        decoder = ConceptPrototypeDecoder(
            feature_dim=2,
            num_concepts=3,
            center_loss_mode="pairwise_gap_hard_negative",
            center_gap_margin=0.20,
            center_gap_topk=2,
            center_negative_margin=0.10,
            center_negative_weight=0.5,
            center_negative_topk=1,
        )
        cosine = torch.tensor([[[[0.20, 0.30, -0.10]]]], requires_grad=True)
        support = torch.tensor([[1.0, 0.0, 0.0]])
        loss = decoder.center_grounding_loss(cosine, torch.tensor([0]), support)

        # Top-2 negative mean: (.30 - .10) / 2 = .10.
        # Gap hinge: .20 - (.20 - .10) = .10. Negative hinge:
        # .5 * (.30 - .10) = .10.
        self.assertTrue(torch.allclose(loss, torch.tensor(0.20)))
        loss.backward()
        self.assertLess(float(cosine.grad[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(cosine.grad[0, 0, 0, 1]), 0.0)
        self.assertGreater(float(cosine.grad[0, 0, 0, 2]), 0.0)

        satisfied = torch.tensor([[[[0.50, 0.10, -0.20]]]])
        zero_loss = decoder.center_grounding_loss(
            satisfied, torch.tensor([0]), support
        )
        self.assertTrue(torch.allclose(zero_loss, torch.zeros_like(zero_loss)))

    def test_hard_negative_separation_ignores_cooccurring_concepts(self) -> None:
        decoder = ConceptPrototypeDecoder(
            feature_dim=2,
            num_concepts=3,
            center_loss_mode="margin_hard_negative",
            center_separation_margin=0.15,
            center_separation_topk=1,
        )
        with torch.no_grad():
            decoder.centers.copy_(torch.tensor([
                [1.0, 0.0],
                [1.0, 0.01],
                [-1.0, 0.0],
            ]))
        cooccurring_support = torch.tensor([
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        separated_support = torch.eye(3)

        ignored = decoder.center_separation_loss(cooccurring_support)
        penalized = decoder.center_separation_loss(separated_support)
        self.assertTrue(torch.allclose(ignored, torch.zeros_like(ignored)))
        self.assertGreater(float(penalized), 0.5)

    def test_joint_stage_weight_and_learning_rate_schedules(self) -> None:
        target = {"concept": 0.25, "graph": 0.10}
        start = {"graph": 0.0}
        expected_graph = [0.0, 0.025, 0.05, 0.075, 0.10, 0.10]
        actual_graph = [
            interpolate_loss_weights(target, start, epoch, 5)["graph"]
            for epoch in range(1, 7)
        ]
        self.assertTrue(torch.allclose(
            torch.tensor(actual_graph), torch.tensor(expected_graph)
        ))
        self.assertTrue(all(
            interpolate_loss_weights(target, start, epoch, 5)["concept"] == 0.25
            for epoch in range(1, 7)
        ))
        schedule = {1: 1e-4, 6: 5e-5}
        self.assertEqual(scheduled_learning_rate(1e-4, schedule, 5), 1e-4)
        self.assertEqual(scheduled_learning_rate(1e-4, schedule, 6), 5e-5)

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

    def test_scope_heads_can_be_the_direct_concept_classifier(self) -> None:
        metadata = [
            {
                "concept_id": "0", "concept_name": "concept_a",
                "scope": "spatial", "parent_name": "parent_a", "level": "2",
            },
            {
                "concept_id": "1", "concept_name": "concept_b",
                "scope": "temporal", "parent_name": "parent_b", "level": "2",
            },
        ]
        decoder = HierarchicalConceptDecoder(
            feature_dim=8,
            metadata=metadata,
            num_concepts=2,
            concept_classifier="scope",
            parent_enabled=False,
        )
        output = decoder(torch.randn(3, 8, 5, 2))

        self.assertIsNone(decoder.base)
        self.assertFalse(decoder.multi_scale)
        self.assertEqual(tuple(output["sequence_probs"].shape), (3, 2))
        self.assertTrue(torch.allclose(
            output["sequence_probs"], output["scope_probs"]
        ))
        self.assertTrue(torch.allclose(
            output["time_logits"].squeeze(1), output["scope_logits"]
        ))
        output["sequence_probs"].sum().backward()
        for parameter in decoder.scope_heads.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_scope_head_can_be_removed_from_prototype_concept_path(self) -> None:
        metadata = [
            {
                "concept_id": "0", "concept_name": "concept_a",
                "scope": "spatial", "parent_name": "parent_a", "level": "2",
            },
            {
                "concept_id": "1", "concept_name": "concept_b",
                "scope": "temporal", "parent_name": "parent_b", "level": "2",
            },
        ]
        decoder = HierarchicalConceptDecoder(
            feature_dim=8,
            metadata=metadata,
            num_concepts=2,
            temporal_pooling="adaptive_mil",
            scope_enabled=False,
        )
        features = torch.randn(3, 8, 5, 2)
        base = decoder.base(features)
        output = decoder(features)

        self.assertEqual(len(decoder.scope_heads), 0)
        self.assertFalse(hasattr(decoder, "scope_gate_raw"))
        self.assertTrue(torch.equal(
            output["scope_term"], torch.zeros_like(output["scope_term"])
        ))
        self.assertTrue(torch.allclose(output["time_logits"], base["time_logits"]))
        self.assertTrue(torch.allclose(
            output["sequence_probs"], base["sequence_probs"]
        ))

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
        legacy_sequence = sequence * (
            1.0 + 0.0 * symbolic.concept_to_parent
        )
        self.assertTrue(torch.allclose(
            symbolic(sequence), symbolic(legacy_sequence), atol=1e-7
        ))

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

    def test_parent_is_auxiliary_only(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
        )
        sequence = torch.rand(3, 102)
        self.assertEqual(tuple(symbolic(sequence).shape), (3, 2))
        self.assertNotIn("parent_raw", dict(symbolic.named_parameters()))
        rulebook = symbolic.export_rulebook()
        self.assertEqual(rulebook["parent_role"], "auxiliary_supervision")
        self.assertNotIn("parent", rulebook["dense_weights"])
        self.assertNotIn("parent_weight", rulebook["coefficients"])
        parent_logits = torch.randn(3, symbolic.num_parents, requires_grad=True)
        symbolic.parent_loss(parent_logits, torch.tensor([0, 1, 0])).backward()
        self.assertIsNotNone(parent_logits.grad)
        self.assertTrue(torch.isfinite(parent_logits.grad).all())

    def test_concept_loss_is_full_binary_cross_entropy(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
        )
        probabilities = torch.full((2, 102), 0.3, requires_grad=True)
        labels = torch.tensor([0, 1])
        expected = torch.nn.functional.binary_cross_entropy(
            probabilities, symbolic.support[labels]
        )
        actual = symbolic.concept_loss(probabilities, labels)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        negative_mask = symbolic.support[labels] == 0
        self.assertTrue((probabilities.grad[negative_mask] > 0).all())

    def test_symbolic_decoder_has_no_class_bias(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
        )
        self.assertNotIn("bias", dict(symbolic.named_parameters()))
        self.assertNotIn("bias", symbolic.export_rulebook()["dense_weights"])

    def test_logit_mode_maps_concepts_to_logit_domain(self) -> None:
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
        config_dir = Path("config/ablation_ntu120_csub_v31")
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
                    self.assertEqual(config["stage2_loss_weights"]["graph"], 0.10)
                for profile in (
                    "loss_weights", "stage1_loss_weights", "stage2_loss_weights"
                ):
                    self.assertIn("concept", config[profile])
                    self.assertNotIn("weak_concept", config[profile])
                    self.assertNotIn("weak_positive", config[profile])
                    self.assertNotIn("weak_negative", config[profile])
                    self.assertNotIn("density", config[profile])
                    self.assertNotIn("hierarchy", config[profile])
                    self.assertNotIn("sparse", config[profile])
                    self.assertNotIn("hard_negative", config[profile])
                self.assertNotIn("parent_weight", config)

    def test_main_config_uses_joint_training_without_mutating_v3_e(self) -> None:
        main = load_config("config/HCG_NS_SkateFormer_ntu120_csub.yaml")
        v3_e = load_config("config/ablation_ntu120_csub_v31/V3_E.yaml")

        self.assertNotIn("base_config", main)
        self.assertEqual(main["work_dir"], "./work_dir/hcg_ns/main_ntu120_csub")
        self.assertEqual(main["stage1_name"], "joint_grounding_composition")
        self.assertEqual(main["stage1_epochs"], 15)
        self.assertEqual(main["stage2_epochs"], 0)
        self.assertEqual(main["stage1_loss_weights"], main["loss_weights"])
        self.assertEqual(main["stage1_loss_warmup_start_weights"], {"graph": 0.0})
        self.assertEqual(
            main["prototype_args"]["center_loss_mode"],
            "pairwise_gap_hard_negative",
        )
        self.assertEqual(main["prototype_args"]["center_gap_margin"], 0.20)
        self.assertEqual(main["prototype_args"]["center_gap_topk"], 5)
        self.assertEqual(v3_e["stage1_epochs"], 5)
        self.assertEqual(v3_e["stage2_epochs"], 10)
        self.assertNotIn("center_loss_mode", v3_e["prototype_args"])

    def test_scope_head_ablation_config_disables_other_concept_paths(self) -> None:
        config = load_config(
            "config/HCG_NS_SkateFormer_ntu120_csub_scope_head.yaml"
        )
        prototype = config["prototype_args"]
        self.assertEqual(prototype["concept_classifier"], "scope")
        self.assertFalse(prototype["multi_scale"])
        self.assertFalse(prototype["interaction_reasoning"])
        self.assertFalse(prototype["parent_enabled"])
        self.assertEqual(config["stage1_epochs"], 0)
        self.assertEqual(config["stage2_epochs"], 15)
        for name in ("parent", "graph", "center"):
            self.assertEqual(config["stage2_loss_weights"][name], 0.0)

    def test_no_scope_head_ablation_changes_only_scope_enabled(self) -> None:
        main = load_config("config/HCG_NS_SkateFormer_ntu120_csub.yaml")
        ablation = load_config(
            "config/HCG_NS_SkateFormer_ntu120_csub_no_scope_head.yaml"
        )
        self.assertFalse(ablation["prototype_args"]["scope_enabled"])
        for key, value in main.items():
            if key == "work_dir":
                continue
            if key == "prototype_args":
                expected = dict(value)
                expected["scope_enabled"] = False
                self.assertEqual(ablation[key], expected)
            else:
                self.assertEqual(ablation[key], value, key)


if __name__ == "__main__":
    unittest.main()
