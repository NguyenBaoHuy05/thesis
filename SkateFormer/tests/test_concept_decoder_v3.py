"""Focused unit tests for the V3 concept decoder changes."""

from __future__ import annotations

import unittest

import torch

from main_hcg_ns import (
    interpolate_loss_weights,
    load_config,
    scheduled_learning_rate,
)
from model.concept_prototypes import (
    AdaptiveTemporalMILPool,
    ConceptPrototypeDecoder,
    TemporalConceptGRU,
)
from model.concept_ontology import load_ontology_metadata
from model.hierarchical_concept_decoder import (
    HierarchicalConceptDecoder,
    OntologyMaskedConceptRefiner,
)
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

    def test_temporal_gru_is_identity_at_initialization(self) -> None:
        temporal_gru = TemporalConceptGRU(
            num_temporal_concepts=3,
            hidden_dim=5,
            residual_init=0.0,
        )
        logits = torch.randn(2, 7, 3, requires_grad=True)
        output = temporal_gru(logits)
        self.assertTrue(torch.equal(output, logits))
        output.sum().backward()
        self.assertIsNotNone(temporal_gru.alpha.grad)
        self.assertTrue(torch.isfinite(temporal_gru.alpha.grad).all())

    def test_temporal_gru_changes_only_temporal_concepts(self) -> None:
        decoder = ConceptPrototypeDecoder(
            feature_dim=8,
            num_concepts=4,
            temporal_concepts=[1, 3],
            temporal_pooling="adaptive_mil",
            temporal_gru_enabled=True,
            temporal_gru_hidden_dim=6,
        )
        logits = torch.randn(2, 9, 4)
        unchanged = decoder._apply_temporal_gru(logits)
        self.assertTrue(torch.equal(unchanged, logits))

        with torch.no_grad():
            decoder.temporal_gru.alpha.fill_(0.5)
        contextualized = decoder._apply_temporal_gru(logits)
        self.assertTrue(torch.equal(contextualized[:, :, 0], logits[:, :, 0]))
        self.assertTrue(torch.equal(contextualized[:, :, 2], logits[:, :, 2]))
        self.assertFalse(torch.equal(contextualized[:, :, 1], logits[:, :, 1]))
        self.assertFalse(torch.equal(contextualized[:, :, 3], logits[:, :, 3]))

    def test_temporal_gru_is_order_sensitive_before_mil(self) -> None:
        temporal_gru = TemporalConceptGRU(
            num_temporal_concepts=2,
            hidden_dim=4,
        )
        with torch.no_grad():
            temporal_gru.alpha.fill_(1.0)
        logits = torch.tensor([[[1.0, -1.0], [0.0, 0.5], [-0.5, 1.0]]])
        forward = temporal_gru(logits)
        reversed_then_restored = temporal_gru(logits.flip(1)).flip(1)
        self.assertFalse(torch.allclose(forward, reversed_then_restored))

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

    def test_scope_metadata_routes_pose_and_motion_projectors(self) -> None:
        decoder = ConceptPrototypeDecoder(
            feature_dim=8,
            num_concepts=2,
            temporal_concepts=[1],
            temporal_pooling="adaptive_mil",
        )
        features = torch.randn(2, 8, 5, 3)
        shifted = features + torch.randn(2, 8, 1, 3)
        original = decoder(features)["time_logits"]
        translated = decoder(shifted)["time_logits"]

        # A time-constant pose translation changes pose evidence but cancels
        # exactly in first-order temporal differences.
        self.assertFalse(torch.allclose(original[:, :, 0], translated[:, :, 0]))
        self.assertTrue(torch.allclose(
            original[:, :, 1], translated[:, :, 1], atol=1e-6
        ))
        static_pose = torch.randn(2, 8, 1, 3).expand(-1, -1, 5, -1)
        static_logits = decoder(static_pose)["time_logits"]
        self.assertTrue(torch.allclose(
            static_logits[:, :, 1], torch.zeros_like(static_logits[:, :, 1])
        ))

    def test_ontology_refiner_is_identity_then_stays_inside_parent_groups(self) -> None:
        refiner = OntologyMaskedConceptRefiner(
            num_concepts=3,
            concept_to_parent=torch.tensor([0, 0, 1]),
            num_parents=2,
            hidden_dim=4,
        )
        logits = torch.randn(2, 3)
        self.assertTrue(torch.equal(refiner(logits), logits))

        with torch.no_grad():
            refiner.alpha.fill_(0.5)
        changed_other_parent = logits.clone()
        changed_other_parent[:, 2] += 10.0
        refined = refiner(logits)
        refined_changed = refiner(changed_other_parent)
        self.assertTrue(torch.allclose(
            refined[:, :2], refined_changed[:, :2], atol=1e-6
        ))

    def test_full_scope_routed_multiscale_decoder_backward(self) -> None:
        metadata = load_ontology_metadata(
            "concepts/ntu_concept_metadata_v2_4.csv", num_concepts=102
        )
        decoder = HierarchicalConceptDecoder(
            feature_dim=8,
            metadata=metadata,
            num_concepts=102,
            multi_scale=True,
            pyramid_dims=(8, 8),
            pyramid_levels=(0, 1),
            temporal_pooling="adaptive_mil",
            scale_fusion="per_concept",
            num_people=2,
            num_points=2,
            interaction_reasoning=True,
            temporal_gru_enabled=True,
            temporal_gru_hidden_dim=16,
        )
        output = decoder([
            torch.randn(2, 8, 6, 4),
            torch.randn(2, 8, 3, 4),
        ])
        self.assertEqual(tuple(output["time_logits"].shape), (2, 6, 102))
        self.assertEqual(tuple(output["sequence_logits"].shape), (2, 102))
        self.assertEqual(decoder.base.temporal_mask.sum().item(), 25)
        self.assertEqual(len(decoder.scope_ids["context"]), 3)
        output["sequence_logits"].sum().backward()
        self.assertIsNotNone(decoder.base.pose_scale_projectors[0][1].weight.grad)
        self.assertIsNotNone(decoder.base.motion_scale_projectors[0][1].weight.grad)
        self.assertIsNotNone(decoder.base.temporal_gru.alpha.grad)
        self.assertIsNotNone(decoder.concept_refiner.alpha.grad)

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
        self.assertGreater(float(penalized.detach()), 0.5)

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
        self.assertEqual(tuple(output["sequence_logits"].shape), (2, 2))
        self.assertNotIn("soft_gate", output)
        self.assertIn("raw_sequence_probs", output)
        self.assertFalse(hasattr(decoder, "scope_heads"))

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
        self.assertGreater(float(loss.detach()), 0.0)
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

    def test_concept_loss_is_full_binary_cross_entropy_with_logits(self) -> None:
        symbolic = HierarchicalSymbolicDecoder(
            concept_csv="concepts/ntu_action_concept_v2_4.csv",
            metadata_csv="concepts/ntu_concept_metadata_v2_4.csv",
            num_classes=2,
            num_concepts=102,
        )
        logits = torch.full((2, 102), -0.3, requires_grad=True)
        labels = torch.tensor([0, 1])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, symbolic.support[labels]
        )
        actual = symbolic.concept_loss(logits, labels)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        negative_mask = symbolic.support[labels] == 0
        self.assertTrue((logits.grad[negative_mask] > 0).all())

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

    def test_main_config_uses_temporal_gru_joint_training(self) -> None:
        main = load_config("config/HCG_NS_SkateFormer_ntu120_csub.yaml")

        self.assertNotIn("base_config", main)
        self.assertEqual(
            main["work_dir"],
            "./work_dir/hcg_ns/main_ntu120_csub_pose_motion_gru_refiner",
        )
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
        self.assertTrue(main["prototype_args"]["refiner_enabled"])
        self.assertEqual(main["prototype_args"]["refiner_hidden_dim"], 128)
        self.assertTrue(main["prototype_args"]["temporal_gru_enabled"])
        self.assertEqual(main["prototype_args"]["temporal_gru_hidden_dim"], 64)
        self.assertEqual(main["prototype_args"]["temporal_gru_layers"], 1)
        self.assertEqual(main["prototype_args"]["temporal_gru_dropout"], 0.0)
        self.assertEqual(main["prototype_args"]["temporal_gru_residual_init"], 0.0)
        self.assertNotIn("scope_fusion", main["prototype_args"])
        self.assertNotIn("scope_scale", main["prototype_args"])


if __name__ == "__main__":
    unittest.main()
