import unittest

import numpy as np
import torch

from model.checkpoint import load_model_weights
from model.hypergcn_large import HYPERGC, Model, SemanticHYPERGC
from model.synergies import (
    NUM_JOINTS, PAIR_SYNERGIES, SINGLE_SYNERGIES, SPINE_SHOULDER, incidence, pair_validity,
    random_bank, validate_bank,
)
from model.synergy_bottleneck import SemanticHyperedges, SynergyBottleneck, SynergyPooling, segment_index
from model.synergy_metrics import effective_synergies, plausibility, readout_faithfulness
from model.training import configure_trainable_parameters

T = 64
MODEL_ARGS = dict(num_class=120, num_point=25, num_person=2, hyper_joints=3,
                  graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))


def to_feeder(actors):
    """Absolute positions [T,V,3] per actor -> NTU joint-mode input [3,T,V,2]."""
    x = torch.zeros(3, T, NUM_JOINTS, 2)
    for m, positions in enumerate(actors):
        relative = positions - positions[:, SPINE_SHOULDER:SPINE_SHOULDER + 1]
        relative[:, SPINE_SHOULDER] = positions[:, SPINE_SHOULDER]
        x[..., m] = relative.permute(2, 0, 1)
    return x


def standing(offset=0.0):
    torch.manual_seed(0)
    pose = torch.randn(NUM_JOINTS, 3) * 0.3
    pose[0], pose[20] = torch.tensor([0., 0., 0.]), torch.tensor([0., .5, 0.])
    return (pose + torch.tensor([offset, 0., 0.])).expand(T, -1, -1).clone()


class BankTests(unittest.TestCase):
    def test_semantic_bank_is_valid(self):
        validate_bank(SINGLE_SYNERGIES, PAIR_SYNERGIES)
        self.assertEqual((len(SINGLE_SYNERGIES), len(PAIR_SYNERGIES)), (19, 5))

    def test_random_bank_matches_sizes(self):
        singles, pairs = random_bank(seed=3)
        validate_bank(singles, pairs)
        self.assertEqual([len(s.joints) for s in singles], [len(s.joints) for s in SINGLE_SYNERGIES])
        self.assertNotEqual([s.joints for s in singles], [s.joints for s in SINGLE_SYNERGIES])
        self.assertEqual(random_bank(seed=3), random_bank(seed=3))

    def test_incidence_leaves_virtual_joints_out(self):
        H = incidence([s.joints for s in SINGLE_SYNERGIES], NUM_JOINTS + 3)
        self.assertEqual(H.shape, (28, 19))
        self.assertEqual(float(H[25:].sum()), 0.0)
        self.assertEqual(H[:, 0].nonzero().flatten().tolist(), sorted(SINGLE_SYNERGIES[0].joints))

    def test_duplicated_second_actor_is_not_a_pair(self):
        x = torch.stack([to_feeder([standing(), standing()]),
                         to_feeder([standing(), standing(1.0)]),
                         to_feeder([standing()])])
        self.assertEqual(pair_validity(x).tolist(), [False, True, False])


class BottleneckTests(unittest.TestCase):
    def make(self, readout='embedding', pairs=PAIR_SYNERGIES):
        torch.manual_seed(0)
        bottleneck = SynergyBottleneck(8, SINGLE_SYNERGIES, pairs, num_classes=5, segments=4,
                                       readout=readout, tied=False, gate_hidden=4)
        features = torch.randn(3, 2, 8, 16, 25)
        actor_valid = torch.tensor([[True, True], [True, False], [True, True]])
        pair_valid = torch.tensor([True, False, True])
        return bottleneck, features, actor_valid, pair_valid

    def test_segment_index_matches_tensor_split(self):
        self.assertEqual(segment_index(10, 4).tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])

    def test_pooling_reads_only_the_synergy_joints(self):
        features = torch.randn(1, 8, 16, 25, requires_grad=True)
        SynergyPooling([s.joints for s in SINGLE_SYNERGIES], 4)(features)[..., 0, :].sum().backward()
        touched = features.grad.abs().sum(dim=(0, 1, 2)).nonzero().flatten().tolist()
        self.assertEqual(touched, sorted(SINGLE_SYNERGIES[0].joints))

    def test_logits_are_the_sum_of_contributions(self):
        for readout in ('embedding', 'scalar'):
            bottleneck, features, actor_valid, pair_valid = self.make(readout)
            result = bottleneck(features, actor_valid, pair_valid)
            self.assertEqual(result['contributions'].shape, (3, 5, 24, 4))
            torch.testing.assert_close(
                result['action_logits'], result['contributions'].sum(dim=(-1, -2)) + bottleneck.class_bias)

    def test_actor_order_does_not_matter(self):
        bottleneck, features, actor_valid, _ = self.make()
        valid = torch.ones(3, 2, dtype=torch.bool)
        pair_valid = torch.ones(3, dtype=torch.bool)
        first = bottleneck(features, valid, pair_valid)['action_logits']
        second = bottleneck(features.flip(1), valid, pair_valid)['action_logits']
        torch.testing.assert_close(first, second)

    def test_invalid_pairs_contribute_nothing(self):
        bottleneck, features, actor_valid, pair_valid = self.make()
        contributions = bottleneck(features, actor_valid, pair_valid)['contributions']
        self.assertEqual(float(contributions[1, :, 19:].abs().sum()), 0.0)

    def test_gate_mask_removes_contributions(self):
        bottleneck, features, actor_valid, pair_valid = self.make()
        mask = torch.ones(3, 4, 24)
        mask[:, 2, 5] = 0
        contributions = bottleneck(features, actor_valid, pair_valid, gate_mask=mask)['contributions']
        self.assertEqual(float(contributions[:, :, 5, 2].abs().sum()), 0.0)


class SemanticHypergraphTests(unittest.TestCase):
    def test_rows_sum_to_the_mean_gate(self):
        groups = [s.joints for s in SINGLE_SYNERGIES]
        layer = SemanticHyperedges(8, groups, 28, segments=4, gate_hidden=4)
        G, gate = layer(torch.randn(2, 8, 16, 28))
        self.assertEqual(G.shape, (2, 16, 28, 28))
        H = incidence(groups, 28)
        expected = (H[None] * gate[:, 0, None]).sum(-1) / H.sum(1).clamp_min(1)
        torch.testing.assert_close(G[:, 0].sum(-1), expected)
        G, _ = layer(torch.randn(2, 8, 16, 28), gate_mask=torch.zeros(2, 4, 19))
        self.assertEqual(float(G.abs().sum()), 0.0)

    def test_zero_beta_keeps_the_pretrained_layer(self):
        from graph.ntu_rgb_d import Graph
        A = Graph(3, labeling_mode='virtual_ensemble').A
        torch.manual_seed(0)
        base = HYPERGC(64, 64, 25, 3, A).eval()
        semantic = SemanticHYPERGC(64, 64, 25, 3, A, groups=[s.joints for s in SINGLE_SYNERGIES]).eval()
        semantic.load_state_dict(base.state_dict(), strict=False)
        x = torch.randn(2, 64, 8, 25)
        torch.testing.assert_close(base(x)[0], semantic(x)[0])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.plain = Model(**MODEL_ARGS).eval()
        cls.model = Model(**MODEL_ARGS, synergy_enabled=True).eval()
        load_model_weights(cls.model, cls.plain.state_dict(), allow_new=True)
        cls.x = torch.stack([to_feeder([standing(), standing(1.0)]), to_feeder([standing()])])

    def test_official_checkpoint_needs_allow_new(self):
        with self.assertRaises(ValueError):
            load_model_weights(Model(**MODEL_ARGS, synergy_enabled=True), self.plain.state_dict())

    def test_forward_and_gate_mask(self):
        with torch.no_grad():
            output = self.model(self.x)
            self.assertEqual(output.action_logits.shape, (2, 120))
            self.assertEqual(output.gates.shape, (2, 4, 24))
            torch.testing.assert_close(
                output.action_logits,
                output.contributions.sum(dim=(-1, -2)) + self.model.synergy.class_bias)
            closed = self.model(self.x, gate_mask=torch.zeros(2, 4, 24)).action_logits
            torch.testing.assert_close(closed, self.model.synergy.class_bias.expand(2, -1))

    def test_tied_gates_come_from_l9(self):
        with torch.no_grad():
            output = self.model(self.x)
        self.assertEqual(self.model.l9.gcn1.gate.shape, (4, 4, 19))
        self.assertTrue(bool(((output.gates >= 0) & (output.gates <= 1)).all()))

    def test_freeze_until_l6(self):
        model = Model(**MODEL_ARGS, synergy_enabled=True)
        configure_trainable_parameters(model, 'l6')
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertFalse(any(name.startswith(('l6.', 'l1.', 'data_bn', 'fc.')) for name in trainable))
        self.assertIn('l7.gcn1.conv_d.weight', trainable)
        self.assertIn('l9.gcn1.semantic.beta', trainable)
        self.assertIn('synergy.class_weight', trainable)


class MetricTests(unittest.TestCase):
    def test_deletion_beats_random_on_a_concentrated_explanation(self):
        contributions = torch.zeros(4, 3, 6, 2)
        contributions[:, 0, 0, 0] = 5.0  # one entry decides class 0
        contributions[:, 0, 1:, :] = 0.01
        result = readout_faithfulness(contributions, torch.zeros(3), torch.Generator().manual_seed(0))
        self.assertLess(float(result['deletion_auc'].mean()), float(result['random_deletion_auc'].mean()))
        self.assertGreater(float(result['insertion_auc'].mean()), float(result['random_insertion_auc'].mean()))
        self.assertEqual(effective_synergies(contributions, torch.zeros(4, dtype=torch.long)).tolist(), [1] * 4)

    def test_plausibility_of_a_perfect_ranking(self):
        annotations = np.array([[2, 1, 0, 0], [0, 0, 2, 0]])
        profile = np.array([[3., 2., 1., 0.], [0., 1., 5., 0.]])
        result = plausibility(profile, annotations, k=1)
        self.assertEqual((result['precision_at_k'], result['ndcg_at_k']), (1.0, 1.0))


if __name__ == '__main__':
    unittest.main()
