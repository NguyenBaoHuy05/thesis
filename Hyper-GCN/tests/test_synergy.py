import unittest

import numpy as np
import torch

from model.checkpoint import load_model_weights
from model.hypergcn_large import Model
from model.synergies import SINGLE_SYNERGIES, incidence, random_bank, validate_bank
from model.synergy_bottleneck import SynergyBottleneck, SynergyPooling
from model.synergy_metrics import effective_synergies, plausibility, readout_faithfulness
from model.training import configure_trainable_parameters

MODEL_ARGS = dict(num_class=120, num_point=25, num_person=2, hyper_joints=3,
                  graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))


class BankTests(unittest.TestCase):
    def test_semantic_bank_is_valid(self):
        validate_bank(SINGLE_SYNERGIES)
        self.assertEqual(len(SINGLE_SYNERGIES), 19)

    def test_random_bank_matches_sizes(self):
        bank = random_bank(seed=3)
        validate_bank(bank)
        self.assertEqual([len(s.joints) for s in bank], [len(s.joints) for s in SINGLE_SYNERGIES])
        self.assertNotEqual([s.joints for s in bank], [s.joints for s in SINGLE_SYNERGIES])
        self.assertEqual(random_bank(seed=3), random_bank(seed=3))

    def test_incidence(self):
        H = incidence([s.joints for s in SINGLE_SYNERGIES])
        self.assertEqual(H.shape, (25, 19))
        self.assertEqual(H[:, 0].nonzero().flatten().tolist(), sorted(SINGLE_SYNERGIES[0].joints))


class BottleneckTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(0)
        bottleneck = SynergyBottleneck(8, SINGLE_SYNERGIES, num_classes=5, segments=4, gate_hidden=4)
        features = torch.randn(3, 2, 8, 16, 25)
        actor_valid = torch.tensor([[True, True], [True, False], [True, True]])
        return bottleneck, features, actor_valid

    def test_pooling_reads_only_the_synergy_joints(self):
        features = torch.randn(1, 8, 16, 25, requires_grad=True)
        SynergyPooling([s.joints for s in SINGLE_SYNERGIES], 4)(features)[..., 0, :].sum().backward()
        touched = features.grad.abs().sum(dim=(0, 1, 2)).nonzero().flatten().tolist()
        self.assertEqual(touched, sorted(SINGLE_SYNERGIES[0].joints))

    def test_logits_are_the_sum_of_contributions(self):
        bottleneck, features, actor_valid = self.make()
        result = bottleneck(features, actor_valid)
        self.assertEqual(result['contributions'].shape, (3, 5, 19, 4))
        torch.testing.assert_close(
            result['action_logits'], result['contributions'].sum(dim=(-1, -2)) + bottleneck.class_bias)

    def test_contribution_is_bounded_by_scale_times_gate(self):
        bottleneck, features, _ = self.make()
        with torch.no_grad():
            bottleneck.class_weight.mul_(1e3)  # growing w must not let g shrink for free
        result = bottleneck(features, torch.tensor([[True, False]] * 3))
        bound = bottleneck.readout_scale * result['gates'].transpose(1, 2)[:, None]
        self.assertTrue(bool((result['contributions'].abs() <= bound + 1e-4).all()))

    def test_actor_order_does_not_matter(self):
        bottleneck, features, _ = self.make()
        valid = torch.ones(3, 2, dtype=torch.bool)
        torch.testing.assert_close(bottleneck(features, valid)['action_logits'],
                                   bottleneck(features.flip(1), valid)['action_logits'])

    def test_gate_mask_removes_contributions(self):
        bottleneck, features, actor_valid = self.make()
        mask = torch.ones(3, 4, 19)
        mask[:, 2, 5] = 0
        contributions = bottleneck(features, actor_valid, gate_mask=mask)['contributions']
        self.assertEqual(float(contributions[:, :, 5, 2].abs().sum()), 0.0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.plain = Model(**MODEL_ARGS).eval()
        cls.x = torch.randn(2, 3, 64, 25, 2)
        cls.x[1, ..., 1] = 0  # second sample has one actor

    def make(self, **kwargs):
        model = Model(**MODEL_ARGS, synergy_enabled=True, **kwargs).eval()
        load_model_weights(model, self.plain.state_dict(), allow_new=True)
        return model

    def test_plain_model_is_unchanged(self):
        with torch.no_grad():
            logits, hyper = self.plain(self.x)
        self.assertEqual(logits.shape, (2, 120))
        self.assertEqual(len(hyper), 9)

    def test_official_checkpoint_needs_allow_new(self):
        with self.assertRaises(ValueError):
            load_model_weights(Model(**MODEL_ARGS, synergy_enabled=True), self.plain.state_dict())

    def test_forward_and_gate_mask(self):
        model = self.make()
        with torch.no_grad():
            output = model(self.x)
            self.assertEqual(output.action_logits.shape, (2, 120))
            self.assertEqual(output.gates.shape, (2, 4, 19))
            closed = model(self.x, gate_mask=torch.zeros(2, 4, 19)).action_logits
        torch.testing.assert_close(closed, model.synergy.class_bias.expand(2, -1))

    def test_feature_level_skips_later_layers(self):
        model = self.make(synergy_feature_level='l6')
        with torch.no_grad():
            output = model(self.x)
        self.assertEqual(len(output.hyper_outputs), 6)
        self.assertEqual(model.synergy.channels, 256)
        configure_trainable_parameters(model, 'l9')
        trainable = {name.split('.')[0] for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {'synergy'})


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
