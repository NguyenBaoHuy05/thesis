import unittest

import torch

from model.hypergcn_large import Model
from tools.cache_features import fc_logits_from_features, pooled_z, run_pass, valid_actor_mask

MODEL_ARGS = dict(num_class=120, num_point=25, num_person=2, hyper_joints=3,
                  graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))


class FcLogitsReplicationTests(unittest.TestCase):
    """fc_logits must come from the single features() pass, not a second forward()."""

    def test_matches_full_forward(self):
        torch.manual_seed(0)
        model = Model(**MODEL_ARGS).eval()
        x = torch.randn(3, 3, 64, 25, 2)
        with torch.no_grad():
            expected, _ = model(x)
            features, _ = model.features(x, until='l9')
            actual = fc_logits_from_features(model, features)
        torch.testing.assert_close(actual, expected)


class ValidActorMaskTests(unittest.TestCase):
    def test_second_actor_all_zero_is_invalid(self):
        x = torch.randn(2, 3, 64, 25, 2)
        x[:, :, :, :, 1] = 0
        valid = valid_actor_mask(x)
        self.assertEqual(valid.tolist(), [[True, False], [True, False]])

    def test_both_actors_present(self):
        x = torch.randn(2, 3, 64, 25, 2)
        valid = valid_actor_mask(x)
        self.assertTrue(valid.all())

    def test_second_actor_copy_of_first_is_invalid(self):
        # align_frames pads single-person samples with a copy of actor 1.
        x = torch.randn(2, 3, 64, 25, 2)
        x[0, :, :, :, 1] = x[0, :, :, :, 0]
        valid = valid_actor_mask(x)
        self.assertEqual(valid.tolist(), [[True, False], [True, True]])


class PooledZTests(unittest.TestCase):
    def test_channelwise_max_over_valid_actors_only(self):
        # [N=1, M=2, C=2, T=1, V=1]: actor 0 wins channel 0, actor 1 wins channel 1.
        features = torch.tensor([[[[[1.0]], [[5.0]]], [[[9.0]], [[2.0]]]]])
        valid = torch.tensor([[True, True]])
        z = pooled_z(features, valid)
        self.assertEqual(z.tolist(), [[9.0, 5.0]])

    def test_invalid_actor_is_ignored_even_if_larger(self):
        features = torch.tensor([[[[[1.0]], [[5.0]]], [[[99.0]], [[99.0]]]]])
        valid = torch.tensor([[True, False]])
        z = pooled_z(features, valid)
        self.assertEqual(z.tolist(), [[1.0, 5.0]])

    def test_no_valid_actor_raises(self):
        features = torch.zeros(1, 2, 2, 1, 1)
        valid = torch.tensor([[False, False]])
        with self.assertRaises(ValueError):
            pooled_z(features, valid)


class EarlierLayerTests(unittest.TestCase):
    def test_hooked_layer_equals_a_pass_stopped_at_that_layer(self):
        torch.manual_seed(0)
        model = Model(**MODEL_ARGS).eval()
        x = torch.randn(3, 3, 64, 25, 2)
        loader = [(x, torch.tensor([0, 1, 2]), torch.tensor([10, 11, 12]))]
        zs, fc_logits, labels, names = run_pass(model, loader, 'cpu', layers=('l6', 'l9'))
        valid = valid_actor_mask(x)
        with torch.no_grad():
            expected_l6 = pooled_z(model.features(x, until='l6')[0], valid)
            expected_l9 = pooled_z(model.features(x, until='l9')[0], valid)
        self.assertEqual(zs['l6'].shape, (3, 256))
        torch.testing.assert_close(torch.from_numpy(zs['l6']), expected_l6)
        torch.testing.assert_close(torch.from_numpy(zs['l9']), expected_l9)
        self.assertEqual(names, [10, 11, 12])
        # Hooks are removed after the pass.
        self.assertFalse(model.l6._forward_hooks)


if __name__ == '__main__':
    unittest.main()
