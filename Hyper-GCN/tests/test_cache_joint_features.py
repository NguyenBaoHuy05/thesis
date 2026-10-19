import unittest

import torch

from model.hypergcn_large import Model
from model.actors import valid_actor_mask
from tools.cache_joint_features import fc_logits_from_features, segment_mean

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


class SegmentMeanTests(unittest.TestCase):
    def test_one_segment_is_the_time_mean(self):
        features = torch.randn(2, 2, 4, 12, 25)
        torch.testing.assert_close(segment_mean(features, 1)[:, :, :, 0], features.mean(3))

    def test_equal_segments(self):
        features = torch.arange(6.).view(1, 1, 1, 6, 1)
        self.assertEqual(segment_mean(features, 3).flatten().tolist(), [0.5, 2.5, 4.5])


if __name__ == '__main__':
    unittest.main()
