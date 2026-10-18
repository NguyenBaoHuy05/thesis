import unittest

import torch

from model.attribute_decoder import AttributeDecoder
from model.kg import NTU_BODY_PARTS, attribute_joint_mask

N, M, C, T, V, A = 3, 2, 8, 4, 25, 3


def make_decoder(**kwargs):
    torch.manual_seed(0)
    mask = torch.zeros(A, V, dtype=torch.bool)
    mask[0, [3]] = True             # attribute 0 sees the head joint only
    mask[1, [10, 11]] = True        # attribute 1 the right wrist / hand
    mask[2] = True                  # attribute 2 every joint
    decoder = AttributeDecoder(C, torch.randn(A, 6), mask, d_model=8, heads=2, layers=2,
                               dropout=0.0, max_frames=T, num_person=M, **kwargs)
    nn_init_output(decoder)
    return decoder.eval()


def nn_init_output(decoder):
    """The output layer starts at zero (a = 0.5); give it weights so a depends on the input."""
    torch.nn.init.normal_(decoder.out.weight)


class AttributeDecoderTests(unittest.TestCase):
    def setUp(self):
        self.decoder = make_decoder()
        self.features = torch.randn(N, M, C, T, V)
        self.valid = torch.tensor([[True, True], [True, False], [True, False]])

    def test_starts_at_one_half(self):
        decoder = AttributeDecoder(C, torch.randn(A, 6), torch.ones(A, V, dtype=torch.bool),
                                   d_model=8, heads=2, max_frames=T)
        a = decoder(self.features, self.valid)
        self.assertEqual(tuple(a.shape), (N, A))
        torch.testing.assert_close(a, torch.full((N, A), 0.5))

    def test_attribute_ignores_joints_outside_its_body_parts(self):
        a = self.decoder(self.features, self.valid)
        changed = self.features.clone()
        changed[..., 20:] += 5.0                 # joints 20-24: outside attributes 0 and 1
        b = self.decoder(changed, self.valid)
        torch.testing.assert_close(b[:, :2], a[:, :2])
        self.assertFalse(torch.allclose(b[:, 2], a[:, 2]))

    def test_invalid_actor_is_ignored(self):
        a = self.decoder(self.features, self.valid)
        changed = self.features.clone()
        changed[:, 1] += 5.0                     # second actor: invalid for samples 1 and 2
        b = self.decoder(changed, self.valid)
        torch.testing.assert_close(b[1:], a[1:])
        self.assertFalse(torch.allclose(b[0], a[0]))

    def test_temporal_order_matters(self):
        a = self.decoder(self.features, self.valid)
        b = self.decoder(self.features.flip(3), self.valid)
        self.assertFalse(torch.allclose(a, b))

    def test_gradient_reaches_features_and_query(self):
        decoder = make_decoder().train()
        features = self.features.clone().requires_grad_(True)
        decoder(features, self.valid).sum().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertGreater(decoder.query.weight.grad.abs().sum().item(), 0)

    def test_rejects_empty_mask_row_and_too_many_frames(self):
        mask = torch.ones(A, V, dtype=torch.bool)
        mask[1] = False
        with self.assertRaises(ValueError):
            AttributeDecoder(C, torch.randn(A, 6), mask, d_model=8, heads=2)
        with self.assertRaises(ValueError):
            self.decoder(torch.randn(N, M, C, T + 1, V), self.valid)


class JointMaskTests(unittest.TestCase):
    KG = {'attributes': {
        'hand_at_mouth': {'body_parts': ['hand_R', 'head']},
        'no_parts': {},
        'walks': {'body_parts': ['feet']},
    }}

    def test_parts_map_to_joints_and_mirror(self):
        mask = attribute_joint_mask(self.KG)
        expected = set(NTU_BODY_PARTS['hand_R'] + NTU_BODY_PARTS['hand_L'] + NTU_BODY_PARTS['head'])
        self.assertEqual(set(mask[0].nonzero().flatten().tolist()), expected)
        self.assertTrue(mask[1].all())
        self.assertEqual(set(mask[2].nonzero().flatten().tolist()), set(NTU_BODY_PARTS['feet']))

    def test_no_mirror_keeps_one_side(self):
        mask = attribute_joint_mask(self.KG, mirror=False)
        self.assertFalse(mask[0, list(NTU_BODY_PARTS['hand_L'][1:])].any())   # 6 is also arm_L
        self.assertTrue(mask[0, list(NTU_BODY_PARTS['hand_R'])].all())

    def test_unknown_part_raises(self):
        with self.assertRaises(ValueError):
            attribute_joint_mask({'attributes': {'x': {'body_parts': ['tail']}}})

    def test_real_kg_parts_are_all_known(self):
        from model.kg import load_kg
        mask = attribute_joint_mask(load_kg('annotations/action_kg.json'))
        self.assertTrue(mask.any(1).all())


if __name__ == '__main__':
    unittest.main()
