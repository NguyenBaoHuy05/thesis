import unittest

import torch

from model.attribute_decoder import AttributeDecoder

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


if __name__ == '__main__':
    unittest.main()
