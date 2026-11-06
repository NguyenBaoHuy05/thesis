import unittest

import torch

from model import training
from model.checkpoint import load_model_weights
from model.hypergcn_large import Model
from model.training import configure_trainable_parameters, frozen_modules

MODEL_ARGS = dict(num_class=120, num_point=25, num_person=2, hyper_joints=3,
                  graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = Model(**MODEL_ARGS).eval()
        cls.x = torch.randn(2, 3, 64, 25, 2)

    def test_forward(self):
        with torch.no_grad():
            logits, hyper = self.model(self.x)
        self.assertEqual(logits.shape, (2, 120))
        self.assertEqual(len(hyper), 9)

    def test_features_stop_at_the_requested_layer(self):
        with torch.no_grad():
            features, hyper = self.model.features(self.x, until='l6')
        self.assertEqual(features.shape[:3], (2, 2, 256))
        self.assertEqual(features.shape[-1], 25)
        self.assertEqual(len(hyper), 6)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Model(**MODEL_ARGS)

    def test_round_trip_with_module_prefix(self):
        state = {'module.' + k: v for k, v in self.model.state_dict().items()}
        report = load_model_weights(Model(**MODEL_ARGS), state)
        self.assertEqual(report['initialized'], [])

    def test_missing_backbone_key_is_rejected(self):
        state = dict(self.model.state_dict())
        state.pop('fc.weight')
        with self.assertRaises(ValueError):
            load_model_weights(Model(**MODEL_ARGS), state, allow_new=True)

    def test_missing_extension_key_needs_allow_new(self):
        model = Model(**MODEL_ARGS)
        model.kg = torch.nn.Linear(4, 4)
        state = self.model.state_dict()
        original = training.EXTENSION_PREFIXES
        training.EXTENSION_PREFIXES = ('kg.',)
        try:
            with self.assertRaises(ValueError):
                load_model_weights(model, state)
            report = load_model_weights(model, state, allow_new=True)
        finally:
            training.EXTENSION_PREFIXES = original
        self.assertEqual(sorted(report['initialized']), ['kg.bias', 'kg.weight'])


class FreezeTests(unittest.TestCase):
    def test_frozen_modules(self):
        self.assertEqual(frozen_modules(None), ())
        self.assertEqual(frozen_modules('l2')[-2:], ('l1', 'l2'))
        with self.assertRaises(ValueError):
            frozen_modules('l10')

    def test_freeze_l9_leaves_only_fc(self):
        model = Model(**MODEL_ARGS)
        configure_trainable_parameters(model, 'l9')
        trainable = {name.split('.')[0] for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {'fc'})


if __name__ == '__main__':
    unittest.main()
