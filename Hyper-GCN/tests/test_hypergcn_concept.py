"""model/hypergcn_concept.py: the refinement starts at K0, loads checkpoints and head.pt, trains l9 + graph."""

import os
import tempfile
import unittest

import torch

from model.checkpoint import load_model_weights
from model.hypergcn_concept import Model, segment_mean
from model.hypergcn_large import Model as HyperGCN
from model.training import configure_trainable_parameters, frozen_modules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKBONE = dict(num_class=120, num_point=25, num_person=2, hyper_joints=3,
                graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))
CONCEPT = dict(matrix=os.path.join(ROOT, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv'),
               metadata=os.path.join(ROOT, 'annotations', 'concepts', 'ntu_concept_metadata_v2_4.csv'),
               head_args=dict(decoder_args=dict(d_model=16, heads=2, layers=1), rgcn_args=dict(bases=2)))


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.backbone = HyperGCN(**BACKBONE).eval()
        cls.model = Model(concept_args=CONCEPT, **BACKBONE)
        load_model_weights(cls.model, cls.backbone.state_dict(), allow_new=True)
        cls.x = torch.randn(3, 3, 64, 25, 2)
        cls.x[2, ..., 1] = 0                                   # one single-actor sample
        cls.labels = torch.tensor([0, 5, 119])

    def test_starts_at_k0(self):
        self.model.eval()
        with torch.no_grad():
            refined, hyper = self.model(self.x)
            k0, _ = self.backbone(self.x)
        self.assertEqual(len(hyper), 9)
        torch.testing.assert_close(refined, k0)
        torch.testing.assert_close(self.model.eval_outputs()['k0'], k0)
        self.assertEqual(tuple(self.model.eval_outputs()['head'].shape), (3, 120))

    def test_gradient_reaches_l9_graph_and_refinement(self):
        model = Model(concept_args=CONCEPT, **BACKBONE)
        torch.nn.init.normal_(model.cg.head.readout.weight, std=0.1)   # a trained head.pt has z != 0
        configure_trainable_parameters(model, 'l8')
        model.train()
        logits, _ = model(self.x)
        extra, terms = model.extra_loss(self.labels)
        (torch.nn.functional.cross_entropy(logits, self.labels) + extra).backward()
        for name in ('l9', 'cg.proj', 'cg.head.decoder', 'cg.head.rgcn', 'fc'):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(name) and p.grad is not None]
            self.assertTrue(any(g.abs().sum() > 0 for g in grads), name)
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n.startswith('l8')))
        self.assertTrue({'head_ce', 'head_bce', 'head_aux', 'head_kl', 'head_margin'} <= set(terms))

    def test_head_weights_are_loaded(self):
        source = Model(concept_args=CONCEPT, **BACKBONE)
        with torch.no_grad():
            source.cg.head.aux.bias.fill_(0.25)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'head.pt')
            torch.save(source.cg.head.state_dict(), path)
            model = Model(concept_args=CONCEPT, head_weights=path, **BACKBONE)
        for key, value in source.cg.head.state_dict().items():
            torch.testing.assert_close(model.cg.head.state_dict()[key], value)

    def test_checkpoint_load_keeps_the_graph(self):
        # main.py loads the Hyper-GCN checkpoint after head.pt: the cg.* keys must survive it.
        model = Model(concept_args=CONCEPT, **BACKBONE)
        before = {k: v.clone() for k, v in model.cg.state_dict().items()}
        report = load_model_weights(model, self.backbone.state_dict(), allow_new=True)
        self.assertTrue(report['initialized'] and all(k.startswith('cg.') for k in report['initialized']))
        for key, value in model.cg.state_dict().items():
            torch.testing.assert_close(value, before[key])
        with self.assertRaises(ValueError):
            load_model_weights(Model(concept_args=CONCEPT, **BACKBONE), self.backbone.state_dict())

    def test_every_frame_online_with_frozen_backbone_and_fc(self):
        # online_e3.yaml: 16 l9 frames, backbone and fc frozen, the graph from scratch.
        model = Model(concept_args=dict(CONCEPT, segments=16), **BACKBONE)
        load_model_weights(model, self.backbone.state_dict(), allow_new=True)
        torch.nn.init.normal_(model.cg.proj.weight, std=0.01)
        configure_trainable_parameters(model, 'fc')
        self.assertIn('fc', frozen_modules('fc'))
        model.train()
        logits, _ = model(self.x)
        extra, _ = model.extra_loss(self.labels)
        (torch.nn.functional.cross_entropy(logits, self.labels) + extra).backward()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.startswith('cg.')))
        self.assertTrue(model.cg.proj.weight.grad.abs().sum() > 0)
        self.assertEqual(len(model.cg.head.decoder.time_emb), 16)

    def test_segment_mean(self):
        features = torch.arange(8.).view(1, 1, 1, 8, 1)
        self.assertEqual(segment_mean(features, 2).flatten().tolist(), [1.5, 5.5])
        self.assertIs(segment_mean(features, 8), features)


if __name__ == '__main__':
    unittest.main()
