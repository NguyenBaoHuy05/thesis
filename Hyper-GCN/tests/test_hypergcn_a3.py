"""model/hypergcn_a3.py: A3 starts at K0, copies l7..l9 into pass 2, concept hyperedges, controls, gradients."""

import os
import unittest

import torch
import torch.nn.functional as F

from model.checkpoint import load_model_weights
from model.hypergcn_a3 import PASS2, Model, hyper_norm, random_joint_sets
from model.hypergcn_large import HYPERGC, Model as HyperGCN
from model.training import configure_trainable_parameters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKBONE = dict(num_class=120, num_point=25, num_person=2, hyper_joints=3,
                graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))
A3 = dict(matrix=os.path.join(ROOT, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv'),
          metadata=os.path.join(ROOT, 'annotations', 'concepts', 'ntu_concept_metadata_v2_4.csv'))


def a3_model(**args):
    return Model(a3_args=dict(A3, **args), **BACKBONE)


class A3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.backbone = HyperGCN(**BACKBONE).eval()
        cls.x = torch.randn(3, 3, 64, 25, 2)
        cls.x[2, ..., 1] = 0                                   # one single-actor sample
        cls.labels = torch.tensor([0, 5, 119])

    def loaded(self, **args):
        model = a3_model(**args)
        load_model_weights(model, self.backbone.state_dict(), allow_new=True)
        return model

    def test_hyper_norm_matches_hypergc(self):
        H, W = torch.rand(2, 1, 28, 74), torch.rand(2, 1, 74)
        torch.testing.assert_close(hyper_norm(H, W), HYPERGC.hyper_norm(None, H, W))

    def test_both_stages_start_at_k0(self):
        with torch.no_grad():
            k0, _ = self.backbone(self.x)
            for stage in ('concept', 'a3'):
                model = self.loaded(stage=stage).eval()
                out, hyper = model(self.x)
                torch.testing.assert_close(out, k0)
                self.assertEqual(len(hyper), 9)
                self.assertEqual(tuple(model.eval_outputs()['concept_match'].shape), (3, 120))
            torch.testing.assert_close(model.eval_outputs()['no_kg'], k0)

    def test_pass2_copies_k0_and_keeps_a_stage1_checkpoint(self):
        model = self.loaded()
        for name in PASS2:
            for key, value in getattr(self.backbone, name).state_dict().items():
                torch.testing.assert_close(model.a3.layers[name].state_dict()[key], value)
        # An S1 checkpoint already has a3.layers: loading it must not overwrite them with l7..l9.
        state = model.state_dict()
        state['a3.layers.l9.gcn1.alpha'] = torch.full_like(state['a3.layers.l9.gcn1.alpha'], 3.0)
        again = a3_model()
        load_model_weights(again, state)
        self.assertEqual(again.a3.layers['l9'].gcn1.alpha.item(), 3.0)
        self.assertNotEqual(again.l9.gcn1.alpha.item(), 3.0)

    def test_concept_hyperedges(self):
        head = a3_model().a3
        self.assertEqual(len(head.spatial), 74)
        self.assertEqual(tuple(head.B.shape), (74, 25))
        c = torch.rand(2, 102)
        G = head.adjacency(c)
        self.assertEqual(tuple(G.shape), (2, 1, 28, 28))
        self.assertEqual(G[:, :, 25:].abs().sum().item(), 0)          # virtual joints are in no concept hyperedge
        self.assertEqual(G[:, :, :, 25:].abs().sum().item(), 0)
        # Joints of no active concept get nothing: with c = 0 except one concept, G lives on that concept's joints.
        k = 0
        c = torch.zeros(1, 102)
        c[0, head.spatial[k]] = 1
        joints = head.B[k].bool()
        G = head.adjacency(c)[0, 0, :25, :25]
        self.assertGreater(G[joints][:, joints].sum().item(), 0)
        self.assertEqual(G[~joints].abs().sum().item(), 0)

    def test_random_mask_control(self):
        concept, random = a3_model().a3.B, a3_model(mask='random', mask_seed=0).a3.B
        torch.testing.assert_close(concept.sum(1), random.sum(1))
        self.assertFalse(torch.equal(concept, random))
        torch.testing.assert_close(random, a3_model(mask='random', mask_seed=0).a3.B)   # rebuilt identically
        self.assertFalse(torch.equal(random, random_joint_sets(concept.bool(), 1).float()))

    def test_gradients_in_stage_a3(self):
        model = self.loaded()
        configure_trainable_parameters(model, 'l9')
        model.train()
        logits, _ = model(self.x)
        F.cross_entropy(logits, self.labels).backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith('a3.layers.l9')))
        self.assertTrue(any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith('fc')))
        self.assertGreater(grads['a3.beta'].abs().sum().item(), 0)    # beta = 0 still gets a gradient
        self.assertFalse(any(n.startswith(('l7', 'l8', 'l9', 'a3.concept')) for n in grads))   # c is detached
        model.zero_grad()
        model(self.x)
        extra, terms = model.extra_loss(self.labels)
        extra.backward()
        self.assertGreater(model.a3.concept.weight.grad.abs().sum().item(), 0)
        self.assertTrue({'concept_bce', 'beta_l7', 'w_kg'} <= set(terms))

    def test_k0ft_control_has_no_kg(self):
        model = self.loaded(use_kg=False)
        configure_trainable_parameters(model, 'l9')
        model.train()
        logits, _ = model(self.x)
        F.cross_entropy(logits, self.labels).backward()
        self.assertIsNone(model.a3.beta.grad)
        model.eval()
        with torch.no_grad():
            model(self.x)
        self.assertNotIn('no_kg', model.eval_outputs())

    def test_stage_concept_trains_only_the_concept_head(self):
        model = self.loaded(stage='concept')
        configure_trainable_parameters(model, 'fc')
        model.train()
        model(self.x)
        extra, _ = model.extra_loss(self.labels)
        extra.backward()
        trained = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        self.assertEqual(trained, {'a3.concept.weight', 'a3.concept.bias'})


if __name__ == '__main__':
    unittest.main()
