"""A3-G (plan KG_RETRIEVAL_PLAN/A3G_PREDICATE_HYPEREDGE/A3G_PLAN.md §6): predicates on synthetic skeletons, hyperedges,
the temporal adjacency path of HYPERGC, every mode starts at K0, controls, gradients."""

import math
import unittest

import torch
import torch.nn.functional as F

from model.checkpoint import load_model_weights
from model.geo_predicates import (HYPEREDGE_NAMES, JOINTS, PREDICATE_NAMES, GeoPredicates, _incidence_table,
                                  random_incidence)
from model.hypergcn_a3g import GRAPH_MODES, MODES, PASS2, Model
from model.hypergcn_large import Model as HyperGCN
from model.training import configure_trainable_parameters

BACKBONE = dict(num_class=120, num_point=25, num_person=2, hyper_joints=3,
                graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))
T = 64
P = {name: k for k, name in enumerate(PREDICATE_NAMES)}
E = {name: e for e, name in enumerate(HYPEREDGE_NAMES)}
WIDE = dict(q5=[-3.0] * len(PREDICATE_NAMES), q95=[3.0] * len(PREDICATE_NAMES))    # no clipping on the test bodies

# Standing person in camera space (metres, y up); right = shoulder_r - shoulder_l = -x, so forward = +z.
STANDING = {
    'sb': (0, 1.0, 3), 'sm': (0, 1.25, 3), 'nk': (0, 1.55, 3), 'hd': (0, 1.7, 3), 'ss': (0, 1.5, 3),
    'shoulder_l': (0.2, 1.5, 3), 'elbow_l': (0.25, 1.25, 3), 'wrist_l': (0.28, 1.0, 3), 'hand_l': (0.28, 0.95, 3),
    'tip_l': (0.28, 0.88, 3), 'thumb_l': (0.25, 0.93, 3),
    'shoulder_r': (-0.2, 1.5, 3), 'elbow_r': (-0.25, 1.25, 3), 'wrist_r': (-0.28, 1.0, 3), 'hand_r': (-0.28, 0.95, 3),
    'tip_r': (-0.28, 0.88, 3), 'thumb_r': (-0.25, 0.93, 3),
    'hip_l': (0.1, 0.95, 3), 'knee_l': (0.1, 0.5, 3), 'ankle_l': (0.1, 0.08, 3), 'foot_l': (0.1, 0.02, 3.1),
    'hip_r': (-0.1, 0.95, 3), 'knee_r': (-0.1, 0.5, 3), 'ankle_r': (-0.1, 0.08, 3), 'foot_r': (-0.1, 0.02, 3.1),
}
HAND_CHAIN = ('wrist', 'hand', 'tip', 'thumb')


def body(changes=None, moving='r'):
    """[T, 25, 3]: the standing person with joints moved by `changes`, and the hand chain of `moving` swinging."""
    joints = dict(STANDING, **(changes or {}))
    X = torch.tensor([joints[name] for name in sorted(JOINTS, key=JOINTS.get)], dtype=torch.float32)
    X = X.unsqueeze(0).repeat(T, 1, 1)
    swing = 0.03 * torch.sin(torch.arange(T) * 2 * math.pi / 16)
    for part in HAND_CHAIN:
        X[:, JOINTS[f'{part}_{moving}'], 0] += swing
    return X


def feeder(*actors):
    """Absolute bodies [T, 25, 3] (one or two) -> one feeder sample [1, 3, T, 25, 2] (relative to joint 20)."""
    x = torch.zeros(1, 3, T, 25, 2)
    for m, X in enumerate(actors):
        rel = X - X[:, JOINTS['ss']:JOINTS['ss'] + 1]
        rel[:, JOINTS['ss']] = X[:, JOINTS['ss']]
        x[0, :, :, :, m] = rel.permute(2, 0, 1)
    return x


def rotate(X, angle=0.3):
    c, s = math.cos(angle), math.sin(angle)
    Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
    return X @ (Rx @ Ry).T + torch.tensor([0.3, -0.1, 0.5])


class PredicateTests(unittest.TestCase):
    def raw(self, *actors):
        return GeoPredicates().raw(feeder(*actors))

    def test_hand_at_mouth_and_active_hand(self):
        mouth = {'wrist_r': (-0.05, 1.55, 3.05), 'hand_r': (-0.02, 1.62, 3.05), 'tip_r': (0, 1.66, 3.07),
                 'thumb_r': (-0.03, 1.6, 3.08), 'elbow_r': (-0.25, 1.35, 3.1)}
        at_mouth, _, active = self.raw(body(mouth))
        neutral, _, _ = self.raw(body())
        self.assertEqual(active[0, 0].item(), 1)                                    # the right hand moves
        self.assertGreater(at_mouth[0, 0, P['near_mouth_a']].mean(), neutral[0, 0, P['near_mouth_a']].mean() + 1)

    def test_left_active_hand_moves_the_hyperedges(self):
        geo = GeoPredicates()
        _, _, active = geo.raw(feeder(body(moving='l')))
        self.assertEqual(active[0, 0].item(), 0)
        B = geo.incidence(active)[0, 0]
        self.assertEqual(B[E['mouth_a'], JOINTS['hand_l']].item(), 1)
        self.assertEqual(B[E['mouth_a'], JOINTS['hand_r']].item(), 0)
        self.assertEqual(B[E['mouth_o'], JOINTS['hand_r']].item(), 1)
        torch.testing.assert_close(B, _incidence_table(0))
        self.assertTrue(torch.equal(B.sum(1), _incidence_table(1).sum(1)))           # same sizes for both hands

    def test_thumb_up_and_down_split_into_two_hyperedges(self):
        geo = GeoPredicates(WIDE)
        for dy, on, off in ((0.05, 'thumb_up_a', 'thumb_down_a'), (-0.05, 'thumb_down_a', 'thumb_up_a')):
            p, _ = geo(feeder(body({'thumb_r': (-0.28, 0.95 + dy, 3)})))
            self.assertGreater(p[0, 0, E[on]].mean().item(), 0)
            self.assertEqual(p[0, 0, E[off]].abs().sum().item(), 0)

    def test_sitting_raises_pelvis_low(self):
        seated = {name: (x, y - 0.4, z) for name, (x, y, z) in STANDING.items()
                  if not name.startswith(('knee', 'ankle', 'foot'))}
        seated.update(knee_l=(0.1, 0.55, 3.4), knee_r=(-0.1, 0.55, 3.4))
        sitting, _, _ = self.raw(body(seated))
        standing, _, _ = self.raw(body())
        self.assertGreater(sitting[0, 0, P['pelvis_low']].mean(), standing[0, 0, P['pelvis_low']].mean() + 0.5)

    def test_rotation_and_translation_invariance(self):
        X = body({'thumb_r': (-0.25, 1.0, 3)})
        raw, _, _ = self.raw(X)
        turned, _, _ = self.raw(rotate(X))
        self.assertLess((raw - turned).abs().max().item(), 1e-4)

    def test_empty_and_copied_actors_give_no_evidence(self):
        geo = GeoPredicates(WIDE)
        X = body()
        for second in (torch.zeros_like(X), X.clone()):
            p, _ = geo(feeder(X, second))
            _, valid, _ = geo.raw(feeder(X, second))
            self.assertEqual(valid[0].tolist(), [True, False])
            self.assertEqual(p[0, 1].abs().sum().item(), 0)
            self.assertGreater(p[0, 0].abs().sum().item(), 0)

    def test_random_incidence(self):
        real = _incidence_table(1)
        random = random_incidence(0)
        torch.testing.assert_close(random, random_incidence(0))                      # rebuilt identically
        torch.testing.assert_close(random.sum(1), real.sum(1))
        self.assertFalse(torch.equal(random, real))
        self.assertFalse(torch.equal(random, random_incidence(1)))

    def test_stale_stats_are_refused(self):
        with self.assertRaises(ValueError):
            GeoPredicates(dict(WIDE, names=list(reversed(PREDICATE_NAMES))))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.backbone = HyperGCN(**BACKBONE).eval()
        cls.x = torch.randn(3, 3, T, 25, 2)
        cls.x[2, ..., 1] = 0                                   # one single-actor sample
        cls.labels = torch.tensor([0, 5, 119])

    def loaded(self, mode='geo'):
        model = Model(a3g_args=dict(mode=mode, stats=WIDE), **BACKBONE)
        load_model_weights(model, self.backbone.state_dict(), allow_new=True)
        return model

    def test_temporal_adjacency_constant_in_time_equals_extra_adjacency(self):
        layer = self.backbone.l8
        x = torch.randn(2, layer.gcn1.in_channels, 16, 25)
        G = torch.rand(2, 1, 28, 28)
        with torch.no_grad():
            torch.testing.assert_close(layer(x, temporal_adjacency=None)[0], layer(x)[0])
            torch.testing.assert_close(layer(x, temporal_adjacency=G[:, 0, None].expand(-1, 16, -1, -1))[0],
                                       layer(x, extra_adjacency=G)[0])

    def test_every_mode_starts_at_k0(self):
        with torch.no_grad():
            k0, _ = self.backbone(self.x)
            for mode in MODES:
                model = self.loaded(mode).eval()
                out, hyper = model(self.x)
                torch.testing.assert_close(out, k0)
                self.assertEqual(len(hyper), 9)
                if mode in GRAPH_MODES:
                    torch.testing.assert_close(model.eval_outputs()['no_geo'], k0)
                else:
                    self.assertEqual(model.eval_outputs(), {})

    def test_pass2_copies_k0(self):
        model = self.loaded()
        for name in PASS2:
            for key, value in getattr(self.backbone, name).state_dict().items():
                torch.testing.assert_close(model.a3g.layers[name].state_dict()[key], value)

    def test_adjacency_shapes_virtual_rows_and_empty_actor(self):
        head = self.loaded().a3g
        p, B = head.hyperedges(self.x)
        self.assertEqual(tuple(p.shape), (6, len(HYPEREDGE_NAMES), T))
        self.assertEqual(tuple(B.shape), (6, len(HYPEREDGE_NAMES), 25))
        for steps in (32, 16):
            G = head.adjacency(p, B, steps)
            self.assertEqual(tuple(G.shape), (6, steps, 28, 28))
            self.assertEqual(G[:, :, 25:].abs().sum().item(), 0)
            self.assertEqual(G[:, :, :, 25:].abs().sum().item(), 0)
            self.assertEqual(G[5].abs().sum().item(), 0)                              # sample 2, actor 1 is empty

    def test_shuffle_and_random_controls(self):
        p, B = self.loaded('geo').a3g.hyperedges(self.x)
        p_shuffle, B_shuffle = self.loaded('shuffle').a3g.hyperedges(self.x)
        torch.testing.assert_close(p_shuffle.view(3, 2, *p.shape[1:])[0], p.view(3, 2, *p.shape[1:])[1])
        torch.testing.assert_close(B_shuffle, B)
        p_random, B_random = self.loaded('random').a3g.hyperedges(self.x)
        torch.testing.assert_close(p_random, p)
        torch.testing.assert_close(B_random, random_incidence(0).expand_as(B))

    def test_gradients(self):
        model = self.loaded('geo')
        configure_trainable_parameters(model, 'l9')
        with torch.no_grad():
            model.a3g.beta.fill_(0.1)                          # w_geo has no gradient while beta = 0
        model.train()
        logits, _ = model(self.x)
        F.cross_entropy(logits, self.labels).backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith('a3g.layers.l9')))
        self.assertTrue(any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith('fc')))
        self.assertGreater(grads['a3g.beta'].abs().sum().item(), 0)
        self.assertGreater(grads['a3g.w_geo'].abs().sum().item(), 0)
        self.assertFalse(any(n.startswith(('l1', 'l6', 'l7', 'l8', 'l9')) for n in grads))
        _, terms = model.extra_loss(self.labels)
        self.assertTrue({'beta_l7', 'beta_l8', 'beta_l9', 'w_geo'} <= set(terms))

    def test_beta_gets_a_gradient_at_zero_and_controls_train_their_own_part(self):
        for mode, name in (('geo', 'a3g.beta'), ('input', 'a3g.input_proj.weight')):
            model = self.loaded(mode)
            configure_trainable_parameters(model, 'l9')
            model.train()
            F.cross_entropy(model(self.x)[0], self.labels).backward()
            self.assertGreater(dict(model.named_parameters())[name].grad.abs().sum().item(), 0)
        model = self.loaded('off')
        configure_trainable_parameters(model, 'l9')
        model.train()
        F.cross_entropy(model(self.x)[0], self.labels).backward()
        self.assertIsNone(model.a3g.beta.grad)


if __name__ == '__main__':
    unittest.main()
