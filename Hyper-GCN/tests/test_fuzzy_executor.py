"""model/fuzzy_executor.py, model/concept_graph_head.py, model/reason_logic.py, model/pair_metrics.py."""

import os
import unittest

import torch

from model.concept_graph import ConceptGraph, build_graph, flat_graph, read_matrix
from model.concept_graph_head import ConceptGraphHead
from model.concept_logic import logic_scores
from model.fuzzy_executor import FuzzyGraphExecutor, log1mexp
from model.pair_metrics import k5_pair, pair_metrics
from model.reason_logic import ReasonLogicHead

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv')
LO, HI = 0.01, 0.99


def toy_graph():
    """concepts 0..5; action 0 = AND(m0 = {0, 1}) & OR{4, 5} & c2; action 1 = m0 & OR{4, 5} & c3;
    pair (0, 1): NOT(1|0) reads {3}, NOT(0|1) reads {2}."""
    return ConceptGraph(actions=['a0', 'a1'], concepts=[f'c{k}' for k in range(6)],
                        rows=[[0, 1, 2, 4], [0, 1, 3, 4]], motifs=[[0, 1]], ors=[[4, 5]],
                        action_motifs=[[0], [0]], action_ors=[[0], [0]], action_direct=[[2], [3]],
                        pairs=[(0, 1)], nots=[(0, 1, [3]), (1, 0, [2])], cfg={})


class FuzzyExecutorTests(unittest.TestCase):
    def test_log1mexp(self):
        x = torch.tensor([-1e-3, -0.5, -0.7, -5.0, -30.0])
        torch.testing.assert_close(log1mexp(x), torch.log(1 - torch.exp(x)), rtol=1e-4, atol=1e-6)

    def test_flat_graph_without_nots_is_cl_v2(self):
        actions, concepts, rows = read_matrix(MATRIX)
        graph = flat_graph(actions, concepts, rows, nots=False)
        ex = FuzzyGraphExecutor(graph, omega=1.0)
        c = torch.rand(7, len(concepts)) * 0.98 + 0.01
        M = ex.mask_row
        torch.testing.assert_close(ex(c), logic_scores(c, M, 1.0, balance=True), rtol=1e-5, atol=1e-5)

    def test_truth_table(self):
        ex = FuzzyGraphExecutor(toy_graph(), omega=1.0, nu=1.0)
        c = torch.full((1, 6), LO)
        c[0, [0, 1, 2, 5]] = HI                      # a0's rule holds through the OR alternative c5
        s = ex(c)
        self.assertGreater(s[0, 0], s[0, 1])
        self.assertGreater(s[0, 0].exp(), 0.9)
        c_bad = c.clone()
        c_bad[0, 3] = HI                             # a1's distinguishing concept on: NOT(1|0) is false
        self.assertLess(ex(c_bad)[0, 0], s[0, 0] - 0.5)
        c_or_off = c.clone()
        c_or_off[0, 5] = LO                          # no posture alternative left
        self.assertLess(ex(c_or_off)[0, 0], s[0, 0] - 0.3)
        v = ex.nodes(c)
        self.assertGreater(v['motif'][0, 0].exp(), 0.98)
        self.assertGreater(v['orr'][0, 0].exp(), 0.98)
        self.assertGreater(v['notv'][0, 0].exp(), 0.98)          # NOT(1|0): c3 low
        self.assertLess(v['notv'][0, 1].exp(), 0.02)             # NOT(0|1): c2 high

    def test_or_alternative_is_not_absent(self):
        ex = FuzzyGraphExecutor(toy_graph())
        self.assertEqual(ex.mask_absent[0, 5].item(), 0.0)
        self.assertEqual(ex.mask_absent[0, 3].item(), 1.0)

    def test_alpha_gradients_and_penalty(self):
        ex = FuzzyGraphExecutor(toy_graph(), learn_alpha=True)
        self.assertAlmostEqual(ex.alpha_penalty().item(), 0.0, places=5)
        c = torch.rand(3, 6, requires_grad=True)
        ex(c.clamp(0.05, 0.95)).sum().backward()
        self.assertIsNotNone(c.grad)
        self.assertIsNotNone(ex.theta['act_not'].grad)
        trace = ex.trace(torch.full((6,), 0.5), 0)
        self.assertEqual(set(trace), {'direct', 'motif', 'orr', 'nots'})
        self.assertEqual(list(trace['nots']), [0])

    def test_real_graph_runs(self):
        actions, concepts, rows = read_matrix(MATRIX)
        ex = FuzzyGraphExecutor(build_graph(actions, concepts, rows))
        s = ex(torch.rand(4, len(concepts)) * 0.98 + 0.01)
        self.assertEqual(tuple(s.shape), (4, 120))
        self.assertTrue(torch.isfinite(s).all() and (s <= 0).all())


class HeadTests(unittest.TestCase):
    def setUp(self):
        self.graph = toy_graph()
        self.mask = torch.ones(6, 25, dtype=torch.bool)
        self.args = dict(d_model=16, heads=2, layers=1, max_frames=2, num_person=2)
        self.feats, self.valid = torch.randn(5, 2, 8, 2, 25), torch.tensor([[True, False]] * 5)

    def test_graph_head(self):
        head = ConceptGraphHead(8, self.graph, self.mask, decoder_args=self.args, learn_alpha=True, eta=1e-4)
        out = head(self.feats, self.valid)
        self.assertEqual(tuple(out['logits'].shape), (5, 2))
        (out['logits'].sum() + head.regularizer()).backward()
        self.assertIsNotNone(head.decoder.query.weight.grad)

    def test_reason_head(self):
        M = torch.tensor([[1., 1, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0]])
        head = ReasonLogicHead(8, M, self.mask, decoder_args=self.args, logic_args=dict(l1_dim=8, l2_dim=8))
        out = head(self.feats, self.valid)
        self.assertEqual(tuple(out['logits'].shape), (5, 2))
        self.assertGreater(head.decoder.out.weight.abs().sum().item(), 0.0)      # not the zero init
        (out['logits'].sum() + head.regularizer()).backward()
        self.assertIsNotNone(head.logic.union1.con_layer.W.grad)
        head.logic.union1.con_layer.W.data += 2
        head.after_step()
        self.assertLessEqual(head.logic.union1.con_layer.W.max().item(), 1.0)


class PairMetricTests(unittest.TestCase):
    def test_pcr(self):
        actions = ['a', 'b', 'c']
        labels = torch.tensor([0, 0, 1, 1, 2])
        pred = torch.tensor([1, 0, 1, 2, 2])
        m = pair_metrics(pred, labels, [(0, 1)], actions)
        self.assertAlmostEqual(m['pcr'], 0.25)                   # 1 of the 4 samples of a / b went to its partner
        self.assertEqual(m['pcr_samples'], 4)
        self.assertAlmostEqual(m['top1_outside_pairs'], 1.0)
        self.assertAlmostEqual(m['per_pair'][0]['rate'], 0.25)

    def test_k5_pair_moves_to_partner(self):
        ex = FuzzyGraphExecutor(toy_graph())
        c = torch.full((1, 6), LO)
        c[0, [0, 1, 2, 4]] = HI                                  # predicted a0
        rows = ex.mask_row
        out = k5_pair(lambda x: ex(x), c, rows, [(0, 1)])
        self.assertEqual(out['samples'], 1)
        self.assertGreaterEqual(out['moved_to_partner'], 0.0)


if __name__ == '__main__':
    unittest.main()
