"""model/rgcn.py and model/concept_action_graph.py on the real v2.4 graph with small decoders."""

import os
import unittest

import torch

from model.concept_action_graph import ConceptActionGraphHead
from model.concept_graph import build_graph, read_matrix
from model.concept_logic import concept_loss
from model.rgcn import StratifiedRGCN, node_counts, relation_adjacency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv')
D = 16


def real_graph():
    actions, concepts, rows = read_matrix(MATRIX)
    return build_graph(actions, concepts, rows)


def random_states(graph, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {t: torch.randn(n, k, D, generator=g) for t, k in node_counts(graph).items()}


class RGCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = real_graph()

    def test_adjacency_rows_are_normalized(self):
        for r, A in relation_adjacency(self.graph).items():
            sums = A.sum(1)
            self.assertTrue(torch.all(((sums - 1).abs() < 1e-5) | (sums == 0)), r)

    def test_gates_do_not_read_actions(self):
        torch.manual_seed(0)
        rgcn = StratifiedRGCN(self.graph, d=D, dropout=0.0).eval()
        h = random_states(self.graph)
        h2 = dict(h, action=h['action'] + 1.0)
        out, out2 = rgcn(h), rgcn(h2)
        torch.testing.assert_close(out['motif'], out2['motif'])
        torch.testing.assert_close(out['orr'], out2['orr'])
        self.assertFalse(torch.allclose(out['notn'], out2['notn']))     # r6: action -> NOT
        self.assertEqual(tuple(out['action'].shape), (3, 120, D))

    def test_action_reads_its_concepts(self):
        torch.manual_seed(0)
        rgcn = StratifiedRGCN(self.graph, d=D, dropout=0.0).eval()
        h = random_states(self.graph)
        c = self.graph.action_direct[0][0]                  # a concept action 0 reads directly
        h2 = {k: v.clone() for k, v in h.items()}
        h2['concept'][:, c] += 1.0
        moved = (rgcn(h)['action'] - rgcn(h2)['action']).abs().sum((0, 2))
        self.assertGreater(moved[0].item(), 0.0)
        readers = {a for a in range(120) if c in set(self.graph.action_direct[a])
                   or any(c in self.graph.motifs[m] for m in self.graph.action_motifs[a])
                   or any(c in self.graph.ors[o] for o in self.graph.action_ors[a])
                   or any(c in d and i == a for i, _j, d in self.graph.nots)}
        for a in range(120):
            if a not in readers:
                self.assertAlmostEqual(moved[a].item(), 0.0, places=5)


class FullHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = real_graph()

    def make(self, **kwargs):
        torch.manual_seed(0)
        mask = torch.ones(102, 25, dtype=torch.bool)
        args = dict(d_model=D, heads=2, layers=1, max_frames=2, num_person=2, dropout=0.0)
        return ConceptActionGraphHead(8, self.graph, mask, decoder_args=args, rgcn_args=dict(dropout=0.0), **kwargs)

    def setUp(self):
        self.feats, self.valid = torch.randn(4, 2, 8, 2, 25), torch.tensor([[True, False]] * 4)
        self.labels = torch.tensor([0, 15, 16, 30])

    def test_shapes_and_logic_first_at_init(self):
        head = self.make()
        out = head(self.feats, self.valid)
        for k in ('c', 'logits', 'logits_sym', 'z', 'aux'):
            self.assertIn(k, out)
        self.assertEqual(tuple(out['logits'].shape), (4, 120))
        torch.testing.assert_close(out['logits'], out['logits_sym'])          # readout starts at zero
        torch.testing.assert_close(head.logits(out['c']), out['logits_sym'])

    def test_losses_reach_every_part(self):
        head = self.make()
        head.train()
        torch.nn.init.normal_(head.readout.weight, std=0.1)
        torch.nn.init.normal_(head.decoder.out.weight)
        out = head(self.feats, self.valid)
        loss, _ = concept_loss(out, self.labels, head.M)
        extra, terms = head.extra_loss(out, self.labels)
        (loss + extra).backward()
        self.assertEqual(set(terms), {'aux', 'kl', 'margin'})
        for name in ('decoder.query.weight', 'action_decoder.query.weight', 'rgcn.bases', 'emb_other',
                     'init_concept.weight', 'executor.theta.act_not', 'aux.weight', 'log_tau'):
            grad = dict(head.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0.0, name)

    def test_margin_only_on_partners(self):
        head = self.make()
        head.eval()
        out = head(self.feats, self.valid)
        no_partner = torch.tensor([a for a in range(120) if not head.partner[a].any()][:4])
        _, terms = head.extra_loss(out, no_partner)
        self.assertEqual(terms['margin'], 0.0)



if __name__ == '__main__':
    unittest.main()
