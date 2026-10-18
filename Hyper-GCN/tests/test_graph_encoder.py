import unittest

import torch
import torch.nn.functional as F

from model.graph_encoder import TAU_MIN, GraphRetrievalScorer, RGCNEncoder
from model.kg import RELATIONS, KGTensors, relation_adjacency


def kg_3x2():
    """3 actions, 2 attributes. a0 has x (w 2), lacks y (w 1); a1 has y (w 1); a2 nothing.
    a0-a1 similar, a1-a2 opposite."""
    W = torch.tensor([[1.0, -0.5], [0.0, 0.5], [0.0, 0.0]])
    S_sim = torch.zeros(3, 3, dtype=torch.bool)
    S_opp = torch.zeros(3, 3, dtype=torch.bool)
    S_sim[0, 1] = S_sim[1, 0] = True
    S_opp[1, 2] = S_opp[2, 1] = True
    return KGTensors(W=W, y_attr=(W > 0).float(), known=W != 0, S_sim=S_sim, S_opp=S_opp)


class RelationAdjacencyTests(unittest.TestCase):
    def test_blocks_and_normalization(self):
        A = relation_adjacency(kg_3x2())
        self.assertEqual(tuple(A.shape), (len(RELATIONS), 5, 5))
        has, has_inv, lacks, lacks_inv, similar, opposite = A
        # attribute x (node 3) receives from a0 only; attribute y (node 4) from a1 only (has).
        torch.testing.assert_close(has[3, :3], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(has[4, :3], torch.tensor([0.0, 1.0, 0.0]))
        # a0 receives from attribute x via has_inv and from y via lacks_inv.
        torch.testing.assert_close(has_inv[0, 3:], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(lacks_inv[0, 3:], torch.tensor([0.0, 1.0]))
        self.assertEqual(similar[0, 1].item(), 1.0)
        self.assertEqual(opposite[2, 1].item(), 1.0)
        # Rows with no edge stay zero; rows with edges sum to 1.
        self.assertEqual(has[0].sum().item(), 0.0)
        self.assertEqual(opposite[0].sum().item(), 0.0)
        sums = A.sum(-1)
        self.assertTrue(torch.all((sums == 0) | torch.isclose(sums, torch.ones_like(sums))))

    def test_similar_and_opposite_are_separate(self):
        A = relation_adjacency(kg_3x2())
        self.assertEqual(A[RELATIONS.index('similar'), 1, 2].item(), 0.0)
        self.assertEqual(A[RELATIONS.index('opposite'), 1, 0].item(), 0.0)


class EncoderTests(unittest.TestCase):
    def test_shapes_and_no_edges_is_identity(self):
        torch.manual_seed(0)
        enc = RGCNEncoder(kg_3x2(), d=8, layers=0)
        torch.testing.assert_close(enc(), enc.x)
        enc = RGCNEncoder(kg_3x2(), d=8, layers=2)
        actions, attributes = enc.split(enc())
        self.assertEqual(tuple(actions.shape), (3, 8))
        self.assertEqual(tuple(attributes.shape), (2, 8))

    def test_edges_change_the_node_vectors(self):
        torch.manual_seed(0)
        enc = RGCNEncoder(kg_3x2(), d=8, layers=2, dropout=0.0).eval()
        before = enc()
        kg = kg_3x2()
        kg.W[2, 0] = 1.0                              # a2 now has x
        enc.adjacency = relation_adjacency(kg)
        self.assertFalse(torch.allclose(before[2], enc()[2]))

    def test_gradient_reaches_nodes_and_relations(self):
        torch.manual_seed(0)
        enc = RGCNEncoder(kg_3x2(), d=8, layers=2)
        enc().pow(2).sum().backward()
        self.assertIsNotNone(enc.x.grad)
        self.assertGreater(enc.layers[0].relation.grad.abs().sum().item(), 0)

    def test_text_init(self):
        torch.manual_seed(0)
        enc = RGCNEncoder(kg_3x2(), d=4, layers=1, init='text',
                          text_embeddings=(torch.randn(3, 6), torch.randn(2, 6)))
        self.assertEqual(tuple(enc.x.shape), (5, 4))
        with self.assertRaises(ValueError):
            RGCNEncoder(kg_3x2(), d=4, init='text')


class GraphRetrievalTests(unittest.TestCase):
    def test_scores_are_cosine_with_graph_nodes_over_tau(self):
        torch.manual_seed(0)
        head = GraphRetrievalScorer(16, kg_3x2(), tau=0.1, d=8, layers=2).eval()
        z = torch.randn(4, 16)
        g = head.class_embeddings
        expected = F.normalize(head.proj(z), dim=-1) @ F.normalize(g, dim=-1).t() / 0.1
        torch.testing.assert_close(head(z), expected)
        self.assertEqual(tuple(head(z).shape), (4, 3))

    def test_tau_clamped_and_backward(self):
        head = GraphRetrievalScorer(16, kg_3x2(), tau=1e-4, d=8)
        self.assertAlmostEqual(head.tau.item(), TAU_MIN, places=6)
        F.cross_entropy(head(torch.randn(4, 16)), torch.tensor([0, 1, 2, 0])).backward()
        self.assertIsNotNone(head.encoder.x.grad)


if __name__ == '__main__':
    unittest.main()
