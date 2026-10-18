"""A1-G options of model/kg_retrieval.py and train_kg_head.py (A1-G plan S2, S5)."""

import os
import tempfile
import unittest

import numpy as np
import torch

from model.kg import KGTensors
from model.kg_retrieval import (KGRetrievalHead, PredicateEvidence, a1_loss, masked_bce,
                                masked_discriminability, opposed_attributes)
from train_kg_head import evaluate_a1, load_predicates, needs_text, predicate_stats


def kg(C=6, A=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randint(-2, 3, (C, A), generator=g).float() / 2
    S = torch.zeros(C, C, dtype=torch.bool)
    S[0, 1] = S[1, 0] = True
    return KGTensors(W=W, y_attr=(W > 0).float(), known=W != 0, S_sim=S, S_opp=torch.zeros_like(S))


def graph_head(**kwargs):
    torch.manual_seed(0)
    args = dict(k=3, retrieval_source='graph', graph_args=dict(d=8, layers=2), verify_mode='opposed',
                normalize_verify=True, use_fuse=False)
    args.update(kwargs)
    return KGRetrievalHead(16, None, None, kg(), **args)


class OpposedTests(unittest.TestCase):
    def test_unknown_is_not_a_middle_value(self):
        # attr 0: has + unknown + unknown -> not opposed; attr 1: has + lacks -> opposed;
        # attr 2: lacks + unknown -> not opposed.
        Wc = torch.tensor([[[1.0, 0.5, -1.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]]])
        torch.testing.assert_close(opposed_attributes(Wc), torch.tensor([[0.0, 1.0, 0.0]]))

    def test_masked_std_ignores_unknown_and_needs_two_known(self):
        Wc = torch.tensor([[[1.0, 1.0], [0.0, -1.0], [0.0, 0.0]]])
        known = (Wc != 0).float()
        d = masked_discriminability(Wc, torch.full((1, 3), 1 / 3), known)
        self.assertEqual(d[0, 0].item(), 0.0)          # only one candidate knows attr 0
        self.assertAlmostEqual(d[0, 1].item(), 1.0, places=5)


class GraphHeadTests(unittest.TestCase):
    def test_forward_shapes_and_decomposition(self):
        head = graph_head().eval()
        out = head(torch.randn(5, 16))
        self.assertEqual(tuple(out['logits'].shape), (5, 6))
        torch.testing.assert_close(out['contributions'].sum(-1), out['lam'] * out['v'])
        self.assertTrue(torch.all(out['f'] == 0))
        self.assertTrue(torch.all((out['d'] == 0) | (out['d'] == 1)))

    def test_normalization_divides_by_used_known_cells(self):
        z = torch.randn(5, 16)
        plain = graph_head(normalize_verify=False).eval()(z)
        norm = graph_head(normalize_verify=True).eval()(z)
        head = graph_head().eval()
        n_c = (norm['d'].unsqueeze(1) * head.known[norm['cand']]).sum(-1).clamp_min(1)
        torch.testing.assert_close(norm['v'], plain['v'] / n_c)

    def test_no_text_embeddings_needed_and_trains(self):
        head = graph_head().train()
        out = head(torch.randn(5, 16), labels=torch.arange(5))
        loss, terms = a1_loss(out, torch.arange(5), head)
        loss.backward()
        self.assertIsNotNone(head.retrieval.encoder.x.grad)

    def test_fuse_on_graph_nodes(self):
        head = graph_head(use_fuse=True, d_model=8, heads=2).eval()
        self.assertEqual(tuple(head(torch.randn(3, 16))['f'].shape), (3, 3))

    def test_replace_classes_refused_on_graph(self):
        with self.assertRaises(ValueError):
            graph_head().replace_classes(torch.randn(6, 4), kg())

    def test_text_head_state_keys_unchanged(self):
        # A1 head.pt files (P3, P6 head_weights) must still load with strict=True.
        torch.manual_seed(0)
        head = KGRetrievalHead(16, torch.randn(6, 12), torch.randn(4, 10), kg(), k=3, d_model=8, heads=2)
        keys = set(head.state_dict())
        self.assertIn('attribute_embeddings', keys)
        self.assertFalse(any(k.startswith('predicates') for k in keys))


def predicate_head(learn_threshold=True):
    columns = [0, 2]                                   # predicates on attributes 0 and 2 only
    return graph_head(attribute_source='predicate',
                      predicate_args=dict(columns=columns, mean=torch.zeros(2), std=torch.ones(2),
                                          learn_threshold=learn_threshold))


class PredicateTests(unittest.TestCase):
    def test_evidence_columns_and_invalid_values(self):
        ev = PredicateEvidence([0, 2], torch.zeros(2), torch.ones(2), num_attributes=4)
        phi = torch.tensor([[5.0, -5.0], [5.0, 5.0]])
        valid = torch.tensor([[True, True], [True, False]])
        a, evidence = ev(phi, valid)
        self.assertGreater(a[0, 0].item(), 0.99)
        self.assertLess(a[0, 2].item(), 0.01)
        self.assertEqual(a[0, 1].item(), 0.5)          # no predicate
        self.assertEqual(a[1, 2].item(), 0.5)          # invalid value
        torch.testing.assert_close(evidence, torch.tensor([[True, False, True, False],
                                                           [True, False, False, False]]))

    def test_scale_is_positive_so_direction_is_fixed(self):
        ev = PredicateEvidence([0], torch.zeros(1), torch.ones(1), num_attributes=1)
        with torch.no_grad():
            ev.log_scale.fill_(-3.0)
        a_low, _ = ev(torch.tensor([[-1.0]]))
        a_high, _ = ev(torch.tensor([[1.0]]))
        self.assertGreater(a_high.item(), a_low.item())

    def test_attributes_without_evidence_do_not_enter_verify(self):
        head = predicate_head().eval()
        out = head(torch.randn(4, 16), phi=torch.randn(4, 2))
        self.assertTrue(torch.all(out['d'][:, [1, 3]] == 0))

    def test_thresholds_get_gradient_and_bce_is_masked(self):
        head = predicate_head().train()
        y = torch.arange(4)
        out = head(torch.randn(4, 16), labels=y, phi=torch.randn(4, 2))
        loss, _ = a1_loss(out, y, head)
        loss.backward()
        self.assertIsNotNone(head.predicates.theta.grad)
        # Changing a column without a predicate cannot change the BCE.
        a2 = out['a'].detach().clone()
        a2[:, 1] = 0.01
        y_attr = (head.W > 0).float()
        torch.testing.assert_close(masked_bce(out['a'].detach(), y, y_attr, head.known, out['evidence']),
                                   masked_bce(a2, y, y_attr, head.known, out['evidence']))

    def test_fixed_thresholds_are_buffers(self):
        head = predicate_head(learn_threshold=False)
        names = dict(head.named_parameters())
        self.assertNotIn('predicates.theta', names)


class TrainScriptTests(unittest.TestCase):
    def test_needs_text(self):
        self.assertTrue(needs_text(dict(head='k1')))
        self.assertFalse(needs_text(dict(head='g1', head_args=dict(init='learned'))))
        self.assertTrue(needs_text(dict(head='g1', head_args=dict(init='text'))))
        self.assertTrue(needs_text(dict(head='a1', head_args={})))
        self.assertFalse(needs_text(dict(head='a1', head_args=dict(retrieval_source='graph',
                                                                    attribute_source='predicate'))))

    def test_load_predicates_keeps_kg_attributes(self):
        kg_json = dict(attributes={'x': {}, 'y': {}, 'z': {}})
        with tempfile.TemporaryDirectory() as folder:
            np.savez(os.path.join(folder, 'p_train.npz'), phi=np.arange(6, dtype=np.float32).reshape(2, 3),
                     valid=np.ones((2, 3), bool), names=np.array(['z', 'not_in_kg', 'x']),
                     label=np.array([0, 1]), index=np.arange(2))
            phi, valid, label, names, columns = load_predicates(folder, 'p', 'train', kg_json)
        self.assertEqual(names, ['z', 'x'])
        self.assertEqual(columns, [2, 0])
        torch.testing.assert_close(phi, torch.tensor([[0.0, 2.0], [3.0, 5.0]]))

    def test_predicate_stats_skip_invalid(self):
        phi = torch.tensor([[1.0], [3.0], [100.0]])
        mean, std = predicate_stats(phi, torch.tensor([[True], [True], [False]]))
        self.assertAlmostEqual(mean.item(), 2.0)
        self.assertAlmostEqual(std.item(), 1.0)

    def test_evaluate_reports_coverage_and_fixed_broken(self):
        head = predicate_head()
        z, y = torch.randn(7, 16), torch.arange(7) % 6
        extra = dict(phi=torch.randn(7, 2), phi_valid=torch.ones(7, 2, dtype=torch.bool))
        _, metrics = evaluate_a1(head, z, y, kg(), batch_size=3, extra=extra, fc_logits=torch.randn(7, 6))
        self.assertIn('verify_coverage', metrics['terms'])
        self.assertEqual(set(metrics['vs_k0']), {'fixed', 'broken'})
        self.assertEqual(set(metrics['vs_retrieval']), {'fixed', 'broken'})


if __name__ == '__main__':
    unittest.main()
