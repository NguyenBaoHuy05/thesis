import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from model.kg import KGTensors, load_embeddings
from model.kg_retrieval import (TAU_MAX, TAU_MIN, KGRetrievalHead, RetrievalScorer,
                                discriminability, topk_metrics, verify)


def tiny_kg():
    return {
        'hash': 'sha256:abc',
        'actions': {'b': {'id': 1}, 'a': {'id': 0}},
        'attributes': {'x': {}, 'y': {}, 'z': {}},
    }


def save_embeddings(folder, **overrides):
    arrays = dict(action=np.eye(2, 4, dtype=np.float32), action_text=np.eye(2, 4, dtype=np.float32),
                  attribute=np.eye(3, 4, dtype=np.float32), action_names=np.array(['a', 'b']),
                  attribute_names=np.array(['x', 'y', 'z']), encoder=np.array('fake'),
                  kg_hash=np.array('sha256:abc'))
    arrays.update(overrides)
    path = os.path.join(folder, 'emb.npz')
    np.savez(path, **arrays)
    return path


class LoadEmbeddingsTests(unittest.TestCase):
    def test_loads_matching_file(self):
        with tempfile.TemporaryDirectory() as folder:
            action, attribute = load_embeddings(save_embeddings(folder), tiny_kg())
        self.assertEqual(tuple(action.shape), (2, 4))
        self.assertEqual(tuple(attribute.shape), (3, 4))

    def test_rejects_stale_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_embeddings(folder, kg_hash=np.array('sha256:old'))
            with self.assertRaises(ValueError):
                load_embeddings(path, tiny_kg())

    def test_rejects_wrong_action_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_embeddings(folder, action_names=np.array(['b', 'a']))
            with self.assertRaises(ValueError):
                load_embeddings(path, tiny_kg())


class RetrievalScorerTests(unittest.TestCase):
    def test_scores_are_cosine_over_tau(self):
        torch.manual_seed(0)
        embeddings = torch.randn(5, 8)
        head = RetrievalScorer(16, embeddings, tau=0.1).eval()
        z = torch.randn(3, 16)
        expected = F.normalize(head.proj(z), dim=-1) @ F.normalize(embeddings, dim=-1).t() / 0.1
        torch.testing.assert_close(head(z), expected)

    def test_tau_is_clamped(self):
        head = RetrievalScorer(4, torch.randn(3, 2), tau=1e-4)
        self.assertAlmostEqual(head.tau.item(), TAU_MIN, places=6)
        head = RetrievalScorer(4, torch.randn(3, 2), tau=5.0)
        self.assertAlmostEqual(head.tau.item(), TAU_MAX, places=6)

    def test_fixed_tau_is_not_a_parameter(self):
        head = RetrievalScorer(4, torch.randn(3, 2), learn_tau=False)
        self.assertNotIn('log_tau', dict(head.named_parameters()))

    def test_class_embeddings_are_not_trained(self):
        head = RetrievalScorer(4, torch.randn(3, 2))
        self.assertNotIn('class_embeddings', dict(head.named_parameters()))

    def test_new_class_needs_no_retraining(self):
        # Swapping in a larger embedding table changes the output width, not the weights.
        head = RetrievalScorer(4, torch.randn(3, 2)).eval()
        head.class_embeddings = F.normalize(torch.randn(4, 2), dim=-1)
        self.assertEqual(tuple(head(torch.randn(2, 4)).shape), (2, 4))


def random_kg(C=8, A=6, seed=0):
    """Random W in {-1, -1/2, 0, 1/2, 1}; attribute A-1 is unknown for every class."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randint(-2, 3, (C, A), generator=g).float() / 2
    W[:, -1] = 0
    S = torch.zeros(C, C, dtype=torch.bool)
    S[0, 1] = S[1, 0] = True
    return KGTensors(W=W, y_attr=(W > 0).float(), known=W != 0, S_sim=S,
                     S_opp=torch.zeros(C, C, dtype=torch.bool))


def make_head(C=8, A=6, k=3, **kwargs):
    torch.manual_seed(0)
    return KGRetrievalHead(16, torch.randn(C, 12), torch.randn(A, 10), random_kg(C, A), k=k,
                           d_model=8, heads=2, **kwargs)


class VerifyTests(unittest.TestCase):
    def test_attribute_shared_by_all_candidates_has_zero_discriminability(self):
        Wc = torch.tensor([[[1.0, 1.0, 0.5], [1.0, -1.0, 0.5], [1.0, 0.0, -0.5]]])  # [1, k=3, A=3]
        d = discriminability(Wc, torch.full((1, 3), 1 / 3))
        self.assertEqual(d[0, 0].item(), 0.0)
        self.assertAlmostEqual(d.max().item(), 1.0, places=5)
        # So the value of that attribute cannot change v.
        a = torch.tensor([[0.9, 0.3, 0.6]])
        a_flipped = a.clone()
        a_flipped[0, 0] = 0.1
        torch.testing.assert_close(verify(Wc, d, a).sum(-1), verify(Wc, d, a_flipped).sum(-1))

    def test_discriminability_gradient_is_finite_when_candidates_agree(self):
        Wc = torch.ones(2, 3, 4)
        pi = torch.full((2, 3), 1 / 3, requires_grad=True)
        discriminability(Wc, pi).sum().backward()
        self.assertTrue(torch.isfinite(pi.grad).all())

    def test_raising_an_attribute_the_class_has_raises_its_evidence(self):
        Wc = torch.tensor([[[1.0, -0.5], [-1.0, 0.5]]])
        d = torch.ones(1, 2)
        low, high = torch.tensor([[0.2, 0.5]]), torch.tensor([[0.8, 0.5]])
        v_low, v_high = verify(Wc, d, low).sum(-1), verify(Wc, d, high).sum(-1)
        self.assertGreater(v_high[0, 0].item(), v_low[0, 0].item())   # W[c0, attr0] > 0
        self.assertLess(v_high[0, 1].item(), v_low[0, 1].item())      # W[c1, attr0] < 0


class KGRetrievalHeadTests(unittest.TestCase):
    def test_logits_decompose_on_candidates_and_are_minus_inf_elsewhere(self):
        head = make_head().eval()
        out = head(torch.randn(4, 16))
        on_cand = out['logits'].gather(1, out['cand'])
        torch.testing.assert_close(on_cand, out['s_c'] + out['lam'] * out['v'] + out['f'])
        torch.testing.assert_close(out['contributions'].sum(-1), out['lam'] * out['v'])
        outside = torch.ones_like(out['logits'], dtype=torch.bool).scatter(1, out['cand'], False)
        self.assertTrue(torch.isneginf(out['logits'][outside]).all())

    def test_ground_truth_forced_into_candidates_only_with_labels(self):
        head = make_head().eval()
        z = torch.randn(4, 16)
        s = head.retrieval(z)
        worst = s.argmin(1)  # never in the top-3 of 8 classes
        out = head(z, labels=worst)
        self.assertTrue((out['cand'] == worst.unsqueeze(1)).any(1).all())
        self.assertTrue((out['cand'].gather(1, out['target'].unsqueeze(1)).squeeze(1) == worst).all())
        out = head(z)
        self.assertIsNone(out['target'])
        self.assertFalse((out['cand'] == worst.unsqueeze(1)).any(1).any())

    def test_nodes_outside_the_subgraph_do_not_change_fuse(self):
        head = make_head().eval()
        z = torch.randn(4, 16)
        f = head(z)['f']
        # The last attribute is unknown for every class, so it is never in G_q.
        head.attribute_embeddings[-1] = torch.randn(10)
        torch.testing.assert_close(head(z)['f'], f)
        # An attribute inside G_q does change f.
        head.attribute_embeddings[:-1] = torch.randn(5, 10)
        self.assertFalse(torch.allclose(head(z)['f'], f))

    def test_ablation_switches(self):
        z = torch.randn(4, 16)
        out = make_head(use_verify=False).eval()(z)
        self.assertEqual(out['contributions'].abs().sum().item(), 0.0)
        out = make_head(use_fuse=False).eval()(z)
        self.assertEqual(out['f'].abs().sum().item(), 0.0)
        out = make_head(uniform_d=True).eval()(z)
        self.assertTrue((out['d'] == 1).all())
        out = make_head(uniform_pi=True).eval()(z)
        self.assertTrue(torch.allclose(out['pi'], torch.full_like(out['pi'], 1 / 3)))
        head = make_head(learn_lambda=False, lambda_init=0.5)
        self.assertNotIn('lambda_raw', dict(head.named_parameters()))
        self.assertAlmostEqual(head.lam.item(), 0.5, places=5)

    def test_kg_and_embeddings_are_not_trained(self):
        names = dict(make_head().named_parameters())
        for buffer in ('W', 'known', 'neighbors', 'attribute_embeddings', 'retrieval.class_embeddings'):
            self.assertNotIn(buffer, names)

    def test_new_class_needs_no_retraining(self):
        head = make_head().eval()
        head.replace_classes(torch.randn(9, 12), random_kg(C=9, A=6, seed=1))
        out = head(torch.randn(2, 16))
        self.assertEqual(tuple(out['logits'].shape), (2, 9))

    def test_gradients_reach_every_branch(self):
        head = make_head().train()
        out = head(torch.randn(4, 16), labels=torch.tensor([0, 1, 2, 3]))
        F.cross_entropy(out['logits'], torch.tensor([0, 1, 2, 3])).backward()
        for name in ('retrieval.proj.weight', 'attribute.weight', 'lambda_raw', 'U_a.weight', 'u_val'):
            grad = dict(head.named_parameters())[name].grad
            self.assertTrue(grad is not None and grad.abs().sum() > 0, name)


class TopkMetricsTests(unittest.TestCase):
    def test_recall_at_k(self):
        scores = torch.tensor([[3., 2., 1.], [1., 3., 2.]])
        labels = torch.tensor([1, 1])
        metrics = topk_metrics(scores, labels, ks=(1, 2))
        self.assertAlmostEqual(metrics['recall@1'], 0.5)
        self.assertAlmostEqual(metrics['recall@2'], 1.0)


if __name__ == '__main__':
    unittest.main()
