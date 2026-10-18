import unittest

import torch

from model.kg import KGTensors
from model.kg_retrieval import KGRetrievalHead
from train_kg_head import attribute_auc, evaluate_a1, loss_terms, masked_bce


def kg_2x2():
    W = torch.tensor([[1.0, 0.0], [-0.5, 0.5]])  # class 0 has attr 0, unknown attr 1
    S = torch.zeros(2, 2, dtype=torch.bool)
    return KGTensors(W=W, y_attr=(W > 0).float(), known=W != 0, S_sim=S, S_opp=S)


class MaskedBceTests(unittest.TestCase):
    def test_unknown_cells_are_ignored(self):
        kg = kg_2x2()
        labels = torch.tensor([0])
        a = torch.tensor([[0.9, 0.5]])
        a_other = torch.tensor([[0.9, 0.01]])  # only the unknown cell changes
        torch.testing.assert_close(masked_bce(a, labels, kg), masked_bce(a_other, labels, kg))

    def test_known_cells_count(self):
        kg = kg_2x2()
        good = masked_bce(torch.tensor([[0.1, 0.9]]), torch.tensor([1]), kg)  # lacks 0, has 1
        bad = masked_bce(torch.tensor([[0.9, 0.1]]), torch.tensor([1]), kg)
        self.assertLess(good.item(), bad.item())


class AttributeAucTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        kg = kg_2x2()
        labels = torch.tensor([0, 0, 1, 1])
        perfect = torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.2, 0.7], [0.1, 0.9]])
        auc, count = attribute_auc(perfect, labels, kg)
        self.assertEqual(count, 1)  # attr 1 is known only for class 1: one value, no AUC
        self.assertAlmostEqual(auc, 1.0)
        auc, _ = attribute_auc(1 - perfect, labels, kg)
        self.assertAlmostEqual(auc, 0.0)


def small_head():
    torch.manual_seed(0)
    C, A = 6, 4
    W = torch.randint(-2, 3, (C, A)).float() / 2
    S = torch.zeros(C, C, dtype=torch.bool)
    kg = KGTensors(W=W, y_attr=(W > 0).float(), known=W != 0, S_sim=S, S_opp=S)
    head = KGRetrievalHead(8, torch.randn(C, 5), torch.randn(A, 5), kg, k=3, d_model=8, heads=2)
    return head, kg


class LossAndEvaluateTests(unittest.TestCase):
    def test_a1_loss_terms_and_backward(self):
        head, kg = small_head()
        config = dict(head='a1', loss=dict(alpha=0.5, beta=0.5, gamma=0.1))
        loss, logits, terms = loss_terms(config, head.train(), torch.randn(5, 8), torch.arange(5), kg)
        self.assertEqual(set(terms), {'ce_cand', 'ce_all', 'bce', 'f_ratio'})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(tuple(logits.shape), (5, 6))

    def test_gamma_denominator_is_stop_gradient(self):
        # The f-ratio penalty must not be lowered by scaling s: its gradient w.r.t. tau is zero
        # when f does not depend on tau.
        head, kg = small_head()
        head.train()
        z, y = torch.randn(5, 8), torch.arange(5)
        out = head(z, labels=y)
        scale = (out['s_c'].abs() + (out['lam'] * out['v']).abs()).mean().detach()
        (out['f'].abs().mean() / scale).backward()
        grad = head.retrieval.log_tau.grad
        self.assertTrue(grad is None or grad.abs().item() == 0.0)

    def test_evaluate_reports_terms_and_flips(self):
        head, kg = small_head()
        logits, metrics = evaluate_a1(head, torch.randn(7, 8), torch.arange(7) % 6, kg, batch_size=3)
        self.assertEqual(tuple(logits.shape), (7, 6))
        for key in ('mean_abs_s', 'spread_lambda_v', 'flip_without_lambda_v', 'flip_without_f'):
            self.assertIn(key, metrics['terms'])
        self.assertTrue(0 <= metrics['terms']['flip_without_f'] <= 1)


if __name__ == '__main__':
    unittest.main()
