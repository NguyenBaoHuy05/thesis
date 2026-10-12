import unittest

import torch

from model.large_symbolic_reasoner import (
    LargeSymbolicReasoner,
    PRIMITIVE_NAMES,
    RELATION_NAMES,
)


class LargeSymbolicReasonerTests(unittest.TestCase):
    def test_composes_actions_from_structured_evidence(self):
        reasoner = LargeSymbolicReasoner(
            feature_dims=(8, 16, 32),
            num_classes=120,
        )
        features = (
            torch.randn(2, 2, 8, 8, 25),
            torch.randn(2, 2, 16, 4, 25),
            torch.randn(2, 2, 32, 2, 25),
        )
        action_logits, primitive_logits, graph = reasoner(features)

        self.assertEqual(action_logits.shape, (2, 120))
        self.assertEqual(primitive_logits.shape, (2, len(PRIMITIVE_NAMES)))
        self.assertEqual(
            graph.relation_probabilities.shape,
            (2, len(RELATION_NAMES)),
        )
        self.assertEqual(reasoner.action_prototypes.in_features,
                         len(PRIMITIVE_NAMES) + len(RELATION_NAMES))

    def test_primitive_loss_uses_weak_action_support(self):
        reasoner = LargeSymbolicReasoner(
            feature_dims=(8, 16, 32),
            num_classes=120,
        )
        logits = torch.randn(3, len(PRIMITIVE_NAMES), requires_grad=True)
        labels = torch.tensor([0, 23, 53])
        loss = reasoner.loss(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
