import unittest

import torch

from model.kg import kg_tensors


def edge(w):
    return {'w': w, 'agree': 1.0, 'evidence': ''}


def tiny_kg():
    # JSON key order differs from label order on purpose; `opposite` listed on one side only.
    return {
        'attributes': {'p': {}, 'q': {}, 'r': {}},
        'actions': {
            'b': {'id': 1, 'has': {'q': edge(1)}, 'lacks': {'p': edge(2)},
                  'similar': ['a'], 'opposite': []},
            'a': {'id': 0, 'has': {'p': edge(2), 'r': edge(1)}, 'lacks': {'q': edge(1)},
                  'similar': ['b'], 'opposite': ['c']},
            'c': {'id': 2, 'has': {'r': edge(2)}, 'lacks': {}, 'similar': [], 'opposite': []},
        },
    }


class KGTensorsTests(unittest.TestCase):
    def test_signed_weights_in_label_and_attribute_order(self):
        t = kg_tensors(tiny_kg())
        expected = torch.tensor([[1.0, -0.5, 0.5],    # a: has p (2), lacks q (1), has r (1)
                                 [-1.0, 0.5, 0.0],    # b: lacks p (2), has q (1)
                                 [0.0, 0.0, 1.0]])    # c: has r (2)
        torch.testing.assert_close(t.W, expected)

    def test_attribute_targets_and_mask(self):
        t = kg_tensors(tiny_kg())
        self.assertEqual(t.y_attr.tolist(), [[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(t.known.tolist(), [[True, True, True], [True, True, False],
                                            [False, False, True]])

    def test_relations_are_symmetric(self):
        t = kg_tensors(tiny_kg())
        self.assertTrue(t.S_sim[0, 1] and t.S_sim[1, 0])
        self.assertTrue(t.S_opp[0, 2] and t.S_opp[2, 0])  # listed only on `a`
        self.assertEqual(int(t.S_sim.sum()), 2)
        self.assertFalse(t.S_sim.diagonal().any() or t.S_opp.diagonal().any())

    def test_rejects_unknown_attribute(self):
        kg = tiny_kg()
        kg['actions']['c']['has']['zz'] = edge(1)
        with self.assertRaises(ValueError):
            kg_tensors(kg)

    def test_rejects_has_and_lacks_on_the_same_attribute(self):
        kg = tiny_kg()
        kg['actions']['c']['lacks']['r'] = edge(1)
        with self.assertRaises(ValueError):
            kg_tensors(kg)

    def test_rejects_unexpected_weight(self):
        kg = tiny_kg()
        kg['actions']['c']['has']['r'] = edge(3)
        with self.assertRaises(ValueError):
            kg_tensors(kg)

    def test_rejects_unknown_related_action(self):
        kg = tiny_kg()
        kg['actions']['c']['similar'] = ['nobody']
        with self.assertRaises(ValueError):
            kg_tensors(kg)


if __name__ == '__main__':
    unittest.main()
