"""model/concept_logic.py, tools/shuffle_concepts.py on the real concept files and toy inputs."""

import os
import random
import unittest

import torch

from model.concept_logic import (ConceptLogicHead, concept_joint_mask, concept_loss, load_concepts,
                                 logic_scores)
from model.concept_schema import NTU_ACTION_NAMES
from tools.shuffle_concepts import read, shuffle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv')
META = os.path.join(ROOT, 'annotations', 'concepts', 'ntu_concept_metadata_v2_4.csv')


class ConceptLogicTests(unittest.TestCase):
    def test_real_matrix_loads_in_label_order(self):
        M, names, meta = load_concepts(MATRIX, META, NTU_ACTION_NAMES)
        self.assertEqual(tuple(M.shape), (120, 102))
        self.assertEqual(len({tuple(r) for r in M.tolist()}), 120)       # every class has its own row
        mask = concept_joint_mask(names, meta)
        self.assertTrue(mask.any(1).all())
        self.assertTrue(mask[names.index('head_nod'), 3] and not mask[names.index('head_nod'), 15])
        self.assertTrue(mask[names.index('hand_contact_mouth_region'), 3])   # contact with the head
        self.assertTrue(mask[names.index('leg_kick'), 19] and not mask[names.index('leg_kick'), 11])

    def test_logic_prefers_the_class_whose_literals_hold(self):
        M = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        c = torch.tensor([[.9, .9, .1]])
        scores = logic_scores(c, M, omega=1.0)
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertTrue((scores <= 0).all())
        # has-only AND cannot tell a class from a superset class when the extra concept is on
        both = logic_scores(torch.tensor([[.9, .9, .9]]), torch.tensor([[1., 0., 0.], [1., 1., 0.]]), omega=0.0)
        self.assertAlmostEqual(both[0, 0].item(), both[0, 1].item(), places=5)

    def test_head_starts_uniform_and_trains(self):
        M = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]])
        mask = torch.ones(4, 25, dtype=torch.bool)
        head = ConceptLogicHead(8, M, mask, decoder_args=dict(d_model=16, heads=2, layers=1, max_frames=2,
                                                              num_person=2))
        feats, valid = torch.randn(5, 2, 8, 2, 25), torch.tensor([[True, False]] * 5)
        out = head(feats, valid)
        self.assertEqual(tuple(out['c'].shape), (5, 4))
        torch.testing.assert_close(out['c'], torch.full((5, 4), .5))
        self.assertTrue(torch.allclose(out['logits'], out['logits'][:, :1].expand(-1, 3)))
        loss, terms = concept_loss(out, torch.tensor([0, 1, 2, 0, 1]), head.M)
        loss.backward()
        self.assertIsNotNone(head.decoder.out.weight.grad)
        self.assertEqual(set(terms), {'ce', 'bce'})

    def test_shuffle_keeps_degrees_and_distinct_rows(self):
        _, _, M = read(MATRIX)
        S = shuffle(M, random.Random(0), 2000)
        self.assertEqual([sum(r) for r in S], [sum(r) for r in M])
        self.assertEqual([sum(c) for c in zip(*S)], [sum(c) for c in zip(*M)])
        self.assertEqual(len({tuple(r) for r in S}), len(S))
        self.assertNotEqual(S, M)


if __name__ == '__main__':
    unittest.main()
