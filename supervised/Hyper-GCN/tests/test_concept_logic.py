"""model/concept_logic.py on the real concept files and toy inputs."""

import os
import unittest

import torch

from model.concept_logic import concept_joint_mask, concept_loss, load_concepts
from model.concept_schema import NTU_ACTION_NAMES

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


    def test_concept_loss(self):
        M = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        out = dict(logits=torch.zeros(2, 2, requires_grad=True), c=torch.full((2, 3), .5, requires_grad=True))
        loss, terms = concept_loss(out, torch.tensor([0, 1]), M, beta=0.5)
        loss.backward()
        self.assertEqual(set(terms), {'ce', 'bce'})
        self.assertIsNotNone(out['c'].grad)


if __name__ == '__main__':
    unittest.main()
