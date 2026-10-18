import unittest

import numpy as np

from tools.build_kg_embeddings import build


def fake_encode(texts):
    """Deterministic unit vectors, one per distinct string."""
    out = np.stack([np.random.default_rng(abs(hash(t)) % 2 ** 32).standard_normal(8) for t in texts])
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def tiny_kg():
    # JSON key order differs from label order on purpose.
    return {
        'hash': 'sha256:abc',
        'actions': {
            'b_action': {'id': 1, 'text': 'b', 'paraphrases': ['b1', 'b2']},
            'a_action': {'id': 0, 'text': 'a', 'paraphrases': []},
        },
        'attributes': {
            'z_attr': {'text': 'z', 'definition': 'z def'},
            'y_attr': {'text': 'y', 'definition': 'y def'},
        },
    }


class BuildKgEmbeddingsTests(unittest.TestCase):
    def test_actions_in_label_order_attributes_in_json_order(self):
        arrays = build(tiny_kg(), fake_encode, 'fake')
        self.assertEqual(list(arrays['action_names']), ['a_action', 'b_action'])
        self.assertEqual(list(arrays['attribute_names']), ['z_attr', 'y_attr'])
        self.assertEqual(str(arrays['kg_hash']), 'sha256:abc')

    def test_rows_are_unit_norm(self):
        arrays = build(tiny_kg(), fake_encode, 'fake')
        for key in ('action', 'action_text', 'attribute'):
            np.testing.assert_allclose(np.linalg.norm(arrays[key], axis=1), 1, rtol=1e-5)

    def test_ensemble_is_normalized_mean_of_descriptions(self):
        arrays = build(tiny_kg(), fake_encode, 'fake')
        mean = fake_encode(['b', 'b1', 'b2']).mean(0)
        np.testing.assert_allclose(arrays['action'][1], mean / np.linalg.norm(mean), rtol=1e-5)
        np.testing.assert_allclose(arrays['action_text'][1], fake_encode(['b'])[0], rtol=1e-5)
        # A single description: the ensemble equals the text embedding.
        np.testing.assert_allclose(arrays['action'][0], arrays['action_text'][0], rtol=1e-5)

    def test_rejects_non_contiguous_ids(self):
        kg = tiny_kg()
        kg['actions']['b_action']['id'] = 5
        with self.assertRaises(ValueError):
            build(kg, fake_encode, 'fake')


if __name__ == '__main__':
    unittest.main()
