import os
import random
import tempfile
import unittest
from collections import Counter

import numpy as np

from model.kg import kg_tensors, load_embeddings
from tools.shuffle_kg import kg_hash, shuffle_kg


def synthetic_kg(seed=0, actions=30, attributes=12):
    rng = random.Random(seed)
    attrs = [f't{j}' for j in range(attributes)]
    kg = {'attributes': {t: {'text': t} for t in attrs},
          'attribute_relations': {'implies': [], 'excludes': [['t0', 't1']]},
          'dropped_edges': [], 'actions': {}}
    for i in range(actions):
        chosen = rng.sample(attrs, 7)
        has = [t for t in chosen[:4] if t != 't1'] if 't0' in chosen[:4] else chosen[:4]
        kg['actions'][f'a{i}'] = {
            'id': i, 'text': f'a{i}',
            'has': {t: {'w': rng.choice([1, 2]), 'agree': 1.0, 'evidence': 'x'} for t in has},
            'lacks': {t: {'w': rng.choice([1, 2]), 'agree': 0.67, 'evidence': 'y'} for t in chosen[4:]},
            'similar': [], 'opposite': [], 'reviewed': False,
        }
    names = list(kg['actions'])
    for relation, count in (('similar', 25), ('opposite', 8)):
        while sum(len(a[relation]) for a in kg['actions'].values()) < 2 * count:
            a, b = rng.sample(names, 2)
            if b not in kg['actions'][a][relation]:
                kg['actions'][a][relation].append(b)
                kg['actions'][b][relation].append(a)
    kg['hash'] = kg_hash(kg)
    return kg


def degrees(kg, kind):
    by_action = {name: len(a[kind]) for name, a in kg['actions'].items()}
    by_attr = Counter(attr for a in kg['actions'].values() for attr in a[kind])
    return by_action, by_attr


class ShuffleKgTests(unittest.TestCase):
    def setUp(self):
        self.kg = synthetic_kg()
        self.out = shuffle_kg(self.kg, seed=0)

    def test_attribute_degrees_preserved_per_kind(self):
        for kind in ('has', 'lacks'):
            self.assertEqual(degrees(self.out, kind), degrees(self.kg, kind))

    def test_weight_multiset_preserved(self):
        for kind in ('has', 'lacks'):
            weights = lambda kg: sorted(e['w'] for a in kg['actions'].values() for e in a[kind].values())
            self.assertEqual(weights(self.out), weights(self.kg))

    def test_edges_actually_move(self):
        changed = self.out['shuffled']['changed']
        for kind in ('has', 'lacks', 'similar'):
            self.assertGreater(changed[kind]['moved'], changed[kind]['edges'] // 2, kind)

    def test_no_attribute_both_has_and_lacks(self):
        for a in self.out['actions'].values():
            self.assertFalse(set(a['has']) & set(a['lacks']))

    def test_excludes_respected(self):
        for a in self.out['actions'].values():
            self.assertFalse({'t0', 't1'} <= set(a['has']))

    def test_relations_symmetric_without_self_links_and_degree_preserved(self):
        for relation in ('similar', 'opposite'):
            for name, a in self.out['actions'].items():
                self.assertNotIn(name, a[relation])
                self.assertEqual(len(a[relation]), len(set(a[relation])))
                self.assertEqual(len(a[relation]), len(self.kg['actions'][name][relation]))
                for other in a[relation]:
                    self.assertIn(name, self.out['actions'][other][relation])

    def test_texts_untouched_and_input_not_modified(self):
        original = synthetic_kg()
        self.assertEqual(self.kg, original)
        for name, a in self.out['actions'].items():
            self.assertEqual(a['text'], original['actions'][name]['text'])
        self.assertEqual(self.out['attributes'], original['attributes'])

    def test_deterministic_per_seed(self):
        self.assertEqual(shuffle_kg(self.kg, seed=0), self.out)
        self.assertNotEqual(shuffle_kg(self.kg, seed=1)['hash'], self.out['hash'])

    def test_hash_and_provenance(self):
        self.assertEqual(self.out['hash'], kg_hash(self.out))
        self.assertEqual(self.out['shuffled']['source_hash'], self.kg['hash'])
        with self.assertRaises(ValueError):
            shuffle_kg(self.out, seed=0)

    def test_loads_into_tensors_and_reuses_real_embeddings(self):
        t = kg_tensors(self.out)
        self.assertEqual(int(t.known.sum()), sum(len(a['has']) + len(a['lacks'])
                                                 for a in self.kg['actions'].values()))
        names = [f'a{i}' for i in range(30)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'emb.npz')
            np.savez(path, action=np.eye(30, 32, dtype=np.float32),
                     attribute=np.eye(12, 32, dtype=np.float32), action_names=np.array(names),
                     attribute_names=np.array(list(self.kg['attributes'])),
                     kg_hash=np.array(self.kg['hash']))
            action, _ = load_embeddings(path, self.out)
        self.assertEqual(tuple(action.shape), (30, 32))


if __name__ == '__main__':
    unittest.main()
