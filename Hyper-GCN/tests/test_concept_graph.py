"""model/concept_graph.py on the real v2.4 matrix (plan §5 checks). No torch needed."""

import os
import tempfile
import unittest

from model.concept_graph import ConceptGraph, build_graph, flat_graph, random_pair_graph, read_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, 'annotations', 'concepts', 'ntu_action_concept_v2_4.csv')


class ConceptGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions, cls.concepts, cls.rows = read_matrix(MATRIX)
        cls.graph = build_graph(cls.actions, cls.concepts, cls.rows)

    def test_every_row_is_split_without_overlap(self):
        g = self.graph
        for a, row in enumerate(g.rows):
            parts = [set(g.action_direct[a])] + [set(g.motifs[m]) for m in g.action_motifs[a]]
            covered = set().union(*parts)
            self.assertEqual(sum(map(len, parts)), len(covered), g.actions[a])     # disjoint
            or_members = set().union(*[set(g.ors[o]) for o in g.action_ors[a]]) if g.action_ors[a] else set()
            self.assertEqual(covered | (set(row) & or_members), set(row), g.actions[a])
            self.assertFalse(covered & or_members, g.actions[a])
            for o in g.action_ors[a]:
                self.assertTrue(set(g.ors[o]) & set(row))

    def test_motifs_are_shared_subsets_without_common_concepts(self):
        g = self.graph
        common = {j for j in range(len(g.concepts)) if sum(j in r for r in g.rows) >= g.cfg['common_min']}
        for m, members in enumerate(g.motifs):
            users = [a for a, ms in enumerate(g.action_motifs) if m in ms]
            self.assertGreaterEqual(len(users), g.cfg['motif_min_support'])
            self.assertFalse(set(members) & common)
            for a in users:
                self.assertTrue(set(members) <= set(g.rows[a]))
        self.assertTrue(all(len(ms) <= g.cfg['motifs_per_action'] for ms in g.action_motifs))

    def test_not_nodes_read_only_the_distinguishing_concepts(self):
        g = self.graph
        pairs = set(g.pairs)
        for i, j, diff in g.nots:
            self.assertIn(tuple(sorted((i, j))), pairs)
            self.assertEqual(set(diff), set(g.rows[j]) - set(g.rows[i]))
            self.assertTrue(diff)
        self.assertEqual(len(g.pairs), 58)               # 55 Hamming <= 4 + 3 put_on / take_off not in them
        names = {(g.actions[i], g.actions[j]) for i, j, _ in g.nots}
        self.assertIn(('put_on_a_shoe', 'take_off_a_shoe'), names)

    def test_no_two_actions_read_the_same_literals(self):
        g = self.graph
        keys = set()
        for a in range(len(g.actions)):
            key = (frozenset(g.action_direct[a]), frozenset(g.action_motifs[a]), frozenset(g.action_ors[a]),
                   frozenset((j, tuple(d)) for i, j, d in g.nots if i == a))
            self.assertNotIn(key, keys, g.actions[a])
            keys.add(key)

    def test_flat_graph_is_the_matrix(self):
        g = flat_graph(self.actions, self.concepts, self.rows, nots=False)
        self.assertEqual(g.action_direct, self.rows)
        self.assertFalse(g.motifs or g.ors or g.nots)
        self.assertTrue(all(not m and not o for m, o in zip(g.action_motifs, g.action_ors)))
        self.assertEqual(len(flat_graph(self.actions, self.concepts, self.rows).nots), len(self.graph.nots))

    def test_random_pairs_keep_sizes(self):
        r = random_pair_graph(self.graph, seed=0)
        self.assertEqual([len(d) for _, _, d in r.nots], [len(d) for _, _, d in self.graph.nots])
        self.assertEqual(r.pairs, self.graph.pairs)
        for i, j, diff in r.nots:
            self.assertNotEqual(i, j)
            self.assertFalse(set(diff) & set(self.rows[i]))

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'g.json')
            self.graph.save(path)
            loaded = ConceptGraph.load(path)
        self.assertEqual(loaded.edges(), self.graph.edges())
        self.assertEqual(loaded.summary(), self.graph.summary())


if __name__ == '__main__':
    unittest.main()
