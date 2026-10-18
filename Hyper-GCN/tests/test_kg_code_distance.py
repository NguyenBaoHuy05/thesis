import unittest

import numpy as np

from tools.kg_code_distance import average_ranks, distances, spearman


class KgCodeDistanceTests(unittest.TestCase):
    def test_distances(self):
        W = np.array([[1.0, -0.5, 0.0], [1.0, 0.5, 0.5], [1.0, -0.5, 0.0]])
        d = distances(W, [(0, 1), (0, 2)])
        self.assertEqual(d['l1'].tolist(), [1.5, 0.0])
        self.assertEqual(d['differ'].tolist(), [2, 0])
        self.assertEqual(d['opposed'].tolist(), [1, 0])  # attr 1: lacks vs has

    def test_average_ranks_share_ties(self):
        self.assertEqual(average_ranks(np.array([0, 0, 5, 1])).tolist(), [1.5, 1.5, 4.0, 3.0])

    def test_spearman_sign(self):
        x = np.array([0, 0, 1, 3, 7])
        self.assertAlmostEqual(spearman(x, -x), -1.0)
        self.assertGreater(spearman(x, x + 1), 0.99)


if __name__ == '__main__':
    unittest.main()
