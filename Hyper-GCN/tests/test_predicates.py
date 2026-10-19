"""model/predicates.py on synthetic skeletons (numpy only, no torch)."""

import unittest

import numpy as np

from model import predicates as pr

T = 64


def standing():
    """One standing person, camera space (y up), [V=25, 3] in meters."""
    J = np.zeros((25, 3))
    J[0], J[1], J[20], J[2], J[3] = (0, 0, 0), (0, .25, 0), (0, .5, 0), (0, .55, 0), (0, .7, 0)
    for side, sx in ((0, 1), (1, -1)):                       # 0 = left (x > 0), 1 = right
        b = 4 * side
        J[4 + b], J[5 + b], J[6 + b], J[7 + b] = (.2 * sx, .5, 0), (.25 * sx, .25, 0), (.27 * sx, .05, 0), (.27 * sx, 0, 0)
        J[12 + b], J[13 + b], J[14 + b], J[15 + b] = (.1 * sx, 0, 0), (.1 * sx, -.45, 0), (.1 * sx, -.85, 0), (.1 * sx, -.9, .1)
        J[21 + 2 * side], J[22 + 2 * side] = (.27 * sx, -.05, 0), (.25 * sx, 0, .02)
    return J


def sequence(edit=None):
    """[T, V, 3]; edit(seq, t, s) changes frame t in place, s in [0, 1] = progress."""
    seq = np.repeat(standing()[None], T, 0)
    if edit:
        for t in range(T):
            edit(seq, t, t / (T - 1))
    return seq


def feeder(*actors):
    """Absolute [T, V, 3] sequences -> feeder joint-mode x [1, 3, T, V, 2] (single person: the second
    slot is a copy of the first, as align_frames does)."""
    actors = list(actors) + [actors[0]] * (2 - len(actors))
    x = np.stack(actors, -1).transpose(2, 0, 1, 3).astype(np.float64)          # [3, T, V, M]
    rel = x - x[:, :, 20:21]
    rel[:, :, 20] = x[:, :, 20]
    return rel[None]


def phi(x, name):
    values, valid = pr.compute(x, (name,))
    return values[0, 0], valid[0, 0]


def bump(s):
    return np.sin(np.pi * s)                                  # 0 -> 1 -> 0


class PredicateTests(unittest.TestCase):
    def test_hand_at_mouth(self):
        def to_mouth(seq, t, s):
            seq[t, 11] = (1 - bump(s)) * seq[t, 11] + bump(s) * (seq[t, 3] + (0, -.08, .08))
        self.assertGreater(phi(feeder(sequence(to_mouth)), 'hand_at_mouth')[0],
                           phi(feeder(sequence()), 'hand_at_mouth')[0] + 0.5)

    def test_hips_lower_and_rise(self):
        def sit(seq, t, s):
            seq[t, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 21, 22, 23, 24]] += (0, -.4 * s, 0)
        x = feeder(sequence(sit))
        self.assertGreater(phi(x, 'hips_lower')[0], 0.5)
        self.assertLess(phi(x, 'hips_rise')[0], -0.5)

    def test_head_bowed_versus_tilted(self):
        def bow(seq, t, s):
            seq[t, 3] += (0, -.03 * bump(s), .12 * bump(s))
        x = feeder(sequence(bow))
        self.assertGreater(phi(x, 'head_bowed_toward_hands')[0], phi(x, 'head_tilted_back')[0])

    def test_rotation_invariance(self):
        def wave(seq, t, s):
            seq[t, 11] = seq[t, 8] + (-.1, .3 + .1 * np.sin(6 * np.pi * s), .1)
        seq = sequence(wave)
        rng = np.random.default_rng(0)
        a, b, c = rng.uniform(-.3, .3, 3)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        rotated = seq @ (Rz @ Ry @ Rx).T
        p0, _ = pr.compute(feeder(seq))
        p1, _ = pr.compute(feeder(rotated))
        np.testing.assert_allclose(p0, p1, atol=1e-6)

    def test_repetitions(self):
        s = np.linspace(0, 1, T)
        self.assertEqual(pr.repetitions(np.sin(2 * np.pi * 3 * s)[None])[0], 3.5)  # 7 half-swings
        self.assertEqual(pr.repetitions(bump(s)[None])[0], 1.0)                    # up and back
        self.assertEqual(pr.repetitions(s[None])[0], 0.5)                          # one way only
        self.assertEqual(pr.repetitions(np.zeros((1, T)))[0], 0.0)

    def test_two_person_distance(self):
        def approach(seq, t, s):
            seq[t] += (1.5 - 1.0 * s, 0, 0)
        x = feeder(sequence(), sequence(approach))
        value, valid = phi(x, 'distance_decreases')
        self.assertTrue(valid)
        self.assertGreater(value, 1.0)
        self.assertLess(phi(x, 'distance_increases')[0], -1.0)

    def test_two_person_predicate_needs_a_second_real_actor(self):
        value, valid = phi(feeder(sequence()), 'distance_decreases')
        self.assertFalse(valid)
        self.assertEqual(value, 0.0)

    def test_copy_of_first_actor_is_not_a_second_actor(self):
        x = feeder(sequence())
        np.testing.assert_array_equal(pr.valid_actors(x), [[True, False]])

    def test_every_predicate_is_finite(self):
        values, valid = pr.compute(feeder(sequence(), sequence(lambda q, t, s: q.__setitem__(t, q[t] + (1, 0, 0)))))
        self.assertEqual(values.shape, (1, len(pr.PREDICATES)))
        self.assertTrue(np.isfinite(values).all() and valid.all())

    def test_then_follows_peak_order(self):
        s = np.linspace(0, 1, T)
        early, late = np.exp(-((s - .2) / .1) ** 2)[None], np.exp(-((s - .8) / .1) ** 2)[None]
        self.assertGreater(pr.then(early, late)[0], 0.99)
        self.assertLess(pr.then(late, early)[0], 0.1)
        flat = pr.normalized_event(np.ones((1, T)))
        np.testing.assert_array_equal(flat, 0.0)

    def test_then_hat_on_versus_off(self):
        """Put on: hand above the head, then down onto it. Take off: hand on the head, then up and away."""
        top, above = (0, .78, 0), (0, 1.05, 0)

        def path(first, second):
            def edit(seq, t, s):                      # right hand: rest, first, second, rest
                if .1 <= s <= .8:
                    seq[t, 11] = first if s < .45 else second
            return edit
        on = feeder(sequence(path(above, top)))
        off = feeder(sequence(path(top, above)))
        self.assertGreater(phi(on, 'then_above_head_to_hand_at_head')[0],
                           phi(on, 'then_hand_at_head_to_above_head')[0])
        self.assertGreater(phi(off, 'then_hand_at_head_to_above_head')[0],
                           phi(off, 'then_above_head_to_hand_at_head')[0])

    def test_unknown_predicate_is_rejected(self):
        with self.assertRaises(ValueError):
            pr.compute(feeder(sequence()), ('not_an_attribute',))


if __name__ == '__main__':
    unittest.main()
