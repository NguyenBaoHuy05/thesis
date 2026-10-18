"""Geometric predicates phi_attr(X) for the A1-G VERIFY step (A1-G plan S3). numpy only, so the cache
tool (tools/cache_predicates.py) and the tests run without torch.

Input: the feeder output in joint mode, x [N, 3, T, V=25, M] (feeders/feeder_ntu.py): every joint is
relative to joint 20 (spine shoulder), and joint 20 itself holds the absolute trajectory.
`absolute` undoes that. Nothing here reads a label or a backbone feature: each phi is a fixed
function of the coordinates, and a higher value means "the attribute is more present".

Body frame, per sample and actor, so that phi does not depend on random_rot (rotation by +-0.3 rad):
    up      = mean direction spine_base -> spine_shoulder over the first 20% of frames
              (people start upright in NTU; a sit_down still starts standing)
    L       = mean torso length (normalizes every distance and height)
    right   = shoulder_R - shoulder_L, orthogonalized against up
    forward = up x right (camera space is right-handed). The sign is checked on data by the AUC
              report of tools/cache_predicates.py: an AUC far below 0.5 means the sign is wrong.

Single-person predicates take the max over the real actors (as z is pooled in model/hypergcn_kg.py);
two-person predicates need two real actors and are invalid otherwise (the head then uses p = 0.5,
i.e. no evidence).

NTU joints (0-based): 0 spine base, 1 spine mid, 2 neck, 3 head, 4-7 left shoulder/elbow/wrist/hand,
8-11 right, 12-15 left hip/knee/ankle/foot, 16-19 right, 20 spine shoulder, 21-24 hand tips/thumbs.
"""

import numpy as np

SPINE_BASE, SPINE_MID, NECK, HEAD, SPINE_SHOULDER = 0, 1, 2, 3, 20
SHOULDER = (4, 8)
ELBOW = (5, 9)
HAND = (7, 11)
HIP = (12, 16)
KNEE = (13, 17)
FOOT = (15, 19)
LEG_JOINTS = (13, 14, 15, 17, 18, 19)
EPS = 1e-6
EDGE = 0.2          # fraction of frames that defines "start" and "end"


def absolute(x):
    """x [N, 3, T, V, M] (relative to joint 20) -> absolute coordinates [N, M, T, V, 3]."""
    out = x + x[:, :, :, SPINE_SHOULDER:SPINE_SHOULDER + 1]
    out[:, :, :, SPINE_SHOULDER] = x[:, :, :, SPINE_SHOULDER]
    return out.transpose(0, 4, 2, 3, 1)


def valid_actors(x, atol=1e-6):
    """[N, M] bool, numpy twin of model/hypergcn_kg.py valid_actor_mask: not all zero and (m > 0)
    not a copy of the first actor."""
    nonzero = np.abs(x).max(axis=(1, 2, 3)) > atol
    copy = np.abs(x - x[..., :1]).max(axis=(1, 2, 3)) <= atol
    copy[:, 0] = False
    return nonzero & ~copy


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), EPS)


class Frame:
    """Body frame of one actor for every sample. P: [N, T, V, 3] absolute coordinates."""

    def __init__(self, P):
        T = P.shape[1]
        head = max(1, int(round(EDGE * T)))
        torso = P[:, :, SPINE_SHOULDER] - P[:, :, SPINE_BASE]                        # [N, T, 3]
        self.up = _unit(torso[:, :head].mean(1))                                       # [N, 3]
        self.L = np.maximum(np.linalg.norm(torso, axis=-1).mean(1), EPS)              # [N]
        right = (P[:, :, SHOULDER[1]] - P[:, :, SHOULDER[0]]).mean(1)
        right = right - (right * self.up).sum(-1, keepdims=True) * self.up
        self.right = _unit(right)
        self.forward = np.cross(self.up, self.right)
        self.P, self.T, self.edge = P, T, head

    def joint(self, v):
        return self.P[:, :, v]                                                         # [N, T, 3]

    def _proj(self, vec, axis):
        return (vec * axis[:, None]).sum(-1) / self.L[:, None]                        # [N, T]

    def height(self, vec):
        return self._proj(vec, self.up)

    def lateral(self, vec):
        return self._proj(vec, self.right)

    def fwd(self, vec):
        return self._proj(vec, self.forward)

    def dist(self, a, b):
        return np.linalg.norm(a - b, axis=-1) / self.L[:, None]                       # [N, T]

    def start(self, series):
        return series[:, :self.edge].mean(1)

    def end(self, series):
        return series[:, -self.edge:].mean(1)

    def middle(self, series):
        third = self.T // 3
        return series[:, third:self.T - third].mean(1)

    def path(self, vec):
        """Path length per frame of a trajectory [N, T, 3], normalized: [N, T - 1]."""
        return np.linalg.norm(np.diff(vec, axis=1), axis=-1) / self.L[:, None]

    def active_hand(self):
        """Trajectory of the hand that moves most, relative to the spine shoulder: [N, T, 3]."""
        rel = [self.joint(h) - self.joint(SPINE_SHOULDER) for h in HAND]
        energy = np.stack([self.path(r).sum(1) for r in rel], 1)                       # [N, 2]
        pick = energy.argmax(1)
        return np.where(pick[:, None, None] == 0, rel[0], rel[1])


def repetitions(signal, prominence=0.15, smooth=5):
    """Number of back-and-forth cycles of each row of signal [N, T]: extrema of the smoothed signal,
    keeping only swings larger than `prominence` x its range."""
    kernel = np.ones(smooth) / smooth
    out = np.zeros(len(signal))
    for n, s in enumerate(signal):
        s = np.convolve(np.pad(s, smooth // 2, mode='edge'), kernel, mode='valid')
        threshold = prominence * (s.max() - s.min())
        if threshold < EPS:
            continue
        # Zigzag: a swing ends when the signal comes back from its running extreme by `threshold`.
        direction, anchor, extreme, swings = 0, s[0], s[0], 0
        for value in s[1:]:
            if direction == 0:
                if abs(value - anchor) >= threshold:
                    direction, extreme = np.sign(value - anchor), value
            elif (value - extreme) * direction > 0:
                extreme = value
            elif abs(value - extreme) >= threshold:
                swings += 1
                direction, extreme = -direction, value
        out[n] = (swings + (direction != 0)) / 2
    return out


def principal_signal(trajectory):
    """Projection of a centered trajectory [N, T, 3] on its direction of largest variance: [N, T]."""
    centered = trajectory - trajectory.mean(1, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered * vt[:, :1]).sum(-1)


def _energy(f, joints, reference):
    return sum(f.path(f.joint(j) - f.joint(reference)).sum(1) for j in joints)


# ---- single-person predicates: f (Frame of one actor) -> phi [N] --------------------------------

def _hand_head(f):
    return np.stack([f.dist(f.joint(h), f.joint(HEAD)) for h in HAND])                # [2, N, T]


def _chest(f):
    return (f.joint(SPINE_MID) + f.joint(SPINE_SHOULDER)) / 2


def _arm_asymmetry(f):
    e = [_energy(f, (h - 1, h), SPINE_SHOULDER) for h in HAND]                         # wrist + hand
    return np.abs(e[0] - e[1]) / np.maximum(e[0] + e[1], EPS)


SINGLE = {
    # contact / position
    'hand_at_mouth': lambda f: -_hand_head(f).min(0).min(1),
    'both_hands_at_face': lambda f: -_hand_head(f).max(0).min(1),
    'hand_on_head_top': lambda f: np.max([f.height(f.joint(h) - f.joint(HEAD)) for h in HAND], 0).max(1),
    'hands_above_head': lambda f: np.min([f.height(f.joint(h) - f.joint(HEAD)) for h in HAND], 0).max(1),
    'hands_touch_each_other': lambda f: -f.dist(f.joint(HAND[0]), f.joint(HAND[1])).min(1),
    'hand_on_own_chest': lambda f: -np.min([f.dist(f.joint(h), _chest(f)) for h in HAND], 0).min(1),
    'hand_at_own_hip': lambda f: -np.min([f.dist(f.joint(h), f.joint(p)) for h, p in zip(HAND, HIP)], 0).min(1),
    'hands_down_to_feet': lambda f: -np.min([f.height(f.joint(h) - f.joint(ft)) for h in HAND for ft in FOOT], 0).min(1),
    'elbow_raised_high': lambda f: np.max([f.height(f.joint(e) - f.joint(s)) for e, s in zip(ELBOW, SHOULDER)], 0).max(1),
    'arms_out_to_sides': lambda f: np.minimum(-f.lateral(f.joint(HAND[0]) - f.joint(SPINE_SHOULDER)),
                                              f.lateral(f.joint(HAND[1]) - f.joint(SPINE_SHOULDER))).max(1),
    'hand_crosses_midline': lambda f: np.maximum(f.lateral(f.joint(HAND[0]) - f.joint(SPINE_MID)),
                                                 -f.lateral(f.joint(HAND[1]) - f.joint(SPINE_MID))).max(1),
    'arm_extends_forward': lambda f: np.max([f.fwd(f.joint(h) - f.joint(s)) for h, s in zip(HAND, SHOULDER)], 0).max(1),
    # head and torso
    'head_tilted_back': lambda f: (-f.fwd(f.joint(HEAD) - f.joint(NECK))).max(1),
    'head_bowed_toward_hands': lambda f: f.fwd(f.joint(HEAD) - f.joint(NECK)).max(1),
    'torso_bent_forward': lambda f: f.fwd(f.joint(SPINE_SHOULDER) - f.joint(SPINE_BASE)).max(1),
    # whole body
    'seated': lambda f: -np.median(f.height(f.joint(SPINE_BASE) - (f.joint(KNEE[0]) + f.joint(KNEE[1])) / 2), 1),
    'hips_lower': lambda f: f.start(f.height(f.joint(SPINE_BASE))) - f.end(f.height(f.joint(SPINE_BASE))),
    'hips_rise': lambda f: f.end(f.height(f.joint(SPINE_BASE))) - f.start(f.height(f.joint(SPINE_BASE))),
    'airborne': lambda f: (lambda low: low.max(1) - np.percentile(low, 20, axis=1))(
        np.minimum(f.height(f.joint(FOOT[0])), f.height(f.joint(FOOT[1])))),
    'on_one_leg': lambda f: np.abs(f.height(f.joint(FOOT[0]) - f.joint(FOOT[1]))).max(1),
    'knee_lifts_forward': lambda f: np.max([f.height(f.joint(k) - f.joint(h)) for k, h in zip(KNEE, HIP)], 0).max(1),
    'walks_across_floor': lambda f: (lambda d: np.linalg.norm(d - (d * f.up).sum(-1, keepdims=True) * f.up, axis=-1) / f.L)(
        f.joint(SPINE_BASE)[:, -f.edge:].mean(1) - f.joint(SPINE_BASE)[:, :f.edge].mean(1)),
    # which body part moves
    'legs_carry_motion': lambda f: _energy(f, LEG_JOINTS, SPINE_BASE) / np.maximum(
        _energy(f, LEG_JOINTS, SPINE_BASE) + _energy(f, (6, 7, 10, 11), SPINE_SHOULDER), EPS),
    'head_carries_motion': lambda f: _energy(f, (HEAD,), SPINE_SHOULDER) / np.maximum(
        _energy(f, (HEAD,), SPINE_SHOULDER) + _energy(f, HAND, SPINE_SHOULDER), EPS),
    'single_arm_active': _arm_asymmetry,
    'both_arms_symmetric': lambda f: 1 - _arm_asymmetry(f),
    # dynamics of the active hand
    'fast_arm_swing': lambda f: f.path(f.active_hand()).max(1),
    'sudden_fast_motion': lambda f: (lambda s: s.max(1) / np.maximum(s.mean(1), EPS))(f.path(f.active_hand())),
    'many_steady_repeats': lambda f: repetitions(principal_signal(f.active_hand())),
    'single_occurrence': lambda f: -repetitions(principal_signal(f.active_hand())),
    'long_hold': lambda f: (lambda s: (s < 0.25 * np.maximum(s.max(1, keepdims=True), EPS)).mean(1))(
        f.path(f.active_hand())),
    'raise_hold_lower': lambda f: np.max([f.middle(f.height(f.joint(h) - f.joint(SPINE_BASE)))
                                          - np.maximum(f.start(f.height(f.joint(h) - f.joint(SPINE_BASE))),
                                                       f.end(f.height(f.joint(h) - f.joint(SPINE_BASE))))
                                          for h in HAND], 0),
    'hand_retracts_to_body': lambda f: (lambda d: d.max(1) - f.end(d))(
        np.linalg.norm(f.active_hand(), axis=-1) / f.L[:, None]),
    'hands_come_together': lambda f: (lambda d: f.start(d) - f.end(d))(f.dist(f.joint(HAND[0]), f.joint(HAND[1]))),
    'hands_move_apart': lambda f: (lambda d: f.end(d) - f.start(d))(f.dist(f.joint(HAND[0]), f.joint(HAND[1]))),
    # temporal direction (A1-G plan S3, not in the KG yet: used only once the KG has the attribute)
    'hands_rise_over_time': lambda f: np.mean([f.end(f.height(f.joint(h) - f.joint(SPINE_BASE)))
                                               - f.start(f.height(f.joint(h) - f.joint(SPINE_BASE)))
                                               for h in HAND], 0),
    'hands_lower_over_time': lambda f: np.mean([f.start(f.height(f.joint(h) - f.joint(SPINE_BASE)))
                                                - f.end(f.height(f.joint(h) - f.joint(SPINE_BASE)))
                                                for h in HAND], 0),
}


# ---- two-person predicates: (f0, f1) -> phi [N] --------------------------------------------------

def _between(f0, f1):
    return np.linalg.norm(f0.joint(SPINE_BASE) - f1.joint(SPINE_BASE), axis=-1) / ((f0.L + f1.L) / 2)[:, None]


PAIR = {
    'distance_decreases': lambda f0, f1: (lambda d: f0.start(d) - f0.end(d))(_between(f0, f1)),
    'distance_increases': lambda f0, f1: (lambda d: f0.end(d) - f0.start(d))(_between(f0, f1)),
}

PREDICATES = tuple(SINGLE) + tuple(PAIR)


def compute(x, names=PREDICATES):
    """x [N, 3, T, V, M] feeder output (joint mode). Returns (phi [N, P] float32, valid [N, P] bool)."""
    unknown = [name for name in names if name not in SINGLE and name not in PAIR]
    if unknown:
        raise ValueError(f'no predicate for {unknown}')
    x = np.asarray(x, dtype=np.float64)
    P = absolute(x)
    actors = valid_actors(x)
    if not actors[:, 0].all():
        raise ValueError('sample whose first actor is empty')
    frames = [Frame(P[:, m]) for m in range(P.shape[1])]
    phi = np.zeros((len(x), len(names)))
    valid = np.ones((len(x), len(names)), dtype=bool)
    for j, name in enumerate(names):
        if name in SINGLE:
            per_actor = np.stack([SINGLE[name](f) for f in frames], 1)                 # [N, M]
            phi[:, j] = np.where(actors, per_actor, -np.inf).max(1)
        else:
            if len(frames) < 2:
                valid[:, j] = False
                continue
            phi[:, j] = PAIR[name](frames[0], frames[1])
            valid[:, j] = actors[:, 1]
    phi = np.where(valid, phi, 0.0)
    if not np.isfinite(phi).all():
        raise FloatingPointError('non-finite predicate value')
    return phi.astype(np.float32), valid
