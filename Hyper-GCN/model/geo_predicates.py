"""Fixed geometric predicates and predicate-defined hyperedges for A3-G
(plan: KG_RETRIEVAL_PLAN/A3G_PREDICATE_HYPEREDGE/A3G_PLAN.md §1.1, definitions: KG_RETRIEVAL_PLAN/PAIR_REFEREE/PREDICATES.md).

Input: the feeder output in joint mode, x [N, 3, T, 25, M]: every joint relative to joint 20 (spine shoulder), joint 20
itself holds the absolute trajectory. Nothing here reads a label or a backbone feature, and nothing has a gradient.

Body frame per sample and actor (ported from the A1-G predicates, so values do not depend on random_rot):
    up      = mean direction spine_base -> spine_shoulder over the first 20% of frames
    L       = mean torso length (every distance, height and velocity is divided by L)
    right   = shoulder_R - shoulder_L, orthogonalized against up
    forward = up x right
Only differences of joints are projected, so no origin is needed.

Active hand a = the hand with the longer path relative to the spine shoulder; o = the other hand.

    raw    [N, M, K=19, T]   PREDICATE_NAMES, higher = more present
    e_hat  [N, M, K, T]      clip((raw - q5) / (q95 - q5), 0, 1); signed predicates: clip(0.5 + raw / 2q, 0, 1),
                             q = max(|q5|, |q95|), so 0.5 = no motion / horizontal
    p      [N, M, E=21, T]   HYPEREDGES weights; signed predicates split into clip(2e-1) and clip(1-2e); 0 for an
                             actor that is not real
    B      [N, M, E, 25]     joints of each hyperedge, resolved with the actor's active hand
"""

import numpy as np
import torch
import torch.nn.functional as F

JOINTS = {
    'sb': 0, 'sm': 1, 'nk': 2, 'hd': 3,
    'shoulder_l': 4, 'elbow_l': 5, 'wrist_l': 6, 'hand_l': 7,
    'shoulder_r': 8, 'elbow_r': 9, 'wrist_r': 10, 'hand_r': 11,
    'hip_l': 12, 'knee_l': 13, 'ankle_l': 14, 'foot_l': 15,
    'hip_r': 16, 'knee_r': 17, 'ankle_r': 18, 'foot_r': 19,
    'ss': 20, 'tip_l': 21, 'thumb_l': 22, 'tip_r': 23, 'thumb_r': 24,
}
NUM_JOINTS = 25
EDGE = 0.2           # fraction of frames that defines "start" (up axis)
SMOOTH = 5           # moving average (frames) before a velocity
FINE_EPS = 0.01      # floor of the wrist speed in fine_motion_a (L / frame)

PREDICATE_NAMES = (
    'near_mouth_a', 'near_mouth_o', 'near_ear_a', 'above_head_a', 'near_chest_a', 'hand_behind', 'hands_close',
    'hand_near_foot', 'thumb_up_a', 'thumb_tip_gap_a', 'elbow_raised_a', 'elbow_raised_o', 'head_back', 'head_down',
    'torso_bend', 'pelvis_low', 'foot_raised', 'rising_a', 'fine_motion_a',
)
SIGNED = ('thumb_up_a', 'rising_a')

# (name, joints, predicate, transform of e_hat). Joints ending in _a / _o follow the active / other hand.
HYPEREDGES = (
    ('mouth_a', ('wrist_a', 'hand_a', 'tip_a', 'nk', 'hd'), 'near_mouth_a', 'id'),
    ('mouth_o', ('wrist_o', 'hand_o', 'tip_o', 'nk', 'hd'), 'near_mouth_o', 'id'),
    ('ear_a', ('wrist_a', 'hand_a', 'tip_a', 'hd'), 'near_ear_a', 'id'),
    ('above_head_a', ('elbow_a', 'wrist_a', 'hand_a', 'hd'), 'above_head_a', 'id'),
    ('chest_a', ('wrist_a', 'hand_a', 'sm', 'ss'), 'near_chest_a', 'id'),
    ('behind_a', ('wrist_a', 'hand_a', 'sb', 'sm'), 'hand_behind', 'id'),
    ('hands_close', ('wrist_l', 'hand_l', 'wrist_r', 'hand_r'), 'hands_close', 'id'),
    ('hand_foot', ('hand_l', 'hand_r', 'ankle_l', 'foot_l', 'ankle_r', 'foot_r'), 'hand_near_foot', 'id'),
    ('thumb_up_a', ('wrist_a', 'hand_a', 'thumb_a'), 'thumb_up_a', 'pos'),
    ('thumb_down_a', ('wrist_a', 'hand_a', 'thumb_a'), 'thumb_up_a', 'neg'),
    ('pinch_a', ('hand_a', 'tip_a', 'thumb_a'), 'thumb_tip_gap_a', 'inv'),
    ('elbow_up_a', ('shoulder_a', 'elbow_a', 'wrist_a'), 'elbow_raised_a', 'id'),
    ('elbow_up_o', ('shoulder_o', 'elbow_o', 'wrist_o'), 'elbow_raised_o', 'id'),
    ('head_back', ('ss', 'nk', 'hd'), 'head_back', 'id'),
    ('head_down', ('ss', 'nk', 'hd'), 'head_down', 'id'),
    ('torso_bend', ('sb', 'sm', 'ss', 'nk', 'hip_l', 'hip_r'), 'torso_bend', 'id'),
    ('pelvis_low', ('sb', 'hip_l', 'hip_r', 'knee_l', 'knee_r'), 'pelvis_low', 'id'),
    ('foot_raised', ('knee_l', 'ankle_l', 'foot_l', 'knee_r', 'ankle_r', 'foot_r'), 'foot_raised', 'id'),
    ('rising_a', ('shoulder_a', 'elbow_a', 'wrist_a', 'hand_a'), 'rising_a', 'pos'),
    ('lowering_a', ('shoulder_a', 'elbow_a', 'wrist_a', 'hand_a'), 'rising_a', 'neg'),
    ('fine_a', ('wrist_a', 'hand_a', 'tip_a', 'thumb_a'), 'fine_motion_a', 'id'),
)
HYPEREDGE_NAMES = tuple(h[0] for h in HYPEREDGES)
HAND_PARTS = ('shoulder', 'elbow', 'wrist', 'hand', 'tip', 'thumb')


def absolute(x):
    """x [N, 3, T, 25, M] (relative to joint 20) -> absolute joints [N, M, T, 25, 3]."""
    if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != NUM_JOINTS:
        raise ValueError('x must have shape [N, 3, T, 25, M]')
    joints = x.permute(0, 4, 2, 3, 1)
    root = joints[..., JOINTS['ss']:JOINTS['ss'] + 1, :]
    keep = torch.ones(NUM_JOINTS, 1, dtype=x.dtype, device=x.device)
    keep[JOINTS['ss']] = 0
    return joints + root * keep


def valid_actors(xa, atol=1e-6):
    """[N, M] bool: not all zero and (m > 0) not a copy of the first actor (as the A1-G predicates)."""
    nonzero = xa.abs().amax(dim=(2, 3, 4)) > atol
    copy = (xa - xa[:, :1]).abs().amax(dim=(2, 3, 4)) <= atol
    copy[:, 0] = False
    return nonzero & ~copy


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _smooth(v):
    """Moving average over SMOOTH frames along dim 2 of [N, M, T, ...], edge-padded."""
    moved = v.movedim(2, -1)
    flat = moved.reshape(-1, 1, moved.shape[-1])
    flat = F.pad(flat, (SMOOTH // 2, SMOOTH - 1 - SMOOTH // 2), mode='replicate')
    return F.avg_pool1d(flat, SMOOTH, stride=1).reshape(moved.shape).movedim(-1, 2)


def _velocity(v):
    """Central difference along dim 2 of [N, M, T, ...] (one-sided at the ends)."""
    out = torch.zeros_like(v)
    if v.shape[2] > 1:
        out[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / 2
        out[:, :, 0] = v[:, :, 1] - v[:, :, 0]
        out[:, :, -1] = v[:, :, -1] - v[:, :, -2]
    return out


class BodyFrame:
    """Body frame of every actor. xa [N, M, T, V, 3]."""

    def __init__(self, xa):
        T = xa.shape[2]
        head = max(1, int(round(EDGE * T)))
        torso = xa[..., JOINTS['ss'], :] - xa[..., JOINTS['sb'], :]                     # [N, M, T, 3]
        self.up = _unit(torso[:, :, :head].mean(2))                                         # [N, M, 3]
        self.L = torso.norm(dim=-1).mean(2).clamp_min(1e-3)                                # [N, M]
        right = (xa[..., JOINTS['shoulder_r'], :] - xa[..., JOINTS['shoulder_l'], :]).mean(2)
        right = right - (right * self.up).sum(-1, keepdim=True) * self.up
        self.right = _unit(right)
        self.forward = torch.cross(self.up, self.right, dim=-1)

    def _expand(self, value, like):
        """[N, M] against [N, M, T, ...], or [N, M, 3] against [N, M, T, ..., 3]: insert the missing middle dims."""
        shape = value.shape[:2] + (1,) * (like.ndim - value.ndim) + value.shape[2:]
        return value.reshape(shape)

    def _project(self, vec, axis):
        """Difference vector(s) [N, M, T, ..., 3] on an axis, divided by L: [N, M, T, ...]."""
        return (vec * self._expand(axis, vec)).sum(-1) / self._expand(self.L, vec[..., 0])

    def height(self, vec):
        return self._project(vec, self.up)

    def fwd(self, vec):
        return self._project(vec, self.forward)

    def norm(self, vec):
        return vec.norm(dim=-1) / self._expand(self.L, vec[..., 0])

    def dist(self, a, b):
        return self.norm(a - b)


def _hand(xa, side, part):
    """Joint `part` of the hand on `side` ([N, M], 0 = left, 1 = right): [N, M, T, 3]."""
    left = xa[..., JOINTS[part + '_l'], :]
    right = xa[..., JOINTS[part + '_r'], :]
    return torch.where((side == 0)[:, :, None, None], left, right)


def _incidence_table(active_side):
    """[E, 25] float: joints of every hyperedge when the active hand is `active_side` (0 left, 1 right)."""
    sides = {'a': 'l' if active_side == 0 else 'r', 'o': 'r' if active_side == 0 else 'l'}
    table = torch.zeros(len(HYPEREDGES), NUM_JOINTS)
    for e, (_, joints, _, _) in enumerate(HYPEREDGES):
        for joint in joints:
            part, _, who = joint.rpartition('_')
            name = f'{part}_{sides[who]}' if part in HAND_PARTS and who in sides else joint
            table[e, JOINTS[name]] = 1
    return table


def random_incidence(seed):
    """[E, 25]: random joint sets with the row sizes of the real table (A3-G control 'random'). CPU generator, so
    the sets are the same on every device and at every construction."""
    generator = torch.Generator().manual_seed(int(seed))
    sizes = _incidence_table(1).sum(1).long().tolist()
    table = torch.zeros(len(HYPEREDGES), NUM_JOINTS)
    for e, size in enumerate(sizes):
        table[e, torch.randperm(NUM_JOINTS, generator=generator)[:size]] = 1
    return table


class GeoPredicates:
    """Predicates, their normalization (train quantiles) and the hyperedges built from them."""

    def __init__(self, stats=None):
        q5, q95 = torch.zeros(len(PREDICATE_NAMES)), torch.ones(len(PREDICATE_NAMES))
        if stats is not None:
            values = np.load(stats) if isinstance(stats, str) else stats
            if 'names' in values and tuple(str(n) for n in values['names']) != PREDICATE_NAMES:
                raise ValueError(f'{stats}: predicate names differ from PREDICATE_NAMES (stale stats file)')
            q5 = torch.as_tensor(np.asarray(values['q5']), dtype=torch.float32)
            q95 = torch.as_tensor(np.asarray(values['q95']), dtype=torch.float32)
            if q5.numel() != len(PREDICATE_NAMES) or q95.numel() != len(PREDICATE_NAMES):
                raise ValueError('predicate stats need one q5 / q95 per predicate')
        signed = torch.tensor([n in SIGNED for n in PREDICATE_NAMES])
        q = torch.maximum(q5.abs(), q95.abs()).clamp_min(1e-6)
        # e_hat = clip(raw * scale + shift, 0, 1)
        self.scale = torch.where(signed, 1 / (2 * q), 1 / (q95 - q5).clamp_min(1e-6))
        self.shift = torch.where(signed, torch.full_like(q, 0.5), -q5 * self.scale)
        self.tables = torch.stack([_incidence_table(0), _incidence_table(1)])            # [2, E, 25]

    @torch.no_grad()
    def raw(self, x):
        """x [N, 3, T, 25, M] -> (raw [N, M, K, T], valid [N, M] bool, active [N, M] long: 0 left / 1 right)."""
        xa = absolute(x.float())
        f = BodyFrame(xa)
        valid = valid_actors(xa)
        J = lambda name: xa[..., JOINTS[name], :]                                           # noqa: E731
        ss = J('ss')
        path = [f.norm(_velocity(J(f'hand_{s}') - ss)).sum(2) for s in 'lr']
        active = (path[1] > path[0]).long()
        other = 1 - active
        hand_a, hand_o = _hand(xa, active, 'hand'), _hand(xa, other, 'hand')
        wrist_a, tip_a, thumb_a = (_hand(xa, active, p) for p in ('wrist', 'tip', 'thumb'))
        mouth = 0.6 * J('hd') + 0.4 * J('nk')
        chest = (J('sm') + ss) / 2
        ear_offset = 0.25 * (f.right * f.L[..., None])[:, :, None]                         # [N, M, 1, 3]
        head_tilt = f.fwd(J('hd') - J('nk'))
        feet = (J('foot_l'), J('foot_r'))
        hand_velocity = _velocity(_smooth(hand_a - J('sb')))
        finger = f.norm(_velocity(_smooth(tip_a - wrist_a)))
        wrist_speed = f.norm(_velocity(_smooth(wrist_a - ss)))
        raw = torch.stack((
            -f.dist(hand_a, mouth),                                                         # near_mouth_a
            -f.dist(hand_o, mouth),                                                         # near_mouth_o
            -torch.minimum(f.dist(hand_a, J('hd') + ear_offset), f.dist(hand_a, J('hd') - ear_offset)),
            f.height(hand_a - J('hd')),                                                     # above_head_a
            -f.dist(hand_a, chest),                                                         # near_chest_a
            -f.fwd(hand_a - J('sm')),                                                       # hand_behind
            -f.dist(hand_a, hand_o),                                                        # hands_close
            -torch.stack([f.dist(h, ft) for h in (hand_a, hand_o) for ft in feet]).amin(0),  # hand_near_foot
            f.height(thumb_a - hand_a),                                                     # thumb_up_a (signed)
            f.dist(thumb_a, tip_a),                                                         # thumb_tip_gap_a
            f.height(_hand(xa, active, 'elbow') - _hand(xa, active, 'shoulder')),           # elbow_raised_a
            f.height(_hand(xa, other, 'elbow') - _hand(xa, other, 'shoulder')),             # elbow_raised_o
            -head_tilt,                                                                     # head_back
            head_tilt,                                                                      # head_down
            f.fwd(ss - J('sb')),                                                            # torso_bend
            torch.minimum(*(f.height(ft - J('sb')) for ft in feet)),                        # pelvis_low
            f.height(feet[0] - feet[1]).abs(),                                              # foot_raised
            f.height(hand_velocity),                                                        # rising_a (signed)
            finger / (wrist_speed + FINE_EPS),                                              # fine_motion_a
        ), dim=2)
        raw = torch.where(valid[:, :, None, None], raw, torch.zeros_like(raw))
        return raw, valid, active

    def normalize(self, raw):
        scale = self.scale.to(raw.device, raw.dtype).view(1, 1, -1, 1)
        shift = self.shift.to(raw.device, raw.dtype).view(1, 1, -1, 1)
        return (raw * scale + shift).clamp(0, 1)

    def weights(self, e_hat, valid):
        """e_hat [N, M, K, T] -> hyperedge weights p [N, M, E, T], 0 for actors that are not real."""
        index = {name: k for k, name in enumerate(PREDICATE_NAMES)}
        out = []
        for _, _, predicate, transform in HYPEREDGES:
            e = e_hat[:, :, index[predicate]]
            out.append({'id': e, 'inv': 1 - e, 'pos': (2 * e - 1).clamp(0, 1), 'neg': (1 - 2 * e).clamp(0, 1)}[transform])
        return torch.stack(out, dim=2) * valid[:, :, None, None].to(e_hat.dtype)

    def incidence(self, active, random=False, seed=0):
        """active [N, M] -> B [N, M, E, 25] (the actor's active hand), or the random table for every actor."""
        if random:
            table = random_incidence(seed).to(active.device)
            return table.expand(*active.shape, *table.shape)
        return self.tables.to(active.device)[active]

    @torch.no_grad()
    def __call__(self, x, random=False, seed=0):
        """x [N, 3, T, 25, M] -> (p [N, M, E, T], B [N, M, E, 25])."""
        raw, valid, active = self.raw(x)
        return self.weights(self.normalize(raw), valid), self.incidence(active, random, seed)


def save_stats(path, q5, q95):
    np.savez(path, q5=np.asarray(q5, dtype=np.float32), q95=np.asarray(q95, dtype=np.float32),
             names=np.asarray(PREDICATE_NAMES))
