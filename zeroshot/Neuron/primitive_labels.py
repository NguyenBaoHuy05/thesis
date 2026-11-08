"""Truth values of the KG primitives computed from the raw skeleton, following the `definition` of each primitive
in primitive*.json. Used for (1) direct supervision of the grounding (BCE on p_k^s(t), lambda_prim in
main_nesy.py) and (2) the rule oracle: the rules evaluated on these values instead of the learned grounding.

    labels = PrimitiveLabeler(prims)(x, dt)      x (n, 3, T, 25, M) metres, dt (n,) seconds per frame
                                                 -> (n, 2, K, T // 4) in [0, 1]  (side 0 = left, 1 = right)

Conventions (zeroshot/prompts/format_primitive_list.py): L = body-size unit, H0 = reference standing height,
f / lat / up = the person's forward / lateral / up axes, pH pT pS pE pW = hand, hand tip, shoulder, elbow, wrist.
Choices made here (the definitions do not fix them):
  - up = camera y (NTU is recorded with level Kinects; the labels use the crop of the training view but NOT its
    random rotation, so they describe the physical pose);
  - L = median length of the chain spine base - spine mid - spine shoulder - neck - head (~0.7 m);
  - H0 = 1.05 x mean leg length (hip - knee - ankle): the hip height of the person standing, which is what the
    body.* thresholds compare the pelvis height with;
  - floor = 5th percentile of the lower foot height; f = horizontal normal of the shoulder line, oriented by the
    feet (and toward the camera, as Kinect assumes);
  - every comparison is soft: sigmoid((a - b) / margin), AND = min, OR = max, NOT = 1 - x;
  - "for >= x s" keeps the frames of runs that long; motion events mark the frames of the event (+- a window);
  - "within x s" counts of reversals are taken over the clip (the clips are short);
  - single-person primitives: max over the tracked persons; inter.*: both directions (A -> B, B -> A);
  - lateral primitives (kg.side_kind): side s uses the arm / leg of side s, the others give the same value on both
    sides, as the grounding's joint masks.
The 64 frames are pooled (max) by 4 to the 16 time steps of the encoder cube.

Self-check on Colab (step 0: are the labels sensible, how good are the rules on them):
    python primitive_labels.py --config config/nesy/ntu60_xsub_55_5.yaml --n 3000
"""
import argparse
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import kg as kg_lib

VERSION = 1          # bump when a detector changes: the label caches are keyed by it
FPS = 30.0           # NTU RGB+D (60 and 120)
ARM = (dict(S=4, E=5, W=6, H=7, T=21, Th=22), dict(S=8, E=9, W=10, H=11, T=23, Th=24))     # left, right
LEG = (dict(Hip=12, K=13, A=14, F=15), dict(Hip=16, K=17, A=18, F=19))


# ====================================================================== soft logic and vector helpers
def gt(a, b, m):
    return torch.sigmoid((a - b) / m)


def lt(a, b, m):
    return torch.sigmoid((b - a) / m)


def between(a, lo, hi, m):
    return torch.minimum(gt(a, lo, m), lt(a, hi, m))


def AND(*xs):
    r = xs[0]
    for x in xs[1:]:
        r = torch.minimum(r, x)
    return r


def OR(*xs):
    r = xs[0]
    for x in xs[1:]:
        r = torch.maximum(r, x)
    return r


def norm(v):
    return v.norm(dim=-1)


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def dot(a, b):
    return (a * b).sum(-1)


def ang(a, b):
    """Angle in degrees between vectors (last dim)."""
    return torch.rad2deg(torch.acos(dot(unit(a), unit(b)).clamp(-1, 1)))


def horiz(v):
    return v * v.new_tensor([1.0, 0.0, 1.0])


def seg_dist(p, a, b):
    """Distance from point p to segment a-b (all (..., 3))."""
    ab = b - a
    t = (dot(p - a, ab) / dot(ab, ab).clamp(min=1e-6)).clamp(0, 1)
    return norm(p - (a + t[..., None] * ab))


# ====================================================================== time helpers (time = last dim)
def _b(v, like):
    """Per-sample tensor (n,) broadcast against like (n, ..., T)."""
    return v.view(-1, *([1] * (like.dim() - 1)))


def _frames(dt, sec, like):
    return _b((sec / dt).round().clamp(min=1), like)


def _shift(x, o, past):
    """x at t - o (past) or t + o, edge-replicated."""
    if past:
        return torch.cat([x[..., :1].expand(*x.shape[:-1], o), x[..., :-o]], -1)
    return torch.cat([x[..., o:], x[..., -1:].expand(*x.shape[:-1], o)], -1)


def win_max(x, dt, sec, mode='c'):
    """Max over [t - w, t] ('b'), [t, t + w] ('f') or a centred window of total length sec ('c')."""
    if mode == 'c':
        return torch.maximum(win_max(x, dt, sec / 2, 'b'), win_max(x, dt, sec / 2, 'f'))
    T = x.shape[-1]
    w = _frames(dt, sec, x)
    out = x
    for o in range(1, min(int(w.max()), T - 1) + 1):
        out = torch.where(o <= w, torch.maximum(out, _shift(x, o, mode == 'b')), out)
    return out


def win_min(x, dt, sec, mode='c'):
    return -win_max(-x, dt, sec, mode)


def win_mean(x, dt, sec):
    """Centred moving average over sec."""
    T = x.shape[-1]
    w = _frames(dt, sec / 2, x)
    s, c = x.clone(), torch.ones_like(x)
    for o in range(1, min(int(w.max()), T - 1) + 1):
        m = (o <= w).float()
        s = s + m * (_shift(x, o, True) + _shift(x, o, False))
        c = c + 2 * m
    return s / c


def tdiff(x, dt):
    """Central time derivative per second of (n, ..., T)."""
    xp = torch.cat([x[..., :1], x, x[..., -1:]], -1)
    return (xp[..., 2:] - xp[..., :-2]) / (2 * _b(dt, x))


def run_containing(b):
    """Length (frames) of the run of True that contains each frame, 0 where False."""
    bf = b.float()
    T = bf.shape[-1]
    fw, bw = torch.zeros_like(bf), torch.zeros_like(bf)
    acc = torch.zeros_like(bf[..., 0])
    for t in range(T):
        acc = (acc + 1) * bf[..., t]
        fw[..., t] = acc
    acc = torch.zeros_like(bf[..., 0])
    for t in reversed(range(T)):
        acc = (acc + 1) * bf[..., t]
        bw[..., t] = acc
    return (fw + bw - 1).clamp(min=0) * bf


def lasting(x, dt, sec):
    """Keep x on the frames of runs (x > 0.5) that last >= sec seconds."""
    return x * (run_containing(x > 0.5) * _b(dt, x) >= sec - 1e-6).float()


def lasting_frames(x, n):
    return x * (run_containing(x > 0.5) >= n).float()


def reversals(d, amp):
    """Direction reversals of a detrended signal, with hysteresis +-amp."""
    state = torch.zeros_like(d[..., 0])
    cnt = torch.zeros_like(d[..., 0])
    one = torch.ones_like(state)
    for t in range(d.shape[-1]):
        hi, lo = d[..., t] > amp, d[..., t] < -amp
        cnt = cnt + ((hi & (state < 0)) | (lo & (state > 0))).float()
        state = torch.where(hi, one, torch.where(lo, -one, state))
    return cnt


def oscillation(s, dt, amp, min_rev, win=1.5, amp_max=None, fmin=None, fmax=None):
    """Frames of an oscillation of s with semi-amplitude >= amp (and < amp_max), >= min_rev reversals over the
    clip and a reversal rate giving a frequency in [fmin, fmax] Hz."""
    d = s - win_mean(s, dt, win)
    rev = reversals(d, 0.6 * amp)
    act = (d.abs() > 0.3 * amp).float()
    ok = rev >= min_rev
    if amp_max is not None:
        ok = ok & (d.abs().amax(-1) < amp_max)
    if fmin is not None or fmax is not None:
        dur = (act.sum(-1) * _b(dt, rev)).clamp(min=1e-3)
        freq = rev / 2 / dur
        if fmin is not None:
            ok = ok & (freq >= fmin)
        if fmax is not None:
            ok = ok & (freq <= fmax)
    return win_max(act, dt, 0.5) * ok[..., None].float()


def up_crossings(h, lo, hi):
    """Number of times h rises above hi after having been below lo."""
    state = torch.zeros_like(h[..., 0])
    cnt = torch.zeros_like(h[..., 0])
    one = torch.ones_like(state)
    for t in range(h.shape[-1]):
        up, down = h[..., t] > hi, h[..., t] < lo
        cnt = cnt + (up & (state < 0)).float()
        state = torch.where(up, one, torch.where(down, -one, state))
    return cnt


def down_crossings(h, lo, hi):
    return up_crossings(-h, -hi, -lo)


# ====================================================================== per-person geometry
def _fill(X, vt):
    """Replace untracked frames by the nearest tracked one (forward, then backward fill)."""
    n, T = vt.shape
    idx = torch.arange(T, device=X.device)[None].expand(n, T)
    last = torch.where(vt, idx, torch.full_like(idx, -1)).cummax(1).values           # last tracked frame <= t
    first = vt.float().argmax(1)                                                     # first tracked frame
    src = torch.where(last >= 0, last, first[:, None])
    return X[torch.arange(n, device=X.device)[:, None], src]


def _masked_stat(v, m, q=None, default=1.0):
    x = v.masked_fill(~m, float('nan'))
    r = torch.nanmedian(x, dim=1).values if q is None else torch.nanquantile(x, q, dim=1)
    return torch.nan_to_num(r, nan=default)


class Person:
    def __init__(self, X, dt):
        """X (n, T, 25, 3) metres, dt (n,) seconds per frame."""
        n, T = X.shape[:2]
        self.dt, self.T = dt, T
        vt = X.abs().sum((2, 3)) > 1e-6
        self.valid_t = vt
        self.valid = vt.float().mean(1) >= 0.5
        X = _fill(X, vt)
        chain = sum(norm(X[:, :, a] - X[:, :, b]) for a, b in ((0, 1), (1, 20), (20, 2), (2, 3)))
        L = _masked_stat(chain, vt).clamp(min=0.3)
        self.L = torch.where(self.valid, L, torch.ones_like(L))
        P = X / self.L[:, None, None, None]
        Pp = torch.cat([P[:, :1], P, P[:, -1:]], 1)
        P = (Pp[:, :-2] + Pp[:, 1:-1] + Pp[:, 2:]) / 3                               # 3-frame smoothing
        self.P = P
        Pp = torch.cat([P[:, :1], P, P[:, -1:]], 1)
        V = (Pp[:, 2:] - Pp[:, :-2]) / (2 * dt[:, None, None, None])
        vp = torch.cat([vt[:, :1], vt, vt[:, -1:]], 1)
        self.V = V * (vp[:, :-2] & vp[:, 1:-1] & vp[:, 2:])[..., None, None]          # no speed across gaps
        y = P[..., 1]
        floor = _masked_stat(torch.minimum(y[..., 15], y[..., 19]), vt, q=0.05, default=0.0)
        self.h = y - floor[:, None, None]                                            # height above floor
        leg = (norm(P[:, :, 12] - P[:, :, 13]) + norm(P[:, :, 13] - P[:, :, 14])
               + norm(P[:, :, 16] - P[:, :, 17]) + norm(P[:, :, 17] - P[:, :, 18])) / 2
        self.H0 = (1.05 * _masked_stat(leg, vt)).clamp(min=0.5)
        self.up = P.new_tensor([0.0, 1.0, 0.0])
        r = horiz(P[:, :, 8] - P[:, :, 4])
        r = torch.where((norm(r) > 0.05)[..., None], r, horiz(P[:, :, 16] - P[:, :, 12]))
        r = unit(r)
        for _ in range(2):
            rp = torch.cat([r[:, :1], r, r[:, -1:]], 1)
            r = unit(rp[:, :-2] + rp[:, 1:-1] + rp[:, 2:])
        self.lat = r                                                                 # left -> right shoulder
        f0 = torch.stack([r[..., 2], torch.zeros_like(r[..., 0]), -r[..., 0]], -1)
        foot = unit(horiz(P[:, :, 15] - P[:, :, 14] + P[:, :, 19] - P[:, :, 18]))
        score = (dot(f0, foot) * vt).sum(1) + 0.3 * (-f0[..., 2] * vt).sum(1)
        self.f = f0 * (score.ge(0).float() * 2 - 1)[:, None, None]
        self.tr = P[:, :, 20] - P[:, :, 0]
        self.trunk_ang = ang(self.tr, self.up.expand_as(self.tr))
        self.X = P * self.L[:, None, None, None]                                     # smoothed, metres
        self.Vm = self.V * self.L[:, None, None, None]

    def j(self, i):
        return self.P[:, :, i]

    def v(self, i):
        return self.V[:, :, i]

    def yj(self, i):
        return self.P[:, :, i, 1]

    def hj(self, i):
        return self.h[:, :, i]

    def angle_at(self, a, b, c):
        """Angle a-b-c at joint b."""
        return ang(self.j(a) - self.j(b), self.j(c) - self.j(b))

    def elbow(self, s):
        a = ARM[s]
        return self.angle_at(a['S'], a['E'], a['W'])

    def knee(self, s):
        g = LEG[s]
        return self.angle_at(g['Hip'], g['K'], g['A'])

    def up_like(self, v):
        return self.up.expand_as(v)


# ====================================================================== detectors
SINGLE, INTER, DIRECTED = {}, {}, set()


def single(*ids):
    def deco(fn):
        for i in ids:
            SINGLE[i] = fn
        return fn
    return deco


def inter(*ids, directed=True):
    def deco(fn):
        for i in ids:
            INTER[i] = fn
            if directed:
                DIRECTED.add(i)
        return fn
    return deco


# ---------------------------------------------------------------- body
@single('body.standing')
def _(p, s):
    return AND(gt(p.knee(0), 150, 5), gt(p.knee(1), 150, 5), gt(p.hj(0), 0.8 * p.H0[:, None], 0.03),
               lt(p.trunk_ang, 30, 5))


def _thigh_elev(p, g):
    return (ang(p.j(g['K']) - p.j(g['Hip']), p.up_like(p.j(0))) - 90).abs()


@single('body.sitting')
def _(p, s):
    H0 = p.H0[:, None]
    return AND(between(p.hj(0), 0.45 * H0, 0.75 * H0, 0.03), lt(_thigh_elev(p, LEG[0]), 30, 5),
               lt(_thigh_elev(p, LEG[1]), 30, 5), between(p.knee(0), 60, 120, 5), between(p.knee(1), 60, 120, 5),
               lt(p.trunk_ang, 40, 5))


@single('body.squatting')
def _(p, s):
    ankles = (p.j(14) + p.j(18)) / 2
    return AND(lt(p.hj(0), 0.55 * p.H0[:, None], 0.03), lt(p.knee(0), 90, 5), lt(p.knee(1), 90, 5),
               lt(norm(horiz(p.j(0) - ankles)), 0.4, 0.03))


def _lying(p):
    return AND(gt(p.trunk_ang, 60, 5), lt(p.hj(20), 0.4 * p.H0[:, None], 0.03))


@single('body.lying')
def _(p, s):
    return _lying(p)


@single('body.lowering')
def _(p, s):
    vy, h0 = p.v(0)[..., 1], p.hj(0)
    move = lasting(lt(vy, -0.3, 0.1), p.dt, 0.3)
    drop = win_max(h0, p.dt, 2.0, 'b') - win_min(h0, p.dt, 2.0, 'f')
    gentle = lt((-vy).amax(1), 1.2, 0.15)[:, None]
    return AND(move, gt(drop, 0.3, 0.05), gentle)


@single('body.rising')
def _(p, s):
    vy, h0 = p.v(0)[..., 1], p.hj(0)
    move = lasting(gt(vy, 0.3, 0.1), p.dt, 0.3)
    rise = win_max(h0, p.dt, 2.0, 'f') - win_min(h0, p.dt, 2.0, 'b')
    feet = lt(torch.maximum(p.hj(15), p.hj(19)), 0.08, 0.02)
    return AND(move, gt(rise, 0.3, 0.05), feet)


@single('body.falling')
def _(p, s):
    h0, T = p.hj(0), p.T
    fast = AND(gt(-p.v(0)[..., 1], 1.5, 0.2), gt(-p.v(20)[..., 1], 1.5, 0.2))
    drop = win_max(h0, p.dt, 1.0, 'b') - win_min(h0, p.dt, 1.5, 'f')
    q = max(T // 4, 1)
    end = OR(gt(p.trunk_ang[:, -q:].mean(1), 60, 5), lt(h0[:, -q:].mean(1), 0.4 * p.H0, 0.03))[:, None]
    return AND(win_max(fast, p.dt, 0.6), gt(drop, 0.5, 0.05), end)


@single('body.airborne')
def _(p, s):
    return lasting_frames(AND(gt(p.hj(15), 0.08, 0.02), gt(p.hj(19), 0.08, 0.02)), 3)


def _net_disp(p, j):
    return norm(horiz(p.j(j)[:, -1] - p.j(j)[:, 0]))


@single('body.translating')
def _(p, s):
    sp = norm(horiz(p.v(0)))
    return AND(lasting(gt(sp, 0.5, 0.1), p.dt, 1.0), gt(_net_disp(p, 0), 1.0, 0.1)[:, None])


@single('body.staggering')
def _(p, s):
    # approximation: lateral sway of the shoulders over the pelvis (>= 2 reversals) while moving
    sway = dot(p.j(20) - p.j(0), p.lat)
    moving = win_max(gt(norm(horiz(p.v(0))), 0.15, 0.05), p.dt, 1.0)
    return AND(oscillation(sway, p.dt, 0.2, 2, win=2.0), moving)


# ---------------------------------------------------------------- torso / head
@single('torso.lean_forward')
def _(p, s):
    return AND(between(p.trunk_ang, 20, 50, 5), gt(dot(p.tr, p.f), 0, 0.02))


@single('torso.bend_deep')
def _(p, s):
    return AND(gt(p.trunk_ang, 50, 5), gt(dot(p.tr, p.f), 0, 0.02), 1 - _lying(p))


@single('torso.lean_back')
def _(p, s):
    return AND(gt(p.trunk_ang, 10, 4), lt(dot(p.tr, p.f), 0, 0.02))


@single('torso.twisted')
def _(p, s):
    return gt(ang(horiz(p.j(8) - p.j(4)), horiz(p.j(16) - p.j(12))), 25, 5)


@single('torso.forward_jerk')
def _(p, s):
    q = dot(p.tr, unit(p.f - p.up_like(p.f)))                                       # forward / down
    rise = q - win_min(q, p.dt, 0.3, 'b')
    back = q - win_min(q, p.dt, 0.6, 'f')
    return win_max(AND(between(rise, 0.1, 0.3, 0.03), gt(back, rise - 0.05, 0.03)), p.dt, 0.3)


def _head_vec(p):
    return p.j(3) - p.j(2)


@single('head.nod')
def _(p, s):
    hv = _head_vec(p)
    a = ang(hv, p.tr) * torch.sign(dot(unit(hv) - unit(p.tr), p.f) + 1e-6)          # + forward / down
    rise = a - win_min(a, p.dt, 2.0, 'b')
    ret = a - win_min(a, p.dt, 2.0, 'f')
    return win_max(AND(gt(rise, 10, 3), gt(ret, 7, 3)), p.dt, 1.0)


@single('head.shake')
def _(p, s):
    return oscillation(dot(_head_vec(p), p.lat), p.dt, 0.03, 2, win=1.0)


@single('head.lowered')
def _(p, s):
    hv = _head_vec(p)
    return AND(gt(ang(hv, p.tr), 20, 4), gt(dot(hv, p.f), 0, 0.01))


@single('head.tilt_back')
def _(p, s):
    hv = _head_vec(p)
    return AND(gt(ang(hv, p.tr), 15, 4), lt(dot(hv, p.f), 0, 0.01))


# ---------------------------------------------------------------- hand height / contact zones (side s)
def _H(p, s, k='H'):
    return p.j(ARM[s][k])


@single('hand.below_hip')
def _(p, s):
    return lt(_H(p, s)[..., 1], p.yj(0), 0.03)


@single('hand.torso_level')
def _(p, s):
    yH = _H(p, s)[..., 1]
    return AND(gt(yH, p.yj(0), 0.03), lt(yH, p.yj(20), 0.03))


@single('hand.head_level')
def _(p, s):
    yH = _H(p, s)[..., 1]
    return AND(gt(yH, p.yj(20), 0.03), lt(yH, p.yj(3) + 0.15, 0.03))


@single('hand.above_head')
def _(p, s):
    return gt(_H(p, s)[..., 1], p.yj(3) + 0.15, 0.03)


@single('both_hands.above_head')
def _(p, s):
    return AND(gt(p.yj(7), p.yj(3) + 0.1, 0.03), gt(p.yj(11), p.yj(3) + 0.1, 0.03))


def _at_face(p, s):
    H, T3 = _H(p, s), p.j(3)
    d = torch.minimum(norm(H - T3), norm(_H(p, s, 'T') - T3))
    return AND(lt(d, 0.35, 0.03), gt(dot(H - T3, p.f), 0, 0.02), lt(dot(H - T3, p.lat).abs(), 0.15, 0.03))


def _at_head_side(p, s):
    H = _H(p, s)
    return AND(lt(norm(H - p.j(3)), 0.35, 0.03), gt(dot(H - p.j(3), p.lat).abs(), 0.15, 0.03))


def _at_head_top(p, s):
    H = _H(p, s)
    return AND(lt(norm(H - p.j(3)), 0.4, 0.03), gt(H[..., 1], p.yj(3) + 0.05, 0.03))


def _at_neck(p, s):
    return AND(lt(norm(_H(p, s) - p.j(2)), 0.2, 0.03), 1 - _at_face(p, s))


def _at_torso_front(p, s):
    H = _H(p, s)
    return AND(lt(seg_dist(H, p.j(0), p.j(20)), 0.25, 0.03), between(dot(H - p.j(1), p.f), 0, 0.2, 0.03))


def _at_hip(p, s):
    H = _H(p, s)
    return lt(torch.minimum(norm(H - p.j(12)), norm(H - p.j(16))), 0.25, 0.03)


def _behind_torso(p, s):
    H = _H(p, s)
    sw = norm(p.j(8) - p.j(4))
    return AND(lt(dot(H - p.j(1), p.f), -0.15, 0.03), lt(dot(H - p.j(1), p.lat).abs(), sw, 0.03))


def _at_foot(p, s):
    H = _H(p, s)
    d = torch.stack([norm(H - p.j(j)) for j in (14, 15, 18, 19)]).amin(0)
    return lt(d, 0.3, 0.03)


for _pid, _fn in (('hand.at_face', _at_face), ('hand.at_head_side', _at_head_side),
                  ('hand.at_head_top', _at_head_top), ('hand.at_neck', _at_neck),
                  ('hand.at_torso_front', _at_torso_front), ('hand.at_hip', _at_hip),
                  ('hand.behind_torso', _behind_torso), ('hand.at_foot', _at_foot)):
    SINGLE[_pid] = _fn


@single('hand.at_other_forearm')
def _(p, s):
    a = lt(torch.minimum(norm(p.j(21) - p.j(10)), norm(p.j(7) - p.j(10))), 0.15, 0.03)
    b = lt(torch.minimum(norm(p.j(23) - p.j(6)), norm(p.j(11) - p.j(6))), 0.15, 0.03)
    return OR(a, b)


@single('hand.in_front_space')
def _(p, s):
    H = _H(p, s)
    other = OR(*(fn(p, s) for fn in (_at_face, _at_head_side, _at_head_top, _at_neck, _at_torso_front, _at_hip,
                                      _behind_torso, _at_foot)))
    return AND(between(dot(H - p.j(1), p.f), 0.2, 0.9, 0.03), gt(H[..., 1], p.yj(0), 0.03),
               lt(H[..., 1], p.yj(3), 0.03), 1 - other)


# ---------------------------------------------------------------- two hands
def _hands_dist(p):
    return torch.minimum(norm(p.j(7) - p.j(11)), norm(p.j(21) - p.j(23)))


def _touching(p):
    return lt(_hands_dist(p), 0.12, 0.02)


@single('hands.touching')
def _(p, s):
    return _touching(p)


@single('hands.near')
def _(p, s):
    d = _hands_dist(p)
    return AND(gt(d, 0.12, 0.02), lt(d, 0.35, 0.03))


@single('hands.crossed')
def _(p, s):
    l = lambda j: dot(p.j(j) - p.j(20), p.lat)
    return AND(gt(-torch.sign(l(4)) * l(6), 0.05, 0.02), gt(-torch.sign(l(8)) * l(10), 0.05, 0.02))


@single('hands.rubbing')
def _(p, s):
    rel = (p.j(7) - p.j(11)).permute(0, 2, 1)                                       # (n, 3, T)
    osc = oscillation(rel, p.dt, 0.03, 3, win=1.0, amp_max=0.15).amax(1)
    return AND(osc, win_max(_touching(p), p.dt, 0.5))


@single('hands.clapping')
def _(p, s):
    d = norm(p.j(7) - p.j(11))
    cycles = down_crossings(d, 0.1, 0.25)
    closing = (-tdiff(d, p.dt)).amax(1)
    ok = ((cycles >= 2) & (closing > 1.0)).float()[:, None]
    return win_max(lt(d, 0.15, 0.03), p.dt, 0.5) * ok


def _speed_rel(p, j, sec=0.25):
    return win_mean(norm(p.v(j) - p.v(1)), p.dt, sec)


@single('hands.both_active')
def _(p, s):
    s7, s11 = _speed_rel(p, 7), _speed_rel(p, 11)
    return lasting(AND(gt(s7, 0.3, 0.08), gt(s11, 0.3, 0.08)), p.dt, 0.5)


@single('hands.one_active')
def _(p, s):
    s7, s11 = _speed_rel(p, 7), _speed_rel(p, 11)
    c = OR(AND(gt(s7, 0.3, 0.08), lt(s11, 0.1, 0.04)), AND(gt(s11, 0.3, 0.08), lt(s7, 0.1, 0.04)))
    return lasting(c, p.dt, 0.5)


@single('hands.pull_apart')
def _(p, s):
    d = norm(p.j(7) - p.j(11))
    return win_max(AND(lt(win_min(d, p.dt, 1.0, 'b'), 0.15, 0.03), gt(d, 0.35, 0.03)), p.dt, 0.5)


# ---------------------------------------------------------------- hand shape / pose (side s)
def _thumb_up_val(p, s):
    H = _H(p, s)
    return (_H(p, s, 'Th') - H)[..., 1], (_H(p, s, 'T') - H)[..., 1]


@single('hand.thumb_up')
def _(p, s):
    th, tp = _thumb_up_val(p, s)
    return AND(gt(th, 0.05, 0.015), gt(th, tp, 0.015))


@single('hand.thumb_down')
def _(p, s):
    th, tp = _thumb_up_val(p, s)
    return AND(lt(th, -0.05, 0.015), lt(th, tp, 0.015))


@single('hand.thumb_index_touch')
def _(p, s):
    return lasting_frames(lt(norm(_H(p, s, 'Th') - _H(p, s, 'T')), 0.04, 0.01), 3)


@single('arm.extended')
def _(p, s):
    return gt(p.elbow(s), 150, 5)


@single('arm.flexed')
def _(p, s):
    return lt(p.elbow(s), 80, 5)


def _upper_arm(p, s):
    return p.j(ARM[s]['E']) - p.j(ARM[s]['S'])


@single('arm.forward_raised')
def _(p, s):
    u = _upper_arm(p, s)
    return AND(gt(dot(unit(u), p.f), 0.6, 0.05), gt(ang(u, -p.up_like(u)), 45, 5))


@single('arm.lateral_raised')
def _(p, s):
    u = _upper_arm(p, s)
    out = unit(horiz(p.j(ARM[s]['S']) - p.j(20)))
    return AND(gt(dot(unit(u), out), 0.6, 0.05), gt(ang(u, -p.up_like(u)), 60, 5))


@single('hand.static_hold')
def _(p, s):
    a = ARM[s]
    sr = _speed_rel(p, a['H'], 0.3)
    c = AND(gt(_H(p, s)[..., 1], p.yj(0) + 0.1, 0.03), lt(sr, 0.08, 0.03), 1 - _at_hip(p, s))
    return lasting(c, p.dt, 1.0)


@single('hand.wrist_twist')
def _(p, s):
    H = _H(p, s)
    ax = unit(H - _H(p, s, 'W'))
    n = torch.cross(_H(p, s, 'T') - H, _H(p, s, 'Th') - H, dim=-1)
    up = p.up_like(ax)
    r0 = unit(up - dot(up, ax)[..., None] * ax)
    r1 = torch.cross(ax, r0, dim=-1)
    th = torch.atan2(dot(n, r1), dot(n, r0))
    dth = torch.diff(th, dim=-1)
    dth = (dth + math.pi) % (2 * math.pi) - math.pi
    th = torch.rad2deg(torch.cat([th[..., :1], th[..., :1] + dth.cumsum(-1)], -1))   # unwrapped
    return oscillation(th, p.dt, 30, 2, win=1.5)


# ---------------------------------------------------------------- hand motion (side s)
@single('hand.move_away')
def _(p, s):
    q = norm(_H(p, s) - p.j(1))
    rise = win_max(q, p.dt, 1.5, 'f') - q
    return win_max(gt(rise, 0.4, 0.05), p.dt, 1.0, 'f')


@single('hand.fast_thrust')
def _(p, s):
    a = ARM[s]
    el = p.elbow(s)
    er = win_max(el, p.dt, 0.4, 'f') - el
    c = AND(gt(dot(p.v(a['H']), p.f), 2, 0.3), gt(win_max(er, p.dt, 0.4, 'b'), 40, 5))
    return win_max(c, p.dt, 0.4)


@single('hand.downward_strike')
def _(p, s):
    a = ARM[s]
    yH = _H(p, s)[..., 1]
    high = win_max(yH - p.yj(20), p.dt, 0.4, 'b')
    drop = win_max(yH, p.dt, 0.4, 'b') - win_min(yH, p.dt, 0.4, 'f')
    c = AND(gt(-p.v(a['H'])[..., 1], 2.5, 0.3), gt(high, 0, 0.03), gt(drop, 0.4, 0.05))
    return win_max(c, p.dt, 0.4)


@single('hand.upward_flick')
def _(p, s):
    a = ARM[s]
    low = win_max(p.yj(20) - _H(p, s)[..., 1], p.dt, 0.4, 'b')
    return win_max(AND(gt(p.v(a['H'])[..., 1], 2, 0.3), gt(low, 0, 0.03)), p.dt, 0.3)


@single('hand.horizontal_swing')
def _(p, s):
    a = ARM[s]
    H = _H(p, s)
    fast = win_max(gt(norm(horiz(p.v(a['H']))), 2, 0.3), p.dt, 1.0)
    step = norm(horiz(torch.diff(H, dim=1, prepend=H[:, :1])))
    path = win_mean(step, p.dt, 1.0) * _b(1.0 / p.dt, step)
    c = dot(H - p.j(20), p.lat)
    cross = AND(gt(win_max(c, p.dt, 1.0), 0.05, 0.02), lt(win_min(c, p.dt, 1.0), -0.05, 0.02))
    return AND(fast, gt(path, 0.8, 0.08), cross)


@single('hand.lateral_oscillation')
def _(p, s):
    a = ARM[s]
    sig = dot(_H(p, s) - p.j(a['E']), p.lat)
    return oscillation(sig, p.dt, 0.1, 4, win=1.5, fmin=0.8, fmax=4.0)


@single('hand.vertical_oscillation')
def _(p, s):
    sig = _H(p, s)[..., 1] - p.yj(1)
    return oscillation(sig, p.dt, 0.1, 4, win=1.5, amp_max=0.5)


@single('hand.small_fast_oscillation')
def _(p, s):
    a = ARM[s]
    rel = (_H(p, s) - p.j(a['E'])).permute(0, 2, 1)                                 # (n, 3, T)
    osc = oscillation(rel, p.dt, 0.02, 4, win=0.5, amp_max=0.1, fmin=2.0).amax(1)
    return lasting(osc, p.dt, 1.0)


@single('arm.circling')
def _(p, s):
    a = ARM[s]
    v = _H(p, s) - p.j(a['S'])                                                       # (n, T, 3)
    m = p.valid_t.float()[..., None]
    mu = (v * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
    c = v - mu
    C = (c * m).transpose(1, 2) @ (c * m) / m.sum(1, keepdim=True).clamp(min=1)
    _, E = torch.linalg.eigh(C + 1e-6 * torch.eye(3, device=v.device))
    e1, e2 = E[..., 2], E[..., 1]                                                    # two largest axes
    th = torch.atan2(dot(c, e2[:, None]), dot(c, e1[:, None]))
    dth = (torch.diff(th, dim=-1) + math.pi) % (2 * math.pi) - math.pi
    cum = torch.cat([torch.zeros_like(th[:, :1]), dth.cumsum(-1)], -1)
    sweep = torch.rad2deg(cum.amax(1) - cum.amin(1))
    arm_len = _masked_stat(sum(norm(p.j(x) - p.j(y)) for x, y in ((a['S'], a['E']), (a['E'], a['W']),
                                                                   (a['W'], a['H']))), p.valid_t)
    return AND(gt(norm(v), 0.6 * arm_len[:, None], 0.03), gt(sweep, 270, 20)[:, None])


# ---------------------------------------------------------------- legs
@single('leg.single_support')
def _(p, s):
    f, o = LEG[s]['F'], LEG[1 - s]['F']
    return lasting_frames(AND(gt(p.hj(f), 0.1, 0.02), lt(p.hj(o), 0.05, 0.02)), 5)


@single('leg.alternating_lifts')
def _(p, s):
    n15, n19 = up_crossings(p.hj(15), 0.05, 0.1), up_crossings(p.hj(19), 0.05, 0.1)
    ok = ((n15 >= 2) & (n19 >= 2) & (_net_disp(p, 0) < 0.3)).float()[:, None]
    return OR(gt(p.hj(15), 0.1, 0.02), gt(p.hj(19), 0.1, 0.02)) * ok


@single('leg.heel_to_buttock')
def _(p, s):
    g = LEG[s]
    return AND(lt(p.knee(s), 60, 5), lt(dot(p.j(g['A']) - p.j(g['K']), p.f), -0.15, 0.03),
               gt(p.yj(g['A']), p.yj(g['K']) - 0.1, 0.03))


@single('leg.kick_forward')
def _(p, s):
    g = LEG[s]
    kn = p.knee(s)
    kr = win_max(kn, p.dt, 0.5, 'f') - kn
    c = AND(gt(dot(p.v(g['F']), p.f), 2, 0.3), gt(win_max(kr, p.dt, 0.5, 'b'), 30, 5),
            gt(win_max(p.hj(g['F']), p.dt, 0.5), 0.3, 0.05))
    return win_max(c, p.dt, 0.4)


@single('leg.kick_lateral')
def _(p, s):
    g = LEG[s]
    out = unit(horiz(p.j(g['Hip']) - p.j(0)))
    d = p.j(g['A']) - p.j(g['Hip'])
    abd = torch.rad2deg(torch.atan2(dot(d, out), -d[..., 1]))
    c = AND(gt(dot(p.v(g['F']), out), 1.5, 0.3), gt(win_max(abd, p.dt, 0.5), 30, 5))
    return win_max(c, p.dt, 0.4)


# ---------------------------------------------------------------- whole-clip motion amount
def _active_time(p):
    ms = norm(p.V - p.V[:, :, 1:2]).mean(-1)                                         # (n, T) mean joint speed
    return (gt(ms, 0.3, 0.08) * p.valid_t).sum(1) * p.dt


@single('body.brief_motion')
def _(p, s):
    return lt(_active_time(p), 1.2, 0.15)[:, None].expand(-1, p.T)


@single('body.sustained_motion')
def _(p, s):
    return gt(_active_time(p), 2.5, 0.2)[:, None].expand(-1, p.T)


# ---------------------------------------------------------------- interactions (A acts on B; metres / mean L)
def _Lp(A, B):
    return ((A.L + B.L) / 2)[:, None]


def _d(A, i, B, j):
    return norm(A.X[:, :, i] - B.X[:, :, j]) / _Lp(A, B)


@inter('inter.approaching', 'inter.separating', directed=False)
def _(A, B, pid):
    d = norm(horiz(A.X[:, :, 0] - B.X[:, :, 0])) / _Lp(A, B)
    vd = tdiff(d, A.dt)
    c = lt(vd, -0.3, 0.1) if pid == 'inter.approaching' else gt(vd, 0.3, 0.1)
    return lasting(c, A.dt, 1.0)


@inter('inter.facing', 'inter.same_heading', directed=False)
def _(A, B, pid):
    u = unit(horiz(B.X[:, :, 0] - A.X[:, :, 0]))
    if pid == 'inter.facing':
        return AND(gt(dot(A.f, u), 0.7, 0.05), gt(dot(B.f, -u), 0.7, 0.05))
    return AND(gt(dot(A.f, B.f), 0.7, 0.05), gt(dot(u, A.f).abs(), 0.7, 0.05))


HANDS = (7, 11, 21, 23)


@inter('inter.hand_to_other_hand', directed=False)
def _(A, B, pid):
    d = torch.stack([_d(A, i, B, j) for i in HANDS for j in HANDS]).amin(0)
    return lt(d, 0.15, 0.03)


def _hand_head_shoulder(A, B):
    d = torch.stack([_d(A, i, B, j) for i in (7, 11) for j in (2, 3, 4, 8, 20)]).amin(0)
    return lt(d, 0.2, 0.03)


def _hand_trunk(A, B):
    Lp = _Lp(A, B)
    out = []
    for h in (7, 11):
        H = A.X[:, :, h]
        d = torch.minimum(seg_dist(H, B.X[:, :, 0], B.X[:, :, 1]),
                          torch.minimum(norm(H - B.X[:, :, 12]), norm(H - B.X[:, :, 16]))) / Lp
        rel = H - B.X[:, :, 1]
        side = OR(gt(dot(rel, B.f) / Lp, 0, 0.02), gt(dot(rel, B.lat).abs() / Lp, 0.15, 0.03))
        out.append(AND(lt(d, 0.25, 0.03), side))
    return OR(*out)


def _hand_back(A, B, h):
    Lp = _Lp(A, B)
    rel = A.X[:, :, h] - B.X[:, :, 1]
    sw = norm(B.X[:, :, 8] - B.X[:, :, 4]) / Lp
    # + proximity (< 0.5 L, not in the definition) so that a person merely standing behind does not count
    return AND(lt(dot(rel, B.f) / Lp, -0.1, 0.03), lt(dot(rel, B.lat).abs() / Lp, sw, 0.03),
               lt(norm(rel) / Lp, 0.5, 0.05))


def _foot_leg(A, B):
    d = torch.stack([_d(A, i, B, j) for i in (14, 15, 18, 19) for j in (13, 14, 15, 17, 18, 19)]).amin(0)
    return lt(d, 0.25, 0.03)


@inter('inter.hand_to_other_head_shoulder')
def _(A, B, pid):
    return _hand_head_shoulder(A, B)


@inter('inter.hand_to_other_trunk')
def _(A, B, pid):
    return _hand_trunk(A, B)


@inter('inter.hand_to_other_back')
def _(A, B, pid):
    return OR(_hand_back(A, B, 7), _hand_back(A, B, 11))


@inter('inter.arms_around_other')
def _(A, B, pid):
    return AND(_hand_back(A, B, 7), _hand_back(A, B, 11), lt(_d(A, 1, B, 1), 1.0, 0.05))


@inter('inter.foot_to_other_leg')
def _(A, B, pid):
    return _foot_leg(A, B)


@inter('inter.other_displaced')
def _(A, B, pid):
    contact = OR(_hand_head_shoulder(A, B), _hand_trunk(A, B), _hand_back(A, B, 7), _hand_back(A, B, 11),
                 _foot_leg(A, B))
    after = win_max(contact, A.dt, 0.5, 'b')
    away = unit(B.X[:, :, 20] - A.X[:, :, 20])
    v_away = dot(B.Vm[:, :, 20], away) / _Lp(A, B)
    acc = tdiff(v_away, A.dt)
    d = _d(A, 20, B, 20)
    disp = win_max(d, A.dt, 0.5, 'f') - d
    return win_max(AND(after, gt(acc, 3, 0.5), gt(win_max(disp, A.dt, 0.5, 'b'), 0.2, 0.05)), A.dt, 0.5)


@inter('inter.head_to_other_head', directed=False)
def _(A, B, pid):
    return lt(_d(A, 3, B, 3), 0.4, 0.03)


@inter('inter.pointing_at_other')
def _(A, B, pid):
    out = []
    for s in (0, 1):
        a = ARM[s]
        S, T = A.X[:, :, a['S']], A.X[:, :, a['T']]
        out.append(AND(gt(A.elbow(s), 150, 5), lt(ang(T - S, B.X[:, :, 3] - S), 15, 3),
                       gt(norm(T - B.X[:, :, 3]) / _Lp(A, B), 1.0, 0.05)))
    return OR(*out)


@inter('inter.mutual_reach', directed=False)
def _(A, B, pid):
    def reach(P, Q):
        d = torch.stack([_d(P, h, Q, 1) for h in (7, 11)]).amin(0)
        return win_max(gt(d - win_min(d, P.dt, 1.0, 'f'), 0.3, 0.05), P.dt, 1.0)
    return AND(reach(A, B), reach(B, A))


TWO_PERSON = 'inter.two_person'


def _two_person(B):
    extent = _masked_stat(B.yj(3) - torch.minimum(B.yj(15), B.yj(19)), B.valid_t, default=0.0)
    ok = B.valid & (extent > 0.5 * B.H0)
    return ok.float()[:, None].expand(-1, B.T)


# ====================================================================== labeler
class PrimitiveLabeler:
    def __init__(self, prims):
        self.ids = [p['id'] for p in prims]
        self.lateral = [kg_lib.side_kind(i) is not None for i in self.ids]
        self.implemented = torch.tensor([i in SINGLE or i in INTER or i == TWO_PERSON for i in self.ids])
        self.missing = [i for i, ok in zip(self.ids, self.implemented.tolist()) if not ok]

    @torch.no_grad()
    def __call__(self, x, dt, pool=4):
        """x (n, 3, T, 25, M) metres (NTU coordinates), dt (n,) -> (n, 2, K, T // pool) in [0, 1]."""
        n, _, T, _, M = x.shape
        persons = [Person(x[..., m].permute(0, 2, 3, 1).float(), dt.float()) for m in range(M)]
        out = x.new_zeros((n, 2, len(self.ids), T), dtype=torch.float32)
        A = persons[0]
        B = persons[1] if M > 1 else None
        both = (A.valid_t & B.valid_t & A.valid[:, None] & B.valid[:, None]).float() if B is not None else None
        for k, pid in enumerate(self.ids):
            if pid == TWO_PERSON:
                if B is not None:
                    out[:, :, k] = _two_person(B)[:, None]
            elif pid in SINGLE:
                fn = SINGLE[pid]
                for s in ((0, 1) if self.lateral[k] else (None,)):
                    v = torch.stack([fn(p, 0 if s is None else s) * (p.valid_t & p.valid[:, None]).float()
                                     for p in persons]).amax(0)
                    if s is None:
                        out[:, :, k] = v[:, None]
                    else:
                        out[:, s, k] = v
            elif pid in INTER and B is not None:
                fn = INTER[pid]
                v = fn(A, B, pid)
                if pid in DIRECTED:
                    v = torch.maximum(v, fn(B, A, pid))
                out[:, :, k] = (v * both)[:, None]
        out = out.nan_to_num(0.0).clamp(0, 1)
        if pool > 1:
            T2 = T // pool
            out = out[..., :T2 * pool].reshape(n, 2, len(self.ids), T2, pool).amax(-1)
        return out


# ====================================================================== data: same crop as the cached view, no rotation
def crop_resize(data, valid_frame_num, p_interval, window):
    """feeders.tools.valid_crop_resize (same RNG calls, so the same crop after the same seeding), also returning
    the number of original frames the window covers."""
    C, T, V, M = data.shape
    begin, end = 0, valid_frame_num
    valid_size = end - begin
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data[:, begin + bias:end - bias, :, :]
        cropped_length = data.shape[1]
    else:
        p = np.random.rand() * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64), valid_size)
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data[:, begin + bias:begin + bias + cropped_length, :, :]
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)[None, None]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear', align_corners=False).squeeze()
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()
    return data, int(cropped_length)


class LabelView(Dataset):
    """Like main_nesy.FixedView (same seeding, same crop as the cached cube of sample i) but without the random
    rotation; returns (skeleton (3, T, 25, M), label, seconds per frame)."""

    def __init__(self, ds, seed):
        self.ds, self.seed = ds, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.seed + int(i)
        np.random.seed(s % 2 ** 32)
        random.seed(s)
        torch.manual_seed(s)
        ds = self.ds
        assert not (ds.bone or ds.vel), 'labels need the joint modality'
        data = np.array(ds.data[i])
        valid = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        data, length = crop_resize(data, valid, ds.p_interval, ds.window_size)
        return data, ds.label[i], np.float32(length / FPS / ds.window_size)


# ====================================================================== rule oracle
def oracle_scores(reasoner, prim, classes, impl):
    """Rule scores of `classes` on primitive truth values prim (n, 2, K, T) in [0, 1] (literals of primitives
    without a detector are left out). Presence = max over time."""
    y = prim.float().clamp(0.02, 0.98)
    logit = torch.log(y) - torch.log1p(-y)
    return reasoner(logit, classes, lit_mask=impl.float(), P=y.amax(-1))


# ====================================================================== self-check (Colab)
def _main():
    import importlib
    import yaml
    from torch.utils.data import DataLoader, Subset
    from model.nesy import RuleReasoner
    from main_nesy import SPLITS

    ap = argparse.ArgumentParser(description='check the primitive labels and the rule oracle')
    ap.add_argument('--config', required=True)
    ap.add_argument('--n', type=int, default=3000, help='samples per part (train subset / unseen test subset)')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ma = cfg['model_args']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    split = cfg['split']
    n_cls = 60 if split.startswith('ntu60') else 120
    unseen = sorted(SPLITS[split])
    seen = [c for c in range(n_cls) if c not in unseen]
    prims = __import__('json').load(open(ma['primitive_path'], encoding='utf-8'))
    kg = kg_lib.load_kg(ma['primitive_path'], ma['rules_path'], n_cls)
    lab = PrimitiveLabeler(prims)
    impl = lab.implemented.to(dev)
    print(f'{len(lab.ids)} primitives, {int(impl.sum())} with a detector; missing: {lab.missing or "-"}')
    reasoner = RuleReasoner(kg['R'], kg['seqs'], kg['groups'], kg['implications']).to(dev).eval()
    mod, _, cls = cfg['feeder'].rpartition('.')
    Feeder = getattr(importlib.import_module(mod), cls)
    seed = cfg.get('seed', 1)

    def run(part, keep):
        ds = LabelView(Feeder(**cfg[f'{part}_feeder_args']), seed)
        idx = [i for i in range(len(ds)) if keep(int(ds.ds.label[i]))]
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(idx, min(a.n, len(idx)), replace=False).tolist())
        L, Y = [], []
        for x, y, dt in DataLoader(Subset(ds, idx), batch_size=a.batch, num_workers=a.workers):
            L.append(lab(x.to(dev), dt.to(dev)))
            Y.append(y.long().to(dev))
        return torch.cat(L), torch.cat(Y)

    def report(name, prim, y, classes):
        pres = prim.amax(-1)                                                         # (n, 2, K)
        on = (pres.amax(1) > 0.5).float()                                            # (n, K)
        rate = on.mean(0)
        order = rate.argsort(descending=True).tolist()
        print(f'\n== {name}: {len(y)} samples')
        print('  presence rate (any side, any time):')
        print('   ', ', '.join(f'{lab.ids[k]} {rate[k]:.2f}' for k in order if impl[k]))
        never = [lab.ids[k] for k in order if impl[k] and rate[k] < 0.005]
        always = [lab.ids[k] for k in order if impl[k] and rate[k] > 0.8]
        print(f'  never on (<0.5%): {never or "-"}\n  almost always on (>80%): {always or "-"}')
        R = torch.as_tensor(kg['R'], device=dev)
        req, forb = [], []
        for c in classes:
            m = y == c
            if m.any():
                req.append(((on[m] * (R[c] > 0) * impl).sum(1) / ((R[c] > 0) * impl).sum().clamp(min=1)).mean())
                forb.append(((on[m] * (R[c] < 0) * impl).sum(1) / ((R[c] < 0) * impl).sum().clamp(min=1)).mean())
        print(f'  rule agreement: require literals on {torch.stack(req).mean():.3f} | '
              f'forbid literals on {torch.stack(forb).mean():.3f} (want high | low)')
        cl = torch.tensor(classes, device=dev)
        s = torch.cat([oracle_scores(reasoner, prim[i:i + 512], cl, impl) for i in range(0, len(y), 512)])
        pred = cl[s.argmax(1)]
        print(f'  rule oracle accuracy over these {len(classes)} classes: {(pred == y).float().mean():.4f}')
        per = [(kg['names'][c], float((pred[y == c] == c).float().mean())) for c in classes if (y == c).any()]
        per.sort(key=lambda t: t[1])
        print('  per class (worst first):', ', '.join(f'{n} {v:.2f}' for n, v in per[:15]))

    prim, y = run('train', lambda c: c in seen)
    report(f'train (seen classes, fixed view of seed {seed})', prim, y, seen)
    prim, y = run('test', lambda c: c in unseen)
    report('test, unseen classes = ZSL oracle', prim, y, unseen)


if __name__ == '__main__':
    _main()
