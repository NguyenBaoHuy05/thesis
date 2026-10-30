"""Per-frame differentiable geometric signals (pipeline v4, docs/architecture_v4.html).

torch port of the per-frame formulas of sip/measures.py, extended with velocities, kept
PER FRAME (no temporal statistic) and differentiable w.r.t. the joint coordinates, so that
(1) the concept layer can quantify them over time with learned thresholds and
(2) the counterfactual generator can edit the skeleton by gradient descent.

Input ``x``: (N, 3, T, V=25, M=2) exactly as the HyperGCN feeder hands it to the model
(joint 20 absolute, every other joint relative to joint 20; y = UP; metres). See the
module docstring of sip/measures.py for the coordinate frame and the validity rules,
which are reproduced here (same constants).

Main actor = person slot 0, unless slot 0 is never valid and slot 1 is (as in v3).
Distances / heights are divided by the main actor's body size L (median spine length,
treated as a constant: no gradient), angles are in degrees, velocities are per
(resampled) frame. Velocity signals at frame t use frames t and t+1; the last frame is
masked out.

``geo_signals`` returns S (N, NS, T), mask (N, NS, T) bool and an info dict. The mask
says on which frames a signal is defined; quantifiers (sip/ccp.py) only look there.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sip.measures import (
    SPINE_BASE, SPINE_MID, NECK, HEAD,
    L_SHOULDER, L_ELBOW, L_WRIST, L_HAND,
    R_SHOULDER, R_ELBOW, R_WRIST, R_HAND,
    L_HIP, L_KNEE, L_ANKLE, L_FOOT,
    R_HIP, R_KNEE, R_ANKLE, R_FOOT,
    SPINE_SHOULDER, L_HAND_TIP, R_HAND_TIP,
    UP, SPINE_RATIO_LO, SPINE_RATIO_HI, BLEND_MARGIN, FAR_VALUE,
)

__all__ = ["SIGNALS", "SIGNAL_UNITS", "SIGNAL_INDEX", "NUM_SIGNALS",
           "to_absolute", "validity", "select_main", "geo_signals", "bone_lengths", "BONES"]

_EPS = 1e-6
_DEFAULT_BODY_SIZE = 0.5

# (name, unit, mask kind): "frame" = valid main-actor frames, "vel" = frames t with t and
# t+1 valid, "pairvel" = both persons valid at t and t+1.
_SIGNAL_SPECS = [
    ("hand_head", "L", "frame"),            # min over hands |p_hand - p_head|
    ("hand_chest", "L", "frame"),           # min over hands |p_hand - chest|
    ("hands_dist", "L", "frame"),           # |p_7 - p_11|
    ("hand_body", "L", "frame"),            # min over hands to torso / leg joints
    ("hand_above_head", "L", "frame"),      # max(y_7, y_11) - y_3
    ("hand_above_shoulder", "L", "frame"),  # max over arms y_hand - y_shoulder
    ("hand_reach_fwd", "L", "frame"),       # max over arms (wrist - shoulder) . forward
    ("hand_lateral", "L", "frame"),         # max over arms outward lateral wrist offset
    ("arms_crossed", "L", "frame"),         # (p_7 - p_11) . lateral (> 0: hands crossed)
    ("elbow_max", "deg", "frame"),          # max over arms elbow angle (180 = straight)
    ("elbow_min", "deg", "frame"),          # min over arms elbow angle
    ("wrist_speed", "L/frame", "vel"),      # max over wrists body-relative speed
    ("hand_vy_max", "L/frame", "vel"),      # max over hands body-relative vertical velocity
    ("hand_vy_min", "L/frame", "vel"),      # min over hands body-relative vertical velocity
    ("hand_fwd_vmax", "L/frame", "vel"),    # max over hands body-relative forward velocity
    ("hand_fwd_vmin", "L/frame", "vel"),    # min over hands body-relative forward velocity
    ("hand_chest_v", "L/frame", "vel"),     # d/dt hand_chest
    ("hands_dist_v", "L/frame", "vel"),     # d/dt hands_dist
    ("raised_hand_speed", "L/frame", "vel"),  # max over hands speed * sigmoid(hand above shoulder)
    ("arm_sym", "cos", "vel"),              # cos(v_L, mirror(v_R)) * motion gate
    ("head_pitch", "deg", "frame"),         # neck->head angle from up, > 0 = forward / down
    ("head_pitch_speed", "deg/frame", "vel"),
    ("head_lat_speed", "L/frame", "vel"),   # |d/dt (head - neck) . lateral|
    ("head_speed", "L/frame", "vel"),       # |d/dt (head - spine shoulder)|
    ("torso_pitch", "deg", "frame"),        # spine (0 -> 20) angle from up
    ("shoulder_yaw_speed", "deg/frame", "vel"),
    ("hip_height", "L", "frame"),           # y_0 - mean ankle y
    ("hip_vy", "L/frame", "vel"),           # d/dt hip_height
    ("hip_lat_speed", "L/frame", "vel"),    # |d/dt p_0 . lateral|
    ("knee_min", "deg", "frame"),           # min over legs knee angle
    ("foot_height_diff", "L", "frame"),     # |y_15 - y_19|
    ("feet_lift", "L", "frame"),            # min foot height above the clip's lowest foot
    ("foot_vy", "L/frame", "vel"),          # max over feet |vertical velocity|
    ("ankle_speed", "L/frame", "vel"),      # max over ankles horizontal speed
    ("leg_alt_speed", "L/frame", "vel"),    # |d/dt (p_14 - p_18) . forward|
    ("body_fwd_v", "L/frame", "vel"),       # spine-shoulder velocity . forward
    ("body_speed", "L/frame", "vel"),       # mean over joints body-relative speed
    ("person_dist", "L", "frame"),          # |p_20(A) - p_20(B)|, FAR_VALUE if B absent
    ("person_dist_v", "L/frame", "pairvel"),  # d/dt person_dist
    ("contact_dist", "L", "frame"),         # min hand-to-other-person distance, FAR if absent
    ("second_present", "frac", "frame"),    # 1 if the partner is valid at t
]
SIGNALS = [s[0] for s in _SIGNAL_SPECS]
SIGNAL_UNITS = {s[0]: s[1] for s in _SIGNAL_SPECS}
SIGNAL_INDEX = {s: i for i, s in enumerate(SIGNALS)}
NUM_SIGNALS = len(SIGNALS)

# NTU bones (0-based), from feeders/bone_pairs.ntu_pairs without the (21, 21) self pair
BONES = [(a - 1, b - 1) for a, b in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12))]

_TORSO_LEG_JOINTS = [SPINE_BASE, SPINE_MID, SPINE_SHOULDER, L_HIP, L_KNEE, L_ANKLE,
                     R_HIP, R_KNEE, R_ANKLE]
_CONTACT_HANDS = [L_HAND, R_HAND, L_HAND_TIP, R_HAND_TIP]


def _norm(v):
    return torch.sqrt((v * v).sum(-1) + 1e-10)


def _angle_deg(a, b, c):
    """Angle a-b-c at vertex b in degrees (atan2 form: finite gradient everywhere)."""
    v1, v2 = a - b, c - b
    cr = _norm(torch.cross(v1, v2, dim=-1))
    dot = (v1 * v2).sum(-1)
    return torch.atan2(cr, dot) * (180.0 / math.pi)


def _vel(s):
    """(N, T) -> (N, T): s[t+1] - s[t], last frame 0."""
    return F.pad(s[:, 1:] - s[:, :-1], (0, 1))


def _pair_mask(m):
    """(N, T) bool -> (N, T): m[t] & m[t+1], last frame False (no bool F.pad)."""
    return torch.cat([m[:, 1:] & m[:, :-1], torch.zeros_like(m[:, :1])], dim=1)


def _vel3(p):
    """(N, T, ..., 3) -> same shape: p[t+1] - p[t], last frame 0."""
    d = p[:, 1:] - p[:, :-1]
    return torch.cat([d, torch.zeros_like(p[:, :1])], dim=1)


def to_absolute(x):
    """(N, 3, T, V, M) feeder input -> (N, M, T, V, 3) absolute coordinates."""
    J = x.permute(0, 4, 2, 3, 1)
    V = J.shape[3]
    keep = torch.ones(V, dtype=J.dtype, device=J.device)
    keep[SPINE_SHOULDER] = 0.0
    return J + J[..., SPINE_SHOULDER:SPINE_SHOULDER + 1, :] * keep[:, None]


@torch.no_grad()
def validity(x):
    """valid (N, M, T) bool and body size L (N, M); same rules as sip/measures.py."""
    J = x.permute(0, 4, 2, 3, 1)
    A = to_absolute(x)
    raw = J.abs().sum(dim=(-1, -2)) > _EPS                                    # (N,M,T)
    spine = _norm(A[..., SPINE_SHOULDER, :] - A[..., SPINE_BASE, :])          # (N,M,T)
    med = torch.nanmedian(torch.where(raw, spine, torch.full_like(spine, float('nan'))), dim=-1).values
    ok = torch.isfinite(med) & (med > 1e-3)
    L = torch.where(ok, med, torch.full_like(med, _DEFAULT_BODY_SIZE))
    Lb = L[..., None]
    valid = raw & ok[..., None] & (spine > SPINE_RATIO_LO * Lb) & (spine < SPINE_RATIO_HI * Lb)
    # frames next to a raw-zero frame are resize blends (see sip/measures.py)
    N, M, T = raw.shape
    zero = (~raw).float().reshape(N * M, 1, T)
    near = F.max_pool1d(zero, kernel_size=2 * BLEND_MARGIN + 1, stride=1, padding=BLEND_MARGIN)
    valid = valid & ~(near.reshape(N, M, T) > 0)
    return valid, L


def select_main(valid):
    """(N, M, T) -> main slot (N,) long: 0 unless slot 0 is never valid and slot 1 is."""
    has0 = valid[:, 0].any(-1)
    has1 = valid[:, 1].any(-1) if valid.shape[1] > 1 else torch.zeros_like(has0)
    return torch.where(~has0 & has1, torch.ones_like(has0, dtype=torch.long),
                       torch.zeros_like(has0, dtype=torch.long))


def bone_lengths(A):
    """(..., V, 3) -> (..., n_bones)."""
    a = torch.tensor([b[0] for b in BONES], device=A.device)
    b = torch.tensor([b[1] for b in BONES], device=A.device)
    return _norm(A[..., a, :] - A[..., b, :])


def geo_signals(x, valid=None, L=None, main=None):
    """x (N, 3, T, V, M) -> S (N, NS, T), mask (N, NS, T) bool, info dict.

    valid / L / main can be passed (e.g. those of the ORIGINAL clip when x is an edited
    counterfactual) so that the validity of frames and the body size stay fixed.
    info: valid_all (N,M,T), L_all (N,M), main (N,), valid (N,T), L (N,), A (N,T,V,3) main
    actor absolute coordinates.
    """
    if valid is None or L is None:
        valid, L = validity(x)
    if main is None:
        main = select_main(valid)
    A_all = to_absolute(x)                       # (N,M,T,V,3)
    N, M, T = A_all.shape[:3]
    ar = torch.arange(N, device=x.device)
    A = A_all[ar, main]                          # (N,T,V,3)
    vm = valid[ar, main]                         # (N,T)
    Lm = L[ar, main].detach()                    # (N,)
    Lb = Lm[:, None]
    y = A[..., UP]

    vel_mask = _pair_mask(vm)                    # (N,T)
    sig, msk = {}, {}

    def put(name, value, mask):
        sig[name] = value
        msk[name] = mask

    lat = A[..., R_SHOULDER, :] - A[..., L_SHOULDER, :]
    lat_hat = lat / (_norm(lat)[..., None] + _EPS)
    fwd = torch.stack([lat[..., 2], torch.zeros_like(lat[..., 0]), -lat[..., 0]], dim=-1)
    fwd = fwd / (_norm(fwd)[..., None] + _EPS)
    R = A - A[..., SPINE_SHOULDER:SPINE_SHOULDER + 1, :]      # body-relative

    pL, pR = A[..., L_HAND, :], A[..., R_HAND, :]
    head = A[..., HEAD, :]
    chest = 0.5 * (A[..., SPINE_SHOULDER, :] + A[..., SPINE_MID, :])

    # --- hand proximity / height / arm configuration
    put("hand_head", torch.minimum(_norm(pL - head), _norm(pR - head)) / Lb, vm)
    hand_chest = torch.minimum(_norm(pL - chest), _norm(pR - chest)) / Lb
    put("hand_chest", hand_chest, vm)
    hands_dist = _norm(pL - pR) / Lb
    put("hands_dist", hands_dist, vm)
    body = A[..., _TORSO_LEG_JOINTS, :]                                   # (N,T,J,3)
    dLb = _norm(pL[..., None, :] - body).amin(-1)
    dRb = _norm(pR[..., None, :] - body).amin(-1)
    put("hand_body", torch.minimum(dLb, dRb) / Lb, vm)
    put("hand_above_head", (torch.maximum(y[..., L_HAND], y[..., R_HAND]) - y[..., HEAD]) / Lb, vm)
    hL = y[..., L_HAND] - y[..., L_SHOULDER]
    hR = y[..., R_HAND] - y[..., R_SHOULDER]
    put("hand_above_shoulder", torch.maximum(hL, hR) / Lb, vm)
    reach_l = ((A[..., L_WRIST, :] - A[..., L_SHOULDER, :]) * fwd).sum(-1)
    reach_r = ((A[..., R_WRIST, :] - A[..., R_SHOULDER, :]) * fwd).sum(-1)
    put("hand_reach_fwd", torch.maximum(reach_l, reach_r) / Lb, vm)
    lat_l = -((A[..., L_WRIST, :] - A[..., L_SHOULDER, :]) * lat_hat).sum(-1)
    lat_r = ((A[..., R_WRIST, :] - A[..., R_SHOULDER, :]) * lat_hat).sum(-1)
    put("hand_lateral", torch.maximum(lat_l, lat_r) / Lb, vm)
    put("arms_crossed", ((pL - pR) * lat_hat).sum(-1) / Lb, vm)
    el_l = _angle_deg(A[..., L_SHOULDER, :], A[..., L_ELBOW, :], A[..., L_WRIST, :])
    el_r = _angle_deg(A[..., R_SHOULDER, :], A[..., R_ELBOW, :], A[..., R_WRIST, :])
    put("elbow_max", torch.maximum(el_l, el_r), vm)
    put("elbow_min", torch.minimum(el_l, el_r), vm)

    # --- arm motion (body-relative)
    vwL, vwR = _vel3(R[..., L_WRIST, :]), _vel3(R[..., R_WRIST, :])
    spL, spR = _norm(vwL), _norm(vwR)
    put("wrist_speed", torch.maximum(spL, spR) / Lb, vel_mask)
    vhL, vhR = _vel3(R[..., L_HAND, :]), _vel3(R[..., R_HAND, :])
    put("hand_vy_max", torch.maximum(vhL[..., UP], vhR[..., UP]) / Lb, vel_mask)
    put("hand_vy_min", torch.minimum(vhL[..., UP], vhR[..., UP]) / Lb, vel_mask)
    fL, fR = (vhL * fwd).sum(-1), (vhR * fwd).sum(-1)
    put("hand_fwd_vmax", torch.maximum(fL, fR) / Lb, vel_mask)
    put("hand_fwd_vmin", torch.minimum(fL, fR) / Lb, vel_mask)
    put("hand_chest_v", _vel(hand_chest), vel_mask)
    put("hands_dist_v", _vel(hands_dist), vel_mask)
    sphL, sphR = _norm(vhL), _norm(vhR)
    up_gate_L = torch.sigmoid(hL / (0.1 * Lb))
    up_gate_R = torch.sigmoid(hR / (0.1 * Lb))
    put("raised_hand_speed", torch.maximum(sphL * up_gate_L, sphR * up_gate_R) / Lb, vel_mask)
    vRm = vwR - 2.0 * (vwR * lat_hat).sum(-1, keepdim=True) * lat_hat     # mirror right arm
    cos_sym = (vwL * vRm).sum(-1) / (spL * spR + 1e-8)
    gate = torch.tanh(torch.minimum(spL, spR) / (0.02 * Lb))
    put("arm_sym", cos_sym * gate, vel_mask)

    # --- head
    hn = A[..., HEAD, :] - A[..., NECK, :]
    # + 1e-7: atan2(0, 0) on all-zero (invalid) frames would give a NaN gradient
    head_pitch = torch.atan2((hn * fwd).sum(-1), hn[..., UP] + 1e-7) * (180.0 / math.pi)
    put("head_pitch", head_pitch, vm)
    put("head_pitch_speed", _vel(head_pitch).abs(), vel_mask)
    put("head_lat_speed", _vel((hn * lat_hat).sum(-1)).abs() / Lb, vel_mask)
    put("head_speed", _norm(_vel3(R[..., HEAD, :])) / Lb, vel_mask)

    # --- torso / legs
    spine_v = A[..., SPINE_SHOULDER, :] - A[..., SPINE_BASE, :]
    horiz = torch.sqrt(spine_v[..., 0] ** 2 + spine_v[..., 2] ** 2 + 1e-10)
    put("torso_pitch", torch.atan2(horiz, spine_v[..., UP]) * (180.0 / math.pi), vm)
    yaw = torch.atan2(lat[..., 2], lat[..., 0] + 1e-7)
    dyaw = _vel(yaw)
    dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))                 # wrap to (-pi, pi]
    put("shoulder_yaw_speed", dyaw.abs() * (180.0 / math.pi), vel_mask)
    hip_height = (y[..., SPINE_BASE] - 0.5 * (y[..., L_ANKLE] + y[..., R_ANKLE])) / Lb
    put("hip_height", hip_height, vm)
    put("hip_vy", _vel(hip_height), vel_mask)
    put("hip_lat_speed", _vel((A[..., SPINE_BASE, :] * lat_hat).sum(-1)).abs() / Lb, vel_mask)
    k_l = _angle_deg(A[..., L_HIP, :], A[..., L_KNEE, :], A[..., L_ANKLE, :])
    k_r = _angle_deg(A[..., R_HIP, :], A[..., R_KNEE, :], A[..., R_ANKLE, :])
    put("knee_min", torch.minimum(k_l, k_r), vm)
    put("foot_height_diff", (y[..., L_FOOT] - y[..., R_FOOT]).abs() / Lb, vm)
    low_foot = torch.minimum(y[..., L_FOOT], y[..., R_FOOT])
    floor = torch.where(vm, low_foot, torch.full_like(low_foot, 1e4)).amin(-1, keepdim=True).detach()
    floor = torch.where(floor > 1e3, torch.zeros_like(floor), floor)
    put("feet_lift", (low_foot - floor) / Lb, vm)
    put("foot_vy", torch.maximum(_vel(y[..., L_FOOT]).abs(), _vel(y[..., R_FOOT]).abs()) / Lb, vel_mask)
    vaL, vaR = _vel3(A[..., L_ANKLE, :]), _vel3(A[..., R_ANKLE, :])
    hsL = torch.sqrt(vaL[..., 0] ** 2 + vaL[..., 2] ** 2 + 1e-10)
    hsR = torch.sqrt(vaR[..., 0] ** 2 + vaR[..., 2] ** 2 + 1e-10)
    put("ankle_speed", torch.maximum(hsL, hsR) / Lb, vel_mask)
    alt = ((A[..., L_ANKLE, :] - A[..., R_ANKLE, :]) * fwd).sum(-1)
    put("leg_alt_speed", _vel(alt).abs() / Lb, vel_mask)
    put("body_fwd_v", (_vel3(A[..., SPINE_SHOULDER, :]) * fwd).sum(-1) / Lb, vel_mask)
    put("body_speed", _norm(_vel3(R)).mean(-1) / Lb, vel_mask)

    # --- interaction (B = the other slot)
    if M >= 2:
        other = 1 - main
        B = A_all[ar, other]
        vB = valid[ar, other]
        both = vm & vB
        far = torch.full_like(hip_height, FAR_VALUE)
        pd = torch.clamp(_norm(A[..., SPINE_SHOULDER, :] - B[..., SPINE_SHOULDER, :]) / Lb, max=FAR_VALUE)
        pd = torch.where(both, pd, far)
        put("person_dist", pd, vm)
        put("person_dist_v", _vel(pd), _pair_mask(both))
        HA, HB = A[:, :, _CONTACT_HANDS], B[:, :, _CONTACT_HANDS]                 # (N,T,4,3)
        dAB = _norm(HA[:, :, :, None, :] - B[:, :, None, :, :]).flatten(2).amin(-1)
        dBA = _norm(HB[:, :, :, None, :] - A[:, :, None, :, :]).flatten(2).amin(-1)
        cd = torch.clamp(torch.minimum(dAB, dBA) / Lb, max=FAR_VALUE)
        put("contact_dist", torch.where(both, cd, far), vm)
        put("second_present", vB.to(A.dtype), vm)
    else:
        far = torch.full_like(hip_height, FAR_VALUE)
        put("person_dist", far, vm)
        put("person_dist_v", torch.zeros_like(far), torch.zeros_like(vm))
        put("contact_dist", far, vm)
        put("second_present", torch.zeros_like(far), vm)

    assert list(sig) == SIGNALS, set(sig) ^ set(SIGNALS)
    S = torch.stack([sig[s] for s in SIGNALS], dim=1)        # (N,NS,T)
    Mk = torch.stack([msk[s] for s in SIGNALS], dim=1)       # (N,NS,T)
    info = dict(valid_all=valid, L_all=L, main=main, valid=vm, L=Lm, A=A)
    return S, Mk, info
