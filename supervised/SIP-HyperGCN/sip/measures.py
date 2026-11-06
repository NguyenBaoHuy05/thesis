"""Sample-level CONTINUOUS geometric measures computed from NTU joint coordinates (v3).

Replaces the v2 binary predicates. No thresholds are applied while measuring: every
measure is a per-frame continuous quantity (or a per-sample quantity for displacements)
summarised by a few temporal statistics computed over the VALID frames only. Thresholds
appear later, only inside induced rules (learned on TRAIN-FIT).

numpy only, vectorised over samples and frames; samples are processed in chunks.

Coordinate frame of the input
-----------------------------
``x`` has shape ``(N, 3, T, 25, M=2)`` exactly as the HyperGCN test feeder hands it to
the model (``input_format="feeder"``, the default):

* metres, Kinect v2 camera space: axis 0 = x (lateral), axis 1 = y (UP), axis 2 = z (depth);
  no rotation / scale normalisation on the test split.
* the feeder (joint modality) makes every joint relative to spine-shoulder joint 20 of the
  same person/frame but keeps joint 20 itself absolute:
  ``x[..., v] = p_v - p_20`` (v != 20), ``x[..., 20] = p_20``. We reconstruct
  ``p_v = x_v + x_20``. ``input_format="absolute"`` skips this.
* an absent 2nd person is all zeros; a partly-present 2nd person has zero frames and
  interpolated "shrunk" frames at the boundary (removed by the BLEND_MARGIN rule below).

Validity (the ONLY constants of this module, all about data validity, none about actions)
----------------------------------------------------------------------------------------
* person-frame valid  <=>  non-zero  AND  spine length |p_20 - p_0| within
  [SPINE_RATIO_LO, SPINE_RATIO_HI] = [0.6, 1.5] x that person's median spine length
  AND no raw-zero frame of that person within BLEND_MARGIN = 2 frames (the feeder resizes
  before the joint-20-relative transform, so frames next to a zero frame are shrunk blends).
* L = per-person median spine length over its non-zero frames (fallback 0.5 m).
* interaction measures need >= MIN_PAIR_FRAMES = 4 frames where BOTH persons are valid,
  otherwise they are set to FAR_VALUE. Interaction distances are capped at FAR_VALUE so a
  missing partner is always "at least as far" as any real partner.
* body_disp_h / body_disp_v compare the mean of the first vs last max(1, n_valid // 8)
  valid frames (DISP_WINDOW_DIV = 8, a smoothing window, not a threshold).

Which person
------------
All per-person measures are taken from the MAIN actor = person slot 0 (``seq_translation``
centres the clip on actor 1, which is valid in essentially all frames). If slot 0 has no
valid frame but slot 1 has, slot 1 is used as main actor and slot 0 as the partner.
Arm measures combine the two arms per frame as stated in the table (min / max over arms,
i.e. "the more extreme arm"); no left/right preference.

Body axes (per frame): up = +y; lateral s = p_8 - p_4 (R shoulder - L shoulder);
forward f = normalize(up x s) = normalize((s_z, 0, -s_x)) (horizontal). See
``forward_axis_check``.

Units: distances / heights divided by the main actor's L; angles in DEGREES; speeds in
L per (resampled) frame - clips are resized to T=64, so speed = motion amount per clip.

Empty valid set: a statistic over an empty frame set (main actor never valid, < 2 valid
frames for speeds, no motion for hand_symmetry) is replaced by the measure's NEUTRAL value
(a rough standing-at-rest value; 0 for motion / symmetry / displacement). Interaction
measures use FAR_VALUE. The output is finite everywhere.

Run ``measure_table_markdown()`` for the measure table (also below).
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "MEASURE_NAMES",
    "MEASURE_REGIONS",
    "MEASURE_STATS",
    "MEASURE_FORMULAS",
    "NEUTRAL",
    "COLUMNS",
    "FAR_VALUE",
    "SPINE_RATIO_LO",
    "SPINE_RATIO_HI",
    "MIN_PAIR_FRAMES",
    "measure_slices",
    "compute_measures",
    "measure_table_markdown",
    "forward_axis_check",
]

# --------------------------------------------------------------------------- joints
SPINE_BASE, SPINE_MID, NECK, HEAD = 0, 1, 2, 3
L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 4, 5, 6, 7
R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 8, 9, 10, 11
L_HIP, L_KNEE, L_ANKLE, L_FOOT = 12, 13, 14, 15
R_HIP, R_KNEE, R_ANKLE, R_FOOT = 16, 17, 18, 19
SPINE_SHOULDER = 20
L_HAND_TIP, L_THUMB, R_HAND_TIP, R_THUMB = 21, 22, 23, 24
UP = 1  # vertical coordinate index (Kinect y)

_EPS = 1e-6
_DEFAULT_BODY_SIZE = 0.5  # metres, fallback spine length

# validity constants (see module docstring)
SPINE_RATIO_LO = 0.6
SPINE_RATIO_HI = 1.5
BLEND_MARGIN = 2  # frames dropped on each side of a raw-zero frame (resize blends)
MIN_PAIR_FRAMES = 4
DISP_WINDOW_DIV = 8

FAR_VALUE: float = 5.0  # in units of L (~2.5 m); fill / cap for interaction distances

_ARMS = ["left_arm", "right_arm"]
_LEGS = ["left_leg", "right_leg"]

# (name, regions, stats, per-frame formula, neutral value)
_SPECS = [
    ("hand_head_dist", _ARMS + ["head"], ["min", "q10", "median"],
     "min_h |p_h - p_3| / L, h in {7, 11}", 1.5),
    ("hand_chest_dist", _ARMS + ["torso"], ["min", "q10", "median"],
     "min_h |p_h - c| / L, c = (p_20 + p_1) / 2, h in {7, 11}", 0.8),
    ("hands_dist", _ARMS, ["min", "q10", "median"],
     "|p_7 - p_11| / L", 0.8),
    ("hand_height_head", _ARMS + ["head"], ["max", "q90", "median"],
     "(max(y_7, y_11) - y_3) / L", -1.5),
    ("hand_height_shoulder", _ARMS, ["max", "q90", "median"],
     "max(y_7 - y_4, y_11 - y_8) / L", -1.0),
    ("hand_floor", _ARMS + _LEGS, ["min", "q10"],
     "(min(y_7, y_11) - (y_13 + y_17) / 2) / L", 0.5),
    ("arm_reach_fwd", _ARMS, ["max", "q90"],
     "max over arms of (p_wrist - p_shoulder) . f / L (wrists 6, 10; shoulders 4, 8)", 0.0),
    ("elbow_angle", _ARMS, ["max", "median"],
     "max over arms of angle(shoulder, elbow, wrist) [deg] (180 = straight)", 160.0),
    ("wrist_speed", _ARMS, ["mean", "q90"],
     "max over wrists of |r_w(t+1) - r_w(t)| / L, r = p - p_20 (body-relative); frame pairs both valid", 0.0),
    ("hand_symmetry", _ARMS, ["wmean"],
     "cos(v_L, mirror_s(v_R)) of body-relative wrist velocities, averaged over frame pairs "
     "weighted by min(|v_L|, |v_R|) (= mean over moving frames, no motion threshold)", 0.0),
    ("torso_pitch", ["torso"], ["max", "median"],
     "angle(p_20 - p_0, +y) [deg] (0 = upright)", 5.0),
    ("hip_height", ["torso"] + _LEGS, ["min", "median"],
     "(y_0 - (y_14 + y_18) / 2) / L", 1.8),
    ("knee_angle", _LEGS, ["min", "median"],
     "min over legs of angle(hip, knee, ankle) [deg] (180 = straight)", 170.0),
    ("foot_height_diff", _LEGS, ["max", "q90"],
     "|y_15 - y_19| / L", 0.0),
    ("leg_alternation", _LEGS, ["std"],
     "(p_14 - p_18) . f / L", 0.0),
    ("body_disp_h", ["torso"], ["delta"],
     "per sample: |horiz(mean_end p_20 - mean_start p_20)| / L", 0.0),
    ("body_disp_v", ["torso"], ["delta"],
     "per sample, SIGNED: (mean_end y_0 - mean_start y_0) / L (> 0 = rises, < 0 = goes down)", 0.0),
    ("person_dist", ["other_person"], ["min", "median"],
     "|p_20(A) - p_20(B)| / L_A on frames where both valid; capped at FAR_VALUE", FAR_VALUE),
    ("contact_dist", ["other_person"] + _ARMS, ["min", "q10"],
     "min over hands {7, 11, 21, 23} of one person to any joint of the other (both directions) "
     "/ L_A on frames where both valid; capped at FAR_VALUE", FAR_VALUE),
    ("second_person", ["other_person"], ["present_frac"],
     "per sample: fraction of the T frames in which the partner (slot 1) is valid", 0.0),
]

MEASURE_NAMES: list[str] = [s[0] for s in _SPECS]
MEASURE_REGIONS: dict[str, list[str]] = {s[0]: list(s[1]) for s in _SPECS}
MEASURE_STATS: dict[str, list[str]] = {s[0]: list(s[2]) for s in _SPECS}
MEASURE_FORMULAS: dict[str, str] = {s[0]: s[3] for s in _SPECS}
NEUTRAL: dict[str, float] = {s[0]: float(s[4]) for s in _SPECS}
COLUMNS: list[str] = [f"{m}:{st}" for m in MEASURE_NAMES for st in MEASURE_STATS[m]]
INTERACTION_MEASURES = ("person_dist", "contact_dist")

_REGION_NAMES = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg", "other_person"]
assert all(r in _REGION_NAMES for rs in MEASURE_REGIONS.values() for r in rs)
assert len(set(MEASURE_NAMES)) == len(MEASURE_NAMES)


def measure_slices() -> dict[str, slice]:
    """measure -> slice of its columns in COLUMNS (MEASURE_NAMES order)."""
    out, i = {}, 0
    for m in MEASURE_NAMES:
        k = len(MEASURE_STATS[m])
        out[m] = slice(i, i + k)
        i += k
    return out


# --------------------------------------------------------------------------- helpers
def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt((v * v).sum(-1))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle a-b-c at vertex b in degrees. a,b,c: (..., 3) -> (...)."""
    v1 = a - b
    v2 = c - b
    cos = (v1 * v2).sum(-1) / (_norm(v1) * _norm(v2) + _EPS)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _to_joints(x: np.ndarray, input_format: str):
    """(n,3,T,V,M) -> J_in (n,M,T,V,3) as given, A (n,M,T,V,3) absolute coordinates."""
    J = np.ascontiguousarray(np.transpose(x, (0, 4, 2, 3, 1))).astype(np.float64)
    if input_format == "feeder":
        A = J + J[..., SPINE_SHOULDER:SPINE_SHOULDER + 1, :]
        A[..., SPINE_SHOULDER, :] = J[..., SPINE_SHOULDER, :]
    elif input_format == "absolute":
        A = J
    else:
        raise ValueError("input_format must be 'feeder' or 'absolute'")
    return J, A


def _valid_and_size(J: np.ndarray, A: np.ndarray):
    """valid (n,M,T) bool and body size L (n,M) (median spine length of non-zero frames)."""
    raw_valid = np.abs(J).sum(axis=(-1, -2)) > _EPS
    spine = _norm(A[..., SPINE_SHOULDER, :] - A[..., SPINE_BASE, :])  # (n,M,T)
    sp = np.where(raw_valid, spine, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(sp, axis=-1)  # (n,M), NaN if never non-zero
    ok = np.isfinite(med) & (med > 1e-3)
    L = np.where(ok, med, _DEFAULT_BODY_SIZE)
    medb = L[..., None]
    valid = (raw_valid & ok[..., None]
             & (spine > SPINE_RATIO_LO * medb)
             & (spine < SPINE_RATIO_HI * medb))
    # The feeder resizes the clip (bilinear in time) BEFORE the joint-20-relative transform,
    # so frames next to a zero (absent) frame are blends alpha * p that shrink the whole
    # skeleton toward the origin; with alpha in (0.6, 1) they pass the spine band. Drop every
    # frame within BLEND_MARGIN frames of a raw-zero frame of the same person.
    T = raw_valid.shape[-1]
    for s in range(1, BLEND_MARGIN + 1):
        if s >= T:
            break
        prev_zero = np.zeros_like(raw_valid)
        prev_zero[..., s:] = ~raw_valid[..., :-s]
        next_zero = np.zeros_like(raw_valid)
        next_zero[..., :-s] = ~raw_valid[..., s:]
        valid &= ~(prev_zero | next_zero)
    return valid, L


def _forward_axis(A: np.ndarray) -> np.ndarray:
    """Horizontal forward unit vector f = normalize(up x s), s = R_sh - L_sh. (..., 3)."""
    s = A[..., R_SHOULDER, :] - A[..., L_SHOULDER, :]
    f = np.zeros_like(s)
    f[..., 0] = s[..., 2]   # e_y x s = (s_z, 0, -s_x)
    f[..., 2] = -s[..., 0]
    return f / (_norm(f)[..., None] + _EPS)


def _masked_stats(v: np.ndarray, m: np.ndarray, stats: list[str]) -> np.ndarray:
    """Temporal statistics of v (n,T') over mask m (n,T') -> (n, len(stats)); NaN if empty.

    Quantiles use linear interpolation between order statistics (numpy default);
    std is the population std. Supported: min, max, median, mean, std, qXX.
    """
    n = v.shape[0]
    cnt = m.sum(-1)  # (n,)
    has = cnt > 0
    cntf = np.maximum(cnt, 1).astype(np.float64)
    out = np.full((n, len(stats)), np.nan)
    vs = None
    mean = None
    for j, st in enumerate(stats):
        if st in ("mean", "std"):
            if mean is None:
                mean = np.where(m, v, 0.0).sum(-1) / cntf
            if st == "mean":
                val = mean
            else:
                var = np.where(m, (v - mean[:, None]) ** 2, 0.0).sum(-1) / cntf
                val = np.sqrt(var)
        else:
            if st == "min":
                q = 0.0
            elif st == "max":
                q = 1.0
            elif st == "median":
                q = 0.5
            elif st.startswith("q") and st[1:].isdigit():
                q = int(st[1:]) / 100.0
            else:
                raise ValueError(f"unknown statistic {st!r}")
            if vs is None:
                vs = np.sort(np.where(m, v, np.inf), axis=-1)  # valid values first
            pos = q * (cntf - 1.0)
            lo = np.floor(pos).astype(np.int64)
            hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
            frac = pos - lo
            a = np.take_along_axis(vs, lo[:, None], axis=-1)[:, 0]
            b = np.take_along_axis(vs, hi[:, None], axis=-1)[:, 0]
            with np.errstate(invalid="ignore"):
                val = np.where(frac > 0, a + frac * (b - a), a)
        out[:, j] = np.where(has, val, np.nan)
    return out


def _window_means(p: np.ndarray, valid: np.ndarray):
    """Mean of p (n,T,D) over the first / last max(1, n_valid // 8) valid frames -> (n,D) x2."""
    rank = np.cumsum(valid, axis=-1) - 1  # (n,T)
    nv = valid.sum(-1, keepdims=True)
    k = np.maximum(nv // DISP_WINDOW_DIV, 1)
    start = valid & (rank < k)
    end = valid & (rank >= nv - k)

    def _m(mask):
        w = mask[..., None].astype(np.float64)
        return (p * w).sum(-2) / np.maximum(w.sum(-2), 1.0)

    return _m(start), _m(end)


# --------------------------------------------------------------------------- core
def _compute_chunk(x: np.ndarray, input_format: str) -> np.ndarray:
    J, A_all = _to_joints(x, input_format)
    n, M, T = A_all.shape[:3]
    valid_all, L_all = _valid_and_size(J, A_all)  # (n,M,T), (n,M)

    # main actor: slot 0, unless slot 0 is never valid and slot 1 is
    has0 = valid_all[:, 0].any(-1)
    if M >= 2:
        main = np.where(~has0 & valid_all[:, 1].any(-1), 1, 0)
    else:
        main = np.zeros(n, dtype=np.int64)
    ar = np.arange(n)
    A = A_all[ar, main]          # (n,T,V,3)
    valid = valid_all[ar, main]  # (n,T)
    L = L_all[ar, main]          # (n,)
    Lb = L[:, None]              # (n,1)
    y = A[..., UP]               # (n,T,V)

    res: dict[str, np.ndarray] = {}

    def put(name, per_frame, mask):
        res[name] = _masked_stats(per_frame, mask, MEASURE_STATS[name])

    # --- hand proximity
    pLh, pRh = A[..., L_HAND, :], A[..., R_HAND, :]
    head = A[..., HEAD, :]
    put("hand_head_dist", np.minimum(_norm(pLh - head), _norm(pRh - head)) / Lb, valid)
    chest = 0.5 * (A[..., SPINE_SHOULDER, :] + A[..., SPINE_MID, :])
    put("hand_chest_dist", np.minimum(_norm(pLh - chest), _norm(pRh - chest)) / Lb, valid)
    put("hands_dist", _norm(pLh - pRh) / Lb, valid)

    # --- hand height
    put("hand_height_head", (np.maximum(y[..., L_HAND], y[..., R_HAND]) - y[..., HEAD]) / Lb, valid)
    put("hand_height_shoulder",
        np.maximum(y[..., L_HAND] - y[..., L_SHOULDER], y[..., R_HAND] - y[..., R_SHOULDER]) / Lb, valid)
    knee_y = 0.5 * (y[..., L_KNEE] + y[..., R_KNEE])
    put("hand_floor", (np.minimum(y[..., L_HAND], y[..., R_HAND]) - knee_y) / Lb, valid)

    # --- arm configuration
    f = _forward_axis(A)  # (n,T,3)
    reach_l = ((A[..., L_WRIST, :] - A[..., L_SHOULDER, :]) * f).sum(-1)
    reach_r = ((A[..., R_WRIST, :] - A[..., R_SHOULDER, :]) * f).sum(-1)
    put("arm_reach_fwd", np.maximum(reach_l, reach_r) / Lb, valid)
    el_l = _angle_deg(A[..., L_SHOULDER, :], A[..., L_ELBOW, :], A[..., L_WRIST, :])
    el_r = _angle_deg(A[..., R_SHOULDER, :], A[..., R_ELBOW, :], A[..., R_WRIST, :])
    put("elbow_angle", np.maximum(el_l, el_r), valid)

    # --- arm motion (body-relative wrists), over consecutive valid frame pairs
    R = A - A[..., SPINE_SHOULDER:SPINE_SHOULDER + 1, :]
    pair_valid = valid[:, 1:] & valid[:, :-1]  # (n,T-1)
    vL = R[:, 1:, L_WRIST] - R[:, :-1, L_WRIST]  # (n,T-1,3)
    vR = R[:, 1:, R_WRIST] - R[:, :-1, R_WRIST]
    spL = _norm(vL)
    spR = _norm(vR)
    put("wrist_speed", np.maximum(spL, spR) / Lb, pair_valid)

    s = A[:, :-1, R_SHOULDER, :] - A[:, :-1, L_SHOULDER, :]
    lat = s / (_norm(s)[..., None] + _EPS)  # (n,T-1,3)
    vRm = vR - 2.0 * (vR * lat).sum(-1, keepdims=True) * lat  # mirror through sagittal plane
    cos_sym = (vL * vRm).sum(-1) / (spL * spR + _EPS)
    w = np.where(pair_valid, np.minimum(spL, spR) / Lb, 0.0)
    wsum = w.sum(-1)
    sym = np.where(wsum > _EPS, (w * cos_sym).sum(-1) / np.maximum(wsum, _EPS), np.nan)
    res["hand_symmetry"] = sym[:, None]

    # --- torso / legs
    spine_v = A[..., SPINE_SHOULDER, :] - A[..., SPINE_BASE, :]
    cos_up = spine_v[..., UP] / (_norm(spine_v) + _EPS)
    put("torso_pitch", np.degrees(np.arccos(np.clip(cos_up, -1.0, 1.0))), valid)
    put("hip_height", (y[..., SPINE_BASE] - 0.5 * (y[..., L_ANKLE] + y[..., R_ANKLE])) / Lb, valid)
    k_l = _angle_deg(A[..., L_HIP, :], A[..., L_KNEE, :], A[..., L_ANKLE, :])
    k_r = _angle_deg(A[..., R_HIP, :], A[..., R_KNEE, :], A[..., R_ANKLE, :])
    put("knee_angle", np.minimum(k_l, k_r), valid)
    put("foot_height_diff", np.abs(y[..., L_FOOT] - y[..., R_FOOT]) / Lb, valid)
    put("leg_alternation", ((A[..., L_ANKLE, :] - A[..., R_ANKLE, :]) * f).sum(-1) / Lb, valid)

    # --- global displacement (start vs end windows of valid frames)
    anyv = valid.any(-1)
    p20_s, p20_e = _window_means(A[..., SPINE_SHOULDER, :], valid)  # (n,3)
    dh = p20_e - p20_s
    disp_h = np.sqrt(dh[:, 0] ** 2 + dh[:, 2] ** 2) / L
    res["body_disp_h"] = np.where(anyv, disp_h, np.nan)[:, None]
    y0_s, y0_e = _window_means(y[..., SPINE_BASE][..., None], valid)  # (n,1)
    res["body_disp_v"] = np.where(anyv, (y0_e[:, 0] - y0_s[:, 0]) / L, np.nan)[:, None]

    # --- interaction (person A = main actor, person B = the other slot)
    if M >= 2:
        other = 1 - main
        B = A_all[ar, other]              # (n,T,V,3)
        validB = valid_all[ar, other]     # (n,T)
        both = valid & validB
        enough = both.sum(-1) >= MIN_PAIR_FRAMES  # (n,)
        d_p = np.minimum(_norm(A[..., SPINE_SHOULDER, :] - B[..., SPINE_SHOULDER, :]) / Lb, FAR_VALUE)
        st = _masked_stats(d_p, both, MEASURE_STATS["person_dist"])
        res["person_dist"] = np.where(enough[:, None], st, FAR_VALUE)
        hands = [L_HAND, R_HAND, L_HAND_TIP, R_HAND_TIP]
        HA, HB = A[:, :, hands], B[:, :, hands]  # (n,T,4,3)
        dAB = _norm(HA[:, :, :, None, :] - B[:, :, None, :, :]).min(axis=(2, 3))  # (n,T)
        dBA = _norm(HB[:, :, :, None, :] - A[:, :, None, :, :]).min(axis=(2, 3))
        d_c = np.minimum(np.minimum(dAB, dBA) / Lb, FAR_VALUE)
        st = _masked_stats(d_c, both, MEASURE_STATS["contact_dist"])
        res["contact_dist"] = np.where(enough[:, None], st, FAR_VALUE)
        res["second_person"] = validB.mean(-1)[:, None]
    else:
        res["person_dist"] = np.full((n, len(MEASURE_STATS["person_dist"])), FAR_VALUE)
        res["contact_dist"] = np.full((n, len(MEASURE_STATS["contact_dist"])), FAR_VALUE)
        res["second_person"] = np.zeros((n, 1))

    assert set(res) == set(MEASURE_NAMES), set(res) ^ set(MEASURE_NAMES)
    cols = []
    for name in MEASURE_NAMES:
        a = np.asarray(res[name], dtype=np.float64).reshape(n, len(MEASURE_STATS[name]))
        cols.append(np.where(np.isfinite(a), a, NEUTRAL[name]))
    return np.concatenate(cols, axis=1).astype(np.float32)


def compute_measures(x: np.ndarray, input_format: str = "feeder", chunk_size: int = 256) -> np.ndarray:
    """Compute all measures.

    Args:
        x: (N, 3, T, 25, M) joint coordinates as fed to the model (zero-padded frames /
           absent second person allowed). M = 1 accepted (interaction = FAR_VALUE,
           second_person = 0).
        input_format: "feeder" (default; joints relative to joint 20, joint 20 absolute)
           or "absolute".
        chunk_size: samples per vectorised chunk (peak memory ~ chunk*T*100*3*8 bytes
           for contact_dist).
    Returns:
        (N, D) float32, columns ordered as COLUMNS, finite everywhere.
    """
    x = np.asarray(x)
    if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != 25:
        raise ValueError(f"expected x of shape (N, 3, T, 25, M), got {x.shape}")
    N = x.shape[0]
    out = np.zeros((N, len(COLUMNS)), dtype=np.float32)
    for i in range(0, N, chunk_size):
        out[i:i + chunk_size] = _compute_chunk(x[i:i + chunk_size], input_format)
    return out


# --------------------------------------------------------------------------- docs / checks
def measure_table_markdown() -> str:
    """Markdown table: measure | regions | per-frame formula | temporal stats | neutral."""
    lines = [
        "| measure | regions | formula (per frame unless stated) | stats | empty-set value |",
        "|---|---|---|---|---|",
    ]
    for name, regions, stats, formula, neutral in _SPECS:
        lines.append(f"| `{name}` | {', '.join(regions)} | {formula} | {', '.join(stats)} | {neutral:g} |")
    lines.append("")
    lines.append(f"D = {len(COLUMNS)} columns; FAR_VALUE = {FAR_VALUE:g} (L units) for "
                 f"interaction measures without a partner (>= {MIN_PAIR_FRAMES} co-valid frames "
                 f"required); valid frame = non-zero and spine length in "
                 f"[{SPINE_RATIO_LO:g}, {SPINE_RATIO_HI:g}] x median.")
    return "\n".join(lines)


def forward_axis_check(x: np.ndarray, input_format: str = "feeder") -> float:
    """Fraction of valid person-frames where f = up x (R_sh - L_sh) agrees (positive dot)
    with the horizontal foot direction (foot - ankle). Should be clearly > 0.5; if << 0.5
    the L/R convention is mirrored (negate f in ``_forward_axis``)."""
    J, A = _to_joints(np.asarray(x), input_format)
    valid, _ = _valid_and_size(J, A)
    f = _forward_axis(A)
    foot = (A[..., L_FOOT, :] - A[..., L_ANKLE, :]) + (A[..., R_FOOT, :] - A[..., R_ANKLE, :])
    foot[..., UP] = 0.0
    agree = (f * foot).sum(-1) > 0
    return float((agree & valid).sum() / max(valid.sum(), 1))
