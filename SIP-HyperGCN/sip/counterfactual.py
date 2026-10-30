"""Automatic counterfactual skeletons (pipeline v4, docs/architecture_v4.html section 2).

For a clip x and a geo concept c that holds on x, find a small edit delta of the joints
of c (main actor, valid frames, joint 20 never moved) such that c no longer holds:

    min  10 * relu(logit_c(x + delta) - target)          switch c off
       + lam_keep   * mean_{c' != c} |z_c'(x + delta) - z_c'(x)|   keep every other geo concept
       + lam_norm   * ||delta||^2 / (L^2 n_frames)           small edit
       + lam_bone   * sum (bone length change)^2 / (L^2 n_frames)   valid skeleton
       + lam_smooth * ||delta_{t+1} - delta_t||^2 / (L^2 n_frames)  no jitter

Only the concept layer (geometry + learned thresholds) is inside the objective: the
network under test never guides the edit, so its reaction to x_cf is an unbiased test of
whether it depends on c. Gradients are taken w.r.t. delta only (torch.autograd.grad), so
model / concept parameters accumulate nothing.

``random_control`` returns an edit of the SAME norm on the SAME joints / frames, smooth in
time but in a random direction: the reference against which the counterfactual effect is
measured (fixes the edit-size mismatch of the v3 controls).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sip.geo_torch import bone_lengths, geo_signals

__all__ = ["editable_geo", "edit_mask", "generate", "random_control"]


def editable_geo(ccp):
    """(G,) bool over geo concepts: True if the concept has joints to edit."""
    gi = ccp.concepts.geo_index.tolist()
    return torch.tensor([len(ccp.specs[i].joints) > 0 for i in gi], device=ccp.concepts.geo_index.device)


def edit_mask(ccp, gpos, info, shape):
    """(N, 1, T, V, M) float: 1 on the editable joints of geo concept gpos[i], in the main
    actor's slot, on its valid frames."""
    N, _, T, V, M = shape
    dev = gpos.device
    gi = ccp.concepts.geo_index
    jm = torch.zeros(N, V, device=dev)
    for i, g in enumerate(gpos.tolist()):
        js = ccp.specs[int(gi[g])].joints
        if js:
            jm[i, js] = 1.0
    mask = torch.zeros(N, 1, T, V, M, device=dev)
    ar = torch.arange(N, device=dev)
    sl = info["valid"].float()[:, :, None] * jm[:, None, :]          # (N,T,V)
    mask[ar, 0, :, :, info["main"]] = sl
    return mask


def generate(ccp, x, gpos, steps: int = 30, lr: float = 0.01, target_logit: float = -2.0,
             lam_keep: float = 1.0, lam_norm: float = 1.0, lam_bone: float = 10.0,
             lam_smooth: float = 1.0):
    """x (N, 3, T, V, M), gpos (N,) long: position of the concept in ccp.concepts.geo_index.
    Returns dict: x_cf, success (N,) bool (concept switched off), edit (N,) RMS edit size
    in L per frame, z_before / z_after (N,) of the target concept, mask."""
    x0 = x.detach().float()
    N = x0.shape[0]
    ar = torch.arange(N, device=x0.device)
    with torch.no_grad():
        S0, Mk0, info = geo_signals(x0)
        z0 = torch.sigmoid(ccp.concepts.geo_logit(S0, Mk0))
    fixed = dict(valid=info["valid_all"], L=info["L_all"], main=info["main"])
    mask = edit_mask(ccp, gpos, info, x0.shape)
    L2 = info["L"] ** 2
    nf = info["valid"].sum(1).clamp_min(1).float()
    vb = info["valid"][:, :, None].float()
    bone0 = bone_lengths(info["A"]).detach()
    keep = torch.ones_like(z0)
    keep[ar, gpos] = 0.0
    keep_n = keep.sum(1).clamp_min(1.0)

    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        with torch.enable_grad():
            d = delta * mask
            S, Mk, inf = geo_signals(x0 + d, **fixed)
            gl = ccp.concepts.geo_logit(S, Mk)
            l_target = F.relu(gl[ar, gpos] - target_logit)
            l_keep = ((torch.sigmoid(gl) - z0).abs() * keep).sum(1) / keep_n
            l_norm = (d ** 2).sum((1, 2, 3, 4)) / (L2 * nf)
            l_bone = (((bone_lengths(inf["A"]) - bone0) ** 2) * vb).sum((1, 2)) / (L2 * nf)
            l_smooth = ((d[:, :, 1:] - d[:, :, :-1]) ** 2).sum((1, 2, 3, 4)) / (L2 * nf)
            total = (10.0 * l_target + lam_keep * l_keep + lam_norm * l_norm
                     + lam_bone * l_bone + lam_smooth * l_smooth).sum()
            g, = torch.autograd.grad(total, delta)
        opt.zero_grad(set_to_none=True)
        delta.grad = g
        opt.step()

    with torch.no_grad():
        d = delta * mask
        x_cf = x0 + d
        S, Mk, _ = geo_signals(x_cf, **fixed)
        z1 = torch.sigmoid(ccp.concepts.geo_logit(S, Mk))
        edit = torch.sqrt((d ** 2).sum((1, 2, 3, 4)) / (L2 * nf))
    return dict(x_cf=x_cf, success=z1[ar, gpos] < 0.5, edit=edit, z_before=z0[ar, gpos],
                z_after=z1[ar, gpos], mask=mask, delta=d)


@torch.no_grad()
def random_control(x, delta, mask, generator=None, smooth: int = 9):
    """Edit of the same per-sample norm as delta, on the same joints / frames (mask),
    temporally smoothed Gaussian noise in a random direction."""
    x0 = x.detach().float()
    N, C, T, V, M = x0.shape
    noise = torch.randn(x0.shape, device=x0.device, generator=generator) * mask
    n = noise.permute(0, 1, 3, 4, 2).reshape(-1, 1, T)
    n = F.avg_pool1d(n, smooth, stride=1, padding=smooth // 2, count_include_pad=False)
    n = n.reshape(N, C, V, M, T).permute(0, 1, 4, 2, 3) * mask
    scale = delta.flatten(1).norm(dim=1) / n.flatten(1).norm(dim=1).clamp_min(1e-8)
    return x0 + n * scale[:, None, None, None, None]
