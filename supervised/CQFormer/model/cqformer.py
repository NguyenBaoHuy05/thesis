"""Concept-Query Transformer: no backbone, every decision goes through joint-bound concepts.

    joints of a set -> shared encoder -> concept timeline C[t, k] -> motifs -> class logits

Guarantee: a concept only depends on the joints of its set (tests/test_guarantee.py checks it).
It holds because joints are selected by index before anything else, bones are computed inside
the set, person validity is read from the set's joints, and there is no BatchNorm or global
token. Keep all of this when editing the concept path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.joint_sets import JOINT_SETS, PARENT, JointSet, concept_names, is_state_concept

ABSENT_LOGIT = -20.0   # concept value of a person slot with no data


class Attention(nn.Module):
    def __init__(self, dim, heads, attn_drop):
        super().__init__()
        self.heads = heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """Attention over the joints of the set, then local attention over time, then MLP."""

    def __init__(self, dim, heads, mlp_ratio, attn_drop, drop_path):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.spatial = Attention(dim, heads, attn_drop)
        self.norm_t = nn.LayerNorm(dim)
        self.temporal = Attention(dim, heads, attn_drop)
        self.norm_m = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = drop_path

    def forward(self, z, time_mask):
        N, T, V, D = z.shape
        s = z.reshape(N * T, V, D)
        s = s + self.spatial(self.norm_s(s))
        t = s.reshape(N, T, V, D).permute(0, 2, 1, 3).reshape(N * V, T, D)
        t = t + self.temporal(self.norm_t(t), time_mask)
        out = t.reshape(N, V, T, D).permute(0, 2, 1, 3)
        out = out + self.mlp(self.norm_m(out))
        if self.training and self.drop_path > 0:
            keep = (torch.rand(N, 1, 1, 1, device=z.device) >= self.drop_path).to(z.dtype)
            out = z + (out - z) * keep / (1 - self.drop_path)
        return out


class CQFormer(nn.Module):
    def __init__(self, num_classes, concept_matrix, sets: list[JointSet] = JOINT_SETS, dim=192, depth=6,
                 heads=6, mlp_ratio=4.0, num_frames=64, patch_t=4, local_window=2, attn_drop=0.1,
                 drop_path=0.2, free_per_set=16, motifs=256, order_filters=256, motif_kernel=5,
                 head_drop=0.3, modality_gain=(1.0, 1.0, 5.0, 5.0)):
        super().__init__()
        if num_frames % patch_t:
            raise ValueError("num_frames must be a multiple of patch_t")
        self.sets = sets
        self.patch_t = patch_t
        self.steps = num_frames // patch_t
        self.dim = dim

        self.modality_gain = nn.Parameter(torch.tensor(modality_gain, dtype=torch.float))
        self.embed = nn.Linear(12 * patch_t, dim)
        self.joint_embed = nn.Embedding(25, dim)
        self.set_embed = nn.Embedding(len(sets), dim)
        self.person_embed = nn.Embedding(2, dim)
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(6, dtype=torch.float), persistent=False)
        self.time_embed = nn.Linear(12, dim)

        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_ratio, attn_drop, drop_path * i / max(1, depth - 1)) for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        steps = torch.arange(self.steps)
        self.register_buffer("time_mask", (steps[None] - steps[:, None]).abs() <= local_window, persistent=False)

        self.key = nn.Linear(dim, dim)
        self.queries = nn.ParameterList()
        self.scorers = nn.ParameterList()
        self.biases = nn.ParameterList()
        named_idx, offset = [], 0
        for i, s in enumerate(sets):
            pos = {j: p for p, j in enumerate(s.joints)}
            parents = [PARENT[j] for j in s.joints]
            parent_idx = [pos.get(par, p) for p, par in enumerate(parents)]
            has_parent = [par in pos and par != j for par, j in zip(parents, s.joints)]
            self.register_buffer(f"joints_{i}", torch.tensor(s.joints), persistent=False)
            self.register_buffer(f"parent_{i}", torch.tensor(parent_idx), persistent=False)
            self.register_buffer(f"has_parent_{i}", torch.tensor(has_parent).view(1, 1, 1, -1), persistent=False)
            M = 2 if s.two_person else 1
            self.register_buffer(f"token_joint_{i}", torch.tensor(s.joints).repeat(M), persistent=False)
            persons = torch.arange(M).repeat_interleave(len(s.joints))
            self.register_buffer(f"token_person_{i}", persons, persistent=False)

            K = len(s.concepts) + free_per_set
            self.queries.append(nn.Parameter(torch.randn(K, dim)))
            self.scorers.append(nn.Parameter(torch.randn(K, dim)))
            self.biases.append(nn.Parameter(torch.full((K,), -1.0)))
            named_idx.extend(range(offset, offset + len(s.concepts)))
            offset += K
        self.num_timeline = offset
        self.register_buffer("named_idx", torch.tensor(named_idx), persistent=False)

        names = concept_names(sets)
        self.tau_raw = nn.Parameter(torch.full((len(names),), 0.37))   # softplus(0.37) + 0.1 is about 1
        sign = [-1.0 if is_state_concept(n) else 1.0 for n in names]
        self.register_buffer("tau_sign", torch.tensor(sign), persistent=False)

        S = torch.as_tensor(concept_matrix, dtype=torch.float)
        self.register_buffer("S", S, persistent=False)
        self.W = nn.Parameter(2 * S - 1)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

        self.motif = nn.Conv1d(self.num_timeline, motifs, motif_kernel, padding=motif_kernel // 2)
        pad = 2 * (motif_kernel // 2)
        self.order = nn.Conv1d(motifs, order_filters, motif_kernel, padding=pad, dilation=2)
        self.head_drop = nn.Dropout(head_drop)
        self.motif_head = nn.Linear(motifs + 2 * order_filters, num_classes)

    def modalities(self, x, i):
        """J, B, JM, BM of one person slot: x (N, 3, T, V) -> (N, 12, T, V). Bones stay in the set."""
        parent = getattr(self, f"parent_{i}")
        has_parent = getattr(self, f"has_parent_{i}")
        bone = torch.where(has_parent, x - x[:, :, :, parent], x - x.mean(dim=3, keepdim=True))
        feats = torch.cat([x, bone, self.motion(x), self.motion(bone)], dim=1)
        return feats * self.modality_gain.repeat_interleave(3).view(1, 12, 1, 1)

    @staticmethod
    def motion(x):
        return torch.cat([x[:, :, 1:] - x[:, :, :-1], torch.zeros_like(x[:, :, :1])], dim=2)

    def encode(self, feats, index_t, i):
        """feats (N, 12, T, V), index_t (N, T) -> concept logits (N, steps, K) of set i."""
        N, _, _, V = feats.shape
        tok = feats.reshape(N, 12, self.steps, self.patch_t, V).permute(0, 2, 4, 1, 3).reshape(N, self.steps, V, -1)
        z = self.embed(tok) + self.joint_embed(getattr(self, f"token_joint_{i}")) + self.set_embed.weight[i]
        if self.sets[i].two_person:
            z = z + self.person_embed(getattr(self, f"token_person_{i}"))
        t = index_t.reshape(N, self.steps, self.patch_t).mean(-1, keepdim=True) * self.freqs
        z = z + self.time_embed(torch.cat([t.sin(), t.cos()], dim=-1)).unsqueeze(2)
        for block in self.blocks:
            z = block(z, self.time_mask)
        z = self.norm(z)
        scale = math.sqrt(self.dim)
        keys = self.key(z)
        attn = torch.softmax(torch.einsum("ntvd,kd->ntvk", keys, self.queries[i].to(keys.dtype)) / scale, dim=2)
        score = torch.einsum("ntvd,kd->ntvk", z, self.scorers[i].to(z.dtype)) / scale
        return (attn * score).sum(2) + self.biases[i]

    def run_set(self, x, index_t, i):
        """Concept logits (B, steps, K) of set i, from the joints of that set only."""
        s = self.sets[i]
        xs = x[:, :, :, getattr(self, f"joints_{i}"), :]   # B C T V M
        B, C, T, V, M = xs.shape
        flat = xs.permute(0, 4, 1, 2, 3).reshape(B * M, C, T, V)
        if s.two_person:
            feats = self.modalities(flat, i).reshape(B, M, 12, T, V)
            feats = feats.permute(0, 2, 3, 1, 4).reshape(B, 12, T, M * V)
            return self.encode(feats, index_t, i)

        valid = flat.abs().sum(dim=(1, 2, 3)) > 0
        K = self.queries[i].shape[0]
        if not valid.any():
            return xs.new_full((B, self.steps, K), ABSENT_LOGIT)
        logits = self.encode(self.modalities(flat[valid], i), index_t.repeat_interleave(M, dim=0)[valid], i)
        full = logits.new_full((B * M, self.steps, K), ABSENT_LOGIT)
        full[valid] = logits
        return full.reshape(B, M, self.steps, K).amax(dim=1)

    def concept_timeline(self, x, index_t):
        """Logits (B, steps, num_timeline): named and free concepts of all sets, set by set."""
        return torch.cat([self.run_set(x, index_t, i) for i in range(len(self.sets))], dim=2)

    def pool_named(self, timeline):
        """Clip-level logit of each named concept: soft max for events, soft min for states."""
        c = timeline[:, :, self.named_idx].float()
        tau = self.tau_sign * (F.softplus(self.tau_raw.float()) + 0.1)
        return tau * (torch.logsumexp(c / tau, dim=1) - math.log(self.steps))

    def forward(self, x, index_t):
        """x (B, 3, T, 25, 2) joint coordinates, index_t (B, T)."""
        timeline = self.concept_timeline(x, index_t)
        concept_logits = self.pool_named(timeline)
        logits_named = torch.sigmoid(concept_logits) @ self.W.t() + self.class_bias

        c = torch.tanh(timeline / 2).transpose(1, 2)
        a1 = F.relu(self.motif(c))
        a2 = F.relu(self.order(a1))
        a = torch.cat([a1.amax(dim=-1), a2.amax(dim=-1), a2.mean(dim=-1)], dim=1)
        logits = logits_named + self.motif_head(self.head_drop(a))
        return {"logits": logits, "logits_named": logits_named, "concept_logits": concept_logits,
                "timeline": timeline}
