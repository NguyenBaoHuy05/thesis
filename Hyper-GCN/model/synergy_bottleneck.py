"""Synergy bottleneck: action logits computed only from gated, named synergies.

    f[e,s] = LN(mean_{joint in e, t in s} feature)       (pooling, no affine LN)
    g[e,s] = sigmoid(MLP_e(f[e,s]))                       (one gate MLP per synergy)
    logit_c = sum_{e,s} g[e,s] * scale/sqrt(C) * <unit(w_c,e), f[e,s]> + b_c

contrib[c,e,s] is each summand, so the logits are exactly the sum of the
contributions. The gate scale is identifiable: |<unit(w), f>| <= sqrt(C) and the
scale is fixed, so g cannot shrink while w grows. See SEMANTIC_HYPERGRAPH_PLAN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from model.synergies import incidence


@dataclass
class SynergyOutput:
    action_logits: torch.Tensor                  # [N,K]
    contributions: torch.Tensor                  # [N,K,E,S], sums to action_logits - bias
    gates: torch.Tensor                          # [N,S,E] of the actor behind the predicted class
    gate_penalty: torch.Tensor                   # mean gate over valid actors
    hyper_outputs: tuple[torch.Tensor, ...]


class SynergyPooling(nn.Module):
    """[..., C, T, V] -> [..., S, E, C]: mean over the joints of each group and
    over each temporal segment."""

    def __init__(self, groups, segments=4, num_joints=25):
        super().__init__()
        self.segments = int(segments)
        self.num_joints = int(num_joints)
        matrix = incidence(groups, num_joints)
        self.register_buffer("assignment", matrix / matrix.sum(0, keepdim=True), persistent=False)

    def forward(self, features):
        if features.shape[-2] < self.segments:
            raise ValueError(f"cannot split {features.shape[-2]} time steps into {self.segments} segments")
        features = features[..., :self.num_joints]
        pooled = torch.einsum("...ctv,ve->...cte", features, self.assignment.to(features))
        chunks = torch.tensor_split(pooled, self.segments, dim=-2)
        return torch.stack([chunk.mean(-2) for chunk in chunks], dim=-3).transpose(-1, -2)


class SynergyGate(nn.Module):
    """One small MLP per synergy: f [..., S, E, C] -> g [..., S, E] in [0, 1]."""

    def __init__(self, channels, num_synergies, hidden=64, temperature=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("gate temperature must be positive")
        self.temperature = float(temperature)
        self.weight1 = nn.Parameter(torch.randn(num_synergies, channels, hidden) / math.sqrt(channels))
        self.bias1 = nn.Parameter(torch.zeros(num_synergies, hidden))
        self.weight2 = nn.Parameter(torch.randn(num_synergies, hidden) / math.sqrt(hidden))
        self.bias2 = nn.Parameter(torch.zeros(num_synergies))

    def forward(self, f):
        hidden = F.gelu(torch.einsum("...ec,ech->...eh", f, self.weight1) + self.bias1)
        return torch.sigmoid(((hidden * self.weight2).sum(-1) + self.bias2) / self.temperature)


class SynergyBottleneck(nn.Module):
    def __init__(self, channels, synergies, num_classes, segments=4, gate_hidden=64,
                 gate_temperature=1.0, gate_weight=1e-3, readout_scale=5.0, dropout=0.0):
        super().__init__()
        if gate_weight < 0 or readout_scale <= 0:
            raise ValueError("gate_weight must be nonnegative and readout_scale positive")
        self.names = tuple(s.name for s in synergies)
        self.segments = int(segments)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.gate_weight = float(gate_weight)
        self.readout_scale = float(readout_scale)  # largest contribution of one open gate

        self.pooling = SynergyPooling([s.joints for s in synergies], segments)
        # No affine terms: a learnable LN gain would rescale f and undo the scale bound.
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.gate = SynergyGate(channels, len(synergies), gate_hidden, gate_temperature)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.class_weight = nn.Parameter(torch.randn(num_classes, len(synergies), channels))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def num_synergies(self):
        return len(self.names)

    def weight(self):
        """scale * unit(w_c,e) / sqrt(C): |<w, f>| <= scale for the normalized f."""
        return F.normalize(self.class_weight, dim=-1) * (self.readout_scale / math.sqrt(self.channels))

    def forward(self, features, actor_valid, gate_mask=None):
        """features [N,M,C,T,V], actor_valid [N,M], gate_mask [N,S,E] (deletion tests)."""
        N = features.shape[0]
        f = self.norm(self.pooling(features))  # N,M,S,E,C
        g = self.gate(f)  # N,M,S,E
        if gate_mask is not None:
            g = g * gate_mask[:, None]
        contributions = (torch.einsum("nmsec,kec->nmkes", self.dropout(f), self.weight())
                         * g.transpose(-1, -2).unsqueeze(2))  # N,M,K,E,S

        # Actor order is arbitrary: every class takes its best valid actor.
        actor_valid = actor_valid.clone()
        actor_valid[:, 0] |= ~actor_valid.any(dim=1)
        scores = contributions.sum(dim=(-1, -2)).masked_fill(~actor_valid[:, :, None], float("-inf"))
        actor = scores.argmax(dim=1)  # N,K
        index = actor[:, None, :, None, None].expand(-1, 1, -1, *contributions.shape[-2:])
        contributions = contributions.gather(1, index).squeeze(1)  # N,K,E,S

        logits = contributions.sum(dim=(-1, -2)) + self.class_bias
        predicted_actor = actor.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)
        valid = actor_valid[..., None, None].to(g)
        return dict(
            action_logits=logits,
            contributions=contributions,
            gates=g[torch.arange(N, device=g.device), predicted_actor],
            gate_penalty=(g * valid).sum() / (valid.sum() * g.shape[-1] * g.shape[-2]).clamp_min(1),
        )

    def total_loss(self, ce_loss, output):
        """L = CE + lambda_g * mean(g)."""
        return ce_loss + self.gate_weight * output.gate_penalty, dict(gate=output.gate_penalty)

    @torch.no_grad()
    def explain(self, contributions, logits, class_names, top_k=5):
        """Top (synergy, segment) contributions to the predicted class of each sample."""
        traces = []
        predicted = logits.argmax(dim=1)
        for n in range(len(predicted)):
            k = int(predicted[n])
            contrib = contributions[n, k].flatten()  # E*S
            total = contrib.abs().sum().clamp_min(1e-8)
            traces.append({
                "predicted": class_names[k] if k < len(class_names) else k,
                "logit": float(logits[n, k]),
                "top": [
                    {"synergy": self.names[i // self.segments], "segment": i % self.segments,
                     "contribution": float(contrib[i]), "share": float(contrib[i] / total)}
                    for i in contrib.argsort(descending=True)[:top_k].tolist()
                ],
            })
        return traces

    def audit(self):
        return {
            "mode": "synergy_bottleneck",
            "segments": self.segments,
            "num_classes": self.num_classes,
            "synergies": list(self.names),
            "readout_scale": self.readout_scale,
            "gate_weight": self.gate_weight,
        }
