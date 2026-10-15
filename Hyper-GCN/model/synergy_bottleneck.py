"""Synergy bottleneck: action logits computed only from gated, named synergies.

    f[e,s] = mean_{joint in e, t in s} feature        (pooling)
    g[e,s] = sigmoid(MLP_e(LN(f[e,s])))                (gate, or tied to W_sem of l9)
    logit_c = sum_{e,s} g[e,s] <w_c,e, f[e,s]> + b_c   (readout "embedding", B2)
    logit_c = sum_{e,s} w_c,e,s g[e,s] + b_c           (readout "scalar", B1)

contrib[c,e,s] is each summand, so the logits are exactly the sum of the
contributions. See SEMANTIC_HYPERGRAPH_PLAN.md, sections 2.2-2.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from model.synergies import incidence

READOUTS = ("embedding", "scalar")


@dataclass
class SynergyOutput:
    action_logits: torch.Tensor                  # [N,K]
    contributions: torch.Tensor                  # [N,K,E,S], sums to action_logits - bias
    gates: torch.Tensor                          # [N,S,E] of the actor behind the predicted class
    gate_penalty: torch.Tensor                   # mean gate over valid actors and pairs
    hyper_outputs: tuple[torch.Tensor, ...]


def segment_index(length, segments, device=None):
    """Segment of every time step, matching torch.tensor_split."""
    if length < segments:
        raise ValueError(f"cannot split {length} time steps into {segments} segments")
    sizes = [len(chunk) for chunk in torch.tensor_split(torch.arange(length), segments)]
    return torch.repeat_interleave(torch.arange(segments, device=device),
                                   torch.tensor(sizes, device=device))


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
        self.norm = nn.LayerNorm(channels)
        self.weight1 = nn.Parameter(torch.randn(num_synergies, channels, hidden) / math.sqrt(channels))
        self.bias1 = nn.Parameter(torch.zeros(num_synergies, hidden))
        self.weight2 = nn.Parameter(torch.randn(num_synergies, hidden) / math.sqrt(hidden))
        self.bias2 = nn.Parameter(torch.zeros(num_synergies))

    def forward(self, f):
        hidden = F.gelu(torch.einsum("...ec,ech->...eh", self.norm(f), self.weight1) + self.bias1)
        return torch.sigmoid(((hidden * self.weight2).sum(-1) + self.bias2) / self.temperature)


class SemanticHyperedges(nn.Module):
    """Segment-gated semantic hypergraph G_sem(s) for one HYPERGC layer.

    G_sem(s) = Dv^-1 H diag(g_s) De^-1 H^T with the *unweighted* vertex degree
    Dv. HYPERGC.hyper_norm divides by the weighted degree, which cancels the
    gate of a joint that lies in one synergy only; here a gate of 0 removes
    the synergy and the row sum of a joint is the mean gate of its synergies.
    """

    def __init__(self, channels, groups, num_vertices, segments=4, gate_hidden=64,
                 gate_temperature=1.0):
        super().__init__()
        self.segments = int(segments)
        self.pooling = SynergyPooling(groups, segments)
        self.gate = SynergyGate(channels, len(groups), gate_hidden, gate_temperature)
        H = incidence(groups, num_vertices)
        self.register_buffer("left", H / H.sum(1, keepdim=True).clamp_min(1), persistent=False)
        self.register_buffer("right", H / H.sum(0, keepdim=True), persistent=False)
        # beta = 0 at initialization keeps a pretrained layer unchanged. It is
        # not passed through relu: relu has zero gradient at 0, so beta would
        # never leave its initial value.
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, gate_mask=None):
        """x [B,C,T,V+virtual] -> per-frame adjacency [B,T,V',V'] and gates [B,S,E]."""
        gate = self.gate(self.pooling(x))
        if gate_mask is not None:
            gate = gate * gate_mask
        G = torch.einsum("ue,bse,ve->bsuv", self.left.to(x), gate, self.right.to(x))
        return G[:, segment_index(x.shape[2], self.segments, x.device)], gate


class SynergyBottleneck(nn.Module):
    def __init__(self, channels, singles, pairs, num_classes, segments=4, readout="embedding",
                 tied=True, gate_hidden=64, gate_temperature=1.0, gate_weight=1e-3,
                 entropy_weight=0.0, dropout=0.0):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"synergy readout must be one of {READOUTS}")
        if gate_weight < 0 or entropy_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        self.readout = readout
        self.tied = bool(tied)
        self.segments = int(segments)
        self.num_classes = int(num_classes)
        self.gate_weight = float(gate_weight)
        self.entropy_weight = float(entropy_weight)
        self.single_names = tuple(s.name for s in singles)
        self.pair_names = tuple(p.name for p in pairs)
        self.names = self.single_names + self.pair_names
        E1, E = len(singles), len(self.names)

        self.pooling = SynergyPooling([s.joints for s in singles], segments)
        self.norm = nn.LayerNorm(channels)
        self.gate = None if self.tied else SynergyGate(channels, E1, gate_hidden, gate_temperature)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        self.pair_pooling = self.pair_phi = self.pair_norm = self.pair_gate = None
        if pairs:
            groups = list(dict.fromkeys(g for p in pairs for g in (p.joints_a, p.joints_b)))
            self.pair_pooling = SynergyPooling(groups, segments)
            self.register_buffer("pair_first", torch.tensor(
                [groups.index(p.joints_a) for p in pairs]), persistent=False)
            self.register_buffer("pair_second", torch.tensor(
                [groups.index(p.joints_b) for p in pairs]), persistent=False)
            self.pair_phi = nn.Sequential(
                nn.Linear(2 * channels, channels), nn.GELU(), nn.Linear(channels, channels),
            )
            self.pair_norm = nn.LayerNorm(channels)
            self.pair_gate = SynergyGate(channels, len(pairs), gate_hidden, gate_temperature)

        if readout == "embedding":
            self.class_weight = nn.Parameter(
                torch.randn(num_classes, E, channels) / math.sqrt(channels * E * segments))
        else:
            self.class_weight = nn.Parameter(
                torch.randn(num_classes, E, segments) / math.sqrt(E * segments))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def num_synergies(self):
        return len(self.names)

    def _contributions(self, f, g, synergies):
        """f [...,S,E,C], g [...,S,E] -> contrib [...,K,E,S] for the given slice of synergies."""
        weight = self.class_weight[:, synergies]
        gate = g.transpose(-1, -2).unsqueeze(-3)  # ...,1,E,S
        if self.readout == "embedding":
            return torch.einsum("...sec,kec->...kes", self.dropout(f), weight) * gate
        return weight * gate

    def _pairs(self, features, pair_valid):
        """Symmetric two-actor synergies: f [N,S,P,C] and gates [N,S,P]."""
        groups = self.norm(self.pair_pooling(features[:, :2]))  # N,2,S,G,C
        a, b = groups[:, 0], groups[:, 1]
        first, second = self.pair_first, self.pair_second
        f = (self.pair_phi(torch.cat((a[..., first, :], b[..., second, :]), dim=-1))
             + self.pair_phi(torch.cat((b[..., first, :], a[..., second, :]), dim=-1)))
        f = self.pair_norm(f)
        g = self.pair_gate(f) * pair_valid[:, None, None].to(f)
        return f, g

    def forward(self, features, actor_valid, pair_valid=None, tied_gate=None, gate_mask=None):
        """features [N,M,C,T,V] (last backbone layer), actor_valid [N,M],
        pair_valid [N], tied_gate [N,M,S,E1] (W_sem of l9), gate_mask [N,S,E]."""
        N, M = features.shape[:2]
        E1 = len(self.single_names)
        f = self.norm(self.pooling(features))  # N,M,S,E1,C
        if self.tied:
            if tied_gate is None:
                raise ValueError("a tied bottleneck needs the gates of the last semantic layer")
            g = tied_gate
        else:
            g = self.gate(f)
        if gate_mask is not None:
            g = g * gate_mask[:, None, :, :E1]
        single = self._contributions(f, g, slice(0, E1))  # N,M,K,E1,S

        # Actor order is arbitrary: every class takes its best valid actor.
        actor_valid = actor_valid.clone()
        actor_valid[:, 0] |= ~actor_valid.any(dim=1)
        actor_scores = single.sum(dim=(-1, -2)).masked_fill(~actor_valid[:, :, None], float("-inf"))
        actor = actor_scores.argmax(dim=1)  # N,K
        index = actor[:, None, :, None, None].expand(-1, 1, -1, *single.shape[-2:])
        contributions = single.gather(1, index).squeeze(1)  # N,K,E1,S
        valid = actor_valid[..., None, None].to(g)
        penalty_sum, penalty_count = (g * valid).sum(), valid.sum() * g.shape[-1] * g.shape[-2]

        pair_gates = None
        if self.pair_gate is not None:
            if M >= 2 and pair_valid is not None:
                fp, gp = self._pairs(features, pair_valid)
                if gate_mask is not None:
                    gp = gp * gate_mask[..., E1:]
                pair = self._contributions(fp, gp, slice(E1, None))  # N,K,P,S
                pair_weight = pair_valid[:, None, None].to(gp)
                penalty_sum = penalty_sum + gp.sum()
                penalty_count = penalty_count + pair_weight.sum() * gp.shape[-1] * gp.shape[-2]
            else:
                P = len(self.pair_names)
                pair = contributions.new_zeros(N, self.num_classes, P, self.segments)
                gp = g.new_zeros(N, self.segments, P)
            contributions = torch.cat((contributions, pair), dim=2)
            pair_gates = gp

        logits = contributions.sum(dim=(-1, -2)) + self.class_bias
        predicted_actor = actor.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)
        gates = g[torch.arange(N, device=g.device), predicted_actor]  # N,S,E1
        if pair_gates is not None:
            gates = torch.cat((gates, pair_gates), dim=-1)
        return dict(
            action_logits=logits,
            contributions=contributions,
            gates=gates,
            gate_penalty=penalty_sum / penalty_count.clamp_min(1),
        )

    def total_loss(self, ce_loss, output, label):
        """L = CE + lambda_g mean(g) + lambda_e entropy of the label's contributions."""
        entropy = ce_loss.new_zeros(())
        if self.entropy_weight > 0:
            rows = torch.arange(len(label), device=label.device)
            share = output.contributions[rows, label].abs().flatten(1)
            share = share / share.sum(-1, keepdim=True).clamp_min(1e-8)
            entropy = -(share * share.clamp_min(1e-8).log()).sum(-1).mean()
        total = ce_loss + self.gate_weight * output.gate_penalty + self.entropy_weight * entropy
        return total, dict(gate=output.gate_penalty, entropy=entropy)

    @torch.no_grad()
    def explain(self, contributions, logits, class_names, top_k=5):
        """Top (synergy, segment) contributions to the predicted class of each sample."""
        traces = []
        predicted = logits.argmax(dim=1)
        for n in range(len(predicted)):
            k = int(predicted[n])
            contrib = contributions[n, k]  # E,S
            total = contrib.abs().sum().clamp_min(1e-8)
            order = contrib.flatten().argsort(descending=True)[:top_k].tolist()
            traces.append({
                "predicted": class_names[k] if k < len(class_names) else k,
                "logit": float(logits[n, k]),
                "top": [
                    {"synergy": self.names[i // self.segments], "segment": i % self.segments,
                     "contribution": float(contrib.flatten()[i]),
                     "share": float(contrib.flatten()[i] / total)}
                    for i in order
                ],
            })
        return traces

    def audit(self):
        return {
            "mode": "synergy_bottleneck",
            "readout": self.readout,
            "tied": self.tied,
            "segments": self.segments,
            "num_classes": self.num_classes,
            "single_synergies": list(self.single_names),
            "pair_synergies": list(self.pair_names),
            "gate_weight": self.gate_weight,
            "entropy_weight": self.entropy_weight,
        }
