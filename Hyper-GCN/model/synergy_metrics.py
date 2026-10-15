"""Explanation metrics of the synergy bottleneck (SEMANTIC_HYPERGRAPH_PLAN.md, section 3).

Readout-level deletion / insertion use the additivity of the readout: setting
g[e,s] = 0 removes contrib[:, e, s] from every class logit. This leaves G_sem in
the backbone untouched; evaluate_synergy.py also runs the whole-network version.
"""

from __future__ import annotations

import json

import numpy as np
import torch


def readout_logits(contributions, bias, order, keep_first):
    """Logits [N,K,P+1] after removing (``keep_first`` False: deletion) or
    keeping only (True: insertion) the first 0..P entries of ``order`` [N,P]."""
    N, K = contributions.shape[:2]
    flat = contributions.flatten(2)
    ordered = flat.gather(2, order[:, None, :].expand(-1, K, -1))
    cumulative = torch.cat((flat.new_zeros(N, K, 1), ordered.cumsum(2)), dim=2)
    bias = bias.to(flat)[None, :, None]
    if keep_first:
        return bias + cumulative
    return flat.sum(2, keepdim=True) + bias - cumulative


def predicted_probability(logits, predicted):
    """Softmax probability of ``predicted`` [N] at every step: [N,P+1]."""
    probability = logits.softmax(dim=1)
    return probability.gather(1, predicted[:, None, None].expand(-1, 1, logits.shape[2])).squeeze(1)


@torch.no_grad()
def readout_faithfulness(contributions, bias, generator=None):
    """Deletion / insertion curves of the predicted class, ordered by its
    contributions (most important first) and in a random order.

    Returns per-sample AUCs (mean probability over the P+1 steps).
    """
    contributions = contributions.float()
    logits = contributions.sum(dim=(-1, -2)) + bias.to(contributions)
    predicted = logits.argmax(dim=1)
    rows = torch.arange(len(predicted), device=predicted.device)
    own = contributions[rows, predicted].flatten(1)  # N,P
    ranked = own.argsort(dim=1, descending=True)
    shuffled = torch.rand(own.shape, generator=generator, device=own.device).argsort(dim=1)
    result = {}
    for name, order in (("", ranked), ("random_", shuffled)):
        for kind, keep_first in (("deletion", False), ("insertion", True)):
            curve = predicted_probability(readout_logits(contributions, bias, order, keep_first), predicted)
            result[name + kind + "_auc"] = curve.mean(dim=1)
    return result


def effective_synergies(contributions, predicted, fraction=0.8):
    """Number of (synergy, segment) entries needed for ``fraction`` of the
    absolute contribution to the predicted class: [N]."""
    rows = torch.arange(len(predicted), device=predicted.device)
    share = contributions[rows, predicted].flatten(1).abs()
    share = share.sort(dim=1, descending=True).values
    cumulative = share.cumsum(1) / share.sum(1, keepdim=True).clamp_min(1e-8)
    return (cumulative < fraction).sum(1) + 1


def load_annotations(path, class_names, synergy_names):
    """Relevance labels {0,1,2} [K,E] from annotations/class_synergy.json."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    labels = np.zeros((len(class_names), len(synergy_names)))
    index = {name: e for e, name in enumerate(synergy_names)}
    for k, name in enumerate(class_names):
        for synergy, value in data["labels"].get(name, {}).items():
            if synergy in index:
                labels[k, index[synergy]] = value
    return labels


def plausibility(profile, annotations, k=3):
    """Precision@k and nDCG@k of each class's top synergies against the annotation.

    profile [K,E]: mean contribution of each synergy to its class. Classes
    without any relevant synergy in the annotation are skipped.
    """
    precision, ndcg = [], []
    for scores, labels in zip(profile, annotations):
        if labels.max() <= 0:
            continue
        top = np.argsort(-scores)[:k]
        precision.append(float((labels[top] > 0).mean()))
        gains = 2.0 ** labels - 1
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        ideal = (np.sort(gains)[::-1][:k] * discounts).sum()
        ndcg.append(float((gains[top] * discounts).sum() / ideal))
    return {"precision_at_k": float(np.mean(precision)) if precision else None,
            "ndcg_at_k": float(np.mean(ndcg)) if ndcg else None,
            "k": k, "classes": len(precision)}


def topk_sets(profile, k=3):
    return [set(np.argsort(-scores)[:k].tolist()) for scores in profile]


def jaccard(first, second):
    """Mean Jaccard similarity of two lists of top-k sets."""
    return float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in zip(first, second)]))


class SynergyAccumulator:
    """Collects bottleneck outputs over an evaluation pass."""

    def __init__(self, bottleneck, class_names, max_explanations=16):
        self.bottleneck = bottleneck
        self.class_names = class_names
        self.max_explanations = max_explanations
        E, K = bottleneck.num_synergies, bottleneck.num_classes
        self.profile_sum = np.zeros((K, E, bottleneck.segments))
        self.profile_count = np.zeros(K)
        self.values = {}
        self.gates = []
        self.explanations = []
        self.generator = None

    @torch.no_grad()
    def update(self, output, label):
        contributions = output.contributions.float()
        bias = self.bottleneck.class_bias.detach()
        if self.generator is None:
            self.generator = torch.Generator(device=contributions.device).manual_seed(0)
        for key, value in readout_faithfulness(contributions, bias, self.generator).items():
            self.values.setdefault(key, []).append(value.cpu().numpy())
        predicted = output.action_logits.argmax(dim=1)
        self.values.setdefault("effective_synergies", []).append(
            effective_synergies(contributions, predicted).float().cpu().numpy())
        rows = torch.arange(len(label), device=label.device)
        own = contributions[rows, label].cpu().numpy()  # N,E,S
        np.add.at(self.profile_sum, label.cpu().numpy(), own)
        np.add.at(self.profile_count, label.cpu().numpy(), 1)
        self.gates.append(output.gates.float().cpu().numpy())
        if len(self.explanations) < self.max_explanations:
            take = self.max_explanations - len(self.explanations)
            self.explanations += self.bottleneck.explain(
                contributions[:take], output.action_logits[:take], self.class_names)

    def profile(self):
        """Mean contribution [K,E,S] of every synergy and segment to its true class."""
        return self.profile_sum / np.maximum(self.profile_count, 1)[:, None, None]

    def summary(self, annotations=None, k=3):
        gates = np.concatenate(self.gates)
        names = self.bottleneck.names
        profile = self.profile()
        class_profile = profile.sum(-1)
        result = {key: float(np.concatenate(value).mean()) for key, value in self.values.items()}
        result["deletion_gap"] = result["random_deletion_auc"] - result["deletion_auc"]
        result["insertion_gap"] = result["insertion_auc"] - result["random_insertion_auc"]
        result["gate_mean"] = float(gates.mean())
        result["gate_saturated_low"] = float((gates < 0.05).mean())
        result["gate_saturated_high"] = float((gates > 0.95).mean())
        result["gate_mean_per_synergy"] = dict(zip(names, gates.mean(axis=(0, 1)).tolist()))
        share = np.abs(class_profile) / np.maximum(np.abs(class_profile).sum(1, keepdims=True), 1e-8)
        result["dominant_synergy_share"] = dict(zip(names, share.mean(0).tolist()))
        result["class_top_synergies"] = {
            self.class_names[c]: [names[e] for e in np.argsort(-class_profile[c])[:k]]
            for c in range(len(class_profile)) if self.profile_count[c] > 0
        }
        if annotations is not None:
            result["plausibility"] = plausibility(class_profile, annotations, k)
        return result
