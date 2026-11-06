"""
Fuzzy AND/OR Logic Tree
========================
A differentiable binary tree whose internal nodes are real AND/OR gates
(t-norm / t-conorm) and whose leaves are (possibly negated) named-concept
literals, trained jointly with a concept-bottleneck extractor and later
hardened into an explicit nested Boolean formula per class.

Node combination (soft, beta in [0,1] learned per node, hardens to AND if
beta > 0.5 else OR):
    AND(a, b) = a * b                      (product t-norm)
    OR(a, b)  = a + b - a * b              (probabilistic sum t-conorm)
    node      = beta * AND(a, b) + (1 - beta) * OR(a, b)

Leaf literal (polarity in [0,1] learned per leaf, hardens to identity if
polarity > 0.5 else NOT):
    literal = polarity * p + (1 - polarity) * (1 - p)

All parameters are class-specific but batched as tensors (no per-class
Python objects). `temperature` is a plain float meant to be annealed by the
training loop (lower -> sharper softmax/sigmoid -> closer to a hard tree).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyLogicTree(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        num_class: int,
        depth: int = 4,
        entropy_weight: float = 0.01,
        diversity_weight: float = 0.05,
        concept_candidate_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.num_concepts = num_concepts
        self.num_class = num_class
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_internal = self.num_leaves - 1
        self.entropy_weight = float(entropy_weight)
        self.diversity_weight = float(diversity_weight)
        self.temperature = 1.0  # annealed externally by the training loop

        if concept_candidate_mask is None:
            concept_candidate_mask = torch.ones(num_class, num_concepts)
        self.register_buffer("concept_candidate_mask", concept_candidate_mask.float())

        self.leaf_selector_logits = nn.Parameter(
            torch.randn(num_class, self.num_leaves, num_concepts) * 0.01
        )
        self.leaf_polarity_logits = nn.Parameter(torch.zeros(num_class, self.num_leaves))
        self.node_gate_logits = nn.Parameter(torch.zeros(num_class, self.num_internal))

    def _masked_leaf_logits(self) -> torch.Tensor:
        """leaf_selector_logits with concepts outside this class's KG-neighborhood
        (see concept_candidate_mask, built from ontology Jaccard similarity) driven
        to ~0 softmax probability -- keeps every leaf's candidate pool restricted to
        concepts that are actually present in this class or one of its nearest
        ontology neighbors, instead of a free choice over all num_concepts."""
        mask = self.concept_candidate_mask.unsqueeze(1)  # (num_class, 1, num_concepts)
        return self.leaf_selector_logits.masked_fill(mask <= 0, -1e9)

    def forward(self, concept_probs: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """concept_probs: (B, num_concepts) -> root_value: (B, num_class) in [0, 1]."""
        T = max(self.temperature, 1e-3)
        leaf_logits = self._masked_leaf_logits()

        if hard:
            sel_idx = leaf_logits.argmax(dim=-1)  # (num_class, num_leaves)
            literal_val = concept_probs[:, sel_idx]             # (B, num_class, num_leaves)
            polarity = (self.leaf_polarity_logits > 0).float()  # (num_class, num_leaves)
        else:
            sel = F.softmax(leaf_logits / T, dim=-1)  # (num_class, num_leaves, num_concepts)
            literal_val = torch.einsum("bk,clk->bcl", concept_probs, sel)  # (B, num_class, num_leaves)
            polarity = torch.sigmoid(self.leaf_polarity_logits / T)  # (num_class, num_leaves)

        polarity = polarity.unsqueeze(0)
        values = polarity * literal_val + (1.0 - polarity) * (1.0 - literal_val)
        if hard:
            # Snap literal truth values to exact {0, 1} too -- otherwise AND (product
            # t-norm) keeps multiplying continuous probabilities and any class whose
            # hardened formula happens to nest AND more deeply shrinks towards 0 purely
            # from chain length, independent of whether the input actually satisfies it.
            # With true booleans, product-AND/prob-sum-OR reduce exactly to boolean
            # AND/OR, so depth no longer biases the comparison across classes.
            values = (values > 0.5).float()

        offset = 0
        level_size = self.num_leaves
        while level_size > 1:
            half = level_size // 2
            left = values[..., 0::2]
            right = values[..., 1::2]
            beta_logit = self.node_gate_logits[:, offset : offset + half]  # (num_class, half)
            if hard:
                beta = (beta_logit > 0).float()
            else:
                beta = torch.sigmoid(beta_logit / T)
            beta = beta.unsqueeze(0)
            and_val = left * right
            or_val = left + right - left * right
            values = beta * and_val + (1.0 - beta) * or_val
            offset += half
            level_size = half

        return values.squeeze(-1)  # (B, num_class)

    def loss(
        self,
        concept_probs: torch.Tensor,
        label: torch.Tensor,
        entropy_weight: float = None,
        diversity_weight: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ew = self.entropy_weight if entropy_weight is None else float(entropy_weight)
        dw = self.diversity_weight if diversity_weight is None else float(diversity_weight)

        root_value = self.forward(concept_probs, hard=False)
        logits = torch.logit(root_value.float().clamp(1e-4, 1.0 - 1e-4))
        ce = F.cross_entropy(logits, label)

        sel_probs = F.softmax(self._masked_leaf_logits(), dim=-1)  # (num_class, num_leaves, num_concepts)
        sel_entropy = -(sel_probs * sel_probs.clamp_min(1e-8).log()).sum(-1).mean()

        # Diversity: each leaf picks independently, so nothing stops two leaves in
        # the same class's tree converging onto the same concept -- restricting
        # candidates to the KG neighborhood (concept_candidate_mask) actually makes
        # this more likely, not less, since there are fewer distinct concepts to
        # spread 2^depth leaves across. Penalize pairwise dot-product overlap
        # between leaves' concept distributions (1.0 = identical pick, ~0 = disjoint).
        overlap = torch.bmm(sel_probs, sel_probs.transpose(-1, -2))  # (num_class, num_leaves, num_leaves)
        off_diag = 1.0 - torch.eye(self.num_leaves, device=overlap.device, dtype=overlap.dtype)
        num_pairs = max(self.num_leaves * (self.num_leaves - 1), 1)
        leaf_diversity_penalty = (overlap * off_diag).sum(dim=(-1, -2)).div(num_pairs).mean()

        gate_p = torch.sigmoid(self.node_gate_logits)
        gate_entropy = -(
            gate_p * gate_p.clamp_min(1e-8).log()
            + (1.0 - gate_p) * (1.0 - gate_p).clamp_min(1e-8).log()
        ).mean()

        pol_p = torch.sigmoid(self.leaf_polarity_logits)
        pol_entropy = -(
            pol_p * pol_p.clamp_min(1e-8).log()
            + (1.0 - pol_p) * (1.0 - pol_p).clamp_min(1e-8).log()
        ).mean()

        entropy = sel_entropy + gate_entropy + pol_entropy
        total = ce + ew * entropy + dw * leaf_diversity_penalty
        return total, {
            "ce": ce.detach(),
            "entropy": entropy.detach(),
            "leaf_diversity": leaf_diversity_penalty.detach(),
            "sel_entropy": sel_entropy.detach(),
            "gate_entropy": gate_entropy.detach(),
            "pol_entropy": pol_entropy.detach(),
        }

    @torch.no_grad()
    def harden(self) -> Dict[str, torch.Tensor]:
        """Snap every soft decision to its discrete counterpart + report confidence."""
        leaf_logits = self._masked_leaf_logits()
        sel_idx = leaf_logits.argmax(dim=-1)  # (num_class, num_leaves)
        sel_probs = F.softmax(leaf_logits, dim=-1)
        sel_purity = sel_probs.gather(-1, sel_idx.unsqueeze(-1)).squeeze(-1)  # (num_class, num_leaves)

        polarity_hard = self.leaf_polarity_logits > 0  # True = keep concept, False = NOT
        polarity_conf = torch.sigmoid(self.leaf_polarity_logits.abs())  # closer to 1 = more decided

        gate_hard = self.node_gate_logits > 0  # True = AND, False = OR
        gate_conf = torch.sigmoid(self.node_gate_logits.abs())

        return {
            "sel_idx": sel_idx,
            "sel_purity": sel_purity,
            "polarity_hard": polarity_hard,
            "polarity_conf": polarity_conf,
            "gate_hard": gate_hard,
            "gate_conf": gate_conf,
        }

    @torch.no_grad()
    def hardening_report(self) -> Dict[str, float]:
        d = self.harden()
        return {
            "mean_leaf_selector_purity": float(d["sel_purity"].mean()),
            "min_leaf_selector_purity": float(d["sel_purity"].min()),
            "mean_polarity_confidence": float(d["polarity_conf"].mean()),
            "mean_gate_confidence": float(d["gate_conf"].mean()),
        }

    @torch.no_grad()
    def explain_hard(
        self,
        concept_probs: torch.Tensor,
        class_idx: int,
        concept_names: List[str],
        threshold: float = 0.5,
    ) -> Dict:
        """Literal-by-literal trace of the hardened rule for `class_idx` on ONE sample.
        concept_probs: (num_concepts,) or (1, num_concepts)."""
        if concept_probs.dim() == 2:
            concept_probs = concept_probs[0]
        d = self.harden()
        sel_idx = d["sel_idx"][class_idx].tolist()
        polarity_hard = d["polarity_hard"][class_idx].tolist()
        gate_hard = d["gate_hard"][class_idx].tolist()

        literals = []
        values = []
        for l in range(self.num_leaves):
            cidx = sel_idx[l]
            p = float(concept_probs[cidx])
            keep = bool(polarity_hard[l])
            truth = (p > threshold) if keep else (p <= threshold)
            name = concept_names[cidx] if keep else f"NOT {concept_names[cidx]}"
            literals.append({"literal": name, "concept_prob": round(p, 4), "satisfied": bool(truth)})
            values.append(bool(truth))

        offset = 0
        level_size = self.num_leaves
        while level_size > 1:
            half = level_size // 2
            new_values = []
            for i in range(half):
                left, right = values[2 * i], values[2 * i + 1]
                is_and = bool(gate_hard[offset + i])
                new_values.append((left and right) if is_and else (left or right))
            values = new_values
            offset += half
            level_size = half

        return {"literals": literals, "result": bool(values[0])}

    @torch.no_grad()
    def export_formula(self, concept_names: List[str]) -> List[str]:
        """One nested Boolean string per class, e.g. 'class_3 <=> ((a AND NOT b) OR c)'."""
        d = self.harden()
        sel_idx = d["sel_idx"].tolist()
        polarity_hard = d["polarity_hard"].tolist()
        gate_hard = d["gate_hard"].tolist()

        formulas = []
        for c in range(self.num_class):
            values = [
                concept_names[sel_idx[c][l]] if polarity_hard[c][l] else f"NOT {concept_names[sel_idx[c][l]]}"
                for l in range(self.num_leaves)
            ]
            offset = 0
            level_size = self.num_leaves
            while level_size > 1:
                half = level_size // 2
                new_values = []
                for i in range(half):
                    left, right = values[2 * i], values[2 * i + 1]
                    # X AND X == X AND X OR X == X (idempotence) -- collapse identical
                    # branches instead of printing a redundant nested duplicate.
                    if left == right:
                        new_values.append(left)
                    else:
                        op = "AND" if gate_hard[c][offset + i] else "OR"
                        new_values.append(f"({left} {op} {right})")
                values = new_values
                offset += half
                level_size = half
            formulas.append(f"class_{c} <=> {values[0]}")
        return formulas
