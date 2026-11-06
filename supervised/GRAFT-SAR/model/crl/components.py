"""
Differentiable First-Order Logic Engine
=======================================
Implements differentiable t-norm logic operations (Conjunction, Disjunction, Negation)
with Soft Temperature-Annealed Binarization and automatic Rule Extraction.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-7


class SoftBinarizeLayer(nn.Module):
    """
    Soft Temperature-Annealed Binarizer.
    Provides smooth sigmoid relaxation with temperature scale k during training,
    and exact step binarization during evaluation/rule extraction.
    """

    def __init__(
        self,
        n_concepts: int,
        use_not: bool = True,
        initial_k: float = 3.0,
        k_max: float = 20.0,
    ):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.use_not = bool(use_not)
        self.input_dim = self.n_concepts
        self.output_dim = 2 * self.n_concepts if self.use_not else self.n_concepts
        self.register_buffer("k", torch.tensor(float(initial_k)))
        self.k_min = float(initial_k)
        self.k_max = float(k_max)

    def set_temperature(self, epoch: int, total_epochs: int):
        """Anneals temperature scale k from k_min to k_max smoothly."""
        progress = min(max(epoch / max(total_epochs, 1), 0.0), 1.0)
        new_k = self.k_min + progress * (self.k_max - self.k_min)
        self.k.fill_(new_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Concept probabilities in [0, 1] or raw logits.
        Returns: Soft binarized literals in [0, 1] (B, output_dim).
        """
        x_float = x.float()
        # If input is already in [0, 1], shift to center at 0.5
        if (x_float >= 0.0).all() and (x_float <= 1.0).all():
            centered = x_float - 0.5
        else:
            centered = x_float

        if self.training:
            # Soft continuous relaxation: sigmoid(k * centered)
            soft = torch.sigmoid(self.k * centered)
            # Straight-Through Estimator with soft backward gradient
            hard = (centered > 0.0).float()
            x_bin = hard + (soft - soft.detach())
        else:
            # Exact binarization at test time
            x_bin = (centered > 0.0).float()

        if self.use_not:
            x_bin = torch.cat([x_bin, 1.0 - x_bin], dim=-1)

        return x_bin.to(dtype=x.dtype)

    @torch.no_grad()
    def binarized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exact non-differentiable binarization for rule extraction."""
        x_float = x.float()
        if (x_float >= 0.0).all() and (x_float <= 1.0).all():
            centered = x_float - 0.5
        else:
            centered = x_float

        hard = (centered > 0.0).float()
        if self.use_not:
            hard = torch.cat([hard, 1.0 - hard], dim=-1)
        return hard.to(dtype=x.dtype)


class ConjunctionLayer(nn.Module):
    """
    Differentiable Conjunction Layer (L1: Learn local / composite clauses).
    Implements Product T-Norm:
        C_j = prod_i (1 - W_ij * (1 - x_i))
    where W_ij in [0, 1] indicates the selection of literal i in clause j.
    """

    def __init__(self, input_dim: int, output_dim: int, init_range: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize small weights so clauses initially pass information smoothly
        self.W = nn.Parameter(torch.empty(input_dim, output_dim).uniform_(0.01, init_range))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim) literals in [0, 1]
        Returns: (B, output_dim) conjunction activations in [0, 1]
        """
        # Clamp W to valid [0, 1] probability range
        W_clamped = torch.clamp(self.W, 0.0, 1.0)
        # 1 - (1 - x_i) * W_ij
        # (B, input_dim, 1) and (1, input_dim, output_dim)
        diff = 1.0 - (1.0 - x.unsqueeze(-1)) * W_clamped.unsqueeze(0)  # (B, input_dim, output_dim)
        # Product over input literals
        # Using log-sum-exp for numerical stability: exp(sum(log(clamp(diff, eps, 1.0))))
        log_diff = torch.log(torch.clamp(diff, min=EPSILON, max=1.0))
        out = torch.exp(torch.sum(log_diff, dim=1))  # (B, output_dim)
        return out

    def get_sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity penalty on selection weights W."""
        return torch.norm(self.W, p=1)


class DisjunctionLayer(nn.Module):
    """
    Differentiable Disjunction Layer (L2: Aggregate clauses into action classes).
    Implements Probabilistic Sum T-Conorm:
        D_k = 1 - prod_j (1 - W_jk * C_j)
    """

    def __init__(self, input_dim: int, output_dim: int, init_range: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(input_dim, output_dim).uniform_(0.01, init_range))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim) conjunction clauses in [0, 1]
        Returns: (B, output_dim) action class probabilities in [0, 1]
        """
        W_clamped = torch.clamp(self.W, 0.0, 1.0)
        # 1 - x_j * W_jk
        diff = 1.0 - x.unsqueeze(-1) * W_clamped.unsqueeze(0)  # (B, input_dim, output_dim)
        log_diff = torch.log(torch.clamp(diff, min=EPSILON, max=1.0))
        out = 1.0 - torch.exp(torch.sum(log_diff, dim=1))  # (B, output_dim)
        return out

    def get_sparsity_loss(self) -> torch.Tensor:
        return torch.norm(self.W, p=1)


class LinearLogicClassifier(nn.Module):
    """
    Linear Symbolic Aggregator with non-negative / bounded weights.
    Maps conjunction clauses to action class logits with interpretable weights.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def get_l1_loss(self) -> torch.Tensor:
        return torch.norm(self.fc.weight, p=1)


class ConceptLogicLayers(nn.Module):
    """
    Hierarchical Concept Reasoning Engine.
    Structure:
      Input (Concept Probabilities / Logits)
         |
      Soft Temperature Binarizer (Generates P and ~P literals)
         |
      L1 Conjunction Layer (M local / intermediate clauses)
         |
      L2 Disjunction / Linear Classifier (Action class logits y_logic)
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        num_rules: int = 128,
        use_not: bool = True,
        use_linear_classifier: bool = True,
        initial_k: float = 3.0,
        k_max: float = 20.0,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.use_not = use_not
        self.use_linear_classifier = use_linear_classifier

        # 1. Soft Binarizer
        self.binarizer = SoftBinarizeLayer(
            n_concepts=num_concepts,
            use_not=use_not,
            initial_k=initial_k,
            k_max=k_max,
        )

        literal_dim = self.binarizer.output_dim

        # 2. Conjunction Layer (L1)
        self.l1_conjunction = ConjunctionLayer(input_dim=literal_dim, output_dim=num_rules)

        # 3. Disjunction or Linear Classifier (L2)
        if use_linear_classifier:
            self.l2_classifier = LinearLogicClassifier(input_dim=num_rules, num_classes=num_classes)
        else:
            self.l2_classifier = DisjunctionLayer(input_dim=num_rules, output_dim=num_classes)

    def forward(self, concept_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            concept_probs: (B, num_concepts) concept activations in [0, 1]
        Returns:
            logic_logits: (B, num_classes) action predictions
            clauses: (B, num_rules) conjunction clause activations
            literals: (B, 2 * num_concepts) literal values
        """
        literals = self.binarizer(concept_probs)
        clauses = self.l1_conjunction(literals)
        logic_logits = self.l2_classifier(clauses)
        return logic_logits, clauses, literals

    def set_temperature(self, epoch: int, total_epochs: int):
        self.binarizer.set_temperature(epoch, total_epochs)

    def get_regularization_loss(self) -> torch.Tensor:
        """Returns L1 sparsity loss over logic weights to promote interpretable minimal rules."""
        l1_reg = self.l1_conjunction.get_sparsity_loss()
        if hasattr(self.l2_classifier, "get_l1_loss"):
            l2_reg = self.l2_classifier.get_l1_loss()
        else:
            l2_reg = self.l2_classifier.get_sparsity_loss()
        return 1e-4 * (l1_reg + l2_reg)

    def extract_rules(
        self,
        concept_names: List[str],
        class_names: Optional[List[str]] = None,
        weight_threshold: float = 0.2,
        max_rules_per_class: int = 3,
    ) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Extracts human-interpretable First-Order Logic rules:
        Class_k <= (Concept_A AND ~Concept_B) OR (Concept_C AND Concept_D)
        """
        if class_names is None or len(class_names) < self.num_classes:
            class_names = [f"Action_{i}" for i in range(self.num_classes)]

        literal_names = list(concept_names)
        if self.use_not:
            literal_names += [f"NOT {name}" for name in concept_names]

        W_l1 = torch.clamp(self.l1_conjunction.W, 0.0, 1.0).detach().cpu().numpy()  # (n_lit, n_rules)

        # Parse L1 conjunctions
        clause_formulas = []
        for r_idx in range(self.num_rules):
            col = W_l1[:, r_idx]
            selected_lits = [
                literal_names[i]
                for i in range(len(literal_names))
                if col[i] > weight_threshold
            ]
            if not selected_lits:
                formula = "TRUE"
            else:
                formula = " ∧ ".join(selected_lits)
            clause_formulas.append(formula)

        # Parse L2 Disjunctions / Linear weights
        rules_by_class = defaultdict(list)
        if self.use_linear_classifier:
            W_l2 = self.l2_classifier.fc.weight.detach().cpu().numpy()  # (n_classes, n_rules)
            for c_idx in range(self.num_classes):
                c_name = class_names[c_idx]
                weights = W_l2[c_idx]
                top_rule_indices = np.argsort(weights)[::-1][:max_rules_per_class]
                for r_idx in top_rule_indices:
                    w = float(weights[r_idx])
                    if abs(w) > 0.05 and clause_formulas[r_idx] != "TRUE":
                        rules_by_class[c_name].append({
                            "rule": clause_formulas[r_idx],
                            "weight": round(w, 4),
                        })
        else:
            W_l2 = torch.clamp(self.l2_classifier.W, 0.0, 1.0).detach().cpu().numpy()  # (n_rules, n_classes)
            for c_idx in range(self.num_classes):
                c_name = class_names[c_idx]
                weights = W_l2[:, c_idx]
                top_rule_indices = np.argsort(weights)[::-1][:max_rules_per_class]
                for r_idx in top_rule_indices:
                    w = float(weights[r_idx])
                    if w > weight_threshold and clause_formulas[r_idx] != "TRUE":
                        rules_by_class[c_name].append({
                            "rule": clause_formulas[r_idx],
                            "weight": round(w, 4),
                        })

        return dict(rules_by_class)
