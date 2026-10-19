"""Concept-graph head (plan §4, §6): concept decoder (stage 1) -> fuzzy executor of the logic graph (stage 3).

P1 / P2 version: no action branch and no R-GCN (lambda = 0), so the class is decided only by executing the
graph on the concepts. Same interface as model/concept_logic.py: ConceptLogicHead (M, logits(c), tau,
forward -> dict(c, logits)), so train_concept_logic.py trains both.
"""

import torch
import torch.nn as nn

from model.attribute_decoder import AttributeDecoder
from model.concept_logic import TAU_MAX, TAU_MIN
from model.fuzzy_executor import FuzzyGraphExecutor


class ConceptGraphHead(nn.Module):
    def __init__(self, in_channels, graph, joint_mask, decoder_args=None, omega=1.0, nu=1.0, learn_alpha=False,
                 tau=0.1, learn_tau=True, eta=0.0):
        super().__init__()
        K = len(graph.concepts)
        if joint_mask.shape[0] != K:
            raise ValueError('one joint_mask row per concept')
        self.executor = FuzzyGraphExecutor(graph, omega=omega, nu=nu, learn_alpha=learn_alpha)
        self.register_buffer('M', self.executor.mask_row.clone())
        self.eta = float(eta)
        self.decoder = AttributeDecoder(in_channels, torch.eye(K), joint_mask, **(decoder_args or {}))
        self.log_tau = nn.Parameter(torch.tensor(float(tau)).log(), requires_grad=learn_tau)

    @property
    def tau(self):
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def logits(self, c):
        return self.executor(c) / self.tau

    def regularizer(self):
        return self.eta * self.executor.alpha_penalty()

    def forward(self, features, valid):
        """features [N, M, C, S, V], valid [N, M] -> dict(c [N, K], logits [N, classes])."""
        c = self.decoder(features, valid)
        return dict(c=c, logits=self.logits(c))
