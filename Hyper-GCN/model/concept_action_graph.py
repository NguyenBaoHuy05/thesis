"""Full concept–action logic graph head (KG_RETRIEVAL_PLAN/CONCEPT_GRAPH/CONCEPT_GRAPH_PLAN.md, drawio
drawio/architecture_concept_graph.drawio). Reads the frozen l9 map (cache or fine-tuning).

Stage 1, two decoders on the tokens of the l9 map:
    concept decoder (102 queries, body-group mask, 2 layers) -> c [N, K] in (0, 1), tokens U [N, K, d]
    action decoder  (120 queries, every joint, 1 layer)      -> tokens A [N, 120, d], aux logits w.a_k
Stage 2, the graph (model/concept_graph.py) and its parameters: one learnable embedding per node (concepts and
    actions start from the decoder queries Q_C, Q_A; gates and NOT nodes at random) and the R-GCN relation
    weights. No separate training.
Stage 3, per-sample execution:
    node init  concept [e_j ; c_j u_j ; c_j], action [e_k ; drop(a_k)] (whole-token dropout), gate / NOT e_v
    neural     stratified R-GCN (model/rgcn.py) -> z_k = w.h_k
    symbolic   fuzzy executor (model/fuzzy_executor.py) on c -> log s
    logits     = log s / tau + lam z
Loss (extra_loss, on top of CE(logits) + beta BCE(c, M[y]) of model/concept_logic.py: concept_loss):
    aux_weight CE(aux) + gamma KL(softmax z || softmax(log s / tau)) + mu margin + eta sum |alpha - 1|
    margin = mean over the NOT(j|y) partners j of y of max(0, m - log s_y + log s_j)

logits(c) (used by K5 / K5-pair) is the symbolic path only: an intervention on c is read through the logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attribute_decoder import AttributeDecoder
from model.concept_logic import TAU_MAX, TAU_MIN
from model.fuzzy_executor import FuzzyGraphExecutor
from model.rgcn import StratifiedRGCN


class ConceptActionGraphHead(nn.Module):
    def __init__(self, in_channels, graph, joint_mask, decoder_args=None, action_decoder_args=None,
                 omega=1.0, nu=1.0, learn_alpha=True, eta=1e-4, tau=0.1, learn_tau=True,
                 lam=0.5, gamma=0.5, mu=0.5, margin=0.1, aux_weight=0.1, token_drop=0.3,
                 rgcn_args=None):
        super().__init__()
        K, A = len(graph.concepts), len(graph.actions)
        if joint_mask.shape[0] != K:
            raise ValueError('one joint_mask row per concept')
        self.lam, self.gamma, self.mu, self.margin = float(lam), float(gamma), float(mu), float(margin)
        self.aux_weight, self.token_drop, self.eta = float(aux_weight), float(token_drop), float(eta)

        # stage 1
        decoder_args = dict(decoder_args or {})
        self.decoder = AttributeDecoder(in_channels, torch.eye(K), joint_mask, **decoder_args)
        action_args = dict(decoder_args, **(action_decoder_args or {}))
        self.action_decoder = AttributeDecoder(in_channels, torch.eye(A), torch.ones(A, joint_mask.shape[1],
                                               dtype=torch.bool), **action_args)
        d = self.decoder.norm.normalized_shape[0]
        if self.action_decoder.norm.normalized_shape[0] != d:
            raise ValueError('both decoders need the same d_model')
        self.aux = nn.Linear(d, 1)

        # stage 2 (graph) + stage 3 symbolic path
        self.executor = FuzzyGraphExecutor(graph, omega=omega, nu=nu, learn_alpha=learn_alpha)
        self.register_buffer('M', self.executor.mask_row.clone())
        partner = torch.zeros(A, A, dtype=torch.bool)
        for i, j, _ in graph.nots:
            partner[i, j] = True
        self.register_buffer('partner', partner)
        self.log_tau = nn.Parameter(torch.tensor(float(tau)).log(), requires_grad=learn_tau)

        # stage 2 node parameters + stage 3 neural path
        with torch.no_grad():
            e_c = self.decoder.query(self.decoder.attribute_embeddings).clone()
            e_a = self.action_decoder.query(self.action_decoder.attribute_embeddings).clone()
        n_other = len(graph.motifs) + len(graph.ors) + len(graph.nots)
        self.emb_concept = nn.Parameter(e_c)
        self.emb_action = nn.Parameter(e_a)
        self.emb_other = nn.Parameter(torch.randn(n_other, d) * 0.02)
        self.sizes = (len(graph.motifs), len(graph.ors), len(graph.nots))
        self.init_concept = nn.Linear(2 * d + 1, d)
        self.init_action = nn.Linear(2 * d, d)
        self.rgcn = StratifiedRGCN(graph, d=d, **(rgcn_args or {}))
        self.readout = nn.Linear(d, 1)
        nn.init.zeros_(self.readout.weight)                # z = 0 at the start: the logic decides first
        nn.init.zeros_(self.readout.bias)

    @property
    def tau(self):
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def logits(self, c):
        """Symbolic path only (lam = 0), used by the K5 interventions on c."""
        return self.executor(c) / self.tau

    def neural(self, c, U, A_tok):
        N = c.shape[0]
        if self.training and self.token_drop > 0:
            keep = (torch.rand(N, A_tok.shape[1], 1, device=A_tok.device) > self.token_drop).float()
            A_tok = A_tok * keep
        h_c = self.init_concept(torch.cat([self.emb_concept.expand(N, -1, -1), c[..., None] * U, c[..., None]], -1))
        h_a = self.init_action(torch.cat([self.emb_action.expand(N, -1, -1), A_tok], -1))
        g, o, n = self.sizes
        other = self.emb_other.expand(N, -1, -1)
        h = self.rgcn(dict(concept=h_c, action=h_a, motif=other[:, :g], orr=other[:, g:g + o],
                           notn=other[:, g + o:g + o + n]))
        return self.readout(h['action']).squeeze(-1), h['action']

    def forward(self, features, valid):
        """features [N, M, C, S, V], valid [N, M] -> dict(c, logits, logits_sym, z, h_action [N, A, d], aux)."""
        c, U = self.decoder(features, valid, return_tokens=True)
        _, A_tok = self.action_decoder(features, valid, return_tokens=True)
        log_s = self.executor(c)
        sym = log_s / self.tau
        z, h_action = self.neural(c, U, A_tok)
        return dict(c=c, log_s=log_s, logits_sym=sym, aux=self.aux(A_tok).squeeze(-1), z=z, h_action=h_action,
                    logits=sym + self.lam * z)

    def extra_loss(self, out, labels):
        """aux CE + gamma KL(z || sym) + mu margin on the NOT(j|y) partners + eta alpha penalty."""
        terms = {}
        loss = self.aux_weight * F.cross_entropy(out['aux'], labels)
        terms['aux'] = loss.item()
        if self.gamma > 0:
            kl = F.kl_div(torch.log_softmax(out['logits_sym'], 1), torch.log_softmax(out['z'], 1),
                          log_target=True, reduction='batchmean')
            loss = loss + self.gamma * kl
            terms['kl'] = kl.item()
        if self.mu > 0:
            mask = self.partner[labels].float()                                   # [N, A]
            gap = out['log_s'].gather(1, labels[:, None]) - out['log_s']          # log s_y - log s_j
            hinge = F.relu(self.margin - gap) * mask
            m = hinge.sum() / mask.sum().clamp_min(1.0)
            loss = loss + self.mu * m
            terms['margin'] = m.item()
        if self.eta > 0:
            loss = loss + self.eta * self.executor.alpha_penalty()
        return loss, terms

    def forward_outputs(self):
        """Extra outputs collected at test time (train_concept_logic.py: predict)."""
        return ('logits_sym', 'z', 'aux')
