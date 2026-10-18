"""Graph encoder and graph retrieval of the A1-G head (A1-G plan S2).

    x_i      learned initial vector of node i (120 actions + A attributes), init='learned';
             init='text' starts x from the text embeddings (fixed random projection) and still learns
    h^{l+1}  = act( W_0 h^l + sum_r A_r h^l W_r^l )        A_r: model/kg.py relation_adjacency
    g        = x + h^L                                     (residual; layers=0 gives g = x, ablation G1-0)

    s_c      = cos(P z, g_c) / tau                         GraphRetrievalScorer

The graph does not depend on the sample: g is recomputed once per forward (181 nodes, cheap) so
gradients reach x and the relation weights. The residual keeps a free per-class component: KG rows
that are nearly identical (put_on / take_off) would otherwise give nearly identical g_c and
retrieval could not tell them apart (A1-G plan S2.2).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kg import RELATIONS, relation_adjacency

TAU_MIN, TAU_MAX = 0.01, 1.0


class RGCNLayer(nn.Module):
    def __init__(self, d, num_relations, dropout=0.0):
        super().__init__()
        self.self_loop = nn.Linear(d, d)
        self.relation = nn.Parameter(torch.randn(num_relations, d, d) / math.sqrt(d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency):
        """h [Nn, d], adjacency [R, Nn, Nn] -> [Nn, d]."""
        messages = torch.einsum('rij,jd,rde->ie', adjacency, h, self.relation)
        return self.self_loop(self.dropout(h)) + messages


class RGCNEncoder(nn.Module):
    def __init__(self, kg, d=256, layers=2, dropout=0.1, relations=RELATIONS, init='learned',
                 text_embeddings=None):
        """kg: model/kg.py KGTensors. text_embeddings: (action [C, D], attribute [A, D]) for init='text'."""
        super().__init__()
        C, A = kg.W.shape
        self.num_actions, self.num_attributes = C, A
        self.relations = tuple(relations)
        self.register_buffer('adjacency', relation_adjacency(kg, self.relations))
        if init == 'learned':
            x = torch.randn(C + A, d) / math.sqrt(d)
        elif init == 'text':
            if text_embeddings is None:
                raise ValueError("init='text' needs text_embeddings (action, attribute)")
            text = torch.cat([F.normalize(t.float(), dim=-1) for t in text_embeddings])
            if len(text) != C + A:
                raise ValueError(f'{len(text)} text embeddings for {C + A} nodes')
            projection = torch.linalg.qr(torch.randn(text.shape[1], d))[0] if text.shape[1] >= d \
                else torch.randn(text.shape[1], d) / math.sqrt(d)
            x = text @ projection
        else:
            raise ValueError(f"init must be 'learned' or 'text', not {init!r}")
        self.x = nn.Parameter(x)
        self.layers = nn.ModuleList(RGCNLayer(d, len(self.relations), dropout) for _ in range(layers))

    def forward(self):
        """Returns g [C + A, d]: actions first, then attributes."""
        h = self.x
        for i, layer in enumerate(self.layers):
            h = layer(h, self.adjacency)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return self.x + h if len(self.layers) else self.x

    def split(self, g):
        return g[:self.num_actions], g[self.num_actions:]


class GraphRetrievalScorer(nn.Module):
    """s_c = cos(P z, g_c) / tau with g from RGCNEncoder. Same interface as RetrievalScorer
    (cosine, forward, tau); `class_embeddings` / `attribute_embeddings` are the current g."""

    def __init__(self, in_channels, kg, tau=0.07, learn_tau=True, feature_dropout=0.0, d=256,
                 layers=2, dropout=0.1, relations=RELATIONS, init='learned', text_embeddings=None):
        super().__init__()
        self.encoder = RGCNEncoder(kg, d, layers, dropout, relations, init, text_embeddings)
        self.proj = nn.Linear(in_channels, d)
        self.feature_dropout = nn.Dropout(feature_dropout)
        log_tau = torch.tensor(math.log(tau))
        if learn_tau:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer('log_tau', log_tau)

    @property
    def tau(self):
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def nodes(self):
        """(action [C, d], attribute [A, d]) node vectors after the encoder."""
        return self.encoder.split(self.encoder())

    @property
    def class_embeddings(self):
        return self.nodes()[0]

    @property
    def attribute_embeddings(self):
        return self.nodes()[1]

    def cosine(self, z):
        q = F.normalize(self.proj(self.feature_dropout(z.float())), dim=-1)
        return q @ F.normalize(self.class_embeddings, dim=-1).t()

    def forward(self, z):
        return self.cosine(z) / self.tau
