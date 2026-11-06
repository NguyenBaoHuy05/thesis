"""KG-conditioned attribute decoder on the backbone map before pooling (A1, option 2).

The cache head reads a = sigmoid(W_a z) from z pooled over (T, V): the attribute evidence then
comes from the same vector as the class decision, and the pooling erases temporal order (the
bit that separates put_on / take_off). Here each attribute queries the tokens (actor, frame,
joint) of the l9 map directly:

    tokens = LN(Linear(features[n, m, :, t, v])) + E_time[t] + E_joint[v] + E_person[m]
    query  = Linear(text embedding of the attribute)
    q      = q + CrossAttn(LN(q), tokens), mask: joints outside the attribute's KG `body_parts`
             and actors that are not real people          (x layers, + feed-forward)
    a_attr = sigmoid(w . LN(q_attr) + b)

Attributes do not attend to each other: each one finds its own evidence. The output layer starts
at zero, so a = 0.5 and VERIFY contributes exactly 0 before training: a head trained on the cache
keeps its retrieval and FUSE behavior at epoch 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(2 * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, tokens, blocked):
        """q [N, A, d], tokens [N, S, d], blocked [N * heads, A, S] (True = not attended)."""
        h, _ = self.attn(self.norm_q(q), tokens, tokens, attn_mask=blocked, need_weights=False)
        q = q + self.dropout(h)
        return q + self.dropout(self.ff(self.norm_ff(q)))


class AttributeDecoder(nn.Module):
    def __init__(self, in_channels, attribute_embeddings, joint_mask, d_model=256, heads=4,
                 layers=1, dropout=0.1, max_frames=64, num_person=2):
        super().__init__()
        joint_mask = joint_mask.bool()
        if not joint_mask.any(1).all():
            raise ValueError('every attribute needs at least one joint in joint_mask')
        self.heads = heads
        self.register_buffer('joint_mask', joint_mask)                                  # [A, V]
        self.register_buffer('attribute_embeddings', F.normalize(attribute_embeddings.float(), dim=-1))
        self.token = nn.Sequential(nn.Linear(in_channels, d_model), nn.LayerNorm(d_model))
        self.time_emb = nn.Parameter(torch.randn(max_frames, d_model) * 0.02)
        self.joint_emb = nn.Parameter(torch.randn(joint_mask.shape[1], d_model) * 0.02)
        self.person_emb = nn.Parameter(torch.randn(num_person, d_model) * 0.02)
        self.query = nn.Linear(attribute_embeddings.shape[1], d_model)
        self.layers = nn.ModuleList(DecoderLayer(d_model, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def replace_attributes(self, attribute_embeddings, joint_mask):
        if len(attribute_embeddings) != len(joint_mask):
            raise ValueError('one joint_mask row per attribute')
        device = self.joint_mask.device
        self.attribute_embeddings = F.normalize(attribute_embeddings.float(), dim=-1).to(device)
        self.joint_mask = joint_mask.bool().to(device)

    def forward(self, features, valid, return_tokens=False):
        """features [N, M, C, T, V] (backbone output before pooling), valid [N, M] real actors.
        Returns a [N, A] in (0, 1); with return_tokens also the query states after the last layer and the
        final LayerNorm, [N, A, d] (the concept / action tokens of model/concept_action_graph.py)."""
        N, M, C, T, V = features.shape
        if T > len(self.time_emb):
            raise ValueError(f'{T} frames, decoder built for max_frames={len(self.time_emb)}')
        if V != self.joint_mask.shape[1] or M > len(self.person_emb):
            raise ValueError(f'features [M={M}, V={V}] do not match the decoder')
        x = self.token(features.float().permute(0, 1, 3, 4, 2))                         # [N, M, T, V, d]
        x = x + self.time_emb[:T, None] + self.joint_emb + self.person_emb[:M, None, None]
        tokens = x.flatten(1, 3)                                                        # [N, S], S = M T V

        # allowed[n, attr, (m, t, v)] = joint v belongs to attr and actor m is real.
        allowed = valid[:, None, :, None, None] & self.joint_mask[None, :, None, None, :]
        allowed = allowed.expand(N, -1, M, T, V).flatten(2)                             # [N, A, S]
        blocked = (~allowed).repeat_interleave(self.heads, dim=0)

        q = self.query(self.attribute_embeddings).unsqueeze(0).expand(N, -1, -1)
        for layer in self.layers:
            q = layer(q, tokens, blocked)
        q = self.norm(q)
        a = torch.sigmoid(self.out(q).squeeze(-1))
        return (a, q) if return_tokens else a
