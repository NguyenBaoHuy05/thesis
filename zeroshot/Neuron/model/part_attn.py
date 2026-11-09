"""PartAttn: a text-queried part / phase attention head on the frozen Shift-GCN encoder. Replaces Neuron's head
(prototype memory + recall / remember gates); the encoder, the CLIP description features and the losses are reused.

skeleton -> Shift-GCN (frozen) -> cube (n, 256, T=16, V=25)
visual tokens (16 per sample): 5 body parts x 3 temporal phases + 1 global token
    part pooling: trunk / left arm / right arm / left leg / right leg (NTU joint groups); phase: T pooled to 3
    each token: LayerNorm -> + learned token embedding -> (key: Linear -> d_att) and (value: MLP -> 768, unit vector)
    one token set per branch (spatial / temporal), so the two branches can specialise
text  : Neuron's CLIP features of the LLM descriptions, 3 levels per branch, (C, 10, 768) each (unchanged)
score : for class c, description k, level l:
            a_t   = softmax_t( key_t . Wq_l(text_ck) / sqrt(d_att) )          text is the QUERY
            s_ckl = sum_t a_t * cos(value_t, text_ck)                          attention-weighted token cosine
        description pooling (mean | lse) -> (n, C).  6 outputs: spa[0..2] (levels cg, mg, fg), tem[0..2].
        Prediction = (spa[2] + tem[2]) / 2, as in Neuron++ (one prediction per sample).

Why this head: nothing in it is tied to the seen classes except the shared projections. The text only chooses WHERE to
look (which part, which phase), so an unseen class description picks tokens the same way a seen one does, and the
cosine is taken per token against that text (dense alignment) instead of through one pooled vector.

Same interface as NeuronPlus (forward_cube, s_neu, losses, second_person, log_ce_scale, ...), so main_plus.py trains it
unchanged: --model model.part_attn.PartAttn. All parameters are new, so they all go to AdamW (neuron_parameters() = []).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neuron_plus import NeuronPlus, class_graph, _as3
from .shiftgcn_match import Model

# NTU RGB+D 25-joint skeleton, 0-based
PARTS = [[0, 1, 2, 3, 20],          # trunk + head
         [4, 5, 6, 7, 21, 22],      # left arm + hand
         [8, 9, 10, 11, 23, 24],    # right arm + hand
         [12, 13, 14, 15],          # left leg
         [16, 17, 18, 19]]          # right leg
N_PHASE = 3
N_TOK = len(PARTS) * N_PHASE + 1


class TokenBranch(nn.Module):
    """cube (n, 256, T, V) -> keys (n, N_TOK, d_att) and unit values (n, N_TOK, 768)."""

    def __init__(self, c_in, d_att, d_out, hidden, drop):
        super().__init__()
        self.norm = nn.LayerNorm(c_in)
        self.pos = nn.Parameter(torch.zeros(N_TOK, c_in))
        nn.init.normal_(self.pos, std=0.02)
        self.key = nn.Linear(c_in, d_att)
        self.val = nn.Sequential(nn.Linear(c_in, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, d_out))
        for p in PARTS:
            assert len(set(p)) == len(p)

    def tokens(self, cube):
        n, c, t, v = cube.shape
        parts = torch.stack([cube[..., p].mean(-1) for p in PARTS], dim=-1)          # (n, c, T, 5)
        parts = F.adaptive_avg_pool2d(parts, (N_PHASE, len(PARTS)))                   # (n, c, 3, 5)
        tok = parts.flatten(2).transpose(1, 2)                                        # (n, 15, c)
        glob = cube.mean((2, 3))[:, None]                                             # (n, 1, c)
        return torch.cat([tok, glob], dim=1)

    def forward(self, cube):
        x = self.norm(self.tokens(cube)) + self.pos
        return self.key(x), F.normalize(self.val(x), dim=-1)


class PartAttn(nn.Module):
    def __init__(self, num_class=55, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 n_cls_total=60, freeze_bn=True, d_att=256, hidden=512, drop=0.1,
                 agg='mean', agg_temp=0.1, ce_scale=10.0, ce_scale_learn=True,
                 graph_k=5, graph_temp=0.1, neg_mode='graph', cal_tau=0.1, tri_margin=0.2):
        super().__init__()
        self.feature_extractor = Model(num_class, num_point, num_person, graph, graph_args, in_channels)
        for p in self.parameters():
            p.requires_grad = False                                  # the encoder is frozen; the head is added below
        self.freeze_bn = freeze_bn
        assert agg in ('mean', 'lse'), agg
        self.agg = agg
        self.log_agg_temp = nn.Parameter(torch.tensor(math.log(agg_temp)), requires_grad=agg == 'lse')
        self.log_ce_scale = nn.Parameter(torch.tensor(math.log(ce_scale)), requires_grad=ce_scale_learn)
        self.branch = nn.ModuleDict({b: TokenBranch(256, d_att, 768, hidden, drop) for b in ('s', 't')})
        self.query = nn.ModuleDict({f'{b}{r}': nn.Linear(768, d_att) for b in ('s', 't') for r in range(3)})
        self.d_att = d_att
        self.prop_alpha = 0.0                                        # no propagation in this head (main_plus logs it)
        self.graph_k, self.graph_temp = graph_k, graph_temp
        self.neg_mode, self.cal_tau, self.tri_margin = neg_mode, cal_tau, tri_margin

    # parameters: everything trainable is new -> AdamW
    def plus_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('feature_extractor.')]

    def neuron_parameters(self):
        return []

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            self.feature_extractor.eval()
        return self

    pool = NeuronPlus.pool
    losses = NeuronPlus.losses
    s_neu = staticmethod(NeuronPlus.s_neu)
    second_person = NeuronPlus.second_person

    def score(self, branch, cube, sem):
        """sem: list of 3 (C, K, 768) -> 3 score matrices (n, C)."""
        key, val = self.branch[branch](cube)                                          # (n, T, a), (n, T, 768)
        out = []
        for r, t in enumerate(sem):
            e = F.normalize(_as3(t), dim=-1)                                          # (C, K, 768)
            q = self.query[f'{branch}{r}'](e)                                         # (C, K, a)
            attn = torch.einsum('nta,cka->ntck', key, q).div(math.sqrt(self.d_att)).softmax(1)
            sim = torch.einsum('ntd,ckd->ntck', val, e)
            out.append(self.pool((attn * sim).sum(1)))                                # (n, C, K) -> (n, C)
        return out

    def forward(self, x, spatial_round, temporal_round, classes=None):
        _, _, cube = self.feature_extractor(x)
        return self.forward_cube(cube, spatial_round, temporal_round)

    def forward_cube(self, cube, spatial_round, temporal_round, classes=None, extra=()):
        """extra is accepted and ignored (main_plus then logs s_neu in the zsl_prop_* columns)."""
        A, nbr = class_graph(list(spatial_round), list(temporal_round), self.graph_k, self.graph_temp)
        return dict(spa=self.score('s', cube, spatial_round), tem=self.score('t', cube, temporal_round), nbr=nbr)
