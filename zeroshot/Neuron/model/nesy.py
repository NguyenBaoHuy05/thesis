"""Neuro-symbolic zero-shot model built on Neuron.

skeleton -> Shift-GCN (frozen) -> cube (n, 256, T=16, V=25)
  neural branch   : Neuron head (spatial + temporal micro-prototypes), unchanged      -> s_neu (n, C)
  symbolic branch :
    grounding     : 71 named primitive slots. Slot k queries the joints with a query made from the CLIP
                    text of primitive k, restricted to its joints (primitive.json), separately for the
                    left- and right-side masks -> truth value p_k^s(t) in [0, 1], s in {left, right}
    reasoning     : rule of class y from kg_rules.json, in log space (product t-norm, mean over literals)
                      lit_s(y) = mean( log P_k^s for k in require(y) ; log(1 - max_s P_k^s) for k in forbid(y) )
                      P_k^s    = soft max over time of p_k^s(t)
                      O_s(y)   = path(k1 -> .. -> km) - path(km -> .. -> k1)   (sequence rules, DP over time)
                      s_sym(y) = soft max over sides of ( lit_s(y) + lam_seq * O_s(y) )        (DNF over sides)
  fusion          : z = tau_neu * s_neu + beta * s_sym      (s_neu detached: Neuron is trained by its own loss)
  inference       : classes whose rule contradicts the number of persons in the sample are masked out.

Loss = Neuron loss (CE on its 6 logits)
     + lambda_sym   * CE(tau_sym * s_sym)
     + lambda_fuse  * CE(z)
     + lambda_logic * (exclusive groups + implications on p_k^s(t))
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import kg as kg_lib
from .shiftgcn_match import ModelMatch

EPS = 1e-6


def path_score(sim, ks):
    """sim: (B, T, K). Best strictly increasing time path through primitives ks; mean of sim, (B,)."""
    dp = sim[:, :, ks[0]]
    for k in ks[1:]:
        prev = torch.cummax(dp, dim=1).values                                       # best path ending <= t
        prev = torch.cat([prev.new_full((prev.shape[0], 1), -1e4), prev[:, :-1]], 1)  # strictly earlier
        dp = prev + sim[:, :, k]
    return dp.max(1).values / len(ks)


class PrimitiveGrounding(nn.Module):
    """cube (n, C, T, V) -> primitive logits (n, 2, K, T). Neuron-style joint attention, one query per named
    primitive. Queries and detector directions are functions of the primitive's CLIP text, so primitives that
    no seen-class rule uses still get a meaningful detector."""

    def __init__(self, prim_text, side_masks, d_in=256, d=128):
        super().__init__()
        self.register_buffer('E0', F.normalize(torch.as_tensor(prim_text, dtype=torch.float32), dim=-1))
        m = torch.as_tensor(side_masks, dtype=torch.float32)                          # (2, K, V)
        self.register_buffer('mask_bias', (m - 1) * 1e4)                               # 0 on its joints, -1e4 else
        D = self.E0.shape[1]
        self.ln = nn.LayerNorm(d_in)
        self.key = nn.Linear(d_in, d)
        self.val = nn.Linear(d_in, d)
        self.query = nn.Linear(D, d)                       # text -> which joints to look at
        self.direction = nn.Linear(D, d)                   # text -> what the pooled joints should look like
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.d = d

    def forward(self, cube):
        f = self.ln(cube.permute(0, 2, 3, 1))                                          # (n, T, V, C)
        k, v = self.key(f), self.val(f)                                                # (n, T, V, d)
        q = self.query(self.E0)                                                        # (K, d)
        att = torch.einsum('ntvd,kd->ntkv', k, q) / math.sqrt(self.d)                  # (n, T, K, V)
        att = att[:, None] + self.mask_bias[None, :, None]                            # (n, 2, T, K, V)
        h = torch.einsum('nstkv,ntvd->nstkd', att.softmax(-1), v)                      # (n, 2, T, K, d)
        h = F.normalize(self.mlp(h), dim=-1)
        r = F.normalize(self.direction(self.E0), dim=-1)                               # (K, d)
        logit = self.alpha * (h * r).sum(-1) + self.bias                               # (n, 2, T, K)
        return logit.permute(0, 1, 3, 2)                                               # (n, 2, K, T)


class RuleReasoner(nn.Module):
    """Primitive logits (n, 2, K, T) -> class scores s_sym (n, C) for the global class ids `classes`."""

    def __init__(self, R, seqs, groups, implications, kappa_t=1.0, kappa_side=10.0, p_drop=0.0):
        super().__init__()
        self.p_drop = p_drop        # rule dropout: each literal (and each sequence rule) dropped while training
        R = torch.as_tensor(R, dtype=torch.float32)
        self.register_buffer('Rp', (R > 0).float())
        self.register_buffer('Rn', (R < 0).float())
        self.register_buffer('n_lit', (R != 0).float().sum(1))
        self.register_buffer('G', torch.as_tensor(groups, dtype=torch.float32))       # (n_groups, K)
        imp = torch.as_tensor(implications, dtype=torch.long).reshape(-1, 2)
        self.register_buffer('imp_a', imp[:, 0].clone())
        self.register_buffer('imp_b', imp[:, 1].clone())
        self.seqs = {int(c): list(ks) for c, ks in seqs.items()}
        self.kappa_t, self.kappa_side = kappa_t, kappa_side
        self.lam_seq = nn.Parameter(torch.tensor(1.0))

    def presence(self, logit):
        """Soft max over time: P_k^s = sum_t softmax_t(kappa * logit) * sigmoid(logit). (n, 2, K)"""
        w = torch.softmax(self.kappa_t * logit, dim=-1)
        return (w * torch.sigmoid(logit)).sum(-1)

    def forward(self, logit, classes):
        n = logit.shape[0]
        P = self.presence(logit)                                                       # (n, 2, K)
        logP = P.clamp(min=EPS).log()
        log_not_any = (1 - P.max(1).values).clamp(min=EPS).log()                       # (n, K) forbid: neither side
        Rp, Rn = self.Rp[classes], self.Rn[classes]                                    # (C, K)
        drop = self.training and self.p_drop > 0
        if drop:                    # rule dropout; a class always keeps at least one require literal
            Rp_d = Rp * (torch.rand_like(Rp) > self.p_drop)
            Rp = torch.where((Rp_d.sum(1) == 0)[:, None], Rp, Rp_d)
            Rn = Rn * (torch.rand_like(Rn) > self.p_drop)
        n_lit = (Rp + Rn).sum(1)
        lit = (logP @ Rp.T + (log_not_any @ Rn.T)[:, None]) / n_lit                   # (n, 2, C)

        O = lit.new_zeros(lit.shape)
        col = {int(c): j for j, c in enumerate(classes.tolist())}
        hit = [(col[c], ks) for c, ks in self.seqs.items() if c in col]
        if drop:
            hit = [h for h in hit if float(torch.rand(())) > self.p_drop]
        if hit:
            sim = F.logsigmoid(logit).permute(0, 1, 3, 2).reshape(n * 2, logit.shape[3], logit.shape[2])
            for j, ks in hit:
                O[:, :, j] = (path_score(sim, ks) - path_score(sim, ks[::-1])).view(n, 2)
        s = lit + self.lam_seq * O
        return torch.logsumexp(self.kappa_side * s, dim=1) / self.kappa_side         # (n, C)

    def logic_loss(self, logit):
        """Exclusive groups: at most one primitive of a group per side and time (sum of pairwise products).
        Implications a -> b: penalise p_a * (1 - p_b)."""
        p = torch.sigmoid(logit).permute(0, 1, 3, 2)                                  # (n, 2, T, K)
        L = p.new_zeros(())
        if self.G.shape[0]:
            s1, s2 = p @ self.G.T, (p * p) @ self.G.T                                  # (n, 2, T, n_groups)
            L = L + ((s1 * s1 - s2) / 2).mean()
        if self.imp_a.numel():
            L = L + (p[..., self.imp_a] * (1 - p[..., self.imp_b])).mean()
        return L


class NeuronNeSy(ModelMatch):
    def __init__(self, num_class=55, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 n_cls_total=60, primitive_path='../primitive.json', rules_path='../kg_rules.json',
                 prim_text_path='./semantics/ntu/primitive_text_clip.pt', d_sym=128, kappa_t=1.0,
                 kappa_side=10.0, beta_init=0.1, two_person_thr=0.5, mask_penalty=1e3, freeze_bn=True,
                 p_drop=0.0, neuron_scale=0.1, neuron_scale_learn=False):
        super().__init__(num_class, num_point, num_person, graph, graph_args, in_channels)   # encoder frozen here
        kg = kg_lib.load_kg(primitive_path, rules_path, n_cls_total)
        prim_text = torch.load(prim_text_path)
        assert prim_text.shape[0] == len(kg['ids']), 'primitive text cache does not match primitive.json'
        self.kg_ids, self.kg_names = kg['ids'], kg['names']
        self.ground = PrimitiveGrounding(prim_text, kg['side_masks'], d_in=256, d=d_sym)
        self.reason = RuleReasoner(kg['R'], kg['seqs'], kg['groups'], kg['implications'], kappa_t, kappa_side,
                                   p_drop=p_drop)
        # Scale of the cosines in Neuron's own loss. Neuron uses 0.1, which keeps its softmax near uniform
        # (its CE stays at ~ln(#classes)); neuron_scale_learn=True learns it (CLIP-style temperature).
        ns = torch.tensor(math.log(neuron_scale))
        if neuron_scale_learn:
            self.log_neuron_scale = nn.Parameter(ns)
        else:
            self.register_buffer('log_neuron_scale', ns)
        self.register_buffer('person_rule', torch.as_tensor(kg['R'][:, kg['two_person']]))   # +1 / -1 / 0
        self.log_tau_neu = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.log_tau_sym = nn.Parameter(torch.tensor(0.0))      # s_sym is a mean log-probability already
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.two_person_thr, self.mask_penalty, self.freeze_bn = two_person_thr, mask_penalty, freeze_bn

    # ---------------------------------------------------------------- parameters
    def nesy_modules(self):
        return [self.ground, self.reason]

    def nesy_parameters(self):
        ps = [p for m in self.nesy_modules() for p in m.parameters()]
        ps += [self.log_tau_neu, self.log_tau_sym, self.beta]
        if isinstance(self.log_neuron_scale, nn.Parameter):     # a scalar: AdamW, not Neuron's SGD at lr 0.1
            ps.append(self.log_neuron_scale)
        return ps

    def neuron_parameters(self):
        nesy = {id(p) for p in self.nesy_parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in nesy]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:                       # frozen encoder: keep its BatchNorm statistics fixed too
            self.feature_extractor.eval()
        return self

    # ---------------------------------------------------------------- forward
    def forward(self, x, spatial_round, temporal_round, classes):
        """x: (n, 3, T, V, M); *_round: Neuron class semantics for `classes` (global ids, LongTensor)."""
        _, _, cube = self.feature_extractor(x)                                        # (n, 256, 16, 25)
        return self.forward_cube(cube, spatial_round, temporal_round, classes)

    def forward_cube(self, cube, spatial_round, temporal_round, classes):
        """Same as forward, from a (cached) encoder cube (n, 256, 16, 25)."""
        spa, tem = self.neural_logits(cube, spatial_round, temporal_round)
        p_logit = self.ground(cube)                                                    # (n, 2, K, T)
        s_sym = self.reason(p_logit, classes)
        s_neu = (spa[2] + tem[2]) / 2 / 0.1        # Neuron's fine-grained logits are cosine * 0.1
        tau_neu = self.log_tau_neu.exp().clamp(max=100)
        z = tau_neu * s_neu.detach() + self.beta * s_sym
        return dict(spa=spa, tem=tem, s_neu=s_neu, s_sym=s_sym, z=z, p_logit=p_logit)

    def losses(self, out, y, lambda_sym=1.0, lambda_fuse=1.0, lambda_logic=0.1):
        """y: labels re-indexed to the columns of `classes`."""
        ce = F.cross_entropy
        f = self.log_neuron_scale.exp().clamp(max=100) / 0.1     # Neuron's logits are cosine * 0.1
        neuron = (sum(ce(o * f, y) for o in out['spa']) / 3 + sum(ce(o * f, y) for o in out['tem']) / 3) / 2
        tau_sym = self.log_tau_sym.exp().clamp(max=100)
        return dict(neuron=neuron,
                    sym=lambda_sym * ce(tau_sym * out['s_sym'], y),
                    fuse=lambda_fuse * ce(out['z'], y),
                    logic=lambda_logic * self.reason.logic_loss(out['p_logit']))

    @torch.no_grad()
    def second_person(self, x):
        """(n,) bool from the skeleton x (n, 3, T, V, M): person 2 has non-zero joints in > thr of the frames."""
        return (x[..., 1].abs().sum((1, 3)) > 0).float().mean(1) > self.two_person_thr

    @torch.no_grad()
    def person_conflict(self, present, classes):
        """(n, C) bool: the class rule requires two persons but the sample has one, or forbids two persons but
        the sample has two. present: (n,) bool from second_person."""
        rule = self.person_rule[classes]                                                     # (C,)
        return ((rule > 0)[None] & ~present[:, None]) | ((rule < 0)[None] & present[:, None])

    @torch.no_grad()
    def final_scores(self, out, present, classes):
        return out['z'] - self.mask_penalty * self.person_conflict(present, classes).float()
