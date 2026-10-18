"""KG retrieval head on cached backbone features (A1 plan S1).

RETRIEVE (`RetrievalScorer`, alone = ablation K1, text only):

    q   = P z                       Linear C -> D
    s_c = cos(q, e_c) / tau         e_c fixed text embedding of class c, tau learned in [0.01, 1]

Full head (`KGRetrievalHead`) on the candidates Cand = top-k(s):

    a        = sigmoid(W_a z)                                    attribute head, [N, A]
               (attribute_source='decoder': model/attribute_decoder.py on the map before pooling)
    pi       = softmax_{Cand}(cos_c / tau_pi)                    tau_pi fixed (see below)
    d_attr   = std_pi(W[Cand, attr]) / max_attr std              in [0, 1]
    v_c      = sum_attr W[c, attr] d_attr (2 a_attr - 1)         VERIFY
    h        = CrossAttn(U_z z, nodes of G_q)                    FUSE, G_q = Cand + their attributes
    f_c      = <W_h h, U_a e_c> / sqrt(d)                                 + similar/opposite neighbors
    logit_c  = s_c + lambda v_c + f_c on Cand, -inf elsewhere,   lambda = softplus(.), init 1

Deviation from the plan: the plan weights candidates with pi = softmax(s), but s = cos / tau
and tau converges to ~0.03 in K1, so pi is nearly one-hot, every std_pi -> 0 and VERIFY switches
itself off exactly where retrieval is confident (also when it is confidently wrong). pi uses its
own fixed temperature on the cosine instead (A1_PROGRESS.md, P3 decisions).

Nothing is indexed by a per-class parameter: classes enter through e_c and the rows of W, so
`replace_classes` adds unseen classes without retraining (zero-shot, K6).

A1-G options (KG_RETRIEVAL_PLAN/A1G_GRAPH_RETRIEVAL_PLAN, S2), all off by default so A1 configs and
head.pt files keep working:

    retrieval_source='graph'   s_c = cos(P z, g_c) / tau, g from the R-GCN over the KG
                               (model/graph_encoder.py); no text embedding needed
    verify_mode='opposed'      d_attr = 1 iff some candidate has attr and another lacks it, else 0
                               (unknown cells no longer count as a middle value); 'masked_std'
                               = std over the candidates that know the attribute (>= 2 of them)
    normalize_verify=True      v_c divided by n_c = sum_attr d_attr known[c, attr] (>= 1), so a
                               class with many KG edges gets no larger evidence range
    attribute_source='predicate'  a from geometric predicates phi (model/predicates.py, cached by
                               tools/cache_predicates.py): p = sigmoid(((phi - mu)/sd - theta) / s),
                               two parameters per predicate; attributes without a predicate (or an
                               invalid one, e.g. two-person without a partner) get a = 0.5 and are
                               left out of d, n_c and the BCE
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attribute_decoder import AttributeDecoder
from model.graph_encoder import GraphRetrievalScorer

VERIFY_MODES = ('std', 'masked_std', 'opposed')

TAU_MIN, TAU_MAX = 0.01, 1.0
EPS = 1e-8


class RetrievalScorer(nn.Module):
    def __init__(self, in_channels, class_embeddings, tau=0.07, learn_tau=True, feature_dropout=0.0):
        super().__init__()
        # Fixed (offline) embeddings: a buffer, so swapping in unseen classes needs no retraining.
        self.register_buffer('class_embeddings', F.normalize(class_embeddings.float(), dim=-1))
        self.proj = nn.Linear(in_channels, class_embeddings.shape[1])
        self.feature_dropout = nn.Dropout(feature_dropout)
        log_tau = torch.tensor(math.log(tau))
        if learn_tau:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer('log_tau', log_tau)

    @property
    def tau(self):
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def cosine(self, z):
        """z: [N, C] -> cos(P z, e_c): [N, num_class]."""
        q = F.normalize(self.proj(self.feature_dropout(z.float())), dim=-1)
        return q @ self.class_embeddings.t()

    def forward(self, z):
        """z: [N, C] -> s: [N, num_class]."""
        return self.cosine(z) / self.tau


def topk_metrics(scores, labels, ks=(1, 5, 10, 20)):
    """Recall@k: fraction of samples whose label is in the top-k scores (Recall@1 = Top1).
    k larger than the number of classes is skipped."""
    ks = [k for k in ks if k <= scores.shape[1]]
    ranked = scores.topk(max(ks), dim=1).indices
    hit = ranked == labels.unsqueeze(1)
    return {f'recall@{k}': hit[:, :k].any(1).float().mean().item() for k in ks}


def discriminability(Wc, pi):
    """Wc: [N, k, A] KG rows of the candidates, pi: [N, k] candidate weights.
    d_attr = std_pi(Wc[:, :, attr]) / max_attr std, in [0, 1]; exactly 0 for an attribute on which
    all candidates agree (set explicitly: float rounding of the weighted mean leaves a tiny var).
    sqrt(var + eps) keeps the gradient finite at var = 0."""
    mean = (pi.unsqueeze(-1) * Wc).sum(1, keepdim=True)
    var = (pi.unsqueeze(-1) * (Wc - mean) ** 2).sum(1)
    std = ((var + EPS).sqrt() - math.sqrt(EPS)).clamp_min(0)
    std = std.masked_fill(Wc.amax(1) == Wc.amin(1), 0.0)
    return std / (std.amax(1, keepdim=True) + EPS)


def masked_discriminability(Wc, pi, known_c):
    """As `discriminability`, but only over the candidates whose KG cell is known (known_c [N, k, A]):
    an unknown cell is not a value between has and lacks. 0 when fewer than 2 candidates know it."""
    weight = pi.unsqueeze(-1) * known_c
    total = weight.sum(1, keepdim=True)
    weight = weight / total.clamp_min(EPS)
    mean = (weight * Wc).sum(1, keepdim=True)
    var = (weight * (Wc - mean) ** 2).sum(1)
    std = ((var + EPS).sqrt() - math.sqrt(EPS)).clamp_min(0)
    big, small = Wc.masked_fill(known_c == 0, -2).amax(1), Wc.masked_fill(known_c == 0, 2).amin(1)
    std = std.masked_fill((known_c.sum(1) < 2) | (big == small), 0.0)
    return std / (std.amax(1, keepdim=True) + EPS)


def opposed_attributes(Wc):
    """[N, A] float: 1 where some candidate has the attribute and another one lacks it."""
    return ((Wc > 0).any(1) & (Wc < 0).any(1)).float()


class PredicateEvidence(nn.Module):
    """a [N, A] from geometric predicates phi [N, P] (A1-G plan S3).

    p_j = sigmoid((u_j - theta_j) / s_j), u_j = (phi_j - mean_j) / std_j (train statistics), s_j > 0
    so the direction "higher phi = attribute present" is fixed by the predicate definition. Only two
    parameters per predicate: they cannot learn to recognize the class (the shortcut of A1).
    Columns without a predicate, and invalid predicate values, are 0.5 (no evidence)."""

    def __init__(self, columns, mean, std, num_attributes, learn_threshold=True):
        super().__init__()
        columns = torch.as_tensor(columns, dtype=torch.long)
        if len(set(columns.tolist())) != len(columns):
            raise ValueError('two predicates on the same attribute')
        self.register_buffer('columns', columns)
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))
        theta, log_scale = torch.zeros(len(columns)), torch.zeros(len(columns))
        if learn_threshold:
            self.theta, self.log_scale = nn.Parameter(theta), nn.Parameter(log_scale)
        else:
            self.register_buffer('theta', theta)
            self.register_buffer('log_scale', log_scale)
        mask = torch.zeros(num_attributes, dtype=torch.bool)
        mask[columns] = True
        self.register_buffer('attribute_mask', mask, persistent=False)

    def forward(self, phi, valid=None):
        """phi, valid [N, P] -> (a [N, A], evidence [N, A] bool: the attribute has a valid predicate)."""
        u = (phi.float() - self.mean) / self.std
        p = torch.sigmoid((u - self.theta) / self.log_scale.exp())
        valid = torch.ones_like(p, dtype=torch.bool) if valid is None else valid.bool()
        p = torch.where(valid, p, torch.full_like(p, 0.5))
        a = torch.full((len(phi), len(self.attribute_mask)), 0.5, device=phi.device, dtype=p.dtype)
        evidence = torch.zeros_like(a).index_copy(1, self.columns, valid.to(a.dtype)).bool()
        return a.index_copy(1, self.columns, p), evidence


def verify(Wc, d, a):
    """Per-attribute evidence Wc d (2a - 1): [N, k, A]; its sum over attributes is v_c."""
    return Wc * d.unsqueeze(1) * (2 * a - 1).unsqueeze(1)


def inverse_softplus(x):
    return math.log(math.expm1(x))


class KGRetrievalHead(nn.Module):
    """RETRIEVE -> VERIFY -> FUSE on cached features z [N, C].

    Ablation switches: use_verify=False (K3, lambda v = 0), use_fuse=False (K4, f = 0),
    uniform_d=True (A1-d, d = 1), uniform_pi=True (pi = 1/k), learn_lambda=False (K4', lambda fixed).

    attribute_source='decoder' reads a from the backbone map before pooling (`features`, `valid`
    in forward) instead of z; it needs joint_mask [A, V] (model/kg.py: attribute_joint_mask) and
    takes decoder_args (model/attribute_decoder.py). Only end to end (model/hypergcn_kg.py): the
    cache holds pooled z only.

    A1-G (module docstring): retrieval_source='graph' with graph_args (model/graph_encoder.py:
    GraphRetrievalScorer; class_embeddings / attribute_embeddings may be None unless
    graph_args.init='text' or a text-based branch needs them), verify_mode, normalize_verify,
    attribute_source='predicate' with predicate_args (PredicateEvidence: columns, mean, std,
    learn_threshold) and phi / phi_valid in forward.
    """

    def __init__(self, in_channels, class_embeddings, attribute_embeddings, kg, k=10,
                 tau=0.07, learn_tau=True, feature_dropout=0.0, tau_pi=0.1, uniform_pi=False,
                 uniform_d=False, use_verify=True, use_fuse=True, lambda_init=1.0, learn_lambda=True,
                 d_model=256, heads=4, attribute_source='z', joint_mask=None, decoder_args=None,
                 retrieval_source='text', graph_args=None, verify_mode='std', normalize_verify=False,
                 predicate_args=None):
        super().__init__()
        if attribute_source not in ('z', 'decoder', 'predicate'):
            raise ValueError(f'attribute_source must be z, decoder or predicate, not {attribute_source}')
        if retrieval_source not in ('text', 'graph'):
            raise ValueError(f'retrieval_source must be text or graph, not {retrieval_source}')
        if verify_mode not in VERIFY_MODES:
            raise ValueError(f'verify_mode must be one of {VERIFY_MODES}, not {verify_mode}')
        self.k, self.tau_pi, self.uniform_pi, self.uniform_d = k, tau_pi, uniform_pi, uniform_d
        self.use_verify, self.use_fuse, self.d_model = use_verify, use_fuse, d_model
        self.verify_mode, self.normalize_verify = verify_mode, normalize_verify
        self.retrieval_source = retrieval_source
        self.feature_dropout = nn.Dropout(feature_dropout)
        # Dropout is applied once here and shared by every branch.
        if retrieval_source == 'graph':
            graph_args = dict(graph_args or {})
            if graph_args.get('init') == 'text':
                graph_args['text_embeddings'] = (class_embeddings, attribute_embeddings)
            self.retrieval = GraphRetrievalScorer(in_channels, kg, tau, learn_tau, **graph_args)
        else:
            self.retrieval = RetrievalScorer(in_channels, class_embeddings, tau, learn_tau)
        self.attribute_source = attribute_source
        A = kg.W.shape[1]
        if attribute_source == 'decoder':
            if joint_mask is None:
                raise ValueError("attribute_source='decoder' needs joint_mask")
            self.decoder = AttributeDecoder(in_channels, attribute_embeddings, joint_mask,
                                            **(decoder_args or {}))
        elif attribute_source == 'predicate':
            if predicate_args is None:
                raise ValueError("attribute_source='predicate' needs predicate_args (columns, mean, std)")
            self.predicates = PredicateEvidence(num_attributes=A, **predicate_args)
        else:
            self.attribute = nn.Linear(in_channels, A)
        if attribute_embeddings is not None:
            self.register_buffer('attribute_embeddings', F.normalize(attribute_embeddings.float(), dim=-1))
        self._set_kg(kg)

        raw = torch.tensor(inverse_softplus(lambda_init))
        if learn_lambda:
            self.lambda_raw = nn.Parameter(raw)
        else:
            self.register_buffer('lambda_raw', raw)

        if use_fuse:
            if retrieval_source == 'graph':
                D_action = D_attr = self.retrieval.proj.out_features
            else:
                if attribute_embeddings is None:
                    raise ValueError('FUSE on text retrieval needs attribute_embeddings')
                D_action, D_attr = class_embeddings.shape[1], attribute_embeddings.shape[1]
            self.U_a = nn.Linear(D_action, d_model)
            self.U_t = nn.Linear(D_attr, d_model)
            self.U_z = nn.Linear(in_channels, d_model)
            self.W_h = nn.Linear(d_model, d_model)
            self.u_val = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
            # Node types: 0 candidate action, 1 neighbor action, 2 attribute.
            self.type_emb = nn.Embedding(3, d_model)
            self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)

    def _set_kg(self, kg):
        self.register_buffer('W', kg.W.float())
        self.register_buffer('known', kg.known.float())
        self.register_buffer('neighbors', (kg.S_sim | kg.S_opp).float())

    def replace_classes(self, class_embeddings, kg):
        """Swap in a new class set (e.g. seen + unseen classes for K6) without touching weights."""
        if self.retrieval_source != 'text':
            raise ValueError('replace_classes needs text retrieval: graph node vectors are learned per class')
        if kg.W.shape != (len(class_embeddings), self.W.shape[1]):
            raise ValueError(f'KG W {tuple(kg.W.shape)} does not match {len(class_embeddings)} classes '
                             f'x {self.W.shape[1]} attributes')
        device = self.W.device
        self.retrieval.class_embeddings = F.normalize(class_embeddings.float(), dim=-1).to(device)
        self.W, self.known = kg.W.float().to(device), kg.known.float().to(device)
        self.neighbors = (kg.S_sim | kg.S_opp).float().to(device)

    @property
    def lam(self):
        return F.softplus(self.lambda_raw)

    def candidates(self, s, labels=None):
        """Top-k classes; with labels (training), a missing ground truth replaces the last
        (lowest-scoring) candidate. Returns cand [N, k] and the label's position (or None)."""
        cand = s.topk(self.k, dim=1).indices
        if labels is None:
            return cand, None
        hit = cand == labels.unsqueeze(1)
        missing = ~hit.any(1)
        cand = cand.clone()
        cand[missing, -1] = labels[missing]
        return cand, (cand == labels.unsqueeze(1)).float().argmax(1)

    def node_tables(self):
        """(action [C, D], attribute [A, D]) vectors the FUSE nodes are built from."""
        if self.retrieval_source == 'graph':
            return self.retrieval.nodes()
        return self.retrieval.class_embeddings, self.attribute_embeddings

    def fuse(self, z, cand, a):
        """h = CrossAttn(U_z z, G_q) and f_c = <W_h h, U_a e_c> / sqrt(d) on the candidates."""
        N, C = z.shape[0], self.W.shape[0]
        in_cand = torch.zeros(N, C, device=z.device).scatter_(1, cand, 1.0).bool()
        neighbor = (in_cand.float() @ self.neighbors > 0) & ~in_cand
        attr_on = in_cand.float() @ self.known > 0

        action_table, attribute_table = self.node_tables()
        action_nodes = self.U_a(action_table)                                           # [C, d]
        action_type = torch.where(in_cand.unsqueeze(-1), self.type_emb.weight[0], self.type_emb.weight[1])
        attr_nodes = (self.U_t(attribute_table) + self.type_emb.weight[2]).unsqueeze(0) \
            + a.unsqueeze(-1) * self.u_val                                              # [N, A, d]
        nodes = torch.cat([action_nodes.unsqueeze(0) + action_type, attr_nodes], dim=1)
        ignore = ~torch.cat([in_cand | neighbor, attr_on], dim=1)                       # True = masked

        h, _ = self.attn(self.U_z(z).unsqueeze(1), nodes, nodes, key_padding_mask=ignore,
                         need_weights=False)
        h = self.W_h(h.squeeze(1))
        return (h.unsqueeze(1) * action_nodes[cand]).sum(-1) / math.sqrt(self.d_model)

    def attributes(self, z, features=None, valid=None, phi=None, phi_valid=None):
        """Returns (a [N, A], evidence [N, A] bool: attributes that carry evidence)."""
        if self.attribute_source == 'predicate':
            if phi is None:
                raise ValueError("attribute_source='predicate' needs phi [N, P] (tools/cache_predicates.py)")
            return self.predicates(phi, phi_valid)
        if self.attribute_source == 'z':
            a = torch.sigmoid(self.attribute(z))
        else:
            if features is None or valid is None:
                raise ValueError("attribute_source='decoder' needs features [N, M, C, T, V] and valid [N, M]")
            a = self.decoder(features, valid)
        return a, torch.ones_like(a, dtype=torch.bool)

    def discriminability(self, Wc, pi, cand, evidence):
        if self.uniform_d:
            d = torch.ones_like(evidence, dtype=Wc.dtype)
        elif self.verify_mode == 'opposed':
            d = opposed_attributes(Wc)
        elif self.verify_mode == 'masked_std':
            d = masked_discriminability(Wc, pi, self.known[cand])
        else:
            d = discriminability(Wc, pi)
        return d * evidence

    def forward(self, z, labels=None, features=None, valid=None, phi=None, phi_valid=None):
        """z: [N, C]; labels only while training (forces the ground truth into Cand);
        features / valid: backbone map before pooling, for attribute_source='decoder';
        phi / phi_valid [N, P]: cached predicates, for attribute_source='predicate'."""
        z = self.feature_dropout(z.float())
        cos = self.retrieval.cosine(z)
        s = cos / self.retrieval.tau
        cand, target = self.candidates(s, labels)
        s_c = s.gather(1, cand)
        a, has_evidence = self.attributes(z, features, valid, phi, phi_valid)
        Wc = self.W[cand]

        if self.uniform_pi:
            pi = torch.full_like(s_c, 1.0 / self.k)
        else:
            pi = torch.softmax(cos.gather(1, cand) / self.tau_pi, dim=1)
        d = self.discriminability(Wc, pi, cand, has_evidence.float())
        evidence = verify(Wc, d, a)
        if self.normalize_verify:
            n_c = (d.unsqueeze(1) * self.known[cand]).sum(-1).clamp_min(1)              # [N, k]
            evidence = evidence / n_c.unsqueeze(-1)
        lam = self.lam if self.use_verify else torch.zeros_like(self.lam)
        v = evidence.sum(-1)
        f = self.fuse(z, cand, a) if self.use_fuse else torch.zeros_like(s_c)

        cand_logits = s_c + lam * v + f
        logits = torch.full_like(s, float('-inf')).scatter(1, cand, cand_logits)
        return dict(logits=logits, s=s, s_c=s_c, v=v, f=f, a=a, d=d, pi=pi, cand=cand,
                    target=target, contributions=lam * evidence, lam=lam, evidence=has_evidence)


def masked_bce(a, labels, y_attr, known, evidence=None):
    """BCE of the attribute head on the cells the KG knows for the sample's class (and, with
    evidence [N, A], that carry evidence: predicate attributes only)."""
    target, mask = y_attr[labels], known[labels].float()
    if evidence is not None:
        mask = mask * evidence.float()
    bce = F.binary_cross_entropy(a.clamp(1e-6, 1 - 1e-6), target, reduction='none')
    return (bce * mask).sum() / mask.sum().clamp_min(1)


def a1_loss(out, labels, head, alpha=0.5, beta=0.5, gamma=0.0):
    """CE_cand(logits) + alpha CE_all(s) + beta BCE_masked(a) + gamma mean|f| / sg(mean(|s_c| + |lambda v|)).
    `out` from head(z, labels=labels) (ground truth forced into the candidates). The gamma
    denominator is a stop-gradient: otherwise the ratio could shrink by inflating s (lowering
    tau) instead of shrinking f. Returns (loss, {term: float})."""
    ce_cand = F.cross_entropy(out['logits'], labels)
    ce_all = F.cross_entropy(out['s'], labels)
    bce = masked_bce(out['a'], labels, (head.W > 0).float(), head.known, out.get('evidence'))
    loss = ce_cand + alpha * ce_all + beta * bce
    terms = dict(ce_cand=ce_cand.item(), ce_all=ce_all.item(), bce=bce.item())
    if gamma:
        scale = (out['s_c'].abs() + (out['lam'] * out['v']).abs()).mean().detach()
        f_ratio = out['f'].abs().mean() / (scale + 1e-8)
        loss = loss + gamma * f_ratio
        terms['f_ratio'] = f_ratio.item()
    return loss, terms
