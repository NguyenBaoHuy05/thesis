"""Group-Factorized Centered CLIPConceptHead + Differential Top-K Concept Arbitration.

Addresses all 5 bottlenecks identified in Step-1 diagnostics:
  1. Centered CLIP Concept Space: removes the shared anisotropic CLIP mean vector across all P primitives
     so primitive prototypes spread uniformly across the hypersphere.
  2. Group-Factorized Multi-Head Projection: combines a shared global projection W_global (for cross-group
     zero-shot transfer to unseen-only primitives) with G=7 body-part-specific projections W_g
     (hand, interaction, body, arm, torso, leg, head) so multiple body parts can activate simultaneously
     without pulling against each other.
  3. Cross-Modal Soft LLM Distillation: distills soft concept targets from seen-class LLM text embeddings
     onto uncovered primitives (1 - seen_prim_mask) so unseen-only primitives receive soft semantic
     supervision during training without violating zero-shot separation.
  4. Untouched Flora VAE + DiT Backbone: keeps Flora's Phase-1 & Phase-2 completely uncorrupted.
  5. IDF-Weighted Differential Top-K Arbitration: at test time, compares DiT's top-1 candidate c_1 against
     top-K alternatives c_k (k=2..K) ONLY on their distinguishing primitives (R_{c_k} \\ R_{c_1} and
     R_{c_1} \\ R_{c_k}, weighted by IDF specificity and forbid polarity) against the relative DiT
     velocity gap (e_k - e_1) / e_1. Shared primitives cancel out identically to zero.
"""
import contextlib
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fm import MappingNet


@contextlib.contextmanager
def _preserve_rng_state():
    """Save and restore Python, NumPy, PyTorch CPU & CUDA RNG states so Flora's RNG stream is untouched."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class ConceptEncoder(nn.Module):
    """MLP mapping frozen 256-D skeleton features to sub-action concept latent z_c."""

    def __init__(self, ske_dim, concept_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(ske_dim),
            nn.Linear(ske_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, concept_dim),
            nn.LayerNorm(concept_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, ske):
        return self.net(ske)


class GroupFactorizedCLIPConceptHead(nn.Module):
    """Centered CLIP concept head with shared global projection + low-rank axis adapters."""

    def __init__(self, concept_dim, prim_emb, prim_groups, num_groups,
                 center_clip=True, use_groups=True, group_rank=32):
        super().__init__()
        self.center_clip = bool(center_clip)
        self.use_groups = bool(use_groups)
        self.num_groups = int(num_groups)
        clip_dim = prim_emb.shape[-1]

        e = F.normalize(prim_emb.float(), dim=-1)
        if self.center_clip:
            e = F.normalize(e - e.mean(dim=0, keepdim=True), dim=-1)
        self.register_buffer('prim_emb', e)  # (P, 768)
        self.register_buffer('prim_groups', prim_groups.long())  # (P,) in 0..G-1

        # Shared global projection transfers alignment to rare / unseen-only concepts
        self.global_proj = nn.Linear(concept_dim, clip_dim)
        # Low-rank group-specific residual adapters avoid memorizing seen classes
        if self.use_groups:
            self.group_down = nn.ModuleList([
                nn.Linear(concept_dim, group_rank) for _ in range(self.num_groups)
            ])
            self.group_up = nn.ModuleList([
                nn.Linear(group_rank, clip_dim, bias=False) for _ in range(self.num_groups)
            ])
            for gu in self.group_up:
                nn.init.normal_(gu.weight, std=0.01)
        else:
            self.group_down = None
            self.group_up = None

        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-1.5))

    def compute_similarity(self, z_c):
        """Return (B, P) cosine similarities between group-projected z_c and centered prim_emb."""
        q_glob = self.global_proj(z_c)  # (B, 768)
        if not self.use_groups or self.group_down is None:
            q = F.normalize(q_glob, dim=-1)
            return q @ self.prim_emb.t()

        # Stack G group-specific queries: (B, G, 768)
        q_groups = torch.stack(
            [F.normalize(q_glob + gu(F.gelu(gd(z_c))), dim=-1)
             for gd, gu in zip(self.group_down, self.group_up)],
            dim=1,
        )
        # Gather the group query corresponding to each primitive p: (B, P, 768)
        g_idx = self.prim_groups.view(1, -1, 1).expand(z_c.shape[0], -1, q_glob.shape[-1])
        q_per_prim = torch.gather(q_groups, dim=1, index=g_idx)  # (B, P, 768)
        sim = (q_per_prim * self.prim_emb.unsqueeze(0)).sum(dim=-1)  # (B, P)
        return sim

    def forward(self, z_c):
        sim = self.compute_similarity(z_c)
        scale = self.logit_scale.exp().clamp(max=15.0)
        logits = scale * sim + self.bias
        return logits, sim


class ConceptArbitratedMappingNet(MappingNet):
    """Flora MappingNet paired with Group-Factorized Centered CLIPConceptHead & Differential Top-K Arbitration."""

    def __init__(self, args):
        super().__init__(args)
        self.concept_dim = int(getattr(args, 'concept_dim', 256))
        self.center_clip = bool(getattr(args, 'center_clip', 1))
        self.use_groups = bool(getattr(args, 'use_groups', 1))
        self.lambda_soft = float(getattr(args, 'lambda_soft', 0.0))
        self.arb_topk = int(getattr(args, 'arb_topk', 5))
        self.arb_alpha = float(getattr(args, 'arb_alpha', 0.08))
        self.arb_gate_tau = float(getattr(args, 'arb_gate_tau', 0.15))
        self.arb_forbid_boost = float(getattr(args, 'arb_forbid_boost', 1.5))

        with _preserve_rng_state():
            self.concept_encoder = ConceptEncoder(args.ske_dim, self.concept_dim)
        self.concept_head = None

        # Populated by Trainer via attach_concepts(...)
        self.register_buffer('class_concepts', None)   # (N_cls, P) binary require
        self.register_buffer('class_forbids', None)    # (N_cls, P) binary forbid
        self.register_buffer('seen_prim_mask', None)   # (P,)
        self.register_buffer('pos_weight', None)       # (P,)
        self.register_buffer('idf_weight', None)       # (P,) IDF specificity
        self.register_buffer('soft_text_targets', None)# (N_cls, P) soft CLIP text teacher
        self.register_buffer('sim_mean', None)         # (1, P) training-set mean of concept sim
        self.register_buffer('sim_std', None)          # (1, P) training-set std of concept sim

    def attach_concepts(self, class_concepts, class_forbids, seen_prim_mask,
                        pos_weight, idf_weight, prim_emb, prim_groups, num_groups):
        device = next(self.parameters()).device
        P = class_concepts.shape[1]
        self.class_concepts = class_concepts.to(device=device, dtype=torch.float32)
        self.class_forbids = class_forbids.to(device=device, dtype=torch.float32)
        self.seen_prim_mask = seen_prim_mask.to(device=device, dtype=torch.float32)
        self.pos_weight = pos_weight.to(device=device, dtype=torch.float32)
        self.idf_weight = idf_weight.to(device=device, dtype=torch.float32)
        self.sim_mean = torch.zeros(1, P, device=device)
        self.sim_std = torch.ones(1, P, device=device)

        self.concept_head = GroupFactorizedCLIPConceptHead(
            self.concept_dim,
            prim_emb.to(device),
            prim_groups.to(device),
            num_groups=num_groups,
            center_clip=self.center_clip,
            use_groups=self.use_groups,
        ).to(device)

        # Build parameter-free soft concept teacher from Flora's class text embeddings (LLM half)
        with torch.no_grad():
            # class_embeddings: (N_cls, L+1, 1536), last token [-1], LLM description half [768:]
            cls_llm = self.class_embeddings[:, -1, 768:].to(device=device, dtype=torch.float32)
            cls_llm = F.normalize(cls_llm, dim=-1)
            if self.center_clip:
                cls_llm = F.normalize(cls_llm - cls_llm.mean(dim=0, keepdim=True), dim=-1)
            sim_t = cls_llm @ self.concept_head.prim_emb.t()  # (N_cls, P)
            # Standardize per class so soft teacher probabilities span (0.05, 0.95) smoothly
            sim_t_z = (sim_t - sim_t.mean(dim=1, keepdim=True)) / (sim_t.std(dim=1, keepdim=True) + 1e-6)
            self.soft_text_targets = torch.sigmoid(1.5 * sim_t_z - 0.5).detach()

    def _get_ske_256(self, skeleton_data):
        if not self.args.use_features:
            return self.shiftgcn(skeleton_data).mean(3).mean(2)
        return skeleton_data

    def forward_concept_stage(self, skeleton_data, seen_classes, label_idx):
        """Train ConceptEncoder + GroupFactorizedCLIPConceptHead on seen classes."""
        ske_256 = self._get_ske_256(skeleton_data)
        z_c = self.concept_encoder(ske_256)
        logits, _ = self.concept_head(z_c)

        seen_idx = torch.as_tensor(seen_classes, device=label_idx.device, dtype=torch.long)
        global_cls = seen_idx[label_idx]
        targets = self.class_concepts[global_cls]

        # 1) Hard Masked Pos-Weighted BCE on seen-covered primitives
        bce_mat = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        mask_cov = self.seen_prim_mask.unsqueeze(0)
        denom_cov = (mask_cov.sum() * logits.shape[0]).clamp(min=1.0)
        loss_hard = (bce_mat * mask_cov).sum() / denom_cov

        # 2) Soft Cross-Modal LLM Distillation on uncovered primitives (1 - seen_prim_mask)
        loss_soft = loss_hard.new_tensor(0.0)
        mask_uncov = 1.0 - mask_cov
        if self.lambda_soft > 0.0 and mask_uncov.sum() > 0:
            soft_t = self.soft_text_targets[global_cls]
            soft_mat = F.binary_cross_entropy_with_logits(logits, soft_t, reduction='none')
            denom_uncov = (mask_uncov.sum() * logits.shape[0]).clamp(min=1.0)
            loss_soft = (soft_mat * mask_uncov).sum() / denom_uncov

        return loss_hard + self.lambda_soft * loss_soft

    @torch.no_grad()
    def infer_concepts(self, skeleton_data):
        """Return (probs, sim, z_sim) where z_sim is smoothly bounded training-standardized activation."""
        ske_256 = self._get_ske_256(skeleton_data)
        z_c = self.concept_encoder(ske_256)
        logits, sim = self.concept_head(z_c)
        probs = torch.sigmoid(logits)
        raw_z = (sim - self.sim_mean) / (self.sim_std + 1e-6)
        z_sim = 4.0 * torch.tanh(raw_z / 4.0)
        return probs, sim, z_sim

    @torch.no_grad()
    def compute_fm_mse(self, skeleton_data, candidate_classes, attuned_semantics):
        """Exact Flora DiT flow-matching velocity MSE matrix (B, N_cand), lower = better."""
        ske_256 = self._get_ske_256(skeleton_data)
        B = ske_256.shape[0]
        ske_rep = ske_256.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
        z1_ske, _, _ = self.motion_encoder(ske_rep)
        z0_text, _, _ = self.semantic_encoder(attuned_semantics[candidate_classes])

        timestep = torch.ones(B, device=ske_256.device) * self.args.step_ratio
        class_chunk = self.args.eval_class_chunk or len(candidate_classes)
        mse_error_list = []
        for start in range(0, len(candidate_classes), class_chunk):
            z0_chunk = z0_text[start:start + class_chunk]
            n_class = z0_chunk.shape[0]
            z0_batch = z0_chunk.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_class, *z0_chunk.shape[1:])
            z1_batch = z1_ske.unsqueeze(1).expand(-1, n_class, -1, -1).reshape(B * n_class, *z1_ske.shape[1:])
            time_batch = timestep.unsqueeze(1).expand(-1, n_class).reshape(-1)
            z_time = self.psi(time_batch, z0_batch, z1_batch)
            pred_velocity = self.velocity_predictor(z_time, time_batch)
            target_velocity = self.Dt_psi(time_batch, z0_batch, z1_batch)
            mse_error_list.append(F.mse_loss(pred_velocity, target_velocity, reduction='none').mean((1, 2)).view(B, n_class))
        return torch.cat(mse_error_list, dim=1)  # (B, N_cand)

    @torch.no_grad()
    def build_pairwise_diff_matrix(self, candidate_classes):
        """Precompute (N_cand, N_cand, P) IDF-weighted differential concept tensor D[k, m, p].

        For candidate pair (c_k, c_m):
          - Shared required primitives (R_{c_k} & R_{c_m}) are ZEROED OUT.
          - Primitives in R_{c_k} \\ R_{c_m} receive positive IDF weight (boosted if in F_{c_m}).
          - Primitives in R_{c_m} \\ R_{c_k} receive negative IDF weight (boosted if in F_{c_k}).
        """
        cand_idx = torch.as_tensor(candidate_classes, device=self.class_concepts.device, dtype=torch.long)
        req = self.class_concepts[cand_idx]   # (C, P)
        forb = self.class_forbids[cand_idx]   # (C, P)
        idf = self.idf_weight.view(1, 1, -1)  # (1, 1, P)

        # Exclusive required primitives: req_k * (1 - req_m) -> shape (C, C, P)
        excl_k = req.unsqueeze(1) * (1.0 - req.unsqueeze(0))
        excl_m = req.unsqueeze(0) * (1.0 - req.unsqueeze(1))

        # Downweight unseen-only primitives (no supervised training signal) so noisy rare tags cannot dominate
        rel = torch.where(
            self.seen_prim_mask > 0,
            torch.ones_like(self.seen_prim_mask),
            torch.full_like(self.seen_prim_mask, 0.15),
        ).view(1, 1, -1)

        # Boost weight when the other class explicitly forbids the primitive
        boost_k = 1.0 + (self.arb_forbid_boost - 1.0) * forb.unsqueeze(0)
        boost_m = 1.0 + (self.arb_forbid_boost - 1.0) * forb.unsqueeze(1)

        w_k = excl_k * idf * rel * boost_k
        w_m = excl_m * idf * rel * boost_m

        # Exact zero-sum normalization per side (+1.0 on excl_k, -1.0 on excl_m) so sum(D[k,m]) == 0
        # This guarantees zero cardinality bias toward classes with more concepts.
        sum_k = w_k.sum(dim=-1, keepdim=True)
        sum_m = w_m.sum(dim=-1, keepdim=True)
        norm_k = torch.where(sum_k > 1e-6, w_k / sum_k.clamp(min=1e-6), torch.zeros_like(w_k))
        norm_m = torch.where(sum_m > 1e-6, w_m / sum_m.clamp(min=1e-6), torch.zeros_like(w_m))
        return norm_k - norm_m  # (C, C, P)

    @torch.no_grad()
    def differential_topk_arbitrate(self, mse_fm, z_sim, diff_mat, topk=5, alpha=0.08, gate_tau=None):
        """Uncertainty-Gated Differential Top-K Arbitration.

        mse_fm  : (B, C) DiT velocity MSE (lower = better)
        z_sim   : (B, P) standardized concept activation per primitive
        diff_mat: (C, C, P) differential concept weight tensor D[k, m, p]
        gate_tau: uncertainty gate temperature tau > 0 (g(x) = exp(-m1/tau)); <=0 disables gating
        """
        if alpha <= 0.0 or topk <= 1:
            return torch.argmin(mse_fm, dim=1)

        if gate_tau is None:
            gate_tau = self.arb_gate_tau

        B, C = mse_fm.shape
        # Adaptive Top-K: avoid arbitrating over 100% of classes when C is small (e.g., C=5 -> K=2)
        K = min(int(topk), max(2, C // 2), C)
        top_mse, top_idx = torch.topk(mse_fm, k=K, dim=1, largest=False)  # (B, K)
        c1_idx = top_idx[:, 0]                                            # (B,)
        e1 = top_mse[:, 0:1].clamp(min=1e-6)                              # (B, 1)
        rel_gap = (top_mse - e1) / e1                                     # (B, K) >= 0, col 0 == 0

        # Uncertainty gate: attenuate concept override when DiT has a large Top-1 vs Top-2 margin
        if gate_tau > 0.0 and K >= 2:
            m1 = rel_gap[:, 1:2]                                          # (B, 1) >= 0
            gate = torch.exp(-m1 / float(gate_tau))                       # (B, 1) in (0, 1]
        else:
            gate = 1.0

        # Gather D[c_k, c_1, :] for each sample b and candidate k in 0..K-1 -> (B, K, P)
        c1_rep = c1_idx.unsqueeze(1).expand(B, K)
        d_k1 = diff_mat[top_idx, c1_rep]  # (B, K, P)

        # Differential concept score: (B, K)
        delta_s = (d_k1 * z_sim.unsqueeze(1)).sum(dim=-1)
        utility = -rel_gap + (alpha * gate) * delta_s  # (B, K), utility[:, 0] == 0
        best_in_k = torch.argmax(utility, dim=1)
        return top_idx[torch.arange(B, device=mse_fm.device), best_in_k]

    @torch.no_grad()
    def predict_zsl(self, skeleton_data, unseen_classes, attuned_semantics, label):
        mse_fm = self.compute_fm_mse(skeleton_data, unseen_classes, attuned_semantics)
        if self.arb_alpha <= 0.0:
            return {'predictions_idx': torch.argmin(mse_fm, dim=1), 'mse_fm': mse_fm}
        _, _, z_sim = self.infer_concepts(skeleton_data)
        diff_mat = self.build_pairwise_diff_matrix(unseen_classes)
        preds = self.differential_topk_arbitrate(
            mse_fm, z_sim, diff_mat,
            topk=self.arb_topk, alpha=self.arb_alpha, gate_tau=self.arb_gate_tau,
        )
        return {'predictions_idx': preds, 'mse_fm': mse_fm, 'z_sim': z_sim}


# Backward-compatible alias
ConceptResidualMappingNet = ConceptArbitratedMappingNet
ConceptAugmentedMappingNet = ConceptArbitratedMappingNet
