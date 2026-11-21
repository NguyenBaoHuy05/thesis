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
    """Dual-modal concept encoder fusing Cross-Modal VAE latent mu_vae with raw skeleton skip."""

    def __init__(self, ske_dim, latent_dim=768, concept_dim=256, hidden_dim=512,
                 dropout=0.25, use_vae_latent=True):
        super().__init__()
        self.use_vae_latent = bool(use_vae_latent)
        self.ske_net = nn.Sequential(
            nn.LayerNorm(ske_dim),
            nn.Linear(ske_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, concept_dim),
            nn.LayerNorm(concept_dim),
            nn.GELU(),
        )
        if self.use_vae_latent:
            self.vae_net = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, concept_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.out_norm = nn.LayerNorm(concept_dim)
        else:
            self.vae_net = None
            self.out_norm = None

    def encode_vae_latent(self, mu_vae):
        """Encode pooled Cross-Modal VAE latent (mu_ske or mu_text) into concept space."""
        return self.out_norm(self.vae_net(mu_vae))

    def forward(self, ske, mu_vae=None):
        h_ske = self.ske_net(ske)
        if self.use_vae_latent and mu_vae is not None:
            return self.out_norm(self.vae_net(mu_vae) + 0.5 * h_ske)
        return h_ske


class GroupFactorizedCLIPConceptHead(nn.Module):
    """Centered CLIP concept head with shared global projection + low-rank axis adapters."""

    def __init__(self, concept_dim, prim_emb, prim_groups, num_groups,
                 center_clip=True, use_groups=True, group_rank=64):
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

        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))
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
        scale = self.logit_scale.exp().clamp(max=20.0)
        logits = scale * sim + self.bias
        return logits, sim


class ConceptArbitratedMappingNet(MappingNet):
    """Flora MappingNet paired with Group-Factorized Centered CLIPConceptHead & Differential Top-K Arbitration."""

    def __init__(self, args):
        super().__init__(args)
        self.concept_dim = int(getattr(args, 'concept_dim', 256))
        self.center_clip = bool(getattr(args, 'center_clip', 1))
        self.use_groups = bool(getattr(args, 'use_groups', 1))
        self.use_vae_latent = bool(getattr(args, 'use_vae_latent', 1))
        self.lambda_vae_c = float(getattr(args, 'lambda_vae_c', 0.15))
        self.lambda_soft = float(getattr(args, 'lambda_soft', 0.0))
        self.arb_topk = int(getattr(args, 'arb_topk', 5))
        self.arb_alpha = float(getattr(args, 'arb_alpha', 0.05))
        self.arb_gate_tau = float(getattr(args, 'arb_gate_tau', 0.15))
        self.arb_forbid_boost = float(getattr(args, 'arb_forbid_boost', 1.5))

        with _preserve_rng_state():
            self.concept_encoder = ConceptEncoder(
                args.ske_dim,
                latent_dim=args.latent_dim,
                concept_dim=self.concept_dim,
                use_vae_latent=self.use_vae_latent,
            )
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

    def _get_vae_ske_mu(self, ske_256):
        """Extract Cross-Modal VAE skeleton mean mu_ske[:, -1] cleanly in token-wise pass."""
        if not self.use_vae_latent:
            return None
        # Encoder layers are purely token-wise, so single-token pass equals pooled mu_ske[:, -1]
        _, mu_ske, _ = self.motion_encoder(ske_256.unsqueeze(1))
        return mu_ske[:, 0, :]

    def _masked_concept_bce(self, logits, targets):
        bce_mat = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        mask_cov = self.seen_prim_mask.unsqueeze(0)
        denom_cov = (mask_cov.sum() * logits.shape[0]).clamp(min=1.0)
        return (bce_mat * mask_cov).sum() / denom_cov

    def forward_phase1(self, skeleton_data, attuned_semantics, seen_classes, label_idx):
        """Flora Phase 1 Cross-Modal VAE + Dual-Modal Concept Grounding on (mu_ske, mu_text)."""
        for p in self.velocity_predictor.parameters():
            p.requires_grad = False
        for mod in (self.semantic_encoder, self.motion_encoder, self.motion_decoder, self.semantic_decoder):
            for p in mod.parameters():
                p.requires_grad = True
        if self.kg_gate is not None:
            self.kg_gate.requires_grad_(True)

        skeleton_embedding, semantics_embedding = self.prepare_embedding(
            skeleton_data, seen_classes, label_idx, attuned_semantics
        )
        z0_text, mu_text, log_var_text = self.semantic_encoder(semantics_embedding)
        z1_ske, mu_ske, log_var_ske = self.motion_encoder(skeleton_embedding)

        loss_mu = F.mse_loss(mu_text, mu_ske, reduction='mean')
        loss_log_var = F.mse_loss(log_var_text, log_var_ske, reduction='mean')
        recon_loss_s2s = F.mse_loss(self.motion_decoder(z1_ske), skeleton_embedding, reduction='mean')
        recon_loss_t2t = F.mse_loss(self.semantic_decoder(z0_text), semantics_embedding, reduction='mean')
        recon_loss_s2t = F.mse_loss(self.semantic_decoder(z1_ske), semantics_embedding, reduction='mean')
        recon_loss_t2s = F.mse_loss(self.motion_decoder(z0_text), skeleton_embedding, reduction='mean')
        loss_recon = (recon_loss_s2s + recon_loss_t2t + recon_loss_s2t + recon_loss_t2s) / 4.0

        total_loss = loss_recon + self.args.lambda_align * (loss_mu + loss_log_var)

        # Ground BOTH VAE modalities (mu_ske, mu_text) onto the 6-axis MSK-CB space
        if self.lambda_vae_c > 0.0 and self.use_vae_latent and self.concept_head is not None:
            self.concept_encoder.vae_net.requires_grad_(True)
            self.concept_encoder.out_norm.requires_grad_(True)
            self.concept_head.requires_grad_(True)

            seen_idx = torch.as_tensor(seen_classes, device=label_idx.device, dtype=torch.long)
            targets = self.class_concepts[seen_idx[label_idx]]
            z_vae_ske = self.concept_encoder.encode_vae_latent(mu_ske[:, -1, :])
            z_vae_txt = self.concept_encoder.encode_vae_latent(mu_text[:, -1, :])
            logits_ske, _ = self.concept_head(z_vae_ske)
            logits_txt, _ = self.concept_head(z_vae_txt)
            loss_vae_c = 0.5 * (
                self._masked_concept_bce(logits_ske, targets)
                + self._masked_concept_bce(logits_txt, targets)
            )
            total_loss = total_loss + self.lambda_vae_c * loss_vae_c

        return {
            'loss': total_loss,
            'loss_mu': loss_mu,
            'loss_log_var': loss_log_var,
            'loss_recon': loss_recon,
        }

    def forward_phase2(self, skeleton_data, skeleton_data_contraastive, label_idx,
                       label_idx_contrastive, seen_classes, attuned_semantics):
        self.concept_encoder.requires_grad_(False)
        if self.concept_head is not None:
            self.concept_head.requires_grad_(False)
        return super().forward_phase2(
            skeleton_data, skeleton_data_contraastive, label_idx,
            label_idx_contrastive, seen_classes, attuned_semantics,
        )

    def forward_concept_stage(self, skeleton_data, seen_classes, label_idx, attuned_semantics=None):
        """Train Dual-Modal ConceptEncoder + GroupFactorizedCLIPConceptHead on frozen Cross-Modal VAE latents."""
        ske_256 = self._get_ske_256(skeleton_data)
        with torch.no_grad():
            mu_ske_p = self._get_vae_ske_mu(ske_256)
        z_c = self.concept_encoder(ske_256, mu_vae=mu_ske_p)
        logits, _ = self.concept_head(z_c)

        seen_idx = torch.as_tensor(seen_classes, device=label_idx.device, dtype=torch.long)
        global_cls = seen_idx[label_idx]
        targets = self.class_concepts[global_cls]

        # 1) Hard Masked Pos-Weighted BCE on skeleton + VAE latent
        loss_hard = self._masked_concept_bce(logits, targets)

        # 2) Dual-Modal Text-Latent Concept Anchoring (transfers through shared Cross-Modal VAE space)
        if self.use_vae_latent and attuned_semantics is not None:
            with torch.no_grad():
                sem_batch = attuned_semantics[global_cls]
                _, mu_txt, _ = self.semantic_encoder(sem_batch)
                mu_txt_p = mu_txt[:, -1, :]
            z_txt = self.concept_encoder.encode_vae_latent(mu_txt_p)
            logits_txt, _ = self.concept_head(z_txt)
            loss_hard = loss_hard + 0.5 * self._masked_concept_bce(logits_txt, targets)

        # 3) Soft Cross-Modal LLM Distillation on uncovered primitives (1 - seen_prim_mask)
        loss_soft = loss_hard.new_tensor(0.0)
        mask_cov = self.seen_prim_mask.unsqueeze(0)
        mask_uncov = 1.0 - mask_cov
        if self.lambda_soft > 0.0 and mask_uncov.sum() > 0:
            soft_t = self.soft_text_targets[global_cls]
            soft_mat = F.binary_cross_entropy_with_logits(logits, soft_t, reduction='none')
            denom_uncov = (mask_uncov.sum() * logits.shape[0]).clamp(min=1.0)
            loss_soft = (soft_mat * mask_uncov).sum() / denom_uncov

        return loss_hard + self.lambda_soft * loss_soft

    @torch.no_grad()
    def infer_concepts(self, skeleton_data, mu_ske_p=None):
        """Return (probs, sim, z_sim) using Cross-Modal VAE latent mu_ske_p + skeleton residual."""
        ske_256 = self._get_ske_256(skeleton_data)
        if mu_ske_p is None:
            mu_ske_p = self._get_vae_ske_mu(ske_256)
        z_c = self.concept_encoder(ske_256, mu_vae=mu_ske_p)
        logits, sim = self.concept_head(z_c)
        probs = torch.sigmoid(logits)
        raw_z = (sim - self.sim_mean) / (self.sim_std + 1e-6)
        z_sim = 4.0 * torch.tanh(raw_z / 4.0)
        return probs, sim, z_sim

    @torch.no_grad()
    def compute_fm_mse(self, skeleton_data, candidate_classes, attuned_semantics, return_mu=False):
        """Exact Flora DiT flow-matching velocity MSE matrix (B, N_cand), lower = better."""
        ske_256 = self._get_ske_256(skeleton_data)
        B = ske_256.shape[0]
        ske_rep = ske_256.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
        z1_ske, mu_ske, _ = self.motion_encoder(ske_rep)
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
        mse_fm = torch.cat(mse_error_list, dim=1)  # (B, N_cand)
        if return_mu:
            return mse_fm, mu_ske[:, -1, :]
        return mse_fm

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
        mse_fm, mu_ske_p = self.compute_fm_mse(skeleton_data, unseen_classes, attuned_semantics, return_mu=True)
        if self.arb_alpha <= 0.0:
            return {'predictions_idx': torch.argmin(mse_fm, dim=1), 'mse_fm': mse_fm}
        _, _, z_sim = self.infer_concepts(skeleton_data, mu_ske_p=mu_ske_p)
        diff_mat = self.build_pairwise_diff_matrix(unseen_classes)
        preds = self.differential_topk_arbitrate(
            mse_fm, z_sim, diff_mat,
            topk=self.arb_topk, alpha=self.arb_alpha, gate_tau=self.arb_gate_tau,
        )
        return {'predictions_idx': preds, 'mse_fm': mse_fm, 'z_sim': z_sim}

    @torch.no_grad()
    def _cached_diff_matrix(self, classes):
        key = tuple(int(c) for c in classes)
        cache = getattr(self, '_diff_cache', None)
        if cache is None:
            cache = self._diff_cache = {}
        if key not in cache:
            cache[key] = self.build_pairwise_diff_matrix(list(key))
        return cache[key]

    @torch.no_grad()
    def predict_gzsl(self, skeleton_data, seen_classes, unseen_classes, attuned_semantics):
        """Flora's seen/unseen domain decision, then concept arbitration inside the chosen domain."""
        alpha = float(getattr(self.args, 'arb_alpha_gzsl', self.arb_alpha))
        if alpha <= 0.0:
            return super().predict_gzsl(skeleton_data, seen_classes, unseen_classes, attuned_semantics)

        dev = self.class_concepts.device
        n_total = len(seen_classes) + len(unseen_classes)
        all_cls = list(range(n_total))
        mse_all, mu_ske_p = self.compute_fm_mse(skeleton_data, all_cls, attuned_semantics, return_mu=True)
        _, _, z_sim = self.infer_concepts(skeleton_data, mu_ske_p=mu_ske_p)

        seen_t = torch.as_tensor(seen_classes, device=dev, dtype=torch.long)
        unseen_t = torch.as_tensor(unseen_classes, device=dev, dtype=torch.long)
        mse_seen, mse_unseen = mse_all[:, seen_t], mse_all[:, unseen_t]

        # Identical domain decision to Flora (calibration_factor on min-MSE ratio)
        ratio = mse_seen.min(dim=1).values / mse_unseen.min(dim=1).values
        # As in Flora: ratio > calibration_factor => seen classes are penalised => sample goes to the UNSEEN domain
        go_unseen = ratio > self.args.calibration_factor  # (B,)

        pred_seen = seen_t[self.differential_topk_arbitrate(
            mse_seen, z_sim, self._cached_diff_matrix(seen_classes),
            topk=self.arb_topk, alpha=alpha, gate_tau=self.arb_gate_tau)]
        pred_unseen = unseen_t[self.differential_topk_arbitrate(
            mse_unseen, z_sim, self._cached_diff_matrix(unseen_classes),
            topk=self.arb_topk, alpha=alpha, gate_tau=self.arb_gate_tau)]
        return {'predictions_idx': torch.where(go_unseen, pred_unseen, pred_seen)}


# Backward-compatible alias
ConceptResidualMappingNet = ConceptArbitratedMappingNet
ConceptAugmentedMappingNet = ConceptArbitratedMappingNet
