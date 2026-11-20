"""Decoupled Dual-Branch Concept-Residual MappingNet (Direction 1).

Architecture:
  - Concept Branch (independent of Flora's VAE reconstruction):
      256-D skeleton -> ConceptEncoder -> z_c (D_c) -> CLIPConceptHead -> q_c (768-D CLIP space)
      Trained with Masked Pos-Weighted BCE against frozen CLIP primitive embeddings
      + optional continuous Concept-Prototype alignment loss.
  - Residual Branch (Flora's VAE + DiT flow matching):
      Full latent_dim (768) preserved so Flora's alignment & Phase-2 DiT are not distorted.
      Optional Gradient Reversal Layer (GRL) adversary gently suppresses concept leakage in z_r.
  - Dual-Path Inference:
      Combines standardized Residual Flow-Matching score (-MSE) with Concept-Space score
      (cosine to centered class concept prototypes or probabilistic rule consistency).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fm import MappingNet


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class ConceptEncoder(nn.Module):
    """Dedicated MLP mapping frozen 256-D skeleton features to sub-action concept latents z_c."""

    def __init__(self, ske_dim, concept_dim=256, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(ske_dim),
            nn.Linear(ske_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, concept_dim),
            nn.LayerNorm(concept_dim),
            nn.GELU(),
        )

    def forward(self, ske):
        return self.net(ske)


class CLIPConceptHead(nn.Module):
    """Projects z_c into frozen CLIP space and scores cosine similarity to concept prototypes."""

    def __init__(self, concept_dim, prim_emb):
        super().__init__()
        self.register_buffer('prim_emb', F.normalize(prim_emb.float(), dim=-1))  # (P, 768)
        self.proj = nn.Linear(concept_dim, prim_emb.shape[-1])
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def encode_clip(self, z_c):
        return F.normalize(self.proj(z_c), dim=-1)  # (B, 768)

    def forward(self, z_c):
        q = self.encode_clip(z_c)
        scale = self.logit_scale.exp().clamp(max=50.0)
        logits = scale * (q @ self.prim_emb.t()) + self.bias
        return logits, q


class LinearConceptHead(nn.Module):
    """Linear multi-label concept head (ablation baseline against CLIPConceptHead)."""

    def __init__(self, concept_dim, num_concepts):
        super().__init__()
        self.head = nn.Linear(concept_dim, num_concepts)

    def encode_clip(self, z_c):
        return None

    def forward(self, z_c):
        return self.head(z_c), None


def _zscore_rows(x, eps=1e-6):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, unbiased=False, keepdim=True) + eps)


class ConceptResidualMappingNet(MappingNet):
    """Decoupled Concept-Residual MappingNet built on Flora's MappingNet."""

    def __init__(self, args):
        super().__init__(args)
        self.concept_dim = int(getattr(args, 'concept_dim', 256))
        self.lambda_concept = float(getattr(args, 'lambda_concept', 1.0))
        self.lambda_proto = float(getattr(args, 'lambda_proto', 0.5))
        self.lambda_adv = float(getattr(args, 'lambda_adv', 0.02))
        self.concept_alpha = float(getattr(args, 'concept_alpha', 0.25))
        self.concept_score_mode = getattr(args, 'concept_score', 'proto')  # proto | prob | both
        self.concept_head_type = getattr(args, 'concept_head', 'clip')

        self.concept_encoder = ConceptEncoder(args.ske_dim, self.concept_dim)
        self.concept_head = None
        self.adv_head = None

        # Populated by Trainer via attach_concepts(...)
        self.register_buffer('class_concepts', None)       # (N_cls, P) binary require
        self.register_buffer('class_forbids', None)        # (N_cls, P) binary forbid
        self.register_buffer('seen_prim_mask', None)       # (P,)
        self.register_buffer('pos_weight', None)           # (P,)
        self.register_buffer('class_proto_clip', None)     # (N_cls, 768) centered & L2-normed

    def attach_concepts(self, class_concepts, class_forbids, seen_prim_mask, pos_weight, prim_emb):
        device = next(self.parameters()).device
        self.class_concepts = class_concepts.to(device=device, dtype=torch.float32)
        self.class_forbids = class_forbids.to(device=device, dtype=torch.float32)
        self.seen_prim_mask = seen_prim_mask.to(device=device, dtype=torch.float32)
        self.pos_weight = pos_weight.to(device=device, dtype=torch.float32)

        prim_norm = F.normalize(prim_emb.to(device=device, dtype=torch.float32), dim=-1)  # (P, 768)
        # Centered class concept prototype in CLIP space (parameter-free)
        w_req = self.class_concepts / self.class_concepts.sum(dim=1, keepdim=True).clamp(min=1.0)
        proto = w_req @ prim_norm  # (N_cls, 768)
        proto = proto - proto.mean(dim=0, keepdim=True)
        self.class_proto_clip = F.normalize(proto, dim=-1)

        if self.concept_head_type == 'clip':
            self.concept_head = CLIPConceptHead(self.concept_dim, prim_norm).to(device)
        elif self.concept_head_type == 'linear':
            self.concept_head = LinearConceptHead(self.concept_dim, class_concepts.shape[1]).to(device)
        else:
            raise ValueError(f'unknown concept_head: {self.concept_head_type}')

        # Adversarial concept probe on Flora's residual latent mu_ske (768 -> P)
        self.adv_head = nn.Sequential(
            nn.LayerNorm(self.args.latent_dim),
            nn.Linear(self.args.latent_dim, class_concepts.shape[1]),
        ).to(device)

    def _masked_bce(self, logits, targets):
        loss_mat = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        mask = self.seen_prim_mask.unsqueeze(0)
        denom = (mask.sum() * logits.shape[0]).clamp(min=1.0)
        return (loss_mat * mask).sum() / denom

    def _get_ske_256(self, skeleton_data):
        if not self.args.use_features:
            return self.shiftgcn(skeleton_data).mean(3).mean(2)
        return skeleton_data

    def forward_phase1(self, skeleton_data, attuned_semantics, seen_classes, label_idx):
        for p in self.velocity_predictor.parameters():
            p.requires_grad = False
        for mod in (self.semantic_encoder, self.motion_encoder, self.motion_decoder,
                    self.semantic_decoder, self.concept_encoder):
            for p in mod.parameters():
                p.requires_grad = True
        for mod in (self.concept_head, self.adv_head):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

        # --- 1) Residual Branch: Flora's VAE alignment ---
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
        loss_flora = loss_recon + self.args.lambda_align * (loss_mu + loss_log_var)

        seen_idx = torch.as_tensor(seen_classes, device=label_idx.device, dtype=torch.long)
        global_cls = seen_idx[label_idx]
        targets = self.class_concepts[global_cls]  # (B, P)

        # --- 2) Concept Branch (decoupled from VAE reconstruction) ---
        loss_concept = loss_flora.new_tensor(0.0)
        loss_proto = loss_flora.new_tensor(0.0)
        if self.concept_head is not None and self.lambda_concept > 0.0:
            ske_256 = self._get_ske_256(skeleton_data)
            z_c = self.concept_encoder(ske_256)
            logits_c, q_clip = self.concept_head(z_c)
            loss_concept = self._masked_bce(logits_c, targets)
            if q_clip is not None and self.lambda_proto > 0.0:
                # Contrastive alignment to seen class concept prototypes in centered CLIP space
                seen_protos = self.class_proto_clip[seen_idx]  # (N_seen, 768)
                q_centered = F.normalize(q_clip - q_clip.mean(dim=0, keepdim=True), dim=-1)
                proto_logits = 10.0 * (q_centered @ seen_protos.t())
                loss_proto = F.cross_entropy(proto_logits, label_idx)

        # --- 3) Adversarial GRL suppression of concept leakage in Residual latent mu_ske ---
        loss_adv = loss_flora.new_tensor(0.0)
        if self.adv_head is not None and self.lambda_adv > 0.0:
            mu_res_pool = mu_ske.mean(dim=1)  # (B, 768)
            adv_logits = self.adv_head(grad_reverse(mu_res_pool, self.lambda_adv))
            loss_adv = self._masked_bce(adv_logits, targets)

        total_loss = (
            loss_flora
            + self.lambda_concept * loss_concept
            + self.lambda_proto * loss_proto
            + loss_adv  # adv_head gets full grad; motion_encoder gets -lambda_adv * grad via GRL
        )
        return {
            'loss': total_loss,
            'loss_mu': loss_mu,
            'loss_log_var': loss_log_var,
            'loss_recon': loss_recon,
            'loss_concept': loss_concept,
            'loss_proto': loss_proto,
            'loss_adv': loss_adv,
        }

    def forward_phase2(self, skeleton_data, skeleton_data_contraastive, label_idx,
                       label_idx_contrastive, seen_classes, attuned_semantics):
        for mod in (self.concept_encoder, self.concept_head, self.adv_head):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False
        return super().forward_phase2(
            skeleton_data, skeleton_data_contraastive, label_idx,
            label_idx_contrastive, seen_classes, attuned_semantics
        )

    @torch.no_grad()
    def extract_concept_latents(self, skeleton_data):
        """Return (probs, q_clip, z_c, mu_res) for diagnostics and concept inference."""
        ske_256 = self._get_ske_256(skeleton_data)
        z_c = self.concept_encoder(ske_256)
        logits, q_clip = self.concept_head(z_c)
        probs = torch.sigmoid(logits)

        ske_rep = ske_256.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
        _, mu_ske, _ = self.motion_encoder(ske_rep)
        mu_res = mu_ske.mean(dim=1)
        return probs, q_clip, z_c, mu_res

    @torch.no_grad()
    def compute_concept_class_scores(self, skeleton_data, candidate_classes):
        """Return (B, N_cand) concept-based class affinity scores."""
        probs, q_clip, _, _ = self.extract_concept_latents(skeleton_data)
        cand_idx = torch.as_tensor(candidate_classes, device=probs.device, dtype=torch.long)

        s_proto = None
        if q_clip is not None:
            q_centered = F.normalize(q_clip - q_clip.mean(dim=0, keepdim=True), dim=-1)
            protos = self.class_proto_clip[cand_idx]  # (N_cand, 768)
            s_proto = q_centered @ protos.t()

        # Probabilistic rule-consistency score: mean log p(require) + 0.5 * mean log(1 - p(forbid))
        p_clamped = probs.clamp(1e-4, 1.0 - 1e-4)
        req = self.class_concepts[cand_idx]  # (N_cand, P)
        forb = self.class_forbids[cand_idx]  # (N_cand, P)
        req_w = req / req.sum(dim=1, keepdim=True).clamp(min=1.0)
        forb_w = forb / forb.sum(dim=1, keepdim=True).clamp(min=1.0)
        has_forb = (forb.sum(dim=1) > 0).float().unsqueeze(0)
        s_prob = (
            torch.log(p_clamped) @ req_w.t()
            + 0.5 * has_forb * (torch.log(1.0 - p_clamped) @ forb_w.t())
        )

        if self.concept_score_mode == 'proto' and s_proto is not None:
            return s_proto
        if self.concept_score_mode == 'prob' or s_proto is None:
            return s_prob
        return 0.5 * (_zscore_rows(s_proto) + _zscore_rows(s_prob))

    @torch.no_grad()
    def predict_zsl_scores(self, skeleton_data, unseen_classes, attuned_semantics):
        """Return (s_fm, s_concept) each of shape (B, N_unseen) where higher = better."""
        ske_256 = self._get_ske_256(skeleton_data)
        B = ske_256.shape[0]
        ske_rep = ske_256.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
        z1_ske, _, _ = self.motion_encoder(ske_rep)
        semantics_embedding = attuned_semantics[unseen_classes]
        z0_text, _, _ = self.semantic_encoder(semantics_embedding)

        timestep = torch.ones(B, device=ske_256.device) * self.args.step_ratio
        class_chunk = self.args.eval_class_chunk or len(unseen_classes)
        mse_error_list = []
        for start in range(0, len(unseen_classes), class_chunk):
            z0_chunk = z0_text[start:start + class_chunk]
            n_class = z0_chunk.shape[0]
            z0_batch = z0_chunk.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_class, *z0_chunk.shape[1:])
            z1_batch = z1_ske.unsqueeze(1).expand(-1, n_class, -1, -1).reshape(B * n_class, *z1_ske.shape[1:])
            time_batch = timestep.unsqueeze(1).expand(-1, n_class).reshape(-1)
            z_time = self.psi(time_batch, z0_batch, z1_batch)
            pred_velocity = self.velocity_predictor(z_time, time_batch)
            target_velocity = self.Dt_psi(time_batch, z0_batch, z1_batch)
            mse_error_list.append(F.mse_loss(pred_velocity, target_velocity, reduction='none').mean((1, 2)).view(B, n_class))
        mse_all = torch.cat(mse_error_list, dim=1)  # (B, N_unseen)
        s_fm = -mse_all
        s_concept = self.compute_concept_class_scores(skeleton_data, unseen_classes)
        return s_fm, s_concept

    @torch.no_grad()
    def predict_zsl(self, skeleton_data, unseen_classes, attuned_semantics, label):
        s_fm, s_concept = self.predict_zsl_scores(skeleton_data, unseen_classes, attuned_semantics)
        if self.concept_alpha == 0.0:
            scores = s_fm
        else:
            scores = _zscore_rows(s_fm) + self.concept_alpha * _zscore_rows(s_concept)
        return {
            'predictions_idx': torch.argmax(scores, dim=1),
            's_fm': s_fm,
            's_concept': s_concept,
        }
