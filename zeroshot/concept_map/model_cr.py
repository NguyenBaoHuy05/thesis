"""Disentangled Concept-Residual MappingNet (Direction 1 prototype).

Subclasses Flora's MappingNet without modifying Flora's core files:
- Keeps total latent_dim = D (768) so Flora's decoders and Phase-2 DiT work unchanged.
- Splits the aligned latent mean mu into:
    mu_c = mu[..., :D_c]   (Concept subspace: predicts sub-action concepts)
    mu_r = mu[..., D_c:]   (Residual subspace: captures fine-grained cues orthogonal to concepts)
- Uses a shared CLIP-anchored (or linear) concept head on both skeleton and text concept latents,
  plus a batch cross-correlation orthogonality loss between mu_c and mu_r.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fm import MappingNet


class CLIPConceptHead(nn.Module):
    """Projects a concept latent vector to CLIP space and scores cosine similarity to frozen concept prototypes."""

    def __init__(self, concept_dim, prim_emb):
        super().__init__()
        # prim_emb: (P, 768) L2-normalized CLIP pooled embeddings of primitive descriptions
        self.register_buffer('prim_emb', F.normalize(prim_emb.float(), dim=-1))
        self.proj = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, prim_emb.shape[-1]),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-2.0))  # shared scalar bias avoids seen-frequency bias

    def forward(self, z_c):
        # z_c: (B, D_c) -> logits: (B, P)
        q = F.normalize(self.proj(z_c), dim=-1)
        scale = self.logit_scale.exp().clamp(max=50.0)
        return scale * (q @ self.prim_emb.t()) + self.bias


class LinearConceptHead(nn.Module):
    """Standard linear multi-label head (ablation baseline against CLIPConceptHead)."""

    def __init__(self, concept_dim, num_concepts):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, num_concepts),
        )

    def forward(self, z_c):
        return self.head(z_c)


def cross_corr_orth_loss(z_c, z_r, eps=1e-6):
    """Mean squared off-block cross-correlation between batch-standardized z_c (B, D_c) and z_r (B, D_r)."""
    if z_r.shape[-1] == 0 or z_c.shape[0] < 2:
        return z_c.new_tensor(0.0)
    zc = z_c - z_c.mean(dim=0, keepdim=True)
    zr = z_r - z_r.mean(dim=0, keepdim=True)
    zc = zc / (zc.std(dim=0, unbiased=False, keepdim=True) + eps)
    zr = zr / (zr.std(dim=0, unbiased=False, keepdim=True) + eps)
    corr = (zc.t() @ zr) / z_c.shape[0]
    return (corr ** 2).mean()


class ConceptResidualMappingNet(MappingNet):
    """Flora MappingNet augmented with a disentangled Concept-Residual Phase-1 objective."""

    def __init__(self, args):
        super().__init__(args)
        self.concept_dim = int(getattr(args, 'concept_dim', 256))
        if not (0 < self.concept_dim <= args.latent_dim):
            raise ValueError(f'concept_dim={self.concept_dim} must be in (0, {args.latent_dim}]')
        self.residual_dim = args.latent_dim - self.concept_dim
        self.lambda_concept = float(getattr(args, 'lambda_concept', 0.1))
        self.lambda_orth = float(getattr(args, 'lambda_orth', 0.1))
        self.text_concept_weight = float(getattr(args, 'text_concept_weight', 0.5))
        self.concept_head_type = getattr(args, 'concept_head', 'clip')
        self.concept_head = None

        # Populated by Trainer via attach_concepts(...)
        self.register_buffer('class_concepts', None)
        self.register_buffer('seen_prim_mask', None)
        self.register_buffer('pos_weight', None)

    def attach_concepts(self, class_concepts, seen_prim_mask, pos_weight, prim_emb):
        """Attach concept targets and initialize the concept readout head."""
        device = next(self.parameters()).device
        self.class_concepts = class_concepts.to(device=device, dtype=torch.float32)
        self.seen_prim_mask = seen_prim_mask.to(device=device, dtype=torch.float32)
        self.pos_weight = pos_weight.to(device=device, dtype=torch.float32)
        if self.concept_head_type == 'clip':
            self.concept_head = CLIPConceptHead(self.concept_dim, prim_emb.to(device)).to(device)
        elif self.concept_head_type == 'linear':
            self.concept_head = LinearConceptHead(self.concept_dim, class_concepts.shape[1]).to(device)
        else:
            raise ValueError(f'unknown concept_head: {self.concept_head_type}')

    def _masked_bce(self, logits, targets):
        loss_mat = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        mask = self.seen_prim_mask.unsqueeze(0)  # (1, P)
        denom = (mask.sum() * logits.shape[0]).clamp(min=1.0)
        return (loss_mat * mask).sum() / denom

    def forward_phase1(self, skeleton_data, attuned_semantics, seen_classes, label_idx):
        for p in self.velocity_predictor.parameters():
            p.requires_grad = False
        for mod in (self.semantic_encoder, self.motion_encoder, self.motion_decoder, self.semantic_decoder):
            for p in mod.parameters():
                p.requires_grad = True
        if self.concept_head is not None:
            for p in self.concept_head.parameters():
                p.requires_grad = True

        skeleton_embedding, semantics_embedding = self.prepare_embedding(
            skeleton_data, seen_classes, label_idx, attuned_semantics
        )
        z0_text, mu_text, log_var_text = self.semantic_encoder(semantics_embedding)  # (B, T, D)
        z1_ske, mu_ske, log_var_ske = self.motion_encoder(skeleton_embedding)        # (B, T, D)

        loss_mu = F.mse_loss(mu_text, mu_ske, reduction='mean')
        loss_log_var = F.mse_loss(log_var_text, log_var_ske, reduction='mean')
        recon_loss_s2s = F.mse_loss(self.motion_decoder(z1_ske), skeleton_embedding, reduction='mean')
        recon_loss_t2t = F.mse_loss(self.semantic_decoder(z0_text), semantics_embedding, reduction='mean')
        recon_loss_s2t = F.mse_loss(self.semantic_decoder(z1_ske), semantics_embedding, reduction='mean')
        recon_loss_t2s = F.mse_loss(self.motion_decoder(z0_text), skeleton_embedding, reduction='mean')
        loss_recon = (recon_loss_s2s + recon_loss_t2t + recon_loss_s2t + recon_loss_t2s) / 4.0
        loss_flora = loss_recon + self.args.lambda_align * (loss_mu + loss_log_var)

        # Pool across token dimension to obtain per-sample concept & residual latents
        mu_ske_pool = mu_ske.mean(dim=1)    # (B, D)
        mu_text_pool = mu_text.mean(dim=1)  # (B, D)
        ske_c, ske_r = mu_ske_pool[:, :self.concept_dim], mu_ske_pool[:, self.concept_dim:]
        txt_c, txt_r = mu_text_pool[:, :self.concept_dim], mu_text_pool[:, self.concept_dim:]

        loss_concept = loss_flora.new_tensor(0.0)
        if self.concept_head is not None and self.lambda_concept > 0.0:
            seen_idx = torch.as_tensor(seen_classes, device=label_idx.device, dtype=torch.long)
            global_cls = seen_idx[label_idx]
            targets = self.class_concepts[global_cls]  # (B, P)
            logits_ske = self.concept_head(ske_c)
            loss_c_ske = self._masked_bce(logits_ske, targets)
            if self.text_concept_weight > 0.0:
                logits_txt = self.concept_head(txt_c)
                loss_c_txt = self._masked_bce(logits_txt, targets)
                loss_concept = loss_c_ske + self.text_concept_weight * loss_c_txt
            else:
                loss_concept = loss_c_ske

        loss_orth = loss_flora.new_tensor(0.0)
        if self.residual_dim > 0 and self.lambda_orth > 0.0:
            loss_orth = 0.5 * (
                cross_corr_orth_loss(ske_c, ske_r) + cross_corr_orth_loss(txt_c, txt_r)
            )

        total_loss = loss_flora + self.lambda_concept * loss_concept + self.lambda_orth * loss_orth
        return {
            'loss': total_loss,
            'loss_mu': loss_mu,
            'loss_log_var': loss_log_var,
            'loss_recon': loss_recon,
            'loss_concept': loss_concept,
            'loss_orth': loss_orth,
        }

    def forward_phase2(self, skeleton_data, skeleton_data_contraastive, label_idx,
                       label_idx_contrastive, seen_classes, attuned_semantics):
        if self.concept_head is not None:
            for p in self.concept_head.parameters():
                p.requires_grad = False
        return super().forward_phase2(
            skeleton_data, skeleton_data_contraastive, label_idx,
            label_idx_contrastive, seen_classes, attuned_semantics
        )

    @torch.no_grad()
    def extract_concept_latents(self, skeleton_data):
        """Return (probs, mu_c, mu_r) for a batch of skeleton features (B, 256)."""
        if not self.args.use_features:
            ske = self.shiftgcn(skeleton_data).mean(3).mean(2)
        else:
            ske = skeleton_data
        ske_rep = ske.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
        _, mu_ske, _ = self.motion_encoder(ske_rep)
        mu_pool = mu_ske.mean(dim=1)
        mu_c = mu_pool[:, :self.concept_dim]
        mu_r = mu_pool[:, self.concept_dim:]
        probs = torch.sigmoid(self.concept_head(mu_c)) if self.concept_head is not None else None
        return probs, mu_c, mu_r
