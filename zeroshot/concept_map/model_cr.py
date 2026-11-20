"""Concept-Augmented Motion Mapping (Cách A — Direction 1).

Two-stage motion-side concept grounding built on Flora:
  Stage 0 (Concept Encoder):
    Frozen 256-D skeleton -> ConceptEncoder -> CLIPConceptHead -> concept probs p_hat (B, P)
    and continuous CLIP concept vector q_clip (B, 768).
    Trained with Masked Pos-Weighted BCE only (no seen-class InfoNCE shortcut, no GRL).
    Frozen after Stage 0; produces a centered, scale-matched concept motion embedding m_c (B, 768):
      - concept_rep='soft_proto': m_c = sum_p p_hat_p * e_p (explicitly reconstructed from concept bank)
      - concept_rep='q_clip'    : m_c = q_clip (continuous CLIP projection)
  Stage 1 & 2 (Flora Phase-1 Alignment & Phase-2 DiT Flow Matching):
    Injects m_c directly into the motion side (no post-hoc score re-ranking!):
      - fusion_mode='concat'    : motion_encoder / motion_decoder operate on [ske_256 || gamma * m_c]
      - fusion_mode='latent_add': adds gamma * m_c to motion latent (motion-side residual skip)
      - fusion_mode='none'      : pure Flora baseline
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fm import MappingNet, Encoder, Decoder


class ConceptEncoder(nn.Module):
    """MLP mapping frozen 256-D skeleton features to sub-action concept latent z_c."""

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

    def forward(self, z_c):
        q = F.normalize(self.proj(z_c), dim=-1)  # (B, 768)
        scale = self.logit_scale.exp().clamp(max=50.0)
        logits = scale * (q @ self.prim_emb.t()) + self.bias
        return logits, q


class ConceptAugmentedMappingNet(MappingNet):
    """Flora MappingNet with motion-side Concept-Augmented representation."""

    def __init__(self, args):
        super().__init__(args)
        self.concept_dim = int(getattr(args, 'concept_dim', 256))
        self.fusion_mode = getattr(args, 'fusion_mode', 'concat')     # none | concat | latent_add
        self.concept_rep = getattr(args, 'concept_rep', 'soft_proto') # soft_proto | q_clip
        self.gamma_c = float(getattr(args, 'gamma_c', 0.5))
        self.clip_dim = 768

        # If concat mode, rebuild motion_encoder / motion_decoder for input_dim = ske_dim + 768
        if self.fusion_mode == 'concat':
            aug_dim = args.ske_dim + self.clip_dim
            self.motion_encoder = Encoder(input_dim=aug_dim, output_dim=args.latent_dim)
            self.motion_decoder = Decoder(input_dim=args.latent_dim, output_dim=aug_dim)

        self.concept_encoder = ConceptEncoder(args.ske_dim, self.concept_dim)
        self.concept_head = None

        # Buffers populated by Trainer
        self.register_buffer('class_concepts', None)    # (N_cls, P)
        self.register_buffer('seen_prim_mask', None)    # (P,)
        self.register_buffer('pos_weight', None)        # (P,)
        self.register_buffer('prim_emb', None)          # (P, 768)
        self.register_buffer('class_proto_clip', None)  # (N_cls, 768)
        self.register_buffer('mc_mean', torch.zeros(1, self.clip_dim))
        self.register_buffer('ske_scale', torch.tensor(1.0))

    def attach_concepts(self, class_concepts, seen_prim_mask, pos_weight, prim_emb):
        device = next(self.parameters()).device
        self.class_concepts = class_concepts.to(device=device, dtype=torch.float32)
        self.seen_prim_mask = seen_prim_mask.to(device=device, dtype=torch.float32)
        self.pos_weight = pos_weight.to(device=device, dtype=torch.float32)
        self.prim_emb = F.normalize(prim_emb.to(device=device, dtype=torch.float32), dim=-1)

        # Centered class concept prototype in CLIP space (for C-Only diagnostic)
        w_req = self.class_concepts / self.class_concepts.sum(dim=1, keepdim=True).clamp(min=1.0)
        proto = w_req @ self.prim_emb
        proto = proto - proto.mean(dim=0, keepdim=True)
        self.class_proto_clip = F.normalize(proto, dim=-1)

        self.concept_head = CLIPConceptHead(self.concept_dim, self.prim_emb).to(device)
        self.mc_mean = self.mc_mean.to(device)
        self.ske_scale = self.ske_scale.to(device)

    def _get_ske_256(self, skeleton_data):
        if not self.args.use_features:
            return self.shiftgcn(skeleton_data).mean(3).mean(2)
        return skeleton_data

    def forward_concept_stage(self, skeleton_data, seen_classes, label_idx):
        """Stage 0: train ConceptEncoder + CLIPConceptHead with Masked Pos-Weighted BCE."""
        ske_256 = self._get_ske_256(skeleton_data)
        z_c = self.concept_encoder(ske_256)
        logits, _ = self.concept_head(z_c)
        seen_idx = torch.as_tensor(seen_classes, device=label_idx.device, dtype=torch.long)
        targets = self.class_concepts[seen_idx[label_idx]]
        loss_mat = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        mask = self.seen_prim_mask.unsqueeze(0)
        denom = (mask.sum() * logits.shape[0]).clamp(min=1.0)
        return (loss_mat * mask).sum() / denom

    @torch.no_grad()
    def infer_concept_raw(self, ske_256):
        """Return (probs, q_clip, m_c_raw) from the frozen concept branch."""
        z_c = self.concept_encoder(ske_256)
        logits, q_clip = self.concept_head(z_c)
        probs = torch.sigmoid(logits)
        if self.concept_rep == 'soft_proto':
            # Reconstruct motion concept embedding strictly from frozen CLIP concept dictionary
            w = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            m_c = F.normalize(w @ self.prim_emb, dim=-1)
        elif self.concept_rep == 'q_clip':
            m_c = q_clip
        else:
            raise ValueError(f'unknown concept_rep: {self.concept_rep}')
        return probs, q_clip, m_c

    @torch.no_grad()
    def infer_concept_centered(self, ske_256):
        """Centered, L2-normalized, scale-matched concept motion embedding (B, 768)."""
        _, _, m_c = self.infer_concept_raw(ske_256)
        m_centered = F.normalize(m_c - self.mc_mean, dim=-1)
        return m_centered * self.ske_scale

    def _augment_skeleton_tokens(self, skeleton_data):
        """Build (B, L+1, D_in) motion input for motion_encoder."""
        ske_256 = self._get_ske_256(skeleton_data)
        if self.fusion_mode == 'concat' and self.gamma_c > 0.0:
            mc = self.infer_concept_centered(ske_256)  # (B, 768)
            ske_in = torch.cat([ske_256, self.gamma_c * mc], dim=-1)  # (B, 1024)
        else:
            ske_in = ske_256
        return ske_in.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)

    def _encode_motion(self, skeleton_data):
        ske_tokens = self._augment_skeleton_tokens(skeleton_data)
        z1_ske, mu_ske, log_var_ske = self.motion_encoder(ske_tokens)
        if self.fusion_mode == 'latent_add' and self.gamma_c > 0.0:
            ske_256 = self._get_ske_256(skeleton_data)
            _, _, m_c = self.infer_concept_raw(ske_256)
            mc_dir = F.normalize(m_c - self.mc_mean, dim=-1).unsqueeze(1)  # (B, 1, 768)
            # Match typical latent norm smoothly
            lat_scale = mu_ske.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            delta = self.gamma_c * mc_dir * lat_scale
            mu_ske = mu_ske + delta
            z1_ske = z1_ske + delta
        return ske_tokens, z1_ske, mu_ske, log_var_ske

    def prepare_embedding(self, skeleton_data, seen_classes, label_idx, attuned_semantics):
        ske_tokens = self._augment_skeleton_tokens(skeleton_data)
        semantics_seen = attuned_semantics[seen_classes]
        semantics_embedding = semantics_seen[label_idx]
        return ske_tokens, semantics_embedding

    def forward_phase1(self, skeleton_data, attuned_semantics, seen_classes, label_idx):
        for p in self.velocity_predictor.parameters():
            p.requires_grad = False
        for mod in (self.concept_encoder, self.concept_head):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False
        for mod in (self.semantic_encoder, self.motion_encoder, self.motion_decoder, self.semantic_decoder):
            for p in mod.parameters():
                p.requires_grad = True

        semantics_seen = attuned_semantics[seen_classes]
        semantics_embedding = semantics_seen[label_idx]
        skeleton_embedding, z1_ske, mu_ske, log_var_ske = self._encode_motion(skeleton_data)
        z0_text, mu_text, log_var_text = self.semantic_encoder(semantics_embedding)

        loss_mu = F.mse_loss(mu_text, mu_ske, reduction='mean')
        loss_log_var = F.mse_loss(log_var_text, log_var_ske, reduction='mean')
        recon_loss_s2s = F.mse_loss(self.motion_decoder(z1_ske), skeleton_embedding, reduction='mean')
        recon_loss_t2t = F.mse_loss(self.semantic_decoder(z0_text), semantics_embedding, reduction='mean')
        recon_loss_s2t = F.mse_loss(self.semantic_decoder(z1_ske), semantics_embedding, reduction='mean')
        recon_loss_t2s = F.mse_loss(self.motion_decoder(z0_text), skeleton_embedding, reduction='mean')
        loss_recon = (recon_loss_s2s + recon_loss_t2t + recon_loss_s2t + recon_loss_t2s) / 4.0
        total_loss = loss_recon + self.args.lambda_align * (loss_mu + loss_log_var)
        return {
            'loss': total_loss,
            'loss_mu': loss_mu,
            'loss_log_var': loss_log_var,
            'loss_recon': loss_recon,
        }

    def forward_phase2(self, skeleton_data, skeleton_data_contraastive, label_idx,
                       label_idx_contrastive, seen_classes, attuned_semantics):
        for p in self.velocity_predictor.parameters():
            p.requires_grad = True
        for mod in (self.semantic_encoder, self.motion_encoder, self.motion_decoder,
                    self.semantic_decoder, self.concept_encoder, self.concept_head):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False

        semantics_seen = attuned_semantics[seen_classes]
        sem_pos = semantics_seen[label_idx]
        sem_con = semantics_seen[label_idx_contrastive]
        z0_text, _, _ = self.semantic_encoder(sem_pos)
        _, z1_ske, _, _ = self._encode_motion(skeleton_data)
        z0_text_con, _, _ = self.semantic_encoder(sem_con)
        _, z1_ske_con, _, _ = self._encode_motion(skeleton_data_contraastive)

        time_steps = self.time_step_sampler.sample_time(z0_text)
        z_time = self.psi(time_steps, z0_text, z1_ske)
        target_velocity = self.Dt_psi(time_steps, z0_text, z1_ske)
        pred_velocity = self.velocity_predictor(z_time, time_steps)
        flow_matching_loss = F.mse_loss(pred_velocity, target_velocity, reduction='mean')

        target_velocity_con = self.Dt_psi(time_steps, z0_text_con, z1_ske_con)
        flow_matching_loss_con = F.mse_loss(pred_velocity, target_velocity_con, reduction='mean')
        total_loss = flow_matching_loss - self.args.lambda_cfm * flow_matching_loss_con
        return {
            'loss': total_loss,
            'loss_flow_matching': flow_matching_loss,
            'loss_flow_matching_contrastive': flow_matching_loss_con,
        }

    @torch.no_grad()
    def predict_zsl(self, skeleton_data, unseen_classes, attuned_semantics, label):
        _, z1_ske, _, _ = self._encode_motion(skeleton_data)
        B = z1_ske.shape[0]
        semantics_embedding = attuned_semantics[unseen_classes]
        z0_text, _, _ = self.semantic_encoder(semantics_embedding)

        timestep = torch.ones(B, device=z1_ske.device) * self.args.step_ratio
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
        mse_error_list = torch.cat(mse_error_list, dim=1)
        return {'predictions_idx': torch.argmin(mse_error_list, dim=1)}

    @torch.no_grad()
    def predict_gzsl(self, skeleton_data, seen_classes, unseen_classes, attuned_semantics):
        _, z1_ske, _, _ = self._encode_motion(skeleton_data)
        B = z1_ske.shape[0]
        z0_text, _, _ = self.semantic_encoder(attuned_semantics)

        timestep = torch.ones(B, device=z1_ske.device) * self.args.step_ratio
        n_total = len(seen_classes) + len(unseen_classes)
        class_chunk = self.args.eval_class_chunk or n_total
        mse_error_list = []
        for start in range(0, n_total, class_chunk):
            z0_chunk = z0_text[start:start + class_chunk]
            n_class = z0_chunk.shape[0]
            z0_batch = z0_chunk.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_class, *z0_chunk.shape[1:])
            z1_batch = z1_ske.unsqueeze(1).expand(-1, n_class, -1, -1).reshape(B * n_class, *z1_ske.shape[1:])
            time_batch = timestep.unsqueeze(1).expand(-1, n_class).reshape(-1)
            z_time = self.psi(time_batch, z0_batch, z1_batch)
            pred_velocity = self.velocity_predictor(z_time, time_batch)
            target_velocity = self.Dt_psi(time_batch, z0_batch, z1_batch)
            mse_error_list.append(F.mse_loss(pred_velocity, target_velocity, reduction='none').mean((1, 2)).view(B, n_class))
        mse_error_list = torch.cat(mse_error_list, dim=1)
        mse_seen_min = torch.min(mse_error_list[:, seen_classes], dim=1).values
        mse_unseen_min = torch.min(mse_error_list[:, unseen_classes], dim=1).values
        ratio = mse_seen_min / mse_unseen_min
        row_seen = (ratio > self.args.calibration_factor).unsqueeze(1)
        row_unseen = ~row_seen
        col_seen = torch.zeros(1, n_total, dtype=torch.bool, device=z1_ske.device)
        col_seen[:, seen_classes] = True
        col_unseen = torch.zeros(1, n_total, dtype=torch.bool, device=z1_ske.device)
        col_unseen[:, unseen_classes] = True
        mse_error_list[row_seen & col_seen] = 1e6
        mse_error_list[row_unseen & col_unseen] = 1e6
        return {'predictions_idx': torch.argmin(mse_error_list, dim=1)}
