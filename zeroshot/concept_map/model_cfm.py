"""Compositional / Concept-Conditioned Flow Matching (Direction 2).

Key design:
  - Phase 1 (Flora VAE alignment) is 100% UNCHANGED -> zero distortion to skeleton/text latents,
    zero seen-class ConceptEncoder domain shift.
  - Phase 2 upgrades Flora's unconditioned DiT into ConceptConditionedDiT:
      c = t_emb + gate * concept_proj(y_concept)
    injected into adaLN-Zero of every DiTBlock and FinalLayer (zero-initialized so training
    starts from exact Flora DiT behavior and smoothly learns concept-conditioned velocity fields).
  - Multi-granularity Compositional Flow Training:
      1) Class-level flow: z0_text(y) -> z1_ske(x) conditioned on class concept vector c_y
         (or randomly dropped to 0 for classifier-free unconditional flow v(z_t, t | 0)).
      2) Sub-action concept flow: neutral anchor z0_bar -> z1_ske(x) conditioned on a single
         positive primitive e_p (p in R_y), with a bounded margin loss against a negative
         primitive e_q (q not in R_y) so DiT learns per-concept velocity fields directly.
  - Dual Inference from ONE DiT (no separate concept classifier needed):
      1) Unseen Class ZSL:
           - mode='adaln'   : v_theta(z_t^c, t* | c_c)
           - mode='superpos': v_0 + gamma * mean_{p in R_c} (v_theta(z_t^c, t* | e_p) - v_0)
      2) Zero-Shot Sub-Action Concept Inference:
           s_flow(x, p) = -MSE(v_theta(z_t^neu, t* | e_p), u^neu) for all p in 1..P.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import DiT
from models.fm import MappingNet


class ConceptConditionedDiT(DiT):
    """Flora's DiT extended with zero-initialized adaLN-Zero concept conditioning."""

    def __init__(self, in_channels=768, hidden_size=768, depth=1, num_heads=12,
                 mlp_ratio=4.0, concept_dim=768):
        super().__init__(
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.concept_proj = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Zero-init final projection so at step 0, c == t_emb (identical to vanilla Flora DiT)
        nn.init.constant_(self.concept_proj[-1].weight, 0.0)
        nn.init.constant_(self.concept_proj[-1].bias, 0.0)

    def forward(self, x, t, y_cond=None):
        x = self.x_embedder(x)
        c = self.t_embedder(t)
        if y_cond is not None:
            c = c + self.concept_proj(y_cond)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x


class CompositionalFMMappingNet(MappingNet):
    """Flora MappingNet with Concept-Conditioned / Compositional Flow Matching in Phase 2."""

    def __init__(self, args):
        super().__init__(args)
        self.cfm_mode = getattr(args, 'cfm_mode', 'adaln')          # none | adaln | superpos
        self.lambda_sub = float(getattr(args, 'lambda_sub', 0.25))  # sub-action concept flow weight
        self.sub_margin = float(getattr(args, 'sub_margin', 0.05))  # margin between pos & neg concept MSE
        self.cond_drop = float(getattr(args, 'cond_drop', 0.15))    # CFG dropout prob for unconditional v_0
        self.superpos_gamma = float(getattr(args, 'superpos_gamma', 1.0))

        # Replace vanilla DiT with ConceptConditionedDiT (same depth/heads/hidden_size)
        self.velocity_predictor = ConceptConditionedDiT(
            in_channels=args.latent_dim,
            hidden_size=args.latent_dim,
            depth=args.dit_layers,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            concept_dim=768,
        )

        # Populated by Trainer via attach_concepts(...)
        self.register_buffer('prim_emb_centered', None)  # (P, 768) centered & L2-normed CLIP concept emb
        self.register_buffer('class_cond_clip', None)    # (N_cls, 768) centered class concept cond
        self.register_buffer('class_concepts', None)     # (N_cls, P) binary require matrix
        self.register_buffer('seen_prim_mask', None)     # (P,)
        self.register_buffer('z0_neutral', None)         # (1, L+1, D) mean seen semantic latent

    def attach_concepts(self, class_concepts, seen_prim_mask, prim_emb):
        device = next(self.parameters()).device
        self.class_concepts = class_concepts.to(device=device, dtype=torch.float32)
        self.seen_prim_mask = seen_prim_mask.to(device=device, dtype=torch.float32)

        prim_norm = F.normalize(prim_emb.to(device=device, dtype=torch.float32), dim=-1)  # (P, 768)
        prim_cent = F.normalize(prim_norm - prim_norm.mean(dim=0, keepdim=True), dim=-1)
        self.prim_emb_centered = prim_cent

        w_req = self.class_concepts / self.class_concepts.sum(dim=1, keepdim=True).clamp(min=1.0)
        cls_raw = w_req @ prim_norm
        self.class_cond_clip = F.normalize(cls_raw - cls_raw.mean(dim=0, keepdim=True), dim=-1)

    @torch.no_grad()
    def cache_neutral_anchor(self, attuned_semantics, seen_classes):
        """Cache mean seen semantic latent z0_bar as class-agnostic origin for sub-action flow."""
        self.semantic_encoder.eval()
        sem_seen = attuned_semantics[seen_classes]
        _, mu_seen, _ = self.semantic_encoder(sem_seen)  # (N_seen, T, D)
        self.z0_neutral = mu_seen.mean(dim=0, keepdim=True).detach()  # (1, T, D)

    def _sample_pos_neg_concepts(self, global_cls):
        """For each sample's global class id, sample one positive p in R_y and one negative q not in R_y."""
        B = global_cls.shape[0]
        device = global_cls.device
        req_mat = self.class_concepts[global_cls]  # (B, P)
        # Positive sampling: uniform over required primitives of each class
        pos_probs = req_mat / req_mat.sum(dim=1, keepdim=True).clamp(min=1.0)
        pos_idx = torch.multinomial(pos_probs, 1).squeeze(1)  # (B,)

        # Negative sampling: uniform over seen-covered primitives NOT required by the class
        neg_mask = (1.0 - req_mat) * self.seen_prim_mask.unsqueeze(0)
        fallback = (neg_mask.sum(dim=1, keepdim=True) < 0.5).float()
        neg_mask = neg_mask + fallback * (1.0 - req_mat)
        neg_probs = neg_mask / neg_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        neg_idx = torch.multinomial(neg_probs, 1).squeeze(1)  # (B,)
        return pos_idx.to(device), neg_idx.to(device)

    def forward_phase2(self, skeleton_data, skeleton_data_contraastive, label_idx,
                       label_idx_contrastive, seen_classes, attuned_semantics):
        for p in self.velocity_predictor.parameters():
            p.requires_grad = True
        for mod in (self.semantic_encoder, self.motion_encoder, self.motion_decoder, self.semantic_decoder):
            for p in mod.parameters():
                p.requires_grad = False

        skeleton_embedding, semantics_embedding = self.prepare_embedding(
            skeleton_data, seen_classes, label_idx, attuned_semantics
        )
        skeleton_embedding_con, semantics_embedding_con = self.prepare_embedding(
            skeleton_data_contraastive, seen_classes, label_idx_contrastive, attuned_semantics
        )
        z0_text, _, _ = self.semantic_encoder(semantics_embedding)
        z1_ske, _, _ = self.motion_encoder(skeleton_embedding)
        z0_text_con, _, _ = self.semantic_encoder(semantics_embedding_con)
        z1_ske_con, _, _ = self.motion_encoder(skeleton_embedding_con)

        time_steps = self.time_step_sampler.sample_time(z0_text)
        z_time = self.psi(time_steps, z0_text, z1_ske)
        target_velocity = self.Dt_psi(time_steps, z0_text, z1_ske)

        # Build concept conditioning for class-level flow
        y_cond = None
        seen_idx = torch.as_tensor(seen_classes, device=label_idx.device, dtype=torch.long)
        global_cls = seen_idx[label_idx]
        if self.cfm_mode != 'none':
            if self.cfm_mode == 'superpos':
                # Train on single-concept conditions + CFG dropout so velocity fields compose additively
                pos_p, _ = self._sample_pos_neg_concepts(global_cls)
                y_cond = self.prim_emb_centered[pos_p]
            else:
                y_cond = self.class_cond_clip[global_cls]
            if self.cond_drop > 0.0:
                keep = (torch.rand(y_cond.shape[0], 1, device=y_cond.device) >= self.cond_drop).float()
                y_cond = y_cond * keep

        pred_velocity = self.velocity_predictor(z_time, time_steps, y_cond=y_cond)
        flow_matching_loss = F.mse_loss(pred_velocity, target_velocity, reduction='mean')

        target_velocity_con = self.Dt_psi(time_steps, z0_text_con, z1_ske_con)
        flow_matching_loss_con = F.mse_loss(pred_velocity, target_velocity_con, reduction='mean')
        total_loss = flow_matching_loss - self.args.lambda_cfm * flow_matching_loss_con

        # Sub-action concept flow loss (neutral anchor z0_bar -> z1_ske conditioned on single concept)
        loss_sub = total_loss.new_tensor(0.0)
        if self.cfm_mode != 'none' and self.lambda_sub > 0.0 and self.z0_neutral is not None:
            B = z1_ske.shape[0]
            pos_idx, neg_idx = self._sample_pos_neg_concepts(global_cls)
            z0_neu = self.z0_neutral.expand(B, -1, -1)
            t_sub = self.time_step_sampler.sample_time(z0_neu)
            z_t_neu = self.psi(t_sub, z0_neu, z1_ske)
            u_neu = self.Dt_psi(t_sub, z0_neu, z1_ske)

            v_pos = self.velocity_predictor(z_t_neu, t_sub, y_cond=self.prim_emb_centered[pos_idx])
            v_neg = self.velocity_predictor(z_t_neu, t_sub, y_cond=self.prim_emb_centered[neg_idx])
            mse_pos = F.mse_loss(v_pos, u_neu, reduction='none').mean(dim=(1, 2))
            mse_neg = F.mse_loss(v_neg, u_neu, reduction='none').mean(dim=(1, 2))
            margin_term = F.relu(mse_pos - mse_neg + self.sub_margin)
            loss_sub = mse_pos.mean() + margin_term.mean()
            total_loss = total_loss + self.lambda_sub * loss_sub

        return {
            'loss': total_loss,
            'loss_flow_matching': flow_matching_loss,
            'loss_flow_matching_contrastive': flow_matching_loss_con,
            'loss_sub': loss_sub,
        }

    def _predict_class_velocity(self, z_time, time_batch, cand_class_ids, B):
        """Predict velocity for (B * N_cand, T, D) given candidate global class IDs."""
        n_class = len(cand_class_ids)
        if self.cfm_mode == 'none':
            return self.velocity_predictor(z_time, time_batch, y_cond=None)
        if self.cfm_mode == 'adaln':
            cond = self.class_cond_clip[cand_class_ids]  # (n_class, 768)
            cond_batch = cond.unsqueeze(0).expand(B, -1, -1).reshape(B * n_class, -1)
            return self.velocity_predictor(z_time, time_batch, y_cond=cond_batch)

        # cfm_mode == 'superpos': v_comp = v_0 + gamma * mean_{p in R_c} (v(p) - v_0)
        v_uncond = self.velocity_predictor(z_time, time_batch, y_cond=None)  # (B*n_class, T, D)
        v_out = v_uncond.clone().view(B, n_class, *v_uncond.shape[1:])
        z_t_view = z_time.view(B, n_class, *z_time.shape[1:])
        t_view = time_batch.view(B, n_class)
        for ci, cid in enumerate(cand_class_ids):
            req_p = torch.nonzero(self.class_concepts[cid] > 0.5, as_tuple=False).squeeze(1)
            if len(req_p) == 0:
                continue
            zt_c = z_t_view[:, ci]  # (B, T, D)
            t_c = t_view[:, ci]     # (B,)
            v0_c = v_out[:, ci]
            n_p = len(req_p)
            zt_rep = zt_c.unsqueeze(1).expand(-1, n_p, -1, -1).reshape(B * n_p, *zt_c.shape[1:])
            t_rep = t_c.unsqueeze(1).expand(-1, n_p).reshape(-1)
            cond_p = self.prim_emb_centered[req_p].unsqueeze(0).expand(B, -1, -1).reshape(B * n_p, -1)
            v_p = self.velocity_predictor(zt_rep, t_rep, y_cond=cond_p).view(B, n_p, *zt_c.shape[1:])
            delta_v = (v_p - v0_c.unsqueeze(1)).mean(dim=1)
            v_out[:, ci] = v0_c + self.superpos_gamma * delta_v
        return v_out.reshape(B * n_class, *v_uncond.shape[1:])

    @torch.no_grad()
    def predict_zsl(self, skeleton_data, unseen_classes, attuned_semantics, label):
        if not self.args.use_features:
            ske = self.shiftgcn(skeleton_data).mean(3).mean(2)
        else:
            ske = skeleton_data
        B = ske.shape[0]
        ske_rep = ske.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
        z1_ske, _, _ = self.motion_encoder(ske_rep)
        z0_text, _, _ = self.semantic_encoder(attuned_semantics[unseen_classes])

        timestep = torch.ones(B, device=ske.device) * self.args.step_ratio
        class_chunk = self.args.eval_class_chunk or len(unseen_classes)
        mse_error_list = []
        for start in range(0, len(unseen_classes), class_chunk):
            z0_chunk = z0_text[start:start + class_chunk]
            cid_chunk = unseen_classes[start:start + class_chunk]
            n_class = z0_chunk.shape[0]
            z0_batch = z0_chunk.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_class, *z0_chunk.shape[1:])
            z1_batch = z1_ske.unsqueeze(1).expand(-1, n_class, -1, -1).reshape(B * n_class, *z1_ske.shape[1:])
            time_batch = timestep.unsqueeze(1).expand(-1, n_class).reshape(-1)
            z_time = self.psi(time_batch, z0_batch, z1_batch)
            pred_velocity = self._predict_class_velocity(z_time, time_batch, cid_chunk, B)
            target_velocity = self.Dt_psi(time_batch, z0_batch, z1_batch)
            mse_error_list.append(F.mse_loss(pred_velocity, target_velocity, reduction='none').mean((1, 2)).view(B, n_class))
        mse_error_list = torch.cat(mse_error_list, dim=1)
        return {'predictions_idx': torch.argmin(mse_error_list, dim=1)}

    @torch.no_grad()
    def infer_concepts_via_flow(self, skeleton_data, prim_chunk=16):
        """Zero-shot sub-action concept inference directly from DiT velocity error (-MSE)."""
        if not self.args.use_features:
            ske = self.shiftgcn(skeleton_data).mean(3).mean(2)
        else:
            ske = skeleton_data
        B = ske.shape[0]
        ske_rep = ske.unsqueeze(1).expand(-1, self.args.max_length + 1, -1)
        z1_ske, _, _ = self.motion_encoder(ske_rep)
        z0_neu = self.z0_neutral.expand(B, -1, -1)
        t_fix = torch.ones(B, device=ske.device) * self.args.step_ratio
        z_t = self.psi(t_fix, z0_neu, z1_ske)
        u_target = self.Dt_psi(t_fix, z0_neu, z1_ske)

        P = self.prim_emb_centered.shape[0]
        scores = []
        for start in range(0, P, prim_chunk):
            end = min(start + prim_chunk, P)
            n_p = end - start
            zt_rep = z_t.unsqueeze(1).expand(-1, n_p, -1, -1).reshape(B * n_p, *z_t.shape[1:])
            u_rep = u_target.unsqueeze(1).expand(-1, n_p, -1, -1).reshape(B * n_p, *u_target.shape[1:])
            t_rep = t_fix.unsqueeze(1).expand(-1, n_p).reshape(-1)
            cond = self.prim_emb_centered[start:end].unsqueeze(0).expand(B, -1, -1).reshape(B * n_p, -1)
            v_pred = self.velocity_predictor(zt_rep, t_rep, y_cond=cond)
            mse = F.mse_loss(v_pred, u_rep, reduction='none').mean(dim=(1, 2)).view(B, n_p)
            scores.append(-mse)
        return torch.cat(scores, dim=1)  # (B, P) higher = concept more likely active
