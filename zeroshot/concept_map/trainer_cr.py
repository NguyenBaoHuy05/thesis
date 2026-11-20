"""Trainer for Concept-Augmented Motion Mapping (Cách A — Direction 1).

Training pipeline:
  Stage 0: Train ConceptEncoder + CLIPConceptHead on seen 256-D skeleton features
           with Masked Pos-Weighted BCE only (~400 iters, ~4s). Freeze & calibrate
           training-set mean and scale of m_c; log unseen concept AUC/AP & Top-5 demo.
  Phase 1: Flora VAE alignment on concept-augmented motion representations.
  Phase 2: Flora DiT flow matching (pure DiT inference, no post-hoc score tuning).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from trainer import Trainer
from kg_semantics import PrimitiveKG, embed_texts
from model_cr import ConceptAugmentedMappingNet


def _binary_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sorted_s = y_score[order]
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    return float((prec * y_sorted).sum() / n_pos)


def _macro_concept_metrics(y_true_mat, y_score_mat):
    aucs, aps = [], []
    for p in range(y_true_mat.shape[1]):
        yt = y_true_mat[:, p]
        if yt.min() == yt.max():
            continue
        aucs.append(_binary_auc(yt, y_score_mat[:, p]))
        aps.append(_average_precision(yt, y_score_mat[:, p]))
    return (float(np.mean(aucs)) if aucs else float('nan'),
            float(np.mean(aps)) if aps else float('nan'),
            len(aucs))


class ConceptAugmentedTrainer(Trainer):
    """Flora Trainer with Stage-0 Concept Encoder + Motion-Side Concept Fusion."""

    def load_mapping_model(self):
        self.mapping_model = ConceptAugmentedMappingNet(args=self.args)
        self.logger.info('Loading ConceptAugmentedMappingNet (Cách A) done.')
        self.mapping_model.cuda()
        self.mapping_model.train()
        self._init_concepts()

        # Separate optimizer for Stage-0 concept branch; Flora optimizer for Phase 1 & 2
        concept_params = (
            list(self.mapping_model.concept_encoder.parameters())
            + list(self.mapping_model.concept_head.parameters())
        )
        concept_ids = {id(p) for p in concept_params}
        flora_params = [p for p in self.mapping_model.parameters() if id(p) not in concept_ids]
        concept_lr = float(getattr(self.args, 'concept_lr', 1e-3))
        self.optimizer_concept = optim.AdamW(concept_params, lr=concept_lr, weight_decay=1e-2)
        self.optimizer_mapping = optim.AdamW(
            flora_params, lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.logger.info('Loading optimizers (concept + Flora) done.')

    def _init_concepts(self):
        prims_path = getattr(self.args, 'concept_prims', 'kg/primitive_v2.json')
        rules_path = getattr(self.args, 'concept_rules', 'kg/kg_rules_v2.json')
        self.kg_concept = PrimitiveKG(prims_path, rules_path, self.num_classes, forbid_weight=0.5)
        W_req = (self.kg_concept.W > 0).astype(np.float32)
        seen_counts = W_req[self.seen_classes].sum(axis=0)
        seen_mask = (seen_counts > 0).astype(np.float32)
        n_seen = float(len(self.seen_classes))
        neg_counts = np.maximum(n_seen - seen_counts, 1.0)
        pos_w = np.clip(neg_counts / np.maximum(seen_counts, 1.0), 1.0, 10.0).astype(np.float32)

        kg_dir = os.path.dirname(os.path.abspath(prims_path))
        p_stem = os.path.splitext(os.path.basename(prims_path))[0]
        cache = os.path.join(kg_dir, f'{p_stem}_clip_L{self.args.max_length}.pt')
        prim_tok = embed_texts(
            self.mapping_model.tokenizer,
            self.mapping_model.text_encoder,
            self.kg_concept.prim_text,
            self.args.max_length,
            'cuda',
            cache,
        )
        prim_pooled = prim_tok[:, -1, :].detach()

        self.class_concepts_np = W_req
        self.seen_mask_np = seen_mask
        self.mapping_model.attach_concepts(
            torch.from_numpy(W_req),
            torch.from_numpy(seen_mask),
            torch.from_numpy(pos_w),
            prim_pooled,
        )
        m = self.mapping_model
        self.logger.info(
            f'[CA] Concepts attached: P={W_req.shape[1]}, seen-cov={int(seen_mask.sum())}/{W_req.shape[1]}, '
            f'fusion={m.fusion_mode}, rep={m.concept_rep}, gamma={m.gamma_c}'
        )

    def train_concept_stage(self):
        """Stage 0: train & freeze ConceptEncoder + CLIPConceptHead, calibrate mean/scale, evaluate."""
        n_iters = int(getattr(self.args, 'concept_iters', 400))
        m = self.mapping_model
        if m.fusion_mode == 'none' or m.gamma_c == 0.0:
            self.logger.info('[CA] fusion_mode=none / gamma=0 -> skipping Stage 0.')
            return

        m.concept_encoder.train()
        m.concept_head.train()
        avg_loss = 0.0
        for it in range(1, n_iters + 1):
            gt_motion, label_idx, _, _, _ = next(self.train_loader_iter)
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            label_idx = label_idx.to(self.device, non_blocking=True)
            targets = self.train_target_lookup[label_idx]

            loss = m.forward_concept_stage(gt_motion, self.seen_classes, targets)
            self.optimizer_concept.zero_grad()
            loss.backward()
            self.optimizer_concept.step()
            avg_loss += loss.item()
            if it % 100 == 0 or it == n_iters:
                self.logger.info(f'Stage 0 Concept. Iter {it}/{n_iters} : BCE Loss. {avg_loss / min(100, it):.5f}')
                avg_loss = 0.0

        # Freeze concept branch and calibrate training-set mean & skeleton norm scale
        m.concept_encoder.eval()
        m.concept_head.eval()
        for mod in (m.concept_encoder, m.concept_head):
            for p in mod.parameters():
                p.requires_grad = False

        mc_list, ske_norms = [], []
        with torch.no_grad():
            for gt_motion, _, _, _, _ in self.data_loader['train']:
                gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
                ske_256 = m._get_ske_256(gt_motion)
                _, _, m_c = m.infer_concept_raw(ske_256)
                mc_list.append(m_c.cpu())
                ske_norms.append(ske_256.norm(dim=-1).cpu())
        mc_all = torch.cat(mc_list, dim=0)
        m.mc_mean.copy_(mc_all.mean(dim=0, keepdim=True).to(self.device))
        m.ske_scale.copy_(torch.cat(ske_norms, dim=0).mean().to(self.device))
        self.logger.info(f'[CA] Calibrated mc_mean & ske_scale={m.ske_scale.item():.3f}.')

        self.eval_unseen_concepts()

    @torch.no_grad()
    def eval_unseen_concepts(self):
        m = self.mapping_model
        m.eval()
        probs_all, mc_all, labels_all = [], [], []
        for gt_motion, label_idx, _, _, _ in self.zsl_loader:
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            ske_256 = m._get_ske_256(gt_motion)
            probs, _, m_c = m.infer_concept_raw(ske_256)
            mc_cent = F.normalize(m_c - m.mc_mean, dim=-1)
            probs_all.append(probs.cpu().numpy())
            mc_all.append(mc_cent.cpu().numpy())
            labels_all.append(label_idx.cpu().numpy())

        probs_te = np.concatenate(probs_all, axis=0)
        mc_te = np.concatenate(mc_all, axis=0)
        y_te_cls = np.concatenate(labels_all, axis=0)
        y_te_c = self.class_concepts_np[y_te_cls]

        head_auc, head_ap, n_eval = _macro_concept_metrics(y_te_c, probs_te)
        cov_idx = np.nonzero(self.seen_mask_np > 0)[0]
        uncov_idx = np.nonzero(self.seen_mask_np == 0)[0]
        cov_auc, _, n_cov = _macro_concept_metrics(y_te_c[:, cov_idx], probs_te[:, cov_idx])
        uncov_auc, _, n_uncov = (
            _macro_concept_metrics(y_te_c[:, uncov_idx], probs_te[:, uncov_idx])
            if len(uncov_idx) > 0 else (float('nan'), float('nan'), 0)
        )

        # Concept-only ZSL via cosine to centered class concept prototypes
        protos_u = m.class_proto_clip[self.unseen_classes].cpu().numpy()  # (N_u, 768)
        scores_u = mc_te @ protos_u.T
        unseen_lookup = {int(c): i for i, c in enumerate(self.unseen_classes)}
        y_local = np.array([unseen_lookup[int(c)] for c in y_te_cls])
        c_only_acc = 100.0 * float((np.argmax(scores_u, axis=1) == y_local).mean())

        self.logger.info(
            f'[CA-EVAL] Unseen Concept Head AUC={head_auc:.4f} (AP={head_ap:.4f}, n_prims={n_eval}) | '
            f'Seen-cov AUC={cov_auc:.4f} ({n_cov}), Unseen-only AUC={uncov_auc:.4f} ({n_uncov}) | '
            f'Concept-only ZSL={c_only_acc:.2f}%'
        )

        seen_demo = set()
        prim_ids = self.kg_concept.prim_ids
        for i in range(len(y_te_cls)):
            cid = int(y_te_cls[i])
            if cid in seen_demo:
                continue
            seen_demo.add(cid)
            top5 = np.argsort(-probs_te[i])[:5]
            gt_prims = [prim_ids[p] for p in np.nonzero(y_te_c[i] > 0)[0]]
            pred_str = ', '.join(f'{prim_ids[p]}:{probs_te[i, p]:.2f}' for p in top5)
            self.logger.info(f'[CA-DEMO] class_id={cid+1} | GT={gt_prims} | Top5=[{pred_str}]')
            if len(seen_demo) >= 3:
                break
        m.train()

    def train(self):
        self.train_concept_stage()
        self.train_learning_phase()
        self.train_deciding_phase()
