"""Trainer for Decoupled Dual-Branch Concept-Residual Mapping (Direction 1).

Logs:
  1. Phase-1 Flora losses + Concept BCE + Concept Prototype alignment + GRL Adversarial loss.
  2. Post-Phase-1 unseen concept metrics:
       - Head AUC / AP (all unseen, seen-covered, unseen-only)
       - Concept-only ZSL Accuracy (from the concept branch alone, before Phase 2!)
       - Disentanglement Leakage Check: Ridge probe AUC on z_c vs Residual mu_res
       - Top-5 qualitative predicted sub-action concepts on 3 unseen classes
  3. Phase-2 Dual-Path ZSL accuracy (Residual-only, Concept+Residual at concept_alpha,
     plus diagnostic sweep over alpha in {0.0, 0.1, 0.2, 0.3, 0.5} at zero extra training cost).
"""
import os
import numpy as np
import torch
import torch.optim as optim

from trainer import Trainer
from kg_semantics import PrimitiveKG, embed_texts
from model_cr import ConceptResidualMappingNet, _zscore_rows


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


def _ridge_probe_auc(x_tr, y_tr, x_te, y_te, reg=10.0):
    if x_tr.shape[1] == 0:
        return float('nan')
    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True) + 1e-6
    xtr = (x_tr - mean) / std
    xte = (x_te - mean) / std
    xtr_b = np.concatenate([xtr, np.ones((xtr.shape[0], 1), dtype=np.float32)], axis=1)
    xte_b = np.concatenate([xte, np.ones((xte.shape[0], 1), dtype=np.float32)], axis=1)
    d = xtr_b.shape[1]
    eye = np.eye(d, dtype=np.float64)
    eye[-1, -1] = 0.0
    w = np.linalg.solve(xtr_b.T @ xtr_b + reg * eye, xtr_b.T @ y_tr)
    pred = xte_b @ w
    auc, _, _ = _macro_concept_metrics(y_te, pred)
    return auc


class ConceptResidualTrainer(Trainer):
    """Flora Trainer with decoupled Concept-Residual MappingNet and Dual-Path ZSL evaluation."""

    def load_mapping_model(self):
        self.mapping_model = ConceptResidualMappingNet(args=self.args)
        self.logger.info('Loading ConceptResidualMappingNet (decoupled dual-branch) done.')
        self.mapping_model.cuda()
        self.mapping_model.train()
        self._init_concepts()
        self.optimizer_mapping = optim.AdamW(
            self.mapping_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.logger.info('Loading ConceptResidual optimizer done.')

    def _init_concepts(self):
        prims_path = getattr(self.args, 'concept_prims', 'kg/primitive_v2.json')
        rules_path = getattr(self.args, 'concept_rules', 'kg/kg_rules_v2.json')
        self.kg_concept = PrimitiveKG(prims_path, rules_path, self.num_classes, forbid_weight=0.5)
        W_req = (self.kg_concept.W > 0).astype(np.float32)
        W_forb = (self.kg_concept.W < 0).astype(np.float32)
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
            torch.from_numpy(W_forb),
            torch.from_numpy(seen_mask),
            torch.from_numpy(pos_w),
            prim_pooled,
        )
        m = self.mapping_model
        self.logger.info(
            f'[CR] Concepts attached: P={W_req.shape[1]}, seen-covered={int(seen_mask.sum())}/{W_req.shape[1]}, '
            f'D_c={m.concept_dim}, head={m.concept_head_type}, score={m.concept_score_mode}, '
            f'lambda_c={m.lambda_concept}, lambda_proto={m.lambda_proto}, lambda_adv={m.lambda_adv}, '
            f'alpha={m.concept_alpha}'
        )

    def train_learning_phase(self):
        avg = dict(loss=0.0, mu=0.0, rec=0.0, c=0.0, proto=0.0, adv=0.0)
        for nb_iter in range(1, self.args.align_iter + 1):
            gt_motion, label_idx, _, _, _ = next(self.train_loader_iter)
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            label_idx = label_idx.to(self.device, non_blocking=True)
            targets = self.train_target_lookup[label_idx]
            metrics = self.mapping_model.forward_phase1(
                gt_motion, self.current_attuned_semantics(), self.seen_classes, targets
            )
            loss = metrics['loss']
            self.optimizer_mapping.zero_grad()
            loss.backward()
            self.optimizer_mapping.step()

            avg['loss'] += loss.item()
            avg['mu'] += metrics['loss_mu'].item()
            avg['rec'] += metrics['loss_recon'].item()
            avg['c'] += metrics['loss_concept'].item()
            avg['proto'] += metrics['loss_proto'].item()
            avg['adv'] += metrics['loss_adv'].item()
            if nb_iter % self.args.print_iter == 0:
                n = float(self.args.print_iter)
                self.logger.info(
                    f'Train Phase 1. Iter {nb_iter} : Loss. {avg["loss"]/n:.5f} '
                    f'\t Recon. {avg["rec"]/n:.5f} \t Concept. {avg["c"]/n:.5f} '
                    f'\t Proto. {avg["proto"]/n:.5f} \t Adv. {avg["adv"]/n:.5f}'
                )
                for k in avg:
                    avg[k] = 0.0
        self.logger.info('Phase 1: Decoupled Concept-Residual Learning Done.')
        self.eval_concepts_and_leakage()

    @torch.no_grad()
    def _collect_loader_latents(self, loader):
        probs_all, zc_all, zr_all, labels_all, cscore_all = [], [], [], [], []
        for gt_motion, label_idx, _, _, _ in loader:
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            probs, _, zc, zr = self.mapping_model.extract_concept_latents(gt_motion)
            cscore = self.mapping_model.compute_concept_class_scores(gt_motion, self.unseen_classes)
            probs_all.append(probs.cpu().numpy())
            zc_all.append(zc.cpu().numpy())
            zr_all.append(zr.cpu().numpy())
            labels_all.append(label_idx.cpu().numpy())
            cscore_all.append(cscore.cpu().numpy())
        return (
            np.concatenate(probs_all, axis=0),
            np.concatenate(zc_all, axis=0),
            np.concatenate(zr_all, axis=0),
            np.concatenate(labels_all, axis=0),
            np.concatenate(cscore_all, axis=0),
        )

    def eval_concepts_and_leakage(self):
        self.mapping_model.eval()
        _, zc_tr, zr_tr, y_tr_cls, _ = self._collect_loader_latents(self.data_loader['train'])
        probs_te, zc_te, zr_te, y_te_cls, cscore_te = self._collect_loader_latents(self.zsl_loader)

        y_tr_c = self.class_concepts_np[y_tr_cls]
        y_te_c = self.class_concepts_np[y_te_cls]

        head_auc, head_ap, n_eval = _macro_concept_metrics(y_te_c, probs_te)
        cov_idx = np.nonzero(self.seen_mask_np > 0)[0]
        uncov_idx = np.nonzero(self.seen_mask_np == 0)[0]
        cov_auc, _, n_cov = _macro_concept_metrics(y_te_c[:, cov_idx], probs_te[:, cov_idx])
        uncov_auc, _, n_uncov = (
            _macro_concept_metrics(y_te_c[:, uncov_idx], probs_te[:, uncov_idx])
            if len(uncov_idx) > 0 else (float('nan'), float('nan'), 0)
        )

        # Concept-only ZSL accuracy (before Phase 2)
        unseen_lookup = {int(c): i for i, c in enumerate(self.unseen_classes)}
        y_te_local = np.array([unseen_lookup[int(c)] for c in y_te_cls])
        concept_only_acc = 100.0 * float((np.argmax(cscore_te, axis=1) == y_te_local).mean())

        # Disentanglement check: Ridge probe on z_c vs Residual mu_res
        probe_zc_auc = _ridge_probe_auc(zc_tr, y_tr_c, zc_te, y_te_c)
        probe_zr_auc = _ridge_probe_auc(zr_tr, y_tr_c, zr_te, y_te_c)

        self.logger.info(
            f'[CR-EVAL] Unseen Concept Head AUC={head_auc:.4f} (AP={head_ap:.4f}, n_prims={n_eval}) | '
            f'Seen-cov AUC={cov_auc:.4f} ({n_cov}), Unseen-only AUC={uncov_auc:.4f} ({n_uncov}) | '
            f'Concept-only ZSL={concept_only_acc:.2f}% | '
            f'Probe(z_c) AUC={probe_zc_auc:.4f}, Leakage Probe(z_r) AUC={probe_zr_auc:.4f}'
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
            self.logger.info(f'[CR-DEMO] class_id={cid+1} | GT={gt_prims} | Top5=[{pred_str}]')
            if len(seen_demo) >= 3:
                break

        self.mapping_model.train()

    def test_zsl(self, nb_iter):
        """Evaluate main ZSL (at configured concept_alpha) plus diagnostic alpha sweep."""
        self.mapping_model.eval()
        s_fm_all, s_c_all, labels_all = [], [], []
        for gt_motion, label_idx, _, _, _ in self.zsl_loader:
            with torch.no_grad():
                gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
                label_idx = label_idx.to(self.device, non_blocking=True)
                labels = self.zsl_target_lookup[label_idx]
                s_fm, s_c = self.mapping_model.predict_zsl_scores(
                    gt_motion, self.unseen_classes, self.current_attuned_semantics()
                )
                s_fm_all.append(s_fm)
                s_c_all.append(s_c)
                labels_all.append(labels)

        s_fm = torch.cat(s_fm_all, dim=0)
        s_c = torch.cat(s_c_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        z_fm = _zscore_rows(s_fm)
        z_c = _zscore_rows(s_c)

        alpha = self.mapping_model.concept_alpha
        scores = s_fm if alpha == 0.0 else (z_fm + alpha * z_c)
        acc = 100.0 * (torch.argmax(scores, dim=1) == labels).float().mean().item()
        self.best_accuracy[0] = max(self.best_accuracy[0], acc)
        self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')

        # Diagnostic sweep over alpha at zero extra forward cost
        if self.mapping_model.lambda_concept > 0.0:
            for a in (0.0, 0.1, 0.25, 0.5):
                sc_a = s_fm if a == 0.0 else (z_fm + a * z_c)
                acc_a = 100.0 * (torch.argmax(sc_a, dim=1) == labels).float().mean().item()
                self.logger.info(f'Iter {nb_iter}: [a={a:.2f}] ZSL Accuracy: {acc_a:.2f}%')
        self.mapping_model.train()
