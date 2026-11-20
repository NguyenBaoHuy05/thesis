"""Trainer for Group-Factorized Centered CLIPConceptHead + Differential Top-K Arbitration."""
import json
import os
import numpy as np
import torch
import torch.optim as optim

from trainer import Trainer
from kg_semantics import PrimitiveKG, embed_texts
from model_cr import ConceptArbitratedMappingNet


PART_ORDER = ('hand', 'interaction', 'body', 'arm', 'torso', 'leg', 'head')


def _load_prim_groups(prim_path):
    prims = json.load(open(prim_path, encoding='utf-8'))
    part_to_idx = {p: i for i, p in enumerate(PART_ORDER)}
    groups = [part_to_idx.get(item.get('part', 'body'), 2) for item in prims]
    return np.array(groups, dtype=np.int64), len(PART_ORDER)


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


class ConceptArbitratedTrainer(Trainer):
    """Flora Trainer with Group-Factorized Centered CLIPConceptHead & Differential Top-K Arbitration."""

    def load_mapping_model(self):
        self.mapping_model = ConceptArbitratedMappingNet(args=self.args)
        self.logger.info('Loading ConceptArbitratedMappingNet done.')
        self.mapping_model.cuda()
        self.mapping_model.train()
        self._init_concepts()

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
        self.logger.info('Loading optimizers (Flora + Group-Factorized Concept) done.')

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

        # IDF specificity weight across all classes (uses rules only, no test labels)
        df = W_req.sum(axis=0)
        idf_w = (np.log((self.num_classes + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)

        prim_groups, num_groups = _load_prim_groups(prims_path)
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
        self.class_forbids_np = W_forb
        self.seen_mask_np = seen_mask
        self.idf_w_np = idf_w
        self.mapping_model.attach_concepts(
            torch.from_numpy(W_req),
            torch.from_numpy(W_forb),
            torch.from_numpy(seen_mask),
            torch.from_numpy(pos_w),
            torch.from_numpy(idf_w),
            prim_pooled,
            torch.from_numpy(prim_groups),
            num_groups=num_groups,
        )
        m = self.mapping_model
        self.logger.info(
            f'[GF-CA] Concepts attached: P={W_req.shape[1]}, seen-cov={int(seen_mask.sum())}/{W_req.shape[1]}, '
            f'groups={num_groups} (use_groups={m.use_groups}), center_clip={m.center_clip}, '
            f'lambda_soft={m.lambda_soft}, topk={m.arb_topk}, alpha={m.arb_alpha}'
        )

    def train_concept_stage(self):
        """Train Group-Factorized Centered CLIPConceptHead and calibrate per-concept stats."""
        n_iters = int(getattr(self.args, 'concept_iters', 500))
        m = self.mapping_model
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
                self.logger.info(f'Stage 0 GF-Concept. Iter {it}/{n_iters} : Loss. {avg_loss / min(100, it):.5f}')
                avg_loss = 0.0

        m.concept_encoder.eval()
        m.concept_head.eval()
        for mod in (m.concept_encoder, m.concept_head):
            for p in mod.parameters():
                p.requires_grad = False

        # Calibrate per-primitive mean & std of similarity on the seen training set
        sim_list = []
        with torch.no_grad():
            for gt_motion, _, _, _, _ in self.data_loader['train']:
                gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
                _, sim, _ = m.infer_concepts(gt_motion)
                sim_list.append(sim.cpu())
        sim_all = torch.cat(sim_list, dim=0)
        m.sim_mean.copy_(sim_all.mean(dim=0, keepdim=True).to(self.device))
        m.sim_std.copy_(sim_all.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-4).to(self.device))
        self.logger.info('[GF-CA] Calibrated per-primitive sim_mean & sim_std on training set.')
        self.eval_unseen_concepts()

    @torch.no_grad()
    def eval_unseen_concepts(self):
        m = self.mapping_model
        m.eval()
        probs_all, zsim_all, labels_all = [], [], []
        for gt_motion, label_idx, _, _, _ in self.zsl_loader:
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            probs, _, z_sim = m.infer_concepts(gt_motion)
            probs_all.append(probs.cpu().numpy())
            zsim_all.append(z_sim.cpu().numpy())
            labels_all.append(label_idx.cpu().numpy())

        probs_te = np.concatenate(probs_all, axis=0)
        zsim_te = np.concatenate(zsim_all, axis=0)
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

        # IDF-weighted Concept-only ZSL accuracy on unseen test samples
        req_u = self.class_concepts_np[self.unseen_classes] * self.idf_w_np[None, :]
        req_u = req_u / np.maximum(req_u.sum(axis=1, keepdims=True), 1e-6)
        forb_u = self.class_forbids_np[self.unseen_classes] * self.idf_w_np[None, :]
        forb_u = forb_u / np.maximum(forb_u.sum(axis=1, keepdims=True), 1e-6)
        scores_u = zsim_te @ req_u.T - 0.3 * (zsim_te @ forb_u.T)
        unseen_lookup = {int(c): i for i, c in enumerate(self.unseen_classes)}
        y_local = np.array([unseen_lookup[int(c)] for c in y_te_cls])
        c_only_acc = 100.0 * float((np.argmax(scores_u, axis=1) == y_local).mean())

        self.logger.info(
            f'[GF-EVAL] Unseen Concept Head AUC={head_auc:.4f} (AP={head_ap:.4f}, n_prims={n_eval}) | '
            f'Seen-cov AUC={cov_auc:.4f} ({n_cov}), Unseen-only AUC={uncov_auc:.4f} ({n_uncov}) | '
            f'IDF Concept-only ZSL={c_only_acc:.2f}%'
        )

        seen_demo = set()
        prim_ids = self.kg_concept.prim_ids
        for i in range(len(y_te_cls)):
            cid = int(y_te_cls[i])
            if cid in seen_demo:
                continue
            seen_demo.add(cid)
            top5 = np.argsort(-zsim_te[i])[:5]
            gt_prims = [prim_ids[p] for p in np.nonzero(y_te_c[i] > 0)[0]]
            pred_str = ', '.join(f'{prim_ids[p]}:{probs_te[i, p]:.2f}(z={zsim_te[i, p]:+.1f})' for p in top5)
            self.logger.info(f'[GF-DEMO] class_id={cid+1} | GT={gt_prims} | Top5=[{pred_str}]')
            if len(seen_demo) >= 3:
                break
        m.train()

    def test_zsl(self, nb_iter):
        """Evaluate Differential Top-K Arbitration across alpha sweep at zero extra forward cost."""
        m = self.mapping_model
        m.eval()
        mse_all, zsim_all, labels_all = [], [], []
        for gt_motion, label_idx, _, _, _ in self.zsl_loader:
            with torch.no_grad():
                gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
                label_idx = label_idx.to(self.device, non_blocking=True)
                labels = self.zsl_target_lookup[label_idx]
                mse_fm = m.compute_fm_mse(gt_motion, self.unseen_classes, self.current_attuned_semantics())
                _, _, z_sim = m.infer_concepts(gt_motion)
                mse_all.append(mse_fm)
                zsim_all.append(z_sim)
                labels_all.append(labels)

        mse_fm = torch.cat(mse_all, dim=0)
        z_sim = torch.cat(zsim_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        diff_mat = m.build_pairwise_diff_matrix(self.unseen_classes)

        preds = m.differential_topk_arbitrate(
            mse_fm, z_sim, diff_mat, topk=m.arb_topk, alpha=m.arb_alpha
        )
        acc = 100.0 * (preds == labels).float().mean().item()
        self.best_accuracy[0] = max(self.best_accuracy[0], acc)
        self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')

        # Diagnostic sweep over alpha (with a=0.00 = pure untouched Flora DiT!)
        for a in (0.0, 0.03, 0.05, 0.08, 0.12, 0.20):
            p_a = m.differential_topk_arbitrate(mse_fm, z_sim, diff_mat, topk=m.arb_topk, alpha=a)
            acc_a = 100.0 * (p_a == labels).float().mean().item()
            self.logger.info(f'Iter {nb_iter}: [a={a:.2f}] ZSL Accuracy: {acc_a:.2f}%')
        m.train()

    def train(self):
        # 1) Train Flora Phase 1 first so Flora RNG stream matches vanilla Flora!
        self.train_learning_phase()
        # 2) Train Group-Factorized Centered CLIPConceptHead (Stage 0)
        self.train_concept_stage()
        # 3) Train Flora Phase 2 DiT with Differential Top-K Concept Arbitration at eval time
        self.train_deciding_phase()


# Backward-compatible aliases
ConceptResidualTrainer = ConceptArbitratedTrainer
ConceptAugmentedTrainer = ConceptArbitratedTrainer
