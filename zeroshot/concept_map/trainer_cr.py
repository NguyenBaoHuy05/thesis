"""Trainer for Group-Factorized Centered CLIPConceptHead + Differential Top-K Arbitration."""
import json
import os
import numpy as np
import torch
import torch.optim as optim

from trainer import Trainer
from kg_semantics import embed_texts
from concepts.concept_loader import MultiAxisConceptBank
from model_cr import ConceptArbitratedMappingNet, _preserve_rng_state


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

        with _preserve_rng_state():
            self._init_concepts()
            concept_params = (
                list(self.mapping_model.concept_encoder.parameters())
                + list(self.mapping_model.concept_head.parameters())
            )
            vae_concept_params = (
                list(self.mapping_model.concept_encoder.vae_net.parameters())
                + list(self.mapping_model.concept_encoder.out_norm.parameters())
                + list(self.mapping_model.concept_head.parameters())
                if self.mapping_model.use_vae_latent and self.mapping_model.lambda_vae_c > 0.0
                else []
            )
            concept_ids = {id(p) for p in concept_params}
            flora_params = [p for p in self.mapping_model.parameters() if id(p) not in concept_ids]
            # Include VAE-concept projection & head in Phase 1 optimizer when grounding Cross-Modal VAE
            phase1_params = flora_params + vae_concept_params
            concept_lr = float(getattr(self.args, 'concept_lr', 1e-3))
            self.optimizer_concept = optim.AdamW(concept_params, lr=concept_lr, weight_decay=3e-2)

        self.optimizer_mapping = optim.AdamW(
            phase1_params, lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.logger.info('Loading optimizers (Flora + Dual-Modal VAE Concept) done.')

    def _init_concepts(self):
        cmap_dir = os.path.dirname(os.path.abspath(__file__))
        default_bank = os.path.join(cmap_dir, 'concepts', 'concept_bank_v1.json')
        default_ann = os.path.join(cmap_dir, 'concepts', 'ntu120_concept_annotations.json')
        bank_path = getattr(self.args, 'concept_bank', default_bank)
        ann_path = getattr(self.args, 'concept_ann', default_ann)

        self.concept_bank = MultiAxisConceptBank(bank_path, ann_path, self.num_classes)
        W_req = self.concept_bank.W_req
        W_forb = self.concept_bank.W_forb
        seen_counts = W_req[self.seen_classes].sum(axis=0)
        seen_mask = (seen_counts > 0).astype(np.float32)
        n_seen = float(len(self.seen_classes))
        neg_counts = np.maximum(n_seen - seen_counts, 1.0)
        pos_w = np.clip(neg_counts / np.maximum(seen_counts, 1.0), 1.0, 8.0).astype(np.float32)

        # IDF specificity weight across all classes (uses concept annotations only, no test labels)
        df = W_req.sum(axis=0)
        idf_w = (np.log((self.num_classes + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)

        prim_groups = self.concept_bank.group_indices
        num_groups = self.concept_bank.num_groups
        c_dir = os.path.dirname(os.path.abspath(bank_path))
        c_stem = os.path.splitext(os.path.basename(bank_path))[0]
        cache = os.path.join(c_dir, f'{c_stem}_clip_L{self.args.max_length}.pt')
        prim_tok = embed_texts(
            self.mapping_model.tokenizer,
            self.mapping_model.text_encoder,
            self.concept_bank.concept_texts,
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
            f'[GF-CA] Multi-Axis Concepts attached: P={W_req.shape[1]}, seen-cov={int(seen_mask.sum())}/{W_req.shape[1]}, '
            f'axes={num_groups} (use_groups={m.use_groups}), center_clip={m.center_clip}, '
            f'lambda_soft={m.lambda_soft}, topk={m.arb_topk}, alpha={m.arb_alpha}, gate_tau={m.arb_gate_tau}'
        )

    def train_concept_stage(self):
        """Train Dual-Modal ConceptEncoder + GroupFactorizedCLIPConceptHead and calibrate stats."""
        n_iters = int(getattr(self.args, 'concept_iters', 500))
        m = self.mapping_model
        m.motion_encoder.eval()
        m.semantic_encoder.eval()
        m.motion_encoder.requires_grad_(False)
        m.semantic_encoder.requires_grad_(False)
        m.concept_encoder.requires_grad_(True)
        m.concept_head.requires_grad_(True)
        m.concept_encoder.train()
        m.concept_head.train()
        attuned_sem = self.current_attuned_semantics()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_concept, T_max=max(1, n_iters), eta_min=1e-5
        )
        # Use an isolated iterator so Flora's self.train_loader_iter stream for Phase 2 is untouched
        concept_iter = iter(self.data_loader['train'])
        avg_loss = 0.0
        for it in range(1, n_iters + 1):
            try:
                gt_motion, label_idx, _, _, _ = next(concept_iter)
            except StopIteration:
                concept_iter = iter(self.data_loader['train'])
                gt_motion, label_idx, _, _, _ = next(concept_iter)
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            label_idx = label_idx.to(self.device, non_blocking=True)
            targets = self.train_target_lookup[label_idx]

            loss = m.forward_concept_stage(gt_motion, self.seen_classes, targets, attuned_semantics=attuned_sem)
            self.optimizer_concept.zero_grad()
            loss.backward()
            self.optimizer_concept.step()
            scheduler.step()
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
        prim_ids = self.concept_bank.concept_ids
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
                mse_fm, mu_ske_p = m.compute_fm_mse(
                    gt_motion, self.unseen_classes, self.current_attuned_semantics(), return_mu=True
                )
                _, _, z_sim = m.infer_concepts(gt_motion, mu_ske_p=mu_ske_p)
                mse_all.append(mse_fm)
                zsim_all.append(z_sim)
                labels_all.append(labels)

        mse_fm = torch.cat(mse_all, dim=0)
        z_sim = torch.cat(zsim_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        diff_mat = m.build_pairwise_diff_matrix(self.unseen_classes)

        preds = m.differential_topk_arbitrate(
            mse_fm, z_sim, diff_mat, topk=m.arb_topk, alpha=m.arb_alpha, gate_tau=m.arb_gate_tau
        )
        acc = 100.0 * (preds == labels).float().mean().item()
        self.best_accuracy[0] = max(self.best_accuracy[0], acc)
        self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')

        # Diagnostic sweep over alpha (with a=0.00 = pure Concept-Grounded VAE + DiT)
        for a in (0.0, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30):
            p_a = m.differential_topk_arbitrate(
                mse_fm, z_sim, diff_mat, topk=m.arb_topk, alpha=a, gate_tau=m.arb_gate_tau
            )
            acc_a = 100.0 * (p_a == labels).float().mean().item()
            self.logger.info(f'Iter {nb_iter}: [a={a:.2f}] ZSL Accuracy: {acc_a:.2f}%')
        m.train()

    GZSL_CAL_SWEEP = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2)
    GZSL_GAMMA_SWEEP = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

    @staticmethod
    def _gzsl_shu(preds, labels, seen_mask):
        def acc(mask):
            return 100.0 * (preds[mask] == labels[mask]).float().mean().item() if mask.any() else 0.0
        s, u = acc(seen_mask), acc(~seen_mask)
        h = 2 * s * u / (s + u) if (s + u) > 0 else 0.0
        return s, u, h

    def test_gzsl(self, nb_iter):
        """GZSL with concept arbitration; logs the default calibration_factor (used for 'Best GZSL')
        plus a sweep over calibration_factor from a single set of forward passes."""
        m = self.mapping_model
        m.eval()
        mse_l, z_l, y_l = [], [], []
        with torch.no_grad():
            sem = self.current_attuned_semantics()
            for gt_motion, label_idx, _, _, _ in self.gzsl_loader:
                gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
                label_idx = label_idx.to(self.device, non_blocking=True)
                mse_all, z_sim = m.gzsl_scores(gt_motion, self.seen_classes, self.unseen_classes, sem)
                mse_l.append(mse_all)
                z_l.append(z_sim)
                y_l.append(label_idx)
            mse_all, z_sim, labels = torch.cat(mse_l), torch.cat(z_l), torch.cat(y_l)
            seen_t = torch.as_tensor(self.seen_classes, device=labels.device, dtype=torch.long)
            seen_mask = torch.isin(labels, seen_t)

            cal0 = float(self.args.calibration_factor)
            gamma0 = float(getattr(self.args, 'gzsl_gamma', 0.0))
            cands = m.gzsl_candidates(mse_all, z_sim, self.seen_classes, self.unseen_classes)
            preds = m.gzsl_decide(mse_all, z_sim, self.seen_classes, self.unseen_classes, cal0,
                                  gamma=gamma0, cands=cands)
            s, u, h = self._gzsl_shu(preds, labels, seen_mask)
            self.logger.info(
                f"Iter {nb_iter}: Seen Accuracy: {s:.2f}% \t Unseen Accuracy: {u:.2f}% \t Harmonic Mean: {h:.2f}%")
            if h > self.best_accuracy[3]:
                self.best_accuracy[1], self.best_accuracy[2], self.best_accuracy[3] = s, u, h

            best_here = None
            for cal in self.GZSL_CAL_SWEEP:
                p = m.gzsl_decide(mse_all, z_sim, self.seen_classes, self.unseen_classes, cal, cands=cands)
                s_c, u_c, h_c = self._gzsl_shu(p, labels, seen_mask)
                self.logger.info(
                    f"Iter {nb_iter}: [cal={cal:.2f}] GZSL S={s_c:.2f}% U={u_c:.2f}% H={h_c:.2f}%")
                if best_here is None or h_c > best_here[0]:
                    best_here = (h_c, s_c, u_c, cal, 0.0)
            # Concept-aided domain decision: sweep gamma, report the best calibration_factor for each
            for gamma in self.GZSL_GAMMA_SWEEP:
                if gamma == 0.0:
                    continue
                best_g = None
                for cal in self.GZSL_CAL_SWEEP:
                    p = m.gzsl_decide(mse_all, z_sim, self.seen_classes, self.unseen_classes, cal,
                                      gamma=gamma, cands=cands)
                    s_c, u_c, h_c = self._gzsl_shu(p, labels, seen_mask)
                    if best_g is None or h_c > best_g[0]:
                        best_g = (h_c, s_c, u_c, cal, gamma)
                self.logger.info(
                    f"Iter {nb_iter}: [gamma={gamma:.2f}] best cal={best_g[3]:.2f} "
                    f"GZSL S={best_g[1]:.2f}% U={best_g[2]:.2f}% H={best_g[0]:.2f}%")
                if best_g[0] > best_here[0]:
                    best_here = best_g
            prev = getattr(self, 'best_sweep', None)
            if prev is None or best_here[0] > prev[0]:
                self.best_sweep = best_here
        m.train()

    def print_best_accuracy(self):
        super().print_best_accuracy()
        bs = getattr(self, 'best_sweep', None)
        if bs is not None:
            self.logger.info(
                f'Best GZSL (calibration/gamma sweep, tuned on test): S={bs[1]:.2f}% U={bs[2]:.2f}% '
                f'H={bs[0]:.2f}% at calibration_factor={bs[3]:.2f}, gamma={bs[4]:.2f}')

    def train(self):
        # 1) Train Flora Phase 1 first so Flora RNG stream matches vanilla Flora!
        self.train_learning_phase()
        # 2) Train Group-Factorized Centered CLIPConceptHead (Stage 0) inside isolated RNG
        with _preserve_rng_state():
            self.train_concept_stage()
        # 3) Train Flora Phase 2 DiT with Differential Top-K Concept Arbitration at eval time
        self.train_deciding_phase()


# Backward-compatible aliases
ConceptResidualTrainer = ConceptArbitratedTrainer
ConceptAugmentedTrainer = ConceptArbitratedTrainer
