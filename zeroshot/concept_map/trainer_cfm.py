"""Trainer for Compositional / Concept-Conditioned Flow Matching (Direction 2)."""
import os
import numpy as np
import torch
import torch.optim as optim

from trainer import Trainer
from kg_semantics import PrimitiveKG, embed_texts
from model_cfm import CompositionalFMMappingNet
from trainer_cr import _macro_concept_metrics


class CompositionalFMTrainer(Trainer):
    """Flora Trainer with Concept-Conditioned / Compositional Flow Matching in Phase 2."""

    def load_mapping_model(self):
        self.mapping_model = CompositionalFMMappingNet(args=self.args)
        self.logger.info('Loading CompositionalFMMappingNet (Direction 2) done.')
        self.mapping_model.cuda()
        self.mapping_model.train()
        self._init_concepts()
        self.optimizer_mapping = optim.AdamW(
            self.mapping_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.logger.info('Loading CompositionalFM optimizer done.')

    def _init_concepts(self):
        prims_path = getattr(self.args, 'concept_prims', 'kg/primitive_v2.json')
        rules_path = getattr(self.args, 'concept_rules', 'kg/kg_rules_v2.json')
        self.kg_concept = PrimitiveKG(prims_path, rules_path, self.num_classes, forbid_weight=0.5)
        W_req = (self.kg_concept.W > 0).astype(np.float32)
        seen_mask = (W_req[self.seen_classes].sum(axis=0) > 0).astype(np.float32)

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
            prim_pooled,
        )
        m = self.mapping_model
        self.logger.info(
            f'[CFM] Concepts attached: P={W_req.shape[1]}, seen-cov={int(seen_mask.sum())}/{W_req.shape[1]}, '
            f'mode={m.cfm_mode}, lambda_sub={m.lambda_sub}, margin={m.sub_margin}, '
            f'cond_drop={m.cond_drop}, gamma={m.superpos_gamma}'
        )

    def train_learning_phase(self):
        # Phase 1 is 100% pure Flora VAE alignment
        super().train_learning_phase()
        self.mapping_model.cache_neutral_anchor(self.current_attuned_semantics(), self.seen_classes)
        self.logger.info('[CFM] Cached neutral semantic anchor z0_bar for sub-action concept flow.')

    def train_deciding_phase(self):
        avg_loss = avg_fm = avg_con = avg_sub = 0.0
        for nb_iter in range(self.args.align_iter + 1, self.args.total_iter + 1):
            gt_motion, label_idx, gt_motion_con, label_idx_con, _ = next(self.train_loader_iter)
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            gt_motion_con = gt_motion_con.to(self.device, dtype=torch.float32, non_blocking=True)
            label_idx = label_idx.to(self.device, non_blocking=True)
            label_idx_con = label_idx_con.to(self.device, non_blocking=True)
            targets = self.train_target_lookup[label_idx]
            con_targets = self.train_target_lookup[label_idx_con]

            metrics = self.mapping_model.forward_phase2(
                gt_motion, gt_motion_con, targets, con_targets,
                self.seen_classes, self.current_attuned_semantics()
            )
            loss = metrics['loss']
            self.optimizer_mapping.zero_grad()
            loss.backward()
            self.optimizer_mapping.step()

            avg_loss += loss.item()
            avg_fm += metrics['loss_flow_matching'].item()
            avg_con += metrics['loss_flow_matching_contrastive'].item()
            avg_sub += metrics['loss_sub'].item()
            if nb_iter % self.args.print_iter == 0:
                n = float(self.args.print_iter)
                self.logger.info(
                    f'Train Phase 2. Iter {nb_iter} : Loss. {avg_loss/n:.5f} '
                    f'\t Loss FM. {avg_fm/n:.5f} \t Loss FM Con. {avg_con/n:.5f} '
                    f'\t Loss Sub. {avg_sub/n:.5f}'
                )
                avg_loss = avg_fm = avg_con = avg_sub = 0.0

            if nb_iter % self.args.eval_iter == 0:
                if self.args.setting == 'ZSL':
                    self.test_zsl(nb_iter)
                elif self.args.setting == 'GZSL':
                    self.test_gzsl(nb_iter)

        # Evaluate zero-shot sub-action concept inference directly from DiT velocity errors
        if self.mapping_model.cfm_mode != 'none':
            self.eval_flow_concepts()
        self.print_best_accuracy()

    @torch.no_grad()
    def eval_flow_concepts(self):
        self.mapping_model.eval()
        scores_all, labels_all = [], []
        for gt_motion, label_idx, _, _, _ in self.zsl_loader:
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            sc = self.mapping_model.infer_concepts_via_flow(gt_motion)
            scores_all.append(sc.cpu().numpy())
            labels_all.append(label_idx.cpu().numpy())

        scores_te = np.concatenate(scores_all, axis=0)  # (N_test, P)
        y_te_cls = np.concatenate(labels_all, axis=0)
        y_te_c = self.class_concepts_np[y_te_cls]

        auc_all, ap_all, n_eval = _macro_concept_metrics(y_te_c, scores_te)
        cov_idx = np.nonzero(self.seen_mask_np > 0)[0]
        uncov_idx = np.nonzero(self.seen_mask_np == 0)[0]
        cov_auc, _, n_cov = _macro_concept_metrics(y_te_c[:, cov_idx], scores_te[:, cov_idx])
        uncov_auc, _, n_uncov = (
            _macro_concept_metrics(y_te_c[:, uncov_idx], scores_te[:, uncov_idx])
            if len(uncov_idx) > 0 else (float('nan'), float('nan'), 0)
        )
        self.logger.info(
            f'[CFM-EVAL] Flow Concept AUC={auc_all:.4f} (AP={ap_all:.4f}, n_prims={n_eval}) | '
            f'Seen-cov AUC={cov_auc:.4f} ({n_cov}), Unseen-only AUC={uncov_auc:.4f} ({n_uncov})'
        )

        # Demo top-5 concepts inferred purely via DiT velocity errors
        seen_demo = set()
        prim_ids = self.kg_concept.prim_ids
        for i in range(len(y_te_cls)):
            cid = int(y_te_cls[i])
            if cid in seen_demo:
                continue
            seen_demo.add(cid)
            top5 = np.argsort(-scores_te[i])[:5]
            gt_prims = [prim_ids[p] for p in np.nonzero(y_te_c[i] > 0)[0]]
            pred_str = ', '.join(f'{prim_ids[p]}:{scores_te[i, p]:.3f}' for p in top5)
            self.logger.info(f'[CFM-DEMO] class_id={cid+1} | GT={gt_prims} | Top5=[{pred_str}]')
            if len(seen_demo) >= 3:
                break
        self.mapping_model.train()
