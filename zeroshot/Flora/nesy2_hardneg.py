"""Hướng 2: negative khó theo KG trong phase 2 của Flora.

Phase 2 của Flora có số hạng contrastive: đẩy vận tốc dự đoán của mẫu (x, lớp y) ra xa vận tốc của một
mẫu thuộc lớp khác, mà feeder chọn NGẪU NHIÊN (feeders/feeder_ntu.py:__getitem__).
Ở đây, với xác suất --hn-prob, mẫu đối chứng được lấy từ một lớp seen GẦN y theo KG:
  P(lớp j | y) = softmax_j( sim_KG(y, j) / tau ),  j là lớp seen khác y
sim_KG = cosine giữa hai hàng require/forbid của KG (kg_semantics.PrimitiveKG.similarity).
Các lớp chung nhiều primitive (vd. mặc/cởi áo, uống nước/ăn) trở thành negative của nhau, nên mô hình
phải học khác biệt ở mức primitive, thứ dùng chung với lớp unseen.

Đây là cách dùng KG ngược với `neighbors`: KG chọn cặp cần TÁCH RA, không phải cặp để trộn ngữ nghĩa.

Chạy: python nesy2_hardneg.py
      python nesy2_hardneg.py --hn-prob 1.0 --hn-tau 0.05
"""
import nesy_common as C

import numpy as np
import torch

from trainer import Trainer
from kg_semantics import PrimitiveKG


class HardNegTrainer(Trainer):
    def __init__(self, args):
        assert args.use_features and args.setting == 'ZSL', 'nesy2 supports ZSL on pre-extracted features'
        super().__init__(args)
        ds = self.data_loader['train'].dataset
        self.train_feats = torch.from_numpy(np.asarray(ds.data)).float().cuda()
        labels = np.asarray(ds.label)
        seen = list(self.seen_classes)
        self.seen_pos = {c: i for i, c in enumerate(seen)}
        self.seen_t = torch.tensor(seen, device='cuda')
        self.cls_index = [torch.from_numpy(np.where(labels == c)[0]).cuda() for c in seen]
        kg = PrimitiveKG(args.kg_prims, args.kg_rules, self.num_classes, args.kg_forbid_weight)
        S = kg.similarity()[np.ix_(seen, seen)].astype(np.float64)
        np.fill_diagonal(S, -np.inf)
        P = np.exp((S - S.max(1, keepdims=True)) / args.hn_tau)
        P /= P.sum(1, keepdims=True)
        self.neg_probs = torch.from_numpy(P).float().cuda()            # (n_seen, n_seen), zero diagonal
        top = np.argsort(-P, axis=1)[:, :1]
        ent = -(P * np.log(P + 1e-12)).sum(1).mean()
        ex = [f'{kg_name(kg, seen[i])} -> {kg_name(kg, seen[top[i, 0]])} ({P[i, top[i, 0]]:.2f})' for i in range(0, len(seen), max(1, len(seen) // 6))]
        self.logger.info(f'[nesy2] KG hard negatives: tau={args.hn_tau}, prob={args.hn_prob}, '
                         f'mean entropy {ent:.2f} nats (uniform = {np.log(len(seen) - 1):.2f}); e.g. ' + ' | '.join(ex))

    def hard_negatives(self, label_idx, x_c, y_c):
        """Replace a fraction hn_prob of the feeder's random contrastive samples by KG-close ones."""
        B = label_idx.shape[0]
        use = torch.rand(B, device='cuda') < self.args.hn_prob
        if not use.any():
            return x_c, y_c
        rows = torch.tensor([self.seen_pos[int(l)] for l in label_idx], device='cuda')
        neg_local = torch.multinomial(self.neg_probs[rows], 1).squeeze(1)
        x_c, y_c = x_c.clone(), y_c.clone()
        for b in torch.nonzero(use).flatten().tolist():
            j = int(neg_local[b])
            pool = self.cls_index[j]
            x_c[b] = self.train_feats[pool[torch.randint(0, pool.numel(), (1,), device='cuda')]].squeeze(0)
            y_c[b] = self.seen_t[j]
        return x_c, y_c

    def train_deciding_phase(self):
        # Flora's phase 2 loop (trainer.py), with the contrastive sample replaced by a KG hard negative
        avg_loss, avg_fm, avg_fm_con = 0., 0., 0.
        for nb_iter in range(self.args.align_iter + 1, self.args.total_iter + 1):
            gt_motion, label_idx, gt_motion_contrastive, label_idx_contrastive, _ = next(self.train_loader_iter)
            gt_motion = gt_motion.cuda().float()
            gt_motion_contrastive = gt_motion_contrastive.cuda().float()
            label_idx = label_idx.cuda()
            label_idx_contrastive = label_idx_contrastive.cuda()
            gt_motion_contrastive, label_idx_contrastive = self.hard_negatives(label_idx, gt_motion_contrastive, label_idx_contrastive)
            true_label_array = torch.tensor([self.train_label_dict[l.item()] for l in label_idx]).cuda()
            true_label_array_contrastive = torch.tensor([self.train_label_dict[l.item()] for l in label_idx_contrastive]).cuda()
            metrics = self.mapping_model.forward_phase2(gt_motion, gt_motion_contrastive, torch.argmax(true_label_array, dim=1),
                                                        torch.argmax(true_label_array_contrastive, dim=1), self.seen_classes, self.attuned_semantics)
            loss = metrics['loss']
            self.optimizer_mapping.zero_grad()
            loss.backward()
            self.optimizer_mapping.step()
            avg_loss += loss.item()
            avg_fm += metrics['loss_flow_matching'].item()
            avg_fm_con += metrics['loss_flow_matching_contrastive'].item()
            if nb_iter % self.args.print_iter == 0:
                n = self.args.print_iter
                self.logger.info(f"Train Phase 3. Iter {nb_iter} : Loss. {avg_loss / n:.5f} \t Loss FM. {avg_fm / n:.5f} \t Loss FM Con. {avg_fm_con / n:.5f}")
                avg_loss, avg_fm, avg_fm_con = 0., 0., 0.
            if nb_iter % self.args.eval_iter == 0:
                self.test_zsl(nb_iter)
        self.print_best_accuracy()


def kg_name(kg, c):
    return kg_name.names[c] if hasattr(kg_name, 'names') else str(c)


def _load_names():
    import json
    import os
    with open(os.path.join(C.HERE, 'kg', 'kg_rules_v2.json'), encoding='utf-8') as f:
        kg_name.names = [r['class'] for r in json.load(f)['rules']]


_load_names()

if __name__ == '__main__':
    C.main('hardneg', HardNegTrainer,
           method_args=[
               ('--hn-prob', dict(type=float, default=0.5, help='fraction of contrastive samples replaced by a KG hard negative')),
               ('--hn-tau', dict(type=float, default=0.1, help='softmax temperature over KG similarity')),
           ],
           label_keys=[('hn_prob', 'p'), ('hn_tau', 't')])
