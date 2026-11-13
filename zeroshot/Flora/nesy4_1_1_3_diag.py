"""Hướng 4.1.1.3 (cải tiến từ nesy4_1_1_fuse.py): chẩn đoán lỗi của DiT trên lớp unseen. Chỉ ghi log, không đổi điểm.

Câu hỏi: khi DiT phân vân giữa hai lớp, KG có còn dư địa để sửa không (ý tưởng "vòng lặp chấm lại theo margin")?
Mỗi lần đánh giá ZSL tính ma trận MSE (mẫu x lớp unseen) đúng như models/fm.py predict_zsl, lấy argmin làm dự đoán
(nên top-1 là chính điểm mà runner báo; encoder lấy mẫu ngẫu nhiên nên KHÔNG đánh giá hai lần), rồi đo:

  1. top-1 / top-2 / top-3. top-2 trừ top-1 là trần của việc chấm lại.
  2. Margin = (MSE lớp 2 - MSE lớp 1) / MSE lớp 1: độ chính xác theo 4 nhóm margin (nhỏ -> lớn).
  3. Cặp lớp bị nhầm nhiều nhất (đúng -> đoán), kèm cờ cặp đảo chiều và độ giống z-score theo text và theo KG.
  4. Với các mẫu bị đoán sai: tỉ lệ (text nói giống hơn KG), d = z(text) - z(KG) trung bình; so với mức nền của mọi cặp unseen.
     d > 0 nghĩa là KG nói hai lớp khác nhau hơn text nói -> KG có thông tin để sửa.
  5. Các cặp đảo chiều chạm lớp unseen: độ chính xác từng lớp unseen; nếu lớp đối xứng cũng unseen thì tỉ lệ bị đoán thành lớp đó.
     (Nếu lớp đối xứng là lớp seen thì ZSL không bao giờ đoán nó, nên không có nhầm lẫn trong cặp.)

Log `[diag]` được ghi tại lần đánh giá cuối (tag last) và tại lần đánh giá có điểm cao nhất (tag best, ghi sau khi huấn luyện xong).

Tiêu chí quyết định (đặt trước khi xem số): vòng lặp chấm lại chỉ đáng làm khi
  (a) top-2 - top-1 >= ~10 điểm, (b) tỉ lệ sai của nhóm margin nhỏ nhất >= 2 lần nhóm lớn nhất,
  (c) >= 50% mẫu sai có d > 0.

Chạy: python nesy4_1_1_3_diag.py --tasks 48 110 96 --seeds 1 2 3 --no-baselines
"""
import json
import os

import nesy_common as C

import numpy as np

from kg_semantics import PrimitiveKG
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_forbid import order_only_pairs
from nesy4_1_1_fuse import FuseTrainer


class DiagTrainer(FuseTrainer):
    def load_prototypes(self):
        super().load_prototypes()
        import torch
        a = self.args
        if a.setting != 'ZSL':
            raise ValueError('nesy4_1_1_3 chỉ hỗ trợ --setting ZSL')
        n = self.text_similarity.shape[0]
        kg = PrimitiveKG(a.kg_prims, a.kg_rules, n, a.kg_forbid_weight)
        off = ~torch.eye(n, dtype=torch.bool)
        z = lambda m: (m - m[off].mean()) / m[off].std().clamp_min(1e-8)
        ts = self.text_similarity.detach().cpu().float()
        ks = torch.from_numpy(kg.similarity()).float()
        self._zt = z(ts).numpy()
        self._zk = z(ks).numpy()
        rules = json.load(open(a.kg_rules, encoding='utf-8'))['rules']
        self._names = {r['id'] - 1: r.get('class', str(r['id'])) for r in rules}
        self._unseen = np.array([int(u) for u in self.unseen_classes])
        self._pairs = order_only_pairs(a.kg_rules, n)
        self._best_diag = None          # (accuracy, iter, lines)

    # ---------------------------------------------------------------- scores (same computation as predict_zsl)
    def _scores(self, x):
        import torch
        a = self.args
        m = self.mapping_model
        with torch.no_grad():
            if not a.use_features:
                sk = m.shiftgcn(x)
                sk = sk.mean(3).mean(2)
            else:
                sk = x
            B = sk.shape[0]
            sk = sk.unsqueeze(1).expand(-1, a.max_length + 1, -1)
            z1, _, _ = m.motion_encoder(sk)
            sem = self.current_attuned_semantics()[self.unseen_classes]
            z0, _, _ = m.semantic_encoder(sem)
            t = torch.ones(B, device=sk.device) * a.step_ratio
            chunk = a.eval_class_chunk or len(self.unseen_classes)
            out = []
            for s in range(0, len(self.unseen_classes), chunk):
                z0c = z0[s:s + chunk]
                nc = z0c.shape[0]
                z0b = z0c.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nc, *z0c.shape[1:])
                z1b = z1.unsqueeze(1).expand(-1, nc, -1, -1).reshape(B * nc, *z1.shape[1:])
                tb = t.unsqueeze(1).expand(-1, nc).reshape(-1)
                zt = m.psi(tb, z0b, z1b)
                pv = m.velocity_predictor(zt, tb)
                tv = m.Dt_psi(tb, z0b, z1b)
                out.append(torch.nn.functional.mse_loss(pv, tv, reduction='none').mean((1, 2)).view(B, nc))
            return torch.cat(out, dim=1)

    # ---------------------------------------------------------------- diagnostics
    def _diagnose(self, S, y):
        """S (N, U) MSE, y (N,) index into the unseen list. Returns log lines."""
        N, U = S.shape
        nm = lambda u: self._names.get(int(self._unseen[u]), str(self._unseen[u]))
        order = np.argsort(S, axis=1)
        rank = np.argmax(order == y[:, None], axis=1)
        top = [100.0 * (rank < k).mean() for k in (1, 2, 3)]
        srt = np.sort(S, axis=1)
        margin = (srt[:, 1] - srt[:, 0]) / np.maximum(srt[:, 0], 1e-12)
        edges = np.quantile(margin, [0.25, 0.5, 0.75])
        grp = np.digitize(margin, edges)
        correct = rank == 0
        by_grp = [100.0 * correct[grp == g].mean() if (grp == g).any() else float('nan') for g in range(4)]
        lines = [f'top1/top2/top3 {top[0]:.2f}/{top[1]:.2f}/{top[2]:.2f}  (top2-top1 {top[1] - top[0]:.2f}); N={N}, U={U}',
                 'accuracy by margin quartile (small->large margin) ' + ' / '.join(f'{v:.1f}' for v in by_grp)
                 + ('; error ratio smallest/largest quartile '
                    + (f'{(100 - by_grp[0]) / (100 - by_grp[3]):.2f}' if 100 - by_grp[3] >= 0.5 else 'n/a (largest-margin error < 0.5%)'))]
        pred = order[:, 0]
        conf = np.zeros((U, U), dtype=np.int64)
        for t_, p_ in zip(y[~correct], pred[~correct]):
            conf[t_, p_] += 1
        per_class = np.bincount(y, minlength=U)
        flat = [(conf[i, j], i, j) for i in range(U) for j in range(U) if i != j and conf[i, j] > 0]
        flat.sort(reverse=True)
        rev = {frozenset(p) for p in self._pairs}
        parts = []
        for c, i, j in flat[:5]:
            gi, gj = int(self._unseen[i]), int(self._unseen[j])
            parts.append(f'{nm(i)}->{nm(j)} {c}/{per_class[i]} ({100 * c / max(per_class[i], 1):.0f}%)'
                         f'{" [reversed pair]" if frozenset((gi, gj)) in rev else ""} zText {self._zt[gi, gj]:.2f} zKG {self._zk[gi, gj]:.2f}')
        lines.append('top confusions: ' + '; '.join(parts) if parts else 'top confusions: none')
        wrong = ~correct
        if wrong.any():
            gt = self._unseen[y[wrong]]
            gp = self._unseen[pred[wrong]]
            d = self._zt[gt, gp] - self._zk[gt, gp]
            uu = self._unseen[:, None], self._unseen[None, :]
            offm = ~np.eye(U, dtype=bool)
            base = (self._zt[uu] - self._zk[uu])[offm].mean()
            lines.append(f'wrong samples {int(wrong.sum())}: fraction with d=zText-zKG>0 {100 * (d > 0).mean():.1f}%, '
                         f'>1 {100 * (d > 1).mean():.1f}%, mean d {d.mean():.3f} (baseline over unseen pairs {base:.3f})')
        idx_of = {int(g): u for u, g in enumerate(self._unseen)}
        seg = []
        for i, j in self._pairs:
            if i not in idx_of and j not in idx_of:
                continue
            for x, other in ((i, j), (j, i)):
                if x not in idx_of:
                    continue
                u = idx_of[x]
                acc = 100 * correct[y == u].mean() if (y == u).any() else float('nan')
                if other in idx_of:
                    rate = 100 * conf[u, idx_of[other]] / max(per_class[u], 1)
                    seg.append(f'{self._names.get(x, x)} ({x}*) acc {acc:.1f}% -> partner {self._names.get(other, other)} ({other}*) {rate:.1f}%')
                else:
                    seg.append(f'{self._names.get(x, x)} ({x}*) acc {acc:.1f}% (partner {other} seen, cannot be predicted in ZSL)')
        lines.append('reversed pairs touching unseen (* = unseen): ' + ('; '.join(seg) if seg else 'none'))
        return lines

    # ---------------------------------------------------------------- evaluation (replaces Trainer.test_zsl)
    def test_zsl(self, nb_iter):
        import torch
        self.mapping_model.eval()
        scores, labels = [], []
        for gt_motion, label_idx, _, _, _ in self.zsl_loader:
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True)
            label_idx = label_idx.to(self.device, non_blocking=True)
            labels.append(self.zsl_target_lookup[label_idx].cpu())
            scores.append(self._scores(gt_motion).cpu())
        S = torch.cat(scores).numpy()
        y = torch.cat(labels).numpy()
        accuracy = 100.0 * float((S.argmin(1) == y).mean())
        self.best_accuracy[0] = max(self.best_accuracy[0], accuracy)
        self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {accuracy:.2f}% ')
        lines = self._diagnose(S, y)
        if self._best_diag is None or accuracy > self._best_diag[0]:
            self._best_diag = (accuracy, nb_iter, lines)
        if nb_iter == self.args.total_iter:
            self.logger.info(f'[diag] last (iter {nb_iter}, acc {accuracy:.2f}%)')
            for ln in lines:
                self.logger.info('[diag]   ' + ln)
        self.mapping_model.train()

    def print_best_accuracy(self):
        super().print_best_accuracy()
        if self._best_diag is not None:
            acc, it, lines = self._best_diag
            self.logger.info(f'[diag] best (iter {it}, acc {acc:.2f}%)')
            for ln in lines:
                self.logger.info('[diag]   ' + ln)


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_forbid_fuse_diag', DiagTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-center', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-orth', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-adaptive', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-clip', dict(type=float, default=3.0)),
           ],
           label_keys=[('fuse_center', 'c'), ('fuse_orth', 'o'), ('fuse_adaptive', 'a')])
