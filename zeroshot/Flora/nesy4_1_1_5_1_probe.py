"""Hướng 4.1.1.5.1 (cải tiến từ nesy4_1_1_5_retrieval.py): DiT (COA-Fuse) + bằng chứng primitive từ bộ dò, cộng bằng log-xác suất.

Giữ DiT nguyên vẹn; chỉ CỘNG thêm một số hạng nhỏ trong top-k ứng viên của DiT:
    điểm_c = z(-MSE_DiT)_c + gamma * L_c
    L_c = mean_{p in require_c} log q_p + mean_{p in forbid_c} log(1 - q_p)        (AND mềm = trung bình log, NOT = 1 - q)
q = xác suất primitive của mẫu test:
  --pr-src probe   bộ dò MLP huấn luyện trên đặc trưng seen, nhãn = hàng `require` của lớp (như nesy0_probe; AUC unseen 0.86-0.88)
  --pr-src knn     hồ sơ kNN như nesy4_1_1_5 (k = --rr-k, tau = --rr-tau)
  --pr-src both    trung bình hai nguồn
Khác nesy4_1_1_5: (1) bộ dò thay cho / cùng với kNN, (2) số hạng luật KHÔNG chuẩn hoá z-score mà giữ đơn vị log-xác suất,
nên gamma nhỏ chỉ phá thế hoà khi DiT phân vân, (3) lưới gamma rất nhỏ.
Bộ dò huấn luyện một lần ở lần đánh giá đầu (vài giây); trạng thái RNG được lưu và khôi phục nên quá trình huấn luyện Flora
không bị đổi so với COA-Fuse.

Log `[pr]` mỗi lần đánh giá: base / rr / rule-only / oracle top-k / sửa / làm sai; và trong các mẫu DiT sai mà lớp đúng nằm
trong top-k: tỉ lệ L xếp lớp đúng trên lớp DiT đoán (độ bổ sung; > 50% mới có hy vọng).
Cuối run: lưới gamma x top tại lần đánh giá có rr cao nhất và tại lần có base cao nhất, cùng ví dụ.

Tiêu chí đặt trước: một ô (gamma, top) cố định vượt base ở cả 48/12 và 110/10 seed 1 thì mới chạy 3 seed;
3 seed phải >= +0.5 so với COA-Fuse ở cả hai split, và --rr-shuffle 1 phải mất phần tăng.

Chạy: python nesy4_1_1_5_1_probe.py --tasks 48 110 --seeds 1 --no-baselines
      python nesy4_1_1_5_1_probe.py --pr-src knn --tasks 48 110 --seeds 1 --no-baselines
"""
import os
import random

import nesy_common as C

import numpy as np

from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_1_5_retrieval import RetrievalTrainer
from nesy4_1_1_fuse import FuseTrainer
from nesy0_probe import train_probe

GAMMAS = (0.02, 0.05, 0.1, 0.25, 0.5)
EPS = 1e-4


class ProbeTrainer(RetrievalTrainer):
    def _build_bank(self):
        super()._build_bank()                       # kNN memory bank (always built; cheap)
        import torch
        a = self.args
        ds = self.data_loader['train'].dataset
        X = np.asarray(ds.data, dtype=np.float32).reshape(len(ds.label), -1)
        y = np.asarray(ds.label).astype(np.int64)
        self._cols = np.nonzero(self._obs > 0)[0]
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-6
        T = self._req[y][:, self._cols]
        # keep Flora's random streams untouched
        st, npst, pyst = torch.get_rng_state(), np.random.get_state(), random.getstate()
        cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self._probe = train_probe((X - self._mu) / self._sd, T, a.pr_model, a.pr_epochs, int(a.seed), self.device)
        finally:
            torch.set_rng_state(st)
            np.random.set_state(npst)
            random.setstate(pyst)
            if cst is not None:
                torch.cuda.set_rng_state_all(cst)
        self.logger.info(f'[pr] primitive probe ({a.pr_model}, {a.pr_epochs} epochs) trained on {len(X)} seen samples, '
                         f'{len(self._cols)} primitives; source {a.pr_src}')

    def _q(self, feat):
        """Primitive probabilities (B, P) on the observable columns (0 elsewhere)."""
        a = self.args
        P = self._req.shape[1]
        q = np.zeros((feat.shape[0], P), dtype=np.float32)
        parts = []
        if a.pr_src in ('probe', 'both'):
            f = feat.detach().cpu().numpy()
            parts.append(self._probe((f - self._mu) / self._sd))
        if a.pr_src in ('knn', 'both'):
            parts.append(self._profile(feat).cpu().numpy()[:, self._cols])
        q[:, self._cols] = np.mean(parts, axis=0)
        return q

    def test_zsl(self, nb_iter):
        import torch
        a = self.args
        if self._bank is None:
            self._build_bank()
        self.mapping_model.eval()
        S, Q, Y = [], [], []
        with torch.no_grad():
            for feat, label_idx, _, _, _ in self.zsl_loader:
                feat = feat.to(self.device, dtype=torch.float32, non_blocking=True)
                label_idx = label_idx.to(self.device, non_blocking=True)
                Y.append(self.zsl_target_lookup[label_idx].cpu())
                S.append(self._mse(feat).cpu())
                Q.append(self._q(feat))
        S = torch.cat(S).numpy()
        q = np.clip(np.concatenate(Q), EPS, 1 - EPS)
        y = torch.cat(Y).numpy()
        rq, fb = self._rq, self._fb                                                  # (U, P), observable only
        L = (np.log(q) @ rq.T) / np.maximum(rq.sum(1), 1) + (np.log(1 - q) @ fb.T) / np.maximum(fb.sum(1), 1)
        zd = (-S - (-S).mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-8)
        order = np.argsort(S, axis=1)
        N, U = S.shape

        def rerank(gamma, top):
            cand = order[:, :min(top, U)]
            return cand[np.arange(N), np.take_along_axis(zd + gamma * L, cand, 1).argmax(1)]

        base_pred = order[:, 0]
        base = 100.0 * (base_pred == y).mean()
        rr_pred = rerank(a.rr_gamma, a.rr_top)
        acc = 100.0 * (rr_pred == y).mean()
        topk = order[:, :min(a.rr_top, U)]
        in_top = (topk == y[:, None]).any(1)
        wrong_in = (base_pred != y) & in_top
        comp = 100.0 * (L[np.arange(N), y] > L[np.arange(N), base_pred])[wrong_in].mean() if wrong_in.any() else float('nan')
        tops = sorted({2, 3, U})
        grid = {(g, k): 100.0 * (rerank(g, k) == y).mean() for g in GAMMAS for k in tops}

        self._base_best = max(self._base_best, base)
        self.best_accuracy[0] = max(self.best_accuracy[0], acc)
        self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')
        self.logger.info(f'[pr] iter {nb_iter}: base {base:.2f} -> rr {acc:.2f} ({acc - base:+.2f}); rule-only {100.0 * (L.argmax(1) == y).mean():.2f}; '
                         f'oracle top{a.rr_top} {100.0 * in_top.mean():.2f}; fixed {int(((base_pred != y) & (rr_pred == y)).sum())}, '
                         f'broken {int(((base_pred == y) & (rr_pred != y)).sum())}; complementarity {comp:.1f}% '
                         f'(DiT-wrong, true in top{a.rr_top}: L prefers true class)')
        entry = (acc, nb_iter, base, grid, self._examples(base_pred, rr_pred, y, q))
        if self._best_rr is None or acc > self._best_rr[0]:
            self._best_rr = entry
        if getattr(self, '_best_base', None) is None or base > self._best_base[2]:
            self._best_base = entry
        self.mapping_model.train()

    def print_best_accuracy(self):
        FuseTrainer.print_best_accuracy(self)
        if self._best_rr is None:
            return
        fmt = lambda grid: '; '.join(f'g{g}: ' + ' / '.join(f'top{k} {grid[(gg, k)]:.2f}' for (gg, k) in sorted(grid) if gg == g)
                                     for g in GAMMAS)
        acc, it, base, grid, ex = self._best_rr
        self.logger.info(f'[pr] best rr (iter {it}): rr {acc:.2f}, base at that eval {base:.2f}; best base over run {self._base_best:.2f} '
                         f'(rr best - base best {acc - self._base_best:+.2f})')
        self.logger.info('[pr] grid at best-rr eval: ' + fmt(grid))
        b = self._best_base
        self.logger.info(f'[pr] grid at best-base eval (iter {b[1]}, base {b[2]:.2f}): ' + fmt(b[3]))
        for ln in ex:
            self.logger.info('[pr]   ' + ln)


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_fuse_rr_probe', ProbeTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-center', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-orth', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-adaptive', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-clip', dict(type=float, default=3.0)),
               ('--rr-k', dict(type=int, default=50)),
               ('--rr-tau', dict(type=float, default=0.05)),
               ('--rr-top', dict(type=int, default=3, help='DiT candidates that are re-ranked')),
               ('--rr-gamma', dict(type=float, default=0.1, help='weight of the log-probability rule term (fixed in advance)')),
               ('--rr-forbid', dict(type=str, default='all', choices=['all', 'order', 'none'])),
               ('--rr-shuffle', dict(type=int, default=0, choices=[0, 1], help='1 = permute KG rows across classes (control)')),
               ('--multi-t', dict(type=str, default='')),
               ('--use-mu', dict(type=int, default=0, choices=[0, 1])),
               ('--pr-src', dict(type=str, default='probe', choices=['probe', 'knn', 'both'])),
               ('--pr-model', dict(type=str, default='mlp', choices=['linear', 'mlp'])),
               ('--pr-epochs', dict(type=int, default=20)),
           ],
           label_keys=[('pr_src', 'src'), ('pr_model', 'm'), ('rr_top', 'top'), ('rr_gamma', 'g'), ('rr_shuffle', 'shuf')])
