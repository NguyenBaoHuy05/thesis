"""Hướng 4.1.1.5 (cải tiến từ nesy4_1_1_fuse.py): chấm lại ứng viên bằng truy hồi mẫu seen + luật KG.

Huấn luyện y hệt COA-Fuse. Chỉ đổi bước dự đoán ZSL:
  1. Điểm DiT: ma trận MSE (mẫu x lớp unseen) như models/fm.py predict_zsl (tuỳ chọn nhiều bước t, dùng mu).
  2. Truy hồi (neural): K mẫu huấn luyện seen gần nhất theo cosine của đặc trưng skeleton 256 chiều (đặc trưng cố định,
     không phụ thuộc mô hình đang học). Trọng số softmax(sim / tau).
  3. Hồ sơ primitive (cầu nối ký hiệu): prof = sum_j w_j * Req[y_j], Req[c, p] = 1 nếu lớp c `require` primitive p.
  4. Khớp luật (symbolic, logic mờ): rule_c = mean_{p in require_c} prof[p] - mean_{p in forbid_c} prof[p],
     chỉ tính trên primitive "quan sát được" (có ít nhất một lớp seen require). AND = trung bình, NOT = trừ.
  5. Kết hợp: trong top-k lớp theo DiT, chọn argmax z(-MSE) + gamma * z(rule)  (z-score theo từng mẫu trên các lớp unseen).
Vẫn là ZSL inductive: chỉ dùng mẫu seen, luật KG và tên lớp; không dùng nhãn hay thống kê của tập test.

Cờ:
  --rr-k 50          số láng giềng seen          --rr-tau 0.05    nhiệt độ softmax trên cosine
  --rr-top 3         số ứng viên DiT được chấm lại  --rr-gamma 1.0   trọng số luật (cố định từ trước)
  --rr-forbid all    forbid dùng trong luật: all | order (chỉ seq.*) | none
  --rr-shuffle 0     1 = hoán vị các hàng luật giữa các lớp (đối chứng; hoán vị như nesy0_shuffle)
  --multi-t ''       ví dụ "0.3,0.5,0.7": trung bình MSE qua nhiều t (mặc định: chỉ step_ratio như Flora)
  --use-mu 0         1 = dùng mu của encoder thay vì lấy mẫu ngẫu nhiên khi đánh giá
Điểm `ZSL Accuracy` mà runner đọc là điểm SAU khi chấm lại (gamma, top đã chọn). Điểm DiT gốc của cùng lần đánh giá được
ghi ở dòng `[rr]`, nên so sánh là ghép cặp (cùng mô hình, cùng mẫu noise).

Log `[rr]` mỗi lần đánh giá: base / rr / rule-only / oracle top-k; lưới gamma x top (miễn phí, không huấn luyện lại).
Cuối run (`[rr] best`): lưới tại lần đánh giá có rr cao nhất, độ phủ primitive, ví dụ mẫu được sửa và bị làm sai.

Tiêu chí đặt trước: >= +1 so với COA-Fuse (TB 3 seed) ở >= 3/4 split, và --rr-shuffle 1 phải mất phần tăng.

Chạy: python nesy4_1_1_5_retrieval.py --tasks 48 110 --seeds 1 2 3 --no-baselines
      python nesy4_1_1_5_retrieval.py --rr-shuffle 1 --tasks 48 110 --seeds 1 2 3 --no-baselines
      python nesy4_1_1_5_retrieval.py --multi-t 0.3,0.5,0.7 --use-mu 1 --tasks 48 110 --seeds 1 2 3 --no-baselines
"""
import json
import os

import nesy_common as C

import numpy as np

from kg_semantics import PrimitiveKG
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_1_fuse import FuseTrainer
from nesy0_shuffle import derangement

GAMMAS = (0.25, 0.5, 1.0, 2.0)
TOPS = (2, 3, 5)


class RetrievalTrainer(FuseTrainer):
    def load_prototypes(self):
        super().load_prototypes()
        a = self.args
        if a.setting != 'ZSL':
            raise ValueError('nesy4_1_1_5 chỉ hỗ trợ --setting ZSL')
        n = self.text_similarity.shape[0]
        kg = PrimitiveKG(a.kg_prims, a.kg_rules, n, a.kg_forbid_weight)
        req = (kg.W > 0).astype(np.float32)
        forb = (kg.W < 0).astype(np.float32)
        if a.rr_forbid == 'order':
            forb *= np.array([p.startswith('seq.') for p in kg.prim_ids], dtype=np.float32)[None, :]
        elif a.rr_forbid == 'none':
            forb[:] = 0
        if a.rr_shuffle:
            perm = derangement(n, np.random.RandomState(int(a.seed) + 1000))
            req, forb = req[perm], forb[perm]
        seen = np.array([int(s) for s in self.seen_classes])
        unseen = np.array([int(u) for u in self.unseen_classes])
        obs = req[seen].any(0).astype(np.float32)                    # primitives some seen class requires
        self._req, self._obs = req, obs
        self._rq = req[unseen] * obs                                  # (U, P)
        self._fb = forb[unseen] * obs
        self._prim_ids = kg.prim_ids
        rules = json.load(open(a.kg_rules, encoding='utf-8'))['rules']
        self._names = {r['id'] - 1: r.get('class', str(r['id'])) for r in rules}
        self._unseen = unseen
        self._bank = None
        self._base_best = 0.0
        self._best_rr = None
        cov = self._rq.sum(1) / np.maximum(req[unseen].sum(1), 1)
        self.logger.info(
            f'[rr] retrieval re-ranking: k {a.rr_k} tau {a.rr_tau} top {a.rr_top} gamma {a.rr_gamma} forbid {a.rr_forbid} '
            f'shuffle {a.rr_shuffle} multi_t "{a.multi_t}" use_mu {a.use_mu}; observable primitives {int(obs.sum())}/{len(obs)}; '
            f'unseen require coverage mean {cov.mean():.2f} min {cov.min():.2f} '
            f'({", ".join(f"{self._names.get(int(u), u)} {c:.2f}" for u, c in zip(unseen, cov) if c < 0.75) or "all >= 0.75"})')

    # ---------------------------------------------------------------- memory bank of seen training features
    def _build_bank(self):
        import torch
        import torch.nn.functional as F
        ds = self.data_loader['train'].dataset
        x = torch.as_tensor(np.asarray(ds.data), dtype=torch.float32, device=self.device).reshape(len(ds.label), -1)
        self._bank = F.normalize(x, dim=-1)
        self._bank_req = torch.as_tensor(self._req[np.asarray(ds.label).astype(np.int64)], device=self.device)   # (N, P)
        self.logger.info(f'[rr] memory bank: {x.shape[0]} seen training samples, dim {x.shape[1]}')

    def _profile(self, feat):
        """feat (B, 256) -> primitive profile (B, P)."""
        import torch
        import torch.nn.functional as F
        a = self.args
        sim = F.normalize(feat.float(), dim=-1) @ self._bank.t()                  # (B, N)
        val, idx = sim.topk(min(a.rr_k, sim.shape[1]), dim=1)
        w = torch.softmax(val / a.rr_tau, dim=1)                                    # (B, k)
        return torch.einsum('bk,bkp->bp', w, self._bank_req[idx])

    # ---------------------------------------------------------------- DiT scores (as predict_zsl; optional multi-t / mu)
    def _mse(self, feat):
        import torch
        import torch.nn.functional as F
        a = self.args
        m = self.mapping_model
        B = feat.shape[0]
        sk = feat.unsqueeze(1).expand(-1, a.max_length + 1, -1)
        z1, mu1, _ = m.motion_encoder(sk)
        sem = self.current_attuned_semantics()[self.unseen_classes]
        z0, mu0, _ = m.semantic_encoder(sem)
        if a.use_mu:
            z1, z0 = mu1, mu0
        ts = [float(t) for t in a.multi_t.split(',')] if a.multi_t else [a.step_ratio]
        U = len(self.unseen_classes)
        chunk = a.eval_class_chunk or U
        total = 0
        for tv in ts:
            t = torch.full((B,), tv, device=feat.device)
            out = []
            for s in range(0, U, chunk):
                z0c = z0[s:s + chunk]
                nc = z0c.shape[0]
                z0b = z0c.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nc, *z0c.shape[1:])
                z1b = z1.unsqueeze(1).expand(-1, nc, -1, -1).reshape(B * nc, *z1.shape[1:])
                tb = t.unsqueeze(1).expand(-1, nc).reshape(-1)
                zt = m.psi(tb, z0b, z1b)
                out.append(F.mse_loss(m.velocity_predictor(zt, tb), m.Dt_psi(tb, z0b, z1b), reduction='none').mean((1, 2)).view(B, nc))
            total = total + torch.cat(out, dim=1)
        return total / len(ts)

    # ---------------------------------------------------------------- evaluation (replaces Trainer.test_zsl)
    def test_zsl(self, nb_iter):
        import torch
        a = self.args
        if not a.use_features:
            raise ValueError('nesy4_1_1_5 cần đặc trưng tính sẵn (use_features)')
        if self._bank is None:
            self._build_bank()
        self.mapping_model.eval()
        S, P, Y = [], [], []
        rq = torch.as_tensor(self._rq)
        fb = torch.as_tensor(self._fb)
        with torch.no_grad():
            for feat, label_idx, _, _, _ in self.zsl_loader:
                feat = feat.to(self.device, dtype=torch.float32, non_blocking=True)
                label_idx = label_idx.to(self.device, non_blocking=True)
                Y.append(self.zsl_target_lookup[label_idx].cpu())
                S.append(self._mse(feat).cpu())
                P.append(self._profile(feat).cpu())
        S = torch.cat(S).numpy()
        prof = torch.cat(P)
        y = torch.cat(Y).numpy()
        r = (prof @ rq.t()) / rq.sum(1).clamp_min(1)
        f = (prof @ fb.t()) / fb.sum(1).clamp_min(1)
        rule = (r - f).numpy()                                                      # (N, U)

        z = lambda m: (m - m.mean(1, keepdims=True)) / np.maximum(m.std(1, keepdims=True), 1e-8)
        zd, zr = z(-S), z(rule)
        order = np.argsort(S, axis=1)
        U = S.shape[1]

        def rerank(gamma, top):
            top = min(top, U)
            cand = order[:, :top]
            sc = np.take_along_axis(zd + gamma * zr, cand, 1)
            return cand[np.arange(len(y)), sc.argmax(1)]

        base_pred = order[:, 0]
        base = 100.0 * (base_pred == y).mean()
        rr_pred = rerank(a.rr_gamma, a.rr_top)
        acc = 100.0 * (rr_pred == y).mean()
        rule_only = 100.0 * (zr.argmax(1) == y).mean()
        oracle = 100.0 * (order[:, :min(a.rr_top, U)] == y[:, None]).any(1).mean()
        grid = {(g, k): 100.0 * (rerank(g, k) == y).mean() for g in GAMMAS for k in TOPS if k <= max(U, 2)}

        self._base_best = max(self._base_best, base)
        self.best_accuracy[0] = max(self.best_accuracy[0], acc)
        self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')
        self.logger.info(f'[rr] iter {nb_iter}: base {base:.2f} -> rr {acc:.2f} ({acc - base:+.2f}); rule-only {rule_only:.2f}; '
                         f'oracle top{a.rr_top} {oracle:.2f}; fixed {int(((base_pred != y) & (rr_pred == y)).sum())}, '
                         f'broken {int(((base_pred == y) & (rr_pred != y)).sum())}')
        if self._best_rr is None or acc > self._best_rr[0]:
            self._best_rr = (acc, nb_iter, base, grid, self._examples(base_pred, rr_pred, y, prof.numpy()))
        self.mapping_model.train()

    def _examples(self, base_pred, rr_pred, y, prof, n_each=3):
        nm = lambda u: self._names.get(int(self._unseen[u]), str(self._unseen[u]))
        top_prims = lambda i: ', '.join(f'{self._prim_ids[p]} {prof[i, p]:.2f}' for p in np.argsort(-prof[i])[:5])
        lines = []
        for tag, mask in (('fixed', (base_pred != y) & (rr_pred == y)), ('broken', (base_pred == y) & (rr_pred != y))):
            idx = np.nonzero(mask)[0][:n_each]
            for i in idx:
                lines.append(f'{tag}: true {nm(y[i])} | DiT {nm(base_pred[i])} -> rr {nm(rr_pred[i])} | profile: {top_prims(i)}')
        return lines

    def print_best_accuracy(self):
        super().print_best_accuracy()
        if self._best_rr is None:
            return
        acc, it, base, grid, ex = self._best_rr
        self.logger.info(f'[rr] best (iter {it}): rr {acc:.2f}, base at that eval {base:.2f}; best base over run {self._base_best:.2f} '
                         f'(rr best - base best {acc - self._base_best:+.2f})')
        ks = sorted({k for _, k in grid})
        self.logger.info('[rr] grid at best eval (gamma x top): ' + '; '.join(
            f'g{g}: ' + ' / '.join(f'top{k} {grid[(g, k)]:.2f}' for k in ks if (g, k) in grid) for g in GAMMAS))
        for ln in ex:
            self.logger.info('[rr]   ' + ln)


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_fuse_rr', RetrievalTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-center', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-orth', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-adaptive', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-clip', dict(type=float, default=3.0)),
               ('--rr-k', dict(type=int, default=50, help='seen neighbours retrieved per test sample')),
               ('--rr-tau', dict(type=float, default=0.05, help='softmax temperature on cosine similarity')),
               ('--rr-top', dict(type=int, default=3, help='DiT candidates that are re-ranked')),
               ('--rr-gamma', dict(type=float, default=1.0, help='weight of the rule score (fixed in advance)')),
               ('--rr-forbid', dict(type=str, default='all', choices=['all', 'order', 'none'])),
               ('--rr-shuffle', dict(type=int, default=0, choices=[0, 1], help='1 = permute KG rows across classes (control)')),
               ('--multi-t', dict(type=str, default='', help='comma-separated t values averaged at inference, e.g. 0.3,0.5,0.7')),
               ('--use-mu', dict(type=int, default=0, choices=[0, 1], help='use encoder means instead of samples at inference')),
           ],
           label_keys=[('rr_k', 'k'), ('rr_top', 'top'), ('rr_gamma', 'g'), ('rr_forbid', 'fb'), ('rr_shuffle', 'shuf'),
                       ('multi_t', 'mt'), ('use_mu', 'mu')])
