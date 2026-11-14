"""Hướng 4.1.1.6 (cải tiến từ nesy4_1_1_fuse.py): DiT dùng sai lệch của mình để bổ sung primitive theo luật KG rồi chấm lại.

Huấn luyện y hệt SeqKG + COA-Fuse. Chỉ thêm một bước ở lúc dự đoán ZSL:
  1. Chấm lần đầu như Flora (encoder lấy mẫu; điểm `base` = đúng điểm COA-Fuse) -> top-k lớp ứng viên.
  2. Với mỗi (mẫu, ứng viên c): ngữ nghĩa'_c = ngữ nghĩa_c + beta * sum_p delta_p * e_p
       e_p = embedding CLIP của primitive p (trừ trung bình các primitive, lặp lên 1536, chuẩn hoá về độ lớn trung bình
             của số hạng KG đã center) -> mọi primitive cùng độ lớn, delta = 1 ~ một số hạng KG.
       delta chỉ khác 0 trên primitive được phép (--rf-allow): kg = `require` của chính c; free = mọi primitive;
       shuffle = `require` của lớp khác (hoán vị); none = không bổ sung (kiểm tra: phải trùng base dùng mu).
  3. K bước gradient (--rf-steps) giảm MSE vận tốc của DiT cho đúng mẫu đó và lớp c, với encoder dùng mu (không nhiễu).
     Bước chuẩn hoá: delta <- delta - eta * g / max|g| (eta = budget / 2), rồi chiếu về {delta >= 0, ngoài tập cho phép = 0,
     sum(delta) <= budget}. Mọi lớp cùng số bước, cùng ngân sách (không lợi cho lớp có nhiều primitive).
     Gradient chỉ chảy vào delta (torch.autograd.grad), trọng số Flora không đổi.
  4. Dự đoán = ứng viên có MSE nhỏ nhất sau K bước. Các delta lớn nhất của lớp thắng = lời giải thích.
Trạng thái RNG được giữ: bước 1 lấy mẫu encoder đúng như predict_zsl, bước 2-4 không dùng số ngẫu nhiên, nên quá trình
huấn luyện trùng COA-Fuse.

Log `[rf]` mỗi lần đánh giá: base (Flora, lấy mẫu) / base dùng mu / sau từng bước 1..K / sửa / làm sai; chỉ số cơ chế:
trên các mẫu có lớp đúng trong top-k, MSE giảm bao nhiêu cho lớp đúng so với ứng viên sai và tỉ lệ lớp đúng giảm nhiều nhất.
Cuối run (`[rf] best`): đường cong theo bước tại lần đánh giá tốt nhất, ví dụ mẫu được sửa / làm sai kèm primitive đã bổ sung.

Tiêu chí đặt trước (seed 1, 48/12 và 110/10): kg >= base + 0.5 ở cả hai split và hơn cả free và shuffle >= 0.5;
rồi mới chạy 3 seed trên 4 split.

Chạy: python nesy4_1_1_6_refine.py --tasks 48 110 --seeds 1 --no-baselines                 (kg)
      python nesy4_1_1_6_refine.py --rf-allow free --tasks 48 110 --seeds 1 --no-baselines
      python nesy4_1_1_6_refine.py --rf-allow shuffle --tasks 48 110 --seeds 1 --no-baselines
      python nesy4_1_1_6_refine.py --rf-allow none --tasks 48 --seeds 1 --no-baselines        (kiểm tra)
"""
import json
import os

import nesy_common as C

import numpy as np

from kg_semantics import PrimitiveKG
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_1_fuse import FuseTrainer
from nesy0_shuffle import derangement


def project(delta, mask, budget):
    """Rows of delta onto {x >= 0, x = 0 outside mask, sum(x) <= budget}."""
    import torch
    x = delta.clamp_min(0) * mask
    over = x.sum(-1) > budget
    if over.any():
        v = x[over]
        u, _ = v.sort(dim=-1, descending=True)
        css = u.cumsum(-1)
        j = torch.arange(1, v.shape[-1] + 1, device=v.device, dtype=v.dtype)
        cond = u - (css - budget) / j > 0
        rho = cond.float().cumsum(-1).argmax(-1, keepdim=True)          # last index where cond holds
        theta = (css.gather(-1, rho) - budget) / (rho + 1).to(v.dtype)
        x[over] = (v - theta).clamp_min(0) * mask[over]
    return x


class RefineTrainer(FuseTrainer):
    def load_prototypes(self):
        super().load_prototypes()
        import torch
        a = self.args
        if a.setting != 'ZSL':
            raise ValueError('nesy4_1_1_6 chỉ hỗ trợ --setting ZSL')
        n = self.text_similarity.shape[0]
        sem = self.attuned_semantics
        dev, dt = sem.device, sem.dtype
        kg = PrimitiveKG(a.kg_prims, a.kg_rules, n, a.kg_forbid_weight)
        req = (kg.W > 0).astype(np.float32)
        if a.rf_allow == 'shuffle':
            req = req[derangement(n, np.random.RandomState(int(a.seed) + 1000))]
        elif a.rf_allow == 'free':
            req = np.ones_like(req)
        self._allow = torch.as_tensor(req, device=dev, dtype=dt)               # (n, P)

        # primitive embeddings: same CLIP cache the trainer used for compose
        kg_dir = os.path.dirname(os.path.abspath(a.kg_prims))
        stem = os.path.splitext(os.path.basename(a.kg_prims))[0]
        path = os.path.join(kg_dir, f'{stem}_clip_L{a.max_length}.pt')
        if os.path.exists(path):
            pe = torch.load(path, map_location=dev)
        else:
            from kg_semantics import embed_texts
            m = self.mapping_model
            pe = embed_texts(m.tokenizer, m.text_encoder, kg.prim_text, a.max_length, 'cuda', path)
        pe = pe.to(dev, dt)
        pe = pe - pe.mean(0, keepdim=True)
        pe = torch.cat([pe] * (sem.shape[-1] // pe.shape[-1]), dim=-1)          # (P, T, 1536)
        k = self.kg_info['signed'].to(dev, dt)
        k = torch.cat([k] * (sem.shape[-1] // k.shape[-1]), dim=-1)
        target = (k - k.mean(0, keepdim=True)).flatten(1).norm(dim=1).mean()
        self._prim = pe / pe.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None] * target
        self._prim_ids = kg.prim_ids
        rules = json.load(open(a.kg_rules, encoding='utf-8'))['rules']
        self._names = {r['id'] - 1: r.get('class', str(r['id'])) for r in rules}
        self._unseen = np.array([int(u) for u in self.unseen_classes])
        self._best = None
        self._base_best = 0.0
        per_class = self._allow[torch.as_tensor(self._unseen, device=dev)].sum(1)
        self.logger.info(f'[rf] refinement: allow {a.rf_allow}, steps {a.rf_steps}, budget {a.rf_budget}, top {a.rf_top}, '
                         f'beta {a.kg_beta}; primitive norm {target.item():.3f}; allowed primitives per unseen class '
                         f'min/mean/max {per_class.min().item():.0f}/{per_class.float().mean().item():.1f}/{per_class.max().item():.0f}')

    # ---------------------------------------------------------------- DiT pieces
    @staticmethod
    def _mu(enc, x):
        return enc.proj_mu(enc.backbone(x))            # encoder mean, no sampling (keeps RNG untouched)

    def _vel_mse(self, z0, z1):
        """z0, z1 (M, T, D) -> per-item MSE (M,)."""
        import torch
        import torch.nn.functional as F
        m = self.mapping_model
        t = torch.full((z0.shape[0],), float(self.args.step_ratio), device=z0.device)
        zt = m.psi(t, z0, z1)
        return F.mse_loss(m.velocity_predictor(zt, t), m.Dt_psi(t, z0, z1), reduction='none').mean((1, 2))

    def _base_sampled(self, feat, sem_u):
        """Exactly predict_zsl: sampled encoders, all unseen classes -> (B, U) MSE."""
        import torch
        a = self.args
        m = self.mapping_model
        B = feat.shape[0]
        sk = feat.unsqueeze(1).expand(-1, a.max_length + 1, -1)
        z1, _, _ = m.motion_encoder(sk)
        z0, _, _ = m.semantic_encoder(sem_u)
        U = sem_u.shape[0]
        chunk = a.eval_class_chunk or U
        out = []
        for s in range(0, U, chunk):
            z0c = z0[s:s + chunk]
            nc = z0c.shape[0]
            out.append(self._vel_mse(z0c.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nc, *z0c.shape[1:]),
                                     z1.unsqueeze(1).expand(-1, nc, -1, -1).reshape(B * nc, *z1.shape[1:])).view(B, nc))
        return torch.cat(out, dim=1)

    def _refine(self, feat, sem_u, allow_u, n_steps):
        """Returns mse_mu (B, U), per-step candidate MSE list [(B, k)] (step 0 .. K), candidates (B, k), final delta (B, k, P)."""
        import torch
        a = self.args
        m = self.mapping_model
        B, U = feat.shape[0], sem_u.shape[0]
        sk = feat.unsqueeze(1).expand(-1, a.max_length + 1, -1)
        with torch.no_grad():
            z1 = self._mu(m.motion_encoder, sk)                                         # (B, T, D)
            z0 = self._mu(m.semantic_encoder, sem_u)                                    # (U, T, D)
            mse_mu = self._vel_mse(z0.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * U, *z0.shape[1:]),
                                   z1.unsqueeze(1).expand(-1, U, -1, -1).reshape(B * U, *z1.shape[1:])).view(B, U)
        k = min(a.rf_top, U)
        cand = mse_mu.argsort(dim=1)[:, :k]                                             # (B, k)
        base_sem = sem_u[cand]                                                          # (B, k, T, W)
        mask = allow_u[cand]                                                            # (B, k, P)
        z1k = z1.unsqueeze(1).expand(-1, k, -1, -1).reshape(B * k, *z1.shape[1:])
        P = self._prim.shape[0]
        delta = torch.zeros(B, k, P, device=feat.device, dtype=base_sem.dtype)
        steps = [mse_mu.gather(1, cand)]
        eta = a.rf_budget / 2.0
        for _ in range(n_steps):
            delta = delta.detach().requires_grad_(True)
            with torch.enable_grad():
                s = base_sem + a.kg_beta * torch.einsum('bkp,ptw->bktw', delta, self._prim)
                z0k = self._mu(m.semantic_encoder, s.reshape(B * k, *s.shape[2:]))
                loss = self._vel_mse(z0k, z1k)
                g, = torch.autograd.grad(loss.sum(), delta)
            with torch.no_grad():
                g = g * mask
                g = g / g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
                delta = project(delta.detach() - eta * g, mask, a.rf_budget)
                s = base_sem + a.kg_beta * torch.einsum('bkp,ptw->bktw', delta, self._prim)
                steps.append(self._vel_mse(self._mu(m.semantic_encoder, s.reshape(B * k, *s.shape[2:])), z1k).view(B, k))
        return mse_mu, steps, cand, delta.detach()

    # ---------------------------------------------------------------- evaluation (replaces Trainer.test_zsl)
    def test_zsl(self, nb_iter):
        import torch
        a = self.args
        if not a.use_features:
            raise ValueError('nesy4_1_1_6 cần đặc trưng tính sẵn (use_features)')
        self.mapping_model.eval()
        sem_u = self.current_attuned_semantics()[self.unseen_classes].detach()
        allow_u = self._allow[torch.as_tensor(self._unseen, device=sem_u.device)]
        Ys, Sb, Mu, Steps, Cand, Delta = [], [], [], [], [], []
        for feat, label_idx, _, _, _ in self.zsl_loader:
            feat = feat.to(self.device, dtype=torch.float32, non_blocking=True)
            label_idx = label_idx.to(self.device, non_blocking=True)
            Ys.append(self.zsl_target_lookup[label_idx].cpu())
            with torch.no_grad():
                Sb.append(self._base_sampled(feat, sem_u).cpu())                        # consumes RNG like Flora
            mse_mu, steps, cand, delta = self._refine(feat, sem_u, allow_u, 0 if a.rf_allow == 'none' else a.rf_steps)
            Mu.append(mse_mu.cpu())
            Steps.append(torch.stack(steps, 0).cpu())                                   # (K+1, B, k)
            Cand.append(cand.cpu())
            Delta.append(delta.cpu())
        y = torch.cat(Ys).numpy()
        Sb = torch.cat(Sb).numpy()
        Mu = torch.cat(Mu).numpy()
        St = torch.cat(Steps, 1).numpy()                                                 # (K+1, N, k)
        cand = torch.cat(Cand).numpy()
        delta = torch.cat(Delta).numpy()
        N = len(y)
        base = 100.0 * (Sb.argmin(1) == y).mean()
        base_mu = 100.0 * (Mu.argmin(1) == y).mean()
        preds = [cand[np.arange(N), St[s].argmin(1)] for s in range(St.shape[0])]
        accs = [100.0 * (p == y).mean() for p in preds]
        acc = accs[-1]
        # mechanism: error reduction of the true class vs wrong candidates
        pos = (cand == y[:, None])
        has = pos.any(1)
        red = St[0] - St[-1]                                                             # (N, k) MSE reduction
        rel = red / np.maximum(St[0], 1e-12)
        r_true = rel[pos].mean() if has.any() else float('nan')
        r_wrong = rel[~pos & has[:, None]].mean() if has.any() else float('nan')
        wins = (rel[has][pos[has]] > np.where(pos[has], -np.inf, rel[has]).max(1)).mean() * 100 if has.any() else float('nan')
        fixed = int(((preds[0] != y) & (preds[-1] == y)).sum())
        broken = int(((preds[0] == y) & (preds[-1] != y)).sum())

        self._base_best = max(self._base_best, base)
        self.best_accuracy[0] = max(self.best_accuracy[0], acc)
        self.logger.info(f'Iter {nb_iter}: ZSL Accuracy: {acc:.2f}% ')
        self.logger.info(f'[rf] iter {nb_iter}: base {base:.2f} (mu {base_mu:.2f}) -> steps ' + ' / '.join(f'{v:.2f}' for v in accs[1:])
                         + f'; fixed {fixed}, broken {broken}; relative MSE reduction true {r_true:.3f} vs wrong {r_wrong:.3f}, '
                         f'true class reduced most {wins:.1f}% (oracle top{cand.shape[1]} {100.0 * has.mean():.2f})')
        if self._best is None or acc > self._best[0]:
            self._best = (acc, nb_iter, base, base_mu, accs, self._examples(preds[0], preds[-1], y, cand, delta, St[-1]))
        self.mapping_model.train()

    def _examples(self, p0, p1, y, cand, delta, last, n_each=3):
        nm = lambda u: self._names.get(int(self._unseen[u]), str(self._unseen[u]))
        lines = []
        for tag, mask in (('fixed', (p0 != y) & (p1 == y)), ('broken', (p0 == y) & (p1 != y))):
            for i in np.nonzero(mask)[0][:n_each]:
                j = int(np.argmin(last[i]))
                d = delta[i, j]
                top = ', '.join(f'{self._prim_ids[p]} {d[p]:.2f}' for p in np.argsort(-d)[:4] if d[p] > 0) or 'none'
                lines.append(f'{tag}: true {nm(y[i])} | before {nm(p0[i])} -> after {nm(p1[i])} | added to winner: {top}')
        return lines

    def print_best_accuracy(self):
        super().print_best_accuracy()
        if self._best is None:
            return
        acc, it, base, base_mu, accs, ex = self._best
        self.logger.info(f'[rf] best (iter {it}): refined {acc:.2f}, base at that eval {base:.2f} (mu {base_mu:.2f}); '
                         f'best base over run {self._base_best:.2f} (refined best - base best {acc - self._base_best:+.2f}); '
                         f'per step ' + ' / '.join(f'{v:.2f}' for v in accs))
        for ln in ex:
            self.logger.info('[rf]   ' + ln)


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_fuse_refine', RefineTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-center', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-orth', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-adaptive', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-clip', dict(type=float, default=3.0)),
               ('--rf-allow', dict(type=str, default='kg', choices=['kg', 'free', 'shuffle', 'none'],
                                   help='primitives a candidate class may add: its own require set / any / another class / none')),
               ('--rf-steps', dict(type=int, default=5, help='gradient steps per (sample, candidate)')),
               ('--rf-budget', dict(type=float, default=1.0, help='max sum of primitive weights added (1 ~ one KG term)')),
               ('--rf-top', dict(type=int, default=3, help='DiT candidates refined')),
           ],
           label_keys=[('rf_allow', 'allow'), ('rf_steps', 'K'), ('rf_budget', 'B'), ('rf_top', 'top')])
