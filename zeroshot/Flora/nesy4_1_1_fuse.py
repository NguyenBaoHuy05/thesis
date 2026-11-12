"""Hướng 4.1.1 (cải tiến từ nesy4_1_forbid.py): fuse theo độ lệch text - KG, thay cho phép cộng đều.

nesy4_1:  sem_c = A_c + beta * k_c       A = ngữ nghĩa Flora (attuned), k = compose SeqKG có forbid
Ở đây (không có tham số học, chỉ beta như cũ):
  1. --fuse-center   k~_c = k_c - mean_c k_c
       CLIP rất đẳng hướng: phần lớn k là một vector chung cộng vào MỌI lớp, không giúp phân biệt.
  2. --fuse-orth     k_perp_c = k~_c - proj(k~_c lên A~_c)   (A~ = A trừ trung bình theo lớp, tính theo từng token)
       KG chỉ thêm phần text của chính lớp đó chưa có; độ lớn được đưa về bằng ||k~_c|| (chỉ đổi hướng).
  3. --fuse-adaptive beta_c = beta * d_c / mean(d),   d_c = max_{j != c} [ z(cos_text(c, j)) - z(s_kg(c, j)) ]_+
       z = chuẩn hoá theo các phần tử ngoài đường chéo (hai ma trận khác thang đo). KG được dùng mạnh ở lớp mà
       text nói "giống lớp j" còn luật KG nói "khác"; cắt trên ở --fuse-clip lần beta.
  sem_c = A_c + beta_c * k_fused_c

Log `[nesy4_1_1]`: phân bố beta_c, các lớp có beta_c lớn nhất, cosine (đã trừ trung bình) của các cặp lớp đảo chiều
trên ngữ nghĩa cuối cùng, trước (nesy4_1) và sau fuse.

Chạy: python nesy4_1_1_fuse.py --tasks 48 110 --seeds 1 --no-baselines                 (đủ 3 bước)
      python nesy4_1_1_fuse.py --fuse-orth 0 --fuse-adaptive 0                          (chỉ bước 1)
      python nesy4_1_1_fuse.py --fuse-adaptive 0                                        (bước 1 + 2)
"""
import json
import os

import nesy_common as C

import numpy as np

from kg_semantics import PrimitiveKG
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_forbid import ForbidTrainer, order_only_pairs


class FuseTrainer(ForbidTrainer):
    def load_prototypes(self):
        super().load_prototypes()
        import torch
        import torch.nn.functional as F
        a = self.args
        if a.kg_compose_at != 'post' or a.kg_compose_half != 'both':
            raise ValueError('nesy4_1_1 cần --kg-compose-at post và --kg-compose-half both')
        sem = self.attuned_semantics
        dev, dt = sem.device, sem.dtype
        k = self.kg_info['signed'].to(dev, dt)                                   # (n, T, 768)
        k = torch.cat([k] * (sem.shape[-1] // k.shape[-1]), dim=-1)              # (n, T, 1536), as compose does
        A0 = sem - a.kg_beta * k                                                 # Flora semantics without KG
        n = sem.shape[0]

        add = k
        if a.fuse_center:
            add = add - add.mean(0, keepdim=True)
        if a.fuse_orth:
            t = A0 - A0.mean(0, keepdim=True)
            coef = (add * t).sum(-1, keepdim=True) / (t * t).sum(-1, keepdim=True).clamp_min(1e-8)
            orth = add - coef * t
            scale = add.flatten(1).norm(dim=1) / orth.flatten(1).norm(dim=1).clamp_min(1e-8)
            add = orth * scale[:, None, None]

        beta_c = torch.full((n,), float(a.kg_beta), device=dev, dtype=dt)
        kg = PrimitiveKG(a.kg_prims, a.kg_rules, n, a.kg_forbid_weight)
        if a.fuse_adaptive:
            ts = self.text_similarity.to(dev).float()
            ks = torch.from_numpy(kg.similarity()).to(dev).float()
            off = ~torch.eye(n, dtype=torch.bool, device=dev)
            z = lambda m: (m - m[off].mean()) / m[off].std().clamp_min(1e-8)
            d = (z(ts) - z(ks)).masked_fill(~off, float('-inf')).max(1).values.clamp_min(0)
            w = (d / d.mean().clamp_min(1e-8)).clamp(max=a.fuse_clip)
            beta_c = (a.kg_beta * w).to(dt)

        self.attuned_semantics = A0 + beta_c[:, None, None] * add

        # ---- log
        def centered(x):
            zz = x[:, -1, :].float()
            zz = F.normalize(zz - zz.mean(0, keepdim=True), dim=-1)
            return zz @ zz.t()
        names = {r['id'] - 1: r.get('class', str(r['id'])) for r in json.load(open(a.kg_rules, encoding='utf-8'))['rules']}
        unseen = set(int(u) for u in self.unseen_classes)
        b = beta_c.float().cpu().numpy()
        top = np.argsort(-b)[:8]
        msg = (f'[nesy4_1_1] fuse center {a.fuse_center} orth {a.fuse_orth} adaptive {a.fuse_adaptive} '
               f'(clip {a.fuse_clip}); beta_c min/mean/max {b.min():.3f}/{b.mean():.3f}/{b.max():.3f}; '
               f'unseen mean beta_c {np.mean([b[c] for c in unseen]):.3f}; top: '
               + ', '.join(f'{c}{"*" if c in unseen else ""} {names.get(c, c)} {b[c]:.2f}' for c in top))
        pairs = order_only_pairs(a.kg_rules, n)
        if pairs:
            c0, c1 = centered(sem), centered(self.attuned_semantics)
            m0 = np.mean([c0[i, j].item() for i, j in pairs])
            m1 = np.mean([c1[i, j].item() for i, j in pairs])
            msg += f'; reversed-pair centered cosine of final semantics {m0:.4f} (nesy4_1) -> {m1:.4f} (fuse)'
        self.logger.info(msg + '   (* = unseen)')


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_forbid_fuse', FuseTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-center', dict(type=int, default=1, choices=[0, 1], help='remove the class-mean of the KG term')),
               ('--fuse-orth', dict(type=int, default=1, choices=[0, 1], help='remove the part along the class text semantics')),
               ('--fuse-adaptive', dict(type=int, default=1, choices=[0, 1], help='per-class beta from text-vs-KG disagreement')),
               ('--fuse-clip', dict(type=float, default=3.0, help='max beta_c / beta')),
           ],
           label_keys=[('fuse_center', 'c'), ('fuse_orth', 'o'), ('fuse_adaptive', 'a')])
