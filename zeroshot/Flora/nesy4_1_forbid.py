"""Hướng 4.1 (cải tiến từ nesy4_seqkg.py): compose có dùng `forbid`.

compose gốc chỉ lấy primitive `require` (kg_semantics.compose_weights cắt W về >= 0), nên 14 luật
`forbid` thứ tự ngược của KG v2seq (seq.b>a) không có tác dụng gì; cosine (đã trừ trung bình) giữa hai lớp
đảo chiều vẫn ~0.989. File này đổi trọng số compose thành
    sem_c = r_c + lam * (r_c - f_c)  =  (1 + lam) r_c - lam f_c
  r_c = trung bình embedding primitive require của lớp c, f_c = trung bình primitive forbid.
Tổng trọng số vẫn bằng 1 nên thành phần chung (CLIP rất đẳng hướng) và độ lớn được giữ, kg_beta không cần chỉnh.
Lớp không có forbid giữ nguyên compose. Không có tham số học.
  --forbid-order-only 1 (mặc định): chỉ dùng forbid thứ tự (primitive seq.*), cô lập đúng hiệu ứng SeqKG.
  --forbid-order-only 0: dùng mọi forbid (đụng gần như mọi lớp).

Log có dòng `[nesy4_1]`: cosine đã trừ trung bình của các cặp lớp chỉ khác nhau ở primitive thứ tự,
compose thường -> compose có forbid.

Chạy: python nesy4_1_forbid.py --tasks 48 110 --seeds 1 --baselines compose
      python nesy4_1_forbid.py --forbid-lam 0.5
      python nesy4_1_forbid.py --forbid-order-only 0
Baseline `compose` dùng chung thư mục với compose KG v2 của notebook; run nào xong trước sẽ được tái sử dụng.
"""
import json
import os

import nesy_common as C

import numpy as np

import kg_semantics as KS
from trainer import Trainer
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES


def signed_compose_weights(kg, lam, order_only=True):
    """(n, P) weights, each row sums to 1: (1 + lam) * mean(require) - lam * mean(forbid)."""
    req = (kg.W > 0).astype(np.float32)
    forb = (kg.W < 0).astype(np.float32)
    if order_only:
        forb *= np.array([p.startswith('seq.') for p in kg.prim_ids], dtype=np.float32)[None, :]
    r = req / np.maximum(req.sum(1, keepdims=True), 1.0)
    f = forb / np.maximum(forb.sum(1, keepdims=True), 1.0)
    has_f = forb.sum(1, keepdims=True) > 0
    return np.where(has_f, (1.0 + lam) * r - lam * f, r).astype(np.float32)


def order_only_pairs(rules_path, n):
    """Class pairs whose rules are identical once the seq.* primitives are removed."""
    strip = lambda ids: tuple(sorted(i for i in ids if not i.startswith('seq.')))
    rules = {r['id'] - 1: r for r in json.load(open(rules_path, encoding='utf-8'))['rules']}
    by_key = {}
    for c in range(n):
        by_key.setdefault((strip(rules[c]['require']), strip(rules[c]['forbid'])), []).append(c)
    return [v for v in by_key.values() if len(v) == 2]


class ForbidTrainer(Trainer):
    def load_prototypes(self):
        a = self.args
        if a.kg_mode != 'compose' or a.kg_compose_src != 'avg' or a.kg_compose_idf or a.kg_learned_gate:
            raise ValueError('nesy4_1 thay compose `avg` uniform: cần --kg-mode compose, src avg, không idf, không gate')
        import torch
        import torch.nn.functional as F
        orig = KS.compose_semantics
        info = {}

        def compose_signed(kg, prim_emb, text_dim_total, weighting='uniform'):
            w = torch.from_numpy(signed_compose_weights(kg, a.forbid_lam, bool(a.forbid_order_only))).to(prim_emb.device, prim_emb.dtype)
            comp = torch.einsum('np,ptd->ntd', w, prim_emb)
            info['plain'] = orig(kg, prim_emb, prim_emb.shape[-1], 'uniform')   # (n, T, 768) for the log
            info['signed'] = comp
            info['n_forbid_classes'] = int((np.abs(signed_compose_weights(kg, a.forbid_lam, bool(a.forbid_order_only))
                                                   - kg.compose_weights()).sum(1) > 1e-6).sum())
            return torch.cat([comp] * (text_dim_total // comp.shape[-1]), dim=-1)

        KS.compose_semantics = compose_signed      # trainer imports it from kg_semantics at call time
        try:
            super().load_prototypes()
        finally:
            KS.compose_semantics = orig
        self.kg_info = info                         # nesy4_1_1_fuse.py re-fuses from these

        def centered(x):
            z = x[:, -1, :].float()
            z = F.normalize(z - z.mean(0, keepdim=True), dim=-1)
            return z @ z.t()

        pairs = order_only_pairs(a.kg_rules, info['signed'].shape[0])
        cp, cs = centered(info['plain']), centered(info['signed'])
        msg = (f'[nesy4_1] forbid compose: lam {a.forbid_lam}, order_only {a.forbid_order_only}, '
               f'{info["n_forbid_classes"]} classes changed, {len(pairs)} order-only class pairs')
        if pairs:
            mp = np.mean([cp[i, j].item() for i, j in pairs])
            ms = np.mean([cs[i, j].item() for i, j in pairs])
            unseen = set(int(u) for u in self.unseen_classes)
            hit = [(i, j) for i, j in pairs if i in unseen or j in unseen]
            msg += f'; centered cosine of these pairs {mp:.4f} (compose) -> {ms:.4f} (forbid); pairs touching unseen: {hit}'
        self.logger.info(msg)


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_forbid', ForbidTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1],
                                            help='1 = only order forbids (seq.*), 0 = every forbid')),
           ],
           label_keys=[('forbid_lam', 'lam'), ('forbid_order_only', 'ord')])
