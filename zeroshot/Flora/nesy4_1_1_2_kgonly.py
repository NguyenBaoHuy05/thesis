"""Hướng 4.1.1.2 (cải tiến từ nesy4_1_1_fuse.py / nesy4_1_forbid.py): bỏ ngữ nghĩa của Flora, chỉ dùng ngữ nghĩa từ KG.

Flora:   sem_c = A_c (csv + mô tả LLM qua CLIP, trộn láng giềng)           [+ beta * k_c khi có KG]
Ở đây:   sem_c = k_c   (Signed Compose của SeqKG, lặp đủ 1536 chiều)        không còn text của Flora
Dùng làm dòng ablation "KG một mình mang được bao nhiêu thông tin" (Flora text / KG only / cả hai).

  --kgonly-center 1      trừ trung bình theo lớp của k (CLIP rất đẳng hướng; giống bước Center của COA-Fuse)
  --kgonly-neighbors 0   0: không trộn láng giềng;  1: trộn láng giềng đúng như Flora (cùng topk, cùng mixed_weights,
                         độ giống lấy từ text CLIP của lớp nên không đổi khi bỏ text khỏi ngữ nghĩa)
  --kgonly-match-norm 1  đưa độ lớn trung bình của sem về bằng ngữ nghĩa Flora không KG (để đầu vào encoder cùng thang đo)

Bước Orthogonal và Adaptive của COA-Fuse cần text của Flora làm tham chiếu nên không áp dụng ở đây.
kg_beta không còn tác dụng (vẫn được gọi trong Flora/ForbidTrainer rồi bị ghi đè ở cuối load_prototypes).

Log `[nesy4_1_1_2]`: độ lớn, hệ số rescale, cosine trung bình giữa các lớp (token cuối) của ngữ nghĩa Flora và của KG-only,
trước và sau khi trừ trung bình; cosine trung bình giữa ngữ nghĩa KG-only và Flora của cùng một lớp.

Chạy: python nesy4_1_1_2_kgonly.py --tasks 48 110 --seeds 1 2 3 --no-baselines                     (center, không láng giềng)
      python nesy4_1_1_2_kgonly.py --kgonly-neighbors 1 --tasks 48 110 --seeds 1 2 3 --no-baselines
      python nesy4_1_1_2_kgonly.py --kgonly-center 0 --kgonly-match-norm 0                         (KG thô)
"""
import os

import nesy_common as C

import numpy as np

from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_forbid import ForbidTrainer


class KGOnlyTrainer(ForbidTrainer):
    def load_prototypes(self):
        super().load_prototypes()
        import torch
        import torch.nn.functional as F
        a = self.args
        if a.kg_compose_at != 'post' or a.kg_compose_half != 'both':
            raise ValueError('nesy4_1_1_2 cần --kg-compose-at post và --kg-compose-half both')
        flora = self.attuned_semantics                                           # Flora (đã cộng beta * k)
        dev, dt = flora.device, flora.dtype
        k = self.kg_info['signed'].to(dev, dt)                                   # (n, T, 768)
        k = torch.cat([k] * (flora.shape[-1] // k.shape[-1]), dim=-1)            # (n, T, 1536)
        flora0 = flora - a.kg_beta * k                                           # Flora không KG, để so sánh

        base = k - k.mean(0, keepdim=True) if a.kgonly_center else k
        if a.kgonly_neighbors:
            sem = base.clone()
            for j in range(a.topk):
                idx = self.topk_similarity_indices[:, j]
                val = self.topk_similarity_values[:, j].to(dev, dt)
                sem += a.mixed_weights / a.topk * val[:, None, None] * base[idx]
        else:
            sem = base

        norm = lambda x: x.flatten(1).norm(dim=1).mean().item()
        scale = 1.0
        if a.kgonly_match_norm:
            scale = norm(flora0) / max(norm(sem), 1e-8)
            sem = sem * scale
        self.attuned_semantics = sem

        # ---- log
        def cos_stats(x, center):
            z = x[:, -1, :].float()
            if center:
                z = z - z.mean(0, keepdim=True)
            z = F.normalize(z, dim=-1)
            c = z @ z.t()
            off = ~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)
            return c[off].mean().item()
        same = F.cosine_similarity(flora0[:, -1, :].float(), sem[:, -1, :].float(), dim=-1).mean().item()
        self.logger.info(
            f'[nesy4_1_1_2] KG-only semantics: center {a.kgonly_center} neighbors {a.kgonly_neighbors} '
            f'match_norm {a.kgonly_match_norm}; mean norm Flora {norm(flora0):.2f} -> KG-only {norm(sem):.2f} (scale {scale:.3f}); '
            f'mean off-diagonal cosine across classes: raw Flora {cos_stats(flora0, False):.4f} / KG-only {cos_stats(sem, False):.4f}, '
            f'centered Flora {cos_stats(flora0, True):.4f} / KG-only {cos_stats(sem, True):.4f}; '
            f'same-class cosine Flora vs KG-only {same:.4f}')


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_kgonly', KGOnlyTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--kgonly-center', dict(type=int, default=1, choices=[0, 1], help='remove the class-mean of the KG semantics')),
               ('--kgonly-neighbors', dict(type=int, default=0, choices=[0, 1], help='apply Flora neighbour mixing to the KG semantics')),
               ('--kgonly-match-norm', dict(type=int, default=1, choices=[0, 1], help='rescale to the mean norm of Flora semantics without KG')),
           ],
           label_keys=[('kgonly_center', 'c'), ('kgonly_neighbors', 'n'), ('kgonly_match_norm', 'm')])
