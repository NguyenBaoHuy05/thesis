"""Hướng 4.1.1.4 (cải tiến từ nesy4_1_1_fuse.py): câu primitive có ngữ cảnh KHÔNG chứa tên lớp.

COA-Fuse hiện nhúng CLIP từng câu primitive trần: "A hand is behind the trunk."
Ở đây mỗi câu được bọc trong một mẫu cố định, giống nhau cho mọi primitive, KHÔNG chứa tên lớp:
  --prim-ctx none     câu trần (giống hệt nesy4_1_1_fuse.py, dùng làm kiểm tra)
  --prim-ctx action   "In this action, a hand is behind the trunk."
  --prim-ctx person   "A person is performing an action in which a hand is behind the trunk."
Câu thứ tự "First, X. Then, Y." đổi thành "first, x; then, y" cho vừa mẫu và vẫn là một câu.
Vì mẫu dùng chung, primitive vẫn là đơn vị dùng chung giữa lớp seen và unseen (khác với việc đưa tên lớp vào câu).
Mọi thứ khác (Signed Compose, Center, Orthogonal, Adaptive, beta) giữ nguyên, nên chênh lệch điểm là do ngữ cảnh.

Cache embedding: <prim>_clip_L35_ctx-<mode>.pt cạnh file cache cũ (không đè lên cache của bản trần).
Log `[nesy4_1_1_4]`: số câu bị cắt vì vượt --max-length (35 token), cosine giữa embedding bọc và trần
(trung bình theo primitive, token cuối), và với k_c của mỗi lớp: cosine cùng lớp (đã trừ trung bình lớp) giữa bản trần và bản bọc,
cosine trung bình giữa các lớp (đã trừ trung bình) của k trần / k bọc (thấp hơn = các lớp tách nhau hơn).

Chạy: python nesy4_1_1_4_ctx.py --prim-ctx action --tasks 48 110 --seeds 1 2 3 --no-baselines
      python nesy4_1_1_4_ctx.py --prim-ctx person --tasks 48 110 --seeds 1 2 3 --no-baselines
Tiêu chí đặt trước: chỉ giữ nếu hơn COA-Fuse (3 seed) >= +1 điểm ở cả 48/12 và 110/10.
"""
import os
import re

import nesy_common as C

import numpy as np

import kg_semantics as KS
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_forbid import signed_compose_weights
from nesy4_1_1_fuse import FuseTrainer

PREFIX = {'action': 'In this action, ',
          'person': 'A person is performing an action in which '}


def lower1(s):
    return s[:1].lower() + s[1:]


def wrap_primitive(desc, mode):
    """Contextualise a primitive sentence without any class name."""
    if mode == 'none':
        return desc
    m = re.match(r'^First,\s*(.*?)\.\s*Then,\s*(.*?)\.?$', desc.strip())
    body = f'first, {lower1(m.group(1))}; then, {lower1(m.group(2))}' if m else lower1(desc.strip().rstrip('.'))
    return PREFIX[mode] + body + '.'


class CtxTrainer(FuseTrainer):
    def load_prototypes(self):
        a = self.args
        mode = a.prim_ctx
        orig = KS.embed_texts
        store = {}

        def embed_ctx(tokenizer, text_encoder, texts, max_length, device, cache_path=None):
            plain = orig(tokenizer, text_encoder, texts, max_length, device, cache_path)
            if mode == 'none' or len(texts) != len(plain):
                return plain
            wrapped = [wrap_primitive(t, mode) for t in texts]
            cache2 = cache_path.replace('.pt', f'_ctx-{mode}.pt') if cache_path else None
            ctx = orig(tokenizer, text_encoder, wrapped, max_length, device, cache2)
            n_cut = sum(len(tokenizer(t, truncation=False).input_ids) > max_length for t in wrapped)
            store.update(plain=plain, ctx=ctx, n_cut=n_cut, n=len(wrapped), example=wrapped[0], example_seq=next((w for w in wrapped if 'then,' in w), ''))
            return ctx

        KS.embed_texts = embed_ctx      # trainer imports it from kg_semantics at call time
        try:
            super().load_prototypes()
        finally:
            KS.embed_texts = orig

        if mode == 'none' or not store:
            self.logger.info(f'[nesy4_1_1_4] prim_ctx {mode}: no change')
            return
        import torch
        import torch.nn.functional as F
        from kg_semantics import PrimitiveKG
        plain, ctx = store['plain'].float().cpu(), store['ctx'].float().cpu()
        pc = F.cosine_similarity(plain[:, -1, :], ctx[:, -1, :], dim=-1)
        n = self.text_similarity.shape[0]
        kg = PrimitiveKG(a.kg_prims, a.kg_rules, n, a.kg_forbid_weight)
        w = torch.from_numpy(signed_compose_weights(kg, a.forbid_lam, bool(a.forbid_order_only))).float()
        kp = torch.einsum('np,pd->nd', w, plain[:, -1, :])
        kc = torch.einsum('np,pd->nd', w, ctx[:, -1, :])

        def cen(x):
            return F.normalize(x - x.mean(0, keepdim=True), dim=-1)

        def offdiag(x):
            c = cen(x) @ cen(x).t()
            return c[~torch.eye(c.shape[0], dtype=torch.bool)].mean().item()
        same = (cen(kp) * cen(kc)).sum(-1).mean().item()
        self.logger.info(
            f'[nesy4_1_1_4] prim_ctx {mode}: {store["n_cut"]}/{store["n"]} sentences exceed max_length (truncated); '
            f'plain vs wrapped primitive cosine mean {pc.mean():.3f} min {pc.min():.3f}; '
            f'per-class k centered cosine plain vs wrapped {same:.3f}; '
            f'mean off-diagonal centered cosine across classes plain {offdiag(kp):.4f} / wrapped {offdiag(kc):.4f}; '
            f'example: "{store["example"]}" | "{store["example_seq"]}"')


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_forbid_fuse_ctx', CtxTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-center', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-orth', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-adaptive', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-clip', dict(type=float, default=3.0)),
               ('--prim-ctx', dict(type=str, default='action', choices=['none', 'action', 'person'],
                                   help='context template wrapped around each primitive sentence (no class name)')),
           ],
           label_keys=[('prim_ctx', 'ctx'), ('fuse_center', 'c'), ('fuse_orth', 'o'), ('fuse_adaptive', 'a')])
