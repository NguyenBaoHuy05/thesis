"""Hướng 6: mạng GCN quan hệ (R-GCN) học trên đồ thị SeqKG, thay cho `compose` cố định.

Đồ thị hai phía lớp - primitive (KG v2seq, 111 primitive), nút primitive khởi tạo bằng embedding CLIP của
câu mô tả (cùng cache với compose). Các loại cạnh:
  require  lớp -> primitive         forbid   lớp -> primitive
  first    seq.a>b <- a             second   seq.a>b <- b      (nối primitive thứ tự với hai primitive của nó)
  cooc     primitive <-> primitive  cùng được một lớp `require`
  excl     primitive <-> primitive  một lớp `require` p và `forbid` q

Mạng (mọi trọng số khởi tạo bằng 0, nên LÚC ĐẦU ĐÚNG BẰNG compose):
  làm mịn primitive (--gcn-layers lần): h_p += sum_r W_r . mean_{q ~r p} h_q     r thuộc {first, second, cooc, excl}
  đọc ra lớp:  h_c = A_req h + W_req (A_req h) + W_forb (A_forb h)               A: trung bình theo hàng
Mỗi W là Linear 768 -> 768 không bias, dùng chung cho mọi token CLIP, nên đầu ra vẫn là (n, token, 768)
và được lặp ×2 lên 1536 như compose. Nút lớp không dùng embedding tên lớp (chỉ từ primitive của nó), để
khó học thuộc danh tính lớp seen;.

Gắn vào Flora: semantics = attuned (Flora thuần, không KG) + kg_beta * GCN(KG). Giống kg_gate của Flora:
GCN chỉ được huấn luyện ở phase 1, sang phase 2 đóng băng và dùng đầu ra cố định.
Chống học thuộc: dropout cạnh lớp-primitive, weight decay, lr riêng nhỏ, và phạt độ lệch tương đối so
với compose (--gcn-reg).

Chạy: python nesy6_gcn.py --feat synse --tasks 96 --seeds 1        (none/compose baseline tự chạy nếu chưa có)
      python nesy6_gcn.py --feat synse --tasks 55 48 110 96 --seeds 1 2 3
      python nesy6_gcn.py --gcn-layers 0                              (chỉ đọc ra, không làm mịn primitive)
      python nesy6_gcn.py --gcn-reg 0 --gcn-lr 1e-4                   (thả lỏng, dễ học thuộc hơn)
Trong notebook: !python nesy6_gcn.py --feat synse --tasks 96 --seeds 1
"""
import json
import os

import nesy_common as C

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import Trainer
from kg_semantics import PrimitiveKG, embed_texts, mask_half

SEQ_PRIMS, SEQ_RULES = 'kg/primitive_v2seq.json', 'kg/kg_rules_v2seq.json'


def _row_norm(m):
    return m / np.maximum(m.sum(axis=1, keepdims=True), 1e-8)


class SeqKGGCN(nn.Module):
    def __init__(self, kg, dim=768, layers=1, edge_drop=0.1, use_order=True):
        super().__init__()
        W = kg.W
        P = W.shape[1]
        a_req = _row_norm(np.clip(W, 0, None))
        a_forb = _row_norm((W < 0).astype(np.float32))
        idx = {p: i for i, p in enumerate(kg.prim_ids)}
        first = np.zeros((P, P), dtype=np.float32)
        second = np.zeros((P, P), dtype=np.float32)
        for pid, s in idx.items():
            if pid.startswith('seq.') and '>' in pid:
                a, b = pid[4:].split('>', 1)
                if a in idx and b in idx:
                    first[s, idx[a]] = 1.0
                    second[s, idx[b]] = 1.0
        rels = {'cooc': _row_norm(a_req.T @ a_req), 'excl': _row_norm(a_req.T @ a_forb)}
        if use_order:
            rels = {'first': first, 'second': second, **rels}
        self.rel_names = list(rels)
        self.register_buffer('A_req', torch.from_numpy(a_req))
        self.register_buffer('A_forb', torch.from_numpy(a_forb))
        for name, m in rels.items():
            self.register_buffer(f'R_{name}', torch.from_numpy(m.astype(np.float32)))
        self.refine = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in self.rel_names]) for _ in range(layers)])
        self.w_req = nn.Linear(dim, dim, bias=False)
        self.w_forb = nn.Linear(dim, dim, bias=False)
        for p in self.parameters():
            nn.init.zeros_(p)            # start exactly at compose
        self.edge_drop = edge_drop

    def _drop(self, A):
        keep = (torch.rand_like(A) > self.edge_drop) & (A > 0)
        A2 = A * keep
        A2 = torch.where(A2.sum(1, keepdim=True) == 0, A, A2)     # never leave a class without edges
        return A2 / A2.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, xp):
        """xp: (P, T, D) CLIP tokens of the primitive descriptions -> (class tokens (n, T, D), compose baseline (n, T, D))."""
        drop = self.training and torch.is_grad_enabled() and self.edge_drop > 0
        a_req = self._drop(self.A_req) if drop else self.A_req
        a_forb = self._drop(self.A_forb) if drop else self.A_forb
        h = xp
        for layer in self.refine:
            agg = 0
            for name, lin in zip(self.rel_names, layer):
                agg = agg + lin(torch.einsum('pq,qtd->ptd', getattr(self, f'R_{name}'), h))
            h = h + agg
        base = torch.einsum('np,ptd->ntd', a_req, xp)           # plain compose with the same (dropped) edges
        req = torch.einsum('np,ptd->ntd', a_req, h)
        forb = torch.einsum('np,ptd->ntd', a_forb, h)
        return req + self.w_req(req) + self.w_forb(forb), base


class GCNTrainer(Trainer):
    def load_prototypes(self):
        super().load_prototypes()                     # Flora attuned semantics; --kg-mode none -> no KG inside
        a = self.args
        kg = PrimitiveKG(a.gcn_prims, a.gcn_rules, self.num_classes, a.kg_forbid_weight)
        m = self.mapping_model
        stem = os.path.splitext(os.path.basename(a.gcn_prims))[0]
        cache = os.path.join(os.path.dirname(os.path.abspath(a.gcn_prims)), f'{stem}_clip_L{a.max_length}.pt')
        dev, dt = self.attuned_semantics.device, self.attuned_semantics.dtype
        self.kg_prim_emb = embed_texts(m.tokenizer, m.text_encoder, kg.prim_text, a.max_length, 'cuda', cache).to(dev, dt)
        self.gcn = SeqKGGCN(kg, self.kg_prim_emb.shape[-1], a.gcn_layers, a.gcn_edge_drop, bool(a.gcn_order)).to(dev)
        self.optimizer_mapping.add_param_group(dict(params=list(self.gcn.parameters()), lr=a.gcn_lr, weight_decay=a.gcn_wd))
        self.gcn_cache = None
        self.gcn_pen = None
        # pairs of classes that differ only by their order primitives (what SeqKG was built to separate)
        strip = lambda ids: tuple(sorted(i for i in ids if not i.startswith('seq.')))
        rules = {r['id'] - 1: r for r in json.load(open(a.gcn_rules, encoding='utf-8'))['rules']}
        by_key = {}
        for c in range(self.num_classes):
            by_key.setdefault((strip(rules[c]['require']), strip(rules[c]['forbid'])), []).append(c)
        self.gcn_pairs = [v for v in by_key.values() if len(v) == 2]
        # phase 1 loss + relative deviation from compose
        orig = m.forward_phase1

        def forward_phase1(*args, **kw):
            out = orig(*args, **kw)
            if self.gcn_pen is not None and a.gcn_reg > 0:
                out['loss'] = out['loss'] + a.gcn_reg * self.gcn_pen
            return out
        m.forward_phase1 = forward_phase1
        n_par = sum(p.numel() for p in self.gcn.parameters())
        self.logger.info(f'[nesy6] SeqKG GCN: {len(kg.prim_ids)} primitives, relations {self.gcn.rel_names}, '
                         f'{a.gcn_layers} refine layer(s), {n_par / 1e6:.2f}M params, lr {a.gcn_lr}, reg {a.gcn_reg}, '
                         f'edge_drop {a.gcn_edge_drop}, {len(self.gcn_pairs)} order-only class pairs')

    def _to_width(self, x):
        reps = self.attuned_semantics.shape[-1] // x.shape[-1]
        return mask_half(torch.cat([x] * reps, dim=-1), self.args.kg_compose_half)

    def kg_semantics(self):
        if self.gcn_cache is not None:
            return self.gcn_cache
        out, base = self.gcn(self.kg_prim_emb)
        self.gcn_pen = ((out - base) ** 2).mean() / (base ** 2).mean().clamp_min(1e-8)
        return self._to_width(out)

    def current_attuned_semantics(self, return_gate=False):
        sem = self.attuned_semantics + self.args.kg_beta * self.kg_semantics()
        return (sem, None) if return_gate else sem

    def train_learning_phase(self):
        super().train_learning_phase()
        # freeze: phase 2 and every evaluation use the final GCN output
        self.gcn.eval()
        with torch.no_grad():
            out, base = self.gcn(self.kg_prim_emb)
            self.gcn_cache = self._to_width(out)
            rel = ((out - base) ** 2).mean().sqrt() / (base ** 2).mean().sqrt().clamp_min(1e-8)
            msg = f'[nesy6] GCN frozen. relative change vs compose = {rel.item():.4f}'
            if self.gcn_pairs:
                cos = lambda x, i, j: F.cosine_similarity(x[i, -1], x[j, -1], dim=-1).item()
                before = np.mean([cos(base, i, j) for i, j in self.gcn_pairs])
                after = np.mean([cos(out, i, j) for i, j in self.gcn_pairs])
                msg += f'; cosine of order-only class pairs {before:.4f} (compose) -> {after:.4f} (GCN)'
        for p in self.gcn.parameters():
            p.requires_grad_(False)
        self.logger.info(msg)


def _prepare(opts):
    if not (os.path.exists(opts.gcn_prims) and os.path.exists(opts.gcn_rules)):
        from nesy4_seqkg import build_seq_kg
        build_seq_kg()


if __name__ == '__main__':
    C.main('gcn', GCNTrainer, default_kg_mode='none', prepare=_prepare,
           method_args=[
               ('--gcn-prims', dict(default=SEQ_PRIMS, help='KG primitives the GCN runs on (nesy4_seqkg.py writes this file)')),
               ('--gcn-rules', dict(default=SEQ_RULES)),
               ('--gcn-layers', dict(type=int, default=1, help='primitive refinement layers before the class read-out (0 = read-out only)')),
               ('--gcn-order', dict(type=int, default=1, choices=[0, 1], help='1 = add first/second edges from seq.a>b to a and b')),
               ('--gcn-edge-drop', dict(type=float, default=0.1, help='dropout of class-primitive edges while training')),
               ('--gcn-lr', dict(type=float, default=1e-5, help='learning rate of the GCN (Flora uses 1e-4)')),
               ('--gcn-wd', dict(type=float, default=0.01, help='weight decay of the GCN, pulls the weights back to compose')),
               ('--gcn-reg', dict(type=float, default=1.0, help='weight of the relative deviation from compose added to the phase 1 loss')),
           ],
           label_keys=[('gcn_layers', 'L'), ('gcn_lr', 'lr'), ('gcn_reg', 'reg')])
