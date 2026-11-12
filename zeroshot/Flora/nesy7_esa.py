"""Hướng 7: ESA (Edge-Set Attention) trên đồ thị con của từng lớp, thay cho R-GCN của nesy6_gcn.py.

Mỗi lớp c có một đồ thị con nhỏ rút từ KG v2seq; MỖI CẠNH là một token = [CLIP(nguồn) ; CLIP(tên cạnh) ; CLIP(đích)]
(3 x 768 = 2304 -> LayerNorm -> Linear -> --esa-dim). Cạnh (triple) của lớp c:
  (lớp, "requires", p)   p thuộc require(c)          nguồn là một vector học được CHUNG cho mọi lớp
  (lớp, "forbids",  q)   q thuộc forbid(c)           (không dùng embedding tên lớp, để khó học thuộc danh tính lớp seen)
  (s, "first step", a)   s = seq.a>b thuộc require(c)
  (s, "second step", b)
Tên cạnh được CLIP-text mã hoá một lần (4 câu ngắn). Hai cạnh "kề" nhau nếu chung một nút (nguồn hoặc đích).

Bộ mã hoá (đúng như ESA): các khối xen kẽ attention CÓ MẶT NẠ (chỉ giữa các cạnh kề nhau) và attention thường
(mọi cạnh với nhau), pre-LayerNorm + FFN; sau đó PMA (pooling by multihead attention, một seed học được) gộp tập cạnh
thành một embedding đồ thị --esa-dim chiều. Chiếu tuyến tính lên 768 (khởi tạo 0) rồi CỘNG VÀO compose:
    class tokens = compose(primitive CLIP tokens) + delta_c  (delta_c cộng vào mọi token của lớp c)
Vì chiếu cuối khởi tạo 0 nên lúc đầu đúng bằng compose (như nesy6). Phần còn lại giống nesy6: semantics = attuned +
kg_beta * (đầu ra này), mạng chỉ huấn luyện ở phase 1 rồi đóng băng, hàm mục tiêu thêm gcn_sim * L_sim (chỉ dựa luật KG).

Chạy: python nesy7_esa.py --feat synse --tasks 48 --seeds 1        (none/compose baseline tự chạy nếu chưa có)
      python nesy7_esa.py --tasks 55 48 110 96 --seeds 1 2 3
      python nesy7_esa.py --esa-layers 3 --esa-dim 128
Trong notebook: !python nesy7_esa.py --tasks 48 --seeds 1
Log có dòng `[nesy6] GCN frozen ...` (dùng chung với nesy6): độ lệch so với compose, cosine cặp đảo chiều, L_sim.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nesy_common as C
import nesy6_gcn as G
from kg_semantics import embed_texts

REL_TEXT = ['this action requires', 'this action must not involve', 'has as its first step', 'has as its second step']
REQ, FORB, FIRST, SECOND = range(4)


def build_triples(kg):
    """Per class a list of (src, rel, dst) over primitive indices; src == P is the shared 'this action' node."""
    P = len(kg.prim_ids)
    idx = {p: i for i, p in enumerate(kg.prim_ids)}
    out = []
    for c in range(kg.W.shape[0]):
        req = np.nonzero(kg.W[c] > 0)[0]
        t = [(P, REQ, p) for p in req] + [(P, FORB, q) for q in np.nonzero(kg.W[c] < 0)[0]]
        for p in req:
            pid = kg.prim_ids[p]
            if pid.startswith('seq.') and '>' in pid:
                a, b = pid[4:].split('>', 1)
                if a in idx and b in idx:
                    t += [(p, FIRST, idx[a]), (p, SECOND, idx[b])]
        out.append(t)
    return out


class Block(nn.Module):
    """Pre-LN transformer block; mask (B, 1|h, M, M or 1) boolean, True = may attend."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mask):
        B, M, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, M, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, M, d))
        return x + self.ff(self.n2(x))


class PMA(nn.Module):
    """Pooling by multihead attention with one learned seed: (B, M, d) edge tokens -> (B, d)."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.n = nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x, valid):
        B, M, d = x.shape
        q = self.q(self.seed).expand(B, -1, -1).view(B, 1, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(self.n(x)).view(B, M, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.o(a.transpose(1, 2).reshape(B, d))


class SeqKGESA(nn.Module):
    def __init__(self, kg, rel_emb, dim=768, d_model=256, layers=2, heads=4, edge_drop=0.1):
        super().__init__()
        P = len(kg.prim_ids)
        trip = build_triples(kg)
        n, M = len(trip), max(len(t) for t in trip)
        src, rel, dst = (torch.zeros(n, M, dtype=torch.long) for _ in range(3))
        valid = torch.zeros(n, M, dtype=torch.bool)
        for c, t in enumerate(trip):
            for j, (s, r, d) in enumerate(t):
                src[c, j], rel[c, j], dst[c, j], valid[c, j] = int(s), int(r), int(d), True
        # edges are adjacent if they share a node (src ids live in 0..P, dst ids in 0..P-1)
        adj = ((src[:, :, None] == src[:, None, :]) | (src[:, :, None] == dst[:, None, :]) |
               (dst[:, :, None] == src[:, None, :]) | (dst[:, :, None] == dst[:, None, :]))
        for name, t in dict(src=src, rel=rel, dst=dst, valid=valid, adj=adj, rel_emb=rel_emb.float(),
                            has_edges=valid.any(1).float(), eye=torch.eye(M, dtype=torch.bool)).items():
            self.register_buffer(name, t)
        self.P, self.edge_drop = P, edge_drop
        self.cls_node = nn.Parameter(torch.zeros(dim))
        self.inp = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, d_model))
        self.blocks = nn.ModuleList([Block(d_model, heads) for _ in range(layers)])
        self.pma = PMA(d_model, heads)
        self.out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                 # start exactly at compose
        self.describe = (f'ESA {layers} blocks (masked/full alternating), {heads} heads, d={d_model}, '
                         f'{n} class graphs, <= {M} edge tokens, edge drop {edge_drop}')

    def forward(self, xp):
        """xp: (P, T, D) CLIP tokens of the primitive descriptions -> (class tokens (n, T, D), compose baseline (n, T, D))."""
        valid = self.valid
        if self.training and torch.is_grad_enabled() and self.edge_drop > 0:
            kept = valid & (torch.rand(valid.shape, device=valid.device) > self.edge_drop)
            valid = torch.where(kept.any(1, keepdim=True), kept, valid)           # never leave a class without edges
        nodes = torch.cat([xp[:, -1, :].float(), self.cls_node[None]], 0)         # (P+1, D); last row = 'this action'
        x = self.inp(torch.cat([nodes[self.src], self.rel_emb[self.rel], nodes[self.dst]], dim=-1))
        ok = valid.clone()
        ok[:, 0] |= ~ok.any(1)                                                    # a class with no triple at all: keep softmax finite
        masked = ((self.adj & ok[:, None, :]) | self.eye)[:, None]                # (n, 1, M, M)
        full = ok[:, None, None, :]                                               # (n, 1, 1, M)
        for i, blk in enumerate(self.blocks):
            x = blk(x, masked if i % 2 == 0 else full)
        delta = self.out(self.pma(x, ok)) * self.has_edges[:, None]               # (n, D)
        # compose over the same (dropped) require edges
        cnt = torch.zeros(valid.shape[0], self.P, device=valid.device).scatter_add_(1, self.dst, (valid & (self.rel == REQ)).float())
        base = torch.einsum('np,ptd->ntd', cnt / cnt.sum(1, keepdim=True).clamp_min(1e-8), xp)
        return base + delta[:, None, :].to(base.dtype), base


class ESATrainer(G.GCNTrainer):
    def build_net(self, kg, a):
        m = self.mapping_model
        rel = embed_texts(m.tokenizer, m.text_encoder, REL_TEXT, a.max_length, 'cuda', None)[:, -1, :]
        return SeqKGESA(kg, rel.to(self.kg_prim_emb.device), self.kg_prim_emb.shape[-1],
                        a.esa_dim, a.esa_layers, a.esa_heads, a.gcn_edge_drop)


if __name__ == '__main__':
    C.main('esa', ESATrainer, default_kg_mode='none', prepare=G._prepare, defaults=dict(gcn_lr=1e-4),
           method_args=G.COMMON_ARGS + [
               ('--esa-dim', dict(type=int, default=256, help='width of edge tokens / graph embedding')),
               ('--esa-layers', dict(type=int, default=2, help='ESA blocks, alternating masked and full attention')),
               ('--esa-heads', dict(type=int, default=4)),
           ],
           label_keys=[('esa_layers', 'L'), ('esa_dim', 'd'), ('gcn_lr', 'lr'), ('gcn_sim', 'sim')])
