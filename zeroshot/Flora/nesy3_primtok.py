"""Hướng 3: primitive làm token riêng trong chuỗi ngữ nghĩa của lớp.

`compose` lấy trung bình token của nhiều câu primitive khác nhau theo vị trí, rồi cộng vào chuỗi của lớp,
nên token bị trộn lệch vị trí. Ở đây mỗi primitive `require` của lớp c trở thành MỘT token riêng
(embedding pooled của CLIP cho câu mô tả primitive, cùng không gian với token của lớp), được NỐI vào
cuối chuỗi ngữ nghĩa sau khi trộn láng giềng:
    attuned[c] = [ tokens của Flora (max_length+1) ; K token primitive ]
DiT của Flora không có positional embedding và encoder/decoder chạy theo từng token, nên chuỗi dài hơn
không cần sửa kiến trúc. Skeleton được nhân bản theo args.max_length + 1 (models/fm.py), nên chỉ cần
tăng args.max_length thêm K sau khi đã trích embedding văn bản.
Lớp có ít hơn K primitive: các slot thừa lấy trung bình các primitive của nó (không thêm thông tin mới).
Vector 768-d được lặp vào cả hai nửa [csv ; llm] của 1536-d, nhân --pt-scale.

Mặc định chạy trên Flora thuần (--kg-mode none), vì đây là thay thế cho `compose`.

Chạy: python nesy3_primtok.py
      python nesy3_primtok.py --kg-mode compose --pt-scale 0.5
"""
import os

import nesy_common as C

import torch

from trainer import Trainer
from kg_semantics import PrimitiveKG, embed_texts


class PrimTokTrainer(Trainer):
    def load_prototypes(self):
        super().load_prototypes()        # Flora attuned semantics (+ compose if --kg-mode compose)
        a = self.args
        kg = PrimitiveKG(a.kg_prims, a.kg_rules, self.num_classes, a.kg_forbid_weight)
        m = self.mapping_model
        stem = os.path.splitext(os.path.basename(a.kg_prims))[0]
        cache = os.path.join(os.path.dirname(os.path.abspath(a.kg_prims)), f'{stem}_clip_L{a.max_length}.pt')
        prim_emb = embed_texts(m.tokenizer, m.text_encoder, kg.prim_text, a.max_length, 'cuda', cache)
        pooled = prim_emb[:, -1, :].to(self.attuned_semantics.device, self.attuned_semantics.dtype)   # (P, 768)
        req = [torch.nonzero(torch.from_numpy(kg.W[c] > 0)).flatten().tolist() for c in range(self.num_classes)]
        K = a.pt_max_tokens if a.pt_max_tokens > 0 else max(len(r) for r in req)
        n, _, width = self.attuned_semantics.shape
        tok = torch.zeros(n, K, pooled.shape[1], device=pooled.device, dtype=pooled.dtype)
        for c, ids in enumerate(req):
            ids = ids[:K]
            if not ids:
                tok[c] = self.attuned_semantics[c, -1, :pooled.shape[1]]   # no primitive: repeat the class's own pooled token
                continue
            tok[c, :len(ids)] = pooled[ids]
            tok[c, len(ids):] = pooled[ids].mean(0)
        tok = torch.cat([tok] * (width // pooled.shape[1]), dim=-1) * a.pt_scale
        self.attuned_semantics = torch.cat([self.attuned_semantics, tok], dim=1)
        # skeleton tokens are expanded to max_length + 1 in MappingNet: grow it to the new sequence length.
        # Text embeddings were already extracted, so nothing else depends on max_length any more.
        a.max_length += K
        assert self.attuned_semantics.shape[1] == a.max_length + 1
        self.logger.info(f'[nesy3] appended {K} primitive tokens per class (scale {a.pt_scale}); '
                         f'sequence length now {a.max_length + 1}')


if __name__ == '__main__':
    C.main('primtok', PrimTokTrainer, default_kg_mode='none',
           method_args=[
               ('--pt-scale', dict(type=float, default=1.0, help='scale of the appended primitive tokens')),
               ('--pt-max-tokens', dict(type=int, default=0, help='K; 0 = max #require over classes')),
           ],
           label_keys=[('pt_scale', 's'), ('pt_max_tokens', 'K')])
