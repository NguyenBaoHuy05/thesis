"""Đối chứng nesy0 (xây trên nesy4_1_1_fuse.py): KG xáo trộn.

Câu hỏi: phần tăng điểm của KG đến từ NỘI DUNG KG (primitive đúng của từng lớp) hay chỉ từ việc cộng thêm một vector
nhiễu/chung (Center, beta, độ lớn)? Cách đo: giữ nguyên toàn bộ cơ chế, chỉ hoán vị các HÀNG của KG giữa các lớp
(lớp c nhận luật require/forbid của lớp pi(c)). Cùng số primitive, cùng phân bố trọng số, cùng bước Center/Orth/Adaptive,
nhưng KG không còn đúng với lớp.

  --shuffle-kg 1        hoán vị hàng W của KG (mọi lớp; hoán vị không có điểm cố định)
  --shuffle-kg 0        KG thật (trùng nesy4_1_1_fuse.py khi cùng cờ fuse; dùng để kiểm tra)
  --shuffle-seed S      hoán vị phụ thuộc (seed chạy + 1000*S); mỗi seed chạy là một hoán vị khác nhau
                        nên trung bình 3 seed đã bao gồm độ may rủi của hoán vị
Hoán vị áp dụng cho MỌI chỗ dùng KG: trọng số compose, trọng số forbid thứ tự, độ giống KG của bước Adaptive.
Cờ --fuse-center/--fuse-orth/--fuse-adaptive như nesy4_1_1_fuse.py nên có thể đối chứng riêng từng thành phần.

Cách đọc kết quả (so cùng cờ fuse, KG thật vs KG xáo, 3 seed):
  - KG xáo ~ KG thật (chênh < ~0.5, trong độ lệch chuẩn): phần tăng KHÔNG đến từ nội dung KG (do Center/beta/nhiễu).
  - KG thật > KG xáo rõ ràng và đều ở các seed: nội dung KG có đóng góp thật; chênh lệch là phần của KG.
  - KG xáo < Flora không KG: KG sai làm hại, tức mô hình có đọc KG (đây là bằng chứng KG được sử dụng).
Log `[nesy0]`: số lớp bị đổi, ví dụ ánh xạ, tương quan Pearson giữa độ giống text CLIP và độ giống KG
(thật: dương rõ; xáo: ~0).

Chạy (xáo trộn; số KG thật đã có từ các lần chạy trước):
  python nesy0_shuffle.py --shuffle-kg 1 --fuse-orth 0 --fuse-adaptive 0 --tasks 48 110 --seeds 1 2 3 --no-baselines   (chỉ Center)
  python nesy0_shuffle.py --shuffle-kg 1 --fuse-center 0 --fuse-orth 0 --fuse-adaptive 0 --tasks 48 110 --seeds 1 2 3 --no-baselines  (cộng thô)
  python nesy0_shuffle.py --shuffle-kg 1 --tasks 48 110 --seeds 1 2 3 --no-baselines                                      (COA-Fuse đầy đủ)
"""
import json
import os

import nesy_common as C

import numpy as np

import kg_semantics as KS
import nesy4_1_1_fuse as FU
from nesy4_seqkg import build_seq_kg, DST_PRIMS, DST_RULES
from nesy4_1_1_fuse import FuseTrainer

REAL_KG = KS.PrimitiveKG


def derangement(n, rng):
    """Random permutation of range(n) with no fixed point."""
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def make_shuffled_kg(seed):
    class ShuffledKG(REAL_KG):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            n = self.W.shape[0]
            self.perm = derangement(n, np.random.RandomState(seed))
            self.W_real = self.W
            self.W = self.W[self.perm]              # class c now carries the rules of class perm[c]
    return ShuffledKG


class ShuffleTrainer(FuseTrainer):
    def load_prototypes(self):
        a = self.args
        if a.shuffle_kg:
            cls = make_shuffled_kg(int(a.seed) + 1000 * int(a.shuffle_seed))
            KS.PrimitiveKG = cls          # Trainer.load_prototypes imports it from kg_semantics at call time
            FU.PrimitiveKG = cls          # nesy4_1_1_fuse imported the name at module level (Adaptive step)
        try:
            super().load_prototypes()
        finally:
            KS.PrimitiveKG = REAL_KG
            FU.PrimitiveKG = REAL_KG

        n = self.text_similarity.shape[0]
        names = {r['id'] - 1: r.get('class', str(r['id'])) for r in json.load(open(a.kg_rules, encoding='utf-8'))['rules']}
        real = REAL_KG(a.kg_prims, a.kg_rules, n, a.kg_forbid_weight)
        off = ~np.eye(n, dtype=bool)
        ts = self.text_similarity.detach().cpu().float().numpy()

        def corr(S):
            return float(np.corrcoef(ts[off], S[off])[0, 1])
        if a.shuffle_kg:
            shuf = make_shuffled_kg(int(a.seed) + 1000 * int(a.shuffle_seed))(a.kg_prims, a.kg_rules, n, a.kg_forbid_weight)
            unseen = set(int(u) for u in self.unseen_classes)
            same_group = float(np.mean([(int(c) in unseen) == (int(shuf.perm[c]) in unseen) for c in range(n)]))
            ex = ', '.join(f'{names.get(c, c)} <- {names.get(int(shuf.perm[c]), int(shuf.perm[c]))}' for c in sorted(unseen)[:4])
            self.logger.info(
                f'[nesy0] SHUFFLED KG (shuffle_seed {a.shuffle_seed}, run seed {a.seed}): {n} classes permuted, 0 fixed points; '
                f'Pearson(text sim, KG sim) real {corr(real.similarity()):.3f} -> shuffled {corr(shuf.similarity()):.3f}; '
                f'fraction of classes whose rules come from a class of the same seen/unseen group {same_group:.2f}; '
                f'unseen examples (class <- rules of): {ex}')
        else:
            self.logger.info(f'[nesy0] real KG; Pearson(text sim, KG sim) {corr(real.similarity()):.3f}')


if __name__ == '__main__':
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()
    C.main('seqkg_shuffle', ShuffleTrainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES),
           method_args=[
               ('--forbid-lam', dict(type=float, default=1.0, help='sem = r + lam * (r - f)')),
               ('--forbid-order-only', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-center', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-orth', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-adaptive', dict(type=int, default=1, choices=[0, 1])),
               ('--fuse-clip', dict(type=float, default=3.0)),
               ('--shuffle-kg', dict(type=int, default=1, choices=[0, 1], help='1 = permute the rows of the KG across classes')),
               ('--shuffle-seed', dict(type=int, default=0, help='permutation seed offset (permutation also depends on the run seed)')),
           ],
           label_keys=[('shuffle_kg', 'shuf'), ('shuffle_seed', 'ss'), ('fuse_center', 'c'), ('fuse_orth', 'o'), ('fuse_adaptive', 'a')])
