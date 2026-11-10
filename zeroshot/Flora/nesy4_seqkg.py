"""Hướng 4: KG có thứ tự (sequence-aware KG).

KG v2 có trường `sequence` (thứ tự primitive) nhưng kg_semantics bỏ qua nó. Vì vậy 7 cặp lớp có
require/forbid giống hệt nhau (mặc/cởi áo, giày, kính, mũ, tai nghe, túi; cho vào/lấy ra khỏi túi) không
phân biệt được, dù `sequence` của chúng ngược nhau.

File này dựng KG v2seq từ v2, KHÔNG viết thêm tri thức mới, chỉ dùng thông tin đã có trong v2:
  - mỗi cặp liên tiếp (a, b) trong `sequence` của một lớp thành một primitive thứ tự `seq.a>b`,
    với mô tả "First, <mô tả a> Then, <mô tả b>", và được thêm vào `require` của lớp đó;
  - nếu thứ tự ngược `seq.b>a` cũng là primitive (lớp khác dùng), nó được thêm vào `forbid`.
Rồi chạy `compose` của Flora với KG mới (so với `compose` dùng KG v2).

Các file khác dùng được KG này: python nesy1_logic.py --kg-prims kg/primitive_v2seq.json --kg-rules kg/kg_rules_v2seq.json

Chạy: python nesy4_seqkg.py                 (dựng KG + chạy compose trên 48/12 và 110/10)
      python nesy4_seqkg.py --build-only
"""
import json
import os

import nesy_common as C

from trainer import Trainer

SRC_PRIMS, SRC_RULES = 'kg/primitive_v2.json', 'kg/kg_rules_v2.json'
DST_PRIMS, DST_RULES = 'kg/primitive_v2seq.json', 'kg/kg_rules_v2seq.json'


def _sentence(desc):
    desc = desc.strip()
    return desc[0].lower() + desc[1:] if desc else desc


def build_seq_kg(verbose=True):
    with open(SRC_PRIMS, encoding='utf-8') as f:
        prims = json.load(f)
    with open(SRC_RULES, encoding='utf-8') as f:
        kg = json.load(f)
    desc = {p['id']: p['description'] for p in prims}
    pairs = []
    for r in kg['rules']:
        for a, b in zip(r['sequence'], r['sequence'][1:]):
            if (a, b) not in pairs:
                pairs.append((a, b))
    seq_ids = {(a, b): f'seq.{a}>{b}' for a, b in pairs}
    new_prims = list(prims) + [{
        'id': seq_ids[(a, b)], 'part': 'sequence', 'type': 'order', 'joints': [],
        'definition': f'{a} occurs before {b}', 'exclusive_group': None,
        'description': f'First, {_sentence(desc[a])} Then, {_sentence(desc[b])}',
    } for a, b in pairs]
    n_forbid = 0
    for r in kg['rules']:
        own = [(a, b) for a, b in zip(r['sequence'], r['sequence'][1:])]
        r['require'] = list(r['require']) + [seq_ids[p] for p in own]
        rev = [seq_ids[(b, a)] for a, b in own if (b, a) in seq_ids and seq_ids[(b, a)] not in r['require']]
        r['forbid'] = list(r['forbid']) + [x for x in rev if x not in r['forbid']]
        n_forbid += len(rev)
    kg['meta'] = dict(kg.get('meta', {}))
    kg['meta']['v2seq'] = ('built by nesy4_seqkg.py from kg_rules_v2.json: consecutive `sequence` pairs become order '
                           'primitives seq.a>b (require), the reverse order is forbidden when it is a primitive; '
                           'no new knowledge')
    with open(DST_PRIMS, 'w', encoding='utf-8') as f:
        json.dump(new_prims, f, ensure_ascii=False, indent=1)
    with open(DST_RULES, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=1)
    if verbose:
        key = lambda r: (tuple(sorted(r['require'])), tuple(sorted(r['forbid'])))
        rules = kg['rules']
        same = sum(1 for i in range(len(rules)) for j in range(i + 1, len(rules)) if key(rules[i]) == key(rules[j]))
        print(f'[nesy] KG v2seq: {len(prims)} + {len(pairs)} order primitives, {n_forbid} reverse-order forbids, '
              f'class pairs with identical rules: {same} (v2 had 7)')
        print(f'[nesy] e.g. {new_prims[len(prims)]["id"]}: {new_prims[len(prims)]["description"]}')
    return DST_PRIMS, DST_RULES


if __name__ == '__main__':
    import sys
    if '--build-only' in sys.argv:
        build_seq_kg()
        sys.exit(0)
    if not (os.path.exists(DST_PRIMS) and os.path.exists(DST_RULES)):
        build_seq_kg()                     # subprocesses need the files; the parent rebuilds them in prepare()
    C.main('seqkg', Trainer, default_kg_mode='compose',
           prepare=lambda opts: build_seq_kg(),
           defaults=dict(kg_prims=DST_PRIMS, kg_rules=DST_RULES))
