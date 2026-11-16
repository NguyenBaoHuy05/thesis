"""KG cho PKU-MMD I (51 lớp), dựng từ KG v2 của NTU.

Cả 51 lớp PKU-MMD I đều có lớp cùng tên (hoặc cùng hành động) trong NTU RGB+D 120, nên luật được chép
nguyên từ lớp NTU tương ứng (require / forbid / sequence / persons), không viết lại. Id của luật là
thứ tự lớp trong semantics/pku51.csv và pku51_llm.txt (1..51), vì kg_semantics.PrimitiveKG lấy luật
của lớp c theo id c + 1.

Ngoại lệ duy nhất: "put something inside pocket" và "take out something from pocket" đều ứng với một lớp
NTU (25, reach into pocket). Nếu chép nguyên thì hai lớp có luật giống hệt nhau. Hai lớp này được viết thêm
(Claude Opus 5.5, 2026-11-16, chỉ từ tên lớp và primitive_v2.json, không dùng dữ liệu skeleton hay kết quả
thí nghiệm): giữ luật của lớp 25, thêm hand.in_front_space vào require, và một `sequence` ngược nhau theo
đúng mẫu của các cặp NTU đội/bỏ mũ, đeo/tháo kính (in_front_space -> vị trí trên cơ thể và ngược lại).

Chạy: python kg_pku51.py    -> kg/kg_rules_pku51_v2.json
"""
import copy
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PRIMS = os.path.join(HERE, 'kg', 'primitive_v2.json')
SRC_RULES = os.path.join(HERE, 'kg', 'kg_rules_v2.json')
DST_RULES = os.path.join(HERE, 'kg', 'kg_rules_pku51_v2.json')

# PKU-MMD I class (order of semantics/pku51_llm.txt) -> NTU RGB+D 120 class id (A-number)
PKU_TO_NTU = [
    ('bow', 35),
    ('brushing hair', 4),
    ('brushing teeth', 3),
    ('check time (from watch)', 33),
    ('cheer up', 22),
    ('clapping', 10),
    ('cross hands in front (say stop)', 40),
    ('drink water', 1),
    ('drop', 5),
    ('eat meal/snack', 2),
    ('falling', 43),
    ('giving something to other person', 56),
    ('hand waving', 23),
    ('handshaking', 58),
    ('hopping (one foot jumping)', 26),
    ('hugging other person', 55),
    ('jump up', 27),
    ('kicking other person', 51),
    ('kicking something', 24),
    ('make a phone call/answer phone', 28),
    ('pat on back of other person', 53),
    ('pickup', 6),
    ('playing with phone/tablet', 29),
    ('point finger at the other person', 54),
    ('pointing to something with finger', 31),
    ('punching/slapping other person', 50),
    ('pushing other person', 52),
    ('put on a hat/cap', 20),
    ('put something inside pocket', 25),
    ('reading', 11),
    ('rub two hands together', 34),
    ('salute', 38),
    ('sitting down', 8),
    ('standing up', 9),
    ('take off a hat/cap', 21),
    ('take off glasses', 19),
    ('take off jacket', 15),
    ('take out something from pocket', 25),
    ('taking a selfie', 32),
    ('tear up paper', 13),
    ('throw', 7),
    ('touch back (backache)', 46),
    ('touch chest (stomachache/heart pain)', 45),
    ('touch head (headache)', 44),
    ('touch neck (neckache)', 47),
    ('typing on a keyboard', 30),
    ('use a fan (with hand or paper)/feeling warm', 49),
    ('wear jacket', 14),
    ('wear on glasses', 18),
    ('wipe face', 37),
    ('writing', 12),
]

# the only rules not copied from NTU (see docstring)
POCKET_SEQUENCE = {
    'put something inside pocket': ['hand.in_front_space', 'hand.at_hip'],
    'take out something from pocket': ['hand.at_hip', 'hand.in_front_space'],
}


def build():
    with open(SRC_RULES, encoding='utf-8') as f:
        ntu = json.load(f)
    with open(SRC_PRIMS, encoding='utf-8') as f:
        prim_ids = {p['id'] for p in json.load(f)}
    by_id = {r['id']: r for r in ntu['rules']}

    # the order must match Flora's PKU semantics, which index classes by line
    llm = [l.strip() for l in open(os.path.join(HERE, 'semantics', 'pku51_llm.txt'), encoding='utf-8')]
    assert len(llm) == len(PKU_TO_NTU) == 51, (len(llm), len(PKU_TO_NTU))
    for (name, _), line in zip(PKU_TO_NTU, llm):
        quoted = line.split('"')[1].lower()
        assert quoted == name, f'class order differs from pku51_llm.txt: {name!r} vs {quoted!r}'

    rules = []
    for i, (name, ntu_id) in enumerate(PKU_TO_NTU, start=1):
        r = copy.deepcopy(by_id[ntu_id])
        r.update(id=i, **{'class': name}, ntu_id=ntu_id)
        if name in POCKET_SEQUENCE:
            seq = POCKET_SEQUENCE[name]
            r['require'] = list(r['require']) + [p for p in seq if p not in r['require']]
            r['sequence'] = list(seq)
        for p in r['require'] + r['forbid'] + r['sequence']:
            assert p in prim_ids, f'{name}: unknown primitive {p}'
        rules.append(r)

    kg = dict(meta=dict(ntu['meta']), constraints=ntu['constraints'], rules=rules)
    kg['meta'].update(
        classes='PKU-MMD I, 51 classes, id = line in semantics/pku51_llm.txt; ntu_id = NTU class the rule is copied from',
        pku51=('built by kg_pku51.py from kg_rules_v2.json: every rule copied from the NTU class of the same action; '
               'only the two pocket classes (both NTU 25) get hand.in_front_space and opposite sequences, written by '
               'Claude Opus 5.5 on 2026-11-16 from class names and primitive_v2.json only'))
    with open(DST_RULES, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=1)

    key = lambda r: (tuple(sorted(r['require'])), tuple(sorted(r['forbid'])))
    same = [(a['class'], b['class']) for i, a in enumerate(rules) for b in rules[i + 1:] if key(a) == key(b)]
    print(f'[kg_pku51] {len(rules)} classes -> {os.path.relpath(DST_RULES, HERE)}; '
          f'{sum(bool(r["sequence"]) for r in rules)} with a sequence; '
          f'pairs with identical require/forbid (before ORE): {len(same)}')
    for a, b in same:
        print(f'  {a}  ==  {b}')
    return DST_RULES


if __name__ == '__main__':
    build()
