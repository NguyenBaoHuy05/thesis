"""Hard class pairs and what the KG can say about them (A1-G plan S4).

  python tools/hard_pairs.py [--kg annotations/action_kg.json] [--out annotations/kg_work/hard_pairs.json]
      [--cache-dir <folder> --cache-name ntu120_xsub --top 25] [--predicates-from <kg_metrics.json>]

Pairs come from the KG itself: every `similar` or `opposite` relation (made by the LLM from the
action definitions, before any model error was seen). Choosing pairs from the test confusion of
K0 would tune the KG to the test split; the cache options only *check* afterwards how many of K0's
top confused test pairs the KG pairs cover.

Per pair:
  opposed            attributes one class has and the other lacks: what VERIFY (verify_mode
                     opposed) can use
  opposed_grounded   the subset with a geometric predicate (model/predicates.py)
  one_sided          attributes known for one class, unknown for the other: candidates for the
                     local closed-world fill (S4 step 2)
Pairs with no grounded opposed attribute come first: they need a new attribute or a fill.
--predicates-from counts as grounded only the predicates a train_kg_head.py run used (e.g. after
min_train_auc), instead of every predicate of model/predicates.py.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.predicates import PREDICATES


# Same as model/kg.py, without importing torch: this tool runs anywhere.
def load_kg(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def action_names(kg):
    return [name for name, _ in sorted(kg['actions'].items(), key=lambda item: item[1]['id'])]


def signed_rows(kg_json):
    """{action: {attribute: +1 has / -1 lacks}}."""
    return {name: {**{a: 1 for a in info['has']}, **{a: -1 for a in info['lacks']}}
            for name, info in kg_json['actions'].items()}


def kg_pairs(kg_json):
    """Sorted (a, b) tuples, a < b in label order, with the relation(s) that link them."""
    order = {name: i for i, name in enumerate(action_names(kg_json))}
    pairs = {}
    for name, info in kg_json['actions'].items():
        for relation in ('similar', 'opposite'):
            for other in info.get(relation, []):
                key = tuple(sorted((name, other), key=order.get))
                pairs.setdefault(key, set()).add(relation)
    return pairs


def describe(a, b, rows, grounded):
    ra, rb = rows[a], rows[b]
    opposed = sorted(x for x in ra if x in rb and ra[x] != rb[x])
    one_sided = sorted([x for x in ra if x not in rb] + [x for x in rb if x not in ra])
    return dict(opposed=opposed, opposed_grounded=[x for x in opposed if x in grounded],
                one_sided=one_sided)


def k0_top_pairs(cache_dir, name, top, names):
    with np.load(os.path.join(cache_dir, f'{name}_test.npz')) as data:
        predicted, labels = data['fc_logits'].argmax(1), data['label']
    C = len(names)
    matrix = np.zeros((C, C), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    sym = matrix + matrix.T
    i, j = np.triu_indices(C, 1)
    order = np.argsort(-sym[i, j], kind='mergesort')[:top]
    errors = int(matrix.sum() - np.trace(matrix))
    return [(names[i[o]], names[j[o]], int(sym[i[o], j[o]])) for o in order], errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg', default='annotations/action_kg.json')
    parser.add_argument('--out', default='annotations/kg_work/hard_pairs.json')
    parser.add_argument('--cache-dir', help='feature cache with K0 fc_logits: coverage check only')
    parser.add_argument('--cache-name', default='ntu120_xsub')
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--predicates-from', metavar='KG_METRICS_JSON',
                        help='grounded = only the predicates of this train_kg_head.py run')
    args = parser.parse_args()

    kg_json = load_kg(args.kg)
    rows = signed_rows(kg_json)
    grounded = set(PREDICATES) & set(kg_json['attributes'])
    if args.predicates_from:
        with open(args.predicates_from, encoding='utf-8') as handle:
            grounded &= set(json.load(handle)['predicates']['names'])
    pairs = []
    for (a, b), relations in kg_pairs(kg_json).items():
        pairs.append(dict(pair=[a, b], relations=sorted(relations), **describe(a, b, rows, grounded)))
    pairs.sort(key=lambda p: (len(p['opposed_grounded']), len(p['opposed']), 'opposite' not in p['relations']))

    out = dict(kg=args.kg, kg_hash=kg_json['hash'], source='kg similar/opposite relations',
               predicates_from=args.predicates_from,
               grounded_attributes=sorted(grounded), pairs=pairs,
               summary=dict(pairs=len(pairs),
                            no_opposed=sum(not p['opposed'] for p in pairs),
                            no_grounded_opposed=sum(not p['opposed_grounded'] for p in pairs)))
    if args.cache_dir:
        names = action_names(kg_json)
        top, errors = k0_top_pairs(args.cache_dir, args.cache_name, args.top, names)
        listed = {tuple(sorted(p['pair'])) for p in pairs}
        check = []
        for a, b, count in top:
            check.append(dict(pair=[a, b], confusions=count, in_kg_pairs=tuple(sorted((a, b))) in listed,
                              **describe(a, b, rows, grounded)))
        out['k0_test_check'] = dict(
            note='coverage check only; not used to choose pairs', top=args.top, k0_errors=errors,
            covered=sum(c['in_kg_pairs'] for c in check),
            with_grounded_opposed=sum(bool(c['opposed_grounded']) for c in check),
            confusions_in_top=sum(c['confusions'] for c in check), pairs=check)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)
    s = out['summary']
    print(f"{s['pairs']} KG pairs: {s['no_opposed']} without any opposed attribute, "
          f"{s['no_grounded_opposed']} without a grounded one -> {args.out}")
    if 'k0_test_check' in out:
        c = out['k0_test_check']
        print(f"K0 top-{c['top']} test pairs ({c['confusions_in_top']} of {c['k0_errors']} errors): "
              f"{c['covered']} among the KG pairs, {c['with_grounded_opposed']} with a grounded opposed attribute")
