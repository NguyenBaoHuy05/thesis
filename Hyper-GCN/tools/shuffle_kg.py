"""Degree-preserving shuffled KGs for ablation K2 (A1 plan S2, S6).

  python tools/shuffle_kg.py [--kg annotations/action_kg.json] [--seeds 0 1 2]
      [--out-dir annotations/kg_shuffled] [--swaps-per-edge 10]

Writes <out-dir>/action_kg_shuffle_s<seed>.json. Edge swaps (a1-t1, a2-t2) -> (a1-t2, a2-t1):

- has and lacks are swapped separately, so every action keeps its number of has and of lacks
  edges and every attribute its number of has and of lacks edges; an edge keeps its weight `w`
  and `agree` (its `evidence` and review flag are dropped: they no longer describe the edge).
- A swap is skipped if it would put an attribute on an action twice (has and lacks) or break an
  `excludes` rule, so the shuffled KG passes tools/validate_kg.py like the real one.
- `similar` and `opposite` are swapped as undirected action-action graphs, keeping each action's
  number of neighbors, without self-links or duplicate pairs.

Only edges move: action and attribute texts are untouched, so the shuffled KG reuses the real
KG's embeddings file. The output records `shuffled.source_hash`, which model/kg.py accepts.
"""

import argparse
import copy
import hashlib
import json
import os
import random


def kg_hash(kg):
    """Same convention as tools/kg_build.py / kg_review.py: sha256 of the KG without its hash."""
    body = {key: value for key, value in kg.items() if key != 'hash'}
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return 'sha256:' + hashlib.sha256(payload).hexdigest()


def excluded_pairs(kg):
    pairs = set()
    for x, y in kg.get('attribute_relations', {}).get('excludes', []):
        pairs |= {(x, y), (y, x)}
    return pairs


def shuffle_attribute_edges(kg, rng, swaps_per_edge):
    """In place on kg['actions'][*]['has'/'lacks']. Returns {kind: (edges, changed)}."""
    actions = kg['actions']
    excluded = excluded_pairs(kg)
    # Attributes an action already touches (has or lacks): a swap must not duplicate one.
    occupied = {name: set(a['has']) | set(a['lacks']) for name, a in actions.items()}
    has_sets = {name: set(a['has']) for name, a in actions.items()}
    stats = {}
    for kind in ('has', 'lacks'):
        edges = [[name, attr, actions[name][kind][attr]] for name in sorted(actions)
                 for attr in sorted(actions[name][kind])]
        original = {(name, attr) for name, attr, _ in edges}

        def allowed(action, old, new):
            if kind != 'has':
                return True
            rest = has_sets[action] - {old}
            return not any((new, other) in excluded for other in rest)

        for _ in range(swaps_per_edge * len(edges)):
            i, j = rng.randrange(len(edges)), rng.randrange(len(edges))
            a1, t1, _ = edges[i]
            a2, t2, _ = edges[j]
            if a1 == a2 or t1 == t2 or t2 in occupied[a1] or t1 in occupied[a2]:
                continue
            if not (allowed(a1, t1, t2) and allowed(a2, t2, t1)):
                continue
            occupied[a1].symmetric_difference_update({t1, t2})
            occupied[a2].symmetric_difference_update({t1, t2})
            if kind == 'has':
                has_sets[a1].symmetric_difference_update({t1, t2})
                has_sets[a2].symmetric_difference_update({t1, t2})
            edges[i][1], edges[j][1] = t2, t1

        for name in actions:
            actions[name][kind] = {}
        for name, attr, payload in edges:
            actions[name][kind][attr] = {key: value for key, value in payload.items()
                                         if key not in ('evidence', 'reviewed')}
        stats[kind] = (len(edges), sum((name, attr) not in original for name, attr, _ in edges))
    return stats


def shuffle_relation(kg, relation, rng, swaps_per_edge):
    """Undirected degree-preserving swap of kg['actions'][*][relation]. Returns (pairs, changed)."""
    actions = kg['actions']
    pairs = sorted({tuple(sorted((a, b))) for a in actions for b in actions[a].get(relation, [])})
    original, present = set(pairs), set(pairs)
    edges = [list(pair) for pair in pairs]
    for _ in range(swaps_per_edge * len(edges)):
        i, j = rng.randrange(len(edges)), rng.randrange(len(edges))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if a == d or c == b or new1 == new2 or new1 in present or new2 in present:
            continue
        present -= {tuple(sorted((a, b))), tuple(sorted((c, d)))}
        present |= {new1, new2}
        edges[i], edges[j] = list(new1), list(new2)

    neighbors = {name: [] for name in actions}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    for name in actions:
        actions[name][relation] = sorted(neighbors[name])
    return len(edges), sum(tuple(edge) not in original for edge in edges)


def shuffle_kg(kg, seed, swaps_per_edge=10):
    if 'shuffled' in kg:
        raise ValueError('input KG is already shuffled')
    rng = random.Random(seed)
    out = copy.deepcopy(kg)
    stats = shuffle_attribute_edges(out, rng, swaps_per_edge)
    for relation in ('similar', 'opposite'):
        stats[relation] = shuffle_relation(out, relation, rng, swaps_per_edge)
    out['shuffled'] = dict(seed=seed, source_hash=kg['hash'], swaps_per_edge=swaps_per_edge,
                           changed={kind: dict(edges=n, moved=moved) for kind, (n, moved) in stats.items()})
    out['dropped_edges'] = []  # refer to edges of the real KG
    for action in out['actions'].values():
        action['reviewed'] = False
    out['hash'] = kg_hash(out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--kg', default=os.path.join(repo, 'annotations', 'action_kg.json'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--out-dir', default=os.path.join(repo, 'annotations', 'kg_shuffled'))
    parser.add_argument('--swaps-per-edge', type=int, default=10)
    args = parser.parse_args()

    with open(args.kg, encoding='utf-8') as handle:
        kg = json.load(handle)
    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seeds:
        out = shuffle_kg(kg, seed, args.swaps_per_edge)
        path = os.path.join(args.out_dir, f'action_kg_shuffle_s{seed}.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(out, handle, indent=1, ensure_ascii=False)
        moved = ', '.join(f"{kind} {c['moved']}/{c['edges']}" for kind, c in out['shuffled']['changed'].items())
        print(f'seed {seed}: moved {moved} -> {path}')
    print('Check each output with: python tools/validate_kg.py --kg <file>')
