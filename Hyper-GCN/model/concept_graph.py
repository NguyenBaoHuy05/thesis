"""Concept–action logic graph (stage 2 of KG_RETRIEVAL_PLAN/CONCEPT_GRAPH/CONCEPT_GRAPH_PLAN.md, §5).

Built offline from the binary action-concept matrix M (annotations/concepts/, rows in label order); nothing here
is trained. Pure Python (no torch), so the graph can be built and checked anywhere.

Nodes: concepts, actions, AND gates (motifs: concept sets shared by several actions), OR gates (concepts that
replace each other, v1: posture stand / sit), and pair nodes NOT(j|i) for confusable pairs (i, j).

Every action's row of M is split, without overlap, into: its motifs, its OR gates (an OR gate replaces the
members of the group the row has) and direct concept edges (the rest). NOT(j|i) reads only the concepts j has
and i lacks (M_j minus M_i) and enters the AND of action i as a NOT literal.

Relations (R-GCN numbering, drawio/architecture_concept_graph.drawio):
    r1 concept in AND    r2 concept in OR    r3 concept -> action    r4 AND -> action    r5 OR -> action
    r6 action j -> NOT(j|i) (auxiliary, R-GCN only)    r7 NOT(j|i) -| action i    r8 concept in M_j minus M_i -> NOT(j|i)
"""

import csv
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass, field

POSTURE_OR = (('hip_static_stand', 'hip_static_sit'), ('leg_static_stand', 'leg_static_sit'))
OPPOSITE_PREFIXES = (('put_on_', 'take_off_'),)

DEFAULT_CFG = dict(
    common_min=60,              # concepts in >= this many rows are never used in a motif
    motif_sizes=(2, 3),
    motif_min_support=3,        # a motif must be contained in >= this many rows
    max_motifs=40,
    min_gain=6,                 # stop when the best motif covers fewer (action, concept) cells
    motifs_per_action=3,
    or_groups=POSTURE_OR,
    pair_hamming=4,             # confusable pairs: rows differing in <= this many concepts
    opposite_prefixes=OPPOSITE_PREFIXES,
)


@dataclass
class ConceptGraph:
    actions: list
    concepts: list
    rows: list                                       # [A] sorted concept ids of each row of M
    motifs: list = field(default_factory=list)       # [G] sorted concept ids
    ors: list = field(default_factory=list)          # [O] concept ids of each OR group
    action_motifs: list = field(default_factory=list)   # [A] motif ids
    action_ors: list = field(default_factory=list)      # [A] OR ids
    action_direct: list = field(default_factory=list)   # [A] concept ids
    pairs: list = field(default_factory=list)        # undirected confusable pairs (i, j), i < j
    nots: list = field(default_factory=list)         # [(i, j, concept ids of M_j minus M_i)]: NOT(j|i) -> action i
    cfg: dict = field(default_factory=dict)

    def edges(self):
        """{relation: [(source, target)]} with node ids local to each node type."""
        e = {r: [] for r in ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8')}
        for g, members in enumerate(self.motifs):
            e['r1'] += [(c, g) for c in members]
        for o, members in enumerate(self.ors):
            e['r2'] += [(c, o) for c in members]
        for a in range(len(self.actions)):
            e['r3'] += [(c, a) for c in self.action_direct[a]]
            e['r4'] += [(g, a) for g in self.action_motifs[a]]
            e['r5'] += [(o, a) for o in self.action_ors[a]]
        for n, (i, j, diff) in enumerate(self.nots):
            e['r6'].append((j, n))
            e['r7'].append((n, i))
            e['r8'] += [(c, n) for c in diff]
        return e

    def literals(self, a):
        """Concept ids action a reads through its motifs, OR gates and direct edges."""
        out = set(self.action_direct[a])
        for g in self.action_motifs[a]:
            out |= set(self.motifs[g])
        for o in self.action_ors[a]:
            out |= set(self.ors[o])
        return out

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(asdict(self), handle, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        data['nots'] = [(i, j, list(d)) for i, j, d in data['nots']]
        data['pairs'] = [tuple(p) for p in data['pairs']]
        return cls(**data)

    def summary(self):
        e = self.edges()
        return dict(actions=len(self.actions), concepts=len(self.concepts), motifs=len(self.motifs),
                    ors=len(self.ors), nots=len(self.nots), pairs=len(self.pairs),
                    edges={r: len(v) for r, v in e.items()},
                    actions_with_motif=sum(bool(m) for m in self.action_motifs),
                    actions_with_not=len({i for i, _, _ in self.nots}),
                    motif_cells=sum(len(self.motifs[g]) for m in self.action_motifs for g in m))


def read_matrix(path):
    """(action names, concept names, rows as sorted concept-id lists) from the CSV."""
    with open(path, encoding='utf-8', newline='') as handle:
        table = list(csv.reader(handle))
    concepts = table[0][1:]
    actions = [r[0] for r in table[1:]]
    rows = [sorted(j for j, v in enumerate(r[1:]) if float(v) == 1) for r in table[1:]]
    return actions, concepts, rows


def confusable_pairs(actions, rows, hamming=4, opposite_prefixes=OPPOSITE_PREFIXES):
    """Undirected pairs (i < j): rows differing in <= `hamming` concepts, plus name opposites (put_on_x /
    take_off_x). Uses only M and the class names, never the test split."""
    sets = [set(r) for r in rows]
    pairs = {(i, j) for i, j in itertools.combinations(range(len(rows)), 2)
             if len(sets[i] ^ sets[j]) <= hamming}
    index = {name: k for k, name in enumerate(actions)}
    for a, b in opposite_prefixes:
        for name, k in index.items():
            if name.startswith(a) and b + name[len(a):] in index:
                pairs.add(tuple(sorted((k, index[b + name[len(a):]]))))
    return sorted(pairs)


def mine_motifs(rows, blocked, cfg):
    """Greedy motif choice (plan §5). rows: per action the concept ids still free for motifs; blocked: ids never
    used in a motif. Each action takes at most `motifs_per_action` disjoint motifs, each a subset of its row.
    Returns (motifs, per-action motif ids)."""
    free = [set(r) - blocked for r in rows]
    support = Counter()
    for r in free:
        for k in cfg['motif_sizes']:
            support.update(itertools.combinations(sorted(r), k))
    candidates = sorted(m for m, n in support.items() if n >= cfg['motif_min_support'])
    motifs, assigned = [], [[] for _ in rows]
    while len(motifs) < cfg['max_motifs']:
        best, best_key, best_users = None, None, None
        for m in candidates:
            s = set(m)
            users = [a for a, r in enumerate(free) if s <= r and len(assigned[a]) < cfg['motifs_per_action']]
            key = (len(users) * len(m), len(m))
            if best_key is None or key > best_key:
                best, best_key, best_users = m, key, users
        if best is None or best_key[0] < cfg['min_gain'] or len(best_users) < cfg['motif_min_support']:
            break
        g = len(motifs)
        motifs.append(list(best))
        for a in best_users:
            assigned[a].append(g)
            free[a] -= set(best)
        candidates.remove(best)
    return motifs, assigned


def build_graph(actions, concepts, rows, cfg=None):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    index = {name: j for j, name in enumerate(concepts)}
    ors = []
    for group in cfg['or_groups']:
        missing = [n for n in group if n not in index]
        if missing:
            raise ValueError(f'OR group {group}: unknown concepts {missing}')
        ors.append(sorted(index[n] for n in group))

    remaining = [set(r) for r in rows]
    action_ors = [[] for _ in rows]
    for o, members in enumerate(ors):
        for a, r in enumerate(remaining):
            if r & set(members):
                action_ors[a].append(o)
                r -= set(members)

    df = Counter(j for r in rows for j in r)
    common = {j for j, n in df.items() if n >= cfg['common_min']}
    motifs, action_motifs = mine_motifs([sorted(r) for r in remaining], common, cfg)
    for a, ms in enumerate(action_motifs):
        for g in ms:
            remaining[a] -= set(motifs[g])

    pairs = confusable_pairs(actions, rows, cfg['pair_hamming'], cfg['opposite_prefixes'])
    nots = []
    for i, j in pairs:
        for a, b in ((i, j), (j, i)):
            diff = sorted(set(rows[b]) - set(rows[a]))
            if diff:
                nots.append((a, b, diff))

    cfg_out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    return ConceptGraph(actions=list(actions), concepts=list(concepts), rows=[list(r) for r in rows],
                        motifs=motifs, ors=ors, action_motifs=action_motifs, action_ors=action_ors,
                        action_direct=[sorted(r) for r in remaining], pairs=[tuple(p) for p in pairs],
                        nots=nots, cfg=cfg_out)


def flat_graph(actions, concepts, rows, nots=True, pair_hamming=4):
    """No motif and no OR gate: every concept of a row is a direct edge (CL v2), optionally with NOT(j|i).
    nots=False gives exactly the flat AND of model/concept_logic.py."""
    cfg = dict(or_groups=(), max_motifs=0, pair_hamming=pair_hamming)
    graph = build_graph(actions, concepts, rows, cfg)
    if not nots:
        graph.nots = []
    return graph


def random_pair_graph(graph, seed=0):
    """E1e control: same number of NOT nodes and the same size of each M_j minus M_i, but the pair (i, j) and
    the concepts are drawn at random (j != i, concepts outside the row of i)."""
    import random
    rng = random.Random(seed)
    n_actions, n_concepts = len(graph.actions), len(graph.concepts)
    nots = []
    for _i, _j, diff in graph.nots:
        i = rng.randrange(n_actions)
        j = rng.choice([a for a in range(n_actions) if a != i])
        outside = [c for c in range(n_concepts) if c not in set(graph.rows[i])]
        nots.append((i, j, sorted(rng.sample(outside, len(diff)))))
    return ConceptGraph(**{**asdict(graph), 'nots': nots})
