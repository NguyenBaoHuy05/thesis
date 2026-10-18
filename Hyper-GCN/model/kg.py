"""Load the action KG and its offline text embeddings (A1 plan S1/S2/S3).

The embeddings file (tools/build_kg_embeddings.py) is only valid for the KG it was built from:
`load_embeddings` refuses a file whose `kg_hash` or action order differs from the KG, so a KG
edited after review cannot silently run with stale vectors.

A shuffled KG (tools/shuffle_kg.py, ablation K2) moves edges only, never texts, so it is
accepted with the embeddings of the KG it was shuffled from (`shuffled.source_hash`).

`kg_tensors` turns the KG into the dense matrices the head works on (the KG is small:
120 actions x ~60 attributes), rows in label order and attribute columns in JSON key order,
the same orders as the embeddings file.
"""

import json
from dataclasses import dataclass, fields

import numpy as np
import torch

EDGE_WEIGHTS = (1, 2)


def load_kg(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def action_names(kg):
    """Action keys in label order (index i = action with `id` i)."""
    return [name for name, _ in sorted(kg['actions'].items(), key=lambda item: item[1]['id'])]


def load_embeddings(path, kg, key='action'):
    """Returns (action [num_class, D], attribute [A, D]) float32 tensors, rows L2-normalized.
    `key` picks the action variant: 'action' (prompt ensemble) or 'action_text' (text only)."""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    expected = kg['shuffled']['source_hash'] if 'shuffled' in kg else kg['hash']
    if str(arrays['kg_hash']) != expected:
        raise ValueError(f"{path} was built from KG {str(arrays['kg_hash'])[:19]}..., "
                         f"not {expected[:19]}...; rerun tools/build_kg_embeddings.py")
    if list(arrays['action_names']) != action_names(kg):
        raise ValueError(f'{path}: action order differs from the KG ids')
    if list(arrays['attribute_names']) != list(kg['attributes']):
        raise ValueError(f'{path}: attribute order differs from the KG')
    return torch.from_numpy(arrays[key]).float(), torch.from_numpy(arrays['attribute']).float()


@dataclass
class KGTensors:
    """W:      [C, A] signed edge weight: has -> +w/2, lacks -> -w/2, unknown -> 0 (w in {1, 2}),
               so W is in {-1, -1/2, 0, 1/2, 1} (VERIFY: v_c = sum_attr W[c, attr] d_attr (2 a_attr - 1)).
    y_attr: [C, A] attribute target of a sample of class c: 1 has, 0 lacks (and 0 where unknown).
    known:  [C, A] bool, the KG says has or lacks; BCE on y_attr is masked to these cells.
    S_sim, S_opp: [C, C] bool, symmetric `similar` / `opposite` relations (diagonal False)."""
    W: torch.Tensor
    y_attr: torch.Tensor
    known: torch.Tensor
    S_sim: torch.Tensor
    S_opp: torch.Tensor

    def to(self, device):
        return KGTensors(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


# KG `body_parts` -> NTU RGB+D joints (0-based: 0 base spine, 1 mid spine, 2 neck, 3 head,
# 4-7 left shoulder/elbow/wrist/hand, 8-11 right, 12-15 left hip/knee/ankle/foot, 16-19 right,
# 20 spine shoulder, 21-22 left hand tip/thumb, 23-24 right).
NTU_BODY_PARTS = {
    'head': (2, 3),
    'torso': (0, 1, 2, 20, 4, 8),
    'hip': (0, 12, 16),
    'arm_L': (4, 5, 6), 'arm_R': (8, 9, 10),
    'hand_L': (6, 7, 21, 22), 'hand_R': (10, 11, 23, 24),
    'leg_L': (12, 13, 14), 'leg_R': (16, 17, 18),
    'feet': (14, 15, 18, 19),
}
MIRROR = {'arm_L': 'arm_R', 'arm_R': 'arm_L', 'hand_L': 'hand_R', 'hand_R': 'hand_L',
          'leg_L': 'leg_R', 'leg_R': 'leg_L'}


def attribute_joint_mask(kg, num_point=25, mirror=True):
    """[A, V] bool: the joints an attribute may look at, from its KG `body_parts` (all joints if
    it lists none). mirror=True also opens the other side: the KG writes `hand_R` for "the
    acting hand", but performers use either hand and the mask should not decide that."""
    if num_point != 25:
        raise ValueError('body-part map is defined for the 25 NTU joints')
    mask = torch.zeros(len(kg['attributes']), num_point, dtype=torch.bool)
    for j, (attr, info) in enumerate(kg['attributes'].items()):
        parts = list(info.get('body_parts') or [])
        if not parts:
            mask[j] = True
            continue
        if mirror:
            parts += [MIRROR[p] for p in parts if p in MIRROR]
        for part in parts:
            if part not in NTU_BODY_PARTS:
                raise ValueError(f'{attr}: unknown body part {part}')
            mask[j, list(NTU_BODY_PARTS[part])] = True
    return mask


# Relation types of the graph encoder (A1-G plan S2). Messages flow from column j to row i of each
# adjacency; has/lacks carry the edge weight w (1 or 2), the action-action relations weight 1.
RELATIONS = ('has', 'has_inv', 'lacks', 'lacks_inv', 'similar', 'opposite')


def relation_adjacency(kg, relations=RELATIONS):
    """[R, C + A, C + A] row-normalized adjacency per relation; nodes are the C actions (label
    order) then the A attributes (JSON order). has: action -> attribute (the attribute receives),
    has_inv: attribute -> action, same for lacks; similar / opposite: symmetric between actions.
    A row with no edge of that relation stays zero (no message, not a uniform average)."""
    C, A = kg.W.shape
    has, lacks = (2 * kg.W).clamp_min(0), (-2 * kg.W).clamp_min(0)          # [C, A], values w
    blocks = {
        'has': (slice(C, None), slice(None, C), has.t()),
        'has_inv': (slice(None, C), slice(C, None), has),
        'lacks': (slice(C, None), slice(None, C), lacks.t()),
        'lacks_inv': (slice(None, C), slice(C, None), lacks),
        'similar': (slice(None, C), slice(None, C), kg.S_sim.float()),
        'opposite': (slice(None, C), slice(None, C), kg.S_opp.float()),
    }
    out = torch.zeros(len(relations), C + A, C + A)
    for r, name in enumerate(relations):
        if name not in blocks:
            raise ValueError(f'unknown relation {name}; expected one of {RELATIONS}')
        rows, cols, block = blocks[name]
        out[r, rows, cols] = block
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)


def kg_tensors(kg):
    names = action_names(kg)
    attributes = list(kg['attributes'])
    row = {name: i for i, name in enumerate(names)}
    col = {attr: j for j, attr in enumerate(attributes)}
    W = torch.zeros(len(names), len(attributes))
    known = torch.zeros(len(names), len(attributes), dtype=torch.bool)
    S_sim = torch.zeros(len(names), len(names), dtype=torch.bool)
    S_opp = torch.zeros(len(names), len(names), dtype=torch.bool)

    for name in names:
        action = kg['actions'][name]
        for kind, sign in (('has', 1.0), ('lacks', -1.0)):
            for attr, edge in action[kind].items():
                if attr not in col:
                    raise ValueError(f'{name}: {kind} unknown attribute {attr}')
                if edge['w'] not in EDGE_WEIGHTS:
                    raise ValueError(f"{name}: {kind} {attr} has weight {edge['w']}, expected {EDGE_WEIGHTS}")
                i, j = row[name], col[attr]
                if known[i, j]:
                    raise ValueError(f'{name}: {attr} is both has and lacks')
                W[i, j] = sign * edge['w'] / 2
                known[i, j] = True
        for relation, S in (('similar', S_sim), ('opposite', S_opp)):
            for other in action.get(relation, []):
                if other not in row:
                    raise ValueError(f'{name}: {relation} unknown action {other}')
                if other == name:
                    raise ValueError(f'{name}: {relation} to itself')
                # Undirected: a one-sided listing still links both actions.
                S[row[name], row[other]] = S[row[other], row[name]] = True

    return KGTensors(W=W, y_attr=(W > 0).float(), known=known, S_sim=S_sim, S_opp=S_opp)
