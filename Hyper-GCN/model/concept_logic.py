"""Concept helpers shared by the concept–action graph (model/concept_action_graph.py, model/hypergcn_concept.py).

M [classes, concepts] is the binary action-concept matrix (annotations/concepts/, rows in label order); each concept
query of the decoder attends only to the joints of its body group (concept_joint_mask).
Loss: CE(logit, y) + beta BCE(c, M[y]) (class-level concept targets).
"""

import csv

import torch
import torch.nn.functional as F

EPS = 1e-6
TAU_MIN, TAU_MAX = 0.005, 1.0

# NTU joints (0-based): 0 spine base, 1 spine mid, 2 neck, 3 head, 4-7 left shoulder/elbow/wrist/hand,
# 8-11 right, 12-15 left hip/knee/ankle/foot, 16-19 right, 20 spine shoulder, 21-24 hand tips/thumbs.
HEAD = (2, 3, 20)
ARMS = (4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24)
TORSO = (0, 1, 2, 20, 4, 8, 12, 16)
LEGS = (0, 12, 13, 14, 15, 16, 17, 18, 19)
HEAD_WORDS = ('head', 'face', 'mouth', 'nose', 'neck')
TORSO_WORDS = ('chest', 'back', 'cross')


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.reader(handle))


def load_concepts(matrix_path, metadata_path, action_names):
    """Returns (M [C, K] float tensor, concept names [K], metadata rows {name: {scope, group, ...}}).
    Rows of the CSV must be the actions in label order (checked against `action_names`)."""
    rows = read_csv(matrix_path)
    header, body = rows[0], rows[1:]
    names = header[1:]
    if [r[0] for r in body] != list(action_names):
        raise ValueError(f'{matrix_path}: action rows differ from the label order')
    values = torch.tensor([[float(v) for v in r[1:]] for r in body])
    if not ((values == 0) | (values == 1)).all():
        raise ValueError(f'{matrix_path}: concept values must be 0 or 1')
    meta_rows = read_csv(metadata_path)
    keys = meta_rows[0]
    meta = {r[keys.index('concept_name')]: dict(zip(keys, r)) for r in meta_rows[1:]}
    missing = [n for n in names if n not in meta]
    if missing:
        raise ValueError(f'{metadata_path}: no metadata for {missing}')
    return values, names, meta


def concept_joint_mask(names, meta, num_point=25):
    """[K, V] bool: joints a concept's decoder query may attend to, from its metadata group.
    head -> head + neck; upper_limb -> both arms and hands (+ head joints for contact with the head /
    face / mouth / nose / neck, + torso for chest / back / crossing); hip_torso -> torso and hips;
    lower_limb -> hips, legs, feet; temporal and participant concepts -> every joint."""
    if num_point != 25:
        raise ValueError('joint groups are defined for the 25 NTU joints')
    mask = torch.zeros(len(names), num_point, dtype=torch.bool)
    for j, name in enumerate(names):
        group = meta[name]['group']
        if group == 'head':
            joints = HEAD
        elif group == 'upper_limb':
            joints = ARMS + (HEAD if any(w in name for w in HEAD_WORDS) else ()) \
                + (TORSO if any(w in name for w in TORSO_WORDS) else ())
        elif group == 'hip_torso':
            joints = TORSO
        elif group == 'lower_limb':
            joints = LEGS
        else:
            joints = tuple(range(num_point))
        mask[j, list(joints)] = True
    return mask


def concept_loss(out, labels, M, beta=0.5):
    """CE on the class logits + beta BCE of the concepts against the class row of M."""
    ce = F.cross_entropy(out['logits'], labels)
    bce = F.binary_cross_entropy(out['c'].clamp(EPS, 1 - EPS), M[labels])
    return ce + beta * bce, dict(ce=ce.item(), bce=bce.item())
