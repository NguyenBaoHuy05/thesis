"""Concept-logic classifier: skeleton -> concepts -> class by soft logic over a fixed concept matrix.

No geometric predicates and no class-specific weight: a class is decided only through its concepts.

    c_j     = ConceptDecoder(l9 map before pooling)_j  in (0, 1)       one query per concept, attending
                                                                   only to the joints of its body group
    score_k = [ sum_j M_kj log c_j + omega sum_j (1 - M_kj) log(1 - c_j) ] / (n_pos_k + omega n_neg_k)
            = log of the weighted geometric mean of the literals: AND over has (c_j) and NOT over the
              other concepts (1 - c_j)
    logit_k = score_k / tau                                          tau learned

M [classes, concepts] is the binary action-concept matrix (annotations/concepts/, rows in label order).
It is dense and was designed so that every class has its own row, so a 0 is read as "absent"
(closed world) and enters the AND as NOT c_j with weight omega (omega = 0: has-only AND).
balance=True averages the has and the NOT literals separately (logic_scores), so that the ~10 concepts a
class has are not drowned by its ~92 NOT literals.

Loss: CE(logit, y) + beta BCE(c, M[y]) (class-level concept targets). Ablations: K2 = a shuffled M
(tools/shuffle_concepts.py, every row and column keeps its number of ones); K5 = flip a concept
(c_j <- 1 - c_j) at test time and count predictions that change.

The decoder is model/attribute_decoder.py with one-hot query inputs (a learned vector per concept,
no text embedding). It reads features [N, M, C, S, V]: S = frames of the l9 map when fine-tuning
through main.py, or the S time segments cached by tools/cache_joint_features.py.
"""

import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attribute_decoder import AttributeDecoder

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


def logic_scores(c, M, omega, balance=False):
    """c [N, K] in (0, 1), M [C, K] binary -> [N, C] log weighted geometric mean of the literals.
    balance=False: one mean over all literals (a class has ~10 has and ~92 NOT literals, so each has
    literal weighs ~1/102). balance=True: mean over the has literals and mean over the NOT literals
    separately, (mean_has log c + omega mean_not log(1 - c)) / (1 + omega): each has literal weighs
    ~1/10, so turning off one concept a class needs can change the decision (K5)."""
    log_c, log_not = torch.log(c.clamp(EPS, 1 - EPS)), torch.log((1 - c).clamp(EPS, 1 - EPS))
    pos, neg = log_c @ M.t(), log_not @ (1 - M).t()
    if balance:
        return (pos / M.sum(1).clamp_min(1) + omega * neg / (1 - M).sum(1).clamp_min(1)) / (1 + omega)
    return (pos + omega * neg) / (M.sum(1) + omega * (1 - M).sum(1))


class ConceptLogicHead(nn.Module):
    def __init__(self, in_channels, M, joint_mask, decoder_args=None, omega=1.0, balance=False, tau=0.1,
                 learn_tau=True):
        super().__init__()
        if M.shape[1] != joint_mask.shape[0]:
            raise ValueError('one joint_mask row per concept')
        self.register_buffer('M', M.float())
        self.omega, self.balance = float(omega), bool(balance)
        self.decoder = AttributeDecoder(in_channels, torch.eye(M.shape[1]), joint_mask, **(decoder_args or {}))
        self.log_tau = nn.Parameter(torch.tensor(float(tau)).log(), requires_grad=learn_tau)

    @property
    def tau(self):
        return self.log_tau.exp().clamp(TAU_MIN, TAU_MAX)

    def logits(self, c):
        return logic_scores(c, self.M, self.omega, self.balance) / self.tau

    def forward(self, features, valid):
        """features [N, M, C, S, V], valid [N, M] -> dict(c [N, K], logits [N, classes])."""
        c = self.decoder(features, valid)
        return dict(c=c, logits=self.logits(c))


def concept_loss(out, labels, M, beta=0.5):
    """CE on the class logits + beta BCE of the concepts against the class row of M."""
    ce = F.cross_entropy(out['logits'], labels)
    bce = F.binary_cross_entropy(out['c'].clamp(EPS, 1 - EPS), M[labels])
    return ce + beta * bce, dict(ce=ce.item(), bce=bce.item())
