"""Knowledge graph for the NeSy branch.

Sources (written from class names only, before any experiment):
  zeroshot/primitive.json : 71 primitives (id, part, joints, description, exclusive_group, ...)
  zeroshot/kg_rules.json  : per class require / forbid / sequence, plus implication constraints

Side variants. Hand / arm / leg primitives list the joints of both sides (e.g. hand.at_face uses joints
7 and 11). For the rule disjunction over sides (DNF: "done with the left side" OR "with the right side"),
each primitive gets two joint masks:
  side 0 (left) : its joints minus the right arm (hand/arm primitives) or the right leg (leg primitives)
  side 1 (right): the mirror image
Non-lateral primitives (body, torso, head, two-hand, interaction) use the same mask on both sides.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

N_JOINTS = 25
ARM = ({4, 5, 6, 7, 21, 22}, {8, 9, 10, 11, 23, 24})     # (left, right), 0-based NTU joints
LEG = ({12, 13, 14, 15}, {16, 17, 18, 19})
# Lateral by prefix, but the primitive itself involves both sides.
BILATERAL = {'leg.alternating_lifts', 'hand.at_other_forearm'}
TWO_PERSON = 'inter.two_person'


def side_kind(pid):
    """'arm' / 'leg' for primitives done by one side, None otherwise."""
    head = pid.split('.')[0]
    if pid in BILATERAL:
        return None
    if head in ('hand', 'arm'):
        return 'arm'
    if head == 'leg':
        return 'leg'
    return None


def load_kg(prim_path, rules_path, n_cls):
    prims = json.load(open(prim_path, encoding='utf-8'))
    ids = [p['id'] for p in prims]
    pidx = {k: i for i, k in enumerate(ids)}
    K = len(prims)

    # joint masks (2, K, V)
    masks = np.zeros((2, K, N_JOINTS), np.float32)
    lateral = np.zeros(K, bool)
    for i, p in enumerate(prims):
        joints = set(p['joints']) or set(range(N_JOINTS))
        kind = side_kind(p['id'])
        if kind is None:
            sides = (joints, joints)
        else:
            lateral[i] = True
            left, right = ARM if kind == 'arm' else LEG
            sides = (joints - right or joints, joints - left or joints)
        for s, js in enumerate(sides):
            masks[s, i, sorted(js)] = 1

    # rules
    kg = json.load(open(rules_path, encoding='utf-8'))
    R = np.zeros((n_cls, K), np.float32)          # +1 require, -1 forbid, 0 don't care
    seqs, names = {}, [None] * n_cls
    for r in kg['rules']:
        c = r['id'] - 1
        if c >= n_cls:
            continue
        names[c] = r['class']
        assert r['require'], r['class']
        assert not set(r['require']) & set(r['forbid']), r['class']
        for k in r['require']:
            R[c, pidx[k]] = 1
        for k in r['forbid']:
            R[c, pidx[k]] = -1
        if len(r.get('sequence', [])) >= 2:
            seqs[c] = [pidx[k] for k in r['sequence']]
    assert all(n is not None for n in names), 'KG must cover every class'

    # constraints
    groups = {}
    for i, p in enumerate(prims):
        if p.get('exclusive_group'):
            groups.setdefault(p['exclusive_group'], []).append(i)
    G = np.zeros((len(groups), K), np.float32)
    for g, members in enumerate(groups.values()):
        G[g, members] = 1
    impl = [(pidx[a], pidx[b]) for a, b in kg.get('constraints', {}).get('implication', [])]

    return dict(ids=ids, prims=prims, names=names, R=R, seqs=seqs, side_masks=masks, lateral=lateral,
                groups=G, group_names=list(groups), implications=impl, two_person=pidx[TWO_PERSON])


@torch.no_grad()
def encode_primitive_text(prim_path, out_path, model_id='openai/clip-vit-large-patch14-336', device='cuda'):
    """CLIP text of each primitive: mean of normalised CLIP(description) and CLIP(phrase from id), normalised.
    Same text encoder as Neuron (ViT-L/14@336). Saved as a (K, 768) float tensor; reused if the file exists."""
    if os.path.exists(out_path):
        return torch.load(out_path)
    from transformers import CLIPModel, CLIPTokenizer
    prims = json.load(open(prim_path, encoding='utf-8'))
    clip = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(model_id)

    def enc(texts):
        b = tok(texts, padding=True, truncation=True, max_length=77, return_tensors='pt').to(device)
        f = clip.get_text_features(**b)
        if not torch.is_tensor(f):     # some transformers versions return an output object
            f = f.text_embeds if getattr(f, 'text_embeds', None) is not None else f.pooler_output
        return F.normalize(f.float(), dim=-1).cpu()

    phrase = [p['id'].replace('.', ' ').replace('_', ' ') for p in prims]
    E = F.normalize(enc([p['description'] for p in prims]) + enc(phrase), dim=-1)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    torch.save(E, out_path)
    del clip
    torch.cuda.empty_cache()
    return E
