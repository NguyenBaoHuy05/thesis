"""Primitive-KG semantics for Flora (soft use of the KG; no rule matching, no grounding on skeletons).

Two independent uses, both only touching the semantic side of Flora:

1. kg_similarity: class-class similarity from shared primitives. Used to blend
   with the CLIP text similarity that picks Flora's neighbours (trainer.load_prototypes).
2. compose: class embedding built as a weighted sum of CLIP embeddings of the
   primitive descriptions. Added on top of Flora's attuned semantics.

The KG (kg/kg_rules_v2.json) was written from class names and the primitive
vocabulary only, never from skeleton data, so using it for unseen classes
uses the same kind of knowledge as the class-name text Flora already uses.

Class index i (Flora, 0-based) <-> rule id i + 1 (A-number).
"""
import json
import os

import numpy as np


class PrimitiveKG:
    def __init__(self, prim_path, rules_path, n_classes, forbid_weight=0.5):
        with open(prim_path, 'r', encoding='utf-8') as f:
            self.prims = json.load(f)
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)['rules']
        if n_classes > len(rules):
            raise ValueError(f'KG has rules for {len(rules)} classes (NTU only), task needs {n_classes}')
        self.prim_ids = [p['id'] for p in self.prims]
        self.prim_text = [p['description'] for p in self.prims]
        idx = {p: i for i, p in enumerate(self.prim_ids)}
        by_id = {r['id']: r for r in rules}
        # W[c, p] = +1 if class c requires p, -forbid_weight if it forbids p, 0 otherwise
        W = np.zeros((n_classes, len(self.prim_ids)), dtype=np.float32)
        for c in range(n_classes):
            r = by_id[c + 1]
            for p in r['require']:
                W[c, idx[p]] = 1.0
            for p in r['forbid']:
                W[c, idx[p]] = -forbid_weight
        self.W = W

    def similarity(self):
        """(n, n) cosine similarity between class rows of W. Self-similarity is 1."""
        Wn = self.W / np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-8)
        return Wn @ Wn.T

    def compose_weights(self):
        """(n, P) non-negative weights, rows sum to 1: only `require` primitives contribute.
        A description of what is forbidden does not describe the action."""
        Wp = np.clip(self.W, 0.0, None)
        return Wp / np.maximum(Wp.sum(axis=1, keepdims=True), 1e-8)


def embed_texts(tokenizer, text_encoder, texts, max_length, device, cache_path=None):
    """Same recipe as MappingNet.extract_text_embeddings: tokens + pooled -> (N, max_length+1, 768)."""
    import torch
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, padding='max_length', max_length=max_length, truncation=True,
                            return_tensors='pt').to(device).input_ids
            e = text_encoder(ids)
            out.append(torch.cat((e['last_hidden_state'], e['pooler_output'].unsqueeze(1)), dim=1))
    emb = torch.cat(out, dim=0).detach().clone()
    if cache_path:
        torch.save(emb.cpu(), cache_path)
    return emb


def compose_semantics(kg, prim_emb, text_dim_total):
    """(n, token, text_dim_total) class embeddings composed from primitive embeddings.
    prim_emb: (P, token, 768). Flora's class embedding is [csv-text ; llm-text] = 1536 wide,
    primitives have one text type, so the composed vector is repeated to fill the width."""
    import torch
    w = torch.from_numpy(kg.compose_weights()).to(prim_emb.device, prim_emb.dtype)  # (n, P)
    comp = torch.einsum('np,ptd->ntd', w, prim_emb)  # (n, token, 768)
    reps = text_dim_total // comp.shape[-1]
    return torch.cat([comp] * reps, dim=-1)


def class_primitive_texts(kg):
    """One sentence per class: its required primitive descriptions joined in KG order."""
    texts = []
    for c in range(kg.W.shape[0]):
        parts = [kg.prim_text[p] for p in np.nonzero(kg.W[c] > 0)[0]]
        texts.append(' '.join(parts) if parts else 'A person moves.')
    return texts


def compose_text_semantics(kg, tokenizer, text_encoder, max_length, text_dim_total, device, cache_path=None):
    """(n, token, text_dim_total): CLIP embedding of each class's primitive sentence, repeated to fill
    the width. Unlike compose_semantics the tokens come from one coherent sequence, so token
    positions line up the same way as in Flora's own class texts."""
    import torch
    emb = embed_texts(tokenizer, text_encoder, class_primitive_texts(kg), max_length, device, cache_path)
    reps = text_dim_total // emb.shape[-1]
    return torch.cat([emb] * reps, dim=-1)


def mask_half(sem, half):
    """Zero one half of a [csv ; llm] class embedding. half: both | csv | llm (the half that is kept)."""
    if half == 'both':
        return sem
    sem = sem.clone()
    d = sem.shape[-1] // 2
    if half == 'csv':
        sem[..., d:] = 0
    elif half == 'llm':
        sem[..., :d] = 0
    else:
        raise ValueError(half)
    return sem


if __name__ == '__main__':
    # Offline sanity check (no torch / CLIP needed): python kg_semantics.py [n_classes]
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    kg = PrimitiveKG(os.path.join(here, 'kg', 'primitive_v2.json'),
                     os.path.join(here, 'kg', 'kg_rules_v2.json'), n)
    import csv
    names = [r['label'] for r in csv.DictReader(open(os.path.join(here, 'semantics', f'ntu{n}.csv'), encoding='utf-8'))] \
        if n in (60, 120) else None
    with open(os.path.join(here, 'kg', 'kg_rules_v2.json'), encoding='utf-8') as f:
        cls = [r['class'] for r in json.load(f)['rules']][:n]
    S = kg.similarity()
    print(f'{n} classes, {len(kg.prim_ids)} primitives, mean #require={(kg.W > 0).sum(1).mean():.1f}, '
          f'mean #forbid={(kg.W < 0).sum(1).mean():.1f}')
    off = S[~np.eye(n, dtype=bool)]
    print(f'KG sim off-diagonal: mean={off.mean():.3f} p90={np.percentile(off, 90):.3f} max={off.max():.3f}')
    iu = np.triu_indices(n, 1)
    ident = [(a, b) for a, b in zip(*iu) if S[a, b] > 0.999]
    print(f'class pairs with identical KG rows (KG cannot tell them apart): {len(ident)}')
    for a, b in ident[:6]:
        print(f'  {a + 1} {cls[a]}  ==  {b + 1} {cls[b]}')
    for c in range(0, n, max(1, n // 8)):
        order = [j for j in np.argsort(-S[c]) if j != c][:3]
        print(f'  {c + 1:3d} {cls[c]:32s} -> ' + ' | '.join(f'{cls[j]} ({S[c, j]:.2f})' for j in order))
