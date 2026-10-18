"""Offline text embeddings of the KG nodes (A1 plan S3).

  python tools/build_kg_embeddings.py [--kg annotations/action_kg.json] [--encoder mpnet|bge|clip]
      [--out annotations/kg_emb_<encoder>.npz] [--device cpu]

Each node is embedded as the L2-normalized mean of its normalized descriptions (a prompt
ensemble): an action uses `text` + `paraphrases`, an attribute uses `text` + `definition`.
`action_text` keeps the single-`text` embedding for an ablation of the ensemble.

The .npz holds `action [120, D]` in label order (row i = action with `id` i, as the feeder's
labels), `attribute [A, D]` in the JSON key order, the matching `action_names` /
`attribute_names`, the encoder id and the KG `hash`, which the head checks on load.
Run locally or on Colab and commit the .npz: Kaggle then needs no text encoder.
"""

import argparse
import json
import os

import numpy as np

ENCODERS = {
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'bge': 'BAAI/bge-large-en-v1.5',
    'clip': 'openai/clip-vit-base-patch32',
}
SENTENCE_TRANSFORMERS = ('mpnet', 'bge')


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_encoder(name, device):
    """Returns encode(list of str) -> [n, D] float32 array, rows L2-normalized."""
    model_id = ENCODERS[name]
    if name in SENTENCE_TRANSFORMERS:
        # bge needs no query instruction here: both sides of every comparison are KG texts.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, device=device)
        return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                          convert_to_numpy=True).astype(np.float32)

    import torch
    from transformers import CLIPModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def encode(texts):
        # CLIP's text tower stops at 77 tokens; longer definitions are truncated.
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors='pt')
        # text_model + text_projection is what get_text_features computes; newer transformers
        # wrap its result in a ModelOutput, so call the two parts directly.
        pooled = model.text_model(**tokens.to(device)).pooler_output
        features = model.text_projection(pooled)
        return normalize(features.float().cpu().numpy())
    return encode


def node_texts(kg):
    """Descriptions per node, actions sorted by label id, attributes in JSON key order."""
    actions = sorted(kg['actions'].items(), key=lambda item: item[1]['id'])
    ids = [a['id'] for _, a in actions]
    if ids != list(range(len(actions))):
        raise ValueError(f'action ids must be 0..{len(actions) - 1}, got {ids}')
    action = [(name, [a['text']] + a.get('paraphrases', [])) for name, a in actions]
    attribute = [(name, [a['text'], a['definition']]) for name, a in kg['attributes'].items()]
    return action, attribute


def ensemble(encode, nodes):
    """One encoder call for all descriptions, then mean + renormalize per node."""
    texts = [t for _, descriptions in nodes for t in descriptions]
    flat = encode(texts)
    out, start = [], 0
    for _, descriptions in nodes:
        out.append(flat[start:start + len(descriptions)].mean(0))
        start += len(descriptions)
    return normalize(np.stack(out))


def build(kg, encode, encoder_id):
    action, attribute = node_texts(kg)
    return dict(
        action=ensemble(encode, action).astype(np.float32),
        action_text=encode([descriptions[0] for _, descriptions in action]).astype(np.float32),
        attribute=ensemble(encode, attribute).astype(np.float32),
        action_names=np.array([name for name, _ in action]),
        attribute_names=np.array([name for name, _ in attribute]),
        encoder=np.array(encoder_id),
        kg_hash=np.array(kg['hash']),
    )


def report(kg, arrays):
    """Diagnostics for retrieval: how crowded the action embedding space is."""
    names = list(arrays['action_names'])
    sim = arrays['action'] @ arrays['action'].T
    np.fill_diagonal(sim, -np.inf)
    nearest = sim.argmax(1)
    off = sim[np.isfinite(sim)]
    related = sum(names[j] in kg['actions'][names[i]]['similar'] + kg['actions'][names[i]]['opposite']
                  for i, j in enumerate(nearest))
    print(f"action: {arrays['action'].shape}, attribute: {arrays['attribute'].shape}, "
          f"encoder {arrays['encoder']}, kg {str(arrays['kg_hash'])[:19]}...")
    index = {name: i for i, name in enumerate(names)}
    opposite = sorted({tuple(sorted((index[a], index[b]))) for a in names
                       for b in kg['actions'][a]['opposite']})
    print(f'action cosine off-diagonal: mean {off.mean():.3f}, max {off.max():.3f}, '
          f'pairs > 0.95: {int((np.triu(sim, 1) > 0.95).sum())}')
    print(f'nearest action listed as similar/opposite in the KG: {related}/{len(names)}')
    if opposite:
        # Cosine scales differ between encoders (bge packs everything near 1), so compare the
        # rank of the opposite partner among the 119 other actions: 1 = nearest; higher is better.
        rank = (-sim).argsort(1).argsort(1) + 1
        ranks = [rank[i, j] for i, j in opposite] + [rank[j, i] for i, j in opposite]
        print(f"KG opposite pairs ({len(opposite)}): mean cosine "
              f"{np.mean([sim[i, j] for i, j in opposite]):.3f}, mean rank of the partner "
              f"{np.mean(ranks):.1f} (1 = nearest)")
    top = np.dstack(np.unravel_index(np.argsort(-np.triu(sim, 1), axis=None)[:10], sim.shape))[0]
    print('closest action pairs (hardest for text-only retrieval):')
    for i, j in top:
        print(f'  {sim[i, j]:.3f}  {names[i]} / {names[j]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--kg', default=os.path.join(repo, 'annotations', 'action_kg.json'))
    parser.add_argument('--encoder', choices=sorted(ENCODERS), default='mpnet')
    parser.add_argument('--out', help='default: annotations/kg_emb_<encoder>.npz')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    out = args.out or os.path.join(repo, 'annotations', f'kg_emb_{args.encoder}.npz')

    with open(args.kg, encoding='utf-8') as handle:
        kg = json.load(handle)
    arrays = build(kg, load_encoder(args.encoder, args.device), ENCODERS[args.encoder])
    np.savez(out, **arrays)
    report(kg, arrays)
    print(f'Saved {out}')
