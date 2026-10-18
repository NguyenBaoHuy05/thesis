"""Hyper-GCN large + the A1 KG head, trained end to end through main.py (A1 plan P6, option A).

    model: model.hypergcn_kg.Model
    model_args: {<hypergcn_large args>, kg: annotations/action_kg.json,
                 embeddings: annotations/kg_emb_mpnet.npz, head_args: {...}, head_weights: <head.pt>}

The head reads the same z as the cached features (tools/cache_features.py): l9 output, mean over
(T, V) per actor, channel-wise max over the real actors. A head trained on the cache
(train_kg_head.py, head.pt) therefore starts from where it ended. Its parameters live under the
`kg.` prefix, an extension key (model/training.py): absent from official checkpoints, always
trainable.

head_args.attribute_source: decoder replaces the attribute head a = sigmoid(W_a z) by
model/attribute_decoder.py on the l9 map before pooling, masked by the KG `body_parts`
(model/kg.py: attribute_joint_mask). z stays the input of RETRIEVE and of the FUSE query. A cache
head (head_weights) still loads: its `attribute.*` is dropped, the decoder starts at a = 0.5
(VERIFY = 0), so epoch 0 is that head without VERIFY.

forward returns (logits, hyper_outputs, kg_outputs). main.py uses logits like the plain model's
(Top-k, eval loss) and the A1 loss on kg_outputs while training. Logits outside the candidates
are -1e4 instead of -inf so main.py's eval cross-entropy stays finite; the ranking is unchanged.
"""

import torch

from model.hypergcn_large import Model as HyperGCN
from model.kg import attribute_joint_mask, kg_tensors, load_embeddings, load_kg
from model.kg_retrieval import KGRetrievalHead

OUTSIDE_CANDIDATES = -1e4


def valid_actor_mask(x, atol=1e-6):
    """x: [N, C, T, V, M] skeleton after feeder preprocessing. An actor is valid when it is a
    real person: not all-zero, and (for m > 0) not a copy of the first actor, which is how
    align_frames fills the second slot of a single-person sample. The feeder transforms every
    actor identically (same rotation, crop/resize, centering, bone/velocity), so a copy stays
    a copy and an all-zero actor stays all-zero."""
    nonzero = x.abs().amax(dim=(1, 2, 3)) > atol
    copy = (x - x[..., :1]).abs().amax(dim=(1, 2, 3)) <= atol
    copy[:, 0] = False
    return nonzero & ~copy


def pooled_z(features, valid):
    """features: [N, M, C, T, V] backbone output. valid: [N, M] from `valid_actor_mask`.
    Returns z [N, C]: mean over (T, V) per actor, then channel-wise max over valid actors."""
    if not torch.all(valid.any(dim=1)):
        raise ValueError('sample with no valid actor at all (first actor should always be present)')
    pooled = features.mean(dim=(3, 4))
    pooled = pooled.masked_fill(~valid.unsqueeze(-1), float('-inf'))
    z, _ = pooled.max(dim=1)
    return z


class Model(HyperGCN):
    def __init__(self, kg, embeddings, embedding_key='action', head_args=None, head_weights=None,
                 **backbone_args):
        super().__init__(**backbone_args)
        kg_json = load_kg(kg)
        class_embeddings, attribute_embeddings = load_embeddings(embeddings, kg_json, embedding_key)
        if len(class_embeddings) != self.num_class:
            raise ValueError(f'{len(class_embeddings)} class embeddings for {self.num_class} classes')
        self.kg_hash = kg_json['hash']
        head_args = dict(head_args or {})
        if head_args.get('attribute_source') == 'decoder':
            head_args['joint_mask'] = attribute_joint_mask(
                kg_json, self.num_point, mirror=head_args.pop('mirror_body_parts', True))
            head_args['decoder_args'] = dict(dict(num_person=backbone_args.get('num_person', 2)),
                                             **(head_args.get('decoder_args') or {}))
        self.kg = KGRetrievalHead(self.fc.in_features, class_embeddings, attribute_embeddings,
                                  kg_tensors(kg_json), **head_args)
        if head_weights:
            # The KG buffers come from `kg` above, not from the file: a head trained with another
            # KG (e.g. the real one) cannot be loaded onto a shuffled KG by mistake.
            state = torch.load(head_weights, map_location='cpu')
            buffers = {name for name, _ in self.kg.named_buffers()}
            mismatched = [name for name in buffers & state.keys()
                          if not torch.equal(state[name].float(), dict(self.kg.named_buffers())[name].float())]
            if mismatched:
                raise ValueError(f'{head_weights} was trained with other KG buffers: {mismatched}')
            if self.kg.attribute_source == 'decoder':
                missing, unexpected = self.kg.load_state_dict(state, strict=False)
                if [k for k in missing if not k.startswith('decoder.')] or                         [k for k in unexpected if not k.startswith('attribute.')]:
                    raise ValueError(f'{head_weights}: missing {missing}, unexpected {unexpected}')
            else:
                self.kg.load_state_dict(state)

    def forward(self, x, labels=None):
        """labels only while training: they force the ground truth into the candidates."""
        features, hyper_outputs = self.features(x)
        valid = valid_actor_mask(x)
        z = pooled_z(features, valid)
        out = self.kg(z, labels=labels if self.training else None, features=features, valid=valid)
        logits = out['logits'].clamp_min(OUTSIDE_CANDIDATES)
        return logits, hyper_outputs, out
