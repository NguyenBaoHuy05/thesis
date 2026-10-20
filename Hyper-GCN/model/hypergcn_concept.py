"""Hyper-GCN large + the concept-action graph (model/concept_action_graph.py) as a feature refinement of K0.

    f    = mean over (actors, T, V) of the l9 map            (exactly Model.forward's pooling)
    out  = ConceptActionGraphHead(l9 map averaged over T (S = 1, as the cache), valid actors)
    r    = sum_k softmax(out.logits)_k h_k                     (R-GCN action nodes weighted by the graph's belief)
    f'   = f + W [LN(r) ; c]                                   (W zero-initialized: f' = f at the start)
    logits = fc(f')                                            (fc of K0)

At initialization the model reproduces the checkpoint (K0) exactly; the graph can only correct K0, not replace it.
Loss (main.py): CE(fc(f')) + divergence of the backbone + head_weight * (CE(head) + beta BCE(c, M[y]) + the head's
extra_loss), so the graph keeps the training signal it had on the cache. The head is loaded from `head_weights`
(head.pt of train_concept_logic.py, same matrix / graph / head_args).

eval_outputs(): logits of the last forward besides the refined ones (k0 = fc(f) of the same backbone, head, head_sym),
collected by main.py at test time. One GPU only (the last forward is kept on the module).
"""

import torch
import torch.nn as nn

from model.actors import valid_actor_mask
from model.concept_action_graph import ConceptActionGraphHead
from model.concept_logic import concept_joint_mask, concept_loss, load_concepts
from model.concept_schema import NTU_ACTION_NAMES
from model.hypergcn_large import Model as HyperGCN


class ConceptRefiner(nn.Module):
    def __init__(self, in_channels, matrix, metadata, graph=None, head_args=None, beta=0.5, head_weight=1.0,
                 num_person=2):
        super().__init__()
        from train_concept_logic import make_graph                 # avoids a model -> script import at load time
        M, names, meta = load_concepts(matrix, metadata, NTU_ACTION_NAMES)
        graph = make_graph(matrix, graph or {})
        head_args = dict(head_args or {})
        decoder_args = dict(dict(num_person=num_person, max_frames=1), **head_args.pop('decoder_args', {}))
        action_args = head_args.pop('action_decoder_args', {})
        self.head = ConceptActionGraphHead(in_channels, graph, concept_joint_mask(names, meta),
                                           decoder_args=decoder_args, action_decoder_args=action_args, **head_args)
        if not self.head.use_rgcn:
            raise ValueError('the refinement reads the R-GCN action nodes: use_rgcn must be True')
        if not torch.equal(self.head.M, M):
            raise ValueError('graph rows differ from the concept matrix')
        d = self.head.decoder.norm.normalized_shape[0]
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d + M.shape[1], in_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.beta, self.head_weight = float(beta), float(head_weight)

    def forward(self, features, valid):
        """features [N, M, C, T, V] -> (delta [N, C], head output dict)."""
        out = self.head(features.mean(3, keepdim=True), valid)
        p = torch.softmax(out['logits'], 1)
        r = torch.einsum('na,nad->nd', p, out['h_action'])
        return self.proj(torch.cat([self.norm(r), out['c']], 1)), out

    def loss(self, out, labels):
        loss, terms = concept_loss(out, labels, self.head.M, beta=self.beta)
        extra, extra_terms = self.head.extra_loss(out, labels)
        terms = {'head_' + k: v for k, v in dict(terms, **extra_terms).items()}
        return self.head_weight * (loss + extra), terms


class Model(HyperGCN):
    def __init__(self, concept_args=None, head_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.cg = ConceptRefiner(self.embedding_channels * 4, num_person=kwargs.get('num_person', 2),
                                 **(concept_args or {}))
        if head_weights:
            state = torch.load(head_weights, map_location='cpu')
            self.cg.head.load_state_dict(state)
        self._last = None

    def forward(self, x):
        features, hyper_outputs = self.features(x)
        pooled = features.flatten(3).mean(3).mean(1)
        delta, out = self.cg(features, valid_actor_mask(x))
        output = self.fc(self.drop_out(pooled + delta))
        self._last = dict(out=out, k0=self.fc(self.drop_out(pooled)).detach())
        return output, hyper_outputs

    def extra_loss(self, labels):
        """Head losses of the last forward (main.py adds them to CE + divergence)."""
        loss, terms = self.cg.loss(self._last['out'], labels)
        terms['delta_norm'] = self.cg.proj.weight.norm().item()
        return loss, terms

    def eval_outputs(self):
        out = self._last['out']
        return dict(k0=self._last['k0'], head=out['logits'].detach(), head_sym=out['logits_sym'].detach())
