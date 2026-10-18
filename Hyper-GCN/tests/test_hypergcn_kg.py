import json
import os
import tempfile
import unittest

import numpy as np
import torch

from model import training
from model.checkpoint import load_model_weights
from model.hypergcn_kg import OUTSIDE_CANDIDATES, Model, pooled_z, valid_actor_mask
from model.hypergcn_large import Model as HyperGCN
from model.kg_retrieval import a1_loss
from tools.shuffle_kg import kg_hash

BACKBONE = dict(num_class=4, num_point=25, num_person=2, hyper_joints=3,
                graph='graph.ntu_rgb_d.Graph', graph_args=dict(labeling_mode='virtual_ensemble'))


def edge(w):
    return {'w': w, 'agree': 1.0, 'evidence': ''}


def write_kg(folder, flip=False):
    attrs = {f't{j}': {'text': f't{j}', 'definition': ''} for j in range(3)}
    actions = {}
    for i in range(4):
        has, lacks = (f't{i % 3}', f't{(i + 1) % 3}') if not flip else (f't{(i + 1) % 3}', f't{i % 3}')
        actions[f'a{i}'] = {'id': i, 'text': f'a{i}', 'has': {has: edge(2)}, 'lacks': {lacks: edge(1)},
                            'similar': [], 'opposite': []}
    kg = {'attributes': attrs, 'actions': actions}
    kg['hash'] = kg_hash(kg)
    kg_path = os.path.join(folder, 'kg_flip.json' if flip else 'kg.json')
    with open(kg_path, 'w', encoding='utf-8') as handle:
        json.dump(kg, handle)
    emb_path = os.path.join(folder, 'emb_flip.npz' if flip else 'emb.npz')
    rng = np.random.default_rng(0)
    np.savez(emb_path, action=rng.standard_normal((4, 6)).astype(np.float32),
             attribute=rng.standard_normal((3, 6)).astype(np.float32),
             action_names=np.array([f'a{i}' for i in range(4)]),
             attribute_names=np.array(list(attrs)), kg_hash=np.array(kg['hash']))
    return kg_path, emb_path


HEAD = dict(k=2, d_model=8, heads=2)


class HypergcnKgTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.kg_path, self.emb_path = write_kg(self.folder.name)
        torch.manual_seed(0)
        self.model = Model(kg=self.kg_path, embeddings=self.emb_path, head_args=HEAD, **BACKBONE)
        self.x = torch.randn(2, 3, 64, 25, 2)

    def tearDown(self):
        self.folder.cleanup()

    def test_eval_logits_are_finite_and_rank_the_candidates_first(self):
        self.model.eval()
        with torch.no_grad():
            logits, hyper, out = self.model(self.x)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.isfinite(logits).all())
        outside = torch.ones_like(logits, dtype=torch.bool).scatter(1, out['cand'], False)
        self.assertTrue((logits[outside] == OUTSIDE_CANDIDATES).all())
        self.assertIsNone(out['target'])

    def test_head_sees_the_cached_pooling(self):
        self.model.eval()
        with torch.no_grad():
            features, _ = self.model.features(self.x)
            z = pooled_z(features, valid_actor_mask(self.x))
            _, _, out = self.model(self.x)
            torch.testing.assert_close(out['s'], self.model.kg.retrieval(z))

    def test_official_checkpoint_loads_with_new_head(self):
        official = HyperGCN(**BACKBONE).state_dict()
        report = load_model_weights(self.model, official, allow_new=True)
        self.assertTrue(report['initialized'])
        self.assertTrue(all(training.is_extension_key(key) for key in report['initialized']))

    def test_training_step_reaches_backbone_and_head(self):
        self.model.train()
        labels = torch.tensor([3, 0])
        logits, hyper, out = self.model(self.x, labels)
        self.assertTrue((out['cand'] == labels.unsqueeze(1)).any(1).all())
        loss, terms = a1_loss(out, labels, self.model.kg, gamma=0.1)
        loss.backward()
        self.assertGreater(self.model.l9.gcn1.conv_d.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.kg.attribute.weight.grad.abs().sum().item(), 0)

    def test_head_weights_round_trip_and_reject_another_kg(self):
        path = os.path.join(self.folder.name, 'head.pt')
        torch.save(self.model.kg.state_dict(), path)
        again = Model(kg=self.kg_path, embeddings=self.emb_path, head_args=HEAD, head_weights=path, **BACKBONE)
        for key, value in self.model.kg.state_dict().items():
            torch.testing.assert_close(again.kg.state_dict()[key], value)
        flip_kg, flip_emb = write_kg(self.folder.name, flip=True)
        with self.assertRaises(ValueError):
            Model(kg=flip_kg, embeddings=flip_emb, head_args=HEAD, head_weights=path, **BACKBONE)



DECODER_HEAD = dict(HEAD, attribute_source='decoder',
                    decoder_args=dict(d_model=8, heads=2, layers=1, dropout=0.0))


class HypergcnKgDecoderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.kg_path, self.emb_path = write_kg(self.folder.name)
        torch.manual_seed(0)
        self.x = torch.randn(2, 3, 64, 25, 2)

    def tearDown(self):
        self.folder.cleanup()

    def test_cache_head_loads_and_decoder_starts_with_verify_off(self):
        cache = Model(kg=self.kg_path, embeddings=self.emb_path, head_args=HEAD, **BACKBONE)
        path = os.path.join(self.folder.name, 'head.pt')
        torch.save(cache.kg.state_dict(), path)
        model = Model(kg=self.kg_path, embeddings=self.emb_path, head_args=DECODER_HEAD,
                      head_weights=path, **BACKBONE)
        model.load_state_dict({k: v for k, v in cache.state_dict().items()
                               if not k.startswith('kg.')}, strict=False)
        self.assertFalse(hasattr(model.kg, 'attribute'))
        model.eval()
        with torch.no_grad():
            _, _, out = model(self.x)
        torch.testing.assert_close(out['a'], torch.full_like(out['a'], 0.5))
        torch.testing.assert_close(out['v'], torch.zeros_like(out['v']))
        # Same retrieval as the cache head: z is unchanged.
        cache.eval()
        with torch.no_grad():
            _, _, ref = cache(self.x)
        torch.testing.assert_close(out['s'], ref['s'])

    def test_training_step_reaches_decoder(self):
        model = Model(kg=self.kg_path, embeddings=self.emb_path, head_args=DECODER_HEAD, **BACKBONE)
        model.train()
        labels = torch.tensor([3, 0])
        _, _, out = model(self.x, labels)
        loss, _ = a1_loss(out, labels, model.kg, gamma=0.1)
        loss.backward()
        self.assertGreater(model.kg.decoder.out.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(training.is_extension_key(n) for n, _ in model.named_parameters()
                            if 'decoder' in n))


if __name__ == '__main__':
    unittest.main()
