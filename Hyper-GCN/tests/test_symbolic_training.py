"""CPU smoke checks of adapter-only gradients, frozen BN and initial checkpoint retention."""

import argparse
import ast
from collections import OrderedDict
import json
import os
from pathlib import Path
import pickle
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import yaml

from model.hypergcn_base import Model as HyperGCNBase
from model.symbolic_hyperconcept import SymbolicHyperConceptBridge
from model.symbolic_metrics import concept_metrics
from model.training import configure_trainable_parameters, set_training_mode


ROOT = Path(__file__).resolve().parents[1]


def processor_namespace():
    # Exercise real Processor methods without optional TensorBoard or CUDA init.
    source = ast.parse((ROOT / 'main.py').read_text())
    definitions = ast.Module(body=[node for node in source.body
                                  if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                                  and node.name in ('Processor', 'DictAction', 'str2bool', 'get_parser')],
                             type_ignores=[])
    namespace = dict(torch=torch, nn=nn, optim=optim, np=np, json=json, os=os,
                     OrderedDict=OrderedDict, pickle=pickle, time=time, argparse=argparse,
                     concept_metrics=concept_metrics, set_training_mode=set_training_mode,
                     tqdm=lambda loader, **kwargs: loader)
    exec(compile(definitions, str(ROOT / 'main.py'), 'exec'), namespace)
    return namespace


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, tag, value, step):
        self.values[tag] = value


class TinyFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout(.5))
        self.fc = nn.Linear(8, 120)
        self.symbolic_graph = SymbolicHyperConceptBridge(
            8, ROOT / 'concepts/ntu_action_concept_v2_4.csv',
            ROOT / 'concepts/ntu_concept_metadata_v2_4.csv', 120)
        self.symbolic_graph_weight = .1

    def forward(self, inputs):
        features = self.backbone(inputs)
        fused, concepts = self.symbolic_graph(features)
        return self.fc(fused.mean((2, 3))), [], concepts


class SymbolicOnlyTrainingTests(unittest.TestCase):
    def test_prototype_mode_bypasses_blackbox_fc(self):
        model = HyperGCNBase(
            num_class=120, num_point=25, num_person=2, hyper_joints=3,
            graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'virtual_ensemble'},
            symbolic_graph_enabled=True,
            concept_csv=ROOT / 'concepts/ntu_action_concept_v2_4.csv',
            metadata_csv=ROOT / 'concepts/ntu_concept_metadata_v2_4.csv',
            symbolic_action_mode='prototype',
            symbolic_action_decoder='tversky',
            symbolic_action_temperature=.15,
            symbolic_hypergraph_mode='learnable_v1',
            symbolic_hypergraph_reg_weight=.01,
        )

        class Block(nn.Module):
            def __init__(self, out_channels, stride=1):
                super().__init__()
                self.out_channels = out_channels
                self.stride = stride

            def forward(self, inputs):
                outputs = inputs[:, :, ::self.stride]
                if outputs.shape[1] < self.out_channels:
                    outputs = torch.cat([outputs, outputs], dim=1)
                return outputs[:, :self.out_channels], torch.zeros(1)

        model.l1 = Block(128)
        model.l2 = Block(128)
        model.l3 = Block(128)
        model.l4 = Block(256, 2)
        model.l5 = Block(256)
        model.l6 = Block(256)
        model.l7 = Block(256, 2)
        model.l8 = Block(256)
        model.l9 = Block(256)
        fc_calls = []
        model.fc.register_forward_hook(lambda *args: fc_calls.append(True))
        configure_trainable_parameters(model, symbolic_only=True)
        model.symbolic_graph.hypergraph_gate.data.fill_(.2)
        output, _, concepts = model(torch.randn(2, 3, 8, 25, 2))
        self.assertEqual(output.shape, (2, 120))
        self.assertEqual(concepts.shape, (2, 102))
        self.assertEqual(fc_calls, [])
        self.assertFalse(model.fc.weight.requires_grad)
        self.assertFalse(model.symbolic_graph.concept_graph_gate.requires_grad)
        self.assertTrue(model.symbolic_graph.hypergraph_delta.requires_grad)
        self.assertTrue(model.symbolic_graph.hyperedge_weight_raw.requires_grad)
        self.assertTrue(model.symbolic_graph.concept_queries.requires_grad)
        self.assertTrue(
            model.symbolic_graph.grounded_encoder.channel_projection[0].weight.requires_grad
        )
        output.sum().backward()
        self.assertIsNone(model.fc.weight.grad)
        self.assertIsNotNone(model.symbolic_graph.concept_queries.grad)
        self.assertIsNotNone(model.symbolic_graph.hypergraph_delta.grad)
        self.assertIsNotNone(
            model.symbolic_graph.grounded_encoder.channel_projection[0].weight.grad
        )

    def test_optimizer_gradients_and_pretrained_state_remain_frozen(self):
        torch.manual_seed(123)
        model = TinyFeatureModel()
        counts = configure_trainable_parameters(model, symbolic_only=True)
        self.assertEqual(counts['trainable'], sum(p.numel() for p in model.symbolic_graph.parameters()))
        frozen = {key: value.clone() for key, value in model.state_dict().items()
                  if not key.startswith('symbolic_graph.')}
        queries = model.symbolic_graph.concept_queries.detach().clone()
        features = torch.randn(4, 8, 3, 50)
        labels = torch.tensor([0, 1, 2, 3])
        model.eval()
        with torch.no_grad():
            baseline = model.fc(model.backbone(features).mean((2, 3)))
            torch.testing.assert_close(model(features)[0], baseline, rtol=0, atol=0)

        processor_type = processor_namespace()['Processor']
        processor = processor_type.__new__(processor_type)
        processor.model = model
        processor.arg = argparse.Namespace(symbolic_only=True, optimizer='SGD', base_lr=.001,
                                           weight_decay=.0004, nesterov=True, warm_up_epoch=0,
                                           step=[12, 17], lr_decay_rate=.1)
        processor.print_log = lambda *args, **kwargs: None
        processor.load_optimizer()
        expected = {id(p) for p in model.symbolic_graph.parameters()}
        actual = {id(p) for group in processor.optimizer.param_groups for p in group['params']}
        self.assertEqual(actual, expected)
        processor.loss = nn.CrossEntropyLoss(label_smoothing=.1)
        processor.h_loss = mock.Mock(side_effect=AssertionError('Divergence must not run'))
        processor.data_loader = {'train': [(features, labels, torch.arange(4))]}
        processor.train_writer = Writer()
        processor.global_step = 0
        processor.output_device = 0
        with mock.patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
            for epoch in range(3):
                processor.train(epoch)
        processor.h_loss.assert_not_called()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.fc.training)
        self.assertTrue(model.symbolic_graph.training)
        for key, before in frozen.items():
            torch.testing.assert_close(model.state_dict()[key], before, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            if not name.startswith('symbolic_graph.'):
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
        self.assertFalse(torch.equal(model.symbolic_graph.concept_queries, queries))
        self.assertEqual(processor.train_writer.values['loss/divergence'], 0.)
        ce = processor.loss(model(features)[0], labels)
        ce.backward()

    def test_modes_after_eval_and_data_parallel(self):
        model = nn.DataParallel(TinyFeatureModel())
        configure_trainable_parameters(model, symbolic_only=True)
        for _ in range(2):
            model.eval()
            set_training_mode(model, symbolic_only=True)
            self.assertFalse(model.module.backbone[1].training)
            self.assertFalse(model.module.backbone[3].training)
            self.assertTrue(model.module.symbolic_graph.training)
        set_training_mode(model, symbolic_only=False)
        self.assertTrue(model.module.backbone[1].training)
        model.module.fc.weight.requires_grad_(False)
        configure_trainable_parameters(model, symbolic_only=False)
        self.assertFalse(model.module.fc.weight.requires_grad)
        with self.assertRaisesRegex(ValueError, 'enabled symbolic_graph'):
            configure_trainable_parameters(nn.Linear(8, 120), symbolic_only=True)

    def test_active_yaml_is_supported_by_parser(self):
        parser = processor_namespace()['get_parser']()
        config = yaml.safe_load((ROOT / 'config/base/nturgbd120-cross-subject/hyper_joint_symbolic.yaml').read_text())
        self.assertFalse(set(config) - set(vars(parser.parse_args([]))))
        parser.set_defaults(**config)
        arg = parser.parse_args(['--weights', 'official.pt'])
        self.assertTrue(arg.symbolic_only)
        self.assertFalse(arg.eval_before_train)
        self.assertEqual(arg.optimizer, 'Adam')
        self.assertEqual(arg.base_lr, 3e-4)
        self.assertEqual(arg.warm_up_epoch, 1)
        self.assertFalse(arg.nesterov)  # parser default; omitted from Adam YAML
        self.assertEqual(arg.num_epoch, 20)
        self.assertEqual(arg.model_args['symbolic_action_decoder'], 'tversky')
        self.assertEqual(arg.model_args['symbolic_concept_dim'], 64)
        self.assertEqual(arg.model_args['symbolic_temporal_bins'], 4)
        self.assertFalse(parser.parse_args(['--symbolic-only', 'False']).symbolic_only)

    def test_start_retains_epoch_zero_when_training_accuracy_worsens(self):
        labels = torch.tensor([0, 1, 0])

        class Scores(nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = nn.Parameter(torch.tensor([[3., -3.], [-3., 3.], [3., -3.]]))

            def forward(self, indices):
                return self.scores[indices.long()]

        class Dataset:
            sample_name = ['a', 'b', 'c']

            def top_k(self, scores, k):
                return float(np.mean([label in row[-k:] for label, row in zip(labels.numpy(), scores.argsort())]))

        class Loader:
            dataset = Dataset()

            def __len__(self):
                return 1

            def __iter__(self):
                yield torch.arange(3), labels, torch.arange(3)

        processor_type = processor_namespace()['Processor']
        processor = processor_type.__new__(processor_type)
        processor.model = Scores()
        initial_scores = processor.model.scores.detach().clone()
        processor.loss = nn.CrossEntropyLoss()
        processor.data_loader = {'train': Loader(), 'test': Loader()}
        processor.global_step = 0
        processor.output_device = 0
        processor.best_acc = 0
        processor.best_acc_epoch = 0
        processor.best_weights_path = None
        processor.baseline_accuracy = {}
        processor.val_writer = Writer()
        processor.print_log = lambda *args, **kwargs: None

        def worsen(epoch, save_model=False):
            with torch.no_grad():
                processor.model.scores.copy_(-initial_scores)
            processor.global_step += 1

        processor.train = worsen
        with tempfile.TemporaryDirectory() as directory:
            processor.arg = argparse.Namespace(work_dir=directory, model_saved_name=os.path.join(directory, 'runs'),
                                               phase='train', feeder='synthetic', show_topk=[1],
                                               symbolic_only=True, eval_before_train=True, start_epoch=0,
                                               num_epoch=1, save_interval=1, save_epoch=99, save_score=False,
                                               weights='official.pt', device=[0], print_log=True,
                                               weight_decay=.0004, base_lr=.001, batch_size=3,
                                               test_batch_size=3, seed=1)
            with mock.patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
                processor.start()
            self.assertTrue(Path(processor.best_weights_path).exists())
            initial = json.loads((Path(directory) / 'initial_evaluation.json').read_text())
            self.assertEqual(initial['top1']['test'], 1.)
            later = json.loads((Path(directory) / 'epoch1_test_action_metrics.json').read_text())
            self.assertEqual(later['delta_from_initial_pp'], -100.)
            self.assertTrue((Path(directory) / 'epoch0_test_score.pkl').exists())
        self.assertEqual(processor.best_acc, 1.)
        self.assertEqual(processor.best_acc_epoch, 0)
        torch.testing.assert_close(processor.model.scores, initial_scores, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
