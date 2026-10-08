"""CPU unit checks for concept schema, loss, metrics and pretrained initialization."""

import argparse
import ast
import csv
from collections import OrderedDict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from model.checkpoint import load_model_weights
from model.concept_schema import (
    BODY_PART_JOINTS, concept_parts, load_concept_tables,
)
from model.symbolic_hyperconcept import SymbolicHyperConceptBridge, _concept_affinity
from model.symbolic_metrics import concept_metrics


ROOT = Path(__file__).resolve().parents[1]
CONCEPT_CSV = ROOT / 'concepts/ntu_action_concept_v2_4.csv'
METADATA_CSV = ROOT / 'concepts/ntu_concept_metadata_v2_4.csv'


def bridge(**kwargs):
    return SymbolicHyperConceptBridge(8, CONCEPT_CSV, METADATA_CSV, 120, **kwargs)


class ConceptSchemaTests(unittest.TestCase):
    def test_existing_tables_and_body_partition(self):
        for classes in (60, 120):
            values, metadata = load_concept_tables(CONCEPT_CSV, METADATA_CSV, classes, 102)
            self.assertEqual(len(values), classes)
            self.assertEqual(len(metadata), 102)
            for row in metadata:
                self.assertTrue(concept_parts(row['concept_name'], row['scope'], row['group']))
        physical = [joint for part, joints in BODY_PART_JOINTS.items() if part != 'full_body' for joint in joints]
        self.assertEqual(sorted(physical), list(range(25)))

    def test_region_and_lower_limb_corrections(self):
        for name in ('hand_lift_to_face', 'hand_contact_nose_region', 'upper_limb_hand_to_head_region'):
            self.assertIn('head', concept_parts(name, 'spatial', 'upper_limb'))
        self.assertEqual(set(concept_parts('lower_limb_kick_targeted', 'spatial', 'lower_limb')),
                         {'left_leg', 'right_leg', 'left_foot', 'right_foot'})
        with self.assertRaisesRegex(ValueError, 'no anatomical mapping'):
            concept_parts('unknown_spatial_concept', 'spatial', 'upper_limb')

    def test_invalid_tables_fail_before_tensor_construction(self):
        with CONCEPT_CSV.open(newline='', encoding='utf-8') as handle:
            original = list(csv.reader(handle))
        with METADATA_CSV.open(newline='', encoding='utf-8') as handle:
            original_metadata = list(csv.reader(handle))
        cases = []
        rows = [row[:] for row in original]
        rows[1], rows[2] = rows[2], rows[1]
        cases.append((rows, original_metadata, 'NTU label'))
        rows = [row[:] for row in original]
        rows[120][1] = 'nan'  # Validate unused rows even for an NTU60 run.
        cases.append((rows, original_metadata, 'binary'))
        rows = [row[:] for row in original]
        rows[1].pop()
        cases.append((rows, original_metadata, 'column count'))
        rows = [row[:] for row in original]
        rows[0][2] = rows[0][1]
        cases.append((rows, original_metadata, 'duplicate'))
        metadata = [row[:] for row in original_metadata]
        metadata[2][0] = metadata[1][0]
        cases.append((original, metadata, 'contiguous'))
        metadata = [row[:] for row in original_metadata]
        metadata[1][1], metadata[2][1] = metadata[2][1], metadata[1][1]
        cases.append((original, metadata, 'order'))
        metadata = [row[:] for row in original_metadata]
        metadata[1].append('extra')
        cases.append((original, metadata, 'column count'))
        metadata = [row[:] for row in original_metadata]
        metadata[1][3] = ''
        cases.append((original, metadata, 'nonempty'))
        with tempfile.TemporaryDirectory() as directory:
            concept, meta = Path(directory) / 'concept.csv', Path(directory) / 'metadata.csv'
            for rows, metadata, message in cases:
                with self.subTest(message=message):
                    with concept.open('w', newline='', encoding='utf-8') as handle:
                        csv.writer(handle).writerows(rows)
                    with meta.open('w', newline='', encoding='utf-8') as handle:
                        csv.writer(handle).writerows(metadata)
                    with self.assertRaisesRegex(ValueError, message):
                        load_concept_tables(concept, meta, 60, 102)


class ConceptGraphLossTests(unittest.TestCase):
    def test_cosine_invariance_and_isolated_concept(self):
        support = torch.tensor([[1., 1., 0.], [1., 0., 0.], [0., 1., 0.]])
        affinity = _concept_affinity(support)
        torch.testing.assert_close(affinity.sum(1), torch.ones(3))
        torch.testing.assert_close(affinity[2], torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(affinity, _concept_affinity(support.repeat(3, 1)))
        torch.testing.assert_close(affinity, torch.tensor([[2/3, 1/3, 0.], [1/3, 2/3, 0.], [0., 0., 1.]]))

    def test_balance_uses_train_counts_caps_and_can_be_disabled(self):
        model = bridge(balance_loss=True, pos_weight_source='train', pos_weight_max=4.)
        logits = torch.zeros(2, 102, requires_grad=True)
        labels = torch.tensor([0, 119])
        with self.assertRaisesRegex(RuntimeError, 'training class counts'):
            model.loss(logits, labels)
        counts = torch.ones(120)
        counts[0] = 500
        model.set_training_class_counts(counts)
        positives = counts @ model.support
        expected_weights = ((counts.sum() - positives) / positives.clamp_min(1)).clamp(1, 4)
        torch.testing.assert_close(model.positive_weight, expected_weights)
        expected = F.binary_cross_entropy_with_logits(logits, model.support[labels], pos_weight=expected_weights)
        torch.testing.assert_close(model.loss(logits, labels), expected)
        expected.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        model.balance_loss = False
        torch.testing.assert_close(model.loss(logits, labels), F.binary_cross_entropy_with_logits(logits, model.support[labels]))
        audit = model.audit()
        self.assertTrue(all(row['pos_weight'] == 1. for row in audit['concepts']))
        self.assertEqual(audit['concepts'][0]['positive_training_samples'], int(positives[0]))
        for invalid in (torch.zeros(120), -torch.ones(120), torch.ones(119), torch.full((120,), float('nan'))):
            with self.assertRaises(ValueError):
                model.set_training_class_counts(invalid)

    def test_training_counts_survive_checkpoint_roundtrip(self):
        model = bridge(balance_loss=True, pos_weight_source='train', pos_weight_max=5)
        model.set_training_class_counts(torch.arange(1, 121))
        restored = bridge(balance_loss=True, pos_weight_source='train', pos_weight_max=5)
        restored.load_state_dict(model.state_dict())
        self.assertTrue(restored._positive_weight_ready)
        torch.testing.assert_close(restored.positive_weight, model.positive_weight)
        self.assertNotIn('concept_affinity', model.state_dict())
        json.dumps(restored.audit(), allow_nan=False)

    def test_zero_positive_and_all_positive_concepts_are_finite(self):
        model = bridge(balance_loss=True, pos_weight_source='train')
        counts = torch.zeros(120)
        counts[0] = 8
        model.set_training_class_counts(counts)
        torch.testing.assert_close(model.positive_weight, torch.ones(102))
        self.assertTrue(torch.isfinite(model.loss(torch.zeros(1, 102), torch.tensor([0]))))


class SymbolicMetricTests(unittest.TestCase):
    def test_perfect_scores_and_no_positive_concept_policy(self):
        targets = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        logits = np.where(targets, 2., -2.)
        metadata = [{'concept_name': str(i), 'scope': ('spatial', 'temporal', 'context')[i]} for i in range(3)]
        result = concept_metrics(logits, targets, metadata)
        self.assertEqual(result['macro_f1'], 1.)
        self.assertEqual(result['mAP'], 1.)
        self.assertEqual(result['concepts_with_positives'], 2)
        self.assertAlmostEqual(result['macro_f1_all'], 2/3)
        self.assertIsNone(result['per_concept'][2]['AP'])
        self.assertIsNone(result['scopes']['context']['mAP'])
        json.dumps(result, allow_nan=False)

    def test_false_positives_and_threshold_are_reported(self):
        metadata = [{'concept_name': 'a', 'scope': 'spatial'}, {'concept_name': 'b', 'scope': 'context'}]
        logits = np.ones((2, 2))
        targets = np.array([[1, 0], [0, 0]])
        result = concept_metrics(logits, targets, metadata)
        self.assertEqual(result['per_concept'][1]['predicted_positives'], 2)
        self.assertAlmostEqual(result['micro_f1'], 2/5)
        self.assertEqual(concept_metrics(logits, targets, metadata, threshold=.9)['macro_f1'], 0.)


class TinyModel(nn.Module):
    def __init__(self, symbolic):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.symbolic_graph = bridge() if symbolic else None


class PretrainedTests(unittest.TestCase):
    def test_official_state_requires_explicit_symbolic_initialization(self):
        baseline = TinyModel(False)
        symbolic = TinyModel(True)
        checkpoint = {'state_dict': {'module.' + key: value for key, value in baseline.state_dict().items()}}
        with self.assertRaisesRegex(ValueError, 'Missing'):
            load_model_weights(symbolic, checkpoint)
        report = load_model_weights(symbolic, checkpoint, allow_new_symbolic=True)
        torch.testing.assert_close(symbolic.fc.weight, baseline.fc.weight)
        self.assertTrue(all(key.startswith('symbolic_graph.') for key in report['initialized']))

    def test_missing_backbone_and_wrong_shapes_fail_even_in_finetune(self):
        symbolic = TinyModel(True)
        with self.assertRaisesRegex(ValueError, 'fc.weight'):
            load_model_weights(symbolic, {'fc.bias': torch.zeros(2)}, allow_new_symbolic=True)
        checkpoint = TinyModel(False).state_dict()
        checkpoint['fc.weight'] = torch.zeros(3, 4)
        with self.assertRaisesRegex(ValueError, 'shapes'):
            load_model_weights(symbolic, checkpoint, allow_new_symbolic=True)

    def test_legacy_csv_buffers_cannot_override_new_graph(self):
        symbolic = TinyModel(True)
        checkpoint = symbolic.state_dict()
        checkpoint['symbolic_graph.concept_affinity'] = torch.zeros(102, 102)
        report = load_model_weights(symbolic, checkpoint)
        self.assertIn('symbolic_graph.concept_affinity', report['rebuilt'])
        torch.testing.assert_close(symbolic.symbolic_graph.concept_affinity.sum(1), torch.ones(102))


class SymbolicConfigTests(unittest.TestCase):
    def test_all_configs_match_baseline_except_declared_finetune_options(self):
        configs = list((ROOT / 'config/symbolic').rglob('*.yaml'))
        configs.append(ROOT / 'config/base/nturgbd120-cross-subject/hyper_joint_symbolic.yaml')
        self.assertEqual(len(configs), 17)
        for path in configs:
            with self.subTest(config=path):
                relative = path.relative_to(ROOT).as_posix()
                baseline_path = ROOT / (relative.replace('config/symbolic/', 'config/base/')
                                        if '/symbolic/' in relative else 'config/base/nturgbd120-cross-subject/hyper_joint.yaml')
                config = yaml.safe_load(path.read_text())
                baseline = yaml.safe_load(baseline_path.read_text())
                self.assertEqual(config['train_feeder_args'], baseline['train_feeder_args'])
                self.assertEqual(config['test_feeder_args'], baseline['test_feeder_args'])
                self.assertEqual({k: v for k, v in config['model_args'].items()
                                  if not k.startswith('symbolic_') and k not in ('concept_csv', 'metadata_csv', 'num_concepts')},
                                 baseline['model_args'])
                for key in ('model', 'weight_decay', 'lr_decay_rate',
                            'batch_size', 'test_batch_size', 'nesterov'):
                    self.assertEqual(config[key], baseline[key])
                self.assertTrue(config['pretrained_finetune'])
                self.assertIsNone(config['weights'])
                self.assertEqual(config['optimizer'], 'SGD')
                if config.get('symbolic_only'):
                    self.assertTrue(config['eval_before_train'])
                    self.assertEqual(config['base_lr'], 1e-3)
                    self.assertEqual(config['num_epoch'], 20)
                    self.assertEqual(config['step'], [12, 17])
                    self.assertEqual(config['warm_up_epoch'], 3)
                    self.assertTrue(config['work_dir'].endswith('joint_symbolic_only'))
                else:
                    self.assertEqual(config['base_lr'], 1e-4)
                    for key in ('step', 'warm_up_epoch', 'num_epoch'):
                        self.assertEqual(config[key], baseline[key])
                self.assertTrue(config['model_args']['symbolic_balance_loss'])
                self.assertEqual(config['model_args']['symbolic_pos_weight_source'], 'train')


class SymbolicEvaluationTests(unittest.TestCase):
    def test_processor_exports_all_samples_with_short_last_batch(self):
        # Execute the actual Processor methods without importing the optional
        # TensorBoard writer or running its CUDA/dataset initialization.
        source = ast.parse((ROOT / 'main.py').read_text())
        definitions = ast.Module(body=[node for node in source.body
                                      if isinstance(node, ast.ClassDef) and node.name == 'Processor'],
                                 type_ignores=[])
        namespace = {'torch': torch, 'np': np, 'json': json, 'os': os, 'OrderedDict': OrderedDict,
                     'concept_metrics': concept_metrics, 'tqdm': lambda loader, **kwargs: loader}
        exec(compile(definitions, str(ROOT / 'main.py'), 'exec'), namespace)
        processor = namespace['Processor'].__new__(namespace['Processor'])
        symbolic = bridge(balance_loss=True, pos_weight_source='train')
        counts = torch.ones(120)
        counts[0] = 100
        symbolic.set_training_class_counts(counts)
        labels = torch.tensor([0, 1, 119])
        concepts = symbolic.support[labels] * 2 - 1
        actions = torch.full((3, 120), -10.)
        actions[torch.arange(3), labels] = 10.

        class Outputs(nn.Module):
            def __init__(self):
                super().__init__()
                self.symbolic_graph = symbolic
                self.symbolic_graph_weight = .1

            def forward(self, indices):
                indices = indices.long()
                return actions[indices], [], concepts[indices]

        class Dataset:
            sample_name = ['a', 'b', 'c']
            label = labels.numpy()

            def top_k(self, scores, k):
                return float(np.mean([label in row[-k:] for label, row in zip(self.label, scores.argsort())]))

        class Loader:
            dataset = Dataset()

            def __iter__(self):
                yield torch.tensor([0, 1]), labels[:2], torch.tensor([0, 1])
                yield torch.tensor([2]), labels[2:], torch.tensor([2])

            def __len__(self):
                return 2

        class Writer:
            def __init__(self):
                self.values = {}

            def add_scalar(self, tag, value, step):
                self.values[tag] = value

        processor.model = Outputs()
        processor.loss = nn.CrossEntropyLoss()
        processor.data_loader = {'test': Loader()}
        processor.output_device = 0
        processor.global_step = 7
        processor.best_acc = 0
        processor.best_acc_epoch = 0
        processor.val_writer = Writer()
        processor.print_log = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory() as directory:
            processor.arg = argparse.Namespace(work_dir=directory, phase='train',
                                               feeder='feeders.feeder_ntu.Feeder',
                                               model_saved_name=os.path.join(directory, 'runs'),
                                               show_topk=[1], concept_threshold=.5)
            with mock.patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor):
                processor.eval(epoch=3)
            metrics = json.loads((Path(directory) / 'epoch4_test_concept_metrics.json').read_text())
        self.assertEqual(metrics['num_samples'], 3)
        self.assertEqual(metrics['macro_f1'], 1.)
        self.assertAlmostEqual(metrics['loss'], symbolic.loss(concepts, labels).item(), places=6)
        self.assertAlmostEqual(metrics['weighted_loss'], .1 * metrics['loss'])
        for tag in ('symbolic/mAP', 'symbolic/macro_f1', 'symbolic/spatial/mAP',
                    'symbolic/residual_gate', 'symbolic/concept_graph_gate'):
            self.assertIn(tag, processor.val_writer.values)


if __name__ == '__main__':
    unittest.main()
