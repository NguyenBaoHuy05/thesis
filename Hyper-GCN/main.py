#!/usr/bin/env python
from __future__ import print_function

import argparse
import csv
import json
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback

import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.checkpoint import load_model_weights
from model.symbolic_metrics import concept_metrics
from model.training import configure_trainable_parameters, set_training_mode

# from torchlight import DictAction



def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


class DivergenceLoss(nn.Module):
    def __init__(self):
        super(DivergenceLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        V, C = x[0].size()
        loss = 0

        for i in x:
            norm = torch.norm(i, dim=-1, keepdim=True, p=2)
            norm = norm @ norm.T
            loss_i = i @ i.T
            loss_i = loss_i / (norm + 1e-8)
            loss_p = self.relu(loss_i)
            loss_p = (loss_p.sum() - V) / (V * (V - 1))
            loss += loss_p

        return loss / len(x)


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=55,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-epoch',
        type=int,
        default=55,
        help='the epoch for evaluating models (#iteration)'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--pretrained-finetune', type=str2bool, default=False,
        help='require --weights and allow newly initialized symbolic parameters during training')
    parser.add_argument(
        '--symbolic-only', type=str2bool, default=False,
        help='train only symbolic_graph; freeze pretrained parameters and BatchNorm statistics')
    parser.add_argument(
        '--eval-before-train', type=str2bool, default=False,
        help='evaluate and retain the initial model before training (automatic for symbolic-only)')
    parser.add_argument(
        '--concept-threshold', type=float, default=0.5,
        help='fixed sigmoid threshold for concept F1; do not tune this on the test set')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        if arg.phase == 'train' and (arg.pretrained_finetune or arg.symbolic_only) and not arg.weights:
            raise ValueError('Pretrained fine-tuning requires --weights with a matching Hyper-GCN checkpoint.')
        if not 0 < arg.concept_threshold < 1:
            raise ValueError('concept_threshold must be between 0 and 1')
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()
        if self.arg.phase == 'train':
            counts = configure_trainable_parameters(self.model, self.arg.symbolic_only)
            self.print_log('Parameters: {}'.format(counts))
            if self.arg.symbolic_only:
                action_mode = getattr(self.model, 'symbolic_action_mode', 'blackbox')
                self.print_log('Symbolic-only: backbone/classifier frozen, pretrained BN in eval; '
                               'action_mode={}; loss = CE + symbolic_graph_weight * concept BCE '
                               '(no divergence).'.format(action_mode))

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
            self.configure_symbolic()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.best_weights_path = None
        self.baseline_accuracy = {}

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        print(self.model)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(output_device)
        self.h_loss = DivergenceLoss().cuda(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, map_location='cpu')
            report = load_model_weights(
                self.model, weights,
                allow_new_symbolic=self.arg.phase == 'train' and (
                    self.arg.pretrained_finetune or self.arg.symbolic_only),
                ignore_weights=self.arg.ignore_weights,
            )
            if self.arg.phase == 'train' and self.arg.symbolic_only:
                missing_frozen = [key for key in report['initialized']
                                  if not key.startswith('symbolic_graph.')]
                if missing_frozen:
                    raise ValueError('symbolic_only cannot freeze uninitialized pretrained keys: {}'.format(missing_frozen))
            self.print_log('Checkpoint load: {}'.format(report))

    def configure_symbolic(self):
        bridge = getattr(self.model, 'symbolic_graph', None)
        if bridge is None:
            return
        if getattr(bridge, 'structured_mode', False):
            audit = bridge.audit()
            with open(os.path.join(self.arg.work_dir, 'symbolic_audit.json'), 'w', encoding='utf-8') as handle:
                json.dump(audit, handle, indent=2, allow_nan=False)
            self.print_log(
                'Symbolic structured mode: {} primitives, {} relations, graph={}, grounding={}'.format(
                    audit['num_primitives'], audit['num_relations'],
                    bridge.hypergraph_mode, audit['grounding_mode'],
                )
            )
            return
        if self.arg.phase == 'train':
            labels = np.asarray(self.data_loader['train'].dataset.label)
            num_classes = bridge.support.shape[0]
            if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
                    or labels.size == 0 or labels.min() < 0 or labels.max() >= num_classes):
                raise ValueError('training labels must be integer NTU indices matching num_class')
            bridge.set_training_class_counts(np.bincount(labels, minlength=num_classes))
        elif bridge.balance_loss and not bridge._positive_weight_ready:
            raise ValueError('Balanced test loss needs a symbolic checkpoint with training_class_counts; '
                             'for legacy weights set symbolic_balance_loss=False explicitly.')
        audit = bridge.audit()
        with open(os.path.join(self.arg.work_dir, 'symbolic_audit.json'), 'w', encoding='utf-8') as handle:
            json.dump(audit, handle, indent=2, allow_nan=False)
        audit_csv = os.path.join(self.arg.work_dir, 'concept_body_joint_audit.csv')
        with open(audit_csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                'concept_id', 'concept_name', 'scope', 'parts', 'joints',
                'interaction', 'positive_classes', 'positive_training_samples',
            ))
            writer.writeheader()
            for concept_id, concept in enumerate(audit['concepts']):
                writer.writerow({
                    'concept_id': concept_id,
                    'concept_name': concept['name'],
                    'scope': concept['scope'],
                    'parts': '|'.join(concept['parts']),
                    'joints': '|'.join(str(joint) for joint in concept['joints']),
                    'interaction': concept['interaction'],
                    'positive_classes': concept['positive_classes'],
                    'positive_training_samples': concept['positive_training_samples'],
                })
        self.print_log(
            'Symbolic: {} concepts, graph={}, grounding={}, decoder={}, '
            'balance={}, source={}, cap={}'.format(
                bridge.num_concepts, bridge.hypergraph_mode,
                audit['grounding_mode'], bridge.action_matching,
                bridge.balance_loss, bridge.pos_weight_source,
                bridge.pos_weight_max,
            )
        )

    def log_symbolic_gates(self, writer=None):
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        bridge = getattr(model_ref, 'symbolic_graph', None)
        if bridge is None:
            return
        if getattr(bridge, 'structured_mode', False):
            self.print_log('\tSymbolic gates: structured event graph')
            return
        values = {}
        if bridge.hypergraph_mode == 'learnable_v1':
            values['hypergraph_gate'] = torch.tanh(bridge.hypergraph_gate).detach().item()
        else:
            values['concept_graph_gate'] = torch.tanh(bridge.concept_graph_gate).detach().item()
        self.print_log('\tSymbolic gates: {}'.format(values))
        if writer is not None:
            for name, value in values.items():
                writer.add_scalar('symbolic/' + name, value, self.global_step)

    def load_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError('No trainable parameters for optimizer')
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        symbolic_only = getattr(self.arg, 'symbolic_only', False)
        set_training_mode(self.model, symbolic_only)
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        component_sums = dict(ce=0.0, divergence=0.0, symbolic=0.0,
                              symbolic_weighted=0.0, hypergraph=0.0,
                              hypergraph_weighted=0.0)
        sample_count = 0
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            h_loss = data.new_zeros(())
            symbolic_loss = data.new_zeros(())
            symbolic_weighted = data.new_zeros(())
            hypergraph_loss = data.new_zeros(())
            hypergraph_weighted = data.new_zeros(())
            if isinstance(output, tuple):
                ce_loss = self.loss(output[0], label)
                if not symbolic_only:
                    h_loss = self.h_loss(output[1])
                loss = ce_loss + h_loss
                if len(output) > 2 and output[2] is not None:
                    model_ref = (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    )
                    symbolic_loss = model_ref.symbolic_graph.loss(output[2], label)
                    symbolic_weighted = model_ref.symbolic_graph_weight * symbolic_loss
                    loss = loss + symbolic_weighted
                    hypergraph_loss = model_ref.symbolic_graph.structure_loss()
                    hypergraph_weighted = (
                        getattr(model_ref, 'symbolic_hypergraph_reg_weight', 0.0)
                        * hypergraph_loss
                    )
                    loss = loss + hypergraph_weighted
                output = output[0]
            else:
                ce_loss = self.loss(output, label)
                loss = ce_loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            components = dict(ce=ce_loss, divergence=h_loss, symbolic=symbolic_loss,
                              symbolic_weighted=symbolic_weighted,
                              hypergraph=hypergraph_loss,
                              hypergraph_weighted=hypergraph_weighted)
            for name, component in components.items():
                value = component.detach().item()
                self.train_writer.add_scalar('loss/' + name, value, self.global_step)
                component_sums[name] += value * label.size(0)
            sample_count += label.size(0)


            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}. Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        self.print_log('\tMean loss components: {}'.format({
            name: total / max(sample_count, 1) for name, total in component_sums.items()
        }))
        self.log_symbolic_gates(self.train_writer)

        if save_model:
            self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch_number):
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        weights = OrderedDict((key, value.detach().cpu()) for key, value in model_ref.state_dict().items())
        path = '{}-{}-{}.pt'.format(self.arg.model_saved_name, epoch_number, int(self.global_step))
        torch.save(weights, path)
        return path

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        bridge = getattr(model_ref, 'symbolic_graph', None)
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        accuracies = {}
        for ln in loader_name:
            if bridge is not None and bridge.action_decoder == 'prototype':
                bridge.reset_grounding_statistics()
            loss_value = []
            score_frag = []
            concept_frag = []
            concept_labels = []
            symbolic_loss_sum = 0.0
            concept_sample_count = 0
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    if isinstance(output, tuple):
                        loss = self.loss(output[0], label)
                        if len(output) > 2 and output[2] is not None:
                            symbolic_loss_sum += bridge.loss(output[2], label).item() * label.size(0)
                            concept_sample_count += label.size(0)
                            concept_frag.append(output[2].detach().cpu().numpy())
                            concept_labels.append(label.cpu().numpy())
                        output = output[0]
                    else:
                        loss = self.loss(output, label)
                    score_frag.append(output.cpu())
                    loss_value.append(loss.data.item())
                    _, predict_label = torch.max(output.data, 1)
                    step += 1
                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate([i.numpy() for i in score_frag])
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            accuracies[ln] = accuracy
            initial_candidate = epoch == -1 and getattr(self, 'best_weights_path', None) is None
            is_best = accuracy > self.best_acc or initial_candidate
            if is_best:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                if self.arg.phase == 'train':
                    # Also save improvements between periodic saves, and retain
                    # epoch 0 if fine-tuning never beats the initial checkpoint.
                    self.best_weights_path = self.save_checkpoint(epoch + 1)

            baseline = getattr(self, 'baseline_accuracy', {}).get(ln)
            delta_pp = None if baseline is None else 100 * (accuracy - baseline)
            action_metrics = dict(epoch=epoch + 1, top1=accuracy, initial_top1=baseline,
                                  delta_from_initial_pp=delta_pp)
            with open(os.path.join(self.arg.work_dir, 'epoch{}_{}_action_metrics.json'.format(epoch + 1, ln)),
                      'w', encoding='utf-8') as handle:
                json.dump(action_metrics, handle, indent=2, allow_nan=False)
            if delta_pp is not None:
                self.print_log('\tTop1 vs initial model: {:+.3f} percentage points.'.format(delta_pp))
                if self.arg.phase == 'train':
                    self.val_writer.add_scalar('acc/delta_from_initial_pp', delta_pp, self.global_step)

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if concept_frag and not getattr(bridge, 'structured_mode', False):
                logits = np.concatenate(concept_frag)
                labels = np.concatenate(concept_labels)
                targets = bridge.support.detach().cpu().numpy()[labels]
                metrics = concept_metrics(logits, targets, bridge.metadata, self.arg.concept_threshold)
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                concept_std = probabilities.std(axis=0)
                dead_indices = np.flatnonzero(concept_std < 1e-4)
                activation_frequency = (
                    probabilities >= self.arg.concept_threshold
                ).mean(axis=0)
                centered = probabilities - probabilities.mean(axis=0, keepdims=True)
                scale = np.sqrt((centered * centered).sum(axis=0))
                normalized = centered / np.maximum(scale, 1e-12)
                correlation = normalized.T @ normalized
                np.fill_diagonal(correlation, 0.)
                pairs = np.argwhere(np.triu(correlation > .95, k=1))
                ranked_pairs = sorted(
                    (
                        (float(correlation[left, right]), int(left), int(right))
                        for left, right in pairs
                    ), reverse=True,
                )[:20]
                metrics['activation_summary'] = {
                    'mean_probability': float(probabilities.mean()),
                    'mean_activation_frequency': float(activation_frequency.mean()),
                    'dead_concept_count': int(dead_indices.size),
                    'dead_concepts': [bridge.concept_names[index] for index in dead_indices],
                    'high_correlation_pair_count': int(pairs.shape[0]),
                    'top_correlated_pairs': [
                        {
                            'left': bridge.concept_names[left],
                            'right': bridge.concept_names[right],
                            'correlation': value,
                        }
                        for value, left, right in ranked_pairs
                    ],
                }
                metrics['loss'] = symbolic_loss_sum / concept_sample_count
                metrics['weighted_loss'] = model_ref.symbolic_graph_weight * metrics['loss']
                metrics['action_top1'] = accuracy
                metrics['delta_from_initial_pp'] = delta_pp
                if bridge.hypergraph_mode == 'learnable_v1':
                    hypergraph = bridge.hypergraph_snapshot()
                    metrics['hypergraph_structure_loss'] = hypergraph['structure_loss']
                    hypergraph_path = os.path.join(
                        self.arg.work_dir,
                        'epoch{}_{}_hypergraph.json'.format(epoch + 1, ln),
                    )
                    with open(hypergraph_path, 'w', encoding='utf-8') as handle:
                        json.dump(hypergraph, handle, indent=2, allow_nan=False)
                    if is_best:
                        with open(os.path.join(self.arg.work_dir, 'best_hypergraph.json'),
                                  'w', encoding='utf-8') as handle:
                            json.dump(hypergraph, handle, indent=2, allow_nan=False)
                    self.print_log(
                        '\tHypergraph: structure={:.6f}, prior={:.4f}, nonprior={:.4f}, changed={}'.format(
                            hypergraph['structure_loss'], hypergraph['mean_prior_membership'],
                            hypergraph['mean_nonprior_membership'],
                            hypergraph['changed_memberships_at_0.5'],
                        )
                    )
                    if self.arg.phase == 'train':
                        for name in ('structure_loss', 'mean_prior_membership',
                                     'mean_nonprior_membership', 'changed_memberships_at_0.5'):
                            self.val_writer.add_scalar('hypergraph/' + name, hypergraph[name], self.global_step)
                if bridge.action_decoder == 'prototype':
                    grounding = bridge.grounding_snapshot()
                    metrics['grounding'] = {
                        key: value for key, value in grounding.items()
                        if key != 'concepts'
                    }
                    grounding_path = os.path.join(
                        self.arg.work_dir,
                        'epoch{}_{}_grounding.json'.format(epoch + 1, ln),
                    )
                    with open(grounding_path, 'w', encoding='utf-8') as handle:
                        json.dump(grounding, handle, indent=2, allow_nan=False)
                    if is_best:
                        with open(os.path.join(self.arg.work_dir, 'best_grounding.json'),
                                  'w', encoding='utf-8') as handle:
                            json.dump(grounding, handle, indent=2, allow_nan=False)
                    self.print_log(
                        '\tGrounding: temporal_entropy={:.4f}, '
                        'person_entropy={:.4f}, token_norm={:.4f}'.format(
                            grounding['mean_temporal_entropy'],
                            grounding['mean_person_entropy'],
                            grounding['mean_token_norm'],
                        )
                    )
                    if self.arg.phase == 'train':
                        for name in ('mean_temporal_entropy', 'mean_person_entropy',
                                     'mean_token_norm'):
                            self.val_writer.add_scalar(
                                'grounding/' + name, grounding[name], self.global_step
                            )
                metrics_path = os.path.join(self.arg.work_dir, 'epoch{}_{}_concept_metrics.json'.format(epoch + 1, ln))
                with open(metrics_path, 'w', encoding='utf-8') as handle:
                    json.dump(metrics, handle, indent=2, allow_nan=False)
                self.print_log('\tSymbolic: loss={:.4f}, macro-F1={}, mAP={}, active={}/{}'.format(
                    metrics['loss'], metrics['macro_f1'], metrics['mAP'],
                    metrics['concepts_with_positives'], metrics['num_concepts']))
                if self.arg.phase == 'train':
                    for name in ('macro_f1', 'macro_f1_all', 'micro_f1', 'mAP', 'loss', 'weighted_loss'):
                        if metrics[name] is not None:
                            self.val_writer.add_scalar('symbolic/' + name, metrics[name], self.global_step)
                    for scope, summary in metrics['scopes'].items():
                        for name in ('macro_f1', 'mAP'):
                            if summary[name] is not None:
                                self.val_writer.add_scalar('symbolic/{}/{}'.format(scope, name),
                                                           summary[name], self.global_step)
                    self.val_writer.add_scalar(
                        'symbolic/dead_concept_count',
                        metrics['activation_summary']['dead_concept_count'],
                        self.global_step,
                    )
                    self.val_writer.add_scalar(
                        'symbolic/high_correlation_pair_count',
                        metrics['activation_summary']['high_correlation_pair_count'],
                        self.global_step,
                    )
                self.log_symbolic_gates(self.val_writer if self.arg.phase == 'train' else None)

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            # label_list = np.concatenate(label_list)
            # pred_list = np.concatenate(pred_list)
            # confusion = confusion_matrix(label_list, pred_list)
            # list_diag = np.diag(confusion)
            # list_raw_sum = np.sum(confusion, axis=1)
            # each_acc = list_diag / list_raw_sum
            # with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(each_acc)
            #     writer.writerows(confusion)


        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()
        return accuracies

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            if self.arg.symbolic_only or self.arg.eval_before_train:
                self.print_log('Initial evaluation before any optimizer step (epoch 0).')
                self.baseline_accuracy = self.eval(epoch=-1, save_score=self.arg.save_score)
                with open(os.path.join(self.arg.work_dir, 'initial_evaluation.json'), 'w', encoding='utf-8') as handle:
                    json.dump(dict(weights=self.arg.weights, top1=self.baseline_accuracy,
                                   checkpoint=self.best_weights_path), handle, indent=2, allow_nan=False)
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                # if epoch + 1 >= self.arg.eval_epoch:
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = self.best_weights_path
            if weights_path is None:
                raise RuntimeError('No evaluated checkpoint available for best-model selection.')
            weights = torch.load(weights_path, map_location='cpu')
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=self.best_acc_epoch - 1, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
