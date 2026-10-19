#!/usr/bin/env python
from __future__ import print_function

import argparse
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
from model.training import configure_trainable_parameters, set_training_mode

# from torchlight import DictAction



def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # TF32 matmuls on Ampere/Ada GPUs (A100, L4): faster, negligible accuracy change.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        '--freeze-until', default=None,
        help='freeze the backbone up to this layer (stem, l1..l9) with BatchNorm in eval mode; '
             'requires --weights. Extension modules always train.')
    parser.add_argument(
        '--eval-before-train', type=str2bool, default=False,
        help='evaluate and retain the initial model before training')
    parser.add_argument(
        '--select-best', type=str2bool, default=True,
        help='keep the epoch with the best test Top1 (the original behavior). False: report the '
             'last epoch, so no choice is made on the test split')
    parser.add_argument(
        '--lr-scale', action=DictAction, default=dict(),
        help='learning-rate multiplier per parameter-name prefix, e.g. {head.decoder.: 10} for a '
             'module trained from scratch next to a fine-tuned backbone')
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
        if arg.phase == 'train' and arg.freeze_until not in (None, 'none') and not arg.weights:
            raise ValueError('freeze_until requires --weights with a matching Hyper-GCN checkpoint.')
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name) and not sys.stdin.isatty():
                    # Non-interactive runs (Kaggle, nohup) keep the old TensorBoard logs.
                    print('log_dir: ', arg.model_saved_name, 'already exist; keeping it')
                elif os.path.isdir(arg.model_saved_name):
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
            counts = configure_trainable_parameters(self.model, self.arg.freeze_until)
            self.print_log('Parameters: {}'.format(counts))

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
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
                drop_last=True,
                pin_memory=True,
                persistent_workers=self.arg.num_worker > 0)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.arg.num_worker > 0)

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
                allow_new=self.arg.phase == 'train',
                ignore_weights=self.arg.ignore_weights,
            )
            self.print_log('Checkpoint load: {}'.format(report))

    def load_optimizer(self):
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not named:
            raise ValueError('No trainable parameters for optimizer')
        # One group per --lr-scale prefix (first match wins); adjust_learning_rate applies lr_scale.
        groups = {}
        for name, parameter in named:
            name = name[len('module.'):] if name.startswith('module.') else name
            scale = next((float(v) for k, v in self.arg.lr_scale.items() if name.startswith(k)), 1.0)
            groups.setdefault(scale, []).append(parameter)
        parameters = [dict(params=params, lr=self.arg.base_lr * scale, lr_scale=scale)
                      for scale, params in groups.items()]
        if self.arg.lr_scale:
            self.print_log('lr_scale groups: {}'.format(
                {scale: sum(p.numel() for p in params) for scale, params in groups.items()}))
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
                param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
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
        set_training_mode(self.model, self.arg.freeze_until)
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        component_sums = dict(ce=0.0, divergence=0.0)
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
            if isinstance(output, tuple):
                ce_loss = self.loss(output[0], label)
                h_loss = self.h_loss(output[1])
                loss = ce_loss + h_loss
                output = output[0]
            else:
                ce_loss = self.loss(output, label)
                h_loss = data.new_zeros(())
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
            components = dict(ce=ce_loss, divergence=h_loss)
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
        mean_components = {
            name: total / max(sample_count, 1) for name, total in component_sums.items()
        }
        self.print_log('\tMean loss components: {}'.format(
            ', '.join('{}={:.4f}'.format(name, value) for name, value in mean_components.items())))
        self.last_train_summary = dict(loss=float(np.mean(loss_value)), acc=float(np.mean(acc_value)),
                                       lr=float(self.lr), **mean_components)

        if save_model:
            self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch_number):
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        weights = OrderedDict((key, value.detach().cpu()) for key, value in model_ref.state_dict().items())
        path = '{}-{}-{}.pt'.format(self.arg.model_saved_name, epoch_number, int(self.global_step))
        torch.save(weights, path)
        return path

    def append_history(self, epoch, ln, test_loss, accuracy, selection_score, is_best):
        """One JSON line per evaluated epoch: train losses, test metrics, best flag."""
        record = dict(epoch=epoch + 1, split=ln, test_loss=test_loss, test_top1=accuracy,
                      selection_score=selection_score, is_best=bool(is_best),
                      checkpoint=self.best_weights_path if is_best else None)
        if epoch >= 0 and getattr(self, 'last_train_summary', None) is not None:
            record['train'] = self.last_train_summary
        with open(os.path.join(self.arg.work_dir, 'history.jsonl'), 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None,
             record_history=True):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        accuracies = {}
        for ln in loader_name:
            loss_value = []
            score_frag = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    if isinstance(output, tuple):
                        loss = self.loss(output[0], label)
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
            selection_score = accuracy
            initial_candidate = epoch == -1 and getattr(self, 'best_weights_path', None) is None
            if self.arg.select_best:
                is_best = selection_score > self.best_acc or initial_candidate
            else:
                # Report the last epoch: no choice is made on the test split.
                is_best = epoch + 1 == self.arg.num_epoch
            if is_best:
                self.best_acc = selection_score
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
            if record_history:
                self.append_history(epoch, ln, float(loss), accuracy, selection_score, is_best)

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
            if self.arg.eval_before_train:
                self.print_log('Initial evaluation before any optimizer step (epoch 0).')
                self.baseline_accuracy = self.eval(epoch=-1, save_score=self.arg.save_score)
                with open(os.path.join(self.arg.work_dir, 'initial_evaluation.json'), 'w', encoding='utf-8') as handle:
                    json.dump(dict(weights=self.arg.weights, top1=self.baseline_accuracy,
                                   checkpoint=self.best_weights_path), handle, indent=2, allow_nan=False)
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                # Sparse evaluation early, every epoch from eval_epoch on, always the last epoch.
                if ((epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 >= self.arg.eval_epoch
                        or epoch + 1 == self.arg.num_epoch):
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
            self.eval(epoch=self.best_acc_epoch - 1, save_score=True, loader_name=['test'],
                      wrong_file=wf, result_file=rf, record_history=False)
            self.arg.print_log = True
            with open(os.path.join(self.arg.work_dir, 'best_checkpoint.json'), 'w', encoding='utf-8') as handle:
                json.dump(dict(path=os.path.abspath(weights_path), epoch=self.best_acc_epoch,
                               score=self.best_acc, selection_metric='top1'),
                          handle, indent=2, allow_nan=False)


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
