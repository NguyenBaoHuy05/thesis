#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction, IO

try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
except (ImportError, Exception):
    pass


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    if not mod_str:
        return globals()[class_str] if class_str in globals() else getattr(sys.modules['__main__'], class_str)
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


def get_parser():
    parser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config_ntu60_cs.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if true, the classification score will be stored')

    # visualize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating model (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print log to output or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which top k accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder_ntu.Feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default='model_dcln.DCLN', help='module class will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--resume', type=str2bool, default=False, help='resume training (model, optimizer, scaler, epoch, best_acc) from work_dir/checkpoint.pt')

    # optimism
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--logic-lr', type=float, default=0.01,
                        help='Dedicated LR for ConceptLogicLayers when trained jointly with the '
                             'backbone (Stage 1 / plain end-to-end mode). Its W matrices are '
                             'clipped to [0,1], so sharing base_lr with the conv backbone makes '
                             'updates saturate at the clip boundary instead of learning smoothly.')
    parser.add_argument('--step', type=int, default=[20], nargs='+', help='the epoch where lr decay')
    parser.add_argument('--device', type=int, default=[0], nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    # 2-stage training
    parser.add_argument('--two-stage', type=str2bool, default=False,
                        help='Enable 2-stage training: Stage1=backbone+concept, Stage2=logic layers')
    parser.add_argument('--stage1-epoch', type=int, default=40,
                        help='Number of epochs for Stage 1 (backbone+concept training). '
                             'Stage 2 runs from stage1_epoch to num_epoch.')
    parser.add_argument('--stage1-step', type=int, default=[25, 35], nargs='+',
                        help='LR decay epochs within Stage 1')
    parser.add_argument('--stage2-backbone-lr', type=float, default=0.001,
                        help='Fine-tuning LR for backbone in Stage 2 (concept modules). '
                             'Set to 0 to fully freeze backbone in Stage 2.')
    parser.add_argument('--stage2-logic-lr', type=float, default=0.01,
                        help='Learning rate for ConceptLogicLayers in Stage 2')
    parser.add_argument('--stage2-step', type=int, default=[15, 22], nargs='+',
                        help='LR decay epochs WITHIN Stage 2 (relative to stage2 start)')

    return parser


class Processor():

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                else:
                    os.makedirs(arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.work_dir, 'test'), 'test')

        self.global_step = 0
        use_cuda = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.load_model()
        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_param_into_net()
            self.preload_checkpoint()
            self.load_optimizer()
            self.load_data()
            self.try_resume_checkpoint()

    def load_model(self):
        output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        self.output_device = output_device
        self.model = IO(self.arg.work_dir).load_model(self.arg.model, **self.arg.model_args)
        self.model = self.model.cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self.print_log(f'Load weights from {self.arg.weights}.')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'rb') as f:
                    weights = pickle.load(f)

                weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
                for i in self.arg.ignore_weights:
                    ignore_name = list()
                    for w in weights:
                        if w.find(i) == 0:
                            ignore_name.append(w)
                    for n in ignore_name:
                        weights.pop(n)
                        self.print_log(f'Filter [{i}] remove weights [{n}].')

                for w in weights:
                    self.print_log(f'Load weights [{w}].')

                try:
                    self.model.load_state_dict(weights)
                except (KeyError, RuntimeError):
                    state = self.model.state_dict()
                    diff = list(set(state.keys()).difference(set(weights.keys())))
                    for d in diff:
                        self.print_log(f'Can not find weights [{d}].')
                    state.update(weights)
                    self.model.load_state_dict(state)
            else:
                self.model = IO(self.arg.work_dir).load_weights(self.model, self.arg.weights, self.arg.ignore_weights)

    def load_param_into_net(self):
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.print_log(f'Using DataParallel across GPUs: {self.arg.device}')
            self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=init_seed)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_optimizer(self):
        """Load optimizer. Stage-aware for 2-stage training."""
        model = self.bare_model()
        if self.arg.two_stage and getattr(self, '_in_stage2', False):
            self.optimizer = self._build_stage2_optimizer()
            return

        # Stage 1 (or plain end-to-end when two_stage=False): train everything
        # jointly, but ConceptLogicLayers gets its own (much smaller) LR — its
        # W matrices are clipped to [0, 1], so base_lr (tuned for the conv
        # backbone) saturates them at the clip boundary every step instead of
        # letting them learn smoothly.
        logic_params = list(model.logic_layers.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith('logic_layers')]
        param_groups = [
            {'params': other_params, 'lr': self.arg.base_lr, 'name': 'backbone_concept'},
            {'params': logic_params, 'lr': self.arg.logic_lr, 'name': 'logic_layers'},
        ]
        self.print_log(
            f'Optimizer param groups: backbone_concept lr={self.arg.base_lr}, '
            f'logic_layers lr={self.arg.logic_lr} '
            f'({sum(p.numel() for p in logic_params):,} logic params).'
        )

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                param_groups,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log(f'using warm up, epoch: {self.arg.warm_up_epoch}')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def bare_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def checkpoint_path(self):
        return os.path.join(self.arg.work_dir, 'checkpoint.pt')

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.bare_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'lr': self.lr,
            'best_acc': self.best_acc,
            'best_acc_epoch': self.best_acc_epoch,
            'global_step': self.global_step,
        }
        ckpt_path = self.checkpoint_path()
        tmp_path = ckpt_path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, ckpt_path)

    def preload_checkpoint(self):
        self.resume_ckpt = None
        self._in_stage2 = False
        if not self.arg.resume:
            return
        ckpt_path = self.checkpoint_path()
        if not os.path.isfile(ckpt_path):
            return
        map_location = f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu'
        self.resume_ckpt = torch.load(ckpt_path, map_location=map_location)
        self.arg.start_epoch = self.resume_ckpt['epoch'] + 1
        self.print_log(f"[Resume] Preloaded checkpoint. Starting epoch set to {self.arg.start_epoch + 1}")

        if self.arg.two_stage and self.arg.start_epoch >= self.arg.stage1_epoch:
            self._in_stage2 = True
            # Freeze backbone and unfreeze logic immediately so load_optimizer builds the Stage 2 version
            self._freeze_backbone()
            self._unfreeze_logic_layers()

    def try_resume_checkpoint(self):
        if not self.arg.resume or getattr(self, 'resume_ckpt', None) is None:
            return
        checkpoint = self.resume_ckpt
        self.bare_model().load_state_dict(checkpoint['model_state_dict'])

        # The temperature-runaway bug (see clip_temperature) already collapsed
        # exp(t) to ~25+ in checkpoints saved before this fix. Clamping only
        # going forward isn't enough — reset it back into range now so the
        # resumed run doesn't inherit a permanently-collapsed classifier.
        logic_layers = self.bare_model().logic_layers
        t_before = torch.exp(logic_layers.t).item()
        logic_layers.clip_temperature()
        t_after = torch.exp(logic_layers.t).item()
        if abs(t_before - t_after) > 1e-6:
            self.print_log(
                f'[Resume] logic_layers temperature was out of range '
                f'(exp(t)={t_before:.4f}) — reset to exp(t)={t_after:.4f}.'
            )

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.lr = checkpoint['lr']
        self.best_acc = checkpoint['best_acc']
        self.best_acc_epoch = checkpoint['best_acc_epoch']
        self.global_step = checkpoint['global_step']
        self.print_log(
            f"Resumed from checkpoint: continuing at epoch {self.arg.start_epoch + 1}, "
            f"lr={self.lr}, best_acc={self.best_acc * 100:.2f}% (epoch {self.best_acc_epoch}).")

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time :  {localtime}')

    def print_log(self, str_msg, print_time=True):
        if print_time:
            str_msg = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str_msg
        if self.arg.print_log:
            print(str_msg)
        if self.arg.phase == 'train':
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(str_msg, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    # ─── 2-Stage: Freeze / Unfreeze helpers ───────────────────────────────────

    def _set_requires_grad(self, module, requires_grad):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _freeze_backbone(self):
        """Stage 2: Freeze all backbone layers (l1-l10 + data_bn)."""
        m = self.bare_model()
        backbone_modules = ['data_bn', 'l1', 'l2', 'l3', 'l4', 'l5',
                            'l6', 'l7', 'l8', 'l9', 'l10']
        for name in backbone_modules:
            self._set_requires_grad(getattr(m, name), False)
        self.print_log('[2-Stage] Backbone (l1-l10 + data_bn) FROZEN.')

    def _unfreeze_logic_layers(self):
        """Stage 2: Unfreeze ConceptLogicLayers for training."""
        self._set_requires_grad(self.bare_model().logic_layers, True)
        self.print_log('[2-Stage] ConceptLogicLayers UNFROZEN.')

    def _build_stage2_optimizer(self):
        """Build a fresh optimizer for Stage 2 with separate LR groups."""
        m = self.bare_model()
        backbone_lr = self.arg.stage2_backbone_lr
        logic_lr    = self.arg.stage2_logic_lr

        # Concept modules get backbone_lr (fine-tune lightly)
        concept_params = [
            p for n, p in m.named_parameters()
            if not n.startswith('logic_layers') and p.requires_grad
        ]
        # Logic layers get a dedicated (typically higher) lr
        logic_params = list(m.logic_layers.parameters())

        param_groups = [
            {'params': concept_params, 'lr': backbone_lr, 'name': 'concept_modules'},
            {'params': logic_params,   'lr': logic_lr,    'name': 'logic_layers'},
        ]

        if self.arg.optimizer == 'SGD':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        else:
            optimizer = optim.Adam(
                param_groups,
                weight_decay=self.arg.weight_decay)

        self.print_log(
            f'[2-Stage] Stage 2 optimizer built. '
            f'Concept LR={backbone_lr}, Logic LR={logic_lr}'
        )
        return optimizer

    def _enter_stage2(self):
        """Transition: freeze backbone, unfreeze logic, rebuild optimizer+scaler."""
        self.print_log('=' * 70)
        self.print_log('[2-Stage] *** ENTERING STAGE 2 ***')
        self.print_log('[2-Stage] Backbone frozen. ConceptLogicLayers unlocked.')
        self.print_log('=' * 70)

        self._freeze_backbone()
        self._unfreeze_logic_layers()

        self.optimizer = self._build_stage2_optimizer()
        use_cuda = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
        self.lr = self.arg.stage2_logic_lr  # Track the main (logic) lr
        self._in_stage2 = True

    # ─── Train (stage-aware) ──────────────────────────────────────────────────

    def train(self, epoch, save_model=False):
        self.model.train()

        # Determine current stage
        in_stage1 = (not self.arg.two_stage) or (epoch < self.arg.stage1_epoch)
        stage_label = 'S1' if in_stage1 else 'S2'
        self.print_log(f'Training epoch: {epoch + 1} [{stage_label}]')

        loader = self.data_loader['train']
        loss_value = []
        acc_value = []
        ce_loss_value = []
        early_loss_value = []
        late_loss_value = []
        output_std_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        process = tqdm(loader, ncols=40)

        # Warm-up applies only within Stage 1. Scale each param group toward
        # its OWN target LR proportionally — do not overwrite every group with
        # one absolute value, or logic_layers' dedicated (smaller) LR gets
        # clobbered back to base_lr.
        if self.arg.warm_up_epoch > 0 and in_stage1:
            if epoch < self.arg.warm_up_epoch:
                ratio = (epoch + 1) / self.arg.warm_up_epoch
                target_lr = {'backbone_concept': self.arg.base_lr, 'logic_layers': self.arg.logic_lr}
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = target_lr.get(param_group.get('name'), self.arg.base_lr) * ratio

        use_cuda = torch.cuda.is_available()
        logic_layers = self.bare_model().logic_layers

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device, non_blocking=True)
                label = label.long().cuda(self.output_device, non_blocking=True)
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_cuda):
                output, early_loss, late_loss = self.model(data, label=label)
                ce_loss = self.loss(output, label)

                if in_stage1:
                    # Stage 1: full concept supervision, no logic L2 penalty
                    # (logic_layers are frozen — their l2_penalty returns 0 grads anyway,
                    #  but we skip it for clarity and to avoid unnecessary computation)
                    loss = ce_loss + 0.5 * early_loss + 0.5 * late_loss
                else:
                    # Stage 2: classification via logic layers is primary;
                    # lighter concept supervision keeps MLDecoder calibrated
                    loss = (ce_loss
                            + 0.1 * late_loss
                            + logic_layers.l2_weight * logic_layers.l2_penalty())

            ce_loss_value.append(ce_loss.data.item())
            early_loss_value.append(early_loss.data.item() if torch.is_tensor(early_loss) else float(early_loss))
            late_loss_value.append(late_loss.data.item() if torch.is_tensor(late_loss) else float(late_loss))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # logic_layers is now trained from Stage 1 onward, so its W matrices
            # (consumed by Product.apply's log()) must be clipped every step to
            # avoid drifting outside [0, 1] and producing NaNs.
            logic_layers.clip_weights()
            # Prevent the learnable temperature from running away (see
            # clip_temperature docstring) — this is what was collapsing CE to
            # ln(num_classes) regardless of how well concepts were learned.
            logic_layers.clip_temperature()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output, 1)
            acc = torch.mean((predict_label == label).float())
            acc_value.append(acc.data.item())
            output_std_value.append(output.detach().float().std().item())

            self.train_writer.add_scalar('acc',        acc.data.item(),        self.global_step)
            self.train_writer.add_scalar('loss',       loss.data.item(),       self.global_step)
            self.train_writer.add_scalar('loss_raw',   loss.data.item(),       self.global_step)
            self.train_writer.add_scalar('loss_ce',    ce_loss.data.item(),    self.global_step)
            self.train_writer.add_scalar('loss_early', early_loss_value[-1],   self.global_step)
            self.train_writer.add_scalar('loss_late',  late_loss_value[-1],    self.global_step)

            timer['statistics'] += self.split_time()

        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }
        self.print_log(
            f'\tMean training loss: {np.mean(loss_value):.4f}.  '
            f'Mean training acc: {np.mean(acc_value) * 100:.2f}%.')
        self.print_log(
            f'\t  |- CE: {np.mean(ce_loss_value):.4f}  '
            f'early_concept: {np.mean(early_loss_value):.4f}  '
            f'late_concept: {np.mean(late_loss_value):.4f}')
        temp = torch.exp(logic_layers.t).item()
        self.print_log(
            f'\t  |- logic_layers temperature exp(t)={temp:.4f}  '
            f'output(scores) std={np.mean(output_std_value):.4f}  '
            f'grad_scaler_scale={self.scaler.get_scale():.1f}')
        self.print_log(
            f'\tTime consumption: [DataLoader] {proportion["dataloader"]} '
            f'| [Model] {proportion["model"]}')

        if save_model:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            state_dict = model_to_save.state_dict()
            weights = OrderedDict([[''.join(k.split('module.')), v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, f'{self.arg.work_dir}/epoch{epoch + 1}_{np.mean(acc_value) * 100:.2f}.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_wrong = open(wrong_file, 'w')
        if result_file is not None:
            f_result = open(result_file, 'w')
        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}')
        use_cuda = torch.cuda.is_available()
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []

            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device, non_blocking=True)
                    label = label.long().cuda(self.output_device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        output = self.model(data)
                        loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = predict_label.cpu().numpy()
                    target = label.cpu().numpy()
                    for i, prod in enumerate(predict):
                        if result_file is not None:
                            f_result.write(f'{prod},{target[i]}\n')
                        if wrong_file is not None and prod != target[i]:
                            f_wrong.write(f'{index[i]},{prod},{target[i]}\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if float(np.mean(loss_value)) < 0.0001:
                loss = 0.0001
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                if self.arg.phase == 'train':
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    state_dict = model_to_save.state_dict()
                    weights = OrderedDict([[''.join(k.split('module.')) , v.cpu()] for k, v in state_dict.items()])
                    torch.save(weights, f'{self.arg.work_dir}/best_model.pt')
                    self.print_log(f'--> Saved new best model to {self.arg.work_dir}/best_model.pt (Test Acc: {accuracy * 100:.2f}%)')

            print(f'Accuracy: {accuracy:.4f}, model_name: {self.arg.model_saved_name}')

            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_raw', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_value):.4f}.')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {self.data_loader[ln].dataset.top_k(score, k) * 100:.2f}%')

            if save_score:
                with open(f'{self.arg.work_dir}/epoch{epoch + 1}_{accuracy * 100:.2f}_score.pkl', 'wb') as f:
                    pickle.dump(score, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
            self.print_log(f'Model total parameters: {sum(p.numel() for p in self.model.parameters()):,}')

            if self.arg.two_stage:
                self.print_log(
                    f'[2-Stage] ENABLED.  '
                    f'Stage 1: epoch 1-{self.arg.stage1_epoch} (backbone+concept).  '
                    f'Stage 2: epoch {self.arg.stage1_epoch + 1}-{self.arg.num_epoch} (logic layers).'
                )

            _in_stage2 = getattr(self, '_in_stage2', False)  # Track whether we already transitioned

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                # ── Stage transition (fires exactly once at stage1_epoch) ──
                if self.arg.two_stage and not _in_stage2 and epoch >= self.arg.stage1_epoch:
                    self._enter_stage2()
                    _in_stage2 = True

                # ── LR decay (stage-aware) ────────────────────────────────
                if not self.arg.two_stage or not _in_stage2:
                    # Stage 1 uses the global step list. Decay each param group's
                    # OWN current lr multiplicatively — do not overwrite every
                    # group with one shared absolute value, or logic_layers'
                    # dedicated (smaller) lr collapses onto the backbone's.
                    if (epoch + 1) in self.arg.stage1_step if self.arg.two_stage else self.arg.step:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * self.arg.lr_decay_rate
                        lrs = [f"{pg.get('name', 'main')}={pg['lr']:.6f}" for pg in self.optimizer.param_groups]
                        self.print_log(f'[S1] LR decayed: {" | ".join(lrs)}')
                else:
                    # Stage 2: decay relative to stage2 start
                    epoch_in_stage2 = epoch - self.arg.stage1_epoch + 1
                    if epoch_in_stage2 in self.arg.stage2_step:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * self.arg.lr_decay_rate
                        lrs = [f"{pg['name']}={pg['lr']:.6f}" for pg in self.optimizer.param_groups]
                        self.print_log(f'[S2] LR decayed: {" | ".join(lrs)}')

                # ── Train & Eval ──────────────────────────────────────────
                save_this = (
                    ((epoch + 1) % self.arg.save_interval == 0)
                    or (epoch + 1 == self.arg.num_epoch)
                )
                self.train(epoch, save_model=save_this)

                if ((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

                self.save_checkpoint(epoch)
                self.print_log(
                    f'Best Top-1 Accuracy: {self.best_acc * 100:.2f}% at epoch {self.best_acc_epoch}'
                )

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please fill in the weights parameter')
            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.')


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
