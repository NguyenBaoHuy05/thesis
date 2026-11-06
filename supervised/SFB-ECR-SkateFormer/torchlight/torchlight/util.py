import argparse
import os
import sys
import traceback
import time
import pickle
from collections import OrderedDict
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported boolean value: ' + str(v))

class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')
        output_dict = getattr(namespace, self.dest, {}) or {}
        output_dict.update(input_dict)
        setattr(namespace, self.dest, output_dict)

class IO():
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.pavi_logger = None
        self.session_file = None
        self.model_text = ''
        os.makedirs(work_dir, exist_ok=True)

    def log(self, *args, **kwargs):
        pass

    def load_model(self, model, **model_args):
        Model = import_class(model)
        model = Model(**model_args)
        self.model_text += '\n\n' + str(model)
        return model

    def load_weights(self, model, weights_path, ignore_weights=None, fix_weights=False):
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        self.print_log(f'Load weights from {weights_path}.')
        weights = torch.load(weights_path, map_location='cpu')
        if isinstance(weights, dict) and 'state_dict' in weights:
            weights = weights['state_dict']
        elif isinstance(weights, dict) and 'model' in weights:
            weights = weights['model']
            
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])

        for i in ignore_weights:
            ignore_name = [k for k in weights.keys() if k.startswith(i)]
            for item in ignore_name:
                self.print_log(f'Ignore weight: {item}')
                weights.pop(item, None)

        missing, unexpected = model.load_state_dict(weights, strict=False)
        self.print_log(f'Loaded state_dict: missing={len(missing)}, unexpected={len(unexpected)}')

        if fix_weights:
            for name, param in model.named_parameters():
                if name in weights:
                    if not any([name.startswith(i) for i in ignore_weights]):
                        param.requires_grad = False

        return model

    def print_log(self, str_content, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str_content = f"[{localtime}] {str_content}"
        if self.print_to_screen:
            print(str_content)
        if self.save_log:
            with open(os.path.join(self.work_dir, 'log.txt'), 'a', encoding='utf-8') as f:
                f.write(str_content + '\n')
