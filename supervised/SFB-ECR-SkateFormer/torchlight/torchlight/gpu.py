import os
import torch

def visible_gpu(gpus):
    if isinstance(gpus, (list, tuple)):
        gpus = ','.join(str(g) for g in gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

def ngpu(gpus):
    if isinstance(gpus, (list, tuple)):
        return len(gpus)
    return len(str(gpus).split(','))
