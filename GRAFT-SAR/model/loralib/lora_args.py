"""
LoRA Configuration Arguments
============================
Defines hyperparameters for Low-Rank Adaptation (LoRA) on CLIP modules.
"""

from dataclasses import dataclass, field
from typing import List
from types import SimpleNamespace


@dataclass
class LoraArgs:
    position: str = "all"
    encoder: str = "text"
    params: List[str] = field(default_factory=lambda: ["q", "k", "v"])
    r: int = 4
    alpha: int = 16
    dropout_rate: float = 0.1
    save_path: str = ""
    filename: str = "lora_weights"
    eval_only: bool = False
    seed: int = 1
    root_path: str = ""
    dataset: str = "ntu"
    shots: int = 16
    backbone: str = "ViT-B/32"
    lr: float = 0.01
    n_iters: int = 500
    batch_size: int = 32


def get_lora_args(**kwargs):
    default_args = {
        'seed': 1,
        'root_path': '',
        'dataset': 'ntu',
        'shots': 16,
        'backbone': 'ViT-B/32',
        'lr': 0.01,
        'n_iters': 500,
        'batch_size': 32,
        'position': 'all',
        'encoder': 'text',
        'params': ['q', 'k', 'v'],
        'r': 4,
        'alpha': 16,
        'dropout_rate': 0.1,
        'save_path': '',
        'filename': 'lora_weights',
        'eval_only': False,
    }
    default_args.update(kwargs)
    return SimpleNamespace(**default_args)
