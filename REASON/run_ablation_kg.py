import os
import copy
import yaml
import json
import random
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG_RANDOM = ROOT / 'config' / 'xsub' / 'config_skeleton_crl_random_pretrain.yaml'
CONFIG_KG = ROOT / 'config' / 'xsub' / 'config_skeleton_crl_kg_pretrain.yaml'
LOG_DIR = ROOT / 'ablation_runs'


@dataclass
class RunConfig:
    name: str
    kg_init_mode: str
    config_path: str
    output_dir: str
    seed: int


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def make_run_config(base_cfg, mode: str, run_name: str, seed: int):
    cfg = copy.deepcopy(base_cfg)
    cfg['logic_model']['kg_init_mode'] = mode
    cfg['work_dir'] = str(ROOT / 'work_dir' / 'ablation' / run_name)
    cfg['seed'] = seed
    cfg['wandb_name'] = f'reason_kg_{mode}_{seed}'
    return cfg


def run_one(run_cfg: RunConfig):
    print(f"\n=== Starting run: {run_cfg.name} ===")
    set_seed(run_cfg.seed)

    out_dir = Path(run_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(run_cfg.config_path)
    cfg = make_run_config(cfg, run_cfg.kg_init_mode, run_cfg.name, run_cfg.seed)
    cfg_path = out_dir / 'config.yaml'
    save_yaml(str(cfg_path), cfg)

    cmd = [
        str(Path(sys_executable())),
        str(ROOT / 'main.py'),
        '--config',
        str(cfg_path),
        '--device',
        '0',
    ]

    print('Command:')
    print(' '.join(cmd))
    import subprocess
    proc = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
    print(f'Exit code: {proc.returncode}')
    return proc.returncode


def sys_executable():
    return os.environ.get('PYTHON', str(Path('venv') / 'Scripts' / 'python.exe')).replace('\\', '/')


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    runs = [
        RunConfig(
            name='random_init',
            kg_init_mode='random',
            config_path=str(CONFIG_RANDOM),
            output_dir=str(LOG_DIR / 'random_init'),
            seed=1,
        ),
        RunConfig(
            name='kg_primed',
            kg_init_mode='kg_primed',
            config_path=str(CONFIG_KG),
            output_dir=str(LOG_DIR / 'kg_primed'),
            seed=1,
        ),
    ]

    results = {}
    for run in runs:
        results[run.name] = run_one(run)

    summary_path = LOG_DIR / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\n=== Summary ===')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
