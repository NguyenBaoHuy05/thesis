"""The main model (online, no cache) in one command.

  python tools/run_refine.py [--config config/concept_graph/run_refine.yaml] [--model-config <main.py config>]

1. tests (tests/test_hypergcn_concept.py, tests/test_concept_action_graph.py), if run_tests;
2. data: <data> -> data/ntu120/NTU120_CSub.npz (path of the feeder configs), unless the same size is there;
3. main.py on <model_config> (default config/concept_graph/online_e3.yaml) with weights = <checkpoint> and
   work_dir = <work_dir>/<config name>; the config actually run is saved there as run_config.yaml.
Hyper-GCN runs on the data at every step: no feature cache. Google Drive must be mounted first (Colab).
"""

import argparse
import os
import shutil
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN_REPO = os.path.join(REPO, 'data', 'ntu120', 'NTU120_CSub.npz')


def run(*cmd):
    print('\n$ ' + ' '.join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=REPO, check=True)


def copy(src, dst):
    """Copy through a temporary name, so an interrupted copy never looks complete (mmap error of 2026-10-20)."""
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f'ok   {dst}')
        return
    print(f'copy {src} -> {dst}', flush=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.part')
    os.replace(dst + '.part', dst)


def need(path, what):
    if not os.path.isfile(path):
        sys.exit(f'{what} not found: {path} (fill it in the config; is Drive mounted?)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/concept_graph/run_refine.yaml')
    parser.add_argument('--model-config', help='overrides model_config, e.g. config/concept_graph/stage1_decoders.yaml')
    args = parser.parse_args()
    with open(os.path.join(REPO, args.config), encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if args.model_config:
        cfg['model_config'] = args.model_config
    need(cfg['checkpoint'], 'checkpoint')

    if cfg.get('run_tests', True):
        run('-m', 'pytest', 'tests/test_hypergcn_concept.py', 'tests/test_concept_action_graph.py', '-q')

    if not os.path.isfile(DATA_IN_REPO) or os.path.isfile(cfg['data']):
        need(cfg['data'], 'NTU120 data')
        copy(cfg['data'], DATA_IN_REPO)

    with open(os.path.join(REPO, cfg['model_config']), encoding='utf-8') as handle:
        model = yaml.safe_load(handle)
    model['work_dir'] = os.path.join(cfg['work_dir'], os.path.splitext(os.path.basename(cfg['model_config']))[0])
    model['weights'] = cfg['checkpoint']
    os.makedirs(model['work_dir'], exist_ok=True)
    run_config = os.path.join(model['work_dir'], 'run_config.yaml')
    with open(run_config, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(model, handle, sort_keys=False)
    run('main.py', '--config', run_config)
