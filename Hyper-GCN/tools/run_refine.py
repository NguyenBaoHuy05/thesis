"""The whole refinement model (KG_RETRIEVAL_PLAN/CONCEPT_GRAPH, ft_e3) in one command, skipping finished steps.

  python tools/run_refine.py [--config config/concept_graph/run_refine.yaml] [--force-head]

1. tests (tests/test_hypergcn_concept.py, tests/test_concept_action_graph.py), if run_tests;
2. data: <data> -> data/ntu120/NTU120_CSub.npz (path of the feeder configs), unless the same size is there;
3-4. only when <work_dir>/e3/head.pt is missing (or --force-head): the S = 1 cache in <cache_dir> (copied from
   <cache_drive> if there, else built from <checkpoint> and copied to <cache_drive>), then E3 on the cache
   (head_config) -> head.pt. The fine-tuning itself does not read the cache;
5. refinement fine-tuning (refine_config) with weights / head_weights / work_dir from this config ->
   <work_dir>/ft_e3 (the config actually run is saved there as run_config.yaml).
Google Drive must be mounted first (Colab).
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


def same_file(src, dst):
    return os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src)


def copy(src, dst):
    """Copy through a temporary name, so an interrupted copy never looks complete (mmap error of 2026-10-20)."""
    if same_file(src, dst):
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
    parser.add_argument('--force-head', action='store_true', help='train E3 on the cache again')
    args = parser.parse_args()
    with open(os.path.join(REPO, args.config) if not os.path.isabs(args.config) else args.config,
              encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    need(cfg['checkpoint'], 'checkpoint')

    if cfg.get('run_tests', True):
        run('-m', 'pytest', 'tests/test_hypergcn_concept.py', 'tests/test_concept_action_graph.py', '-q')

    if not os.path.isfile(DATA_IN_REPO) or os.path.isfile(cfg['data']):
        need(cfg['data'], 'NTU120 data')
        copy(cfg['data'], DATA_IN_REPO)

    # The cache (frozen l9 map) is only read to train E3 (step 4); the fine-tuning runs the backbone on the data.
    head = os.path.join(cfg['work_dir'], 'e3', 'head.pt')
    if args.force_head or not os.path.isfile(head):
        files = [f"{cfg['cache_name']}_{split}{ext}" for split in ('train', 'test') for ext in ('_feat.npy', '.npz')]
        on_drive = all(os.path.isfile(os.path.join(cfg['cache_drive'], f)) for f in files)
        if on_drive:
            for f in files:
                copy(os.path.join(cfg['cache_drive'], f), os.path.join(cfg['cache_dir'], f))
        elif not all(os.path.isfile(os.path.join(cfg['cache_dir'], f)) for f in files):
            run('tools/cache_joint_features.py', '--weights', cfg['checkpoint'], '--out-dir', cfg['cache_dir'],
                '--name', cfg['cache_name'], '--num-worker', '2')
            for f in files:
                copy(os.path.join(cfg['cache_dir'], f), os.path.join(cfg['cache_drive'], f))
        run('train_concept_logic.py', '--config', cfg['head_config'], '--cache-dir', cfg['cache_dir'],
            '--work-dir', os.path.dirname(head))
    else:
        print(f'ok   {head} (E3 already trained; --force-head to train it again)')

    with open(os.path.join(REPO, cfg['refine_config']), encoding='utf-8') as handle:
        refine = yaml.safe_load(handle)
    refine['work_dir'] = os.path.join(cfg['work_dir'], 'ft_e3')
    refine['weights'] = cfg['checkpoint']
    refine['model_args']['head_weights'] = head
    os.makedirs(refine['work_dir'], exist_ok=True)
    run_config = os.path.join(refine['work_dir'], 'run_config.yaml')
    with open(run_config, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(refine, handle, sort_keys=False)
    run('main.py', '--config', run_config)
