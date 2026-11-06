"""A3 (model/hypergcn_a3.py) in one command: S1, then every S2 config from the S1 checkpoint.

  python tools/run_a3.py [--config config/a3/run_a3.yaml] [--only s2_a3 ...]

1. tests (tests/test_hypergcn_a3.py), if run_tests;
2. data: <data> -> data/ntu120/NTU120_CSub.npz (path of the feeder configs), unless the same size is there;
3. S1 (config/a3/s1_concept.yaml) with weights = <checkpoint> (K0);
4. each S2 config of <stages> with weights = the S1 checkpoint (<work_dir>/s1_concept/best_checkpoint.json).
Each run goes to <work_dir>/<config name>, with the config actually run saved there as run_config.yaml. A run whose
best_checkpoint.json exists is skipped, so the command resumes after a Colab disconnect (an interrupted run restarts).
Hyper-GCN runs on the data at every step: no feature cache. Google Drive must be mounted first (Colab).
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_refine import DATA_IN_REPO, REPO, copy, need, run   # noqa: E402


def stage(config, weights, work_dir):
    """main.py on config/a3/<config>.yaml; returns the path of its last-epoch checkpoint."""
    with open(os.path.join(REPO, 'config', 'a3', config + '.yaml'), encoding='utf-8') as handle:
        model = yaml.safe_load(handle)
    model['work_dir'] = os.path.join(work_dir, config)
    done = os.path.join(model['work_dir'], 'best_checkpoint.json')
    if os.path.isfile(done):
        print(f'\nskip {config}: {done} exists')
    else:
        model['weights'] = weights
        os.makedirs(model['work_dir'], exist_ok=True)
        run_config = os.path.join(model['work_dir'], 'run_config.yaml')
        with open(run_config, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(model, handle, sort_keys=False)
        run('main.py', '--config', run_config)
    with open(done, encoding='utf-8') as handle:
        return json.load(handle)['path']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/a3/run_a3.yaml')
    parser.add_argument('--only', nargs='+', help='S2 configs to run instead of <stages>, e.g. s2_a3')
    args = parser.parse_args()
    with open(os.path.join(REPO, args.config), encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    need(cfg['checkpoint'], 'checkpoint')

    if cfg.get('run_tests', True):
        run('-m', 'pytest', 'tests/test_hypergcn_a3.py', '-q')

    if not os.path.isfile(DATA_IN_REPO) or os.path.isfile(cfg['data']):
        need(cfg['data'], 'NTU120 data')
        copy(cfg['data'], DATA_IN_REPO)

    s1 = stage('s1_concept', cfg['checkpoint'], cfg['work_dir'])
    print(f'\nS1 checkpoint: {s1}')
    for config in args.only or cfg['stages']:
        stage(config, s1, cfg['work_dir'])
