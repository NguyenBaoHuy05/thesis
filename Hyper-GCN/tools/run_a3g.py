"""A3-G (model/hypergcn_a3g.py) in one command (plan: KG_RETRIEVAL_PLAN/A3G_PREDICATE_HYPEREDGE/A3G_PLAN.md §7).

  python tools/run_a3g.py [--config config/a3g/run_a3g.yaml] [--only g0 | a3g_off a3g ...]

1. tests (tests/test_a3g.py), if run_tests;
2. data: <data> -> data/ntu120/NTU120_CSub.npz, unless the same size is there;
3. G0 (tools/predicate_stats.py) -> <work_dir>/predicate_stats.npz + .json, if the report is missing or --only g0;
4. stop unless the report says "passed": true (plan §4); a failed G0 is never skipped by a rerun;
5. each G1 config of <stages> (or --only) from the K0 checkpoint: config/a3g/a3g.yaml merged with
   config/a3g/<name>.yaml, stats = <work_dir>/predicate_stats.npz, results in <work_dir>/<name>.
A run whose best_checkpoint.json exists is skipped, so the command resumes after a Colab disconnect.
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_refine import DATA_IN_REPO, REPO, copy, need, run   # noqa: E402


def merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load(name):
    with open(os.path.join(REPO, 'config', 'a3g', name + '.yaml'), encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def stage(name, cfg, stats):
    """main.py on a3g.yaml + <name>.yaml from the K0 checkpoint."""
    model = load('a3g') if name == 'a3g' else merge(load('a3g'), load(name))
    model['work_dir'] = os.path.join(cfg['work_dir'], name)
    model['weights'] = cfg['checkpoint']
    model['model_args']['a3g_args']['stats'] = stats
    done = os.path.join(model['work_dir'], 'best_checkpoint.json')
    if os.path.isfile(done):
        print(f'\nskip {name}: {done} exists')
        return
    os.makedirs(model['work_dir'], exist_ok=True)
    run_config = os.path.join(model['work_dir'], 'run_config.yaml')
    with open(run_config, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(model, handle, sort_keys=False)
    run('main.py', '--config', run_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/a3g/run_a3g.yaml')
    parser.add_argument('--only', nargs='+', help="'g0', or G1 configs to run instead of <stages>")
    args = parser.parse_args()
    with open(os.path.join(REPO, args.config), encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    need(cfg['checkpoint'], 'checkpoint')

    if cfg.get('run_tests', True):
        run('-m', 'pytest', 'tests/test_a3g.py', '-q')

    if not os.path.isfile(DATA_IN_REPO) or os.path.isfile(cfg['data']):
        need(cfg['data'], 'NTU120 data')
        copy(cfg['data'], DATA_IN_REPO)

    stats = os.path.join(cfg['work_dir'], 'predicate_stats.npz')
    report = os.path.splitext(stats)[0] + '.json'
    g0_only = args.only == ['g0']
    if g0_only or not os.path.isfile(report):
        command = ['tools/predicate_stats.py', '--data', DATA_IN_REPO, '--output', stats]
        if cfg.get('k0_scores'):
            command += ['--k0-scores', cfg['k0_scores']]
        try:
            run(*command)
        except Exception as error:                     # G0 stop: the report is written, read it below
            print(f'G0 exited with an error: {error}')
    if not os.path.isfile(report):
        sys.exit(f'no G0 report at {report}')
    with open(report, encoding='utf-8') as handle:
        g0 = json.load(handle)
    if not g0['passed']:
        sys.exit('G0 did not pass, G1 is not run (plan §4): ' + '; '.join(g0['failed']))
    if g0_only:
        sys.exit(0)

    for name in args.only or cfg['stages']:
        stage(name, cfg, stats)
