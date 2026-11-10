"""Shared runner for the NeSy experiments on Flora (nesy1_logic.py ... nesy4_seqkg.py).

Each experiment file defines its method (usually a Trainer subclass) and calls `main(...)`.
`python nesyX.py` then:
  1. runs the plain-Flora baselines it compares against (none, compose), skipping finished runs;
  2. runs the method on every (split, seed);
  3. prints and saves a table of best / last10 ZSL accuracy and the gain over the baselines.

Run directories follow flora_nesy.ipynb, so baselines already trained there are reused:
  <out-root>/none/<task>_ZSL_seed<k>/run.log
  <out-root>/kg_compose_a0.5_b0.3/<task>_ZSL_seed<k>/run.log
  <out-root>/<method label>/<task>_ZSL_seed<k>/run.log

Every run is a separate subprocess (the same script with --single), so a crash or an OOM in one
run does not stop the grid, and CUDA memory is released between runs.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                      # Flora configs and data paths are relative to this folder
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# short name -> (task, Flora config with the split's hyper-parameters, ZSL top-1 reported by Flora on STAR features)
SPLITS = {
    '55':  ('ntu60_xsub_seen55_unseen5',    'configs/star/ntu60_xsub_unseen5.yaml',    85.8),
    '48':  ('ntu60_xsub_seen48_unseen12',   'configs/synse/ntu60_xsub_unseen12.yaml',  61.5),
    '110': ('ntu120_xsub_seen110_unseen10', 'configs/synse/ntu120_xsub_unseen10.yaml', 80.7),
    '96':  ('ntu120_xsub_seen96_unseen24',  'configs/synse/ntu120_xsub_unseen24.yaml', 64.1),
}
FEAT = 'star'
DRIVE_OUT = '/content/drive/MyDrive/BaoHuy/zeroshot/flora_nesy'
BASELINE_LABEL = {'none': 'none', 'compose': 'kg_compose_a0.5_b0.3'}   # names used by flora_nesy.ipynb

ZSL_RE = re.compile(r'Iter (\d+): ZSL Accuracy: ([\d.]+)%')
TAG_RE = re.compile(r'Iter (\d+): \[([^\]]+)\] ZSL Accuracy: ([\d.]+)%')   # extra accuracies a method logs


# ---------------------------------------------------------------- config

def build_args(task, seed, out_dir, flora_overrides=None, extra=None):
    """Flora args exactly as main.py builds them from the split's config, plus our overrides.
    flora_overrides: keys Flora's parser knows (kg_mode, kg_beta, ...). extra: method-only attributes."""
    import yaml
    import options
    cfg_path = next(v[1] for v in SPLITS.values() if v[0] == task)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg.update(task_name=task, data_path=f'skeleton_features/{FEAT}_features/{task}.npz', use_features=True,
               shiftgcn_checkpoint_path=None, clip_checkpoint_path='openai/clip-vit-large-patch14-336',
               num_worker=2, setting='ZSL', seed=seed, out_dir=out_dir)
    cfg.update(flora_overrides or {})
    parser = options.get_args_parser()
    known = {a.dest for a in parser._actions}
    bad = [k for k in cfg if k not in known]
    assert not bad, f'unknown Flora args: {bad}'
    parser.set_defaults(**cfg)
    args = parser.parse_args([])
    for k, v in (extra or {}).items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------- logs

def run_done(run_dir):
    log = os.path.join(run_dir, 'run.log')
    return os.path.exists(log) and 'Best ZSL' in open(log).read()


def read_log(run_dir):
    """(main ZSL accuracies per eval, {tag: accuracies per eval}) or (None, {})."""
    log = os.path.join(run_dir, 'run.log')
    if not os.path.exists(log):
        return None, {}
    text = open(log).read()
    main_acc = [float(m.group(2)) for m in ZSL_RE.finditer(text)]
    tags = {}
    for m in TAG_RE.finditer(text):
        tags.setdefault(m.group(2), []).append(float(m.group(3)))
    return (main_acc or None), tags


def stats(acc):
    return dict(best=max(acc), last10=sum(acc[-10:]) / len(acc[-10:]))


# ---------------------------------------------------------------- one run (inside the subprocess)

def run_single(trainer_cls, task, seed, run_dir, flora_overrides, extra):
    import utils.utils as utils
    os.makedirs(run_dir, exist_ok=True)
    args = build_args(task, seed, run_dir, flora_overrides, extra)
    utils.init_seed(seed)
    trainer = trainer_cls(args)
    trainer.train()


def spawn(argv, label, run_dir):
    if run_done(run_dir):
        print(f'[skip] {label}: đã xong')
        return
    shutil.rmtree(run_dir, ignore_errors=True)   # Flora's logger appends; never mix two runs in one log
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.abspath(sys.argv[0])] + argv, capture_output=True, text=True)
    print(f'[{label}] exit={p.returncode}  {(time.time() - t0) / 60:.1f} phút', flush=True)
    if p.returncode != 0:
        print(p.stderr[-3000:], flush=True)
    else:
        keep = lambda l: 'Best ZSL' in l or 'primitive acc' in l or '[nesy' in l or \
            ('Iter 1200:' in l and 'ZSL Accuracy' in l)          # all Flora configs use total_iter = 1200
        tail = [l for l in p.stdout.splitlines() if keep(l)]
        print('\n'.join(tail[-14:]), flush=True)


# ---------------------------------------------------------------- the experiment grid

def main(name, trainer_cls, method_args=(), default_kg_mode='compose', label_keys=(), prepare=None,
         flora_from_method=None, defaults=None):
    """name: label prefix. trainer_cls: Trainer subclass running the method.
    method_args: [(flag, kwargs for add_argument)], attributes set on Flora args.
    label_keys: [(dest, short)] that go into the run-dir label.
    prepare(opts): optional, runs once in the parent before the grid (e.g. build a KG file).
    flora_from_method(opts) -> dict: extra Flora args derived from method options."""
    ap = argparse.ArgumentParser(description=f'{name} on Flora', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--tasks', nargs='+', default=['48', '110'], choices=list(SPLITS), help='splits to run')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    out_default = DRIVE_OUT if os.path.isdir(os.path.dirname(DRIVE_OUT)) else os.path.join(HERE, 'output', 'nesy')
    ap.add_argument('--out-root', default=out_default)
    ap.add_argument('--kg-mode', default=default_kg_mode, choices=['none', 'compose'],
                    help='semantic side under the method: Flora alone, or with compose')
    ap.add_argument('--kg-beta', type=float, default=0.3)
    ap.add_argument('--kg-prims', default='kg/primitive_v2.json',
                    help='KG used by the method and by compose; nesy4_seqkg.py writes kg/primitive_v2seq.json')
    ap.add_argument('--kg-rules', default='kg/kg_rules_v2.json')
    ap.add_argument('--no-baselines', action='store_true', help='do not train missing none/compose baselines')
    ap.add_argument('--single', nargs=2, metavar=('TASK', 'SEED'), help=argparse.SUPPRESS)
    ap.add_argument('--baseline', choices=['none', 'compose'], help=argparse.SUPPRESS)
    ap.add_argument('--run-dir', help=argparse.SUPPRESS)
    for flag, kw in method_args:
        ap.add_argument(flag, **kw)
    if defaults:
        ap.set_defaults(**defaults)
    opts = ap.parse_args()
    method_dests = [ap._option_string_actions[flag].dest for flag, _ in method_args]
    extra = {d: getattr(opts, d) for d in method_dests}
    flora = dict(kg_mode=opts.kg_mode, kg_beta=opts.kg_beta, kg_prims=opts.kg_prims, kg_rules=opts.kg_rules)
    if flora_from_method:
        flora.update(flora_from_method(opts))

    # ---- inside a subprocess: one run
    if opts.single:
        task, seed = opts.single[0], int(opts.single[1])
        if opts.baseline:
            from trainer import Trainer
            run_single(Trainer, task, seed, opts.run_dir, dict(kg_mode=opts.baseline), {})
        else:
            run_single(trainer_cls, task, seed, opts.run_dir, flora, extra)
        return

    # ---- parent: the grid
    if prepare:
        prepare(opts)
    kg_tag = '' if opts.kg_prims.endswith('primitive_v2.json') else '_' + os.path.splitext(os.path.basename(opts.kg_prims))[0]
    label = name + ''.join(f'_{short}{getattr(opts, d)}' for d, short in label_keys) + f'_on-{opts.kg_mode}' + \
        (f'{opts.kg_beta}' if opts.kg_mode == 'compose' else '') + kg_tag
    print(f'[nesy] {label}  tasks={opts.tasks} seeds={opts.seeds}  out={opts.out_root}', flush=True)
    passthrough = [a for a in sys.argv[1:]]
    for short in opts.tasks:
        task = SPLITS[short][0]
        if not os.path.exists(f'skeleton_features/{FEAT}_features/{task}.npz'):
            print(f'[thiếu feature] {task}: bỏ qua')
            continue
        for seed in opts.seeds:
            if not opts.no_baselines:
                for b in ('none', 'compose'):
                    d = f'{opts.out_root}/{BASELINE_LABEL[b]}/{task}_ZSL_seed{seed}'
                    spawn(passthrough + ['--single', task, str(seed), '--baseline', b, '--run-dir', d],
                          f'{b} {short} seed{seed}', d)
            d = f'{opts.out_root}/{label}/{task}_ZSL_seed{seed}'
            spawn(passthrough + ['--single', task, str(seed), '--run-dir', d], f'{label} {short} seed{seed}', d)
    summarize(opts, label)


def summarize(opts, label):
    import pandas as pd
    rows = []
    for short in opts.tasks:
        task, _, paper = SPLITS[short]
        for seed in opts.seeds:
            base = {}
            for b in ('none', 'compose'):
                acc, _ = read_log(f'{opts.out_root}/{BASELINE_LABEL[b]}/{task}_ZSL_seed{seed}')
                base[b] = stats(acc) if acc else None
            acc, tags = read_log(f'{opts.out_root}/{label}/{task}_ZSL_seed{seed}')
            if not acc:
                continue
            s = stats(acc)
            row = dict(split=short, seed=seed, paper=paper, best=s['best'], last10=s['last10'])
            for b in ('none', 'compose'):
                row[f'd_best_vs_{b}'] = s['best'] - base[b]['best'] if base[b] else float('nan')
            for t, a in tags.items():
                row[f'best[{t}]'] = max(a)
            rows.append(row)
    if not rows:
        print('[nesy] chưa có run nào xong')
        return
    df = pd.DataFrame(rows).round(2)
    out = f'{opts.out_root}/{label}/summary.csv'
    df.to_csv(out, index=False)
    with pd.option_context('display.max_columns', 50, 'display.width', 250):
        print(df.to_string(index=False))
        mean = df.drop(columns=['seed']).groupby('split', sort=False).mean(numeric_only=True).round(2)
        print('\nmean over seeds:\n' + mean.to_string())
    print(f'[nesy] saved {out}')
