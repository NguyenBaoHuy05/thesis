"""Runner for Direction 2: Compositional / Concept-Conditioned Flow Matching.

Variants:
  - cfm_adaln      : Concept-Conditioned DiT (adaLN-Zero on class concept) + sub-action concept flow (lambda_sub=0.25)
  - cfm_adaln_nosub: Concept-Conditioned DiT without sub-action concept flow (ablation: lambda_sub=0.0)
  - cfm_superpos   : Explicit Velocity Superposition over required primitives + sub-action concept flow

Usage (Colab):
  !python zeroshot/concept_map/run_cfm.py --feat star --tasks 48 110 --seeds 1 --force
"""
import argparse
import os
import re
import subprocess
import sys

CMAP_DIR = os.path.dirname(os.path.abspath(__file__))
FLORA_DIR = os.path.normpath(os.path.join(CMAP_DIR, '..', 'Flora'))
if FLORA_DIR not in sys.path:
    sys.path.insert(0, FLORA_DIR)
if CMAP_DIR not in sys.path:
    sys.path.insert(0, CMAP_DIR)

import nesy_common as C

DRIVE_OUT_CFM = '/content/drive/MyDrive/BaoHuy/zeroshot/concept_cfm_dir2'

VARIANTS = {
    'cfm_adaln_nosub': dict(cfm_mode='adaln',    lambda_sub=0.0,  sub_margin=0.05, cond_drop=0.1,  superpos_gamma=1.0),
    'cfm_adaln':       dict(cfm_mode='adaln',    lambda_sub=0.25, sub_margin=0.05, cond_drop=0.1,  superpos_gamma=1.0),
    'cfm_superpos':    dict(cfm_mode='superpos', lambda_sub=0.25, sub_margin=0.05, cond_drop=0.15, superpos_gamma=1.0),
    'cfm_superpos_g5': dict(cfm_mode='superpos', lambda_sub=0.25, sub_margin=0.05, cond_drop=0.15, superpos_gamma=0.5),
}

CFM_EVAL_RE = re.compile(
    r'\[CFM-EVAL\] Flow Concept AUC=([\d.nan]+) \(AP=([\d.nan]+), n_prims=(\d+)\) \| '
    r'Seen-cov AUC=([\d.nan]+) \(\d+\), Unseen-only AUC=([\d.nan]+) \(\d+\)'
)


def parse_cfm_log(run_dir):
    acc_list, _ = C.read_log(run_dir)
    if not acc_list:
        return None
    text = open(os.path.join(run_dir, 'run.log'), encoding='utf-8').read()
    m = CFM_EVAL_RE.search(text)
    return {
        'best': max(acc_list),
        'last10': sum(acc_list[-10:]) / min(len(acc_list), 10),
        'flow_auc': float(m.group(1)) if m else float('nan'),
        'flow_ap': float(m.group(2)) if m else float('nan'),
        'unseen_auc': float(m.group(5)) if m else float('nan'),
    }


def run_single(opts):
    os.chdir(FLORA_DIR)
    C.set_feat(opts.feat)
    import utils.utils as U
    from trainer_cfm import CompositionalFMTrainer

    task = C.SPLITS[opts.task][0]
    prims, rules = C.kg_for_task(task, opts.concept_prims, opts.concept_rules)
    vcfg = dict(VARIANTS[opts.variant])
    if opts.lambda_sub is not None:
        vcfg['lambda_sub'] = opts.lambda_sub
    if opts.superpos_gamma is not None:
        vcfg['superpos_gamma'] = opts.superpos_gamma
    vcfg['concept_prims'] = prims
    vcfg['concept_rules'] = rules

    out_dir = os.path.join(opts.out_root, opts.feat, opts.variant, f'{task}_{opts.setting}_seed{opts.seed}')
    os.makedirs(out_dir, exist_ok=True)
    if C.run_done(out_dir, opts.setting) and not opts.force:
        print(f'[skip] {out_dir}')
        return

    flora_ov = dict(kg_mode='none', setting=opts.setting)
    args = C.build_args(task, opts.seed, out_dir, flora_overrides=flora_ov, extra=vcfg)
    U.init_seed(args.seed)
    CompositionalFMTrainer(args).train()


def main():
    ap = argparse.ArgumentParser(description='Direction 2: Compositional Flow Matching runner')
    ap.add_argument('--feat', choices=['star', 'synse', 'sadave'], default='star')
    ap.add_argument('--setting', choices=['ZSL', 'GZSL'], default='ZSL')
    ap.add_argument('--tasks', nargs='+', default=['48', '110'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--variants', nargs='+', default=['cfm_adaln_nosub', 'cfm_adaln', 'cfm_superpos'])
    ap.add_argument('--out-root', default=DRIVE_OUT_CFM if os.path.isdir('/content/drive/MyDrive')
                    else os.path.join(CMAP_DIR, 'runs_cfm'))
    ap.add_argument('--concept-prims', default='kg/primitive_v2.json')
    ap.add_argument('--concept-rules', default='kg/kg_rules_v2.json')
    ap.add_argument('--lambda-sub', type=float, default=None)
    ap.add_argument('--superpos-gamma', type=float, default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--task', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--seed', type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument('--variant', default=None, help=argparse.SUPPRESS)
    opts = ap.parse_args()

    if opts.single:
        run_single(opts)
        return

    C.set_feat(opts.feat)
    for variant in opts.variants:
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant {variant}; choose from {list(VARIANTS)}')

    for variant in opts.variants:
        for t in opts.tasks:
            for s in opts.seeds:
                task = C.SPLITS[t][0]
                run_dir = os.path.join(opts.out_root, opts.feat, variant, f'{task}_{opts.setting}_seed{s}')
                if C.run_done(run_dir, opts.setting) and not opts.force:
                    print(f'[skip] {variant} | {t} | seed {s}')
                    continue
                cmd = [
                    sys.executable, os.path.abspath(__file__),
                    '--single', '--feat', opts.feat, '--setting', opts.setting,
                    '--task', t, '--seed', str(s), '--variant', variant,
                    '--out-root', opts.out_root,
                    '--concept-prims', opts.concept_prims,
                    '--concept-rules', opts.concept_rules,
                ]
                if opts.lambda_sub is not None:
                    cmd += ['--lambda-sub', str(opts.lambda_sub)]
                if opts.superpos_gamma is not None:
                    cmd += ['--superpos-gamma', str(opts.superpos_gamma)]
                if opts.force:
                    cmd.append('--force')
                print(f'\n[run] {variant} | split={t} | seed={s}')
                subprocess.run(cmd, check=True)

    print(f'\n=== Direction 2: Compositional Flow Matching Summary (feat={opts.feat}, setting={opts.setting}) ===')
    hdr = f'{"Variant":<16} {"Split":<6} {"Best":>7} {"Last10":>7} {"FlowAUC":>8} {"FlowAP":>7} {"UnseenC":>8}'
    print(hdr)
    print('-' * len(hdr))
    for variant in opts.variants:
        for t in opts.tasks:
            task = C.SPLITS[t][0]
            for s in opts.seeds:
                run_dir = os.path.join(opts.out_root, opts.feat, variant, f'{task}_{opts.setting}_seed{s}')
                res = parse_cfm_log(run_dir)
                if not res:
                    print(f'{variant:<16} {t:<6}   (incomplete)')
                else:
                    print(
                        f'{variant:<16} {t:<6} '
                        f'{res["best"]:7.2f} {res["last10"]:7.2f} '
                        f'{res["flow_auc"]:8.4f} {res["flow_ap"]:7.4f} {res["unseen_auc"]:8.4f}'
                    )


if __name__ == '__main__':
    main()
