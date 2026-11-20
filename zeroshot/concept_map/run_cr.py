"""Step-1 prototype runner for Direction 1 (Disentangled Concept-Residual Mapping).

Runs isolated subprocesses per (variant, split, seed) using Flora's pre-extracted 256-D features
and compares:
  - flora_base  : lambda_c=0, lambda_orth=0 (sanity check against Flora baseline)
  - cr_no_orth  : D_c=256, lambda_c=0.1, lambda_orth=0.0, CLIP-anchored head
  - cr_orth     : D_c=256, lambda_c=0.1, lambda_orth=0.1, CLIP-anchored head
  - cr_linear   : D_c=256, lambda_c=0.1, lambda_orth=0.1, Linear head (ablation)
  - cr_cbm_only : D_c=768 (D_r=0), lambda_c=0.1, lambda_orth=0.0 (pure bottleneck ablation)

Usage (Colab):
  !python zeroshot/concept_map/run_cr.py --tasks 48 110 --seeds 1
  !python zeroshot/concept_map/run_cr.py --tasks 48 110 --seeds 1 --variants flora_base cr_no_orth cr_orth
  !python zeroshot/concept_map/run_cr.py --tasks 48 110 --seeds 1 --feat synse
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

DRIVE_OUT_CR = '/content/drive/MyDrive/BaoHuy/zeroshot/concept_map_step1'

VARIANTS = {
    'flora_base':  dict(concept_dim=256, lambda_concept=0.0, lambda_orth=0.0, concept_head='clip'),
    'cr_no_orth':  dict(concept_dim=256, lambda_concept=0.1, lambda_orth=0.0, concept_head='clip'),
    'cr_orth':     dict(concept_dim=256, lambda_concept=0.1, lambda_orth=0.1, concept_head='clip'),
    'cr_orth_s2':  dict(concept_dim=256, lambda_concept=0.2, lambda_orth=0.2, concept_head='clip'),
    'cr_linear':   dict(concept_dim=256, lambda_concept=0.1, lambda_orth=0.1, concept_head='linear'),
    'cr_cbm_only': dict(concept_dim=768, lambda_concept=0.1, lambda_orth=0.0, concept_head='clip'),
}

CR_EVAL_RE = re.compile(
    r'\[CR-EVAL\] Unseen Concept Head AUC=([\d.nan]+) \(AP=([\d.nan]+), n_prims=(\d+)\) \| '
    r'Seen-cov AUC=([\d.nan]+) \(\d+\), Unseen-only AUC=([\d.nan]+) \(\d+\) \| '
    r'Probe\(z_c\) AUC=([\d.nan]+), Leakage Probe\(z_r\) AUC=([\d.nan]+)'
)


def parse_cr_log(run_dir):
    """Return dict with ZSL best/last10 and CR-EVAL concept metrics."""
    acc_list, _ = C.read_log(run_dir)
    if not acc_list:
        return None
    text = open(os.path.join(run_dir, 'run.log'), encoding='utf-8').read()
    m = CR_EVAL_RE.search(text)
    out = {
        'best': max(acc_list),
        'last10': sum(acc_list[-10:]) / min(len(acc_list), 10),
        'head_auc': float(m.group(1)) if m else float('nan'),
        'head_ap': float(m.group(2)) if m else float('nan'),
        'probe_zc': float(m.group(6)) if m else float('nan'),
        'probe_zr': float(m.group(7)) if m else float('nan'),
    }
    return out


def run_single(opts):
    os.chdir(FLORA_DIR)
    C.set_feat(opts.feat)
    import utils.utils as U
    from trainer_cr import ConceptResidualTrainer

    task = C.SPLITS[opts.task][0]
    prims, rules = C.kg_for_task(task, opts.concept_prims, opts.concept_rules)
    vcfg = dict(VARIANTS[opts.variant])
    if opts.concept_dim is not None:
        vcfg['concept_dim'] = opts.concept_dim
    if opts.lambda_concept is not None:
        vcfg['lambda_concept'] = opts.lambda_concept
    if opts.lambda_orth is not None:
        vcfg['lambda_orth'] = opts.lambda_orth
    vcfg['concept_prims'] = prims
    vcfg['concept_rules'] = rules
    vcfg['text_concept_weight'] = opts.text_concept_weight

    out_dir = os.path.join(opts.out_root, opts.feat, opts.variant, f'{task}_{opts.setting}_seed{opts.seed}')
    os.makedirs(out_dir, exist_ok=True)
    if C.run_done(out_dir, opts.setting) and not opts.force:
        print(f'[skip] {out_dir}')
        return

    flora_ov = dict(kg_mode='none', setting=opts.setting)
    args = C.build_args(task, opts.seed, out_dir, flora_overrides=flora_ov, extra=vcfg)
    U.init_seed(args.seed)
    ConceptResidualTrainer(args).train()


def main():
    ap = argparse.ArgumentParser(description='Step-1 Concept-Residual Mapping prototype runner')
    ap.add_argument('--feat', choices=['star', 'synse', 'sadave'], default='star')
    ap.add_argument('--setting', choices=['ZSL', 'GZSL'], default='ZSL')
    ap.add_argument('--tasks', nargs='+', default=['48', '110'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--variants', nargs='+', default=['flora_base', 'cr_no_orth', 'cr_orth', 'cr_linear'])
    ap.add_argument('--out-root', default=DRIVE_OUT_CR if os.path.isdir('/content/drive/MyDrive')
                    else os.path.join(CMAP_DIR, 'runs'))
    ap.add_argument('--concept-prims', default='kg/primitive_v2.json')
    ap.add_argument('--concept-rules', default='kg/kg_rules_v2.json')
    ap.add_argument('--text-concept-weight', type=float, default=0.5)
    ap.add_argument('--concept-dim', type=int, default=None)
    ap.add_argument('--lambda-concept', type=float, default=None)
    ap.add_argument('--lambda-orth', type=float, default=None)
    ap.add_argument('--force', action='store_true')
    # Internal subprocess flags
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
                    '--text-concept-weight', str(opts.text_concept_weight),
                ]
                if opts.concept_dim is not None:
                    cmd += ['--concept-dim', str(opts.concept_dim)]
                if opts.lambda_concept is not None:
                    cmd += ['--lambda-concept', str(opts.lambda_concept)]
                if opts.lambda_orth is not None:
                    cmd += ['--lambda-orth', str(opts.lambda_orth)]
                if opts.force:
                    cmd.append('--force')
                print(f'\n[run] {variant} | split={t} | seed={s}')
                subprocess.run(cmd, check=True)

    # Summary table
    print(f'\n=== Concept-Residual Step-1 Summary (feat={opts.feat}, setting={opts.setting}) ===')
    hdr = f'{"Variant":<13} {"Split":<6} {"Seed":<5} {"Best":>7} {"Last10":>7} {"HeadAUC":>8} {"HeadAP":>7} {"P(z_c)":>7} {"Leak(z_r)":>9}'
    print(hdr)
    print('-' * len(hdr))
    for variant in opts.variants:
        for t in opts.tasks:
            task = C.SPLITS[t][0]
            for s in opts.seeds:
                run_dir = os.path.join(opts.out_root, opts.feat, variant, f'{task}_{opts.setting}_seed{s}')
                res = parse_cr_log(run_dir)
                if not res:
                    print(f'{variant:<13} {t:<6} {s:<5}   (incomplete)')
                else:
                    print(
                        f'{variant:<13} {t:<6} {s:<5} '
                        f'{res["best"]:7.2f} {res["last10"]:7.2f} '
                        f'{res["head_auc"]:8.4f} {res["head_ap"]:7.4f} '
                        f'{res["probe_zc"]:7.4f} {res["probe_zr"]:9.4f}'
                    )


if __name__ == '__main__':
    main()
