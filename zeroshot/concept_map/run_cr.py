"""Runner for Decoupled Dual-Branch Concept-Residual Mapping (Direction 1).

Compares variants on pre-extracted 256-D features (STAR / SynSE):
  - flora_base  : Flora baseline (lambda_c=0, lambda_proto=0, lambda_adv=0, alpha=0)
  - cr_dual     : Decoupled ConceptEncoder (CLIP head + proto align) + Dual-Path Inference (alpha=0.25)
  - cr_dual_grl : cr_dual + GRL adversarial concept suppression on Residual latent (lambda_adv=0.05)
  - cr_prob     : cr_dual using probabilistic rule-consistency score instead of continuous CLIP prototype
  - cr_linear   : Ablation replacing CLIPConceptHead with LinearConceptHead (prob score)

Usage (Colab):
  !python zeroshot/concept_map/run_cr.py --tasks 48 110 --seeds 1 --force
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
    'flora_base':  dict(concept_dim=256, lambda_concept=0.0, lambda_proto=0.0, lambda_adv=0.0,
                        concept_alpha=0.0, concept_head='clip', concept_score='proto'),
    'cr_dual':     dict(concept_dim=256, lambda_concept=1.0, lambda_proto=0.5, lambda_adv=0.0,
                        concept_alpha=0.25, concept_head='clip', concept_score='proto'),
    'cr_dual_grl': dict(concept_dim=256, lambda_concept=1.0, lambda_proto=0.5, lambda_adv=0.05,
                        concept_alpha=0.25, concept_head='clip', concept_score='proto'),
    'cr_prob':     dict(concept_dim=256, lambda_concept=1.0, lambda_proto=0.5, lambda_adv=0.05,
                        concept_alpha=0.25, concept_head='clip', concept_score='prob'),
    'cr_both':     dict(concept_dim=256, lambda_concept=1.0, lambda_proto=0.5, lambda_adv=0.05,
                        concept_alpha=0.25, concept_head='clip', concept_score='both'),
    'cr_linear':   dict(concept_dim=256, lambda_concept=1.0, lambda_proto=0.0, lambda_adv=0.05,
                        concept_alpha=0.25, concept_head='linear', concept_score='prob'),
}

CR_EVAL_RE = re.compile(
    r'\[CR-EVAL\] Unseen Concept Head AUC=([\d.nan]+) \(AP=([\d.nan]+), n_prims=(\d+)\) \| '
    r'Seen-cov AUC=([\d.nan]+) \(\d+\), Unseen-only AUC=([\d.nan]+) \(\d+\) \| '
    r'Concept-only ZSL=([\d.nan]+)% \| '
    r'Probe\(z_c\) AUC=([\d.nan]+), Leakage Probe\(z_r\) AUC=([\d.nan]+)'
)


def parse_cr_log(run_dir):
    acc_list, tags = C.read_log(run_dir)
    if not acc_list:
        return None
    text = open(os.path.join(run_dir, 'run.log'), encoding='utf-8').read()
    m = CR_EVAL_RE.search(text)
    out = {
        'best': max(acc_list),
        'last10': sum(acc_list[-10:]) / min(len(acc_list), 10),
        'head_auc': float(m.group(1)) if m else float('nan'),
        'unseen_auc': float(m.group(5)) if m else float('nan'),
        'c_only_zsl': float(m.group(6)) if m else float('nan'),
        'probe_zc': float(m.group(7)) if m else float('nan'),
        'probe_zr': float(m.group(8)) if m else float('nan'),
        'tags': {k: max(v) for k, v in tags.items()},
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
    if opts.lambda_proto is not None:
        vcfg['lambda_proto'] = opts.lambda_proto
    if opts.lambda_adv is not None:
        vcfg['lambda_adv'] = opts.lambda_adv
    if opts.concept_alpha is not None:
        vcfg['concept_alpha'] = opts.concept_alpha
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
    ConceptResidualTrainer(args).train()


def main():
    ap = argparse.ArgumentParser(description='Decoupled Concept-Residual Mapping runner')
    ap.add_argument('--feat', choices=['star', 'synse', 'sadave'], default='star')
    ap.add_argument('--setting', choices=['ZSL', 'GZSL'], default='ZSL')
    ap.add_argument('--tasks', nargs='+', default=['48', '110'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--variants', nargs='+', default=['flora_base', 'cr_dual', 'cr_dual_grl', 'cr_prob', 'cr_linear'])
    ap.add_argument('--out-root', default=DRIVE_OUT_CR if os.path.isdir('/content/drive/MyDrive')
                    else os.path.join(CMAP_DIR, 'runs'))
    ap.add_argument('--concept-prims', default='kg/primitive_v2.json')
    ap.add_argument('--concept-rules', default='kg/kg_rules_v2.json')
    ap.add_argument('--concept-dim', type=int, default=None)
    ap.add_argument('--lambda-concept', type=float, default=None)
    ap.add_argument('--lambda-proto', type=float, default=None)
    ap.add_argument('--lambda-adv', type=float, default=None)
    ap.add_argument('--concept-alpha', type=float, default=None)
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
                for flag, val in (
                    ('--concept-dim', opts.concept_dim),
                    ('--lambda-concept', opts.lambda_concept),
                    ('--lambda-proto', opts.lambda_proto),
                    ('--lambda-adv', opts.lambda_adv),
                    ('--concept-alpha', opts.concept_alpha),
                ):
                    if val is not None:
                        cmd += [flag, str(val)]
                if opts.force:
                    cmd.append('--force')
                print(f'\n[run] {variant} | split={t} | seed={s}')
                subprocess.run(cmd, check=True)

    print(f'\n=== Concept-Residual Summary (feat={opts.feat}, setting={opts.setting}) ===')
    hdr = (
        f'{"Variant":<13} {"Split":<6} {"Best":>7} {"Last10":>7} {"C-Only":>7} '
        f'{"HeadAUC":>8} {"UnseenC":>8} {"P(z_c)":>7} {"Leak(z_r)":>9}  AlphaSweep'
    )
    print(hdr)
    print('-' * (len(hdr) + 18))
    for variant in opts.variants:
        for t in opts.tasks:
            task = C.SPLITS[t][0]
            for s in opts.seeds:
                run_dir = os.path.join(opts.out_root, opts.feat, variant, f'{task}_{opts.setting}_seed{s}')
                res = parse_cr_log(run_dir)
                if not res:
                    print(f'{variant:<13} {t:<6}   (incomplete)')
                else:
                    sw = ' '.join(f'{k}:{v:.2f}' for k, v in sorted(res['tags'].items()))
                    print(
                        f'{variant:<13} {t:<6} '
                        f'{res["best"]:7.2f} {res["last10"]:7.2f} {res["c_only_zsl"]:7.2f} '
                        f'{res["head_auc"]:8.4f} {res["unseen_auc"]:8.4f} '
                        f'{res["probe_zc"]:7.4f} {res["probe_zr"]:9.4f}  {sw}'
                    )


if __name__ == '__main__':
    main()
