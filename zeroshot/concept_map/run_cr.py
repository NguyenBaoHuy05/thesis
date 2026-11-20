"""Runner for Group-Factorized Centered CLIPConceptHead + Differential Top-K Arbitration.

Ablates each new component cleanly on 48/12 and 110/10:
  - gf_arb       : Full model (Centered CLIP + 7 Body-Part Group Heads + Soft Text Distill + Top-3 Diff Arbitration)
  - gf_no_soft   : Without Soft LLM Text Distillation (lambda_soft=0.0)
  - gf_no_group  : Single shared CLIP projection without 7 Body-Part Group Heads (use_groups=0)
  - gf_uncentered: Uncentered CLIP primitive space (center_clip=0, use_groups=0, lambda_soft=0)

Note: Every variant logs `a=0.00` (pure untouched Flora DiT baseline) alongside `a in {0.03, 0.05, 0.08, 0.12, 0.20}`
in AlphaSweep, so you get the exact same-run Flora baseline and the arbitrated accuracy in ONE run!

Usage (Colab):
  !python zeroshot/concept_map/run_cr.py --feat star --tasks 48 110 --seeds 1 --force
"""
import argparse
import os
import re
import shutil
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
    'gf_arb':        dict(center_clip=1, use_groups=1, lambda_soft=0.0, arb_topk=3, arb_alpha=0.025),
    'gf_soft':       dict(center_clip=1, use_groups=1, lambda_soft=0.2, arb_topk=3, arb_alpha=0.025),
    'gf_no_group':   dict(center_clip=1, use_groups=0, lambda_soft=0.0, arb_topk=3, arb_alpha=0.025),
    'gf_uncentered': dict(center_clip=0, use_groups=0, lambda_soft=0.0, arb_topk=3, arb_alpha=0.025),
}

GF_EVAL_RE = re.compile(
    r'\[GF-EVAL\] Unseen Concept Head AUC=([\d.nan]+) \(AP=([\d.nan]+), n_prims=(\d+)\) \| '
    r'Seen-cov AUC=([\d.nan]+) \(\d+\), Unseen-only AUC=([\d.nan]+) \(\d+\) \| '
    r'IDF Concept-only ZSL=([\d.nan]+)%'
)


def parse_gf_log(run_dir):
    acc_list, tags = C.read_log(run_dir)
    if not acc_list:
        return None
    text = open(os.path.join(run_dir, 'run.log'), encoding='utf-8').read()
    m = GF_EVAL_RE.search(text)
    return {
        'best': max(acc_list),
        'last10': sum(acc_list[-10:]) / min(len(acc_list), 10),
        'head_auc': float(m.group(1)) if m else float('nan'),
        'head_ap': float(m.group(2)) if m else float('nan'),
        'seen_auc': float(m.group(4)) if m else float('nan'),
        'unseen_auc': float(m.group(5)) if m else float('nan'),
        'c_only_zsl': float(m.group(6)) if m else float('nan'),
        'tags': {k: max(v) for k, v in tags.items()},
    }


def run_single(opts):
    os.chdir(FLORA_DIR)
    C.set_feat(opts.feat)
    import utils.utils as U
    from trainer_cr import ConceptArbitratedTrainer

    task = C.SPLITS[opts.task][0]
    vcfg = dict(VARIANTS[opts.variant])
    if opts.arb_alpha is not None:
        vcfg['arb_alpha'] = opts.arb_alpha
    if opts.arb_topk is not None:
        vcfg['arb_topk'] = opts.arb_topk
    vcfg['concept_bank'] = os.path.abspath(opts.concept_bank)
    vcfg['concept_ann'] = os.path.abspath(opts.concept_ann)
    vcfg['concept_iters'] = opts.concept_iters
    vcfg['concept_lr'] = opts.concept_lr

    out_dir = os.path.join(opts.out_root, opts.feat, opts.variant, f'{task}_{opts.setting}_seed{opts.seed}')
    if C.run_done(out_dir, opts.setting) and not opts.force:
        print(f'[skip] {out_dir}')
        return
    if opts.force and os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    flora_ov = dict(kg_mode='none', setting=opts.setting)
    args = C.build_args(task, opts.seed, out_dir, flora_overrides=flora_ov, extra=vcfg)
    U.init_seed(args.seed)
    ConceptArbitratedTrainer(args).train()


def main():
    ap = argparse.ArgumentParser(description='Group-Factorized CLIPConceptHead + Differential Top-K Arbitration')
    ap.add_argument('--feat', choices=['star', 'synse', 'sadave'], default='star')
    ap.add_argument('--setting', choices=['ZSL', 'GZSL'], default='ZSL')
    ap.add_argument('--tasks', '--splits', dest='tasks', nargs='+', default=['48', '110'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--variants', nargs='+', default=['gf_arb', 'gf_soft', 'gf_no_group', 'gf_uncentered'])
    ap.add_argument('--out-root', '--out_dir', dest='out_root',
                    default=DRIVE_OUT_CR if os.path.isdir('/content/drive/MyDrive')
                    else os.path.join(CMAP_DIR, 'runs'))
    ap.add_argument('--concept-bank', default=os.path.join(CMAP_DIR, 'concepts', 'concept_bank_v1.json'))
    ap.add_argument('--concept-ann', default=os.path.join(CMAP_DIR, 'concepts', 'ntu120_concept_annotations.json'))
    ap.add_argument('--concept-iters', type=int, default=500)
    ap.add_argument('--concept-lr', type=float, default=1e-3)
    ap.add_argument('--arb-alpha', '--arb_alpha', dest='arb_alpha', type=float, default=None)
    ap.add_argument('--arb-topk', '--arb_topk', '--arb_top_k', dest='arb_topk', type=int, default=None)
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
                    '--concept-bank', opts.concept_bank,
                    '--concept-ann', opts.concept_ann,
                    '--concept-iters', str(opts.concept_iters),
                    '--concept-lr', str(opts.concept_lr),
                ]
                if opts.arb_alpha is not None:
                    cmd += ['--arb-alpha', str(opts.arb_alpha)]
                if opts.arb_topk is not None:
                    cmd += ['--arb-topk', str(opts.arb_topk)]
                if opts.force:
                    cmd.append('--force')
                print(f'\n[run] {variant} | split={t} | seed={s}')
                subprocess.run(cmd, check=True)

    print(f'\n=== Group-Factorized CLIPConceptHead + Differential Top-K Arbitration (feat={opts.feat}) ===')
    hdr = (
        f'{"Variant":<14} {"Split":<6} {"Best":>7} {"Last10":>7} {"C-Only":>7} '
        f'{"HeadAUC":>8} {"SeenAUC":>8} {"UnseenC":>8}  Differential Top-3 AlphaSweep (a=0.00 is Flora)'
    )
    print(hdr)
    print('-' * (len(hdr) + 12))
    for variant in opts.variants:
        for t in opts.tasks:
            task = C.SPLITS[t][0]
            for s in opts.seeds:
                run_dir = os.path.join(opts.out_root, opts.feat, variant, f'{task}_{opts.setting}_seed{s}')
                res = parse_gf_log(run_dir)
                if not res:
                    print(f'{variant:<14} {t:<6}   (incomplete)')
                else:
                    sw = ' '.join(f'{k}:{v:.2f}' for k, v in sorted(res['tags'].items()))
                    print(
                        f'{variant:<14} {t:<6} '
                        f'{res["best"]:7.2f} {res["last10"]:7.2f} {res["c_only_zsl"]:7.2f} '
                        f'{res["head_auc"]:8.4f} {res["seen_auc"]:8.4f} {res["unseen_auc"]:8.4f}  {sw}'
                    )


if __name__ == '__main__':
    main()
