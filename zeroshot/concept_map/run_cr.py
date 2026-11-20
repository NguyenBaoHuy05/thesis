"""Runner for Concept-Augmented Motion Mapping (Cách A — Direction 1).

Tests motion-side concept fusion directly inside Flora's Phase-1 & Phase-2 (no post-hoc re-ranking):
  - ca_concat_soft : Concat [ske_256 || gamma * soft_proto(p_hat)] into motion_encoder/decoder (gamma=0.5)
  - ca_concat_q    : Concat [ske_256 || gamma * q_clip] into motion_encoder/decoder (gamma=0.5)
  - ca_concat_s1   : Concat soft_proto with stronger gamma=1.0
  - ca_lat_soft    : Residual skip mu_ske += gamma * soft_proto in latent space (gamma=0.2)
  - ca_lat_q       : Residual skip mu_ske += gamma * q_clip in latent space (gamma=0.2)

Usage (Colab):
  !python zeroshot/concept_map/run_cr.py --feat star --tasks 48 110 --seeds 1 --force
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
    'flora_base':     dict(fusion_mode='none',       concept_rep='soft_proto', gamma_c=0.0),
    'ca_concat_soft': dict(fusion_mode='concat',     concept_rep='soft_proto', gamma_c=0.5),
    'ca_concat_q':    dict(fusion_mode='concat',     concept_rep='q_clip',     gamma_c=0.5),
    'ca_concat_s1':   dict(fusion_mode='concat',     concept_rep='soft_proto', gamma_c=1.0),
    'ca_lat_soft':    dict(fusion_mode='latent_add', concept_rep='soft_proto', gamma_c=0.2),
    'ca_lat_q':       dict(fusion_mode='latent_add', concept_rep='q_clip',     gamma_c=0.2),
}

CA_EVAL_RE = re.compile(
    r'\[CA-EVAL\] Unseen Concept Head AUC=([\d.nan]+) \(AP=([\d.nan]+), n_prims=(\d+)\) \| '
    r'Seen-cov AUC=([\d.nan]+) \(\d+\), Unseen-only AUC=([\d.nan]+) \(\d+\) \| '
    r'Concept-only ZSL=([\d.nan]+)%'
)


def parse_ca_log(run_dir):
    acc_list, _ = C.read_log(run_dir)
    if not acc_list:
        return None
    text = open(os.path.join(run_dir, 'run.log'), encoding='utf-8').read()
    m = CA_EVAL_RE.search(text)
    return {
        'best': max(acc_list),
        'last10': sum(acc_list[-10:]) / min(len(acc_list), 10),
        'head_auc': float(m.group(1)) if m else float('nan'),
        'head_ap': float(m.group(2)) if m else float('nan'),
        'unseen_auc': float(m.group(5)) if m else float('nan'),
        'c_only_zsl': float(m.group(6)) if m else float('nan'),
    }


def run_single(opts):
    os.chdir(FLORA_DIR)
    C.set_feat(opts.feat)
    import utils.utils as U
    from trainer_cr import ConceptAugmentedTrainer

    task = C.SPLITS[opts.task][0]
    prims, rules = C.kg_for_task(task, opts.concept_prims, opts.concept_rules)
    vcfg = dict(VARIANTS[opts.variant])
    if opts.gamma_c is not None:
        vcfg['gamma_c'] = opts.gamma_c
    vcfg['concept_prims'] = prims
    vcfg['concept_rules'] = rules
    vcfg['concept_iters'] = opts.concept_iters
    vcfg['concept_lr'] = opts.concept_lr

    out_dir = os.path.join(opts.out_root, opts.feat, opts.variant, f'{task}_{opts.setting}_seed{opts.seed}')
    os.makedirs(out_dir, exist_ok=True)
    if C.run_done(out_dir, opts.setting) and not opts.force:
        print(f'[skip] {out_dir}')
        return

    flora_ov = dict(kg_mode='none', setting=opts.setting)
    args = C.build_args(task, opts.seed, out_dir, flora_overrides=flora_ov, extra=vcfg)
    U.init_seed(args.seed)
    ConceptAugmentedTrainer(args).train()


def main():
    ap = argparse.ArgumentParser(description='Concept-Augmented Motion Mapping (Cách A) runner')
    ap.add_argument('--feat', choices=['star', 'synse', 'sadave'], default='star')
    ap.add_argument('--setting', choices=['ZSL', 'GZSL'], default='ZSL')
    ap.add_argument('--tasks', nargs='+', default=['48', '110'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--variants', nargs='+',
                    default=['ca_concat_soft', 'ca_concat_q', 'ca_lat_soft', 'ca_lat_q'])
    ap.add_argument('--out-root', default=DRIVE_OUT_CR if os.path.isdir('/content/drive/MyDrive')
                    else os.path.join(CMAP_DIR, 'runs'))
    ap.add_argument('--concept-prims', default='kg/primitive_v2.json')
    ap.add_argument('--concept-rules', default='kg/kg_rules_v2.json')
    ap.add_argument('--concept-iters', type=int, default=400)
    ap.add_argument('--concept-lr', type=float, default=1e-3)
    ap.add_argument('--gamma-c', type=float, default=None)
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
                    '--concept-iters', str(opts.concept_iters),
                    '--concept-lr', str(opts.concept_lr),
                ]
                if opts.gamma_c is not None:
                    cmd += ['--gamma-c', str(opts.gamma_c)]
                if opts.force:
                    cmd.append('--force')
                print(f'\n[run] {variant} | split={t} | seed={s}')
                subprocess.run(cmd, check=True)

    print(f'\n=== Concept-Augmented Motion Mapping Summary (feat={opts.feat}, setting={opts.setting}) ===')
    hdr = f'{"Variant":<16} {"Split":<6} {"Best":>7} {"Last10":>7} {"C-Only":>7} {"HeadAUC":>8} {"HeadAP":>7} {"UnseenC":>8}'
    print(hdr)
    print('-' * len(hdr))
    for variant in opts.variants:
        for t in opts.tasks:
            task = C.SPLITS[t][0]
            for s in opts.seeds:
                run_dir = os.path.join(opts.out_root, opts.feat, variant, f'{task}_{opts.setting}_seed{s}')
                res = parse_ca_log(run_dir)
                if not res:
                    print(f'{variant:<16} {t:<6}   (incomplete)')
                else:
                    print(
                        f'{variant:<16} {t:<6} '
                        f'{res["best"]:7.2f} {res["last10"]:7.2f} {res["c_only_zsl"]:7.2f} '
                        f'{res["head_auc"]:8.4f} {res["head_ap"]:7.4f} {res["unseen_auc"]:8.4f}'
                    )


if __name__ == '__main__':
    main()
