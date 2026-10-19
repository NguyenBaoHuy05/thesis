"""Print one markdown table of the concept runs (kg_metrics.json of each work dir), to paste into the progress md.

  python tools/summarize_runs.py <W> [e1a e1b_v2 e1c ...]     (default: every sub-folder of W with kg_metrics.json)
"""

import json
import os
import sys


def pct(x):
    return '' if x is None else f'{100 * x:.2f}'


if __name__ == '__main__':
    root = sys.argv[1]
    names = sys.argv[2:] or sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, 'kg_metrics.json')))
    print('| run | Top1 / Top5 | PCR | Top1 in / out pairs | fixed / broken vs K0 | K5 support / random | K5-pair '
          '| fusion Top1 / Top5 | fusion fixed / broken | fusion PCR |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    k0 = None
    for name in names:
        path = os.path.join(root, name, 'kg_metrics.json')
        if not os.path.isfile(path):
            print(f'| {name} | missing | | | | | | | | |')
            continue
        with open(path, encoding='utf-8') as handle:
            m = json.load(handle)
        h, p, k5 = m['head'], m['pairs'], m['k5']
        fu = m.get('fusion')
        fus = (f"{pct(fu['topk']['recall@1'])} / {pct(fu['topk']['recall@5'])} | "
               f"{fu['vs_k0']['fixed']} / {fu['vs_k0']['broken']} | {pct(fu['pairs']['pcr'])}") if fu else ' | | '
        print(f"| {name} | {pct(h['recall@1'])} / {pct(h['recall@5'])} | {pct(p['pcr'])} | "
              f"{pct(p['top1_in_pairs'])} / {pct(p['top1_outside_pairs'])} | "
              f"{m['vs_k0']['fixed']} / {m['vs_k0']['broken']} | {pct(k5['drop_support'])} / {pct(k5['drop_random_absent'])} | "
              f"{pct(m['k5_pair']['moved_to_partner'])} | {fus} |")
        k0 = k0 or (m['k0_fc_logits'], m['pairs_k0'])
    if k0:
        print(f"| K0 | {pct(k0[0]['recall@1'])} / {pct(k0[0]['recall@5'])} | {pct(k0[1]['pcr'])} | "
              f"{pct(k0[1]['top1_in_pairs'])} / {pct(k0[1]['top1_outside_pairs'])} | | | | | | |")
