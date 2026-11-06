"""Confusable-pair metrics of plan §2: PCR (pair confusion rate), per-pair rates, Top1 outside the pairs, K5-pair.

Pairs come from model/concept_graph.py: confusable_pairs (only M and class names, never the test split).
"""

import torch


def partner_matrix(pairs, num_classes):
    """[A, A] bool: j is a partner of i."""
    P = torch.zeros(num_classes, num_classes, dtype=torch.bool)
    for i, j in pairs:
        P[i, j] = P[j, i] = True
    return P


def pair_metrics(pred, labels, pairs, actions):
    """PCR over test samples whose label has a partner; per-pair rate (samples of i or j predicted as the
    other one, over samples of i or j); Top1 on the samples whose label has no partner."""
    P = partner_matrix(pairs, len(actions))
    in_pair = P[labels].any(1)
    confused = P[labels, pred]
    per_pair = []
    for i, j in pairs:
        both = (labels == i) | (labels == j)
        swaps = ((labels == i) & (pred == j)) | ((labels == j) & (pred == i))
        per_pair.append(dict(pair=[actions[i], actions[j]], samples=int(both.sum()),
                             rate=float(swaps.sum()) / max(int(both.sum()), 1)))
    out_pair = ~in_pair
    return dict(pcr=float(confused[in_pair].float().mean()), pcr_samples=int(in_pair.sum()),
                top1_in_pairs=float((pred == labels)[in_pair].float().mean()),
                top1_outside_pairs=float((pred == labels)[out_pair].float().mean()) if out_pair.any() else None,
                per_pair=sorted(per_pair, key=lambda r: -r['rate']))


@torch.no_grad()
def k5_pair(logits_fn, c, rows, pairs, batch=2048):
    """For test samples predicted i that have a partner j (the partner with the largest logit): switch on the
    concepts j has and i lacks (c <- 1 on M_j minus M_i) and count predictions that move to j.
    c [N, K] on the CPU; logits_fn: CPU c [n, K] -> logits [n, A]; rows: M [A, K] binary (CPU)."""
    A = rows.shape[0]
    P = partner_matrix(pairs, A)
    rows = rows.bool()
    moved, total = 0, 0
    for s in range(0, len(c), batch):
        cb = c[s:s + batch]
        logits = logits_fn(cb).cpu()
        pred = logits.argmax(1)
        partner_logits = logits.masked_fill(~P[pred], float('-inf'))
        j = partner_logits.argmax(1)
        diff = rows[j] & ~rows[pred]                                  # [n, K]
        keep = P[pred].any(1) & diff.any(1)
        if not keep.any():
            continue
        c2 = torch.where(diff[keep], torch.ones_like(cb[keep]), cb[keep])
        new = logits_fn(c2).cpu().argmax(1)
        moved += int((new == j[keep]).sum())
        total += int(keep.sum())
    return dict(moved_to_partner=moved / max(total, 1), samples=total)
