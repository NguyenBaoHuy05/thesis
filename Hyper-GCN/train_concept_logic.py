"""Train / evaluate a concept head on cached l9 joint features.

  python train_concept_logic.py --config config/concept/cl_full.yaml [--cache-dir <folder>]
      [--work-dir <folder>] [--matrix <csv>] [--seed 0] [--device cuda]

config `head`:
- cl     (default): flat soft AND over M, model/concept_logic.py (CL v1 / v2);
- graph: concept-graph head, model/concept_graph_head.py, on the graph of config `graph`
         (kind full | flat | random, nots, seed; model/concept_graph.py), plan KG_RETRIEVAL_PLAN/CONCEPT_GRAPH;
- reason: REASON's logic layers on our concept decoder (model/reason_logic.py, baseline E1b);
- full:   the whole proposed model (model/concept_action_graph.py): concept + action decoders, graph, stratified
          R-GCN + fuzzy executor, logits = log s / tau + lam z; also reports the symbolic path alone (head_sym,
          lam = 0), the R-GCN path alone (head_neural) and the action decoder's auxiliary head (head_aux).

Reads <cache_dir>/<cache_name>_{train,test}_feat.npy + .npz (tools/cache_joint_features.py).
No validation split: the recipe is fixed in the config, test is scored once.

Reported (kg_metrics.json in work_dir):
- head Top1/Top5 against K0 (cached fc_logits), samples fixed / broken against K0;
- concept AUC (macro over concepts) of c against the class row of the matrix;
- K5 on the test split: zero the concept that supports the predicted class most (has, largest c) and
  count changed predictions, against zeroing a random concept the predicted class does not have;
- confusable pairs (model/pair_metrics.py, for every head and for K0): PCR, per-pair rate, Top1 inside /
  outside the pairs, K5-pair (switch on M_j minus M_i, count predictions that move to j);
- late fusion with K0: log_softmax(fc_logits) + fusion_lambda * log_softmax(head logits), fusion_lambda fixed in
  the config (default 1, never tuned on test; train cannot tune it, the head reaches ~100% there): Top1 / Top5,
  pairs, fixed / broken against K0. Answers whether the concept head corrects K0 more than it breaks it;
- fusion_sweep: the same fusion for several lambda, chosen on test. DIAGNOSTIC ONLY (never a claimed number): its
  best Top1 is the ceiling of late fusion with this head. Test logits of the head are saved (test_logits.npy).
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from model.concept_graph import build_graph, confusable_pairs, flat_graph, random_pair_graph, read_matrix
from model.concept_action_graph import ConceptActionGraphHead
from model.concept_graph_head import ConceptGraphHead
from model.concept_logic import ConceptLogicHead, concept_joint_mask, concept_loss, load_concepts
from model.concept_schema import NTU_ACTION_NAMES
from model.pair_metrics import k5_pair, pair_metrics
from model.reason_logic import ReasonLogicHead

EVAL_BATCH = 512
SWEEP = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)          # fusion_sweep: diagnostic, lambda chosen on test


def topk_metrics(scores, labels, ks=(1, 5)):
    """Fraction of samples whose label is in the top-k scores (recall@1 = Top1)."""
    ranked = scores.topk(max(ks), dim=1).indices
    hit = ranked == labels.unsqueeze(1)
    return {f'recall@{k}': hit[:, :k].any(1).float().mean().item() for k in ks}


def load_split(cache_dir, name, split, device=None, chunk=4096):
    """feat [N, M, C, S, V] float16 (on `device` if given, else a CPU memmap), valid, fc_logits, label."""
    feat = np.load(os.path.join(cache_dir, f'{name}_{split}_feat.npy'), mmap_mode='r')
    with np.load(os.path.join(cache_dir, f'{name}_{split}.npz')) as data:
        rest = {key: torch.from_numpy(data[key]) for key in ('valid', 'fc_logits', 'label')}
    if device is not None:
        out = torch.empty(feat.shape, dtype=torch.float16, device=device)
        for i in range(0, len(feat), chunk):
            out[i:i + chunk] = torch.from_numpy(np.array(feat[i:i + chunk])).to(device)
        feat = out
    return feat, rest['valid'].bool(), rest['fc_logits'].float(), rest['label'].long()


def batches(feat, idx, device):
    block = feat[idx] if torch.is_tensor(feat) else torch.from_numpy(np.array(feat[idx.numpy()]))
    return block.to(device).float()


def channel_std(feat, valid, chunk=2048):
    """Per-channel std of the train features over real actors, segments and joints: [C]."""
    total = torch.zeros(feat.shape[2], dtype=torch.float64, device=feat.device)
    total_sq, count = torch.zeros_like(total), 0
    for i in range(0, len(feat), chunk):
        x = feat[i:i + chunk].float()                                          # [n, M, C, S, V]
        w = valid[i:i + chunk].to(x.device)[:, :, None, None, None].float()
        total += (x * w).sum((0, 1, 3, 4)).double()
        total_sq += (x * x * w).sum((0, 1, 3, 4)).double()
        count += w.sum().item() * x.shape[3] * x.shape[4]
    mean = total / count
    return (total_sq / count - mean ** 2).clamp_min(0).sqrt().float()


def concept_auc(c, target):
    """Macro AUC over concepts (columns with both classes present); c, target [N, K]."""
    aucs = []
    for j in range(c.shape[1]):
        t = target[:, j]
        n_pos = int(t.sum())
        n_neg = len(t) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = c[:, j].argsort().argsort().double() + 1
        aucs.append(((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())
    return float(np.mean(aucs)), aucs


@torch.no_grad()
def predict(head, feat, valid, device):
    """-> c [N, K], logits [N, A], {extra output: [N, A]} (head.forward_outputs(), e.g. logits_sym, z)."""
    head.eval()
    names = head.forward_outputs() if hasattr(head, 'forward_outputs') else ()
    cs, logits, extra = [], [], {k: [] for k in names}
    for i in range(0, len(valid), EVAL_BATCH):
        idx = torch.arange(i, min(i + EVAL_BATCH, len(valid)))
        out = head(batches(feat, idx, device), valid[idx].to(device))
        cs.append(out['c'].cpu())
        logits.append(out['logits'].cpu())
        for k in names:
            extra[k].append(out[k].float().cpu())
    return torch.cat(cs), torch.cat(logits), {k: torch.cat(v) for k, v in extra.items()}


def fusion_metrics(fc_logits, head_logits, labels, pairs, actions, lam=1.0):
    """Late fusion log_softmax(K0) + lam * log_softmax(head) (plan: concept head as a correction of K0)."""
    fused = torch.log_softmax(fc_logits.float(), 1) + lam * torch.log_softmax(head_logits.float(), 1)
    pred, k0 = fused.argmax(1), fc_logits.argmax(1)
    return dict(lam=lam, topk=topk_metrics(fused, labels, ks=(1, 5)),
                vs_k0=dict(fixed=int(((pred == labels) & (k0 != labels)).sum()),
                           broken=int(((pred != labels) & (k0 == labels)).sum())),
                pairs={k: v for k, v in pair_metrics(pred, labels, pairs, actions).items() if k != 'per_pair'})


def make_graph(matrix, spec):
    """Concept graph of config `graph`: kind full (plan §5) | flat (no motif / OR) | random (E1e: NOT nodes on
    random pairs, built from the flat graph); nots False drops the NOT(j|i) nodes."""
    actions, concepts, rows = read_matrix(matrix)
    kind = spec.get('kind', 'full')
    if kind == 'full':
        graph = build_graph(actions, concepts, rows, spec.get('cfg'))
    elif kind in ('flat', 'random'):
        graph = flat_graph(actions, concepts, rows)
        if kind == 'random':
            graph = random_pair_graph(graph, seed=spec.get('seed', 0))
    else:
        raise ValueError(f'unknown graph kind {kind}')
    if not spec.get('nots', True):
        graph.nots = []
    return graph


@torch.no_grad()
def batched_logits(head, c, batch=EVAL_BATCH):
    """head.logits on CPU c [N, K] in batches (REASON's logic layers hold [n, 512, 128] per batch) -> CPU."""
    head.eval()
    device = head.M.device
    return torch.cat([head.logits(c[i:i + batch].to(device)).cpu() for i in range(0, len(c), batch)])


@torch.no_grad()
def intervention(head, c, generator):
    """K5: change rate of the prediction when the strongest supporting concept of the predicted class
    is set to 0, against setting a random concept outside its has-set to 0."""
    M = head.M.cpu()
    pred = batched_logits(head, c).argmax(1)
    rows = torch.arange(len(c))
    has = M[pred].bool()
    support = torch.where(has, c, torch.full_like(c, -1)).argmax(1)
    weights = (~has).float()
    other = torch.multinomial(weights, 1, generator=generator).squeeze(1)
    out = {}
    for name, j in (('drop_support', support), ('drop_random_absent', other)):
        c2 = c.clone()
        c2[rows, j] = 0.0
        out[name] = (batched_logits(head, c2).argmax(1) != pred).float().mean().item()
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--work-dir')
    parser.add_argument('--matrix', help='overrides matrix')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--feature-noise', type=float, help='overrides feature_noise')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval-only', action='store_true', help='load <work_dir>/head.pt, skip training')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    for key in ('cache_dir', 'work_dir', 'matrix', 'seed', 'feature_noise'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    os.makedirs(config['work_dir'], exist_ok=True)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    device = torch.device(args.device)

    M, names, meta = load_concepts(config['matrix'], config['metadata'], NTU_ACTION_NAMES)
    joint_mask = concept_joint_mask(names, meta)

    feat_train, valid_train, _, y_train = load_split(config['cache_dir'], config['cache_name'], 'train', device)
    feat_test, valid_test, fc_test, y_test = load_split(config['cache_dir'], config['cache_name'], 'test')
    N, n_person, C, S, V = feat_train.shape
    print(f'train feat {tuple(feat_train.shape)}, test feat {tuple(feat_test.shape)}, '
          f'{len(names)} concepts from {config["matrix"]}')

    head_args = dict(config.get('head_args', {}))
    decoder_args = dict(dict(num_person=n_person, max_frames=S), **head_args.pop('decoder_args', {}))
    kind = config.get('head', 'cl')
    graph = None
    if kind == 'cl':
        head = ConceptLogicHead(C, M, joint_mask, decoder_args=decoder_args, **head_args)
    elif kind == 'graph':
        graph = make_graph(config['matrix'], config.get('graph', {}))
        graph.save(os.path.join(config['work_dir'], 'graph.json'))
        print(f'graph: {graph.summary()}')
        head = ConceptGraphHead(C, graph, joint_mask, decoder_args=decoder_args, **head_args)
        if not torch.equal(head.M, M):
            raise ValueError('graph rows differ from the concept matrix')
    elif kind == 'full':
        graph = make_graph(config['matrix'], config.get('graph', {}))
        graph.save(os.path.join(config['work_dir'], 'graph.json'))
        print(f'graph: {graph.summary()}')
        action_args = head_args.pop('action_decoder_args', {})
        head = ConceptActionGraphHead(C, graph, joint_mask, decoder_args=decoder_args,
                                      action_decoder_args=action_args, **head_args)
        if not torch.equal(head.M, M):
            raise ValueError('graph rows differ from the concept matrix')
    elif kind == 'reason':
        head = ReasonLogicHead(C, M, joint_mask, decoder_args=decoder_args, **head_args)
    else:
        raise ValueError(f'unknown head {kind}')
    head = head.to(device)
    wd = config.get('weight_decay', 0.0)
    groups = head.param_groups(config['lr'], wd) if hasattr(head, 'param_groups') else head.parameters()
    optimizer = torch.optim.Adam(groups, lr=config['lr'], weight_decay=wd)
    steps = config['epochs'] * ((N + config['batch_size'] - 1) // config['batch_size'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    M_dev, y_train_dev, valid_train_dev = head.M, y_train.to(device), valid_train.to(device)
    # Gaussian noise on the train features, in units of each channel's std (train only): the head
    # reaches 100% train top1 on clean cached features.
    noise = config.get('feature_noise', 0.0)
    std = channel_std(feat_train, valid_train_dev)[None, None, :, None, None] if noise else None

    history = []
    head_path = os.path.join(config['work_dir'], 'head.pt')
    if args.eval_only:
        head.load_state_dict(torch.load(head_path, map_location=device))
        print(f'loaded {head_path}, no training')
    for epoch in range(0 if args.eval_only else config['epochs']):
        head.train()
        order = torch.randperm(N, device=device)
        sums, correct = dict(loss=0.0, ce=0.0, bce=0.0), 0
        for start in range(0, N, config['batch_size']):
            idx = order[start:start + config['batch_size']]
            x = feat_train[idx].float()
            if noise:
                x = x + noise * std * torch.randn_like(x)
            out = head(x, valid_train_dev[idx])
            loss, terms = concept_loss(out, y_train_dev[idx], M_dev, beta=config['loss']['beta'])
            if hasattr(head, 'regularizer'):
                loss = loss + head.regularizer()
            if hasattr(head, 'extra_loss'):
                extra_loss, extra_terms = head.extra_loss(out, y_train_dev[idx])
                loss = loss + extra_loss
                terms.update(extra_terms)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if hasattr(head, 'clip_grads'):
                head.clip_grads()
            optimizer.step()
            scheduler.step()
            if hasattr(head, 'after_step'):
                head.after_step()
            sums['loss'] += loss.item() * len(idx)
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + value * len(idx)
            correct += (out['logits'].argmax(1) == y_train_dev[idx]).sum().item()
        row = dict(epoch=epoch + 1, **{k: v / N for k, v in sums.items()}, train_top1=correct / N,
                   tau=head.tau.item())
        history.append(row)
        print(f"epoch {epoch + 1}/{config['epochs']}: loss {row['loss']:.4f} (ce {row['ce']:.4f}, "
              f"bce {row['bce']:.4f}" + ''.join(f", {k} {row[k]:.4f}" for k in sums if k not in ('loss', 'ce', 'bce'))
              + f"), train top1 {100 * row['train_top1']:.2f}%, tau {row['tau']:.4f}")

    if not args.eval_only:
        torch.save(head.state_dict(), head_path)                # before evaluation: a crash there keeps the model
    c_test, logits, extra = predict(head, feat_test, valid_test, device)
    pred, k0 = logits.argmax(1), fc_test.argmax(1)
    actions, _, rows = read_matrix(config['matrix'])
    pairs = confusable_pairs(actions, rows)                 # evaluation pairs: always the real ones (plan §2)

    def logits_fn(c):
        return batched_logits(head, c)
    auc_own, aucs_own = concept_auc(c_test, M[y_test])
    groups = {}
    for name, a in zip(names, aucs_own):
        groups.setdefault(meta[name]['group'], []).append(a)
    metrics = dict(
        config=config,
        head=topk_metrics(logits, y_test, ks=(1, 5)), k0_fc_logits=topk_metrics(fc_test, y_test, ks=(1, 5)),
        vs_k0=dict(fixed=int(((pred == y_test) & (k0 != y_test)).sum()),
                   broken=int(((pred != y_test) & (k0 == y_test)).sum())),
        concept_auc=auc_own,
        concept_auc_by_group={g: float(np.mean(v)) for g, v in groups.items()},
        concepts_auc_below_0_6=[n for n, a in zip(names, aucs_own) if a < 0.6],
        k5=intervention(head, c_test, torch.Generator().manual_seed(config['seed'])),
        pairs=pair_metrics(pred, y_test, pairs, actions),
        pairs_k0=pair_metrics(k0, y_test, pairs, actions),
        fusion=fusion_metrics(fc_test, logits, y_test, pairs, actions, config.get('fusion_lambda', 1.0)),
        fusion_sweep={str(lam): fusion_metrics(fc_test, logits, y_test, pairs, actions, lam)['topk']['recall@1']
                      for lam in SWEEP},
        k5_pair=k5_pair(logits_fn, c_test, M.cpu(), pairs),
        graph=graph.summary() if graph is not None else None,
        tau=head.tau.item(),
    )
    for key, name in (('logits_sym', 'head_sym'), ('z', 'head_neural'), ('aux', 'head_aux')):
        if key in extra:
            metrics[name] = topk_metrics(extra[key], y_test, ks=(1, 5))
            metrics[f'pairs_{name}'] = {k: v for k, v in pair_metrics(extra[key].argmax(1), y_test, pairs, actions).items()
                                        if k != 'per_pair'}
    np.save(os.path.join(config['work_dir'], 'test_concepts.npy'), c_test.numpy().astype(np.float16))
    np.save(os.path.join(config['work_dir'], 'test_logits.npy'), logits.numpy().astype(np.float32))
    if history:
        with open(os.path.join(config['work_dir'], 'history.json'), 'w', encoding='utf-8') as handle:
            json.dump(history, handle, indent=2)
    with open(os.path.join(config['work_dir'], 'kg_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    for key in ('head', 'head_sym', 'head_neural', 'head_aux', 'k0_fc_logits'):
        if key in metrics:
            extra_pcr = f", PCR {100 * metrics['pairs_' + key]['pcr']:.2f}%" if 'pairs_' + key in metrics else ''
            print(f'{key:>13}: ' + ', '.join(f'{k} {100 * v:.2f}%' for k, v in metrics[key].items()) + extra_pcr)
    print(f"fixed / broken vs K0: {metrics['vs_k0']['fixed']} / {metrics['vs_k0']['broken']}")
    print(f"concept AUC {auc_own:.3f}; by group: "
          + ', '.join(f'{g} {a:.3f}' for g, a in metrics['concept_auc_by_group'].items()))
    print(f"{len(metrics['concepts_auc_below_0_6'])} concepts with AUC < 0.6: "
          + ', '.join(metrics['concepts_auc_below_0_6']))
    print(f"K5: prediction changes when the strongest supporting concept is zeroed "
          f"{100 * metrics['k5']['drop_support']:.2f}%, a random absent concept "
          f"{100 * metrics['k5']['drop_random_absent']:.2f}%")
    fu = metrics['fusion']
    print(f"fusion K0 + {fu['lam']} head: " + ', '.join(f'{k} {100 * v:.2f}%' for k, v in fu['topk'].items())
          + f", fixed / broken vs K0 {fu['vs_k0']['fixed']} / {fu['vs_k0']['broken']}, PCR {100 * fu['pairs']['pcr']:.2f}%")
    best = max(metrics['fusion_sweep'], key=metrics['fusion_sweep'].get)
    print('fusion sweep (DIAGNOSTIC, lambda chosen on test): '
          + ', '.join(f'{lam}: {100 * v:.2f}%' for lam, v in metrics['fusion_sweep'].items())
          + f' -> ceiling {100 * metrics["fusion_sweep"][best]:.2f}% at lambda {best} (K0 {100 * metrics["k0_fc_logits"]["recall@1"]:.2f}%)')
    for key in ('pairs', 'pairs_k0'):
        pm = metrics[key]
        print(f"{key:>8}: PCR {100 * pm['pcr']:.2f}% on {pm['pcr_samples']} samples, top1 in pairs "
              f"{100 * pm['top1_in_pairs']:.2f}%, outside {100 * pm['top1_outside_pairs']:.2f}%")
    print('most confused pairs: ' + '; '.join(f"{a}/{b} {100 * r['rate']:.1f}%"
                                              for r in metrics['pairs']['per_pair'][:8] for a, b in [r['pair']]))
    print(f"K5-pair: {100 * metrics['k5_pair']['moved_to_partner']:.2f}% move to the partner "
          f"({metrics['k5_pair']['samples']} samples)")
