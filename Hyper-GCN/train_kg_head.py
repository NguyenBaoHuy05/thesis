"""Train/evaluate a KG head on cached Hyper-GCN features (A1 plan S1.1/S1.2, S4).

  python train_kg_head.py --config config/kg/a1.yaml [--cache-dir <folder>] [--work-dir <folder>]
      [--kg annotations/kg_shuffled/action_kg_shuffle_s0.json] [--embeddings <npz>] [--seed 1]
      [--device cuda]

Reads <cache_dir>/<cache_name>_{train,test}.npz from tools/cache_features.py. Heads:

  k1  RETRIEVE only (text embeddings, no KG edges); loss CE(s, y).
  g1  RETRIEVE only on the graph (A1-G: s_c = cos(P z, g_c) / tau, g from the R-GCN over the KG,
      model/graph_encoder.py); head_args go to GraphRetrievalScorer; loss CE(s, y). No embeddings
      file needed (unless head_args.init = text).
  a1  RETRIEVE -> VERIFY -> FUSE (model/kg_retrieval.py: KGRetrievalHead); loss
        CE_cand(logits, y) + alpha CE_all(s, y) + beta BCE_masked(a, y_attr[y])
        + gamma mean|f| / sg(mean(|s_c| + |lambda v|))
      The ground truth is forced into the candidates while training only. sg = stop-gradient:
      without it the gamma term could shrink by inflating s (lowering tau) instead of f.
      K2 = the same config with --kg pointing at a shuffled KG (same embeddings file).
      A1-G configs set head_args.retrieval_source: graph (graph_args), verify_mode, normalize_verify
      and, for attribute_source: predicate, a `predicates` block:
          predicates: {cache_name: ntu120_xsub_pred, names: all | [list], learn_threshold: True}
      phi comes from <cache_dir>/<cache_name>_{train,test}.npz (tools/cache_predicates.py), one
      deterministic view per split; its labels must match the feature cache row by row. Only
      predicates whose name is a KG attribute are used; mean / std come from the train split.

Training draws, for every sample of every batch, one of the cached random_rot views. There is
no validation split (every train subject was seen by the backbone), so the recipe is fixed in
the config and the test split is evaluated once, after the last epoch. The train split is also
scored (view 0, eval mode) to show the train/test gap. Paths are relative to the current
folder, as for main.py: run from the repo root.

Writes to work_dir: head.pt, history.json (per epoch), kg_metrics.json (head vs K0 = the cached
fc_logits; for a1 also lambda, logit-term magnitudes, prediction flips, attribute AUC),
test_scores.npy (float16 [N, num_class]; for a1 the logits, -inf outside the candidates).
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.graph_encoder import GraphRetrievalScorer
from model.kg import kg_tensors, load_embeddings, load_kg
from model.kg_retrieval import KGRetrievalHead, RetrievalScorer, a1_loss, topk_metrics
from model.kg_retrieval import masked_bce as kg_masked_bce

RECALL_TARGET = ('recall@10', 0.98)  # P2 pass criterion, on the retrieval scores s
EVAL_BATCH = 1024                    # FUSE builds [N, C + A, d_model] node tensors


def load_cache(cache_dir, name, split):
    with np.load(os.path.join(cache_dir, f'{name}_{split}.npz')) as data:
        z = torch.from_numpy(data['z'])
        label = torch.from_numpy(data['label']).long()
        fc_logits = torch.from_numpy(data['fc_logits']).float()
    if z.dim() == 2:  # single view: add the view axis
        z, fc_logits = z.unsqueeze(0), fc_logits.unsqueeze(0)
    return z, label, fc_logits


def load_predicates(cache_dir, name, split, kg_json, wanted='all'):
    """phi / valid [N, P] of tools/cache_predicates.py, restricted to predicates that are KG
    attributes (and to `wanted` unless 'all'). Returns (phi, valid, label, names, KG columns)."""
    with np.load(os.path.join(cache_dir, f'{name}_{split}.npz')) as data:
        phi, valid, label = data['phi'], data['valid'], data['label']
        names = [str(n) for n in data['names']]
    attributes = list(kg_json['attributes'])
    if wanted != 'all':
        missing = [n for n in wanted if n not in names]
        if missing:
            raise ValueError(f'{name}_{split}.npz has no predicate {missing}')
    keep = [j for j, n in enumerate(names) if n in attributes and (wanted == 'all' or n in wanted)]
    if not keep:
        raise ValueError('no cached predicate is an attribute of this KG')
    names = [names[j] for j in keep]
    return (torch.from_numpy(phi[:, keep]).float(), torch.from_numpy(valid[:, keep]).bool(),
            torch.from_numpy(label).long(), names, [attributes.index(n) for n in names])


def predicate_train_auc(phi, valid, labels, kg, columns):
    """AUC of each predicate against has (1) / lacks (0) of the sample's class, on valid cells the
    KG knows, train split only (the selection never sees the test split). nan if one side is empty."""
    aucs = []
    for j, column in enumerate(columns):
        mask = valid[:, j] & kg.known[labels, column].bool()
        target, score = (kg.W[labels[mask], column] > 0).float(), phi[mask, j]
        n_pos = target.sum().item()
        n_neg = len(target) - n_pos
        if n_pos == 0 or n_neg == 0:
            aucs.append(float('nan'))
            continue
        ranks = score.argsort().argsort().double() + 1
        aucs.append(((ranks[target == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())
    return aucs


def predicate_stats(phi, valid):
    """Per-predicate mean / std over the valid train values."""
    w = valid.float()
    count = w.sum(0).clamp_min(1)
    mean = (phi * w).sum(0) / count
    std = (((phi - mean) ** 2 * w).sum(0) / count).sqrt()
    return mean, std


def build_head(config, in_channels, class_embeddings, attribute_embeddings, kg, predicate_args=None):
    args = dict(config.get('head_args', {}))
    if config['head'] == 'k1':
        return RetrievalScorer(in_channels, class_embeddings, **args)
    if config['head'] == 'g1':
        if args.get('init') == 'text':
            args['text_embeddings'] = (class_embeddings, attribute_embeddings)
        return GraphRetrievalScorer(in_channels, kg, **args)
    if config['head'] == 'a1':
        if predicate_args is not None:
            args['predicate_args'] = predicate_args
        return KGRetrievalHead(in_channels, class_embeddings, attribute_embeddings, kg, **args)
    raise ValueError(f"unknown head {config['head']!r}; expected 'k1', 'g1' or 'a1'")


def needs_text(config):
    """Whether the head reads the text embeddings file."""
    args = config.get('head_args', {})
    if config['head'] == 'k1':
        return True
    if config['head'] == 'g1':
        return args.get('init') == 'text'
    graph = args.get('retrieval_source', 'text') == 'graph'
    return (not graph or (args.get('graph_args') or {}).get('init') == 'text'
            or args.get('attribute_source') == 'decoder')


def masked_bce(a, labels, kg):
    """BCE of the attribute head on the cells the KG knows for the sample's class."""
    return kg_masked_bce(a, labels, kg.y_attr, kg.known)


def loss_terms(config, head, z, labels, kg, extra=None):
    """Returns (total loss, logits used for train accuracy, {term: value}). extra: keyword
    tensors for the head (phi, phi_valid), already indexed to the batch."""
    if config['head'] in ('k1', 'g1'):
        s = head(z)
        loss = F.cross_entropy(s, labels)
        return loss, s, dict(ce_all=loss.item())

    out = head(z, labels=labels, **(extra or {}))
    loss, terms = a1_loss(out, labels, head, **config.get('loss', {}))
    return loss, out['logits'], terms


def train(config, head, z, label, kg, device, extra=None):
    """z: [V, N, C] float16 on device; extra: {name: [N, ...] tensor} per sample (one view).
    Returns the per-epoch history."""
    views, n, _ = z.shape
    epochs, batch_size = config['epochs'], config['batch_size']
    steps = epochs * ((n + batch_size - 1) // batch_size)
    optimizer = torch.optim.Adam(head.parameters(), lr=config['lr'],
                                 weight_decay=config.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    # Gaussian noise in units of each channel's std over the train set.
    noise = config.get('feature_noise', 0.0)
    channel_std = z[0].float().std(0) if noise else None

    history = []
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(n, device=device)
        sums, correct = defaultdict(float), 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            view = torch.randint(views, (len(idx),), device=device)
            zb = z[view, idx].float()
            if noise:
                zb = zb + noise * channel_std * torch.randn_like(zb)
            batch_extra = {key: value[idx] for key, value in (extra or {}).items()}
            loss, logits, terms = loss_terms(config, head, zb, label[idx], kg, batch_extra)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            sums['loss'] += loss.item() * len(idx)
            for key, value in terms.items():
                sums[key] += value * len(idx)
            correct += (logits.argmax(1) == label[idx]).sum().item()
        retrieval = head.retrieval if config['head'] == 'a1' else head
        row = dict(epoch=epoch + 1, **{key: value / n for key, value in sums.items()},
                   train_top1=correct / n, tau=retrieval.tau.item(), lr=scheduler.get_last_lr()[0])
        if config['head'] == 'a1':
            row['lambda'] = head.lam.item()
        history.append(row)
        lambda_note = f", lambda {row['lambda']:.3f}" if 'lambda' in row else ''
        print(f"epoch {epoch + 1}/{epochs}: loss {row['loss']:.4f}, "
              f"train top1 {100 * row['train_top1']:.2f}%, tau {row['tau']:.4f}{lambda_note}")
    return history


@torch.no_grad()
def score(head, z, batch_size=EVAL_BATCH):
    head.eval()
    return torch.cat([head(z[i:i + batch_size]) for i in range(0, len(z), batch_size)])


def attribute_auc(a, labels, kg, evidence=None):
    """Macro AUC of the attribute head against y_attr[label], on cells the KG knows (and, with
    evidence [N, A], that carry evidence: predicate attributes only).
    Diagnostic only: the targets come from class labels, not from per-sample annotation."""
    target, known = kg.y_attr[labels], kg.known[labels]
    if evidence is not None:
        known = known & evidence
    aucs = []
    for j in range(a.shape[1]):
        mask = known[:, j]
        t, s = target[mask, j], a[mask, j]
        n_pos = t.sum().item()
        n_neg = len(t) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = s.argsort().argsort().double() + 1
        aucs.append(((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item())
    return float(np.mean(aucs)), len(aucs)


def fixed_broken(pred, reference, labels):
    """Samples the head gets right where `reference` was wrong (fixed), and the reverse (broken)."""
    right, ref_right = pred == labels, reference == labels
    return dict(fixed=int((right & ~ref_right).sum()), broken=int((~right & ref_right).sum()))


@torch.no_grad()
def evaluate_a1(head, z, labels, kg, batch_size=EVAL_BATCH, extra=None, fc_logits=None):
    """Logits plus what the plan's S5 asks for: lambda, size of each logit term on the
    candidates (mean |.| and spread across candidates, which is what can change a decision),
    how often dropping lambda v or f changes the prediction, and the attribute AUC. A1-G (S6) adds
    the coverage of VERIFY (samples with v != 0 on some candidate) and the samples fixed / broken
    against the retrieval inside the head and against K0 (fc_logits, if given)."""
    head.eval()
    logits, s_all, a_all, ev_all = [], [], [], []
    sums, n = defaultdict(float), 0
    for i in range(0, len(z), batch_size):
        batch_extra = {key: value[i:i + batch_size] for key, value in (extra or {}).items()}
        out = head(z[i:i + batch_size], **batch_extra)
        lv = out['lam'] * out['v']
        full = out['s_c'] + lv + out['f']
        pred = full.argmax(1)
        for name, term in (('s', out['s_c']), ('lambda_v', lv), ('f', out['f'])):
            sums[f'mean_abs_{name}'] += term.abs().mean(1).sum().item()
            sums[f'spread_{name}'] += term.std(1).sum().item()
        sums['flip_without_lambda_v'] += (pred != (full - lv).argmax(1)).sum().item()
        sums['flip_without_f'] += (pred != (full - out['f']).argmax(1)).sum().item()
        sums['verify_coverage'] += (out['v'].abs() > 1e-8).any(1).sum().item()
        n += len(pred)
        logits.append(out['logits'].cpu())
        s_all.append(out['s'].cpu())
        a_all.append(out['a'].cpu())
        ev_all.append(out['evidence'].cpu())
    logits, s_all, a_all, ev_all = torch.cat(logits), torch.cat(s_all), torch.cat(a_all), torch.cat(ev_all)
    labels = labels.cpu()
    terms = {key: value / n for key, value in sums.items()}
    auc, auc_attrs = attribute_auc(a_all, labels, kg.to('cpu'), ev_all)
    pred = logits.argmax(1)
    metrics = dict(
        head=topk_metrics(logits, labels, ks=(1, 5)),
        retrieval=topk_metrics(s_all, labels),
        lam=head.lam.item(),
        tau=head.retrieval.tau.item(),
        terms=terms,
        attribute_auc=auc,
        attribute_auc_count=auc_attrs,
        vs_retrieval=fixed_broken(pred, s_all.argmax(1), labels),
    )
    if fc_logits is not None:
        metrics['vs_k0'] = fixed_broken(pred, fc_logits.cpu().argmax(1), labels)
    return logits, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache-dir', help='overrides cache_dir')
    parser.add_argument('--cache-name', help='overrides cache_name (e.g. ntu120_xsub_l6 for another layer)')
    parser.add_argument('--work-dir', help='overrides work_dir')
    parser.add_argument('--kg', help='overrides kg (e.g. a shuffled KG for K2)')
    parser.add_argument('--embeddings', help='overrides embeddings (e.g. the CLIP file, A1-enc)')
    parser.add_argument('--seed', type=int, help='overrides seed')
    parser.add_argument('--feature-noise', type=float,
                        help='overrides feature_noise (Gaussian noise on z in train-std units, train only)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--predicates-from', metavar='KG_METRICS_JSON',
                        help='use exactly the predicates selected by an earlier run (its kg_metrics.json), '
                             'e.g. K2 with the set chosen on the real KG; disables min_train_auc')
    parser.add_argument('--head-arg', action='append', default=[], metavar='KEY=VALUE',
                        help='overrides head_args[KEY] (YAML value; dotted KEY for nested, e.g. '
                             'graph_args.layers=0); repeatable, for ablations')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    for key in ('cache_dir', 'cache_name', 'work_dir', 'kg', 'embeddings', 'seed', 'feature_noise'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.predicates_from:
        with open(args.predicates_from, encoding='utf-8') as handle:
            chosen = json.load(handle)['predicates']['names']
        config['predicates'] = dict(config['predicates'], names=chosen, min_train_auc=None,
                                    names_from=args.predicates_from)
    for item in args.head_arg:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'--head-arg expects KEY=VALUE, got {item!r}')
        target = config.setdefault('head_args', {})
        *parents, leaf = key.split('.')
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = yaml.safe_load(value)
    os.makedirs(config['work_dir'], exist_ok=True)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    kg_json = load_kg(config['kg'])
    if needs_text(config):
        class_embeddings, attribute_embeddings = load_embeddings(
            config['embeddings'], kg_json, config.get('embedding_key', 'action'))
        text_note = f'class embeddings {tuple(class_embeddings.shape)} from {config["embeddings"]}'
    else:
        class_embeddings = attribute_embeddings = None
        text_note = 'no text embeddings (graph node vectors learned)'
    kg = kg_tensors(kg_json)
    z_train, y_train, _ = load_cache(config['cache_dir'], config['cache_name'], 'train')
    z_test, y_test, fc_test = load_cache(config['cache_dir'], config['cache_name'], 'test')
    if kg.W.shape[0] != int(fc_test.shape[-1]):
        raise ValueError(f'{kg.W.shape[0]} KG actions, {fc_test.shape[-1]} classes in the cache')
    shuffled = kg_json.get('shuffled')
    print(f'train z {tuple(z_train.shape)}, test z {tuple(z_test.shape)}, {text_note}, '
          f'KG {config["kg"]}' + (f' (shuffled, seed {shuffled["seed"]})' if shuffled else ''))

    device = torch.device(args.device)
    extra_train = extra_test = predicate_args = None
    if config['head'] == 'a1' and config.get('head_args', {}).get('attribute_source') == 'predicate':
        spec = config['predicates']
        loaded = {split: load_predicates(config['cache_dir'], spec['cache_name'], split, kg_json,
                                         spec.get('names', 'all')) for split in ('train', 'test')}
        for split, y in (('train', y_train), ('test', y_test)):
            if not torch.equal(loaded[split][2], y):
                raise ValueError(f'predicate cache {split} rows do not match the feature cache')
        phi_train, valid_train, _, names, columns = loaded['train']
        phi_test, valid_test = loaded['test'][0], loaded['test'][1]
        aucs = predicate_train_auc(phi_train, valid_train, y_train, kg, columns)
        min_auc = spec.get('min_train_auc')
        if min_auc is not None:
            keep = [j for j, auc in enumerate(aucs) if auc == auc and auc >= min_auc]
            if not keep:
                raise ValueError(f'no predicate reaches train AUC {min_auc}')
            dropped = [f'{names[j]} {aucs[j]:.2f}' for j in range(len(names)) if j not in keep]
            print(f'min_train_auc {min_auc}: dropped {len(dropped)}: {", ".join(dropped)}')
            phi_train, valid_train = phi_train[:, keep], valid_train[:, keep]
            phi_test, valid_test = phi_test[:, keep], valid_test[:, keep]
            names, columns, aucs = [names[j] for j in keep], [columns[j] for j in keep], [aucs[j] for j in keep]
        mean, std = predicate_stats(phi_train, valid_train)
        predicate_args = dict(columns=columns, mean=mean, std=std,
                              learn_threshold=spec.get('learn_threshold', True))
        extra_train = dict(phi=phi_train.to(device), phi_valid=valid_train.to(device))
        extra_test = dict(phi=phi_test.to(device), phi_valid=valid_test.to(device))
        print(f'{len(names)} predicates from {spec["cache_name"]} (train AUC): '
              + ', '.join(f'{n} {a:.2f}' for n, a in zip(names, aucs)))

    kg = kg.to(device)
    head = build_head(config, z_train.shape[-1], class_embeddings, attribute_embeddings, kg,
                      predicate_args).to(device)
    z_train, y_train = z_train.to(device), y_train.to(device)
    history = train(config, head, z_train, y_train, kg, device, extra_train)

    if config['head'] == 'a1':
        test_scores, metrics = evaluate_a1(head, z_test[0].to(device), y_test, kg, extra=extra_test,
                                           fc_logits=fc_test[0])
        _, train_metrics = evaluate_a1(head, z_train[0], y_train, kg, extra=extra_train)
        metrics['head_train_view0'] = train_metrics['head']
        if predicate_args is not None:
            metrics['predicates'] = dict(names=names, train_auc=aucs, theta=head.predicates.theta.tolist(),
                                         scale=head.predicates.log_scale.exp().tolist())
    else:
        test_scores = score(head, z_test[0].to(device)).cpu()
        metrics = dict(head=topk_metrics(test_scores, y_test), retrieval=None,
                       head_train_view0=topk_metrics(score(head, z_train[0]).cpu(), y_train.cpu()),
                       tau=head.tau.item())
    metrics = dict(config=config, kg_hash=kg_json['hash'], k0_fc_logits=topk_metrics(fc_test[0], y_test),
                   **metrics)
    torch.save(head.state_dict(), os.path.join(config['work_dir'], 'head.pt'))
    np.save(os.path.join(config['work_dir'], 'test_scores.npy'), test_scores.numpy().astype(np.float16))
    with open(os.path.join(config['work_dir'], 'history.json'), 'w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
    with open(os.path.join(config['work_dir'], 'kg_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    for name in ('head', 'retrieval', 'k0_fc_logits', 'head_train_view0'):
        if metrics.get(name):
            print(f'{name:>16}: ' + ', '.join(f'{k} {100 * v:.2f}%' for k, v in metrics[name].items()))
    if config['head'] == 'a1':
        t = metrics['terms']
        print(f"lambda {metrics['lam']:.3f}, tau {metrics['tau']:.4f}, attribute AUC "
              f"{metrics['attribute_auc']:.3f} ({metrics['attribute_auc_count']} attributes)")
        print('candidate logit terms (mean |.| / spread across candidates): ' + ', '.join(
            f"{name} {t[f'mean_abs_{name}']:.2f} / {t[f'spread_{name}']:.2f}" for name in ('s', 'lambda_v', 'f')))
        print(f"prediction changes without lambda*v: {100 * t['flip_without_lambda_v']:.2f}%, "
              f"without f: {100 * t['flip_without_f']:.2f}%")
        print(f"VERIFY coverage (v != 0): {100 * t['verify_coverage']:.2f}%; fixed / broken vs retrieval "
              f"{metrics['vs_retrieval']['fixed']} / {metrics['vs_retrieval']['broken']}, vs K0 "
              f"{metrics['vs_k0']['fixed']} / {metrics['vs_k0']['broken']}")
    key, target = RECALL_TARGET
    retrieval = metrics['retrieval'] or metrics['head']
    verdict = 'pass' if retrieval[key] >= target else 'below target'
    print(f"retrieval check: test {key} {100 * retrieval[key]:.2f}% (target >= {100 * target:.0f}%): {verdict}")
