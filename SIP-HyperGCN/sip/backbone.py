"""Load a trained HyperGCN and run it with taps / interventions at each layer.

`forward_tapped` re-implements `model.hypergcn_base.Model.forward` step by step
(same embedding, same residual wiring x1 -> l3, x4 -> l6, x7 -> l9, same pooling).

Intervention semantics
----------------------
Layer 0 is the input embedding: to_joint_embedding + pos_embedding + tanh + data_bn,
reshaped to (N*M, 128, T, V); it is the input of l1 (v3 region ablation point).
`intervene[k]` (k = 0..9) is applied to the OUTPUT of layer lk, before that output is
used anywhere downstream. For k in (1, 4, 7) the output is also stored as a residual
(x1 / x4 / x7) that is added to the input of l3 / l6 / l9; the residual is taken
AFTER the intervention, i.e. the edited representation fully replaces the
original one (no "clean" copy leaks through the skip connection).
The tapped feature of layer k is also recorded after the intervention.

State API (clean-prefix trick)
------------------------------
A state is a dict {x, N, M, next, [x1, x4, x7]} where x is the output of layer next-1.
`init_state` -> state after layer 0 (next=1); `run_layers(state, end)` -> state after
layer `end`; `intervene_state(state, fn)` applies fn to the output of layer next-1
(updating the residual copy when needed). A clean state after layer l can therefore be
shared by many interventions at l: `run_layers(intervene_state(prefix_l, fn), 9)`.
"""
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

NUM_LAYERS = 9
RESIDUAL_SAVE = {1: 'x1', 4: 'x4', 7: 'x7'}   # layer whose output is kept as residual
RESIDUAL_ADD = {3: 'x1', 6: 'x4', 9: 'x7'}    # layer whose input gets the residual

DEFAULT_MODEL_ARGS = dict(
    num_point=25,
    num_person=2,
    hyper_joints=3,
    graph='graph.ntu_rgb_d.Graph',
    graph_args={'labeling_mode': 'virtual_ensemble'},
)


def add_hypergcn_path(hypergcn_root):
    hypergcn_root = os.path.abspath(hypergcn_root)
    if hypergcn_root not in sys.path:
        sys.path.insert(0, hypergcn_root)
    return hypergcn_root


DEFAULT_CONFIG = os.path.join('config', 'base', 'nturgbd-cross-subject', 'hyper_joint.yaml')


def build_feeder(hypergcn_root, data_path, split, config=None):
    """Deterministic feeder: TEST feeder args from the yaml for both splits
    (center crop p=0.95, no rotation), only `split` and `data_path` overridden."""
    import yaml
    add_hypergcn_path(hypergcn_root)
    from feeders.feeder_ntu import Feeder

    if config is None:
        config = os.path.join(hypergcn_root, DEFAULT_CONFIG)
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    args = dict(cfg['test_feeder_args'])
    args['data_path'] = data_path
    args['split'] = split
    args['debug'] = False
    for k in ('random_rot', 'random_choose', 'random_shift', 'random_move'):
        args[k] = False
    print('[build_feeder] split={} args={}'.format(split, args))
    return Feeder(**args)


def _load_state_dict(checkpoint):
    try:
        weights = torch.load(checkpoint, map_location='cpu')
    except Exception:
        # torch >= 2.6 defaults to weights_only=True; fall back for pickled extras
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(weights, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'net'):
            if key in weights and isinstance(weights[key], dict):
                weights = weights[key]
                break
    state = OrderedDict()
    for k, v in weights.items():
        # hypergcn/main.py saves with k.split('module.')[-1]; handle DataParallel too
        if k.startswith('module.'):
            k = k[len('module.'):]
        state[k] = v
    return state


def load_hypergcn(hypergcn_root, checkpoint, num_class=60, device='cuda'):
    add_hypergcn_path(hypergcn_root)
    from model.hypergcn_base import Model

    model = Model(num_class=num_class, **DEFAULT_MODEL_ARGS)
    state = _load_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print('[load_hypergcn] unexpected keys:', unexpected)
    if missing:
        raise RuntimeError('[load_hypergcn] missing keys in checkpoint: {}'.format(missing))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print('[load_hypergcn] loaded {} ({} tensors)'.format(checkpoint, len(state)))
    return model


def _pool(h, N, M):
    # (N*M, C, T, V) -> (N, C): mean over T*V, then over persons (same as the head)
    c = h.size(1)
    return h.view(N, M, c, -1).mean(3).mean(1)


def _pool_joint(h, N, M):
    # (N*M, C, T, V) -> (N, C, V): mean over T, then over persons
    nm, c, t, v = h.size()
    return h.view(N, M, c, t, v).mean(3).mean(1)


def embed(model, x):
    """Input embedding part of Model.forward. x: (N, C, T, V, M) -> (N*M, 128, T, V)."""
    N, C, T, V, M = x.size()
    x = x.permute(0, 4, 2, 3, 1).contiguous()
    x = model.to_joint_embedding(x)
    x = x + model.pos_embedding[:, :model.num_point]
    x = model.tanh(x)
    x = x.permute(0, 1, 3, 4, 2).contiguous()

    x = x.view(N, M * V * model.embedding_channels, T)
    x = model.data_bn(x)
    x = x.view(N, M, V, model.embedding_channels, T).permute(0, 1, 3, 4, 2).contiguous()

    x = x.view(N * M, model.embedding_channels, T, V)
    return x, N, M


def person_joint_mean(h, N, M):
    """(N*M, C, T, V) -> (N, M, C, V): mean over T, kept per person slot.
    Used to estimate the TRAIN-FIT region-ablation means (M, C, V) of any layer (incl. 0)."""
    nm, c, t, v = h.size()
    return h.view(N, M, c, t, v).mean(3)


def person_present(x):
    """x: raw feeder input (N, C, T, V, M) -> (N, M) bool, True where the person slot
    is non-zero somewhere. The NTU feeder zero-fills a missing 2nd person and the
    joint-20-relative transform / crop-resize keeps it exactly zero."""
    return x.abs().sum(dim=(1, 2, 3)) > 0


def init_state(model, x, intervene=None, feats=None):
    """State after layer 0 (embedding + data_bn), i.e. right before l1.
    The state is a dict: x, N, M, next (index of next layer), residuals.
    intervene[0] (if given) is applied to the embedding output; feats[0] (if feats is
    a dict) receives its pooled (N, 128) feature, recorded after the intervention."""
    h, N, M = embed(model, x)
    if intervene is not None and 0 in intervene:
        h = intervene[0](h)
    if feats is not None:
        feats[0] = _pool(h, N, M)
    return {'x': h, 'N': N, 'M': M, 'next': 1}


def intervene_state(state, fn):
    """Apply fn to the output of the last completed layer k = state['next'] - 1
    (k = 0 is the embedding). Same semantics as `intervene[k]` in run_layers: if k
    keeps a residual (1, 4, 7) the residual is replaced too. Returns a NEW state."""
    s = dict(state)
    k = s['next'] - 1
    h = fn(s['x'])
    s['x'] = h
    if k in RESIDUAL_SAVE:
        s[RESIDUAL_SAVE[k]] = h
    return s


def run_layers(model, state, end, intervene=None, feats=None, jfeats=None, joint_layers=()):
    """Run layers state['next'] .. end (inclusive). Returns a NEW state dict
    (input state is not modified, so it can be reused as a shared prefix).
    feats / jfeats: optional dicts filled with pooled / joint-level features."""
    s = dict(state)
    N, M = s['N'], s['M']
    for k in range(s['next'], end + 1):
        layer = getattr(model, 'l{}'.format(k))
        inp = s['x']
        if k in RESIDUAL_ADD:
            inp = inp + s[RESIDUAL_ADD[k]]
        h, _ = layer(inp)
        if intervene is not None and k in intervene:
            h = intervene[k](h)
        if k in RESIDUAL_SAVE:
            s[RESIDUAL_SAVE[k]] = h
        s['x'] = h
        if feats is not None:
            feats[k] = _pool(h, N, M)
        if jfeats is not None and k in joint_layers:
            jfeats[k] = _pool_joint(h, N, M)
    s['next'] = end + 1
    return s


def head(model, state):
    """Pooling + fc on a state that has run all 9 layers. Returns (logits, pooled l9)."""
    assert state['next'] == NUM_LAYERS + 1, 'run all layers before head()'
    z = _pool(state['x'], state['N'], state['M'])
    return model.fc(model.drop_out(z)), z


def forward_tapped(model, x, intervene=None, joint_layers=(3, 6, 9)):
    """Exact re-implementation of Model.forward with taps.

    x: (N, 3, T, V, M) tensor on the model device.
    intervene: {k: fn}, k in 0..9 (0 = embedding output), fn maps the layer-k output
        (N*M, C, T', V) -> same shape.
    Returns logits (N, K), feats {k: (N, C_k)} for k=1..9, jfeats {k: (N, C_k, V)}.
    """
    feats, jfeats = {}, {}
    state = init_state(model, x, intervene=intervene)
    state = run_layers(model, state, NUM_LAYERS, intervene=intervene, feats=feats,
                       jfeats=jfeats, joint_layers=joint_layers)
    logits, _ = head(model, state)
    return logits, feats, jfeats


@torch.no_grad()
def check_equivalence(model, x, atol=1e-4, rtol=1e-4):
    """Assert forward_tapped (no intervention, and identity intervention) == model(x)[0]."""
    model.eval()
    ref = model(x)[0]
    logits, feats, _ = forward_tapped(model, x)
    diff = (ref - logits).abs().max().item()
    ident = {k: (lambda h: h) for k in range(0, NUM_LAYERS + 1)}
    logits_id, _, _ = forward_tapped(model, x, intervene=ident)
    diff_id = (ref - logits_id).abs().max().item()
    # state API: identity intervention at every layer via intervene_state == plain
    for k in range(0, NUM_LAYERS + 1):
        st = run_layers(model, init_state(model, x), k)
        st = run_layers(model, intervene_state(st, lambda h: h.clone()), NUM_LAYERS)
        d = (ref - head(model, st)[0]).abs().max().item()
        assert d <= atol + rtol * ref.abs().max().item(),             'intervene_state identity at l{} changed logits ({:.3e})'.format(k, d)
    # the head must be fc(pooled l9)
    diff_fc = (model.fc(feats[NUM_LAYERS]) - ref).abs().max().item()
    print('[check_equivalence] max|diff| plain={:.3e} identity={:.3e} fc(feat_l9)={:.3e}'.format(
        diff, diff_id, diff_fc))
    assert torch.allclose(ref, logits, atol=atol, rtol=rtol), 'forward_tapped != model.forward'
    assert torch.allclose(ref, logits_id, atol=atol, rtol=rtol), 'identity intervention changed logits'
    assert diff_fc <= atol + rtol * ref.abs().max().item(), 'fc(feat_l9) != logits'
    return diff


# ---------------------------------------------------------------------------
# Shared machinery of run_intervention.py (measures, LEACE) and
# run_region_intervention.py (regions, mean-ablation).
# ---------------------------------------------------------------------------

def stratified_val_index(labels, val_frac, seed):
    """Sorted row indices of a per-class `val_frac` subset. Depends only on
    (labels, val_frac, seed) so every run reproduces the same VAL."""
    assert 0.0 < val_frac < 1.0, 'val_frac must be in (0, 1)'
    labels = np.asarray(labels).astype(np.int64)
    rng = np.random.RandomState(seed)
    val = []
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        n_c = len(rows)
        n_val = int(round(val_frac * n_c))
        if n_c >= 2:
            n_val = min(max(n_val, 1), n_c - 1)
        else:
            n_val = 0
        perm = rng.permutation(n_c)
        val.append(rows[perm[:n_val]])
    return np.sort(np.concatenate(val)).astype(np.int64)


def split_train_rows(train_labels, val_frac, seed):
    """(val_index, fit_rows): VAL and TRAIN-FIT row indices into train_feats.npz."""
    n_train = len(train_labels)
    val_index = stratified_val_index(train_labels, val_frac, seed)
    fit_mask = np.ones(n_train, dtype=bool)
    fit_mask[val_index] = False
    fit_rows = np.flatnonzero(fit_mask)
    print('[split] train rows={} VAL={} TRAIN-FIT={} (val_frac={}, seed={})'.format(
        n_train, len(val_index), len(fit_rows), val_frac, seed))
    return val_index, fit_rows


def build_eval_dataset(hypergcn_root, data_path, eval_split, config, val_index,
                       train_feeder_index, train_labels, max_samples=None):
    """Evaluation set. eval_split 'train' -> the VAL rows of train_feats.npz (Subset of
    the train feeder at their feeder indices); 'test' -> the test feeder (first
    max_samples rows). Returns (dataset, eval_index, expected_feeder, expected_labels)
    where eval_index are row indices into <eval_split>_feats.npz.
    For 'test' the test_feats.npz rows are assumed to be in feeder order
    (extract_features.py, shuffle=False, no --max_samples or the same prefix)."""
    dataset = build_feeder(hypergcn_root, data_path, eval_split, config=config)
    if eval_split == 'train':
        eval_index = np.asarray(val_index).copy()
        if max_samples is not None and max_samples < len(eval_index):
            eval_index = eval_index[:max_samples]
        expected_feeder = np.asarray(train_feeder_index)[eval_index]
        assert expected_feeder.max() < len(dataset), 'train_feats index out of range of the train feeder'
        expected_labels = np.asarray(train_labels)[eval_index]
        dataset = torch.utils.data.Subset(dataset, expected_feeder.tolist())
    else:
        n_eval = len(dataset)
        if max_samples is not None and max_samples < n_eval:
            n_eval = max_samples
            dataset = torch.utils.data.Subset(dataset, list(range(n_eval)))
        eval_index = np.arange(n_eval, dtype=np.int64)
        expected_feeder = eval_index
        expected_labels = None
    return dataset, eval_index.astype(np.int64), expected_feeder, expected_labels


@torch.no_grad()
def intervention_sweep(model, loader, layers, fns, num_class, store_final=True,
                       device='cuda', check=True, tag='sweep', on_batch=None):
    """Traverse `loader` once. Per batch: one clean pass that keeps the clean state
    AFTER each layer l in `layers` (layer 0 = embedding); then, for every unit u,
    `run_layers(intervene_state(prefix[l], fns[li][u]), 9)` + head. Only layers
    l+1..9 are recomputed per unit. fns[li][u] = None means skipped (the clean
    prediction / feature is copied). `on_batch(x)`, if given, is called with each raw
    input batch before any pass (e.g. to update a per-batch person-presence mask).

    Returns dict: base_logits (N,K) f32, base_final (N,C) f32, labels (N,), feeder_index (N,),
    correct (L,U,N) bool, pred_label (L,U,N) int16, final_feat (L,U,N,C) f16 or None.
    """
    L = len(layers)
    U = len(fns[0]) if L else 0
    N, K = len(loader.dataset), num_class
    C_final = model.fc.in_features
    n_runs = sum(f is not None for row in fns for f in row)
    print('[{}] N={} L={} U={} -> 1 clean + {} intervened partial passes per batch'.format(
        tag, N, L, U, n_runs))

    base_logits = np.zeros((N, K), dtype=np.float32)
    base_final = np.zeros((N, C_final), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    feeder_index = np.zeros(N, dtype=np.int64)
    correct = np.zeros((L, U, N), dtype=bool)
    pred_label = np.zeros((L, U, N), dtype=np.int16)
    final_feat = None
    if store_final:
        final_feat = np.zeros((L, U, N, C_final), dtype=np.float16)
        print('final_feat buffer: {:.0f} MB'.format(final_feat.nbytes / 2 ** 20))

    n_batch = len(loader)
    t0 = time.time()
    pos = 0
    for b, (data, label, index) in enumerate(loader):
        x = data.float().to(device)
        n = x.size(0)
        sl = slice(pos, pos + n)
        labels[sl] = np.asarray(label).astype(np.int64)
        feeder_index[sl] = np.asarray(index).astype(np.int64)
        if on_batch is not None:
            on_batch(x)
        if b == 0 and check:
            check_equivalence(model, x)

        # clean pass, keeping the state right AFTER each intervened layer
        state = init_state(model, x)
        prefix = {}
        for l in layers:
            state = run_layers(model, state, l)
            prefix[l] = state
        state = run_layers(model, state, NUM_LAYERS)
        logits, z = head(model, state)
        base_logits[sl] = logits.float().cpu().numpy()
        base_final[sl] = z.float().cpu().numpy()
        bpred = logits.argmax(1).cpu().numpy()

        for li, l in enumerate(layers):
            for ui in range(U):
                fn = fns[li][ui]
                if fn is None:
                    pred_np, z_np = bpred, base_final[sl]
                else:
                    s = run_layers(model, intervene_state(prefix[l], fn), NUM_LAYERS)
                    lg, zz = head(model, s)
                    pred_np = lg.argmax(1).cpu().numpy()
                    z_np = None if final_feat is None else zz.float().cpu().numpy()
                pred_label[li, ui, sl] = pred_np.astype(np.int16)
                correct[li, ui, sl] = pred_np == labels[sl]
                if final_feat is not None:
                    final_feat[li, ui, sl] = z_np.astype(np.float16)
        pos += n
        del prefix, state

        el = time.time() - t0
        if (b + 1) % 5 == 0 or b + 1 == n_batch or b == 0:
            print('[{}] batch {}/{}  elapsed {:.0f}s  ETA {:.0f}s'.format(
                tag, b + 1, n_batch, el, el / (b + 1) * (n_batch - b - 1)))
    assert pos == N
    return dict(base_logits=base_logits, base_final=base_final, labels=labels,
                feeder_index=feeder_index, correct=correct, pred_label=pred_label,
                final_feat=final_feat)


def sweep_outputs(res, num_class, layers, unit_names, skipped, eval_split, eval_index,
                  val_index, n_random, val_frac, seed, expected_feeder, expected_labels):
    """Check the evaluated rows and build the npz dict common to both intervention
    scripts (docs/PIPELINE_V3.md keys)."""
    assert np.array_equal(res['feeder_index'], expected_feeder), 'feeder order != eval_index mapping'
    labels = res['labels']
    if expected_labels is not None:
        assert np.array_equal(labels, expected_labels), 'feeder labels != train_feats labels on VAL rows'
    K = num_class
    L, U = res['correct'].shape[:2]
    assert len(unit_names) == U
    base_pred = res['base_logits'].argmax(1).astype(np.int64)
    count = np.bincount(labels, minlength=K).astype(np.float64)
    denom = np.maximum(count, 1)
    base_acc_per_class = np.bincount(labels, weights=(base_pred == labels).astype(np.float64),
                                     minlength=K) / denom
    acc_per_class = np.zeros((L, U, K), dtype=np.float64)
    for li in range(L):
        for ui in range(U):
            acc_per_class[li, ui] = np.bincount(
                labels, weights=res['correct'][li, ui].astype(np.float64), minlength=K) / denom
    if (count == 0).any():
        print('warning: {} classes absent from the evaluated subset (acc set to 0)'.format(
            int((count == 0).sum())))
    out = dict(
        layers=np.array(layers, dtype=np.int64),
        unit_names=np.array(unit_names, dtype=str),
        n_random=np.array(n_random, dtype=np.int64),
        labels=labels,
        base_logits=res['base_logits'],
        base_pred=base_pred,
        correct=res['correct'],
        pred_label=res['pred_label'],
        acc_per_class=acc_per_class,
        base_acc_per_class=base_acc_per_class,
        class_count=count.astype(np.int64),
        skipped=np.asarray(skipped, dtype=bool),
        eval_split=np.array(eval_split),
        eval_index=np.asarray(eval_index, dtype=np.int64),
        val_index=np.asarray(val_index, dtype=np.int64),
        val_frac=np.array(val_frac, dtype=np.float64),
        seed=np.array(seed, dtype=np.int64),
        base_final_feat=res['base_final'],
    )
    if res['final_feat'] is not None:
        out['final_feat'] = res['final_feat']
    return out


def print_sweep_summary(out):
    labels, base_pred = out['labels'], out['base_pred']
    base_acc = (base_pred == labels).mean()
    print('split={} N={} base top1={:.4f}'.format(str(out['eval_split']), len(labels), base_acc))
    for li, l in enumerate(out['layers']):
        for ui, name in enumerate(out['unit_names']):
            acc = out['correct'][li, ui].mean()
            print('  l{} {:28s} top1={:.4f}  drop={:+.4f}{}'.format(
                int(l), str(name), acc, acc - base_acc,
                '  (skipped)' if out['skipped'][li, ui] else ''))
