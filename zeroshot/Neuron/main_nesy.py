"""Train / evaluate the NeSy model built on Neuron (model/nesy.py).

    python main_nesy.py --config config/nesy/ntu60_xsub_55_5.yaml

Feature modes
  cache  (default): the frozen encoder runs ONCE. Every training sample gets ONE fixed augmentation (Neuron's
                    random crop + rotation, with the RNGs seeded by seed + sample index, so the view is the same
                    in every run); test samples use the deterministic centre crop. The encoder cubes
                    (n, 256, 16, 25, fp16) are kept on the GPU, in RAM or memory-mapped on disk
                    (cache_device; auto picks by free memory, disk fits a T4) and saved to cache_dir for
                    reuse. Batches are index-sorted, so every place gives the same batches.
  online          : Neuron's way, a new random augmentation every epoch through the encoder (slow).

Checkpoints in work_dir
  last.pt : model + optimisers + history, written every epoch; with resume=True a restarted run continues from it.
  best.pt : model at the epoch with the highest `best_metric` (default zsl_final). Selected on the test set,
            like Neuron's best epoch, so it is optimistic.

Differences from main_match.py (Neuron):
  - every reported accuracy is a single prediction per sample (Neuron's "spatial OR temporal" accuracy is
    logged only as `neuron_or`, for comparison with the paper);
  - one test pass per epoch: GZSL scores on all classes; ZSL = the unseen samples restricted to the unseen
    columns (scores are per class, so this equals evaluating with the unseen classes only);
  - paths come from the config; no interactive prompt; per-epoch metrics go to <work_dir>/metrics.csv.
GZSL uses calibrated stacking with gamma searched on the test set, as Neuron does (optimistic).

Primitive labels (lambda_prim > 0 or prim_eval, cache mode): primitive_labels.py computes the truth value of every
primitive from the skeleton (its `definition`), on the same crop as the cached cube of each sample but without its
random rotation, once per cache (saved next to the cubes as *_prim<hash>.npy, uint8, (N, 2, K, 16)). Used for
  - lambda_prim: BCE supervision of the grounding (model.prim_loss), with pos_weight from the label rates;
  - prim_eval: zsl_oracle (the rules on the test labels instead of the grounding: the ceiling of the KG) and
    prim_auc_unseen / prim_auc_seen (mean AUC of the grounding's presence vs the labels on test samples of unseen /
    seen classes; per-primitive values in <work_dir>/prim_auc.json).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import OrderedDict

import numpy as np
import psutil
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import kg as kg_lib
import primitive_labels as pl
from feeders.feeder_ntu_mmap import count_samples

SPLITS = {  # 0-based class ids, same as feeders/feeder_ntu.py
    'ntu60_seen55_unseen5':    [10, 11, 19, 26, 56],
    'ntu60_seen48_unseen12':   [3, 5, 9, 12, 15, 40, 42, 47, 51, 56, 58, 59],
    'ntu120_seen110_unseen10': [4, 13, 37, 43, 49, 65, 88, 95, 99, 106],
    'ntu120_seen96_unseen24':  [5, 9, 11, 16, 18, 20, 22, 29, 35, 39, 45, 49, 59, 68, 70, 81, 84, 87, 93, 94,
                                104, 113, 114, 119],
}
SEMANTICS = {  # Neuron's CLIP features of the LLM descriptions, (120, 10, 768) each
    'spatial': ['spatial/ntu120_gpt4omini_incontext_spatial_cg.tar',
                'spatial/ntu120_gpt4omini_incontext_spatial_mg.tar',
                'spatial/ntu120_gpt4omini_incontext_spatial_fg.tar'],
    'temporal': ['temporal/ntu120_gpt4omini_incontext_temporal_fp.tar',
                 'temporal/ntu120_gpt4omini_incontext_temporal_fsp.tar',
                 'temporal/ntu120_gpt4omini_incontext_temporal_fstp.tar'],
}
CUBE_SHAPE = (256, 16, 25)


def import_class(name):
    mod, _, cls = name.rpartition('.')
    __import__(mod)
    return getattr(sys.modules[mod], cls)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class DictAction(argparse.Action):
    """--model-args "d_sym=128,beta_init=0.5" style overrides (same as torchlight.DictAction)."""

    def __call__(self, parser, namespace, values, option_string=None):
        d = dict(getattr(namespace, self.dest) or {})
        d.update(eval(f'dict({values})'))
        setattr(namespace, self.dest, d)


class FixedView(Dataset):
    """Feeder wrapper: sample i always gets the same augmentation (numpy / random / torch RNGs seeded with
    seed + i before the feeder draws its crop and rotation), whatever the number of workers."""

    def __init__(self, ds, seed):
        self.ds, self.seed = ds, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.seed + int(i)
        np.random.seed(s % 2 ** 32)
        random.seed(s)
        torch.manual_seed(s)
        return self.ds[i]


def best_gamma(z, y, classes, seen, span=None, n=101):
    """Calibrated stacking: subtract gamma from seen-class scores; returns (gamma, S, U, H) with max H.
    span: range of the gamma grid (default: range of z; pass the unmasked range for masked scores)."""
    is_seen_col = torch.isin(classes, seen).float()
    s_mask = torch.isin(y, seen)
    span = float(z.max() - z.min()) if span is None else span
    best = (0.0, 0.0, 0.0, -1.0)
    for g in np.linspace(-0.2 * span, span, n):
        pred = classes[(z - g * is_seen_col).argmax(1)]
        S = float((pred[s_mask] == y[s_mask]).float().mean())
        U = float((pred[~s_mask] == y[~s_mask]).float().mean())
        H = 2 * S * U / (S + U) if S + U > 0 else 0.0
        if H > best[3]:
            best = (float(g), S, U, H)
    return best


def mean_auc(score, target, impl, min_n=20):
    """Per-column ROC AUC of score (N, K) against target > 0.5; ok marks columns of implemented primitives with
    at least min_n positives and negatives."""
    b = target > 0.5
    r = score.argsort(0).argsort(0).float() + 1
    npos, nneg = b.sum(0).float(), (~b).sum(0).float()
    auc = ((r * b).sum(0) - npos * (npos + 1) / 2) / (npos * nneg).clamp(min=1)
    return auc, impl & (npos >= min_n) & (nneg >= min_n)


def save_npy(t, path, chunk=4096):
    """Tensor (CPU or GPU) -> .npy, written in chunks."""
    arr = np.lib.format.open_memmap(path, mode='w+', dtype=str(t.dtype).replace('torch.', ''), shape=tuple(t.shape))
    for i in range(0, len(t), chunk):
        arr[i:i + chunk] = t[i:i + chunk].cpu().numpy()
    arr.flush()
    del arr


def load_npy(path, device='cpu', chunk=4096):
    """.npy -> tensor on device, read in chunks (no second full copy in RAM)."""
    arr = np.load(path, mmap_mode='r')
    t = torch.empty(arr.shape, dtype=getattr(torch, str(arr.dtype)), device=device)
    for i in range(0, len(arr), chunk):
        t[i:i + chunk] = torch.from_numpy(np.ascontiguousarray(arr[i:i + chunk])).to(device)
    return t


class Processor:
    def __init__(self, arg):
        self.arg = arg
        os.makedirs(arg.work_dir, exist_ok=True)
        with open(f'{arg.work_dir}/config.yaml', 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(vars(arg), f)
        self.dev = torch.device('cuda')

        self.unseen = sorted(SPLITS[arg.split])
        self.n_cls = 60 if arg.split.startswith('ntu60') else 120
        self.seen = [c for c in range(self.n_cls) if c not in self.unseen]
        self.cls_seen = torch.tensor(self.seen, device=self.dev)
        self.cls_all = torch.arange(self.n_cls, device=self.dev)
        self.unseen_t = torch.tensor(self.unseen, device=self.dev)
        self.remap = torch.full((self.n_cls,), -1, dtype=torch.long, device=self.dev)
        self.remap[self.cls_seen] = torch.arange(len(self.seen), device=self.dev)

        load = lambda p: torch.load(os.path.join(arg.semantic_dir, p), map_location='cpu', weights_only=False)
        self.sem = {k: [load(p).float().to(self.dev) for p in v] for k, v in SEMANTICS.items()}

        ma = dict(arg.model_args)
        kg_lib.encode_primitive_text(ma['primitive_path'], ma['prim_text_path'])
        ma.update(n_cls_total=self.n_cls, num_class=len(self.seen))
        self.model = import_class(arg.model)(**ma)
        self.weights_path = self.load_encoder()
        self.model = self.model.to(self.dev)

        n_neu = sum(p.numel() for p in self.model.neuron_parameters())
        n_sym = sum(p.numel() for p in self.model.nesy_parameters())
        self.log(f'trainable params: Neuron head {n_neu:,} | NeSy {n_sym:,}')
        self.opt_neu = torch.optim.SGD(self.model.neuron_parameters(), lr=arg.base_lr, momentum=0.9,
                                       nesterov=arg.nesterov, weight_decay=arg.weight_decay)
        self.opt_sym = torch.optim.AdamW(self.model.nesy_parameters(), lr=arg.nesy_lr, weight_decay=arg.nesy_wd)

        self.need_prim = arg.lambda_prim > 0 or arg.prim_eval
        self.prim_impl, self.prim_auc = None, {}
        if self.need_prim:
            assert arg.feature_mode == 'cache', 'primitive labels (lambda_prim / prim_eval) need feature_mode cache'
            with open(ma['primitive_path'], 'rb') as f:
                raw = f.read()
            self.labeler = pl.PrimitiveLabeler(json.loads(raw))
            self.prim_impl = self.labeler.implemented.to(self.dev)
            self.prim_tag = hashlib.md5(raw + f'labeler v{pl.VERSION}'.encode()).hexdigest()[:8]
            self.log(f'primitive labels: {int(self.prim_impl.sum())}/{len(self.labeler.ids)} primitives have a '
                     f'detector (missing: {self.labeler.missing or "-"}) | lambda_prim {arg.lambda_prim} | '
                     f'ground_detach {self.model.ground_detach}')

        self.Feeder = import_class(arg.feeder)
        if arg.feature_mode == 'cache':
            n = {part: self.cached_count(part) or count_samples(**self.feeder_args(part)) for part in ('train', 'test')}
            K = self.arg.cache_views
            self.where = self.cache_place(K * n['train'] + n['test'])
            # train: K fixed views per sample (epoch e uses view e % K); view 0 is the single-view cache
            self.cache = dict(train=[self.build_cache('train', n['train'], view=k) for k in range(K)],
                              test=self.build_cache('test', n['test']))
            n_tr, n_te = n['train'], n['test']
            if self.need_prim:
                for k, D in enumerate(self.cache['train']):
                    D['prim'] = self.build_labels('train', D['y'], view=k)
                self.cache['test']['prim'] = self.build_labels('test', self.cache['test']['y'])
                self.set_prim_pos_weight(self.cache['train'][0]['prim'])
        elif arg.feature_mode == 'online':
            self.loader = dict(
                train=DataLoader(self.Feeder(**arg.train_feeder_args), batch_size=arg.batch_size, shuffle=True,
                                 num_workers=arg.num_worker, drop_last=True, worker_init_fn=init_seed,
                                 pin_memory=True),
                test=DataLoader(self.Feeder(**arg.test_feeder_args), batch_size=arg.test_batch_size,
                                shuffle=False, num_workers=arg.num_worker, pin_memory=True))
            n_tr, n_te = len(self.loader['train'].dataset), len(self.loader['test'].dataset)
        else:
            raise ValueError(f'feature_mode must be cache or online, got {arg.feature_mode}')
        self.log(f'split {arg.split}: {len(self.seen)} seen / {len(self.unseen)} unseen | '
                 f'train {n_tr} | test {n_te} | feature_mode {arg.feature_mode}')

    # ------------------------------------------------------------------ utils
    def log(self, s):
        s = f'[{time.strftime("%H:%M:%S")}] {s}'
        print(s, flush=True)
        with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
            print(s, file=f)

    def load_encoder(self):
        hits = sorted(glob.glob(self.arg.weights, recursive=True))
        assert len(hits) == 1, f'weights pattern {self.arg.weights} matches {hits}'
        sd = torch.load(hits[0], map_location='cpu')
        sd = OrderedDict(('feature_extractor.' + k.split('module.')[-1], v) for k, v in sd.items())
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        miss_enc = [k for k in missing if k.startswith('feature_extractor.')]
        assert not miss_enc, f'encoder keys missing from the checkpoint: {miss_enc}'
        self.log(f'encoder weights: {hits[0]}' + (f' | ignored keys: {unexpected}' if unexpected else ''))
        return hits[0]

    def semantics(self, classes):
        return [t[classes] for t in self.sem['spatial']], [t[classes] for t in self.sem['temporal']]

    def adjust_lr(self, epoch):
        f = self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step))
        for g in self.opt_neu.param_groups:
            g['lr'] = self.arg.base_lr * f
        for g in self.opt_sym.param_groups:
            g['lr'] = self.arg.nesy_lr * f

    # ------------------------------------------------------------------ feature cache
    def feeder_args(self, part):
        return self.arg.train_feeder_args if part == 'train' else self.arg.test_feeder_args

    def cache_place(self, n_total):
        """Where the cubes live: 'cuda' (fastest), 'cpu' (RAM) or 'disk' (memory-mapped .npy in cache_dir).
        auto: GPU if they fit with room to train, else RAM if it has room, else disk."""
        gb = n_total * int(np.prod(CUBE_SHAPE)) * 2 / 2 ** 30
        where = self.arg.cache_device
        free_gpu = torch.cuda.mem_get_info()[0] / 2 ** 30
        free_ram = psutil.virtual_memory().available / 2 ** 30
        if where == 'auto':
            if gb < min(self.arg.gpu_cache_gb, free_gpu - 4):          # keep ~4 GB for training
                where = 'cuda'
            elif gb < free_ram - 6:                                    # keep ~6 GB for workers / Python
                where = 'cpu'
            else:
                where = 'disk'
        if where == 'disk':
            assert self.arg.cache_dir, 'cache_device disk needs cache_dir'
            free_disk = shutil.disk_usage(os.path.dirname(os.path.abspath(self.arg.cache_dir)) or '/').free / 2 ** 30
            self.log(f'free disk {free_disk:.0f} GB')
        self.log(f'cache: {gb:.1f} GB of cubes -> {where} (free GPU {free_gpu:.1f} GB, free RAM {free_ram:.1f} GB)')
        return where

    def cache_paths(self, part, view=0):
        a = self.arg
        if not a.cache_dir:
            return {}
        key = dict(feeder=self.feeder_args(part), weights=os.path.basename(self.weights_path), seed=a.seed)
        if view:                                       # view 0 keeps the single-view name, so old caches are reused
            key['view'] = view
        sig = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
        tag = part if not view else f'{part}v{view}'
        return {k: f'{a.cache_dir}/{a.split}_{tag}_{sig}_{k}.npy' for k in ('cube', 'person', 'y')}

    def cached_count(self, part):
        """Number of samples of a complete cache for this part, or 0 (then the data are not needed)."""
        paths = self.cache_paths(part)
        if paths and all(os.path.exists(p) for p in paths.values()):
            return int(np.load(paths['y'], mmap_mode='r').shape[0])
        return 0

    @torch.no_grad()
    def extract(self, part, N, paths, view=0):
        """Frozen encoder over one feeder, one fixed view per sample -> cube (N, 256, 16, 25) fp16 written
        straight to its place (GPU / RAM tensor, or a memory-mapped .npy for 'disk'), labels, second-person flag.
        View k seeds sample i with seed + k * 1e8 + i."""
        ds = FixedView(self.Feeder(**self.feeder_args(part)), self.arg.seed + view * 10 ** 8)
        assert len(ds) == N, (len(ds), N)
        loader = DataLoader(ds, batch_size=self.arg.extract_batch_size, shuffle=False,
                            num_workers=self.arg.num_worker, pin_memory=True)
        small = 'cuda' if self.where == 'cuda' else 'cpu'
        if self.where == 'disk':
            os.makedirs(self.arg.cache_dir, exist_ok=True)
            cube = np.lib.format.open_memmap(paths['cube'], mode='w+', dtype=np.float16, shape=(N, *CUBE_SHAPE))
        else:
            cube = torch.empty((N, *CUBE_SHAPE), dtype=torch.float16, device=self.where)
        out = dict(cube=cube, y=torch.empty(N, dtype=torch.long, device=small),
                   person=torch.empty(N, dtype=torch.bool, device=small))
        enc = self.model.feature_extractor.eval()
        i = 0
        for x, y, _ in tqdm(loader, ncols=60, desc=f'extract {part}' + (f' view {view}' if view else ''),
                            mininterval=30):
            x = x.float().to(self.dev, non_blocking=True)
            _, _, c = enc(x)
            n = len(x)
            if self.where == 'disk':
                cube[i:i + n] = c.half().cpu().numpy()
            else:
                cube[i:i + n] = c.half()
            out['y'][i:i + n] = y.long().to(small)
            out['person'][i:i + n] = self.model.second_person(x).to(small)
            i += n
        assert i == N
        if self.where == 'disk':
            cube.flush()
        del loader, ds
        return out

    def build_cache(self, part, N, view=0):
        a, paths, t0 = self.arg, self.cache_paths(part, view), time.time()
        name = part if not view else f'{part} view {view}'
        small = 'cuda' if self.where == 'cuda' else 'cpu'
        if paths and all(os.path.exists(p) for p in paths.values()):
            out = dict(y=load_npy(paths['y'], small), person=load_npy(paths['person'], small))
            out['cube'] = np.load(paths['cube'], mmap_mode='r') if self.where == 'disk' \
                else load_npy(paths['cube'], self.where)
            assert len(out['y']) == N
            self.log(f'cache {name}: loaded {N} samples from {a.cache_dir} ({time.time() - t0:.0f}s)')
            return out
        out = self.extract(part, N, paths, view)
        msg = f'cache {name}: encoded {N} samples ({time.time() - t0:.0f}s)'
        if paths:
            os.makedirs(a.cache_dir, exist_ok=True)
            if self.where != 'disk':                   # disk: the cube file was written during extraction
                save_npy(out['cube'], paths['cube'])
            save_npy(out['person'], paths['person'])
            save_npy(out['y'], paths['y'])             # labels last: marks the cache as complete
            msg += f', saved to {a.cache_dir} ({time.time() - t0:.0f}s total)'
        if self.where == 'disk':
            out['cube'] = np.load(paths['cube'], mmap_mode='r')
        self.log(msg)
        return out

    # ------------------------------------------------------------------ primitive labels
    @torch.no_grad()
    def build_labels(self, part, y_ref, view=0):
        """Primitive labels (N, 2, K, 16) uint8 of one cached part / view: same sample order and crop as the cubes
        (checked through the class labels), loaded from cache_dir when present."""
        a, t0, N = self.arg, time.time(), len(y_ref)
        name = part if not view else f'{part} view {view}'
        small = 'cuda' if self.where == 'cuda' else 'cpu'
        paths = self.cache_paths(part, view)
        path = paths['y'].replace('_y.npy', f'_prim{self.prim_tag}.npy') if paths else None
        if path and os.path.exists(path):
            out = load_npy(path, small)
            assert len(out) == N, (path, len(out), N)
            self.log(f'primitive labels {name}: loaded from {path} ({time.time() - t0:.0f}s)')
            return out
        ds = pl.LabelView(self.Feeder(**self.feeder_args(part)), a.seed + view * 10 ** 8)
        assert len(ds) == N, (len(ds), N)
        loader = DataLoader(ds, batch_size=a.extract_batch_size, shuffle=False, num_workers=a.num_worker,
                            pin_memory=True)
        out, i = None, 0
        y_cpu = y_ref.cpu().long()
        for x, y, dt in tqdm(loader, ncols=60, desc=f'labels {name}', mininterval=30):
            lab = self.labeler(x.to(self.dev, non_blocking=True), dt.to(self.dev, non_blocking=True))
            if out is None:
                out = torch.empty((N, *lab.shape[1:]), dtype=torch.uint8, device=small)
            n = len(x)
            assert torch.equal(y.long(), y_cpu[i:i + n]), f'{name}: sample order differs from the cube cache'
            out[i:i + n] = (lab * 255).round().to(torch.uint8).to(small)
            i += n
        assert i == N
        if path:
            os.makedirs(a.cache_dir, exist_ok=True)
            save_npy(out, path)
        self.log(f'primitive labels {name}: computed for {N} samples ({time.time() - t0:.0f}s)'
                 + (f', saved to {path}' if path else ''))
        return out

    def set_prim_pos_weight(self, prim):
        """pos_weight of the primitive BCE = (1 - rate) / rate of each primitive's labels, clipped to
        [1, prim_pos_weight_max]."""
        tot = torch.zeros(prim.shape[2], dtype=torch.float64, device=prim.device)
        for i in range(0, len(prim), 4096):
            tot += prim[i:i + 4096].double().sum((0, 1, 3))
        rate = (tot / (len(prim) * prim.shape[1] * prim.shape[3] * 255)).float().to(self.dev)
        pw = ((1 - rate) / rate.clamp(min=1e-4)).clamp(1, self.arg.prim_pos_weight_max)
        self.model.prim_pos_weight.copy_(pw)
        ids, impl = self.labeler.ids, self.prim_impl
        order = rate.argsort().tolist()
        rare = [f'{ids[k]} {rate[k]:.4f}' for k in order if impl[k]][:8]
        common = [f'{ids[k]} {rate[k]:.2f}' for k in order if impl[k]][-6:]
        self.log(f'primitive label rate over train frames: mean {rate[impl].mean():.3f} | rarest: {", ".join(rare)}'
                 f' | most common: {", ".join(common)}')

    # ------------------------------------------------------------------ batches
    def batches(self, part, shuffle, epoch=0):
        """Yields (input, y, second_person, is_cube, primitive labels or None) on the GPU."""
        bs = self.arg.batch_size if part == 'train' else self.arg.test_batch_size
        if self.arg.feature_mode == 'online':
            for x, y, _ in tqdm(self.loader[part], ncols=60, desc=part, mininterval=30):
                x = x.float().to(self.dev, non_blocking=True)
                yield x, y.long().to(self.dev, non_blocking=True), self.model.second_person(x), False, None
            return
        D = self.cache[part][epoch % len(self.cache[part])] if part == 'train' else self.cache[part]
        N = len(D['y'])
        if shuffle:
            order = torch.randperm(N, generator=torch.Generator().manual_seed(self.arg.seed * 1000 + epoch))
            stop = N - bs + 1                          # drop_last, as Neuron's train loader
        else:
            order, stop = torch.arange(N), N
        for i in range(0, stop, bs):
            # sorted inside the batch: same samples per batch in every cache_device, sequential-ish disk reads
            idx = order[i:i + bs].sort().values
            if self.where == 'disk':
                cube = torch.from_numpy(D['cube'][idx.numpy()])           # reads only these rows
            else:
                cube = D['cube'][idx.to(D['cube'].device)]
            idx_s = idx.to(D['y'].device)
            prim = D['prim'][idx_s].to(self.dev, non_blocking=True) if 'prim' in D else None
            yield (cube.to(self.dev, non_blocking=True).float(), D['y'][idx_s].to(self.dev),
                   D['person'][idx_s].to(self.dev), True, prim)

    def forward(self, inp, is_cube, classes):
        spa, tem = self.semantics(classes)
        if is_cube:
            return self.model.forward_cube(inp, spa, tem, classes)
        return self.model(inp, spa, tem, classes)

    # ------------------------------------------------------------------ train / eval
    def train_epoch(self, epoch):
        self.model.train()
        self.adjust_lr(epoch)
        lw = dict(lambda_sym=self.arg.lambda_sym, lambda_fuse=self.arg.lambda_fuse,
                  lambda_logic=self.arg.lambda_logic, lambda_prim=self.arg.lambda_prim, prim_impl=self.prim_impl)
        tot, n = {}, 0
        for inp, y, _, is_cube, prim in self.batches('train', shuffle=True, epoch=epoch):
            out = self.forward(inp, is_cube, self.cls_seen)
            L = self.model.losses(out, self.remap[y], prim=prim, **lw)
            self.opt_neu.zero_grad(set_to_none=True)
            self.opt_sym.zero_grad(set_to_none=True)
            sum(L.values()).backward()
            self.opt_neu.step()
            self.opt_sym.step()
            for k, v in L.items():
                tot[k] = tot.get(k, 0.0) + float(v.detach()) * len(y)
            n += len(y)
        return {f'L_{k}': v / n for k, v in tot.items()}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        keys = ('spa', 'tem', 's_neu', 'neu3', 's_sym', 'z', 'final') + (('oracle',) if self.need_prim else ())
        acc = {k: [] for k in keys}
        ys, p_pred, p_tgt = [], [], []
        for inp, y, person, is_cube, prim in self.batches('test', shuffle=False):
            out = self.forward(inp, is_cube, self.cls_all)
            out['neu3'] = (sum(out['spa']) + sum(out['tem'])) / 6       # diagnostic: all 3 rounds of both branches
            out['spa'], out['tem'] = out['spa'][2], out['tem'][2]           # fine-grained round, as Neuron
            out['final'] = self.model.final_scores(out, person, self.cls_all)
            if prim is not None:
                tgt = prim.float() / 255
                out['oracle'] = pl.oracle_scores(self.model.reason, tgt, self.cls_all, self.prim_impl)
                p_pred.append(self.model.reason.presence(out['p_logit']).float())     # (n, 2, K)
                p_tgt.append(tgt.amax(-1))
            for k in keys:
                acc[k].append(out[k].float())
            ys.append(y)
        S = {k: torch.cat(v) for k, v in acc.items()}
        y = torch.cat(ys)

        u_mask = torch.isin(y, self.unseen_t)
        yu = y[u_mask]
        zsl_pred = {k: self.unseen_t[v[u_mask][:, self.unseen_t].argmax(1)] for k, v in S.items()}
        m = {f'zsl_{k}': float((p == yu).float().mean()) for k, p in zsl_pred.items()}
        m['zsl_neuron_or'] = float(((zsl_pred['spa'] == yu) | (zsl_pred['tem'] == yu)).float().mean())
        for k, ref in (('s_neu', 's_neu'), ('final', 'z')):
            g, s, u, h = best_gamma(S[k], y, self.cls_all, self.cls_seen, span=float(S[ref].max() - S[ref].min()))
            m.update({f'gzsl_{k}_S': s, f'gzsl_{k}_U': u, f'gzsl_{k}_H': h, f'gzsl_{k}_gamma': g})
        if p_pred:          # grounding vs labels, per primitive, on test samples of unseen / seen classes
            Pp, Pt = torch.cat(p_pred), torch.cat(p_tgt)
            for name, msk in (('unseen', u_mask), ('seen', ~u_mask)):
                auc, ok = mean_auc(Pp[msk].flatten(0, 1), Pt[msk].flatten(0, 1), self.prim_impl)
                m[f'prim_auc_{name}'] = float(auc[ok].mean()) if ok.any() else float('nan')
                m[f'prim_n_{name}'] = int(ok.sum())
                self.prim_auc[name] = {self.labeler.ids[k]: round(float(auc[k]), 4)
                                       for k in torch.nonzero(ok).flatten().tolist()}
        m.update(beta=float(self.model.beta), tau_neu=float(self.model.log_tau_neu.exp()),
                 lam_seq=float(self.model.reason.lam_seq), neuron_scale=float(self.model.log_neuron_scale.exp()))
        return m

    # ------------------------------------------------------------------ checkpoints
    def save_last(self, epoch, rows, best):
        torch.save(dict(model=self.model.state_dict(), opt_neu=self.opt_neu.state_dict(),
                        opt_sym=self.opt_sym.state_dict(), epoch=epoch, rows=rows, best=best),
                   f'{self.arg.work_dir}/last.pt.tmp')
        os.replace(f'{self.arg.work_dir}/last.pt.tmp', f'{self.arg.work_dir}/last.pt')   # no half-written file

    def try_resume(self):
        path = f'{self.arg.work_dir}/last.pt'
        if not (self.arg.resume and os.path.exists(path)):
            return 0, [], dict(value=-1.0, epoch=0)
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(ck['model'])
        self.opt_neu.load_state_dict(ck['opt_neu'])
        self.opt_sym.load_state_dict(ck['opt_sym'])
        self.log(f'resumed from {path}: {ck["epoch"]} epochs done, best {self.arg.best_metric} '
                 f'{ck["best"]["value"]:.4f} @ep{ck["best"]["epoch"]}')
        return ck['epoch'], ck['rows'], ck['best']

    def start(self):
        csv_path = f'{self.arg.work_dir}/metrics.csv'
        done, rows, best = self.try_resume()
        for epoch in range(done, self.arg.num_epoch):
            t0 = time.time()
            rec = dict(epoch=epoch + 1, **self.train_epoch(epoch))
            rec.update(self.evaluate())
            rec['time_s'] = round(time.time() - t0)
            rows.append(rec)
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            tag = ''
            if rec[self.arg.best_metric] > best['value']:
                best = dict(value=rec[self.arg.best_metric], epoch=epoch + 1)
                torch.save(dict(model=self.model.state_dict(), epoch=epoch + 1, metrics=rec),
                           f'{self.arg.work_dir}/best.pt')
                tag = ' *best'
            self.save_last(epoch + 1, rows, best)
            if self.prim_auc:
                json.dump(dict(epoch=epoch + 1, **self.prim_auc), open(f'{self.arg.work_dir}/prim_auc.json', 'w'),
                          indent=1)
            prim_s = (f'oracle {rec["zsl_oracle"]:.4f} primAUC u {rec["prim_auc_unseen"]:.3f} '
                      f's {rec["prim_auc_seen"]:.3f} | ') if 'zsl_oracle' in rec else ''
            self.log(f'ep {epoch + 1:2d} | ZSL neu {rec["zsl_s_neu"]:.4f} sym {rec["zsl_s_sym"]:.4f} '
                     f'fused {rec["zsl_z"]:.4f} final {rec["zsl_final"]:.4f} '
                     f'(spa {rec["zsl_spa"]:.4f} tem {rec["zsl_tem"]:.4f} neuron-OR {rec["zsl_neuron_or"]:.4f}) | '
                     f'H neu {rec["gzsl_s_neu_H"]:.4f} final {rec["gzsl_final_H"]:.4f} | {prim_s}'
                     f'beta {rec["beta"]:.3f} | ' + ' '.join(f'{k} {v:.3f}' for k, v in rec.items() if k.startswith('L_'))
                     + f' | {rec["time_s"]}s{tag}')

        last = rows[-1]
        best_row = rows[best['epoch'] - 1]
        per_metric = {k: max(r[k] for r in rows) for k in last if k.startswith(('zsl_', 'gzsl_')) and k.endswith(
            ('_neu', '_sym', '_z', '_final', '_or', '_spa', '_tem', '_H', '_oracle'))}
        summary = dict(split=self.arg.split, best_metric=self.arg.best_metric, best_epoch=best['epoch'],
                       best_checkpoint=best_row, last_epoch=last, best_on_test=per_metric)
        json.dump(summary, open(f'{self.arg.work_dir}/summary.json', 'w'), indent=1)
        show = lambda r: ' '.join(f'{k} {v:.4f}' for k, v in r.items() if k.startswith('zsl_') or k.endswith('_H'))
        self.log(f'last epoch     : {show(last)}')
        self.log(f'best.pt (ep {best["epoch"]}, max {self.arg.best_metric} on test): {show(best_row)}')


def get_parser():
    p = argparse.ArgumentParser(description='NeSy zero-shot skeleton action recognition on Neuron')
    p.add_argument('--config', required=True)
    p.add_argument('--work-dir', default='./work_dir/nesy')
    p.add_argument('--split', default='ntu60_seen55_unseen5', choices=list(SPLITS))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--semantic-dir', default='./semantics/ntu')
    # data
    p.add_argument('--feeder', default='feeders.feeder_ntu.Feeder_Shiftgcn_Match')
    p.add_argument('--num-worker', type=int, default=8)
    p.add_argument('--train-feeder-args', action=DictAction, default=dict())
    p.add_argument('--test-feeder-args', action=DictAction, default=dict())
    p.add_argument('--feature-mode', default='cache', choices=['cache', 'online'])
    p.add_argument('--cache-dir', default=None, help='save / reuse encoder cubes here (None: keep in memory only)')
    p.add_argument('--cache-device', default='auto', choices=['auto', 'cuda', 'cpu', 'disk'],
                   help='disk: memory-mapped .npy in cache_dir (low RAM / VRAM, e.g. T4)')
    p.add_argument('--gpu-cache-gb', type=float, default=14.0, help='auto: cubes go to the GPU below this size')
    p.add_argument('--extract-batch-size', type=int, default=256)
    p.add_argument('--cache-views', type=int, default=1,
                   help='fixed augmented views per training sample; epoch e trains on view e %% K')
    # model
    p.add_argument('--model', default='model.nesy.NeuronNeSy')
    p.add_argument('--model-args', action=DictAction, default=dict())
    p.add_argument('--weights', required=False, help='Shift-GCN checkpoint (glob pattern allowed)')
    # optim (Neuron head: SGD as in Neuron; NeSy branch: AdamW), same step decay for both
    p.add_argument('--base-lr', type=float, default=0.1)
    p.add_argument('--nesy-lr', type=float, default=1e-3)
    p.add_argument('--nesy-wd', type=float, default=1e-4)
    p.add_argument('--step', type=int, default=[4, 20], nargs='+')
    p.add_argument('--lr-decay-rate', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--nesterov', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--test-batch-size', type=int, default=64)
    p.add_argument('--num-epoch', type=int, default=30)
    # loss weights
    p.add_argument('--lambda-sym', type=float, default=1.0)
    p.add_argument('--lambda-fuse', type=float, default=1.0)
    p.add_argument('--lambda-logic', type=float, default=0.1)
    p.add_argument('--lambda-prim', type=float, default=0.0,
                   help='BCE of the grounding against the primitive labels from the definitions (primitive_labels.py)')
    p.add_argument('--prim-eval', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False,
                   help='log zsl_oracle (rules on the test labels) and the primitive AUC of the grounding')
    p.add_argument('--prim-pos-weight-max', type=float, default=10.0)
    # checkpoints
    p.add_argument('--best-metric', default='zsl_final', help='metrics.csv column used for best.pt')
    p.add_argument('--resume', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True)
    return p


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    with open(p.config) as f:
        cfg = yaml.safe_load(f)
    known = vars(p)
    bad = [k for k in cfg if k not in known]
    assert not bad, f'unknown config keys: {bad}'
    parser.set_defaults(**cfg)
    arg = parser.parse_args()
    init_seed(arg.seed)
    Processor(arg).start()
