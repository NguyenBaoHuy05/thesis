"""Train / evaluate Neuron++ (model/neuron_plus.py): Neuron's head with a stronger text side and loss.

    python main_plus.py --config config/plus/ntu60_xsub_55_5.yaml

Feature modes
  cache  (default): the frozen encoder runs ONCE. Every training sample gets ONE fixed augmentation (Neuron's
                    random crop + rotation, with the RNGs seeded by seed + sample index, so the view is the same
                    in every run); test samples use the deterministic centre crop. The encoder cubes
                    (n, 256, 16, 25, fp16) are kept on the GPU, in RAM or memory-mapped on disk
                    (cache_device; auto picks by free memory, disk fits a T4) and saved to cache_dir for
                    reuse. Cube caches written by main_nesy.py are reused (same file names and format).
  online          : Neuron's way, a new random augmentation every epoch through the encoder (slow).

Checkpoints in work_dir
  last.pt : model + optimisers + history, written every epoch; with resume=True a restarted run continues from it.
  best.pt : model at the epoch with the highest `best_metric` (default zsl_s_neu). Selected on the test set,
            like Neuron's best epoch, so it is optimistic.

Differences from main_match.py (Neuron):
  - every reported accuracy is a single prediction per sample (Neuron's "spatial OR temporal" accuracy is
    logged only as `zsl_neuron_or`, for comparison with the paper);
  - one test pass per epoch: GZSL scores on all classes; ZSL = the unseen samples restricted to the unseen
    columns (scores are per class, so this equals evaluating with the unseen classes only);
  - paths come from the config; no interactive prompt; per-epoch metrics go to <work_dir>/metrics.csv.
GZSL uses calibrated stacking with gamma searched on the test set, as Neuron does (optimistic).

eval_alphas x eval_prop_modes: extra ZSL columns `zsl_prop_<mode>_a<alpha>`: the same model scored with the
propagation of strength alpha at test time only (training-free diagnosis; picking alpha on the test set is
optimistic). graph = class graph; self and random are controls (self: only smoothing towards the own class mean;
random: random neighbours): the graph is only worth keeping if it beats both.
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
    """--model-args "agg=lse,adapter_rank=32" style overrides (same as torchlight.DictAction)."""

    def __call__(self, parser, namespace, values, option_string=None):
        d = dict(getattr(namespace, self.dest) or {})
        words = {w: w for w in ('mean', 'max', 'lse', 'graph', 'random', 'all')}   # bare words -> strings
        d.update(eval(f'dict({values})', words))
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
    span: range of the gamma grid (default: range of z)."""
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
        ma.update(n_cls_total=self.n_cls, num_class=len(self.seen))
        self.model = import_class(arg.model)(**ma)
        self.weights_path = self.load_encoder()
        self.model = self.model.to(self.dev)

        n_neu = sum(p.numel() for p in self.model.neuron_parameters())
        n_new = sum(p.numel() for p in self.model.plus_parameters())
        self.log(f'trainable params: Neuron head {n_neu:,} | new {n_new:,}')
        self.opt_neu = torch.optim.SGD(self.model.neuron_parameters(), lr=arg.base_lr, momentum=0.9,
                                       nesterov=arg.nesterov, weight_decay=arg.weight_decay)
        self.opt_new = torch.optim.AdamW(self.model.plus_parameters(), lr=arg.new_lr, weight_decay=arg.new_wd) \
            if n_new else None
        self.log('losses: ' + ' '.join(f'{k} {getattr(arg, "lambda_" + k)}' for k in ('ce', 'xnce', 'cal', 'tri'))
                 + f' | neg_mode {self.model.neg_mode} | agg {self.model.agg} | prop_alpha {self.model.prop_alpha}')

        self.Feeder = import_class(arg.feeder)
        if arg.feature_mode == 'cache':
            n = {part: self.cached_count(part) or count_samples(**self.feeder_args(part)) for part in ('train', 'test')}
            K = self.arg.cache_views
            self.where = self.cache_place(K * n['train'] + n['test'])
            # train: K fixed views per sample (epoch e uses view e % K); view 0 is the single-view cache
            self.cache = dict(train=[self.build_cache('train', n['train'], view=k) for k in range(K)],
                              test=self.build_cache('test', n['test']))
            n_tr, n_te = n['train'], n['test']
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
        if self.opt_new is not None:
            for g in self.opt_new.param_groups:
                g['lr'] = self.arg.new_lr * f

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

    # ------------------------------------------------------------------ batches
    def batches(self, part, shuffle, epoch=0):
        """Yields (input, y, is_cube) on the GPU."""
        bs = self.arg.batch_size if part == 'train' else self.arg.test_batch_size
        if self.arg.feature_mode == 'online':
            for x, y, _ in tqdm(self.loader[part], ncols=60, desc=part, mininterval=30):
                yield x.float().to(self.dev, non_blocking=True), y.long().to(self.dev, non_blocking=True), False
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
            yield cube.to(self.dev, non_blocking=True).float(), D['y'][idx.to(D['y'].device)].to(self.dev), True

    def forward(self, inp, is_cube, classes, **kw):
        spa, tem = self.semantics(classes)
        if is_cube:
            return self.model.forward_cube(inp, spa, tem, **kw)
        return self.model(inp, spa, tem)

    # ------------------------------------------------------------------ train / eval
    def train_epoch(self, epoch):
        self.model.train()
        self.adjust_lr(epoch)
        lw = dict(lambda_ce=self.arg.lambda_ce, lambda_xnce=self.arg.lambda_xnce, lambda_cal=self.arg.lambda_cal,
                  lambda_tri=self.arg.lambda_tri)
        tot, n = {}, 0
        for inp, y, is_cube in self.batches('train', shuffle=True, epoch=epoch):
            out = self.forward(inp, is_cube, self.cls_seen)
            L = self.model.losses(out, self.remap[y], **lw)
            self.opt_neu.zero_grad(set_to_none=True)
            if self.opt_new is not None:
                self.opt_new.zero_grad(set_to_none=True)
            sum(L.values()).backward()
            self.opt_neu.step()
            if self.opt_new is not None:
                self.opt_new.step()
            for k, v in L.items():
                tot[k] = tot.get(k, 0.0) + float(v.detach()) * len(y)
            n += len(y)
        return {f'L_{k}': v / n for k, v in tot.items()}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        extra = [(md, a) for md in self.arg.eval_prop_modes for a in self.arg.eval_alphas]
        pname = lambda md, a: f'prop_{md}_a{a}'
        names = ['s_neu', 'neu3', 'spa', 'tem'] + [pname(*e) for e in extra]
        acc = {k: [] for k in names}
        ys = []
        for inp, y, is_cube in self.batches('test', shuffle=False):
            kw = dict(extra=extra) if (extra and is_cube) else {}
            out = self.forward(inp, is_cube, self.cls_all, **kw)
            cur = dict(s_neu=self.model.s_neu(out), neu3=(sum(out['spa']) + sum(out['tem'])) / 6,
                       spa=out['spa'][2], tem=out['tem'][2])
            for e in extra:
                cur[pname(*e)] = out['extra'][e] if 'extra' in out else cur['s_neu']
            for k in names:
                acc[k].append(cur[k].float())
            ys.append(y)
        S = {k: torch.cat(v) for k, v in acc.items()}
        y = torch.cat(ys)

        u_mask = torch.isin(y, self.unseen_t)
        yu = y[u_mask]
        zsl_pred = {k: self.unseen_t[v[u_mask][:, self.unseen_t].argmax(1)] for k, v in S.items()}
        m = {f'zsl_{k}': float((p == yu).float().mean()) for k, p in zsl_pred.items()}
        m['zsl_neuron_or'] = float(((zsl_pred['spa'] == yu) | (zsl_pred['tem'] == yu)).float().mean())
        for k in ['s_neu'] + [pname(*e) for e in extra]:
            g, s, u, h = best_gamma(S[k], y, self.cls_all, self.cls_seen)
            m.update({f'gzsl_{k}_S': s, f'gzsl_{k}_U': u, f'gzsl_{k}_H': h, f'gzsl_{k}_gamma': g})
        # per unseen class accuracy of s_neu (which classes fail)
        pc = {int(c): float((zsl_pred['s_neu'][yu == c] == c).float().mean()) for c in self.unseen_t.tolist()}
        self.per_class = {str(c): round(v, 4) for c, v in pc.items()}
        m['ce_scale'] = float(self.model.log_ce_scale.exp())
        return m

    # ------------------------------------------------------------------ checkpoints
    def save_last(self, epoch, rows, best):
        torch.save(dict(model=self.model.state_dict(), opt_neu=self.opt_neu.state_dict(),
                        opt_new=self.opt_new.state_dict() if self.opt_new is not None else None,
                        epoch=epoch, rows=rows, best=best), f'{self.arg.work_dir}/last.pt.tmp')
        os.replace(f'{self.arg.work_dir}/last.pt.tmp', f'{self.arg.work_dir}/last.pt')   # no half-written file

    def try_resume(self):
        path = f'{self.arg.work_dir}/last.pt'
        if not (self.arg.resume and os.path.exists(path)):
            return 0, [], dict(value=-1.0, epoch=0)
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(ck['model'])
        self.opt_neu.load_state_dict(ck['opt_neu'])
        if self.opt_new is not None and ck.get('opt_new') is not None:
            self.opt_new.load_state_dict(ck['opt_new'])
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
                json.dump(dict(epoch=epoch + 1, zsl_acc_per_unseen_class=self.per_class),
                          open(f'{self.arg.work_dir}/per_class_best.json', 'w'), indent=1)
                tag = ' *best'
            self.save_last(epoch + 1, rows, best)
            prop = ' '.join(f'{md[0]}[' + ' '.join(f'{rec[f"zsl_prop_{md}_a{a}"]:.3f}' for a in self.arg.eval_alphas) + ']'
                            for md in self.arg.eval_prop_modes)
            self.log(f'ep {epoch + 1:2d} | ZSL neu {rec["zsl_s_neu"]:.4f} neu3 {rec["zsl_neu3"]:.4f} '
                     f'(spa {rec["zsl_spa"]:.4f} tem {rec["zsl_tem"]:.4f} neuron-OR {rec["zsl_neuron_or"]:.4f})'
                     + (f' | prop a={self.arg.eval_alphas} {prop}' if prop else '') + f' | H neu {rec["gzsl_s_neu_H"]:.4f} | '
                     f'scale {rec["ce_scale"]:.2f} | ' + ' '.join(f'{k} {v:.3f}' for k, v in rec.items()
                                                                  if k.startswith('L_'))
                     + f' | {rec["time_s"]}s{tag}')

        last = rows[-1]
        best_row = rows[best['epoch'] - 1]
        per_metric = {k: max(r[k] for r in rows) for k in last if k.startswith(('zsl_', 'gzsl_'))}
        summary = dict(split=self.arg.split, best_metric=self.arg.best_metric, best_epoch=best['epoch'],
                       best_checkpoint=best_row, last_epoch=last, best_on_test=per_metric)
        json.dump(summary, open(f'{self.arg.work_dir}/summary.json', 'w'), indent=1)
        show = lambda r: ' '.join(f'{k} {v:.4f}' for k, v in r.items() if k.startswith('zsl_') or k.endswith('_H'))
        self.log(f'last epoch     : {show(last)}')
        self.log(f'best.pt (ep {best["epoch"]}, max {self.arg.best_metric} on test): {show(best_row)}')


def get_parser():
    p = argparse.ArgumentParser(description='Neuron++ zero-shot skeleton action recognition')
    p.add_argument('--config', required=True)
    p.add_argument('--work-dir', default='./work_dir/plus')
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
    p.add_argument('--model', default='model.neuron_plus.NeuronPlus')
    p.add_argument('--model-args', action=DictAction, default=dict())
    p.add_argument('--weights', required=False, help='Shift-GCN checkpoint (glob pattern allowed)')
    # optim (Neuron head: SGD as in Neuron; new parameters: AdamW), same step decay for both
    p.add_argument('--base-lr', type=float, default=0.1)
    p.add_argument('--new-lr', type=float, default=1e-3)
    p.add_argument('--new-wd', type=float, default=1e-4)
    p.add_argument('--step', type=int, default=[4, 20], nargs='+')
    p.add_argument('--lr-decay-rate', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--nesterov', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--test-batch-size', type=int, default=64)
    p.add_argument('--num-epoch', type=int, default=30)
    # loss weights (see model/neuron_plus.py)
    p.add_argument('--lambda-ce', type=float, default=1.0)
    p.add_argument('--lambda-xnce', type=float, default=0.0)
    p.add_argument('--lambda-cal', type=float, default=0.0)
    p.add_argument('--lambda-tri', type=float, default=0.0)
    # evaluation
    p.add_argument('--eval-alphas', type=float, nargs='*', default=[],
                   help='extra ZSL columns with propagation of these strengths at test time only')
    p.add_argument('--eval-prop-modes', nargs='*', default=['graph', 'self', 'random'],
                   choices=['graph', 'self', 'random'], help='propagation modes of the extra columns (self / random: controls)')
    # checkpoints
    p.add_argument('--best-metric', default='zsl_s_neu', help='metrics.csv column used for best.pt')
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
