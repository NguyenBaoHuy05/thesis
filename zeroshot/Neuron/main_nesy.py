"""Train / evaluate the NeSy model built on Neuron (model/nesy.py).

    python main_nesy.py --config config/nesy/ntu60_xsub_55_5.yaml

Differences from main_match.py (Neuron):
  - every reported accuracy is a single prediction per sample (Neuron's "spatial OR temporal" accuracy is
    logged only as `neuron_or`, for comparison with the paper);
  - one test pass per epoch: GZSL scores on all classes; ZSL = the unseen samples restricted to the unseen
    columns (scores are per class, so this equals evaluating with the unseen classes only);
  - paths come from the config; no interactive prompt; per-epoch metrics go to <work_dir>/metrics.csv.
GZSL uses calibrated stacking with gamma searched on the test set, as Neuron does (optimistic).
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import kg as kg_lib

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
        self.load_encoder()
        self.model = self.model.to(self.dev)

        n_neu = sum(p.numel() for p in self.model.neuron_parameters())
        n_sym = sum(p.numel() for p in self.model.nesy_parameters())
        self.log(f'trainable params: Neuron head {n_neu:,} | NeSy {n_sym:,}')
        self.opt_neu = torch.optim.SGD(self.model.neuron_parameters(), lr=arg.base_lr, momentum=0.9,
                                       nesterov=arg.nesterov, weight_decay=arg.weight_decay)
        self.opt_sym = torch.optim.AdamW(self.model.nesy_parameters(), lr=arg.nesy_lr, weight_decay=arg.nesy_wd)

        Feeder = import_class(arg.feeder)
        self.loader = dict(
            train=torch.utils.data.DataLoader(Feeder(**arg.train_feeder_args), batch_size=arg.batch_size,
                                              shuffle=True, num_workers=arg.num_worker, drop_last=True,
                                              worker_init_fn=init_seed, pin_memory=True),
            test=torch.utils.data.DataLoader(Feeder(**arg.test_feeder_args), batch_size=arg.test_batch_size,
                                             shuffle=False, num_workers=arg.num_worker, pin_memory=True))
        self.log(f'split {arg.split}: {len(self.seen)} seen / {len(self.unseen)} unseen | '
                 f'train {len(self.loader["train"].dataset)} | test {len(self.loader["test"].dataset)}')

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

    def semantics(self, classes):
        return [t[classes] for t in self.sem['spatial']], [t[classes] for t in self.sem['temporal']]

    def adjust_lr(self, epoch):
        f = self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step))
        for g in self.opt_neu.param_groups:
            g['lr'] = self.arg.base_lr * f
        for g in self.opt_sym.param_groups:
            g['lr'] = self.arg.nesy_lr * f

    # ------------------------------------------------------------------ train / eval
    def train_epoch(self, epoch):
        self.model.train()
        self.adjust_lr(epoch)
        spa, tem = self.semantics(self.cls_seen)
        lw = dict(lambda_sym=self.arg.lambda_sym, lambda_fuse=self.arg.lambda_fuse,
                  lambda_logic=self.arg.lambda_logic)
        tot, n = {}, 0
        for x, y, _ in tqdm(self.loader['train'], ncols=60, desc=f'train {epoch + 1}', mininterval=10):
            x, y = x.float().to(self.dev, non_blocking=True), y.long().to(self.dev, non_blocking=True)
            out = self.model(x, spa, tem, self.cls_seen)
            L = self.model.losses(out, self.remap[y], **lw)
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
        spa, tem = self.semantics(self.cls_all)
        keys = ('spa', 'tem', 's_neu', 's_sym', 'z', 'final')
        acc = {k: [] for k in keys}
        ys = []
        for x, y, _ in tqdm(self.loader['test'], ncols=60, desc='test', mininterval=10):
            x = x.float().to(self.dev, non_blocking=True)
            out = self.model(x, spa, tem, self.cls_all)
            out['spa'], out['tem'] = out['spa'][2], out['tem'][2]           # fine-grained round, as Neuron
            out['final'] = self.model.final_scores(out, x, self.cls_all)
            for k in keys:
                acc[k].append(out[k].float())
            ys.append(y.to(self.dev))
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
        m.update(beta=float(self.model.beta), tau_neu=float(self.model.log_tau_neu.exp()),
                 lam_seq=float(self.model.reason.lam_seq))
        return m

    def start(self):
        rows, csv_path = [], f'{self.arg.work_dir}/metrics.csv'
        for epoch in range(self.arg.num_epoch):
            t0 = time.time()
            rec = dict(epoch=epoch + 1, **self.train_epoch(epoch))
            rec.update(self.evaluate())
            rec['time_s'] = round(time.time() - t0)
            rows.append(rec)
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            self.log(f'ep {epoch + 1:2d} | ZSL neu {rec["zsl_s_neu"]:.4f} sym {rec["zsl_s_sym"]:.4f} '
                     f'fused {rec["zsl_z"]:.4f} final {rec["zsl_final"]:.4f} '
                     f'(spa {rec["zsl_spa"]:.4f} tem {rec["zsl_tem"]:.4f} neuron-OR {rec["zsl_neuron_or"]:.4f}) | '
                     f'H neu {rec["gzsl_s_neu_H"]:.4f} final {rec["gzsl_final_H"]:.4f} | '
                     f'beta {rec["beta"]:.3f} | ' + ' '.join(f'{k} {v:.3f}' for k, v in rec.items() if k.startswith('L_'))
                     + f' | {rec["time_s"]}s')
        torch.save(self.model.state_dict(), f'{self.arg.work_dir}/last.pt')

        last = rows[-1]
        best = {k: max(r[k] for r in rows) for k in last if k.startswith(('zsl_', 'gzsl_')) and k.endswith(
            ('_neu', '_sym', '_z', '_final', '_or', '_spa', '_tem', '_H'))}
        summary = dict(split=self.arg.split, last_epoch=last, best_on_test=best)
        json.dump(summary, open(f'{self.arg.work_dir}/summary.json', 'w'), indent=1)
        self.log('last epoch : ' + ' '.join(f'{k} {v:.4f}' for k, v in last.items()
                                            if k.startswith('zsl_') or k.endswith('_H')))
        self.log('best-on-test (max over epochs, optimistic): '
                 + ' '.join(f'{k} {v:.4f}' for k, v in best.items()))


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
