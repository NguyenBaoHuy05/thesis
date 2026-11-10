"""Hướng 1: suy luận primitive ở bước quyết định (NeSy decision).

Neural: một MLP nhỏ đọc feature skeleton 256-d của Flora và dự đoán xác suất của từng primitive.
  Nhãn chỉ lấy từ KG trên lớp seen: require = 1, forbid = 0, còn lại = 0 với trọng số nhỏ
  (--unk-weight), vì luật không liệt kê một primitive không có nghĩa là nó bị cấm.
  Primitive không lớp seen nào require thì không học được, nên bị bỏ khỏi điểm logic.
Symbolic: điểm của lớp c là log-likelihood (logic mềm, t-norm tích) rằng luật của c được thoả:
  logic(c) = mean_{p in require(c)} log p_hat + w_f * mean_{p in forbid(c)} log(1 - p_hat)
Quyết định: Flora lấy argmin MSE vận tốc. Ở đây
  score(c) = z(-MSE_c) + lambda * z(logic(c)),  z = chuẩn hoá theo từng mẫu trên các lớp unseen.

Đầu primitive train trước Flora (feature cố định nên độc lập), Flora train như cũ.
Mỗi lần đánh giá log thêm accuracy với nhiều lambda ([lam0] = Flora thuần, cùng checkpoint),
logic một mình, và độ chính xác dự đoán primitive trên lớp seen / unseen.

Chạy: python nesy1_logic.py            (48/12 + 110/10, seed 1, nền compose)
      python nesy1_logic.py --tasks 48 55 110 96 --seeds 1 2 3 --logic-lambda 0.5
"""
import nesy_common as C

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import Trainer
from kg_semantics import PrimitiveKG

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def flow_errors(model, x, classes, attuned):
    """(B, len(classes)) velocity MSE; same computation as MappingNet.predict_zsl (use_features=True)."""
    args = model.args
    B = x.shape[0]
    ske = x.unsqueeze(1).expand(-1, args.max_length + 1, -1)
    z1, _, _ = model.motion_encoder(ske)
    z0, _, _ = model.semantic_encoder(attuned[classes])
    t = torch.ones(B, device=x.device) * args.step_ratio
    errs = []
    for i in range(len(classes)):
        z0c = z0[i].unsqueeze(0).expand(B, -1, -1)
        zt = model.psi(t, z0c, z1)
        pv = model.velocity_predictor(zt, t)
        tv = model.Dt_psi(t, z0c, z1)
        errs.append(F.mse_loss(pv, tv, reduction='none').mean(2).mean(1))
    return torch.stack(errs, 1)


def zrow(s):
    return (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)


class PrimitiveHead(nn.Module):
    def __init__(self, d_in, n_prim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_prim))

    def forward(self, x):
        return self.net(x)


class LogicTrainer(Trainer):
    def __init__(self, args):
        assert args.use_features and args.setting == 'ZSL', 'nesy1 supports ZSL on pre-extracted features'
        super().__init__(args)
        self.kg = PrimitiveKG(args.kg_prims, args.kg_rules, self.num_classes, args.kg_forbid_weight)
        W = torch.from_numpy(self.kg.W).cuda()                                    # (n, P): +1 require, <0 forbid
        seen = torch.tensor(self.seen_classes, device='cuda')
        self.trainable = (W[seen] > 0).any(0)                                     # (P,) has a seen positive
        self.req = ((W > 0) & self.trainable).float()                             # (n, P)
        self.forb = ((W < 0) & self.trainable).float()
        self.target = (W > 0).float()
        self.weight = torch.where(W != 0, torch.ones_like(W), torch.full_like(W, args.unk_weight))
        self.head = None   # built in train_head, under a forked RNG
        self.logger.info(f'[nesy1] {int(self.trainable.sum())}/{W.shape[1]} primitives trainable on seen classes; '
                         f'unseen classes with no trainable require: '
                         f'{int((self.req[self.unseen_classes].sum(1) == 0).sum())}')

    # ---- neural part
    def train_head(self):
        ds = self.data_loader['train'].dataset
        X = torch.from_numpy(np.asarray(ds.data)).float().cuda()
        Y = torch.from_numpy(np.asarray(ds.label)).long().cuda()
        torch.manual_seed(self.args.seed)
        self.head = PrimitiveHead(X.shape[1], self.kg.W.shape[1], self.args.head_hidden, self.args.head_dropout).cuda()
        opt = torch.optim.AdamW(self.head.parameters(), lr=self.args.head_lr, weight_decay=1e-4)
        g = torch.Generator(device='cuda').manual_seed(self.args.seed)
        self.head.train()
        for it in range(1, self.args.head_iters + 1):
            idx = torch.randint(0, X.shape[0], (self.args.head_batch,), device='cuda', generator=g)
            logits = self.head(X[idx])
            y = Y[idx]
            loss = F.binary_cross_entropy_with_logits(logits, self.target[y], weight=self.weight[y] * self.trainable,
                                                      reduction='sum') / (self.weight[y] * self.trainable).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 500 == 0:
                self.logger.info(f'[nesy1] head iter {it}: BCE {loss.item():.4f}')
        self.head.eval()
        self.log_primitive_quality()

    @torch.no_grad()
    def log_primitive_quality(self):
        """Accuracy (threshold 0.5) on the primitives a class's rule states (require/forbid), separately for
        seen-class test samples and unseen-class test samples. Unseen = does the primitive knowledge transfer."""
        ds = self.gzsl_loader.dataset
        X = torch.from_numpy(np.asarray(ds.data)).float().cuda()
        Y = torch.from_numpy(np.asarray(ds.label)).long().cuda()
        pred = torch.sigmoid(self.head(X)) > 0.5
        known = ((self.req + self.forb)[Y] > 0)
        correct = (pred == (self.target[Y] > 0)) & known
        is_unseen = torch.zeros(self.num_classes, dtype=torch.bool, device='cuda')
        is_unseen[self.unseen_classes] = True
        for name, m in (('seen', ~is_unseen[Y]), ('unseen', is_unseen[Y])):
            k = known[m].sum().item()
            acc = 100 * correct[m].sum().item() / max(k, 1)
            pos = known[m] & (self.target[Y][m] > 0)
            rec = 100 * (correct[m] & pos).sum().item() / max(pos.sum().item(), 1)
            self.logger.info(f'[nesy1] primitive acc on {name}-class test: {acc:.2f}% (require recall {rec:.2f}%)')

    # ---- symbolic part
    def logic_scores(self, x, classes):
        z = self.head(x)
        logp, log1mp = F.logsigmoid(z), F.logsigmoid(-z)
        req, forb = self.req[classes], self.forb[classes]
        s_req = logp @ req.T / req.sum(1).clamp(min=1)
        s_forb = log1mp @ forb.T / forb.sum(1).clamp(min=1)
        return s_req + self.args.forbid_logic_weight * s_forb

    # ---- training / evaluation
    def train(self):
        # fork the RNG so Flora's own training sees the same random stream as the baseline run
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.train_head()
        super().train()

    @torch.no_grad()
    def test_zsl(self, nb_iter):
        self.mapping_model.eval()
        lams = sorted(set(LAMBDAS) | {self.args.logic_lambda})
        correct = {l: 0 for l in lams}
        correct_logic, total = 0, 0
        for gt_motion, label_idx, _, _, _ in self.zsl_loader:
            x = gt_motion.cuda().float()
            y = torch.tensor([self.unseen_classes.index(int(l)) for l in label_idx], device='cuda')
            s_flow = zrow(-flow_errors(self.mapping_model, x, self.unseen_classes, self.attuned_semantics))
            s_logic = zrow(self.logic_scores(x, self.unseen_classes))
            for l in lams:
                correct[l] += ((s_flow + l * s_logic).argmax(1) == y).sum().item()
            correct_logic += (s_logic.argmax(1) == y).sum().item()
            total += x.shape[0]
        acc = {l: 100 * c / total for l, c in correct.items()}
        main_acc = acc[self.args.logic_lambda]
        self.best_accuracy[0] = max(self.best_accuracy[0], main_acc)
        self.logger.info(f"Iter {nb_iter}: ZSL Accuracy: {main_acc:.2f}% ")
        for l in lams:
            self.logger.info(f"Iter {nb_iter}: [lam{l:g}] ZSL Accuracy: {acc[l]:.2f}% ")
        self.logger.info(f"Iter {nb_iter}: [logic-only] ZSL Accuracy: {100 * correct_logic / total:.2f}% ")
        self.mapping_model.train()


if __name__ == '__main__':
    C.main('logic', LogicTrainer,
           method_args=[
               ('--logic-lambda', dict(type=float, default=0.5, help='weight of the logic score in the decision')),
               ('--forbid-logic-weight', dict(type=float, default=0.5, help='w_f: weight of forbid terms in logic(c)')),
               ('--unk-weight', dict(type=float, default=0.1, help='BCE weight of primitives a rule does not mention (target 0)')),
               ('--head-iters', dict(type=int, default=3000)),
               ('--head-batch', dict(type=int, default=256)),
               ('--head-lr', dict(type=float, default=1e-3)),
               ('--head-hidden', dict(type=int, default=512)),
               ('--head-dropout', dict(type=float, default=0.3)),
           ],
           label_keys=[('logic_lambda', 'lam'), ('forbid_logic_weight', 'wf'), ('unk_weight', 'unk')])
