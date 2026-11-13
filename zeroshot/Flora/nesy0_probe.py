"""Đối chứng nesy0 (không huấn luyện Flora): đặc trưng skeleton 256 chiều có chứa thông tin primitive không?

Lý do: mọi cách đưa KG vào phía skeleton / bước quyết định đều thất bại (logic head, primitive token, chấm lại theo margin,
truy hồi + luật trong nesy4_1_1_5). File này đo trực tiếp bằng một bộ dò (probe):
  - Huấn luyện bộ phân loại đa nhãn trên đặc trưng của mẫu huấn luyện SEEN, nhãn = hàng `require` của lớp trong KG
    (primitive p có mặt nếu lớp của mẫu require p). Chỉ dùng primitive có ít nhất một lớp seen require.
  - Đo AUC từng primitive trên mẫu test UNSEEN (nhãn primitive lấy từ luật của lớp unseen thật).
    Chỉ tính primitive "đo được": vừa học được từ seen, vừa phân chia các lớp unseen (có lớp unseen có, có lớp không).
  - Tham chiếu: AUC trên mẫu test SEEN (cùng lớp đã học; mức "trần" khi không phải tổng quát hoá).
  - Đối chứng: cùng bộ dò nhưng huấn luyện với luật bị hoán vị giữa các lớp -> AUC unseen phải ~0.5.
  - So sánh: hồ sơ primitive từ truy hồi kNN (đúng như nesy4_1_1_5) trên cùng các primitive.
  - Phân loại ZSL chỉ bằng luật (DAP mờ: mean p(require) - mean p(forbid)) với xác suất của bộ dò và của kNN.

Đọc kết quả (đặt trước): AUC unseen trung bình >= 0.75 -> đặc trưng có thông tin primitive, lỗi nằm ở cách dùng;
<= 0.65 -> đặc trưng gộp gần như không mang primitive chuyển giao được sang lớp mới (hạn chế của hướng NeSy phía skeleton).

Chạy: python nesy0_probe.py --tasks 55 48 110 96                (STAR, linear)
      python nesy0_probe.py --tasks 55 48 110 96 --model mlp
      python nesy0_probe.py --feat synse --tasks 55 48 110 96
Kết quả in ra màn hình và lưu <out-root>/probe/probe_<feat>_<model>.csv.
"""
import argparse
import os

import nesy_common as C

import numpy as np

from kg_semantics import PrimitiveKG
from nesy0_shuffle import derangement


def auc(scores, labels):
    """Mann-Whitney AUC; labels in {0, 1}."""
    pos = labels > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind='mergesort')] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_probe(X, T, model, epochs, seed, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d, p = X.shape[1], T.shape[1]
    net = nn.Linear(d, p) if model == 'linear' else nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, p))
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Tt = torch.as_tensor(T, dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        net.train()
        for s in range(0, n, 1024):
            i = perm[s:s + 1024]
            opt.zero_grad()
            lossf(net(Xt[i]), Tt[i]).backward()
            opt.step()
    net.eval()

    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.as_tensor(Z, dtype=torch.float32, device=device))).cpu().numpy()
    return predict


def knn_profile(Xs, Ts, Xq, k=50, tau=0.05, device='cpu'):
    import torch
    import torch.nn.functional as F
    B = F.normalize(torch.as_tensor(Xs, dtype=torch.float32, device=device), dim=-1)
    Tb = torch.as_tensor(Ts, dtype=torch.float32, device=device)
    out = []
    for s in range(0, len(Xq), 2048):
        q = F.normalize(torch.as_tensor(Xq[s:s + 2048], dtype=torch.float32, device=device), dim=-1)
        val, idx = (q @ B.t()).topk(k, dim=1)
        w = torch.softmax(val / tau, dim=1)
        out.append(torch.einsum('bk,bkp->bp', w, Tb[idx]).cpu())
    return torch.cat(out).numpy()


def rule_accuracy(prob, req_u, forb_u, y_idx):
    """prob (N, P) over trained primitives; req_u / forb_u (U, P) masks. Fuzzy rule score, argmax over unseen."""
    r = prob @ req_u.T / np.maximum(req_u.sum(1), 1)
    f = prob @ forb_u.T / np.maximum(forb_u.sum(1), 1)
    return 100.0 * ((r - f).argmax(1) == y_idx).mean()


def run_task(short, opts, device):
    from utils.utils import task_definition
    task = C.SPLITS[short][0]
    n_cls, unseen, seen = task_definition(task)[:3]
    unseen = [int(u) for u in unseen]
    seen = [int(s) for s in seen]
    z = np.load(C.data_path(task))
    Xs, ys = np.asarray(z['train_data'], dtype=np.float32).reshape(len(z['train_label']), -1), np.asarray(z['train_label']).astype(int)
    Xu, yu = np.asarray(z['zsl_data'], dtype=np.float32).reshape(len(z['zsl_label']), -1), np.asarray(z['zsl_label']).astype(int)
    Xg, yg = np.asarray(z['gzsl_data'], dtype=np.float32).reshape(len(z['gzsl_label']), -1), np.asarray(z['gzsl_label']).astype(int)
    seen_test = np.isin(yg, seen)
    Xst, yst = Xg[seen_test], yg[seen_test]

    kg = PrimitiveKG(opts.kg_prims, opts.kg_rules, n_cls, 0.5)
    req = (kg.W > 0).astype(np.float32)
    forb = (kg.W < 0).astype(np.float32)
    trained = req[seen].any(0)                                       # primitives the probe can learn
    varies_u = req[unseen].any(0) & ~req[unseen].all(0)              # split unseen classes
    varies_s = req[seen].any(0) & ~req[seen].all(0)
    ev_u = np.nonzero(trained & varies_u)[0]
    ev_s = np.nonzero(trained & varies_s)[0]
    cols = np.nonzero(trained)[0]
    col_of = {p: i for i, p in enumerate(cols)}

    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    nz = lambda X: (X - mu) / sd
    Ts = req[ys][:, cols]
    pred = train_probe(nz(Xs), Ts, opts.model, opts.epochs, opts.seed, device)
    Pu, Pst = pred(nz(Xu)), pred(nz(Xst))
    perm = derangement(n_cls, np.random.RandomState(opts.seed + 1000))
    pred_sh = train_probe(nz(Xs), req[perm][ys][:, cols], opts.model, opts.epochs, opts.seed, device)
    Pu_sh = pred_sh(nz(Xu))
    Ku = knn_profile(Xs, Ts, Xu, device=device)

    Tu, Tst = req[yu], req[yst]
    au = {p: auc(Pu[:, col_of[p]], Tu[:, p]) for p in ev_u}
    a_sh = {p: auc(Pu_sh[:, col_of[p]], Tu[:, p]) for p in ev_u}
    a_knn = {p: auc(Ku[:, col_of[p]], Tu[:, p]) for p in ev_u}
    a_st = {p: auc(Pst[:, col_of[p]], Tst[:, p]) for p in ev_s}
    m = lambda d: float(np.nanmean(list(d.values()))) if d else float('nan')

    uidx = {c: i for i, c in enumerate(unseen)}
    y_idx = np.array([uidx[c] for c in yu])
    req_u, forb_u = req[unseen][:, cols], forb[unseen][:, cols]
    acc_probe = rule_accuracy(Pu, req_u, forb_u, y_idx)
    acc_knn = rule_accuracy(Ku, req_u, forb_u, y_idx)

    names = kg.prim_ids
    srt = sorted(au.items(), key=lambda kv: -kv[1])
    print(f'\n[probe] {opts.feat} {short} ({task}), model {opts.model}: train {len(Xs)} seen samples, '
          f'{len(cols)} trainable primitives; evaluable on unseen {len(ev_u)}, on seen test {len(ev_s)}')
    print(f'[probe]   mean AUC  unseen {m(au):.3f} | seen test (reference) {m(a_st):.3f} | unseen, shuffled-rule probe {m(a_sh):.3f} '
          f'| unseen, kNN profile {m(a_knn):.3f}')
    print(f'[probe]   unseen primitives with AUC >= 0.75: {sum(v >= 0.75 for v in au.values())}/{len(au)}; <= 0.6: {sum(v <= 0.6 for v in au.values())}/{len(au)}')
    print(f'[probe]   rule-only ZSL accuracy (chance {100 / len(unseen):.1f}%): probe {acc_probe:.2f} | kNN {acc_knn:.2f}')
    print('[probe]   best unseen primitives: ' + ', '.join(f'{names[p]} {v:.2f}' for p, v in srt[:6]))
    print('[probe]   worst unseen primitives: ' + ', '.join(f'{names[p]} {v:.2f}' for p, v in srt[-6:]))
    return dict(feat=opts.feat, split=short, model=opts.model, n_eval_unseen=len(ev_u), auc_unseen=m(au), auc_seen_test=m(a_st),
                auc_unseen_shuffled=m(a_sh), auc_unseen_knn=m(a_knn), rule_acc_probe=acc_probe, rule_acc_knn=acc_knn,
                chance=100 / len(unseen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', default='star', choices=list(C.FEAT_SPLITS))
    ap.add_argument('--tasks', nargs='+', default=['55', '48', '110', '96'])
    ap.add_argument('--model', default='linear', choices=['linear', 'mlp'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--kg-prims', default='kg/primitive_v2seq.json')
    ap.add_argument('--kg-rules', default='kg/kg_rules_v2seq.json')
    ap.add_argument('--out-root', default=C.DRIVE_OUT)
    opts = ap.parse_args()
    C.set_feat(opts.feat)
    if not (os.path.exists(opts.kg_prims) and os.path.exists(opts.kg_rules)):
        from nesy4_seqkg import build_seq_kg
        build_seq_kg()
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = [run_task(t, opts, device) for t in opts.tasks]
    import pandas as pd
    df = pd.DataFrame(rows)
    print('\n' + df.round(3).to_string(index=False))
    out = os.path.join(opts.out_root, 'probe')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'probe_{opts.feat}_{opts.model}.csv')
    df.to_csv(path, index=False)
    print(f'[probe] saved {path}')


if __name__ == '__main__':
    main()
