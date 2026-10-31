"""Sanity check of the SLiM encoder (KAIST-VICLab/SLiM, NeurIPS 2026) on our NTU60 data,
before it is used as a concept-feature backbone.

Runs the released ntu60_xsub encoder + its trained linear probes on TEST, with two samplers:
  slim : SLiM's own test pipeline (NTU_Classification + DownstreamAugmentation)
  ours : the HyperGCN feeder of the CCP (center crop p=0.95, window 64)
The SLiM README reports 87.88 top-1. `slim` close to that means NTU60_CS.npz has SLiM's
preprocessing; `ours` close to `slim` means the CCP clips can be fed to SLiM as they are.

xFormers is not needed: the temporal-RoPE attention is re-routed to torch SDPA (same math:
memory_efficient_attention and SDPA both scale by 1/sqrt(head_dim)), and SwiGLU falls back to
SLiM's own SwiGLUFFN (same w12 / w3 weights).

Colab:
    git clone https://github.com/KAIST-VICLab/SLiM /content/SLiM
    python scripts/slim_check.py --slim_root /content/SLiM --hypergcn_root $H --data_path $D
"""
import argparse
import os
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sip.backbone import build_feeder  # noqa: E402


def get_parser():
    p = argparse.ArgumentParser(description="Check the SLiM encoder + linear probe on NTU60 TEST")
    p.add_argument("--slim_root", required=True, help="clone of github.com/KAIST-VICLab/SLiM")
    p.add_argument("--hypergcn_root", required=True)
    p.add_argument("--data_path", required=True, help="NTU60_CS.npz")
    p.add_argument("--config", default=None)
    p.add_argument("--hf_repo", default="JeonghyeokDo/SLiM")
    p.add_argument("--subfolder", default="ntu60_xsub")
    p.add_argument("--probe", default=None, help="probe file in the HF repo (default: the .pth under <subfolder>/linear/)")
    p.add_argument("--sampling", nargs="+", default=["slim", "ours"], choices=["slim", "ours"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_worker", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p


def import_slim(slim_root):
    os.environ["XFORMERS_DISABLED"] = "1"   # read by slim.layers at import time
    sys.path.insert(0, slim_root)
    from slim.layers import attention as A

    def rope_sdpa(self, x, num_t, attn_bias=None):
        B, N, C = x.shape
        q, k, v = torch.unbind(self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads), 2)
        g = 1 + self.num_registers   # CLS + registers get no RoPE
        cos, sin = self.rope(num_t, (N - g) // num_t, x.device, x.dtype)
        qd, kd = A.apply_rotary_pos_emb(q[:, g:], k[:, g:], cos, sin)
        q, k = torch.cat([q[:, :g], qd], 1), torch.cat([k[:, :g], kd], 1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))

    A.MemEffAttention_RoPE.forward = rope_sdpa
    from slim.hub import SLiMEncoder
    return SLiMEncoder


def load_probes(arg):
    """{name: (n_blocks, avgpool, W, b)} from the released DINOv2-style classifier grid."""
    from huggingface_hub import hf_hub_download, list_repo_files
    name = arg.probe
    if name is None:
        files = list_repo_files(arg.hf_repo)
        cands = [f for f in files if f.startswith(arg.subfolder + "/linear/") and f.endswith(".pth")]
        if not cands:
            raise SystemExit("no probe under {}/linear/ in {}; files: {}".format(arg.subfolder, arg.hf_repo, files))
        name = cands[0]
    ck = torch.load(hf_hub_download(arg.hf_repo, name), map_location="cpu", weights_only=True)
    state = ck["model"] if "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    probes = {}
    for n in sorted({k.split(".")[1] for k in state}):
        f = n.split("_")   # classifier_<n>_blocks_avgpool_<bool>_lr_...
        probes[n] = (int(f[1]), f[4] == "True", state["classifiers_dict.{}.linear.weight".format(n)].float(),
                     state["classifiers_dict.{}.linear.bias".format(n)].float())
    print("[probe] {}: {} classifiers".format(name, len(probes)))
    return probes


@torch.no_grad()
def slim_features(model, x, person):
    """Last-4-block (patch tokens, CLS) summed over valid persons, as in slim/eval/linear.py."""
    B, M = person.shape
    with torch.autocast("cuda", dtype=torch.float16):
        feats = model.get_intermediate_layers(x, 4, return_class_token=True)
    w = person.float()
    return [((p.float().view(B, M, *p.shape[1:]) * w[:, :, None, None]).sum(1),
             (c.float().view(B, M, -1) * w[:, :, None]).sum(1)) for p, c in feats]


def linear_input(feats, n_blocks, avgpool):
    out = torch.cat([c for _, c in feats[-n_blocks:]], -1)
    if avgpool:
        out = torch.cat([out, feats[-1][0].mean(1)], -1)
    return out


def make_loader(kind, arg):
    if kind == "slim":
        from slim.data.augmentations import DownstreamAugmentation
        from slim.data.datasets.ntu import NTU_Classification
        root = tempfile.mkdtemp()
        os.symlink(os.path.abspath(arg.data_path), os.path.join(root, "NTU60_XSub.npz"))   # name SLiM expects
        ds = NTU_Classification(root_path=root, split="test", num_classes=60, benchmark="xsub", min_buffer=150,
                                transform=DownstreamAugmentation(out_size=64, split="test"))
        unpack = lambda b: (b[0], b[1], b[2])
    else:
        ds = build_feeder(arg.hypergcn_root, arg.data_path, "test", config=arg.config)
        unpack = lambda b: (b[0], b[0].abs().sum((1, 2, 3)) > 1e-5, b[1])
    loader = torch.utils.data.DataLoader(ds, batch_size=arg.batch_size, shuffle=False,
                                         num_workers=arg.num_worker, pin_memory=True)
    return loader, unpack


def run(kind, model, probes, arg):
    loader, unpack = make_loader(kind, arg)
    dev = arg.device
    W = {n: (nb, ap, w.to(dev), b.to(dev)) for n, (nb, ap, w, b) in probes.items()}
    correct = {n: 0 for n in W}
    total, t0 = 0, time.time()
    for b, batch in enumerate(loader):
        if arg.max_batches is not None and b >= arg.max_batches:
            break
        x, person, y = unpack(batch)
        x, person, y = x.float().to(dev), person.to(dev), y.long().to(dev)
        feats = slim_features(model, x, person)
        if b == 0:
            pt = feats[-1][0]
            print("[{}] x {} -> patch tokens {} (= 8 x 25 grid of dim {}), CLS {}".format(
                kind, tuple(x.shape), tuple(pt.shape), pt.shape[-1], tuple(feats[-1][1].shape)))
        for n, (nb, ap, w, bias) in W.items():
            correct[n] += int((F.linear(linear_input(feats, nb, ap), w, bias).argmax(1) == y).sum())
        total += len(y)
        if (b + 1) % 50 == 0:
            print("[{}] batch {}/{} {:.0f}s".format(kind, b + 1, len(loader), time.time() - t0))
    acc = {n: c / max(total, 1) for n, c in correct.items()}
    ranked = sorted(acc, key=acc.get, reverse=True)
    print("[{}] {} clips; best {} = {:.4f}".format(kind, total, ranked[0], acc[ranked[0]]))
    for n in ranked[1:4]:
        print("    {} = {:.4f}".format(n, acc[n]))
    return acc[ranked[0]]


def main():
    arg = get_parser().parse_args()
    SLiMEncoder = import_slim(arg.slim_root)
    model = SLiMEncoder.from_pretrained(arg.hf_repo, subfolder=arg.subfolder).eval().to(arg.device)
    print("[slim] {} / {}: {:.2f}M params".format(arg.hf_repo, arg.subfolder,
                                                 sum(p.numel() for p in model.parameters()) / 1e6))
    probes = load_probes(arg)
    res = {kind: run(kind, model, probes, arg) for kind in arg.sampling}
    print("[done] top-1 (README 0.8788):", {k: round(v, 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
