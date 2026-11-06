"""Training entrypoint cho GLIR-HyperGCN — hien thuc hoa 2 cau hinh V2 va V3
tu Architecture/10_CLIR-SkateFormer_v2_Proposal.md (Muc 7):

  V2 = G1 (BodyPartGroundedConceptHeads) + G2 (LightConceptGrounding)
       — lich huan luyen 2-phase ROI RAC (model.set_phase), giong CLIR goc.
  V3 = G1 + G2 + G3 (annealing LIEN TUC — model.set_progress thay set_phase)

Chay:
    python train.py --variant v2 --config config/ntu60_cs/glir_hypergcn.yaml
    python train.py --variant v3 --config config/ntu60_cs/glir_hypergcn.yaml

Can bo du lieu NTU RGB+D da tien xu ly (xem CLIR-SkateFormer/feeders/feeder_ntu.py
va concepts/ntu_rgbd_60_120.csv, da duoc copy nguyen ven vao repo nay) — script nay
CHUA duoc chay full-training trong moi truong phat trien (khong co du lieu that o
day), chi da xac nhan kien truc dung qua verify_model.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.glir_hypergcn import GLIRHyperGCN


def load_concept_names(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    return header[1:]  # bo cot 'action_class'


def build_model(cfg: dict, concept_names) -> GLIRHyperGCN:
    return GLIRHyperGCN(
        num_classes=cfg["num_classes"],
        num_person=cfg.get("num_person", 2),
        num_point=cfg.get("num_point", 25),
        embedding_channels=cfg.get("embedding_channels", 128),
        backbone_weights=cfg.get("backbone_weights"),
        concept_names=concept_names,
        num_loops=cfg.get("num_loops", 2),
        beta_max=cfg.get("beta_max", 0.5),
        beta_init=cfg.get("beta_init", 0.1),
        literal_mode=cfg.get("literal_mode", "soft"),
        use_grounding=cfg.get("use_grounding", True),
        clip_backbone=cfg.get("clip_backbone", "ViT-B/32"),
        lambda_ground=cfg.get("lambda_ground", 0.1),
    )


class GLIRTrainer:
    """Tuong tu CLIRTrainer trong CLIR-SkateFormer/generate_notebook.py, nhung ho tro
    ca 2 lich huan luyen (V2: set_phase roi rac, V3: set_progress lien tuc)."""

    def __init__(self, cfg: dict, variant: str, train_loader, test_loader):
        assert variant in ("v2", "v3")
        self.cfg = cfg
        self.variant = variant
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        concept_names = load_concept_names(cfg["concepts_csv"])
        self.model = build_model(cfg, concept_names).to(self.device)

        self.best_acc = 0.0
        self.global_step = 0

        base_lr = cfg["base_lr"]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=cfg.get("weight_decay", 0.05))

        if self.variant == "v2":
            self.model.set_phase(1)
        else:
            self.model.set_progress(0.0)

    def _set_epoch_schedule(self, epoch: int, num_epoch: int):
        if self.variant == "v2":
            phase1_epochs = self.cfg.get("phase1_epochs", num_epoch // 2)
            new_phase = 1 if epoch < phase1_epochs else 2
            current = getattr(self.model.reasoner, "literal_mode", "soft")
            want = "soft" if new_phase == 1 else "ste"
            if current != want:
                print(f"[V2] Chuyen Phase {new_phase} tai epoch {epoch + 1} (dot ngot, dung lam baseline so sanh voi V3)")
                self.model.set_phase(new_phase)
        else:
            progress = epoch / max(1, num_epoch - 1)
            self.model.set_progress(progress)

    def train_epoch(self, epoch: int, num_epoch: int):
        self.model.train()
        self._set_epoch_schedule(epoch, num_epoch)
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"[{self.variant.upper()}] Epoch {epoch + 1}/{num_epoch}")
        for batch in pbar:
            if len(batch) == 5:
                data, _index_t, label, _idx, c_targets = batch
            else:
                data, _index_t, label, _idx = batch[:4]
                c_targets = None

            data = data.float().to(self.device)
            label = label.long().to(self.device)
            c_targets = c_targets.float().to(self.device) if c_targets is not None else None

            self.optimizer.zero_grad()
            out = self.model(data, concept_targets=c_targets, action_labels=label)
            loss = torch.nn.functional.cross_entropy(out["action_logits"], label)
            if "aux_loss" in out:
                loss = loss + out["aux_loss"]
            loss.backward()
            if self.cfg.get("grad_clip", 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.optimizer.step()
            self.global_step += 1

            total_loss += loss.item()
            pred = out["action_logits"].argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total * 100:.2f}%"})

        return total_loss / max(1, len(self.train_loader)), correct / max(1, total)

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()
        correct, total = 0, 0
        for batch in tqdm(self.test_loader, desc="Eval"):
            data, _index_t, label = batch[0].float().to(self.device), batch[1], batch[2].long().to(self.device)
            out = self.model(data)
            logits = out["action_logits"] if isinstance(out, dict) else out
            pred = logits.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        return correct / max(1, total)

    def _build_checkpoint(self, epoch: int) -> dict:
        return {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "variant": self.variant,
            "best_acc": self.best_acc,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def run(self):
        num_epoch = self.cfg["num_epoch"]
        work_dir = self.cfg["work_dir"]
        os.makedirs(work_dir, exist_ok=True)
        for epoch in range(num_epoch):
            train_loss, train_acc = self.train_epoch(epoch, num_epoch)
            torch.save(self._build_checkpoint(epoch), os.path.join(work_dir, "checkpoint_latest.pt"))

            if (epoch + 1) % self.cfg.get("eval_interval", 5) == 0 or (epoch + 1) == num_epoch:
                val_acc = self.eval_epoch()
                print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Val Acc: {val_acc * 100:.2f}%")
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    torch.save(self.model.state_dict(), os.path.join(work_dir, "best_model.pt"))
                    torch.save(self._build_checkpoint(epoch), os.path.join(work_dir, "best_checkpoint.pt"))
                    print(f"  * Saved best (acc={self.best_acc * 100:.2f}%)")


def build_dataloaders(cfg: dict):
    from feeders.feeder_ntu import Feeder  # noqa: import cuc bo, can du lieu that

    train_set = Feeder(**cfg["train_feeder_args"])
    test_set = Feeder(**cfg["test_feeder_args"])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.get("test_batch_size", cfg["batch_size"]), shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v2", "v3"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("concepts_csv", str(ROOT_DIR / "concepts" / "ntu_rgbd_60_120.csv"))
    cfg.setdefault("work_dir", str(ROOT_DIR / "work_dir" / args.variant))

    train_loader, test_loader = build_dataloaders(cfg)
    trainer = GLIRTrainer(cfg, args.variant, train_loader, test_loader)
    trainer.run()


if __name__ == "__main__":
    main()
