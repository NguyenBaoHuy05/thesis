"""Export test predictions and ensemble joint/bone checkpoints.

Examples:
  python ensemble_predictions.py extract --config config/ntu60_skateformer_logic.yaml \
      --checkpoint work_dir/ntu60_skateformer_logic_j/best_model.pt \
      --output predictions/pred_joint.pkl

  python ensemble_predictions.py ensemble \
      --joint predictions/pred_joint.pkl --bone predictions/pred_bone.pkl \
      --output predictions/ensemble.pkl --joint-weight 0.6 --bone-weight 0.4
"""

import argparse
import os
import pickle
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from feeders.feeder_ntu import NTUFeeder, custom_collate_fn
from model.thesis_model import ThesisNeurosymbolicModel


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_model(cfg, dataset, checkpoint, device):
    data = dataset.data
    channels = int(data.shape[1])
    points = int(getattr(dataset, "num_points", data.shape[3]))
    people = int(data.shape[4]) if data.ndim == 5 else 1
    backbone_cfg = cfg.setdefault("model_args", {}).setdefault("backbone_cfg", {})
    backbone_cfg["in_channels"] = channels
    backbone_cfg["num_points"] = points
    backbone_cfg["num_people"] = people

    model = ThesisNeurosymbolicModel(**cfg["model_args"], device=device).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def extract_predictions(args):
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = NTUFeeder(**cfg["test_feeder_args"])
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("test_batch_size", 32),
        shuffle=False,
        num_workers=0 if os.name == "nt" else int(cfg.get("num_worker", 0)),
        collate_fn=custom_collate_fn,
    )
    model = prepare_model(cfg, dataset, args.checkpoint, device)

    final_scores, logic_scores, labels = [], [], []
    with torch.no_grad():
        for data, batch_labels, concept_gt, _, index_t, prompts_by_group in loader:
            data = data.to(device)
            concept_gt = concept_gt.to(device)
            if index_t is not None:
                index_t = index_t.to(device)
            outputs = model(
                x=data,
                index_t=index_t,
                labels=None,
                concept_gt=concept_gt,
                prompts_by_group=prompts_by_group,
            )
            final_scores.append(outputs["y_final"].cpu().numpy())
            logic_scores.append(outputs["y_logic"].cpu().numpy())
            labels.append(batch_labels.numpy())

    final_scores = np.concatenate(final_scores).astype(np.float32)
    result = {
        "scores": final_scores,
        "probabilities": torch.softmax(torch.from_numpy(final_scores), dim=1).numpy(),
        "logic_scores": np.concatenate(logic_scores).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.int64),
        "indices": np.arange(len(final_scores), dtype=np.int64),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    accuracy = (result["probabilities"].argmax(1) == result["labels"]).mean() * 100
    print(f"[Export] Saved {len(final_scores)} predictions to {output}")
    print(f"[Export] Single-model final accuracy: {accuracy:.2f}%")


def ensemble_predictions(args):
    with open(args.joint, "rb") as handle:
        joint = pickle.load(handle)
    with open(args.bone, "rb") as handle:
        bone = pickle.load(handle)

    if not np.array_equal(joint["indices"], bone["indices"]):
        raise ValueError("Joint and bone prediction files have different sample ordering.")
    if not np.array_equal(joint["labels"], bone["labels"]):
        raise ValueError("Joint and bone prediction files have different labels/order.")

    total_weight = args.joint_weight + args.bone_weight
    if total_weight <= 0:
        raise ValueError("The sum of ensemble weights must be positive.")
    joint_prob = torch.softmax(torch.from_numpy(joint["scores"]), dim=1).numpy()
    bone_prob = torch.softmax(torch.from_numpy(bone["scores"]), dim=1).numpy()
    scores = (
        args.joint_weight * joint_prob + args.bone_weight * bone_prob
    ) / total_weight
    labels = joint["labels"]
    predictions = scores.argmax(axis=1)
    result = {
        "scores": scores.astype(np.float32),
        "probabilities": scores.astype(np.float32),
        "predictions": predictions.astype(np.int64),
        "labels": labels.astype(np.int64),
        "indices": joint["indices"],
        "weights": {"joint": args.joint_weight, "bone": args.bone_weight},
        "source_files": {"joint": args.joint, "bone": args.bone},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    accuracy = (predictions == labels).mean() * 100
    print(f"[Ensemble] Saved {len(predictions)} predictions to {output}")
    print(f"[Ensemble] Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Export and ensemble skeleton predictions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default=None)
    extract.set_defaults(function=extract_predictions)

    ensemble = subparsers.add_parser("ensemble")
    ensemble.add_argument("--joint", required=True)
    ensemble.add_argument("--bone", required=True)
    ensemble.add_argument("--output", required=True)
    ensemble.add_argument("--joint-weight", type=float, default=0.5)
    ensemble.add_argument("--bone-weight", type=float, default=0.5)
    ensemble.set_defaults(function=ensemble_predictions)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
