"""Generate an evaluation report (confusion matrix + per-class metrics).

This is designed for your final-year report/viva.

Example:
  python Model/generate_evaluation_report.py --data Dataset \
    --weights "Flask Deployed App/plant_disease_model_1_latest.pt" \
    --output-dir Model/reports

Outputs (in a timestamped folder):
- confusion_matrix.csv
- metrics_per_class.csv
- summary.json

Notes:
- Supports both legacy CNN state_dict weights and TL checkpoint dict weights.
- Does not require sklearn.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from plant_labels import class_list
from Model.plantvillage_dataset import (
    PlantVillageDataset,
    discover_plantvillage_samples,
    split_samples_stratified,
)


def _import_cnn_module() -> object:
    repo_root = Path(__file__).resolve().parents[1]
    flask_dir = repo_root / "Flask Deployed App"
    if str(flask_dir) not in sys.path:
        sys.path.insert(0, str(flask_dir))
    import CNN  # type: ignore

    return CNN


def _build_transfer_model(arch: str, num_classes: int) -> nn.Module:
    from torchvision import models

    arch = (arch or "").lower().strip()
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch '{arch}'")


def load_model(weights_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = torch.load(weights_path, map_location=device)

    meta: Dict[str, Any] = {}
    if isinstance(payload, dict) and "state_dict" in payload:
        arch = str(payload.get("arch") or "")
        num_classes = int(payload.get("num_classes") or 39)
        model = _build_transfer_model(arch, num_classes)
        model.load_state_dict(payload["state_dict"])
        meta = {"type": "checkpoint", "arch": arch, "num_classes": num_classes}
        return model, meta

    CNN = _import_cnn_module()
    model = CNN.CNN(39)
    model.load_state_dict(payload)
    meta = {"type": "cnn_state_dict", "arch": "cnn", "num_classes": 39}
    return model, meta


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[int] = []
    ps: List[int] = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
        ps.extend(pred.tolist())
        ys.extend([int(v) for v in y.numpy().tolist()])

    return np.asarray(ys, dtype=int), np.asarray(ps, dtype=int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> List[Dict[str, Any]]:
    k = cm.shape[0]
    rows: List[Dict[str, Any]] = []

    for i in range(k):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class_idx": i,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return rows


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate evaluation report for PlantVillage")
    p.add_argument("--data", required=True, help="PlantVillage dataset root")
    p.add_argument("--weights", required=True, help="Path to .pt weights/checkpoint")
    p.add_argument("--output-dir", required=True, help="Directory to store reports")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")

    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    samples, class_names = discover_plantvillage_samples(Path(args.data))
    num_classes = len(class_names)

    transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_idxs, val_idxs, test_idxs = split_samples_stratified(
        samples=samples,
        num_classes=num_classes,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    ds = PlantVillageDataset(samples=samples, transform=transform)
    test_ds = Subset(ds, test_idxs)

    loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    weights_path = Path(args.weights)
    model, model_meta = load_model(weights_path, device)
    model.to(device)

    y_true, y_pred = predict_all(model, loader, device)

    cm = confusion_matrix(y_true, y_pred, k=num_classes)
    metrics = per_class_metrics(cm)

    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0

    out_root = Path(args.output_dir)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"report_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix CSV
    cm_path = out_dir / "confusion_matrix.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + list(range(num_classes)))
        for i in range(num_classes):
            w.writerow([i] + cm[i, :].tolist())

    # Save per-class metrics CSV
    m_path = out_dir / "metrics_per_class.csv"
    with m_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_idx", "class_name", "precision", "recall", "f1", "support"])
        w.writeheader()
        for row in metrics:
            i = int(row["class_idx"])
            w.writerow(
                {
                    "class_idx": i,
                    "class_name": class_names[i],
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1": f"{row['f1']:.6f}",
                    "support": row["support"],
                }
            )

    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": str(Path(args.data).resolve()),
        "weights": str(weights_path.resolve()),
        "num_classes": num_classes,
        "accuracy": accuracy,
        "model": model_meta,
        "splits": {"val_ratio": float(args.val_ratio), "test_ratio": float(args.test_ratio), "seed": int(args.seed)},
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved report to: {out_dir}")
    print(f"Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
