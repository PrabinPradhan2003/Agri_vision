"""Train a transfer-learning model (EfficientNet/ResNet) on PlantVillage.

This script exports a *checkpoint dict* that the Flask app can load.

Example:
  python Model/train_transfer_learning.py --data Dataset --arch efficientnet_b0 --epochs 5 \
    --output "Flask Deployed App/plant_disease_model_tl.pt"

The exported file contains:
- arch
- num_classes
- class_names (canonical 39 order)
- state_dict
- metrics

Notes:
- Requires torch + torchvision.
- Keeps label order stable with plant_labels.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from Model.plantvillage_dataset import (
    PlantVillageDataset,
    discover_plantvillage_samples,
    split_samples_stratified,
)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * int(y.numel())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += int(y.numel())

    if total_n == 0:
        return 0.0, 0.0
    return total_loss / total_n, total_correct / total_n


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    arch = arch.lower().strip()

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError("Unsupported --arch. Use: efficientnet_b0, resnet50")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer learning training for PlantVillage")
    p.add_argument("--data", required=True, help="Path to PlantVillage dataset root (39 class folders)")
    p.add_argument("--output", required=True, help="Path to write checkpoint (.pt)")

    p.add_argument("--arch", default="efficientnet_b0", help="efficientnet_b0 | resnet50")
    p.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", default=None, help="Resume from a checkpoint or state_dict")

    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data_root = Path(args.data)
    samples, class_names = discover_plantvillage_samples(data_root)
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
    train_ds = Subset(ds, train_idxs)
    val_ds = Subset(ds, val_idxs)
    test_ds = Subset(ds, test_idxs)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.arch, num_classes, pretrained=bool(args.pretrained))
    model.to(device)

    if args.resume:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(y.numel())
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_n += int(y.numel())

        train_loss = total_loss / max(1, total_n)
        train_acc = total_correct / max(1, total_n)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{int(args.epochs)} "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "arch": args.arch,
        "pretrained": bool(args.pretrained),
        "num_classes": num_classes,
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "metrics": {"best_val_acc": best_val_acc, "test_acc": test_acc, "test_loss": test_loss},
    }

    torch.save(checkpoint, out_path)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "created_at": checkpoint["created_at"],
                "arch": checkpoint["arch"],
                "pretrained": checkpoint["pretrained"],
                "num_classes": checkpoint["num_classes"],
                "class_names": checkpoint["class_names"],
                "metrics": checkpoint["metrics"],
                "torch_version": torch.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved checkpoint: {out_path}")
    print(f"Saved meta:       {meta_path}")
    print(f"Test accuracy:    {test_acc*100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
