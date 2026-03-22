"""Train the repo's CNN on the PlantVillage dataset.

Usage (from repo root):

  python Model/train_plantvillage.py --data Dataset --epochs 10 \
    --output "Flask Deployed App/plant_disease_model_1_latest.pt"

Notes:
- `--data` should point to a folder with 39 class subfolders.
- The class index order is fixed to match `Flask Deployed App/disease_info.csv`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    return float(correct) / float(y.numel())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()

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


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data_root = Path(args.data)
    samples, class_names = discover_plantvillage_samples(data_root)

    num_classes = len(class_names)
    if num_classes != 39:
        raise ValueError(f"Expected 39 classes, found {num_classes}")

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

    CNN = _import_cnn_module()
    model: nn.Module = CNN.CNN(num_classes)
    model.to(device)

    if args.resume:
        resume_path = Path(args.resume)
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            running_acc += _accuracy(logits.detach(), y)
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{int(args.epochs)} "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate the best model on test split (if any)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "PlantVillage",
        "data_root": str(data_root.resolve()),
        "model": "CNN",
        "num_classes": num_classes,
        "class_names": class_names,
        "split": {
            "train": len(train_idxs),
            "val": len(val_idxs),
            "test": len(test_idxs),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "seed": int(args.seed),
        },
        "metrics": {
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
        },
        "device": str(device),
        "torch_version": torch.__version__,
    }

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved weights: {out_path}")
    print(f"Saved meta:    {meta_path}")
    print(f"Test accuracy: {test_acc*100:.2f}%")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN on PlantVillage dataset")
    p.add_argument("--data", required=True, help="Path to PlantVillage dataset root (39 class folders)")
    p.add_argument("--output", required=True, help="Where to save .pt state_dict")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--resume", default=None, help="Optional path to .pt weights to resume from")

    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main(sys.argv[1:])
