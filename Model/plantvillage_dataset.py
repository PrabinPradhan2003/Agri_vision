"""Utilities to load the PlantVillage dataset with a stable class index order.

Why this file exists:
- `torchvision.datasets.ImageFolder` assigns indices alphabetically based on
  folder names. That is NOT guaranteed to match this repo's `disease_info.csv`
  ordering.
- The Flask app expects indices 0..38 to match the CSV rows.

This loader enforces the canonical class list from `plant_labels.IDX_TO_CLASS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from plant_labels import class_list, match_expected_to_available


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def _list_class_dirs(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def discover_plantvillage_samples(dataset_root: Path) -> Tuple[List[Sample], List[str]]:
    """Discover image samples and return (samples, class_names_in_index_order)."""

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {dataset_root}")

    expected = class_list()
    available = _list_class_dirs(dataset_root)

    matches = match_expected_to_available(expected=expected, available=available)

    samples: List[Sample] = []
    for idx, m in enumerate(matches):
        class_dir = dataset_root / m.found
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class folder missing: {class_dir}")

        count = 0
        for img_path in _iter_images(class_dir):
            samples.append(Sample(path=img_path, label=idx))
            count += 1

        if count == 0:
            raise ValueError(f"No images found under class folder: {class_dir}")

    return samples, expected


def split_samples_stratified(
    samples: Sequence[Sample],
    num_classes: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return (train_idxs, val_idxs, test_idxs) into the `samples` list."""

    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    import random

    rng = random.Random(seed)

    by_class: List[List[int]] = [[] for _ in range(num_classes)]
    for i, s in enumerate(samples):
        by_class[s.label].append(i)

    train_idxs: List[int] = []
    val_idxs: List[int] = []
    test_idxs: List[int] = []

    for c in range(num_classes):
        idxs = by_class[c]
        rng.shuffle(idxs)

        n = len(idxs)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        test_part = idxs[:n_test]
        val_part = idxs[n_test : n_test + n_val]
        train_part = idxs[n_test + n_val :]

        # Ensure each class contributes at least 1 training sample.
        if len(train_part) == 0 and len(val_part) > 0:
            train_part.append(val_part.pop())
        if len(train_part) == 0 and len(test_part) > 0:
            train_part.append(test_part.pop())

        train_idxs.extend(train_part)
        val_idxs.extend(val_part)
        test_idxs.extend(test_part)

    rng.shuffle(train_idxs)
    rng.shuffle(val_idxs)
    rng.shuffle(test_idxs)

    return train_idxs, val_idxs, test_idxs


class PlantVillageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(s.label)
