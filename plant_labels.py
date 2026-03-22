"""Canonical PlantVillage 39-class label order used across this repo.

The Flask app indexes into `disease_info.csv` by predicted class index, so
training MUST use the same class order.

Folder names in PlantVillage downloads vary slightly across sources (Kaggle,
Mendeley, GitHub mirrors). The helpers here normalize names and allow
lightweight aliasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


IDX_TO_CLASS: Dict[int, str] = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Background_without_leaves",
    5: "Blueberry___healthy",
    6: "Cherry___Powdery_mildew",
    7: "Cherry___healthy",
    8: "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    9: "Corn___Common_rust",
    10: "Corn___Northern_Leaf_Blight",
    11: "Corn___healthy",
    12: "Grape___Black_rot",
    13: "Grape___Esca_(Black_Measles)",
    14: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    15: "Grape___healthy",
    16: "Orange___Haunglongbing_(Citrus_greening)",
    17: "Peach___Bacterial_spot",
    18: "Peach___healthy",
    19: "Pepper,_bell___Bacterial_spot",
    20: "Pepper,_bell___healthy",
    21: "Potato___Early_blight",
    22: "Potato___Late_blight",
    23: "Potato___healthy",
    24: "Raspberry___healthy",
    25: "Soybean___healthy",
    26: "Squash___Powdery_mildew",
    27: "Strawberry___Leaf_scorch",
    28: "Strawberry___healthy",
    29: "Tomato___Bacterial_spot",
    30: "Tomato___Early_blight",
    31: "Tomato___Late_blight",
    32: "Tomato___Leaf_Mold",
    33: "Tomato___Septoria_leaf_spot",
    34: "Tomato___Spider_mites Two-spotted_spider_mite",
    35: "Tomato___Target_Spot",
    36: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    37: "Tomato___Tomato_mosaic_virus",
    38: "Tomato___healthy",
}


def class_list() -> List[str]:
    return [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]


def normalize_label(name: str) -> str:
    """Normalize a PlantVillage class string for fuzzy matching.

    Examples of differences across sources:
    - "Pepper,_bell___healthy" vs "Pepper__bell___healthy"
    - "..._(Black_Measles)" vs "..._Black_Measles"
    """

    text = name.strip().lower()

    # Treat comma the same as underscore.
    text = text.replace(",", "_")

    # Normalize whitespace to underscore.
    text = re.sub(r"\s+", "_", text)

    # Replace other punctuation with underscores.
    text = re.sub(r"[^a-z0-9_]+", "_", text)

    # Collapse runs of underscores.
    text = re.sub(r"_+", "_", text).strip("_")

    # Common Kaggle naming: double underscores for bell pepper.
    text = text.replace("pepper__bell", "pepper_bell")

    return text


@dataclass(frozen=True)
class LabelMatch:
    expected: str
    found: str


def build_normalized_lookup(names: Iterable[str]) -> Dict[str, str]:
    """Map normalized -> original name (first occurrence wins)."""
    lookup: Dict[str, str] = {}
    for n in names:
        key = normalize_label(n)
        lookup.setdefault(key, n)
    return lookup


def match_expected_to_available(expected: Iterable[str], available: Iterable[str]) -> List[LabelMatch]:
    """Match expected labels to the actual folder names available on disk."""
    available_list = list(available)
    lookup = build_normalized_lookup(available_list)

    matches: List[LabelMatch] = []
    for exp in expected:
        key = normalize_label(exp)
        found = lookup.get(key)

        # Try a couple extra alias tweaks.
        if found is None:
            alt = key.replace("_black_measles_", "_black_measles_")
            found = lookup.get(alt)

        if found is None:
            # Another common variation: remove parentheses tokens entirely.
            alt2 = key.replace("_black_measles", "_black_measles")
            found = lookup.get(alt2)

        if found is None:
            raise ValueError(
                "Could not match expected class folder.\n"
                f"Missing: {exp}\n\n"
                "Tips:\n"
                "- Ensure your dataset directory contains 39 subfolders (one per class).\n"
                "- If your download uses different names, rename folders to match `plant_labels.IDX_TO_CLASS`.\n"
                f"- Available folders (sample): {available_list[:20]}"
            )

        matches.append(LabelMatch(expected=exp, found=found))

    return matches
