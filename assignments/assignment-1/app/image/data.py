"""
Utilities for Stanford Dogs metadata used by the Assignment 1 image demo.

The Streamlit demo now prefers the fair-benchmark artifacts downloaded from the
final notebook/W&B export workflow under `image/artifacts/download/`. If those
files are not present, it falls back to the older in-repo artifact layout.
"""

from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


ASSIGNMENT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PREFERRED_ARTIFACT_ROOT = (
    ASSIGNMENT_ROOT / "image" / "artifacts" / "download" / "artifacts" / "stanford_dogs"
)
LEGACY_ARTIFACT_ROOT = ASSIGNMENT_ROOT / "image" / "artifacts" / "stanford_dogs"


def resolve_stanford_dogs_artifact_root() -> Path:
    """Return the preferred Stanford Dogs artifact root for the image demo."""
    if PREFERRED_ARTIFACT_ROOT.exists():
        return PREFERRED_ARTIFACT_ROOT
    return LEGACY_ARTIFACT_ROOT


def resolve_stanford_dogs_artifact_path(*parts: str) -> Path:
    """Resolve a path inside the active Stanford Dogs artifact tree."""
    return resolve_stanford_dogs_artifact_root().joinpath(*parts)


EDA_METADATA_CSV = resolve_stanford_dogs_artifact_path("eda", "split_metadata.csv")
MODEL_COMPARISON_CSV = resolve_stanford_dogs_artifact_path("model_comparison.csv")


def _normalize_lookup_value(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


def make_model_lookup_key(model_name: str, training_strategy: str) -> str:
    return f"{_normalize_lookup_value(model_name)}::{_normalize_lookup_value(training_strategy)}"


@lru_cache(maxsize=1)
def load_stanford_dogs_class_labels(
    metadata_csv_path: str = str(EDA_METADATA_CSV),
) -> List[str]:
    """Return Stanford Dogs class names ordered by integer label."""
    metadata_path = Path(metadata_csv_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Stanford Dogs metadata CSV not found at {metadata_path}."
        )

    label_to_name: Dict[int, str] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            class_name = row["class_name"].strip()
            label_to_name.setdefault(label, class_name)

    if not label_to_name:
        raise ValueError("No Stanford Dogs label metadata could be loaded.")

    max_label = max(label_to_name)
    return [label_to_name[idx] for idx in range(max_label + 1)]


@lru_cache(maxsize=1)
def load_model_comparison_table(
    csv_path: str = str(MODEL_COMPARISON_CSV),
) -> List[Dict[str, str]]:
    """Load the exported fair-benchmark comparison table."""
    comparison_path = Path(csv_path)
    if not comparison_path.exists():
        return []

    with comparison_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def load_model_comparison_rows(
    csv_path: str = str(MODEL_COMPARISON_CSV),
) -> Dict[str, Dict[str, str]]:
    """Load exported comparison rows keyed by model + training strategy."""
    rows: Dict[str, Dict[str, str]] = {}
    for row in load_model_comparison_table(csv_path):
        model_name = row.get("Model", "").strip()
        strategy = row.get("Training strategy", "").strip()
        if model_name and strategy:
            rows[make_model_lookup_key(model_name, strategy)] = row
    return rows


def get_model_comparison_row(
    model_name: str,
    training_strategy: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Return the exported comparison row for a model.

    If `training_strategy` is omitted and multiple rows exist for the same model,
    the row with the best test accuracy is returned.
    """
    if training_strategy:
        return load_model_comparison_rows().get(
            make_model_lookup_key(model_name, training_strategy)
        )

    candidates = [
        row
        for row in load_model_comparison_table()
        if row.get("Model", "").strip() == model_name
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda row: float(row.get("Test accuracy", 0.0) or 0.0),
    )
