"""
Utilities for Stanford Dogs metadata used by the Assignment 1 image demo.

The demo only needs lightweight metadata from exported notebook artifacts:
- the ordered class-label list for prediction display
- summary metrics for the final comparison/calibration panel
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from typing import Dict, List, Optional


ASSIGNMENT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ARTIFACT_ROOT = os.path.join(ASSIGNMENT_ROOT, "image", "artifacts", "stanford_dogs")
EDA_METADATA_CSV = os.path.join(ARTIFACT_ROOT, "eda", "split_metadata.csv")
MODEL_COMPARISON_CSV = os.path.join(ARTIFACT_ROOT, "model_comparison.csv")


@lru_cache(maxsize=1)
def load_stanford_dogs_class_labels(
    metadata_csv_path: str = EDA_METADATA_CSV,
) -> List[str]:
    """Return Stanford Dogs class names ordered by integer label."""
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(
            f"Stanford Dogs metadata CSV not found at {metadata_csv_path}."
        )

    label_to_name: Dict[int, str] = {}
    with open(metadata_csv_path, "r", encoding="utf-8", newline="") as f:
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
def load_model_comparison_rows(
    csv_path: str = MODEL_COMPARISON_CSV,
) -> Dict[str, Dict[str, str]]:
    """Load the final exported model comparison table by model name."""
    if not os.path.exists(csv_path):
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Model"].strip()] = row
    return rows


def get_model_comparison_row(model_name: str) -> Optional[Dict[str, str]]:
    """Return the exported comparison row for a given model if available."""
    return load_model_comparison_rows().get(model_name)
