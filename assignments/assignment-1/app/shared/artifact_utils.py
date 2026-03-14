"""
Helpers for reading notebook-generated artifacts and training metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .model_registry import CalibrationResult


ASSIGNMENT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ARTIFACTS_DIR = ASSIGNMENT_ROOT / "image" / "artifacts"


def get_best_accuracy_from_history(history: Optional[Dict[str, Any]]) -> Optional[float]:
    """Return the best validation accuracy found in a checkpoint history."""
    if not history:
        return None

    val_acc = history.get("val_acc")
    if isinstance(val_acc, list) and val_acc:
        return float(max(val_acc))

    return None


def load_precomputed_calibration_result(
    model_tag: str,
    sample_tag: str = "full",
) -> Optional[CalibrationResult]:
    """
    Load notebook-generated calibration metrics and figure from image/artifacts/.

    The function searches recursively so nested folders like artifacts/cnn and
    artifacts/vit are both supported.
    """
    if not ARTIFACTS_DIR.exists():
        return None

    metrics_name = f"{model_tag}_calibration_metrics_{sample_tag}.json"
    image_name = f"{model_tag}_calibration_{sample_tag}.png"

    metrics_path = next(ARTIFACTS_DIR.rglob(metrics_name), None)
    image_path = next(ARTIFACTS_DIR.rglob(image_name), None)

    if metrics_path is None or image_path is None:
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return CalibrationResult(
        ece=float(metrics["ece"]),
        bin_accuracies=[float(x) for x in metrics["bin_accuracies"]],
        bin_confidences=[float(x) for x in metrics["bin_confidences"]],
        bin_counts=[int(x) for x in metrics["bin_counts"]],
        reliability_diagram=str(image_path),
        source=f"Notebook artifact ({metrics_path.parent.name})",
    )
