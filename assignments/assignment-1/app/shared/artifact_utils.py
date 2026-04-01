"""
Helpers for reading notebook-generated artifacts and training metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .model_registry import CalibrationResult


ASSIGNMENT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ARTIFACTS_DIR = ASSIGNMENT_ROOT / "image" / "artifacts"


def _render_reliability_diagram_from_metrics(metrics: Dict[str, Any]) -> np.ndarray:
    """Render a reliability diagram directly from saved calibration metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_accuracies = [float(x) for x in metrics["bin_accuracies"]]
    bin_confidences = [float(x) for x in metrics["bin_confidences"]]
    bin_counts = [int(x) for x in metrics["bin_counts"]]
    ece = float(metrics["ece"])

    n_bins = len(bin_accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = [
        (bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)
    ]
    total = max(sum(bin_counts), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    ax1.set_facecolor("#161b22")
    width = 0.08
    ax1.bar(
        [c - width / 2 for c in bin_centers],
        bin_accuracies,
        width,
        label="Accuracy",
        color="#58a6ff",
        alpha=0.9,
        edgecolor="#58a6ff",
    )
    ax1.bar(
        [c + width / 2 for c in bin_centers],
        bin_confidences,
        width,
        label="Avg Confidence",
        color="#f97583",
        alpha=0.9,
        edgecolor="#f97583",
    )
    ax1.plot(
        [0, 1],
        [0, 1],
        "--",
        color="#8b949e",
        linewidth=2,
        label="Perfect Calibration",
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence", color="white", fontsize=12)
    ax1.set_ylabel("Accuracy / Confidence", color="white", fontsize=12)
    ax1.set_title(
        f"Reliability Diagram (ECE: {ece:.4f})",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        fontsize=10,
    )
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")
    ax1.grid(True, alpha=0.1, color="white")

    ax2.set_facecolor("#161b22")
    ax2.bar(
        bin_centers,
        [count / total for count in bin_counts],
        0.08,
        color="#56d364",
        alpha=0.9,
        edgecolor="#56d364",
    )
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence", color="white", fontsize=12)
    ax2.set_ylabel("Fraction of Samples", color="white", fontsize=12)
    ax2.set_title(
        "Confidence Distribution",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    ax2.grid(True, alpha=0.1, color="white")

    plt.tight_layout(pad=3)
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    diagram = np.array(rgba_buffer)[:, :, :3]
    plt.close(fig)
    return diagram


def normalize_history(history: Optional[Any]) -> Dict[str, Any]:
    """
    Normalize checkpoint history to a dict-of-lists structure.

    Supported formats:
    - {"val_acc": [...], "train_acc": [...], ...}
    - [{"epoch": 1, "val_acc": 0.8, ...}, {"epoch": 2, "val_acc": 0.82, ...}]
    """
    if not history:
        return {}

    if isinstance(history, dict):
        return history

    if isinstance(history, list) and history and all(isinstance(item, dict) for item in history):
        normalized: Dict[str, Any] = {}
        for item in history:
            for key, value in item.items():
                normalized.setdefault(key, []).append(value)
        return normalized

    return {}


def get_best_accuracy_from_history(history: Optional[Any]) -> Optional[float]:
    """Return the best validation accuracy found in a checkpoint history."""
    history = normalize_history(history)
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
    metrics_path = next(ARTIFACTS_DIR.rglob(metrics_name), None)
    image_name = f"{model_tag}_calibration_{sample_tag}.png"
    image_path = next(ARTIFACTS_DIR.rglob(image_name), None)

    if metrics_path is None:
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if image_path is not None:
        reliability_diagram = np.array(Image.open(image_path).convert("RGB"))
    else:
        reliability_diagram = _render_reliability_diagram_from_metrics(metrics)

    return CalibrationResult(
        ece=float(metrics["ece"]),
        bin_accuracies=[float(x) for x in metrics["bin_accuracies"]],
        bin_confidences=[float(x) for x in metrics["bin_confidences"]],
        bin_counts=[int(x) for x in metrics["bin_counts"]],
        reliability_diagram=reliability_diagram,
        source=f"Notebook artifact ({metrics_path.parent.name})",
    )
