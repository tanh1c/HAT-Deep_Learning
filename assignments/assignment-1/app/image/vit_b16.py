"""
Stanford Dogs ViT-B/16 model handler.

This handler powers the Assignment 1 image demo for the Stanford Dogs
benchmark, including prediction, attention visualization, and loading the
exported calibration artifact from the final notebook outputs.
"""

from __future__ import annotations

import os
import types
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vit_b_16

from app.image.data import get_model_comparison_row, load_stanford_dogs_class_labels
from app.shared.model_registry import (
    BaseModelHandler,
    CalibrationResult,
    PredictionResult,
)


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OFFICIAL_TEST_SAMPLES = 8580

ASSIGNMENT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CALIBRATION_IMAGE_PATH = os.path.join(
    ASSIGNMENT_ROOT,
    "image",
    "artifacts",
    "stanford_dogs",
    "vit",
    "vit_b16_calibration.png",
)


def create_vit_classifier(num_classes: int) -> nn.Module:
    """Create a ViT-B/16 classifier with a custom output head."""
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


class ViTAttentionVisualizer:
    """Attention visualization helper for the last ViT encoder block."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.attentions = None
        self._patch_last_encoder_block()

    def _patch_last_encoder_block(self) -> None:
        last_block = self.model.encoder.layers[-1]
        visualizer = self

        def forward_with_attention(block, input_tensor):
            torch._assert(
                input_tensor.dim() == 3,
                f"Expected (batch_size, seq_length, hidden_dim) got {input_tensor.shape}",
            )

            x = block.ln_1(input_tensor)
            attn_output, attn_weights = block.self_attention(
                x,
                x,
                x,
                need_weights=True,
                average_attn_weights=False,
            )
            visualizer.attentions = attn_weights.detach()

            x = block.dropout(attn_output)
            x = x + input_tensor

            y = block.ln_2(x)
            y = block.mlp(y)
            return x + y

        last_block.forward = types.MethodType(forward_with_attention, last_block)

    def generate_attention_map(self, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        if self.attentions is None:
            return None

        cls_attention = self.attentions[0, :, 0, 1:].mean(dim=0)
        num_patches = int(cls_attention.shape[0] ** 0.5)

        if num_patches * num_patches != cls_attention.shape[0]:
            return cls_attention.cpu().numpy()

        attention_map = cls_attention.reshape(num_patches, num_patches).cpu().numpy()
        attention_map = attention_map - attention_map.min()
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        return attention_map


def create_attention_overlay(image_np: np.ndarray, attention_map: Optional[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Create a three-panel attention visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    if attention_map is None:
        return image_np

    attention_uint8 = (attention_map * 255).astype(np.uint8)
    attention_resized = Image.fromarray(attention_uint8).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.BILINEAR,
    )
    attention_resized = np.array(attention_resized).astype(np.float32) / 255.0

    if image_np.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image_np = np.array(
            Image.fromarray(image_np).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        )

    colormap = cm.jet(attention_resized)[:, :, :3]
    colormap = (colormap * 255).astype(np.uint8)
    overlay = (alpha * colormap + (1 - alpha) * image_np).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")

    titles = ["Original Image", "Attention Map", "Overlay"]
    images = [image_np, colormap, overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    plt.tight_layout(pad=2)
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    result = np.array(rgba_buffer)[:, :, :3]
    plt.close(fig)
    return result


def load_cached_calibration_result() -> Optional[CalibrationResult]:
    """Load the exported ViT-B/16 calibration artifact for Stanford Dogs."""
    row = get_model_comparison_row("ViT-B/16")
    if row is None or not os.path.exists(CALIBRATION_IMAGE_PATH):
        return None

    reliability_diagram = np.array(Image.open(CALIBRATION_IMAGE_PATH).convert("RGB"))
    ece = float(row["ECE"])
    return CalibrationResult(
        ece=ece,
        bin_accuracies=[],
        bin_confidences=[],
        bin_counts=[OFFICIAL_TEST_SAMPLES],
        reliability_diagram=reliability_diagram,
        source="Notebook artifact (official Stanford Dogs test set)",
    )


class StanfordDogsViTHandler(BaseModelHandler):
    """Model handler for the Stanford Dogs ViT-B/16 checkpoint."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = load_stanford_dogs_class_labels()
        self.model: Optional[nn.Module] = None
        self.attention_viz: Optional[ViTAttentionVisualizer] = None
        self.history: Dict[str, Any] = {}
        self.best_accuracy: Optional[float] = None
        self._calibration_cache: Dict[str, CalibrationResult] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self._load_model()

    def _load_model(self) -> None:
        self.model = create_vit_classifier(num_classes=len(self.class_labels))

        if os.path.exists(self.model_path):
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
            if isinstance(checkpoint, dict):
                self.history = checkpoint.get("history", {}) or {}
                val_acc = self.history.get("val_acc")
                if isinstance(val_acc, list) and val_acc:
                    self.best_accuracy = float(max(val_acc))

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.attention_viz = ViTAttentionVisualizer(self.model)

        precomputed_full = load_cached_calibration_result()
        if precomputed_full is not None:
            self._calibration_cache["full"] = precomputed_full

    def get_model_name(self) -> str:
        return "ViT-B/16"

    def get_dataset_name(self) -> str:
        return "Stanford Dogs"

    def get_data_type(self) -> str:
        return "image"

    def get_class_labels(self) -> List[str]:
        return self.class_labels

    def get_model_info(self) -> Dict[str, str]:
        total_params = sum(p.numel() for p in self.model.parameters())
        comparison_row = get_model_comparison_row("ViT-B/16")

        info = {
            "Architecture": "ViT-B/16 (Transfer Learning from ImageNet)",
            "Dataset": "Stanford Dogs (120 classes, 20,580 images)",
            "Parameters": f"{total_params:,}",
            "Input Size": f"{IMAGE_SIZE}×{IMAGE_SIZE}×3",
            "Training": "Head 3 epochs + full fine-tuning 8 epochs",
            "Device": str(self.device),
        }

        if self.best_accuracy is not None:
            info["Best Validation Accuracy"] = f"{self.best_accuracy * 100:.2f}%"

        if comparison_row is not None:
            info["Test Accuracy"] = f"{float(comparison_row['Test accuracy']) * 100:.2f}%"
            info["Macro F1"] = f"{float(comparison_row['Macro F1']):.4f}"
            info["Train Time"] = f"{float(comparison_row['Train time (s)']):.2f} s"
            info["Official-Test ECE"] = f"{float(comparison_row['ECE']):.6f}"

        return info

    def predict(self, input_data) -> PredictionResult:
        if input_data is None:
            raise ValueError("No input image provided")

        if isinstance(input_data, np.ndarray):
            original_image = input_data.copy()
            pil_image = Image.fromarray(input_data).convert("RGB")
        else:
            pil_image = input_data.convert("RGB")
            original_image = np.array(pil_image)

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        probs = probabilities.cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_label = self.class_labels[pred_idx]
        pred_conf = float(probs[pred_idx])

        attention_map = self.attention_viz.generate_attention_map(input_tensor)
        explanation_image = create_attention_overlay(original_image, attention_map)

        return PredictionResult(
            label=pred_label,
            confidence=pred_conf,
            all_labels=self.class_labels,
            all_confidences=probs.tolist(),
            explanation_image=explanation_image,
        )

    def get_example_inputs(self) -> List[Any]:
        return []

    def get_calibration_data(
        self, max_samples: Optional[int] = None
    ) -> Optional[CalibrationResult]:
        return self._calibration_cache.get("full")
