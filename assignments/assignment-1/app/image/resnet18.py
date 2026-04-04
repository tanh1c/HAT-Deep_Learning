"""
Stanford Dogs ResNet handlers.

The module name is kept for backward compatibility with older imports, but the
current Streamlit demo uses the fair-benchmark ResNet-50 checkpoints exported
from the latest notebook workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

from app.image.data import (
    get_model_comparison_row,
    load_stanford_dogs_class_labels,
    resolve_stanford_dogs_artifact_path,
)
from app.shared.artifact_utils import normalize_history
from app.shared.model_registry import (
    BaseModelHandler,
    CalibrationResult,
    PredictionResult,
)


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OFFICIAL_TEST_SAMPLES = 8580


@dataclass(frozen=True)
class ResNetVariantConfig:
    display_name: str
    model_name: str
    training_strategy: str
    architecture: str
    calibration_image_path: Path


RESNET50_FULL_CONFIG = ResNetVariantConfig(
    display_name="ResNet-50 · Full fine-tuning",
    model_name="ResNet-50",
    training_strategy="Full fine-tuning for 12 epochs",
    architecture="ResNet-50 (Transfer Learning from ImageNet)",
    calibration_image_path=resolve_stanford_dogs_artifact_path(
        "cnn",
        "restnet50_full_finetune",
        "resnet_50___full_fine_tuning_for_12_epochs_calibration.png",
    ),
)

RESNET50_STAGED_CONFIG = ResNetVariantConfig(
    display_name="ResNet-50 · Head 3 + Full 8",
    model_name="ResNet-50",
    training_strategy="Head 3 + full fine-tune 8 epochs",
    architecture="ResNet-50 (Transfer Learning from ImageNet)",
    calibration_image_path=resolve_stanford_dogs_artifact_path(
        "cnn",
        "restnet50_head_then_full",
        "resnet_50___head_3_+_full_fine_tune_8_epochs_calibration.png",
    ),
)


def create_resnet50_classifier(num_classes: int) -> nn.Module:
    """Create a ResNet-50 classifier with a custom output head."""
    model = resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class GradCAM:
    """Minimal Grad-CAM implementation for the last convolutional block."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return cam.squeeze().cpu().numpy()


def create_gradcam_overlay(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create a three-panel Grad-CAM visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    colormap = cm.jet(heatmap)[:, :, :3]
    colormap = (colormap * 255).astype(np.uint8)

    if image_np.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image_np = np.array(
            Image.fromarray(image_np).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        )

    overlay = (alpha * colormap + (1 - alpha) * image_np).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")

    titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
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


def load_cached_calibration_result(
    variant: ResNetVariantConfig,
) -> Optional[CalibrationResult]:
    """Load the exported ResNet-50 calibration artifact for a specific run."""
    row = get_model_comparison_row(
        variant.model_name,
        variant.training_strategy,
    )
    if row is None or not variant.calibration_image_path.exists():
        return None

    reliability_diagram = np.array(
        Image.open(variant.calibration_image_path).convert("RGB")
    )
    ece = float(row["ECE"])
    return CalibrationResult(
        ece=ece,
        bin_accuracies=[],
        bin_confidences=[],
        bin_counts=[OFFICIAL_TEST_SAMPLES],
        reliability_diagram=reliability_diagram,
        source=f"Notebook artifact ({variant.training_strategy})",
    )


class StanfordDogsResNet50Handler(BaseModelHandler):
    """Model handler for a Stanford Dogs ResNet-50 checkpoint variant."""

    def __init__(self, model_path: str, variant: ResNetVariantConfig):
        self.model_path = model_path
        self.variant = variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = load_stanford_dogs_class_labels()
        self.model: Optional[nn.Module] = None
        self.grad_cam: Optional[GradCAM] = None
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
        checkpoint = None
        model_path = Path(self.model_path)
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                checkpoint_class_names = checkpoint.get("class_names")
                if isinstance(checkpoint_class_names, list) and checkpoint_class_names:
                    self.class_labels = [str(name) for name in checkpoint_class_names]
                self.history = normalize_history(checkpoint.get("history", {}))
                val_acc = self.history.get("val_acc")
                if isinstance(val_acc, list) and val_acc:
                    self.best_accuracy = float(max(val_acc))

        self.model = create_resnet50_classifier(num_classes=len(self.class_labels))

        if checkpoint is not None:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.grad_cam = GradCAM(self.model, self.model.layer4[-1])

        precomputed = load_cached_calibration_result(self.variant)
        if precomputed is not None:
            self._calibration_cache["official_test"] = precomputed

    def get_model_name(self) -> str:
        return self.variant.display_name

    def get_dataset_name(self) -> str:
        return "Stanford Dogs"

    def get_data_type(self) -> str:
        return "image"

    def get_class_labels(self) -> List[str]:
        return self.class_labels

    def get_model_info(self) -> Dict[str, str]:
        total_params = sum(p.numel() for p in self.model.parameters())
        comparison_row = get_model_comparison_row(
            self.variant.model_name,
            self.variant.training_strategy,
        )

        info = {
            "Architecture": self.variant.architecture,
            "Model Variant": self.variant.display_name,
            "Dataset": "Stanford Dogs (120 classes, 20,580 images)",
            "Parameters": f"{total_params:,}",
            "Input Size": f"{IMAGE_SIZE}x{IMAGE_SIZE}x3",
            "Training": self.variant.training_strategy,
            "Device": str(self.device),
        }

        if self.best_accuracy is not None:
            info["Best Validation Accuracy"] = f"{self.best_accuracy * 100:.2f}%"

        if comparison_row is not None:
            info["Test Accuracy"] = f"{float(comparison_row['Test accuracy']) * 100:.2f}%"
            info["Macro F1"] = f"{float(comparison_row['Macro F1']):.4f}"
            info["Weighted F1"] = f"{float(comparison_row['Weighted F1']):.4f}"
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

        input_tensor_grad = self.transform(pil_image).unsqueeze(0).to(self.device)
        input_tensor_grad.requires_grad_(True)
        heatmap = self.grad_cam.generate(input_tensor_grad, target_class=pred_idx)
        explanation_image = create_gradcam_overlay(original_image, heatmap)

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
        self,
        max_samples: Optional[int] = None,
    ) -> Optional[CalibrationResult]:
        return self._calibration_cache.get("official_test")


class StanfordDogsResNet50FullHandler(StanfordDogsResNet50Handler):
    def __init__(self, model_path: str):
        super().__init__(model_path, RESNET50_FULL_CONFIG)


class StanfordDogsResNet50StagedHandler(StanfordDogsResNet50Handler):
    def __init__(self, model_path: str):
        super().__init__(model_path, RESNET50_STAGED_CONFIG)

