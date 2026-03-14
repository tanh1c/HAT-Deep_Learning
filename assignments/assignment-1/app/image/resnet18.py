"""
CIFAR-10 ResNet-18 Model Handler

Handles prediction, Grad-CAM visualization, and calibration
for the ResNet-18 model trained on CIFAR-10.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any
import torchvision.transforms as transforms
from torchvision.models import resnet18

from app.shared.model_registry import (
    BaseModelHandler,
    PredictionResult,
    CalibrationResult,
)
from app.shared.artifact_utils import (
    get_best_accuracy_from_history,
    load_precomputed_calibration_result,
)
from app.image.data import create_cifar10_test_dataset

# CIFAR-10 class labels
CIFAR10_LABELS = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# CIFAR-10 normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Image size ResNet expects
IMAGE_SIZE = 224


def create_resnet18_cifar10(num_classes=10):
    """Create ResNet-18 with modified classifier for CIFAR-10."""
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class GradCAM:
    """
    Grad-CAM implementation for visual explanation.
    Generates heatmap showing which regions the model focuses on.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = torch.nn.functional.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False
        )
        return cam.squeeze().cpu().numpy()


def create_gradcam_overlay(image_np, heatmap, alpha=0.5):
    """Create overlay of Grad-CAM heatmap on original image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Apply colormap to heatmap
    colormap = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    colormap = (colormap * 255).astype(np.uint8)

    # Resize image to match heatmap
    if image_np.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img_pil = Image.fromarray(image_np).resize((IMAGE_SIZE, IMAGE_SIZE))
        image_np = np.array(img_pil)

    # Create overlay
    overlay = (alpha * colormap + (1 - alpha) * image_np).astype(np.uint8)

    # Create figure with original + heatmap + overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')

    titles = ['Original Image', 'Grad-CAM Heatmap', 'Overlay']
    images = [image_np, colormap, overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    plt.tight_layout(pad=2)

    # Convert figure to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() which is more robust in newer matplotlib versions
    rgba_buffer = fig.canvas.buffer_rgba()
    result = np.array(rgba_buffer)[:, :, :3]  # Strip alpha channel
    plt.close(fig)

    return result


class Cifar10ResNet18Handler(BaseModelHandler):
    """Model handler for CIFAR-10 ResNet-18."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.grad_cam = None
        self.history = {}
        self.config = {}
        self.best_accuracy = None
        self._calibration_cache = {}
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        self.model = create_resnet18_cifar10(num_classes=10)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device,
                                    weights_only=True)
            if isinstance(checkpoint, dict):
                self.history = checkpoint.get('history', {}) or {}
                self.config = checkpoint.get('config', {}) or {}
                self.best_accuracy = get_best_accuracy_from_history(self.history)
            # Handle both state_dict and full model saves
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM with the last conv layer
        self.grad_cam = GradCAM(self.model, self.model.layer4[-1])

        precomputed_full = load_precomputed_calibration_result("resnet18")
        if precomputed_full is not None:
            self._calibration_cache["full"] = precomputed_full

    def get_model_name(self) -> str:
        return "ResNet-18"

    def get_dataset_name(self) -> str:
        return "CIFAR-10"

    def get_data_type(self) -> str:
        return "image"

    def get_class_labels(self) -> List[str]:
        return CIFAR10_LABELS

    def get_model_info(self) -> Dict[str, str]:
        total_params = sum(p.numel() for p in self.model.parameters())
        best_accuracy = (
            f"{self.best_accuracy:.2f}%"
            if self.best_accuracy is not None
            else "N/A"
        )
        info = {
            "Architecture": "ResNet-18 (Transfer Learning from ImageNet)",
            "Dataset": "CIFAR-10 (10 classes, 60,000 images)",
            "Parameters": f"{total_params:,}",
            "Input Size": f"{IMAGE_SIZE}×{IMAGE_SIZE}×3",
            "Training": "Full fine-tune, AdamW, Cosine Annealing LR",
            "Best Accuracy": best_accuracy,
            "Device": str(self.device),
        }
        if "epochs" in self.config:
            info["Epochs"] = str(self.config["epochs"])
        full_result = self._calibration_cache.get("full")
        if full_result is not None:
            info["Full-Test ECE"] = f"{full_result.ece:.6f}"
        return info

    def predict(self, input_data) -> PredictionResult:
        """Run prediction with Grad-CAM visualization."""
        if input_data is None:
            raise ValueError("No input image provided")

        # Convert to PIL Image if numpy array
        if isinstance(input_data, np.ndarray):
            original_image = input_data.copy()
            pil_image = Image.fromarray(input_data).convert('RGB')
        else:
            pil_image = input_data.convert('RGB')
            original_image = np.array(pil_image)

        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        probs = probabilities.cpu().numpy()
        pred_idx = probs.argmax()
        pred_label = CIFAR10_LABELS[pred_idx]
        pred_conf = float(probs[pred_idx])

        # Generate Grad-CAM
        # Need to re-run with gradients enabled
        input_tensor_grad = self.transform(pil_image).unsqueeze(0).to(self.device)
        input_tensor_grad.requires_grad_(True)

        heatmap = self.grad_cam.generate(input_tensor_grad, target_class=pred_idx)
        explanation_image = create_gradcam_overlay(original_image, heatmap)

        return PredictionResult(
            label=pred_label,
            confidence=pred_conf,
            all_labels=CIFAR10_LABELS,
            all_confidences=probs.tolist(),
            explanation_image=explanation_image,
        )

    def get_example_inputs(self) -> List[Any]:
        """Return example images from CIFAR-10 test set if available."""
        return []

    def get_calibration_data(
        self, max_samples: Optional[int] = None
    ) -> Optional[CalibrationResult]:
        """
        Compute calibration metrics on test set.
        This runs evaluation on the full test set - can be slow on CPU.
        """
        cache_key = "full" if max_samples is None else f"subset:{max_samples}"
        if cache_key in self._calibration_cache:
            return self._calibration_cache[cache_key]

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            test_dataset = create_cifar10_test_dataset(transform=self.transform)
            if max_samples is not None and 0 < max_samples < len(test_dataset):
                indices = np.linspace(
                    0, len(test_dataset) - 1, num=max_samples, dtype=int
                ).tolist()
                test_dataset = torch.utils.data.Subset(test_dataset, indices)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=128, shuffle=False, num_workers=0
            )

            all_probs = []
            all_preds = []
            all_targets = []

            self.model.eval()
            with torch.inference_mode():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(1)

                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.numpy())

            all_probs = np.array(all_probs)
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            # Compute ECE (Expected Calibration Error)
            n_bins = 10
            max_probs = np.max(all_probs, axis=1)
            correctness = (all_preds == all_targets).astype(float)

            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []

            for i in range(n_bins):
                lower = bin_boundaries[i]
                upper = bin_boundaries[i + 1]
                mask = (max_probs > lower) & (max_probs <= upper)
                count = mask.sum()
                bin_counts.append(int(count))

                if count > 0:
                    bin_acc = correctness[mask].mean()
                    bin_conf = max_probs[mask].mean()
                else:
                    bin_acc = 0.0
                    bin_conf = 0.0

                bin_accuracies.append(float(bin_acc))
                bin_confidences.append(float(bin_conf))

            # Compute ECE
            total = len(all_preds)
            ece = sum(
                (count / total) * abs(acc - conf)
                for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
            )

            # Create reliability diagram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.patch.set_facecolor('#0d1117')

            # Reliability Diagram
            ax1.set_facecolor('#161b22')
            bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]
            width = 0.08

            bars1 = ax1.bar(
                [c - width/2 for c in bin_centers], bin_accuracies, width,
                label='Accuracy', color='#58a6ff', alpha=0.9, edgecolor='#58a6ff'
            )
            bars2 = ax1.bar(
                [c + width/2 for c in bin_centers], bin_confidences, width,
                label='Avg Confidence', color='#f97583', alpha=0.9, edgecolor='#f97583'
            )

            ax1.plot([0, 1], [0, 1], '--', color='#8b949e', linewidth=2,
                     label='Perfect Calibration')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Confidence', color='white', fontsize=12)
            ax1.set_ylabel('Accuracy / Confidence', color='white', fontsize=12)
            ax1.set_title(
                f'Reliability Diagram (ECE: {ece:.4f})',
                color='white', fontsize=14, fontweight='bold', pad=15
            )
            ax1.legend(facecolor='#161b22', edgecolor='#30363d',
                       labelcolor='white', fontsize=10)
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#30363d')
            ax1.grid(True, alpha=0.1, color='white')

            # Confidence histogram
            ax2.set_facecolor('#161b22')
            ax2.bar(
                bin_centers, [c / total for c in bin_counts], 0.08,
                color='#56d364', alpha=0.9, edgecolor='#56d364'
            )
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Confidence', color='white', fontsize=12)
            ax2.set_ylabel('Fraction of Samples', color='white', fontsize=12)
            ax2.set_title(
                'Confidence Distribution',
                color='white', fontsize=14, fontweight='bold', pad=15
            )
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#30363d')
            ax2.grid(True, alpha=0.1, color='white')

            plt.tight_layout(pad=3)

            # Convert to numpy
            fig.canvas.draw()
            rgba_buffer = fig.canvas.buffer_rgba()
            diagram = np.array(rgba_buffer)[:, :, :3]  # Strip alpha channel
            plt.close(fig)

            self._calibration_cache[cache_key] = CalibrationResult(
                ece=ece,
                bin_accuracies=bin_accuracies,
                bin_confidences=bin_confidences,
                bin_counts=bin_counts,
                reliability_diagram=diagram,
                source="Live computation",
            )
            return self._calibration_cache[cache_key]

        except Exception as e:
            print(f"Error computing calibration: {e}")
            return None
