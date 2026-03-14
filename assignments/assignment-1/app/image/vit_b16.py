"""
CIFAR-10 ViT-B/16 Model Handler

Handles prediction, Grad-CAM visualization, and calibration
for the ViT-B/16 model trained on CIFAR-10.
"""

import os
import types
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any
import torchvision.transforms as transforms
from torchvision.models import vit_b_16

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
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10 normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Image size ViT expects
IMAGE_SIZE = 224


def create_vit_model(num_classes=10):
    """Create ViT-B/16 with modified classifier for CIFAR-10."""
    model = vit_b_16(weights=None)
    # Replace classifier head
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


class ViTAttentionVisualizer:
    """
    Attention visualization for ViT.
    Shows which patches the model attends to.
    """

    def __init__(self, model):
        self.model = model
        self.attentions = None
        self._patch_last_encoder_block()

    def _patch_last_encoder_block(self):
        """
        Torchvision's ViT encoder block calls MultiheadAttention with
        need_weights=False, so a normal forward hook never receives attention
        maps. We patch only the last block to request weights during inference.
        """
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

    def generate_attention_map(self, input_tensor):
        """Generate attention map from input tensor."""
        self.model.eval()

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)

        if self.attentions is None:
            return None

        # Get the [CLS] token attention across all heads
        # Shape: (batch, heads, seq_len, seq_len) -> take cls token row
        cls_attention = self.attentions[0, :, 0, 1:].mean(dim=0)  # Average over heads

        # Reshape to patch grid (assuming 16x16 patches for 224x224 image)
        num_patches = int(cls_attention.shape[0] ** 0.5)

        if num_patches * num_patches != cls_attention.shape[0]:
            # Fallback: just return raw attention
            return cls_attention.cpu().numpy()

        # Reshape to 2D grid
        attention_map = cls_attention.reshape(num_patches, num_patches).cpu().numpy()

        # Normalize
        attention_map = attention_map - attention_map.min()
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()

        return attention_map


def create_attention_overlay(image_np, attention_map, alpha=0.5):
    """Create overlay of attention map on original image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if attention_map is None:
        return image_np

    # Resize attention map to image size
    from PIL import Image as PILImage
    attention_uint8 = (attention_map * 255).astype(np.uint8)
    attention_resized = PILImage.fromarray(attention_uint8).resize(
        (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR
    )
    attention_resized = np.array(attention_resized).astype(np.float32) / 255.0

    if image_np.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image_np = np.array(
            PILImage.fromarray(image_np).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)
        )

    # Apply colormap
    colormap = cm.jet(attention_resized)[:, :, :3]
    colormap = (colormap * 255).astype(np.uint8)

    # Create overlay
    overlay = (alpha * colormap + (1 - alpha) * image_np).astype(np.uint8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')

    titles = ['Original Image', 'Attention Map', 'Overlay']
    images = [image_np, colormap, overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
        ax.set_facecolor('#0d1117')

    plt.tight_layout(pad=2)

    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    result = np.array(rgba_buffer)[:, :, :3]
    plt.close(fig)

    return result


class Cifar10ViTHandler(BaseModelHandler):
    """Model handler for CIFAR-10 ViT-B/16."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.attention_viz = None
        self.history = {}
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
        self.model = create_vit_model(num_classes=10)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device,
                                 weights_only=True)
            if isinstance(checkpoint, dict):
                self.history = checkpoint.get('history', {}) or {}
                self.best_accuracy = get_best_accuracy_from_history(self.history)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize attention visualizer
        self.attention_viz = ViTAttentionVisualizer(self.model)

        precomputed_full = load_precomputed_calibration_result("vit_b16")
        if precomputed_full is not None:
            self._calibration_cache["full"] = precomputed_full

    def get_model_name(self) -> str:
        return "ViT-B/16"

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
            "Architecture": "ViT-B/16 (Transfer Learning from ImageNet)",
            "Dataset": "CIFAR-10 (10 classes, 60,000 images)",
            "Parameters": f"{total_params:,}",
            "Input Size": f"{IMAGE_SIZE}×{IMAGE_SIZE}×3",
            "Training": "Full fine-tune, AdamW, Cosine Annealing LR",
            "Best Accuracy": best_accuracy,
            "Device": str(self.device),
        }
        if self.history:
            info["Epochs"] = str(len(self.history.get("val_acc", [])))
        full_result = self._calibration_cache.get("full")
        if full_result is not None:
            info["Full-Test ECE"] = f"{full_result.ece:.6f}"
        return info

    def predict(self, input_data) -> PredictionResult:
        """Run prediction with attention visualization."""
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

        # Generate attention visualization
        attention_map = self.attention_viz.generate_attention_map(input_tensor)
        explanation_image = create_attention_overlay(original_image, attention_map)

        return PredictionResult(
            label=pred_label,
            confidence=pred_conf,
            all_labels=CIFAR10_LABELS,
            all_confidences=probs.tolist(),
            explanation_image=explanation_image,
        )

    def get_example_inputs(self) -> List[Any]:
        return []

    def get_calibration_data(
        self, max_samples: Optional[int] = None
    ) -> Optional[CalibrationResult]:
        """Compute calibration metrics on test set."""
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

            # Compute ECE
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

            ax1.bar([c - width/2 for c in bin_centers], bin_accuracies, width,
                   label='Accuracy', color='#58a6ff', alpha=0.9, edgecolor='#58a6ff')
            ax1.bar([c + width/2 for c in bin_centers], bin_confidences, width,
                   label='Avg Confidence', color='#f97583', alpha=0.9, edgecolor='#f97583')

            ax1.plot([0, 1], [0, 1], '--', color='#8b949e', linewidth=2,
                    label='Perfect Calibration')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Confidence', color='white', fontsize=12)
            ax1.set_ylabel('Accuracy / Confidence', color='white', fontsize=12)
            ax1.set_title(f'Reliability Diagram (ECE: {ece:.4f})',
                          color='white', fontsize=14, fontweight='bold', pad=15)
            ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=10)
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#30363d')
            ax1.grid(True, alpha=0.1, color='white')

            # Confidence histogram
            ax2.set_facecolor('#161b22')
            ax2.bar(bin_centers, [c / total for c in bin_counts], 0.08,
                   color='#56d364', alpha=0.9, edgecolor='#56d364')
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Confidence', color='white', fontsize=12)
            ax2.set_ylabel('Fraction of Samples', color='white', fontsize=12)
            ax2.set_title('Confidence Distribution',
                          color='white', fontsize=14, fontweight='bold', pad=15)
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#30363d')
            ax2.grid(True, alpha=0.1, color='white')

            plt.tight_layout(pad=3)

            fig.canvas.draw()
            rgba_buffer = fig.canvas.buffer_rgba()
            diagram = np.array(rgba_buffer)[:, :, :3]
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
