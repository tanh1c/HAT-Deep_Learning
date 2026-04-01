# Assignment 1 - Image Track Progress Report

## Scope

This document summarizes the final progress of the `image classification` part
of Assignment 1 after replacing the old `CIFAR-10` setup with the final
`Stanford Dogs` benchmark.

The image track now includes:

- a fine-grained image dataset with `120` classes
- two compared model families:
  - `CNN`: `ResNet-18`
  - `ViT`: `ViT-B/16`
- exported figures and tables for the web report
- interpretability, calibration, and ablation extensions

---

## 1. Requirement Mapping

The image track already satisfies the core assignment requirements:

- `Image dataset`: completed
- `CNN model`: completed
- `ViT model`: completed
- `Evaluation and comparison`: completed
- `Tables and figures for report`: completed
- `Interpretability`: completed
- `Calibration`: completed
- `Ablation / extension work`: completed
- `GitHub Pages integration`: completed

---

## 2. Final dataset used

### Dataset

- Dataset: `Stanford Dogs`
- Total images: `20,580`
- Official train images: `12,000`
- Official test images: `8,580`
- Number of classes: `120`

### Why this dataset is valid

Stanford Dogs is more appropriate than the earlier CIFAR-10 setup because:

- it remains comfortably above the assignment threshold for training size
- it contains many more classes
- it is a fine-grained benchmark, so CNN-vs-ViT comparison is more meaningful
- it supports stronger qualitative analysis through confusion matrices and
  breed-level error inspection

---

## 3. Completed experiments

### 3.1 ResNet-18 baseline

Checkpoint:

- `../models/stanforddogs_resnet18.pth`

Main setup:

- backbone: `ResNet-18`
- initialization: `ImageNet-pretrained`
- input size: `224 × 224`
- strategy: `full fine-tuning for 12 epochs`

Final result:

- test accuracy: `0.7273`
- macro F1: `0.7193`
- weighted F1: `0.7283`
- ECE: `0.0394`

### 3.2 ViT-B/16 baseline

Checkpoint:

- `../models/stanforddogs_vit_b16.pth`

Main setup:

- backbone: `ViT-B/16`
- initialization: `ImageNet-pretrained`
- input size: `224 × 224`
- strategy:
  - head-only training for `3` epochs
  - full fine-tuning for `8` epochs

Final result:

- test accuracy: `0.9378`
- macro F1: `0.9343`
- weighted F1: `0.9378`
- ECE: `0.0161`

### 3.3 Final comparison

| Model | Family | Test accuracy | Macro F1 | Weighted F1 | ECE | Train time (s) |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 | CNN | 0.7273 | 0.7193 | 0.7283 | 0.0394 | 166.10 |
| ViT-B/16 | Transformer | 0.9378 | 0.9343 | 0.9378 | 0.0161 | 318.85 |

Current conclusion:

- `ViT-B/16` is the stronger final model on Stanford Dogs
- `ResNet-18` remains useful as a smaller and faster CNN baseline
- the final dataset/model pair is now more aligned with the assignment brief

---

## 4. Completed extension work

### 4.1 Interpretability

Implemented:

- `Grad-CAM` for `ResNet-18`
- attention visualization for `ViT-B/16`

Available exported figures:

- `../artifacts/stanford_dogs/cnn/resnet18_gradcam_gallery.png`
- `../artifacts/stanford_dogs/vit/vit_b16_attention_gallery.png`

### 4.2 Calibration

Implemented:

- `ECE`
- reliability diagram
- confidence histogram

Exported figures:

- `../artifacts/stanford_dogs/cnn/resnet18_calibration.png`
- `../artifacts/stanford_dogs/vit/vit_b16_calibration.png`

### 4.3 Misclassified gallery

Available exported figures:

- `../artifacts/stanford_dogs/cnn/resnet18_misclassified_gallery.png`
- `../artifacts/stanford_dogs/vit/vit_b16_misclassified_gallery.png`

These galleries show that the remaining errors are mostly fine-grained breed
confusions rather than completely unrelated predictions.

### 4.4 Ablation studies

Completed:

- `augmentation vs no augmentation`
- `freeze backbone vs full fine-tune`

Main findings:

- `ViT-B/16` benefited more clearly from augmentation than `ResNet-18`
- in the current setup, `freeze-backbone` ablations produced stronger metrics
  than full fine-tuning for both models

---

## 5. Current repository status

The image section now has:

- final Stanford Dogs notebook
- updated Stanford Dogs report draft
- updated GitHub Pages image report page
- exported figures for EDA, calibration, confusion, interpretability, and
  ablations
- updated Stanford Dogs checkpoints

This means the image track is now ready for:

- GitHub Pages presentation
- final report integration
- slide preparation
- push to the main repository after the remaining repo-wide cleanup is complete
