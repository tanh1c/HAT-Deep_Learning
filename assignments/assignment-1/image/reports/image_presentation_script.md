# Image Track Presentation Script

This note now includes the opening slides as well, so you can present from the beginning and then move naturally into the image part.

Suggested slide flow:
- Slide 1: Introduction
- Slide 2: Project Overview
- Slide 3: Dataset Overview
- Slide 4: Image Task and Motivation
- Slide 5: Image Pipeline
- Slide 6: Preprocessing and Data Augmentation
- Slide 7: Models and Training Strategy
- Slide 8: Main Benchmark Results
- Slide 9: Calibration and Interpretability
- Slide 10: Key Takeaways for the Image Track

---

## Slide 1. Introduction

### Slide title
`Assignment 1: Classification on Image, Text, and Multimodal Data`

### What to put on the slide
- Course: `Deep Learning and Its Applications`
- Team name and members
- Three modalities:
  - image
  - text
  - multimodal
- One short line:
  - `Goal: compare representative deep learning model families across modalities`

### Suggested visual
- Clean title slide
- Team name and university
- Optional small icons for image, text, and multimodal

### Speaking script
"Good morning everyone. We are Group HAT, and in this presentation we report the results of Assignment 1 for the course Deep Learning and Its Applications. Our project studies classification in three different modalities: image, text, and multimodal image-text data. The main idea is not only to build working pipelines, but also to compare representative model families fairly and analyze their behavior beyond raw accuracy."

---

## Slide 2. Project Overview

### Slide title
`Project Overview`

### What to put on the slide
- Image track:
  - `CNN vs Vision Transformer`
- Text track:
  - `RNN vs Transformer`
- Multimodal track:
  - `Zero-shot vs Few-shot`
- Shared evaluation goals:
  - accuracy
  - F1-score
  - calibration / analysis when relevant

### Suggested visual
- Three-column layout, one column per modality
- Or one pipeline overview graphic for the whole project

### Speaking script
"Our project is organized into three tracks. In the image track, we compare a CNN backbone with a Vision Transformer. In the text track, we compare an RNN-based approach with a Transformer-based approach. In the multimodal track, we compare zero-shot and few-shot learning settings. Across all three tracks, we aim to keep the comparisons fair and reproducible, and we evaluate the models not only by accuracy but also by F1-score and additional analyses such as calibration and interpretability when appropriate."

---

## Slide 3. Dataset Overview

### Slide title
`Dataset Overview`

### What to put on the slide
- Image:
  - `Stanford Dogs`
  - `20,580 images`
  - `120 classes`
- Text:
  - `20 Newsgroups`
  - `20 classes`
  - `18,000+ documents`
- Multimodal:
  - `UPMC-Food101 subset`
  - `10 classes`
  - paired image-text samples
- Transition line:
  - `In the next slides, I will focus on the image track`

### Suggested visual
- A compact comparison table
- Or 3 cards, one card per dataset

### Speaking script
"For the datasets, we use Stanford Dogs for the image track, 20 Newsgroups for the text track, and a 10-class subset of UPMC-Food101 for the multimodal track. Stanford Dogs is a fine-grained visual classification benchmark with 20,580 images and 120 breeds. 20 Newsgroups gives us a multi-class text classification problem with more than 18,000 documents. For the multimodal setting, we use paired image-text samples so that both modalities are semantically aligned. In the next part, I will focus on the image track, which is the first technical section of our presentation."

---

## Slide 4. Image Task and Motivation

### Slide title
`Image Track: Stanford Dogs Fine-Grained Classification`

### What to put on the slide
- Problem: classify dog breed from an input image
- Dataset: Stanford Dogs, 120 breeds, 20,580 images
- Challenge: fine-grained classification
- Main comparison: `ResNet-50` vs `ViT-B/16`
- Goal: compare CNN and Transformer fairly on the same benchmark

### Suggested visual
- One or two breed sample images
- Or reuse `stanforddogs_random_class_samples.png`

### Speaking script
"For the image part, our task is fine-grained dog breed classification on Stanford Dogs. This is harder than ordinary image classification because many breeds look visually similar, so the model must focus on subtle details such as ear shape, muzzle structure, and fur pattern. We use Stanford Dogs with 120 classes and 20,580 images, and our main goal is to compare a CNN backbone, ResNet-50, with a Transformer backbone, ViT-B/16, under the same training protocol."

---

## Slide 5. Image Pipeline

### Slide title
`Image Pipeline: Input -> Backbone -> Head -> Prediction`

### What to put on the slide
- Input image
- Preprocessing
- Two backbone branches:
  - `ResNet-50 backbone`
  - `ViT-B/16 backbone`
- Classification head
- Output: 120-class breed logits
- Shared transfer-learning schedules

### Suggested visual
- Use the pipeline diagram already added in LaTeX
- Or redraw a clean version in PowerPoint/Figma using the same flow

### Speaking script
"This is the overall pipeline for the image track. We start from a raw RGB dog image, apply preprocessing, then pass the standardized tensor into one of two backbones: either ResNet-50 or ViT-B/16. After feature extraction, each model uses a classifier head to map features into 120 breed classes. This lets us isolate the effect of the backbone architecture while keeping the task and output space the same."

---

## Slide 6. Preprocessing and Data Augmentation

### Slide title
`Preprocessing and Data Augmentation`

### What to put on the slide
- Evaluation pipeline:
  - `Resize(256,256)`
  - `CenterCrop(224)`
  - `ToTensor()`
  - `Normalize(ImageNet mean/std)`
- Training input size: `224 x 224`
- Batch size: `32`
- ResNet-50 augmentation:
  - stronger crop range
  - flip
  - rotation
  - color jitter
  - random erasing
- ViT-B/16 augmentation:
  - milder crop range
  - lighter perturbation

### Suggested visual
- `augmented_batch_preview.png`
- Small flow diagram for preprocessing

### Speaking script
"Before training, all images are standardized through a common preprocessing pipeline. We first resize to 256 by 256, center crop to 224, convert to tensor, and normalize using ImageNet statistics because both backbones start from ImageNet-pretrained checkpoints. For training, ResNet-50 uses a stronger augmentation policy to improve robustness, while ViT-B/16 uses milder augmentation so that fine-grained breed cues are not overly distorted."

---

## Slide 7. Models and Training Strategy

### Slide title
`Backbones and Fair Training Strategy`

### What to put on the slide
- Models:
  - `ResNet-50`: CNN baseline
  - `ViT-B/16`: Transformer baseline
- Both adapted to `120` classes
- Same split:
  - Train: `10,200`
  - Val: `1,800`
  - Test: `8,580`
- Two strategies for both models:
  - Full fine-tuning for `12 epochs`
  - Head-only `3 epochs` + full fine-tune `8 epochs`
- Optimizer: `AdamW`
- Loss: `CrossEntropyLoss`

### Suggested visual
- Simple 2x2 table:
  - ResNet-50 full
  - ResNet-50 staged
  - ViT-B/16 full
  - ViT-B/16 staged

### Speaking script
"To make the comparison fair, we use exactly the same split, input resolution, batch size, and evaluation metrics for both model families. We test each backbone with two transfer-learning strategies: full fine-tuning for 12 epochs, and a staged strategy with 3 head-only epochs followed by 8 full fine-tuning epochs. In all cases we use cross-entropy loss and AdamW. This design allows us to compare not only CNN versus Transformer, but also the effect of the training schedule."

---

## Slide 8. Main Benchmark Results

### Slide title
`Main Benchmark Results`

### What to put on the slide

Use this table:

| Model | Strategy | Test Acc | Macro F1 | ECE |
|---|---|---:|---:|---:|
| ResNet-50 | Full fine-tuning | 85.57% | 0.8485 | 0.0468 |
| ResNet-50 | Head 3 + full 8 | 86.55% | 0.8599 | 0.0408 |
| ViT-B/16 | Full fine-tuning | 90.77% | 0.9026 | 0.0178 |
| ViT-B/16 | Head 3 + full 8 | 93.48% | 0.9311 | 0.0198 |

### Suggested visual
- `stanforddogs_comparison_overview.png`
- Or a bar chart with test accuracy

### Speaking script
"These are the main benchmark results. The best overall model is ViT-B/16 with the staged strategy, reaching 93.48 percent accuracy and 0.9311 macro F1. ResNet-50 also improves when we switch from full fine-tuning to the staged schedule, from 85.57 to 86.55 percent. So one key takeaway is that the staged schedule helps both model families, but the Transformer backbone still achieves the strongest final performance."

---

## Slide 9. Calibration and Interpretability

### Slide title
`Beyond Accuracy: Calibration and Interpretability`

### What to put on the slide
- Calibration:
  - ResNet-50 staged: `ECE = 0.0408`
  - ViT-B/16 staged: `ECE = 0.0198`
- Interpretability:
  - Grad-CAM for ResNet-50
  - Attention gallery for ViT-B/16
- Message:
  - ViT is more accurate and better calibrated
  - Both models focus on meaningful dog regions

### Suggested visual
- Left: `resnet50_calibration_staged.png`
- Right: `vit_b16_calibration_staged.png`
- Optional extra slide element:
  - `resnet50_gradcam_gallery_staged.png`
  - `vit_b16_attention_gallery_staged.png`

### Speaking script
"We also evaluated the models beyond raw accuracy. In terms of calibration, ViT-B/16 is clearly better, with a lower expected calibration error than ResNet-50. This means its confidence scores are more reliable. For interpretability, we used Grad-CAM for ResNet-50 and attention visualization for ViT-B/16. In both cases, the visualizations show that the models focus mainly on the dog region rather than irrelevant background, which increases confidence that the models learn meaningful features."

---

## Slide 10. Key Takeaways for the Image Track

### Slide title
`Key Takeaways`

### What to put on the slide
- Stanford Dogs is a challenging fine-grained benchmark
- Fair comparison was ensured by identical split and protocol
- Staged transfer learning improved both backbones
- `ViT-B/16 staged` achieved the best performance
- `ResNet-50` remains lighter and more deployment-friendly

### Suggested visual
- One summary box:
  - Best accuracy: `ViT-B/16 staged = 93.48%`
  - Best CNN baseline: `ResNet-50 staged = 86.55%`

### Speaking script
"To conclude the image track, Stanford Dogs is a challenging fine-grained benchmark that clearly separates stronger representation learning from weaker ones. Under a fair comparison protocol, the staged transfer-learning schedule improves both ResNet-50 and ViT-B/16. The best overall result comes from ViT-B/16 staged, while ResNet-50 remains the lighter CNN baseline and may still be preferable when deployment cost matters."

---

## Short Version If Time Is Limited

If you need to shorten the image part, keep only:
- Slide 4: task and motivation
- Slide 6: preprocessing
- Slide 8: benchmark results
- Slide 10: key takeaways

Short closing sentence:
"In short, our image experiments show that under the same Stanford Dogs protocol, staged transfer learning improves both families, and ViT-B/16 gives the strongest overall performance while ResNet-50 remains a practical lighter baseline."
