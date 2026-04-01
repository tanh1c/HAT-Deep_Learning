# Assignment 1 - Image Presentation Content

This file summarizes the final `image` track of Assignment 1 using the
`Stanford Dogs` dataset and the final comparison between `ResNet-18` and
`ViT-B/16`.

Main source notebook:

- `../notebooks/stanforddogs_resnet18_vit_report_workflow.ipynb`

Main exported report draft:

- `./stanford_dogs_report_draft.md`

Main artifacts:

- `../artifacts/stanford_dogs/comparison_overview.png`
- `../artifacts/stanford_dogs/cnn/resnet18_calibration.png`
- `../artifacts/stanford_dogs/vit/vit_b16_calibration.png`
- `../artifacts/stanford_dogs/cnn/resnet18_gradcam_gallery.png`
- `../artifacts/stanford_dogs/vit/vit_b16_attention_gallery.png`
- `../artifacts/stanford_dogs/cnn/resnet18_misclassified_gallery.png`
- `../artifacts/stanford_dogs/vit/vit_b16_misclassified_gallery.png`

---

## 1. Problem and dataset exploration (EDA)

### Problem statement

The image task is a fine-grained image classification problem:

- dataset: `Stanford Dogs`
- number of classes: `120 dog breeds`
- compared model families:
  - `CNN`: `ResNet-18`
  - `Vision Transformer`: `ViT-B/16`

The objective is to compare these two pretrained architectures under the same
data split and evaluation protocol.

### Why Stanford Dogs is suitable

Stanford Dogs matches the assignment requirements well:

- `120` classes, well above the minimum requirement
- `12,000` official training images
- `8,580` official test images
- fine-grained breed differences make the task meaningfully harder than very
  simple image benchmarks

### EDA highlights

Key dataset statistics:

- total images: `20,580`
- classes: `120`
- min images per class: `148`
- max images per class: `252`
- mean images per class: `171.5`
- average brightness: `0.4522`
- average contrast std: `0.2261`
- average saturation: `0.3047`
- average width: `442.5 px`
- average height: `385.9 px`
- average aspect ratio: `1.19`

Main observations:

- the dataset contains noticeable variation in illumination and saturation
- image resolution and aspect ratio are not uniform
- some breeds remain visually close, which supports the use of confusion
  matrices and qualitative analysis later
- the train/test split is large enough for a fair comparison between CNN and ViT

Important EDA figures:

- `../artifacts/stanford_dogs/eda/quality_distributions.png`
- `../artifacts/stanford_dogs/eda/dataset_distributions.png`
- `../artifacts/stanford_dogs/eda/image_size_distribution.png`
- `../artifacts/stanford_dogs/eda/random_class_samples.png`
- `../artifacts/stanford_dogs/eda/darkest_examples.png`
- `../artifacts/stanford_dogs/eda/brightest_examples.png`
- `../artifacts/stanford_dogs/eda/rgb_channel_summary.png`

---

## 2. Dataset, DataLoader, and Augmentation setup

### Split strategy

We preserved the official dataset split and only created validation data from
the training side:

| Split | Images | Classes | Min class count | Max class count |
|---|---:|---:|---:|---:|
| Train | 10,200 | 120 | 85 | 85 |
| Val | 1,800 | 120 | 15 | 15 |
| Test | 8,580 | 120 | 48 | 152 |

This gives:

- official benchmark-style test evaluation
- a balanced internal validation split
- training size still comfortably above the assignment threshold

### DataLoader setup

Both models use the same split structure:

- ResNet loaders: `319 / 57 / 269`
- ViT loaders: `319 / 57 / 269`

### Preprocessing design

Shared choices:

- resize to `256`
- crop to `224`
- normalize with `ImageNet mean/std`

Reason:

- both backbones are initialized from ImageNet-pretrained weights
- `224 × 224` keeps enough spatial detail for breed-level cues

### Augmentation policy

`ResNet-18`:

- `RandomResizedCrop`
- horizontal flip
- rotation
- color jitter
- random erasing

`ViT-B/16`:

- milder crop
- horizontal flip
- color jitter
- random erasing

Rationale:

- ResNet uses stronger augmentation for robustness
- ViT keeps augmentation more moderate to preserve subtle breed cues

Main figure:

- `../artifacts/stanford_dogs/eda/augmented_batch_preview.png`

---

## 3. Model building, training, evaluation, and comparison

### ResNet-18

- family: `CNN`
- parameters: `11,238,072`
- training strategy: `full fine-tuning for 12 epochs`

Final training trend from the exported history:

| Epoch | Train acc | Val acc | Val macro F1 |
|---|---:|---:|---:|
| 8 | 0.8931 | 0.7044 | 0.7018 |
| 9 | 0.9341 | 0.7289 | 0.7267 |
| 10 | 0.9515 | 0.7417 | 0.7412 |
| 11 | 0.9640 | 0.7461 | 0.7466 |
| 12 | 0.9701 | 0.7489 | 0.7482 |

Test result:

- accuracy: `0.7273`
- macro F1: `0.7193`
- weighted F1: `0.7283`

### ViT-B/16

- family: `Transformer`
- parameters: `85,890,936`
- training strategy: `head-only training for 3 epochs, then full fine-tuning for 8 epochs`

Final fine-tuning trend:

| Epoch | Train acc | Val acc | Val macro F1 |
|---|---:|---:|---:|
| 4 | 0.9851 | 0.9239 | 0.9233 |
| 5 | 0.9914 | 0.9350 | 0.9349 |
| 6 | 0.9927 | 0.9372 | 0.9368 |
| 7 | 0.9931 | 0.9372 | 0.9369 |
| 8 | 0.9943 | 0.9394 | 0.9391 |

Test result:

- accuracy: `0.9378`
- macro F1: `0.9343`
- weighted F1: `0.9378`

---

## 4. Experimental results: tables, figures, analysis, and discussion

### Final comparison table

| Model | Family | Params | Training strategy | Test accuracy | Macro F1 | Weighted F1 | ECE | Train time (s) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| ResNet-18 | CNN | 11,238,072 | Full fine-tuning for 12 epochs | 0.7273 | 0.7193 | 0.7283 | 0.0394 | 166.10 |
| ViT-B/16 | Transformer | 85,890,936 | Head 3 + full fine-tune 8 epochs | 0.9378 | 0.9343 | 0.9378 | 0.0161 | 318.85 |

Main figure:

- `../artifacts/stanford_dogs/comparison_overview.png`

### Main conclusions

- `ViT-B/16` strongly outperforms `ResNet-18` on Stanford Dogs
- the gap is visible in both accuracy and macro F1
- `ViT-B/16` is also better calibrated because it achieves lower `ECE`
- `ResNet-18` remains much lighter and faster, so it is still a meaningful baseline

### Qualitative evidence

ResNet artifacts:

- `../artifacts/stanford_dogs/cnn/resnet_18_confusion_matrix_counts.png`
- `../artifacts/stanford_dogs/cnn/resnet_18_confusion_matrix_normalized.png`
- `../artifacts/stanford_dogs/cnn/resnet18_calibration.png`
- `../artifacts/stanford_dogs/cnn/resnet18_gradcam_gallery.png`
- `../artifacts/stanford_dogs/cnn/resnet18_misclassified_gallery.png`

ViT artifacts:

- `../artifacts/stanford_dogs/vit/vit_b_16_confusion_matrix_counts.png`
- `../artifacts/stanford_dogs/vit/vit_b_16_confusion_matrix_normalized.png`
- `../artifacts/stanford_dogs/vit/vit_b16_calibration.png`
- `../artifacts/stanford_dogs/vit/vit_b16_attention_gallery.png`
- `../artifacts/stanford_dogs/vit/vit_b16_misclassified_gallery.png`

Representative class-wise metrics:

ResNet-18:

- Chihuahua: precision/recall/F1 = `0.6538 / 0.6538 / 0.6538`
- Japanese spaniel: `0.8488 / 0.8588 / 0.8538`
- Maltese dog: `0.8750 / 0.7829 / 0.8264`

ViT-B/16:

- Chihuahua: precision/recall/F1 = `0.8750 / 0.9423 / 0.9074`
- Japanese spaniel: `0.9512 / 0.9176 / 0.9341`
- Maltese dog: `0.9862 / 0.9408 / 0.9630`

---

## 5. Other extension reports

### Interpretability

Completed:

- `Grad-CAM` for `ResNet-18`
- attention-map visualization for `ViT-B/16`

These outputs show whether each model focuses on the dog body, face, fur region,
or surrounding context.

### Misclassified gallery

The misclassified-example galleries show that the remaining mistakes are often
fine-grained breed confusions:

- visually similar small-breed dogs
- spaniel-to-spaniel confusions
- shepherd / terrier / retriever variants with overlapping appearance

### Augmentation vs no augmentation

| Setting | Accuracy | Macro F1 | Weighted F1 | ECE | Train time (s) |
|---|---:|---:|---:|---:|---:|
| ResNet-18 with augmentation | 0.7287 | 0.7183 | 0.7288 | 0.0572 | 55.56 |
| ResNet-18 no augmentation | 0.7335 | 0.7239 | 0.7346 | 0.0477 | 37.31 |
| ViT-B/16 with augmentation | 0.9127 | 0.9073 | 0.9128 | 0.0733 | 139.15 |
| ViT-B/16 no augmentation | 0.8990 | 0.8947 | 0.8994 | 0.0633 | 138.39 |

Interpretation:

- for this setup, augmentation did not improve ResNet-18
- augmentation improved ViT-B/16 accuracy and macro F1, but slightly worsened calibration

### Freeze backbone vs full fine-tune

| Setting | Accuracy | Macro F1 | Weighted F1 | ECE | Train time (s) | Trainable params |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 freeze backbone | 0.7416 | 0.7322 | 0.7420 | 0.1480 | 55.88 | 61,560 |
| ResNet-18 full fine-tune | 0.7359 | 0.7244 | 0.7356 | 0.0643 | 56.19 | 11,238,072 |
| ViT-B/16 freeze backbone | 0.9457 | 0.9423 | 0.9459 | 0.0074 | 57.29 | 92,280 |
| ViT-B/16 full fine-tune | 0.9172 | 0.9127 | 0.9174 | 0.0753 | 138.37 | 85,890,936 |

Interpretation:

- in this specific experimental setup, freeze-backbone training performed better
  than full fine-tuning for both models
- the result is especially strong for `ViT-B/16`, which reached the best
  ablation performance with far fewer trainable parameters

### Final extension summary

The image track goes beyond the minimum assignment requirement by including:

- interpretability
- calibration
- misclassification analysis
- augmentation ablation
- freeze-vs-full fine-tuning ablation

This makes the Stanford Dogs image track ready for GitHub Pages, presentation
slides, and the final report.
