# Assignment 1 - Image Section Progress Report

## Scope

This document summarizes the current progress of the `image classification` part of Assignment 1.

Assumption for team roles:

- I am responsible for the `image` section.
- My teammates are responsible for the `text` and `multimodal` sections.

This report focuses on:

- what has already been completed for the image part,
- the current experimental results,
- extension work that has been finished,
- and the remaining tasks needed for report, GitHub Pages, video, and slide submission.

## 1. Assignment Requirement Mapping

According to the assignment specification, the image section must:

- use an image classification dataset,
- compare two model families:
  - `CNN`
  - `ViT`
- train/fine-tune pretrained models,
- evaluate and compare the results fairly,
- optionally include extensions such as:
  - `Application demo`
  - `Interpretability`
  - `Calibration`

Current status:

- `Image dataset`: completed
- `CNN model`: completed
- `ViT model`: completed
- `Evaluation and comparison`: completed at experiment level
- `Application demo`: completed
- `Interpretability`: completed
- `Calibration`: completed
- `Report packaging / presentation packaging`: not finished yet

## 2. Dataset Used

### Image dataset

- Dataset: `CIFAR-10`
- Task: image classification
- Number of classes: `10`
- Test set size used in evaluation/calibration: `10,000` samples

### Why this dataset is valid for the assignment

- It satisfies the requirement of having at least `5` classes.
- It is a standard benchmark for image classification.
- It is more suitable than very simple datasets such as MNIST for comparing `CNN` and `ViT`.
- It supports both model comparison and extension analysis such as confusion matrix, interpretability, and calibration.

## 3. Completed Work

### 3.1 CNN model

Notebook:

- `../notebooks/cifar10_resnet18_transfer_learning.ipynb`

Model:

- `../models/resnet18_cifar10.pth`

Main setup:

- Backbone: `ResNet-18`
- Initialization: pretrained on `ImageNet`
- Input size: `224 x 224`
- Batch size: `64`
- Epochs: `30`
- Fine-tuning mode: `full fine-tuning` (`FREEZE_BACKBONE = False`)
- Optimizer: `AdamW`
- Scheduler: `Linear warmup + CosineAnnealingLR`

Data preprocessing / augmentation observed in the notebook:

- `RandomHorizontalFlip`
- `Normalize`
- resize to the image size expected by the pretrained model

Result:

- Final test accuracy: `96.55%`

### 3.2 ViT model

Notebook:

- `../notebooks/cifar10_vit_transfer_learning.ipynb`

Model:

- `../models/vit_b16_cifar10.pth`

Main setup:

- Backbone: `ViT-B/16`
- Initialization: pretrained on `ImageNet`
- Input size: `224 x 224`
- Training strategy:
  - Phase 1: freeze backbone for `3` epochs with `lr = 0.001`
  - Phase 2: fine-tune the full model for `15` epochs with `lr = 3e-05`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`

Data preprocessing / augmentation observed in the notebook:

- `Resize`
- `Normalize`

Result:

- Final test accuracy: `98.36%`

### 3.3 Saved experiment outputs

The project already contains trained models and exported calibration artifacts:

- `../models/resnet18_cifar10.pth`
- `../models/vit_b16_cifar10.pth`
- `../artifacts/cnn/resnet18_calibration_full.png`
- `../artifacts/cnn/resnet18_calibration_metrics_full.json`
- `../artifacts/vit/vit_b16_calibration_full.png`
- `../artifacts/vit/vit_b16_calibration_metrics_full.json`

## 4. Evaluation Summary

### Main comparison

| Model | Family | Final Test Accuracy | Notes |
|---|---|---:|---|
| ResNet-18 | CNN | `96.55%` | Transfer learning from ImageNet, full fine-tuning |
| ViT-B/16 | Vision Transformer | `98.36%` | Two-phase training: freeze then full fine-tune |

### Current conclusion

- Both models achieved strong performance on CIFAR-10.
- `ViT-B/16` achieved higher final test accuracy than `ResNet-18`.
- The image section already satisfies the core comparison requirement for `CNN vs ViT`.

## 5. Completed Extension Work

I selected and completed three extension directions for the image part:

- `Application demo`
- `Interpretability`
- `Calibration`

### 5.1 Application demo

Implementation:

- `../../app/main.py`
- `../../app/image/resnet18.py`
- `../../app/image/vit_b16.py`

What the demo currently supports:

- image upload,
- prediction with `CNN` or `ViT`,
- interpretability visualization,
- calibration analysis.

### 5.2 Interpretability

Implemented methods:

- `Grad-CAM` for the CNN model
- attention-based visualization for the ViT model

Purpose:

- to show which image regions influenced the classification decision,
- to support qualitative comparison between the two model families.

### 5.3 Calibration

Metrics and outputs:

- `ECE (Expected Calibration Error)`
- `Reliability Diagram`
- `Confidence Distribution`

Full-test calibration results from exported notebook artifacts:

| Model | ECE | Evaluated Samples |
|---|---:|---:|
| ResNet-18 | `0.020006` | `10,000` |
| ViT-B/16 | `0.006917` | `10,000` |

Current interpretation:

- Lower ECE is better.
- The current exported results indicate that `ViT-B/16` is better calibrated than `ResNet-18` on the CIFAR-10 test set.

### 5.4 Extension conclusion

The image extension part is already complete and covers:

- a usable web demo,
- model interpretability,
- confidence calibration.

This is a strong extension package for the image section.

## 6. What Has Been Finished vs What Is Still Missing

### Finished

- Selected a valid image dataset: `CIFAR-10`
- Built and trained a `CNN` model
- Built and trained a `ViT` model
- Saved trained model checkpoints
- Produced evaluation outputs in notebooks
- Produced calibration artifacts
- Built a Gradio-based image demo
- Added interpretability support
- Added calibration support

### Not finished yet

- convert the completed work into polished report content,
- prepare final tables and figures for GitHub Pages / slides,
- prepare short written discussion and conclusion,
- prepare demo video assets,
- integrate the image section into the full team deliverable page,
- coordinate with teammates so the final Assignment 1 page includes `image + text + multimodal`.

## 7. Remaining Tasks

### 7.1 Tasks for the image report

I still need to write a clean report section for the image part, including:

1. problem statement and dataset choice,
2. dataset exploration and justification,
3. preprocessing, DataLoader, and augmentation setup,
4. CNN model design and training setup,
5. ViT model design and training setup,
6. evaluation protocol and fair comparison,
7. result analysis,
8. extension analysis:
   - demo,
   - interpretability,
   - calibration,
9. final conclusion for the image section.

### 7.2 Figures and tables I should prepare next

Recommended assets for the final report / GitHub Pages / slides:

- one comparison table for `CNN vs ViT`,
- confusion matrix for `ResNet-18`,
- confusion matrix for `ViT-B/16`,
- calibration figure for `ResNet-18`,
- calibration figure for `ViT-B/16`,
- 2 to 4 interpretability examples,
- screenshots of the web demo,
- optional training curves from the notebooks.

### 7.3 Suggested discussion points

For the final written analysis, I should explain:

- why transfer learning was used,
- why CIFAR-10 is suitable for this assignment,
- why ViT achieved better accuracy,
- how the calibration results differ between CNN and ViT,
- whether the interpretability outputs support the prediction behavior,
- the trade-off between model performance and model complexity.

### 7.4 Group-level tasks still needed

Even if the image extension is done, the team still needs to complete:

- the GitHub Pages `home page`,
- the dedicated `Assignment 1` page,
- the `demo video` link,
- the `presentation video` link,
- the code link,
- the presentation/report content required by the assignment,
- the slide deck for LMS submission.

Important note:

- `GitHub Pages` is static, so it is suitable for presenting results, screenshots, figures, markdown content, and links.
- The live Gradio app itself is better shown through screenshots, video, or an external deployment link if needed.

## 8. Recommended Immediate Next Steps

My suggested order of work from now:

1. Write the final `image report` text.
2. Export or collect all final figures and screenshots.
3. Create one clean `CNN vs ViT` summary table.
4. Record a short image-demo video script.
5. Hand off the image section content to the teammate preparing GitHub Pages.
6. Merge the image section into the group presentation slides.

## 9. Deadline Reminder

- `Report 1`: due before `23:59, 26 March 2026`
- `Final report`: due before `23:59, 06 April 2026`

## 10. Final Status

### Current status of the image section

The `image` part of Assignment 1 is already complete at the implementation level:

- core model comparison is done,
- trained models are saved,
- extension work is done,
- and the demo app is working.

### What remains

The remaining work is mainly:

- documentation,
- packaging,
- presentation,
- and integration into the final team deliverables.

In short:

`The technical work for the image section is mostly finished. The next phase is report-writing and presentation packaging.`


