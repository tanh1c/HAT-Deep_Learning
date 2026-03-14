# Repo Structure Guide

This guide proposes a cleaner repository layout for the full course project so all three members can work in one repo without mixing files.

## Recommended target structure

```text
docs/
  index.html
  assignment-1/
  assignment-2/
  assignment-3/
  assets/

assignments/
  assignment-1/
    app/
      main.py
      shared/
      image/
      text/
      multimodal/
    image/
      notebooks/
      reports/
      artifacts/
      models/
    text/
      notebooks/
      reports/
      artifacts/
      models/
    multimodal/
      notebooks/
      reports/
      artifacts/
      models/
    shared/
      brief/

  assignment-2/
  assignment-3/
```

## Why this layout is better

- `docs/` stays dedicated to GitHub Pages only.
- `assignments/assignment-1/` becomes the real working area for the team.
- Each member gets a clean space:
  - `image/`
  - `text/`
  - `multimodal/`
- The web app for Assignment 1 stays in one place, but the code can still be split by modality inside `app/`.
- Future assignments can reuse the same pattern without making the root folder messy.

## Suggested mapping from the current repo

### Move image notebooks and reports

```text
cifar10_resnet18_transfer_learning.ipynb
-> assignments/assignment-1/image/notebooks/

cifar10_vit_transfer_learning.ipynb
-> assignments/assignment-1/image/notebooks/

image_progress_report.md
-> assignments/assignment-1/image/reports/

image_presentation_content.md
-> assignments/assignment-1/image/reports/
```

### Move image artifacts and models

```text
artifacts/cnn/*
artifacts/vit/*
-> assignments/assignment-1/image/artifacts/

models/*
-> assignments/assignment-1/image/models/
```

Note:
- `models/vit_b16_cifar10.pth` is too large for a normal GitHub push.
- For checkpoints, prefer one of these:
  - Git LFS
  - Google Drive link
  - GitHub Release asset

### Move Assignment 1 app code

```text
app/main.py
-> assignments/assignment-1/app/main.py

app/artifact_utils.py
-> assignments/assignment-1/app/shared/artifact_utils.py

app/model_registry.py
-> assignments/assignment-1/app/shared/model_registry.py

app/cifar10_data.py
-> assignments/assignment-1/app/image/data.py

app/cifar10_resnet18.py
-> assignments/assignment-1/app/image/resnet18.py

app/cifar10_vit.py
-> assignments/assignment-1/app/image/vit_b16.py

app/style.css
-> assignments/assignment-1/app/assets/style.css
```

When text and multimodal are added later, teammates can use:

```text
assignments/assignment-1/app/text/
assignments/assignment-1/app/multimodal/
```

### Move the assignment brief

```text
assignment-one.md
assignment-one.pdf
-> assignments/assignment-1/shared/brief/
```

## Team allocation

### Chu Nguyen Tuan Anh

Owns:

```text
assignments/assignment-1/image/
assignments/assignment-1/app/image/
docs/assignment-1/
```

### Nguyen Quoc Hieu

Owns:

```text
assignments/assignment-1/text/
assignments/assignment-1/app/text/
```

### Vu Hai Tuan

Owns:

```text
assignments/assignment-1/multimodal/
assignments/assignment-1/app/multimodal/
```

### Shared ownership

```text
docs/
assignments/assignment-1/app/main.py
assignments/assignment-1/app/shared/
```

## What should stay at the repo root

Keep only high-level folders at the root:

- `docs/`
- `assignments/`
- `.gitignore`
- `README.md`

Avoid leaving Assignment 1 notebooks, models, data archives, or temporary experiments directly in the root.

## Recommended push strategy

### Safe first push

- push `docs/`
- push `app` code after it is reorganized
- push notebooks and reports
- do **not** push `data/`
- do **not** push large checkpoints unless using Git LFS
- do **not** push `.claude/`, `.codex/`, `old/`, `ref/`

### Best practice

- keep source code and reports in Git
- keep dataset archives out of Git
- keep large weights in Git LFS or external storage
- keep GitHub Pages assets in `docs/assets/`

## Practical advice

If you want the lowest-risk path:

1. Keep the current repo working.
2. Create the new `assignments/assignment-1/...` folders.
3. Move image files first.
4. Let teammates fill `text/` and `multimodal/`.
5. Move the Assignment 1 app only after all three tracks agree on the shared app structure.

This avoids breaking the current working demo while still giving the team a clean target structure.
