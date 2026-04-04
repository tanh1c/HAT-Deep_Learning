# Streamlit Community Cloud Deployment

This folder deploys the **Assignment 1 image demo** on **Streamlit Community Cloud**.

The app is intentionally limited to the **Stanford Dogs** image track so it stays
lighter than the older Gradio multi-tab demo.

## Entrypoint

- App file: `deploy/streamlit-cloud/app.py`
- Dependency file: `deploy/streamlit-cloud/requirements.txt`

When you create the app on Community Cloud, use:

- **Repository:** this repository
- **Branch:** `main`
- **File path:** `deploy/streamlit-cloud/app.py`

## Local run

Always run Streamlit from the **repo root** so the file paths behave the same way
as Community Cloud:

```bash
cd /path/to/HAT-Deep_Learning
streamlit run deploy/streamlit-cloud/app.py
```

## Model checkpoint strategy

The app resolves model files in this order:

1. local repository file under `assignments/assignment-1/image/artifacts/download/models/`
2. configured local absolute/relative path
3. cached downloaded file
4. remote Google Drive file or direct URL

This lets you:

- keep local development simple when `.pth` files already exist
- avoid committing large checkpoints into GitHub for Streamlit deployment

## Recommended secrets

In Streamlit Community Cloud, open **Advanced settings** and paste something like:

```toml
[models]
STANFORDDOGS_RESNET50_FULL_GDRIVE_ID = "your-google-drive-file-id"
STANFORDDOGS_RESNET50_STAGED_GDRIVE_ID = "your-google-drive-file-id"
STANFORDDOGS_VIT_B16_FULL_GDRIVE_ID = "your-google-drive-file-id"
STANFORDDOGS_VIT_B16_STAGED_GDRIVE_ID = "your-google-drive-file-id"
```

You can also use direct URLs instead:

```toml
[models]
STANFORDDOGS_RESNET50_FULL_URL = "https://example.com/stanforddogs_resnet50_full_finetune.pth"
STANFORDDOGS_RESNET50_STAGED_URL = "https://example.com/stanforddogs_resnet50_head_then_full.pth"
STANFORDDOGS_VIT_B16_FULL_URL = "https://example.com/stanforddogs_vit_b16_full_finetune.pth"
STANFORDDOGS_VIT_B16_STAGED_URL = "https://example.com/stanforddogs_vit_b16_head_then_full.pth"
```

Or local custom paths for local development:

```toml
[models]
STANFORDDOGS_RESNET50_FULL_PATH = "assignments/assignment-1/image/artifacts/download/models/stanforddogs_resnet50_full_finetune.pth"
STANFORDDOGS_RESNET50_STAGED_PATH = "assignments/assignment-1/image/artifacts/download/models/stanforddogs_resnet50_head_then_full.pth"
STANFORDDOGS_VIT_B16_FULL_PATH = "assignments/assignment-1/image/artifacts/download/models/stanforddogs_vit_b16_full_finetune.pth"
STANFORDDOGS_VIT_B16_STAGED_PATH = "assignments/assignment-1/image/artifacts/download/models/stanforddogs_vit_b16_head_then_full.pth"
```

## What the app shows

- upload one dog image
- choose one of the 4 fair-benchmark checkpoints
- top-k predictions
- Grad-CAM or attention-based explanation
- calibration artifact exported from the final notebook
- model info from the final Stanford Dogs experiment

## Notes

- The old Gradio app remains in the repo as a backup workflow.
- The Streamlit version is better suited for a simple free deployment target.
