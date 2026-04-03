"""
Streamlit image demo for the Assignment 1 Stanford Dogs models.

This app is intentionally focused on the image track only so it stays lightweight
enough for Streamlit Community Cloud. It reuses the existing model handlers and
artifact outputs from the final notebook.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT_ROOT))

import numpy as np
from PIL import Image
import streamlit as st
import torch
import wandb
from transformers import BertTokenizer, DistilBertForSequenceClassification

from app.image.resnet18 import StanfordDogsResNet18Handler
from app.image.vit_b16 import StanfordDogsViTHandler
from app.shared.model_assets import ensure_model_checkpoint


REPO_ROOT = ASSIGNMENT_ROOT.parent.parent
MODEL_ROOT = ASSIGNMENT_ROOT / "image" / "models"
TEXT_MODEL_ROOT = ASSIGNMENT_ROOT / "text" / "models"

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "stanforddogs_resnet18": {
        "label": "ResNet-18",
        "filename": "stanforddogs_resnet18.pth",
        "handler_cls": StanfordDogsResNet18Handler,
        "configured_path_key": "STANFORDDOGS_RESNET18_PATH",
        "configured_url_key": "STANFORDDOGS_RESNET18_URL",
        "configured_gdrive_key": "STANFORDDOGS_RESNET18_GDRIVE_ID",
        "family": "CNN",
    },
    "stanforddogs_vit_b16": {
        "label": "ViT-B/16",
        "filename": "stanforddogs_vit_b16.pth",
        "handler_cls": StanfordDogsViTHandler,
        "configured_path_key": "STANFORDDOGS_VIT_B16_PATH",
        "configured_url_key": "STANFORDDOGS_VIT_B16_URL",
        "configured_gdrive_key": "STANFORDDOGS_VIT_B16_GDRIVE_ID",
        "family": "Transformer",
    },
}

TEXT_ARTIFACT_NAME = (
    "nguyenquochieujff7-ho-chi-minh-city-university-of-technology/"
    "bert-models/DistilBERT_Full:v0"
)
TEXT_BASE_MODEL_NAME = "distilbert-base-uncased"
TEXT_NUM_LABELS = 20
TEXT_MAX_LENGTH = 512
TEXT_MAX_WORDS = 400
TEXT_LOCAL_MODEL_DIR = TEXT_MODEL_ROOT / "DistilBERT_Full-v0"

TEXT_CLASS_NAMES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

TEXT_CLASS_DISPLAY_NAMES = {
    "alt.atheism": "Atheism",
    "comp.graphics": "Computer Graphics",
    "comp.os.ms-windows.misc": "MS Windows",
    "comp.sys.ibm.pc.hardware": "IBM PC Hardware",
    "comp.sys.mac.hardware": "Mac Hardware",
    "comp.windows.x": "X Window System",
    "misc.forsale": "For Sale",
    "rec.autos": "Autos",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine",
    "sci.space": "Space",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "Politics - Guns",
    "talk.politics.mideast": "Politics - Middle East",
    "talk.politics.misc": "Politics - Misc",
    "talk.religion.misc": "Religion - Misc",
}


def _get_prediction_state_key(model_key: str) -> str:
    return f"prediction::{model_key}"


def _get_upload_state_key() -> str:
    return "active_image_upload"


def _serialize_uploaded_image(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def stash_uploaded_image(file_bytes: bytes, file_name: str, source: str) -> None:
    st.session_state[_get_upload_state_key()] = {
        "bytes": file_bytes,
        "name": file_name,
        "source": source,
    }


def get_stashed_image_payload() -> Dict[str, str] | None:
    payload = st.session_state.get(_get_upload_state_key())
    if not isinstance(payload, dict):
        return None
    return payload


def extract_submission_files(submission: Any) -> List[Any]:
    """Handle Streamlit chat_input submissions across dict-like/object variants."""
    if submission is None:
        return []

    if isinstance(submission, dict):
        files = submission.get("files")
        return files if isinstance(files, list) else []

    files = getattr(submission, "files", None)
    return files if isinstance(files, list) else []


@st.cache_resource(show_spinner=False)
def load_model_handler(model_key: str, resolved_model_path: str):
    spec = MODEL_SPECS[model_key]
    handler_cls = spec["handler_cls"]
    return handler_cls(resolved_model_path)


def get_model_handler(model_key: str):
    spec = MODEL_SPECS[model_key]
    resolved_path, source = ensure_model_checkpoint(
        filename=spec["filename"],
        default_local_dir=MODEL_ROOT,
        configured_path_key=spec["configured_path_key"],
        configured_url_key=spec["configured_url_key"],
        configured_gdrive_key=spec["configured_gdrive_key"],
        secrets=st.secrets,
    )
    handler = load_model_handler(model_key, str(resolved_path))
    return handler, resolved_path, source


def render_sidebar(selected_model_key: str) -> None:
    st.sidebar.title("Image Demo")
    st.sidebar.caption("Stanford Dogs · Streamlit deployment")
    st.sidebar.markdown(
        "[Assignment page](https://tanh1c.github.io/HAT-Deep_Learning/assignment-1/image.html)  \n"
        "[GitHub repository](https://github.com/tanh1c/HAT-Deep_Learning)"
    )

    spec = MODEL_SPECS[selected_model_key]
    st.sidebar.markdown("### Model Source Keys")
    st.sidebar.code(
        "\n".join(
            [
                spec["configured_path_key"],
                spec["configured_gdrive_key"],
                spec["configured_url_key"],
            ]
        ),
        language="text",
    )
    st.sidebar.info(
        "The app looks for a local `.pth` first. If it is missing, it can download "
        "the checkpoint from Google Drive or a direct URL defined in Streamlit secrets."
    )


def clean_text(text: str) -> str:
    lines = str(text).split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">") or stripped.startswith(":") or stripped.startswith("|"):
            continue

        if any(
            line.startswith(header)
            for header in [
                "From:",
                "Subject:",
                "Organization:",
                "Lines:",
                "Reply-To:",
                "NNTP-Posting-Host:",
            ]
        ):
            continue

        if stripped == "--":
            break

        cleaned.append(line)

    cleaned_text = " ".join(cleaned).strip()
    cleaned_text = re.sub(r"http\S+|ftp\S+|www\.\S+", "", cleaned_text)
    cleaned_text = re.sub(r"\S+@\S+", "", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned_text = re.sub(r"\[.*?deletia.*?\]", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\[.*?snip.*?\]", "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text


def truncate_text(text: str, max_words: int = TEXT_MAX_WORDS) -> str:
    words = str(text).split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return str(text)


def preprocess_text(text: str) -> str:
    return truncate_text(clean_text(text))


def resolve_text_checkpoint_path(artifact_dir: Path) -> Path:
    preferred_name = "best_model_distilbert_full_fine_tune.pt"
    direct_preferred = artifact_dir / preferred_name
    if direct_preferred.exists():
        return direct_preferred

    nested_preferred = sorted(artifact_dir.rglob(preferred_name))
    if nested_preferred:
        return nested_preferred[0]

    checkpoints = sorted(artifact_dir.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoint found inside artifact folder: {artifact_dir}"
        )
    return checkpoints[0]


def download_text_wandb_artifact() -> Path:
    TEXT_LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if any(TEXT_LOCAL_MODEL_DIR.rglob("*.pt")):
        return TEXT_LOCAL_MODEL_DIR

    model_secrets = st.secrets.get("models", {})
    wandb_api_key = (
        st.secrets.get("WANDB_API_KEY")
        or st.secrets.get("wandb_key")
        or model_secrets.get("WANDB_API_KEY")
        or model_secrets.get("wandb_key")
    )
    if not wandb_api_key:
        raise RuntimeError(
            "Missing W&B API key in Streamlit secrets. Set `WANDB_API_KEY` "
            "or `wandb_key`."
        )

    wandb.login(key=wandb_api_key, relogin=True)
    artifact = wandb.Api().artifact(TEXT_ARTIFACT_NAME, type="model")
    artifact_path = artifact.download(root=str(TEXT_LOCAL_MODEL_DIR))
    return Path(artifact_path)


@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_text_model_and_tokenizer():
    artifact_dir = download_text_wandb_artifact()
    checkpoint_path = resolve_text_checkpoint_path(artifact_dir)
    tokenizer = BertTokenizer.from_pretrained(TEXT_BASE_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        TEXT_BASE_MODEL_NAME,
        num_labels=TEXT_NUM_LABELS,
        seq_classif_dropout=0.1,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer, checkpoint_path


def predict_text(model, tokenizer, text: str, top_k: int = 5):
    normalized_text = preprocess_text(text)
    encoded = tokenizer(
        normalized_text,
        truncation=True,
        max_length=TEXT_MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, TEXT_NUM_LABELS))
    predictions = [
        {
            "rank": rank + 1,
            "class_id": int(idx),
            "label": TEXT_CLASS_DISPLAY_NAMES.get(
                TEXT_CLASS_NAMES[int(idx)],
                TEXT_CLASS_NAMES[int(idx)],
            ),
            "raw_label": TEXT_CLASS_NAMES[int(idx)],
            "probability": float(prob),
        }
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]
    return normalized_text, predictions


def render_text_sidebar() -> None:
    st.sidebar.title("Text Demo")
    st.sidebar.caption("20 Newsgroups - DistilBERT")
    st.sidebar.markdown(
        "[Assignment page](https://tanh1c.github.io/HAT-Deep_Learning/assignment-1/text.html)  \n"
        "[GitHub repository](https://github.com/HatakekkSheeshh/classification-finetune-bert)"
    )
    st.sidebar.markdown("### W&B Artifact")
    st.sidebar.code(TEXT_ARTIFACT_NAME, language="text")


def run_text_app() -> None:
    render_text_sidebar()
    st.title("20 Newsgroups Text Classification")
    st.caption("DistilBERT_Full fine-tuned checkpoint from W&B artifact")

    try:
        model, tokenizer, checkpoint_path = load_text_model_and_tokenizer()
        st.success(f"Loaded checkpoint: {checkpoint_path}")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    sample_text = (
        "NASA recently announced a new space telescope mission focused on observing "
        "distant galaxies and studying exoplanet atmospheres."
    )
    text_input = st.text_area("Input text to classify", value=sample_text, height=220)
    top_k = st.slider("Top-K labels", min_value=1, max_value=10, value=5)

    if st.button("Predict", type="primary"):
        if not text_input.strip():
            st.warning("Please enter text before prediction.")
            st.stop()

        cleaned_input, predictions = predict_text(
            model,
            tokenizer,
            text_input,
            top_k=top_k,
        )
        best = predictions[0]

        st.subheader("Prediction")
        st.metric("Predicted label", best["label"], f"{best['probability']:.2%}")

        st.subheader("Top-K probabilities")
        st.dataframe(predictions, use_container_width=True, hide_index=True)

        with st.expander("Preprocessed text"):
            st.write(cleaned_input)


def render_model_info(handler, source: str, resolved_path: Path) -> None:
    info = handler.get_model_info()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Architecture", info.get("Architecture", "N/A"))
    metric_cols[1].metric("Dataset", info.get("Dataset", "N/A"))
    metric_cols[2].metric("Test Accuracy", info.get("Test Accuracy", "N/A"))
    metric_cols[3].metric("Official-Test ECE", info.get("Official-Test ECE", "N/A"))

    with st.expander("Model details", expanded=False):
        st.markdown(f"- **Resolved source:** {source}")
        st.markdown(f"- **Resolved path:** `{resolved_path.as_posix()}`")
        for key, value in info.items():
            st.markdown(f"- **{key}:** {value}")


def render_top_predictions(labels: List[str], confidences: List[float], top_k: int = 5) -> None:
    top_predictions = sorted(
        zip(labels, confidences),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    st.subheader("Top predictions")
    for label, confidence in top_predictions:
        st.progress(
            min(100, max(0, int(round(confidence * 100)))),
            text=f"{label} — {confidence:.2%}",
        )


def render_calibration(handler) -> None:
    calibration = handler.get_calibration_data(max_samples=None)
    if calibration is None or calibration.reliability_diagram is None:
        st.warning("Calibration artifact is not available for this model.")
        return

    cols = st.columns([0.3, 0.7])
    cols[0].metric("ECE", f"{calibration.ece:.6f}")
    cols[0].caption(calibration.source or "Notebook artifact")
    cols[1].image(
        calibration.reliability_diagram,
        caption="Reliability diagram exported from the final notebook",
        width="stretch",
    )


def run_image_app() -> None:
    st.title("Stanford Dogs Image Demo")
    st.caption(
        "A lightweight Streamlit deployment for the Assignment 1 image track. "
        "The app reuses the final Stanford Dogs checkpoints, explanation outputs, "
        "and calibration artifacts."
    )

    selected_model_key = st.radio(
        "Choose model",
        options=list(MODEL_SPECS.keys()),
        format_func=lambda key: f"{MODEL_SPECS[key]['label']} · {MODEL_SPECS[key]['family']}",
        horizontal=True,
    )
    render_sidebar(selected_model_key)

    try:
        with st.spinner("Loading model..."):
            handler, resolved_path, source = get_model_handler(selected_model_key)
    except Exception as exc:
        st.error(f"Failed to initialize the selected model: {exc}")
        st.markdown("### Suggested Streamlit secrets")
        st.code(
            "\n".join(
                [
                    "[models]",
                    'STANFORDDOGS_RESNET18_GDRIVE_ID = "your-google-drive-file-id"',
                    'STANFORDDOGS_VIT_B16_GDRIVE_ID = "your-google-drive-file-id"',
                ]
            ),
            language="toml",
        )
        return

    render_model_info(handler, source, resolved_path)

    uploaded_file = st.file_uploader(
        "Upload a dog image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear dog breed image for the best qualitative explanation.",
    )

    attachment_submission = st.chat_input(
        "Paste an image from your clipboard or attach one here",
        accept_file=True,
        file_type=["jpg", "jpeg", "png", "webp"],
        key="clipboard_image_input",
        width="stretch",
    )

    attachment_files = extract_submission_files(attachment_submission)
    if attachment_files:
        attachment = attachment_files[0]
        stash_uploaded_image(
            attachment.getvalue(),
            attachment.name or "clipboard-image",
            "Clipboard / chat attachment",
        )
    elif uploaded_file is not None:
        stash_uploaded_image(
            uploaded_file.getvalue(),
            uploaded_file.name or "uploaded-image",
            "File uploader",
        )

    image_payload = get_stashed_image_payload()
    if image_payload is None:
        st.info("Upload one image to run prediction, explanation, and calibration review.")
        with st.expander("Calibration artifact preview", expanded=True):
            render_calibration(handler)
        return

    file_bytes = image_payload["bytes"]
    image_hash = _serialize_uploaded_image(file_bytes)
    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    preview_cols = st.columns([0.42, 0.58], gap="large")

    with preview_cols[0]:
        st.image(
            pil_image,
            caption=f"Active image · {image_payload.get('source', 'Unknown source')}",
            width="stretch",
        )
        predict_clicked = st.button(
            "Predict & Explain",
            type="primary",
            width="stretch",
        )
        with st.expander("Calibration artifact", expanded=False):
            render_calibration(handler)

    if predict_clicked:
        with st.spinner("Running inference and explanation..."):
            result = handler.predict(np.array(pil_image))
        st.session_state[_get_prediction_state_key(selected_model_key)] = {
            "image_hash": image_hash,
            "result": result,
        }

    cached_result = st.session_state.get(_get_prediction_state_key(selected_model_key))
    if not cached_result or cached_result["image_hash"] != image_hash:
        with preview_cols[1]:
            st.info("Click **Predict & Explain** to generate top-k predictions and the interpretability view.")
        return

    result = cached_result["result"]
    with preview_cols[1]:
        st.subheader("Prediction summary")
        st.metric("Predicted breed", result.label)
        st.metric("Top-1 confidence", f"{result.confidence:.2%}")
        render_top_predictions(result.all_labels, result.all_confidences)

    st.subheader("Interpretability view")
    st.image(
        result.explanation_image,
        caption=(
            "Grad-CAM for ResNet-18 or attention overlay for ViT-B/16. "
            "This is generated directly from the loaded checkpoint."
        ),
        width="stretch",
    )


def run_app() -> None:
    st.set_page_config(
        page_title="Assignment 1 Demo",
        page_icon="A1",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    selected_track = st.sidebar.radio(
        "Choose demo track",
        ["Image", "Text"],
        horizontal=True,
    )

    if selected_track == "Text":
        run_text_app()
        return

    run_image_app()


if __name__ == "__main__":
    run_app()
