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
import sys
from pathlib import Path
from typing import Any, Dict, List

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT_ROOT))

import numpy as np
from PIL import Image
import streamlit as st

from app.image.resnet18 import StanfordDogsResNet18Handler
from app.image.vit_b16 import StanfordDogsViTHandler
from app.shared.model_assets import ensure_model_checkpoint


REPO_ROOT = ASSIGNMENT_ROOT.parent.parent
MODEL_ROOT = ASSIGNMENT_ROOT / "image" / "models"

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


def run_app() -> None:
    st.set_page_config(
        page_title="Stanford Dogs Image Demo",
        page_icon="🐶",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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


if __name__ == "__main__":
    run_app()
