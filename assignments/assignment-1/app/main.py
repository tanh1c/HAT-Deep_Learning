"""
Deep Learning Assignment 1 - Application Demo
===============================================
A modular Gradio application for demonstrating
trained models on Image, Text, and Multimodal datasets.

Features:
- Image classification with Grad-CAM / attention visualization
- Model Calibration analysis (ECE + Reliability Diagram)
- Easy to extend with new models/datasets

Usage:
    python assignments/assignment-1/app/main.py
"""

import sys
import os

# Add assignment root to path so `app.*` imports keep working.
ASSIGNMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ASSIGNMENT_ROOT)

import gradio as gr
from typing import Dict

from app.shared.model_registry import (
    register_model,
    get_all_model_keys,
    get_models_by_type,
    BaseModelHandler,
)
from app.image.resnet18 import StanfordDogsResNet18Handler
from app.image.vit_b16 import StanfordDogsViTHandler


# ============================================================================
# CONFIGURATION
# ============================================================================

APP_TITLE = "🧠 Deep Learning Assignment 1 - Demo"
APP_DESCRIPTION = """
<div style="text-align: center; padding: 10px 0;">
    <p style="font-size: 16px; color: #8b949e; margin: 5px 0;">
        Classification on Images, Text, and Multimodal Data
    </p>
    <p style="font-size: 14px; color: #58a6ff; margin: 5px 0;">
        CO3091 · HCM University of Technology · 2025-2026 Semester 2
    </p>
</div>
"""

# Load custom CSS from external file
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        CUSTOM_CSS = f.read()
else:
    CUSTOM_CSS = ""


CUSTOM_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0d1117",
    body_background_fill_dark="#0d1117",
    block_background_fill="#161b22",
    block_background_fill_dark="#161b22",
    block_border_color="#30363d",
    block_border_color_dark="#30363d",
    block_label_text_color="#c9d1d9",
    block_label_text_color_dark="#c9d1d9",
    block_title_text_color="#f0f6fc",
    block_title_text_color_dark="#f0f6fc",
    body_text_color="#c9d1d9",
    body_text_color_dark="#c9d1d9",
    body_text_color_subdued="#8b949e",
    body_text_color_subdued_dark="#8b949e",
    button_primary_background_fill="#238636",
    button_primary_background_fill_dark="#238636",
    button_primary_background_fill_hover="#2ea043",
    button_primary_background_fill_hover_dark="#2ea043",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
    input_border_color="#30363d",
    input_border_color_dark="#30363d",
    shadow_drop="none",
    shadow_drop_lg="none",
)


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def init_models():
    """Initialize and register all available models."""
    model_dir = os.path.join(ASSIGNMENT_ROOT, "image", "models")

    # Stanford Dogs ResNet-18
    resnet18_path = os.path.join(model_dir, "stanforddogs_resnet18.pth")
    if os.path.exists(resnet18_path):
        try:
            handler = StanfordDogsResNet18Handler(resnet18_path)
            register_model("stanforddogs_resnet18", handler)
            print(f"✅ Loaded: Stanford Dogs ResNet-18 from {resnet18_path}")
        except Exception as e:
            print(f"❌ Failed to load Stanford Dogs ResNet-18: {e}")
    else:
        print(f"⚠️ Model file not found: {resnet18_path}")

    # Stanford Dogs ViT-B/16
    vit_path = os.path.join(model_dir, "stanforddogs_vit_b16.pth")
    if os.path.exists(vit_path):
        try:
            handler = StanfordDogsViTHandler(vit_path)
            register_model("stanforddogs_vit", handler)
            print(f"✅ Loaded: Stanford Dogs ViT-B/16 from {vit_path}")
        except Exception as e:
            print(f"❌ Failed to load Stanford Dogs ViT-B/16: {e}")
    else:
        print(f"⚠️ Model file not found: {vit_path}")


# ============================================================================
# UI BUILDER FUNCTIONS
# ============================================================================

def format_confidence_label(labels, confidences, top_k=5):
    """Format top-k predictions as a dictionary for gr.Label."""
    paired = sorted(zip(labels, confidences), key=lambda x: x[1], reverse=True)
    return {label: float(conf) for label, conf in paired[:top_k]}


def build_model_info_markdown(handler: BaseModelHandler) -> str:
    """Build formatted model info markdown."""
    info = handler.get_model_info()
    lines = ["### 📋 Model Information\n"]
    for key, val in info.items():
        lines.append(f"| **{key}** | {val} |")

    header = "| Property | Value |\n|:---|:---|\n"
    table_lines = [line for line in lines[1:]]
    return lines[0] + header + "\n".join(table_lines)


def build_image_prediction_tab(model_key: str, handler: BaseModelHandler):
    """Build the prediction tab UI for image models."""
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📸 Upload Image",
                type="numpy",
                height=300,
                sources=["upload", "clipboard"],
            )
            predict_btn = gr.Button(
                "🔍 Predict & Explain",
                variant="primary",
                size="lg",
            )
            gr.Markdown(
                f"*Classes: {', '.join(handler.get_class_labels())}*",
                elem_classes=["text-sm"],
            )

        with gr.Column(scale=1):
            output_label = gr.Label(
                label="📊 Prediction Results (Top-5)",
                num_top_classes=5,
            )

    with gr.Row():
        explanation_image = gr.Image(
            label="🔥 Model Explanation (Interpretability)",
            interactive=False,
            height=350,
        )

    def do_predict(image):
        if image is None:
            return None, None
        try:
            result = handler.predict(image)
            conf_dict = format_confidence_label(
                result.all_labels, result.all_confidences
            )
            return conf_dict, result.explanation_image
        except Exception as e:
            raise gr.Error(f"Prediction failed: {str(e)}")

    predict_btn.click(
        fn=do_predict,
        inputs=[input_image],
        outputs=[output_label, explanation_image],
    )


def build_calibration_tab(model_key: str, handler: BaseModelHandler):
    """Build the calibration analysis tab."""
    gr.Markdown("""
    ### 📐 Model Calibration Analysis

    Calibration measures how well the model's confidence matches its actual accuracy.
    A perfectly calibrated model has **confidence = accuracy** for all predictions.

    - **ECE (Expected Calibration Error)**: Lower is better (0 = perfect calibration)
    - **Reliability Diagram**: Compares predicted confidence vs actual accuracy per bin
    - **Notebook Artifact**: Uses the exported official-test calibration figure from the final notebook
    """)

    calibration_mode = gr.Radio(
        choices=[
            "Notebook Artifact (official test set)",
        ],
        value="Notebook Artifact (official test set)",
        label="Calibration Mode",
    )

    compute_btn = gr.Button(
        "📊 Compute Calibration",
        variant="primary",
        size="lg",
    )

    ece_display = gr.Markdown(visible=False)
    calibration_plot = gr.Image(
        label="📈 Calibration Analysis",
        interactive=False,
        visible=False,
        height=450,
    )

    def compute_calibration(mode):
        try:
            result = handler.get_calibration_data(max_samples=None)
            if result is None:
                raise gr.Error("Could not compute calibration data")

            sample_note = f"Official {handler.get_dataset_name()} test set artifact"
            source_note = result.source or "Live computation"
            ece_md = f"""
### Calibration Metrics

| Metric | Value |
|:---|:---|
| **Mode** | {sample_note} |
| **Source** | {source_note} |
| **Expected Calibration Error (ECE)** | `{result.ece:.6f}` |
| **Interpretation** | {'✅ Well calibrated' if result.ece < 0.05 else '⚠️ Moderately calibrated' if result.ece < 0.15 else '❌ Poorly calibrated'} |
| **Total evaluated samples** | {sum(result.bin_counts):,} |
"""
            return (
                gr.update(value=ece_md, visible=True),
                gr.update(value=result.reliability_diagram, visible=True),
            )
        except Exception as e:
            raise gr.Error(f"Calibration computation failed: {str(e)}")

    compute_btn.click(
        fn=compute_calibration,
        inputs=[calibration_mode],
        outputs=[ece_display, calibration_plot],
    )


def build_model_tabs(model_key: str, handler: BaseModelHandler):
    """Build all tabs for a specific model."""
    gr.Markdown(build_model_info_markdown(handler))

    with gr.Tabs():
        with gr.Tab("🎯 Predict & Explain", id="predict"):
            data_type = handler.get_data_type()
            if data_type == "image":
                build_image_prediction_tab(model_key, handler)
            elif data_type == "text":
                gr.Markdown("### 📝 Text Classification\n*Coming soon...*")
            elif data_type == "multimodal":
                gr.Markdown("### 🖼️+📝 Multimodal Classification\n*Coming soon...*")

        with gr.Tab("📐 Calibration", id="calibration"):
            build_calibration_tab(model_key, handler)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def create_app() -> gr.Blocks:
    """Create the main Gradio application."""
    init_models()

    with gr.Blocks(
        title="DL Assignment 1 - Demo",
    ) as app:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        model_keys = get_all_model_keys()

        if not model_keys:
            gr.Markdown("""
            ## ⚠️ No Models Loaded

            Please ensure model files are in the `image/models/` directory.
            See the README for instructions on adding models.
            """)
        else:
            image_models = get_models_by_type("image")
            text_models = get_models_by_type("text")
            multimodal_models = get_models_by_type("multimodal")

            with gr.Tabs():
                if image_models:
                    with gr.Tab("🖼️ Image Classification", id="image_tab"):
                        if len(image_models) > 1:
                            with gr.Tabs():
                                for key, handler in image_models.items():
                                    tab_name = f"{handler.get_model_name()} ({handler.get_dataset_name()})"
                                    with gr.Tab(tab_name):
                                        build_model_tabs(key, handler)
                        else:
                            key, handler = next(iter(image_models.items()))
                            build_model_tabs(key, handler)

                if text_models:
                    with gr.Tab("📝 Text Classification", id="text_tab"):
                        if len(text_models) > 1:
                            with gr.Tabs():
                                for key, handler in text_models.items():
                                    tab_name = f"{handler.get_model_name()} ({handler.get_dataset_name()})"
                                    with gr.Tab(tab_name):
                                        build_model_tabs(key, handler)
                        else:
                            key, handler = next(iter(text_models.items()))
                            build_model_tabs(key, handler)

                if multimodal_models:
                    with gr.Tab("🔀 Multimodal Classification", id="mm_tab"):
                        if len(multimodal_models) > 1:
                            with gr.Tabs():
                                for key, handler in multimodal_models.items():
                                    tab_name = f"{handler.get_model_name()} ({handler.get_dataset_name()})"
                                    with gr.Tab(tab_name):
                                        build_model_tabs(key, handler)
                        else:
                            key, handler = next(iter(multimodal_models.items()))
                            build_model_tabs(key, handler)

                if not text_models:
                    with gr.Tab("📝 Text Classification", id="text_tab"):
                        gr.Markdown("""
                        ### 📝 Text Classification Models

                        *No text models loaded yet. Add your text model handler
                        and register it in `app/main.py`.*
                        """)

                if not multimodal_models:
                    with gr.Tab("🔀 Multimodal Classification", id="mm_tab"):
                        gr.Markdown("""
                        ### 🔀 Multimodal Classification Models

                        *No multimodal models loaded yet. Add your multimodal
                        model handler and register it in `app/main.py`.*
                        """)

        gr.Markdown("""
        <div class="app-footer">
            <p>Deep Learning and Its Applications · Assignment 1</p>
            <p>HCM University of Technology (HCMUT) · VNUHCM</p>
        </div>
        """)

    return app


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=5555,
        share=False,
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
        allowed_paths=[os.path.join(ASSIGNMENT_ROOT, "image", "artifacts")],
    )
