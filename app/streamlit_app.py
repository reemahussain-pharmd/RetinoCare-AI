"""
RetinoCare AI — Streamlit Web Application
Multi-page interface for retinopathy detection, Grad-CAM, analytics dashboard.
"""

import os
import sys
import io
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.image_preprocessor import preprocess_single, enhance_contrast_clahe, resize_image
from src.explainability.grad_cam import GradCAM, SaliencyMap
from src.inference.report_generator import generate_pdf_report

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinoCare AI",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]
SEVERITY_COLORS = {
    "No/Mild DR":              "#4CAF50",
    "Moderate DR":             "#FFC107",
    "Severe/Proliferative DR": "#F44336",
}
SEVERITY_EMOJI = {
    "No/Mild DR":              "✅",
    "Moderate DR":             "🟠",
    "Severe/Proliferative DR": "🚨",
}

# Support both .keras and .h5 model formats
_keras_path = ROOT / "models" / "best_model.keras"
_h5_path    = ROOT / "models" / "best_model.h5"
MODEL_PATH  = str(_keras_path if _keras_path.exists() else _h5_path)

DISCLAIMER = (
    "⚠️ **AI DISCLAIMER:** This system is an **assistive diagnostic tool** and is "
    "**NOT a replacement for professional ophthalmological evaluation**. All predictions "
    "must be reviewed and confirmed by a qualified ophthalmologist before any clinical "
    "decision is made."
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        color: white; padding: 20px 30px; border-radius: 12px;
        text-align: center; margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #1565C0;
        padding: 15px; border-radius: 8px; margin: 8px 0;
    }
    .severity-badge {
        display: inline-block; padding: 6px 16px;
        border-radius: 20px; font-weight: bold;
        font-size: 16px; color: white;
    }
    .disclaimer-box {
        background: #fff3e0; border: 1px solid #ff9800;
        border-radius: 8px; padding: 12px; font-size: 13px;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Model Loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import traceback
    path = Path(MODEL_PATH)
    if not path.exists():
        return None, f"File not found: {MODEL_PATH}"
    size_bytes = path.stat().st_size
    if size_bytes < 10_000:
        # LFS pointer files are ~134 bytes; real models are several MB
        try:
            preview = path.read_text(errors="replace")[:200]
        except Exception:
            preview = ""
        return None, f"Model file looks like an LFS pointer ({size_bytes} bytes). LFS may not have been resolved in the Docker build context.\n\nFile preview:\n{preview}"
    try:
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception:
        return None, traceback.format_exc()


# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px;">
        <h2 style="color:#1565C0;">👁 RetinoCare AI</h2>
        <p style="color:#666; font-size:12px;">AI-Powered Retinopathy Detection</p>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload & Predict", "Prediction Result",
                 "Grad-CAM Explainability", "Analytics Dashboard", "About"],
        icons=["house", "upload", "activity", "eye", "bar-chart", "info-circle"],
        default_index=0,
        styles={
            "container":     {"background-color": "#f0f2f6"},
            "icon":          {"color": "#1565C0", "font-size": "16px"},
            "nav-link":      {"font-size": "14px"},
            "nav-link-selected": {"background-color": "#1565C0"},
        },
    )

    st.markdown("---")
    model, _model_err = load_model()
    if model:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ No model found. Train first.")
        if _model_err:
            with st.expander("Show error details"):
                st.code(_model_err)

    st.markdown("""
    <div style="font-size:11px; color:#999; text-align:center; margin-top:20px;">
        RetinoCare AI v1.0<br>For Research Purposes Only
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Home":
    st.markdown("""
    <div class="main-header">
        <h1>👁 RetinoCare AI</h1>
        <h3>AI-Powered Retinopathy Detection & Severity Classification</h3>
        <p>Deep Learning | Explainable AI | Clinical Decision Support</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(DISCLAIMER)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Model Accuracy",  "~85%+",  "varies by model")
    col2.metric("📊 AUC Score",       "0.90+",  "macro OvR")
    col3.metric("🔬 F1 Score",        "0.85+",  "macro avg")
    col4.metric("⚡ Models Trained",  "2",      "")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🏥 Clinical Use Case")
        st.markdown("""
        Diabetic retinopathy affects **~103 million people** worldwide (2020) and is the
        leading cause of preventable blindness. Early AI-assisted detection enables:

        - **Faster screening** at scale in remote/resource-limited settings
        - **Consistent grading** across all patient images
        - **Explainable decisions** clinicians can verify with Grad-CAM
        - **Automated referral triage** based on severity classification
        """)

    with col_b:
        st.subheader("🔬 Retinopathy Severity Scale")
        severity_data = {
            "Stage":    CLASS_NAMES,
            "Risk":     ["Low", "Moderate", "High / Critical"],
            "Action":   ["Annual screening",
                         "Ophthalmology referral 3-6 months",
                         "URGENT referral — vision at risk"],
        }
        st.dataframe(pd.DataFrame(severity_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("🏗 System Architecture")
    st.markdown("""
    ```
    User Uploads Retina Image
            ↓
    Frontend UI (Streamlit)
            ↓
    Preprocessing Layer (CLAHE + Normalization)
            ↓
    Deep Learning Model (EfficientNet / ResNet / DenseNet)
            ↓
    Explainable AI Layer (Grad-CAM + SHAP + Saliency)
            ↓
    Clinical Report Generator (PDF)
            ↓
    Prediction Dashboard
    ```
    """)

    st.info("👆 Use the sidebar to navigate to **Upload & Predict** to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Upload & Predict":
    st.title("📤 Upload Retinal Image")
    st.markdown(DISCLAIMER)
    st.markdown("---")

    col_upload, col_options = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Supported formats: JPG, PNG, BMP, TIFF",
        )

    with col_options:
        st.subheader("⚙️ Options")
        apply_clahe = st.checkbox("Apply CLAHE Enhancement", value=True)
        use_tta     = st.checkbox("Test-Time Augmentation (TTA)", value=False)
        gen_report  = st.checkbox("Generate PDF Report", value=True)
        show_gcam   = st.checkbox("Generate Grad-CAM", value=True)

    if uploaded_file:
        # Show image
        img = Image.open(uploaded_file).convert("RGB")
        img_arr = np.array(img)

        col1, col2 = st.columns(2)
        col1.subheader("📷 Uploaded Image")
        col1.image(img, caption="Original Retinal Image", use_column_width=True)

        col2.subheader("📊 Image Properties")
        col2.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Filename | `{uploaded_file.name}` |
        | Size     | `{img.size[0]} × {img.size[1]} px` |
        | Format   | `{img.format or 'N/A'}` |
        | File size| `{uploaded_file.size / 1024:.1f} KB` |
        """)

        if apply_clahe:
            enhanced = enhance_contrast_clahe(img_arr)
            col2.image(enhanced, caption="CLAHE Enhanced", use_column_width=True)

        st.markdown("---")
        predict_btn = st.button("🔍 Run Prediction", type="primary", use_container_width=True)

        if predict_btn:
            if model is None:
                st.error("No trained model found. Please train a model first (run `main.py`).")
            else:
                with st.spinner("Analyzing retinal image..."):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        img.save(tmp.name)
                        tmp_path = tmp.name

                    from src.inference.predictor import RetinopathyPredictor
                    predictor = RetinopathyPredictor(MODEL_PATH)

                    if use_tta:
                        result = predictor.test_time_augmentation(tmp_path, n_augments=10)
                    else:
                        result = predictor.predict_from_path(tmp_path)

                    st.session_state["prediction"] = result
                    st.session_state["tmp_path"]   = tmp_path
                    st.session_state["img_array"]  = img_arr

                    if show_gcam:
                        gcam_path = "outputs/latest_gradcam.png"
                        gcam_result = predictor.predict_with_gradcam(tmp_path, save_path=gcam_path)
                        st.session_state["gradcam"] = gcam_result.get("gradcam_overlay")

                st.success("✅ Analysis complete! Navigate to **Prediction Result** page.")
                st.balloons()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION RESULT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Prediction Result":
    st.title("🎯 Prediction Result")

    if "prediction" not in st.session_state:
        st.warning("No prediction yet. Please upload an image first.")
        st.stop()

    result = st.session_state["prediction"]
    cls    = result["predicted_class"]
    conf   = result["confidence"]
    color  = result["risk_color"]
    emoji  = SEVERITY_EMOJI.get(cls, "🔵")

    # ── Result Banner ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{color}; color:white; padding:20px; border-radius:12px;
                text-align:center; margin-bottom:20px;">
        <h2>{emoji} {cls}</h2>
        <h4>Confidence: {conf:.1f}% | Risk: {result['risk_level']}</h4>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Row ──────────────────────────────────────────────────────────
    cols = st.columns(3)
    cols[0].metric("Predicted Class",  cls)
    cols[1].metric("Confidence",       f"{conf:.1f}%")
    cols[2].metric("Risk Level",       result["risk_level"])

    st.markdown("---")

    # ── Probability Chart ────────────────────────────────────────────────────
    col_chart, col_rec = st.columns([1, 1])

    with col_chart:
        st.subheader("📊 Class Probabilities")
        probs = result["all_probabilities"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(
            list(probs.keys()),
            list(probs.values()),
            color=[SEVERITY_COLORS[k] for k in probs],
        )
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        for bar, val in zip(bars, probs.values()):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9)
        ax.set_title("Retinopathy Severity Probabilities")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_rec:
        st.subheader("🏥 Clinical Recommendation")
        st.info(result["clinical_recommendation"])
        st.markdown(f"<div class='disclaimer-box'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Download Report ──────────────────────────────────────────────────────
    if st.button("📄 Generate & Download PDF Report", type="secondary"):
        with st.spinner("Generating clinical report..."):
            report_path = "reports/retinopathy_report.pdf"
            generate_pdf_report(
                prediction=result,
                original_image_array=st.session_state.get("img_array"),
                gradcam_array=st.session_state.get("gradcam"),
                save_path=report_path,
            )
            with open(report_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=f,
                    file_name="RetinoCare_AI_Report.pdf",
                    mime="application/pdf",
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GRAD-CAM EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Grad-CAM Explainability":
    st.title("🔍 Grad-CAM Explainability")

    st.markdown("""
    **Grad-CAM (Gradient-weighted Class Activation Mapping)** highlights the retinal regions
    that most influenced the AI prediction. Red/warm areas = high importance.

    > *"This shows the ophthalmologist exactly where the AI is looking — enabling trust and verification."*
    """)

    if "prediction" not in st.session_state or model is None:
        st.warning("Please upload an image and run prediction first.")
        st.stop()

    result    = st.session_state["prediction"]
    img_array = st.session_state.get("img_array")
    gradcam   = st.session_state.get("gradcam")

    col1, col2 = st.columns(2)

    if img_array is not None:
        col1.subheader("Original Image")
        col1.image(img_array, caption="Retinal Fundus Image", use_column_width=True)

    if gradcam is not None:
        col2.subheader("Grad-CAM Overlay")
        col2.image(gradcam, caption=f"AI Focus Areas — {result['predicted_class']}", use_column_width=True)

    st.markdown("---")
    st.subheader("📖 Clinical Interpretation of Grad-CAM")
    st.markdown(f"""
    | Region | Clinical Significance |
    |--------|----------------------|
    | 🔴 Red (high activation)  | Primary retinal lesion sites influencing prediction |
    | 🟡 Yellow (medium)        | Secondary contributing retinal features |
    | 🔵 Blue (low activation)  | Background regions with minimal AI influence |

    **Predicted Class:** `{result['predicted_class']}` | **Confidence:** `{result['confidence']:.1f}%`

    The highlighted areas correspond to retinal pathologies such as microaneurysms, exudates,
    haemorrhages, or neovascularization patterns — depending on severity stage.
    """)

    st.markdown(DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Analytics Dashboard":
    st.title("📈 Analytics Dashboard")

    # Load comparison if available
    comp_path = ROOT / "outputs" / "model_comparison.csv"
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)

        st.subheader("🏆 Model Comparison")
        st.dataframe(df_comp.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
            color="#c8f7c5",
        ), use_container_width=True)

        st.markdown("---")

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        metrics = ["Accuracy", "F1 Score", "AUC"]
        x = np.arange(len(df_comp))
        width = 0.25
        for i, m in enumerate(metrics):
            ax.bar(x + i*width, df_comp[m], width, label=m)
        ax.set_xticks(x + width)
        ax.set_xticklabels(df_comp["Model"], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info("Train models first to see the analytics dashboard. Run `main.py`.")

    # Session history
    st.markdown("---")
    st.subheader("📋 Prediction Session History")
    if "prediction" in st.session_state:
        r = st.session_state["prediction"]
        st.success(f"Last prediction: **{r['predicted_class']}** ({r['confidence']:.1f}% confidence)")
    else:
        st.info("No predictions in this session yet.")

    st.markdown("---")
    st.subheader("📊 Retinopathy Severity Distribution (Training Data)")
    ref_counts = [811, 569, 384]  # actual dataset distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_list = ["#4CAF50", "#FFC107", "#F44336"]
    ax.bar(CLASS_NAMES, ref_counts, color=colors_list, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(ref_counts):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)
    ax.set_title("Class Distribution in Dataset", fontweight="bold")
    ax.set_ylabel("Number of Images")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_column_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "About":
    st.title("ℹ️ About RetinoCare AI")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## AI-Powered Retinopathy Detection System

        **RetinoCare AI** is a portfolio-grade healthcare AI project demonstrating the application
        of deep learning and explainable AI for diabetic retinopathy grading from retinal fundus images.

        ### 🧠 Models Used
        | Model | Type | Parameters |
        |-------|------|-----------|
        | CNN (Scratch) | Baseline | ~2M |
        | MobileNetV2 | Transfer Learning | ~3.4M |
        | ResNet50 | Transfer Learning | ~25M |
        | EfficientNetB0 | Transfer Learning | ~5.3M |
        | EfficientNetB3 | Transfer Learning | ~12M |
        | DenseNet121 | Transfer Learning | ~8M |
        | InceptionV3 | Transfer Learning | ~23M |

        ### 🔬 Explainability Methods
        - **Grad-CAM**: Visualizes convolutional feature maps
        - **SHAP**: Pixel-level attribution scores
        - **Saliency Maps**: Gradient-based attention

        ### 📦 Technology Stack
        `Python` · `TensorFlow/Keras` · `OpenCV` · `SHAP` · `Streamlit` · `ReportLab`

        ### ⚠️ Ethical Disclaimer
        This system is developed for **academic and research purposes only**.
        It must not be used as a standalone clinical diagnostic tool.
        All AI predictions require validation by a qualified ophthalmologist.

        ### 📚 References
        - APTOS 2019 Blindness Detection Dataset
        - EyePACS Diabetic Retinopathy Dataset
        - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
        - Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
        """)

    with col2:
        st.markdown("""
        ### 📊 Project Stats
        """)
        st.metric("Models Trained", "2")
        st.metric("Best AUC", "0.90+")
        st.metric("Dataset Size", "1,764 images")
        st.metric("Classes", "3")

        st.markdown("---")
        st.markdown("""
        ### 🏗 Architecture
        ```
        Input Image (224×224)
              ↓
        CLAHE Preprocessing
              ↓
        Deep Learning Model
              ↓
        Grad-CAM + SHAP
              ↓
        Clinical Report PDF
        ```
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#999; font-size:12px;">
        RetinoCare AI · Built with TensorFlow, Keras & Streamlit ·
        For Educational Purposes Only
    </div>
    """, unsafe_allow_html=True)
