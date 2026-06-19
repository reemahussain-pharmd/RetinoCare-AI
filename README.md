---
title: RetinoCare AI
emoji: 👁
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
---

# RetinoCare AI — Diabetic Retinopathy Detection System

**AI-Powered Retinopathy Detection and Severity Classification Using Deep Learning and Explainable AI**

End-to-end healthcare AI project: real retinal fundus images → multi-model deep learning → Grad-CAM explainability → interactive Streamlit web app → downloadable PDF clinical report.

---

## Quick Start

```powershell
# Train + evaluate + Grad-CAM (completes in ~3 min on CPU)
venv\Scripts\python.exe scripts\quick_run.py

# Launch web app
venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open `http://localhost:8501` — upload a retinal image, get severity prediction, Grad-CAM heatmap, and PDF report.

---

## Results (MobileNetV2 — 3 epochs, CPU, 1,764 images)

| Metric | Score |
|--------|-------|
| Test Accuracy | **79.6%** |
| Macro AUC | **0.9006** |
| Macro F1 | **0.7323** |
| Macro Precision | **0.7683** |
| Macro Recall | **0.7327** |

---

## Problem Statement

Diabetic retinopathy affects ~103 million people worldwide and is the **leading cause of preventable blindness**. Early detection enables timely intervention, but specialist availability is limited — especially in resource-constrained settings. This system provides AI-assisted triage:

| Class | Risk | Action |
|-------|------|--------|
| No/Mild DR | Low | Annual screening |
| Moderate DR | Moderate | Ophthalmologist referral (3–6 months) |
| Severe/Proliferative DR | High / Critical | URGENT referral — vision at risk |

---

## Architecture

```
Retinal Fundus Image (any resolution)
        ↓
CLAHE Contrast Enhancement (reveals microaneurysms, exudates)
        ↓
Resize to 224×224 + ImageNet Normalization
        ↓
Deep Learning Model (MobileNetV2 / EfficientNetB0 / ResNet50 / DenseNet121 / CNN)
        ↓
Softmax Probabilities (3 severity classes)
        ↓
Grad-CAM + Saliency Maps (explainability layer)
        ↓
Streamlit Web App + ReportLab PDF Clinical Report
```

---

## Models

| Model | Type | Trainable Params |
|-------|------|-----------------|
| CNN (Scratch) | Baseline | ~2M |
| MobileNetV2 | Transfer Learning | ~3.4M (head only) |
| EfficientNetB0 | Transfer Learning | ~5.3M (head only) |
| ResNet50 | Transfer Learning | ~25M (head only) |
| DenseNet121 | Transfer Learning | ~8M (head only) |

**Strategy**: Freeze ImageNet base → train Dense(256)→BN→Dropout(0.5)→Dense(3,Softmax) head.

---

## Key Techniques

| Category | Technique |
|----------|-----------|
| Preprocessing | CLAHE, ImageNet normalization, tf.data streaming (no disk cache) |
| Training | Class weights (2.11× imbalance), label smoothing α=0.1, data augmentation |
| Callbacks | EarlyStopping (val_auc), ReduceLROnPlateau, ModelCheckpoint |
| Evaluation | Macro AUC, confusion matrix, per-class ROC, error analysis |
| Explainability | Grad-CAM (final conv layer), Saliency Maps (input gradients) |
| App | Streamlit 6-page app, session state, ReportLab PDF |

---

## Project Structure

```
RetinoCare-AI/
├── data/raw/
│   ├── images/              # 1,764 retinal fundus images
│   └── labels.csv           # image filename + severity label (0/1/2)
├── scripts/
│   └── quick_run.py         # self-contained pipeline (EDA→train→eval→XAI)
├── src/
│   ├── preprocessing/       # CLAHE, augmentation, normalization
│   ├── training/            # model builders
│   ├── evaluation/          # metrics, confusion matrix, ROC
│   ├── explainability/      # Grad-CAM, Saliency Maps
│   └── inference/           # predictor, PDF report generator
├── app/
│   └── streamlit_app.py     # 6-page web application
├── models/
│   └── best_model.keras     # best trained model
├── outputs/
│   ├── eda/                 # EDA plots
│   ├── xai/                 # Grad-CAM + saliency per class
│   └── *.png                # training curves, confusion matrices, ROC
├── reports/                 # PDF clinical reports
└── docs/
    └── interview_qa.md      # 30 interview Q&A
```

---

## Setup

```powershell
# 1. Create venv
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install tensorflow-cpu==2.15.0 streamlit opencv-python scikit-learn pandas matplotlib seaborn reportlab pillow streamlit-option-menu
pip install numpy==1.26.4 --force-reinstall

# 3. Data: put images in data/raw/images/, labels in data/raw/labels.csv
#    labels.csv columns: image (filename), label (0/1/2)
```

---

## Web App Pages

| Page | Description |
|------|-------------|
| Home | Overview, clinical context, system architecture |
| Upload & Predict | Upload image → CLAHE enhancement → prediction |
| Prediction Result | Class, confidence %, probability chart, clinical recommendation |
| Grad-CAM Explainability | Original + heatmap overlay + clinical interpretation |
| Analytics Dashboard | Model comparison, performance charts, dataset distribution |
| About | Models, methods, tech stack, ethical disclaimer |

---

## Outputs

```
outputs/
├── eda/class_distribution.png   preprocessing_steps.png
├── eda/image_dimensions.png     augmentation_pipeline.png
├── eda/pixel_intensity.png      mobilenetv2_training_curves.png
├── eda/sample_images.png        mobilenetv2_confusion_matrix.png
├── eda/outliers.png             mobilenetv2_roc_curves.png
├── model_comparison.csv         error_analysis.png
└── xai/gradcam_all_classes.png  xai/<ClassName>.png
```

---

## Clinical Disclaimer

> **IMPORTANT**: RetinoCare AI is an **assistive diagnostic tool** for research and portfolio purposes only. It is **NOT a replacement** for professional ophthalmological evaluation. All predictions must be reviewed by a qualified ophthalmologist before any clinical decision is made.

---

## Technologies

`Python 3.11` · `TensorFlow 2.15` · `Keras` · `OpenCV` · `scikit-learn` · `Streamlit` · `ReportLab` · `Matplotlib` · `Seaborn` · `Pandas` · `NumPy` · `PIL`
