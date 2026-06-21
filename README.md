---
title: RetinaIQ
emoji: 👁
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
license: mit
---

<div align="center">

# 👁 RetinaIQ

### AI-Powered Retinal Disease Screening Platform

[![Healthcare AI](https://img.shields.io/badge/Healthcare-AI-1B4F72?style=for-the-badge)](https://huggingface.co/spaces/reemahussain/RetinoCare_AI)
[![Computer Vision](https://img.shields.io/badge/Computer-Vision-117A65?style=for-the-badge)](https://huggingface.co/spaces/reemahussain/RetinoCare_AI)
[![Explainable AI](https://img.shields.io/badge/Explainable-AI-2E86C1?style=for-the-badge)](https://huggingface.co/spaces/reemahussain/RetinoCare_AI)
[![Clinical DSS](https://img.shields.io/badge/Clinical-Decision_Support-7D3C98?style=for-the-badge)](https://huggingface.co/spaces/reemahussain/RetinoCare_AI)
[![Deep Learning](https://img.shields.io/badge/Deep-Learning-E67E22?style=for-the-badge)](https://huggingface.co/spaces/reemahussain/RetinoCare_AI)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](.)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](.)
[![HF Spaces](https://img.shields.io/badge/Live_Demo-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/reemahussain/RetinoCare_AI)

**Built by a PharmD professional with AI/ML expertise — demonstrating end-to-end clinical AI development**

[🚀 Live Demo — Hugging Face Spaces](https://huggingface.co/spaces/reemahussain/RetinoCare_AI) &nbsp;·&nbsp;
[📂 GitHub Repository](https://github.com/reemahussain-pharmd/RetinaIQ) &nbsp;·&nbsp;
[🐛 Report Issue](https://github.com/reemahussain-pharmd/RetinaIQ/issues)

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Screenshots](#2-screenshots)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Features](#4-features)
5. [Dataset](#5-dataset)
6. [Model Performance](#6-model-performance)
7. [Clinical DSS](#7-clinical-decision-support-system)
8. [Explainability (Grad-CAM)](#8-explainability-grad-cam)
9. [Installation](#9-installation)
10. [Future Work](#10-future-work)
11. [Limitations](#11-limitations--disclaimer)
12. [License](#12-license)

---

## 1. Project Overview

Diabetic retinopathy (DR) is the **leading cause of preventable blindness** in working-age adults globally, affecting approximately **103 million people**. Up to **80% of vision loss** from DR is preventable through timely detection and treatment — yet specialist ophthalmologist availability remains severely limited in many healthcare settings.

**RetinaIQ** is a deep learning-powered retinal disease screening platform designed to assist in early detection, classification, and explainable analysis of retinal disorders from fundus images.

> **Current Module: Diabetic Retinopathy Detection & Classification** — using deep learning and computer vision.
> Future modules planned: Glaucoma screening, Age-Related Macular Degeneration (AMD), and broader retinal disease AI.

RetinaIQ demonstrates how **deep learning and explainable AI** can assist ophthalmologists in:

- **Automated severity grading** of diabetic retinopathy from retinal fundus photographs
- **Explainable predictions** via Grad-CAM activation maps that clinicians can verify
- **Personalised risk stratification** by combining image evidence with patient clinical metadata
- **Standardised clinical documentation** via structured PDF reports with evidence-based guidelines

### Clinical Classification System

| DR Grade | Risk Level | Clinical Action | ICD-10 |
|----------|-----------|-----------------|--------|
| No/Mild DR | Low | Annual routine screening | E11.319 |
| Moderate NPDR | Moderate | Ophthalmology referral within 3–6 months | E11.339 |
| Severe/Proliferative DR | High / Critical | **URGENT** referral within 24–72 hours | E11.359 |

### Why This Project Matters

This system is designed to illustrate a **complete clinical AI pipeline** — from raw image acquisition through to a clinician-ready report — including quality gates, uncertainty quantification, explainability, and regulatory-aware disclaimers. It reflects the skills required for **Clinical Data Scientist**, **Healthcare AI Analyst**, **Medical AI**, and **Digital Health** roles.

---

## 2. Screenshots

> _Screenshots from the deployed Hugging Face Spaces application_

| Page | Description |
|------|-------------|
| **Home** | Hero banner, KPI metrics, recruiter badges, model summary card, workflow pipeline |
| **Upload & Predict** | Image quality assessment, side-by-side CLAHE preview, inference timing |
| **Prediction Result** | Confidence warning system, probability chart, clinical recommendation, PDF download |
| **Grad-CAM XAI** | Heatmap overlay, dynamic clinical interpretation, retinal region reference |
| **Analytics Dashboard** | 6-model comparison, training splits, ROC curves, error analysis, dataset distribution |
| **Clinical DSS** | 6-factor risk scoring, methodology expander, automated clinical interpretation, guidelines |
| **Model Validation** | Strengths, limitations, future roadmap |
| **About** | Project objective, key contributions, clinical impact, tech stack |

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RetinaIQ Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

  Retinal Fundus Image (any resolution, RGB)
              │
              ▼
  ┌─────────────────────┐
  │  Image Quality Gate │  ← Laplacian blur score + brightness check
  │  (Excellent/Good/   │     Warns on blurry / dark / overexposed images
  │   Moderate/Poor)    │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  CLAHE Enhancement  │  ← Contrast Limited Adaptive Histogram Equalisation
  │  (OpenCV)           │     Reveals microaneurysms and exudates
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Resize 224×224     │
  │  ImageNet Norm      │  ← (pixel - mean) / std
  └──────────┬──────────┘
             │
             ▼
  ┌──────────────────────────────────────────┐
  │           MobileNetV2 (frozen)           │  ← Transfer Learning
  │           + Dense(256)                   │     ImageNet pretrained
  │           + BatchNorm                    │
  │           + Dropout(0.5)                 │
  │           + Dense(3, Softmax)            │
  └──────────┬───────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  Softmax      Grad-CAM
  Probs        Heatmap       ← Gradient-weighted class activation mapping
      │             │
      └──────┬──────┘
             │
             ▼
  ┌──────────────────────────────────────────┐
  │     Reliability Score                    │
  │     (Margin × 0.65 + Entropy × 0.35)    │
  └──────────┬───────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────┐
  │     Composite Risk Score                 │  ← DR Grade + HbA1c + Duration
  │     (6-factor weighted composite)        │     + BP + Age + Smoking
  └──────────┬───────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────┐
  │     PDF Clinical Report (v2.1)           │  ← ReportLab
  │     10 sections, matplotlib charts       │     Patient ID, Grad-CAM, Guidelines
  └──────────────────────────────────────────┘
```

---

## 4. Features

### Core Features

| Feature | Description |
|---------|-------------|
| **Multi-Class DR Grading** | 3-class classification: No/Mild, Moderate, Severe/Proliferative DR |
| **Image Quality Assessment** | Laplacian blur + brightness scoring with Excellent/Good/Moderate/Poor grading |
| **CLAHE Enhancement** | Side-by-side preprocessing comparison before inference |
| **Grad-CAM Explainability** | Class-specific activation heatmaps with dynamic clinical text interpretation |
| **Confidence Warning System** | High (>90%) / Moderate (70–90%) / Low (<70%) — with manual review flag |
| **Reliability Score** | Novel metric: prediction margin (65%) + Shannon entropy (35%) |
| **Composite Risk Scoring** | 6-factor weighted clinical risk score with component breakdown |
| **Automated Clinical Interpretation** | HbA1c, BP, duration, smoking — automated flag generation |
| **PDF Clinical Report v2.1** | 10-section structured report with embedded matplotlib charts |
| **Multi-Model Benchmarking** | 6-model comparison against published APTOS/EyePACS baselines |
| **Error Analysis Dashboard** | Systematic misclassification analysis with cause explanations |
| **Model Validation Page** | Scientific maturity: strengths, limitations, and future roadmap |
| **Evidence-Based Guidelines** | NICE NG28, AAO PPP 2019, IDF Atlas, ADA Standards — per DR grade |
| **Synthetic Demo Images** | 3 on-the-fly generated fundus images for hands-on demo without patient data |
| **Test-Time Augmentation** | Optional TTA over 10 augmented views for robustness |

### App Pages (8 total)

```
Home → Upload & Predict → Prediction Result → Grad-CAM XAI
     → Analytics Dashboard → Clinical DSS → Model Validation → About
```

---

## 5. Dataset

### Source
APTOS-style retinal fundus photographs, compiled for a 3-class severity grading task.

### Statistics

| Split | Images | Percentage |
|-------|--------|-----------|
| Training | 1,234 | 70% |
| Validation | 265 | 15% |
| Test | 265 | 15% |
| **Total** | **1,764** | 100% |

### Class Distribution

| Class | Count | Proportion | Weight Applied |
|-------|-------|-----------|----------------|
| No/Mild DR | 811 | 46.0% | 1.00× |
| Moderate DR | 569 | 32.3% | 1.43× |
| Severe/Proliferative DR | 384 | 21.8% | 2.11× |

### Class Imbalance Mitigation

- **Inverse-frequency class weights** — minority classes penalised more on misclassification
- **Label smoothing** (alpha=0.1) — reduces overconfident predictions, improves calibration
- **Data augmentation** — horizontal flip, rotation ±15°, brightness/contrast jitter

### Preprocessing Pipeline

```
Raw Image → CLAHE Enhancement → Resize 224×224 → ImageNet Normalisation
```

CLAHE (Contrast Limited Adaptive Histogram Equalisation) with clip limit 2.0 and 8×8 tile grid
improves visibility of microaneurysms, haemorrhages, and hard exudates.

---

## 6. Model Performance

### MobileNetV2 (Deployed)

| Metric | Score |
|--------|-------|
| Test Accuracy | **79.6%** |
| Macro AUC | **0.9006** |
| Weighted AUC | **0.9081** |
| Macro F1 | **0.7323** |
| Macro Precision | **0.7683** |
| Macro Recall | **0.7327** |
| Parameters | ~3.4M (head only) |
| Training | 3 epochs, CPU, EarlyStopping |

### Multi-Model Comparison

| Model | Accuracy | AUC | F1 Score | Status |
|-------|----------|-----|----------|--------|
| CNN (Scratch) | 0.731 | 0.832 | 0.701 | Trained on this dataset |
| **MobileNetV2** | **0.796** | **0.901** | **0.732** | **Deployed ✓** |
| EfficientNetB0 | 0.831 | 0.923 | 0.798 | Literature benchmark† |
| ResNet50 | 0.812 | 0.908 | 0.779 | Literature benchmark† |
| DenseNet121 | 0.841 | 0.931 | 0.817 | Literature benchmark† |
| InceptionV3 | 0.824 | 0.916 | 0.791 | Literature benchmark† |

_† Published values from APTOS/EyePACS literature. Actual performance varies by dataset and training setup._

### Training Configuration

```python
Model:       MobileNetV2 (ImageNet pretrained, frozen base)
Head:        Dense(256) → BatchNorm → Dropout(0.5) → Dense(3, Softmax)
Loss:        CategoricalCrossentropy(label_smoothing=0.1)
Optimiser:   Adam(lr=1e-4)
Callbacks:   EarlyStopping(monitor='val_auc', patience=5)
             ReduceLROnPlateau(factor=0.5, patience=3)
             ModelCheckpoint(save_best_only=True)
Batch Size:  32
Image Size:  224 × 224
Augmentation: HorizontalFlip, Rotation±15, BrightnessContrast
Class Weights: {0: 1.00, 1: 1.43, 2: 2.11}
```

---

## 7. Clinical Decision Support System

The Clinical DSS goes beyond image classification to integrate **patient metadata** for personalised risk assessment.

### Composite Risk Score Formula

```
Risk Score = DR_Grade×0.40 + HbA1c×0.22 + Duration×0.16 + BP×0.12 + Age×0.06 + Smoking×0.04
```

| Factor | Weight | Clinical Rationale |
|--------|--------|--------------------|
| DR Grade (AI result) | 40% | Strongest predictor — direct retinal imaging evidence |
| HbA1c | 22% | Chronic glycaemic control — most modifiable risk factor |
| Diabetes Duration | 16% | Cumulative microvascular damage |
| Blood Pressure | 12% | Hypertension accelerates DR progression |
| Age | 6% | Age-related vascular fragility |
| Smoking Status | 4% | Vasoconstriction and oxidative stress |

### Risk Categories

| Score | Category | Action |
|-------|----------|--------|
| 0–39 | Low Risk | Routine annual screening |
| 40–69 | Moderate Risk | Ophthalmology referral within 3–6 months |
| 70–100 | High Risk | Urgent specialist referral within 48 hours |

### Automated Clinical Interpretation

The system generates rule-based clinical flags for each risk factor:

- _"HbA1c critically elevated (>9%) — intensive glycaemic management essential."_
- _"Blood pressure hypertensive (>=140 mmHg) — antihypertensive therapy review required."_
- _"Active smoker — smoking cessation counselling strongly recommended."_

### Evidence-Based Guidelines Integration

NICE NG28 (2016) · AAO PPP 2019 · IDF Atlas 10th Ed. · ADA Standards 2023 · RCOphth 2020

---

## 8. Explainability (Grad-CAM)

**Gradient-weighted Class Activation Mapping (Grad-CAM)** (Selvaraju et al., ICCV 2017) identifies
the retinal regions that most influenced the model's classification decision.

### How It Works

```
Forward pass → Extract final conv layer activations
             → Compute gradient of class score w.r.t. activations
             → Global average pool gradients → class weights
             → Weighted sum of feature maps → ReLU → Overlay on image
```

### Dynamic Clinical Interpretation

Each predicted class generates a class-specific textual interpretation of the Grad-CAM:

| Class | Focus Pattern | Clinical Meaning |
|-------|--------------|------------------|
| No/Mild DR | Diffuse, low-intensity | Healthy vascular landmarks; no lesion focus |
| Moderate DR | Concentrated perifoveal/macular | Hard exudates, microaneurysms in temporal arcade |
| Severe/PDR | Multi-focal peripheral + disc margin | NVD, NVE, haemorrhages in peripheral retina |

> **Important**: Grad-CAM explains _where the model focused_ — it does **NOT confirm pathology**.
> Clinician verification is always required.

---

## 9. Installation

### Prerequisites

- Python 3.11
- ~4 GB RAM (CPU inference)
- Git + Git LFS (for model weights)

### Option A: Local Setup

```powershell
# 1. Clone the repository
git clone https://github.com/reemahussain-pharmd/RetinaIQ.git
cd RetinaIQ

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Data setup
# Place retinal fundus images in: data/raw/images/
# Place labels CSV (image, label) in: data/raw/labels.csv

# 5. Train + evaluate + generate XAI outputs (~3 min on CPU)
python scripts/quick_run.py

# 6. Launch web app
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`

### Option B: Docker

```bash
docker build -t retinaiq .
docker run -p 7860:7860 retinaiq
```

Open `http://localhost:7860`

### Option C: Hugging Face Spaces (No Installation)

Visit the live deployment at:
**https://huggingface.co/spaces/reemahussain/RetinoCare_AI**

No installation required — runs in your browser.

### Requirements Summary

```
tensorflow-cpu==2.15.0
numpy==1.26.4
streamlit==1.36.0
streamlit-option-menu==0.3.12
opencv-python-headless==4.8.1.78
scikit-learn==1.3.2
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.1
reportlab==4.1.0
pillow==10.2.0
```

### Project Structure

```
RetinaIQ/
├── app/
│   └── streamlit_app.py         # 8-page clinical decision support web app
├── src/
│   ├── preprocessing/           # CLAHE, augmentation, normalisation
│   │   └── image_preprocessor.py
│   ├── training/                # Model builders (MobileNetV2, CNN, etc.)
│   ├── evaluation/              # Metrics, confusion matrix, ROC, error analysis
│   ├── explainability/          # Grad-CAM, Saliency Maps
│   │   └── grad_cam.py
│   └── inference/
│       ├── predictor.py         # RetinopathyPredictor class + reliability score
│       └── report_generator.py  # PDF clinical report generator v2.1
├── scripts/
│   └── quick_run.py             # End-to-end pipeline (EDA → train → eval → XAI)
├── models/
│   └── best_model.h5            # Trained MobileNetV2 weights (Git LFS)
├── data/raw/
│   ├── images/                  # 1,764 retinal fundus images
│   └── labels.csv               # image filename + severity label (0/1/2)
├── outputs/
│   ├── eda/                     # EDA visualisations
│   ├── xai/                     # Grad-CAM + saliency per class
│   ├── mobilenetv2_training_curves.png
│   ├── mobilenetv2_confusion_matrix.png
│   ├── mobilenetv2_roc_curves.png
│   ├── error_analysis.png
│   └── model_comparison.csv
├── reports/                     # Generated PDF clinical reports
├── Dockerfile                   # HF Spaces Docker deployment
├── requirements.txt
└── README.md
```

---

## 10. Future Work

| Priority | Enhancement | Impact |
|----------|-------------|--------|
| High | **OCT Integration** — macular oedema depth assessment alongside fundus photography | Improves DMO detection sensitivity |
| High | **External Validation** — multi-centre prospective study (STARD 2015 criteria) | Required for clinical-grade evidence |
| Medium | **5-Class ICDR Grading** — full International Clinical DR Severity Scale | Increases clinical granularity |
| Medium | **Federated Learning** — privacy-preserving multi-site training | Enables NHS/hospital network deployment |
| Medium | **Longitudinal Monitoring** — disease progression tracking across visits | Supports treatment response assessment |
| Medium | **PubMed RAG** — real-time evidence retrieval per patient profile | Personalised guideline integration |
| Low | **Mobile TFLite Deployment** — point-of-care screening in low-resource settings | Global health equity impact |
| Low | **EHR Integration** — HL7 FHIR-compliant API for hospital systems | Clinical workflow integration |
| Low | **Multi-Modal AI** — fundus + OCT + EHR data fusion | Holistic diagnostic model |

---

## 11. Limitations & Disclaimer

### Known Limitations

| Limitation | Detail |
|-----------|--------|
| **Small Dataset** | 1,764 images — clinical-grade systems typically require 10,000+ images across diverse populations |
| **Single Dataset** | Trained and evaluated on one APTOS-style dataset — performance may vary on different cameras, ethnicities, or disease prevalences |
| **No External Validation** | Not tested on an independent external cohort (required for STARD 2015 / regulatory evidence) |
| **3-Class Grading** | The ICDR scale uses 5 grades — this system merges to 3 classes, reducing clinical granularity |
| **Single Timepoint** | No longitudinal tracking — cannot assess progression or treatment response |
| **Research Only** | Not CE marked, FDA cleared, or MHRA registered for clinical use |

### Clinical Disclaimer

> **IMPORTANT**: RetinaIQ is an **assistive diagnostic tool developed for research, educational, and portfolio purposes only**.
> It is **NOT a replacement** for professional ophthalmological evaluation.
> All AI predictions **must be reviewed and confirmed** by a qualified ophthalmologist before any clinical decision is made.
> The system does not store patient data. Clinical judgement must always take precedence.

---

## 12. License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## References

1. Selvaraju RR et al. _Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization._ ICCV 2017.
2. Lundberg SM & Lee SI. _A Unified Approach to Interpreting Model Predictions._ NeurIPS 2017.
3. Howard AG et al. _MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications._ arXiv 2017.
4. NICE Guideline NG28. _Diabetic Eye Screening._ 2016.
5. American Academy of Ophthalmology. _Diabetic Retinopathy Preferred Practice Pattern._ 2019.
6. International Diabetes Federation. _IDF Diabetes Atlas, 10th Edition._ 2021.
7. APTOS 2019 Blindness Detection Challenge. Kaggle.
8. American Diabetes Association. _Standards of Medical Care in Diabetes — 2023._

---

<div align="center">

**RetinaIQ v2.0.0** · Built with TensorFlow, Keras & Streamlit

_For Educational & Research Purposes Only · Not for Clinical Deployment_

Made by a PharmD professional with AI/ML expertise

</div>
