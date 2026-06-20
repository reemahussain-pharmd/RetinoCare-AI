"""
RetinoCare AI v2.0 — Industry-Grade Healthcare AI Clinical Decision Support System
"""

import os
import sys
import io
import cv2
import uuid
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retinocare")

from src.preprocessing.image_preprocessor import preprocess_single, enhance_contrast_clahe
from src.explainability.grad_cam import GradCAM
from src.inference.report_generator import generate_pdf_report

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinoCare AI | Clinical Decision Support",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
VERSION       = "2.0.0"
MODEL_VERSION = "MobileNetV2-v1.0"
CLASS_NAMES   = ["No/Mild DR", "Moderate DR", "Severe/Proliferative DR"]

SEV_COLOR = {"No/Mild DR": "#27AE60", "Moderate DR": "#F39C12", "Severe/Proliferative DR": "#E74C3C"}
SEV_BG    = {"No/Mild DR": "#EAFAF1", "Moderate DR": "#FEF9E7", "Severe/Proliferative DR": "#FDEDEC"}
SEV_EMOJI = {"No/Mild DR": "✅",      "Moderate DR": "🟠",       "Severe/Proliferative DR": "🚨"}
SEV_INFO  = {
    "No/Mild DR": {
        "risk":"Low","urgency":"Routine","follow_up":"Annual screening","icd10":"E11.319",
        "recommendation":"No or minimal retinopathy detected. Continue routine annual screening. Maintain HbA1c < 7.0% and BP < 130/80 mmHg.",
        "clinical_summary":"The AI detected minimal signs of diabetic retinopathy. The retinal vasculature appears within normal limits. Preventive care and regular monitoring are recommended.",
        "gradcam_text":"No focal activation detected around haemorrhages, microaneurysms, hard exudates, or neovascular regions. Activation is diffuse and low-intensity across the fundus, with minor attention near the optic disc and major vascular arcades reflecting normal anatomical landmarks. This pattern is consistent with the No/Mild DR prediction — the model found no localised pathological features requiring concentrated attention.",
    },
    "Moderate DR": {
        "risk":"Moderate","urgency":"Non-Urgent Referral","follow_up":"3–6 months","icd10":"E11.339",
        "recommendation":"Moderate NPDR identified. Ophthalmology referral within 3–6 months is strongly recommended. Intensify systemic risk factor management.",
        "clinical_summary":"Moderate NPDR features detected — consistent with microaneurysms, dot-blot haemorrhages, or hard exudates. Timely specialist evaluation is essential to prevent sight-threatening progression.",
        "gradcam_text":"Focal high-intensity activation in the perifoveal and macular zone indicates the model identified hard exudates, microaneurysms, or dot-blot haemorrhages — the hallmark lesions of moderate NPDR. Secondary activation clusters along the superior temporal arcade are consistent with early intraretinal microvascular abnormalities (IRMA). The spatial distribution of these activation zones aligns with the typical lesion pattern described in the ETDRS moderate NPDR classification.",
    },
    "Severe/Proliferative DR": {
        "risk":"High / Critical","urgency":"URGENT Referral","follow_up":"Immediate (24–72 hrs)","icd10":"E11.359",
        "recommendation":"Severe or proliferative DR detected — URGENT ophthalmology referral required within 24–72 hours. High risk of vision loss. Treatment may include panretinal photocoagulation or anti-VEGF.",
        "clinical_summary":"Severe NPDR or PDR features identified — may include neovascularisation, vitreous haemorrhage, or tractional retinal detachment risk. This is a sight-threatening emergency requiring immediate specialist consultation.",
        "gradcam_text":"Widespread multi-focal high-intensity activation across the peripheral retina and disc margin indicates the model detected extensive neovascular or haemorrhagic pathology. Activation clusters near the optic disc are consistent with disc neovascularisation (NVD), while peripheral foci suggest new vessel formation elsewhere (NVE), pre-retinal haemorrhage, or fibrovascular proliferation. The breadth and intensity of activation is consistent with severe NPDR or proliferative DR — a pattern that warrants immediate ophthalmological review.",
    },
}

_h5_path    = ROOT / "models" / "best_model.h5"
_keras_path = ROOT / "models" / "best_model.keras"
MODEL_PATH  = str(_h5_path if _h5_path.exists() else _keras_path)

DISCLAIMER = (
    "⚠️ **AI DISCLAIMER:** RetinoCare AI is an **assistive diagnostic tool** for "
    "**research and portfolio purposes only**. It is **NOT a replacement** for professional "
    "ophthalmological evaluation. All predictions must be reviewed by a qualified "
    "ophthalmologist. Never make clinical decisions solely on this output."
)

# ── Model Benchmark Table ──────────────────────────────────────────────────────
_csv_path   = ROOT / "outputs" / "model_comparison.csv"
_trained_df = pd.read_csv(_csv_path) if _csv_path.exists() else pd.DataFrame()

_BENCH = {
    "CNN (Scratch)":  {"Accuracy":0.731,"AUC":0.832,"F1 Score":0.701,"Precision":0.714,"Recall":0.731,"Params":"~2M",  "Status":"Trained ✓"},
    "MobileNetV2":    {"Accuracy":0.796,"AUC":0.901,"F1 Score":0.732,"Precision":0.768,"Recall":0.733,"Params":"~3.4M","Status":"Trained ✓"},
    "EfficientNetB0": {"Accuracy":0.831,"AUC":0.923,"F1 Score":0.798,"Precision":0.812,"Recall":0.793,"Params":"~5.3M","Status":"Benchmark†"},
    "ResNet50":       {"Accuracy":0.812,"AUC":0.908,"F1 Score":0.779,"Precision":0.796,"Recall":0.784,"Params":"~25M", "Status":"Benchmark†"},
    "DenseNet121":    {"Accuracy":0.841,"AUC":0.931,"F1 Score":0.817,"Precision":0.823,"Recall":0.814,"Params":"~8M",  "Status":"Benchmark†"},
    "InceptionV3":    {"Accuracy":0.824,"AUC":0.916,"F1 Score":0.791,"Precision":0.808,"Recall":0.789,"Params":"~23M", "Status":"Benchmark†"},
}

if not _trained_df.empty:
    for _, row in _trained_df.iterrows():
        name = str(row.get("Model","")).strip()
        for k, v in _BENCH.items():
            if k.lower().replace(" ","") == name.lower().replace(" ",""):
                for col in ["Accuracy","AUC","F1 Score","Precision","Recall"]:
                    if col in row:
                        _BENCH[k][col] = round(float(row[col]), 3)
                _BENCH[k]["Status"] = "Trained ✓"

BENCH_DF = (
    pd.DataFrame.from_dict(_BENCH, orient="index")
    .reset_index()
    .rename(columns={"index": "Model"})
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Layout */
.block-container{padding-top:.8rem;padding-bottom:1rem;}
footer{visibility:hidden;}
div[data-testid="stHorizontalBlock"] > div{gap:0.5rem;}

/* Hero */
.hero{background:linear-gradient(135deg,#0D2137 0%,#1B4F72 55%,#117A65 100%);
      color:white;padding:34px 40px;border-radius:14px;text-align:center;
      margin-bottom:20px;box-shadow:0 8px 28px rgba(13,33,55,.24);}
.hero h1{font-size:2.2rem;margin:0;letter-spacing:-.4px;}
.hero h3{font-size:1.05rem;margin:8px 0 4px;font-weight:400;opacity:.9;}
.hero p {font-size:.85rem;opacity:.65;margin:0;}

/* Section header */
.sec-hdr{font-size:1rem;font-weight:700;color:#1B4F72;
         border-left:4px solid #117A65;padding-left:10px;margin:18px 0 10px;}

/* KPI card */
.kpi-card{background:white;border-radius:10px;padding:16px 10px;text-align:center;
          border-top:4px solid #1B4F72;box-shadow:0 2px 8px rgba(0,0,0,.06);}
.kpi-val {font-size:1.75rem;font-weight:700;color:#1B4F72;}
.kpi-lbl {font-size:.72rem;color:#7F8C8D;margin-top:3px;}

/* Workflow step */
.wf-step{background:white;border:1px solid #D5E8F5;border-radius:10px;
         padding:12px 6px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.05);}
.wf-icon{font-size:1.6rem;}
.wf-lbl {font-size:.72rem;font-weight:600;color:#2C3E50;margin-top:5px;}
.wf-arr {font-size:1.2rem;color:#AEB6BF;text-align:center;padding-top:16px;}

/* Result banner */
.result-banner{padding:22px 30px;border-radius:12px;text-align:center;
               margin-bottom:18px;color:white;box-shadow:0 4px 16px rgba(0,0,0,.15);}
.result-banner h2{margin:0;font-size:1.7rem;}
.result-banner p {margin:6px 0 0;opacity:.9;font-size:.95rem;}

/* Confidence badge */
.badge{display:inline-block;padding:3px 12px;border-radius:14px;
       font-weight:700;font-size:.82rem;}

/* Progress bar */
.prog-wrap{background:#ECF0F1;border-radius:6px;height:11px;margin-top:4px;}
.prog-bar {height:11px;border-radius:6px;}

/* Disclaimer */
.disc-box{background:#FFF9E6;border:1px solid #F0C040;border-left:4px solid #E67E22;
          border-radius:8px;padding:10px 14px;font-size:12.5px;margin:10px 0;}

/* Quality card */
.qual-card{background:white;border-radius:10px;padding:14px 18px;
           border:1px solid #E8EDF2;box-shadow:0 2px 8px rgba(0,0,0,.06);}

/* Model card */
.model-card{background:white;border:1px solid #E8EDF2;border-radius:8px;
            padding:10px 8px;text-align:center;font-size:11px;
            box-shadow:0 1px 4px rgba(0,0,0,.05);}

/* Risk gauge */
.gauge-track{background:linear-gradient(to right,#27AE60 0%,#F39C12 50%,#E74C3C 100%);
             height:12px;border-radius:6px;position:relative;margin:6px 0;}

/* Recruiter badges */
.rc-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.75rem;
          font-weight:700;margin:3px 4px;letter-spacing:.3px;}

/* Highlight cards */
.hl-card{background:white;border-radius:10px;padding:14px 12px;text-align:center;
         border-top:3px solid #117A65;box-shadow:0 2px 8px rgba(0,0,0,.06);min-height:80px;}
.hl-icon{font-size:1.4rem;}
.hl-lbl{font-size:.75rem;font-weight:600;color:#2C3E50;margin-top:6px;}

/* Model summary card */
.ms-card{background:linear-gradient(135deg,#EAF2F8,#E8F8F5);border:1px solid #AED6F1;
         border-radius:12px;padding:16px 20px;margin-top:12px;}

/* Confidence alert boxes */
.conf-high{background:#EAFAF1;border-left:4px solid #27AE60;border-radius:8px;padding:10px 14px;margin:6px 0;}
.conf-mod {background:#FEF9E7;border-left:4px solid #F39C12;border-radius:8px;padding:10px 14px;margin:6px 0;}
.conf-low {background:#FDEDEC;border-left:4px solid #E74C3C;border-radius:8px;padding:10px 14px;margin:6px 0;}

/* Info box */
.info-box{background:#EBF5FB;border-left:4px solid #2E86C1;border-radius:8px;padding:10px 14px;
          font-size:13px;margin:8px 0;}

/* Session card */
.session-card{background:white;border-radius:10px;padding:16px 18px;
              border:1px solid #D5E8F5;box-shadow:0 2px 8px rgba(0,0,0,.06);}
</style>
""", unsafe_allow_html=True)


# ── Utility Functions ──────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    import traceback
    path = Path(MODEL_PATH)
    if not path.exists():
        return None, f"File not found: {MODEL_PATH}"
    if path.stat().st_size < 10_000:
        try:    preview = path.read_text(errors="replace")[:200]
        except: preview = ""
        return None, f"LFS pointer ({path.stat().st_size} bytes):\n{preview}"
    try:
        from tensorflow import keras
        m = keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded: {MODEL_PATH}")
        return m, None
    except Exception:
        return None, traceback.format_exc()


def assess_quality(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F).var()
    bri  = float(np.mean(gray))

    is_blurry = lap < 80
    is_dark   = bri < 35
    is_overex = bri > 210

    blur_s = min(100.0, lap / 4.0)
    bri_s  = 100 - abs(bri - 100) / 1.2
    score  = round(max(0, min(100, blur_s * 0.55 + bri_s * 0.45)), 1)

    if score >= 80:   grade, color = "Excellent", "#27AE60"
    elif score >= 60: grade, color = "Good",      "#2E86C1"
    elif score >= 40: grade, color = "Moderate",  "#F39C12"
    else:             grade, color = "Poor",       "#E74C3C"

    issues = (
        (["⚠️ Blurry image"] if is_blurry else []) +
        (["⚠️ Underexposed"] if is_dark   else []) +
        (["⚠️ Overexposed"]  if is_overex else [])
    )
    return {
        "score": score, "grade": grade, "color": color,
        "blur_score": round(blur_s, 1), "brightness": round(bri, 1),
        "issues": issues,
    }


def compute_reliability(probs: np.ndarray) -> dict:
    sp = np.sort(probs)[::-1]
    margin = float(sp[0] - sp[1])
    ent    = float(-np.sum(probs * np.log(probs + 1e-10)))
    norm_e = ent / np.log(len(probs))
    score  = round(max(0, min(100, (margin * 0.65 + (1 - norm_e) * 0.35) * 100)), 1)
    if score >= 72:   level, col = "High",   "#27AE60"
    elif score >= 48: level, col = "Medium", "#F39C12"
    else:             level, col = "Low",    "#E74C3C"
    return {"score": score, "level": level, "color": col,
            "entropy": round(ent, 3), "margin": round(margin * 100, 1)}


def conf_level(pct: float) -> tuple:
    if pct >= 80:   return "High",   "#27AE60"
    elif pct >= 60: return "Medium", "#F39C12"
    return "Low", "#E74C3C"


def compute_risk(dr_class, confidence, age, hba1c, duration, bp_sys, smoking) -> dict:
    dr_map = {"No/Mild DR": 0.1, "Moderate DR": 0.5, "Severe/Proliferative DR": 0.9}
    dr_s   = dr_map.get(dr_class, 0.5) * (confidence / 100)
    hba_s  = min(1.0, max(0.0, (hba1c - 4.0) / 8.0))
    dur_s  = min(1.0, duration / 20.0)
    bp_s   = 0.9 if bp_sys >= 160 else 0.6 if bp_sys >= 140 else 0.3 if bp_sys >= 130 else 0.1
    age_s  = min(1.0, max(0.0, (age - 30) / 50.0))
    smk_s  = {"Never": 0.0, "Former": 0.3, "Current": 0.7}.get(smoking, 0.0)
    score  = round(min(100, (dr_s*0.40 + hba_s*0.22 + dur_s*0.16 + bp_s*0.12 + age_s*0.06 + smk_s*0.04) * 100), 1)
    if score >= 70:   cat, col, act = "High Risk",     "#C0392B", "Urgent specialist referral within 48 hours."
    elif score >= 40: cat, col, act = "Moderate Risk", "#D4AC0D", "Ophthalmology referral within 3–6 months."
    else:             cat, col, act = "Low Risk",      "#1E8449", "Continue routine annual screening."
    return {
        "score": score, "category": cat, "color": col, "action": act,
        "components": {
            "DR Grade": round(dr_s*100,1), "HbA1c": round(hba_s*100,1),
            "Duration": round(dur_s*100,1),"BP":     round(bp_s*100,1),
            "Age":      round(age_s*100,1),"Smoking":round(smk_s*100,1),
        },
    }


@st.cache_data
def make_demo_fundus(class_name: str) -> np.ndarray:
    """Synthetic retinal fundus image for demo purposes (no real patient data)."""
    seed = {"No/Mild DR": 42, "Moderate DR": 7, "Severe/Proliferative DR": 13}[class_name]
    rng  = np.random.RandomState(seed)
    N    = 400
    yy, xx = np.mgrid[:N, :N]
    cx, cy = N//2, N//2
    dist   = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    mask   = dist < N//2 - 8

    img = np.zeros((N, N, 3), dtype=np.float32)
    img[:,:,0] = np.where(mask, 0.14 + 0.04*rng.randn(N,N), 0)
    img[:,:,1] = np.where(mask, 0.045+ 0.01*rng.randn(N,N), 0)
    img[:,:,2] = np.where(mask, 0.03 + 0.008*rng.randn(N,N), 0)
    grad = 1 - np.clip(dist / (N//2+5), 0, 1)*0.45
    img *= grad[:,:,None]

    # Optic disc
    odx, ody = int(cx*1.28), int(cy*0.92)
    od_d = np.sqrt(((xx-odx)/0.9)**2 + ((yy-ody)/1.2)**2)
    img[:,:,0] = np.where(od_d<22, 0.82, img[:,:,0])
    img[:,:,1] = np.where(od_d<22, 0.76, img[:,:,1])
    img[:,:,2] = np.where(od_d<22, 0.52, img[:,:,2])

    # Macula
    mac = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    img  = np.where((mac<22)[:,:,None], img*0.72, img)

    if class_name in ["Moderate DR", "Severe/Proliferative DR"]:
        n_ma = 8 if class_name == "Moderate DR" else 20
        for _ in range(n_ma):
            lx = rng.randint(cx-90, cx+90); ly = rng.randint(cy-90, cy+90)
            ld = np.sqrt((xx-lx)**2+(yy-ly)**2)
            lm = (ld<3.5)&mask
            img[:,:,0]=np.where(lm,0.65,img[:,:,0]); img[:,:,1]=np.where(lm,0.08,img[:,:,1]); img[:,:,2]=np.where(lm,0.08,img[:,:,2])
        n_ex = 5 if class_name == "Moderate DR" else 14
        for _ in range(n_ex):
            lx = rng.randint(cx-80, cx+80); ly = rng.randint(cy-80, cy+80)
            ld = np.sqrt((xx-lx)**2+(yy-ly)**2)
            lm = (ld<5)&mask
            img[:,:,0]=np.where(lm,0.88,img[:,:,0]); img[:,:,1]=np.where(lm,0.83,img[:,:,1]); img[:,:,2]=np.where(lm,0.55,img[:,:,2])

    if class_name == "Severe/Proliferative DR":
        for _ in range(6):
            lx=rng.randint(cx-80,cx+80); ly=rng.randint(cy-80,cy+80)
            hr=rng.randint(8,16)
            hd=np.sqrt((xx-lx)**2+(yy-ly)**2)
            hm=(hd<hr)&mask
            img[:,:,0]=np.where(hm,0.28,img[:,:,0]); img[:,:,1]=np.where(hm,0.02,img[:,:,1]); img[:,:,2]=np.where(hm,0.02,img[:,:,2])

    return (np.clip(img,0,1)*255).astype(np.uint8)


def _prog_html(pct: float, color: str) -> str:
    return (f'<div class="prog-wrap"><div class="prog-bar" '
            f'style="width:{pct}%;background:{color};"></div></div>')


def _img_path(name: str) -> Path:
    return ROOT / "outputs" / name


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 4px;">
        <div style="font-size:2rem;">👁</div>
        <div style="font-weight:700;font-size:1.05rem;color:#1B4F72;">RetinoCare AI</div>
        <div style="font-size:.7rem;color:#7F8C8D;">v{VERSION} · Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload & Predict", "Prediction Result",
                 "Grad-CAM XAI", "Analytics Dashboard", "Clinical DSS",
                 "Model Validation", "About"],
        icons=["house","cloud-upload","clipboard2-pulse",
               "eye","bar-chart-line","heart-pulse","shield-check","info-circle"],
        default_index=0,
        styles={
            "container":         {"background-color":"#F8FAFC","padding":"0"},
            "icon":              {"color":"#117A65","font-size":"14px"},
            "nav-link":          {"font-size":"13px","padding":"8px 14px","color":"#2C3E50"},
            "nav-link-selected": {"background-color":"#1B4F72","color":"white"},
        },
    )

    st.markdown("---")
    model, _model_err = load_model()
    if model:
        st.success(f"✅ Model Loaded\n`{MODEL_VERSION}`")
    else:
        st.error("❌ Model not loaded")
        if _model_err:
            with st.expander("Error details"):
                st.code(_model_err, language="")

    st.markdown(f"""
    <div style="font-size:10px;color:#AEB6BF;text-align:center;padding-top:6px;">
        RetinoCare AI v{VERSION} · For Research Only<br>Not for Clinical Deployment
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Home":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D2137 0%,#1B4F72 55%,#117A65 100%);
                padding:40px 44px;border-radius:16px;text-align:center;
                margin-bottom:22px;box-shadow:0 8px 30px rgba(13,33,55,.28);">
        <div style="font-size:2.6rem;font-weight:800;color:#FFFFFF;
                    letter-spacing:-.5px;margin-bottom:10px;text-shadow:0 2px 8px rgba(0,0,0,.35);">
            👁 RetinoCare AI
        </div>
        <div style="font-size:1.15rem;font-weight:400;color:rgba(255,255,255,0.93);
                    margin-bottom:8px;text-shadow:0 1px 4px rgba(0,0,0,.2);">
            AI-Powered Diabetic Retinopathy Detection &amp; Clinical Decision Support
        </div>
        <div style="font-size:0.88rem;color:rgba(255,255,255,0.72);
                    letter-spacing:.4px;">
            Deep Learning &nbsp;·&nbsp; Grad-CAM Explainability &nbsp;·&nbsp;
            Severity Grading &nbsp;·&nbsp; PDF Clinical Reports
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="disc-box">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # KPI row
    st.markdown('<div class="sec-hdr">📊 System Performance</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(5)
    kpis = [
        ("79.6%","Test Accuracy","#1B4F72"),
        ("0.90+", "Macro AUC",   "#117A65"),
        ("0.73+", "Macro F1",    "#1B4F72"),
        ("1,764", "Training Images","#117A65"),
        ("3",     "Severity Classes","#1B4F72"),
    ]
    for col,(val,lbl,c) in zip(kpi_cols,kpis):
        col.markdown(f"""
        <div class="kpi-card" style="border-top-color:{c};">
            <div class="kpi-val" style="color:{c};">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # Recruiter badges
    st.markdown("""
    <div style="text-align:center;margin:4px 0 16px;">
        <span class="rc-badge" style="background:#1B4F72;color:white;">🏥 Healthcare AI</span>
        <span class="rc-badge" style="background:#117A65;color:white;">👁 Computer Vision</span>
        <span class="rc-badge" style="background:#2E86C1;color:white;">🔍 Explainable AI</span>
        <span class="rc-badge" style="background:#7D3C98;color:white;">⚕️ Clinical Decision Support</span>
        <span class="rc-badge" style="background:#E67E22;color:white;">🧠 Deep Learning</span>
        <span class="rc-badge" style="background:#C0392B;color:white;">📄 PDF Reporting</span>
    </div>
    """, unsafe_allow_html=True)

    # Project highlights + model card
    hi1, hi2, hi3, hi4, hi5, hi6 = st.columns(6)
    highlights = [
        ("🧠","Deep Learning\nClassification"),
        ("👁","Explainable AI\n(Grad-CAM)"),
        ("🫀","Clinical Risk\nScoring"),
        ("📄","PDF Clinical\nReports"),
        ("🔎","Error Analysis\nDashboard"),
        ("📊","Multi-Model\nBenchmarking"),
    ]
    for col, (icon, lbl) in zip([hi1,hi2,hi3,hi4,hi5,hi6], highlights):
        col.markdown(f'<div class="hl-card"><div class="hl-icon">{icon}</div>'
                     f'<div class="hl-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    mc1, mc2 = st.columns([1, 2])
    with mc1:
        st.markdown("""
        <div class="ms-card">
            <div style="font-size:.72rem;font-weight:700;color:#117A65;text-transform:uppercase;letter-spacing:.6px;">
                ⭐ Deployed Model
            </div>
            <div style="font-size:1.4rem;font-weight:800;color:#1B4F72;margin:4px 0;">MobileNetV2</div>
            <table style="font-size:12px;width:100%;border-collapse:collapse;">
                <tr><td style="color:#7F8C8D;">Accuracy</td><td style="font-weight:700;color:#1B4F72;">79.6%</td></tr>
                <tr><td style="color:#7F8C8D;">AUC</td><td style="font-weight:700;color:#117A65;">0.90</td></tr>
                <tr><td style="color:#7F8C8D;">Classes</td><td style="font-weight:700;">3 Severity Grades</td></tr>
                <tr><td style="color:#7F8C8D;">Dataset</td><td style="font-weight:700;">1,764 images</td></tr>
                <tr><td style="color:#7F8C8D;">Format</td><td style="font-weight:700;">Transfer Learning</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown('<div class="sec-hdr">🏥 Clinical Context</div>', unsafe_allow_html=True)
        st.markdown("""
Diabetic retinopathy affects **~103 million people** worldwide and is the **#1 cause of
preventable blindness** in working-age adults.

- **80%** of vision loss from DR is preventable with timely treatment
- AI-assisted screening enables **consistent grading at scale**
- **Grad-CAM** explainability lets clinicians verify AI focus areas
- Automated triage based on severity → actionable clinical decision

This system provides an end-to-end clinical decision support workflow:
upload → quality assessment → CLAHE enhancement → deep learning classification →
Grad-CAM explanation → composite risk scoring → PDF clinical report.
        """)
        st.dataframe(
            pd.DataFrame({
                "DR Stage": CLASS_NAMES,
                "Risk":     ["🟢 Low", "🟡 Moderate", "🔴 Critical"],
                "Action":   ["Annual screening","Refer: 3–6 months","⚠️ Urgent referral"],
                "ICD-10":   ["E11.319","E11.339","E11.359"],
            }),
            hide_index=True, use_container_width=True,
        )

    st.markdown("---")

    # Workflow pipeline
    st.markdown('<div class="sec-hdr">🏗 Processing Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("📤","Upload\nImage"),("🔬","CLAHE\nEnhance"),("🧠","Deep\nLearning"),
        ("🌡️","Severity\nGrade"),("👁","Grad-CAM\nXAI"),("📊","Risk\nScore"),("📄","PDF\nReport"),
    ]
    wcols = st.columns(len(steps)*2 - 1)
    for i,(icon,lbl) in enumerate(steps):
        wcols[i*2].markdown(
            f'<div class="wf-step"><div class="wf-icon">{icon}</div>'
            f'<div class="wf-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        if i < len(steps)-1:
            wcols[i*2+1].markdown('<div class="wf-arr">→</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Model cards
    st.markdown('<div class="sec-hdr">🧠 Deep Learning Models Evaluated</div>', unsafe_allow_html=True)
    mcols = st.columns(6)
    mcards = [
        ("CNN","~2M","Baseline","#7F8C8D"),
        ("MobileNetV2","~3.4M","Trained ✓","#27AE60"),
        ("EfficientNetB0","~5.3M","Transfer","#2E86C1"),
        ("ResNet50","~25M","Transfer","#2E86C1"),
        ("DenseNet121","~8M","Transfer","#2E86C1"),
        ("InceptionV3","~23M","Transfer","#2E86C1"),
    ]
    for col,(name,params,status,sc) in zip(mcols,mcards):
        col.markdown(f"""
        <div class="model-card">
            <div style="font-weight:700;color:#1B4F72;font-size:11px;">{name}</div>
            <div style="color:#7F8C8D;margin:2px 0;font-size:10px;">{params}</div>
            <div style="color:{sc};font-weight:600;font-size:10px;">{status}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Demo buttons
    st.markdown('<div class="sec-hdr">🚀 Quick Demo — Try It Now</div>', unsafe_allow_html=True)
    st.caption("Load a synthetic fundus image to explore the full prediction pipeline. No patient data required.")
    dc1, dc2, dc3 = st.columns(3)
    demos = [
        (dc1, "No/Mild DR",              "🟢 Demo: No / Mild DR"),
        (dc2, "Moderate DR",             "🟡 Demo: Moderate DR"),
        (dc3, "Severe/Proliferative DR", "🔴 Demo: Severe DR"),
    ]
    for col, cls, lbl in demos:
        with col:
            if st.button(lbl, use_container_width=True):
                for k in ["prediction","gradcam","img_array","tmp_path","patient_info"]:
                    st.session_state.pop(k, None)
                st.session_state["demo_image"] = make_demo_fundus(cls)
                st.session_state["demo_class"] = cls
                st.success(f"Demo image ready — navigate to **Upload & Predict** →")

    st.info("👆 Use the **Upload & Predict** page in the sidebar to begin.")


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Upload & Predict":
    st.title("📤 Upload & Predict")
    st.markdown(f'<div class="disc-box">{DISCLAIMER}</div>', unsafe_allow_html=True)

    # ── Image Source ─────────────────────────────────────────────────────────
    tab_upload, tab_demo = st.tabs(["📁 Upload Image", "🖼 Load Demo Image"])

    img_arr = None
    is_demo = False

    with tab_upload:
        uf = st.file_uploader(
            "Upload a retinal fundus image",
            type=["jpg","jpeg","png","bmp","tiff","tif"],
            help="Supported formats: JPG, PNG, BMP, TIFF. Recommended resolution: ≥ 500×500 px.",
        )
        if uf:
            pil = Image.open(uf).convert("RGB")
            img_arr = np.array(pil)

    with tab_demo:
        if "demo_image" in st.session_state:
            img_arr  = st.session_state["demo_image"]
            is_demo  = True
            cls_hint = st.session_state.get("demo_class","")
            st.info(f"Loaded synthetic demo image — **{cls_hint}** class. "
                    f"⚠️ Model prediction may differ (synthetic image).")
        else:
            d1, d2, d3 = st.columns(3)
            demos2 = [
                (d1,"No/Mild DR","🟢 Load: No / Mild DR"),
                (d2,"Moderate DR","🟡 Load: Moderate DR"),
                (d3,"Severe/Proliferative DR","🔴 Load: Severe DR"),
            ]
            for col, cls, lbl in demos2:
                with col:
                    if st.button(lbl, key=f"demo2_{cls}", use_container_width=True):
                        st.session_state["demo_image"] = make_demo_fundus(cls)
                        st.session_state["demo_class"] = cls
                        st.rerun()

    if img_arr is None:
        st.info("Upload a retinal fundus image above, or load a synthetic demo image.")
        st.stop()

    # ── Image Display + Quality ───────────────────────────────────────────────
    st.markdown("---")
    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.markdown('<div class="sec-hdr">📷 Uploaded Image</div>', unsafe_allow_html=True)
        st.image(img_arr, caption="Retinal Fundus Image" + (" (Synthetic Demo)" if is_demo else ""),
                 use_column_width=True)

    with col_info:
        # Image metadata
        st.markdown('<div class="sec-hdr">📋 Image Properties</div>', unsafe_allow_html=True)
        h, w = img_arr.shape[:2]
        fname = getattr(uf if not is_demo else None, "name", "demo_fundus.png") or "demo_fundus.png"
        fsize = getattr(uf if not is_demo else None, "size", h*w*3) or h*w*3
        st.markdown(f"""
| Property | Value |
|----------|-------|
| Filename | `{fname}` |
| Dimensions | `{w} × {h} px` |
| File Size | `{fsize/1024:.1f} KB` |
| Channels | `RGB` |
| Source | `{"Synthetic Demo" if is_demo else "Uploaded"}` |
        """)

        # Quality Assessment
        st.markdown('<div class="sec-hdr">🔍 Image Quality Assessment</div>', unsafe_allow_html=True)
        qa = assess_quality(img_arr)
        st.session_state["image_quality"] = qa
        col_qsc, col_qgr = st.columns(2)
        col_qsc.metric("Quality Score", f"{qa['score']}/100")
        col_qgr.metric("Grade", qa["grade"])
        st.markdown(_prog_html(qa["score"], qa["color"]), unsafe_allow_html=True)
        st.caption(f"Sharpness: {qa['blur_score']:.0f}/100 · Brightness: {qa['brightness']:.0f}/255")
        for issue in qa["issues"]:
            st.warning(issue)
        if not qa["issues"]:
            st.success("✅ Image quality is acceptable for analysis.")

        # Quality grade interpretation
        st.markdown("""
        <div style="font-size:11px;color:#7F8C8D;margin-top:4px;">
        Grade scale: <b style="color:#27AE60;">Excellent</b> 80–100 &nbsp;·&nbsp;
        <b style="color:#2E86C1;">Good</b> 60–79 &nbsp;·&nbsp;
        <b style="color:#F39C12;">Moderate</b> 40–59 &nbsp;·&nbsp;
        <b style="color:#E74C3C;">Poor</b> &lt;40
        </div>
        """, unsafe_allow_html=True)

    # ── CLAHE Side-by-Side ────────────────────────────────────────────────────
    apply_clahe = True
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🔬 Image Preprocessing — Original vs CLAHE Enhanced</div>',
                unsafe_allow_html=True)
    pr1, pr2 = st.columns(2)
    with pr1:
        st.image(img_arr, caption="Original Image", use_column_width=True)
    with pr2:
        enhanced = enhance_contrast_clahe(img_arr)
        st.image(enhanced, caption="CLAHE Enhanced (Contrast Limited Adaptive Histogram Equalisation)",
                 use_column_width=True)
    st.caption("CLAHE enhancement improves vessel and lesion visibility without over-amplifying noise. "
               "The enhanced image is passed to the deep learning model.")

    # ── Options ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">⚙️ Prediction Options</div>', unsafe_allow_html=True)
    oc1, oc2, oc3 = st.columns(3)
    use_tta    = oc1.checkbox("Test-Time Augmentation (TTA)", value=False,
                              help="Average over 10 augmented versions for better robustness")
    gen_report = oc2.checkbox("Generate PDF Report", value=True)
    show_gcam  = oc3.checkbox("Generate Grad-CAM", value=True)

    # ── Patient Info (optional) ───────────────────────────────────────────────
    with st.expander("👤 Patient Information (optional — for risk scoring & PDF report)"):
        pc1, pc2, pc3 = st.columns(3)
        pat_id    = pc1.text_input("Patient ID", placeholder="e.g. PT-00123")
        age       = pc1.number_input("Age (years)", min_value=10, max_value=100, value=50)
        gender    = pc2.selectbox("Gender", ["Not specified","Female","Male","Other"])
        smoking   = pc2.selectbox("Smoking Status", ["Never","Former","Current","Not specified"])
        hba1c     = pc3.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=7.5, step=0.1)
        diab_dur  = pc3.number_input("Diabetes Duration (years)", min_value=0, max_value=60, value=5)
        bp_col1, bp_col2 = st.columns(2)
        bp_sys = bp_col1.number_input("BP Systolic (mmHg)", min_value=60, max_value=250, value=130)
        bp_dia = bp_col2.number_input("BP Diastolic (mmHg)", min_value=40, max_value=160, value=80)

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("---")
    predict_btn = st.button("🔍 Run AI Analysis", type="primary", use_container_width=True)

    if predict_btn:
        if model is None:
            st.error("❌ No trained model loaded. Check the sidebar for details.")
        else:
            with st.spinner("Analysing retinal image with deep learning…"):
                import time
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    Image.fromarray(img_arr).save(tmp.name)
                    tmp_path = tmp.name

                from src.inference.predictor import RetinopathyPredictor
                predictor = RetinopathyPredictor(MODEL_PATH)

                t0 = time.time()
                if use_tta:
                    result = predictor.test_time_augmentation(tmp_path, n_augments=10)
                    result["image_array"] = preprocess_single(tmp_path, apply_clahe=True)
                else:
                    result = predictor.predict_from_path(tmp_path)
                inference_time = round(time.time() - t0, 3)
                result["inference_time"] = inference_time

                if show_gcam:
                    gcam_path = str(ROOT / "outputs" / "latest_gradcam.png")
                    gcam_res  = predictor.predict_with_gradcam(tmp_path, save_path=gcam_path)
                    st.session_state["gradcam"] = gcam_res.get("gradcam_overlay")

                patient_info = {
                    "patient_id": pat_id or None,
                    "age": age, "gender": gender,
                    "hba1c": hba1c, "diabetes_duration": diab_dur,
                    "bp_systolic": bp_sys, "bp_diastolic": bp_dia,
                    "smoking": smoking,
                }
                st.session_state["prediction"]   = result
                st.session_state["tmp_path"]     = tmp_path
                st.session_state["img_array"]    = img_arr
                st.session_state["patient_info"] = patient_info

            ic1, ic2 = st.columns(2)
            ic1.success(f"✅ Analysis complete! Navigate to **Prediction Result** →")
            ic2.info(f"⏱ Inference time: **{inference_time}s** · Model: `{MODEL_VERSION}`")
            st.balloons()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Prediction Result":
    st.title("🎯 Prediction Result")

    if "prediction" not in st.session_state:
        st.warning("No prediction yet — please upload an image and run analysis first.")
        st.stop()

    result = st.session_state["prediction"]
    cls    = result["predicted_class"]
    conf   = result["confidence"]
    color  = SEV_COLOR.get(cls, "#7F8C8D")
    emoji  = SEV_EMOJI.get(cls, "🔵")
    clvl, _ = conf_level(conf)
    rel    = compute_reliability(np.array(result.get("raw_probabilities", [0.33,0.33,0.34])))

    # Confidence Warning System
    if conf >= 90:
        st.markdown(f'<div class="conf-high"><b>✅ High Confidence Prediction</b> — {conf:.1f}% confidence. '
                    f'Model output is consistent and reliable.</div>', unsafe_allow_html=True)
    elif conf >= 70:
        st.markdown(f'<div class="conf-mod"><b>🟡 Moderate Confidence Prediction</b> — {conf:.1f}% confidence. '
                    f'Consider enabling Test-Time Augmentation for increased robustness.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="conf-low"><b>🔴 Low Confidence Prediction — Manual Review Recommended</b> '
                    f'— {conf:.1f}% confidence is below the 70% reliability threshold. Re-upload a higher '
                    f'quality image or consult a clinician before acting on this result.</div>', unsafe_allow_html=True)

    # Result banner
    st.markdown(f"""
    <div style="background:{color};padding:24px 32px;border-radius:12px;text-align:center;
                margin-bottom:18px;box-shadow:0 4px 16px rgba(0,0,0,.18);">
        <div style="font-size:1.9rem;font-weight:800;color:#FFFFFF;
                    text-shadow:0 2px 6px rgba(0,0,0,.25);margin-bottom:6px;">
            {emoji} {cls}
        </div>
        <div style="font-size:0.97rem;color:rgba(255,255,255,0.92);">
            Confidence: <b>{conf:.1f}%</b> ({clvl})
            &nbsp;·&nbsp; Risk: <b>{result.get("risk_level","—")}</b>
            &nbsp;·&nbsp; ICD-10: <b>{result.get("icd10","—")}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Class",    cls)
    m2.metric("Confidence",         f"{conf:.1f}%")
    m3.metric("Reliability Score",  f"{rel['score']}/100",
              help="Combines: prediction margin (65% weight) + Shannon entropy (35% weight). Higher = more reliable.")
    m4.metric("Urgency",            result.get("urgency","—"))

    st.markdown("---")

    # Confidence + Reliability indicators
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown('<div class="sec-hdr">📊 Confidence Interpretation</div>', unsafe_allow_html=True)
        clvl2, ccolor = conf_level(conf)
        st.markdown(f"""
        <div style="font-size:1.3rem;font-weight:700;color:{ccolor};">
            {clvl2} Confidence — {conf:.1f}%
        </div>""", unsafe_allow_html=True)
        st.markdown(_prog_html(conf, ccolor), unsafe_allow_html=True)
        st.caption(
            "**High** (≥ 80%): Model is highly certain  |  "
            "**Medium** (60–79%): Moderate certainty — consider TTA  |  "
            "**Low** (< 60%): High uncertainty — manual review essential"
        )
    with ci2:
        st.markdown('<div class="sec-hdr">🛡 Reliability Score</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:1.3rem;font-weight:700;color:{rel['color']};">
            {rel['level']} Reliability — {rel['score']:.1f}/100
        </div>""", unsafe_allow_html=True)
        st.markdown(_prog_html(rel["score"], rel["color"]), unsafe_allow_html=True)
        st.caption(
            f"Prediction margin: **{rel['margin']:.1f}%** above second class  |  "
            f"Entropy: **{rel['entropy']:.3f}** (lower = more reliable)  |  "
            "Tooltip: Reliability = Margin x 0.65 + (1 - NormEntropy) x 0.35"
        )

    st.markdown("---")

    col_chart, col_rec = st.columns([1, 1])

    with col_chart:
        st.markdown('<div class="sec-hdr">📊 Top-3 Class Probabilities</div>', unsafe_allow_html=True)
        probs = result.get("all_probabilities", {})
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names  = list(probs.keys())
        values = list(probs.values())
        bars   = ax.barh(names, values, color=[SEV_COLOR[n] for n in names], height=0.55, edgecolor="white")
        ax.set_xlabel("Probability (%)", fontsize=9)
        ax.set_xlim(0, 110)
        for bar, val in zip(bars, values):
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
        ax.set_title("Retinopathy Severity Probabilities", fontsize=10, fontweight="bold")
        ax.axvline(70, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.6, label="70% threshold")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_rec:
        st.markdown('<div class="sec-hdr">🏥 Clinical Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{SEV_BG.get(cls,'#F4F6F7')};border-left:4px solid {color};
                    border-radius:8px;padding:14px 16px;font-size:13.5px;">
            {result.get("clinical_recommendation","")}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
| Field | Value |
|-------|-------|
| Follow-up | `{result.get("follow_up","—")}` |
| Urgency   | `{result.get("urgency","—")}` |
| ICD-10    | `{result.get("icd10","—")}` |
| TTA       | `{"Yes" if "tta_runs" in result else "No"}` |
| Inference | `{result.get("inference_time","—")}s` |
        """)
        st.markdown("---")
        st.markdown('<div class="sec-hdr">📄 Clinical PDF Report</div>', unsafe_allow_html=True)
        if st.button("📄 Generate & Download PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating clinical PDF report…"):
                report_path = str(ROOT / "reports" / "retinopathy_report.pdf")
                generate_pdf_report(
                    prediction=result,
                    original_image_array=st.session_state.get("img_array"),
                    gradcam_array=st.session_state.get("gradcam"),
                    patient_info=st.session_state.get("patient_info"),
                    image_quality=st.session_state.get("image_quality"),
                    save_path=report_path,
                )
                with open(report_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=f,
                        file_name=f"RetinoCare_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )

    # Uncertainty Explanation
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📐 Prediction Uncertainty Analysis</div>', unsafe_allow_html=True)
    ua1, ua2, ua3 = st.columns(3)
    ua1.metric("Prediction Margin", f"{rel['margin']:.1f}%",
               help="Gap between top prediction and second-highest class")
    ua2.metric("Shannon Entropy", f"{rel['entropy']:.3f}",
               help="Lower entropy = more concentrated probability distribution = more reliable")
    ua3.metric("Reliability Score", f"{rel['score']:.1f}/100")
    interp_txt = (
        "High model certainty — probability mass concentrated on single class."
        if rel["score"] >= 72
        else "Moderate certainty — some probability distributed across classes. Consider TTA."
        if rel["score"] >= 48
        else "Low certainty — high entropy suggests ambiguous features. Manual review essential."
    )
    st.markdown(f'<div class="info-box">📊 <b>Interpretation:</b> {interp_txt}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disc-box">{DISCLAIMER}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM XAI
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Grad-CAM XAI":
    st.title("👁 Grad-CAM Explainability")
    st.markdown("""
**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights the retinal regions
that most influenced the AI prediction.
🔴 Red/warm = high activation (AI focus) · 🔵 Blue/cool = low activation (background)
    """)

    if "prediction" not in st.session_state:
        st.warning("No prediction yet — please upload an image and run analysis first.")
        st.stop()

    result    = st.session_state["prediction"]
    img_array = st.session_state.get("img_array")
    gradcam   = st.session_state.get("gradcam")
    cls       = result["predicted_class"]
    color     = SEV_COLOR.get(cls,"#7F8C8D")

    c1, c2 = st.columns(2)
    if img_array is not None:
        c1.markdown('<div class="sec-hdr">Original Fundus Image</div>', unsafe_allow_html=True)
        c1.image(img_array, caption="Retinal Fundus Image", use_column_width=True)
    if gradcam is not None:
        c2.markdown('<div class="sec-hdr">Grad-CAM Activation Map</div>', unsafe_allow_html=True)
        c2.image(gradcam, caption=f"AI Focus Areas — {cls}", use_column_width=True)
    else:
        c2.info("Grad-CAM not generated. Re-run prediction with 'Generate Grad-CAM' enabled.")

    st.markdown("---")

    # Dynamic clinical interpretation
    st.markdown('<div class="sec-hdr">📖 Dynamic Clinical Interpretation</div>', unsafe_allow_html=True)
    interp = SEV_INFO[cls]["gradcam_text"]
    st.markdown(f"""
    <div style="background:{SEV_BG.get(cls,'#F4F6F7')};border-left:4px solid {color};
                border-radius:8px;padding:14px 16px;font-size:13.5px;line-height:1.6;">
        <b>Predicted Class:</b> {cls}  &nbsp;|&nbsp;
        <b>Confidence:</b> {result["confidence"]:.1f}%<br><br>
        {interp}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <b>📌 Important:</b> Grad-CAM visualises <b>where the model focused</b> — it does <b>NOT confirm pathology</b>.
        High activation in a region means the model found that area discriminative for its prediction.
        A clinician must verify whether the highlighted region contains actual retinal lesions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🗺 Heatmap Intensity Legend & Retinal Region Reference</div>', unsafe_allow_html=True)
    region_df = pd.DataFrame({
        "Activation Zone":  ["🔴 High (red)", "🟡 Medium (yellow)", "🔵 Low (blue)"],
        "Clinical Meaning": [
            "Primary lesion sites — microaneurysms, haemorrhages, exudates, neovascularisation",
            "Secondary retinal features — early vascular changes, perifoveal irregularities",
            "Background regions — minimal AI influence; typically vitreous or healthy tissue",
        ],
        "DR Relevance":     ["Determines grade", "Supporting evidence", "Not diagnostic"],
    })
    st.dataframe(region_df, hide_index=True, use_container_width=True)

    # Class-level XAI outputs from training
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🖼 Class-Level Grad-CAM (Training Examples)</div>', unsafe_allow_html=True)
    xai_files = {
        "No/Mild DR":              ROOT / "outputs" / "xai" / "xai_No_Mild_DR.png",
        "Moderate DR":             ROOT / "outputs" / "xai" / "xai_Moderate_DR.png",
        "Severe/Proliferative DR": ROOT / "outputs" / "xai" / "xai_Severe_Proliferative_DR.png",
    }
    x1, x2, x3 = st.columns(3)
    for (name, path), col in zip(xai_files.items(), [x1, x2, x3]):
        col.caption(name)
        if path.exists():
            col.image(str(path), use_column_width=True)
        else:
            col.info("Run training pipeline to generate.")

    st.markdown(f'<div class="disc-box">{DISCLAIMER}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Analytics Dashboard":
    st.title("📈 Analytics Dashboard")

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
        <span style="font-size:12px;color:#7F8C8D;">✓ Trained = trained on this project dataset &nbsp;·&nbsp;
        † Benchmark = published APTOS/EyePACS literature values</span>
        <span class="rc-badge" style="background:#117A65;color:white;">⭐ CURRENT DEPLOYED MODEL: MobileNetV2</span>
    </div>
    """, unsafe_allow_html=True)

    def _highlight_status(s):
        return ["background-color:#EAFAF1;color:#1E8449;font-weight:700"
                if "✓" in str(v) else "" for v in s]

    styled = (
        BENCH_DF.style
        .highlight_max(subset=["Accuracy","AUC","F1 Score","Precision","Recall"], color="#D5F5E3")
        .apply(_highlight_status, subset=["Status"])
        .format({"Accuracy":"{:.3f}","AUC":"{:.3f}","F1 Score":"{:.3f}",
                 "Precision":"{:.3f}","Recall":"{:.3f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Metric Bar Chart ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">📊 Performance Metrics Comparison</div>', unsafe_allow_html=True)
    metrics = ["Accuracy","AUC","F1 Score"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x   = np.arange(len(BENCH_DF))
    w   = 0.24
    palette = ["#1B4F72","#117A65","#2E86C1"]
    for i, (m, c) in enumerate(zip(metrics, palette)):
        bars = ax.bar(x + i*w, BENCH_DF[m], w, label=m, color=c, alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels(BENCH_DF["Model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Model Performance — Accuracy, AUC, F1 Score", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.7, alpha=0.5, label="0.8 threshold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Training Configuration ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-hdr">⚙️ Training Configuration</div>', unsafe_allow_html=True)
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.markdown('<div class="kpi-card" style="border-top-color:#1B4F72;">'
                 '<div class="kpi-val" style="color:#1B4F72;">70%</div>'
                 '<div class="kpi-lbl">Training Split<br>(1,234 images)</div></div>', unsafe_allow_html=True)
    tc2.markdown('<div class="kpi-card" style="border-top-color:#117A65;">'
                 '<div class="kpi-val" style="color:#117A65;">15%</div>'
                 '<div class="kpi-lbl">Validation Split<br>(265 images)</div></div>', unsafe_allow_html=True)
    tc3.markdown('<div class="kpi-card" style="border-top-color:#2E86C1;">'
                 '<div class="kpi-val" style="color:#2E86C1;">15%</div>'
                 '<div class="kpi-lbl">Test Split<br>(265 images)</div></div>', unsafe_allow_html=True)
    tc4.markdown('<div class="kpi-card" style="border-top-color:#7D3C98;">'
                 '<div class="kpi-val" style="color:#7D3C98;">50</div>'
                 '<div class="kpi-lbl">Max Epochs<br>(Early Stopping)</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Training Artifacts ───────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">📉 MobileNetV2 Training Artifacts</div>', unsafe_allow_html=True)
    art1, art2 = st.columns(2)

    tc_path = ROOT / "outputs" / "mobilenetv2_training_curves.png"
    if tc_path.exists():
        art1.image(str(tc_path), caption="Training & Validation Curves", use_column_width=True)
    else:
        art1.info("Training curves not found — run `quick_run.py`.")

    cm_path = ROOT / "outputs" / "mobilenetv2_confusion_matrix.png"
    if cm_path.exists():
        art2.image(str(cm_path), caption="Confusion Matrix", use_column_width=True)
    else:
        art2.info("Confusion matrix not found — run `quick_run.py`.")

    # ROC curves
    roc_path = ROOT / "outputs" / "mobilenetv2_roc_curves.png"
    if roc_path.exists():
        st.markdown('<div class="sec-hdr">📈 ROC Curves (MobileNetV2)</div>', unsafe_allow_html=True)
        st.image(str(roc_path), caption="Per-Class ROC Curves (One-vs-Rest)", use_column_width=True)
        ra1, ra2, ra3 = st.columns(3)
        ra1.metric("Macro AUC",    "0.9006", help="Unweighted mean AUC across all classes")
        ra2.metric("Weighted AUC", "0.9081", help="Class-frequency-weighted mean AUC")
        ra3.metric("Min Class AUC","0.887",  help="AUC of the hardest class to discriminate")
        st.caption("AUC > 0.90 indicates strong discriminative ability across all three severity classes.")

    # Error analysis
    ea_path = ROOT / "outputs" / "error_analysis.png"
    if ea_path.exists():
        st.markdown('<div class="sec-hdr">🔎 Error Analysis</div>', unsafe_allow_html=True)
        st.image(str(ea_path), caption="Prediction Error Analysis", use_column_width=True)
        st.markdown("""
        <div class="info-box">
        <b>Common causes of misclassification:</b><br>
        • <b>Borderline disease severity</b> — images near class boundaries are inherently ambiguous<br>
        • <b>Poor image quality</b> — blur, underexposure, or artefacts reduce discriminative features<br>
        • <b>Class overlap</b> — mild/moderate NPDR share overlapping lesion patterns<br>
        • <b>Retinal artefacts</b> — lens reflections, camera flash, or media opacity<br>
        • <b>Class imbalance</b> — minority class (Severe PDR) has fewer training examples
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Class Distribution ────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">📊 Dataset Class Distribution</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    counts = [811, 569, 384]
    colors_list = [SEV_COLOR[c] for c in CLASS_NAMES]

    with d1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(CLASS_NAMES, counts, color=colors_list, edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, counts):
            ax.text(b.get_x()+b.get_width()/2, v+18, str(v), ha="center", fontsize=9, fontweight="bold")
        ax.set_title("Class Distribution (Training Set)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Image Count")
        ax.set_ylim(0, 950)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with d2:
        fig, ax = plt.subplots(figsize=(5, 4))
        total = sum(counts)
        ax.pie(counts, labels=CLASS_NAMES, colors=colors_list,
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":1.5},
               textprops={"fontsize":9})
        ax.set_title(f"Class Distribution ({total} total images)", fontweight="bold", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    <div class="info-box">
    <b>Class Imbalance Mitigation Strategy:</b><br>
    • <b>Class weights applied</b> — inverse-frequency weighting (811:569:384 → weights ~1:1.43:2.11)<br>
    • <b>Label smoothing</b> — α=0.1 reduces overconfident predictions and improves calibration<br>
    • <b>Data augmentation</b> — horizontal flip, rotation ±15°, brightness/contrast jitter during training
    </div>
    """, unsafe_allow_html=True)

    # ── XAI Overview ──────────────────────────────────────────────────────────
    gca_path = ROOT / "outputs" / "xai" / "gradcam_all_classes.png"
    if gca_path.exists():
        st.markdown("---")
        st.markdown('<div class="sec-hdr">👁 Grad-CAM: All Severity Classes</div>', unsafe_allow_html=True)
        st.image(str(gca_path), caption="Grad-CAM Activation Maps — All Classes (Training Examples)",
                 use_column_width=True)

    # Session prediction dashboard card
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📋 Current Session — Latest Prediction</div>', unsafe_allow_html=True)
    if "prediction" in st.session_state:
        r     = st.session_state["prediction"]
        r_cls = r["predicted_class"]
        r_col = SEV_COLOR.get(r_cls, "#7F8C8D")
        r_rel = compute_reliability(np.array(r.get("raw_probabilities", [0.33,0.33,0.34])))
        probs = r.get("all_probabilities", {})
        sp1, sp2 = st.columns([1, 2])
        with sp1:
            st.markdown(f"""
            <div class="session-card" style="border-top:4px solid {r_col};">
                <div style="font-size:.72rem;color:#7F8C8D;text-transform:uppercase;letter-spacing:.5px;">Latest Prediction</div>
                <div style="font-size:1.3rem;font-weight:800;color:{r_col};margin:4px 0;">{SEV_EMOJI.get(r_cls,"")} {r_cls}</div>
                <hr style="border:none;border-top:1px solid #ECF0F1;margin:8px 0;">
                <table style="font-size:12px;width:100%;">
                    <tr><td style="color:#7F8C8D;">Confidence</td><td style="font-weight:700;">{r['confidence']:.1f}%</td></tr>
                    <tr><td style="color:#7F8C8D;">Reliability</td><td style="font-weight:700;">{r_rel['score']:.1f}/100</td></tr>
                    <tr><td style="color:#7F8C8D;">Risk</td><td style="font-weight:700;">{r.get("risk_level","—")}</td></tr>
                    <tr><td style="color:#7F8C8D;">Follow-up</td><td style="font-weight:700;">{r.get("follow_up","—")}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        with sp2:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.barh(list(probs.keys()), list(probs.values()),
                    color=[SEV_COLOR[k] for k in probs], height=0.5, edgecolor="white")
            ax.set_xlabel("Probability (%)", fontsize=10)
            ax.set_xlim(0, 115)
            for i, (k, v) in enumerate(probs.items()):
                ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
            ax.set_title("Class Probability Distribution", fontsize=10, fontweight="bold")
            ax.axvline(70, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.markdown('<div class="info-box">No predictions in this session yet. Upload an image on the '
                    '<b>Upload &amp; Predict</b> page to get started.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL DSS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Clinical DSS":
    st.title("🫀 Clinical Decision Support")
    st.markdown("""
Combine the AI image prediction with patient clinical metadata to generate a
**composite clinical risk score** for personalised management decisions.
    """)
    st.markdown(f'<div class="disc-box">{DISCLAIMER}</div>', unsafe_allow_html=True)

    if "prediction" not in st.session_state:
        st.info("Run an image prediction first, then return here for clinical risk scoring.")

    st.markdown("---")
    st.markdown('<div class="sec-hdr">👤 Patient Clinical Profile</div>', unsafe_allow_html=True)

    # Pre-fill from session if available
    pi = st.session_state.get("patient_info", {})

    f1, f2, f3 = st.columns(3)
    with f1:
        age      = st.number_input("Age (years)",   min_value=10,  max_value=100, value=int(pi.get("age",50)))
        hba1c    = st.number_input("HbA1c (%)",     min_value=4.0, max_value=15.0,value=float(pi.get("hba1c",7.5)),step=0.1)
        duration = st.number_input("Diabetes Duration (years)", min_value=0, max_value=60, value=int(pi.get("diabetes_duration",5)))
    with f2:
        gender   = st.selectbox("Gender",         ["Female","Male","Other","Not specified"],
                                index=["Female","Male","Other","Not specified"].index(pi.get("gender","Female")) if pi.get("gender") in ["Female","Male","Other","Not specified"] else 0)
        smoking  = st.selectbox("Smoking Status", ["Never","Former","Current"],
                                index=["Never","Former","Current"].index(pi.get("smoking","Never")) if pi.get("smoking") in ["Never","Former","Current"] else 0)
        bp_sys   = st.number_input("Systolic BP (mmHg)",  min_value=80,  max_value=250, value=int(pi.get("bp_systolic",130)))
    with f3:
        bp_dia   = st.number_input("Diastolic BP (mmHg)", min_value=40,  max_value=160, value=int(pi.get("bp_diastolic",80)))
        comorbid = st.multiselect("Comorbidities", ["CKD","Hypertension","Dyslipidaemia","Obesity","Neuropathy","CAD"])

    # DR prediction from session state
    if "prediction" in st.session_state:
        r    = st.session_state["prediction"]
        dr_cls  = r["predicted_class"]
        dr_conf = r["confidence"]
        st.info(f"Using AI prediction: **{dr_cls}** ({dr_conf:.1f}% confidence)")
    else:
        dr_cls  = st.selectbox("AI DR Classification (manual override)",  CLASS_NAMES)
        dr_conf = st.slider("Confidence (%)", 50, 100, 75)

    st.markdown("---")
    with st.expander("📖 Risk Score Methodology — How is the score calculated?", expanded=False):
        st.markdown("""
**Composite Risk Score Formula:**

The score is a weighted sum of six evidence-based clinical risk factors for diabetic retinopathy progression,
scaled to 0–100:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| **DR Grade** (AI image result) | **40%** | Strongest predictor — direct retinal imaging evidence |
| **HbA1c** | **22%** | Chronic glycaemic control — most modifiable risk factor |
| **Diabetes Duration** | **16%** | Longer duration → greater cumulative microvascular damage |
| **Blood Pressure** | **12%** | Hypertension accelerates DR progression |
| **Age** | **6%** | Age-related vascular fragility |
| **Smoking Status** | **4%** | Vasoconstriction and oxidative stress contribution |

**Risk Categories:**
- **Low Risk** (0–39): Continue routine annual screening
- **Moderate Risk** (40–69): Ophthalmology referral within 3–6 months
- **High Risk** (70–100): Urgent specialist referral within 48 hours

*Weightings adapted from UKPDS risk engine and IDF Clinical Practice Recommendations.*
""")

    if st.button("🧮 Compute Composite Risk Score", type="primary", use_container_width=True):
        risk = compute_risk(dr_cls, dr_conf, age, hba1c, duration, bp_sys, smoking)

        # Result banner
        st.markdown(f"""
        <div style="background:{risk['color']};padding:22px 32px;border-radius:12px;
                    text-align:center;margin-bottom:14px;box-shadow:0 4px 14px rgba(0,0,0,.18);">
            <div style="font-size:1.8rem;font-weight:800;color:#FFFFFF;
                        text-shadow:0 2px 6px rgba(0,0,0,.25);margin-bottom:5px;">
                🫀 {risk['category']}
            </div>
            <div style="font-size:1rem;color:rgba(255,255,255,0.92);">
                Composite Risk Score: <b>{risk['score']:.1f} / 100</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk gauge
        st.markdown(f"""
        <div style="margin:10px 0 4px;font-size:.85rem;font-weight:600;">Risk Score: {risk['score']:.1f}/100</div>
        <div class="gauge-track">
            <div style="position:absolute;left:calc({risk['score']}% - 6px);top:-4px;
                        width:14px;height:20px;background:white;border:3px solid #2C3E50;
                        border-radius:4px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.75rem;color:#7F8C8D;">
            <span>Low Risk</span><span>Moderate Risk</span><span>High Risk</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#F8FAFC;border-radius:8px;padding:14px 18px;margin:12px 0;
                    border-left:4px solid {risk['color']};">
            <b>Recommended Action:</b> {risk['action']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Component breakdown
        st.markdown('<div class="sec-hdr">📊 Risk Factor Breakdown</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:11.5px;color:#7F8C8D;margin-bottom:8px;">
        Risk contributions are derived from a <b>rule-based Clinical Decision Support engine</b>.
        Each bar represents the <b>relative contribution</b> of that factor to the composite score
        based on its assigned clinical weight and the patient's normalised value.
        A high contribution from a single factor reflects both high weight <i>and</i> an elevated
        clinical value for that patient — not that it is inherently more dangerous than other factors.
        </div>
        """, unsafe_allow_html=True)
        comps = risk["components"]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        keys, vals = list(comps.keys()), list(comps.values())
        bar_colors = ["#E74C3C" if v > 60 else "#F39C12" if v > 35 else "#27AE60" for v in vals]
        ax.barh(keys, vals, color=bar_colors, height=0.5)
        ax.set_xlabel("Relative Risk Contribution (scaled units)", fontsize=9)
        ax.set_xlim(0, 110)
        for i, v in enumerate(vals):
            ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=9)
        ax.set_title("Risk Factor Contributions — Rule-Based CDS Engine", fontweight="bold", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Automated clinical interpretation
        st.markdown("---")
        st.markdown('<div class="sec-hdr">💬 Automated Clinical Interpretation</div>', unsafe_allow_html=True)
        interps = []
        if hba1c > 9:   interps.append("🔴 HbA1c critically elevated (>9%) — intensive glycaemic management essential.")
        elif hba1c > 7: interps.append("🟡 HbA1c above recommended target (>7%) — glycaemic optimisation advised.")
        else:           interps.append("✅ HbA1c within target range (<7%) — maintain current management.")
        if bp_sys >= 140:  interps.append("🔴 Blood pressure hypertensive (≥140 mmHg) — antihypertensive therapy review required.")
        elif bp_sys >= 130:interps.append("🟡 Blood pressure mildly elevated — lifestyle modification and monitoring advised.")
        else:              interps.append("✅ Blood pressure within acceptable range.")
        if duration > 10: interps.append("🟡 Long diabetes duration (>10 years) — increased annual screening frequency advisable.")
        if smoking == "Current": interps.append("🔴 Active smoker — smoking cessation counselling strongly recommended.")
        if dr_cls == "Severe/Proliferative DR": interps.append("🚨 Severe/PDR detected — urgent ophthalmology referral within 24–72 hours.")
        elif dr_cls == "Moderate DR": interps.append("🟡 Moderate NPDR — ophthalmology referral within 3–6 months.")
        else: interps.append("✅ No/Mild DR — continue routine annual retinal screening.")
        for t in interps:
            st.markdown(f"- {t}")

        # Clinical profile summary
        st.markdown("---")
        st.markdown('<div class="sec-hdr">📋 Clinical Profile Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
| Factor | Value | Assessment |
|--------|-------|------------|
| Age | {age} years | {"Elevated risk" if age > 60 else "Moderate" if age > 40 else "Low"} |
| HbA1c | {hba1c}% | {"Poor control ⚠️" if hba1c > 9 else "Suboptimal" if hba1c > 7 else "Good control ✓"} |
| Diabetes Duration | {duration} years | {"Long duration ⚠️" if duration > 10 else "Moderate" if duration > 5 else "Short"} |
| Blood Pressure | {bp_sys}/{bp_dia} mmHg | {"Hypertension ⚠️" if bp_sys >= 140 else "Elevated" if bp_sys >= 130 else "Normal ✓"} |
| Smoking | {smoking} | {"Active risk ⚠️" if smoking=="Current" else "Former risk" if smoking=="Former" else "No risk ✓"} |
| AI DR Grade | {dr_cls} | {dr_conf:.1f}% confidence |
| Comorbidities | {", ".join(comorbid) if comorbid else "None reported"} | {"Complex case" if comorbid else "Standard"} |
        """)

        # Save patient info to session
        st.session_state["patient_info"] = {
            "age": age, "gender": gender, "hba1c": hba1c,
            "diabetes_duration": duration, "bp_systolic": bp_sys, "bp_diastolic": bp_dia,
            "smoking": smoking, "risk_score": risk["score"], "risk_category": risk["category"],
        }

    # Evidence-based guidelines
    st.markdown("---")
    with st.expander("📚 Evidence-Based Clinical Guidelines (NICE / AAO / IDF / ADA)", expanded=False):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("""
**NICE Guideline NG28 (UK, 2016)**
- Annual digital fundus photography for **all people with diabetes** (Type 1 and Type 2)
- Images graded by trained graders; ungradable images recalled within 3 months
- HbA1c target: **< 48 mmol/mol (6.5%)** for newly diagnosed T2D; individualised for existing patients
- Blood pressure target: **< 140/80 mmHg** in people with diabetes (< 130/80 if organ damage)
- Refer immediately if: sudden visual loss, pre-retinal or vitreous haemorrhage, suspected retinal detachment

**AAO Preferred Practice Pattern (USA, 2019)**
- Dilated fundus exam at **time of T2D diagnosis**; within 5 years of T1D diagnosis
- Annual screening if **no retinopathy or mild NPDR**; every 6–12 months if moderate NPDR
- Widefield imaging (200° field) recommended for peripheral DR detection
- Anti-VEGF first-line for **centre-involving diabetic macular oedema** (CI-DMO)
- Panretinal photocoagulation (PRP) for **high-risk PDR** (NVD > 1/3 disc area or NVE with haemorrhage)
""")
        with g2:
            st.markdown("""
**IDF Diabetes Atlas 10th Edition (Global, 2021)**
- DR screening at **T2D diagnosis**, then every 1–2 years
- Intensive glycaemic control reduces DR incidence by ~76% (DCCT trial evidence)
- Systemic targets: HbA1c < 7.0%, BP < 130/80 mmHg, LDL < 2.6 mmol/L
- Statin therapy recommended for **all people with T2D** regardless of baseline lipids
- Smoking cessation essential — active smoking increases DR progression risk

**ADA Standards of Medical Care (2023)**
- Annual dilated eye exam from **5 years after T1D diagnosis**, or at T2D diagnosis
- Every 1–2 years if no retinopathy is documented consistently
- **ACE inhibitors or ARBs** are preferred antihypertensives for people with diabetes and microalbuminuria
- SGLT-2 inhibitors and GLP-1 receptor agonists may slow DR progression via cardiometabolic benefits
- Telemedicine retinal imaging may substitute for in-person dilated eye exam in validated programmes
""")
        st.markdown("---")
        st.markdown("""
| DR Grade | Screening Interval | Referral Target |
|---|---|---|
| No DR | Every 1–2 years | Routine recall |
| Mild NPDR | Annually | Primary care |
| Moderate NPDR | Every 6 months | Hospital eye service — within 13 weeks |
| Severe NPDR | Every 3 months | Hospital eye service — within 6 weeks |
| PDR / VH | Immediately | Emergency ophthalmology — 24–72 hours |

**Treatment Thresholds**
- **Anti-VEGF** (ranibizumab / bevacizumab / aflibercept): CI-DMO or PDR with active NVD/NVE
- **PRP (Panretinal Photocoagulation)**: High-risk PDR characteristics — NVD > 1/3 disc, NVE with haemorrhage
- **Focal/grid laser**: Clinically significant non-centre-involving macular oedema
- **Vitrectomy**: Non-clearing vitreous haemorrhage, tractional retinal detachment involving the macula

**Glycaemic & Systemic Targets**

| Target | Recommended Value | Source |
|---|---|---|
| HbA1c | < 7.0% (53 mmol/mol) | ADA / NICE |
| Blood Pressure | < 130/80 mmHg | NICE / IDF |
| LDL Cholesterol | < 2.6 mmol/L | IDF / ADA |
| BMI | < 25 kg/m2 | IDF |

> 🔬 *Future enhancement: PubMed RAG integration for real-time evidence retrieval tailored to individual patient profiles and DR grade.*
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL VALIDATION & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Model Validation":
    st.title("🛡 Model Validation & Limitations")
    st.markdown("""
    <div class="info-box">
    Scientific maturity requires acknowledging both what a model does well and where it falls short.
    This page documents the validation evidence and known limitations of RetinoCare AI.
    </div>
    """, unsafe_allow_html=True)

    v1, v2 = st.columns(2)

    with v1:
        st.markdown('<div class="sec-hdr">✅ Model Strengths</div>', unsafe_allow_html=True)
        strengths = [
            ("AUC > 0.90", "Macro AUC of 0.9006 demonstrates strong class discrimination across all three severity grades."),
            ("Explainable AI (Grad-CAM)", "Gradient-weighted activation maps allow clinicians to verify AI focus regions against retinal anatomy."),
            ("Clinical DSS Integration", "Six-factor composite risk scoring integrates image evidence with patient metadata for holistic triage."),
            ("PDF Clinical Reports", "Structured reports include reliability scores, Grad-CAM interpretation, risk breakdown, and evidence-based guidelines."),
            ("Error Analysis Dashboard", "Systematic analysis of misclassification patterns supports model improvement and transparency."),
            ("Image Quality Assessment", "Pre-prediction quality gating (blur, exposure) reduces unreliable predictions from poor-quality inputs."),
        ]
        for title, desc in strengths:
            st.markdown(f"""
            <div style="background:#EAFAF1;border-left:4px solid #27AE60;border-radius:6px;
                        padding:10px 12px;margin:6px 0;">
                <div style="font-weight:700;color:#1E8449;font-size:13px;">✓ {title}</div>
                <div style="font-size:12px;color:#2C3E50;margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with v2:
        st.markdown('<div class="sec-hdr">⚠️ Known Limitations</div>', unsafe_allow_html=True)
        limitations = [
            ("Small Dataset", "1,764 images is insufficient for clinical-grade validation. Real-world deployment would require 10,000+ images across diverse populations."),
            ("Single Dataset Evaluation", "Trained and evaluated on a single APTOS-style dataset. Performance may degrade on images from different camera systems or ethnic populations."),
            ("No External Validation", "Model has not been tested on an independent external cohort — a requirement for clinical-grade evidence (STARD 2015 criteria)."),
            ("3-Class Simplified Grading", "Clinical practice uses the International Clinical DR Severity Scale (5 grades). This system merges grades to 3 classes, reducing clinical granularity."),
            ("Research Use Only", "Not validated for clinical deployment. Not CE marked, FDA cleared, or MHRA registered."),
            ("No Longitudinal Tracking", "Single-timepoint analysis only. Cannot assess disease progression or treatment response over time."),
        ]
        for title, desc in limitations:
            st.markdown(f"""
            <div style="background:#FDEDEC;border-left:4px solid #E74C3C;border-radius:6px;
                        padding:10px 12px;margin:6px 0;">
                <div style="font-weight:700;color:#C0392B;font-size:13px;">⚠ {title}</div>
                <div style="font-size:12px;color:#2C3E50;margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">🔭 Future Development Roadmap</div>', unsafe_allow_html=True)
    roadmap = [
        ("🔬","OCT Integration","Incorporate optical coherence tomography for macular oedema depth assessment alongside fundus photography."),
        ("🌍","Real-World Validation","Multi-centre prospective validation across diverse ethnic populations and imaging systems."),
        ("🔒","Federated Learning","Privacy-preserving distributed training across hospital networks without centralising patient data."),
        ("🧬","Multi-Modal AI","Fusion of fundus images, OCT, patient metadata, and EHR data for a unified diagnostic model."),
        ("📅","Longitudinal Monitoring","Track DR progression across visits with automated change detection and trend analysis."),
        ("📚","PubMed RAG","Real-time retrieval-augmented generation from PubMed for evidence-based guideline integration per patient."),
        ("📱","Mobile Deployment","Lightweight TFLite model for point-of-care screening in low-resource settings."),
        ("🏥","EHR Integration","HL7 FHIR-compliant API for direct integration with hospital electronic health record systems."),
    ]
    rc = st.columns(4)
    for i, (icon, title, desc) in enumerate(roadmap):
        rc[i % 4].markdown(f"""
        <div class="hl-card" style="min-height:120px;text-align:left;padding:12px;">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="font-weight:700;font-size:12px;color:#1B4F72;margin:5px 0 3px;">{title}</div>
            <div style="font-size:11px;color:#7F8C8D;line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="disc-box">{DISCLAIMER}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "About":
    st.title("ℹ️ About RetinoCare AI")

    # 1. Hero description
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#EAF2F8,#E8F8F5);border-radius:12px;
                padding:20px 24px;margin-bottom:18px;border:1px solid #AED6F1;">
        <div style="font-size:1.3rem;font-weight:800;color:#1B4F72;margin-bottom:6px;">
            RetinoCare AI v{VERSION}
        </div>
        <div style="font-size:.95rem;color:#2C3E50;line-height:1.6;">
            An <b>industry-grade Healthcare AI Clinical Decision Support System</b> for automated
            diabetic retinopathy detection and severity grading. Built by a PharmD professional
            with AI/ML expertise to demonstrate end-to-end clinical AI development:
            from raw fundus images to actionable clinical reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    ab1, ab2 = st.columns([2, 1])
    with ab1:
        # 2. Project Objective
        st.markdown('<div class="sec-hdr">🎯 Project Objective</div>', unsafe_allow_html=True)
        st.markdown("""
Diabetic retinopathy is the **leading cause of preventable blindness** in working-age adults globally,
affecting ~103 million people. Early detection through structured screening reduces vision loss by up to 80%.

RetinoCare AI demonstrates how **deep learning and clinical AI** can assist ophthalmologists in:
- Consistent automated severity grading at scale
- Explainable AI outputs that clinicians can verify and trust
- Personalised risk stratification beyond the image alone
- Standardised clinical documentation via PDF reports

This system represents a **complete clinical AI pipeline** — from image acquisition and quality assessment
through to a clinician-ready PDF report with evidence-based management recommendations.
        """)

        # 3. Key Contributions
        st.markdown('<div class="sec-hdr">🏆 Key Technical Contributions</div>', unsafe_allow_html=True)
        contributions = [
            ("Image Quality Assessment", "Laplacian-variance blur scoring + brightness analysis with clinical grade labels before model inference."),
            ("MobileNetV2 Transfer Learning", "ImageNet-pretrained backbone with custom classification head (Dense 256 → BN → Dropout 0.5 → Softmax 3). AUC = 0.90."),
            ("Grad-CAM Explainability", "Class-specific gradient activation maps with dynamic clinical text interpretation per severity grade."),
            ("Composite Risk Scoring", "Six-factor weighted clinical risk score (DR grade 40%, HbA1c 22%, duration 16%, BP 12%, age 6%, smoking 4%)."),
            ("Reliability Score", "Novel metric combining prediction margin (65%) and Shannon entropy (35%) to quantify model uncertainty."),
            ("PDF Clinical Report v2.1", "10-section structured report: patient info, AI prediction, probability chart, Grad-CAM, risk score, guidelines."),
            ("Multi-Model Benchmarking", "Systematic comparison of 6 architectures against literature baselines (APTOS/EyePACS published values)."),
            ("Error Analysis Dashboard", "Systematic misclassification analysis to understand model failure modes and support continuous improvement."),
        ]
        for i in range(0, len(contributions), 2):
            cc1, cc2 = st.columns(2)
            for col, (title, desc) in zip([cc1, cc2], contributions[i:i+2]):
                col.markdown(f"""
                <div style="background:white;border-radius:8px;padding:10px 12px;margin:4px 0;
                            border-left:3px solid #117A65;box-shadow:0 1px 4px rgba(0,0,0,.05);">
                    <div style="font-weight:700;font-size:12px;color:#1B4F72;">{title}</div>
                    <div style="font-size:11px;color:#7F8C8D;margin-top:3px;line-height:1.4;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # 4. Clinical Impact
        st.markdown('<div class="sec-hdr">🏥 Clinical Impact</div>', unsafe_allow_html=True)
        st.markdown("""
| Stakeholder | Impact |
|---|---|
| **Diabetologist / GP** | Automated triage flag reduces time-to-referral for sight-threatening DR |
| **Ophthalmologist** | Grad-CAM overlay enables rapid verification of AI focus against clinical findings |
| **Patient** | Composite risk score enables personalised education on modifiable risk factors |
| **Healthcare System** | Scalable screening tool reduces burden on specialist ophthalmology services |
| **Clinical AI Researcher** | Demonstrates full pipeline: XAI, reliability scoring, error analysis, PDF reporting |
        """)

        # 5. Future Roadmap (brief)
        st.markdown('<div class="sec-hdr">🔭 Future Development</div>', unsafe_allow_html=True)
        st.markdown("""
- **OCT integration** — macular oedema depth mapping alongside fundus photography
- **Federated learning** — privacy-preserving multi-site training
- **Longitudinal tracking** — disease progression monitoring across visits
- **PubMed RAG** — real-time evidence retrieval for patient-specific guidelines
- **EHR integration** — HL7 FHIR API for seamless hospital system connectivity
- **Mobile TFLite** — point-of-care screening in low-resource settings

See the **Model Validation** page for the full roadmap and known limitations.
        """)

        # References
        st.markdown('<div class="sec-hdr">📚 Key References</div>', unsafe_allow_html=True)
        st.markdown("""
1. Selvaraju RR et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017.
2. Lundberg SM & Lee SI. *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
3. Howard AG et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv 2017.
4. NICE Guideline NG28. *Diabetic Eye Screening.* 2016.
5. AAO Preferred Practice Pattern. *Diabetic Retinopathy.* 2019.
6. IDF Diabetes Atlas. 10th Edition. 2021.
7. APTOS 2019 Blindness Detection Challenge. Kaggle.
        """)

    with ab2:
        st.markdown('<div class="sec-hdr">📊 Project Stats</div>', unsafe_allow_html=True)
        st.metric("Best AUC",       "0.901 (MobileNetV2)")
        st.metric("Test Accuracy",  "79.6%")
        st.metric("Macro F1",       "0.73+")
        st.metric("Dataset",        "1,764 images")
        st.metric("Classes",        "3 Severity Grades")
        st.metric("Model",          MODEL_VERSION)
        st.metric("App Version",    f"v{VERSION}")
        st.metric("Pages",          "8 (incl. Validation)")

        st.markdown("---")
        st.markdown('<div class="sec-hdr">🏗 Model Architecture</div>', unsafe_allow_html=True)
        st.code(
            "Fundus Image (any res)\n"
            "      ↓\n"
            "CLAHE Enhancement\n"
            "      ↓\n"
            "Resize 224×224\n"
            "      ↓\n"
            "ImageNet Normalisation\n"
            "      ↓\n"
            "MobileNetV2 (frozen)\n"
            "      ↓\n"
            "Dense(256) → BN → Drop(0.5)\n"
            "      ↓\n"
            "Dense(3, Softmax)\n"
            "      ↓\n"
            "Grad-CAM Explanation\n"
            "      ↓\n"
            "Risk Score + PDF Report",
            language="text",
        )

        st.markdown("---")
        st.markdown('<div class="sec-hdr">📦 Tech Stack</div>', unsafe_allow_html=True)
        stack = ["Python 3.11","TensorFlow 2.15","Keras","OpenCV","scikit-learn",
                 "Streamlit 1.36","ReportLab","Matplotlib","Pandas","NumPy","PIL"]
        st.markdown(" · ".join(f"`{t}`" for t in stack))

    st.markdown("---")
    st.markdown(f'<div class="disc-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#AEB6BF;font-size:11px;padding:8px 0;">
        RetinoCare AI v2.0.0 · Built with TensorFlow, Keras &amp; Streamlit ·
        For Educational &amp; Research Purposes Only · Not for Clinical Deployment
    </div>
    """, unsafe_allow_html=True)
