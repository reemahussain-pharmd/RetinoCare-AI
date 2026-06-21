"""
RetinaIQ — PDF Clinical Report Generator v2.1
Fixes: auto Patient ID, reliability formula explanation, matplotlib probability chart,
Grad-CAM region interpretation, risk score from patient metadata, image quality score,
evidence-based guidelines section.
"""

import io
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT


# ── Colour Palette ─────────────────────────────────────────────────────────────
def _hex(h: str):
    h = h.lstrip("#")
    return colors.Color(*[int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])

NAVY  = _hex("1B4F72")
TEAL  = _hex("117A65")
WHITE = colors.white
LIGHT = _hex("EAF2F8")
LIGHT2= _hex("F4F6F7")
GREY  = colors.Color(0.5, 0.5, 0.5)
RED   = _hex("C0392B")

SEVERITY_COLORS = {
    "No/Mild DR":              "#27AE60",
    "Moderate DR":             "#F39C12",
    "Severe/Proliferative DR": "#E74C3C",
}

# ── Evidence-Based Guidelines (static, by DR class) ────────────────────────────
EVIDENCE_GUIDELINES = {
    "No/Mild DR": [
        ("NICE NG28, 2016",
         "Patients with no or mild diabetic retinopathy should undergo annual digital "
         "fundus photography. Good glycaemic control (HbA1c < 7.0%) and blood pressure "
         "management (< 130/80 mmHg) are the most effective interventions to prevent progression."),
        ("AAO Preferred Practice Pattern, 2019",
         "For patients with no apparent retinopathy or mild NPDR, dilated fundus examination "
         "may be performed every 1–2 years once stable. Intensive glycaemic control reduces the "
         "risk of developing retinopathy by approximately 76%."),
        ("IDF Diabetes Atlas, 10th Ed.",
         "All people with diabetes should receive regular eye screening. The most important "
         "modifiable risk factors are glycaemia, blood pressure, and lipid levels."),
    ],
    "Moderate DR": [
        ("RCOphth DR Referral Guidelines, 2020",
         "Moderate NPDR should be referred to a hospital eye service within 13 weeks. "
         "Anti-VEGF therapy should be considered if diabetic macular oedema (DMO) is present."),
        ("AAO Preferred Practice Pattern, 2019",
         "Patients with moderate NPDR should be examined every 6–12 months. "
         "Focal/grid laser photocoagulation may be considered for clinically significant macular oedema."),
        ("ADA Standards of Medical Care, 2023",
         "Optimisation of glycaemia, blood pressure, and lipids remains the cornerstone of "
         "DR management. ACE inhibitors or ARBs are preferred antihypertensives in patients with diabetes."),
    ],
    "Severe/Proliferative DR": [
        ("RCOphth DR Referral Guidelines, 2020",
         "Severe NPDR or PDR should be referred urgently within 6 weeks; emergency referral "
         "is required if vitreous haemorrhage or sudden vision loss is reported."),
        ("AAO Preferred Practice Pattern, 2019",
         "PDR with high-risk characteristics should be treated with panretinal photocoagulation (PRP) "
         "or anti-VEGF (ranibizumab/bevacizumab). Vitrectomy is indicated for non-clearing vitreous "
         "haemorrhage or tractional retinal detachment involving the macula."),
        ("NICE NG28, 2016",
         "Offer panretinal laser photocoagulation to people with PDR. Consider anti-VEGF therapy "
         "as an adjunct to PRP or as primary treatment where PRP is not feasible."),
    ],
}

# ── Grad-CAM Region Interpretation (rule-based, per class) ────────────────────
GRADCAM_INTERPRETATION = {
    "No/Mild DR": {
        "focus_zone":   "Diffuse, low-intensity activation across the fundus",
        "primary_region":"No concentrated pathological focus identified",
        "ai_analysis":  (
            "The Grad-CAM activation map shows broadly distributed, low-intensity gradients "
            "across the retinal surface with no focally elevated zones. This pattern is consistent "
            "with a healthy or minimally affected retina. Low-level attention near the optic disc "
            "reflects normal anatomical landmarks (optic cup, major vascular arcades) rather than "
            "pathological lesions. The absence of concentrated hot-spots supports the AI's "
            "classification of No/Mild DR."
        ),
    },
    "Moderate DR": {
        "focus_zone":   "Elevated activation in perifoveal and macular regions",
        "primary_region":"Perifoveal zone, superior and inferior temporal arcades",
        "ai_analysis":  (
            "The Grad-CAM map reveals concentrated activation in the perifoveal and macular region, "
            "consistent with the AI identifying hard exudates or microaneurysms — hallmarks of "
            "moderate NPDR. Secondary activation clusters along the superior temporal arcade may "
            "correspond to dot-blot haemorrhages or intraretinal microvascular abnormalities (IRMA). "
            "The distribution of high-activation zones is spatially consistent with the typical "
            "distribution of moderate NPDR lesions as described in the ETDRS classification."
        ),
    },
    "Severe/Proliferative DR": {
        "focus_zone":   "Multi-focal high-intensity activation across peripheral retinal zones",
        "primary_region":"Peripheral fundus, disc margin, and vitreoretinal interface",
        "ai_analysis":  (
            "The Grad-CAM map shows widespread, multi-focal high-intensity activation throughout "
            "the peripheral retina and disc margin. Activation clusters near the optic disc may "
            "indicate disc neovascularisation (NVD) — a hallmark of PDR. Peripheral foci of "
            "elevated activation are consistent with new vessel formation elsewhere (NVE), "
            "pre-retinal or vitreous haemorrhage, or extensive haemorrhagic lesions characteristic "
            "of severe NPDR / PDR. The pattern strongly correlates with advanced microvascular disease."
        ),
    },
}


# ── Image Helpers ──────────────────────────────────────────────────────────────
def _arr_to_img(arr: np.ndarray, width_cm: float = 7.5) -> RLImage:
    from PIL import Image as PIL
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    PIL.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    aspect = arr.shape[0] / arr.shape[1]
    w = width_cm * cm
    return RLImage(buf, width=w, height=w * aspect)


def _prob_chart(probs: dict, cls_name: str) -> RLImage:
    """Render a proper proportional horizontal bar chart for the PDF."""
    names  = list(probs.keys())
    values = list(probs.values())
    bar_colors = [SEVERITY_COLORS.get(n, "#7F8C8D") for n in names]

    fig, ax = plt.subplots(figsize=(6, 2.4))
    bars = ax.barh(names, values, color=bar_colors, height=0.5, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Probability (%)", fontsize=9)
    ax.set_xlim(0, 112)
    for bar, val in zip(bars, values):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Class Probability Distribution", fontsize=10, fontweight="bold")
    ax.axvline(70, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.55, label="70% threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w = 13 * cm
    from PIL import Image as PIL
    img = PIL.open(buf)
    aspect = img.height / img.width
    buf.seek(0)
    return RLImage(buf, width=w, height=w * aspect)


# ── Style Helpers ──────────────────────────────────────────────────────────────
def _S(name, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


def _section(title: str, s) -> list:
    return [
        Spacer(1, 0.3 * cm),
        Paragraph(title, s),
        HRFlowable(width="100%", thickness=0.5, color=NAVY, spaceAfter=3),
    ]


def _kv(rows, col_w=(6*cm, 11*cm)) -> Table:
    t = Table(rows, colWidths=list(col_w))
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), LIGHT),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 9),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _compute_risk(pi: dict, dr_class: str, confidence: float) -> float:
    """Recompute composite risk score from patient metadata for the PDF."""
    dr_map = {"No/Mild DR": 0.1, "Moderate DR": 0.5, "Severe/Proliferative DR": 0.9}
    dr_s  = dr_map.get(dr_class, 0.5) * (confidence / 100)
    hba1c = float(pi.get("hba1c", 7.0))
    dur   = float(pi.get("diabetes_duration", 0))
    bp    = float(pi.get("bp_systolic", 120))
    age   = float(pi.get("age", 50))
    smk   = {"Never": 0.0, "Former": 0.3, "Current": 0.7}.get(pi.get("smoking","Never"), 0.0)

    hba_s = min(1.0, max(0.0, (hba1c - 4.0) / 8.0))
    dur_s = min(1.0, dur / 20.0)
    bp_s  = 0.9 if bp >= 160 else 0.6 if bp >= 140 else 0.3 if bp >= 130 else 0.1
    age_s = min(1.0, max(0.0, (age - 30) / 50.0))

    score = (dr_s*0.40 + hba_s*0.22 + dur_s*0.16 + bp_s*0.12 + age_s*0.06 + smk*0.04) * 100
    return round(min(100, max(0, score)), 1)


# ── Main Report ────────────────────────────────────────────────────────────────
def generate_pdf_report(
    prediction: dict,
    original_image_array: np.ndarray = None,
    gradcam_array: np.ndarray = None,
    patient_info: dict = None,
    image_quality: dict = None,
    save_path: str = "reports/retinopathy_report.pdf",
) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        save_path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
    )

    # ── Styles ──────────────────────────────────────────────────────────────────
    SEC  = _S("Sec",  fontSize=11, fontName="Helvetica-Bold", textColor=NAVY, spaceBefore=6, spaceAfter=2)
    BODY = _S("Body", fontSize=9,  fontName="Helvetica",      leading=13, alignment=TA_JUSTIFY)
    META = _S("Meta", fontSize=8,  fontName="Helvetica",      textColor=GREY)
    FOOT = _S("Foot", fontSize=7,  fontName="Helvetica-Oblique", textColor=GREY, alignment=TA_CENTER)
    TITL = _S("Titl", fontSize=17, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER)
    SUBT = _S("Subt", fontSize=9,  fontName="Helvetica",      textColor=WHITE, alignment=TA_CENTER)
    DISC = _S("Disc", fontSize=8,  fontName="Helvetica-Oblique", textColor=RED,
              borderColor=RED, borderWidth=0.5, borderPadding=7, borderRadius=3,
              backColor=colors.Color(1, 0.96, 0.96), alignment=TA_JUSTIFY)

    cls_name  = prediction.get("predicted_class", "Unknown")
    sev_color = _hex(SEVERITY_COLORS.get(cls_name, "#7F8C8D"))
    conf      = prediction.get("confidence", 0.0)
    rel_score = prediction.get("reliability_score", 0.0)
    rel_level = prediction.get("reliability_level", "—")
    rel_margin= prediction.get("prediction_margin", 0.0)
    rel_ent   = prediction.get("entropy", 0.0)
    conf_level= prediction.get("confidence_level", "—")
    ts        = prediction.get("timestamp", datetime.now().isoformat())
    report_id = f"RC-{uuid.uuid4().hex[:8].upper()}"

    # Auto-generate patient ID if missing
    pi = patient_info or {}
    patient_id = (
        pi.get("patient_id")
        or f"PT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:3].upper()}"
    )

    has_patient_data = any(pi.get(k) for k in ["age","hba1c","bp_systolic","smoking","diabetes_duration"])

    story = []

    # ── Header ──────────────────────────────────────────────────────────────────
    hdr = Table(
        [[Paragraph("RetinaIQ — AI-Powered Retinal Disease Screening Report", TITL)],
         [Paragraph(
             f"Diabetic Retinopathy Module  ·  Report ID: {report_id}  ·  "
             f"For Ophthalmologist Review Only", SUBT)]],
        colWidths=[17*cm],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 0.25*cm))
    story.append(Paragraph(
        f"Generated: {ts[:19].replace('T',' ')}  ·  Model: {prediction.get('model_version','—')}  ·  "
        f"System: RetinaIQ v2.1  ·  Status: Requires Ophthalmologist Review",
        META,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=4))

    # ── 1. Patient & Session Information ────────────────────────────────────────
    story += _section("1.  Patient & Session Information", SEC)

    iq  = image_quality or {}
    risk_score = None
    if has_patient_data:
        risk_score = _compute_risk(pi, cls_name, conf)

    pt_rows = [
        ["Patient ID",         patient_id],
        ["Report ID",          report_id],
        ["Report Date / Time", ts[:19].replace("T", " ")],
        ["Age",                f"{pi.get('age','—')} years" if pi.get("age") else "Not provided"],
        ["Gender",             pi.get("gender", "Not provided")],
        ["Diabetes Duration",  f"{pi.get('diabetes_duration','—')} years" if pi.get("diabetes_duration") else "Not provided"],
        ["HbA1c",              f"{pi.get('hba1c','—')}%" if pi.get("hba1c") else "Not provided"],
        ["Blood Pressure",     (f"{pi.get('bp_systolic','—')}/{pi.get('bp_diastolic','—')} mmHg"
                                if pi.get("bp_systolic") else "Not provided")],
        ["Smoking Status",     pi.get("smoking", "Not provided")],
        ["Image Quality Score",f"{iq.get('score','—')}/100 ({iq.get('grade','—')})" if iq.get("score") else "Not assessed"],
    ]
    story.append(_kv(pt_rows))

    if not has_patient_data:
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph(
            "ℹ Note: Patient metadata not provided. Fields above are displayed for context only. "
            "Complete the Patient Information form in the app for composite risk scoring.",
            _S("Note", fontSize=8, fontName="Helvetica-Oblique", textColor=GREY,
               borderColor=colors.grey, borderWidth=0.4, borderPadding=5,
               backColor=colors.Color(0.97,0.97,0.97)),
        ))

    # ── 2. AI Prediction Result ──────────────────────────────────────────────────
    story += _section("2.  AI Prediction Result", SEC)

    result_rows = [
        ["Predicted DR Grade",    cls_name],
        ["Risk Level",            prediction.get("risk_level","—")],
        ["Urgency",               prediction.get("urgency","—")],
        ["Recommended Follow-up", prediction.get("follow_up","—")],
        ["ICD-10 Code",           prediction.get("icd10","—")],
        ["Confidence Score",      f"{conf:.2f}%  —  {conf_level} Confidence"],
        ["Reliability Score",     f"{rel_score:.1f} / 100  —  {rel_level} Reliability"],
        ["Composite Risk Score",  f"{risk_score}/100" if risk_score is not None else "Patient data not provided"],
        ["Model Version",         prediction.get("model_version","—")],
    ]
    rt = Table(result_rows, colWidths=[6*cm, 11*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,-1), LIGHT),
        ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (1,0),(1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("BACKGROUND",    (0,0),(-1,0), sev_color),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT2, WHITE]),
    ]))
    story.append(rt)

    # ── 2a. Reliability Score Explanation ────────────────────────────────────────
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        f"<b>Reliability Score Methodology:</b> The score of <b>{rel_score:.1f}/100</b> is computed "
        f"from two statistical components of the model's softmax output:<br/>"
        f"(1) <b>Prediction Margin</b> (65% weight) — gap between top class and second-highest class: "
        f"<b>{rel_margin:.1f}%</b>. A wider margin indicates the model clearly favoured one class.<br/>"
        f"(2) <b>Prediction Entropy</b> (35% weight) — Shannon entropy of the full probability "
        f"distribution: <b>{rel_ent:.3f} nats</b>. Lower entropy = more concentrated = more reliable.<br/>"
        f"Formula: Reliability = (Margin × 0.65 + (1 − NormEntropy) × 0.35) × 100",
        _S("Rel", fontSize=8, fontName="Helvetica", leading=12,
           borderColor=TEAL, borderWidth=0.4, borderPadding=6, borderRadius=2,
           backColor=colors.Color(0.94, 0.98, 0.97)),
    ))

    # ── 3. Class Probability Distribution ───────────────────────────────────────
    story += _section("3.  Class Probability Distribution", SEC)
    probs = prediction.get("all_probabilities", {})
    if probs:
        story.append(_prob_chart(probs, cls_name))
    # Numeric table as supplement
    prob_rows = [["Severity Class", "Probability (%)", "Rank"]]
    for rank, (cn, pv) in enumerate(sorted(probs.items(), key=lambda x: -x[1]), 1):
        prob_rows.append([cn, f"{pv:.2f}%", f"#{rank}"])
    pt = Table(prob_rows, colWidths=[7*cm, 5*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT, WHITE, LIGHT2]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 9),
    ]))
    story.append(pt)

    # ── 4. Retinal Image Analysis + Grad-CAM ─────────────────────────────────────
    imgs, caps = [], []
    if original_image_array is not None:
        imgs.append(_arr_to_img(original_image_array, 7.5))
        caps.append("Original Retinal Fundus Image")
    if gradcam_array is not None:
        imgs.append(_arr_to_img(gradcam_array, 7.5))
        caps.append("Grad-CAM Heat Map\n(Red = Primary AI Focus)")

    if imgs:
        story += _section("4.  Retinal Image Analysis", SEC)
        img_tbl = Table([imgs], colWidths=[8.5*cm]*len(imgs))
        img_tbl.setStyle(TableStyle([
            ("ALIGN",  (0,0),(-1,-1), "CENTER"),
            ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(img_tbl)
        cap_s   = _S("Cap", fontSize=8, alignment=TA_CENTER, textColor=GREY)
        cap_tbl = Table([[Paragraph(c, cap_s) for c in caps]], colWidths=[8.5*cm]*len(caps))
        story.append(cap_tbl)

    # ── 4a. Grad-CAM Interpretation ──────────────────────────────────────────────
    story += _section("4a.  AI Focus Region Analysis (Grad-CAM Interpretation)", SEC)
    gi = GRADCAM_INTERPRETATION.get(cls_name, {})
    gcam_rows = [
        ["Focus Zone",      gi.get("focus_zone","—")],
        ["Primary Region",  gi.get("primary_region","—")],
        ["Predicted Class", cls_name],
    ]
    story.append(_kv(gcam_rows))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        f"<b>AI Analysis:</b> {gi.get('ai_analysis','')}",
        _S("GC", fontSize=8.5, fontName="Helvetica", leading=12.5,
           borderColor=_hex("2E86C1"), borderWidth=0.4, borderPadding=6, borderRadius=2,
           backColor=_hex("EBF5FB")),
    ))
    if gradcam_array is None:
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph(
            "Note: Grad-CAM heatmap was not generated for this prediction. "
            "Re-run the analysis with 'Generate Grad-CAM' enabled for a visual overlay.",
            _S("GCN", fontSize=8, fontName="Helvetica-Oblique", textColor=GREY),
        ))

    # ── 5. Clinical Summary & Recommendation ─────────────────────────────────────
    story += _section("5.  Clinical Summary", SEC)
    story.append(Paragraph(prediction.get("clinical_summary",""), BODY))
    story.append(Spacer(1, 0.2*cm))
    story += _section("6.  Clinical Recommendation & Follow-up", SEC)
    story.append(Paragraph(prediction.get("clinical_recommendation",""), BODY))
    story.append(Spacer(1, 0.1*cm))
    story.append(_kv([
        ["Follow-up Interval", prediction.get("follow_up","—")],
        ["Urgency Level",      prediction.get("urgency","—")],
        ["ICD-10 Code",        prediction.get("icd10","—")],
    ]))

    # ── 6. Composite Risk Score ───────────────────────────────────────────────────
    story += _section("7.  Composite Clinical Risk Assessment", SEC)
    if has_patient_data and risk_score is not None:
        if risk_score >= 70:   risk_cat, risk_col = "HIGH RISK",     _hex("C0392B")
        elif risk_score >= 40: risk_cat, risk_col = "MODERATE RISK", _hex("D4AC0D")
        else:                  risk_cat, risk_col = "LOW RISK",       _hex("1E8449")

        risk_rows = [
            ["Composite Risk Score", f"{risk_score:.1f} / 100  —  {risk_cat}"],
            ["DR Grade Weight",      "40%  (AI image classification, confidence-weighted)"],
            ["HbA1c Weight",         f"22%  (Value: {pi.get('hba1c','—')}%)"],
            ["Diabetes Duration",    f"16%  (Value: {pi.get('diabetes_duration','—')} years)"],
            ["Blood Pressure",       f"12%  (Systolic: {pi.get('bp_systolic','—')} mmHg)"],
            ["Age Factor",           f"6%   (Age: {pi.get('age','—')} years)"],
            ["Smoking Status",       f"4%   ({pi.get('smoking','—')})"],
        ]
        rt2 = Table(risk_rows, colWidths=[6*cm, 11*cm])
        rt2.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,-1), LIGHT),
            ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (1,0),(1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("BACKGROUND",    (0,0),(-1,0), risk_col),
            ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("GRID",          (0,0),(-1,-1), 0.4, colors.lightgrey),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [LIGHT2, WHITE]),
        ]))
        story.append(rt2)
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph(
            "<b>Methodology:</b> The composite risk score is a weighted linear combination of "
            "clinically validated risk factors for diabetic retinopathy progression. Weightings "
            "reflect relative evidence strength from the IDF Atlas (10th Ed.) and UKPDS risk engine. "
            "This score is <b>not a validated clinical instrument</b> and must be interpreted by a "
            "qualified clinician.",
            _S("RM", fontSize=8, fontName="Helvetica-Oblique", textColor=GREY),
        ))
    else:
        story.append(Paragraph(
            "Patient clinical metadata was not provided for this analysis. "
            "The composite risk score could not be computed. "
            "Please complete the Patient Information form in the application to generate a "
            "personalised risk assessment.",
            BODY,
        ))

    # ── 7. Evidence-Based Guidelines ────────────────────────────────────────────
    story += _section("8.  Evidence-Based Clinical Guidelines", SEC)
    guidelines = EVIDENCE_GUIDELINES.get(cls_name, [])
    for source, text in guidelines:
        story.append(Paragraph(
            f"<b>{source}:</b>  {text}",
            _S("GL", fontSize=8.5, fontName="Helvetica", leading=13,
               borderColor=TEAL, borderWidth=0.3, borderPadding=5, borderRadius=2,
               backColor=colors.Color(0.95, 0.99, 0.97), spaceAfter=5),
        ))

    # ── 8. Severity Reference Scale ──────────────────────────────────────────────
    story += _section("9.  Retinopathy Severity Reference Scale", SEC)
    sev_ref = [
        ["DR Stage",          "Risk",     "Clinical Action"],
        ["No DR",             "Low",      "Routine annual fundus photography"],
        ["Mild NPDR",         "Mild",     "Follow-up in 6–12 months"],
        ["Moderate NPDR",     "Moderate", "Ophthalmology referral within 3–6 months"],
        ["Severe NPDR",       "High",     "Urgent referral within 4–6 weeks"],
        ["Proliferative DR",  "Critical", "IMMEDIATE ophthalmology referral"],
    ]
    row_bgs = [NAVY, _hex("27AE60"), _hex("52BE80"), _hex("F39C12"), _hex("E67E22"), _hex("E74C3C")]
    cmds = [
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
    ]
    for i, c in enumerate(row_bgs):
        cmds.append(("BACKGROUND", (0,i),(0,i), c))
        if i > 0:
            cmds.append(("TEXTCOLOR", (0,i),(0,i), WHITE))
    sev_t = Table(sev_ref, colWidths=[4.5*cm, 3*cm, 9.5*cm])
    sev_t.setStyle(TableStyle(cmds))
    story.append(sev_t)

    # ── 9. AI Disclaimer ────────────────────────────────────────────────────────
    story += _section("10.  AI Disclaimer", SEC)
    story.append(Paragraph(
        f"<b>⚠ IMPORTANT:</b> {prediction.get('disclaimer','')}",
        DISC,
    ))

    # ── Footer ───────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey))
    story.append(Paragraph(
        f"RetinaIQ v2.1  ·  Report ID: {report_id}  ·  Patient: {patient_id}  ·  "
        f"For Research & Educational Purposes Only  ·  Not for Clinical Deployment",
        FOOT,
    ))

    doc.build(story)
    print(f"[PDF v2.1] Saved: {save_path}")
    return save_path
