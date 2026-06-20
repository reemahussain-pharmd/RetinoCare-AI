"""
RetinoCare AI — PDF Clinical Report Generator v2.0
Produces a professional diagnostic support report with patient metadata,
reliability scores, Grad-CAM visualization, and clinical recommendations.
"""

import io
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT


# ── Colour Palette ─────────────────────────────────────────────────────────────
def _hex(h: str):
    h = h.lstrip("#")
    return colors.Color(*[int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])

NAVY    = _hex("1B4F72")
TEAL    = _hex("117A65")
WHITE   = colors.white
BLACK   = colors.black
LIGHT   = _hex("EAF2F8")
LIGHT2  = _hex("F4F6F7")
GREY    = colors.Color(0.5, 0.5, 0.5)
AMBER   = _hex("D4AC0D")
RED     = _hex("C0392B")
GREEN   = _hex("1E8449")

SEVERITY_COLORS = {
    "No/Mild DR":              "#27AE60",
    "Moderate DR":             "#F39C12",
    "Severe/Proliferative DR": "#E74C3C",
}


# ── Image helpers ──────────────────────────────────────────────────────────────
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


# ── Style helpers ──────────────────────────────────────────────────────────────
def _style(name, **kw):
    return ParagraphStyle(name, **kw)


def _section(title: str, styles) -> list:
    return [
        Spacer(1, 0.35 * cm),
        Paragraph(title, styles["section"]),
        HRFlowable(width="100%", thickness=0.5, color=NAVY, spaceAfter=4),
    ]


def _kv_table(rows: list, col_w=(6 * cm, 11 * cm)) -> Table:
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


# ── Main Report Function ───────────────────────────────────────────────────────
def generate_pdf_report(
    prediction: dict,
    original_image_array: np.ndarray = None,
    gradcam_array: np.ndarray = None,
    patient_info: dict = None,
    save_path: str = "reports/retinopathy_report.pdf",
) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )

    # ── Styles ──────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()
    S = {
        "title":   _style("T",  fontSize=17, fontName="Helvetica-Bold",   textColor=WHITE,  alignment=TA_CENTER),
        "sub":     _style("Su", fontSize=9,  fontName="Helvetica",        textColor=WHITE,  alignment=TA_CENTER),
        "section": _style("Sc", fontSize=11, fontName="Helvetica-Bold",   textColor=NAVY,   spaceBefore=6, spaceAfter=2),
        "body":    _style("B",  fontSize=9,  fontName="Helvetica",        leading=13,       alignment=TA_JUSTIFY),
        "small":   _style("Sm", fontSize=7.5,fontName="Helvetica",        textColor=GREY,   alignment=TA_CENTER),
        "label":   _style("L",  fontSize=8,  fontName="Helvetica-Bold",   textColor=NAVY),
        "foot":    _style("F",  fontSize=7,  fontName="Helvetica-Oblique",textColor=GREY,   alignment=TA_CENTER),
    }

    cls_name  = prediction.get("predicted_class", "Unknown")
    sev_color = _hex(SEVERITY_COLORS.get(cls_name, "#7F8C8D"))
    conf      = prediction.get("confidence", 0)
    rel_score = prediction.get("reliability_score", 0)
    rel_level = prediction.get("reliability_level", "—")
    conf_level= prediction.get("confidence_level", "—")
    ts        = prediction.get("timestamp", datetime.now().isoformat())
    report_id = f"RC-{uuid.uuid4().hex[:8].upper()}"
    patient_id= (patient_info or {}).get("patient_id", f"PT-{uuid.uuid4().hex[:6].upper()}")

    story = []

    # ── Header ──────────────────────────────────────────────────────────────────
    header_data = [
        [Paragraph("RetinoCare AI — Clinical Decision Support Report", S["title"])],
        [Paragraph(
            f"AI-Assisted Diabetic Retinopathy Screening  |  Report ID: {report_id}  |  "
            f"For Ophthalmologist Review Only",
            S["sub"],
        )],
    ]
    hdr = Table(header_data, colWidths=[17 * cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 0.3 * cm))

    # Report meta row
    meta_s = _style("M", fontSize=8, textColor=GREY)
    story.append(Paragraph(
        f"Generated: {ts[:19].replace('T', ' ')}  ·  "
        f"Model: {prediction.get('model_version', '—')}  ·  "
        f"System: RetinoCare AI v2.0  ·  Status: Requires Ophthalmologist Review",
        meta_s,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=4))

    # ── 1. Patient Information ───────────────────────────────────────────────────
    story += _section("1.  Patient & Session Information", S)

    pi = patient_info or {}
    pt_rows = [
        ["Patient ID",          patient_id],
        ["Report ID",           report_id],
        ["Report Date/Time",    ts[:19].replace("T", " ")],
        ["Age",                 str(pi.get("age", "Not provided"))],
        ["Gender",              pi.get("gender", "Not provided")],
        ["Diabetes Duration",   f"{pi.get('diabetes_duration', '—')} years" if pi.get("diabetes_duration") else "Not provided"],
        ["HbA1c",               f"{pi.get('hba1c', '—')}%" if pi.get("hba1c") else "Not provided"],
        ["Blood Pressure",      f"{pi.get('bp_systolic', '—')}/{pi.get('bp_diastolic', '—')} mmHg" if pi.get("bp_systolic") else "Not provided"],
        ["Smoking Status",      pi.get("smoking", "Not provided")],
    ]
    story.append(_kv_table(pt_rows))

    # ── 2. AI Prediction Result ──────────────────────────────────────────────────
    story += _section("2.  AI Prediction Result", S)

    result_rows = [
        ["Predicted DR Grade",     cls_name],
        ["Risk Level",             prediction.get("risk_level", "—")],
        ["Urgency",                prediction.get("urgency", "—")],
        ["ICD-10 Code",            prediction.get("icd10", "—")],
        ["Confidence Score",       f"{conf:.2f}%  ({conf_level} Confidence)"],
        ["Reliability Score",      f"{rel_score:.1f} / 100  ({rel_level} Reliability)"],
        ["Recommended Follow-up",  prediction.get("follow_up", "—")],
        ["Model Version",          prediction.get("model_version", "—")],
    ]

    rt = Table(result_rows, colWidths=[6 * cm, 11 * cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), LIGHT),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        # Highlight predicted grade row
        ("BACKGROUND",    (0, 0), (-1, 0), sev_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT2, WHITE]),
    ]))
    story.append(rt)

    # ── 3. Class Probability Distribution ───────────────────────────────────────
    story += _section("3.  Class Probability Distribution", S)

    probs = prediction.get("all_probabilities", {})
    prob_rows = [["Severity Class", "Probability", "Visual Bar"]]
    for cn, prob in probs.items():
        filled = int(prob / 100 * 28)
        bar    = "█" * filled + "░" * (28 - filled)
        prob_rows.append([cn, f"{prob:.2f}%", bar])

    pt = Table(prob_rows, colWidths=[5.5 * cm, 3 * cm, 8.5 * cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE, LIGHT2]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.2 * cm))

    # ── 4. Retinal Image Analysis ────────────────────────────────────────────────
    imgs, caps = [], []
    if original_image_array is not None:
        imgs.append(_arr_to_img(original_image_array, 7.5))
        caps.append("Original Retinal Fundus Image")
    if gradcam_array is not None:
        imgs.append(_arr_to_img(gradcam_array, 7.5))
        caps.append("Grad-CAM Heat Map\n(Red = High AI Attention)")

    if imgs:
        story += _section("4.  Retinal Image Analysis", S)
        img_table = Table([imgs], colWidths=[8.5 * cm] * len(imgs))
        img_table.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(img_table)
        cap_s   = _style("Cap", fontSize=8, alignment=TA_CENTER, textColor=GREY)
        cap_tbl = Table([[Paragraph(c, cap_s) for c in caps]], colWidths=[8.5 * cm] * len(caps))
        story.append(cap_tbl)
        story.append(Spacer(1, 0.2 * cm))

    # ── 5. Clinical Summary ──────────────────────────────────────────────────────
    story += _section("5.  Clinical Summary", S)
    story.append(Paragraph(prediction.get("clinical_summary", prediction.get("clinical_recommendation", "")), S["body"]))

    # ── 6. Clinical Recommendation & Follow-up ───────────────────────────────────
    story += _section("6.  Clinical Recommendation & Follow-up", S)
    story.append(Paragraph(prediction.get("clinical_recommendation", ""), S["body"]))
    story.append(Spacer(1, 0.15 * cm))

    fu_rows = [
        ["Follow-up Interval",  prediction.get("follow_up", "—")],
        ["Urgency Level",       prediction.get("urgency", "—")],
        ["ICD-10 Code",         prediction.get("icd10", "—")],
    ]
    story.append(_kv_table(fu_rows))

    # ── 7. Severity Reference Scale ───────────────────────────────────────────────
    story += _section("7.  Diabetic Retinopathy Severity Reference", S)
    sev_ref = [
        ["DR Stage",             "Risk",     "Typical Action"],
        ["No DR",                "Low",      "Routine annual fundus photography"],
        ["Mild NPDR",            "Mild",     "Follow-up in 6–12 months"],
        ["Moderate NPDR",        "Moderate", "Ophthalmology referral within 3–6 months"],
        ["Severe NPDR",          "High",     "Urgent referral within 1 month"],
        ["Proliferative DR",     "Critical", "IMMEDIATE ophthalmology referral"],
    ]
    row_colors = [NAVY, _hex("27AE60"), _hex("52BE80"), _hex("F39C12"), _hex("E67E22"), _hex("E74C3C")]
    st_cmds = [
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ]
    for i, rc in enumerate(row_colors):
        st_cmds.append(("BACKGROUND", (0, i), (0, i), rc))
        if i > 0:
            st_cmds.append(("TEXTCOLOR", (0, i), (0, i), WHITE))
    sev_t = Table(sev_ref, colWidths=[4.5 * cm, 3 * cm, 9.5 * cm])
    sev_t.setStyle(TableStyle(st_cmds))
    story.append(sev_t)

    # ── 8. AI Disclaimer ─────────────────────────────────────────────────────────
    story += _section("8.  AI Disclaimer", S)
    disc_s = _style("D", fontSize=8, fontName="Helvetica-Oblique", textColor=RED,
                    borderColor=RED, borderWidth=0.5, borderPadding=8, borderRadius=3,
                    backColor=colors.Color(1, 0.96, 0.96), alignment=TA_JUSTIFY)
    story.append(Paragraph(
        f"<b>⚠ IMPORTANT:</b> {prediction.get('disclaimer', '')}",
        disc_s,
    ))

    # ── Footer ───────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey))
    story.append(Paragraph(
        f"RetinoCare AI v2.0  ·  Report ID: {report_id}  ·  "
        f"For Research & Educational Purposes Only  ·  Not for Clinical Deployment",
        S["foot"],
    ))

    doc.build(story)
    print(f"[PDF] Report saved: {save_path}")
    return save_path
