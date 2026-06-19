"""
PDF Clinical Report Generator using ReportLab.
Produces a professional diagnostic support report with Grad-CAM visualization.
"""

import io
import os
import numpy as np
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# ── Colour Palette ─────────────────────────────────────────────────────────────
SEVERITY_COLORS_HEX = {
    "No/Mild DR":              "#4CAF50",
    "Moderate DR":             "#FFC107",
    "Severe/Proliferative DR": "#F44336",
}

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return colors.Color(*[int(h[i:i+2], 16)/255 for i in (0, 2, 4)])

NAVY   = colors.Color(0.10, 0.20, 0.45)
LIGHT  = colors.Color(0.95, 0.97, 1.00)
WHITE  = colors.white
BLACK  = colors.black


# ── Image helper ───────────────────────────────────────────────────────────────
def _array_to_rl_image(arr: np.ndarray, width_cm: float = 7.0) -> RLImage:
    """Convert numpy [H,W,3] float/uint8 array to a ReportLab Image."""
    from PIL import Image as PILImage
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    pil = PILImage.open(buf)
    aspect = pil.height / pil.width
    w = width_cm * cm
    h = w * aspect
    buf.seek(0)
    return RLImage(buf, width=w, height=h)


def _image_path_to_rl(path: str, width_cm: float = 7.0) -> RLImage:
    from PIL import Image as PILImage
    pil = PILImage.open(path)
    aspect = pil.height / pil.width
    w = width_cm * cm
    return RLImage(path, width=w, height=w * aspect)


# ── Report Builder ─────────────────────────────────────────────────────────────
def generate_pdf_report(
    prediction: dict,
    original_image_path: str = None,
    original_image_array: np.ndarray = None,
    gradcam_array: np.ndarray = None,
    save_path: str = "reports/retinopathy_report.pdf",
) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    cls_name = prediction.get("predicted_class", "Unknown")
    sev_color = _hex_to_rgb(SEVERITY_COLORS_HEX.get(cls_name, "#9E9E9E"))

    # ── Header ──────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title", fontSize=18, fontName="Helvetica-Bold",
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", fontSize=10, fontName="Helvetica",
        textColor=WHITE, alignment=TA_CENTER,
    )

    header_data = [[
        Paragraph("RetinoCare AI — Diagnostic Support Report", title_style),
    ], [
        Paragraph("AI-Powered Retinopathy Detection System | For Clinical Review Only", sub_style),
    ]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Report Metadata ──────────────────────────────────────────────────────────
    meta_style = ParagraphStyle("Meta", fontSize=9, textColor=colors.grey)
    story.append(Paragraph(
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"System: RetinoCare AI v1.0 | Status: AI-Assisted — Requires Ophthalmologist Review",
        meta_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=6))

    # ── Prediction Result ────────────────────────────────────────────────────────
    section_style = ParagraphStyle(
        "Section", fontSize=13, fontName="Helvetica-Bold",
        textColor=NAVY, spaceBefore=10, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", fontSize=10, fontName="Helvetica",
        leading=14, alignment=TA_JUSTIFY, spaceAfter=6,
    )

    story.append(Paragraph("1. Prediction Result", section_style))

    result_data = [
        ["Parameter",         "Value"],
        ["Predicted Class",   cls_name],
        ["Risk Level",        prediction.get("risk_level", "—")],
        ["Confidence Score",  f"{prediction.get('confidence', 0):.2f}%"],
        ["AI Model",          "Best-Performing Model (Auto-Selected)"],
    ]
    result_table = Table(result_data, colWidths=[6*cm, 11*cm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("BACKGROUND",    (0, 1), (-1, 1),  sev_color),
        ("TEXTCOLOR",     (0, 1), (-1, 1),  WHITE),
        ("FONTNAME",      (0, 1), (-1, 1),  "Helvetica-Bold"),
        ("BACKGROUND",    (0, 2), (-1, -1), LIGHT),
        ("ROWBACKGROUNDS",(0, 2), (-1, -1), [LIGHT, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*cm))

    # ── Probability Distribution ─────────────────────────────────────────────────
    story.append(Paragraph("2. Class Probability Distribution", section_style))
    probs = prediction.get("all_probabilities", {})
    prob_data = [["Severity Class", "Probability (%)", "Visual Bar"]]
    for cn, prob in probs.items():
        bar_len = int(prob / 100 * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        prob_data.append([cn, f"{prob:.2f}%", bar])

    prob_table = Table(prob_data, colWidths=[5.5*cm, 4*cm, 7.5*cm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.3*cm))

    # ── Images ───────────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Retinal Image Analysis", section_style))

    images_row = []
    captions   = []

    if original_image_path and Path(original_image_path).exists():
        images_row.append(_image_path_to_rl(original_image_path, width_cm=7.5))
        captions.append("Original Retinal Image")
    elif original_image_array is not None:
        images_row.append(_array_to_rl_image(original_image_array, width_cm=7.5))
        captions.append("Original Retinal Image")

    if gradcam_array is not None:
        images_row.append(_array_to_rl_image(gradcam_array, width_cm=7.5))
        captions.append("Grad-CAM Explanation\n(Red regions = AI focus areas)")

    if images_row:
        img_table = Table([images_row], colWidths=[8.5*cm] * len(images_row))
        img_table.setStyle(TableStyle([
            ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(img_table)

        cap_style = ParagraphStyle("Cap", fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
        cap_table = Table([[Paragraph(c, cap_style) for c in captions]],
                          colWidths=[8.5*cm] * len(captions))
        story.append(cap_table)
        story.append(Spacer(1, 0.3*cm))

    # ── Clinical Recommendation ──────────────────────────────────────────────────
    story.append(Paragraph("4. Clinical Interpretation & Recommendation", section_style))
    story.append(Paragraph(prediction.get("clinical_recommendation", ""), body_style))

    # ── Severity Scale ───────────────────────────────────────────────────────────
    story.append(Paragraph("5. Retinopathy Severity Reference Scale", section_style))
    severity_data = [
        ["Stage",             "Risk",     "Clinical Action"],
        ["No DR",             "Low",      "Routine annual screening"],
        ["Mild DR",           "Mild",     "Follow-up in 6-12 months"],
        ["Moderate DR",       "Moderate", "Ophthalmology referral within 3-6 months"],
        ["Severe DR",         "High",     "Urgent referral within 1 month"],
        ["Proliferative DR",  "Critical", "IMMEDIATE ophthalmology referral"],
    ]
    sev_colors_rl = [WHITE, _hex_to_rgb("#4CAF50"), _hex_to_rgb("#8BC34A"),
                     _hex_to_rgb("#FFC107"), _hex_to_rgb("#FF5722"), _hex_to_rgb("#F44336")]
    sev_table = Table(severity_data, colWidths=[5*cm, 3*cm, 9*cm])
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]
    for row_idx, rc in enumerate(sev_colors_rl[1:], start=1):
        style_cmds.append(("BACKGROUND", (0, row_idx), (0, row_idx), rc))
        style_cmds.append(("TEXTCOLOR",  (0, row_idx), (0, row_idx), WHITE))
    sev_table.setStyle(TableStyle(style_cmds))
    story.append(sev_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Disclaimer ───────────────────────────────────────────────────────────────
    disclaimer_style = ParagraphStyle(
        "Disclaimer", fontSize=8, fontName="Helvetica-Oblique",
        textColor=colors.darkgrey, borderColor=colors.red,
        borderWidth=1, borderPadding=8, borderRadius=4,
        backColor=colors.Color(1, 0.97, 0.97),
        spaceBefore=10, alignment=TA_JUSTIFY,
    )
    story.append(Paragraph(
        f"<b>⚠ AI DISCLAIMER:</b> {prediction.get('disclaimer', '')}",
        disclaimer_style,
    ))

    # ── Footer ───────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    footer_style = ParagraphStyle("Footer", fontSize=7, textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(
        "RetinoCare AI | AI-Powered Retinopathy Detection | For Research & Educational Purposes Only",
        footer_style,
    ))

    doc.build(story)
    print(f"PDF report saved: {save_path}")
    return save_path
