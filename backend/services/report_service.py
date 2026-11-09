"""PDF report generation (ReportLab). Contains no patient identifiers or original filenames."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ml import config


def generate_report(result: dict, original_png: str, gradcam_figure_png: str, report_id: str | None = None) -> str:
    """Build reports/<id>.pdf from a real prediction result and saved images. Returns the PDF path."""
    config.ensure_dirs()
    report_id = report_id or uuid.uuid4().hex[:12]
    path = config.REPORTS_DIR / f"neurovision_report_{report_id}.pdf"
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=22, textColor=colors.HexColor("#123c4f"), spaceAfter=2)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=11, textColor=colors.HexColor("#4a6572"), alignment=1)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#123c4f"), spaceBefore=10)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=14)
    warn = ParagraphStyle("warn", parent=body, textColor=colors.HexColor("#7a3b00"),
                          backColor=colors.HexColor("#fff4e5"), borderPadding=8, leading=14)
    sev = result.get("severity") if result.get("severity_available") and result.get("severity") else "Not Available"
    story = [Paragraph("NEUROVISION AI", h1), Paragraph("Retinal Image Analysis Report", sub), Spacer(1, 14)]
    story.append(Paragraph("Input Image", h2))
    story.append(Image(original_png, width=6 * cm, height=6 * cm))
    story.append(Paragraph("AI-assisted screening prediction", h2))
    rows = [["Prediction", result["prediction"]],
            ["Confidence", f"{result['confidence'] * 100:.2f}%"],
            ["Estimated P(ASD-associated class)", f"{result['probability_asd'] * 100:.2f}%"],
            ["Severity", sev],
            ["Model", result["model"]],
            ["Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")],
            ["Report ID", report_id]]
    t = Table(rows, colWidths=[6 * cm, 10 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6dc")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4f7")),
                           ("FONTSIZE", (0, 0), (-1, -1), 10), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t, Spacer(1, 6)]
    if result.get("severity_note"):
        story.append(Paragraph(result["severity_note"], body))
    story.append(Paragraph(result["prediction_statement"], body))
    story.append(Paragraph("Grad-CAM (CNN component)", h2))
    story.append(Image(gradcam_figure_png, width=16 * cm, height=5.8 * cm))
    story.append(Paragraph("Grad-CAM highlights image regions that influenced the CNN component's output. It does not "
                           "demonstrate that these regions are clinically established ASD biomarkers.", body))
    if result.get("quality_warnings"):
        story.append(Paragraph("Image quality notes", h2))
        story.append(Paragraph("; ".join(result["quality_warnings"]), body))
    story += [Spacer(1, 12), Paragraph("Disclaimer", h2), Paragraph(config.MEDICAL_DISCLAIMER, warn)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="NEUROVISION AI Report", author="NEUROVISION AI").build(story)
    return str(path)
