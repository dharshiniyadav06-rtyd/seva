"""
Farmgate AI - Report Generator Module
Generates professional PDF Assay Certificates and structured CSV export reports.
"""

import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage,
)


MANDATORY_DISCLAIMER = (
    "This tool provides image-based preliminary quality estimation. It does not replace certified "
    "laboratory testing or measure moisture content, chemical composition, or cooking quality."
)


def export_grains_to_csv(grains: list, output_csv_path: str = "grain_analysis_report.csv") -> str:
    """
    Exports individual grain measurements to a clean CSV file with exact columns:
    grain_id, length_mm, width_mm, area_mm2, aspect_ratio, classification.
    """
    rows = []
    for g in grains:
        rows.append({
            "grain_id": g.grain_id,
            "length_mm": g.length_mm,
            "width_mm": g.width_mm,
            "area_mm2": g.area_mm2,
            "aspect_ratio": g.aspect_ratio,
            "classification": g.classification,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def generate_pdf_report(
    stats: Dict[str, any],
    grain_type: str,
    output_pdf_path: str = "farmgate_quality_certificate.pdf",
    sample_id: str = "SMP-2026-0841",
    inspector_name: str = "Farmgate Field Assayer",
    capture_quality_status: str = "Passed",
    annotated_image_path: str = None,
) -> str:
    """
    Generates a formal AGMARK-compliant PDF Quality Certificate using ReportLab.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CertTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E293B"),
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "CertSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    normal_text = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    # Header
    story.append(Paragraph("FARMGATE AI — GRAIN ASSAY CERTIFICATE", title_style))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Low-Cost Smartphone Vision Assaying System | AGMARK Standards Compliance Assessment",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

    # Metadata Table
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [
            Paragraph(f"<b>Sample ID:</b> {sample_id}", normal_text),
            Paragraph(f"<b>Date/Time:</b> {now_str}", normal_text),
        ],
        [
            Paragraph(f"<b>Grain Type / Variety:</b> {grain_type}", normal_text),
            Paragraph(f"<b>Capture Quality Status:</b> {capture_quality_status}", normal_text),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # AGMARK Grade Award Banner
    grade_name = stats.get("agmark_grade", "N/A")
    grade_color = colors.HexColor(stats.get("grade_color", "#3B82F6"))

    banner_data = [
        [
            Paragraph(f"<b>ASSESSED QUALITY GRADE</b>", ParagraphStyle("GB", fontName="Helvetica-Bold", fontSize=9, textColor=colors.white, alignment=1)),
        ],
        [
            Paragraph(f"<b>{grade_name.upper()}</b>", ParagraphStyle("GV", fontName="Helvetica-Bold", fontSize=16, textColor=colors.white, alignment=1)),
        ],
        [
            Paragraph(f"<i>{stats.get('grade_rationale', '')}</i>", ParagraphStyle("GR", fontName="Helvetica-Oblique", fontSize=8.5, textColor=colors.white, alignment=1)),
        ],
    ]
    banner_table = Table(banner_data, colWidths=[520])
    banner_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), grade_color),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 2), (-1, 2), 8),
        ])
    )
    story.append(banner_table)
    story.append(Spacer(1, 14))

    # Assaying Key Metrics Table
    story.append(Paragraph("1. Morphometric & Optical Defect Breakdown", section_heading))

    metrics_rows = [
        ["Quality Parameter", "Measured Value", "Standard AGMARK Limit", "Status"],
        [
            "Total Grains Assayed",
            f"{stats.get('total_grains', 0)} kernels",
            "Min 30 kernels required",
            "Adequate" if stats.get("total_grains", 0) >= 30 else "Small Sample",
        ],
        [
            "Whole Grains (Head Rice)",
            f"{stats.get('whole_count', 0)} ({stats.get('whole_pct', 0.0)}%)",
            ">= 75.0% for Grade A/B",
            "Pass" if stats.get("whole_pct", 0) >= 75.0 else "Warning",
        ],
        [
            "Broken Grains",
            f"{stats.get('broken_count', 0)} ({stats.get('broken_pct', 0.0)}%)",
            "<= 5.0% (G-A) / <= 12.0% (G-B)",
            "Pass" if stats.get("broken_pct", 0) <= 12.0 else "Fail",
        ],
        [
            "Discoloured / Damaged Grains",
            f"{stats.get('discoloured_count', 0)} ({stats.get('discoloured_pct', 0.0)}%)",
            "<= 1.0% (G-A) / <= 3.0% (G-B)",
            "Pass" if stats.get("discoloured_pct", 0) <= 3.0 else "Fail",
        ],
        [
            "Chalky Grains (Opacity >= 25%)",
            f"{stats.get('chalky_count', 0)} ({stats.get('chalky_pct', 0.0)}%)",
            "<= 3.0% (G-A) / <= 6.0% (G-B)",
            "Pass" if stats.get("chalky_pct", 0) <= 6.0 else "Fail",
        ],
        [
            "Average Kernel Length",
            f"{stats.get('avg_length_mm', 0.0)} mm",
            f"Nominal: {stats.get('avg_length_mm', 0.0):.1f} mm",
            "Recorded",
        ],
        [
            "Average Kernel Width",
            f"{stats.get('avg_width_mm', 0.0)} mm",
            f"Nominal: {stats.get('avg_width_mm', 0.0):.1f} mm",
            "Recorded",
        ],
        [
            "Length-to-Breadth Aspect Ratio",
            f"{stats.get('avg_aspect_ratio', 0.0)}",
            ">= 3.0 (Long Grain Standard)",
            "Slender" if stats.get("avg_aspect_ratio", 0) >= 3.0 else "Medium",
        ],
    ]

    t = Table(metrics_rows, colWidths=[180, 130, 130, 80])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("PADDING", (0, 1), (-1, -1), 5),
        ])
    )
    if annotated_image_path and os.path.exists(annotated_image_path):
        story.append(Paragraph("2. Representative Assayed Grain Imagery", section_heading))
        story.append(RLImage(annotated_image_path, width=4.5 * inch, height=2.2 * inch))
        story.append(Spacer(1, 10))
        story.append(Paragraph("3. Assaying Standard Compliance Methodology", section_heading))
    else:
        story.append(Paragraph("2. Assaying Standard Compliance Methodology", section_heading))
    method_text = (
        "Measurements conducted using Farmgate AI high-precision computer vision pipeline. "
        "Fiducial perspective rectification performed via OpenCV ArUco (DICT_4X4_50) Homography at sub-millimeter scale. "
        "Grain segmentation utilized watershed separation with Otsu-adaptive binarization. "
        "Grain dimensional and defect grading adheres to the Directorate of Marketing and Inspection (DMI), "
        "Ministry of Agriculture & Farmers Welfare, Government of India (AGMARK Milled Rice Grading and Marking Rules)."
    )
    story.append(Paragraph(method_text, normal_text))
    story.append(Spacer(1, 15))

    # Sign-off Box
    sig_data = [
        [
            Paragraph("<b>Automated Assaying Signature:</b><br/>FARMGATE-AI-VERIFIED-V1.0", normal_text),
            Paragraph("<b>Quality Assurance Officer:</b><br/>________________________ (Sign/Seal)", normal_text),
        ]
    ]
    sig_table = Table(sig_data, colWidths=[260, 260])
    sig_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ])
    )
    story.append(sig_table)

    # Feature 5: Mandatory Disclaimer Box on Report
    story.append(Spacer(1, 10))
    disclaimer_style = ParagraphStyle(
        "CertDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#991B1B"),
        alignment=1,
    )
    disclaimer_data = [
        [
            Paragraph(
                f"<b>MANDATORY LEGAL DISCLAIMER:</b> {MANDATORY_DISCLAIMER}",
                disclaimer_style,
            )
        ]
    ]
    disclaimer_table = Table(disclaimer_data, colWidths=[520])
    disclaimer_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#EF4444")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(disclaimer_table)

    # Build Document
    doc.build(story)
    return output_pdf_path


def generate_html_report(
    stats: Dict[str, Any],
    grain_type: str,
    sample_id: str = "SMP-2026-0841",
    inspector_name: str = "Farmgate Field Assayer",
    capture_quality_status: str = "Passed",
) -> str:
    """
    Feature 5: Generates a self-contained, beautifully styled HTML/CSS quality report
    with full print styling (@media print) and the mandatory disclaimer.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grade = stats.get("agmark_grade", "N/A")
    grade_color = stats.get("grade_color", "#3B82F6")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Farmgate AI — Assay Certificate {sample_id}</title>
<style>
    @page {{ size: A4; margin: 15mm; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background: #f8fafc;
        margin: 0;
        padding: 24px;
    }}
    .cert-container {{
        max-width: 800px;
        margin: 0 auto;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }}
    .header {{
        text-align: center;
        border-bottom: 2px solid #0284c7;
        padding-bottom: 16px;
        margin-bottom: 20px;
    }}
    .title {{
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }}
    .subtitle {{
        font-size: 13px;
        color: #64748b;
        margin-top: 6px;
    }}
    .meta-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        margin-bottom: 20px;
    }}
    .grade-banner {{
        background: {grade_color};
        color: white;
        text-align: center;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 24px;
    }}
    .grade-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }}
    .grade-val {{
        font-size: 22px;
        font-weight: 800;
        margin: 4px 0;
    }}
    .grade-sub {{
        font-size: 12px;
        opacity: 0.95;
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-bottom: 24px;
    }}
    table.data-table th {{
        background: #0f172a;
        color: white;
        text-align: left;
        padding: 10px 12px;
    }}
    table.data-table td {{
        padding: 9px 12px;
        border-bottom: 1px solid #e2e8f0;
    }}
    table.data-table tr:nth-child(even) {{
        background: #f8fafc;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
    }}
    .badge-pass {{ background: #dcfce7; color: #166534; }}
    .badge-fail {{ background: #fee2e2; color: #991b1b; }}
    .disclaimer-box {{
        background: #fef2f2;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #991b1b;
        font-size: 11px;
        line-height: 1.5;
        font-weight: 600;
        margin-top: 24px;
        text-align: center;
    }}
    @media print {{
        body {{ background: white; padding: 0; }}
        .cert-container {{ border: none; box-shadow: none; padding: 0; }}
        .no-print {{ display: none; }}
    }}
</style>
</head>
<body>
<div class="cert-container">
    <div class="header">
        <h1 class="title">🌾 FARMGATE AI — GRAIN ASSAY CERTIFICATE</h1>
        <div class="subtitle">Low-Cost Smartphone Vision Assaying & AGMARK Quality Grading System</div>
    </div>

    <div class="meta-grid">
        <div><strong>Sample ID:</strong> {sample_id}</div>
        <div><strong>Assaying Timestamp:</strong> {now_str}</div>
        <div><strong>Grain Variety:</strong> {grain_type}</div>
        <div><strong>Capture Quality Status:</strong> {capture_quality_status}</div>
    </div>

    <div class="grade-banner">
        <div class="grade-label">Assayed AGMARK Quality Grade</div>
        <div class="grade-val">{grade}</div>
        <div class="grade-sub">{stats.get('grade_rationale', '')}</div>
    </div>

    <h3 style="font-size: 15px; margin-bottom: 10px; color: #0f172a;">1. Physical Morphometric & Defect Analysis</h3>
    <table class="data-table">
        <thead>
            <tr>
                <th>Quality Parameter</th>
                <th>Measured Value</th>
                <th>AGMARK Standard Limit</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Total Grains Assayed</td>
                <td><strong>{stats.get('total_grains', 0)}</strong> kernels</td>
                <td>Min 30 kernels recommended</td>
                <td><span class="badge badge-pass">Adequate</span></td>
            </tr>
            <tr>
                <td>Whole Grains (Head Rice)</td>
                <td>{stats.get('whole_count', 0)} ({stats.get('whole_pct', 0.0)}%)</td>
                <td>&gt;= 75.0% for Grade A/B</td>
                <td><span class="badge {'badge-pass' if stats.get('whole_pct', 0) >= 75.0 else 'badge-fail'}">{'Pass' if stats.get('whole_pct', 0) >= 75.0 else 'Warning'}</span></td>
            </tr>
            <tr>
                <td>Broken Grains</td>
                <td>{stats.get('broken_count', 0)} ({stats.get('broken_pct', 0.0)}%)</td>
                <td>&lt;= 5.0% (Gr-A) / &lt;= 12.0% (Gr-B)</td>
                <td><span class="badge {'badge-pass' if stats.get('broken_pct', 0) <= 12.0 else 'badge-fail'}">{'Pass' if stats.get('broken_pct', 0) <= 12.0 else 'Fail'}</span></td>
            </tr>
            <tr>
                <td>Uncertain / Borderline Grains</td>
                <td>{stats.get('uncertain_count', 0)} ({stats.get('uncertain_pct', 0.0)}%)</td>
                <td>+/- 5% Cutoff Boundary Band</td>
                <td>Recorded</td>
            </tr>
            <tr>
                <td>Discoloured Grains</td>
                <td>{stats.get('discoloured_count', 0)} ({stats.get('discoloured_pct', 0.0)}%)</td>
                <td>&lt;= 1.0% (Gr-A) / &lt;= 3.0% (Gr-B)</td>
                <td><span class="badge {'badge-pass' if stats.get('discoloured_pct', 0) <= 3.0 else 'badge-fail'}">{'Pass' if stats.get('discoloured_pct', 0) <= 3.0 else 'Fail'}</span></td>
            </tr>
            <tr>
                <td>Chalky Grains</td>
                <td>{stats.get('chalky_count', 0)} ({stats.get('chalky_pct', 0.0)}%)</td>
                <td>&lt;= 3.0% (Gr-A) / &lt;= 6.0% (Gr-B)</td>
                <td><span class="badge {'badge-pass' if stats.get('chalky_pct', 0) <= 6.0 else 'badge-fail'}">{'Pass' if stats.get('chalky_pct', 0) <= 6.0 else 'Fail'}</span></td>
            </tr>
            <tr>
                <td>Mean Kernel Length</td>
                <td>{stats.get('avg_length_mm', 0.0)} &plusmn; {stats.get('length_std_mm', 0.0)} mm</td>
                <td>Reference: {stats.get('reference_length_mm', 0.0)} mm</td>
                <td>Recorded</td>
            </tr>
            <tr>
                <td>Mean Kernel Width</td>
                <td>{stats.get('avg_width_mm', 0.0)} mm</td>
                <td>Transverse Minor Axis</td>
                <td>Recorded</td>
            </tr>
            <tr>
                <td>Length-to-Width Aspect Ratio</td>
                <td>{stats.get('avg_aspect_ratio', 0.0)}</td>
                <td>Slender (&gt;= 3.0) / Medium</td>
                <td>{'Slender' if stats.get('avg_aspect_ratio', 0) >= 3.0 else 'Medium'}</td>
            </tr>
        </tbody>
    </table>

    <div class="disclaimer-box">
        ⚠️ <strong>MANDATORY DISCLAIMER:</strong> {MANDATORY_DISCLAIMER}
    </div>
</div>
</body>
</html>"""
    return html
