"""
Farmgate AI - Report Generator Module
Generates professional PDF Assay Certificates and structured CSV export reports.
"""

import os
from datetime import datetime
from typing import List, Dict
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
)


def export_grains_to_csv(grains: list, output_csv_path: str = "grain_analysis_report.csv") -> str:
    """Exports individual grain measurements to a clean CSV file."""
    rows = []
    for g in grains:
        rows.append({
            "Grain ID": g.grain_id,
            "Classification": g.classification,
            "Length (mm)": g.length_mm,
            "Width (mm)": g.width_mm,
            "Aspect Ratio": g.aspect_ratio,
            "Area (mm2)": g.area_mm2,
            "Chalky Ratio": g.chalky_ratio,
            "Is Broken": g.is_broken,
            "Is Discoloured": g.is_discoloured,
            "Is Chalky": g.is_chalky,
            "Mean R": g.mean_r,
            "Mean G": g.mean_g,
            "Mean B": g.mean_b,
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
            Paragraph(f"<b>Grain Type:</b> {grain_type}", normal_text),
            Paragraph(f"<b>Inspector:</b> {inspector_name}", normal_text),
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
    story.append(t)
    story.append(Spacer(1, 14))

    # Notes & Standards Reference
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

    # Build Document
    doc.build(story)
    return output_pdf_path
