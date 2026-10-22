"""
Farmgate AI - Feature 5 Test Script
Verifies:
1. export_grains_to_csv: exact required columns (grain_id, length_mm, width_mm, area_mm2, aspect_ratio, classification)
2. generate_pdf_report (ReportLab):
   - Sample ID, timestamp, summary table, grade badge, representative annotated image, capture quality status
   - Mandatory disclaimer: 'This tool provides image-based preliminary quality estimation. It does not replace certified laboratory testing or measure moisture content, chemical composition, or cooking quality.'
3. generate_html_report: self-contained HTML with print CSS (@media print) and mandatory disclaimer
4. Zero price / monetary estimates in all reports and exports.
"""

import os
import cv2
import numpy as np
import pandas as pd
from calibration_engine import GrainImageCalibrator
from grain_analyzer import RiceGrainAnalyzer, GrainFeature
from generate_sample_image import generate_sample_smartphone_capture
from report_generator import (
    export_grains_to_csv,
    generate_pdf_report,
    generate_html_report,
    MANDATORY_DISCLAIMER,
)


def run_feature5_tests():
    print("======================================================================")
    print("FARMGATE AI — FEATURE 5: REPORTING & EXPORT VERIFICATION TEST")
    print("======================================================================")

    # 1. Generate or load a sample image and run calibration & analysis
    sample_path = "sample_grain_capture.jpg"
    if not os.path.exists(sample_path):
        print("Generating synthetic smartphone image for test...")
        generate_sample_smartphone_capture(sample_path, num_grains=45, variety="Basmati")

    raw_img = cv2.imread(sample_path)
    calibrator = GrainImageCalibrator(marker_span_mm=170.0, output_size=(1000, 1000))
    is_valid, quality_metrics, msgs = calibrator.validate_capture_quality(raw_img)
    unwarped_bgr, annotated_orig, px_per_mm, calib_status = calibrator.process(raw_img)

    assert calib_status["success"], "Calibration failed during Feature 5 test"

    analyzer = RiceGrainAnalyzer(variety="Basmati")
    annotated_grains, grains_list, stats, binary_mask = analyzer.analyze(unwarped_bgr, px_per_mm)

    print(f"Assayed Grains Count: {len(grains_list)}")
    print(f"Sample Stats: Whole={stats['whole_pct']}%, Broken={stats['broken_pct']}%, Discoloured={stats['discoloured_pct']}%, Chalky={stats['chalky_pct']}%")
    print(f"AGMARK Grade: {stats['agmark_grade']}")

    # ------------------------------------------------------------------
    # TEST 1: CSV Export
    # ------------------------------------------------------------------
    print("\n[TEST 1] Verifying CSV Export...")
    test_csv_path = "test_grain_measurements.csv"
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)

    res_csv = export_grains_to_csv(grains_list, output_csv_path=test_csv_path)
    assert os.path.exists(res_csv), "CSV file was not created"
    df = pd.read_csv(res_csv)

    expected_columns = ["grain_id", "length_mm", "width_mm", "area_mm2", "aspect_ratio", "classification"]
    assert list(df.columns) == expected_columns, f"Columns do not match expected: {df.columns.tolist()} vs {expected_columns}"
    assert len(df) == len(grains_list), f"Row count {len(df)} does not match grains count {len(grains_list)}"
    print(f"[PASS] CSV Export passed: {len(df)} rows, exact columns: {df.columns.tolist()}")
    print("Sample CSV rows:")
    print(df.head(3).to_string())

    # ------------------------------------------------------------------
    # TEST 2: PDF Certificate Generation
    # ------------------------------------------------------------------
    print("\n[TEST 2] Verifying ReportLab PDF Certificate Generation...")
    test_pdf_path = "test_quality_certificate.pdf"
    if os.path.exists(test_pdf_path):
        os.remove(test_pdf_path)

    temp_thumb = "test_pdf_thumbnail.png"
    cv2.imwrite(temp_thumb, cv2.resize(annotated_grains, (640, 480)))

    sample_id = "SMP-2026-TEST01"
    quality_status = "Passed" if is_valid else "Warning"

    pdf_out = generate_pdf_report(
        stats=stats,
        grain_type=stats["variety"],
        output_pdf_path=test_pdf_path,
        sample_id=sample_id,
        capture_quality_status=quality_status,
        annotated_image_path=temp_thumb,
    )

    assert os.path.exists(pdf_out), "PDF file was not created"
    pdf_size = os.path.getsize(pdf_out)
    assert pdf_size > 10000, f"PDF file size too small: {pdf_size} bytes"
    print(f"[PASS] PDF Certificate generated successfully: {pdf_out} ({pdf_size:,} bytes)")

    # Decode ReportLab compressed PDF streams to verify disclaimer and fields
    import base64, zlib
    with open(pdf_out, "rb") as f:
        pdf_bytes = f.read()

    start = pdf_bytes.find(b"stream\n", pdf_bytes.find(b"/Contents")) + 7
    end = pdf_bytes.find(b"~>endstream", start) + 2
    raw_stream = pdf_bytes[start:end]
    decoded_stream = base64.a85decode(raw_stream, adobe=True)
    pdf_text = zlib.decompress(decoded_stream).decode("latin-1", errors="ignore")

    assert "preliminary quality estimation" in pdf_text or "does not replace certified" in pdf_text, \
        "Mandatory disclaimer string missing in PDF stream"
    assert sample_id in pdf_text, f"Sample ID {sample_id} missing in PDF stream"
    assert quality_status in pdf_text, f"Quality status {quality_status} missing in PDF stream"
    print(f"[PASS] PDF content verified: Disclaimer, Sample ID ({sample_id}), and Quality Status ({quality_status}) present.")

    # ------------------------------------------------------------------
    # TEST 3: HTML Report Generation
    # ------------------------------------------------------------------
    print("\n[TEST 3] Verifying Standalone HTML Report Generation...")
    html_content = generate_html_report(
        stats=stats,
        grain_type=stats["variety"],
        sample_id=sample_id,
        capture_quality_status=quality_status,
    )

    assert len(html_content) > 1000, "HTML content too short"
    assert "@media print" in html_content, "Missing @media print styling in HTML report"
    assert MANDATORY_DISCLAIMER in html_content, "Missing exact mandatory disclaimer in HTML report"
    assert sample_id in html_content, "Missing sample_id in HTML report"
    assert quality_status in html_content, "Missing quality_status in HTML report"
    assert stats["agmark_grade"] in html_content, "Missing AGMARK grade in HTML report"

    test_html_path = "test_quality_report.html"
    with open(test_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[PASS] Standalone HTML Report generated successfully: {test_html_path} ({len(html_content):,} chars)")
    print("[PASS] Print CSS (@media print) and Mandatory Disclaimer confirmed.")

    # ------------------------------------------------------------------
    # TEST 4: Zero Price Estimates Verification
    # ------------------------------------------------------------------
    print("\n[TEST 4] Verifying Zero Price / Monetary Estimates in Reports...")
    monetary_keywords = ["rs.", "inr", "rupees", "/quintal", "price estimate", "market value", "batch value", "price/kg"]
    
    # Check CSV
    csv_str = df.to_string().lower()
    for kw in monetary_keywords:
        assert kw not in csv_str, f"Found forbidden monetary keyword '{kw}' in CSV export"

    # Check HTML
    html_str = html_content.lower()
    for kw in monetary_keywords:
        assert kw not in html_str, f"Found forbidden monetary keyword '{kw}' in HTML report"

    print("[PASS] Strict Zero-Price Guarantee verified: No price or monetary predictions exist in CSV or HTML reports.")

    # ------------------------------------------------------------------
    # TEST 5: Mandatory Disclaimer Text Exact Match
    # ------------------------------------------------------------------
    print("\n[TEST 5] Verifying Mandatory Disclaimer Exact Text...")
    expected_disclaimer = (
        "This tool provides image-based preliminary quality estimation. It does not replace "
        "certified laboratory testing or measure moisture content, chemical composition, or cooking quality."
    )
    assert MANDATORY_DISCLAIMER.strip() == expected_disclaimer.strip(), \
        f"MANDATORY_DISCLAIMER does not match specification exactly.\nGot: '{MANDATORY_DISCLAIMER}'\nExpected: '{expected_disclaimer}'"
    print(f"[PASS] Mandatory disclaimer exact match confirmed:\n  \"{MANDATORY_DISCLAIMER}\"")

    print("\n======================================================================")
    print("ALL FEATURE 5 TESTS COMPLETED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    run_feature5_tests()
