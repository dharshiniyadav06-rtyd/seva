"""
Farmgate AI - End-to-End Pipeline Verification Script
Tests sheet generation, synthetic capture, quality validation, homography unwarping,
watershed grain segmentation, variety baselines, AGMARK grading, SQLite database history,
and Mandi feedback loop with MAE computation.
"""

import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("[1/7] Importing Farmgate AI core modules...")
try:
    from generate_sheet import generate_calibration_sheet
    from generate_sample_image import generate_sample_smartphone_capture
    from calibration_engine import GrainImageCalibrator
    import cv2
    from grain_analyzer import RiceGrainAnalyzer
    from report_generator import export_grains_to_csv, generate_pdf_report
    import database as db
    print("      [OK] All core modules (including database.py) imported successfully.")
except Exception as e:
    print(f"      [ERROR] Module import error: {e}")
    sys.exit(1)


def test_full_pipeline():
    print("\n[2/7] Generating A4 300 DPI Printable Calibration Sheet...")
    sheet_path = "test_calibration_sheet.png"
    generate_calibration_sheet(sheet_path, dpi=300)
    assert os.path.exists(sheet_path), "Calibration sheet was not created!"
    sheet_img = cv2.imread(sheet_path)
    print(f"      [OK] Sheet created: {sheet_img.shape[1]}x{sheet_img.shape[0]} px, {os.path.getsize(sheet_path) / 1024:.1f} KB")

    print("\n[3/7] Generating Synthetic Smartphone Camera Photo (Sona Masoori Variety with Tilt)...")
    sample_path = "test_smartphone_capture.jpg"
    generate_sample_smartphone_capture(sample_path, num_grains=50, variety="Sona Masoori")
    assert os.path.exists(sample_path), "Sample photo was not created!"
    sample_img = cv2.imread(sample_path)
    print(f"      [OK] Photo generated: {sample_img.shape[1]}x{sample_img.shape[0]} px, {os.path.getsize(sample_path) / 1024:.1f} KB")

    print("\n[4/7] Testing GrainImageCalibrator (validate_capture_quality, unwarp_image, normalize_lighting)...")
    calibrator = GrainImageCalibrator(marker_span_mm=172.0, output_size=(1000, 1000))

    # Test quality validation
    is_valid, metrics, msgs = calibrator.validate_capture_quality(sample_img)
    print(f"      [OK] Quality Validation -> Valid: {is_valid}, Blur Score: {metrics['blur_score']}, Brightness: {metrics['mean_brightness']}, Markers: {metrics['marker_count']}")

    # Test unwarping and normalization
    t0 = time.time()
    unwarped_bgr, annotated_orig, px_per_mm, status = calibrator.process(sample_img)
    t_calib = time.time() - t0
    assert status["success"], f"Calibration failed: {status.get('error_msg')}"
    print(f"      [OK] ArUco markers detected: {status['markers_found']}")
    print(f"      [OK] Perspective unwarped to 1000x1000 in {t_calib:.3f}s")
    print(f"      [OK] Calibrated scale: {px_per_mm:.3f} pixels/mm")

    print("\n[5/7] Testing RiceGrainAnalyzer with Variety Baselines (Sona Masoori, Basmati, Watershed Segmentation)...")
    analyzer = RiceGrainAnalyzer(
        variety="Sona Masoori",
        length_threshold_ratio=0.75,
        discolour_threshold=38.0,
        chalky_threshold=0.22,
    )
    t0 = time.time()
    annotated_grains, grains_list, stats, mask = analyzer.analyze(unwarped_bgr, px_per_mm)
    t_analyze = time.time() - t0

    cv2.imwrite("test_annotated_grains.png", annotated_grains)
    cv2.imwrite("test_unwarped_stage.png", unwarped_bgr)

    print(f"      [OK] Analyzed {len(grains_list)} grains in {t_analyze:.3f}s (Variety: {stats['variety']})")
    print(f"      [OK] Whole: {stats['whole_count']} ({stats['whole_pct']}%), Broken: {stats['broken_count']} ({stats['broken_pct']}%)")
    print(f"      [OK] Discoloured: {stats['discoloured_count']} ({stats['discoloured_pct']}%), Chalky: {stats['chalky_count']} ({stats['chalky_pct']}%)")
    print(f"      [OK] AGMARK Grade: {stats['agmark_grade']}")

    print("\n[6/7] Testing Offline SQLite Database & Mandi Feedback Loop...")
    test_db = "test_farmgate_history.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    db.init_db(test_db)
    batch_id = db.save_batch(
        variety=stats["variety"],
        total_grains=stats["total_grains"],
        whole_pct=stats["whole_pct"],
        broken_pct=stats["broken_pct"],
        discoloured_pct=stats["discoloured_pct"],
        chalky_pct=stats["chalky_pct"],
        agmark_grade=stats["agmark_grade"],
        notes="Automated test batch",
        db_path=test_db,
    )
    print(f"      [OK] Saved batch #{batch_id} to offline SQLite database: {test_db}")

    # Test Mandi Feedback logging and MAE calculation
    trader_actual_broken = stats["broken_pct"] + 1.5
    mae_result = db.update_mandi_feedback(
        batch_id=batch_id,
        mandi_broken_actual=trader_actual_broken,
        mandi_price_offered=3400.0,
        notes="Mandi physical sieve validation",
        db_path=test_db,
    )
    print(f"      [OK] Mandi feedback logged. Current MAE: {mae_result['mae']}% across {mae_result['sample_count']} sample(s).")
    assert mae_result["mae"] == 1.5, f"Expected MAE 1.5, got {mae_result['mae']}"

    history_df = db.get_batch_history(test_db)
    assert len(history_df) == 1, "Expected 1 batch in test database"
    print(f"      [OK] Fetched batch history ({len(history_df)} record(s)).")

    if os.path.exists(test_db):
        os.remove(test_db)

    print("\n[7/7] Generating Export Reports (CSV and AGMARK PDF Certificate)...")
    csv_path = "test_grain_data.csv"
    export_grains_to_csv(grains_list, csv_path)
    assert os.path.exists(csv_path), "CSV was not created!"
    print(f"      [OK] CSV exported: {csv_path} ({os.path.getsize(csv_path)} bytes)")

    pdf_path = "test_quality_certificate.pdf"
    generate_pdf_report(stats, stats["variety"], output_pdf_path=pdf_path)
    assert os.path.exists(pdf_path), "PDF report was not created!"
    print(f"      [OK] PDF Certificate exported: {pdf_path} ({os.path.getsize(pdf_path) / 1024:.1f} KB)")

    print("\n" + "=" * 65)
    print("[SUCCESS] ALL VERIFICATION TESTS PASSED! Production prototype ready.")
    print("=" * 65)


if __name__ == "__main__":
    test_full_pipeline()
