"""
Farmgate AI - Feature 6 Test Script
Verifies:
1. SQLite mandi_feedback table schema:
   sample_id, timestamp, variety, app_broken_pct, actual_broken_pct, capture_quality_status, notes
2. Feedback entry & update:
   recording actual broken % and lighting/angle notes for a sample
3. Real-time Mean Absolute Error (MAE) calculation:
   MAE = mean(|app_broken_pct - actual_broken_pct|) across all verified samples
4. Comparison table generation:
   sample_id, timestamp, variety, app_broken_pct, actual_broken_pct, absolute_error, capture_quality_status, notes
5. CSV export of the feedback log
6. Strict Zero-Price Guarantee: No price/monetary estimates.
"""

import os
import sqlite3
import pandas as pd
import database as db


def run_feature6_tests():
    print("======================================================================")
    print("FARMGATE AI - FEATURE 6: MANDI FEEDBACK LOOP & MAE VERIFICATION TEST")
    print("======================================================================")

    test_db = "test_farmgate_feedback.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    # Initialize database
    db.init_db(test_db)

    # ------------------------------------------------------------------
    # TEST 1: Schema Verification
    # ------------------------------------------------------------------
    print("\n[TEST 1] Verifying SQLite Schema for mandi_feedback table...")
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(mandi_feedback)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    required_columns = [
        "sample_id",
        "timestamp",
        "variety",
        "app_broken_pct",
        "actual_broken_pct",
        "capture_quality_status",
        "notes",
    ]
    for col in required_columns:
        assert col in columns, f"Column '{col}' missing in mandi_feedback table schema"

    print(f"[PASS] Table mandi_feedback schema verified: {list(columns.keys())}")

    # ------------------------------------------------------------------
    # TEST 2: Save Batches & Verify Feedback Logging
    # ------------------------------------------------------------------
    print("\n[TEST 2] Verifying Batch Saving & Feedback Recording...")
    # Insert Sample 1: Basmati with 5.5% broken
    id1 = db.save_batch(
        sample_id="SMP-TEST-001",
        variety="Basmati",
        total_grains=60,
        whole_pct=91.0,
        broken_pct=5.5,
        discoloured_pct=2.0,
        chalky_pct=1.5,
        agmark_grade="Grade A (Special / Premium)",
        capture_quality_status="Passed",
        notes="Diffuse top-down daylight, 90 deg angle",
        timestamp="2026-10-22 10:00:00",
        db_path=test_db,
    )

    # Insert Sample 2: Sona Masoori with 14.0% broken
    id2 = db.save_batch(
        sample_id="SMP-TEST-002",
        variety="Sona Masoori",
        total_grains=75,
        whole_pct=81.0,
        broken_pct=14.0,
        discoloured_pct=3.0,
        chalky_pct=2.0,
        agmark_grade="Grade B (Standard / Good)",
        capture_quality_status="Passed",
        notes="Even illumination, flat surface",
        timestamp="2026-10-22 11:30:00",
        db_path=test_db,
    )

    # Insert Sample 3: Swarna with 22.0% broken (warning capture)
    id3 = db.save_batch(
        sample_id="SMP-TEST-003",
        variety="Swarna",
        total_grains=50,
        whole_pct=70.0,
        broken_pct=22.0,
        discoloured_pct=4.0,
        chalky_pct=4.0,
        agmark_grade="Grade C (Fair Commercial)",
        capture_quality_status="Warning",
        notes="Slight camera tilt 15 deg",
        timestamp="2026-10-22 13:00:00",
        db_path=test_db,
    )

    # Now record ground truth feedback for Sample 1: Trader actual broken = 6.5% (error = 1.0%)
    db.record_feedback(
        sample_id="SMP-TEST-001",
        actual_broken_pct=6.5,
        notes="Manual tray count verified by Mandi trader",
        db_path=test_db,
    )

    # Record ground truth feedback for Sample 2: Trader actual broken = 11.5% (error = 2.5%)
    db.record_feedback(
        sample_id="SMP-TEST-002",
        actual_broken_pct=11.5,
        notes="Sub-sample 100 grains hand sorted",
        db_path=test_db,
    )

    # Sample 3 remains pending (no actual feedback yet)

    print("[PASS] Batches saved and trader ground-truth feedback recorded successfully.")

    # ------------------------------------------------------------------
    # TEST 3: MAE Calculation Verification
    # ------------------------------------------------------------------
    print("\n[TEST 3] Verifying Mean Absolute Error (MAE) Calculation...")
    mae_stats = db.calculate_feedback_mae(db_path=test_db)
    print(f"MAE Results: {mae_stats}")

    # Sample 1: |5.5 - 6.5| = 1.0%
    # Sample 2: |14.0 - 11.5| = 2.5%
    # Expected MAE = (1.0 + 2.5) / 2 = 1.75%
    expected_mae = 1.75
    assert mae_stats["sample_count"] == 2, f"Expected 2 evaluated samples, got {mae_stats['sample_count']}"
    assert abs(mae_stats["mae"] - expected_mae) < 1e-4, f"Expected MAE {expected_mae}, got {mae_stats['mae']}"
    print(f"[PASS] MAE accurately computed: {mae_stats['mae']}% across {mae_stats['sample_count']} samples (Expected: {expected_mae}%)")

    # Now record feedback for Sample 3: Trader actual = 20.0% (error = |22.0 - 20.0| = 2.0%)
    # New expected MAE = (1.0 + 2.5 + 2.0) / 3 = 5.5 / 3 = 1.83%
    db.record_feedback(
        sample_id="SMP-TEST-003",
        actual_broken_pct=20.0,
        notes="APMC electronic sieving result",
        db_path=test_db,
    )
    mae_stats_3 = db.calculate_feedback_mae(db_path=test_db)
    expected_mae_3 = round((1.0 + 2.5 + 2.0) / 3.0, 2)
    assert abs(mae_stats_3["mae"] - expected_mae_3) < 1e-2, f"Expected MAE {expected_mae_3}, got {mae_stats_3['mae']}"
    print(f"[PASS] Updated MAE after 3rd feedback: {mae_stats_3['mae']}% across {mae_stats_3['sample_count']} samples")

    # ------------------------------------------------------------------
    # TEST 4: Comparison Table DataFrame Verification
    # ------------------------------------------------------------------
    print("\n[TEST 4] Verifying Feedback Comparison Table...")
    comp_df = db.get_feedback_comparison_df(db_path=test_db)
    print("Comparison DataFrame:")
    print(comp_df.to_string())

    assert len(comp_df) == 3, f"Expected 3 rows in comparison df, got {len(comp_df)}"
    assert "absolute_error" in comp_df.columns, "Missing absolute_error column in comparison df"
    assert "capture_quality_status" in comp_df.columns, "Missing capture_quality_status in comparison df"

    # Verify individual absolute errors
    s1_row = comp_df[comp_df["sample_id"] == "SMP-TEST-001"].iloc[0]
    assert abs(s1_row["absolute_error"] - 1.0) < 1e-4, f"Sample 1 error expected 1.0, got {s1_row['absolute_error']}"

    s2_row = comp_df[comp_df["sample_id"] == "SMP-TEST-002"].iloc[0]
    assert abs(s2_row["absolute_error"] - 2.5) < 1e-4, f"Sample 2 error expected 2.5, got {s2_row['absolute_error']}"

    print("[PASS] Feedback comparison table contains exact absolute errors and capture quality statuses.")

    # ------------------------------------------------------------------
    # TEST 5: CSV Export Verification
    # ------------------------------------------------------------------
    print("\n[TEST 5] Verifying CSV Export...")
    test_csv = "test_mandi_feedback_export.csv"
    if os.path.exists(test_csv):
        os.remove(test_csv)

    exported_path = db.export_feedback_csv(test_csv, db_path=test_db)
    assert os.path.exists(exported_path), "CSV file was not created"

    read_df = pd.read_csv(exported_path)
    assert len(read_df) == 3, f"Expected 3 rows in exported CSV, got {len(read_df)}"
    print(f"[PASS] Mandi feedback CSV exported successfully: {exported_path} ({len(read_df)} rows)")

    # ------------------------------------------------------------------
    # TEST 6: Zero Price / Monetary Guarantee
    # ------------------------------------------------------------------
    print("\n[TEST 6] Verifying Zero Price / Monetary Guarantee...")
    price_keywords = ["rs.", "inr", "rupees", "/quintal", "price", "mandi_price", "rate"]
    for col in comp_df.columns:
        assert col.lower() not in ["price", "mandi_price", "price_offered"], f"Forbidden price column in feedback: {col}"
    for kw in ["rs.", "rupees", "/quintal"]:
        assert kw not in read_df.to_string().lower(), f"Forbidden price keyword '{kw}' found in CSV"
    print("[PASS] Strict Zero-Price Guarantee verified: No price or monetary predictions exist in feedback loop.")

    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists(test_csv):
        os.remove(test_csv)

    print("\n======================================================================")
    print("ALL FEATURE 6 TESTS COMPLETED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    run_feature6_tests()
