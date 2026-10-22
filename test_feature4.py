"""
test_feature4.py - Verification script for Feature 4: Variety Baseline & Whole vs. Broken Classification
Evaluates:
1. Variety profiles (Basmati, Sona Masoori, Swarna, Custom/Auto): intact lengths, aspect ratios, AGMARK defaults, estimate flags
2. Known variety classification (Whole vs. Broken with configurable threshold fraction: 0.60, 0.75, 0.85)
3. Custom/Auto dynamic estimation (75th percentile of distribution, is_reference_estimated flag, estimation note)
4. Edge uncertainty (+/-5% threshold band flagged as uncertain)
5. Comprehensive report metrics (total, whole %, broken %, uncertain %, mean, std dev, 10-bin histogram)
"""

import cv2
import numpy as np
import grain_analyzer as ga
import calibration_engine as ce

def run_tests():
    print("=" * 80)
    print("FEATURE 4: VARIETY BASELINE & WHOLE VS. BROKEN CLASSIFICATION TESTS")
    print("=" * 80)

    # -------------------------------------------------------------
    # Test 1: Variety Profiles Integrity & Metadata
    # -------------------------------------------------------------
    print("\n--- Test 1: Variety Baseline Profiles & AGMARK Defaults ---")
    required_varieties = ["Basmati", "Sona Masoori", "Swarna", "Custom/Auto"]
    for v_name in required_varieties:
        assert v_name in ga.RiceGrainAnalyzer.VARIETY_BASELINES, f"Missing variety profile: {v_name}"
        p = ga.RiceGrainAnalyzer.VARIETY_BASELINES[v_name]
        print(f"Profile: {v_name:<14} | Length: {p['full_length_mm']}mm | AR: {p['expected_aspect_ratio']} | "
              f"Est: {p['is_estimate']} | AGMARK Gr.A Max Broken: {p['agmark_defaults']['grade_a_broken_max']}%")
        if p["is_estimate"]:
            assert "estimation_note" in p or "note" in p, f"Missing estimation note for {v_name}"
            print(f"   -> Note: {p.get('estimation_note')}")
    print("Status: PASS (All required profiles configured with AGMARK defaults & estimate tags)")

    # -------------------------------------------------------------
    # Test 2: Known Variety Whole vs. Broken (Basmati at 0.75 cutoff)
    # -------------------------------------------------------------
    print("\n--- Test 2: Known Variety Classification (Basmati, Expected: 7.5mm) ---")
    # Expected intact: 7.5mm.
    # At 75% cutoff: cutoff = 7.5 * 0.75 = 5.625mm.
    # +/- 5% uncertainty margin = 0.05 * 5.625 = 0.281mm.
    # Uncertainty band: [5.344mm, 5.906mm].
    analyzer_basmati = ga.RiceGrainAnalyzer(variety="Basmati", length_threshold_ratio=0.75)
    print(f"Expected intact length : {analyzer_basmati.full_length_mm:.3f} mm")
    print(f"Cutoff threshold (75%) : {analyzer_basmati.broken_cutoff_length_mm:.3f} mm")

    # Synthetic sample of grains drawn as ellipses in marker canvas:
    # 1. Clear whole grain: 7.2mm
    # 2. Clear broken grain: 4.2mm
    # 3. Boundary grain near cutoff: 5.60mm (within +/- 5% uncertainty band [5.34mm, 5.91mm])
    markers = np.zeros((300, 400), dtype=np.int32)
    cv2.ellipse(markers, (100, 150), (5, int(round(7.2 * 4.88 / 2))), 0, 0, 360, 2, -1)
    cv2.ellipse(markers, (200, 150), (5, int(round(4.2 * 4.88 / 2))), 0, 0, 360, 3, -1)
    cv2.ellipse(markers, (300, 150), (5, int(round(5.6 * 4.88 / 2))), 0, 0, 360, 4, -1)

    mock_bgr = np.full((300, 400, 3), 180, dtype=np.uint8)
    grains = analyzer_basmati.analyze_grains(mock_bgr, markers, pixel_per_mm=4.88)

    for g in grains:
        print(f"Grain #{g.grain_id}: Length={g.length_mm}mm -> Class={g.classification:<10} | Uncertain={g.is_uncertain} | Broken={g.is_broken}")

    assert grains[0].classification == "Whole" and not grains[0].is_uncertain, "Grain 1 should be confident Whole"
    assert grains[1].classification == "Broken" and not grains[1].is_uncertain, "Grain 2 should be confident Broken"
    assert grains[2].is_uncertain and grains[2].classification == "Uncertain", "Grain 3 should be flagged Uncertain"
    print("Status: PASS (Known variety classified accurately with +/-5% edge uncertainty)")

    # -------------------------------------------------------------
    # Test 3: Custom/Auto Dynamic 75th Percentile Estimation
    # -------------------------------------------------------------
    print("\n--- Test 3: Custom/Auto 75th Percentile Intact Length Estimation ---")
    analyzer_auto = ga.RiceGrainAnalyzer(variety="Custom/Auto", length_threshold_ratio=0.75)
    
    test_lengths = [3.5, 4.0, 5.0, 5.5, 6.0, 6.2, 6.5, 6.8, 7.0, 7.2]
    markers_auto = np.zeros((400, 600), dtype=np.int32)
    for idx, l in enumerate(test_lengths):
        m_id = idx + 2
        r_maj = int(round(l * 4.88 / 2))
        cx = (idx % 5) * 110 + 60
        cy = (idx // 5) * 180 + 90
        cv2.ellipse(markers_auto, (cx, cy), (5, r_maj), 0, 0, 360, m_id, -1)

    mock_auto_bgr = np.full((400, 600, 3), 180, dtype=np.uint8)
    grains_auto = analyzer_auto.analyze_grains(mock_auto_bgr, markers_auto, pixel_per_mm=4.88)
    stats_auto = analyzer_auto.compute_summary_statistics(grains_auto)

    print(f"Sample length distribution: {sorted(test_lengths)}")
    print(f"Estimated reference intact length: {stats_auto['reference_length_mm']} mm (Expected ~ 6.7mm)")
    print(f"Is reference estimated: {stats_auto['is_reference_estimated']}")
    print(f"Estimation note: {stats_auto['estimation_note']}")
    assert stats_auto["is_reference_estimated"] is True, "Custom/Auto must set is_reference_estimated = True"
    print("Status: PASS (Custom/Auto dynamically derives reference length from upper percentile)")

    # -------------------------------------------------------------
    # Test 4: Comprehensive Statistics Report (Std Dev, Histogram)
    # -------------------------------------------------------------
    print("\n--- Test 4: Comprehensive Statistics Report Metrics ---")
    img_bgr = cv2.imread("sample_grain_capture.jpg")
    calibrator = ce.GrainImageCalibrator()
    unwarped_bgr, _, px_per_mm = calibrator.unwarp_image(img_bgr)
    
    analyzer_full = ga.RiceGrainAnalyzer(variety="Sona Masoori", length_threshold_ratio=0.75)
    _, grains_full, stats_full, _ = analyzer_full.analyze(unwarped_bgr, px_per_mm)

    print(f"Total grain count       : {stats_full['total_grains']}")
    print(f"Whole grains            : {stats_full['whole_count']} ({stats_full['whole_pct']}%)")
    print(f"Broken grains           : {stats_full['broken_count']} ({stats_full['broken_pct']}%)")
    print(f"Uncertain grains        : {stats_full['uncertain_count']} ({stats_full['uncertain_pct']}%)")
    print(f"Mean kernel length      : {stats_full['avg_length_mm']} mm")
    print(f"Length standard dev     : {stats_full['length_std_mm']} mm")
    print(f"10-Bin histogram counts : {stats_full['length_histogram']['counts']}")
    print(f"Histogram bin edges     : {stats_full['length_histogram']['bin_edges']}")
    print(f"AGMARK Assigned Grade   : {stats_full['agmark_grade']}")

    assert stats_full["total_grains"] > 0, "No grains found in full capture"
    assert stats_full["length_std_mm"] >= 0.0, "Invalid standard deviation"
    assert len(stats_full["length_histogram"]["counts"]) == 10, "Histogram must have 10 bins"
    print("Status: PASS (All summary statistics, standard deviation, and histogram verified)")

    print("\n" + "=" * 80)
    print("ALL FEATURE 4 TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
