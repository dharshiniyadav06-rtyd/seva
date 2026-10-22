"""
Test script for evaluate.py
Creates real test images and real ground-truth CSV, runs evaluate.py, and verifies outputs.
"""

import os
import cv2
import pandas as pd
from generate_sample_image import generate_sample_smartphone_capture
import evaluate


def test_evaluate_script():
    print("======================================================================")
    print("TESTING EVALUATE.PY SCRIPT WITH REAL SYNTHETIC IMAGE PAIR")
    print("======================================================================")

    test_eval_dir = "test_eval_dir"
    os.makedirs(test_eval_dir, exist_ok=True)

    gt_csv_path = os.path.join(test_eval_dir, "ground_truth.csv")
    with_img_path = os.path.join(test_eval_dir, "SMP_EVAL_01_withsheet.jpg")
    no_img_path = os.path.join(test_eval_dir, "SMP_EVAL_01_nosheet.jpg")

    # 1. Generate real with-sheet capture (48 grains)
    generate_sample_smartphone_capture(with_img_path, num_grains=48, variety="Basmati")

    # 2. Generate no-sheet image (e.g., crop or center patch of the image to simulate no ArUco markers)
    full_img = cv2.imread(with_img_path)
    # Crop central area without ArUco corners
    h, w = full_img.shape[:2]
    crop_center = full_img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    cv2.imwrite(no_img_path, crop_center)

    # 3. Create ground-truth CSV
    gt_df = pd.DataFrame([
        {
            "sample_id": "SMP_EVAL_01",
            "total_grains": 48,
            "broken_grains": 6,
            "mean_length_mm": 7.5,
        }
    ])
    gt_df.to_csv(gt_csv_path, index=False)

    # 4. Run evaluate.py
    out_csv = os.path.join(test_eval_dir, "eval_test_results.csv")
    ret = evaluate.evaluate(
        data_dir=test_eval_dir,
        ground_truth_path=gt_csv_path,
        variety="Basmati",
        uncalib_mm_px=0.205,
        output_csv=out_csv,
    )

    assert ret == 0, f"evaluate() returned error code {ret}"
    assert os.path.exists(out_csv), "Evaluation results CSV was not generated"

    res_df = pd.read_csv(out_csv)
    assert len(res_df) == 1, f"Expected 1 evaluated sample row, got {len(res_df)}"
    assert "calib_cnt_err" in res_df.columns, "Missing calib_cnt_err in results"
    assert "uncalib_cnt_err" in res_df.columns, "Missing uncalib_cnt_err in results"
    print("\n[PASS] evaluate.py executed and produced valid results table and CSV!")
    print(res_df[["sample_id", "gt_total", "calib_total", "calib_cnt_err", "uncalib_total", "uncalib_cnt_err"]].to_string())

    # Clean up test files
    for f in [with_img_path, no_img_path, gt_csv_path, out_csv]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(test_eval_dir):
        os.rmdir(test_eval_dir)

    print("\n======================================================================")
    print("EVALUATE.PY TEST PASSED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    test_evaluate_script()
