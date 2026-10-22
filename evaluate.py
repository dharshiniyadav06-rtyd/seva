"""
Farmgate AI - Evaluation Script (evaluate.py)
Empirical validation comparing:
1. Calibrated Condition ("with sheet"): ArUco fiducials + perspective homography unwarping
2. Uncalibrated Baseline ("no sheet"): Skips homography, uses fixed or user-supplied mm_per_pixel

Input Requirements:
- Folder of images containing pairs: `<sampleID>_withsheet.jpg` and `<sampleID>_nosheet.jpg`
- CSV file of hand-counted ground truth:
    Required columns: `sample_id, total_grains, broken_grains`
    Optional column: `mean_length_mm`

Rules:
- Never print or invent results without real input files. If no data is present, say so clearly.
- Computes per-sample and averaged comparisons:
    * Grain count error (|est - gt|)
    * Broken % error (|est_broken_pct - gt_broken_pct|)
    * Mean grain length error (|est_length - gt_length|, if gt provided)
"""

import os
import sys
import glob
import argparse
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import pandas as pd

# Core Farmgate AI Modules
from calibration_engine import GrainImageCalibrator
from grain_analyzer import RiceGrainAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Farmgate AI - Calibrated vs Uncalibrated Assaying Evaluation Benchmark"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./eval_dataset",
        help="Path to folder containing sampleID_withsheet.jpg and sampleID_nosheet.jpg",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="./eval_dataset/ground_truth.csv",
        help="Path to CSV file with hand-counted ground truth (sample_id, total_grains, broken_grains)",
    )
    parser.add_argument(
        "--variety",
        type=str,
        default="Long Grain Rice (Default)",
        help="Rice variety profile for broken threshold cutoff (e.g., Basmati, Sona Masoori, Swarna)",
    )
    parser.add_argument(
        "--uncalibrated_mm_per_pixel",
        type=float,
        default=0.205,
        help="Fixed mm/px scale to use for uncalibrated baseline (skips ArUco homography)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="Path to export the final comparison results table",
    )
    return parser.parse_args()


def run_calibrated_pipeline(
    image_path: str,
    variety: str,
    marker_span_mm: float = 170.0,
) -> Optional[Dict[str, Any]]:
    """
    Runs full calibrated computer vision pipeline:
    ArUco corner detection -> perspective homography unwarp -> watershed segmentation -> morphometrics.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    calibrator = GrainImageCalibrator(marker_span_mm=marker_span_mm, output_size=(1000, 1000))
    is_valid, quality_metrics, _ = calibrator.validate_capture_quality(img)
    unwarped_bgr, _, px_per_mm, calib_status = calibrator.process(img)

    if not calib_status.get("success", False) or unwarped_bgr is None:
        return {
            "success": False,
            "error_reason": "ArUco detection or unwarping failed (fewer than 4 markers found)",
            "total_grains": 0,
            "broken_grains": 0,
            "broken_pct": 0.0,
            "avg_length_mm": 0.0,
            "px_per_mm": 0.0,
        }

    analyzer = RiceGrainAnalyzer(variety=variety)
    _, grains_list, stats, _ = analyzer.analyze(unwarped_bgr, px_per_mm)

    return {
        "success": True,
        "total_grains": stats["total_grains"],
        "broken_grains": stats["broken_count"],
        "broken_pct": stats["broken_pct"],
        "avg_length_mm": stats["avg_length_mm"],
        "px_per_mm": px_per_mm,
    }


def run_uncalibrated_pipeline(
    image_path: str,
    variety: str,
    fixed_mm_per_pixel: float = 0.205,
) -> Optional[Dict[str, Any]]:
    """
    Runs baseline uncalibrated pipeline:
    Skips ArUco detection and perspective homography. Directly runs watershed segmentation
    and morphological analysis using a fixed/assumed mm_per_pixel.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    px_per_mm = 1.0 / fixed_mm_per_pixel if fixed_mm_per_pixel > 0 else 4.88
    analyzer = RiceGrainAnalyzer(variety=variety)
    _, grains_list, stats, _ = analyzer.analyze(img, px_per_mm)

    return {
        "success": True,
        "total_grains": stats["total_grains"],
        "broken_grains": stats["broken_count"],
        "broken_pct": stats["broken_pct"],
        "avg_length_mm": stats["avg_length_mm"],
        "px_per_mm": px_per_mm,
    }


def evaluate(data_dir: str, ground_truth_path: str, variety: str, uncalib_mm_px: float, output_csv: str):
    print("=" * 80)
    print("FARMGATE AI — CALIBRATED VS UNCALIBRATED PIPELINE EVALUATION")
    print("=" * 80)

    # 1. Check if ground-truth file exists
    if not os.path.exists(ground_truth_path):
        print(f"\n[ERROR] Ground truth file not found: '{ground_truth_path}'")
        print("Never inventing results without real ground-truth files. Please provide a valid CSV containing:")
        print("  sample_id, total_grains, broken_grains, [mean_length_mm]")
        return 1

    # 2. Check if data directory exists
    if not os.path.isdir(data_dir):
        print(f"\n[ERROR] Image data directory not found: '{data_dir}'")
        print("Please provide a folder containing '<sampleID>_withsheet.jpg' and '<sampleID>_nosheet.jpg'.")
        return 1

    # 3. Read ground truth CSV
    try:
        gt_df = pd.read_csv(ground_truth_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to read ground truth CSV: {e}")
        return 1

    req_cols = ["sample_id", "total_grains", "broken_grains"]
    for c in req_cols:
        if c not in gt_df.columns:
            print(f"\n[ERROR] Required column '{c}' is missing in '{ground_truth_path}'.")
            print(f"Columns present: {gt_df.columns.tolist()}")
            return 1

    if gt_df.empty:
        print(f"\n[WARNING] Ground truth CSV '{ground_truth_path}' has 0 rows. No evaluation data present.")
        return 1

    has_gt_length = "mean_length_mm" in gt_df.columns

    print(f"Loaded {len(gt_df)} ground-truth sample(s) from '{ground_truth_path}'.")
    print(f"Searching for paired images in '{data_dir}'...")

    results = []

    for _, row in gt_df.iterrows():
        sid = str(row["sample_id"]).strip()
        gt_total = int(row["total_grains"])
        gt_broken = int(row["broken_grains"])
        gt_broken_pct = round((gt_broken / gt_total * 100.0) if gt_total > 0 else 0.0, 2)
        gt_length = float(row["mean_length_mm"]) if has_gt_length and pd.notnull(row["mean_length_mm"]) else None

        # Look for images with multiple common extensions (.jpg, .jpeg, .png)
        with_candidates = glob.glob(os.path.join(data_dir, f"{sid}_withsheet.*"))
        no_candidates = glob.glob(os.path.join(data_dir, f"{sid}_nosheet.*"))

        if not with_candidates:
            print(f"  [-] Sample {sid}: Missing with-sheet image ('{sid}_withsheet.*'). Skipping.")
            continue
        if not no_candidates:
            print(f"  [-] Sample {sid}: Missing no-sheet image ('{sid}_nosheet.*'). Skipping.")
            continue

        with_img_path = with_candidates[0]
        no_img_path = no_candidates[0]

        # 1. Run Calibrated Pipeline
        calib_res = run_calibrated_pipeline(with_img_path, variety=variety)

        # 2. Run Uncalibrated Baseline Pipeline
        uncalib_res = run_uncalibrated_pipeline(no_img_path, variety=variety, fixed_mm_per_pixel=uncalib_mm_px)

        if not calib_res or not uncalib_res:
            print(f"  [-] Sample {sid}: Could not read image files. Skipping.")
            continue

        # Errors for Calibrated
        calib_cnt_err = abs(calib_res["total_grains"] - gt_total)
        calib_brk_err = round(abs(calib_res["broken_pct"] - gt_broken_pct), 2)
        calib_len_err = round(abs(calib_res["avg_length_mm"] - gt_length), 2) if gt_length is not None else None

        # Errors for Uncalibrated
        uncalib_cnt_err = abs(uncalib_res["total_grains"] - gt_total)
        uncalib_brk_err = round(abs(uncalib_res["broken_pct"] - gt_broken_pct), 2)
        uncalib_len_err = round(abs(uncalib_res["avg_length_mm"] - gt_length), 2) if gt_length is not None else None

        results.append({
            "sample_id": sid,
            "gt_total": gt_total,
            "gt_broken_pct": gt_broken_pct,
            "gt_mean_len": gt_length,
            # Calibrated Outputs & Errors
            "calib_total": calib_res["total_grains"],
            "calib_cnt_err": calib_cnt_err,
            "calib_broken_pct": calib_res["broken_pct"],
            "calib_brk_err": calib_brk_err,
            "calib_mean_len": calib_res["avg_length_mm"],
            "calib_len_err": calib_len_err,
            # Uncalibrated Outputs & Errors
            "uncalib_total": uncalib_res["total_grains"],
            "uncalib_cnt_err": uncalib_cnt_err,
            "uncalib_broken_pct": uncalib_res["broken_pct"],
            "uncalib_brk_err": uncalib_brk_err,
            "uncalib_mean_len": uncalib_res["avg_length_mm"],
            "uncalib_len_err": uncalib_len_err,
        })

    if not results:
        print("\n[INFO] No evaluated sample pairs could be processed.")
        print("Check that your image files are named like '<sampleID>_withsheet.jpg' and '<sampleID>_nosheet.jpg'.")
        return 0

    res_df = pd.DataFrame(results)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("DETAILED PER-SAMPLE EVALUATION RESULTS")
    print("=" * 80)
    
    # Display table
    summary_cols = [
        "sample_id",
        "gt_total", "calib_cnt_err", "uncalib_cnt_err",
        "gt_broken_pct", "calib_brk_err", "uncalib_brk_err",
    ]
    if has_gt_length:
        summary_cols.extend(["gt_mean_len", "calib_len_err", "uncalib_len_err"])

    print(res_df[summary_cols].to_string(index=False))

    # Averages
    mean_calib_cnt_err = res_df["calib_cnt_err"].mean()
    mean_uncalib_cnt_err = res_df["uncalib_cnt_err"].mean()
    mean_calib_brk_err = res_df["calib_brk_err"].mean()
    mean_uncalib_brk_err = res_df["uncalib_brk_err"].mean()

    print("\n" + "=" * 80)
    print("AVERAGED ERROR COMPARISON SUMMARY (CALIBRATED VS UNCALIBRATED)")
    print("=" * 80)
    print(f"Total Evaluated Sample Pairs: {len(res_df)}")
    print(f"{'Metric':<35} | {'Calibrated (With Sheet)':<25} | {'Uncalibrated (No Sheet)':<25}")
    print("-" * 90)
    print(f"{'Mean Grain Count Error (kernels)':<35} | {mean_calib_cnt_err:<25.2f} | {mean_uncalib_cnt_err:<25.2f}")
    print(f"{'Mean Broken % Error (MAE %)':<35} | {mean_calib_brk_err:<25.2f} | {mean_uncalib_brk_err:<25.2f}")

    if has_gt_length and res_df["calib_len_err"].notnull().any():
        mean_calib_len_err = res_df["calib_len_err"].dropna().mean()
        mean_uncalib_len_err = res_df["uncalib_len_err"].dropna().mean()
        print(f"{'Mean Grain Length Error (mm)':<35} | {mean_calib_len_err:<25.2f} | {mean_uncalib_len_err:<25.2f}")

    # Export to CSV
    res_df.to_csv(output_csv, index=False)
    print("\n[PASS] Results saved to:", output_csv)
    return 0


def main():
    args = parse_args()
    return evaluate(
        data_dir=args.data_dir,
        ground_truth_path=args.ground_truth,
        variety=args.variety,
        uncalib_mm_px=args.uncalibrated_mm_per_pixel,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
