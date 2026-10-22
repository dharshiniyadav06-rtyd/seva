"""
test_feature2.py - Verification script for Feature 2: Image Quality Validation Pipeline
Evaluates:
1. Baseline valid sample (sample_grain_capture.jpg) -> status: pass
2. Missing ArUco marker -> status: fail
3. Moderate blur (sharpness in [45, 80]) -> status: warn
4. Heavy blur (sharpness < 45) -> status: fail
5. Severe underexposure -> status: fail
6. Severe overexposure -> status: fail
7. Specular glare in grain staging region -> status: fail / warn
8. Uneven quadrant shadow / lighting gradient -> status: warn
9. Touching / clumped grains -> status: warn
"""

import cv2
import numpy as np
import calibration_engine as ce

def run_tests():
    calibrator = ce.GrainImageCalibrator()
    img_baseline = cv2.imread("sample_grain_capture.jpg")
    assert img_baseline is not None, "Failed to load sample_grain_capture.jpg"
    
    results = []

    # Test 1: Baseline sample
    s1, m1, met1 = ce.validate_image(img_baseline, calibrator)
    results.append(("1. Baseline Sample", s1, m1, {"blur": round(met1["blur_score"], 1), "brightness": round(met1["mean_brightness"], 1)}))

    # Test 2: Missing marker (black out marker 0 at y=350:500, x=100:300)
    img_missing = img_baseline.copy()
    img_missing[350:500, 100:300] = 0
    s2, m2, met2 = ce.validate_image(img_missing, calibrator)
    results.append(("2. Missing Marker (Marker 0)", s2, m2, {"detected": met2["marker_count"], "missing": met2["missing_ids"]}))

    # Test 3: Moderate blur (warning threshold: between 45 and 80)
    img_mod_blur = cv2.GaussianBlur(img_baseline, (5, 5), 1.0)
    s3, m3, met3 = ce.validate_image(img_mod_blur, calibrator)
    results.append(("3. Moderate Blur (Warning)", s3, m3, {"blur": round(met3["blur_score"], 1)}))

    # Test 4: Heavy blur (failure threshold: < 45)
    img_heavy_blur = cv2.GaussianBlur(img_baseline, (5, 5), 1.5)
    s4, m4, met4 = ce.validate_image(img_heavy_blur, calibrator)
    results.append(("4. Heavy Blur (Failure)", s4, m4, {"blur": round(met4["blur_score"], 1)}))

    # Test 5: Severe underexposure (mean < 35)
    img_dark = (img_baseline * 0.14).astype(np.uint8)
    s5, m5, met5 = ce.validate_image(img_dark, calibrator)
    results.append(("5. Severe Underexposure", s5, m5, {"mean_brightness": round(met5["mean_brightness"], 1)}))

    # Test 6: Severe overexposure (mean > 248)
    img_bright = np.clip(img_baseline.astype(np.float32) + 125, 0, 255).astype(np.uint8)
    s6, m6, met6 = ce.validate_image(img_bright, calibrator)
    results.append(("6. Severe Overexposure", s6, m6, {"mean_brightness": round(met6["mean_brightness"], 1)}))

    # Test 7: Specular glare on grains
    h, w = img_baseline.shape[:2]
    centers, _, _, _, _ = calibrator.detect_markers(img_baseline)
    pts = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.float32)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    inner_pts = (pts - [cx, cy]) * 0.85 + [cx, cy]
    x1, x2 = int(max(0, np.min(inner_pts[:, 0]))), int(min(w, np.max(inner_pts[:, 0])))
    y1, y2 = int(max(0, np.min(inner_pts[:, 1]))), int(min(h, np.max(inner_pts[:, 1])))

    gray = cv2.cvtColor(img_baseline, cv2.COLOR_BGR2GRAY)
    grain_mask = np.zeros_like(gray)
    grain_mask[y1:y2, x1:x2] = (gray[y1:y2, x1:x2] < 220).astype(np.uint8) * 255

    img_glare = img_baseline.copy()
    roi_pts = np.argwhere(grain_mask[y1:y2, x1:x2] > 0)
    for r, c in roi_pts[:int(len(roi_pts) * 0.20)]:
        img_glare[y1 + r, x1 + c] = (245, 245, 245)
    s7, m7, met7 = ce.validate_image(img_glare, calibrator, grain_mask=grain_mask)
    results.append(("7. Glare in Grain Region (20% saturated)", s7, m7, {"glare_fraction": round(met7["glare_fraction"], 3)}))

    # Test 8: Uneven shadow across staging area
    img_shadow = img_baseline.copy().astype(np.float32)
    gradient = np.linspace(0.2, 1.0, w)
    img_shadow = (img_shadow * gradient[None, :, None]).astype(np.uint8)
    s8, m8, met8 = ce.validate_image(img_shadow, calibrator)
    results.append(("8. Uneven Shadow Gradient", s8, m8, {"shadow_delta": round(met8["shadow_delta"], 1)}))

    # Test 9: Touching grains
    s9, m9, met9 = ce.validate_image(img_baseline, calibrator, touching_grains_flag=True, touching_ratio=0.35)
    results.append(("9. Touching Grains Clumping", s9, m9, {"touching_ratio": round(met9["touching_ratio"], 2)}))

    print("=" * 80)
    print("FEATURE 2 VALIDATION PIPELINE TEST RESULTS")
    print("=" * 80)
    for title, status, msgs, met in results:
        status_badge = f"[{status.upper()}]"
        print(f"\n{status_badge:<8} {title}")
        print(f"         Metrics : {met}")
        print(f"         Messages: {msgs}")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
