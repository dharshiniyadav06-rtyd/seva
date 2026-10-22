"""
Farmgate AI - Calibration Engine
Module for fiducial detection, capture quality validation, perspective unwarping
via homography transformation, pixel-to-millimeter scaling, and color/exposure normalization.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import cv2


# =============================================================================
# FEATURE 2: IMAGE QUALITY VALIDATION CONFIGURATION THRESHOLDS
# =============================================================================
# How to tune these thresholds on real smartphone photos:
#
# 1. BLUR_WARN_THRESHOLD (80.0) & BLUR_FAIL_THRESHOLD (45.0):
#    Measured using Variance of Laplacian (cv2.Laplacian(gray, cv2.CV_64F).var()).
#    Sharp smartphone photos generally score between 120 and 800+.
#    Mild motion blur drops to 50-80. Below 45, grain contour edges blur together.
#    To make the blur check more forgiving, lower BLUR_WARN_THRESHOLD to 60.0.
#
# 2. BRIGHTNESS_MIN_FAIL (35.0) & BRIGHTNESS_MIN_WARN (60.0):
#    Mean 8-bit grayscale intensity across the whole image [0 - 255].
#    Adequately lit paper measures 110-185. Below 60 indicates poor indoor light.
#    Below 35 is too dark for watershed segmentation to separate grains from paper.
#
# 3. BRIGHTNESS_MAX_WARN (215.0) & BRIGHTNESS_MAX_FAIL (238.0):
#    Mean grayscale intensity above 215 indicates strong overexposure.
#    Above 238 washes out grain body details into the background white paper.
#
# 4. HIST_CLIP_SHADOW_WARN (0.12) & HIST_CLIP_HIGHLIGHT_WARN (0.18):
#    Fraction of pixels clipped to pure black (< 8) or pure white (> 248).
#    High clipping fractions indicate sensor dynamic range saturation.
#
# 5. GLARE_WARN_THRESHOLD (0.04) & GLARE_FAIL_THRESHOLD (0.12):
#    Fraction of specular/washed-out pixels (> 245 luminance, low saturation)
#    inside the central grain staging area.
#    Direct flashlight / overhead LED reflection creates false grain detections.
#
# 6. SHADOW_DIFF_WARN (38.0):
#    Maximum difference in mean luminance between staging quadrants (TL, TR, BL, BR).
#    A difference > 38 indicates a phone shadow or directional side-window lighting.
#
# 7. TOUCHING_RATIO_WARN (0.25):
#    Fraction of grains touching or clustered. Above 25% degrades measurement accuracy.
# =============================================================================
BLUR_FAIL_THRESHOLD = 45.0
BLUR_WARN_THRESHOLD = 80.0
BRIGHTNESS_MIN_FAIL = 35.0
BRIGHTNESS_MIN_WARN = 60.0
BRIGHTNESS_MAX_WARN = 232.0
BRIGHTNESS_MAX_FAIL = 248.0
HIST_CLIP_SHADOW_WARN = 0.12
HIST_CLIP_HIGHLIGHT_WARN = 0.55
GLARE_WARN_THRESHOLD = 0.05
GLARE_FAIL_THRESHOLD = 0.15
SHADOW_DIFF_WARN = 38.0
TOUCHING_RATIO_WARN = 0.25


class GrainImageCalibrator:
    """
    Calibrates smartphone images of grain samples placed on the Farmgate AI
    calibration sheet. Computes perspective unwarping via ArUco markers,
    validates capture quality (blur, glare, marker visibility),
    calculates exact pixel-to-millimeter scaling, and normalizes lighting.
    """

    def __init__(
        self,
        marker_span_mm: float = 172.0,
        marker_size_mm: float = 22.0,
        output_size: Tuple[int, int] = (1000, 1000),
        canvas_margin_px: int = 80,
    ):
        """
        Args:
            marker_span_mm: Physical distance (mm) between marker centers along X and Y axes (default: 172mm).
            marker_size_mm: Physical side length (mm) of each ArUco marker (default: 22mm).
            output_size: Desired dimensions (width, height) of unwarped rectified image (default: 1000x1000px).
            canvas_margin_px: Padding (px) from unwarped image boundary to marker centers.
        """
        self.marker_span_mm = marker_span_mm
        self.marker_size_mm = marker_size_mm
        self.output_width, self.output_height = output_size
        self.margin_px = canvas_margin_px

        # Calculate exact pixel-to-mm and mm-per-pixel ratio for the unwarped canvas:
        # Distance between TL center and TR center in unwarped pixels:
        self.unwarped_span_px = float(self.output_width - 2 * self.margin_px)
        self.pixel_per_mm = self.unwarped_span_px / self.marker_span_mm
        self.mm_per_pixel = self.marker_span_mm / self.unwarped_span_px

        # Target 2D coordinates for marker centers in the unwarped coordinate frame:
        # ID 0: Top-Left
        # ID 1: Top-Right
        # ID 2: Bottom-Right
        # ID 3: Bottom-Left
        self.dst_pts = np.array(
            [
                [self.margin_px, self.margin_px],                                          # TL (ID 0)
                [self.output_width - self.margin_px, self.margin_px],                      # TR (ID 1)
                [self.output_width - self.margin_px, self.output_height - self.margin_px], # BR (ID 2)
                [self.margin_px, self.output_height - self.margin_px],                      # BL (ID 3)
            ],
            dtype=np.float32,
        )

        # Initialize ArUco dictionary & detector
        self._init_aruco_detector()

    def _init_aruco_detector(self):
        """Initializes ArUco detector compatible with OpenCV 4.5 through 4.10+."""
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        try:
            # OpenCV 4.7+ syntax
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self._use_new_detector = True
        except AttributeError:
            # OpenCV <= 4.6 syntax
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._use_new_detector = False

    def detect_markers(
        self, image_bgr: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, Any, Any]:
        """
        Detects ArUco markers in the image.

        Returns:
            centers: Dict mapping marker_id -> (cx, cy) center coordinate
            corners_dict: Dict mapping marker_id -> 4 corner coordinates (4, 2)
            annotated_img: Copy of image with detected markers drawn
            raw_corners: Raw detected corners tuple/list from OpenCV
            raw_ids: Raw detected marker IDs array
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        annotated_img = image_bgr.copy()

        if self._use_new_detector:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        centers = {}
        corners_dict = {}

        if ids is not None and len(ids) > 0:
            flat_ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(annotated_img, corners, ids)

            for idx, m_id in enumerate(flat_ids):
                m_corners = corners[idx][0]  # shape: (4, 2)
                cx = float(np.mean(m_corners[:, 0]))
                cy = float(np.mean(m_corners[:, 1]))
                centers[int(m_id)] = np.array([cx, cy], dtype=np.float32)
                corners_dict[int(m_id)] = m_corners

                # Draw center point
                cv2.circle(annotated_img, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1)
                cv2.putText(
                    annotated_img,
                    f"ID:{m_id}",
                    (int(round(cx)) + 10, int(round(cy)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        return centers, corners_dict, annotated_img, corners, ids

    def validate_image(
        self,
        image_bgr: np.ndarray,
        touching_grains_flag: bool = False,
        touching_ratio: float = 0.0,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Instance method wrapper for validate_image."""
        return validate_image(
            image_bgr,
            calibrator=self,
            touching_grains_flag=touching_grains_flag,
            touching_ratio=touching_ratio,
        )

    def validate_capture_quality(
        self,
        image_bgr: np.ndarray,
        corners: Optional[Any] = None,
        ids: Optional[Any] = None,
        touching_grains_flag: bool = False,
        touching_ratio: float = 0.0,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validates capture quality (backward compatible interface for existing calls).
        Returns:
            is_valid (bool): True if status is not 'fail'
            metrics (dict): Dict of measured metrics
            messages (list): List of guidance alerts/feedback messages
        """
        status, messages, metrics = self.validate_image(
            image_bgr,
            touching_grains_flag=touching_grains_flag,
            touching_ratio=touching_ratio,
        )
        is_valid = (status != "fail")
        return is_valid, metrics, messages

    def unwarp_image(
        self,
        image_bgr: np.ndarray,
        centers: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Calculates the 3x3 Homography Matrix using cv2.getPerspectiveTransform
        to flatten the photo to a 1000x1000px top-down view and computes
        the exact pixel-to-millimeter ratio.

        Returns:
            unwarped_bgr: 1000x1000 top-down perspective-corrected image (or None if failed)
            homography_matrix: 3x3 np.ndarray Homography Matrix
            pixel_per_mm: Millimeter scale factor for the unwarped canvas
        """
        if centers is None:
            centers, _, _, _, _ = self.detect_markers(image_bgr)

        required_ids = [0, 1, 2, 3]
        if any(mid not in centers for mid in required_ids):
            return None, None, self.pixel_per_mm

        # Gather source points in order: TL (0), TR (1), BR (2), BL (3)
        src_pts = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.float32)

        # Compute 3x3 Homography Matrix
        H = cv2.getPerspectiveTransform(src_pts, self.dst_pts)

        # Flatten the photo to a 1000x1000px top-down view
        unwarped_bgr = cv2.warpPerspective(
            image_bgr,
            H,
            (self.output_width, self.output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return unwarped_bgr, H, self.pixel_per_mm

    def normalize_lighting(
        self,
        unwarped_bgr: np.ndarray,
        use_clahe: bool = True,
        use_white_balance: bool = True,
    ) -> np.ndarray:
        """
        Samples reference areas to adjust white balance, exposure, and glare normalization:
        1. Normalizes illuminance gradients using CIELAB and CLAHE on luminance channel.
        2. Adjusts white-balance so reference paper and swatches have normalized neutral gains.
        """
        normalized = unwarped_bgr.copy()

        # Step 1: Illuminance / Contrast Normalization using CIELAB and CLAHE
        if use_clahe:
            lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l_channel)

            lab_eq = cv2.merge((l_eq, a_channel, b_channel))
            normalized = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Step 2: Automatic White Balance (White-Patch Reference Normalization)
        if use_white_balance:
            gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            threshold_val = np.percentile(gray, 98)
            bright_mask = gray >= threshold_val

            if np.sum(bright_mask) > 100:
                b_high = np.mean(normalized[:, :, 0][bright_mask])
                g_high = np.mean(normalized[:, :, 1][bright_mask])
                r_high = np.mean(normalized[:, :, 2][bright_mask])

                target_val = 250.0
                scale_b = target_val / max(b_high, 1e-3)
                scale_g = target_val / max(g_high, 1e-3)
                scale_r = target_val / max(r_high, 1e-3)

                b = np.clip(normalized[:, :, 0] * scale_b, 0, 255).astype(np.uint8)
                g = np.clip(normalized[:, :, 1] * scale_g, 0, 255).astype(np.uint8)
                r = np.clip(normalized[:, :, 2] * scale_r, 0, 255).astype(np.uint8)
                normalized = cv2.merge([b, g, r])

        return normalized

    def calibrate_image(
        self,
        image_bgr: np.ndarray,
        apply_white_balance: bool = True,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        """
        Feature 1: Detects 4 corner ArUco markers, computes homography, warps to top-down view,
        and computes mm_per_pixel from known marker span and size.
        
        Returns:
            warped_image (np.ndarray or None): 1000x1000 top-down perspective-corrected image, or None if failed.
            mm_per_pixel (float): Scale factor in millimeters per pixel (0.0 if failed).
            status (dict): Confidence and execution metadata:
                - success: bool
                - confidence: float (0.0 to 1.0)
                - markers_found: int (count of corner markers detected)
                - detected_ids: list of int
                - missing_ids: list of int
                - mm_per_pixel: float
                - pixel_per_mm: float
                - homography_matrix: Optional[np.ndarray]
                - message: str
        """
        centers, corners_dict, _, _, ids = self.detect_markers(image_bgr)
        detected_ids = sorted([int(k) for k in centers.keys()])
        required_ids = [0, 1, 2, 3]
        missing_ids = [mid for mid in required_ids if mid not in centers]

        status: Dict[str, Any] = {
            "success": False,
            "confidence": 0.0,
            "markers_found": len([m for m in required_ids if m in centers]),
            "detected_ids": detected_ids,
            "missing_ids": missing_ids,
            "mm_per_pixel": 0.0,
            "pixel_per_mm": 0.0,
            "homography_matrix": None,
            "message": "",
        }

        # Handle failure cleanly when fewer than 4 corner markers are found
        if missing_ids:
            status["message"] = (
                f"Fewer than 4 corner ArUco markers found ({len(detected_ids)}/4 detected: {detected_ids}). "
                f"Missing markers: {missing_ids}. Please sweep grains away from corner markers and ensure all 4 are visible."
            )
            return None, 0.0, status

        # Compute homography and unwarp image
        unwarped_bgr, H, px_per_mm = self.unwarp_image(image_bgr, centers=centers)
        if unwarped_bgr is None or H is None:
            status["message"] = "Homography computation failed. Marker points may be degenerate or colinear."
            return None, 0.0, status

        # Compute confidence based on quadrilateral geometry
        src_pts = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.float32)
        area = cv2.contourArea(src_pts)
        img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
        area_coverage = area / max(img_area, 1.0)

        # Opposite edge ratio consistency
        w_top = float(np.linalg.norm(src_pts[1] - src_pts[0]))
        w_bot = float(np.linalg.norm(src_pts[2] - src_pts[3]))
        h_left = float(np.linalg.norm(src_pts[3] - src_pts[0]))
        h_right = float(np.linalg.norm(src_pts[2] - src_pts[1]))

        w_ratio = min(w_top, w_bot) / max(max(w_top, w_bot), 1e-3)
        h_ratio = min(h_left, h_right) / max(max(h_left, h_right), 1e-3)

        # Confidence score calculation (higher when sheet is well framed and less tilted)
        conf = 0.5 + 0.25 * min(1.0, area_coverage * 3.0) + 0.25 * (w_ratio * h_ratio)
        confidence = float(np.clip(conf, 0.1, 0.99))

        # Apply neutral white balance if requested
        if apply_white_balance:
            processed_warped = self.normalize_lighting(unwarped_bgr)
        else:
            processed_warped = unwarped_bgr

        status["success"] = True
        status["confidence"] = round(confidence, 3)
        status["mm_per_pixel"] = round(self.mm_per_pixel, 5)
        status["pixel_per_mm"] = round(self.pixel_per_mm, 3)
        status["homography_matrix"] = H
        status["message"] = f"Calibration successful (scale: {self.mm_per_pixel:.4f} mm/px, confidence: {confidence:.1%})."

        return processed_warped, self.mm_per_pixel, status

    def process(
        self, image_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray, float, Dict[str, Any]]:
        """
        Unified end-to-end calibration pipeline:
        1. Marker detection and capture quality validation
        2. 3x3 Homography perspective unwarping to 1000x1000px
        3. Color, white-balance, and exposure normalization
        """
        centers, corners_dict, annotated_orig, raw_corners, raw_ids = self.detect_markers(image_bgr)
        is_valid, quality_metrics, messages = self.validate_capture_quality(
            image_bgr, corners=raw_corners, ids=raw_ids
        )

        # Call calibrate_image
        warped_img, mm_per_px, calib_status = self.calibrate_image(image_bgr, apply_white_balance=True)

        status_info = {
            "markers_found": list(centers.keys()),
            "success": calib_status["success"],
            "confidence": calib_status["confidence"],
            "error_msg": calib_status["message"] if not calib_status["success"] else "",
            "quality_metrics": quality_metrics,
            "quality_messages": messages,
            "is_valid": is_valid,
            "mm_per_pixel": calib_status["mm_per_pixel"],
            "pixel_per_mm": self.pixel_per_mm,
            "homography_matrix": calib_status.get("homography_matrix"),
        }

        # Draw quadrilateral connecting the 4 marker centers on the original image
        if all(k in centers for k in [0, 1, 2, 3]):
            src_pts = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.int32)
            cv2.polylines(annotated_orig, [src_pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 255), thickness=3)

        if not calib_status["success"]:
            if not status_info["error_msg"]:
                status_info["error_msg"] = messages[0] if messages else "Fewer than 4 corner markers detected."
            return None, annotated_orig, self.pixel_per_mm, status_info

        return warped_img, annotated_orig, self.pixel_per_mm, status_info


def validate_image(
    image_bgr: np.ndarray,
    calibrator: Optional[GrainImageCalibrator] = None,
    grain_mask: Optional[np.ndarray] = None,
    touching_grains_flag: bool = False,
    touching_ratio: float = 0.0,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Feature 2: Image Quality Validation Pipeline.
    Evaluates 6 specific capture quality checks:
    1. Markers: Checks if all 4 corner ArUco markers (IDs 0, 1, 2, 3) are found.
    2. Blur: Evaluates sharpness using variance of Laplacian.
    3. Lighting / Exposure: Checks mean luminance and histogram shadow/highlight clipping.
    4. Glare: Checks fraction of specular/saturated pixels in the grain staging region.
    5. Shadows: Checks brightness variance across quadrants of the grain staging region.
    6. Touching Grains: Evaluates clumped/touching grain flag or ratio from segmentation.

    Returns:
        status (str): "pass", "warn", or "fail"
        messages (List[str]): Plain-language English descriptions of issues detected.
        metrics (Dict[str, Any]): Dictionary of measured metrics for logging and display.
    """
    if calibrator is None:
        calibrator = GrainImageCalibrator()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    messages: List[str] = []
    has_fail = False
    has_warn = False

    # 1. Markers Check
    centers, _, _, _, ids = calibrator.detect_markers(image_bgr)
    detected_ids = sorted([int(k) for k in centers.keys()])
    required_ids = {0, 1, 2, 3}
    missing_ids = sorted(list(required_ids - set(detected_ids)))

    if len(detected_ids) < 4 or missing_ids:
        has_fail = True
        messages.append(
            f"Fewer than 4 corner ArUco markers detected ({len(detected_ids)}/4 found: {detected_ids}). "
            f"Missing markers: {missing_ids}. Sweep grains away from corner boxes and retake."
        )

    # 2. Blur Check (Variance of Laplacian)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < BLUR_FAIL_THRESHOLD:
        has_fail = True
        messages.append(
            f"Photo is blurry, hold the phone steady and retake. "
            f"(Sharpness: {blur_score:.1f} < {BLUR_FAIL_THRESHOLD:.0f})"
        )
    elif blur_score < BLUR_WARN_THRESHOLD:
        has_warn = True
        messages.append(
            f"Photo sharpness is low, hold the phone steady and tap to focus. "
            f"(Sharpness: {blur_score:.1f} < {BLUR_WARN_THRESHOLD:.0f})"
        )

    # 3. Too Dark / Too Bright & Histogram Clipping
    mean_brightness = float(np.mean(gray))
    dark_clipped = float(np.sum(gray < 8)) / float(max(gray.size, 1))
    bright_clipped = float(np.sum(gray > 248)) / float(max(gray.size, 1))

    if mean_brightness < BRIGHTNESS_MIN_FAIL:
        has_fail = True
        messages.append(
            f"Photo is too dark to analyze grains (Mean: {mean_brightness:.1f} < {BRIGHTNESS_MIN_FAIL:.0f}). "
            f"Increase room lighting or move closer to light."
        )
    elif mean_brightness < BRIGHTNESS_MIN_WARN or dark_clipped > HIST_CLIP_SHADOW_WARN:
        has_warn = True
        messages.append(
            f"Photo is dimly lit or has deep shadow clipping ({dark_clipped*100:.1f}% shadow clipped). "
            f"Adding more light will improve grain measurement accuracy."
        )

    if mean_brightness > BRIGHTNESS_MAX_FAIL:
        has_fail = True
        messages.append(
            f"Photo is severely overexposed / washed out (Mean: {mean_brightness:.1f} > {BRIGHTNESS_MAX_FAIL:.0f}). "
            f"Move away from direct harsh lighting."
        )
    elif mean_brightness > BRIGHTNESS_MAX_WARN or bright_clipped > HIST_CLIP_HIGHLIGHT_WARN:
        has_warn = True
        messages.append(
            f"Photo is brightly exposed with highlight clipping ({bright_clipped*100:.1f}% clipped). "
            f"Beware of washed-out grain boundaries."
        )

    # Determine grain staging region for glare and shadow analysis
    h, w = gray.shape
    if len(centers) == 4 and all(k in centers for k in [0, 1, 2, 3]):
        pts = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.float32)
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        inner_pts = (pts - [cx, cy]) * 0.85 + [cx, cy]
        x1 = int(max(0, np.min(inner_pts[:, 0])))
        x2 = int(min(w, np.max(inner_pts[:, 0])))
        y1 = int(max(0, np.min(inner_pts[:, 1])))
        y2 = int(min(h, np.max(inner_pts[:, 1])))
    else:
        x1, x2 = int(w * 0.2), int(w * 0.8)
        y1, y2 = int(h * 0.2), int(h * 0.8)

    stage_roi_gray = gray[y1:y2, x1:x2]

    # 4. Glare Check (Fraction of saturated pixels in the grain region)
    if grain_mask is not None:
        grain_mask_roi = grain_mask[y1:y2, x1:x2]
        grain_pixels = stage_roi_gray[grain_mask_roi > 0]
        if len(grain_pixels) > 20:
            glare_count = np.sum(grain_pixels > 230)
            glare_fraction = float(glare_count) / float(len(grain_pixels))
        else:
            glare_fraction = 0.0
    else:
        # Segment candidate grain regions on staging area via Otsu
        _, otsu_inv = cv2.threshold(stage_roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Eroded slightly to keep only interior of grains, avoiding white paper boundary transitions
        eroded = cv2.erode(otsu_inv, np.ones((3, 3), np.uint8))
        grain_pixels = stage_roi_gray[eroded > 0]

        # Check if grain interior pixels exhibit blown-out specular highlights (> 225)
        if len(grain_pixels) > 50:
            glare_count = np.sum(grain_pixels > 225)
            glare_fraction = float(glare_count) / float(len(grain_pixels))
        else:
            # Fallback if paper background is darker/diffuse (< 240) and exhibits a specular flash hotspot (> 252)
            med_stage = float(np.median(stage_roi_gray))
            if med_stage < 240:
                glare_count = np.sum(stage_roi_gray > 252)
                glare_fraction = float(glare_count) / float(max(stage_roi_gray.size, 1))
            else:
                glare_fraction = 0.0

    if glare_fraction > GLARE_FAIL_THRESHOLD:
        has_fail = True
        messages.append(
            f"Severe glare detected in the grain region ({glare_fraction*100:.1f}% saturated). "
            f"Tilt phone slightly or diffuse light to remove reflections."
        )
    elif glare_fraction > GLARE_WARN_THRESHOLD:
        has_warn = True
        messages.append(
            f"Noticeable specular reflection detected in the grain region ({glare_fraction*100:.1f}% glare). "
            f"Tilt phone slightly to eliminate reflections."
        )

    # 5. Strong Shadow Check (Brightness variance across stage quadrants)
    rh, rw = stage_roi_gray.shape
    if rh > 10 and rw > 10:
        mid_y, mid_x = rh // 2, rw // 2
        q1 = float(np.mean(stage_roi_gray[:mid_y, :mid_x]))
        q2 = float(np.mean(stage_roi_gray[:mid_y, mid_x:]))
        q3 = float(np.mean(stage_roi_gray[mid_y:, :mid_x]))
        q4 = float(np.mean(stage_roi_gray[mid_y:, mid_x:]))
        quad_means = [q1, q2, q3, q4]
        shadow_delta = float(max(quad_means) - min(quad_means))
    else:
        shadow_delta = 0.0

    if shadow_delta > SHADOW_DIFF_WARN:
        has_warn = True
        messages.append(
            f"Strong uneven shadow detected across the staging area (Delta {shadow_delta:.1f} > {SHADOW_DIFF_WARN:.0f}). "
            f"Hold phone directly overhead without blocking the light."
        )

    # 6. Touching Grains Check
    if touching_grains_flag or touching_ratio > TOUCHING_RATIO_WARN:
        has_warn = True
        messages.append(
            f"Too many grains are clumped or touching together ({touching_ratio*100:.1f}% touching). "
            f"Gently spread grains into a single non-overlapping layer."
        )

    # Final overall status
    if has_fail:
        status = "fail"
    elif has_warn:
        status = "warn"
    else:
        status = "pass"
        messages.append("All capture quality checks passed: sharp focus, balanced exposure, and 4 markers verified.")

    metrics = {
        "status": status,
        "is_valid": (status != "fail"),
        "marker_count": len(detected_ids),
        "detected_ids": detected_ids,
        "missing_ids": missing_ids,
        "blur_score": round(blur_score, 1),
        "mean_brightness": round(mean_brightness, 1),
        "dark_clip_fraction": round(dark_clipped, 3),
        "bright_clip_fraction": round(bright_clipped, 3),
        "glare_fraction": round(glare_fraction, 3),
        "shadow_delta": round(shadow_delta, 1),
        "touching_grains_flag": touching_grains_flag,
        "touching_ratio": round(touching_ratio, 3),
    }

    return status, messages, metrics
