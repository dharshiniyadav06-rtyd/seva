"""
Farmgate AI - Calibration Engine
Module for fiducial detection, capture quality validation, perspective unwarping
via homography transformation, pixel-to-millimeter scaling, and color/exposure normalization.
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import cv2


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
        output_size: Tuple[int, int] = (1000, 1000),
        canvas_margin_px: int = 80,
    ):
        """
        Args:
            marker_span_mm: Physical distance (mm) between marker centers along X and Y axes (default: 172mm).
            output_size: Desired dimensions (width, height) of unwarped rectified image (default: 1000x1000px).
            canvas_margin_px: Padding (px) from unwarped image boundary to marker centers.
        """
        self.marker_span_mm = marker_span_mm
        self.output_width, self.output_height = output_size
        self.margin_px = canvas_margin_px

        # Calculate exact pixel-to-mm ratio for the unwarped canvas:
        # Distance between TL center and TR center in unwarped pixels:
        self.unwarped_span_px = self.output_width - 2 * self.margin_px
        self.pixel_per_mm = self.unwarped_span_px / self.marker_span_mm

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

    def validate_capture_quality(
        self,
        image_bgr: np.ndarray,
        corners: Optional[Any] = None,
        ids: Optional[Any] = None,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validates the smartphone capture quality prior to grain analysis:
        1. Checks if all 4 corner ArUco markers (IDs 0, 1, 2, 3) are detected.
           If < 4: returns "⚠️ Fewer than 4 corner markers detected. Sweep grains away from corner boxes and retake."
        2. Calculates blur score using Laplacian variance (cv2.Laplacian).
           If < 80: flags image as blurry.
        3. Checks mean brightness.
           If > 220: flags overexposed/glare detected.
           If < 40: flags too dark.

        Returns:
            is_valid (bool): True if critical quality checks pass.
            metrics (dict): Dict of measured metrics (blur_score, brightness, marker_count, detected_ids).
            messages (list): List of guidance alerts/feedback messages.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        messages = []
        is_valid = True

        # Marker validation
        if ids is None:
            _, _, _, corners, ids = self.detect_markers(image_bgr)

        detected_ids = []
        if ids is not None and len(ids) > 0:
            detected_ids = [int(i) for i in ids.flatten()]

        # Check required IDs 0, 1, 2, 3
        required_ids = {0, 1, 2, 3}
        missing_ids = required_ids - set(detected_ids)

        if len(detected_ids) < 4 or missing_ids:
            is_valid = False
            messages.append("⚠️ Fewer than 4 corner markers detected. Sweep grains away from corner boxes and retake.")

        # Blur calculation using Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 80.0:
            messages.append(f"⚠️ Image is blurry (Sharpness: {laplacian_var:.1f} < 80). Hold device steady and retake.")
            if laplacian_var < 45.0:
                is_valid = False

        # Brightness / Exposure & Glare check
        mean_brightness = float(np.mean(gray))
        if mean_brightness > 220.0:
            messages.append(f"⚠️ Image is overexposed/glare detected (Brightness: {mean_brightness:.1f} > 220). Move away from direct harsh lighting.")
        elif mean_brightness < 40.0:
            messages.append(f"⚠️ Image is too dark (Brightness: {mean_brightness:.1f} < 40). Ensure adequate ambient lighting.")
            is_valid = False

        metrics = {
            "marker_count": len(detected_ids),
            "detected_ids": detected_ids,
            "missing_ids": list(missing_ids),
            "blur_score": round(laplacian_var, 1),
            "mean_brightness": round(mean_brightness, 1),
            "is_blurry": laplacian_var < 80.0,
            "is_overexposed": mean_brightness > 220.0,
            "is_too_dark": mean_brightness < 40.0,
        }

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

        status_info = {
            "markers_found": list(centers.keys()),
            "success": False,
            "error_msg": "",
            "quality_metrics": quality_metrics,
            "quality_messages": messages,
            "is_valid": is_valid,
        }

        if not is_valid and len(quality_metrics["detected_ids"]) < 4:
            status_info["error_msg"] = (
                messages[0] if messages else "Fewer than 4 corner markers detected."
            )
            return None, annotated_orig, self.pixel_per_mm, status_info

        # Draw quadrilateral connecting the 4 marker centers on the original image
        if all(k in centers for k in [0, 1, 2, 3]):
            src_pts = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.int32)
            cv2.polylines(annotated_orig, [src_pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 255), thickness=3)

        unwarped_bgr, H, px_per_mm = self.unwarp_image(image_bgr, centers=centers)

        if unwarped_bgr is None:
            status_info["error_msg"] = "Could not compute perspective transform. Check ArUco marker visibility."
            return None, annotated_orig, px_per_mm, status_info

        # Normalize lighting & white balance
        normalized_unwarped = self.normalize_lighting(unwarped_bgr)

        status_info["success"] = True
        status_info["homography_matrix"] = H
        return normalized_unwarped, annotated_orig, px_per_mm, status_info
