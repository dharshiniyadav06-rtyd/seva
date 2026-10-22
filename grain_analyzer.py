"""
Farmgate AI - Grain Analyzer Module
Computer Vision pipeline for rice grain segmentation using Distance Transform + Watershed,
variety-aware morphometrics, defect classification (Whole, Broken, Discoloured, Chalky),
and AGMARK quality grading.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2


@dataclass
class GrainFeature:
    grain_id: int
    length_mm: float
    width_mm: float
    aspect_ratio: float
    area_mm2: float
    center_x: float
    center_y: float
    mean_r: float
    mean_g: float
    mean_b: float
    mean_hsv_s: float
    chalky_ratio: float
    classification: str  # "Whole", "Broken", "Discoloured", "Chalky"
    is_broken: bool
    is_discoloured: bool
    is_chalky: bool
    box_contour: list  # list of (x, y) coordinates for rendering


class RiceGrainAnalyzer:
    """
    Analyzes calibrated top-down images of rice grains.
    Features:
    - Variety-Aware Baselines: Sona Masoori (5.2mm), Basmati (7.5mm), Swarna (5.0mm), IR64 (6.0mm), Auto-Detect
    - Watershed segmentation to separate touching grains
    - Sub-millimeter major axis length measurement via fitted ellipse
    - Whole vs. Broken classification (length >= 75% cutoff)
    - Discolouration and Chalkiness defect classification
    - Color-coded bounding box visualization
    - AGMARK grading standard
    """

    # Variety-Aware Baselines (in millimeters)
    VARIETY_BASELINES = {
        "Sona Masoori": {"full_length_mm": 5.2, "name_ta": "சோனா மசூரி"},
        "Basmati": {"full_length_mm": 7.5, "name_ta": "பாசுமதி"},
        "Swarna": {"full_length_mm": 5.0, "name_ta": "ஸ்வர்ணா"},
        "IR64": {"full_length_mm": 6.0, "name_ta": "ஐஆர்64"},
        "Auto-Detect": {"full_length_mm": 6.0, "name_ta": "தானியங்கி கண்டறிதல்"},
    }

    # BGR colors for visualization:
    # Green = Whole, Red = Broken, Yellow = Discoloured, Blue = Chalky
    CLASS_COLORS = {
        "Whole": (0, 200, 0),         # Green
        "Broken": (0, 0, 230),        # Red
        "Discoloured": (0, 215, 255), # Yellow / Amber
        "Chalky": (230, 140, 20),     # Blue / Cyan
    }

    def __init__(
        self,
        variety: str = "Sona Masoori",
        length_threshold_ratio: float = 0.75,
        discolour_threshold: float = 38.0,
        chalky_threshold: float = 0.22,
    ):
        """
        Args:
            variety: Name from VARIETY_BASELINES ("Sona Masoori", "Basmati", "Swarna", "IR64", "Auto-Detect")
            length_threshold_ratio: Minimum ratio of expected length to be deemed 'Whole' (default: 0.75 = 75%)
            discolour_threshold: Color deviation / saturation cutoff for discolouration
            chalky_threshold: Fractional area ratio of opaque white chalkiness inside kernel
        """
        self.variety = variety if variety in self.VARIETY_BASELINES else "Sona Masoori"
        self.full_length_mm = self.VARIETY_BASELINES[self.variety]["full_length_mm"]
        self.length_threshold_ratio = length_threshold_ratio
        self.broken_cutoff_length_mm = self.full_length_mm * self.length_threshold_ratio
        self.discolour_threshold = discolour_threshold
        self.chalky_threshold = chalky_threshold
        self.detected_variety = self.variety

    def segment_touching_grains(
        self,
        gray_img: np.ndarray,
        pixel_per_mm: float = 4.88,
        image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Distance Transform + Watershed Segmentation (cv2.distanceTransform, cv2.watershed)
        to cleanly separate touching and overlapping rice grains inside the active 150mm staging boundary.

        Returns:
            binary_mask: Cleaned binary mask of grains
            markers: Watershed segmented labeled marker mask
        """
        h, w = gray_img.shape

        # Active staging area is 150mm x 150mm centered in unwarped canvas
        center_x = w // 2
        center_y = h // 2
        stage_span_px = int(round(146.0 * pixel_per_mm))  # stay 2mm inside the 150mm border line

        x1 = max(0, center_x - stage_span_px // 2)
        y1 = max(0, center_y - stage_span_px // 2)
        x2 = min(w, center_x + stage_span_px // 2)
        y2 = min(h, center_y + stage_span_px // 2)

        roi_mask = np.zeros_like(gray_img, dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255

        # Extract inner stage pixels to compute background threshold
        stage_gray = gray_img[y1:y2, x1:x2]
        mean_bg = np.mean(stage_gray)

        if mean_bg > 120:
            # White / light paper background: invert so grains are bright foreground
            inv_gray = cv2.bitwise_not(gray_img)
            inv_stage = inv_gray[y1:y2, x1:x2]

            blurred_stage = cv2.GaussianBlur(inv_stage, (5, 5), 0)
            otsu_val, _ = cv2.threshold(blurred_stage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            blurred_full = cv2.GaussianBlur(inv_gray, (5, 5), 0)
            thresh_cutoff = max(float(otsu_val), 26.0)
            _, binary = cv2.threshold(blurred_full, thresh_cutoff, 255, cv2.THRESH_BINARY)
        else:
            # Dark background
            blurred_stage = cv2.GaussianBlur(stage_gray, (5, 5), 0)
            otsu_val, _ = cv2.threshold(blurred_stage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blurred_full = cv2.GaussianBlur(gray_img, (5, 5), 0)
            _, binary = cv2.threshold(blurred_full, otsu_val, 255, cv2.THRESH_BINARY)

        # Restrict strictly inside the active staging boundary (ignore corner ArUco markers)
        binary = cv2.bitwise_and(binary, binary, mask=roi_mask)

        # Morphological opening to filter 1-2px thin grid lines and speckle noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Morphological close to bridge internal grain gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Filter contours by realistic rice grain area
        min_grain_area_px = (1.5 * pixel_per_mm) * (0.8 * pixel_per_mm) * 0.55
        max_grain_area_px = (16.0 * pixel_per_mm) * (6.5 * pixel_per_mm)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(cleaned)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_grain_area_px <= area <= max_grain_area_px:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Distance Transform & Watershed Segmentation
        dist_transform = cv2.distanceTransform(filtered_mask, cv2.DIST_L2, 5)
        # Threshold distance transform to obtain sure grain centers
        _, sure_fg = cv2.threshold(dist_transform, 0.28 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Dilate filtered mask to obtain sure background
        sure_bg = cv2.dilate(filtered_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling for watershed
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Execute watershed
        if image_bgr is not None:
            watershed_img = image_bgr.copy()
        else:
            watershed_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        cv2.watershed(watershed_img, markers)

        return filtered_mask, markers

    def analyze_grains(
        self,
        image_bgr: np.ndarray,
        markers: np.ndarray,
        pixel_per_mm: float,
    ) -> List[GrainFeature]:
        """
        For each isolated grain contour from watershed segmentation:
        - Measures major axis length in mm using fitted ellipse / minimum bounding box.
        - Classifies as:
          - Whole (length >= 75% of variety reference length)
          - Broken (length < 75%)
        - Checks mean RGB/HSV variance for Discolouration and brightness variance for Chalkiness.
        - Prepares annotated bounding boxes (Green = Whole, Red = Broken, Yellow = Discoloured, Blue = Chalky).
        """
        hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        unique_markers = np.unique(markers)

        raw_candidates = []
        min_grain_area_px = (1.5 * pixel_per_mm) * (0.8 * pixel_per_mm)

        for m_id in unique_markers:
            # Skip background (1) and watershed boundaries (-1 or 0)
            if m_id <= 1:
                continue

            grain_mask = (markers == m_id).astype(np.uint8)
            cnts, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = max(cnts, key=cv2.contourArea)
            area_px = cv2.contourArea(cnt)
            if area_px < min_grain_area_px:
                continue

            # Dimensional measurements via fitEllipse or minAreaRect
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (axis_a, axis_b), angle = ellipse
                major_px = max(axis_a, axis_b)
                minor_px = min(axis_a, axis_b)
                box_pts = cv2.boxPoints(((cx, cy), (axis_a, axis_b), angle)).astype(int).tolist()
            else:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rect_w, rect_h), angle = rect
                major_px = max(rect_w, rect_h)
                minor_px = min(rect_w, rect_h)
                box_pts = cv2.boxPoints(rect).astype(int).tolist()

            length_mm = major_px / pixel_per_mm
            width_mm = minor_px / pixel_per_mm
            area_mm2 = area_px / (pixel_per_mm ** 2)
            aspect_ratio = length_mm / max(width_mm, 0.01)

            # Filter out unrealistic aspect ratios or dimensions
            if length_mm > 15.0 or length_mm < 1.0 or width_mm > 5.0 or width_mm < 0.5:
                continue

            # Pixel color and defect features
            grain_pixels_bgr = image_bgr[grain_mask == 1]
            grain_pixels_hsv = hsv_image[grain_mask == 1]
            grain_pixels_lab = lab_image[grain_mask == 1]

            mean_b = float(np.mean(grain_pixels_bgr[:, 0]))
            mean_g = float(np.mean(grain_pixels_bgr[:, 1]))
            mean_r = float(np.mean(grain_pixels_bgr[:, 2]))
            mean_s = float(np.mean(grain_pixels_hsv[:, 1]))  # Saturation
            mean_l = float(np.mean(grain_pixels_lab[:, 0]))  # Luminance

            raw_candidates.append({
                "length_mm": length_mm,
                "width_mm": width_mm,
                "aspect_ratio": aspect_ratio,
                "area_mm2": area_mm2,
                "cx": cx,
                "cy": cy,
                "box_pts": box_pts,
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "mean_s": mean_s,
                "mean_l": mean_l,
                "grain_pixels_lab": grain_pixels_lab,
            })

        # Auto-Detect Variety Baseline if requested
        if self.variety == "Auto-Detect" and raw_candidates:
            lengths = [c["length_mm"] for c in raw_candidates]
            # Use upper 65th percentile as proxy for intact whole kernels
            p70_len = float(np.percentile(lengths, 65))
            # Match closest known baseline:
            # Sona Masoori: 5.2mm, Basmati: 7.5mm, Swarna: 5.0mm, IR64: 6.0mm
            known = [("Swarna", 5.0), ("Sona Masoori", 5.2), ("IR64", 6.0), ("Basmati", 7.5)]
            best_match = min(known, key=lambda x: abs(x[1] - p70_len))
            self.detected_variety = best_match[0]
            self.full_length_mm = best_match[1]
            self.broken_cutoff_length_mm = self.full_length_mm * self.length_threshold_ratio

        grain_list: List[GrainFeature] = []
        grain_id = 1

        for c in raw_candidates:
            length_mm = c["length_mm"]
            width_mm = c["width_mm"]
            mean_r = c["mean_r"]
            mean_g = c["mean_g"]
            mean_b = c["mean_b"]
            mean_s = c["mean_s"]
            grain_pixels_lab = c["grain_pixels_lab"]

            # 1. Broken Grain Criteria: length < 75% of variety reference length
            is_broken = length_mm < self.broken_cutoff_length_mm

            # 2. Discolouration Criteria: elevated color variance/saturation from creamy white
            color_deviation = abs(mean_r - mean_b) + (mean_s * 0.4)
            is_discoloured = (color_deviation > self.discolour_threshold) or (mean_s > 65)

            # 3. Chalkiness Criteria: opaque milky-white endosperm with elevated luminance
            chalky_px_count = np.sum(grain_pixels_lab[:, 0] > 215)
            chalky_ratio = float(chalky_px_count) / float(max(len(grain_pixels_lab), 1))
            is_chalky = chalky_ratio >= self.chalky_threshold

            # Final Classification assignment:
            # Broken has priority in trade size grading, then Discoloured, then Chalky, then Whole
            if is_broken:
                classification = "Broken"
            elif is_discoloured:
                classification = "Discoloured"
            elif is_chalky:
                classification = "Chalky"
            else:
                classification = "Whole"

            feature = GrainFeature(
                grain_id=grain_id,
                length_mm=round(length_mm, 2),
                width_mm=round(width_mm, 2),
                aspect_ratio=round(c["aspect_ratio"], 2),
                area_mm2=round(c["area_mm2"], 2),
                center_x=round(float(c["cx"]), 1),
                center_y=round(float(c["cy"]), 1),
                mean_r=round(mean_r, 1),
                mean_g=round(mean_g, 1),
                mean_b=round(mean_b, 1),
                mean_hsv_s=round(mean_s, 1),
                chalky_ratio=round(chalky_ratio, 3),
                classification=classification,
                is_broken=is_broken,
                is_discoloured=is_discoloured,
                is_chalky=is_chalky,
                box_contour=c["box_pts"],
            )
            grain_list.append(feature)
            grain_id += 1

        return grain_list

    def draw_annotated_results(
        self,
        image_bgr: np.ndarray,
        grains: List[GrainFeature],
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Draws color-coded bounding boxes:
        - Green = Whole (length >= 75%)
        - Red = Broken (length < 75%)
        - Yellow = Discoloured
        - Blue = Chalky
        """
        annotated = image_bgr.copy()

        for g in grains:
            color = self.CLASS_COLORS.get(g.classification, (0, 200, 0))
            pts = np.array(g.box_contour, dtype=np.int32).reshape((-1, 1, 2))

            # Draw bounding box
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            # Draw centroid
            cx, cy = int(round(g.center_x)), int(round(g.center_y))
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            if draw_labels:
                label = f"#{g.grain_id} {g.length_mm}mm"
                font_scale = 0.38
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                tx1 = cx - tw // 2
                ty1 = cy - 8
                # Label backdrop for legibility
                cv2.rectangle(annotated, (tx1 - 2, ty1 - th - 2), (tx1 + tw + 2, ty1 + 2), (255, 255, 255), -1)
                cv2.rectangle(annotated, (tx1 - 2, ty1 - th - 2), (tx1 + tw + 2, ty1 + 2), color, 1)
                cv2.putText(annotated, label, (tx1, ty1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 1)

        return annotated

    def compute_summary_statistics(self, grains: List[GrainFeature]) -> Dict[str, Any]:
        """
        Calculates aggregate sample metrics and computes AGMARK Quality Grade.
        """
        total = len(grains)
        if total == 0:
            return {
                "total_grains": 0,
                "whole_count": 0,
                "broken_count": 0,
                "discoloured_count": 0,
                "chalky_count": 0,
                "whole_pct": 0.0,
                "broken_pct": 0.0,
                "discoloured_pct": 0.0,
                "chalky_pct": 0.0,
                "avg_length_mm": 0.0,
                "avg_width_mm": 0.0,
                "avg_aspect_ratio": 0.0,
                "agmark_grade": "N/A (No Grains Detected)",
                "grade_color": "gray",
                "grade_rationale": "No grains detected in the active calibration area.",
                "variety": self.detected_variety,
                "reference_length_mm": self.full_length_mm,
            }

        whole_count = sum(1 for g in grains if g.classification == "Whole")
        broken_count = sum(1 for g in grains if g.classification == "Broken")
        discoloured_count = sum(1 for g in grains if g.classification == "Discoloured")
        chalky_count = sum(1 for g in grains if g.classification == "Chalky")

        whole_pct = (whole_count / total) * 100.0
        broken_pct = (broken_count / total) * 100.0
        discoloured_pct = (discoloured_count / total) * 100.0
        chalky_pct = (chalky_count / total) * 100.0

        avg_len = float(np.mean([g.length_mm for g in grains]))
        avg_wid = float(np.mean([g.width_mm for g in grains]))
        avg_ar = float(np.mean([g.aspect_ratio for g in grains]))

        # AGMARK Standard Grading Logic for Milled / Raw Rice:
        # Grade A (Special): Broken <= 5.0%, Discoloured <= 1.0%, Chalky <= 3.0%
        # Grade B (Standard/Good): Broken <= 12.0%, Discoloured <= 3.0%, Chalky <= 6.0%
        # Grade C (Fair Commercial): Broken <= 25.0%, Discoloured <= 6.0%, Chalky <= 12.0%
        # Rejection: Broken > 25.0% or Discoloured > 6.0% or Chalky > 12.0%
        if broken_pct <= 5.0 and discoloured_pct <= 1.0 and chalky_pct <= 3.0:
            grade = "Grade A (Special / Premium)"
            grade_color = "#10B981"  # Emerald Green
            rationale = "Exceptional quality. Broken grains <= 5.0%, discoloured <= 1.0%, chalkiness <= 3.0%."
        elif broken_pct <= 12.0 and discoloured_pct <= 3.0 and chalky_pct <= 6.0:
            grade = "Grade B (Standard / Good)"
            grade_color = "#3B82F6"  # Blue
            rationale = "Commercial standard quality. Broken grains <= 12.0%, defect thresholds within permissible limits."
        elif broken_pct <= 25.0 and discoloured_pct <= 6.0 and chalky_pct <= 12.0:
            grade = "Grade C (Fair Commercial)"
            grade_color = "#F59E0B"  # Amber
            rationale = "Fair grade. Moderate broken grains (<= 25%) and slight color/chalkiness variance."
        else:
            grade = "Rejection / Sub-standard"
            grade_color = "#EF4444"  # Red
            reasons = []
            if broken_pct > 25.0:
                reasons.append(f"Broken grains ({broken_pct:.1f}%) exceed 25.0% limit")
            if discoloured_pct > 6.0:
                reasons.append(f"Discoloured grains ({discoloured_pct:.1f}%) exceed 6.0% limit")
            if chalky_pct > 12.0:
                reasons.append(f"Chalky grains ({chalky_pct:.1f}%) exceed 12.0% limit")
            rationale = "Exceeds tolerance limits: " + "; ".join(reasons) + "."

        return {
            "total_grains": total,
            "whole_count": whole_count,
            "broken_count": broken_count,
            "discoloured_count": discoloured_count,
            "chalky_count": chalky_count,
            "whole_pct": round(whole_pct, 1),
            "broken_pct": round(broken_pct, 1),
            "discoloured_pct": round(discoloured_pct, 1),
            "chalky_pct": round(chalky_pct, 1),
            "avg_length_mm": round(avg_len, 2),
            "avg_width_mm": round(avg_wid, 2),
            "avg_aspect_ratio": round(avg_ar, 2),
            "agmark_grade": grade,
            "grade_color": grade_color,
            "grade_rationale": rationale,
            "variety": self.detected_variety,
            "reference_length_mm": round(self.full_length_mm, 2),
        }

    def analyze(
        self,
        unwarped_bgr: np.ndarray,
        pixel_per_mm: float,
    ) -> Tuple[np.ndarray, List[GrainFeature], Dict[str, Any], np.ndarray]:
        """
        Complete analysis pipeline:
        segment_touching_grains -> analyze_grains -> draw_annotated_results -> compute_summary_statistics
        """
        gray_img = cv2.cvtColor(unwarped_bgr, cv2.COLOR_BGR2GRAY)
        binary_mask, markers = self.segment_touching_grains(gray_img, pixel_per_mm, image_bgr=unwarped_bgr)
        grains = self.analyze_grains(unwarped_bgr, markers, pixel_per_mm)
        annotated_img = self.draw_annotated_results(unwarped_bgr, grains)
        stats = self.compute_summary_statistics(grains)
        return annotated_img, grains, stats, binary_mask
