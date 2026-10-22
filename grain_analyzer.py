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
    is_uncertain: bool = False


# Pipeline Defaults & Thresholds
DEFAULT_THRESHOLD_METHOD = "otsu"  # "otsu" or "adaptive"
DEFAULT_BLUR_KSIZE = 3
MIN_GRAIN_AREA_MM2 = 1.5           # Minimum grain area in mm^2 (filters dust/noise)
MAX_GRAIN_AREA_MM2 = 50.0          # Maximum grain area in mm^2 (filters artifacts/lines)
DIST_PEAK_RATIO = 0.28             # Relative threshold on distance transform peaks for seed separation
DIST_SMOOTH_SIGMA = 1.0            # Gaussian smoothing sigma for distance transform to avoid spine over-segmentation


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

    # Feature 4: Variety Baseline Profiles & AGMARK Grade Threshold Defaults
    # Explicitly identifies intact length (mm), expected aspect ratio, AGMARK defaults, and estimates.
    VARIETY_BASELINES = {
        "Basmati": {
            "full_length_mm": 7.5,
            "expected_aspect_ratio": 3.6,
            "agmark_defaults": {
                "grade_a_broken_max": 5.0,
                "grade_b_broken_max": 10.0,
                "grade_c_broken_max": 20.0,
            },
            "is_estimate": False,
            "name_ta": "பாசுமதி",
        },
        "Sona Masoori": {
            "full_length_mm": 5.2,
            "expected_aspect_ratio": 2.6,
            "agmark_defaults": {
                "grade_a_broken_max": 5.0,
                "grade_b_broken_max": 12.0,
                "grade_c_broken_max": 25.0,
            },
            "is_estimate": False,
            "name_ta": "சோனா மசூரி",
        },
        "Swarna": {
            "full_length_mm": 5.0,
            "expected_aspect_ratio": 2.3,
            "agmark_defaults": {
                "grade_a_broken_max": 6.0,
                "grade_b_broken_max": 15.0,
                "grade_c_broken_max": 25.0,
            },
            "is_estimate": False,
            "name_ta": "ஸ்வர்ணா",
        },
        "Custom/Auto": {
            "full_length_mm": 6.6,
            "expected_aspect_ratio": 3.0,
            "agmark_defaults": {
                "grade_a_broken_max": 5.0,
                "grade_b_broken_max": 12.0,
                "grade_c_broken_max": 25.0,
            },
            "is_estimate": True,
            "estimation_note": "Intact reference length estimated dynamically from 75th percentile of sample length distribution.",
            "name_ta": "விருப்பம் / தானியங்கி",
        },
        "IR64": {
            "full_length_mm": 6.0,
            "expected_aspect_ratio": 3.0,
            "agmark_defaults": {
                "grade_a_broken_max": 5.0,
                "grade_b_broken_max": 12.0,
                "grade_c_broken_max": 25.0,
            },
            "is_estimate": False,
            "name_ta": "ஐஆர்64",
        },
        "Long Grain Rice (Default)": {
            "full_length_mm": 6.6,
            "expected_aspect_ratio": 3.1,
            "agmark_defaults": {
                "grade_a_broken_max": 5.0,
                "grade_b_broken_max": 12.0,
                "grade_c_broken_max": 25.0,
            },
            "is_estimate": False,
            "name_ta": "நீண்ட தானிய அரிசி (இயல்பு)",
        },
        "Auto-Detect": {
            "full_length_mm": 6.6,
            "expected_aspect_ratio": 3.0,
            "agmark_defaults": {
                "grade_a_broken_max": 5.0,
                "grade_b_broken_max": 12.0,
                "grade_c_broken_max": 25.0,
            },
            "is_estimate": True,
            "estimation_note": "Intact reference length estimated dynamically from 75th percentile of sample length distribution.",
            "name_ta": "தானியங்கி கண்டறிதல்",
        },
    }

    # BGR colors for visualization:
    # Green = Whole, Red = Broken, Yellow = Discoloured, Blue = Chalky, Orange = Uncertain
    CLASS_COLORS = {
        "Whole": (0, 200, 0),          # Green
        "Broken": (0, 0, 230),         # Red
        "Uncertain": (0, 165, 255),    # Orange / Amber
        "Discoloured": (0, 215, 255),  # Yellow
        "Chalky": (230, 140, 20),      # Blue / Cyan
    }

    def __init__(
        self,
        variety: str = "Long Grain Rice (Default)",
        length_threshold_ratio: float = 0.75,
        discolour_threshold: float = 38.0,
        chalky_threshold: float = 0.22,
        threshold_method: str = DEFAULT_THRESHOLD_METHOD,
        min_grain_area_mm2: float = MIN_GRAIN_AREA_MM2,
        max_grain_area_mm2: float = MAX_GRAIN_AREA_MM2,
        dist_peak_ratio: float = DIST_PEAK_RATIO,
        dist_smooth_sigma: float = DIST_SMOOTH_SIGMA,
    ):
        """
        Args:
            variety: Name from VARIETY_BASELINES ("Basmati", "Sona Masoori", "Swarna", "Custom/Auto", "IR64", "Long Grain Rice (Default)")
            length_threshold_ratio: Minimum ratio of expected length to be deemed 'Whole' (default: 0.75 = 75%, range: 0.60 - 0.85)
            discolour_threshold: Color deviation / saturation cutoff for discolouration
            chalky_threshold: Fractional area ratio of opaque white chalkiness inside kernel
            threshold_method: Binarization method ("otsu" or "adaptive")
            min_grain_area_mm2: Minimum area cutoff in mm^2 to eliminate noise/dust
            max_grain_area_mm2: Maximum area cutoff in mm^2 to eliminate calibration lines
            dist_peak_ratio: Relative peak height ratio on distance transform
            dist_smooth_sigma: Gaussian smoothing sigma on distance transform
        """
        # Normalize variety selection (handle aliases)
        v_key = variety
        if v_key not in self.VARIETY_BASELINES:
            if v_key in ["Custom", "Auto", "Auto-Detect"]:
                v_key = "Custom/Auto"
            else:
                v_key = "Long Grain Rice (Default)"

        self.variety = v_key
        v_profile = self.VARIETY_BASELINES[self.variety]
        self.full_length_mm = float(v_profile["full_length_mm"])
        self.expected_aspect_ratio = float(v_profile.get("expected_aspect_ratio", 3.0))
        self.agmark_defaults = v_profile.get("agmark_defaults", {"grade_a_broken_max": 5.0, "grade_b_broken_max": 12.0, "grade_c_broken_max": 25.0})
        self.is_estimate = bool(v_profile.get("is_estimate", False))
        self.is_reference_estimated = self.is_estimate
        self.estimation_note = str(v_profile.get("estimation_note", ""))

        self.length_threshold_ratio = float(length_threshold_ratio)
        self.broken_cutoff_length_mm = self.full_length_mm * self.length_threshold_ratio
        self.discolour_threshold = discolour_threshold
        self.chalky_threshold = chalky_threshold
        self.threshold_method = threshold_method.lower() if threshold_method in ["otsu", "adaptive"] else "otsu"
        self.min_grain_area_mm2 = min_grain_area_mm2
        self.max_grain_area_mm2 = max_grain_area_mm2
        self.dist_peak_ratio = dist_peak_ratio
        self.dist_smooth_sigma = dist_smooth_sigma
        self.detected_variety = self.variety

    def segment_touching_grains(
        self,
        gray_img: np.ndarray,
        pixel_per_mm: float = 4.88,
        image_bgr: Optional[np.ndarray] = None,
        threshold_method: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature 3: Distance Transform + Watershed Segmentation Pipeline:
        1. Staging ROI Isolation (146mm x 146mm centered boundary)
        2. Gaussian Blur (5x5 kernel)
        3. Parameterized Thresholding (Otsu or Adaptive)
        4. Morphological Open / Close to remove dust & bridge voids
        5. Area Filtering in real mm^2 (min_grain_area_mm2 to max_grain_area_mm2)
        6. Distance Transform (cv2.DIST_L2, 5)
        7. Gaussian-smoothed Local Peak Detection to separate side-touching grains
        8. Marker Seed Generation via Connected Components
        9. Topological Watershed Segmentation on image canvas

        Returns:
            binary_mask: Cleaned binary mask of grains
            markers: Watershed segmented labeled marker mask
        """
        h, w = gray_img.shape
        t_method = (threshold_method or self.threshold_method).lower()

        # Active staging area is 146mm x 146mm centered in unwarped canvas
        center_x = w // 2
        center_y = h // 2
        stage_span_px = int(round(146.0 * pixel_per_mm))

        x1 = max(0, center_x - stage_span_px // 2)
        y1 = max(0, center_y - stage_span_px // 2)
        x2 = min(w, center_x + stage_span_px // 2)
        y2 = min(h, center_y + stage_span_px // 2)

        roi_mask = np.zeros_like(gray_img, dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255

        stage_gray = gray_img[y1:y2, x1:x2]
        mean_bg = np.mean(stage_gray)

        # Step 2 & 3: Blur & Parameterized Thresholding (Otsu or Adaptive)
        if mean_bg > 120:
            # White / light paper background: grains are darker than background
            if t_method == "adaptive":
                blurred_stage = cv2.GaussianBlur(stage_gray, (DEFAULT_BLUR_KSIZE, DEFAULT_BLUR_KSIZE), 0)
                binary_stage = cv2.adaptiveThreshold(
                    blurred_stage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
                )
                binary = np.zeros_like(gray_img)
                binary[y1:y2, x1:x2] = binary_stage
            else:
                # Otsu thresholding on inverted stage
                inv_gray = cv2.bitwise_not(gray_img)
                inv_stage = inv_gray[y1:y2, x1:x2]
                blurred_stage = cv2.GaussianBlur(inv_stage, (DEFAULT_BLUR_KSIZE, DEFAULT_BLUR_KSIZE), 0)
                otsu_val, _ = cv2.threshold(blurred_stage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                blurred_full = cv2.GaussianBlur(inv_gray, (DEFAULT_BLUR_KSIZE, DEFAULT_BLUR_KSIZE), 0)
                thresh_cutoff = max(float(otsu_val), 26.0)
                _, binary = cv2.threshold(blurred_full, thresh_cutoff, 255, cv2.THRESH_BINARY)
        else:
            # Dark background: grains are brighter than background
            blurred_stage = cv2.GaussianBlur(stage_gray, (DEFAULT_BLUR_KSIZE, DEFAULT_BLUR_KSIZE), 0)
            if t_method == "adaptive":
                binary_stage = cv2.adaptiveThreshold(
                    blurred_stage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
                )
                binary = np.zeros_like(gray_img)
                binary[y1:y2, x1:x2] = binary_stage
            else:
                otsu_val, _ = cv2.threshold(blurred_stage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                blurred_full = cv2.GaussianBlur(gray_img, (DEFAULT_BLUR_KSIZE, DEFAULT_BLUR_KSIZE), 0)
                _, binary = cv2.threshold(blurred_full, otsu_val, 255, cv2.THRESH_BINARY)

        # Restrict strictly inside the active staging boundary
        binary = cv2.bitwise_and(binary, binary, mask=roi_mask)

        # Step 4: Morphological opening & closing to clean noise & fill small voids
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Step 5: Filter blobs using real physical area in mm^2 (not arbitrary pixel counts)
        min_grain_area_px = self.min_grain_area_mm2 * (pixel_per_mm ** 2)
        max_grain_area_px = self.max_grain_area_mm2 * (pixel_per_mm ** 2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(cleaned)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_grain_area_px <= area <= max_grain_area_px:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Step 6: Distance Transform
        dist_transform = cv2.distanceTransform(filtered_mask, cv2.DIST_L2, 5)

        # Step 7: Smooth distance transform & detect local peaks for touching grain separation
        if dist_transform.max() > 1.5:
            dist_smooth = cv2.GaussianBlur(dist_transform, (0, 0), self.dist_smooth_sigma)
            # Dilation peak detector (3x3 kernel resolves touching grains with fine contact grooves)
            dilated = cv2.dilate(dist_smooth, np.ones((3, 3), np.uint8))
            peaks = (
                (dist_smooth == dilated)
                & (dist_smooth >= self.dist_peak_ratio * dist_transform.max())
                & (dist_transform >= 1.2)
            )
            sure_fg = cv2.morphologyEx(np.uint8(peaks) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            if np.sum(sure_fg) == 0:
                _, sure_fg = cv2.threshold(dist_transform, 0.28 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
        else:
            sure_fg = np.zeros_like(filtered_mask)

        # Step 8: Marker labeling for watershed
        sure_bg = cv2.dilate(filtered_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Step 9: Execute watershed
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
        min_grain_area_px = self.min_grain_area_mm2 * (pixel_per_mm ** 2)

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

        # Feature 4: Custom/Auto Variety Baseline Estimation
        if self.variety in ["Custom/Auto", "Auto-Detect"] and raw_candidates:
            lengths = [c["length_mm"] for c in raw_candidates]
            # Estimate intact reference length from the upper 75th percentile (assuming majority intact)
            p75_len = float(np.percentile(lengths, 75))
            self.full_length_mm = round(p75_len, 2)
            self.broken_cutoff_length_mm = self.full_length_mm * self.length_threshold_ratio
            self.is_reference_estimated = True
            self.estimation_note = f"Intact reference length ({self.full_length_mm:.2f} mm) estimated dynamically from 75th percentile of sample ({len(lengths)} grains)."

        grain_list: List[GrainFeature] = []
        grain_id = 1

        cutoff = self.broken_cutoff_length_mm
        uncertain_margin = 0.05 * cutoff  # +/- 5% boundary uncertainty

        for c in raw_candidates:
            length_mm = c["length_mm"]
            width_mm = c["width_mm"]
            mean_r = c["mean_r"]
            mean_g = c["mean_g"]
            mean_b = c["mean_b"]
            mean_s = c["mean_s"]
            grain_pixels_lab = c["grain_pixels_lab"]

            # Feature 4: Whole vs. Broken Classification with Edge Uncertainty
            is_broken = length_mm < cutoff
            is_uncertain = abs(length_mm - cutoff) <= uncertain_margin

            # Color and Quality Defect Classification
            color_deviation = abs(mean_r - mean_b) + (mean_s * 0.4)
            is_discoloured = (color_deviation > self.discolour_threshold) or (mean_s > 65)

            chalky_px_count = np.sum(grain_pixels_lab[:, 0] > 215)
            chalky_ratio = float(chalky_px_count) / float(max(len(grain_pixels_lab), 1))
            is_chalky = chalky_ratio >= self.chalky_threshold

            # Final classification tag
            if is_uncertain:
                classification = "Uncertain"
            elif is_broken:
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
                is_uncertain=is_uncertain,
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
            if g.is_uncertain:
                color = self.CLASS_COLORS["Uncertain"]
            else:
                color = self.CLASS_COLORS.get(g.classification, (0, 200, 0))

            pts = np.array(g.box_contour, dtype=np.int32).reshape((-1, 1, 2))

            # Draw bounding box
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            # Draw centroid
            cx, cy = int(round(g.center_x)), int(round(g.center_y))
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            if draw_labels:
                unc_tag = " (UNC)" if g.is_uncertain else ""
                label = f"#{g.grain_id} {g.length_mm}mm{unc_tag}"
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
        Feature 4: Includes mean length, length std dev, length histogram, uncertain counts,
        and dynamically estimated reference metadata.
        """
        total = len(grains)
        if total == 0:
            return {
                "total_grains": 0,
                "whole_count": 0,
                "broken_count": 0,
                "uncertain_count": 0,
                "discoloured_count": 0,
                "chalky_count": 0,
                "whole_pct": 0.0,
                "broken_pct": 0.0,
                "uncertain_pct": 0.0,
                "discoloured_pct": 0.0,
                "chalky_pct": 0.0,
                "avg_length_mm": 0.0,
                "length_std_mm": 0.0,
                "avg_width_mm": 0.0,
                "avg_aspect_ratio": 0.0,
                "length_histogram": {"counts": [], "bin_edges": []},
                "agmark_grade": "N/A (No Grains Detected)",
                "grade_color": "gray",
                "grade_rationale": "No grains detected in the active calibration area.",
                "variety": self.detected_variety,
                "reference_length_mm": round(self.full_length_mm, 2),
                "broken_cutoff_length_mm": round(self.broken_cutoff_length_mm, 2),
                "threshold_fraction": round(self.length_threshold_ratio, 2),
                "is_reference_estimated": self.is_reference_estimated,
                "estimation_note": self.estimation_note,
            }

        lengths = [g.length_mm for g in grains]
        widths = [g.width_mm for g in grains]
        aspect_ratios = [g.aspect_ratio for g in grains]

        whole_count = sum(1 for g in grains if g.classification == "Whole")
        broken_count = sum(1 for g in grains if g.classification == "Broken")
        uncertain_count = sum(1 for g in grains if g.is_uncertain)
        discoloured_count = sum(1 for g in grains if g.classification == "Discoloured")
        chalky_count = sum(1 for g in grains if g.classification == "Chalky")

        whole_pct = (whole_count / total) * 100.0
        broken_pct = (broken_count / total) * 100.0
        uncertain_pct = (uncertain_count / total) * 100.0
        discoloured_pct = (discoloured_count / total) * 100.0
        chalky_pct = (chalky_count / total) * 100.0

        avg_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))
        avg_wid = float(np.mean(widths))
        avg_ar = float(np.mean(aspect_ratios))

        # 10-bin distribution histogram
        hist_counts, bin_edges = np.histogram(lengths, bins=10)
        length_histogram = {
            "counts": hist_counts.tolist(),
            "bin_edges": [round(float(b), 2) for b in bin_edges],
        }

        # AGMARK Standard Grading Logic for Milled / Raw Rice:
        # Grade A: Broken <= 5.0%, Discoloured <= 1.0%, Chalky <= 3.0%
        # Grade B: Broken <= 12.0%, Discoloured <= 3.0%, Chalky <= 6.0%
        # Grade C: Broken <= 25.0%, Discoloured <= 6.0%, Chalky <= 12.0%
        # Rejection: Exceeds Grade C limits
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

        if self.is_reference_estimated and self.estimation_note:
            rationale += f" [Note: {self.estimation_note}]"

        return {
            "total_grains": total,
            "whole_count": whole_count,
            "broken_count": broken_count,
            "uncertain_count": uncertain_count,
            "discoloured_count": discoloured_count,
            "chalky_count": chalky_count,
            "whole_pct": round(whole_pct, 1),
            "broken_pct": round(broken_pct, 1),
            "uncertain_pct": round(uncertain_pct, 1),
            "discoloured_pct": round(discoloured_pct, 1),
            "chalky_pct": round(chalky_pct, 1),
            "avg_length_mm": round(avg_len, 2),
            "length_std_mm": round(std_len, 2),
            "avg_width_mm": round(avg_wid, 2),
            "avg_aspect_ratio": round(avg_ar, 2),
            "length_histogram": length_histogram,
            "agmark_grade": grade,
            "grade_color": grade_color,
            "grade_rationale": rationale,
            "variety": self.detected_variety,
            "reference_length_mm": round(self.full_length_mm, 2),
            "broken_cutoff_length_mm": round(self.broken_cutoff_length_mm, 2),
            "threshold_fraction": round(self.length_threshold_ratio, 2),
            "uncertainty_range_mm": [
                round(self.broken_cutoff_length_mm * 0.95, 2),
                round(self.broken_cutoff_length_mm * 1.05, 2),
            ],
            "is_reference_estimated": self.is_reference_estimated,
            "estimation_note": self.estimation_note,
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
