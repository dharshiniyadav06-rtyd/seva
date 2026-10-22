"""
Farmgate AI - Realistic Synthetic Sample Photo Generator
Simulates a smartphone photo taken at an angle with perspective distortion,
shadows, and realistic rice grains (whole, broken, discoloured, chalky).
Used for instant testing and demonstration without needing a physical camera.
"""

import os
import random
import numpy as np
import cv2
from generate_sheet import generate_calibration_sheet, MARKER_SPAN_MM, ACTIVE_SIZE_MM


def create_synthetic_rice_kernel(
    length_px: float,
    width_px: float,
    angle_deg: float,
    kernel_type: str = "Whole",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes a realistic rice grain kernel with natural texture and color.
    Types: "Whole", "Broken", "Discoloured", "Chalky".
    """
    pad = int(max(length_px, width_px) * 1.5) + 10
    h = w = pad * 2
    cx, cy = pad, pad

    # Base canvas
    grain_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    grain_mask = np.zeros((h, w), dtype=np.uint8)

    # Base ellipse contour
    axes = (int(round(length_px / 2.0)), int(round(width_px / 2.0)))
    cv2.ellipse(grain_mask, (cx, cy), axes, angle_deg, 0, 360, 255, -1)

    # If broken, cut one end cleanly with a random fracture line
    if kernel_type == "Broken":
        rad = np.radians(angle_deg)
        # Vector along length
        dx = np.cos(rad) * (length_px * 0.25)
        dy = np.sin(rad) * (length_px * 0.25)
        cut_cx = int(round(cx + dx))
        cut_cy = int(round(cy + dy))
        # Cut rect
        cut_w = int(length_px)
        cut_h = int(width_px * 3)
        cut_rect = ((cut_cx, cut_cy), (cut_w, cut_h), angle_deg + 85 + random.uniform(-10, 10))
        cut_box = cv2.boxPoints(cut_rect).astype(np.int32)
        cv2.fillPoly(grain_mask, [cut_box], 0)

    # Determine base grain color
    if kernel_type == "Discoloured":
        # Amber / brownish / yellow / red streaks
        base_b = random.randint(80, 130)
        base_g = random.randint(140, 185)
        base_r = random.randint(190, 235)
    else:
        # Healthy polished translucent cream/pearl white
        base_b = random.randint(195, 215)
        base_g = random.randint(215, 230)
        base_r = random.randint(225, 240)

    # Apply base color
    grain_bgr[grain_mask > 0] = [base_b, base_g, base_r]

    # Add realistic subtle noise & gradient texture
    noise = np.random.normal(0, 7, (h, w, 3)).astype(np.int16)
    temp = np.clip(grain_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    grain_bgr = np.where(grain_mask[:, :, None] > 0, temp, 0)

    # If chalky, create an opaque milky-white patch in center/belly
    if kernel_type == "Chalky":
        chalk_mask = np.zeros((h, w), dtype=np.uint8)
        chalk_axes = (int(round(axes[0] * random.uniform(0.4, 0.65))), int(round(axes[1] * random.uniform(0.5, 0.75))))
        cv2.ellipse(chalk_mask, (cx, cy), chalk_axes, angle_deg + random.uniform(-5, 5), 0, 360, 255, -1)
        chalk_mask = cv2.bitwise_and(chalk_mask, grain_mask)
        chalk_mask = cv2.GaussianBlur(chalk_mask, (5, 5), 0)

        # Very bright milky white
        chalk_bgr = np.zeros_like(grain_bgr)
        chalk_bgr[:] = [245, 250, 255]
        alpha = (chalk_mask.astype(float) / 255.0)[:, :, None] * 0.85
        grain_bgr = (grain_bgr * (1.0 - alpha) + chalk_bgr * alpha).astype(np.uint8)

    # Create subtle contact shadow under grain
    shadow_mask = cv2.dilate(grain_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0)
    # Shift shadow slightly
    m_shift = np.float32([[1, 0, 3], [0, 1, 3]])
    shadow_shifted = cv2.warpAffine(shadow_mask, m_shift, (w, h))

    return grain_bgr, grain_mask, shadow_shifted


def generate_sample_smartphone_capture(
    output_path: str = "sample_grain_capture.jpg",
    num_grains: int = 48,
    add_perspective_tilt: bool = True,
    add_lighting_gradient: bool = True,
    variety: str = "Sona Masoori",
) -> str:
    """
    Generates a full synthetic smartphone photo of the calibration sheet
    with distributed rice kernels, simulated camera tilt, and flash/lighting variations.
    """
    # 1. Generate base calibration sheet at 150 DPI for fast high-quality simulation
    sheet_tmp_path = "temp_sheet.png"
    dpi = 150
    generate_calibration_sheet(sheet_tmp_path, dpi=dpi)
    sheet_bgr = cv2.imread(sheet_tmp_path)
    os.remove(sheet_tmp_path)

    h, w, _ = sheet_bgr.shape
    px_per_mm = dpi / 25.4
    center_x = w // 2
    center_y = h // 2

    # Active area boundaries (150mm x 150mm)
    active_span_px = int(round(135.0 * px_per_mm))  # stay slightly inside 150mm
    x_min = center_x - active_span_px // 2
    x_max = center_x + active_span_px // 2
    y_min = center_y - active_span_px // 2
    y_max = center_y + active_span_px // 2

    # Mix of grains: ~65% Whole, ~18% Broken, ~9% Discoloured, ~8% Chalky
    grain_types = (
        ["Whole"] * int(num_grains * 0.65)
        + ["Broken"] * int(num_grains * 0.18)
        + ["Discoloured"] * int(num_grains * 0.09)
        + ["Chalky"] * int(num_grains * 0.08)
    )
    random.shuffle(grain_types)

    # Variety-specific dimension baselines
    variety_lengths = {
        "Sona Masoori": 5.2,
        "Basmati": 7.5,
        "Swarna": 5.0,
        "IR64": 6.0,
        "Auto-Detect": 5.5,
    }
    base_len_mm = variety_lengths.get(variety, 5.2)
    broken_cutoff = base_len_mm * 0.75

    placed_positions = []

    for k_type in grain_types:
        if k_type == "Broken":
            len_mm = random.uniform(base_len_mm * 0.35, broken_cutoff * 0.95)
        else:
            len_mm = random.uniform(broken_cutoff * 1.05, base_len_mm * 1.15)

        wid_mm = random.uniform(base_len_mm * 0.28, base_len_mm * 0.38)
        len_px = len_mm * px_per_mm
        wid_px = wid_mm * px_per_mm
        angle = random.uniform(0, 360)

        # Find a non-overlapping or lightly touching position
        for _ in range(30):
            gx = random.randint(x_min, x_max)
            gy = random.randint(y_min, y_max)
            # Distance from already placed grains
            min_dist = min([np.hypot(gx - px, gy - py) for px, py in placed_positions], default=999)
            if min_dist > (len_px * 0.85):
                placed_positions.append((gx, gy))
                break
        else:
            placed_positions.append((gx, gy))

        # Generate kernel
        kernel_bgr, k_mask, k_shadow = create_synthetic_rice_kernel(len_px, wid_px, angle, k_type)
        kh, kw, _ = kernel_bgr.shape
        kx1 = gx - kw // 2
        ky1 = gy - kh // 2
        kx2 = kx1 + kw
        ky2 = ky1 + kh

        if kx1 < 0 or ky1 < 0 or kx2 >= w or ky2 >= h:
            continue

        # Composite contact shadow
        bg_roi = sheet_bgr[ky1:ky2, kx1:kx2]
        shadow_factor = 1.0 - (k_shadow.astype(float) / 255.0) * 0.35
        bg_roi = (bg_roi * shadow_factor[:, :, None]).astype(np.uint8)

        # Composite grain kernel
        mask_f = (k_mask.astype(float) / 255.0)[:, :, None]
        blended = (bg_roi * (1.0 - mask_f) + kernel_bgr * mask_f).astype(np.uint8)
        sheet_bgr[ky1:ky2, kx1:kx2] = blended

    # 2. Add realistic smartphone lighting gradient (vignette / warm incandescent cast)
    if add_lighting_gradient:
        # Create subtle 2D radial light gradient (phone flash in upper-center)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        flash_cx = int(w * 0.45)
        flash_cy = int(h * 0.40)
        dist_sq = ((x_coords - flash_cx) ** 2 + (y_coords - flash_cy) ** 2) / (w ** 2)
        light_gain = 1.05 - 0.22 * dist_sq  # falloff towards corners

        # Slight warm color temperature (boost Red/Green, attenuate Blue slightly)
        b_gain = light_gain * 0.96
        g_gain = light_gain * 1.00
        r_gain = light_gain * 1.03

        b = np.clip(sheet_bgr[:, :, 0] * b_gain, 0, 255).astype(np.uint8)
        g = np.clip(sheet_bgr[:, :, 1] * g_gain, 0, 255).astype(np.uint8)
        r = np.clip(sheet_bgr[:, :, 2] * r_gain, 0, 255).astype(np.uint8)
        sheet_bgr = cv2.merge([b, g, r])

    # 3. Simulate realistic perspective tilt (smartphone held at an angle)
    if add_perspective_tilt:
        # Source rectangle corners
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # Tilted phone camera view (trapezoidal distortion)
        dx_top = int(w * 0.08)
        dy_top = int(h * 0.06)
        dx_bot = int(w * 0.04)
        dy_bot = int(h * 0.03)

        dst = np.array(
            [
                [dx_top + random.randint(-10, 10), dy_top + random.randint(-10, 10)],
                [w - dx_top + random.randint(-10, 10), dy_top + random.randint(-10, 10)],
                [w - dx_bot + random.randint(-10, 10), h - dy_bot + random.randint(-10, 10)],
                [dx_bot + random.randint(-10, 10), h - dy_bot + random.randint(-10, 10)],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src, dst)
        # Warp with neutral desk background (wood/desk texture color)
        desk_bg = (180, 190, 200)
        tilted = cv2.warpPerspective(sheet_bgr, M, (w, h), borderValue=desk_bg)
        sheet_bgr = tilted

    # Save realistic photo
    cv2.imwrite(output_path, sheet_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[Farmgate AI] Synthetic sample photo generated: {output_path} ({num_grains} grains)")
    return output_path


if __name__ == "__main__":
    generate_sample_smartphone_capture()
