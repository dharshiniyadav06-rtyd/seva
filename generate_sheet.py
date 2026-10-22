"""
Farmgate AI - Printable Calibration Sheet Generator
Generates a high-resolution (300 DPI, A4 size) PNG/PDF calibration sheet
with ArUco markers, active grain area, color blocks, and millimeter scale bars.
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# Physical Dimensions (in millimeters)
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DPI = 300
MM_TO_INCH = 1.0 / 25.4
PX_PER_MM = (DPI * MM_TO_INCH)  # approx 11.811 pixels per mm

# Sheet Geometry
SHEET_WIDTH_PX = int(round(A4_WIDTH_MM * PX_PER_MM))    # 2480 px
SHEET_HEIGHT_PX = int(round(A4_HEIGHT_MM * PX_PER_MM))  # 3508 px

# Calibration Grid Specifications
ACTIVE_SIZE_MM = 150.0  # 150 mm x 150 mm active staging area
ACTIVE_SIZE_PX = int(round(ACTIVE_SIZE_MM * PX_PER_MM))
GRID_STEP_MM = 10.0     # 10 mm grid lines
GRID_STEP_PX = int(round(GRID_STEP_MM * PX_PER_MM))

# ArUco Marker Specifications
MARKER_SIZE_MM = 22.0   # 22 mm x 22 mm marker
MARKER_SIZE_PX = int(round(MARKER_SIZE_MM * PX_PER_MM))
MARKER_SPAN_MM = 172.0  # Center-to-center span between opposite corners (X and Y)
MARKER_SPAN_PX = int(round(MARKER_SPAN_MM * PX_PER_MM))

# Reference Color Swatches (RGB) for White-Balance & Glare Normalization
COLOR_BLOCKS = [
    ("White (100%)", (255, 255, 255), (200, 200, 200)),  # name, fill RGB, outline RGB
    ("50% Gray", (128, 128, 128), (80, 80, 80)),
    ("Black (0%)", (0, 0, 0), (0, 0, 0)),
    ("Pure Red", (255, 0, 0), (180, 0, 0)),
    ("Pure Green", (0, 255, 0), (0, 180, 0)),
    ("Pure Blue", (0, 0, 255), (0, 0, 180)),
]


def create_aruco_marker(marker_id: int, size_px: int) -> np.ndarray:
    """
    Generate an ArUco marker image using DICT_4X4_50.
    Supports both OpenCV 4.7+ (cv2.aruco.ArucoDetector/generateImageMarker)
    and older versions (drawMarker).
    """
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    try:
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    except AttributeError:
        marker_img = np.zeros((size_px, size_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, size_px, marker_img, 1)

    return marker_img


def generate_calibration_sheet(output_path: str = "calibration_sheet_A4.png", dpi: int = 300) -> str:
    """
    Generates a full A4 300 DPI calibration sheet with ArUco markers,
    active boundary area, millimeter scale bars, and color calibration blocks.
    """
    px_per_mm = dpi * MM_TO_INCH
    sheet_w = int(round(A4_WIDTH_MM * px_per_mm))
    sheet_h = int(round(A4_HEIGHT_MM * px_per_mm))
    center_x = sheet_w // 2
    center_y = sheet_h // 2

    # Create pure white RGB canvas
    canvas = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Load default system fonts or fall back to PIL basic font
    try:
        title_font = ImageFont.truetype("arialbd.ttf", int(round(36 * (dpi / 300))))
        subtitle_font = ImageFont.truetype("arial.ttf", int(round(20 * (dpi / 300))))
        label_font = ImageFont.truetype("arialbd.ttf", int(round(18 * (dpi / 300))))
        small_font = ImageFont.truetype("arial.ttf", int(round(14 * (dpi / 300))))
        tiny_font = ImageFont.truetype("arial.ttf", int(round(11 * (dpi / 300))))
    except IOError:
        title_font = subtitle_font = label_font = small_font = tiny_font = ImageFont.load_default()

    # 1. Header Banner & Instructions
    header_y = int(round(20 * px_per_mm))
    title_text = "FARMGATE AI — GRAIN QUALITY CALIBRATION SHEET"
    draw.text((center_x, header_y), title_text, fill=(20, 35, 60), font=title_font, anchor="mm")

    subtitle_text = "Standard ISO A4 (210 × 297 mm) @ 300 DPI  |  DICT_4X4_50 Fiducial Calibration System"
    draw.text((center_x, header_y + int(round(7 * px_per_mm))), subtitle_text, fill=(80, 90, 105), font=subtitle_font, anchor="mm")

    warn_text = "IMPORTANT: Print at 100% scale (Do NOT select 'Fit to Printable Area'). Verify scale with ruler."
    draw.text((center_x, header_y + int(round(13 * px_per_mm))), warn_text, fill=(180, 40, 30), font=small_font, anchor="mm")

    # 2. Central Active Boundary (150mm x 150mm)
    active_size_px = int(round(ACTIVE_SIZE_MM * px_per_mm))
    active_x1 = center_x - active_size_px // 2
    active_y1 = center_y - active_size_px // 2
    active_x2 = active_x1 + active_size_px
    active_y2 = active_y1 + active_size_px

    # Light background fill for active area to differentiate stage
    draw.rectangle([active_x1, active_y1, active_x2, active_y2], fill=(253, 253, 254), outline=(40, 40, 40), width=3)

    # Light 10mm x 10mm grid inside active area
    grid_step_px = int(round(GRID_STEP_MM * px_per_mm))
    for x in range(active_x1 + grid_step_px, active_x2, grid_step_px):
        draw.line([(x, active_y1), (x, active_y2)], fill=(230, 233, 238), width=1)
    for y in range(active_y1 + grid_step_px, active_y2, grid_step_px):
        draw.line([(active_x1, y), (active_x2, y)], fill=(230, 233, 238), width=1)

    # Active Area Center Crosshair
    ch_len = int(round(8 * px_per_mm))
    draw.line([(center_x - ch_len, center_y), (center_x + ch_len, center_y)], fill=(120, 130, 150), width=2)
    draw.line([(center_x, center_y - ch_len), (center_x, center_y + ch_len)], fill=(120, 130, 150), width=2)

    # Active Area Label
    stage_label = "ACTIVE GRAIN ASSAYING STAGE (150 mm × 150 mm)"
    draw.text((center_x, active_y1 + int(round(7 * px_per_mm))), stage_label, fill=(100, 115, 130), font=label_font, anchor="mm")
    sub_stage_label = "Spread single layer of grains inside this boundary. Ensure grains do not overlap."
    draw.text((center_x, active_y1 + int(round(12 * px_per_mm))), sub_stage_label, fill=(130, 140, 155), font=tiny_font, anchor="mm")

    # 3. Precision Millimeter Scale Ticks along Active Border
    tick_len_5mm = int(round(2.5 * px_per_mm))
    tick_len_10mm = int(round(4.5 * px_per_mm))
    for i in range(0, int(ACTIVE_SIZE_MM) + 1):
        pos_offset = int(round(i * px_per_mm))
        t_len = tick_len_10mm if (i % 10 == 0) else (tick_len_5mm if (i % 5 == 0) else int(round(1.5 * px_per_mm)))
        
        # Top border ticks
        draw.line([(active_x1 + pos_offset, active_y1), (active_x1 + pos_offset, active_y1 - t_len)], fill=(50, 50, 50), width=1)
        # Bottom border ticks
        draw.line([(active_x1 + pos_offset, active_y2), (active_x1 + pos_offset, active_y2 + t_len)], fill=(50, 50, 50), width=1)
        # Left border ticks
        draw.line([(active_x1, active_y1 + pos_offset), (active_x1 - t_len, active_y1 + pos_offset)], fill=(50, 50, 50), width=1)
        # Right border ticks
        draw.line([(active_x2, active_y1 + pos_offset), (active_x2 + t_len, active_y1 + pos_offset)], fill=(50, 50, 50), width=1)

        # Scale labels every 20mm
        if i % 20 == 0 and i > 0 and i < ACTIVE_SIZE_MM:
            draw.text((active_x1 + pos_offset, active_y1 - tick_len_10mm - int(round(3 * px_per_mm))), f"{i}mm", fill=(60, 60, 60), font=tiny_font, anchor="mm")
            draw.text((active_x1 - tick_len_10mm - int(round(4 * px_per_mm)), active_y1 + pos_offset), f"{i}", fill=(60, 60, 60), font=tiny_font, anchor="mm")

    # 4. ArUco Markers Placement (Corners)
    marker_size_px = int(round(MARKER_SIZE_MM * px_per_mm))
    marker_span_px = int(round(MARKER_SPAN_MM * px_per_mm))

    # Coordinates of marker centers:
    # Marker 0: Top-Left
    # Marker 1: Top-Right
    # Marker 2: Bottom-Right
    # Marker 3: Bottom-Left
    marker_centers = [
        (center_x - marker_span_px // 2, center_y - marker_span_px // 2),  # ID 0 (TL)
        (center_x + marker_span_px // 2, center_y - marker_span_px // 2),  # ID 1 (TR)
        (center_x + marker_span_px // 2, center_y + marker_span_px // 2),  # ID 2 (BR)
        (center_x - marker_span_px // 2, center_y + marker_span_px // 2),  # ID 3 (BL)
    ]

    for m_id, (mcx, mcy) in enumerate(marker_centers):
        marker_arr = create_aruco_marker(m_id, marker_size_px)
        marker_pil = Image.fromarray(marker_arr).convert("RGB")
        m_x1 = mcx - marker_size_px // 2
        m_y1 = mcy - marker_size_px // 2
        canvas.paste(marker_pil, (m_x1, m_y1))

        # Thin border around marker
        draw.rectangle([m_x1 - 1, m_y1 - 1, m_x1 + marker_size_px + 1, m_y1 + marker_size_px + 1], outline=(150, 150, 150), width=1)

        # Label marker ID
        id_label = f"ArUco #{m_id}"
        label_y = m_y1 - int(round(4 * px_per_mm)) if m_id in [0, 1] else m_y1 + marker_size_px + int(round(4 * px_per_mm))
        draw.text((mcx, label_y), id_label, fill=(40, 40, 40), font=tiny_font, anchor="mm")

    # 5. Reference Color Calibration Bar
    color_bar_y = active_y2 + int(round(22 * px_per_mm))
    block_w_px = int(round(16 * px_per_mm))
    block_h_px = int(round(12 * px_per_mm))
    gap_px = int(round(4 * px_per_mm))
    total_blocks_w = len(COLOR_BLOCKS) * block_w_px + (len(COLOR_BLOCKS) - 1) * gap_px
    start_block_x = center_x - total_blocks_w // 2

    bar_title = "REFERENCE COLOR & EXPOSURE CALIBRATION BLOCKS"
    draw.text((center_x, color_bar_y - int(round(5 * px_per_mm))), bar_title, fill=(50, 60, 75), font=label_font, anchor="mm")

    for idx, (b_name, b_rgb, b_outline) in enumerate(COLOR_BLOCKS):
        bx1 = start_block_x + idx * (block_w_px + gap_px)
        by1 = color_bar_y
        bx2 = bx1 + block_w_px
        by2 = by1 + block_h_px

        draw.rectangle([bx1, by1, bx2, by2], fill=b_rgb, outline=b_outline, width=2)
        # Short label
        draw.text((bx1 + block_w_px // 2, by2 + int(round(3 * px_per_mm))), b_name.split()[0], fill=(70, 70, 70), font=tiny_font, anchor="mm")

    # 6. Physical 50mm Calibration Scale Bar (Ruler Verification)
    scale_bar_y = color_bar_y + int(round(24 * px_per_mm))
    scale_bar_len_mm = 50.0
    scale_bar_len_px = int(round(scale_bar_len_mm * px_per_mm))
    scale_bar_x1 = center_x - scale_bar_len_px // 2
    scale_bar_x2 = scale_bar_x1 + scale_bar_len_px
    scale_bar_h_px = int(round(4 * px_per_mm))

    draw.text((center_x, scale_bar_y - int(round(4 * px_per_mm))), "PHYSICAL VERIFICATION RULER (Exact 50.0 mm)", fill=(30, 40, 50), font=small_font, anchor="mm")

    # Draw 5 alternating 10mm segments (Black / White)
    seg_px = int(round(10.0 * px_per_mm))
    for seg_idx in range(5):
        sx1 = scale_bar_x1 + seg_idx * seg_px
        sx2 = sx1 + seg_px if seg_idx < 4 else scale_bar_x2
        fill_color = (0, 0, 0) if (seg_idx % 2 == 0) else (255, 255, 255)
        draw.rectangle([sx1, scale_bar_y, sx2, scale_bar_y + scale_bar_h_px], fill=fill_color, outline=(0, 0, 0), width=1)
        draw.text((sx1 + seg_px // 2, scale_bar_y + scale_bar_h_px + int(round(3 * px_per_mm))), "10mm", fill=(50, 50, 50), font=tiny_font, anchor="mm")

    # End brackets for scale bar
    draw.line([(scale_bar_x1, scale_bar_y - 4), (scale_bar_x1, scale_bar_y + scale_bar_h_px + 4)], fill=(0, 0, 0), width=2)
    draw.line([(scale_bar_x2, scale_bar_y - 4), (scale_bar_x2, scale_bar_y + scale_bar_h_px + 4)], fill=(0, 0, 0), width=2)

    # 7. Footer & Copyright
    footer_y = sheet_h - int(round(15 * px_per_mm))
    footer_text = "Farmgate AI — Low-Cost Grain Quality Assaying System | Open Calibrated Paper Standard"
    draw.text((center_x, footer_y), footer_text, fill=(140, 145, 155), font=tiny_font, anchor="mm")

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi), quality=95)
    print(f"[Farmgate AI] Calibration sheet generated successfully: {output_path} ({sheet_w}x{sheet_h} px @ {dpi} DPI)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Farmgate AI printable calibration sheet")
    parser.add_argument("--output", type=str, default="calibration_sheet_A4.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300, help="Print DPI (default: 300)")
    args = parser.parse_args()

    generate_calibration_sheet(output_path=args.output, dpi=args.dpi)
