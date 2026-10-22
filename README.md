# 🌾 Farmgate AI — Offline Smartphone Rice Grain Quality Assaying System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![SQLite3](https://img.shields.io/badge/Database-Offline%20SQLite3-blue.svg)](https://www.sqlite.org/)
[![Standard](https://img.shields.io/badge/Compliance-AGMARK%20Standards-orange.svg)](https://agmarknet.gov.in/)

> **Farmgate AI** transforms ordinary Android and iOS smartphones into lab-grade, offline rice grain quality assaying instruments. By utilizing a low-cost, printable A4 calibration sheet with ArUco markers and reference color swatches, Farmgate AI un-warps camera tilt, normalizes lighting/exposure, separates touching grains via Distance Transform + Watershed segmentation, classifies defect parameters (whole, broken, discoloured, chalky), and provides an official **AGMARK-aligned quality certificate** with an offline SQLite history and Mandi trader feedback loop.

---

## 🌟 Key Features & Architectural Modules

### 1. `generate_sheet.py` (Printable Calibration Sheet Generator)
- Renders a high-resolution (**300 DPI, ISO A4 PNG**) calibration sheet.
- **Corner Fiducials**: Four ArUco markers (`DICT_4X4_50`, IDs 0, 1, 2, 3) placed at 172.0 mm center-to-center span.
- **Active Assaying Stage**: Central $150\text{ mm} \times 150\text{ mm}$ boundary with light 10mm grid lines and millimeter scale ticks.
- **Color & Exposure Swatches**: Reference calibration blocks (**White, 50% Gray, Black, Red, Green, Blue**) for white-balance and glare normalization.
- **Physical Verification Ruler**: Exactly 50.0 mm alternating scale bar for manual ruler calibration check.

### 2. `calibration_engine.py` (Validation & Homography Transformation)
Contains class `GrainImageCalibrator`:
- `validate_capture_quality(image, corners, ids)`:
  - Validates all 4 corner ArUco markers. If $< 4$: returns `"⚠️ Fewer than 4 corner markers detected. Sweep grains away from corner boxes and retake."`
  - Calculates blur score using Laplacian variance (`cv2.Laplacian(gray, cv2.CV_64F).var()`). If $< 80$, flags image as blurry.
  - Checks mean brightness. Flags overexposed/glare ($> 220$) or too dark ($< 40$) captures.
- `unwarp_image()`: Calculates the $3 \times 3$ Homography Matrix via `cv2.getPerspectiveTransform` to flatten the smartphone photo to a rectified $1000 \times 1000\text{ px}$ top-down view and computes the exact pixel-to-millimeter ratio.
- `normalize_lighting()`: Samples reference areas to adjust white balance and exposure via CIELAB CLAHE and White-Patch chromatic adaptation.

### 3. `grain_analyzer.py` (Watershed Segmentation & Variety-Aware Analysis)
Contains class `RiceGrainAnalyzer`:
- **Variety-Aware Baselines**:
  - `Sona Masoori` ($5.2\text{ mm}$)
  - `Basmati` ($7.5\text{ mm}$)
  - `Swarna` ($5.0\text{ mm}$)
  - `IR64` ($6.0\text{ mm}$)
  - `Auto-Detect` (dynamically estimates variety from whole-kernel mode)
- `segment_touching_grains(gray_img)`: Applies Distance Transform + Watershed Segmentation (`cv2.distanceTransform`, `cv2.watershed`) to separate touching/clustered kernels inside the active 150mm boundary.
- `analyze_grains()`: For each isolated grain contour:
  - Measures major axis length in mm using fitted ellipse (`cv2.fitEllipse`).
  - Classifies as:
    - 🟢 **Whole**: Length $\ge 75\%$ of variety reference length.
    - 🔴 **Broken**: Length $< 75\%$ of variety reference length.
    - 🟡 **Discoloured**: Color deviation and saturation variance from pearly white standard.
    - 🔵 **Chalky**: Fractional area ratio of opaque milky endosperm ($\ge 22\%$).
  - Draws annotated bounding boxes: **Green = Whole, Red = Broken, Yellow = Discoloured, Blue = Chalky**.
- **AGMARK Quality Grading Engine**: Assigns **Grade A (Special)**, **Grade B (Good)**, **Grade C (Fair)**, or **Rejection** based on Indian DMI Agricultural Produce rules.

### 4. `database.py` (Offline SQLite Manager)
- `init_db()`: Initializes `farmgate_history.db` with table `batch_history` storing `id`, `timestamp`, `variety`, `total_grains`, `whole_pct`, `broken_pct`, `discoloured_pct`, `chalky_pct`, `agmark_grade`, `mandi_broken_actual`, and `mandi_price_offered`.
- `save_batch()`: Auto-saves new test results.
- `update_mandi_feedback()`: Logs actual Mandi trader inspection results and calculates **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N}\sum |\text{Predicted Broken \%} - \text{Mandi Actual Broken \%}|$$
- `get_batch_history()`: Fetches past scans for interactive side-by-side comparison.

### 5. `app.py` (Bilingual Streamlit Dashboard)
- **Language Toggle**: Sidebar radio button switching between **English** and **Tamil (தமிழ்)** for all labels, guidance alerts, metric cards, and reports.
- **Workflow Tabs**:
  - **Tab 1: New Scan (புதிய ஆய்வு)**: Upload photo or load synthetic demo, select rice variety, execute pre-analysis quality checks, view side-by-side raw vs. unwarped vs. annotated images, metric KPI cards, and mandatory disclaimer:
    > *"Visual preliminary decision-support tool; does not measure moisture or internal chemical parameters."*
  - **Tab 2: Mandi Feedback Loop (மண்டி கருத்துப் பின்னூட்டம்)**: Select a past batch ID and input actual trader price/broken % to compare against app predictions and track MAE.
  - **Tab 3: Batch History (தொகுதி வரலாறு)**: Interactive table of past offline scans with filtering and CSV export.
  - **Tab 4: Printable Calibration Sheet**: Preview and download 300 DPI A4 PNG.
  - **Tab 5: AGMARK Standards**: Official grading tolerance table.

---

## 🛠️ Tech Stack & Requirements

- Python 3.10+
- OpenCV (`cv2` >= 4.8.0)
- NumPy (>= 1.24.0)
- Matplotlib (>= 3.7.0)
- Pillow (>= 9.5.0)
- Streamlit (>= 1.30.0)
- Pandas (>= 2.0.0)
- ReportLab (>= 4.0.0)
- SQLite3 (Python Standard Library - fully offline)

---

## 💻 Terminal Execution Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate the A4 Calibration Sheet
```bash
python generate_sheet.py --output calibration_sheet_A4.png --dpi 300
```
*Print the resulting `calibration_sheet_A4.png` at 100% scale (Do NOT scale to fit page) on standard A4 paper.*

### Step 3: Run the Automated Verification Pipeline
```bash
python test_pipeline.py
```
This tests:
1. Sheet generation (300 DPI A4).
2. Synthetic camera capture with perspective tilt.
3. ArUco detection & quality checks (marker count, blur, exposure).
4. Homography unwarping to $1000 \times 1000\text{ px}$.
5. Distance transform + watershed grain separation.
6. Variety-aware defect classification & AGMARK grading.
7. Offline SQLite batch storage & Mandi feedback MAE loop.
8. PDF Certificate and CSV export.

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 📜 AGMARK Rice Quality Standards (Milled / Raw Rice)

| Grade Designation | Max Broken Grains (%) | Max Discoloured (%) | Max Chalky Grains (%) | Min Whole Length | Market Segment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Grade A (Special / Premium)** | $\le 5.0\%$ | $\le 1.0\%$ | $\le 3.0\%$ | $\ge 75\%$ of Reference | Export / Premium Retail |
| **Grade B (Standard / Good)** | $\le 12.0\%$ | $\le 3.0\%$ | $\le 6.0\%$ | $\ge 75\%$ of Reference | Domestic Commercial / Table Rice |
| **Grade C (Fair Commercial)** | $\le 25.0\%$ | $\le 6.0\%$ | $\le 12.0\%$ | $\ge 75\%$ of Reference | Institutional / Blended Rice |
| **Rejection / Sub-standard** | $> 25.0\%$ | $> 6.0\%$ | $> 12.0\%$ | $< 75\%$ | Discounted / Industrial Byproduct |

---

## 📄 License & Attribution
Developed for the **Farmgate AI** decentralized farmgate assaying initiative. Built with Python, OpenCV, and Streamlit.
