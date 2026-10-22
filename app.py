"""
Farmgate AI - Smartphone Rice Grain Quality Assaying Dashboard
Bilingual (English & Tamil / தமிழ்) Streamlit Application
Features:
- ArUco Fiducial Perspective Unwarping (1000x1000px, Sub-MM Precision)
- Pre-Analysis Quality Check (Markers, Blur, Glare/Exposure)
- Distance Transform + Watershed Segmentation for Touching Grains
- Variety-Aware Baselines (Long Grain Rice, Sona Masoori, Basmati, Swarna, IR64, Auto-Detect)
- AGMARK Quality Grading (Whole, Broken, Discoloured, Chalky)
- Offline SQLite Database History & Mandi Feedback Loop with MAE Tracking
"""

import os
import io
import time
from datetime import datetime
import re
from typing import Dict, Any, Optional
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Farmgate AI Core Modules
from calibration_engine import GrainImageCalibrator
from grain_analyzer import RiceGrainAnalyzer, GrainFeature
from generate_sheet import generate_calibration_sheet
from generate_sample_image import generate_sample_smartphone_capture
from report_generator import (
    export_grains_to_csv,
    generate_pdf_report,
    generate_html_report,
    MANDATORY_DISCLAIMER,
)
import database as db


# Page Configuration
st.set_page_config(
    page_title="Farmgate AI — Grain Assaying & AGMARK Quality Grading",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database and Seed initial data if empty
db.init_db()
db.seed_sample_history_if_empty()


# Bilingual Dictionary (English & Tamil)
TEXTS = {
    "en": {
        "app_title": "Farmgate AI",
        "app_subtitle": "Low-Cost Smartphone Vision Assaying & AGMARK Quality Grading for Rice",
        "badge_text": "ARUCO CALIBRATED • SUB-MM PRECISION",
        "lang_select": "Language / மொழி",
        "sidebar_sample_input": "📷 Sample Input",
        "upload_label": "Upload Smartphone Photo",
        "upload_help": "Take a photo of grains spread on the Farmgate AI printable sheet (Max 200MB, JPG/PNG).",
        "demo_btn": "⚡ Demo Sample",
        "demo_help": "Load standard calibrated grain capture photo",
        "reset_btn": "🔄 Reset",
        "reset_help": "Clear the current image and reset view",
        "demo_loading": "Loading calibrated smartphone capture with grains...",
        "demo_success": "Demo sample loaded successfully!",
        "variety_header": "🌾 Grain Assaying Standards",
        "variety_select": "Grain Profile",
        "variety_help": "Select grain variety baseline profile for dimensional standards.",
        "target_length": "Target Full Length (mm)",
        "target_length_help": "Reference unbroken kernel length in millimetres.",
        "broken_cutoff_label": "Broken Threshold (% of Target)",
        "broken_cutoff_help": "Kernels shorter than this percentage of target length are deemed broken.",
        "advanced_tuner": "⚙️ Advanced Calibration & Vision Tuner",
        "discolour_label": "Discolouration Sensitivity",
        "chalky_label": "Chalkiness Threshold (% Area)",
        "marker_span_label": "ArUco Span (mm)",
        "show_labels": "Overlay Grain ID & Millimeter Tags",
        "welcome_title": "Welcome to Farmgate AI",
        "welcome_sub": "Upload a smartphone photo of rice grains on the printable calibration sheet or test instantly with a demo sample.",
        "step1_title": "1. 🖨️ Calibrate",
        "step1_desc": "Print the low-cost A4 sheet with 4 corner ArUco markers and central 150mm staging boundary.",
        "step2_title": "2. 📱 Capture",
        "step2_desc": "Spread a single layer of grains and snap a photo with any smartphone at a natural angle.",
        "step3_title": "3. ⚖️ Grade",
        "step3_desc": "Instant perspective unwarping, watershed grain separation, and AGMARK quality grading.",
        "tab_cv_inspection": "🔬 Computer Vision Inspection",
        "tab_morphometrics": "📊 Morphometrics & Distribution",
        "tab_grain_dataset": "📋 Grain Dataset",
        "tab_sheet": "🖨️ Printable Calibration Sheet",
        "tab_standards": "📜 AGMARK Standards & Tolerances",
        "tab_mandi": "🤝 Mandi Feedback Loop",
        "tab_history": "📁 Batch History",
        "quality_check_header": "🔍 Pre-Analysis Quality Validation",
        "quality_passed": "✅ All capture quality checks passed (Sharp focus, balanced exposure, 4 markers verified).",
        "blur_score_label": "Sharpness / Blur Score",
        "brightness_label": "Mean Brightness",
        "markers_detected_label": "Corner Markers Detected",
        "err_fewer_markers": "⚠️ Fewer than 4 corner markers detected. Sweep grains away from corner boxes and retake.",
        "err_blurry": "⚠️ Image is blurry (Sharpness < 80). Hold device steady and retake.",
        "err_glare": "⚠️ Image is overexposed/glare detected (Brightness > 220). Move away from harsh direct lighting.",
        "err_dark": "⚠️ Image is too dark (Brightness < 40). Ensure adequate ambient lighting.",
        "disclaimer_text": "⚠️ Visual preliminary decision-support tool; does not measure moisture or internal chemical parameters.",
        "agmark_banner_title": "ASSAYED AGMARK QUALITY GRADE",
        "total_grains": "TOTAL GRAINS",
        "whole_pct": "WHOLE GRAIN %",
        "broken_pct": "BROKEN %",
        "discoloured_pct": "DISCOLOURED %",
        "chalky_pct": "CHALKY %",
        "panel_orig_title": "Panel 1: Smartphone Photo & Unwarped View",
        "panel_annotated_title": "Panel 2: Computer Vision Defect Map",
        "legend_whole": "● Whole Grain (Green)",
        "legend_broken": "● Broken Grain (Red)",
        "legend_discoloured": "● Discoloured (Amber)",
        "legend_chalky": "● Chalky (Blue)",
        "caption_raw": "1. Raw Smartphone Photo (ArUco Detection)",
        "caption_unwarped": "2. Calibrated Top-Down View (1000x1000px)",
        "caption_defects": "3. Watershed Separated & Defect Classified",
        "dist_chart_title": "Kernel Length Distribution (mm)",
        "defect_chart_title": "Defect Classification Breakdown",
        "avg_length": "Average Kernel Length",
        "avg_width": "Average Kernel Width",
        "avg_ar": "Average Aspect Ratio (L/W)",
        "filter_class": "Filter by Defect Class",
        "download_csv": "📥 Download CSV Dataset",
        "download_pdf": "📄 Download PDF Certificate",
        "download_img": "🖼️ Download Annotated Image",
        "batch_saved_alert": "✅ Assay result auto-saved to offline SQLite database as Batch ID #{}.",
        "mandi_header": "🤝 Mandi Trader Feedback & MAE Ground-Truth Loop",
        "mandi_desc": "Log actual physical sieve inspection results and trader prices from the Mandi APMC to continuously validate and calibrate model predictions.",
        "select_batch_feedback": "Select Historical Batch ID",
        "actual_broken_input": "Actual Mandi Broken % (Trader Inspection)",
        "price_offered_input": "Trader Price Offered (₹ / Quintal)",
        "feedback_notes": "Trader / Market Notes",
        "submit_feedback_btn": "💾 Save Mandi Inspection Feedback",
        "feedback_success": "Mandi inspection feedback saved successfully for Batch #{}!",
        "current_mae": "Overall Mean Absolute Error (MAE)",
        "mae_desc": "Difference between Farmgate AI predicted broken % and actual Mandi trader inspection.",
        "trader_price_avg": "Average Trader Price",
        "evaluated_batches": "Evaluated Mandi Batches",
        "comparison_title": "Prediction vs. Mandi Actual Comparison",
        "pred_broken": "App Predicted Broken %",
        "actual_broken": "Trader Actual Broken %",
        "pred_diff": "Absolute Error",
        "history_header": "📋 Offline Assay History (farmgate_history.db)",
        "history_desc": "Historical records stored permanently on this device. Accessible without internet connectivity.",
        "filter_variety": "Filter by Variety",
        "compare_batches_label": "Select Batches for Side-by-Side Comparison",
        "export_history_csv": "📥 Export All History to CSV",
        "sheet_title": "Printable Calibration Sheet Generator",
        "sheet_desc": "Print this sheet at 100% scale (Do NOT select 'Fit to Printable Area') on standard A4 paper.",
        "download_sheet": "📥 Download Printable Calibration Sheet (A4 / 300 DPI PNG)",
    },
    "ta": {
        "app_title": "Farmgate AI",
        "app_subtitle": "குறைந்த செலவு ஸ்மார்ட்போன் பார்வை நெல் தானிய தர ஆய்வு மற்றும் அக்மார்க் தரம்",
        "badge_text": "அரூக்கோ அளவீடு • மில்லிமீட்டர் துல்லியம்",
        "lang_select": "Language / மொழி",
        "sidebar_sample_input": "📷 மாதிரி படம் உள்ளீடு",
        "upload_label": "ஸ்மார்ட்போன் புகைப்படத்தைப் பதிவேற்றவும்",
        "upload_help": "ஃபார்ம்கேட் AI அச்சிடக்கூடிய தாளில் பரப்பப்பட்ட தானியங்களின் படத்தை எடுக்கவும் (அதிகபட்சம் 200MB, JPG/PNG).",
        "demo_btn": "⚡ மாதிரி டெமோ",
        "demo_help": "அளவுத்திருத்தம் செய்யப்பட்ட மாதிரி புகைப்படத்தை ஏற்றவும்",
        "reset_btn": "🔄 மீட்டமை",
        "reset_help": "தற்போதைய படத்தை அழித்து மீட்டமைக்கவும்",
        "demo_loading": "மாதிரி புகைப்படத்தை உருவாக்குகிறது...",
        "demo_success": "மாதிரி புகைப்படம் வெற்றிகரமாக ஏற்றப்பட்டது!",
        "variety_header": "🌾 தானிய தர நிர்ணய அளவீடுகள்",
        "variety_select": "Grain Profile",
        "variety_help": "தானிய ரகத்தின் அளவுகோல் சுயவிவரத்தைத் தேர்ந்தெடுக்கவும்.",
        "target_length": "முழு தானிய குறிப்பு நீளம் (மி.மீ)",
        "target_length_help": "முழுமையான தானியத்தின் மாதிரி நீளம் (மி.மீ).",
        "broken_cutoff_label": "Broken Threshold (% of Target)",
        "broken_cutoff_help": "இந்த சதவீதத்திற்கு குறைவான நீளம் கொண்டவை உடைசலாகக் கருதப்படும்.",
        "advanced_tuner": "⚙️ மேம்பட்ட பார்வை & அளவுத்திருத்த அமைப்புகள்",
        "discolour_label": "நிறமாற்ற உணர்திறன்",
        "chalky_label": "சுண்ணாம்புத் தன்மை வரம்பு (% பரப்பளவு)",
        "marker_span_label": "குறிப்பான் இடைவெளி (மி.மீ)",
        "show_labels": "தானிய எண் மற்றும் மி.மீ குறிப்புகளைக் காண்பி",
        "welcome_title": "ஃபார்ம்கேட் AI-க்கு வரவேற்கிறோம்",
        "welcome_sub": "அளவுத்திருத்த தாளில் பரப்பப்பட்ட நெல் தானியங்களின் புகைப்படத்தைப் பதிவேற்றவும் அல்லது டெமோ மாதிரியை சோதிக்கவும்.",
        "step1_title": "1. 🖨️ அச்சிடுக",
        "step1_desc": "4 மூலை அரூக்கோ குறியீடுகள் கொண்ட A4 தாளை அச்சிடுங்கள்.",
        "step2_title": "2. 📱 படம் எடுக்கவும்",
        "step2_desc": "தானியங்களை பரப்பி ஸ்மார்ட்போனில் இயற்கை கோணத்தில் படம் எடுக்கவும்.",
        "step3_title": "3. ⚖️ தரம் அறிக",
        "step3_desc": "தானியங்கி திருத்தம், நீர்நிலை பிரிப்பு மற்றும் உடனடி அக்மார்க் சான்றிதழ்.",
        "tab_cv_inspection": "🔬 கணினி பார்வை ஆய்வு",
        "tab_morphometrics": "📊 உருவவியல் & விநியோகம்",
        "tab_grain_dataset": "📋 தானிய தரவுத்தொகுப்பு",
        "tab_sheet": "🖨️ அளவுத்திருத்த தாள்",
        "tab_standards": "📜 அக்மார்க் விதிகள்",
        "tab_mandi": "🤝 மண்டி பின்னூட்டம்",
        "tab_history": "📁 தொகுதி வரலாறு",
        "quality_check_header": "🔍 படத் தர சரிபார்ப்பு (Pre-Analysis Quality Check)",
        "quality_passed": "✅ அனைத்து படத் தர சோதனைகளும் தேர்ச்சி பெற்றன (தெளிவான படம், சரியான வெளிச்சம், 4 குறியீடுகள் உறுதி செய்யப்பட்டன).",
        "blur_score_label": "படத் தெளிவு / கூர்மை",
        "brightness_label": "சராசரி பிரகாசம்",
        "markers_detected_label": "கண்டறியப்பட்ட மூலைக் குறியீடுகள்",
        "err_fewer_markers": "⚠️ 4 மூலை குறிப்பான்களுக்கும் குறைவாக கண்டறியப்பட்டுள்ளது. தானியங்களை மூலை பெட்டிகளிலிருந்து அகற்றிவிட்டு மீண்டும் எடுக்கவும்.",
        "err_blurry": "⚠️ படம் மங்கலாக உள்ளது (கூர்மை < 80). சாதனத்தை அசையாமல் பிடித்து மீண்டும் எடுக்கவும்.",
        "err_glare": "⚠️ படம் அதிக வெளிச்சம்/பளபளப்புடன் உள்ளது (பிரகாசம் > 220). நேரடி கடுமையான வெளிச்சத்திலிருந்து விலகி எடுக்கவும்.",
        "err_dark": "⚠️ படம் மிகவும் இருட்டாக உள்ளது (பிரகாசம் < 40). போதுமான வெளிச்சத்தை உறுதி செய்யவும்.",
        "disclaimer_text": "⚠️ காட்சி அடிப்படையிலான ஆரம்ப முடிவு-ஆதரவு கருவி; இது ஈரப்பதம் அல்லது உள் வேதியியல் அளவுருக்களை அளவிடாது.",
        "agmark_banner_title": "ASSAYED AGMARK QUALITY GRADE",
        "total_grains": "TOTAL GRAINS",
        "whole_pct": "WHOLE GRAIN %",
        "broken_pct": "BROKEN %",
        "discoloured_pct": "DISCOLOURED %",
        "chalky_pct": "CHALKY %",
        "panel_orig_title": "பலகை 1: ஸ்மார்ட்போன் புகைப்படம் & சீரமைக்கப்பட்ட காட்சி",
        "panel_annotated_title": "பலகை 2: குறைபாடு வரைபடம்",
        "legend_whole": "● முழு தானியம் (பச்சை)",
        "legend_broken": "● உடைசல் தானியம் (சிவப்பு)",
        "legend_discoloured": "● நிறமாற்றம் (மஞ்சள்)",
        "legend_chalky": "● சுண்ணாம்புத் தன்மை (நீலம்)",
        "caption_raw": "1. அசல் ஸ்மார்ட்போன் புகைப்படம்",
        "caption_unwarped": "2. சீரமைக்கப்பட்ட காட்சி (1000x1000 பிக்சல்)",
        "caption_defects": "3. வகைப்படுத்தப்பட்ட தானியங்கள்",
        "dist_chart_title": "தானிய நீள விநியோகம் (மி.மீ)",
        "defect_chart_title": "குறைபாடுகள் ஒப்பீடு",
        "avg_length": "சராசரி தானிய நீளம்",
        "avg_width": "சராசரி தானிய அகலம்",
        "avg_ar": "விகிதம் (நீளம்/அகலம்)",
        "filter_class": "வகைப்படி வடிகட்டவும்",
        "download_csv": "📥 CSV தரவிறக்கம்",
        "download_pdf": "📄 PDF சான்றிதழ் தரவிறக்கம்",
        "download_img": "🖼️ குறிக்கப்பட்ட படம் தரவிறக்கம்",
        "batch_saved_alert": "✅ ஆய்வு முடிவு ஆஃப்லைன் SQLite தரவுத்தளத்தில் தொகுதி #{} ஆக சேமிக்கப்பட்டது.",
        "mandi_header": "🤝 மண்டி வியாபாரி பின்னூட்டம் மற்றும் MAE மதிப்பீடு",
        "mandi_desc": "மண்டி சந்தையில் வியாபாரி பரிசோதித்த உண்மையான உடைசல் சதவீதம் மற்றும் விலையைப் பதிவு செய்து துல்லியத்தை அளவிடவும்.",
        "select_batch_feedback": "முந்தைய தொகுதி எண்ணைத் தேர்ந்தெடுக்கவும்",
        "actual_broken_input": "உண்மையான மண்டி உடைசல் %",
        "price_offered_input": "வழங்கப்பட்ட மண்டி விலை (₹ / குவிண்டால்)",
        "feedback_notes": "மண்டி / வியாபாரி குறிப்புகள்",
        "submit_feedback_btn": "💾 மண்டி முடிவுகளைப் பதிவு செய்",
        "feedback_success": "தொகுதி #{} க்கான மண்டி கருத்து வெற்றிகரமாகப் பதிவு செய்யப்பட்டது!",
        "current_mae": "சராசரி பிழை (MAE)",
        "mae_desc": "செயலி கணித்த உடைசலுக்கும் மண்டி உண்மையான உடைசலுக்கும் உள்ள சராசரி வித்தியாசம்.",
        "trader_price_avg": "சராசரி மண்டி விலை",
        "evaluated_batches": "சரிபார்க்கப்பட்ட மண்டி தொகுதிகள்",
        "comparison_title": "கணிப்பு vs மண்டி உண்மை ஒப்பீடு",
        "pred_broken": "செயலி கணித்த உடைசல் %",
        "actual_broken": "வியாபாரி உண்மையான உடைசல் %",
        "pred_diff": "பிழை அளவு",
        "history_header": "📋 ஆஃப்லைன் தொகுதி வரலாறு (farmgate_history.db)",
        "history_desc": "இணைய இணைப்பு இல்லாமல் உங்கள் சாதனத்தில் நிரந்தரமாக சேமிக்கப்பட்ட முந்தைய ஆய்வுகள்.",
        "filter_variety": "ரகம் வாரியாக வடிகட்டவும்",
        "compare_batches_label": "ஒப்பீடு செய்ய தொகுதிகளைத் தேர்ந்தெடுக்கவும்",
        "export_history_csv": "📥 முழு வரலாற்றையும் CSV-ஆக ஏற்றுமதி செய்",
        "sheet_title": "Printable Calibration Sheet Generator",
        "sheet_desc": "இந்தத் தாளை 100% அளவில் (அளவு மாற்றாமல்) வழக்கமான A4 தாளில் அச்சிடவும்.",
        "download_sheet": "📥 அளவுத்திருத்த தாள் தரவிறக்கம் (A4 / 300 DPI)",
    },
}


# Custom CSS for UI Match with Reference Screenshot
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Noto+Sans+Tamil:wght@400;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', 'Noto Sans Tamil', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
    }

    /* Container Spacing & Layout */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
    }

    /* 1. Header Card Design */
    .farmgate-header {
        background: linear-gradient(135deg, #091a27 0%, #0d2737 40%, #0b3d45 75%, #0d5452 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .farmgate-header-left {
        display: flex;
        flex-direction: column;
    }
    .farmgate-title-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .farmgate-title {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        line-height: 1.15;
        letter-spacing: -0.4px;
    }
    .farmgate-subtitle {
        font-size: 13.5px;
        color: #88d4ce;
        margin: 8px 0 0 0;
        font-weight: 400;
        letter-spacing: 0.15px;
    }
    .farmgate-badge {
        border: 1.5px solid #14b8a6;
        background: rgba(20, 184, 166, 0.06);
        color: #2dd4bf;
        padding: 7px 18px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        display: inline-block;
        white-space: nowrap;
    }

    /* 2. AGMARK Quality Result Banner */
    .agmark-grade-banner {
        border-radius: 14px;
        padding: 22px 28px;
        color: #ffffff;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    }
    .agmark-banner-label {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 4px;
    }
    .agmark-banner-grade {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .agmark-banner-sub {
        font-size: 13.5px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.95);
        margin: 0;
        line-height: 1.4;
    }

    /* 3. Five Quality Metric Cards */
    .metric-card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        min-height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 10px;
    }
    .metric-card-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .metric-card-header {
        font-size: 11px;
        font-weight: 700;
        color: #0284c7;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .metric-card-val {
        font-size: 28px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .metric-card-sub {
        font-size: 11.5px;
        color: #94a3b8;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    button[data-baseweb="tab"] {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        color: #475569 !important;
        border: none !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e11d48 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #e11d48 !important;
        height: 3px !important;
        border-radius: 2px !important;
    }

    /* Sidebar controls */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stFileUploader {
        background-color: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 8px;
    }
    /* Red Accent for Slider */
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #ef4444 !important;
        border-color: #ef4444 !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[aria-hidden="true"] {
        background-color: #ef4444 !important;
    }

    /* Legend Chips */
    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .chip-whole { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .chip-broken { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .chip-discoloured { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
    .chip-chalky { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

    .disclaimer-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 10px 14px;
        color: #92400e;
        font-size: 13px;
        font-weight: 500;
        margin-top: 14px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session():
    """Initializes persistent application session variables."""
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "was_reset" not in st.session_state:
        st.session_state.was_reset = False
    if "current_image_bgr" not in st.session_state:
        if not st.session_state.was_reset and os.path.exists("sample_grain_capture.jpg"):
            st.session_state.current_image_bgr = cv2.imread("sample_grain_capture.jpg")
            st.session_state.image_source_name = "sample_grain_capture.jpg"
        else:
            st.session_state.current_image_bgr = None
            st.session_state.image_source_name = None
    if "image_source_name" not in st.session_state:
        st.session_state.image_source_name = None
    if "last_saved_batch_id" not in st.session_state:
        st.session_state.last_saved_batch_id = None


def get_txt(key: str) -> str:
    """Helper to get translated string for active language."""
    lang = st.session_state.get("lang", "en")
    return TEXTS.get(lang, TEXTS["en"]).get(key, TEXTS["en"].get(key, key))


def render_top_header():
    """Renders the top hero navigation banner matching the reference screenshot."""
    grain_svg = """<svg width="26" height="30" viewBox="0 0 26 30" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 6px;">
        <path d="M13 28V12" stroke="#22c55e" stroke-width="2.2" stroke-linecap="round"/>
        <path d="M13 23C9 21 8.5 17 8.5 17" stroke="#22c55e" stroke-width="1.8" stroke-linecap="round"/>
        <path d="M13 19C17.5 17 18 13 18 13" stroke="#22c55e" stroke-width="1.8" stroke-linecap="round"/>
        <ellipse cx="8" cy="15" rx="2.2" ry="3.6" transform="rotate(-30 8 15)" fill="#eab308" stroke="#ca8a04" stroke-width="0.5"/>
        <ellipse cx="18" cy="11" rx="2.2" ry="3.6" transform="rotate(30 18 11)" fill="#eab308" stroke="#ca8a04" stroke-width="0.5"/>
        <ellipse cx="9" cy="8" rx="2.2" ry="3.6" transform="rotate(-25 9 8)" fill="#eab308" stroke="#ca8a04" stroke-width="0.5"/>
        <ellipse cx="17" cy="5" rx="2.2" ry="3.6" transform="rotate(25 17 5)" fill="#eab308" stroke="#ca8a04" stroke-width="0.5"/>
        <ellipse cx="13" cy="3" rx="2" ry="3.5" fill="#eab308" stroke="#ca8a04" stroke-width="0.5"/>
    </svg>"""
    st.markdown(
        f"""
        <div class="farmgate-header">
            <div class="farmgate-header-left">
                <div class="farmgate-title-row">
                    <span style="display: inline-flex; align-items: center;">{grain_svg}</span>
                    <h1 class="farmgate-title">{get_txt('app_title')}</h1>
                </div>
                <p class="farmgate-subtitle">{get_txt('app_subtitle')}</p>
            </div>
            <div class="farmgate-header-right">
                <span class="farmgate-badge">{get_txt('badge_text')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agmark_banner(stats: Dict[str, Any]):
    """Renders the full-width AGMARK quality grade result banner with dynamic results."""
    grade = stats["agmark_grade"]
    rationale = stats["grade_rationale"]

    # Assign background color based on actual grade
    if "Rejection" in grade:
        bg_color = "#e53935"  # Vibrant Red
    elif "Grade A" in grade:
        bg_color = "#059669"  # Emerald Green
    elif "Grade B" in grade:
        bg_color = "#2563eb"  # Blue
    elif "Grade C" in grade:
        bg_color = "#d97706"  # Amber
    else:
        bg_color = stats.get("grade_color", "#475569")

    st.markdown(
        f"""
        <div class="agmark-grade-banner" style="background-color: {bg_color};">
            <div class="agmark-banner-label">{get_txt('agmark_banner_title')}</div>
            <div class="agmark-banner-grade">{grade}</div>
            <div class="agmark-banner-sub">{rationale}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(stats: Dict[str, Any], analyzer: RiceGrainAnalyzer):
    """Renders the five quality metric cards horizontally with computed values."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">{get_txt('total_grains')}</div>
                <div class="metric-card-val" style="color: #0f172a;">{stats['total_grains']}</div>
                <div class="metric-card-sub">Analyzed Kernels</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">{get_txt('whole_pct')}</div>
                <div class="metric-card-val" style="color: #059669;">{stats['whole_pct']}%</div>
                <div class="metric-card-sub">{stats['whole_count']} kernels (&gt;= {analyzer.broken_cutoff_length_mm:.1f}mm)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">{get_txt('broken_pct')}</div>
                <div class="metric-card-val" style="color: #dc2626;">{stats['broken_pct']}%</div>
                <div class="metric-card-sub">{stats['broken_count']} broken (&lt; {analyzer.broken_cutoff_length_mm:.1f}mm)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">{get_txt('discoloured_pct')}</div>
                <div class="metric-card-val" style="color: #ea580c;">{stats['discoloured_pct']}%</div>
                <div class="metric-card-sub">{stats['discoloured_count']} defective/spotted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">{get_txt('chalky_pct')}</div>
                <div class="metric-card-val" style="color: #2563eb;">{stats['chalky_pct']}%</div>
                <div class="metric-card-sub">{stats['chalky_count']} opaque endosperms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_onscreen_summary_card(stats: Dict[str, Any], quality_status: str, sample_id: str, timestamp_str: str):
    """
    Feature 5: On-screen summary report card displaying:
    Sample ID, timestamp, total grains, whole/broken/discoloured/chalky counts and %,
    assigned AGMARK grade with rationale, capture quality status (passed / warning / failed).
    """
    status_lower = quality_status.lower()
    if "pass" in status_lower:
        status_bg = "#dcfce7"
        status_fg = "#166534"
        icon = "✅"
    elif "warn" in status_lower:
        status_bg = "#fef3c7"
        status_fg = "#92400e"
        icon = "⚠️"
    else:
        status_bg = "#fee2e2"
        status_fg = "#991b1b"
        icon = "❌"

    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px 24px; margin: 12px 0 16px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 16px; font-weight: 700; color: #0f172a;">📋 Assay Summary Report Card</span>
                    <span style="margin-left: 10px; font-size: 13px; color: #64748b; font-family: monospace;">Sample ID: <strong>{sample_id}</strong></span>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 12px; color: #64748b;">🕒 {timestamp_str}</span>
                    <span style="background-color: {status_bg}; color: {status_fg}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 9999px; text-transform: uppercase;">
                        {icon} Capture Quality: {quality_status}
                    </span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; text-align: center;">
                <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Total Assayed</div>
                    <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 2px;">{stats['total_grains']}</div>
                    <div style="font-size: 11px; color: #64748b;">Kernels</div>
                </div>
                <div style="background: #f0fdf4; padding: 10px; border-radius: 8px; border: 1px solid #bbf7d0;">
                    <div style="font-size: 11px; color: #166534; text-transform: uppercase; font-weight: 600;">Whole Grains</div>
                    <div style="font-size: 18px; font-weight: 800; color: #15803d; margin-top: 2px;">{stats['whole_count']} <span style="font-size: 13px;">({stats['whole_pct']}%)</span></div>
                    <div style="font-size: 11px; color: #166534;">Head Rice</div>
                </div>
                <div style="background: #fef2f2; padding: 10px; border-radius: 8px; border: 1px solid #fecaca;">
                    <div style="font-size: 11px; color: #991b1b; text-transform: uppercase; font-weight: 600;">Broken Grains</div>
                    <div style="font-size: 18px; font-weight: 800; color: #dc2626; margin-top: 2px;">{stats['broken_count']} <span style="font-size: 13px;">({stats['broken_pct']}%)</span></div>
                    <div style="font-size: 11px; color: #991b1b;">Fragmented</div>
                </div>
                <div style="background: #fffbeb; padding: 10px; border-radius: 8px; border: 1px solid #fde68a;">
                    <div style="font-size: 11px; color: #92400e; text-transform: uppercase; font-weight: 600;">Discoloured</div>
                    <div style="font-size: 18px; font-weight: 800; color: #d97706; margin-top: 2px;">{stats['discoloured_count']} <span style="font-size: 13px;">({stats['discoloured_pct']}%)</span></div>
                    <div style="font-size: 11px; color: #92400e;">Spotted / Stained</div>
                </div>
                <div style="background: #eff6ff; padding: 10px; border-radius: 8px; border: 1px solid #bfdbfe;">
                    <div style="font-size: 11px; color: #1e40af; text-transform: uppercase; font-weight: 600;">Chalky Grains</div>
                    <div style="font-size: 18px; font-weight: 800; color: #2563eb; margin-top: 2px;">{stats['chalky_count']} <span style="font-size: 13px;">({stats['chalky_pct']}%)</span></div>
                    <div style="font-size: 11px; color: #1e40af;">Opaque Kernels</div>
                </div>
            </div>
            <div style="margin-top: 12px; padding: 8px 12px; background: #f8fafc; border-radius: 6px; font-size: 12px; color: #475569; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                <span><strong>Assigned Grade:</strong> <span style="color: #0f172a; font-weight: 700;">{stats['agmark_grade']}</span> &mdash; <em>{stats.get('grade_rationale', '')}</em></span>
                <span style="font-size: 11px; color: #64748b;">Grain Variety: <strong>{stats['variety']}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_calibration_sheet_tab():
    """Renders the Printable Calibration Sheet generator and instructions."""
    st.markdown(f"### {get_txt('sheet_title')}")
    st.write(get_txt("sheet_desc"))

    sheet_png = "calibration_sheet_A4.png"
    if not os.path.exists(sheet_png):
        with st.spinner("Generating 300 DPI A4 Calibration Sheet..."):
            generate_calibration_sheet(sheet_png, dpi=300)

    cs1, cs2 = st.columns([1, 1])
    with cs1:
        st.image(sheet_png, caption="A4 Calibration Standard (300 DPI)", use_container_width=True)
    with cs2:
        st.markdown(
            """
            ### 📋 Printing Specifications:
            1. **Paper**: Standard White A4 Copier Paper (80 GSM or 100 GSM).
            2. **Scale**: **100% (Actual Size / Do Not Scale)**.
            3. **Active Box**: Place grains strictly inside the **150 mm × 150 mm** inner box.
            4. **Ruler Check**: Verify bottom verification bar measures **exactly 50.0 mm**.
            5. **Corner Markers**: Keep all 4 ArUco markers clean and free of grain dust.
            """
        )
        with open(sheet_png, "rb") as f:
            s_bytes = f.read()
        st.download_button(
            label=get_txt("download_sheet"),
            data=s_bytes,
            file_name="Farmgate_AI_Calibration_Sheet_A4.png",
            mime="image/png",
            use_container_width=True,
        )


def render_standards_tab():
    """Renders the official AGMARK grading standards."""
    st.markdown(f"### {get_txt('tab_standards')}")
    st.write(
        "Indian AGMARK (Agricultural Produce Grading and Marking Act) standards "
        "for milled and raw rice classification."
    )

    standards_df = pd.DataFrame([
        {
            "Grade": "Grade A (Special / Premium)",
            "Max Broken (%)": "<= 5.0%",
            "Max Discoloured (%)": "<= 1.0%",
            "Max Chalky (%)": "<= 3.0%",
            "Min Kernel Length": ">= 75% of Variety Length",
            "Market Segment": "Export / Premium Retail",
        },
        {
            "Grade": "Grade B (Standard / Good)",
            "Max Broken (%)": "<= 12.0%",
            "Max Discoloured (%)": "<= 3.0%",
            "Max Chalky (%)": "<= 6.0%",
            "Min Kernel Length": ">= 75% of Variety Length",
            "Market Segment": "Commercial Retail / Table Rice",
        },
        {
            "Grade": "Grade C (Fair Commercial)",
            "Max Broken (%)": "<= 25.0%",
            "Max Discoloured (%)": "<= 6.0%",
            "Max Chalky (%)": "<= 12.0%",
            "Min Kernel Length": ">= 75% of Variety Length",
            "Market Segment": "Institutional / Blended",
        },
        {
            "Grade": "Rejection / Sub-standard",
            "Max Broken (%)": "> 25.0%",
            "Max Discoloured (%)": "> 6.0%",
            "Max Chalky (%)": "> 12.0%",
            "Min Kernel Length": "< 75%",
            "Market Segment": "Industrial / Discounted",
        },
    ])
    st.table(standards_df)


def render_mandi_tab():
    """
    Feature 6: Renders the Mandi Feedback Loop and MAE Tracking.
    Stores and compares app-estimated broken % against actual broken %
    entered by user/trader, computes MAE across saved samples, and offers CSV export.
    """
    st.markdown("### 🌾 Mandi & Ground-Truth Feedback Loop")
    st.write(
        "Enter ground-truth broken grain percentages from manual counts or Mandi trader assessments "
        "to validate accuracy and track real-time Mean Absolute Error (MAE)."
    )

    db.seed_sample_history_if_empty()
    mae_info = db.calculate_feedback_mae()
    comp_df = db.get_feedback_comparison_df()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        mae_val = mae_info["mae"]
        mae_color = "#10b981" if mae_val <= 2.0 else ("#0284c7" if mae_val <= 4.0 else "#f59e0b")
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">Mean Absolute Error (MAE)</div>
                <div class="metric-card-val" style="color: {mae_color};">±{mae_val:.2f}%</div>
                <div class="metric-card-sub">App vs Ground-Truth Broken Deviation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">Evaluated Samples</div>
                <div class="metric-card-val" style="color: #0f172a;">{mae_info['sample_count']}</div>
                <div class="metric-card-sub">Ground-Truth Calibrated Lots</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m3:
        tier_label = "High Agreement (≤ 2.0%)" if mae_val <= 2.0 else ("Acceptable (≤ 4.0%)" if mae_val <= 4.0 else "Review Needed")
        tier_color = "#10b981" if mae_val <= 2.0 else ("#0284c7" if mae_val <= 4.0 else "#f59e0b")
        st.markdown(
            f"""
            <div class="metric-card-box">
                <div class="metric-card-header">Accuracy Benchmark</div>
                <div class="metric-card-val" style="color: {tier_color}; font-size: 20px; line-height: 1.4;">{tier_label}</div>
                <div class="metric-card-sub">Based on {mae_info['sample_count']} verified lots</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if comp_df.empty:
        st.info("No batches found in SQLite database yet. Perform a scan in Tab 1 first.")
    else:
        st.markdown("#### 📝 Record Ground-Truth Actual Broken %")
        col_form1, col_form2 = st.columns([1, 1])

        with col_form1:
            sample_options = [
                f"{row['sample_id']} — {row['variety']} ({row['timestamp']}) [App: {row['app_broken_pct']}%]"
                for _, row in comp_df.iterrows()
            ]
            selected_sample_str = st.selectbox(
                "Select Saved Sample to Update:", options=sample_options
            )

            selected_sample_id = selected_sample_str.split(" — ")[0].strip()
            selected_matches = comp_df[comp_df["sample_id"] == selected_sample_id]
            if selected_matches.empty:
                st.warning("The selected sample could not be found.")
                return
            selected_row = selected_matches.iloc[0]

            existing_actual = (
                float(selected_row["actual_broken_pct"])
                if pd.notnull(selected_row["actual_broken_pct"])
                else float(selected_row["app_broken_pct"])
            )
            existing_notes = (
                str(selected_row["notes"])
                if pd.notnull(selected_row["notes"]) and selected_row["notes"] is not None
                else ""
            )

            actual_broken = st.number_input(
                "Actual Broken Grain Percentage (Trader / Manual Count %):",
                min_value=0.0,
                max_value=100.0,
                value=max(0.0, min(100.0, existing_actual)),
                step=0.1,
                format="%.1f",
            )
            notes_text = st.text_input(
                "Inspection Notes (Lighting conditions, angle, camera distance, tray count):",
                value=existing_notes,
                placeholder="e.g. Diffuse morning sunlight, perpendicular 90 deg, manual tray count",
            )

            if st.button("💾 Save Feedback & Recalculate MAE", use_container_width=True):
                db.record_feedback(
                    sample_id=selected_sample_id,
                    actual_broken_pct=actual_broken,
                    notes=notes_text,
                )
                st.success(f"✅ Feedback saved for {selected_sample_id}! Updated MAE: ±{db.calculate_feedback_mae()['mae']:.2f}%")
                st.rerun()

        with col_form2:
            st.markdown("#### 🔍 Sample Accuracy Inspection")
            pred_broken_val = float(selected_row["app_broken_pct"])
            diff_val = abs(pred_broken_val - actual_broken)
            q_status = str(selected_row.get("capture_quality_status", "Passed"))
            variety = str(selected_row.get("variety", "N/A"))

            status_bg = "#dcfce7" if "pass" in q_status.lower() else ("#fef3c7" if "warn" in q_status.lower() else "#fee2e2")
            status_fg = "#166534" if "pass" in q_status.lower() else ("#92400e" if "warn" in q_status.lower() else "#991b1b")

            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 14px; font-weight: 700; color: #0f172a;">Sample: {selected_sample_id} ({variety})</span>
                        <span style="background: {status_bg}; color: {status_fg}; padding: 3px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700;">
                            {q_status}
                        </span>
                    </div>
                    <p style="margin: 0 0 10px 0; font-size: 12px; color: #64748b;">
                        Recorded: {selected_row['timestamp']}
                    </p>
                    <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 10px 0;"/>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;">
                        <span>App Estimated Broken:</span>
                        <span style="font-weight: 700; color: #0284c7;">{pred_broken_val:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;">
                        <span>Actual Broken (Ground-Truth):</span>
                        <span style="font-weight: 700; color: #10b981;">{actual_broken:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;">
                        <span>Absolute Error (|App - Actual|):</span>
                        <span style="font-weight: 800; color: {'#10b981' if diff_val <= 2.5 else '#f59e0b'};">
                            {diff_val:.1f}% ({'High Agreement' if diff_val <= 2.5 else 'Acceptable Tolerance'})
                        </span>
                    </div>
                    <div style="margin-top: 8px; font-size: 11.5px; color: #64748b;">
                        <em>Notes: {notes_text or 'No notes provided yet'}</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 📊 Ground-Truth vs Predicted Comparison Table")

        # Display full comparison table
        display_df = comp_df.rename(columns={
            "sample_id": "Sample ID",
            "timestamp": "Timestamp",
            "variety": "Variety",
            "app_broken_pct": "App Broken (%)",
            "actual_broken_pct": "Actual Broken (%)",
            "absolute_error": "Absolute Error (%)",
            "capture_quality_status": "Capture Quality",
            "notes": "Inspection Notes",
        })
        st.dataframe(display_df, use_container_width=True, height=280)

        # CSV Export button
        csv_mandi_buf = io.StringIO()
        comp_df.to_csv(csv_mandi_buf, index=False)
        st.download_button(
            label="📥 Download Mandi Feedback Log (CSV)",
            data=csv_mandi_buf.getvalue(),
            file_name="mandi_feedback_log.csv",
            mime="text/csv",
            use_container_width=False,
        )


def render_history_tab():
    """Renders the offline batch history table and comparison tool."""
    st.markdown(f"### {get_txt('history_header')}")
    st.write(get_txt("history_desc"))

    df_all = db.get_batch_history()
    if df_all.empty:
        st.info("No batches recorded in offline database yet.")
    else:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            variety_list = ["All Varieties"] + list(df_all["variety"].dropna().unique())
            v_filter = st.selectbox(get_txt("filter_variety"), options=variety_list)
        with col_f2:
            st.write("")

        filtered_history = (
            df_all if v_filter == "All Varieties"
            else df_all[df_all["variety"] == v_filter]
        )
        st.dataframe(filtered_history, use_container_width=True, height=320)

        csv_hist_buf = io.StringIO()
        filtered_history.to_csv(csv_hist_buf, index=False)
        st.download_button(
            label=get_txt("export_history_csv"),
            data=csv_hist_buf.getvalue(),
            file_name="farmgate_offline_batch_history.csv",
            mime="text/csv",
            use_container_width=False,
        )

        st.markdown("---")
        st.markdown(f"#### {get_txt('compare_batches_label')}")
        compare_ids = st.multiselect(
            "Select 2 or more batches to compare:",
            options=df_all["id"].tolist(),
            default=df_all["id"].tolist()[:2] if len(df_all) >= 2 else df_all["id"].tolist(),
        )
        if len(compare_ids) >= 2:
            comp_df = df_all[df_all["id"].isin(compare_ids)].set_index("id").T
            st.table(comp_df)


def render_welcome_cards():
    """Renders initial getting-started guide when no sample is loaded."""
    st.info(f"👋 {get_txt('welcome_sub')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            ### {get_txt('step1_title')}
            {get_txt('step1_desc')}
            """
        )
    with col2:
        st.markdown(
            f"""
            ### {get_txt('step2_title')}
            {get_txt('step2_desc')}
            """
        )
    with col3:
        st.markdown(
            f"""
            ### {get_txt('step3_title')}
            {get_txt('step3_desc')}
            """
        )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <span>ℹ️</span> {get_txt('disclaimer_text')}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    initialize_session()

    # --- SIDEBAR: Language Toggle & Input Controls ---
    with st.sidebar:
        # Language Selector
        st.markdown(f"#### 🌐 {get_txt('lang_select')}")
        selected_lang = st.radio(
            "Select UI Language",
            options=["English", "தமிழ் (Tamil)"],
            index=0 if st.session_state.lang == "en" else 1,
            label_visibility="collapsed",
        )
        st.session_state.lang = "ta" if "தமிழ்" in selected_lang else "en"

        st.markdown("---")
        st.markdown(f"### {get_txt('sidebar_sample_input')}")

        uploaded_file = st.file_uploader(
            get_txt("upload_label"),
            type=["jpg", "jpeg", "png"],
            help=get_txt("upload_help"),
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            load_demo = st.button(
                get_txt("demo_btn"),
                use_container_width=True,
                help=get_txt("demo_help"),
            )
        with col_b2:
            clear_btn = st.button(
                get_txt("reset_btn"),
                use_container_width=True,
                help=get_txt("reset_help"),
            )

        if clear_btn:
            st.session_state.current_image_bgr = None
            st.session_state.image_source_name = None
            st.session_state.last_saved_batch_id = None
            st.session_state.was_reset = True
            st.rerun()

        # Variety & Analysis Settings
        st.markdown("---")
        st.markdown(f"### {get_txt('variety_header')}")

        varieties = list(RiceGrainAnalyzer.VARIETY_BASELINES.keys())
        variety_choice = st.selectbox(
            get_txt("variety_select"),
            options=varieties,
            index=0,  # "Long Grain Rice (Default)"
            help=get_txt("variety_help"),
        )

        default_len = RiceGrainAnalyzer.VARIETY_BASELINES[variety_choice]["full_length_mm"]
        custom_ref_length = st.number_input(
            get_txt("target_length"),
            min_value=3.0,
            max_value=12.0,
            value=float(default_len),
            step=0.1,
            format="%.2f",
            help=get_txt("target_length_help"),
        )

        if variety_choice in ["Custom/Auto", "Auto-Detect"]:
            st.caption("ℹ️ *Custom/Auto: Intact length estimated from 75th percentile of distribution.*")

        broken_cutoff_pct = st.slider(
            get_txt("broken_cutoff_label"),
            min_value=60,
            max_value=85,
            value=75,
            step=1,
            help=get_txt("broken_cutoff_help"),
        )

        with st.expander(get_txt("advanced_tuner")):
            discolour_sens = st.slider(
                get_txt("discolour_label"),
                min_value=20.0,
                max_value=70.0,
                value=38.0,
                step=2.0,
            )
            chalky_sens = st.slider(
                get_txt("chalky_label"),
                min_value=10,
                max_value=40,
                value=22,
                step=2,
            )
            marker_span_mm = st.number_input(
                get_txt("marker_span_label"),
                min_value=120.0,
                max_value=240.0,
                value=172.0,
                step=1.0,
            )
            show_grain_labels = st.checkbox(get_txt("show_labels"), value=True)

        if load_demo:
            with st.spinner(get_txt("demo_loading")):
                sample_file = "sample_grain_capture.jpg"
                if not os.path.exists(sample_file):
                    generate_sample_smartphone_capture(
                        output_path=sample_file,
                        num_grains=48,
                        variety=variety_choice if variety_choice != "Auto-Detect" else "Long Grain Rice (Default)",
                    )
                demo_img = cv2.imread(sample_file)
                st.session_state.current_image_bgr = demo_img
                st.session_state.image_source_name = f"Sample Capture ({variety_choice})"
                st.session_state.last_saved_batch_id = None
                st.session_state.was_reset = False
                st.rerun()

        if uploaded_file is not None and st.session_state.image_source_name != uploaded_file.name:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.current_image_bgr = img_bgr
            st.session_state.image_source_name = uploaded_file.name
            st.session_state.last_saved_batch_id = None
            st.session_state.was_reset = False

    # Top Header Card
    render_top_header()

    # Welcome view if no image is present
    if st.session_state.current_image_bgr is None:
        render_welcome_cards()
        tab_sheet_view, tab_agmark_view, tab_history_view = st.tabs([
            get_txt("tab_sheet"),
            get_txt("tab_standards"),
            get_txt("tab_history"),
        ])
        with tab_sheet_view:
            render_calibration_sheet_tab()
        with tab_agmark_view:
            render_standards_tab()
        with tab_history_view:
            render_history_tab()
        return

    # Image is loaded: Run Calibration and Grain Analysis
    raw_img = st.session_state.current_image_bgr
    calibrator = GrainImageCalibrator(
        marker_span_mm=marker_span_mm,
        output_size=(1000, 1000),
    )

    # Pre-Analysis Quality Validation
    is_valid_capture, quality_metrics, quality_messages = calibrator.validate_capture_quality(raw_img)

    # Run Homography Unwarping
    unwarped_bgr, annotated_orig, px_per_mm, calib_status = calibrator.process(raw_img)

    if not calib_status["success"] or unwarped_bgr is None:
        st.error(f"❌ {get_txt('err_fewer_markers')}")
        st.image(
            cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB),
            caption=get_txt("caption_raw"),
            use_container_width=True,
        )
        return

    # Run Grain Segmentation and Measurement
    analyzer = RiceGrainAnalyzer(
        variety=variety_choice,
        length_threshold_ratio=broken_cutoff_pct / 100.0,
        discolour_threshold=discolour_sens,
        chalky_threshold=chalky_sens / 100.0,
    )
    if custom_ref_length > 0:
        analyzer.full_length_mm = custom_ref_length
        analyzer.broken_cutoff_length_mm = custom_ref_length * (broken_cutoff_pct / 100.0)

    annotated_grains, grains_list, stats, binary_mask = analyzer.analyze(unwarped_bgr, px_per_mm)

    current_quality_status = "Passed" if quality_metrics.get("status") == "pass" else ("Warning" if quality_metrics.get("status") == "warn" else "Failed")
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Auto-save batch to SQLite if not saved for this image
    if st.session_state.last_saved_batch_id is None and stats["total_grains"] > 0:
        new_id = db.save_batch(
            variety=stats["variety"],
            total_grains=stats["total_grains"],
            whole_pct=stats["whole_pct"],
            broken_pct=stats["broken_pct"],
            discoloured_pct=stats["discoloured_pct"],
            chalky_pct=stats["chalky_pct"],
            agmark_grade=stats["agmark_grade"],
            capture_quality_status=current_quality_status,
            notes="Offline smartphone scan",
        )
        st.session_state.last_saved_batch_id = new_id

    current_sample_id = f"SMP-{st.session_state.last_saved_batch_id or 1001:04d}"

    # 1. Full-width AGMARK Quality Result Banner
    render_agmark_banner(stats)

    # 2. Five Quality Metric Cards in one row
    render_metric_cards(stats, analyzer)

    # Feature 5: On-screen summary report card
    render_onscreen_summary_card(stats, current_quality_status, current_sample_id, current_timestamp)

    # 3. Seven Analysis Tabs below Metric Cards
    tab_cv, tab_morph, tab_data, tab_sheet, tab_standards, tab_mandi, tab_history = st.tabs([
        get_txt("tab_cv_inspection"),
        get_txt("tab_morphometrics"),
        get_txt("tab_grain_dataset"),
        get_txt("tab_sheet"),
        get_txt("tab_standards"),
        get_txt("tab_mandi"),
        get_txt("tab_history"),
    ])

    # =========================================================================
    # TAB 1: COMPUTER VISION INSPECTION
    # =========================================================================
    with tab_cv:
        st.markdown(f"#### {get_txt('quality_check_header')}")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            m_count = quality_metrics["marker_count"]
            m_color = "#10b981" if m_count == 4 else "#ef4444"
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header">{get_txt('markers_detected_label')}</div>
                    <div class="metric-card-val" style="color: {m_color};">{m_count} / 4</div>
                    <div class="metric-card-sub">ArUco Corner IDs: {quality_metrics['detected_ids']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_q2:
            b_score = quality_metrics["blur_score"]
            b_color = "#10b981" if b_score >= 80.0 else "#f59e0b"
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header">{get_txt('blur_score_label')}</div>
                    <div class="metric-card-val" style="color: {b_color};">{b_score}</div>
                    <div class="metric-card-sub">{'Sharp Focus (>= 80)' if b_score >= 80 else 'Blurry (< 80)'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_q3:
            bright_val = quality_metrics["mean_brightness"]
            br_color = "#10b981" if (40.0 <= bright_val <= 220.0) else "#ef4444"
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-header">{get_txt('brightness_label')}</div>
                    <div class="metric-card-val" style="color: {br_color};">{bright_val}</div>
                    <div class="metric-card-sub">{'Balanced Exposure' if (40 <= bright_val <= 220) else 'Glare / Dark'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if quality_messages:
            for q_msg in quality_messages:
                if "Fewer than 4" in q_msg:
                    st.error(get_txt("err_fewer_markers"))
                elif "blurry" in q_msg:
                    st.warning(get_txt("err_blurry"))
                elif "overexposed" in q_msg:
                    st.warning(get_txt("err_glare"))
                elif "too dark" in q_msg:
                    st.error(get_txt("err_dark"))
                else:
                    st.warning(q_msg)
        else:
            st.success(get_txt("quality_passed"))

        # Decision-Support Disclaimer
        st.markdown(
            f"""
            <div class="disclaimer-box">
                <span>ℹ️</span> {get_txt('disclaimer_text')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="margin-bottom: 12px;">
                <span class="legend-chip chip-whole">{get_txt('legend_whole')}</span>
                <span class="legend-chip chip-broken">{get_txt('legend_broken')}</span>
                <span class="legend-chip chip-discoloured">{get_txt('legend_discoloured')}</span>
                <span class="legend-chip chip-chalky">{get_txt('legend_chalky')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_disp1, col_disp2 = st.columns(2)
        with col_disp1:
            st.markdown(f"##### {get_txt('panel_orig_title')}")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.image(
                    cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB),
                    caption=get_txt("caption_raw"),
                    use_container_width=True,
                )
            with sc2:
                st.image(
                    cv2.cvtColor(unwarped_bgr, cv2.COLOR_BGR2RGB),
                    caption=f"{get_txt('caption_unwarped')} ({px_per_mm:.2f} px/mm)",
                    use_container_width=True,
                )
        with col_disp2:
            st.markdown(f"##### {get_txt('panel_annotated_title')}")
            st.image(
                cv2.cvtColor(annotated_grains, cv2.COLOR_BGR2RGB),
                caption=f"{get_txt('caption_defects')} ({len(grains_list)} kernels)",
                use_container_width=True,
            )

    # =========================================================================
    # TAB 2: MORPHOMETRICS & DISTRIBUTION
    # =========================================================================
    with tab_morph:
        st.markdown(f"### {get_txt('tab_morphometrics')}")
        if grains_list:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-header">{get_txt('avg_length')}</div>
                        <div class="metric-card-val" style="color: #0284c7;">{stats['avg_length_mm']} ± {stats.get('length_std_mm', 0.0):.2f} mm</div>
                        <div class="metric-card-sub">Baseline: {stats['reference_length_mm']} mm (Std: {stats.get('length_std_mm', 0.0):.2f} mm)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-header">{get_txt('avg_width')}</div>
                        <div class="metric-card-val" style="color: #059669;">{stats['avg_width_mm']} mm</div>
                        <div class="metric-card-sub">Transverse Minor Axis</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m3:
                st.markdown(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-header">{get_txt('avg_ar')}</div>
                        <div class="metric-card-val" style="color: #7c3aed;">{stats['avg_aspect_ratio']}</div>
                        <div class="metric-card-sub">Length-to-Width Ratio</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br/>", unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)
            lengths = [g.length_mm for g in grains_list]

            with ch1:
                fig1, ax1 = plt.subplots(figsize=(6, 3.4), dpi=140)
                ax1.hist(lengths, bins=16, color="#0284c7", edgecolor="white", alpha=0.85)
                ax1.axvline(
                    analyzer.broken_cutoff_length_mm,
                    color="#dc2626",
                    linestyle="--",
                    linewidth=2,
                    label=f"Cutoff ({analyzer.broken_cutoff_length_mm:.1f}mm)",
                )
                ax1.axvline(
                    stats["avg_length_mm"],
                    color="#10b981",
                    linestyle="-",
                    linewidth=2,
                    label=f"Mean ({stats['avg_length_mm']:.2f}mm)",
                )
                ax1.set_title(get_txt("dist_chart_title"), fontsize=11, fontweight="bold")
                ax1.set_xlabel("Length (mm)", fontsize=9)
                ax1.set_ylabel("Kernel Count", fontsize=9)
                ax1.legend(fontsize=8)
                ax1.grid(True, linestyle=":", alpha=0.5)
                st.pyplot(fig1)

            with ch2:
                fig2, ax2 = plt.subplots(figsize=(6, 3.4), dpi=140)
                cats = ["Whole", "Broken", "Discoloured", "Chalky"]
                counts = [
                    stats["whole_count"],
                    stats["broken_count"],
                    stats["discoloured_count"],
                    stats["chalky_count"],
                ]
                bar_colors = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6"]
                bars = ax2.bar(cats, counts, color=bar_colors, edgecolor="white", width=0.52)
                for bar in bars:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, int(yval), ha="center", va="bottom", fontsize=9, fontweight="bold")
                ax2.set_title(get_txt("defect_chart_title"), fontsize=11, fontweight="bold")
                ax2.set_ylabel("Count", fontsize=9)
                ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
                st.pyplot(fig2)

    # =========================================================================
    # TAB 3: GRAIN DATASET
    # =========================================================================
    with tab_data:
        st.markdown(f"### {get_txt('tab_grain_dataset')}")
        if grains_list:
            filter_opt = st.selectbox(
                get_txt("filter_class"),
                options=["All Grains", "Whole", "Broken", "Discoloured", "Chalky"],
            )
            filtered_grains = (
                grains_list if filter_opt == "All Grains"
                else [g for g in grains_list if g.classification == filter_opt]
            )

            df_grains = pd.DataFrame([
                {
                    "Grain ID": g.grain_id,
                    "Classification": g.classification,
                    "Length (mm)": g.length_mm,
                    "Width (mm)": g.width_mm,
                    "Aspect Ratio": g.aspect_ratio,
                    "Area (mm²)": g.area_mm2,
                    "Chalky Ratio": g.chalky_ratio,
                    "Is Broken": g.is_broken,
                    "Is Discoloured": g.is_discoloured,
                    "Is Chalky": g.is_chalky,
                }
                for g in filtered_grains
            ])
            st.dataframe(df_grains, use_container_width=True, height=340)

            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            with col_e1:
                # Raw grain measurements CSV with exact required columns:
                # grain_id, length_mm, width_mm, area_mm2, aspect_ratio, classification
                csv_buf = io.StringIO()
                df_all_export = pd.DataFrame([
                    {
                        "grain_id": g.grain_id,
                        "length_mm": g.length_mm,
                        "width_mm": g.width_mm,
                        "area_mm2": g.area_mm2,
                        "aspect_ratio": g.aspect_ratio,
                        "classification": g.classification,
                    }
                    for g in grains_list
                ])
                df_all_export.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📊 Download CSV (Raw Data)",
                    data=csv_buf.getvalue(),
                    file_name=f"farmgate_grain_batch_{st.session_state.last_saved_batch_id or 'scan'}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_e2:
                pdf_file = "farmgate_quality_certificate.pdf"
                current_sample_id = f"SMP-{st.session_state.last_saved_batch_id or 1001:04d}"
                current_quality_status = "Passed" if quality_metrics.get("status") == "pass" else ("Warning" if quality_metrics.get("status") == "warn" else "Failed")
                temp_thumb = "temp_pdf_thumbnail.png"
                cv2.imwrite(temp_thumb, cv2.resize(annotated_grains, (640, 480)))
                generate_pdf_report(
                    stats,
                    stats["variety"],
                    output_pdf_path=pdf_file,
                    sample_id=current_sample_id,
                    capture_quality_status=current_quality_status,
                    annotated_image_path=temp_thumb,
                )
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📄 Download PDF Certificate",
                    data=pdf_bytes,
                    file_name=f"farmgate_certificate_batch_{st.session_state.last_saved_batch_id or 'scan'}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with col_e3:
                html_report_content = generate_html_report(
                    stats,
                    stats["variety"],
                    sample_id=current_sample_id,
                    capture_quality_status=current_quality_status,
                )
                st.download_button(
                    label="🌐 Download HTML Report",
                    data=html_report_content,
                    file_name=f"farmgate_report_batch_{st.session_state.last_saved_batch_id or 'scan'}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with col_e4:
                _, enc_img = cv2.imencode(".png", annotated_grains)
                st.download_button(
                    label="🖼️ Download Annotated Image",
                    data=enc_img.tobytes(),
                    file_name="farmgate_annotated_defects.png",
                    mime="image/png",
                    use_container_width=True,
                )

            # Feature 5: Mandatory Legal Disclaimer Callout
            st.markdown(
                f"""
                <div style="background-color: #fef2f2; border: 1px solid #ef4444; border-radius: 8px; padding: 12px 16px; margin-top: 16px; color: #991b1b; font-size: 13px; font-weight: 600; text-align: center;">
                    ⚠️ <strong>MANDATORY DISCLAIMER:</strong> {MANDATORY_DISCLAIMER}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 4: PRINTABLE CALIBRATION SHEET
    # =========================================================================
    with tab_sheet:
        render_calibration_sheet_tab()

    # =========================================================================
    # TAB 5: AGMARK QUALITY STANDARDS
    # =========================================================================
    with tab_standards:
        render_standards_tab()

    # =========================================================================
    # TAB 6: MANDI FEEDBACK LOOP
    # =========================================================================
    with tab_mandi:
        render_mandi_tab()

    # =========================================================================
    # TAB 7: BATCH HISTORY
    # =========================================================================
    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()
