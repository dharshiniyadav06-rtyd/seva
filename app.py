"""
Farmgate AI - Smartphone Rice Grain Quality Assaying Dashboard
Bilingual (English & Tamil / தமிழ்) Streamlit Application
Features:
- ArUco Fiducial Perspective Unwarping (1000x1000px, Sub-MM Precision)
- Pre-Analysis Quality Check (Markers, Blur, Glare/Exposure)
- Distance Transform + Watershed Segmentation for Touching Grains
- Variety-Aware Baselines (Sona Masoori, Basmati, Swarna, IR64, Auto-Detect)
- AGMARK Quality Grading (Whole, Broken, Discoloured, Chalky)
- Offline SQLite Database History & Mandi Feedback Loop with MAE Tracking
"""

import os
import io
import time
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
from report_generator import export_grains_to_csv, generate_pdf_report
import database as db


# Page Configuration
st.set_page_config(
    page_title="Farmgate AI — Grain Quality Assaying System",
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
        "app_subtitle": "Offline Smartphone Rice Grain Quality Assaying & AGMARK Grading",
        "badge_text": "ArUco Calibrated • Sub-MM Precision",
        "lang_select": "Language / மொழி",
        "sidebar_sample_input": "📷 Sample Capture",
        "upload_label": "Upload Smartphone Photo",
        "upload_help": "Take a photo of grains spread on the Farmgate AI printable sheet.",
        "demo_btn": "⚡ Demo Sample",
        "demo_help": "Synthesize a realistic smartphone capture with perspective tilt",
        "reset_btn": "🔄 Reset",
        "demo_loading": "Synthesizing camera capture with perspective tilt & grains...",
        "demo_success": "Demo sample loaded successfully!",
        "variety_header": "🌾 Rice Variety Baseline",
        "variety_select": "Select Rice Variety",
        "variety_help": "Applies variety-specific kernel length baseline and 75% broken cutoff.",
        "target_length": "Reference Full Length (mm)",
        "broken_cutoff_label": "Broken Threshold (% of Reference)",
        "advanced_tuner": "🔬 Advanced Vision Tuner",
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
        "tab_new_scan": "🔬 Tab 1: New Scan",
        "tab_mandi": "🤝 Tab 2: Mandi Feedback Loop",
        "tab_history": "📋 Tab 3: Batch History",
        "tab_sheet": "🖨️ Calibration Sheet",
        "tab_standards": "📜 AGMARK Standards",
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
        "agmark_banner_title": "AGMARK QUALITY GRADE",
        "total_grains": "Total Grains",
        "whole_pct": "Whole Grain %",
        "broken_pct": "Broken %",
        "discoloured_pct": "Discoloured %",
        "chalky_pct": "Chalky %",
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
        "sheet_desc": "Print this sheet at 100% scale (Do NOT select 'Fit to Printable Area') on standard A4 paper.",
        "download_sheet": "📥 Download Printable Calibration Sheet (A4 / 300 DPI PNG)",
    },
    "ta": {
        "app_title": "ஃபார்ம்கேட் AI (Farmgate AI)",
        "app_subtitle": "ஆஃப்லைன் ஸ்மார்ட்போன் நெல் தானிய தர ஆய்வு மற்றும் அக்மார்க் தரம் நிர்ணயம்",
        "badge_text": "அரூக்கோ அளவீடு • மில்லிமீட்டர் துல்லியம்",
        "lang_select": "Language / மொழி",
        "sidebar_sample_input": "📷 மாதிரி படம் உள்ளீடு",
        "upload_label": "ஸ்மார்ட்போன் புகைப்படத்தைப் பதிவேற்றவும்",
        "upload_help": "ஃபார்ம்கேட் AI அச்சிடக்கூடிய தாளில் பரப்பப்பட்ட தானியங்களின் படத்தை எடுக்கவும்.",
        "demo_btn": "⚡ மாதிரி டெமோ",
        "demo_help": "சாய்வு கோணத்துடன் கூடிய மாதிரி புகைப்படத்தை உடனடியாக உருவாக்கவும்",
        "reset_btn": "🔄 மீட்டமை",
        "demo_loading": "மாதிரி புகைப்படத்தை உருவாக்குகிறது...",
        "demo_success": "மாதிரி புகைப்படம் வெற்றிகரமாக ஏற்றப்பட்டது!",
        "variety_header": "🌾 நெல் ரகம் & அடிப்படை அளவுகள்",
        "variety_select": "நெல் ரகத்தைத் தேர்ந்தெடுக்கவும்",
        "variety_help": "தேர்ந்தெடுக்கப்பட்ட ரகத்தின் முழு நீளம் மற்றும் 75% உடைசல் அளவுகோல் பயன்படுத்தப்படும்.",
        "target_length": "முழு தானிய குறிப்பு நீளம் (மி.மீ)",
        "broken_cutoff_label": "உடைசல் வரம்பு (% அளவு)",
        "advanced_tuner": "🔬 மேம்பட்ட பார்வை அமைப்புகள்",
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
        "tab_new_scan": "🔬 பகுதி 1: புதிய ஆய்வு",
        "tab_mandi": "🤝 பகுதி 2: மண்டி கருத்துப் பின்னூட்டம்",
        "tab_history": "📋 பகுதி 3: தொகுதி வரலாறு",
        "tab_sheet": "🖨️ அளவுத்திருத்த தாள்",
        "tab_standards": "📜 அக்மார்க் விதிகள்",
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
        "agmark_banner_title": "அக்மார்க் தர மதிப்பீடு (AGMARK GRADE)",
        "total_grains": "மொத்த தானியங்கள்",
        "whole_pct": "முழு தானியம் %",
        "broken_pct": "உடைசல் %",
        "discoloured_pct": "நிறமாற்றம் %",
        "chalky_pct": "சுண்ணாம்புத் தன்மை %",
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
        "sheet_desc": "இந்தத் தாளை 100% அளவில் (அளவு மாற்றாமல்) வழக்கமான A4 தாளில் அச்சிடவும்.",
        "download_sheet": "📥 அளவுத்திருத்த தாள் தரவிறக்கம் (A4 / 300 DPI)",
    },
}


# Custom CSS for Professional Modern UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Tamil:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Tamil', sans-serif;
    }

    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #0D9488 100%);
        border-radius: 14px;
        padding: 22px 28px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 25px;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .hero-badge {
        background: rgba(13, 148, 136, 0.35);
        border: 1px solid #14B8A6;
        color: #5EEAD4;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 2px;
    }

    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .chip-whole { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .chip-broken { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    .chip-discoloured { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .chip-chalky { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }

    .grade-banner {
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .disclaimer-box {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 10px 14px;
        color: #92400E;
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
    if "current_image_bgr" not in st.session_state:
        st.session_state.current_image_bgr = None
    if "image_source_name" not in st.session_state:
        st.session_state.image_source_name = None
    if "last_saved_batch_id" not in st.session_state:
        st.session_state.last_saved_batch_id = None


def get_txt(key: str) -> str:
    """Helper to get translated string for active language."""
    lang = st.session_state.get("lang", "en")
    return TEXTS.get(lang, TEXTS["en"]).get(key, TEXTS["en"].get(key, key))


def render_top_header():
    """Renders the top hero navigation banner."""
    st.markdown(
        f"""
        <div class="hero-header">
            <div>
                <h1 class="hero-title">🌾 {get_txt('app_title')}</h1>
                <p class="hero-subtitle">{get_txt('app_subtitle')}</p>
            </div>
            <div>
                <span class="hero-badge">{get_txt('badge_text')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    initialize_session()

    # --- SIDEBAR: Language Toggle & Controls ---
    with st.sidebar:
        st.markdown(f"### 🌐 {get_txt('lang_select')}")
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

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            load_demo = st.button(
                get_txt("demo_btn"),
                use_container_width=True,
                help=get_txt("demo_help"),
            )
        with col_btn2:
            clear_btn = st.button(get_txt("reset_btn"), use_container_width=True)

        if clear_btn:
            st.session_state.current_image_bgr = None
            st.session_state.image_source_name = None
            st.session_state.last_saved_batch_id = None
            st.rerun()

        # Variety Selection
        st.markdown("---")
        st.markdown(f"### {get_txt('variety_header')}")

        varieties = list(RiceGrainAnalyzer.VARIETY_BASELINES.keys())
        variety_choice = st.selectbox(
            get_txt("variety_select"),
            options=varieties,
            index=0,  # Default: Sona Masoori
            help=get_txt("variety_help"),
        )

        default_len = RiceGrainAnalyzer.VARIETY_BASELINES[variety_choice]["full_length_mm"]
        custom_ref_length = st.number_input(
            get_txt("target_length"),
            min_value=3.0,
            max_value=12.0,
            value=float(default_len),
            step=0.1,
        )

        broken_cutoff_pct = st.slider(
            get_txt("broken_cutoff_label"),
            min_value=50,
            max_value=90,
            value=75,
            step=5,
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
                generate_sample_smartphone_capture(
                    output_path=sample_file,
                    num_grains=48,
                    variety=variety_choice if variety_choice != "Auto-Detect" else "Sona Masoori",
                )
                demo_img = cv2.imread(sample_file)
                st.session_state.current_image_bgr = demo_img
                st.session_state.image_source_name = f"Synthetic Demo ({variety_choice})"
                st.session_state.last_saved_batch_id = None
                st.success(get_txt("demo_success"))
                st.rerun()

        if uploaded_file is not None and st.session_state.image_source_name != uploaded_file.name:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.current_image_bgr = img_bgr
            st.session_state.image_source_name = uploaded_file.name
            st.session_state.last_saved_batch_id = None

    render_top_header()

    # --- MAIN TABS ---
    tab_scan, tab_mandi_loop, tab_history_view, tab_sheet_view, tab_agmark_view = st.tabs([
        get_txt("tab_new_scan"),
        get_txt("tab_mandi"),
        get_txt("tab_history"),
        get_txt("tab_sheet"),
        get_txt("tab_standards"),
    ])

    # =========================================================================
    # TAB 1: NEW SCAN WORKFLOW
    # =========================================================================
    with tab_scan:
        if st.session_state.current_image_bgr is None:
            render_welcome_cards()
        else:
            raw_img = st.session_state.current_image_bgr
            calibrator = GrainImageCalibrator(
                marker_span_mm=marker_span_mm,
                output_size=(1000, 1000),
            )

            # Pre-Analysis Quality Validation
            is_valid_capture, quality_metrics, quality_messages = calibrator.validate_capture_quality(raw_img)

            st.markdown(f"#### {get_txt('quality_check_header')}")
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                m_count = quality_metrics["marker_count"]
                m_color = "#10B981" if m_count == 4 else "#EF4444"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{get_txt('markers_detected_label')}</div>
                        <div class="metric-val" style="color: {m_color};">{m_count} / 4</div>
                        <div class="metric-sub">ArUco Corner IDs: {quality_metrics['detected_ids']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_q2:
                b_score = quality_metrics["blur_score"]
                b_color = "#10B981" if b_score >= 80.0 else "#F59E0B"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{get_txt('blur_score_label')}</div>
                        <div class="metric-val" style="color: {b_color};">{b_score}</div>
                        <div class="metric-sub">{'Sharp Focus (>= 80)' if b_score >= 80 else 'Blurry (< 80)'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_q3:
                bright_val = quality_metrics["mean_brightness"]
                br_color = "#10B981" if (40.0 <= bright_val <= 220.0) else "#EF4444"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{get_txt('brightness_label')}</div>
                        <div class="metric-val" style="color: {br_color};">{bright_val}</div>
                        <div class="metric-sub">{'Balanced Exposure' if (40 <= bright_val <= 220) else 'Glare / Dark'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Display Quality Guidance Alerts if any issues detected
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

            # Run Homography and Grain Analysis
            unwarped_bgr, annotated_orig, px_per_mm, calib_status = calibrator.process(raw_img)

            if not calib_status["success"] or unwarped_bgr is None:
                st.error(f"❌ {get_txt('err_fewer_markers')}")
                st.image(
                    cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB),
                    caption=get_txt("caption_raw"),
                    use_container_width=True,
                )
            else:
                # Initialize Analyzer with user and baseline settings
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

                # Mandatory Decision-Support Disclaimer
                st.markdown(
                    f"""
                    <div class="disclaimer-box">
                        <span>ℹ️</span> {get_txt('disclaimer_text')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # AGMARK Quality Grade Banner
                grade_color = stats["grade_color"]
                st.markdown(
                    f"""
                    <div class="grade-banner" style="background: {grade_color};">
                        <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.9;">
                            {get_txt('agmark_banner_title')} • {stats['variety']}
                        </div>
                        <div style="font-size: 26px; font-weight: 800; margin: 4px 0;">
                            {stats['agmark_grade']}
                        </div>
                        <div style="font-size: 13px; opacity: 0.95;">
                            {stats['grade_rationale']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Summary Metric Cards
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{get_txt('total_grains')}</div>
                            <div class="metric-val">{stats['total_grains']}</div>
                            <div class="metric-sub">{stats['variety']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{get_txt('whole_pct')}</div>
                            <div class="metric-val" style="color: #059669;">{stats['whole_pct']}%</div>
                            <div class="metric-sub">{stats['whole_count']} kernels (>= {analyzer.broken_cutoff_length_mm:.1f}mm)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{get_txt('broken_pct')}</div>
                            <div class="metric-val" style="color: #DC2626;">{stats['broken_pct']}%</div>
                            <div class="metric-sub">{stats['broken_count']} kernels (< {analyzer.broken_cutoff_length_mm:.1f}mm)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{get_txt('discoloured_pct')}</div>
                            <div class="metric-val" style="color: #D97706;">{stats['discoloured_pct']}%</div>
                            <div class="metric-sub">{stats['discoloured_count']} defective</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c5:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{get_txt('chalky_pct')}</div>
                            <div class="metric-val" style="color: #2563EB;">{stats['chalky_pct']}%</div>
                            <div class="metric-sub">{stats['chalky_count']} opaque</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Auto-save to SQLite if not already saved for this scan
                if st.session_state.last_saved_batch_id is None and stats["total_grains"] > 0:
                    new_id = db.save_batch(
                        variety=stats["variety"],
                        total_grains=stats["total_grains"],
                        whole_pct=stats["whole_pct"],
                        broken_pct=stats["broken_pct"],
                        discoloured_pct=stats["discoloured_pct"],
                        chalky_pct=stats["chalky_pct"],
                        agmark_grade=stats["agmark_grade"],
                        notes="Offline smartphone scan",
                    )
                    st.session_state.last_saved_batch_id = new_id

                if st.session_state.last_saved_batch_id:
                    st.info(get_txt("batch_saved_alert").format(st.session_state.last_saved_batch_id))

                st.markdown("<br/>", unsafe_allow_html=True)

                # Side-by-Side Computer Vision Display
                st.markdown(
                    f"""
                    <div style="margin-bottom: 10px;">
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

                # Charts Section
                if grains_list:
                    ch1, ch2 = st.columns(2)
                    lengths = [g.length_mm for g in grains_list]

                    with ch1:
                        fig1, ax1 = plt.subplots(figsize=(6, 3.2), dpi=140)
                        ax1.hist(lengths, bins=16, color="#0284C7", edgecolor="white", alpha=0.85)
                        ax1.axvline(
                            analyzer.broken_cutoff_length_mm,
                            color="#DC2626",
                            linestyle="--",
                            linewidth=2,
                            label=f"75% Cutoff ({analyzer.broken_cutoff_length_mm:.2f}mm)",
                        )
                        ax1.axvline(
                            stats["avg_length_mm"],
                            color="#10B981",
                            linestyle="-",
                            linewidth=2,
                            label=f"Mean ({stats['avg_length_mm']:.2f}mm)",
                        )
                        ax1.set_title(get_txt("dist_chart_title"), fontsize=10, fontweight="bold")
                        ax1.set_xlabel("mm", fontsize=8)
                        ax1.legend(fontsize=7)
                        ax1.grid(True, linestyle=":", alpha=0.5)
                        st.pyplot(fig1)

                    with ch2:
                        fig2, ax2 = plt.subplots(figsize=(6, 3.2), dpi=140)
                        cats = ["Whole", "Broken", "Discoloured", "Chalky"]
                        counts = [
                            stats["whole_count"],
                            stats["broken_count"],
                            stats["discoloured_count"],
                            stats["chalky_count"],
                        ]
                        bar_colors = ["#10B981", "#EF4444", "#F59E0B", "#3B82F6"]
                        bars = ax2.bar(cats, counts, color=bar_colors, edgecolor="white", width=0.52)
                        for bar in bars:
                            yval = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, int(yval), ha="center", va="bottom", fontsize=8, fontweight="bold")
                        ax2.set_title(get_txt("defect_chart_title"), fontsize=10, fontweight="bold")
                        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
                        st.pyplot(fig2)

                    # Export Buttons
                    col_e1, col_e2, col_e3 = st.columns(3)
                    with col_e1:
                        csv_buf = io.StringIO()
                        export_df = pd.DataFrame([
                            {
                                "Grain ID": g.grain_id,
                                "Classification": g.classification,
                                "Length (mm)": g.length_mm,
                                "Width (mm)": g.width_mm,
                                "Aspect Ratio": g.aspect_ratio,
                                "Area (mm²)": g.area_mm2,
                                "Chalky Ratio": g.chalky_ratio,
                            }
                            for g in grains_list
                        ])
                        export_df.to_csv(csv_buf, index=False)
                        st.download_button(
                            label=get_txt("download_csv"),
                            data=csv_buf.getvalue(),
                            file_name=f"farmgate_grain_batch_{st.session_state.last_saved_batch_id or 'scan'}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with col_e2:
                        pdf_file = "farmgate_quality_certificate.pdf"
                        generate_pdf_report(stats, stats["variety"], output_pdf_path=pdf_file)
                        with open(pdf_file, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            label=get_txt("download_pdf"),
                            data=pdf_bytes,
                            file_name=f"farmgate_certificate_batch_{st.session_state.last_saved_batch_id or 'scan'}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    with col_e3:
                        _, enc_img = cv2.imencode(".png", annotated_grains)
                        st.download_button(
                            label=get_txt("download_img"),
                            data=enc_img.tobytes(),
                            file_name="farmgate_annotated_defects.png",
                            mime="image/png",
                            use_container_width=True,
                        )

    # =========================================================================
    # TAB 2: MANDI FEEDBACK LOOP
    # =========================================================================
    with tab_mandi_loop:
        st.markdown(f"### {get_txt('mandi_header')}")
        st.write(get_txt("mandi_desc"))

        # Fetch current MAE metrics
        mae_info = db.calculate_feedback_mae()

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{get_txt('current_mae')}</div>
                    <div class="metric-val" style="color: #0284C7;">±{mae_info['mae']}%</div>
                    <div class="metric-sub">{get_txt('mae_desc')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{get_txt('trader_price_avg')}</div>
                    <div class="metric-val" style="color: #059669;">₹{mae_info['mean_trader_price']:.0f}</div>
                    <div class="metric-sub">Per Quintal (₹/Qtl)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{get_txt('evaluated_batches')}</div>
                    <div class="metric-val">{mae_info['sample_count']}</div>
                    <div class="metric-sub">Ground-Truth Calibrated Lots</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Feedback Input Form
        history_df = db.get_batch_history()

        if len(history_df) == 0:
            st.info("No batches found in SQLite database yet. Perform a scan in Tab 1 first.")
        else:
            col_form1, col_form2 = st.columns([1, 1])

            with col_form1:
                batch_options = [
                    f"Batch #{row['id']} — {row['variety']} ({row['timestamp']}) [Predicted Broken: {row['broken_pct']}%]"
                    for _, row in history_df.iterrows()
                ]
                selected_batch_str = st.selectbox(get_txt("select_batch_feedback"), options=batch_options)
                selected_batch_id = int(selected_batch_str.split(" ")[0].replace("Batch", "").replace("#", ""))
                selected_row = history_df[history_df["id"] == selected_batch_id].iloc[0]

                existing_actual = (
                    float(selected_row["mandi_broken_actual"])
                    if pd.notnull(selected_row["mandi_broken_actual"])
                    else float(selected_row["broken_pct"])
                )
                existing_price = (
                    float(selected_row["mandi_price_offered"])
                    if pd.notnull(selected_row["mandi_price_offered"])
                    else 3200.0
                )
                existing_notes = str(selected_row["notes"]) if pd.notnull(selected_row["notes"]) else ""

                actual_broken = st.number_input(
                    get_txt("actual_broken_input"),
                    min_value=0.0,
                    max_value=100.0,
                    value=float(existing_actual),
                    step=0.5,
                )
                price_offered = st.number_input(
                    get_txt("price_offered_input"),
                    min_value=500.0,
                    max_value=15000.0,
                    value=float(existing_price),
                    step=50.0,
                )
                notes_text = st.text_input(get_txt("feedback_notes"), value=existing_notes)

                if st.button(get_txt("submit_feedback_btn"), use_container_width=True):
                    db.update_mandi_feedback(
                        batch_id=selected_batch_id,
                        mandi_broken_actual=actual_broken,
                        mandi_price_offered=price_offered,
                        notes=notes_text,
                    )
                    st.success(get_txt("feedback_success").format(selected_batch_id))
                    st.rerun()

            with col_form2:
                st.markdown(f"#### {get_txt('comparison_title')}")
                pred_broken_val = float(selected_row["broken_pct"])
                diff_val = abs(pred_broken_val - actual_broken)

                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px;">
                        <p style="margin: 0 0 6px 0; font-size: 13px; color: #64748B;">
                            <strong>Batch ID:</strong> #{selected_batch_id} | <strong>Variety:</strong> {selected_row['variety']}
                        </p>
                        <p style="margin: 0 0 6px 0; font-size: 13px; color: #64748B;">
                            <strong>Assigned AGMARK Grade:</strong> {selected_row['agmark_grade']}
                        </p>
                        <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 10px 0;"/>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span>{get_txt('pred_broken')}:</span>
                            <span style="font-weight: 700; color: #0284C7;">{pred_broken_val:.1f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span>{get_txt('actual_broken')}:</span>
                            <span style="font-weight: 700; color: #10B981;">{actual_broken:.1f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span>{get_txt('pred_diff')}:</span>
                            <span style="font-weight: 800; color: {'#10B981' if diff_val <= 2.5 else '#F59E0B'};">
                                {diff_val:.1f}% ({'High Accuracy' if diff_val <= 2.5 else 'Acceptable Tolerance'})
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # TAB 3: BATCH HISTORY (OFFLINE SQLITE)
    # =========================================================================
    with tab_history_view:
        st.markdown(f"### {get_txt('history_header')}")
        st.write(get_txt("history_desc"))

        df_all = db.get_batch_history()

        if len(df_all) == 0:
            st.info("No batches recorded in offline database yet.")
        else:
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                variety_list = ["All Varieties"] + list(df_all["variety"].unique())
                v_filter = st.selectbox(get_txt("filter_variety"), options=variety_list)
            with col_f2:
                st.write("")

            filtered_history = df_all if v_filter == "All Varieties" else df_all[df_all["variety"] == v_filter]

            # Interactive Data Table
            st.dataframe(
                filtered_history,
                use_container_width=True,
                height=320,
            )

            # Export History CSV
            csv_hist_buf = io.StringIO()
            filtered_history.to_csv(csv_hist_buf, index=False)
            st.download_button(
                label=get_txt("export_history_csv"),
                data=csv_hist_buf.getvalue(),
                file_name="farmgate_offline_batch_history.csv",
                mime="text/csv",
                use_container_width=False,
            )

            # Side-by-Side Batch Comparison
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

    # =========================================================================
    # TAB 4: PRINTABLE CALIBRATION SHEET
    # =========================================================================
    with tab_sheet_view:
        st.markdown(f"### {get_txt('tab_sheet')}")
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

    # =========================================================================
    # TAB 5: AGMARK QUALITY STANDARDS
    # =========================================================================
    with tab_agmark_view:
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


def render_welcome_cards():
    """Renders the initial getting-started guide when no image is loaded."""
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


if __name__ == "__main__":
    main()
