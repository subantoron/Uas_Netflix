import io
import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="🎬 Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# Custom CSS - Netflix Black & Red Theme (FIXED DROPDOWN)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;700;800;900&display=swap');

* {
    font-family: 'Netflix Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background: #000000 !important;
    color: #FFFFFF !important;
    overflow-x: hidden !important;
}

/* Netflix Header Style */
.netflix-header {
    background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.7), rgba(0,0,0,0.9)),
                url('https://assets.nflxext.com/ffe/siteui/vlv3/9c5457b8-9ab0-4a04-9fc1-e608d5670f1a/710d74e0-7158-408e-8d9b-23c219dee5df/IN-en-20210719-popsignuptwoweeks-perspective_alpha_website_small.jpg');
    background-size: cover;
    background-position: center 30%;
    padding: 3rem 2rem 4rem 2rem;
    border-radius: 0;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
    border-bottom: 6px solid #E50914;
    box-shadow: 0 15px 40px rgba(0,0,0,0.9);
    min-height: 220px;
}

.netflix-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(to right, rgba(229,9,20,0.4) 0%, transparent 40%, transparent 60%, rgba(229,9,20,0.4) 100%);
    z-index: 1;
}

.netflix-title {
    font-size: 3rem !important;
    font-weight: 900 !important;
    color: #E50914 !important;
    text-shadow: 3px 3px 0px rgba(0,0,0,0.9), 
                 0 0 30px rgba(229,9,20,0.7) !important;
    letter-spacing: -1px;
    margin-bottom: 0.5rem !important;
    position: relative;
    z-index: 2;
    line-height: 1.1 !important;
}

.netflix-subtitle {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8) !important;
    margin-bottom: 1.5rem !important;
    position: relative;
    z-index: 2;
    opacity: 0.95;
    line-height: 1.3 !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Netflix Sans', sans-serif !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    line-height: 1.2 !important;
    margin-bottom: 0.8rem !important;
}

/* ========== FIXED SELECTBOX - DROPDOWN TEXT COLOR ========== */
/* Container utama selectbox */
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 2px solid #E50914 !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.6rem !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
}

/* Text yang ditampilkan di selectbox */
.stSelectbox > div > div > div {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Dropdown/menu options */
div[role="listbox"] {
    background: #FFFFFF !important;
    border: 2px solid #E50914 !important;
    border-radius: 8px !important;
    margin-top: 4px !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.8) !important;
}

/* Setiap option dalam dropdown */
div[role="listbox"] div[role="option"] {
    color: #000000 !important;
    font-weight: 600 !important;
    background: #FFFFFF !important;
    font-size: 14px !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid #EEEEEE !important;
}

/* Option hover state */
div[role="listbox"] div[role="option"]:hover {
    background: #E50914 !important;
    color: #FFFFFF !important;
}

/* Option selected state */
div[role="listbox"] div[role="option"][aria-selected="true"] {
    background: #E50914 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Input field text color */
.stTextInput > div > div > input {
    color: #000000 !important;
    font-weight: 600 !important;
    background: #FFFFFF !important;
}

/* Label selectbox */
.stSelectbox label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
}

/* Enhanced Cards */
.recommendation-card {
    background: linear-gradient(145deg, #0F0F0F 0%, #1A1A1A 100%) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    border-left: 5px solid #E50914 !important;
    border: 1px solid #2D2D2D !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.8) !important;
}

.recommendation-card:hover {
    transform: translateY(-3px) !important;
    border-left: 5px solid #FF0000 !important;
    border-color: #E50914 !important;
    box-shadow: 0 12px 30px rgba(229, 9, 20, 0.4) !important;
}

.recommendation-card h3 {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 1rem !important;
}

/* Enhanced Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1rem !important;
    text-align: center !important;
    border: 1px solid #2D2D2D !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7) !important;
    margin-bottom: 1rem !important;
    min-height: 150px !important;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #E50914, #FF0000, #E50914) !important;
}

.metric-card:hover {
    transform: translateY(-3px) !important;
    border-color: #E50914 !important;
    box-shadow: 0 12px 30px rgba(229, 9, 20, 0.3) !important;
}

.metric-card h3 {
    color: #FFFFFF !important;
    font-size: 2rem !important;
    margin-bottom: 0.3rem !important;
    font-weight: 900 !important;
}

.metric-card div:not(.metric-card h3) {
    color: #E0E0E0 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    margin-bottom: 0.3rem !important;
}

.metric-card div:last-child {
    color: #BBBBBB !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* Enhanced Badges */
.badge {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0.5rem 1rem !important;
    background: linear-gradient(135deg, #E50914 0%, #B81D24 100%) !important;
    color: white !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    margin-right: 0.6rem !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4) !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
    border: 1px solid transparent !important;
    min-height: 34px !important;
}

.badge:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(229, 9, 20, 0.6) !important;
    border-color: #FFFFFF !important;
}

.badge-movie {
    background: linear-gradient(135deg, #E50914 0%, #8B0000 100%) !important;
}

.badge-year {
    background: linear-gradient(135deg, #333333 0%, #666666 100%) !important;
}

.badge-rating {
    background: linear-gradient(135deg, #F5C518 0%, #FFD700 100%) !important;
    color: #000000 !important;
    font-weight: 800 !important;
}

/* Enhanced Buttons */
.stButton > button {
    background: linear-gradient(135deg, #E50914 0%, #B81D24 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.8rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.4) !important;
    position: relative !important;
    overflow: hidden !important;
    min-height: 48px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(229, 9, 20, 0.6) !important;
    background: linear-gradient(135deg, #FF0000 0%, #E50914 100%) !important;
}

/* Enhanced Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%) !important;
    padding: 0.8rem !important;
    border-radius: 12px !important;
    margin-bottom: 1.5rem !important;
    border: 1px solid #2D2D2D !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.7) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 700 !important;
    color: #AAAAAA !important;
    border: 1px solid transparent !important;
    transition: all 0.3s ease !important;
    font-size: 0.95rem !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    border-color: #E50914 !important;
    background: rgba(229, 9, 20, 0.1) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E50914 0%, #B81D24 100%) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4) !important;
    border-color: #E50914 !important;
    transform: translateY(-1px) !important;
}

/* Enhanced Input Fields */
.stTextInput > div > div {
    background: #FFFFFF !important;
    border-radius: 8px !important;
    border: 2px solid #E50914 !important;
    padding: 0.4rem 0.6rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

.stTextInput > div > div:hover {
    border-color: #FF0000 !important;
    box-shadow: 0 0 12px rgba(229,9,20,0.4) !important;
}

/* Glass Panel Effect */
.glass-panel {
    background: rgba(15, 15, 15, 0.9) !important;
    backdrop-filter: blur(15px);
    border-radius: 12px;
    border: 1px solid rgba(229, 9, 20, 0.4);
    padding: 1.5rem;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7);
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem !important;
}

/* Stats Cards */
.stats-card {
    background: linear-gradient(145deg, #0F0F0F, #1A1A1A);
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #2D2D2D;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem !important;
}

.stats-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #E50914, #FF0000);
}

.stats-card:hover {
    border-color: #E50914;
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(229, 9, 20, 0.3);
}

/* Tags */
.tag {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    margin: 0.3rem;
    border-radius: 18px;
    font-size: 0.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #E50914 0%, #B81D24 100%);
    color: white !important;
    border: 1px solid transparent;
    box-shadow: 0 3px 10px rgba(229, 9, 20, 0.3);
    transition: all 0.2s ease;
}

.tag:hover {
    transform: translateY(-2px);
    border-color: #FFFFFF;
    box-shadow: 0 5px 15px rgba(229, 9, 20, 0.5);
}

/* Sidebar Enhancement */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%) !important;
    border-right: 3px solid #E50914 !important;
    box-shadow: 5px 0 20px rgba(0, 0, 0, 0.8) !important;
    padding: 1.5rem 1rem !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #E50914 0%, #B81D24 100%) !important;
    color: white !important;
    border: 1px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4) !important;
    min-height: 42px !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6) !important;
}

/* Radio Button Enhancement */
.stRadio > div {
    background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%) !important;
    padding: 1rem !important;
    border-radius: 12px !important;
    border: 1px solid #2D2D2D !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.7) !important;
    margin-bottom: 1.2rem !important;
}

.stRadio > div > label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Expander Enhancement */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: 1px solid #2D2D2D !important;
    padding: 1rem 1.2rem !important;
    font-size: 1rem !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.6) !important;
    transition: all 0.2s ease !important;
    margin: 1.2rem 0 0.5rem 0 !important;
}

.streamlit-expanderHeader:hover {
    border-color: #E50914 !important;
}

.streamlit-expanderContent {
    background: #0F0F0F !important;
    border-radius: 0 0 8px 8px !important;
    border: 1px solid #2D2D2D !important;
    border-top: none !important;
    padding: 1.2rem !important;
    margin-bottom: 1.2rem !important;
}

/* Divider Enhancement */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #E50914, transparent);
    margin: 2rem 0;
    border-radius: 2px;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0F0F0F;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #E50914, #B81D24);
    border-radius: 4px;
    border: 1px solid #0F0F0F;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #FF0000, #E50914);
}

/* Alert Box Enhancement */
.stAlert {
    background: rgba(15, 15, 15, 0.95) !important;
    border: 1px solid #E50914 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(15px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7) !important;
    margin: 1.2rem 0 !important;
}

/* Column Spacing Fix */
[data-testid="column"] {
    padding: 0.3rem !important;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .netflix-title {
        font-size: 2.2rem !important;
    }
    
    .netflix-subtitle {
        font-size: 1.2rem !important;
    }
    
    .metric-card h3 {
        font-size: 1.7rem !important;
    }
    
    .recommendation-card h3 {
        font-size: 1.3rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1rem !important;
        font-size: 0.85rem !important;
    }
}

/* Ensure proper text rendering */
* {
    text-rendering: optimizeLegibility !important;
    -webkit-font-smoothing: antialiased !important;
}

/* Text color for regular content */
p, span, div, label {
    color: #F5F5F5 !important;
}

/* Special styling for sidebar content */
[data-testid="stSidebar"] div {
    color: #F5F5F5 !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4 {
    color: #E50914 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.8rem !important;
}

/* Fix for overlapping content */
.css-1lcbmhc {
    overflow-wrap: break-word !important;
    word-wrap: break-word !important;
}

/* Ensure proper line heights */
.css-1d391kg {
    line-height: 1.4 !important;
}

/* Better spacing in sidebar */
.sidebar-section {
    margin-bottom: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    border-bottom: 1px solid rgba(229, 9, 20, 0.2) !important;
}

.sidebar-section:last-child {
    border-bottom: none !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* ========== TITLE SELECTOR STYLING ========== */
/* Make sure the selected text in dropdown is visible */
div[data-baseweb="select"] > div:first-child {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Select label styling */
.stSelectbox label[data-testid="stWidgetLabel"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
}

/* Slider labels */
.stSlider label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Checkbox labels */
.stCheckbox label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Text input labels */
.stTextInput label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI helper (alert HTML aman)
# -----------------------------
def ui_alert(kind: str, html: str):
    palette = {
        "info": ("#E50914", "rgba(229, 9, 20, 0.15)", "ℹ️"),
        "success": ("#00B894", "rgba(0, 184, 148, 0.15)", "✅"),
        "warning": ("#FDCB6E", "rgba(253, 203, 110, 0.15)", "⚠️"),
        "error": ("#E50914", "rgba(229, 9, 20, 0.15)", "⛔"),
    }
    border, bg, icon = palette.get(kind, palette["info"])
    st.markdown(
        f"""
        <div style="
            border: 1px solid {border}40;
            background: {bg};
            border-left: 5px solid {border};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 1.2rem 0;
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
            word-wrap: break-word;
            overflow-wrap: break-word;
        ">
            <div style="display: flex; gap: 1rem; align-items: flex-start;">
                <div style="font-size: 1.5rem; line-height: 1; color: {border}; flex-shrink: 0;">{icon}</div>
                <div style="flex: 1; color: #FFFFFF !important; font-size: 0.95rem; font-weight: 600; line-height: 1.4;">{html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Helper Functions
# -----------------------------
def _normalize_text(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"unknown", "nan", "none", "null", ""}:
        return ""
    s = s.replace("&", " and ")
    s = s.lower()
    s = re.sub(r"[^09a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x)
    if s.strip().lower() in {"unknown", "nan", "none", "null", ""}:
        return ""
    return s

@st.cache_data(show_spinner=False)
def load_data_from_path(path_str: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_str)
        if len(df) == 0:
            st.error("File CSV kosong!")
        return df
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        if len(df) == 0:
            st.error("File CSV kosong!")
        return df
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    column_mapping = {
        "show id": "show_id",
        "show_id": "show_id",
        "type": "type",
        "title": "title",
        "director": "director",
        "director_list": "director",
        "cast": "cast",
        "cast_list": "cast",
        "actor": "cast",
        "country": "country",
        "country_list": "country",
        "date_added": "date_added",
        "date_added_iso": "date_added",
        "release year": "release_year",
        "release_year": "release_year",
        "rating": "rating",
        "duration": "duration",
        "listed in": "listed_in",
        "listed_in": "listed_in",
        "listed_in_list": "listed_in",
        "description": "description",
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    expected = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    df["type"] = df["type"].astype(str).str.strip()
    df["type"] = df["type"].apply(lambda x: "TV Show" if str(x).lower() == "tv show" else x)
    df["type"] = df["type"].apply(lambda x: "Movie" if str(x).lower() == "movie" else x)

    text_cols = ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
        df[c] = df[c].replace({"unknown": "", "Unknown": "", "nan": "", "NaN": "", "None": "", "none": ""})

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

    df["soup"] = (
        df["title"].map(_normalize_text)
        + " "
        + df["type"].map(_normalize_text)
        + " "
        + df["director"].map(_normalize_text)
        + " "
        + df["cast"].map(_normalize_text)
        + " "
        + df["country"].map(_normalize_text)
        + " "
        + df["listed_in"].map(_normalize_text)
        + " "
        + df["rating"].map(_normalize_text)
        + " "
        + df["description"].map(_normalize_text)
    ).str.strip()

    df["display_title"] = df["title"].astype(str) + " (" + df["type"].astype(str) + ", " + df["release_year"].astype(str) + ")"

    dup = df["display_title"].duplicated(keep=False)
    if dup.any():
        df.loc[dup, "display_title"] = df.loc[dup].apply(
            lambda r: f"{r['title']} ({r['type']}, {r['release_year']}) — {r.get('show_id','')}",
            axis=1,
        )

    if df["show_id"].astype(str).duplicated().any():
        df["show_id"] = df.apply(lambda r: f"{r.get('show_id','')}_{r.name}", axis=1)

    return df

@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(corpus: pd.Series):
    if corpus is None or len(corpus) == 0:
        return None, None
    if corpus.astype(str).str.strip().eq("").all():
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus.astype(str).values)
    return vectorizer, tfidf_matrix

def recommend_by_index(
    idx: int,
    df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
    same_type: bool = True,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    try:
        if idx < 0 or idx >= len(df):
            return pd.DataFrame()

        sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        order = sims.argsort()[::-1]
        order = order[order != idx]

        recs = df.iloc[order].copy()
        recs["similarity"] = sims[order]

        if same_type:
            selected_type = df.iloc[idx].get("type", "")
            if selected_type:
                recs = recs[recs["type"] == selected_type]

        if year_min is not None:
            recs = recs[recs["release_year"] >= year_min]
        if year_max is not None:
            recs = recs[recs["release_year"] <= year_max]

        return recs.head(top_n)
    except Exception as e:
        st.error(f"Error rekomendasi: {e}")
        return pd.DataFrame()

def recommend_by_query(
    query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 10,
    type_filter: str = "All",
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    try:
        q = _normalize_text(query)
        if not q:
            return pd.DataFrame()

        q_vec = vectorizer.transform([q])
        if q_vec.nnz == 0:
            return pd.DataFrame()

        sims = linear_kernel(q_vec, tfidf_matrix).flatten()
        order = sims.argsort()[::-1]

        recs = df.iloc[order].copy()
        recs["similarity"] = sims[order]

        if type_filter != "All":
            recs = recs[recs["type"] == type_filter]

        if year_min is not None:
            recs = recs[recs["release_year"] >= year_min]
        if year_max is not None:
            recs = recs[recs["release_year"] <= year_max]

        return recs.head(top_n)
    except Exception as e:
        st.error(f"Error pencarian: {e}")
        return pd.DataFrame()

def split_and_count(series: pd.Series, sep: str = ",", top_k: int = 10) -> pd.Series:
    try:
        s = series.fillna("").astype(str).replace({"unknown": "", "Unknown": "", "nan": "", "NaN": ""})
        exploded = s.str.split(sep).explode().astype(str).str.strip()
        exploded = exploded[exploded != ""]
        return exploded.value_counts().head(top_k)
    except Exception:
        return pd.Series(dtype=int)

def display_recommendation_card(r: pd.Series, rank: int):
    try:
        similarity = float(r.get("similarity", 0.0))
        title = _safe_str(r.get("title", ""))
        content_type = _safe_str(r.get("type", ""))
        year = r.get("release_year", "")
        rating = _safe_str(r.get("rating", ""))
        genre = _safe_str(r.get("listed_in", ""))
        description = _safe_str(r.get("description", "No description available"))
        director = _safe_str(r.get("director", "Not specified"))
        country = _safe_str(r.get("country", "Not specified"))
        duration = _safe_str(r.get("duration", "Not specified"))

        st.markdown(
            f"""
        <div class="recommendation-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem; flex-wrap: wrap; gap: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                    <div style="background: linear-gradient(135deg, #E50914 0%, #B81D24 100%); color: white; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 900; font-size: 1.1rem; border: 1px solid white; box-shadow: 0 4px 12px rgba(229,9,20,0.5); flex-shrink: 0;">
                        #{rank}
                    </div>
                    <h3 style="margin: 0; color: #FFFFFF !important; font-size: 1.5rem; font-weight: 800; text-shadow: 2px 2px 0 #000000; word-break: break-word; line-height: 1.2;">{title}</h3>
                </div>
                <div style="background: linear-gradient(135deg, #E50914 0%, #B81D24 100%); color: #FFFFFF !important; font-weight: 900 !important; font-size: 1rem; padding: 0.5rem 1rem; border-radius: 18px; border: 1px solid #FFFFFF; box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4); min-width: 85px; text-align: center; text-shadow: 1px 1px 0px rgba(0,0,0,0.5); white-space: nowrap; flex-shrink: 0;">
                    {similarity:.1%}
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<span class="badge badge-movie">🎬 {content_type}</span>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span class="badge badge-year">📅 {year}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="badge badge-rating">⭐ {rating}</span>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<span class="badge">⏱️ {duration}</span>', unsafe_allow_html=True)

        if genre and genre != "Not specified":
            st.markdown(
                """
            <div style="margin: 1.2rem 0;">
                <strong style="color: #E50914 !important; font-size: 1.1rem; font-weight: 800; text-shadow: 1px 1px 0 #000000;">🎭 GENRE</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )
            genres = [g.strip() for g in str(genre).split(",") if g.strip()]
            tags_html = "".join([f'<span class="tag">{g}</span>' for g in genres[:5]])
            st.markdown(f'<div style="margin-bottom: 1.2rem; word-wrap: break-word;">{tags_html}</div>', unsafe_allow_html=True)

        if description and description != "No description available":
            with st.expander("📖 DESKRIPSI LENGKAP", expanded=False):
                st.markdown(
                    f"""
                <div style="font-size: 0.95rem; color: #CCCCCC !important; line-height: 1.5; padding: 1.2rem; background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%); border-radius: 10px; border-left: 4px solid #E50914; border: 1px solid #2D2D2D; box-shadow: inset 0 1px 6px rgba(0,0,0,0.5); word-wrap: break-word;">
                    {description}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        col_details1, col_details2 = st.columns(2)
        with col_details1:
            if director and director != "Not specified":
                st.markdown(
                    f"""
                <div class="stats-card">
                    <strong style="color: #E50914 !important; font-size: 0.95rem; font-weight: 800; text-shadow: 1px 1px 0 #000000;">🎬 SUTRADARA</strong>
                    <div style="color: #FFFFFF !important; font-size: 0.95rem; font-weight: 700; margin-top: 0.6rem; padding: 0.6rem; background: rgba(229,9,20,0.1); border-radius: 6px; border-left: 3px solid #E50914; word-wrap: break-word;">{director}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        with col_details2:
            if country and country != "Not specified":
                st.markdown(
                    f"""
                <div class="stats-card">
                    <strong style="color: #E50914 !important; font-size: 0.95rem; font-weight: 800; text-shadow: 1px 1px 0 #000000;">🌍 NEGARA</strong>
                    <div style="color: #FFFFFF !important; font-size: 0.95rem; font-weight: 700; margin-top: 0.6rem; padding: 0.6rem; background: rgba(229,9,20,0.1); border-radius: 6px; border-left: 3px solid #E50914; word-wrap: break-word;">{country}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Error card: {e}")

def display_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    st.markdown(
        f"""
    <div class="metric-card">
        <div style="font-size: 2.2rem; margin-bottom: 0.8rem; color: #E50914; filter: drop-shadow(0 0 6px rgba(229,9,20,0.5));">{icon}</div>
        <h3>{value}</h3>
        <div style="font-weight: 800; color: #FFFFFF !important; font-size: 1rem; margin-bottom: 0.4rem; text-shadow: 1px 1px 0 #000000; word-wrap: break-word;">{title}</div>
        <div style="font-size: 0.9rem; color: #BBBBBB !important; font-weight: 600; text-shadow: 1px 1px 1px rgba(0,0,0,0.5); word-wrap: break-word;">{subtitle}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def create_dashboard_stats(df: pd.DataFrame):
    stats = {}
    stats["total"] = len(df)
    stats["movies"] = len(df[df["type"] == "Movie"])
    stats["tv_shows"] = len(df[df["type"] == "TV Show"])
    stats["other"] = stats["total"] - stats["movies"] - stats["tv_shows"]

    valid_years = df["release_year"][df["release_year"] > 0]
    if len(valid_years) > 0:
        stats["min_year"] = int(valid_years.min())
        stats["max_year"] = int(valid_years.max())
        stats["avg_year"] = int(valid_years.mean())
    else:
        stats["min_year"] = 1900
        stats["max_year"] = datetime.now().year
        stats["avg_year"] = 2000

    stats["unique_types"] = len(df["type"].unique())
    stats["unique_countries"] = len(df["country"].unique())
    stats["unique_genres"] = len(df["listed_in"].unique())
    return stats

# -----------------------------
# Netflix Style Header
# -----------------------------
st.markdown(
    """
<div class="netflix-header">
    <div style="position: relative; z-index: 2;">
        <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <div style="font-size: 3.5rem; color: #E50914; filter: drop-shadow(0 0 15px rgba(229,9,20,0.7)); flex-shrink: 0;">🎬</div>
            <div style="flex: 1; min-width: 250px;">
                <h1 class="netflix-title" style="word-break: break-word; overflow-wrap: break-word;">NETFLIX RECOMMENDER</h1>
                <p class="netflix-subtitle" style="word-break: break-word; overflow-wrap: break-word;">Sistem Rekomendasi Berbasis Konten</p>
            </div>
        </div>
        <p style="position: relative; z-index: 2; color: #F0F0F0 !important; font-weight: 600; font-size: 1rem; max-width: 100%; line-height: 1.5; text-shadow: 1px 1px 3px rgba(0,0,0,0.8); margin-bottom: 1.2rem; word-wrap: break-word;">
            Temukan konten yang paling sesuai dengan preferensi Anda menggunakan <strong style="color: #E50914 !important; font-weight: 800; text-shadow: 1px 1px 0 #000000;">Content-Based Filtering</strong> dengan <strong style="color: #E50914 !important; font-weight: 800; text-shadow: 1px 1px 0 #000000;">TF-IDF</strong> dan <strong style="color: #E50914 !important; font-weight: 800; text-shadow: 1px 1px 0 #000000;">Cosine Similarity</strong>.
        </p>
        <div style="display: flex; gap: 0.8rem; margin-top: 1.5rem; flex-wrap: wrap; position: relative; z-index: 2;">
            <span class="badge badge-movie">🎯 REKOMENDASI PRESISI</span>
            <span class="badge badge-year">⚡ PEMROSESAN CEPAT</span>
            <span class="badge badge-rating">📌 BERBASIS KONTEN</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar - FIXED & ORGANIZED
# -----------------------------
with st.sidebar:
    # Sidebar Header
    st.markdown(
        """
    <div class="sidebar-section" style="text-align: center; padding: 1.5rem 1rem;
                background: linear-gradient(135deg, rgba(229, 9, 20, 0.2) 0%, rgba(184, 29, 36, 0.2) 100%);
                border-radius: 10px; border: 1px solid rgba(229, 9, 20, 0.4);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(15px);
                position: relative;
                overflow: hidden;
                word-wrap: break-word;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #E50914, #FF0000);"></div>
        <div style="font-size: 3rem; color: #E50914; margin-bottom: 0.8rem; filter: drop-shadow(0 0 10px rgba(229,9,20,0.5));">🎬</div>
        <h3 style="color: #FFFFFF !important; margin: 0; font-size: 1.5rem; font-weight: 900; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); line-height: 1.1;">NETFLIX</h3>
        <p style="color: rgba(255, 255, 255, 0.95); font-size: 0.95rem; margin-top: 0.3rem; font-weight: 700; line-height: 1.2;">Recommender System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation Menu
    st.markdown('<div class="sidebar-section"><h4 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 800; font-size: 1.1rem;">🧭 MENU NAVIGASI</h4></div>', unsafe_allow_html=True)
    page = st.radio(
        "Pilih Halaman:",
        ["🎯 REKOMENDASI", "📊 DASHBOARD ANALITIK", "🤖 TENTANG SISTEM"],
        index=0,
        label_visibility="collapsed",
        key="nav_menu"
    )

    st.divider()

    # Data Configuration
    st.markdown('<div class="sidebar-section"><h4 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 800; font-size: 1.1rem;">⚙️ KONFIGURASI DATA</h4></div>', unsafe_allow_html=True)
    with st.expander("📁 PENGATURAN DATASET", expanded=True):
        uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"], help="Unggah file dataset Netflix dalam format CSV")
        use_local = st.checkbox("Gunakan dataset lokal", value=True, help="Gunakan file netflix_titles.csv yang tersedia")

    # System Status
    st.divider()
    st.markdown('<div class="sidebar-section"><h4 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 800; font-size: 1.1rem;">📊 STATUS SISTEM</h4></div>', unsafe_allow_html=True)

    # Quick Filter
    st.markdown('<div class="sidebar-section"><h4 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 800; font-size: 1.1rem;">🔍 FILTER CEPAT</h4></div>', unsafe_allow_html=True)

    # Initialize session state
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "tfidf_matrix" not in st.session_state:
        st.session_state.tfidf_matrix = None

# -----------------------------
# Load Data
# -----------------------------
raw_df = None
data_loaded = False

if uploaded is not None:
    with st.spinner("🔄 Memuat dataset..."):
        raw_df = load_data_from_upload(uploaded.getvalue())
        if not raw_df.empty:
            data_loaded = True
            ui_alert("success", f"<b>Dataset berhasil dimuat:</b> {len(raw_df):,} baris data")
        else:
            ui_alert("error", "Dataset kosong atau tidak valid!")
elif use_local:
    if DEFAULT_DATA_PATH.exists():
        with st.spinner("🔄 Memuat dataset lokal..."):
            raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
            if not raw_df.empty:
                data_loaded = True
                ui_alert("success", f"<b>Dataset lokal berhasil dimuat:</b> {len(raw_df):,} baris data")
            else:
                ui_alert("error", "File lokal kosong!")
    else:
        ui_alert("warning", "File lokal tidak ditemukan. Pastikan <code>netflix_titles.csv</code> ada di folder yang sama.")

if not data_loaded:
    ui_alert(
        "info",
        """
        <div style="padding: 1rem;">
            <h4 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 800; text-shadow: 1px 1px 0 #000000; font-size: 1.2rem;">📋 CARA MENGGUNAKAN</h4>
            <ol style="color: #CCCCCC !important; margin-left: 1.2rem; line-height: 1.5; font-weight: 600; font-size: 0.9rem;">
                <li>Unggah file CSV dataset Netflix, atau</li>
                <li>Letakkan file <code style="color: #E50914 !important; font-weight: 800;">netflix_titles.csv</code> di folder aplikasi</li>
            </ol>
            <div style="color: #CCCCCC !important; margin-top: 0.8rem; font-weight: 600; font-size: 0.9rem;">
                Format dataset: CSV dengan kolom standar Netflix (title, type, description, dsb).
            </div>
        </div>
        """,
    )

    with st.expander("🎮 COBA DEMO DENGAN DATA CONTOH", expanded=False):
        if st.button("Muat Data Contoh", type="primary", use_container_width=True):
            sample_data = {
                "show_id": ["s1", "s2", "s3", "s4", "s5"],
                "type": ["Movie", "Movie", "TV Show", "Movie", "TV Show"],
                "title": ["The Dark Knight", "Inception", "Stranger Things", "Pulp Fiction", "Breaking Bad"],
                "director": ["Christopher Nolan", "Christopher Nolan", "The Duffer Brothers", "Quentin Tarantino", "Vince Gilligan"],
                "cast": [
                    "Christian Bale, Heath Ledger",
                    "Leonardo DiCaprio, Joseph Gordon-Levitt",
                    "Millie Bobby Brown, Finn Wolfhard",
                    "John Travolta, Uma Thurman",
                    "Bryan Cranston, Aaron Paul",
                ],
                "country": ["USA", "USA", "USA", "USA", "USA"],
                "release_year": [2008, 2010, 2016, 1994, 2008],
                "rating": ["PG-13", "PG-13", "TV-14", "R", "TV-MA"],
                "duration": ["152 min", "148 min", "4 Seasons", "154 min", "5 Seasons"],
                "listed_in": ["Action, Crime, Drama", "Action, Sci-Fi, Thriller", "Drama, Fantasy, Horror", "Crime, Drama", "Crime, Drama, Thriller"],
                "description": [
                    "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham...",
                    "A thief who steals corporate secrets through the use of dream-sharing technology...",
                    "When a young boy vanishes, a small town uncovers a mystery...",
                    "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine...",
                    "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing...",
                ],
            }
            raw_df = pd.DataFrame(sample_data)
            data_loaded = True
            ui_alert("success", "🎉 Data contoh berhasil dimuat! Sistem siap digunakan.")

    if not data_loaded:
        st.stop()

# -----------------------------
# Process Data
# -----------------------------
try:
    with st.spinner("⚙️ Memproses data..."):
        df = prepare_data(raw_df)

        if df.empty:
            ui_alert("error", "DataFrame kosong setelah pemrosesan!")
            st.stop()

        vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])
        if vectorizer is None or tfidf_matrix is None:
            ui_alert("error", "Model tidak dapat dibangun (teks kosong / vocabulary kosong).")
            st.stop()

        st.session_state.df = df
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.data_loaded = True
except Exception as e:
    ui_alert("error", f"Error dalam memproses data: <code>{_safe_str(e)}</code>")
    st.stop()

df = st.session_state.df
vectorizer = st.session_state.vectorizer
tfidf_matrix = st.session_state.tfidf_matrix

stats = create_dashboard_stats(df)

# Sidebar status + stats
with st.sidebar:
    # Display system status
    if st.session_state.data_loaded:
        ui_alert(
            "success",
            """
            <div style="display: flex; align-items: center; gap: 0.8rem; word-wrap: break-word;">
                <div style="font-size: 1.5rem; color: #00B894; flex-shrink: 0;">✅</div>
                <div style="flex: 1;">
                    <div style="font-weight: 800; color: #00B894 !important; font-size: 0.95rem; text-shadow: 1px 1px 0 #000000; line-height: 1.1;">SISTEM AKTIF</div>
                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 600; line-height: 1.2;">Dataset berhasil diproses</div>
                </div>
            </div>
            """,
        )
    
    # Display dataset stats
    st.markdown(
        f"""
        <div class="glass-panel" style="padding: 1.2rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 1.2rem; flex-wrap: wrap; gap: 0.8rem;">
                <div style="flex: 1; min-width: 100px;">
                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 600; margin-bottom: 0.4rem; line-height: 1.1;">TOTAL DATA</div>
                    <div style="font-size: 1.7rem; font-weight: 900; color: #E50914 !important; text-shadow: 1px 1px 0 #000000; line-height: 1;">{stats['total']:,}</div>
                </div>
                <div style="flex: 1; min-width: 100px; text-align: right;">
                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 600; margin-bottom: 0.4rem; line-height: 1.1;">JENIS UNIK</div>
                    <div style="font-size: 1.7rem; font-weight: 900; color: #E50914 !important; text-shadow: 1px 1px 0 #000000; line-height: 1;">{stats['unique_types']}</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem;">
                <div style="flex: 1; min-width: 80px;">
                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 600; margin-bottom: 0.4rem; line-height: 1.1;">FILM</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: #00B894 !important; text-shadow: 1px 1px 0 #000000; line-height: 1;">{stats['movies']:,}</div>
                </div>
                <div style="flex: 1; min-width: 80px; text-align: right;">
                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 600; margin-bottom: 0.4rem; line-height: 1.1;">SERIAL TV</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: #FDCB6E !important; text-shadow: 1px 1px 0 #000000; line-height: 1;">{stats['tv_shows']:,}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Filter Slider
    min_year = stats.get("min_year", 1900)
    max_year = stats.get("max_year", datetime.now().year)
    sidebar_year_range = st.slider("Tahun Rilis", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="sidebar_year")

unique_types = sorted([t for t in df["type"].unique().tolist() if t and str(t) != "nan"])
type_options = ["All"] + unique_types

min_year = stats.get("min_year", 1900)
max_year = stats.get("max_year", datetime.now().year)

# -----------------------------
# Page: Recommendation (DIPERBAIKI)
# -----------------------------
if page == "🎯 REKOMENDASI":
    tabs = st.tabs(["🎬 BERDASARKAN JUDUL", "🔍 BERDASARKAN KATA KUNCI", "⭐ KONTEN POPULER"])

    # Tab 1 - Berdasarkan Judul (DIPERBAIKI)
    with tabs[0]:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<h3 style="color: #E50914 !important; margin-bottom: 1.2rem; font-weight: 800;">🎯 PILIH KONTEN UNTUK DIREKOMENDASIKAN</h3>', unsafe_allow_html=True)

            # Filter Tipe Konten - DIPERBAIKI dengan label yang jelas
            st.markdown('<div class="glass-panel" style="padding: 1.5rem;">', unsafe_allow_html=True)
            
            # Filter Tipe Konten dengan label terpisah
            st.markdown('<p style="color: #FFFFFF !important; font-weight: 700; margin-bottom: 0.5rem; font-size: 0.95rem;">🎭 FILTER TIPE KONTEN</p>', unsafe_allow_html=True)
            
            filter_type_for_selector = st.selectbox(
                "Pilih tipe konten:",
                options=type_options,
                index=0,
                help="Filter berdasarkan Movie atau TV Show",
                label_visibility="collapsed",
                key="filter_type_title",
            )

            if filter_type_for_selector == "All":
                selector_df = df
            else:
                selector_df = df[df["type"] == filter_type_for_selector]

            # Pilih Judul - DIPERBAIKI dengan label terpisah
            st.markdown('<p style="color: #FFFFFF !important; font-weight: 700; margin: 1.2rem 0 0.5rem 0; font-size: 0.95rem;">📝 PILIH JUDUL</p>', unsafe_allow_html=True)

            if len(selector_df) > 0:
                options = selector_df["display_title"].tolist()
                selected_display = st.selectbox(
                    "Pilih judul untuk rekomendasi:",
                    options=options,
                    index=0,
                    help="Pilih satu judul untuk mendapatkan rekomendasi serupa",
                    label_visibility="collapsed",
                    key="title_selector",
                )
            else:
                selected_display = None
                ui_alert("warning", f"Tidak ada konten dengan tipe: <b>{filter_type_for_selector}</b>")

            st.markdown("</div>", unsafe_allow_html=True)

            # Pengaturan Rekomendasi
            st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1rem 0; font-weight: 800;">⚙️ PENGATURAN REKOMENDASI</h4>', unsafe_allow_html=True)
            col_settings1, col_settings2, col_settings3 = st.columns(3)

            with col_settings1:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<p style="color: #CCCCCC !important; margin-bottom: 0.6rem; font-weight: 700; font-size: 0.9rem;">JUMLAH REKOMENDASI</p>', unsafe_allow_html=True)
                top_n = st.slider("Jumlah:", 5, 20, 10, 1, label_visibility="collapsed", key="top_n_slider")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_settings2:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<p style="color: #CCCCCC !important; margin-bottom: 0.6rem; font-weight: 700; font-size: 0.9rem;">FILTER TIPE</p>', unsafe_allow_html=True)
                same_type = st.checkbox("Hanya tipe yang sama", value=True, help="Movie↔Movie atau TV Show↔TV Show", key="same_type_check")
                st.markdown("</div>", unsafe_allow_html=True)

            # Filter Tahun Rilis - DIPERBAIKI dengan label
            st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1rem 0; font-weight: 800;">📅 FILTER TAHUN RILIS</h4>', unsafe_allow_html=True)
            year_range = st.slider(
                "Pilih rentang tahun:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                format="%d",
                label_visibility="collapsed",
                key="year_range_title",
            )
            year_min, year_max = year_range

            # Tombol Dapatkan Rekomendasi
            st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
            if selected_display and st.button("🚀 DAPATKAN REKOMENDASI", type="primary", use_container_width=True, key="get_recs_btn"):
                matches = df[df["display_title"] == selected_display]
                if len(matches) == 0:
                    ui_alert("error", "Judul tidak ditemukan!")
                else:
                    idx = matches.index[0]

                    st.divider()
                    
                    # Tampilkan Konten yang Dipilih
                    st.markdown('<h3 style="color: #E50914 !important; margin: 1.5rem 0 1.2rem 0; font-weight: 800;">🎬 KONTEN YANG DIPILIH</h3>', unsafe_allow_html=True)
                    selected_item = df.loc[idx]

                    col_selected1, col_selected2 = st.columns([2, 1])
                    with col_selected1:
                        st.markdown(
                            f"""
                            <div class="glass-panel">
                                <h4 style="color: #FFFFFF !important; margin-bottom: 1rem; font-weight: 800; font-size: 1.3rem; word-break: break-word;">{_safe_str(selected_item['title'])}</h4>
                                <div style="display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap;">
                                    <span class="badge badge-movie">{_safe_str(selected_item['type'])}</span>
                                    <span class="badge badge-year">{int(selected_item['release_year'])}</span>
                                    <span class="badge badge-rating">{_safe_str(selected_item['rating'])}</span>
                                    <span class="badge">⏱️ {_safe_str(selected_item['duration'])}</span>
                                </div>
                                <div style="color: #CCCCCC !important; line-height: 1.5; padding: 1.2rem; background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%); border-radius: 10px; border-left: 4px solid #E50914; font-weight: 600; box-shadow: inset 0 1px 6px rgba(0,0,0,0.5); word-wrap: break-word;">
                                    {_safe_str(selected_item['description']) or "Tidak ada deskripsi tersedia"}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_selected2:
                        st.markdown(
                            f"""
                            <div class="stats-card">
                                <div style="margin-bottom: 1.2rem;">
                                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 700; margin-bottom: 0.5rem;">🎭 GENRE</div>
                                    <div style="font-weight: 800; color: #FFFFFF !important; font-size: 0.9rem; padding: 0.6rem; background: rgba(229,9,20,0.1); border-radius: 6px; border-left: 3px solid #E50914; word-wrap: break-word;">{_safe_str(selected_item['listed_in']) or "Tidak tersedia"}</div>
                                </div>
                                <div style="margin-bottom: 1.2rem;">
                                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 700; margin-bottom: 0.5rem;">🌍 NEGARA</div>
                                    <div style="font-weight: 800; color: #FFFFFF !important; font-size: 0.9rem; padding: 0.6rem; background: rgba(229,9,20,0.1); border-radius: 6px; border-left: 3px solid #E50914; word-wrap: break-word;">{_safe_str(selected_item['country']) or "Tidak tersedia"}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.85rem; color: #CCCCCC !important; font-weight: 700; margin-bottom: 0.5rem;">🎬 SUTRADARA</div>
                                    <div style="font-weight: 800; color: #FFFFFF !important; font-size: 0.9rem; padding: 0.6rem; background: rgba(229,9,20,0.1); border-radius: 6px; border-left: 3px solid #E50914; word-wrap: break-word;">{_safe_str(selected_item['director']) or "Tidak tersedia"}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Cari Rekomendasi
                    with st.spinner("🔎 Mencari rekomendasi terbaik..."):
                        recs = recommend_by_index(
                            idx=idx,
                            df=df,
                            tfidf_matrix=tfidf_matrix,
                            top_n=top_n,
                            same_type=same_type,
                            year_min=year_min,
                            year_max=year_max,
                        )

                    st.divider()

                    # Tampilkan Hasil Rekomendasi
                    if recs.empty:
                        ui_alert(
                            "warning",
                            """
                            <div class="glass-panel" style="padding: 1.2rem;">
                                <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                                    <div style="font-size: 2.5rem; color: #E50914; flex-shrink: 0;">🤖</div>
                                    <div style="flex: 1; min-width: 200px;">
                                        <h4 style="margin: 0; color: #E50914 !important; font-weight: 800; font-size: 1.2rem;">TIDAK MENEMUKAN REKOMENDASI</h4>
                                        <div style="margin-top: 0.5rem; color: #CCCCCC !important; font-weight: 600; font-size: 0.9rem;">Coba sesuaikan filter untuk hasil lebih baik.</div>
                                    </div>
                                </div>
                            </div>
                            """,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(229, 9, 20, 0.15) 0%, rgba(184, 29, 36, 0.15) 100%);
                                        border-radius: 12px; margin: 1.5rem 0; border: 2px solid #E50914;
                                        box-shadow: 0 10px 25px rgba(229, 9, 20, 0.3);
                                        position: relative;
                                        overflow: hidden;
                                        word-wrap: break-word;">
                                <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #E50914, #FF0000);"></div>
                                <div style="font-size: 3rem; color: #E50914; margin-bottom: 0.8rem; filter: drop-shadow(0 0 15px rgba(229,9,20,0.6));">🎯</div>
                                <h3 style="color: #FFFFFF !important; margin: 0; font-weight: 900; font-size: 1.5rem; text-shadow: 2px 2px 0 #000000; line-height: 1.2;">{len(recs)} REKOMENDASI TERBAIK</h3>
                                <p style="color: #CCCCCC !important; margin: 0.5rem 0 0 0; font-weight: 600; font-size: 0.95rem; line-height: 1.3;">Berdasarkan kemiripan konten (TF-IDF + Cosine Similarity)</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                        # Tampilkan setiap rekomendasi
                        for i, (_, r) in enumerate(recs.iterrows(), 1):
                            display_recommendation_card(r, i)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Dashboard Dataset
            st.markdown('<h3 style="color: #E50914 !important; margin-bottom: 1.2rem; font-weight: 800;">📊 DASHBOARD DATASET</h3>', unsafe_allow_html=True)
            display_metric_card("Total Konten", f"{stats['total']:,}", "Movies & TV Shows", "📈")
            display_metric_card("Movies", f"{stats['movies']:,}", f"{(stats['movies']/stats['total']*100):.1f}%" if stats["total"] else "0%", "🎥")
            display_metric_card("TV Shows", f"{stats['tv_shows']:,}", f"{(stats['tv_shows']/stats['total']*100):.1f}%" if stats["total"] else "0%", "📺")
            display_metric_card("Tahun Terbaru", str(stats["max_year"]), "Konten terupdate", "🚀")

            st.divider()
            
            # Genre Populer
            st.markdown('<h4 style="color: #E50914 !important; margin: 1rem 0 0.8rem 0; font-weight: 800;">🎭 GENRE POPULER</h4>', unsafe_allow_html=True)
            top_genres = split_and_count(df["listed_in"], sep=",", top_k=5)
            if len(top_genres) > 0:
                for genre, count in top_genres.items():
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.8rem 1rem;
                                    background: linear-gradient(135deg, rgba(229, 9, 20, 0.15) 0%, rgba(184, 29, 36, 0.15) 100%);
                                    border-radius: 8px; margin: 0.6rem 0; border: 1px solid #2D2D2D;
                                    transition: all 0.2s ease;
                                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
                                    word-wrap: break-word;">
                            <span style="color: #FFFFFF !important; font-weight: 700; font-size: 0.9rem; word-break: break-word;">{genre[:18]}{'...' if len(genre)>18 else ''}</span>
                            <span class="badge" style="font-size: 0.8rem; padding: 0.3rem 0.6rem; min-width: 40px; justify-content: center; height: 30px; flex-shrink: 0;">{count}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Tab 2 - Berdasarkan Kata Kunci (DIPERBAIKI)
    with tabs[1]:
        st.markdown('<h3 style="color: #E50914 !important; margin-bottom: 1.2rem; font-weight: 800;">🔍 PENCARIAN DENGAN KATA KUNCI</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #CCCCCC !important; font-size: 0.95rem; font-weight: 600; margin-bottom: 1.5rem; line-height: 1.4;">Masukkan kata kunci untuk menemukan konten yang sesuai.</p>', unsafe_allow_html=True)

        # Input Pencarian dengan label yang jelas
        col_search1, col_search2 = st.columns([3, 1])
        with col_search1:
            st.markdown('<p style="color: #FFFFFF !important; font-weight: 700; margin-bottom: 0.5rem; font-size: 0.95rem;">🔍 MASUKKAN KATA KUNCI</p>', unsafe_allow_html=True)
            query = st.text_input(
                "Masukkan kata kunci pencarian:",
                placeholder="Contoh: action adventure, romantic comedy, sci-fi, crime drama",
                help="Gunakan bahasa Inggris untuk hasil terbaik",
                label_visibility="collapsed",
                key="search_query",
            )
        with col_search2:
            st.markdown('<div style="height: 28px;"></div>')
            search_btn = st.button("🔍 CARI", type="primary", use_container_width=True, key="search_btn")

        # Filter Pencarian
        st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1rem 0; font-weight: 800;">⚙️ FILTER PENCARIAN</h4>', unsafe_allow_html=True)
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<p style="color: #CCCCCC !important; margin-bottom: 0.6rem; font-weight: 700; font-size: 0.9rem;">FILTER TIPE</p>', unsafe_allow_html=True)
            type_filter = st.selectbox("Tipe:", options=type_options, index=0, label_visibility="collapsed", key="type_filter_search")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_filter2:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<p style="color: #CCCCCC !important; margin-bottom: 0.6rem; font-weight: 700; font-size: 0.9rem;">JUMLAH HASIL</p>', unsafe_allow_html=True)
            top_n_q = st.slider("Hasil:", 5, 20, 10, label_visibility="collapsed", key="top_n_search")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_filter3:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<p style="color: #CCCCCC !important; margin-bottom: 0.6rem; font-weight: 700; font-size: 0.9rem;">TAHUN RILIS</p>', unsafe_allow_html=True)
            year_range_q = st.slider("Tahun:", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="year_range_search", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

        year_min_q, year_max_q = year_range_q

        # Proses Pencarian
        st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
        if search_btn:
            if not query:
                ui_alert("warning", "Masukkan kata kunci dulu ya 🙂")
            else:
                with st.spinner("🔎 Menganalisis kata kunci..."):
                    recs_q = recommend_by_query(
                        query=query,
                        df=df,
                        vectorizer=vectorizer,
                        tfidf_matrix=tfidf_matrix,
                        top_n=top_n_q,
                        type_filter=type_filter if type_filter != "All" else "All",
                        year_min=year_min_q,
                        year_max=year_max_q,
                    )

                if recs_q.empty:
                    ui_alert(
                        "error",
                        f"""
                        <div class="glass-panel" style="padding: 1.2rem;">
                            <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                                <div style="font-size: 2rem; color: #E50914; flex-shrink: 0;">❌</div>
                                <div style="flex: 1; min-width: 200px;">
                                    <h4 style="margin: 0; color: #E50914 !important; font-weight: 800; font-size: 1.1rem;">TIDAK MENEMUKAN HASIL</h4>
                                    <div style="margin-top: 0.4rem; color: #CCCCCC !important; font-weight: 600; font-size: 0.9rem;">Untuk pencarian: <b>{_safe_str(query)}</b></div>
                                    <div style="margin-top: 0.8rem; color: #CCCCCC !important; font-weight: 600; font-size: 0.85rem;">
                                        Tips: pakai keyword bahasa Inggris, kurangi filter, dan gunakan kata yang lebih umum.
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                    )
                else:
                    ui_alert(
                        "success",
                        f"""
                        <div style="text-align: center; padding: 1.2rem;">
                            <div style="font-size: 2rem; color: #00B894; filter: drop-shadow(0 0 10px rgba(0,184,148,0.5));">🎉</div>
                            <div style="font-weight: 800; color: #00B894 !important; font-size: 1.3rem; text-shadow: 1px 1px 0 #000000; line-height: 1.1;">{len(recs_q)} HASIL DITEMUKAN!</div>
                            <div style="color: #CCCCCC !important; margin-top: 0.5rem; font-weight: 600; font-size: 0.9rem; line-height: 1.2;">Untuk pencarian: <b>{_safe_str(query)}</b></div>
                        </div>
                        """,
                    )
                    # Tampilkan hasil
                    for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                        display_recommendation_card(r, i)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3 - Konten Populer
    with tabs[2]:
        st.markdown('<h3 style="color: #E50914 !important; margin-bottom: 1.2rem; font-weight: 800;">⭐ KONTEN POPULER</h3>', unsafe_allow_html=True)

        # Statistik Konten Populer
        col_pop1, col_pop2, col_pop3, col_pop4 = st.columns(4)
        with col_pop1:
            most_common_rating = df["rating"].value_counts().index[0] if len(df["rating"].value_counts()) > 0 else "N/A"
            display_metric_card("Rating Terpopuler", most_common_rating, "Paling sering muncul", "⭐")
        with col_pop2:
            display_metric_card("Tahun Rata-rata", str(stats.get("avg_year", 2000)), "Rata-rata rilis", "📅")
        with col_pop3:
            top_countries = split_and_count(df["country"], sep=",", top_k=1)
            most_common_country = top_countries.index[0] if len(top_countries) > 0 else "N/A"
            display_metric_card("Negara Teratas", most_common_country[:10], "Produksi terbanyak", "🌍")
        with col_pop4:
            top_genres = split_and_count(df["listed_in"], sep=",", top_k=1)
            most_common_genre = top_genres.index[0] if len(top_genres) > 0 else "N/A"
            display_metric_card("Genre Terpopuler", most_common_genre[:12], "Paling banyak", "🎭")

        st.divider()
        
        # Rekomendasi Acak
        st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1.2rem 0; font-weight: 800;">🎲 REKOMENDASI ACAK</h4>', unsafe_allow_html=True)

        if len(df) > 0:
            sample_size = min(6, len(df))
            random_sample = df.sample(sample_size)
            cols = st.columns(2)

            for i, (_, item) in enumerate(random_sample.iterrows()):
                with cols[i % 2]:
                    st.markdown(
                        f"""
                        <div class="glass-panel" style="padding: 1.2rem;">
                            <h5 style="margin: 0 0 1rem 0; color: #FFFFFF !important; font-size: 1.1rem; font-weight: 800; word-break: break-word;">
                                {_safe_str(item['title'])[:30]}{'...' if len(_safe_str(item['title']))>30 else ''}
                            </h5>
                            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem;">
                                <span class="badge badge-movie" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">{_safe_str(item['type'])}</span>
                                <span class="badge badge-year" style="font-size: 0.8rem; padding: 0.4rem 0.8rem;">{int(item['release_year'])}</span>
                            </div>
                            <div style="color: #CCCCCC !important; font-size: 0.9rem; line-height: 1.4; margin-bottom: 1.2rem; font-weight: 600; word-wrap: break-word;">
                                {_safe_str(item['description'])[:90]}{'...' if len(_safe_str(item['description']))>90 else ''}
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; padding-top: 0.8rem; border-top: 1px solid #2D2D2D; flex-wrap: wrap; gap: 0.4rem;">
                                <span style="color: #CCCCCC !important; font-weight: 700; display: flex; align-items: center; gap: 0.3rem; white-space: nowrap;">⭐ {_safe_str(item['rating']) or 'N/A'}</span>
                                <span style="color: #CCCCCC !important; font-weight: 700; display: flex; align-items: center; gap: 0.3rem; white-space: nowrap;">⏱️ {_safe_str(item['duration']) or 'N/A'}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# -----------------------------
# Page: Data Analysis
# -----------------------------
elif page == "📊 DASHBOARD ANALITIK":
    st.markdown('<h2 style="color: #E50914 !important; margin-bottom: 1rem; font-weight: 900;">📊 DASHBOARD ANALITIK NETFLIX</h2>', unsafe_allow_html=True)
    st.markdown('<h4 style="color: #E50914 !important; margin-bottom: 1.5rem; font-weight: 800;">📈 METRIK UTAMA</h4>', unsafe_allow_html=True)

    # Metrik Utama
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        display_metric_card("Total Konten", f"{stats['total']:,}", "Movies & TV Shows", "📊")
    with col_m2:
        display_metric_card("Movies", f"{stats['movies']:,}", f"{(stats['movies']/stats['total']*100):.1f}%" if stats["total"] else "0%", "🎥")
    with col_m3:
        display_metric_card("TV Shows", f"{stats['tv_shows']:,}", f"{(stats['tv_shows']/stats['total']*100):.1f}%" if stats["total"] else "0%", "📺")
    with col_m4:
        display_metric_card("Rata-rata Tahun", str(stats["avg_year"]), f"{stats['min_year']} - {stats['max_year']}", "📅")

    st.divider()
    
    # Visualisasi
    st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1.2rem 0; font-weight: 800;">📈 VISUALISASI INTERAKTIF</h4>', unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<h5 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 700;">🎭 DISTRIBUSI TIPE KONTEN</h5>', unsafe_allow_html=True)
        type_counts = df["type"].value_counts()
        if len(type_counts) > 0:
            st.bar_chart(type_counts)

        st.markdown('<h5 style="color: #E50914 !important; margin: 1.5rem 0 0.8rem 0; font-weight: 700;">🌍 TOP 10 NEGARA</h5>', unsafe_allow_html=True)
        top_countries = split_and_count(df["country"], sep=",", top_k=10)
        if len(top_countries) > 0:
            st.bar_chart(top_countries)

    with col_chart2:
        st.markdown('<h5 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 700;">🎬 TOP 10 GENRE</h5>', unsafe_allow_html=True)
        top_genres = split_and_count(df["listed_in"], sep=",", top_k=10)
        if len(top_genres) > 0:
            st.bar_chart(top_genres)

        st.markdown('<h5 style="color: #E50914 !important; margin: 1.5rem 0 0.8rem 0; font-weight: 700;">📅 TREN TAHUN RILIS</h5>', unsafe_allow_html=True)
        year_counts = df["release_year"][df["release_year"] > 0].value_counts().sort_index()
        if len(year_counts) > 0:
            st.line_chart(year_counts)

    st.divider()
    
    # Eksplorasi Data
    st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1.2rem 0; font-weight: 800;">🔍 PRATINJAU & EKSPLORASI DATA</h4>', unsafe_allow_html=True)

    # Search input dengan label yang jelas
    st.markdown('<p style="color: #FFFFFF !important; font-weight: 700; margin-bottom: 0.5rem; font-size: 0.95rem;">🔍 CARI DALAM DATASET</p>', unsafe_allow_html=True)
    search_term = st.text_input("Masukkan kata kunci pencarian:", placeholder="Cari berdasarkan judul, deskripsi, sutradara...", key="data_search", label_visibility="collapsed")

    if search_term:
        search_cols = ["title", "description", "director", "cast", "listed_in"]
        mask = pd.Series(False, index=df.index)
        for col in search_cols:
            mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
        preview_df = df[mask]
        if len(preview_df) > 0:
            ui_alert("info", f"Ditemukan <b>{len(preview_df)}</b> hasil untuk pencarian: <b>{_safe_str(search_term)}</b>")
        else:
            ui_alert("warning", f"Tidak ditemukan hasil untuk pencarian: <b>{_safe_str(search_term)}</b>")
            preview_df = df
    else:
        preview_df = df

    # Slider dengan label yang jelas
    st.markdown('<p style="color: #FFFFFF !important; font-weight: 700; margin: 1rem 0 0.5rem 0; font-size: 0.95rem;">📊 PILIH JUMLAH SAMPEL</p>', unsafe_allow_html=True)
    sample_size = st.slider("", 5, 100, 20, key="sample_size", label_visibility="collapsed")
    
    display_cols = ["title", "type", "release_year", "rating", "duration", "listed_in"]
    available_cols = [c for c in display_cols if c in preview_df.columns]

    if len(preview_df) > 0:
        st.dataframe(
            preview_df[available_cols].head(sample_size),
            use_container_width=True,
            hide_index=True,
            column_config={
                "title": "🎬 Judul",
                "type": "🎭 Tipe",
                "release_year": "📅 Tahun",
                "rating": "⭐ Rating",
                "duration": "⏱️ Durasi",
                "listed_in": "🎨 Genre",
            },
        )
    else:
        ui_alert("warning", "Tidak ada data untuk ditampilkan")

    st.divider()
    
    # Kualitas Data
    st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1.2rem 0; font-weight: 800;">🔍 PEMERIKSAAN KUALITAS DATA</h4>', unsafe_allow_html=True)

    col_quality1, col_quality2, col_quality3 = st.columns(3)
    with col_quality1:
        missing_director = (df["director"].fillna("").astype(str).str.strip() == "").sum()
        display_metric_card("Director Kosong", f"{missing_director:,}", f"{(missing_director/len(df)*100):.1f}%", "🎬")
    with col_quality2:
        missing_cast = (df["cast"].fillna("").astype(str).str.strip() == "").sum()
        display_metric_card("Cast Kosong", f"{missing_cast:,}", f"{(missing_cast/len(df)*100):.1f}%", "👥")
    with col_quality3:
        missing_country = (df["country"].fillna("").astype(str).str.strip() == "").sum()
        display_metric_card("Negara Kosong", f"{missing_country:,}", f"{(missing_country/len(df)*100):.1f}%", "🌍")

    st.divider()
    
    # Ekspor Data
    st.markdown('<h4 style="color: #E50914 !important; margin: 1.5rem 0 1.2rem 0; font-weight: 800;">💾 EKSPOR DATA</h4>', unsafe_allow_html=True)

    col_export1, col_export2 = st.columns([2, 1])
    with col_export1:
        st.markdown(
            """
            <div class="glass-panel">
                <h5 style="color: #E50914 !important; margin-bottom: 0.8rem; font-weight: 800; font-size: 1.1rem;">📥 DOWNLOAD DATASET</h5>
                <p style="color: #CCCCCC !important; margin-bottom: 0; font-weight: 600; font-size: 0.9rem;">Unduh dataset yang telah diproses dalam format CSV.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_export2:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 DOWNLOAD CSV",
            data=csv,
            file_name="netflix_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

# -----------------------------
# Page: About System
# -----------------------------
elif page == "🤖 TENTANG SISTEM":
    col_about1, col_about2 = st.columns([2, 1])

    with col_about1:
        st.markdown('<h2 style="color: #E50914 !important; margin-bottom: 1.2rem; font-weight: 900;">🤖 TENTANG SISTEM REKOMENDASI</h2>', unsafe_allow_html=True)
        st.markdown(
            """
        <div class="glass-panel">
            <h4 style="color: #E50914 !important; margin-bottom: 1rem; font-weight: 800; font-size: 1.2rem;">📌 RINGKASAN</h4>
            <p style="color: #CCCCCC !important; line-height: 1.5; margin-bottom: 0; font-weight: 600; font-size: 0.95rem; word-wrap: break-word;">
                Sistem ini menggunakan <b style="color: #E50914 !important; font-weight: 800;">Content-Based Filtering</b> dengan <b style="color: #E50914 !important; font-weight: 800;">TF-IDF</b> dan <b style="color: #E50914 !important; font-weight: 800;">Cosine Similarity</b>
                untuk memberikan rekomendasi berdasarkan kemiripan metadata (genre, deskripsi, cast, director, dsb).
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_about2:
        st.markdown('<h4 style="color: #E50914 !important; margin-bottom: 1.2rem; font-weight: 800;">📈 INFO TEKNIS</h4>', unsafe_allow_html=True)
        display_metric_card("Ukuran Dataset", f"{len(df):,}", "item konten", "📁")
        display_metric_card("Dimensi Vektor", f"{tfidf_matrix.shape[1]:,}", "fitur TF-IDF", "🔢")
        display_metric_card("Waktu Respon", "< 500ms", "rata-rata", "⚡")

    st.divider()
    
    # Footer About
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem; border-radius: 12px; border: 2px solid #E50914;
                background: linear-gradient(135deg, rgba(229, 9, 20, 0.15) 0%, rgba(184, 29, 36, 0.15) 50%, rgba(139, 0, 0, 0.15) 100%);
                box-shadow: 0 10px 25px rgba(229, 9, 20, 0.3);
                margin: 2rem 0;
                position: relative;
                overflow: hidden;
                word-wrap: break-word;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #E50914, #FF0000);"></div>
        <div style="font-size: 3rem; color: #E50914; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(229,9,20,0.6));">🎬</div>
        <h4 style="color: #E50914 !important; margin: 1rem 0; font-weight: 900; font-size: 1.5rem; text-shadow: 2px 2px 0 #000000; line-height: 1.2;">TERIMA KASIH!</h4>
        <p style="color: #CCCCCC !important; margin: 0; font-size: 1rem; font-weight: 600; line-height: 1.2;">Built with Streamlit + Scikit-learn</p>
        <div style="display: flex; justify-content: center; gap: 0.8rem; margin-top: 1.5rem; flex-wrap: wrap;">
            <span class="badge" style="background: linear-gradient(135deg, #E50914 0%, #8B0000 100%) !important; font-size: 0.9rem;">NETFLIX STYLE</span>
            <span class="badge" style="background: linear-gradient(135deg, #2D2D2D 0%, #4A4A4A 100%) !important; font-size: 0.9rem;">MACHINE LEARNING</span>
            <span class="badge" style="background: linear-gradient(135deg, #00B894 0%, #00A884 100%) !important; font-size: 0.9rem;">CONTENT-BASED</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div style="text-align: center; margin-top: 3rem; padding: 1.5rem 1rem; 
                border-top: 2px solid #E50914;
                background: linear-gradient(to bottom, transparent, rgba(229, 9, 20, 0.1));
                position: relative;
                overflow: hidden;
                word-wrap: break-word;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #E50914, #FF0000, #E50914);"></div>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <span style="color: #FFFFFF !important; font-weight: 700; font-size: 0.9rem; line-height: 1.1;">🎬 Netflix Recommender System</span>
            <span style="color: #E50914 !important; font-weight: 800; font-size: 0.9rem;">•</span>
            <span style="color: #FFFFFF !important; font-weight: 700; font-size: 0.9rem; line-height: 1.1;">Powered by Content-Based Filtering</span>
            <span style="color: #E50914 !important; font-weight: 800; font-size: 0.9rem;">•</span>
            <span style="color: #E50914 !important; font-weight: 800; font-size: 0.9rem; text-shadow: 1px 1px 0 #000000; line-height: 1.1;">NETFLIX</span>
        </div>
        <p style="color: #888888 !important; margin: 0; font-size: 0.8rem; font-weight: 600; line-height: 1.2;">
            © 2024 Netflix Recommender System. All rights reserved.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
