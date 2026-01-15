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
# Custom CSS - Netflix Style with Improved Contrast
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Netflix+Sans:wght@400;500;700;800;900&display=swap');

* {
    font-family: 'Netflix Sans', 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #000000 100%) !important;
    color: #FFFFFF !important;
    min-height: 100vh;
}

/* TYPOGRAPHY - SUPER CLEAR & BOLD */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Netflix Sans', sans-serif !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
    line-height: 1.2 !important;
    margin-bottom: 1.5rem !important;
}

h1 { 
    font-size: 3.5rem !important; 
    color: #FFD700 !important;
    text-shadow: 0 4px 8px rgba(0,0,0,0.5);
}
h2 { 
    font-size: 2.5rem !important; 
    color: #FFD700 !important;
}
h3 { 
    font-size: 2rem !important; 
    color: #FFFFFF !important;
}
h4 { 
    font-size: 1.5rem !important; 
    color: #FFFFFF !important;
}

p, span, div, label {
    color: #F8F8F8 !important;
    font-weight: 600 !important;
    line-height: 1.6 !important;
}

strong, b {
    color: #FFD700 !important;
    font-weight: 900 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* MAIN HEADER */
.main-header {
    background: linear-gradient(135deg, 
                rgba(229, 9, 20, 0.15) 0%, 
                rgba(255, 215, 0, 0.1) 100%),
                linear-gradient(rgba(10, 10, 10, 0.95), rgba(0, 0, 0, 0.98));
    background-size: cover;
    background-position: center;
    padding: 4rem 3rem;
    border-radius: 20px;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
    border: 4px solid #E50914;
    box-shadow: 0 25px 50px rgba(229, 9, 20, 0.2);
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(229, 9, 20, 0.1), transparent 40%, rgba(255, 215, 0, 0.1));
    z-index: 1;
}

.main-header-content {
    position: relative;
    z-index: 2;
}

/* METRIC CARDS - EXTRA CLEAR */
.metric-card {
    background: linear-gradient(145deg, #1A1A1A, #121212);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    border: 3px solid #2A2A2A;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
    margin-bottom: 1.5rem;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
    background: linear-gradient(90deg, #E50914, #FFD700);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: #FFD700;
    box-shadow: 0 30px 60px rgba(255, 215, 0, 0.3);
}

.metric-card h3 {
    color: #FFD700 !important;
    font-size: 3.5rem !important;
    margin-bottom: 0.8rem;
    font-weight: 1000 !important;
    text-shadow: 4px 4px 0 rgba(0,0,0,0.5);
    letter-spacing: -1.5px;
}

.metric-card-title {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    font-size: 1.4rem !important;
    margin-bottom: 0.5rem;
    display: block;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

.metric-card-subtitle {
    color: #CCCCCC !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    opacity: 0.9;
}

.metric-card-icon {
    font-size: 3.5rem;
    margin-bottom: 1.5rem;
    color: #E50914;
    display: block;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
}

/* RECOMMENDATION CARDS */
.recommendation-card {
    background: linear-gradient(145deg, #1A1A1A, #141414);
    border-radius: 20px;
    padding: 2.5rem;
    margin: 2rem 0;
    border-left: 8px solid #FFD700;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
    border: 2px solid #2A2A2A;
}

.recommendation-card:hover {
    transform: translateY(-10px) scale(1.02);
    border-left: 8px solid #E50914;
    box-shadow: 0 35px 70px rgba(255, 215, 0, 0.3);
    background: linear-gradient(145deg, #222222, #1A1A1A);
}

.recommendation-card::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 150px;
    height: 150px;
    background: linear-gradient(135deg, transparent 50%, rgba(255, 215, 0, 0.08) 50%);
    border-radius: 0 0 0 150px;
}

.similarity-score {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%);
    color: #000000 !important;
    font-weight: 1000 !important;
    font-size: 1.4rem !important;
    padding: 1rem 2rem;
    border-radius: 50px;
    display: inline-flex;
    align-items: center;
    border: 4px solid #FFFFFF;
    box-shadow: 0 10px 30px rgba(255, 215, 0, 0.5);
    min-width: 120px;
    justify-content: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* BADGES - YELLOW ACCENT */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.8rem 1.5rem;
    background: linear-gradient(135deg, #E50914 0%, #B81D24 100%);
    color: #FFFFFF !important;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 800;
    margin-right: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
    transition: all 0.3s ease;
    border: 2px solid rgba(255, 255, 255, 0.2);
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.badge:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(229, 9, 20, 0.6);
    background: linear-gradient(135deg, #FF0000 0%, #E50914 100%);
}

.badge-movie {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
    color: #000000 !important;
    font-weight: 900 !important;
}

.badge-year {
    background: linear-gradient(135deg, #2A2A2A 0%, #4A4A4A 100%) !important;
    color: #FFFFFF !important;
}

.badge-rating {
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
    color: #000000 !important;
    font-weight: 900 !important;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
    color: #000000 !important;
    border: none !important;
    padding: 1.2rem 3rem !important;
    border-radius: 15px !important;
    font-weight: 1000 !important;
    font-size: 1.2rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 20px 40px rgba(229, 9, 20, 0.4) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.stButton > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 25px 50px rgba(255, 215, 0, 0.5) !important;
    background: linear-gradient(135deg, #FF0000 0%, #FFD700 100%) !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: #141414 !important;
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 3rem;
    border: 3px solid #2A2A2A;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    padding: 1.2rem 2.5rem !important;
    font-weight: 800 !important;
    color: #AAAAAA !important;
    border: 3px solid transparent !important;
    transition: all 0.3s ease !important;
    font-size: 1.1rem !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
    color: #000000 !important;
    font-weight: 1000 !important;
    box-shadow: 0 15px 30px rgba(255, 215, 0, 0.4) !important;
    border-color: #FFD700 !important;
}

/* INPUTS - IMPROVED CONTRAST */
.stTextInput > div > div > input {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    border-radius: 15px !important;
    border: 3px solid #333333 !important;
    padding: 1.2rem 1.8rem !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
}

.stTextInput > div > div > input:focus {
    border-color: #FFD700 !important;
    box-shadow: 0 0 0 4px rgba(255, 215, 0, 0.2), 0 15px 35px rgba(0, 0, 0, 0.6) !important;
    background: #1A1A1A !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: #AAAAAA !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

/* SELECTBOX - IMPROVED */
.stSelectbox > div > div {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    border: 3px solid #333333 !important;
    border-radius: 15px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding: 1rem 1.5rem !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
}

.stSelectbox > div > div:hover {
    border-color: #FFD700 !important;
    background: #1A1A1A !important;
}

/* SLIDER - IMPROVED */
.stSlider > div > div {
    background: #1A1A1A !important;
    border-radius: 15px !important;
    padding: 1.5rem !important;
    border: 2px solid #333333 !important;
}

.stSlider > div > div > div {
    color: #FFD700 !important;
    font-weight: 800 !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #1A1A1A, #141414) !important;
    color: #FFD700 !important;
    font-weight: 900 !important;
    border-radius: 15px !important;
    border: 3px solid #2A2A2A !important;
    font-size: 1.2rem !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* GLASS PANEL */
.glass-panel {
    background: rgba(20, 20, 20, 0.95) !important;
    backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 3px solid #2A2A2A;
    padding: 3rem;
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6);
    margin-bottom: 2rem;
}

.stats-card {
    background: linear-gradient(145deg, #1A1A1A, #141414);
    border-radius: 20px;
    padding: 2.5rem;
    border: 3px solid #2A2A2A;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6);
}

/* TAGS */
.tag {
    display: inline-block;
    padding: 0.6rem 1.4rem;
    margin: 0.5rem;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFD700 0%, #E50914 100%);
    color: #000000 !important;
    border: none;
    box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 14px;
    background: #0A0A0A;
}

::-webkit-scrollbar-track {
    background: #1A1A1A;
    border-radius: 8px;
    border: 2px solid #2A2A2A;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%);
    border-radius: 8px;
    border: 2px solid #1A1A1A;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #FF0000 0%, #FFD700 100%);
}

/* ALERTS */
.stAlert {
    border-radius: 15px !important;
    border: 3px solid !important;
    font-weight: 700 !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5) !important;
}

/* DATA FRAME */
.stDataFrame {
    border-radius: 15px !important;
    border: 3px solid #2A2A2A !important;
    background: #141414 !important;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5) !important;
}

.dataframe th {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
    color: #000000 !important;
    font-weight: 1000 !important;
    font-size: 1.1rem !important;
    padding: 1.5rem !important;
}

.dataframe td {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    background: #1A1A1A !important;
    padding: 1.2rem !important;
}

/* DIVIDER */
hr {
    border-color: #FFD700 !important;
    opacity: 0.6;
    margin: 4rem 0 !important;
    border-width: 3px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0A0A0A 0%, #000000 100%) !important;
    border-right: 4px solid #E50914;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* CHECKBOX */
.stCheckbox > label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

.stCheckbox > label > div:first-child {
    background: #1A1A1A !important;
    border: 3px solid #333333 !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

.stCheckbox > label > div:first-child:hover {
    border-color: #FFD700 !important;
}

/* ANIMATIONS */
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-15px); }
}

.floating {
    animation: float 3s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 25px rgba(255, 215, 0, 0.4); }
    50% { box-shadow: 0 0 50px rgba(255, 215, 0, 0.7); }
}

.glow {
    animation: glow 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .main-header { padding: 2.5rem 1.5rem; }
    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2.2rem !important; }
    .metric-card h3 { font-size: 3rem !important; }
    .recommendation-card { padding: 2rem; }
    .badge { padding: 0.7rem 1.3rem; font-size: 0.95rem; }
    .glass-panel { padding: 2rem; }
}

/* SPECIAL: CLEAR INPUT LABELS */
.stTextInput > label,
.stSelectbox > label,
.stSlider > label {
    color: #FFD700 !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    margin-bottom: 1rem !important;
    display: block !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* SEARCH SECTION */
.search-section {
    background: linear-gradient(145deg, #141414, #0F0F0F);
    border-radius: 25px;
    padding: 3.5rem;
    margin: 3rem 0;
    border: 3px solid #2A2A2A;
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.7);
}

.search-title {
    color: #FFD700 !important;
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    margin-bottom: 2.5rem !important;
    text-align: center;
    text-shadow: 0 4px 8px rgba(0,0,0,0.5);
}

.search-input-container {
    background: #0F0F0F;
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2.5rem;
    border: 3px solid #333333;
    box-shadow: inset 0 5px 15px rgba(0,0,0,0.3);
}

.search-label {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.4rem !important;
    margin-bottom: 1.2rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 1rem !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* TEXT SHADOWS FOR BETTER READABILITY */
.text-shadow {
    text-shadow: 0 2px 4px rgba(0,0,0,0.7);
}

.text-shadow-heavy {
    text-shadow: 0 4px 8px rgba(0,0,0,0.9);
}

/* BORDER GLOW */
.border-glow {
    border: 3px solid transparent;
    background: linear-gradient(#1A1A1A, #1A1A1A) padding-box,
                linear-gradient(135deg, #E50914, #FFD700) border-box;
    box-shadow: 0 20px 40px rgba(255, 215, 0, 0.2);
}

/* CONTRAST BOX */
.contrast-box {
    background: rgba(15, 15, 15, 0.95);
    border: 3px solid #FFD700;
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 25px 50px rgba(255, 215, 0, 0.2);
    backdrop-filter: blur(10px);
}

/* CLEAR TEXT */
.clear-text {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    line-height: 1.8 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI helper
# -----------------------------
def ui_alert(kind: str, html: str):
    palette = {
        "info": ("#FFD700", "rgba(255, 215, 0, 0.15)", "ℹ️"),
        "success": ("#00B894", "rgba(0, 184, 148, 0.15)", "✅"),
        "warning": ("#FFD700", "rgba(255, 215, 0, 0.15)", "⚠️"),
        "error": ("#E50914", "rgba(229, 9, 20, 0.15)", "⛔"),
    }
    border, bg, icon = palette.get(kind, palette["info"])
    st.markdown(
        f"""
        <div class="contrast-box" style="margin: 2rem 0;">
            <div style="display: flex; gap: 1.5rem; align-items: flex-start;">
                <div style="font-size: 2.5rem; line-height: 1; color: {border}; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));">{icon}</div>
                <div class="clear-text" style="flex: 1;">{html}</div>
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
    s = re.sub(r"[^0-9a-z]+", " ", s)
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
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2rem;">
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem;">
                        <div style="background: linear-gradient(135deg, #E50914 0%, #FFD700 100%); color: #000000; padding: 0.8rem 2rem; border-radius: 50px; font-weight: 1000; font-size: 1.4rem; box-shadow: 0 8px 25px rgba(255, 215, 0, 0.5);">
                            #{rank}
                        </div>
                        <h3 style="margin: 0; color: #FFD700 !important; font-size: 2.5rem; font-weight: 900; text-shadow: 0 4px 8px rgba(0,0,0,0.5);">{title}</h3>
                    </div>
                </div>
                <div class="similarity-score">{similarity:.1%}</div>
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
            <div style="margin: 2.5rem 0 1.5rem 0;">
                <strong style="color: #FFD700 !important; font-size: 1.5rem; font-weight: 900; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">🎭 GENRE</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )
            genres = [g.strip() for g in str(genre).split(",") if g.strip()]
            tags_html = "".join([f'<span class="tag">{g}</span>' for g in genres[:5]])
            st.markdown(f'<div style="margin-bottom: 2.5rem;">{tags_html}</div>', unsafe_allow_html=True)

        if description and description != "No description available":
            with st.expander("📖 DESKRIPSI LENGKAP", expanded=False):
                st.markdown(
                    f"""
                <div class="clear-text" style="padding: 2rem; background: rgba(20, 20, 20, 0.8); border-radius: 15px; border-left: 5px solid #FFD700;">
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
                <div class="stats-card" style="margin-bottom: 1.5rem;">
                    <strong style="color: #FFD700 !important; font-size: 1.2rem; font-weight: 900; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">🎬 DIRECTOR</strong>
                    <div class="clear-text" style="margin-top: 1rem;">{director}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        with col_details2:
            if country and country != "Not specified":
                st.markdown(
                    f"""
                <div class="stats-card" style="margin-bottom: 1.5rem;">
                    <strong style="color: #FFD700 !important; font-size: 1.2rem; font-weight: 900; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">🌍 NEGARA</strong>
                    <div class="clear-text" style="margin-top: 1rem;">{country}</div>
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
        <div class="metric-card-icon floating">{icon}</div>
        <h3>{value}</h3>
        <span class="metric-card-title">{title}</span>
        <span class="metric-card-subtitle">{subtitle}</span>
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
# Header - Netflix Style
# -----------------------------
st.markdown(
    """
<div class="main-header">
    <div class="main-header-content">
        <div style="display: flex; align-items: center; gap: 2.5rem; margin-bottom: 2.5rem;">
            <div style="font-size: 7rem; color: #FFD700; filter: drop-shadow(0 8px 16px rgba(0,0,0,0.5));" class="floating">🎬</div>
            <div>
                <h1 style="color: #FFD700 !important; text-shadow: 0 6px 12px rgba(0,0,0,0.7); margin-bottom: 1rem; font-size: 4rem;">NETFLIX RECOMMENDER</h1>
                <p style="margin: 0; font-size: 2rem; font-weight: 800; color: #FFFFFF !important; line-height: 1.6; text-shadow: 0 3px 6px rgba(0,0,0,0.5);">
                    Sistem Rekomendasi Berbasis Machine Learning
                </p>
            </div>
        </div>
        <p style="color: #F8F8F8 !important; font-weight: 700; font-size: 1.5rem; line-height: 1.8; margin-bottom: 2.5rem; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">
            Temukan konten yang paling sesuai dengan preferensi Anda menggunakan <strong>Content-Based Filtering</strong>
            dengan <strong>TF-IDF Vectorization</strong> dan <strong>Cosine Similarity</strong>.
        </p>
        <div style="display: flex; gap: 1.2rem; flex-wrap: wrap;">
            <span class="badge badge-movie">🎯 REKOMENDASI PRESISI</span>
            <span class="badge badge-year">⚡ PEMROSESAN CEPAT</span>
            <span class="badge badge-rating">📌 BERBASIS KONTEN</span>
            <span class="badge" style="background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;">📊 ANALISIS DATA</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
    <div style="text-align: center; margin-bottom: 3rem; padding: 3rem 2rem;
                background: linear-gradient(135deg, rgba(229, 9, 20, 0.25) 0%, rgba(255, 215, 0, 0.2) 100%);
                border-radius: 20px; border: 4px solid #FFD700; box-shadow: 0 20px 50px rgba(255, 215, 0, 0.3);">
        <div style="font-size: 6rem; color: #FFD700; margin-bottom: 1.5rem; filter: drop-shadow(0 8px 16px rgba(0,0,0,0.3));" class="floating">🎬</div>
        <h2 style="color: #FFD700 !important; margin: 0; font-size: 2.8rem; font-weight: 900; text-shadow: 0 4px 8px rgba(0,0,0,0.5);">NETFLIX</h2>
        <p style="color: rgba(255, 255, 255, 0.95); font-size: 1.4rem; margin-top: 0.8rem; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">Recommender System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧭 MENU NAVIGASI")
    page = st.radio(
        "Pilih Halaman:",
        ["🎯 REKOMENDASI", "📊 DASHBOARD ANALITIK", "🤖 TENTANG SISTEM"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### ⚙️ KONFIGURASI DATA")
    with st.expander("📁 PENGATURAN DATASET", expanded=True):
        uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"], help="Unggah file dataset Netflix dalam format CSV")
        use_local = st.checkbox("Gunakan dataset lokal", value=True, help="Gunakan file netflix_titles.csv yang tersedia")

    st.divider()

    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "tfidf_matrix" not in st.session_state:
        st.session_state.tfidf_matrix = None

# -----------------------------
# Load Data - SIMPLIFIED VERSION
# -----------------------------
raw_df = None
data_loaded = False

# Simple data loading without complex logic
if uploaded is not None:
    raw_df = load_data_from_upload(uploaded.getvalue())
    if not raw_df.empty:
        data_loaded = True
        ui_alert("success", f"<b>Dataset berhasil dimuat:</b> {len(raw_df):,} baris data")
    else:
        ui_alert("error", "Dataset kosong atau tidak valid!")
elif DEFAULT_DATA_PATH.exists():
    raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
    if not raw_df.empty:
        data_loaded = True
        ui_alert("success", f"<b>Dataset lokal berhasil dimuat:</b> {len(raw_df):,} baris data")
    else:
        ui_alert("error", "File lokal kosong!")
else:
    ui_alert("info", "Silakan unggah dataset Netflix CSV atau pastikan file netflix_titles.csv tersedia.")

# -----------------------------
# Process Data
# -----------------------------
if data_loaded:
    try:
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
        
        stats = create_dashboard_stats(df)
        
    except Exception as e:
        ui_alert("error", f"Error dalam memproses data: <code>{_safe_str(e)}</code>")
        st.stop()
else:
    st.warning("Data belum dimuat. Silakan unggah dataset atau gunakan dataset lokal.")
    st.stop()

# Get data from session state
df = st.session_state.df
vectorizer = st.session_state.vectorizer
tfidf_matrix = st.session_state.tfidf_matrix
stats = create_dashboard_stats(df)

# Sidebar status + stats
with st.sidebar:
    ui_alert(
        "success",
        """
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2.5rem; color: #00B894; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));">✅</div>
            <div>
                <div style="font-weight: 900; color: #00B894 !important; font-size: 1.4rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">SISTEM AKTIF</div>
                <div style="font-size: 1.1rem; color: #CCCCCC !important;">Dataset berhasil diproses</div>
            </div>
        </div>
        """,
    )

    st.markdown("### 📊 STATISTIK DATASET")
    st.markdown(
        f"""
        <div class="glass-panel" style="padding: 2.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 2rem;">
                <div>
                    <div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.5rem;">Total Data</div>
                    <div style="font-size: 2.5rem; font-weight: 900; color: #FFD700 !important; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">{stats['total']:,}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.5rem;">Unique Types</div>
                    <div style="font-size: 2.5rem; font-weight: 900; color: #FFD700 !important; text-shadow: 0 3px 6px rgba(0,0,0,0.3);">{stats['unique_types']}</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.5rem;">Movies</div>
                    <div style="font-size: 2rem; font-weight: 900; color: #00B894 !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{stats['movies']:,}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.5rem;">TV Shows</div>
                    <div style="font-size: 2rem; font-weight: 900; color: #FDCB6E !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{stats['tv_shows']:,}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 🔍 FILTER CEPAT")

    min_year = stats.get("min_year", 1900)
    max_year = stats.get("max_year", datetime.now().year)

    sidebar_year_range = st.slider("Tahun Rilis", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="sidebar_year")

unique_types = sorted([t for t in df["type"].unique().tolist() if t and str(t) != "nan"])
type_options = ["All"] + unique_types

min_year = stats.get("min_year", 1900)
max_year = stats.get("max_year", datetime.now().year)

# -----------------------------
# Page: Recommendation - SIMPLIFIED
# -----------------------------
if page == "🎯 REKOMENDASI":
    tabs = st.tabs(["🎬 BERDASARKAN JUDUL", "🔍 BERDASARKAN KATA KUNCI", "⭐ KONTEN POPULER"])

    # Tab 1: BERDASARKAN JUDUL
    with tabs[0]:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Simple and clear UI
            st.markdown('<div class="search-section">', unsafe_allow_html=True)
            st.markdown('<div class="search-title">🎬 PILIH KONTEN UNTUK DIREKOMENDASIKAN</div>', unsafe_allow_html=True)
            
            # Filter Type
            st.markdown('<div class="search-input-container">', unsafe_allow_html=True)
            st.markdown('<div class="search-label">🎭 FILTER TIPE KONTEN</div>', unsafe_allow_html=True)
            
            filter_type_for_selector = st.selectbox(
                "Pilih tipe konten:",
                options=type_options,
                index=0,
                key="filter_type_title",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if filter_type_for_selector == "All":
                selector_df = df
            else:
                selector_df = df[df["type"] == filter_type_for_selector]

            # Select Title
            st.markdown('<div class="search-input-container">', unsafe_allow_html=True)
            st.markdown('<div class="search-label">📝 PILIH JUDUL</div>', unsafe_allow_html=True)
            
            if len(selector_df) > 0:
                options = selector_df["display_title"].tolist()
                selected_display = st.selectbox(
                    "Pilih judul untuk rekomendasi:",
                    options=options,
                    index=0,
                    key="title_selector",
                )
            else:
                selected_display = None
                ui_alert("warning", f"Tidak ada konten dengan tipe: <b>{filter_type_for_selector}</b>")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Settings
            st.markdown("### ⚙️ PENGATURAN REKOMENDASI")
            col_settings1, col_settings2 = st.columns(2)
            
            with col_settings1:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<div style="color: #FFD700 !important; font-size: 1.2rem; font-weight: 800; margin-bottom: 1rem;">📊 JUMLAH REKOMENDASI</div>', unsafe_allow_html=True)
                top_n = st.slider("", 5, 20, 10, 1, label_visibility="collapsed", key="top_n_slider")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_settings2:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<div style="color: #FFD700 !important; font-size: 1.2rem; font-weight: 800; margin-bottom: 1rem;">🎯 FILTER TIPE</div>', unsafe_allow_html=True)
                same_type = st.checkbox("Hanya tipe yang sama", value=True, key="same_type_check")
                st.markdown('</div>', unsafe_allow_html=True)

            # Year Filter
            st.markdown('<div class="search-input-container">', unsafe_allow_html=True)
            st.markdown('<div class="search-label">📅 FILTER TAHUN RILIS</div>', unsafe_allow_html=True)
            
            year_range = st.slider(
                "Pilih rentang tahun:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                format="%d",
                key="year_range_title",
            )
            year_min, year_max = year_range
            st.markdown('</div>', unsafe_allow_html=True)

            # Get Recommendations Button
            st.markdown('</div>', unsafe_allow_html=True)  # Close search-section
            
            if selected_display and st.button("🚀 DAPATKAN REKOMENDASI", type="primary", use_container_width=True, key="get_recs_btn"):
                matches = df[df["display_title"] == selected_display]
                if len(matches) == 0:
                    ui_alert("error", "Judul tidak ditemukan!")
                else:
                    idx = matches.index[0]
                    
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
                    
                    if recs.empty:
                        ui_alert("warning", "Tidak menemukan rekomendasi. Coba sesuaikan filter.")
                    else:
                        st.markdown(f'<div class="search-title">🎯 {len(recs)} REKOMENDASI TERBAIK</div>', unsafe_allow_html=True)
                        for i, (_, r) in enumerate(recs.iterrows(), 1):
                            display_recommendation_card(r, i)

        with col2:
            st.markdown("### 📊 DASHBOARD DATASET")
            display_metric_card("Total Konten", f"{stats['total']:,}", "Movies & TV Shows", "📈")
            display_metric_card("Movies", f"{stats['movies']:,}", f"{(stats['movies']/stats['total']*100):.1f}%" if stats["total"] else "0%", "🎥")
            display_metric_card("TV Shows", f"{stats['tv_shows']:,}", f"{(stats['tv_shows']/stats['total']*100):.1f}%" if stats["total"] else "0%", "📺")
            display_metric_card("Tahun Terbaru", str(stats["max_year"]), "Konten terupdate", "🚀")

    # Tab 2: BERDASARKAN KATA KUNCI
    with tabs[1]:
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        st.markdown('<div class="search-title">🔍 PENCARIAN DENGAN KATA KUNCI</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="search-input-container">', unsafe_allow_html=True)
        st.markdown('<div class="search-label">🔎 MASUKKAN KATA KUNCI</div>', unsafe_allow_html=True)
        
        query = st.text_input(
            "Masukkan kata kunci:",
            placeholder="Contoh: action, comedy, sci-fi, drama",
            key="search_query",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        search_btn = st.button("🔍 LAKUKAN PENCARIAN", type="primary", use_container_width=True, key="search_btn")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if search_btn:
            if not query:
                ui_alert("warning", "Masukkan kata kunci terlebih dahulu")
            else:
                with st.spinner("🔎 Mencari konten..."):
                    recs_q = recommend_by_query(
                        query=query,
                        df=df,
                        vectorizer=vectorizer,
                        tfidf_matrix=tfidf_matrix,
                        top_n=10,
                    )
                
                if recs_q.empty:
                    ui_alert("error", f'Tidak menemukan hasil untuk: "{query}"')
                else:
                    st.markdown(f'<div class="search-title">🎬 HASIL PENCARIAN</div>', unsafe_allow_html=True)
                    for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                        display_recommendation_card(r, i)

    # Tab 3: KONTEN POPULER
    with tabs[2]:
        st.markdown("## ⭐ KONTEN POPULER")
        
        if len(df) > 0:
            sample_size = min(8, len(df))
            random_sample = df.sample(sample_size)
            cols = st.columns(2)

            for i, (_, item) in enumerate(random_sample.iterrows()):
                with cols[i % 2]:
                    st.markdown(
                        f"""
                        <div class="glass-panel">
                            <h4 style="margin: 0 0 1.2rem 0; color: #FFD700 !important; font-size: 1.4rem; font-weight: 900;">
                                {_safe_str(item['title'])[:40]}{'...' if len(_safe_str(item['title']))>40 else ''}
                            </h4>
                            <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
                                <span class="badge badge-movie" style="font-size: 0.8rem; padding: 0.5rem 1rem;">{_safe_str(item['type'])}</span>
                                <span class="badge badge-year" style="font-size: 0.8rem; padding: 0.5rem 1rem;">{int(item['release_year'])}</span>
                            </div>
                            <div style="color: #F5F5F5 !important; font-size: 1rem; line-height: 1.7; margin-bottom: 2rem;">
                                {_safe_str(item['description'])[:120]}{'...' if len(_safe_str(item['description']))>120 else ''}
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 1rem;">
                                <span style="color: #FFD700 !important; font-weight: 700;">⭐ {_safe_str(item['rating']) or 'N/A'}</span>
                                <span style="color: #FFD700 !important; font-weight: 700;">⏱️ {_safe_str(item['duration']) or 'N/A'}</span>
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
    <div style="text-align: center; margin-top: 5rem; padding: 3rem; border-top: 3px solid #FFD700;">
        <p style="color: #F5F5F5 !important; margin: 0; font-size: 1.1rem; font-weight: 600;">
            🎬 Netflix Recommender System • Powered by Content-Based Filtering • 
            <span style="color: #FFD700 !important; font-weight: 900;">NETFLIX</span>
        </p>
        <p style="color: #CCCCCC !important; margin: 1rem 0 0 0; font-size: 0.9rem;">
            © 2024 Sistem Rekomendasi Netflix • All rights reserved
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
