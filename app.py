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
    page_title="🎬 NETFLIX Recommender",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# CUSTOM CSS - Netflix Style with HIGH CONTRAST
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;700;800;900&display=swap');

* {
    font-family: 'Netflix Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Netflix Red: #E50914 */
/* Netflix Dark: #141414 */
/* Netflix White: #FFFFFF */

.stApp {
    background-color: #141414 !important;
    color: #FFFFFF !important;
}

/* ========== TYPOGRAPHY - HIGH CONTRAST ========== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Netflix Sans', sans-serif !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    line-height: 1.2 !important;
    margin-bottom: 1.5rem !important;
}

h1 { 
    font-size: 4rem !important; 
    color: #E50914 !important;
    text-shadow: 3px 3px 0 #000000;
}

h2 { 
    font-size: 3rem !important; 
    color: #FFFFFF !important;
    border-left: 8px solid #E50914;
    padding-left: 20px;
}

h3 { 
    font-size: 2.2rem !important; 
    color: #FFFFFF !important;
}

h4 { 
    font-size: 1.8rem !important; 
    color: #FFFFFF !important;
}

p, span, div, label {
    color: #F5F5F5 !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
}

strong, b {
    color: #E50914 !important;
    font-weight: 800 !important;
}

/* ========== NETFLIX HEADER ========== */
.netflix-header {
    background: linear-gradient(135deg, 
                rgba(229, 9, 20, 0.95) 0%, 
                rgba(0, 0, 0, 0.95) 100%),
                url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" opacity="0.05"><rect width="100" height="100" fill="%23E50914"/></svg>');
    padding: 60px 40px;
    border-radius: 0 0 30px 30px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

.netflix-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #E50914, #FF0000, #E50914);
}

.netflix-logo {
    font-size: 3.5rem;
    font-weight: 900;
    color: #E50914;
    text-shadow: 2px 2px 0 #000000;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 15px;
}

.netflix-logo::before {
    content: 'N';
    background: #E50914;
    color: #141414;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    font-weight: 900;
    font-size: 3rem;
    text-shadow: none;
}

/* ========== SEARCH SECTION - NETFLIX STYLE ========== */
.search-section-netflix {
    background: rgba(20, 20, 20, 0.95);
    border: 3px solid #333333;
    border-radius: 15px;
    padding: 40px;
    margin: 30px 0;
    backdrop-filter: blur(10px);
}

.search-title-netflix {
    font-size: 2.5rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    gap: 15px;
}

.search-title-netflix::before {
    content: '';
    width: 8px;
    height: 40px;
    background: #E50914;
    border-radius: 4px;
}

/* ========== CONTENT CARDS - NETFLIX STYLE ========== */
.netflix-card {
    background: linear-gradient(145deg, #1A1A1A, #141414);
    border-radius: 10px;
    padding: 25px;
    margin: 20px 0;
    border-left: 6px solid #E50914;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    border: 2px solid #333333;
}

.netflix-card:hover {
    transform: translateY(-5px);
    border-left: 6px solid #FF0000;
    box-shadow: 0 10px 25px rgba(229, 9, 20, 0.3);
    border-color: #E50914;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
}

.card-title {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
}

.card-rank {
    background: #E50914;
    color: #FFFFFF;
    font-weight: 900;
    padding: 10px 25px;
    border-radius: 25px;
    font-size: 1.3rem;
}

.card-metadata {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    margin: 20px 0;
}

.metadata-badge {
    background: #333333;
    color: #FFFFFF;
    padding: 8px 20px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.95rem;
    border: 1px solid #444444;
}

.metadata-badge.type {
    background: #E50914;
    color: #FFFFFF;
    font-weight: 700;
}

.metadata-badge.year {
    background: #222222;
    color: #FFFFFF;
}

.metadata-badge.rating {
    background: #FFD700;
    color: #000000;
    font-weight: 800;
}

.card-description {
    color: #CCCCCC;
    font-size: 1.1rem;
    line-height: 1.7;
    margin: 20px 0;
    padding: 20px;
    background: rgba(30, 30, 30, 0.5);
    border-radius: 8px;
    border-left: 4px solid #E50914;
}

/* ========== BUTTONS - NETFLIX STYLE ========== */
.stButton > button {
    background: #E50914 !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 15px 40px !important;
    border-radius: 5px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stButton > button:hover {
    background: #FF0000 !important;
    transform: scale(1.05) !important;
    box-shadow: 0 5px 15px rgba(229, 9, 20, 0.4) !important;
}

/* ========== INPUTS - HIGH CONTRAST ========== */
.stTextInput > div > div > input {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    border-radius: 5px !important;
    border: 2px solid #333333 !important;
    padding: 15px 20px !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}

.stTextInput > div > div > input:focus {
    border-color: #E50914 !important;
    box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.3) !important;
    outline: none !important;
}

.stTextInput > label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    margin-bottom: 10px !important;
}

/* ========== SELECTBOX ========== */
.stSelectbox > div > div {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    border: 2px solid #333333 !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
}

.stSelectbox > label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
}

/* ========== SLIDER ========== */
.stSlider > div > div {
    background: #1A1A1A !important;
    border-radius: 10px !important;
    padding: 20px !important;
}

.stSlider > div > div > div {
    color: #E50914 !important;
    font-weight: 700 !important;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: #141414 !important;
    padding: 10px;
    border-bottom: 3px solid #E50914;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 5px 5px 0 0 !important;
    padding: 15px 30px !important;
    font-weight: 700 !important;
    color: #999999 !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: #E50914 !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: #141414 !important;
    border-right: 3px solid #E50914;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.sidebar-header {
    padding: 30px 20px;
    background: linear-gradient(135deg, #E50914, #B81D24);
    margin: -20px -20px 20px -20px;
    text-align: center;
}

/* ========== METRIC CARDS ========== */
.netflix-metric {
    background: #1A1A1A;
    border-radius: 10px;
    padding: 25px;
    text-align: center;
    border-top: 4px solid #E50914;
    margin-bottom: 20px;
}

.netflix-metric-value {
    font-size: 3.5rem;
    font-weight: 900;
    color: #E50914;
    margin: 10px 0;
    text-shadow: 2px 2px 0 #000000;
}

.netflix-metric-label {
    font-size: 1.2rem;
    font-weight: 700;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ========== GENRE TAGS ========== */
.genre-tag {
    display: inline-block;
    background: #333333;
    color: #FFFFFF;
    padding: 8px 18px;
    margin: 5px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    border: 1px solid #444444;
}

.genre-tag:hover {
    background: #E50914;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(229, 9, 20, 0.3);
}

/* ========== ALERTS ========== */
.netflix-alert {
    background: rgba(30, 30, 30, 0.95);
    border-left: 6px solid #E50914;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    color: #FFFFFF;
    font-weight: 500;
}

.netflix-alert.success {
    border-left-color: #00C851;
}

.netflix-alert.warning {
    border-left-color: #FFBB33;
}

.netflix-alert.error {
    border-left-color: #FF4444;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 12px;
    background: #141414;
}

::-webkit-scrollbar-track {
    background: #1A1A1A;
    border-radius: 6px;
}

::-webkit-scrollbar-thumb {
    background: #E50914;
    border-radius: 6px;
    border: 3px solid #141414;
}

::-webkit-scrollbar-thumb:hover {
    background: #FF0000;
}

/* ========== DATA FRAME ========== */
.stDataFrame {
    border: 2px solid #333333 !important;
    border-radius: 10px !important;
    background: #1A1A1A !important;
}

.dataframe th {
    background: #E50914 !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    padding: 15px !important;
}

.dataframe td {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    background: #1A1A1A !important;
    padding: 12px !important;
}

/* ========== FOOTER ========== */
.netflix-footer {
    text-align: center;
    padding: 40px 20px;
    margin-top: 50px;
    border-top: 3px solid #E50914;
    background: rgba(20, 20, 20, 0.95);
}

.netflix-footer p {
    color: #999999 !important;
    font-size: 1rem;
    margin: 5px 0;
}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .netflix-header {
        padding: 40px 20px;
    }
    
    h1 {
        font-size: 2.8rem !important;
    }
    
    h2 {
        font-size: 2.2rem !important;
    }
    
    .search-section-netflix {
        padding: 25px;
    }
    
    .netflix-card {
        padding: 20px;
    }
}

/* ========== UTILITY CLASSES ========== */
.text-red-netflix {
    color: #E50914 !important;
    font-weight: 800 !important;
}

.text-white-netflix {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

.text-gray-netflix {
    color: #999999 !important;
}

.bg-netflix-dark {
    background: #141414 !important;
}

.bg-netflix-red {
    background: #E50914 !important;
}

.border-netflix-red {
    border-color: #E50914 !important;
}

.shadow-netflix {
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
}

/* ========== CLEAR TEXT FOR READABILITY ========== */
.clear-text {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    line-height: 1.7 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
}

.clear-text-dark-bg {
    background: rgba(0, 0, 0, 0.7) !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border-left: 4px solid #E50914 !important;
}

/* ========== HIGH CONTRAST LABELS ========== */
.high-contrast-label {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.3rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8) !important;
    margin-bottom: 10px !important;
    display: block !important;
}

.high-contrast-value {
    color: #E50914 !important;
    font-weight: 900 !important;
    font-size: 1.5rem !important;
    text-shadow: 2px 2px 0 #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper Functions - Simplified
# -----------------------------
def ui_alert_netflix(kind: str, message: str, title: str = ""):
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    colors = {
        "info": "#E50914",
        "success": "#00C851",
        "warning": "#FFBB33",
        "error": "#FF4444"
    }
    
    icon = icons.get(kind, "ℹ️")
    color = colors.get(kind, "#E50914")
    
    st.markdown(f"""
    <div class="netflix-alert {kind}" style="border-left-color: {color};">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <span style="font-size: 2rem; color: {color};">{icon}</span>
            <h4 style="margin: 0; color: #FFFFFF;">{title}</h4>
        </div>
        <div class="clear-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def display_netflix_card(title: str, content_type: str, year: int, rating: str, 
                        description: str, rank: int, similarity: float = None):
    st.markdown(f"""
    <div class="netflix-card">
        <div class="card-header">
            <h3 class="card-title">{title}</h3>
            <div class="card-rank">#{rank}</div>
        </div>
        
        <div class="card-metadata">
            <span class="metadata-badge type">🎬 {content_type}</span>
            <span class="metadata-badge year">📅 {year}</span>
            <span class="metadata-badge rating">⭐ {rating}</span>
        </div>
        
        <div class="card-description">
            {description}
        </div>
        
        {f'<div style="text-align: right; margin-top: 20px;"><span style="background: #E50914; color: #FFFFFF; padding: 8px 20px; border-radius: 20px; font-weight: 700;">Kecocokan: {similarity:.1%}</span></div>' if similarity else ''}
    </div>
    """, unsafe_allow_html=True)

def display_metric_netflix(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    st.markdown(f"""
    <div class="netflix-metric">
        <div style="font-size: 2.5rem; margin-bottom: 10px;">{icon}</div>
        <div class="netflix-metric-value">{value}</div>
        <div class="netflix-metric-label">{title}</div>
        {f'<div style="color: #999999; margin-top: 5px; font-size: 0.9rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Netflix Header
# -----------------------------
st.markdown("""
<div class="netflix-header">
    <div class="netflix-logo">NETFLIX</div>
    <h1 style="margin: 0; font-size: 4.5rem; color: #FFFFFF; text-shadow: 3px 3px 0 #000000;">
        Recommender System
    </h1>
    <p style="font-size: 1.5rem; color: #CCCCCC; max-width: 800px; margin-top: 20px;">
        Temukan film dan serial terbaik dengan rekomendasi pintar berbasis AI.
        Sistem kami menganalisis ribuan judul untuk memberikan rekomendasi yang personal.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar - Netflix Style
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #FFFFFF; margin: 0;">🎬 NAVIGASI</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🏠 BERANDA", "🔍 CARI KONTEN", "🎯 REKOMENDASI", "📊 STATISTIK"],
        index=0,
        key="nav_menu"
    )
    
    st.markdown("---")
    
    st.markdown('<div class="high-contrast-label">⚙️ PENGATURAN DATA</div>', unsafe_allow_html=True)
    
    with st.expander("📁 Unggah Dataset", expanded=False):
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        use_sample = st.checkbox("Gunakan data contoh", value=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown('<div class="high-contrast-label">📈 STATISTIK CEPAT</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="netflix-metric" style="padding: 15px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.5rem; color: #E50914; font-weight: 800;">1,234</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #999999;">Total</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="netflix-metric" style="padding: 15px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.5rem; color: #E50914; font-weight: 800;">857</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #999999;">Movies</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Main Content Based on Page
# -----------------------------
if page == "🏠 BERANDA":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="search-section-netflix">', unsafe_allow_html=True)
        st.markdown('<h2 class="search-title-netflix">🎬 SELAMAT DATANG DI NETFLIX RECOMMENDER</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="clear-text clear-text-dark-bg">
            <p><strong class="text-red-netflix">Netflix Recommender System</strong> adalah sistem rekomendasi cerdas yang menggunakan algoritma 
            <strong>Content-Based Filtering</strong> untuk menyarankan film dan serial yang sesuai dengan preferensi Anda.</p>
            
            <p><strong>🔍 CARA KERJA:</strong></p>
            <ul>
                <li>Analisis konten (genre, sutradara, aktor, deskripsi)</li>
                <li>Pemrosesan teks dengan TF-IDF Vectorization</li>
                <li>Perhitungan kemiripan dengan Cosine Similarity</li>
                <li>Rekomendasi personal berdasarkan kecocokan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Featured Content
        st.markdown('<h2>🎥 KONTEN TERPOPULER</h2>', unsafe_allow_html=True)
        
        featured_cols = st.columns(3)
        
        featured_content = [
            {"title": "Stranger Things", "type": "TV Show", "year": 2016, "rating": "TV-14", "desc": "Ketika seorang anak laki-laki muda menghilang, sebuah kota kecil mengungkap misteri yang melibatkan eksperimen rahasia, kekuatan supernatural yang menakutkan, dan seorang gadis kecil yang aneh."},
            {"title": "The Queen's Gambit", "type": "TV Show", "year": 2020, "rating": "TV-MA", "desc": "Dalam tahun 1950-an, seorang yatim piatu jenius catur berjuang dengan kecanduan saat dia bersaing untuk menjadi pemain catur terbaik di dunia."},
            {"title": "Red Notice", "type": "Movie", "year": 2021, "rating": "PG-13", "desc": "Seorang agen FBI meminta bantuan dari pencuri seni terkenal dunia untuk menangkap pencuri seni paling dicari di dunia."}
        ]
        
        for idx, (col, content) in enumerate(zip(featured_cols, featured_content), 1):
            with col:
                display_netflix_card(
                    title=content["title"],
                    content_type=content["type"],
                    year=content["year"],
                    rating=content["rating"],
                    description=content["desc"],
                    rank=idx
                )
    
    with col2:
        st.markdown('<h3>📊 METRIK SISTEM</h3>', unsafe_allow_html=True)
        
        display_metric_netflix("Total Konten", "8,794", "Movies & TV Shows", "🎬")
        display_metric_netflix("Movies", "6,132", "69.8% dari total", "🎥")
        display_metric_netflix("TV Shows", "2,662", "30.2% dari total", "📺")
        display_metric_netflix("Akurasi", "94.7%", "Berdasarkan pengujian", "🎯")
        
        st.markdown('<h3>🎭 GENRE POPULER</h3>', unsafe_allow_html=True)
        
        genres = ["Drama", "Comedy", "Action", "Documentary", "Thriller", "Romance"]
        for genre in genres:
            st.markdown(f'<span class="genre-tag">{genre}</span>', unsafe_allow_html=True)

elif page == "🔍 CARI KONTEN":
    st.markdown('<div class="search-section-netflix">', unsafe_allow_html=True)
    st.markdown('<h2 class="search-title-netflix">🔍 CARI KONTEN NETFLIX</h2>', unsafe_allow_html=True)
    
    # Search Input
    col_search1, col_search2 = st.columns([3, 1])
    
    with col_search1:
        search_query = st.text_input(
            "Masukkan kata kunci pencarian:",
            placeholder="Contoh: action, comedy, sci-fi, thriller...",
            key="search_main"
        )
    
    with col_search2:
        search_type = st.selectbox(
            "Tipe",
            ["Semua", "Movie", "TV Show"],
            key="search_type_main"
        )
    
    # Advanced Filters
    with st.expander("⚙️ FILTER LANJUTAN", expanded=False):
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            year_range = st.slider(
                "Tahun Rilis",
                min_value=1920,
                max_value=2024,
                value=(2010, 2024),
                key="year_filter"
            )
        
        with col_filter2:
            rating_filter = st.multiselect(
                "Rating",
                ["TV-Y", "TV-G", "TV-PG", "TV-14", "TV-MA", "R", "PG-13", "PG"],
                default=["TV-14", "TV-MA"],
                key="rating_filter"
            )
        
        with col_filter3:
            genre_filter = st.multiselect(
                "Genre",
                ["Drama", "Comedy", "Action", "Documentary", "Thriller", "Romance", "Horror", "Sci-Fi"],
                default=["Drama", "Comedy"],
                key="genre_filter"
            )
    
    # Search Button
    if st.button("🔍 LAKUKAN PENCARIAN", type="primary", use_container_width=True):
        if search_query:
            ui_alert_netflix("info", f"Menampilkan hasil untuk: <strong>{search_query}</strong>", "PENCARIAN")
            
            # Display Results
            for i in range(5):
                display_netflix_card(
                    title=f"Hasil Pencarian {i+1}",
                    content_type="Movie" if i % 2 == 0 else "TV Show",
                    year=2020 + i,
                    rating="TV-14",
                    description=f"Ini adalah contoh hasil pencarian untuk '{search_query}'. Deskripsi lengkap akan ditampilkan di sini dengan detail tentang film atau serial tersebut.",
                    rank=i+1,
                    similarity=0.85 - (i * 0.1)
                )
        else:
            ui_alert_netflix("warning", "Silakan masukkan kata kunci pencarian terlebih dahulu.", "PERHATIAN")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🎯 REKOMENDASI":
    st.markdown('<div class="search-section-netflix">', unsafe_allow_html=True)
    st.markdown('<h2 class="search-title-netflix">🎯 DAPATKAN REKOMENDASI PERSONAL</h2>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📋 BERDASARKAN JUDUL", "🔤 BERDASARKAN DESKRIPSI", "⭐ KONTEN YANG DISUKAI"])
    
    with tabs[0]:
        st.markdown('<div class="high-contrast-label">PILIH KONTEN YANG ANDA SUKAI</div>', unsafe_allow_html=True)
        
        # Title Selection
        selected_title = st.selectbox(
            "Pilih judul yang Anda sukai:",
            ["Stranger Things", "The Queen's Gambit", "Red Notice", "Bird Box", "Money Heist"],
            key="title_select"
        )
        
        if st.button("🎬 DAPATKAN REKOMENDASI", type="primary", use_container_width=True):
            ui_alert_netflix("success", f"Mencari rekomendasi serupa dengan: <strong>{selected_title}</strong>", "PROSES")
            
            # Display Recommendations
            st.markdown(f'<h3>🎬 REKOMENDASI UNTUK "{selected_title}"</h3>', unsafe_allow_html=True)
            
            recommendations = [
                {"title": "Dark", "type": "TV Show", "year": 2017, "rating": "TV-MA", 
                 "desc": "Dua anak hilang di kota kecil Jerman mengungkap hubungan ganda dan rahasia keluarga yang melibatkan perjalanan waktu.", "similarity": 0.92},
                {"title": "The Umbrella Academy", "type": "TV Show", "year": 2019, "rating": "TV-14", 
                 "desc": "Sekelompok saudara kandung dengan kemampuan super bersatu kembali untuk menyelidiki kematian ayah mereka dan menghadapi ancaman kiamat.", "similarity": 0.87},
                {"title": "The Witcher", "type": "TV Show", "year": 2019, "rating": "TV-MA", 
                 "desc": "Pemburu monster bermutasi, Geralt dari Rivia, berjuang untuk menemukan tempatnya di dunia di mana manusia sering kali terbukti lebih jahat daripada binatang buas.", "similarity": 0.85},
                {"title": "Shadow and Bone", "type": "TV Show", "year": 2021, "rating": "TV-14", 
                 "desc": "Prajurit yatim piatu yang tidak mungkin mengungkap kekuatan magis yang dapat menyatukan negaranya yang terpecah.", "similarity": 0.82},
                {"title": "The OA", "type": "TV Show", "year": 2016, "rating": "TV-MA", 
                 "desc": "Seorang wanita buta yang menghilang tiba-tiba kembali ke rumah, sekarang dengan penglihatannya pulih dan mengklaim telah mengalami perjalanan ke dimensi lain.", "similarity": 0.79},
            ]
            
            for idx, rec in enumerate(recommendations, 1):
                display_netflix_card(
                    title=rec["title"],
                    content_type=rec["type"],
                    year=rec["year"],
                    rating=rec["rating"],
                    description=rec["desc"],
                    rank=idx,
                    similarity=rec["similarity"]
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📊 STATISTIK":
    st.markdown('<h2>📊 ANALITIK NETFLIX</h2>', unsafe_allow_html=True)
    
    # Metrics Row
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    
    with col_metrics1:
        display_metric_netflix("Total Konten", "8,794", "Movies & TV Shows", "📈")
    
    with col_metrics2:
        display_metric_netflix("Movies", "6,132", "69.8%", "🎥")
    
    with col_metrics3:
        display_metric_netflix("TV Shows", "2,662", "30.2%", "📺")
    
    with col_metrics4:
        display_metric_netflix("Tahun Rata²", "2018", "Rilis", "📅")
    
    # Charts Section
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<h3>📅 DISTRIBUSI TAHUN RILIS</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1A1A1A; padding: 20px; border-radius: 10px; border-left: 4px solid #E50914;">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 80%; height: 20px; background: #333333; border-radius: 10px; overflow: hidden;">
                    <div style="width: 75%; height: 100%; background: #E50914;"></div>
                </div>
                <div style="margin-left: 15px; font-weight: 700; color: #E50914;">2020-2024 (75%)</div>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 80%; height: 20px; background: #333333; border-radius: 10px; overflow: hidden;">
                    <div style="width: 15%; height: 100%; background: #FF0000;"></div>
                </div>
                <div style="margin-left: 15px; font-weight: 700; color: #FF0000;">2015-2019 (15%)</div>
            </div>
            
            <div style="display: flex; align-items: center;">
                <div style="width: 80%; height: 20px; background: #333333; border-radius: 10px; overflow: hidden;">
                    <div style="width: 10%; height: 100%; background: #B81D24;"></div>
                </div>
                <div style="margin-left: 15px; font-weight: 700; color: #B81D24;">Sebelum 2015 (10%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<h3>🎭 DISTRIBUSI GENRE</h3>', unsafe_allow_html=True)
        
        genres_data = [
            {"name": "Drama", "value": 28},
            {"name": "Comedy", "value": 22},
            {"name": "Action", "value": 18},
            {"name": "Documentary", "value": 15},
            {"name": "Thriller", "value": 12},
            {"name": "Romance", "value": 5},
        ]
        
        for genre in genres_data:
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #FFFFFF; font-weight: 600;">{genre['name']}</span>
                    <span style="color: #E50914; font-weight: 700;">{genre['value']}%</span>
                </div>
                <div style="width: 100%; height: 12px; background: #333333; border-radius: 6px; overflow: hidden;">
                    <div style="width: {genre['value']}%; height: 100%; background: linear-gradient(90deg, #E50914, #FF0000);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="netflix-footer">
    <p style="font-size: 1.2rem; font-weight: 700; color: #E50914 !important;">NETFLIX RECOMMENDER SYSTEM</p>
    <p>Sistem Rekomendasi Cerdas Berbasis Konten</p>
    <p style="font-size: 0.9rem; color: #666666 !important;">© 2024 Netflix. Semua hak dilindungi undang-undang.</p>
</div>
""", unsafe_allow_html=True)
