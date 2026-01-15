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
# Custom CSS - Modern Netflix Style
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;700;800&display=swap');
    
    * {
        font-family: 'Netflix Sans', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Netflix Red Color Scheme */
    :root {
        --netflix-red: #E50914;
        --netflix-dark: #141414;
        --netflix-gray: #808080;
        --netflix-light: #F5F5F1;
    }
    
    .main { padding: 1rem; }
    
    /* Netflix Header */
    .netflix-header {
        background: linear-gradient(to bottom, rgba(20, 20, 20, 0.95), rgba(20, 20, 20, 0.7));
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        border-bottom: 4px solid var(--netflix-red);
    }
    
    .netflix-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('https://assets.nflxext.com/ffe/siteui/vlv3/5eab1b22-c5ea-48b0-8ef4-862b3fa6df2c/7a5a83e2-2a9d-4f37-9dcf-61b9d3b6a6e5/ID-id-20240422-popsignuptwoweeks-perspective_alpha_website_small.jpg') center/cover no-repeat;
        opacity: 0.4;
        z-index: -1;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .netflix-logo {
        font-family: 'Netflix Sans', sans-serif !important;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        color: var(--netflix-red) !important;
        text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.5);
        letter-spacing: -1px;
    }
    
    .netflix-badge {
        background: var(--netflix-red);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Card Styling - Netflix Style */
    .netflix-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--netflix-red);
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .netflix-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 35px rgba(229, 9, 20, 0.2);
        background: linear-gradient(145deg, #222222, #333333);
    }
    
    .netflix-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--netflix-red), #B20710);
    }
    
    /* Search Button - Netflix Style */
    .stButton > button {
        background: var(--netflix-red) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        background: #B20710 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(229, 9, 20, 0.4) !important;
    }
    
    /* Search Input */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--netflix-red) !important;
        box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: var(--netflix-red) !important;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    /* Checkbox Styling */
    .stCheckbox > label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(20, 20, 20, 0.8) !important;
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        color: var(--netflix-gray) !important;
        border: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--netflix-red) !important;
        color: white !important;
        border-color: var(--netflix-red) !important;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #141414, #1a1a1a) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Metric Cards */
    .netflix-metric {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-bottom: 3px solid var(--netflix-red);
        transition: all 0.3s ease;
    }
    
    .netflix-metric:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    
    .netflix-metric h3 {
        color: var(--netflix-red) !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        font-weight: 800 !important;
    }
    
    /* Badge Styling */
    .netflix-badge-item {
        display: inline-block;
        background: rgba(229, 9, 20, 0.15);
        color: var(--netflix-red);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    
    /* Similarity Score */
    .similarity-badge {
        background: linear-gradient(135deg, #E50914, #B20710);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    /* Content Type Badges */
    .badge-movie {
        background: linear-gradient(135deg, #E50914, #B20710) !important;
        color: white !important;
    }
    
    .badge-tv {
        background: linear-gradient(135deg, #0071EB, #00A8FF) !important;
        color: white !important;
    }
    
    .badge-year {
        background: linear-gradient(135deg, #00B140, #00D46A) !important;
        color: white !important;
    }
    
    .badge-rating {
        background: linear-gradient(135deg, #FFB800, #FF9500) !important;
        color: white !important;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .netflix-card {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 20, 20, 0.8);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--netflix-red);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #B20710;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .netflix-logo { font-size: 2.5rem !important; }
        .netflix-header { padding: 2rem 1rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.6rem 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

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
        "cast": "cast",
        "country": "country",
        "release year": "release_year",
        "release_year": "release_year",
        "rating": "rating",
        "duration": "duration",
        "listed in": "listed_in",
        "listed_in": "listed_in",
        "description": "description",
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    expected = [
        "show_id", "type", "title", "director", "cast", "country",
        "release_year", "rating", "duration", "listed_in", "description"
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
        df["title"].map(_normalize_text) + " " +
        df["type"].map(_normalize_text) + " " +
        df["director"].map(_normalize_text) + " " +
        df["cast"].map(_normalize_text) + " " +
        df["country"].map(_normalize_text) + " " +
        df["listed_in"].map(_normalize_text) + " " +
        df["rating"].map(_normalize_text) + " " +
        df["description"].map(_normalize_text)
    ).str.strip()

    df["display_title"] = df["title"].astype(str) + " (" + df["type"].astype(str) + ", " + df["release_year"].astype(str) + ")"

    dup = df["display_title"].duplicated(keep=False)
    if dup.any():
        df.loc[dup, "display_title"] = df.loc[dup].apply(
            lambda r: f"{r['title']} ({r['type']}, {r['release_year']}) — {r.get('show_id','')}",
            axis=1,
        )

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
        st.error(f"Error: {e}")
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
        st.error(f"Error: {e}")
        return pd.DataFrame()

def split_and_count(series: pd.Series, sep: str = ",", top_k: int = 10) -> pd.Series:
    try:
        s = series.fillna("").astype(str).replace({"unknown": "", "Unknown": "", "nan": "", "NaN": ""})
        exploded = s.str.split(sep).explode().astype(str).str.strip()
        exploded = exploded[exploded != ""]
        return exploded.value_counts().head(top_k)
    except Exception:
        return pd.Series(dtype=int)

def display_content_card(r: pd.Series, rank: int):
    try:
        similarity = float(r.get("similarity", 0.0))
        title = _safe_str(r.get("title", ""))
        content_type = _safe_str(r.get("type", ""))
        year = r.get("release_year", "")
        rating = _safe_str(r.get("rating", ""))
        genre = _safe_str(r.get("listed_in", ""))
        description = _safe_str(r.get("description", "No description"))
        director = _safe_str(r.get("director", "Not specified"))
        country = _safe_str(r.get("country", "Not specified"))
        duration = _safe_str(r.get("duration", "Not specified"))

        st.markdown(f"""
        <div class="netflix-card">
            <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:1rem;">
                <div>
                    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                        <div class="similarity-badge">{similarity:.1%}</div>
                        <div style="color:#E50914; font-weight:700; font-size:0.9rem;">#{rank}</div>
                    </div>
                    <h3 style="color:white; margin:0; font-size:1.3rem; font-weight:700;">{title}</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            badge_class = "badge-movie" if content_type == "Movie" else "badge-tv"
            st.markdown(f'<span class="{badge_class}">🎬 {content_type}</span>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span class="badge-year">📅 {year}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="badge-rating">⭐ {rating}</span>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<span class="netflix-badge-item">⏱️ {duration}</span>', unsafe_allow_html=True)

        if genre and genre != "Not specified":
            st.markdown("""
            <div style="margin:1rem 0;">
                <div style="color:#E50914; font-weight:600; font-size:0.9rem; margin-bottom:0.3rem;">🎭 Genre</div>
            </div>
            """, unsafe_allow_html=True)
            
            genres = [g.strip() for g in str(genre).split(",") if g.strip()]
            for g in genres[:3]:
                st.markdown(f'<span class="netflix-badge-item">{g}</span>', unsafe_allow_html=True)

        if description and description != "No description":
            with st.expander("📖 Deskripsi", expanded=False):
                st.markdown(f"""
                <div style="color:rgba(255,255,255,0.8); font-size:0.9rem; line-height:1.6; padding:0.5rem 0;">
                    {description}
                </div>
                """, unsafe_allow_html=True)

        col_details1, col_details2 = st.columns(2)
        with col_details1:
            if director and director != "Not specified":
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:0.8rem; margin:0.5rem 0;">
                    <div style="color:#E50914; font-size:0.85rem; font-weight:600; margin-bottom:0.2rem;">🎬 Director</div>
                    <div style="color:white; font-size:0.9rem;">{director}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_details2:
            if country and country != "Not specified":
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:0.8rem; margin:0.5rem 0;">
                    <div style="color:#E50914; font-size:0.85rem; font-weight:600; margin-bottom:0.2rem;">🌍 Negara</div>
                    <div style="color:white; font-size:0.9rem;">{country}</div>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error: {e}")

def display_metric(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    st.markdown(f"""
    <div class="netflix-metric">
        <div style="font-size: 2rem; margin-bottom: 0.8rem; color: #E50914;">{icon}</div>
        <h3>{value}</h3>
        <div style="font-weight: 600; color: #F5F5F1; font-size: 1rem; margin-bottom: 0.3rem;">{title}</div>
        <div style="font-size: 0.8rem; color: #808080;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def create_dashboard_stats(df: pd.DataFrame):
    stats = {}
    stats["total"] = len(df)
    stats["movies"] = len(df[df["type"] == "Movie"])
    stats["tv_shows"] = len(df[df["type"] == "TV Show"])

    valid_years = df["release_year"][df["release_year"] > 0]
    if len(valid_years) > 0:
        stats["min_year"] = int(valid_years.min())
        stats["max_year"] = int(valid_years.max())
        stats["avg_year"] = int(valid_years.mean())
    else:
        stats["min_year"] = 1990
        stats["max_year"] = datetime.now().year
        stats["avg_year"] = 2010

    return stats

# -----------------------------
# Netflix Header
# -----------------------------
st.markdown("""
<div class="netflix-header">
    <div class="logo-container">
        <h1 class="netflix-logo">NETFLIX</h1>
        <span class="netflix-badge">Recommender</span>
    </div>
    <div style="color: #F5F5F1; font-size: 1.2rem; max-width: 800px; line-height: 1.6;">
        Temukan film dan serial TV yang paling sesuai dengan selera Anda. 
        Sistem kami menganalisis ribuan konten untuk memberikan rekomendasi terbaik.
    </div>
    <div style="display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #F5F5F1;">
            <span style="color: #E50914;">▶</span>
            <span>Rekomendasi Personal</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #F5F5F1;">
            <span style="color: #E50914;">🎯</span>
            <span>Berdasarkan Konten</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #F5F5F1;">
            <span style="color: #E50914;">⚡</span>
            <span>Real-time</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; color: #E50914; margin-bottom: 0.5rem;">🎬</div>
        <h3 style="color: #F5F5F1; margin: 0;">Menu</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Pilih Menu:",
        ["🎬 Cari Konten", "🔍 Berdasarkan Judul", "📊 Statistik", "ℹ️ Tentang"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("""
    <div style="padding: 1rem; background: rgba(229, 9, 20, 0.1); border-radius: 10px; border: 1px solid rgba(229, 9, 20, 0.3);">
        <div style="color: #E50914; font-weight: 600; margin-bottom: 0.5rem;">📁 Dataset</div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Unggah file CSV",
        type=["csv"],
        help="Unggah dataset Netflix (format CSV)",
        label_visibility="collapsed"
    )
    
    use_local = st.checkbox(
        "Gunakan dataset contoh",
        value=True,
        help="Gunakan data contoh jika tidak ada dataset"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
raw_df = None
data_loaded = False

if uploaded is not None:
    try:
        raw_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        if not raw_df.empty:
            data_loaded = True
            st.sidebar.success(f"✅ {len(raw_df)} data dimuat")
    except:
        st.sidebar.error("File tidak valid")
elif use_local:
    try:
        if DEFAULT_DATA_PATH.exists():
            raw_df = pd.read_csv(str(DEFAULT_DATA_PATH))
            data_loaded = True
            st.sidebar.success(f"✅ {len(raw_df)} data contoh dimuat")
        else:
            st.sidebar.info("Buat dataset atau unggah file")
    except:
        st.sidebar.error("Error memuat data")

if not data_loaded:
    st.warning("""
    ⚠️ **Dataset belum dimuat**
    
    Silakan:
    1. Unggah file CSV Netflix, atau
    2. Aktifkan "Gunakan dataset contoh" dan letakkan file `netflix_titles.csv` di folder aplikasi
    """)
    
    # Buat data contoh minimal
    sample_data = {
        'show_id': ['s1', 's2', 's3', 's4', 's5'],
        'type': ['Movie', 'Movie', 'TV Show', 'Movie', 'TV Show'],
        'title': ['The Dark Knight', 'Inception', 'Stranger Things', 'Pulp Fiction', 'Breaking Bad'],
        'director': ['Christopher Nolan', 'Christopher Nolan', 'Duffer Bros', 'Quentin Tarantino', 'Vince Gilligan'],
        'cast': ['Christian Bale', 'Leonardo DiCaprio', 'Millie Bobby Brown', 'John Travolta', 'Bryan Cranston'],
        'country': ['USA', 'USA', 'USA', 'USA', 'USA'],
        'release_year': [2008, 2010, 2016, 1994, 2008],
        'rating': ['PG-13', 'PG-13', 'TV-14', 'R', 'TV-MA'],
        'duration': ['152 min', '148 min', '4 Seasons', '154 min', '5 Seasons'],
        'listed_in': ['Action, Crime', 'Action, Sci-Fi', 'Drama, Horror', 'Crime, Drama', 'Crime, Drama'],
        'description': [
            'Batman vs Joker',
            'Dream within a dream',
            'Mystery in small town',
            'Crime stories',
            'Chemistry teacher turns meth maker'
        ]
    }
    raw_df = pd.DataFrame(sample_data)
    data_loaded = True
    st.sidebar.info("Menggunakan data contoh")

# Process data
if data_loaded:
    df = prepare_data(raw_df)
    vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])
    
    if vectorizer is None or tfidf_matrix is None:
        st.error("Gagal memproses data teks")
        st.stop()
    
    stats = create_dashboard_stats(df)

# -----------------------------
# Page 1: Cari Konten (Pencarian)
# -----------------------------
if page == "🎬 Cari Konten":
    st.markdown("## 🔍 Cari Konten Netflix")
    
    col_search1, col_search2 = st.columns([3, 1])
    
    with col_search1:
        query = st.text_input(
            "Masukkan kata kunci:",
            placeholder="Cari film atau serial TV...",
            help="Cari berdasarkan judul, genre, aktor, atau deskripsi",
            label_visibility="collapsed"
        )
    
    with col_search2:
        search_clicked = st.button("🔍 Cari", type="primary", use_container_width=True)
    
    st.markdown("### ⚙️ Filter Pencarian")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        type_options = ["Semua"] + sorted(df["type"].unique().tolist())
        selected_type = st.selectbox("Tipe Konten:", options=type_options)
    
    with col_filter2:
        year_range = st.slider(
            "Tahun Rilis:",
            min_value=int(df["release_year"].min()),
            max_value=int(df["release_year"].max()),
            value=(int(df["release_year"].min()), int(df["release_year"].max()))
        )
    
    with col_filter3:
        result_count = st.slider("Jumlah Hasil:", 5, 20, 10)
    
    if search_clicked and query:
        with st.spinner("Mencari..."):
            results = recommend_by_query(
                query=query,
                df=df,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                top_n=result_count,
                type_filter=selected_type if selected_type != "Semua" else "All",
                year_min=year_range[0],
                year_max=year_range[1]
            )
        
        if results.empty:
            st.info("Tidak ditemukan hasil untuk pencarian ini.")
        else:
            st.markdown(f"### 📋 **{len(results)} Hasil Ditemukan**")
            
            for i, (_, row) in enumerate(results.iterrows(), 1):
                display_content_card(row, i)
    
    elif not search_clicked:
        st.info("""
        ### 💡 Tips Pencarian:
        - Gunakan kata kunci spesifik seperti "action adventure", "romantic comedy"
        - Cari berdasarkan aktor favorit
        - Gunakan genre seperti "sci-fi", "drama", "thriller"
        - Kombinasikan dengan filter untuk hasil lebih akurat
        """)

# -----------------------------
# Page 2
