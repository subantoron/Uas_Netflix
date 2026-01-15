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
    page_title="🎬 Netflix AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Using local dataset
DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# Custom CSS - Ultra Modern Design
# -----------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * { 
        font-family: 'Inter', sans-serif; 
        -webkit-font-smoothing: antialiased; 
        -moz-osx-font-smoothing: grayscale; 
    }
    
    h1, h2, h3, h4, h5, h6 { 
        font-family: 'Space Grotesk', sans-serif !important; 
        font-weight: 700 !important; 
        letter-spacing: -0.5px; 
    }
    
    .main { 
        padding: 0rem 1rem 1rem 1rem; 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        min-height: 100vh;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-section {
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.15) 0%, 
            rgba(118, 75, 162, 0.15) 50%, 
            rgba(240, 147, 251, 0.1) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4rem 3rem;
        border-radius: 32px;
        margin: 1rem 0 3rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.3);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(240, 147, 251, 0.1) 0%, transparent 50%);
        z-index: -1;
    }
    
    .hero-content {
        max-width: 800px;
    }
    
    .hero-title {
        font-size: 4rem !important;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.35rem;
        line-height: 1.8;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    @keyframes float { 
        0%, 100% { transform: translateY(0); } 
        50% { transform: translateY(-20px); } 
    }
    
    .floating-icon { 
        animation: float 6s ease-in-out infinite; 
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #667eea;
    }
    
    .recommendation-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #667eea, #764ba2);
        border-radius: 4px 0 0 4px;
    }
    
    .recommendation-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.25),
                    0 0 20px rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.1) 0%, 
            rgba(118, 75, 162, 0.1) 100%);
        z-index: -1;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.6rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .badge-movie { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
    }
    
    .badge-year { 
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%) !important; 
    }
    
    .badge-rating { 
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%) !important; 
    }
    
    .badge-genre { 
        background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%) !important; 
    }
    
    .similarity-score {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 50px;
        display: inline-flex;
        align-items: center;
        font-weight: 800;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.5);
        padding: 0.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 16px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        color: #cbd5e1;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .stSlider > div > div > div > div {
        background: white;
    }
    
    .filter-panel {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .search-box {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        color: white;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .search-box:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .content-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .content-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .chart-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fadeInUp {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .glow-effect {
        position: relative;
    }
    
    .glow-effect::after {
        content: '';
        position: absolute;
        inset: -2px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        border-radius: inherit;
        z-index: -1;
        filter: blur(10px);
        opacity: 0.3;
    }
    
    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
    }
    
    .tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(102, 126, 234, 0.2);
        color: #c7d2fe;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .section-title {
        position: relative;
        padding-left: 1rem;
        margin-bottom: 2rem;
    }
    
    .section-title::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: linear-gradient(to bottom, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
    }
    
    .loading-spinner::after {
        content: '';
        width: 40px;
        height: 40px;
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @media (max-width: 768px) {
        .hero-section { padding: 2rem; }
        .hero-title { font-size: 2.5rem !important; }
        .feature-grid { grid-template-columns: 1fr; }
        .content-grid { grid-template-columns: 1fr; }
        .stTabs [data-baseweb="tab"] { padding: 0.8rem 1.5rem; }
    }
</style>
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
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_PATH)
        if len(df) == 0:
            st.error("Dataset kosong!")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
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
        "date_added": "date_added",
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

    return df


@st.cache_resource(show_spinner=False)
def build_model(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None, None
    
    corpus = df["soup"]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus.astype(str).values)
    return vectorizer, tfidf_matrix


def recommend_by_title(
    title_idx: int,
    df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
    same_type: bool = True,
    year_range: tuple = None,
) -> pd.DataFrame:
    try:
        if title_idx < 0 or title_idx >= len(df):
            return pd.DataFrame()

        sims = linear_kernel(tfidf_matrix[title_idx], tfidf_matrix).flatten()
        order = sims.argsort()[::-1]
        order = order[order != title_idx]

        recs = df.iloc[order].copy()
        recs["similarity"] = sims[order]

        if same_type:
            selected_type = df.iloc[title_idx].get("type", "")
            if selected_type:
                recs = recs[recs["type"] == selected_type]

        if year_range:
            year_min, year_max = year_range
            recs = recs[(recs["release_year"] >= year_min) & (recs["release_year"] <= year_max)]

        return recs.head(top_n)
    except Exception as e:
        st.error(f"Error rekomendasi: {e}")
        return pd.DataFrame()


def recommend_by_keywords(
    query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 10,
    type_filter: str = "All",
    year_range: tuple = None,
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

        if year_range:
            year_min, year_max = year_range
            recs = recs[(recs["release_year"] >= year_min) & (recs["release_year"] <= year_max)]

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
        <div class="recommendation-card animate-fadeInUp">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.5rem;">
                <div style="flex:1;">
                    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
                        <div class="badge" style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);">#{rank}</div>
                        <h3 style="margin:0; color:white; font-size:1.5rem; font-weight:700;">{title}</h3>
                    </div>
                    <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1rem;">
                        <span class="badge badge-movie">🎬 {content_type}</span>
                        <span class="badge badge-year">📅 {year}</span>
                        <span class="badge badge-rating">⭐ {rating}</span>
                        <span class="badge">⏱️ {duration}</span>
                    </div>
                </div>
                <div class="similarity-score">{similarity:.1%}</div>
            </div>
            
            <div style="margin:1.2rem 0;">
                <div style="color:#cbd5e1; font-size:0.95rem; line-height:1.6;">
                    {description[:200]}{'...' if len(description) > 200 else ''}
                </div>
            </div>
            
            {genre and genre != "Not specified" and f'''
            <div style="margin:1rem 0;">
                <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;">🎭 Genre</div>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                    {''.join([f'<span class="tag">{g.strip()}</span>' for g in str(genre).split(",")[:5] if g.strip()])}
                </div>
            </div>
            ''' or ''}
            
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.5rem;">
                <div style="background:rgba(255,255,255,0.05); padding:1rem; border-radius:12px;">
                    <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.25rem;">🎬 Director</div>
                    <div style="color:white; font-weight:600;">{director}</div>
                </div>
                <div style="background:rgba(255,255,255,0.05); padding:1rem; border-radius:12px;">
                    <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.25rem;">🌍 Country</div>
                    <div style="color:white; font-weight:600;">{country}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:1rem; color:#667eea;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div style="font-weight:600; color:#cbd5e1; font-size:1rem; margin-bottom:0.25rem;">{title}</div>
            <div style="font-size:0.85rem; color:#94a3b8; font-weight:500;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        stats["min_year"] = 1900
        stats["max_year"] = datetime.now().year
        stats["avg_year"] = 2000
    
    stats["unique_countries"] = len(df["country"].unique())
    stats["unique_genres"] = len(df["listed_in"].unique())
    
    top_genre = split_and_count(df["listed_in"], sep=",", top_k=1)
    stats["top_genre"] = top_genre.index[0] if len(top_genre) > 0 else "N/A"
    
    return stats


def display_feature_grid():
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3 style="color:white; margin-bottom:0.5rem;">AI-Powered</h3>
                <p style="color:#94a3b8; font-size:0.9rem;">Rekomendasi cerdas dengan algoritma machine learning</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3 style="color:white; margin-bottom:0.5rem;">Real-time</h3>
                <p style="color:#94a3b8; font-size:0.9rem;">Hasil instan dengan pemrosesan cepat</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3 style="color:white; margin-bottom:0.5rem;">Personalized</h3>
                <p style="color:#94a3b8; font-size:0.9rem;">Rekomendasi sesuai preferensi Anda</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3 style="color:white; margin-bottom:0.5rem;">Analytics</h3>
                <p style="color:#94a3b8; font-size:0.9rem;">Dashboard analitik lengkap</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Initialize Session State
# -----------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "df" not in st.session_state:
    st.session_state.df = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "stats" not in st.session_state:
    st.session_state.stats = None

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:2rem; padding:2rem;
                    background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
                    border-radius:24px; border:1px solid rgba(255,255,255,0.2);">
            <div style="font-size:3.5rem; color:#667eea; margin-bottom:0.8rem;" class="floating-icon">🎬</div>
            <h2 style="color:white; margin:0; font-size:1.8rem;">Netflix AI</h2>
            <p style="color:rgba(255,255,255,0.9); font-size:0.9rem; margin-top:0.25rem;">Recommender System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["🎯 Recommendations", "📊 Analytics", "🤖 About"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Quick Stats in Sidebar
    if st.session_state.data_loaded and st.session_state.stats:
        stats = st.session_state.stats
        st.markdown("### 📈 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Movies", f"{stats['movies']:,}")
        with col2:
            st.metric("TV Shows", f"{stats['tv_shows']:,}")
        
        st.metric("Total Content", f"{stats['total']:,}")
        st.metric("Year Range", f"{stats['min_year']}-{stats['max_year']}")

# -----------------------------
# Load and Prepare Data
# -----------------------------
if not st.session_state.data_loaded:
    with st.spinner("🚀 Loading Netflix dataset..."):
        raw_df = load_data()
        if not raw_df.empty:
            df = prepare_data(raw_df)
            vectorizer, tfidf_matrix = build_model(df)
            stats = create_dashboard_stats(df)
            
            st.session_state.df = df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.stats = stats
            st.session_state.data_loaded = True
        else:
            st.error("Failed to load data. Please check your dataset.")
            st.stop()

# Get data from session state
df = st.session_state.df
vectorizer = st.session_state.vectorizer
tfidf_matrix = st.session_state.tfidf_matrix
stats = st.session_state.stats

# -----------------------------
# Hero Section
# -----------------------------
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">Discover Your Next<br>Favorite Movie or Show</h1>
            <p class="hero-subtitle">
                Powered by advanced AI algorithms, our recommendation system analyzes thousands of titles 
                to find the perfect match for your taste. Whether you're into action, romance, or documentaries, 
                we've got you covered.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature Grid
display_feature_grid()

# -----------------------------
# Page: Recommendations
# -----------------------------
if page == "🎯 Recommendations":
    tabs = st.tabs(["🎬 By Title", "🔍 By Keywords", "⭐ Popular"])
    
    # Tab 1: By Title
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<div class="section-title"><h2 style="color:white; margin:0;">Find Similar Content</h2></div>', unsafe_allow_html=True)
            
            # Filter Panel
            with st.container():
                st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    content_type = st.selectbox(
                        "Content Type",
                        ["All", "Movie", "TV Show"],
                        index=0,
                        key="title_content_type"
                    )
                    
                with col_f2:
                    top_n = st.slider(
                        "Number of Recommendations",
                        min_value=5,
                        max_value=20,
                        value=10,
                        key="title_top_n"
                    )
                
                # Year Range Filter
                year_min, year_max = st.slider(
                    "Release Year Range",
                    min_value=stats["min_year"],
                    max_value=stats["max_year"],
                    value=(stats["min_year"], stats["max_year"]),
                    key="title_year_range"
                )
                
                # Title Selection
                if content_type == "All":
                    title_options = df["display_title"].tolist()
                else:
                    title_options = df[df["type"] == content_type]["display_title"].tolist()
                
                selected_title = st.selectbox(
                    "Select a title",
                    title_options,
                    key="title_selector"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Get Recommendations Button
            if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                if selected_title:
                    idx = df[df["display_title"] == selected_title].index[0]
                    selected_item = df.iloc[idx]
                    
                    # Display Selected Item
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); 
                                    padding:2rem; border-radius:20px; margin:2rem 0;">
                            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                                <div style="font-size:2rem;">🎬</div>
                                <div>
                                    <h3 style="color:white; margin:0;">{selected_item['title']}</h3>
                                    <p style="color:#94a3b8; margin:0;">{selected_item['type']} • {selected_item['release_year']}</p>
                                </div>
                            </div>
                            <p style="color:#cbd5e1;">{selected_item['description'][:200]}...</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    # Get Recommendations
                    with st.spinner("🔍 Finding similar content..."):
                        recs = recommend_by_title(
                            title_idx=idx,
                            df=df,
                            tfidf_matrix=tfidf_matrix,
                            top_n=top_n,
                            same_type=(content_type != "All"),
                            year_range=(year_min, year_max)
                        )
                    
                    # Display Recommendations
                    if not recs.empty:
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin:3rem 0 2rem 0;">
                                <h2 style="color:white; margin-bottom:1rem;">🎯 Recommended for You</h2>
                                <p style="color:#94a3b8;">Based on "{selected_item['title']}"</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                        for i, (_, row) in enumerate(recs.iterrows(), 1):
                            display_recommendation_card(row, i)
                    else:
                        st.warning("No recommendations found. Try adjusting your filters.")
    
    # Tab 2: By Keywords
    with tabs[1]:
        st.markdown('<div class="section-title"><h2 style="color:white; margin:0;">Search by Keywords</h2></div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
            
            col_k1, col_k2 = st.columns([3, 1])
            with col_k1:
                query = st.text_input(
                    "Enter keywords",
                    placeholder="e.g., action adventure, romantic comedy, sci-fi",
                    key="keyword_query"
                )
            
            with col_k2:
                top_n_k = st.slider(
                    "Results",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key="keyword_top_n"
                )
            
            col_k3, col_k4 = st.columns(2)
            with col_k3:
                search_type = st.selectbox(
                    "Content Type",
                    ["All", "Movie", "TV Show"],
                    index=0,
                    key="search_content_type"
                )
            
            with col_k4:
                search_year_min, search_year_max = st.slider(
                    "Release Year",
                    min_value=stats["min_year"],
                    max_value=stats["max_year"],
                    value=(stats["min_year"], stats["max_year"]),
                    key="search_year_range"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 Search Recommendations", type="primary", use_container_width=True):
            if query:
                with st.spinner("🔍 Analyzing keywords..."):
                    recs = recommend_by_keywords(
                        query=query,
                        df=df,
                        vectorizer=vectorizer,
                        tfidf_matrix=tfidf_matrix,
                        top_n=top_n_k,
                        type_filter=search_type,
                        year_range=(search_year_min, search_year_max)
                    )
                
                if not recs.empty:
                    st.markdown(
                        f"""
                        <div style="text-align:center; margin:3rem 0 2rem 0;">
                            <h2 style="color:white; margin-bottom:1rem;">🔍 Search Results</h2>
                            <p style="color:#94a3b8;">Found {len(recs)} results for "{query}"</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    for i, (_, row) in enumerate(recs.iterrows(), 1):
                        display_recommendation_card(row, i)
                else:
                    st.info("No results found. Try different keywords or adjust filters.")

# -----------------------------
# Page: Analytics
# -----------------------------
elif page == "📊 Analytics":
    st.markdown('<div class="section-title"><h2 style="color:white; margin:0;">Data Analytics Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_card("Total Content", f"{stats['total']:,}", "Movies & TV Shows", "📊")
    with col2:
        display_metric_card("Movies", f"{stats['movies']:,}", f"{stats['movies']/stats['total']*100:.1f}%", "🎥")
    with col3:
        display_metric_card("TV Shows", f"{stats['tv_shows']:,}", f"{stats['tv_shows']/stats['total']*100:.1f}%", "📺")
    with col4:
        display_metric_card("Year Range", f"{stats['min_year']}-{stats['max_year']}", "Release years", "📅")
    
    st.divider()
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🎭 Content Type Distribution")
        type_counts = df["type"].value_counts()
        st.bar_chart(type_counts)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🌍 Top Countries")
        top_countries = split_and_count(df["country"], sep=",", top_k=10)
        st.bar_chart(top_countries)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🎬 Top Genres")
        top_genres = split_and_count(df["listed_in"], sep=",", top_k=10)
        st.bar_chart(top_genres)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📅 Release Year Trends")
        year_counts = df["release_year"][df["release_year"] > 0].value_counts().sort_index()
        st.line_chart(year_counts.tail(20))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Data Preview
    st.markdown("### 🔍 Data Preview")
    preview_size = st.slider("Sample Size", 5, 100, 20)
    
    display_cols = ["title", "type", "release_year", "rating", "duration", "listed_in"]
    available_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(
        df[available_cols].head(preview_size),
        use_container_width=True,
        hide_index=True,
        column_config={
            "title": "🎬 Title",
            "type": "🎭 Type",
            "release_year": "📅 Year",
            "rating": "⭐ Rating",
            "duration": "⏱️ Duration",
            "listed_in": "🎨 Genre"
        }
    )

# -----------------------------
# Page: About
# -----------------------------
elif page == "🤖 About":
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.markdown('<div class="section-title"><h2 style="color:white; margin:0;">About the System</h2></div>', unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="glass-panel" style="margin-bottom:2rem;">
                <h3 style="color:#667eea; margin-bottom:1rem;">🤖 How It Works</h3>
                <p style="color:#cbd5e1; line-height:1.8; margin-bottom:1.5rem;">
                    This recommendation system uses <strong>Content-Based Filtering</strong> with 
                    <strong>TF-IDF Vectorization</strong> and <strong>Cosine Similarity</strong> 
                    to analyze and compare Netflix titles based on their metadata.
                </p>
                
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:1rem; margin-top:1.5rem;">
                    <div style="background:rgba(102,126,234,0.1); padding:1rem; border-radius:12px;">
                        <div style="color:#667eea; font-weight:600; margin-bottom:0.5rem;">1. Data Processing</div>
                        <div style="color:#94a3b8; font-size:0.9rem;">Clean and prepare the dataset</div>
                    </div>
                    <div style="background:rgba(118,75,162,0.1); padding:1rem; border-radius:12px;">
                        <div style="color:#764ba2; font-weight:600; margin-bottom:0.5rem;">2. Feature Extraction</div>
                        <div style="color:#94a3b8; font-size:0.9rem;">Convert text to numerical vectors</div>
                    </div>
                    <div style="background:rgba(240,147,251,0.1); padding:1rem; border-radius:12px;">
                        <div style="color:#f093fb; font-weight:600; margin-bottom:0.5rem;">3. Similarity Calculation</div>
                        <div style="color:#94a3b8; font-size:0.9rem;">Compute cosine similarity</div>
                    </div>
                    <div style="background:rgba(0,176,155,0.1); padding:1rem; border-radius:12px;">
                        <div style="color:#00b09b; font-weight:600; margin-bottom:0.5rem;">4. Recommendations</div>
                        <div style="color:#94a3b8; font-size:0.9rem;">Generate personalized suggestions</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_about2:
        st.markdown("## 🛠️ Technical Info")
        
        display_metric_card("Dataset Size", f"{len(df):,}", "titles", "📁")
        display_metric_card("Features", f"{tfidf_matrix.shape[1]:,}", "TF-IDF dimensions", "🔢")
        display_metric_card("Algorithm", "TF-IDF", "Cosine Similarity", "⚡")
        display_metric_card("Response Time", "< 500ms", "average", "⏱️")
    
    st.divider()
    
    # Footer
    st.markdown(
        """
        <div style="text-align:center; padding:3rem; border-radius:32px; 
                    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
                    border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size:3rem;" class="floating-icon">🎬</div>
            <h3 style="color:#667eea; margin:1rem 0;">Netflix AI Recommender</h3>
            <p style="color:#94a3b8; margin:0;">Powered by Streamlit + Scikit-learn</p>
            <div style="display:flex; justify-content:center; gap:1rem; margin-top:1.5rem;">
                <span class="badge">🎯 Content-Based Filtering</span>
                <span class="badge">⚡ Real-time Processing</span>
                <span class="badge">📊 Data Analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
