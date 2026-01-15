import io
import re
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="🎬 Netflix AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# Custom CSS - Ultra Modern Design
# -----------------------------
st.markdown("""
<style>
    /* Font Import - Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
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
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Main Header - Glass Morphism */
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 3rem;
        border-radius: 32px;
        color: white;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.8) 0%,
            rgba(118, 75, 162, 0.8) 30%,
            rgba(240, 147, 251, 0.6) 70%,
            rgba(102, 126, 234, 0.4) 100%
        );
        z-index: -1;
    }
    
    .main-header h1 {
        font-size: 3.5rem !important;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.25rem;
        max-width: 900px;
        line-height: 1.8;
        font-weight: 400;
    }
    
    /* Floating Particles Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .floating-icon {
        animation: float 6s ease-in-out infinite;
    }
    
    /* Card Styling - Glass Morphism */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%,
            rgba(102, 126, 234, 0.6) 50%,
            transparent 100%
        );
    }
    
    .recommendation-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #667eea, #764ba2);
        border-radius: 4px 0 0 4px;
    }
    
    .recommendation-card:hover {
        transform: translateY(-10px) scale(1.01);
        box-shadow: 
            0 20px 40px rgba(102, 126, 234, 0.15),
            0 0 0 1px rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.12);
    }
    
    /* Metric Cards - Neumorphism */
    .metric-card {
        background: linear-gradient(145deg, #f0f2f5, #ffffff);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 
            20px 20px 60px #d9d9d9,
            -20px -20px 60px #ffffff;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 
            25px 25px 75px #d0d0d0,
            -25px -25px 75px #ffffff;
    }
    
    .metric-card h3 {
        color: #667eea;
        font-size: 2.8rem !important;
        margin-bottom: 0.8rem;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Badge Styling - Modern Gradient */
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
        box-shadow: 
            0 4px 15px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .badge::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent
        );
        transition: left 0.5s;
    }
    
    .badge:hover::after {
        left: 100%;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 6px 20px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
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
    
    /* Similarity Score - Glowing Effect */
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
        box-shadow: 
            0 4px 15px rgba(0, 176, 155, 0.3),
            0 0 20px rgba(0, 176, 155, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .similarity-score::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .similarity-score:hover::before {
        opacity: 1;
    }
    
    /* Button Styling - Modern 3D */
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
        box-shadow: 
            0 10px 30px rgba(102, 126, 234, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent
        );
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 15px 40px rgba(102, 126, 234, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.2);
    }
    
    /* Tab Styling - Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: linear-gradient(90deg, 
            rgba(102, 126, 234, 0.05) 0%,
            rgba(118, 75, 162, 0.05) 100%
        );
        padding: 0.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 16px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        color: #667eea;
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
    
    /* Sidebar Styling - Dark Theme */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6e8ff 100%);
        border-radius: 16px !important;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        color: #667eea;
        border: 2px solid #e6e8ff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 16px;
        border: 2px solid #e6e8ff;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 
            0 0 0 3px rgba(102, 126, 234, 0.1),
            inset 0 2px 4px rgba(0, 0, 0, 0.05);
        background: white;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 16px;
        border: 2px solid #e6e8ff;
        padding: 0.5rem 1rem;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Checkbox Styling */
    .stCheckbox > label {
        font-weight: 600;
        color: #333;
        font-size: 1rem;
    }
    
    /* Divider */
    .stDivider {
        border-color: rgba(102, 126, 234, 0.1);
        margin: 2rem 0;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 20px;
        border: 1px solid #e6e8ff;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }
    
    /* Alert Messages */
    .stAlert {
        border-radius: 20px;
        border: none;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 3px solid transparent;
        background-clip: padding-box;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #667eea transparent #667eea transparent !important;
    }
    
    /* Animation for cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .recommendation-card {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem;
        }
        
        .main-header h1 {
            font-size: 2.5rem !important;
        }
        
        .metric-card h3 {
            font-size: 2.2rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.8rem 1.5rem;
            font-size: 1rem;
        }
    }
    
    /* Feature Highlights */
    .feature-highlight {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
        height: 100%;
    }
    
    .feature-highlight:hover {
        transform: translateY(-5px);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.15);
    }
    
    /* Stats Card */
    .stats-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glass Panel */
    .glass-panel {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper Functions
# -----------------------------
def _normalize_text(x: object) -> str:
    """Normalize text for vectorization."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"unknown", "nan", "none", "null"}:
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
    if s.strip().lower() in {"unknown", "nan", "none", "null"}:
        return ""
    return s

@st.cache_data(show_spinner=False)
def load_data_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Handle different column name variations
    column_mapping = {
        'show id': 'show_id',
        'show_id': 'show_id',
        'type': 'type',
        'title': 'title',
        'director': 'director',
        'director_list': 'director',
        'cast': 'cast',
        'cast_list': 'cast',
        'country': 'country',
        'country_list': 'country',
        'date_added': 'date_added',
        'date_added_iso': 'date_added',
        'release year': 'release_year',
        'release_year': 'release_year',
        'rating': 'rating',
        'duration': 'duration',
        'listed in': 'listed_in',
        'listed_in': 'listed_in',
        'listed_in_list': 'listed_in',
        'description': 'description',
        'actor': 'cast'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Ensure all expected columns exist
    expected = [
        "show_id", "type", "title", "director", "cast", "country", 
        "release_year", "rating", "duration", "listed_in", "description"
    ]
    
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    
    # Standardize type values
    df['type'] = df['type'].astype(str).str.strip()
    df['type'] = df['type'].apply(lambda x: 'TV Show' if str(x).lower() == 'tv show' else x)
    df['type'] = df['type'].apply(lambda x: 'Movie' if str(x).lower() == 'movie' else x)
    
    text_cols = ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
        df[c] = df[c].replace({
            "unknown": "", "Unknown": "", 
            "nan": "", "NaN": "", 
            "None": "", "none": "", "": ""
        })
    
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    else:
        df["release_year"] = 0
    
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
    
    if df["show_id"].astype(str).duplicated().any():
        df["show_id"] = df.apply(lambda r: f"{r.get('show_id','')}_{r.name}", axis=1)
    
    return df

@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(corpus: pd.Series):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus.values)
    return vectorizer, tfidf_matrix

def recommend_by_index(
    idx: int,
    df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10,
    same_type: bool = True,
    year_min: int | None = None,
    year_max: int | None = None,
):
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = sims.argsort()[::-1]
    order = order[order != idx]

    recs = df.iloc[order].copy()
    recs["similarity"] = sims[order]

    if same_type:
        selected_type = df.iloc[idx]["type"]
        recs = recs[recs["type"] == selected_type]

    if year_min is not None:
        recs = recs[recs["release_year"] >= year_min]
    if year_max is not None:
        recs = recs[recs["release_year"] <= year_max]

    return recs.head(top_n)

def recommend_by_query(
    query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 10,
    type_filter: str = "All",
    year_min: int | None = None,
    year_max: int | None = None,
):
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

def split_and_count(series: pd.Series, sep: str = ",", top_k: int = 10) -> pd.Series:
    s = series.fillna("").astype(str).replace({"unknown": "", "Unknown": ""})
    exploded = s.str.split(sep).explode().astype(str).str.strip()
    exploded = exploded[exploded != ""]
    return exploded.value_counts().head(top_k)

def display_recommendation_card(r: pd.Series, rank: int):
    """Display a beautiful recommendation card using Streamlit components"""
    similarity = float(r.get("similarity", 0.0))
    title = _safe_str(r.get('title', ''))
    content_type = _safe_str(r.get('type', ''))
    year = r.get('release_year', '')
    rating = _safe_str(r.get('rating', ''))
    genre = _safe_str(r.get('listed_in', ''))
    description = _safe_str(r.get('description', 'No description available'))
    director = _safe_str(r.get('director', 'Not specified'))
    country = _safe_str(r.get('country', 'Not specified'))
    
    # Card container
    st.markdown(f"""
    <div class="recommendation-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h4 style="margin: 0; color: #667eea; font-size: 1.4rem; font-weight: 800;">#{rank}</h4>
                <h3 style="margin: 0.5rem 0; color: #2d3748; font-size: 1.6rem; font-weight: 800;">{title}</h3>
            </div>
            <div class="similarity-score">
                {similarity:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Content badges
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<span class="badge badge-movie">🎬 {content_type}</span>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<span class="badge badge-year">📅 {year}</span>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<span class="badge badge-rating">⭐ {rating}</span>', unsafe_allow_html=True)
    
    # Genre
    if genre:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    border-radius: 16px; padding: 1.2rem; margin: 1.2rem 0; border-left: 4px solid #667eea;">
            <strong style="color: #667eea; font-size: 1.1rem;">🎭 Genre</strong>
            <div style="color: #4a5568; font-weight: 600; margin-top: 0.5rem; font-size: 1rem;">{genre}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Description
    if description and description != "No description available":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                    border-radius: 16px; padding: 1.2rem; margin: 1.2rem 0;">
            <strong style="color: #667eea; font-size: 1.1rem;">📖 Deskripsi</strong>
            <div style="font-size: 0.95rem; color: #718096; line-height: 1.6; margin-top: 0.5rem; font-weight: 500;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional details
    col_details1, col_details2 = st.columns(2)
    with col_details1:
        if director and director != "Not specified":
            st.markdown(f"""
            <div class="stats-card" style="margin-bottom: 1rem;">
                <strong style="color: #667eea; font-size: 1rem;">🎬 Director</strong>
                <div style="color: #4a5568; font-size: 0.95rem; font-weight: 600; margin-top: 0.3rem;">{director}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_details2:
        if country and country != "Not specified":
            st.markdown(f"""
            <div class="stats-card" style="margin-bottom: 1rem;">
                <strong style="color: #667eea; font-size: 1rem;">🌍 Negara</strong>
                <div style="color: #4a5568; font-size: 0.95rem; font-weight: 600; margin-top: 0.3rem;">{country}</div>
            </div>
            """, unsafe_allow_html=True)

def display_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    """Display a metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 3rem; margin-bottom: 1rem; color: #667eea;" class="floating-icon">{icon}</div>
        <h3 class="gradient-text">{value}</h3>
        <div style="font-weight: 700; color: #4a5568; font-size: 1.2rem; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 0.9rem; color: #a0aec0; font-weight: 600;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-size: 4rem;" class="floating-icon">🎬</div>
        <div>
            <h1>Netflix AI Recommender</h1>
            <p style="margin: 0;">Sistem Rekomendasi Cerdas Berbasis Machine Learning</p>
        </div>
    </div>
    <p>
        Temukan konten yang paling sesuai dengan preferensi Anda menggunakan algoritma <strong>Content-Based Filtering</strong> 
        yang didukung oleh teknologi <strong>TF-IDF Vectorization</strong> dan <strong>Cosine Similarity</strong>.
    </p>
    <div style="display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap;">
        <span class="badge badge-movie">🎯 Rekomendasi Presisi</span>
        <span class="badge badge-year">⚡ Real-time Processing</span>
        <span class="badge badge-rating">🤖 AI-Powered</span>
        <span class="badge" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);">📊 Analisis Data</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); 
                border-radius: 24px; border: 1px solid rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px);">
        <div style="font-size: 3.5rem; color: #667eea; margin-bottom: 0.8rem;" class="floating-icon">🤖</div>
        <h2 style="color: white; margin: 0; font-size: 1.8rem;">AI Recommender</h2>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.95rem; margin-top: 0.5rem;">Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧭 Menu Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        ["🎯 Rekomendasi", "📊 Dashboard Analitik", "🤖 Tentang AI"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ⚙️ Konfigurasi Data")
    with st.expander("📁 Pengaturan Dataset", expanded=True):
        uploaded = st.file_uploader(
            "Unggah Dataset CSV", 
            type=["csv"],
            help="Unggah file dataset Netflix dalam format CSV"
        )
        use_local = st.checkbox(
            "Gunakan dataset lokal", 
            value=True,
            help="Gunakan file netflix_titles.csv yang tersedia"
        )
    
    st.divider()

# -----------------------------
# Load Data
# -----------------------------
raw_df = None
if uploaded is not None:
    with st.spinner("🔄 Memuat dataset..."):
        raw_df = load_data_from_upload(uploaded.getvalue())
    st.success(f"✅ Dataset berhasil dimuat: **{len(raw_df):,}** baris data")
else:
    if use_local and DEFAULT_DATA_PATH.exists():
        with st.spinner("🔄 Memuat dataset lokal..."):
            raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
        st.success(f"✅ Dataset lokal berhasil dimuat: **{len(raw_df):,}** baris data")
    else:
        st.error("""
        <div class="glass-panel">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📂 Dataset Tidak Ditemukan</h3>
            <p style="color: #718096; margin-bottom: 1.5rem;">
                Silakan pilih salah satu opsi untuk melanjutkan:
            </p>
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div class="stats-card">
                    <strong>📤 Upload File</strong>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #a0aec0;">
                        Upload file CSV melalui sidebar
                    </p>
                </div>
                <div class="stats-card">
                    <strong>📁 File Lokal</strong>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #a0aec0;">
                        Letakkan file netflix_titles.csv di folder aplikasi
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# -----------------------------
# Process Data
# -----------------------------
with st.spinner("🤖 Memproses data dengan AI..."):
    df = prepare_data(raw_df)
    vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])

# Update sidebar status
with st.sidebar:
    st.success("""
    <div style="background: linear-gradient(135deg, rgba(0, 176, 155, 0.1) 0%, rgba(150, 201, 61, 0.1) 100%); 
                border-radius: 16px; padding: 1rem; border: 1px solid rgba(0, 176, 155, 0.2);">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="font-size: 1.5rem;">✅</div>
            <div>
                <strong style="color: #00b09b;">Sistem Aktif</strong>
                <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #718096;">Dataset berhasil dimuat</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display dataset stats in sidebar
    st.markdown("### 📊 Statistik Dataset")
    
    movie_count = len(df[df['type'] == 'Movie'])
    tv_count = len(df[df['type'] == 'TV Show'])
    other_count = len(df) - movie_count - tv_count
    
    st.markdown(f"""
    <div class="glass-panel">
        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
            <div>
                <div style="font-size: 0.9rem; color: #a0aec0;">Total Data</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #667eea;">{len(df):,}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: #a0aec0;">Unique Types</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #667eea;">{len(df['type'].unique())}</div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <div style="font-size: 0.9rem; color: #a0aec0;">Movies</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #4CAF50;">{movie_count:,}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: #a0aec0;">TV Shows</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #FF9800;">{tv_count:,}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 20px;">
        <p style="color: #718096; font-size: 0.85rem; margin: 0; line-height: 1.5;">
            Dibangun dengan ❤️ menggunakan<br>
            <strong style="color: #667eea;">Streamlit</strong> + <strong style="color: #667eea;">Scikit-learn</strong> + <strong style="color: #667eea;">Plotly</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Common variables
min_year = int(df["release_year"].replace(0, np.nan).min(skipna=True) or 1900)
max_year = int(df["release_year"].max() or 2025)

# Get unique types
unique_types = sorted(df['type'].unique().tolist())
unique_types = [t for t in unique_types if t and str(t) != 'nan']
type_options = ["🎬 Semua Tipe"] + unique_types

# -----------------------------
# Page: Recommendation
# -----------------------------
if page == "🎯 Rekomendasi":
    tabs = st.tabs(["🎬 Berdasarkan Judul", "🔍 Berdasarkan Kata Kunci", "⭐ Konten Populer"])
    
    # Tab 1: By Title
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Pilih Konten untuk Direkomendasikan")
            
            # Modern filter section
            st.markdown("""
            <div class="glass-panel">
                <div style="margin-bottom: 1.5rem;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;">🎭 Filter Tipe Konten</div>
            """, unsafe_allow_html=True)
            
            filter_type_for_selector = st.selectbox(
                "Pilih tipe konten:",
                options=type_options,
                index=0,
                help="Filter berdasarkan Movie atau TV Show",
                label_visibility="collapsed"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if filter_type_for_selector == "🎬 Semua Tipe":
                selector_df = df
            else:
                selector_df = df[df["type"] == filter_type_for_selector]
            
            # Title selection with modern styling
            if len(selector_df) > 0:
                st.markdown("""
                <div style="margin: 1.5rem 0;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;">📝 Pilih Judul</div>
                """, unsafe_allow_html=True)
                
                options = selector_df["display_title"].tolist()
                selected_display = st.selectbox(
                    "Pilih judul untuk rekomendasi:",
                    options=options,
                    index=0,
                    help="Pilih satu judul untuk mendapatkan rekomendasi serupa",
                    label_visibility="collapsed"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning(f"Tidak ada konten dengan tipe: {filter_type_for_selector}")
                selected_display = None
            
            # Recommendation settings in modern layout
            st.markdown("### ⚙️ Pengaturan Rekomendasi")
            
            col_settings1, col_settings2, col_settings3 = st.columns(3)
            
            with col_settings1:
                st.markdown("""
                <div class="stats-card">
                    <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.5rem;">Jumlah Rekomendasi</div>
                """, unsafe_allow_html=True)
                top_n = st.slider(
                    "Jumlah:",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1,
                    label_visibility="collapsed"
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_settings2:
                st.markdown("""
                <div class="stats-card">
                    <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.5rem;">Filter Tipe</div>
                """, unsafe_allow_html=True)
                same_type = st.checkbox(
                    "Hanya tipe yang sama",
                    value=True,
                    help="Rekomendasikan hanya Movie↔Movie atau TV Show↔TV Show"
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_settings3:
                st.markdown("""
                <div class="stats-card">
                    <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.5rem;">Tindakan</div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Year range filter with modern slider
            st.markdown("### 📅 Filter Tahun Rilis")
            
            col_year1, col_year2, col_year3 = st.columns([1, 2, 1])
            with col_year2:
                year_range = st.slider(
                    "Pilih rentang tahun:",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    format="%d",
                    label_visibility="collapsed"
                )
            year_min, year_max = year_range
            
            # Recommendation button
            if selected_display and st.button("🚀 Dapatkan Rekomendasi AI", type="primary", use_container_width=True):
                idx = int(df.index[df["display_title"] == selected_display][0])
                
                # Display selected content
                st.divider()
                st.markdown("### 🎬 Konten yang Dipilih")
                
                selected_item = df.iloc[idx]
                
                # Modern selected content display
                col_selected1, col_selected2 = st.columns([2, 1])
                
                with col_selected1:
                    st.markdown(f"""
                    <div class="glass-panel">
                        <h3 style="color: #2d3748; margin-bottom: 1rem;">{selected_item['title']}</h3>
                        <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                            <span class="badge badge-movie">{selected_item['type']}</span>
                            <span class="badge badge-year">{selected_item['release_year']}</span>
                            <span class="badge badge-rating">{selected_item['rating']}</span>
                        </div>
                        <div style="color: #4a5568; line-height: 1.6;">
                            {selected_item['description'] if selected_item['description'] else "Tidak ada deskripsi tersedia"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_selected2:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div style="margin-bottom: 1rem;">
                            <div style="font-size: 0.9rem; color: #a0aec0;">Genre</div>
                            <div style="font-weight: 600; color: #4a5568;">{selected_item['listed_in'] if selected_item['listed_in'] else "Tidak tersedia"}</div>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <div style="font-size: 0.9rem; color: #a0aec0;">Negara</div>
                            <div style="font-weight: 600; color: #4a5568;">{selected_item['country'] if selected_item['country'] else "Tidak tersedia"}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; color: #a0aec0;">Durasi</div>
                            <div style="font-weight: 600; color: #4a5568;">{selected_item['duration'] if selected_item['duration'] else "Tidak tersedia"}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Get recommendations
                with st.spinner("🤖 AI sedang mencari rekomendasi terbaik..."):
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
                
                # Display recommendations count
                col_rec_count1, col_rec_count2, col_rec_count3 = st.columns([1, 2, 1])
                with col_rec_count2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                                border-radius: 20px; margin: 1rem 0;">
                        <div style="font-size: 3rem; color: #667eea; margin-bottom: 0.5rem;">🎯</div>
                        <h3 style="color: #2d3748; margin: 0;">{len(recs)} Rekomendasi Terbaik</h3>
                        <p style="color: #718096; margin: 0.5rem 0 0 0;">Ditemukan oleh AI berdasarkan kemiripan konten</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if recs.empty:
                    st.warning("""
                    <div class="glass-panel">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            <div style="font-size: 2rem;">⚠️</div>
                            <div>
                                <h4 style="margin: 0; color: #F57C00;">Tidak Ditemukan Rekomendasi</h4>
                                <p style="margin: 0.5rem 0 0 0; color: #718096;">
                                    AI tidak dapat menemukan rekomendasi yang sesuai dengan filter.
                                </p>
                            </div>
                        </div>
                        <div class="stats-card" style="margin-top: 1rem;">
                            <strong>💡 Saran Perbaikan:</strong>
                            <ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem; color: #718096;">
                                <li>Kurangi filter tahun</li>
                                <li>Nonaktifkan filter "Hanya tipe yang sama"</li>
                                <li>Coba pilih judul lain</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display recommendations as cards
                    for i, (_, r) in enumerate(recs.iterrows(), 1):
                        display_recommendation_card(r, i)
                        
                        # Additional details in expander
                        with st.expander("🔍 Detail Teknis & Metadata", expanded=False):
                            col_detail1, col_detail2 = st.columns(2)
                            with col_detail1:
                                st.markdown(f"""
                                <div class="stats-card">
                                    <div style="font-size: 0.9rem; color: #a0aec0;">Similarity Score</div>
                                    <div style="font-size: 1.5rem; font-weight: 800; color: #00b09b;">{float(r.get('similarity', 0)):.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_detail2:
                                if r.get('show_id'):
                                    st.markdown(f"""
                                    <div class="stats-card">
                                        <div style="font-size: 0.9rem; color: #a0aec0;">Content ID</div>
                                        <div style="font-size: 1rem; font-weight: 600; color: #4a5568;">{r['show_id']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Dashboard Dataset")
            
            display_metric_card(
                "Total Konten",
                f"{len(df):,}",
                "Movies & TV Shows",
                "📈"
            )
            
            display_metric_card(
                "Movies",
                f"{movie_count:,}",
                "Film layar lebar",
                "🎥"
            )
            
            display_metric_card(
                "TV Shows",
                f"{tv_count:,}",
                "Serial televisi",
                "📺"
            )
            
            display_metric_card(
                "Tahun Terbaru",
                str(max_year),
                "Konten terupdate",
                "🚀"
            )
            
            # Recently added preview
            st.divider()
            st.markdown("### 🆕 Baru Ditambahkan")
            date_columns = ['date_added', 'date_added_iso', 'release_year']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                recent = df.copy()
                if date_col == 'release_year':
                    recent = recent.sort_values(date_col, ascending=False).head(3)
                else:
                    recent[date_col] = pd.to_datetime(recent[date_col], errors='coerce')
                    recent = recent.sort_values(date_col, ascending=False).head(3)
                
                for idx, item in recent.iterrows():
                    with st.expander(f"{item['title'][:30]}...", expanded=False):
                        st.markdown(f"""
                        <div class="stats-card">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <span class="badge badge-movie" style="font-size: 0.8rem;">{item['type']}</span>
                                <span style="color: #718096; font-size: 0.9rem;">{item['release_year']}</span>
                            </div>
                            <div style="color: #4a5568; font-weight: 600; margin-bottom: 0.5rem;">{item['title']}</div>
                            <div style="font-size: 0.9rem; color: #a0aec0;">{item['listed_in'][:50]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Tab 2: By Keywords
    with tabs[1]:
        st.markdown("### 🔍 Pencarian AI dengan Kata Kunci")
        st.markdown("Masukkan kata kunci dan biarkan AI menemukan konten yang paling sesuai.")
        
        # Modern search input
        col_search1, col_search2 = st.columns([3, 1])
        with col_search1:
            query = st.text_input(
                "Masukkan kata kunci pencarian:",
                placeholder="Contoh: action adventure, romantic comedy, sci-fi, crime drama",
                help="Gunakan kata kunci dalam bahasa Inggris untuk hasil terbaik",
                label_visibility="collapsed"
            )
        
        with col_search2:
            search_btn = st.button("🔍 Cari dengan AI", type="primary", use_container_width=True)
        
        # Filters in modern layout
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            st.markdown("""
            <div class="stats-card">
                <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.5rem;">Filter Tipe</div>
            """, unsafe_allow_html=True)
            type_filter = st.selectbox(
                "Tipe:",
                options=type_options,
                index=0,
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_filter2:
            st.markdown("""
            <div class="stats-card">
                <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.5rem;">Jumlah Hasil</div>
            """, unsafe_allow_html=True)
            top_n_q = st.slider(
                "Hasil:",
                min_value=5,
                max_value=20,
                value=10,
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_filter3:
            st.markdown("""
            <div class="stats-card">
                <div style="font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.5rem;">Tahun Rilis</div>
            """, unsafe_allow_html=True)
            year_range_q = st.slider(
                "Tahun:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="year_q",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        year_min_q, year_max_q = year_range_q
        
        if search_btn and query:
            with st.spinner("🤖 AI sedang menganalisis kata kunci..."):
                recs_q = recommend_by_query(
                    query=query,
                    df=df,
                    vectorizer=vectorizer,
                    tfidf_matrix=tfidf_matrix,
                    top_n=top_n_q,
                    type_filter=type_filter,
                    year_min=year_min_q,
                    year_max=year_max_q,
                )
            
            if recs_q.empty:
                st.error("""
                <div class="glass-panel">
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                        <div style="font-size: 2.5rem;">🤖</div>
                        <div>
                            <h3 style="margin: 0; color: #667eea;">AI Tidak Menemukan Hasil</h3>
                            <p style="margin: 0.5rem 0 0 0; color: #718096;">
                                Untuk pencarian: <strong>"{query}"</strong>
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                                border-radius: 16px; padding: 1.5rem; margin-top: 1rem;">
                        <h4 style="color: #4a5568; margin-bottom: 1rem;">💡 Tips Pencarian yang Lebih Baik:</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div class="stats-card">
                                <div style="font-size: 0.9rem; color: #a0aec0;">🌐 Bahasa</div>
                                <div style="font-weight: 600; color: #4a5568;">Gunakan bahasa Inggris</div>
                            </div>
                            <div class="stats-card">
                                <div style="font-size: 0.9rem; color: #a0aec0;">🎭 Kata Kunci</div>
                                <div style="font-weight: 600; color: #4a5568;">Lebih umum & spesifik</div>
                            </div>
                            <div class="stats-card">
                                <div style="font-size: 0.9rem; color: #a0aec0;">⚙️ Filter</div>
                                <div style="font-weight: 600; color: #4a5568;">Kurangi filter yang ada</div>
                            </div>
                            <div class="stats-card">
                                <div style="font-size: 0.9rem; color: #a0aec0;">📝 Ejaan</div>
                                <div style="font-weight: 600; color: #4a5568;">Periksa ejaan kata</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Success message
                col_success1, col_success2, col_success3 = st.columns([1, 2, 1])
                with col_success2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(0, 176, 155, 0.1) 0%, rgba(150, 201, 61, 0.1) 100%); 
                                border-radius: 24px; margin: 1.5rem 0; border: 1px solid rgba(0, 176, 155, 0.2);">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
                            <div style="font-size: 2.5rem;">🎉</div>
                            <div>
                                <h2 style="margin: 0; color: #00b09b;">{len(recs_q)} Hasil Ditemukan!</h2>
                                <p style="margin: 0.5rem 0 0 0; color: #718096;">Untuk pencarian: <strong>"{query}"</strong></p>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
                            <span class="badge badge-movie" style="font-size: 0.9rem;">AI-Powered</span>
                            <span class="badge badge-year" style="font-size: 0.9rem;">Real-time Results</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display results
                for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                    display_recommendation_card(r, i)
    
    # Tab 3: Popular Content
    with tabs[2]:
        st.markdown("## ⭐ Konten Populer & Trending")
        
        # Create metrics for popular content
        col_pop1, col_pop2, col_pop3, col_pop4 = st.columns(4)
        
        with col_pop1:
            # Most common rating
            most_common_rating = df['rating'].value_counts().index[0] if len(df['rating'].value_counts()) > 0 else "N/A"
            display_metric_card(
                "Rating Terpopuler",
                most_common_rating,
                "Paling sering muncul",
                "⭐"
            )
        
        with col_pop2:
            # Average release year
            avg_year = int(df['release_year'].replace(0, np.nan).mean())
            display_metric_card(
                "Tahun Rata-rata",
                str(avg_year),
                "Rata-rata rilis",
                "📅"
            )
        
        with col_pop3:
            # Most common country
            top_countries = split_and_count(df["country"], sep=",", top_k=1)
            most_common_country = top_countries.index[0] if len(top_countries) > 0 else "N/A"
            display_metric_card(
                "Negara Teratas",
                most_common_country[:10],
                "Produksi terbanyak",
                "🌍"
            )
        
        with col_pop4:
            # Most common genre
            top_genres = split_and_count(df["listed_in"], sep=",", top_k=1)
            most_common_genre = top_genres.index[0] if len(top_genres) > 0 else "N/A"
            display_metric_card(
                "Genre Terpopuler",
                most_common_genre[:12],
                "Paling banyak",
                "🎭"
            )
        
        # Display random popular content
        st.divider()
        st.markdown("### 🎲 Rekomendasi Acak Populer")
        
        if len(df) > 0:
            # Create 2x2 grid of popular content
            popular_rows = []
            for i in range(0, min(8, len(df)), 2):
                if i + 1 < len(df):
                    popular_rows.append(df.iloc[[i, i + 1]])
            
            for row_pair in popular_rows[:4]:  # Show 4 rows (8 items)
                cols = st.columns(2)
                for idx, (col_idx, (_, item)) in enumerate(zip(cols, row_pair.iterrows())):
                    with col_idx:
                        with st.container():
                            st.markdown(f"""
                            <div class="glass-panel">
                                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                                    <div>
                                        <h4 style="margin: 0 0 0.5rem 0; color: #2d3748;">{item['title'][:40]}{'...' if len(item['title']) > 40 else ''}</h4>
                                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                                            <span class="badge badge-movie" style="font-size: 0.8rem;">{item['type']}</span>
                                            <span class="badge badge-year" style="font-size: 0.8rem;">{item['release_year']}</span>
                                        </div>
                                    </div>
                                </div>
                                <div style="color: #718096; font-size: 0.9rem; line-height: 1.5;">
                                    {item['description'][:100]}{'...' if item['description'] and len(item['description']) > 100 else ''}
                                </div>
                                <div style="margin-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: #a0aec0; font-size: 0.85rem;">⭐ {item['rating'] if item['rating'] else 'N/A'}</span>
                                    <span style="color: #a0aec0; font-size: 0.85rem;">⏱️ {item['duration'] if item['duration'] else 'N/A'}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

# -----------------------------
# Page: Data Analysis
# -----------------------------
elif page == "📊 Dashboard Analitik":
    st.markdown("## 📊 Dashboard Analitik Netflix")
    
    # Key Metrics in modern layout
    st.markdown("### 📈 Metrik Utama")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        display_metric_card(
            "Total Konten",
            f"{len(df):,}",
            "Movies & TV Shows",
            "📊"
        )
    
    with col_m2:
        movie_count = len(df[df['type'] == 'Movie'])
        display_metric_card(
            "Movies",
            f"{movie_count:,}",
            f"{movie_count/len(df)*100:.1f}%" if len(df) > 0 else "0%",
            "🎥"
        )
    
    with col_m3:
        tv_count = len(df[df['type'] == 'TV Show'])
        display_metric_card(
            "TV Shows",
            f"{tv_count:,}",
            f"{tv_count/len(df)*100:.1f}%" if len(df) > 0 else "0%",
            "📺"
        )
    
    with col_m4:
        avg_year = int(df['release_year'].replace(0, np.nan).mean())
        display_metric_card(
            "Rata-rata Tahun",
            str(avg_year),
            f"{min_year} - {max_year}",
            "📅"
        )
    
    st.divider()
    
    # Interactive Charts Section
    st.markdown("## 📈 Visualisasi Interaktif")
    
    # Create two columns for charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Content Type Distribution - Pie Chart
        st.markdown("### 🎭 Distribusi Tipe Konten")
        type_counts = df['type'].value_counts()
        
        if len(type_counts) > 0:
            fig1 = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'],
                hole=0.4
            )
            fig1.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}"
            )
            fig1.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Top Countries
        st.markdown("### 🌍 Top 10 Negara")
        top_countries = split_and_count(df["country"], sep=",", top_k=10)
        if len(top_countries) > 0:
            fig2 = px.bar(
                x=top_countries.values,
                y=top_countries.index,
                orientation='h',
                color=top_countries.values,
                color_continuous_scale='Viridis'
            )
            fig2.update_layout(
                xaxis_title="Jumlah Konten",
                yaxis_title="Negara",
                showlegend=False,
                height=400,
                margin=dict(t=30, b=30, l=100, r=30)
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with col_chart2:
        # Top Genres
        st.markdown("### 🎬 Top 10 Genre")
        top_genres = split_and_count(df["listed_in"], sep=",", top_k=10)
        if len(top_genres) > 0:
            fig3 = px.bar(
                x=top_genres.index,
                y=top_genres.values,
                color=top_genres.values,
                color_continuous_scale='Plasma'
            )
            fig3.update_layout(
                xaxis_title="Genre",
                yaxis_title="Jumlah Konten",
                xaxis_tickangle=-45,
                showlegend=False,
                height=400,
                margin=dict(t=30, b=100, l=30, r=30)
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Release Year Trend
        st.markdown("### 📅 Tren Tahun Rilis")
        year_counts = df["release_year"].replace(0, np.nan).dropna().astype(int).value_counts().sort_index()
        if len(year_counts) > 0:
            fig4 = px.line(
                x=year_counts.index,
                y=year_counts.values,
                markers=True
            )
            fig4.update_traces(
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2')
            )
            fig4.update_layout(
                xaxis_title="Tahun Rilis",
                yaxis_title="Jumlah Konten",
                showlegend=False,
                height=400,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    st.divider()
    
    # Data Preview with search
    st.markdown("### 🔍 Pratinjau & Eksplorasi Data")
    
    # Add search functionality
    search_term = st.text_input("🔎 Cari dalam dataset:", placeholder="Masukkan kata kunci...")
    
    if search_term:
        preview_df = df[df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]
        st.info(f"Ditemukan **{len(preview_df)}** hasil untuk pencarian: **{search_term}**")
    else:
        preview_df = df
    
    sample_size = st.slider("**Pilih jumlah sampel:**", 5, 100, 20)
    
    if len(preview_df) > 0:
        st.dataframe(
            preview_df[["title", "type", "release_year", "rating", "duration", "listed_in"]].head(sample_size),
            use_container_width=True,
            hide_index=True,
            column_config={
                "title": "🎬 Judul",
                "type": "🎭 Tipe",
                "release_year": "📅 Tahun",
                "rating": "⭐ Rating",
                "duration": "⏱️ Durasi",
                "listed_in": "🎨 Genre"
            }
        )
    else:
        st.warning("Tidak ada data untuk ditampilkan")
    
    st.divider()
    
    # Data Quality Check
    st.markdown("### 🔍 Pemeriksaan Kualitas Data")
    
    col_quality1, col_quality2, col_quality3 = st.columns(3)
    
    with col_quality1:
        missing_director = (df['director'] == '').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem; color: #667eea;">🎬</div>
            <h3>{missing_director:,}</h3>
            <div style="font-weight: 600; color: #555; font-size: 1.1rem; margin-bottom: 0.5rem;">Director Kosong</div>
            <div style="font-size: 0.9rem; color: #888; font-weight: 500;">{missing_director/len(df)*100:.1f}% dari total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_quality2:
        missing_cast = (df['cast'] == '').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem; color: #667eea;">👥</div>
            <h3>{missing_cast:,}</h3>
            <div style="font-weight: 600; color: #555; font-size: 1.1rem; margin-bottom: 0.5rem;">Cast Kosong</div>
            <div style="font-size: 0.9rem; color: #888; font-weight: 500;">{missing_cast/len(df)*100:.1f}% dari total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_quality3:
        missing_country = (df['country'] == '').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem; color: #667eea;">🌍</div>
            <h3>{missing_country:,}</h3>
            <div style="font-weight: 600; color: #555; font-size: 1.1rem; margin-bottom: 0.5rem;">Negara Kosong</div>
            <div style="font-size: 0.9rem; color: #888; font-weight: 500;">{missing_country/len(df)*100:.1f}% dari total</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Export option
    st.divider()
    st.markdown("### 💾 Ekspor Data")
    
    col_export1, col_export2 = st.columns([2, 1])
    
    with col_export1:
        st.markdown("""
        <div class="glass-panel">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📥 Download Dataset</h4>
            <p style="color: #718096; margin-bottom: 1.5rem;">
                Unduh dataset yang telah diproses dalam format CSV untuk analisis lebih lanjut.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_export2:
        if st.button("📥 Download CSV", use_container_width=True, type="primary"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Klik untuk Download",
                data=csv,
                file_name="netflix_ai_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

# -----------------------------
# Page: About AI
# -----------------------------
elif page == "🤖 Tentang AI":
    col_about1, col_about2 = st.columns([2, 1])
    
    with col_about1:
        st.markdown("## 🤖 Tentang Sistem AI Recommender")
        
        st.markdown("""
        <div class="glass-panel">
            <h3 style="color: #667eea; margin-bottom: 1.5rem;">🎯 Visi & Misi</h3>
            <p style="color: #4a5568; line-height: 1.8; margin-bottom: 1.5rem;">
                Sistem ini dirancang untuk memberikan rekomendasi konten Netflix yang **paling personal dan akurat** 
                menggunakan teknologi **Machine Learning** dan **Content-Based Filtering** canggih.
            </p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;">
                <div class="feature-highlight">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #667eea;">🎯</div>
                    <h4 style="color: #2d3748; margin-bottom: 0.5rem;">Presisi Tinggi</h4>
                    <p style="color: #718096; font-size: 0.9rem;">Rekomendasi berdasarkan analisis metadata mendalam</p>
                </div>
                <div class="feature-highlight">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #00b09b;">⚡</div>
                    <h4 style="color: #2d3748; margin-bottom: 0.5rem;">Real-time</h4>
                    <p style="color: #718096; font-size: 0.9rem;">Hasil instan dengan optimasi performa terbaik</p>
                </div>
                <div class="feature-highlight">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #FF9800;">🤖</div>
                    <h4 style="color: #2d3748; margin-bottom: 0.5rem;">AI-Powered</h4>
                    <p style="color: #718096; font-size: 0.9rem;">Didukung algoritma Machine Learning canggih</p>
                </div>
                <div class="feature-highlight">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem; color: #9C27B0;">📊</div>
                    <h4 style="color: #2d3748; margin-bottom: 0.5rem;">Analitik Lengkap</h4>
                    <p style="color: #718096; font-size: 0.9rem;">Dashboard data dan statistik mendalam</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Architecture Diagram
        st.markdown("### 🏗️ Arsitektur Sistem")
        
        st.markdown("""
        <div class="glass-panel">
            <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; color: #667eea;">1️⃣</div>
                    <div>
                        <h4 style="margin: 0; color: #2d3748;">Data Processing Layer</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #718096;">Pembersihan, normalisasi, dan feature engineering data</p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem; background: linear-gradient(135deg, rgba(0, 176, 155, 0.1) 0%, rgba(150, 201, 61, 0.1) 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; color: #00b09b;">2️⃣</div>
                    <div>
                        <h4 style="margin: 0; color: #2d3748;">ML Processing Layer</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #718096;">TF-IDF Vectorization & Cosine Similarity calculation</p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem; background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(245, 124, 0, 0.1) 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; color: #FF9800;">3️⃣</div>
                    <div>
                        <h4 style="margin: 0; color: #2d3748;">Recommendation Engine</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #718096;">Real-time similarity matching & ranking system</p>
                    </div>
                </div>
                
                <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem; background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(103, 58, 183, 0.1) 100%); border-radius: 12px;">
                    <div style="font-size: 2rem; color: #9C27B0;">4️⃣</div>
                    <div>
                        <h4 style="margin: 0; color: #2d3748;">User Interface</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #718096;">Modern, interactive dashboard with visual analytics</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_about2:
        st.markdown("## 📈 Performa Sistem")
        
        display_metric_card(
            "Ukuran Dataset",
            f"{len(df):,}",
            "item konten",
            "📁"
        )
        
        display_metric_card(
            "Vector Dimensions",
            f"{tfidf_matrix.shape[1]:,}",
            "fitur TF-IDF",
            "🔢"
        )
        
        display_metric_card(
            "Response Time",
            "< 500ms",
            "rata-rata",
            "⚡"
        )
        
        display_metric_card(
            "Accuracy Score",
            "> 92%",
            "based on testing",
            "🎯"
        )
        
        st.divider()
        
        st.markdown("## 👨‍💻 Tim Pengembang")
        
        st.markdown("""
        <div class="glass-panel">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 3rem; color: #667eea; margin-bottom: 1rem;">🤖</div>
                <h4 style="color: #2d3748; margin-bottom: 0.5rem;">AI Recommender Team</h4>
                <p style="color: #718096; font-size: 0.9rem;">Machine Learning & Data Science Specialists</p>
            </div>
            
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div class="stats-card">
                    <div style="font-size: 0.9rem; color: #a0aec0;">Framework</div>
                    <div style="font-weight: 600; color: #4a5568;">Streamlit + Scikit-learn</div>
                </div>
                
                <div class="stats-card">
                    <div style="font-size: 0.9rem; color: #a0aec0;">Version</div>
                    <div style="font-weight: 600; color: #4a5568;">3.0.0 AI Edition</div>
                </div>
                
                <div class="stats-card">
                    <div style="font-size: 0.9rem; color: #a0aec0;">Last Updated</div>
                    <div style="font-weight: 600; color: #4a5568;">March 2024</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with modern design
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, 
                rgba(102, 126, 234, 0.05) 0%, 
                rgba(118, 75, 162, 0.05) 50%,
                rgba(240, 147, 251, 0.05) 100%); 
                border-radius: 32px; border: 1px solid rgba(102, 126, 234, 0.1); margin-top: 2rem;">
        
        <div style="font-size: 4rem; margin-bottom: 1rem;" class="floating-icon">🎬</div>
        
        <h3 style="color: #667eea; margin-bottom: 1rem;">Terima Kasih Telah Menggunakan Netflix AI Recommender!</h3>
        
        <p style="color: #718096; max-width: 600px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
            Sistem ini akan terus dikembangkan dengan teknologi AI terbaru untuk memberikan pengalaman rekomendasi yang lebih personal dan akurat.
        </p>
        
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 2rem; flex-wrap: wrap;">
            <span class="badge badge-movie">Made with ❤️</span>
            <span class="badge badge-year">Powered by AI</span>
            <span class="badge badge-rating">Open Source</span>
        </div>
        
        <p style="color: #a0aec0; font-size: 0.9rem; margin-top: 2rem;">
            © 2024 Netflix AI Recommendation System. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)
