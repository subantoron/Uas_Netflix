import io
import re
from pathlib import Path

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
    
    /* Success Message Styling */
    .custom-success {
        background: linear-gradient(135deg, rgba(0, 176, 155, 0.1) 0%, rgba(150, 201, 61, 0.1) 100%);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(0, 176, 155, 0.2);
        margin: 1rem 0;
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
    
    /* Custom Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
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

def create_custom_chart(title, data, chart_type="bar", color=None):
    """Create custom styled chart container"""
    st.markdown(f"""
    <div class="chart-container">
        <h4 style="color: #667eea; margin-bottom: 1rem;">{title}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if chart_type == "bar":
        st.bar_chart(data)
    elif chart_type == "line":
        st.line_chart(data)
    elif chart_type == "area":
        st.area_chart(data)

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
st.mark
