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
# Custom CSS - Netflix Style with Yellow Accent
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
    background: #000000 !important;
    color: #FFFFFF !important;
}

/* TYPOGRAPHY - SUPER CLEAR & BOLD */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Netflix Sans', sans-serif !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    line-height: 1.2 !important;
}

h1 { font-size: 3.5rem !important; }
h2 { font-size: 2.5rem !important; }
h3 { font-size: 2rem !important; }
h4 { font-size: 1.5rem !important; }

p, span, div, label {
    color: #F5F5F5 !important;
    font-weight: 600 !important;
    line-height: 1.6 !important;
}

strong, b {
    color: #FFD700 !important;
    font-weight: 800 !important;
}

/* MAIN HEADER */
.main-header {
    background: linear-gradient(rgba(20, 20, 20, 0.98), rgba(0, 0, 0, 0.95)),
                url('https://assets.nflxext.com/ffe/siteui/vlv3/9c5457b8-9ab0-4a04-9fc1-e608d5670f1a/710d74e0-7158-408e-8d9b-23c219dee5df/IN-en-20210719-popsignuptwoweeks-perspective_alpha_website_small.jpg');
    background-size: cover;
    background-position: center;
    padding: 4rem 3rem;
    border-radius: 12px;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
    border: 3px solid #E50914;
    box-shadow: 0 20px 60px rgba(229, 9, 20, 0.3);
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(to right, rgba(229, 9, 20, 0.2), transparent 60%);
    z-index: 1;
}

.main-header-content {
    position: relative;
    z-index: 2;
}

/* METRIC CARDS - EXTRA CLEAR */
.metric-card {
    background: linear-gradient(145deg, #1A1A1A, #0F0F0F);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    border: 2px solid #333333;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
    margin-bottom: 1.5rem;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    background: linear-gradient(90deg, #E50914, #FFD700);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: #FFD700;
    box-shadow: 0 25px 50px rgba(255, 215, 0, 0.3);
}

.metric-card h3 {
    color: #FFD700 !important;
    font-size: 3.5rem !important;
    margin-bottom: 0.8rem;
    font-weight: 1000 !important;
    text-shadow: 3px 3px 0 rgba(0,0,0,0.5);
    letter-spacing: -1.5px;
}

.metric-card-title {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    font-size: 1.4rem !important;
    margin-bottom: 0.5rem;
    display: block;
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
}

/* RECOMMENDATION CARDS */
.recommendation-card {
    background: linear-gradient(145deg, #1A1A1A, #141414);
    border-radius: 16px;
    padding: 2.5rem;
    margin: 2rem 0;
    border-left: 8px solid #FFD700;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
    border: 1px solid #333333;
}

.recommendation-card:hover {
    transform: translateY(-10px) scale(1.02);
    border-left: 8px solid #E50914;
    box-shadow: 0 30px 60px rgba(255, 215, 0, 0.3);
    background: linear-gradient(145deg, #222222, #1A1A1A);
}

.recommendation-card::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 120px;
    height: 120px;
    background: linear-gradient(135deg, transparent 50%, rgba(255, 215, 0, 0.1) 50%);
    border-radius: 0 0 0 120px;
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
    border: 3px solid #FFFFFF;
    box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
    min-width: 120px;
    justify-content: center;
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
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.3);
    transition: all 0.3s ease;
    border: 2px solid rgba(255, 255, 255, 0.1);
}

.badge:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(229, 9, 20, 0.5);
    background: linear-gradient(135deg, #FF0000 0%, #E50914 100%);
}

.badge-movie {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
    color: #000000 !important;
    font-weight: 900 !important;
}

.badge-year {
    background: linear-gradient(135deg, #2D2D2D 0%, #4A4A4A 100%) !important;
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
    border-radius: 12px !important;
    font-weight: 1000 !important;
    font-size: 1.2rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 15px 35px rgba(229, 9, 20, 0.3) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 20px 40px rgba(255, 215, 0, 0.4) !important;
    background: linear-gradient(135deg, #FF0000 0%, #FFD700 100%) !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: #141414 !important;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 3rem;
    border: 2px solid #333333;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    padding: 1.2rem 2.5rem !important;
    font-weight: 800 !important;
    color: #999999 !important;
    border: 2px solid transparent !important;
    transition: all 0.3s ease !important;
    font-size: 1.1rem !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
    color: #000000 !important;
    font-weight: 1000 !important;
    box-shadow: 0 10px 25px rgba(255, 215, 0, 0.3) !important;
    border-color: #FFD700 !important;
}

/* INPUTS */
.stTextInput > div > div > input {
    background: #1A1A1A !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    border: 2px solid #333333 !important;
    padding: 1.2rem 1.8rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #FFD700 !important;
    box-shadow: 0 0 0 4px rgba(255, 215, 0, 0.1) !important;
    background: #222222 !important;
}

.stSelectbox > div > div > div {
    background: #1A1A1A !important;
    color: #FFFFFF !important;
    border: 2px solid #333333 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #1A1A1A, #141414) !important;
    color: #FFD700 !important;
    font-weight: 900 !important;
    border-radius: 12px !important;
    border: 2px solid #333333 !important;
    font-size: 1.1rem !important;
}

/* GLASS PANEL */
.glass-panel {
    background: rgba(26, 26, 26, 0.95) !important;
    backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 2px solid #333333;
    padding: 2.5rem;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
}

.stats-card {
    background: linear-gradient(145deg, #1A1A1A, #141414);
    border-radius: 16px;
    padding: 2rem;
    border: 2px solid #333333;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
}

/* TAGS */
.tag {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    margin: 0.4rem;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFD700 0%, #E50914 100%);
    color: #000000 !important;
    border: none;
    box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 12px;
    background: #0F0F0F;
}

::-webkit-scrollbar-track {
    background: #1A1A1A;
    border-radius: 6px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%);
    border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #FF0000 0%, #FFD700 100%);
}

/* ALERTS */
.stAlert {
    border-radius: 12px !important;
    border: 2px solid !important;
    font-weight: 600 !important;
}

/* DATA FRAME */
.stDataFrame {
    border-radius: 12px !important;
    border: 2px solid #333333 !important;
    background: #141414 !important;
}

.dataframe th {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
    color: #000000 !important;
    font-weight: 1000 !important;
    font-size: 1rem !important;
}

.dataframe td {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    background: #1A1A1A !important;
}

/* DIVIDER */
hr {
    border-color: #FFD700 !important;
    opacity: 0.5;
    margin: 3rem 0 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #0F0F0F !important;
    border-right: 3px solid #E50914;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* ANIMATIONS */
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.floating {
    animation: float 3s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
    50% { box-shadow: 0 0 40px rgba(255, 215, 0, 0.6); }
}

.glow {
    animation: glow 2s ease-in-out infinite;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .main-header { padding: 2.5rem 1.5rem; }
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 2rem !important; }
    .metric-card h3 { font-size: 2.8rem !important; }
    .recommendation-card { padding: 2rem; }
    .badge { padding: 0.6rem 1.2rem; font-size: 0.9rem; }
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
        <div style="
            border: 3px solid {border}40;
            background: {bg};
            border-left: 8px solid {border};
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin: 1.5rem 0;
            backdrop-filter: blur(10px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        ">
            <div style="display: flex; gap: 1.2rem; align-items: flex-start;">
                <div style="font-size: 2rem; line-height: 1; color: {border};">{icon}</div>
                <div style="flex: 1; color: #FFFFFF !important; font-weight: 600; line-height: 1.6;">{html}</div>
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
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                        <div style="background: linear-gradient(135deg, #E50914 0%, #FFD700 100%); color: #000000; padding: 0.5rem 1.5rem; border-radius: 50px; font-weight: 1000; font-size: 1.2rem;">
                            #{rank}
                        </div>
                        <h3 style="margin: 0; color: #FFD700 !important; font-size: 2.2rem; font-weight: 900;">{title}</h3>
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
            <div style="margin: 2rem 0 1rem 0;">
                <strong style="color: #FFD700 !important; font-size: 1.3rem; font-weight: 900;">🎭 GENRE</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )
            genres = [g.strip() for g in str(genre).split(",") if g.strip()]
            tags_html = "".join([f'<span class="tag">{g}</span>' for g in genres[:5]])
            st.markdown(f'<div style="margin-bottom: 2rem;">{tags_html}</div>', unsafe_allow_html=True)

        if description and description != "No description available":
            with st.expander("📖 DESKRIPSI LENGKAP", expanded=False):
                st.markdown(
                    f"""
                <div style="font-size: 1.1rem; color: #F5F5F5 !important; line-height: 1.8; padding: 1.5rem; background: rgba(255, 215, 0, 0.05); border-radius: 12px; border-left: 4px solid #FFD700;">
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
                    <strong style="color: #FFD700 !important; font-size: 1.1rem; font-weight: 900;">🎬 DIRECTOR</strong>
                    <div style="color: #FFFFFF !important; font-size: 1.1rem; font-weight: 700; margin-top: 0.8rem;">{director}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        with col_details2:
            if country and country != "Not specified":
                st.markdown(
                    f"""
                <div class="stats-card" style="margin-bottom: 1.5rem;">
                    <strong style="color: #FFD700 !important; font-size: 1.1rem; font-weight: 900;">🌍 NEGARA</strong>
                    <div style="color: #FFFFFF !important; font-size: 1.1rem; font-weight: 700; margin-top: 0.8rem;">{country}</div>
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
            <div style="font-size: 6rem; color: #FFD700;" class="floating">🎬</div>
            <div>
                <h1 style="color: #FFD700 !important; text-shadow: 4px 4px 0 rgba(0,0,0,0.5); margin-bottom: 1rem;">NETFLIX RECOMMENDER</h1>
                <p style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #FFFFFF !important; line-height: 1.6;">
                    Sistem Rekomendasi Berbasis Machine Learning
                </p>
            </div>
        </div>
        <p style="color: #F5F5F5 !important; font-weight: 600; font-size: 1.3rem; line-height: 1.8; margin-bottom: 2.5rem;">
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
                background: linear-gradient(135deg, rgba(229, 9, 20, 0.2) 0%, rgba(255, 215, 0, 0.2) 100%);
                border-radius: 16px; border: 3px solid #FFD700;">
        <div style="font-size: 5rem; color: #FFD700; margin-bottom: 1.5rem;" class="floating">🎬</div>
        <h2 style="color: #FFD700 !important; margin: 0; font-size: 2.5rem; font-weight: 900;">NETFLIX</h2>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.2rem; margin-top: 0.8rem; font-weight: 700;">Recommender System</p>
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
        <div>
            <h3 style="color: #FFD700 !important; margin-bottom: 1rem;">📋 CARA MENGGUNAKAN</h3>
            <ol style="color: #F5F5F5 !important; margin-left: 1.5rem; line-height: 1.8;">
                <li>Unggah file CSV dataset Netflix, atau</li>
                <li>Letakkan file <code style="color: #FFD700 !important;">netflix_titles.csv</code> di folder aplikasi</li>
            </ol>
            <div style="color: #CCCCCC !important; margin-top: 1rem;">
                Format dataset: CSV dengan kolom standar Netflix (title, type, description, dsb).
            </div>
        </div>
        """,
    )

    with st.expander("🎮 COBA DEMO DENGAN DATA CONTOH", expanded=False):
        if st.button("MUAT DATA CONTOH", use_container_width=True):
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
            ui_alert("success", "Data contoh berhasil dimuat!")

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
    ui_alert(
        "success",
        """
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2rem; color: #00B894;">✅</div>
            <div>
                <div style="font-weight: 900; color: #00B894 !important; font-size: 1.2rem;">SISTEM AKTIF</div>
                <div style="font-size: 1rem; color: #CCCCCC !important;">Dataset berhasil diproses</div>
            </div>
        </div>
        """,
    )

    st.markdown("### 📊 STATISTIK DATASET")
    st.markdown(
        f"""
        <div class="glass-panel" style="padding: 2rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 1.5rem;">
                <div>
                    <div style="font-size: 1rem; color: #CCCCCC !important;">Total Data</div>
                    <div style="font-size: 2rem; font-weight: 900; color: #FFD700 !important;">{stats['total']:,}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1rem; color: #CCCCCC !important;">Unique Types</div>
                    <div style="font-size: 2rem; font-weight: 900; color: #FFD700 !important;">{stats['unique_types']}</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="font-size: 1rem; color: #CCCCCC !important;">Movies</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: #00B894 !important;">{stats['movies']:,}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1rem; color: #CCCCCC !important;">TV Shows</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: #FDCB6E !important;">{stats['tv_shows']:,}</div>
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
# Page: Recommendation
# -----------------------------
if page == "🎯 REKOMENDASI":
    tabs = st.tabs(["🎬 BERDASARKAN JUDUL", "🔍 BERDASARKAN KATA KUNCI", "⭐ KONTEN POPULER"])

    # Tab 1
    with tabs[0]:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🎯 PILIH KONTEN UNTUK DIREKOMENDASIKAN")

            st.markdown('<div class="glass-panel" style="padding: 2rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.3rem; font-weight: 900; color: #FFD700 !important; margin-bottom: 1.5rem;">🎭 FILTER TIPE KONTEN</div>', unsafe_allow_html=True)

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

            st.markdown('<div style="font-size: 1.3rem; font-weight: 900; color: #FFD700 !important; margin: 2rem 0 1.5rem;">📝 PILIH JUDUL</div>', unsafe_allow_html=True)

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

            st.markdown("### ⚙️ PENGATURAN REKOMENDASI")
            col_settings1, col_settings2, col_settings3 = st.columns(3)

            with col_settings1:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.8rem;">JUMLAH REKOMENDASI</div>', unsafe_allow_html=True)
                top_n = st.slider("Jumlah:", 5, 20, 10, 1, label_visibility="collapsed", key="top_n_slider")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_settings2:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.8rem;">FILTER TIPE</div>', unsafe_allow_html=True)
                same_type = st.checkbox("Hanya tipe yang sama", value=True, help="Movie↔Movie atau TV Show↔TV Show", key="same_type_check")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_settings3:
                st.markdown('<div class="stats-card"><div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.8rem;">TINDAKAN</div></div>', unsafe_allow_html=True)

            st.markdown("### 📅 FILTER TAHUN RILIS")
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

            if selected_display and st.button("🚀 DAPATKAN REKOMENDASI", type="primary", use_container_width=True, key="get_recs_btn"):
                matches = df[df["display_title"] == selected_display]
                if len(matches) == 0:
                    ui_alert("error", "Judul tidak ditemukan!")
                else:
                    idx = matches.index[0]

                    st.divider()
                    st.markdown("### 🎬 KONTEN YANG DIPILIH")
                    selected_item = df.loc[idx]

                    col_selected1, col_selected2 = st.columns([2, 1])
                    with col_selected1:
                        st.markdown(
                            f"""
                            <div class="glass-panel">
                                <h3 style="color: #FFD700 !important; margin-bottom: 1.5rem; font-weight: 900;">{_safe_str(selected_item['title'])}</h3>
                                <div style="display: flex; gap: 1.2rem; margin-bottom: 2rem; flex-wrap: wrap;">
                                    <span class="badge badge-movie">{_safe_str(selected_item['type'])}</span>
                                    <span class="badge badge-year">{int(selected_item['release_year'])}</span>
                                    <span class="badge badge-rating">{_safe_str(selected_item['rating'])}</span>
                                    <span class="badge">⏱️ {_safe_str(selected_item['duration'])}</span>
                                </div>
                                <div style="color: #F5F5F5 !important; line-height: 1.8; padding: 2rem; background: rgba(255, 215, 0, 0.05); border-radius: 12px; border-left: 5px solid #FFD700;">
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
                                <div style="margin-bottom: 2rem;">
                                    <div style="font-size: 1rem; color: #CCCCCC !important;">🎭 GENRE</div>
                                    <div style="font-weight: 900; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem;">{_safe_str(selected_item['listed_in']) or "Tidak tersedia"}</div>
                                </div>
                                <div style="margin-bottom: 2rem;">
                                    <div style="font-size: 1rem; color: #CCCCCC !important;">🌍 NEGARA</div>
                                    <div style="font-weight: 900; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem;">{_safe_str(selected_item['country']) or "Tidak tersedia"}</div>
                                </div>
                                <div style="margin-bottom: 2rem;">
                                    <div style="font-size: 1rem; color: #CCCCCC !important;">🎬 DIRECTOR</div>
                                    <div style="font-weight: 900; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem;">{_safe_str(selected_item['director']) or "Tidak tersedia"}</div>
                                </div>
                                <div>
                                    <div style="font-size: 1rem; color: #CCCCCC !important;">🆔 ID</div>
                                    <div style="font-weight: 900; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem;">{_safe_str(selected_item.get('show_id','N/A'))}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

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

                    if recs.empty:
                        ui_alert(
                            "warning",
                            """
                            <div class="glass-panel" style="padding: 2rem;">
                                <div style="display: flex; align-items: center; gap: 2rem;">
                                    <div style="font-size: 4rem; color: #FFD700;">🤖</div>
                                    <div>
                                        <h3 style="margin: 0; color: #FFD700 !important;">TIDAK MENEMUKAN REKOMENDASI</h3>
                                        <div style="margin-top: 0.8rem; color: #F5F5F5 !important;">Coba sesuaikan filter untuk hasil lebih baik.</div>
                                    </div>
                                </div>
                            </div>
                            """,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(255, 215, 0, 0.1) 100%);
                                        border-radius: 16px; margin: 3rem 0; border: 3px solid #FFD700;">
                                <div style="font-size: 5rem; color: #FFD700; margin-bottom: 1.5rem;" class="floating">🎯</div>
                                <h2 style="color: #FFD700 !important; margin: 0; font-weight: 900;">{len(recs)} REKOMENDASI TERBAIK</h2>
                                <p style="color: #F5F5F5 !important; margin: 1rem 0 0 0; font-size: 1.2rem;">Berdasarkan kemiripan konten (TF-IDF + Cosine Similarity)</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        for i, (_, r) in enumerate(recs.iterrows(), 1):
                            display_recommendation_card(r, i)

        with col2:
            st.markdown("### 📊 DASHBOARD DATASET")
            display_metric_card("Total Konten", f"{stats['total']:,}", "Movies & TV Shows", "📈")
            display_metric_card("Movies", f"{stats['movies']:,}", f"{(stats['movies']/stats['total']*100):.1f}%" if stats["total"] else "0%", "🎥")
            display_metric_card("TV Shows", f"{stats['tv_shows']:,}", f"{(stats['tv_shows']/stats['total']*100):.1f}%" if stats["total"] else "0%", "📺")
            display_metric_card("Tahun Terbaru", str(stats["max_year"]), "Konten terupdate", "🚀")

            st.divider()
            st.markdown("### 🎭 GENRE POPULER")
            top_genres = split_and_count(df["listed_in"], sep=",", top_k=5)
            if len(top_genres) > 0:
                for genre, count in top_genres.items():
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1.2rem;
                                    background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(229, 9, 20, 0.1) 100%);
                                    border-radius: 12px; margin: 1rem 0; border: 2px solid #333333;">
                            <span style="color: #FFFFFF !important; font-weight: 700; font-size: 1rem;">{genre[:20]}{'...' if len(genre)>20 else ''}</span>
                            <span class="badge" style="font-size: 0.9rem; padding: 0.5rem 1rem; min-width: 45px; justify-content: center;">{count}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Tab 2
    with tabs[1]:
        st.markdown("### 🔍 PENCARIAN DENGAN KATA KUNCI")
        st.markdown("Masukkan kata kunci untuk menemukan konten yang sesuai.")

        col_search1, col_search2 = st.columns([3, 1])
        with col_search1:
            query = st.text_input(
                "Masukkan kata kunci pencarian:",
                placeholder="Contoh: action adventure, romantic comedy, sci-fi, crime drama",
                help="Gunakan bahasa Inggris untuk hasil terbaik",
                label_visibility="collapsed",
                key="search_query",
            )
        with col_search2:
            search_btn = st.button("🔍 CARI", type="primary", use_container_width=True, key="search_btn")

        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.8rem;">FILTER TIPE</div>', unsafe_allow_html=True)
            type_filter = st.selectbox("Tipe:", options=type_options, index=0, label_visibility="collapsed", key="type_filter_search")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_filter2:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.8rem;">JUMLAH HASIL</div>', unsafe_allow_html=True)
            top_n_q = st.slider("Hasil:", 5, 20, 10, label_visibility="collapsed", key="top_n_search")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_filter3:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.1rem; color: #CCCCCC !important; margin-bottom: 0.8rem;">TAHUN RILIS</div>', unsafe_allow_html=True)
            year_range_q = st.slider("Tahun:", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="year_range_search", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

        year_min_q, year_max_q = year_range_q

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
                        <div class="glass-panel" style="padding: 2rem;">
                            <h3 style="margin: 0; color: #FFD700 !important;">TIDAK MENEMUKAN HASIL</h3>
                            <div style="margin-top: 1rem; color: #F5F5F5 !important;">Untuk pencarian: <b>{_safe_str(query)}</b></div>
                            <div style="margin-top: 2rem; color: #CCCCCC !important;">
                                Tips: pakai keyword bahasa Inggris, kurangi filter, dan gunakan kata yang lebih umum.
                            </div>
                        </div>
                        """,
                    )
                else:
                    ui_alert(
                        "success",
                        f"""
                        <div style="text-align: center;">
                            <div style="font-size: 3rem; color: #00B894;">🎉</div>
                            <div style="font-weight: 900; color: #00B894 !important; font-size: 1.8rem;">{len(recs_q)} HASIL DITEMUKAN!</div>
                            <div style="color: #F5F5F5 !important; margin-top: 1rem; font-size: 1.1rem;">Untuk pencarian: <b>{_safe_str(query)}</b></div>
                        </div>
                        """,
                    )
                    for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                        display_recommendation_card(r, i)

    # Tab 3
    with tabs[2]:
        st.markdown("## ⭐ KONTEN POPULER")

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
        st.markdown("### 🎲 REKOMENDASI ACAK")

        if len(df) > 0:
            sample_size = min(8, len(df))
            random_sample = df.sample(sample_size)
            cols = st.columns(2)

            for i, (_, item) in enumerate(random_sample.iterrows()):
                with cols[i % 2]:
                    st.markdown(
                        f"""
                        <div class="glass-panel" style="padding: 2rem;">
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
# Page: Data Analysis
# -----------------------------
elif page == "📊 DASHBOARD ANALITIK":
    st.markdown("## 📊 DASHBOARD ANALITIK NETFLIX")
    st.markdown("### 📈 METRIK UTAMA")

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
    st.markdown("## 📈 VISUALISASI INTERAKTIF")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("### 🎭 DISTRIBUSI TIPE KONTEN")
        type_counts = df["type"].value_counts()
        if len(type_counts) > 0:
            st.bar_chart(type_counts)

        st.markdown("### 🌍 TOP 10 NEGARA")
        top_countries = split_and_count(df["country"], sep=",", top_k=10)
        if len(top_countries) > 0:
            st.bar_chart(top_countries)

    with col_chart2:
        st.markdown("### 🎬 TOP 10 GENRE")
        top_genres = split_and_count(df["listed_in"], sep=",", top_k=10)
        if len(top_genres) > 0:
            st.bar_chart(top_genres)

        st.markdown("### 📅 TREN TAHUN RILIS")
        year_counts = df["release_year"][df["release_year"] > 0].value_counts().sort_index()
        if len(year_counts) > 0:
            st.line_chart(year_counts)

    st.divider()
    st.markdown("### 🔍 PRATINJAU & EKSPLORASI DATA")

    search_term = st.text_input("🔎 Cari dalam dataset:", placeholder="Masukkan kata kunci...", key="data_search")

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

    sample_size = st.slider("**Pilih jumlah sampel:**", 5, 100, 20, key="sample_size")
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
    st.markdown("### 🔍 PEMERIKSAAN KUALITAS DATA")

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
    st.markdown("### 💾 EKSPOR DATA")

    col_export1, col_export2 = st.columns([2, 1])
    with col_export1:
        st.markdown(
            """
            <div class="glass-panel">
                <h4 style="color: #FFD700 !important; margin-bottom: 1.5rem;">📥 DOWNLOAD DATASET</h4>
                <p style="color: #F5F5F5 !important; margin-bottom: 0; font-size: 1.1rem;">Unduh dataset yang telah diproses dalam format CSV.</p>
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
        st.markdown("## 🤖 TENTANG SISTEM REKOMENDASI")
        st.markdown(
            """
        <div class="glass-panel">
            <h3 style="color: #FFD700 !important; margin-bottom: 2rem;">📌 RINGKASAN</h3>
            <p style="color: #F5F5F5 !important; line-height: 1.8; margin-bottom: 0; font-size: 1.1rem;">
                Sistem ini menggunakan <strong>Content-Based Filtering</strong> dengan <strong>TF-IDF</strong> dan <strong>Cosine Similarity</strong>
                untuk memberikan rekomendasi berdasarkan kemiripan metadata (genre, deskripsi, cast, director, dsb).
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_about2:
        st.markdown("## 📈 INFO TEKNIS")
        display_metric_card("Ukuran Dataset", f"{len(df):,}", "item konten", "📁")
        display_metric_card("Dimensi Vektor", f"{tfidf_matrix.shape[1]:,}", "fitur TF-IDF", "🔢")
        display_metric_card("Waktu Respon", "< 500ms", "rata-rata", "⚡")

    st.divider()
    st.markdown(
        """
    <div style="text-align: center; padding: 4rem; border-radius: 20px; border: 3px solid #FFD700;
                background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(255, 215, 0, 0.1) 100%);">
        <div style="font-size: 5rem; color: #FFD700; margin-bottom: 2rem;" class="floating">🎬</div>
        <h2 style="color: #FFD700 !important; margin: 0; font-weight: 900;">TERIMA KASIH!</h2>
        <p style="color: #F5F5F5 !important; margin: 1rem 0 0 0; font-size: 1.2rem;">Built with Streamlit + Scikit-learn</p>
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 3rem;">
            <span class="badge" style="background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;">NETFLIX STYLE</span>
            <span class="badge" style="background: linear-gradient(135deg, #2D2D2D 0%, #4A4A4A 100%) !important;">MACHINE LEARNING</span>
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
