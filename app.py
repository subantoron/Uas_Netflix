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

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# Custom CSS - Ultra Modern Design
# -----------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
    h1,h2,h3,h4,h5,h6 { font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important; letter-spacing:-0.5px; }

    .main { padding: 1rem; }

    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

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
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
    }
    .main-header::before {
        content:'';
        position:absolute; inset:0;
        background: linear-gradient(135deg,
            rgba(102,126,234,0.8) 0%,
            rgba(118,75,162,0.8) 30%,
            rgba(240,147,251,0.6) 70%,
            rgba(102,126,234,0.4) 100%
        );
        z-index:-1;
    }
    .main-header h1 {
        font-size: 3.5rem !important;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.25rem;
        max-width: 900px;
        line-height: 1.8;
        font-weight: 400;
    }

    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-20px)} }
    .floating-icon { animation: float 6s ease-in-out infinite; }

    .recommendation-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
        position: relative;
        overflow: hidden;
    }
    .recommendation-card::before {
        content:'';
        position:absolute; left:0; right:0; top:0; height:1px;
        background: linear-gradient(90deg, transparent 0%, rgba(102,126,234,0.6) 50%, transparent 100%);
    }
    .recommendation-card::after {
        content:'';
        position:absolute; left:0; top:0; width:4px; height:100%;
        background: linear-gradient(to bottom, #667eea, #764ba2);
        border-radius: 4px 0 0 4px;
    }
    .recommendation-card:hover {
        transform: translateY(-10px) scale(1.01);
        box-shadow: 0 20px 40px rgba(102,126,234,0.15), 0 0 0 1px rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.12);
    }

    .metric-card {
        background: linear-gradient(145deg, #f0f2f5, #ffffff);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 20px 20px 60px #d9d9d9, -20px -20px 60px #ffffff;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content:'';
        position:absolute; left:0; right:0; top:0; height:4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 25px 25px 75px #d0d0d0, -25px -25px 75px #ffffff;
    }
    .metric-card h3 {
        color:#667eea;
        font-size:2.8rem !important;
        margin-bottom:0.8rem;
        font-weight:800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .badge {
        display:inline-flex; align-items:center;
        padding:0.5rem 1.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color:white;
        border-radius:50px;
        font-size:0.85rem;
        font-weight:600;
        margin-right:0.6rem;
        margin-bottom:0.6rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3), inset 0 1px 0 rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .badge::after {
        content:'';
        position:absolute; top:0; left:-100%;
        width:100%; height:100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition:left 0.5s;
    }
    .badge:hover::after { left:100%; }
    .badge:hover { transform: translateY(-2px); }

    .badge-movie { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
    .badge-year { background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%) !important; }
    .badge-rating { background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%) !important; }

    .similarity-score {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color:white;
        padding:0.7rem 1.5rem;
        border-radius:50px;
        display:inline-flex;
        align-items:center;
        font-weight:800;
        font-size:1.1rem;
        margin:0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,176,155,0.3), 0 0 20px rgba(0,176,155,0.2);
        position:relative;
        overflow:hidden;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color:white; border:none;
        padding:1rem 3rem;
        border-radius:16px;
        font-weight:700;
        font-size:1.1rem;
        transition: all 0.3s ease;
        width:100%;
        box-shadow: 0 10px 30px rgba(102,126,234,0.3), 0 0 0 1px rgba(255,255,255,0.1);
        position:relative;
        overflow:hidden;
        letter-spacing:0.5px;
        font-family:'Space Grotesk',sans-serif;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap:0.5rem;
        background: linear-gradient(90deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
        padding:0.5rem;
        border-radius:20px;
        margin-bottom:2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background:transparent;
        border-radius:16px;
        padding:1rem 2.5rem;
        font-weight:700;
        font-family:'Space Grotesk',sans-serif;
        color:#667eea;
        border:2px solid transparent;
        transition: all 0.3s ease;
        font-size:1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color:white !important;
        box-shadow: 0 10px 25px rgba(102,126,234,0.3);
        border-color: rgba(255,255,255,0.2);
    }

    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6e8ff 100%);
        border-radius:16px !important;
        font-weight:700;
        font-family:'Space Grotesk',sans-serif;
        color:#667eea;
        border:2px solid #e6e8ff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .stTextInput > div > div > input {
        border-radius:16px;
        border:2px solid #e6e8ff;
        padding:1rem 1.5rem;
        font-size:1rem;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.9);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTextInput > div > div > input:focus {
        border-color:#667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1), inset 0 2px 4px rgba(0,0,0,0.05);
        background:white;
    }

    .stDataFrame {
        border-radius:20px;
        border:1px solid #e6e8ff;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    .glass-panel {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
        border-radius:24px;
        border:1px solid rgba(255,255,255,0.1);
        padding:2rem;
    }

    .stats-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .tag {
        display:inline-block;
        padding:0.3rem 0.8rem;
        margin:0.2rem;
        border-radius:20px;
        font-size:0.8rem;
        font-weight:600;
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        color:#667eea;
        border:1px solid rgba(102,126,234,0.3);
    }

    @keyframes fadeInUp {
        from { opacity:0; transform: translateY(30px); }
        to { opacity:1; transform: translateY(0); }
    }
    .recommendation-card { animation: fadeInUp 0.6s ease-out; }

    @media (max-width: 768px) {
        .main-header { padding: 2rem; }
        .main-header h1 { font-size:2.5rem !important; }
        .metric-card h3 { font-size:2.2rem !important; }
        .stTabs [data-baseweb="tab"] { padding:0.8rem 1.5rem; font-size:1rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# UI helpers (FIX: no unsafe_allow_html on st.info/success/error)
# -----------------------------
def ui_alert(kind: str, html: str):
    """Render an HTML alert box via st.markdown. kind: info|success|warning|error"""
    palette = {
        "info": ("#667eea", "rgba(102,126,234,0.12)", "ℹ️"),
        "success": ("#00b09b", "rgba(0,176,155,0.12)", "✅"),
        "warning": ("#FF9800", "rgba(255,152,0,0.12)", "⚠️"),
        "error": ("#e53e3e", "rgba(229,62,62,0.12)", "⛔"),
    }
    border, bg, icon = palette.get(kind, palette["info"])
    st.markdown(
        f"""
        <div style="
            border:1px solid {border}33;
            background:{bg};
            border-left:6px solid {border};
            border-radius:20px;
            padding:1.25rem 1.25rem;
            margin: 0.75rem 0;
            backdrop-filter: blur(10px);
        ">
            <div style="display:flex; gap:0.75rem; align-items:flex-start;">
                <div style="font-size:1.5rem; line-height:1;">{icon}</div>
                <div style="flex:1;">
                    {html}
                </div>
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

    # Standardize type values
    df["type"] = df["type"].astype(str).str.strip()
    df["type"] = df["type"].apply(lambda x: "TV Show" if str(x).lower() == "tv show" else x)
    df["type"] = df["type"].apply(lambda x: "Movie" if str(x).lower() == "movie" else x)

    # Clean text columns
    text_cols = ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
        df[c] = df[c].replace({"unknown": "", "Unknown": "", "nan": "", "NaN": "", "None": "", "none": ""})

    # release_year
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

    # soup for vectorization
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

    # display_title
    df["display_title"] = df["title"].astype(str) + " (" + df["type"].astype(str) + ", " + df["release_year"].astype(str) + ")"

    # duplicates display_title
    dup = df["display_title"].duplicated(keep=False)
    if dup.any():
        df.loc[dup, "display_title"] = df.loc[dup].apply(
            lambda r: f"{r['title']} ({r['type']}, {r['release_year']}) — {r.get('show_id','')}",
            axis=1,
        )

    # duplicates show_id
    if df["show_id"].astype(str).duplicated().any():
        df["show_id"] = df.apply(lambda r: f"{r.get('show_id','')}_{r.name}", axis=1)

    return df


@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(corpus: pd.Series):
    """
    FIX: Jangan filter corpus keluar dari df, supaya tfidf_matrix barisnya SELARAS dengan df.
    Dengan begitu, df.iloc[idx] <-> tfidf_matrix[idx] selalu cocok.
    """
    if corpus is None or len(corpus) == 0:
        return None, None

    # Pastikan ada minimal 1 dokumen non-kosong agar vocabulary tidak empty
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

        # Jika soup kosong, similarity akan 0 semua -> tetap aman tapi hasil random-ish.
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
        st.error(f"Error in recommendation: {e}")
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
        st.error(f"Error in query recommendation: {e}")
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
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
                <div>
                    <h4 style="margin:0; color:#667eea; font-size:1.4rem; font-weight:800;">#{rank}</h4>
                    <h3 style="margin:0.5rem 0; color:#2d3748; font-size:1.6rem; font-weight:800;">{title}</h3>
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
            <div style="margin:1.2rem 0;">
                <strong style="color:#667eea; font-size:1rem;">🎭 Genre</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )
            genres = [g.strip() for g in str(genre).split(",") if g.strip()]
            tags_html = "".join([f'<span class="tag">{g}</span>' for g in genres[:5]])
            st.markdown(f'<div style="margin-bottom:1rem;">{tags_html}</div>', unsafe_allow_html=True)

        if description and description != "No description available":
            with st.expander("📖 Deskripsi", expanded=False):
                st.markdown(
                    f"""
                <div style="font-size:0.95rem; color:#718096; line-height:1.6; padding:0.5rem 0;">
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
                <div class="stats-card" style="margin-bottom:1rem;">
                    <strong style="color:#667eea; font-size:1rem;">🎬 Director</strong>
                    <div style="color:#4a5568; font-size:0.95rem; font-weight:600; margin-top:0.3rem;">{director}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        with col_details2:
            if country and country != "Not specified":
                st.markdown(
                    f"""
                <div class="stats-card" style="margin-bottom:1rem;">
                    <strong style="color:#667eea; font-size:1rem;">🌍 Negara</strong>
                    <div style="color:#4a5568; font-size:0.95rem; font-weight:600; margin-top:0.3rem;">{country}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Error displaying card: {e}")


def display_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    st.markdown(
        f"""
    <div class="metric-card">
        <div style="font-size:3rem; margin-bottom:1rem; color:#667eea;" class="floating-icon">{icon}</div>
        <h3 class="gradient-text">{value}</h3>
        <div style="font-weight:700; color:#4a5568; font-size:1.2rem; margin-bottom:0.5rem;">{title}</div>
        <div style="font-size:0.9rem; color:#a0aec0; font-weight:600;">{subtitle}</div>
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
# Header
# -----------------------------
st.markdown(
    """
<div class="main-header">
    <div style="display:flex; align-items:center; gap:1.5rem; margin-bottom:1.5rem;">
        <div style="font-size:4rem;" class="floating-icon">🎬</div>
        <div>
            <h1>Netflix AI Recommender</h1>
            <p style="margin:0;">Sistem Rekomendasi Cerdas Berbasis Machine Learning</p>
        </div>
    </div>
    <p>
        Temukan konten yang paling sesuai dengan preferensi Anda menggunakan algoritma <strong>Content-Based Filtering</strong>
        yang didukung oleh teknologi <strong>TF-IDF Vectorization</strong> dan <strong>Cosine Similarity</strong>.
    </p>
    <div style="display:flex; gap:1rem; margin-top:2rem; flex-wrap:wrap;">
        <span class="badge badge-movie">🎯 Rekomendasi Presisi</span>
        <span class="badge badge-year">⚡ Real-time Processing</span>
        <span class="badge badge-rating">🤖 AI-Powered</span>
        <span class="badge" style="background:linear-gradient(135deg,#00b09b 0%,#96c93d 100%);">📊 Analisis Data</span>
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
    <div style="text-align:center; margin-bottom:2.5rem; padding:2rem;
                background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
                border-radius:24px; border:1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px);">
        <div style="font-size:3.5rem; color:#667eea; margin-bottom:0.8rem;" class="floating-icon">🤖</div>
        <h2 style="color:white; margin:0; font-size:1.8rem;">AI Recommender</h2>
        <p style="color:rgba(255,255,255,0.9); font-size:0.95rem; margin-top:0.5rem;">Powered by Machine Learning</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧭 Menu Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        ["🎯 Rekomendasi", "📊 Dashboard Analitik", "🤖 Tentang AI"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### ⚙️ Konfigurasi Data")
    with st.expander("📁 Pengaturan Dataset", expanded=True):
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
        <div class="glass-panel" style="padding:1.25rem;">
            <h3 style="color:#667eea; margin-bottom:0.75rem;">📋 Cara Menggunakan</h3>
            <ol style="color:#718096; margin-left:1.25rem; line-height:1.7;">
                <li>Unggah file CSV dataset Netflix, atau</li>
                <li>Letakkan file <code>netflix_titles.csv</code> di folder aplikasi</li>
            </ol>
            <div style="color:#718096; margin-top:0.75rem;">
                Format dataset: CSV dengan kolom standar Netflix (title, type, description, dsb).
            </div>
        </div>
        """,
    )

    with st.expander("🎮 Coba Demo dengan Data Contoh", expanded=False):
        if st.button("Muat Data Contoh"):
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
    with st.spinner("🤖 Memproses data dengan AI..."):
        df = prepare_data(raw_df)

        if df.empty:
            ui_alert("error", "DataFrame kosong setelah pemrosesan!")
            st.stop()

        vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])
        if vectorizer is None or tfidf_matrix is None:
            ui_alert("error", "Gagal membangun model AI (teks kosong / vocabulary kosong).")
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
        <div style="display:flex; align-items:center; gap:0.6rem;">
            <div style="font-size:1.25rem;">✅</div>
            <div>
                <div style="font-weight:800; color:#00b09b;">Sistem Aktif</div>
                <div style="font-size:0.9rem; color:#718096;">Dataset berhasil diproses</div>
            </div>
        </div>
        """,
    )

    st.markdown("### 📊 Statistik Dataset")
    st.markdown(
        f"""
        <div class="glass-panel" style="padding:1.25rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.9rem;">
                <div>
                    <div style="font-size:0.9rem; color:#a0aec0;">Total Data</div>
                    <div style="font-size:1.5rem; font-weight:800; color:#667eea;">{stats['total']:,}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.9rem; color:#a0aec0;">Unique Types</div>
                    <div style="font-size:1.5rem; font-weight:800; color:#667eea;">{stats['unique_types']}</div>
                </div>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div style="font-size:0.9rem; color:#a0aec0;">Movies</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#4CAF50;">{stats['movies']:,}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.9rem; color:#a0aec0;">TV Shows</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#FF9800;">{stats['tv_shows']:,}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 🔍 Filter Cepat")

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
if page == "🎯 Rekomendasi":
    tabs = st.tabs(["🎬 Berdasarkan Judul", "🔍 Berdasarkan Kata Kunci", "⭐ Konten Populer"])

    # Tab 1
    with tabs[0]:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🎯 Pilih Konten untuk Direkomendasikan")

            st.markdown('<div class="glass-panel" style="padding:1.25rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:1.1rem; font-weight:700; color:#667eea; margin-bottom:0.5rem;">🎭 Filter Tipe Konten</div>', unsafe_allow_html=True)

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

            st.markdown('<div style="font-size:1.1rem; font-weight:700; color:#667eea; margin:1rem 0 0.5rem;">📝 Pilih Judul</div>', unsafe_allow_html=True)

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

            st.markdown("### ⚙️ Pengaturan Rekomendasi")
            col_settings1, col_settings2, col_settings3 = st.columns(3)

            with col_settings1:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.9rem; color:#a0aec0; margin-bottom:0.5rem;">Jumlah Rekomendasi</div>', unsafe_allow_html=True)
                top_n = st.slider("Jumlah:", 5, 20, 10, 1, label_visibility="collapsed", key="top_n_slider")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_settings2:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.9rem; color:#a0aec0; margin-bottom:0.5rem;">Filter Tipe</div>', unsafe_allow_html=True)
                same_type = st.checkbox("Hanya tipe yang sama", value=True, help="Movie↔Movie atau TV Show↔TV Show", key="same_type_check")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_settings3:
                st.markdown('<div class="stats-card"><div style="font-size:0.9rem; color:#a0aec0; margin-bottom:0.5rem;">Tindakan</div></div>', unsafe_allow_html=True)

            st.markdown("### 📅 Filter Tahun Rilis")
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

            if selected_display and st.button("🚀 Dapatkan Rekomendasi AI", type="primary", use_container_width=True, key="get_recs_btn"):
                matches = df[df["display_title"] == selected_display]
                if len(matches) == 0:
                    ui_alert("error", "Judul tidak ditemukan!")
                else:
                    idx = matches.index[0]

                    st.divider()
                    st.markdown("### 🎬 Konten yang Dipilih")
                    selected_item = df.loc[idx]

                    col_selected1, col_selected2 = st.columns([2, 1])
                    with col_selected1:
                        st.markdown(
                            f"""
                            <div class="glass-panel">
                                <h3 style="color:#2d3748; margin-bottom:1rem;">{_safe_str(selected_item['title'])}</h3>
                                <div style="display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap;">
                                    <span class="badge badge-movie">{_safe_str(selected_item['type'])}</span>
                                    <span class="badge badge-year">{int(selected_item['release_year'])}</span>
                                    <span class="badge badge-rating">{_safe_str(selected_item['rating'])}</span>
                                    <span class="badge">⏱️ {_safe_str(selected_item['duration'])}</span>
                                </div>
                                <div style="color:#4a5568; line-height:1.6; padding:1rem; background:rgba(102,126,234,0.05); border-radius:12px;">
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
                                <div style="margin-bottom:1rem;">
                                    <div style="font-size:0.9rem; color:#a0aec0;">🎭 Genre</div>
                                    <div style="font-weight:600; color:#4a5568; font-size:0.9rem;">{_safe_str(selected_item['listed_in']) or "Tidak tersedia"}</div>
                                </div>
                                <div style="margin-bottom:1rem;">
                                    <div style="font-size:0.9rem; color:#a0aec0;">🌍 Negara</div>
                                    <div style="font-weight:600; color:#4a5568; font-size:0.9rem;">{_safe_str(selected_item['country']) or "Tidak tersedia"}</div>
                                </div>
                                <div style="margin-bottom:1rem;">
                                    <div style="font-size:0.9rem; color:#a0aec0;">🎬 Director</div>
                                    <div style="font-weight:600; color:#4a5568; font-size:0.9rem;">{_safe_str(selected_item['director']) or "Tidak tersedia"}</div>
                                </div>
                                <div>
                                    <div style="font-size:0.9rem; color:#a0aec0;">🆔 ID</div>
                                    <div style="font-weight:600; color:#4a5568; font-size:0.9rem;">{_safe_str(selected_item.get('show_id','N/A'))}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

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

                    if recs.empty:
                        ui_alert(
                            "warning",
                            """
                            <div class="glass-panel" style="padding:1.25rem;">
                                <div style="display:flex; align-items:center; gap:1rem;">
                                    <div style="font-size:2.5rem;">🤖</div>
                                    <div>
                                        <h3 style="margin:0; color:#667eea;">AI Tidak Menemukan Rekomendasi</h3>
                                        <div style="margin-top:0.35rem; color:#718096;">Coba sesuaikan filter untuk hasil lebih baik.</div>
                                    </div>
                                </div>
                            </div>
                            """,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="text-align:center; padding:1.5rem; background:linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
                                        border-radius:20px; margin:1rem 0;">
                                <div style="font-size:3rem; color:#667eea; margin-bottom:0.5rem;">🎯</div>
                                <h3 style="color:#2d3748; margin:0;">{len(recs)} Rekomendasi Terbaik</h3>
                                <p style="color:#718096; margin:0.5rem 0 0 0;">Berdasarkan kemiripan konten (TF-IDF + Cosine Similarity)</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        for i, (_, r) in enumerate(recs.iterrows(), 1):
                            display_recommendation_card(r, i)

        with col2:
            st.markdown("### 📊 Dashboard Dataset")
            display_metric_card("Total Konten", f"{stats['total']:,}", "Movies & TV Shows", "📈")
            display_metric_card("Movies", f"{stats['movies']:,}", f"{(stats['movies']/stats['total']*100):.1f}%" if stats["total"] else "0%", "🎥")
            display_metric_card("TV Shows", f"{stats['tv_shows']:,}", f"{(stats['tv_shows']/stats['total']*100):.1f}%" if stats["total"] else "0%", "📺")
            display_metric_card("Tahun Terbaru", str(stats["max_year"]), "Konten terupdate", "🚀")

            st.divider()
            st.markdown("### 🎭 Genre Populer")
            top_genres = split_and_count(df["listed_in"], sep=",", top_k=5)
            if len(top_genres) > 0:
                for genre, count in top_genres.items():
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.8rem;
                                    background: linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
                                    border-radius:12px; margin:0.5rem 0;">
                            <span style="color:#4a5568; font-weight:600;">{genre[:20]}{'...' if len(genre)>20 else ''}</span>
                            <span class="badge" style="font-size:0.8rem; padding:0.3rem 0.8rem;">{count}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Tab 2
    with tabs[1]:
        st.markdown("### 🔍 Pencarian AI dengan Kata Kunci")
        st.markdown("Masukkan kata kunci dan biarkan AI menemukan konten yang paling sesuai.")

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
            search_btn = st.button("🔍 Cari dengan AI", type="primary", use_container_width=True, key="search_btn")

        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.9rem; color:#a0aec0; margin-bottom:0.5rem;">Filter Tipe</div>', unsafe_allow_html=True)
            type_filter = st.selectbox("Tipe:", options=type_options, index=0, label_visibility="collapsed", key="type_filter_search")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_filter2:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.9rem; color:#a0aec0; margin-bottom:0.5rem;">Jumlah Hasil</div>', unsafe_allow_html=True)
            top_n_q = st.slider("Hasil:", 5, 20, 10, label_visibility="collapsed", key="top_n_search")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_filter3:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.9rem; color:#a0aec0; margin-bottom:0.5rem;">Tahun Rilis</div>', unsafe_allow_html=True)
            year_range_q = st.slider("Tahun:", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="year_range_search", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

        year_min_q, year_max_q = year_range_q

        if search_btn:
            if not query:
                ui_alert("warning", "Masukkan kata kunci dulu ya 🙂")
            else:
                with st.spinner("🤖 AI sedang menganalisis kata kunci..."):
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
                        <div class="glass-panel" style="padding:1.25rem;">
                            <h3 style="margin:0; color:#667eea;">AI Tidak Menemukan Hasil</h3>
                            <div style="margin-top:0.4rem; color:#718096;">Untuk pencarian: <b>{_safe_str(query)}</b></div>
                            <div style="margin-top:1rem; color:#718096;">
                                Tips: pakai keyword bahasa Inggris, kurangi filter, dan gunakan kata yang lebih umum.
                            </div>
                        </div>
                        """,
                    )
                else:
                    ui_alert(
                        "success",
                        f"""
                        <div style="text-align:center;">
                            <div style="font-size:2rem;">🎉</div>
                            <div style="font-weight:800; color:#00b09b;">{len(recs_q)} hasil ditemukan!</div>
                            <div style="color:#718096; margin-top:0.25rem;">Untuk pencarian: <b>{_safe_str(query)}</b></div>
                        </div>
                        """,
                    )
                    for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                        display_recommendation_card(r, i)

    # Tab 3
    with tabs[2]:
        st.markdown("## ⭐ Konten Populer & Trending")

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
        st.markdown("### 🎲 Rekomendasi Acak Populer")

        if len(df) > 0:
            sample_size = min(8, len(df))
            random_sample = df.sample(sample_size)
            cols = st.columns(2)

            for i, (_, item) in enumerate(random_sample.iterrows()):
                with cols[i % 2]:
                    st.markdown(
                        f"""
                        <div class="glass-panel" style="padding:1.25rem;">
                            <h4 style="margin:0 0 0.5rem 0; color:#2d3748; font-size:1.1rem;">
                                {_safe_str(item['title'])[:40]}{'...' if len(_safe_str(item['title']))>40 else ''}
                            </h4>
                            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.75rem;">
                                <span class="badge badge-movie" style="font-size:0.7rem; padding:0.3rem 0.6rem;">{_safe_str(item['type'])}</span>
                                <span class="badge badge-year" style="font-size:0.7rem; padding:0.3rem 0.6rem;">{int(item['release_year'])}</span>
                            </div>
                            <div style="color:#718096; font-size:0.85rem; line-height:1.5; margin-bottom:1rem;">
                                {_safe_str(item['description'])[:90]}{'...' if len(_safe_str(item['description']))>90 else ''}
                            </div>
                            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem;">
                                <span style="color:#a0aec0;">⭐ {_safe_str(item['rating']) or 'N/A'}</span>
                                <span style="color:#a0aec0;">⏱️ {_safe_str(item['duration']) or 'N/A'}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# -----------------------------
# Page: Data Analysis
# -----------------------------
elif page == "📊 Dashboard Analitik":
    st.markdown("## 📊 Dashboard Analitik Netflix")
    st.markdown("### 📈 Metrik Utama")

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
    st.markdown("## 📈 Visualisasi Interaktif")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("### 🎭 Distribusi Tipe Konten")
        type_counts = df["type"].value_counts()
        if len(type_counts) > 0:
            st.bar_chart(type_counts)

        st.markdown("### 🌍 Top 10 Negara")
        top_countries = split_and_count(df["country"], sep=",", top_k=10)
        if len(top_countries) > 0:
            st.bar_chart(top_countries)

    with col_chart2:
        st.markdown("### 🎬 Top 10 Genre")
        top_genres = split_and_count(df["listed_in"], sep=",", top_k=10)
        if len(top_genres) > 0:
            st.bar_chart(top_genres)

        st.markdown("### 📅 Tren Tahun Rilis")
        year_counts = df["release_year"][df["release_year"] > 0].value_counts().sort_index()
        if len(year_counts) > 0:
            st.line_chart(year_counts)

    st.divider()
    st.markdown("### 🔍 Pratinjau & Eksplorasi Data")

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
    st.markdown("### 🔍 Pemeriksaan Kualitas Data")

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
    st.markdown("### 💾 Ekspor Data")

    col_export1, col_export2 = st.columns([2, 1])
    with col_export1:
        st.markdown(
            """
            <div class="glass-panel">
                <h4 style="color:#667eea; margin-bottom:1rem;">📥 Download Dataset</h4>
                <p style="color:#718096; margin-bottom:0;">Unduh dataset yang telah diproses dalam format CSV.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_export2:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="netflix_ai_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

# -----------------------------
# Page: About AI
# -----------------------------
elif page == "🤖 Tentang AI":
    col_about1, col_about2 = st.columns([2, 1])

    with col_about1:
        st.markdown("## 🤖 Tentang Sistem AI Recommender")
        st.markdown(
            """
        <div class="glass-panel">
            <h3 style="color:#667eea; margin-bottom:1rem;">🎯 Visi & Misi</h3>
            <p style="color:#4a5568; line-height:1.8; margin-bottom:0;">
                Sistem ini menggunakan <b>Content-Based Filtering</b> dengan <b>TF-IDF</b> dan <b>Cosine Similarity</b>
                untuk memberikan rekomendasi konten berdasarkan metadata (genre, deskripsi, cast, director, dsb).
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_about2:
        st.markdown("## 📈 Performa Sistem")
        display_metric_card("Ukuran Dataset", f"{len(df):,}", "item konten", "📁")
        display_metric_card("Vector Dimensions", f"{tfidf_matrix.shape[1]:,}", "fitur TF-IDF", "🔢")
        display_metric_card("Response Time", "< 500ms", "rata-rata", "⚡")

    st.divider()
    st.markdown(
        """
    <div style="text-align:center; padding:2.25rem; border-radius:32px; border:1px solid rgba(102,126,234,0.1);
                background: linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 50%, rgba(240,147,251,0.05) 100%);">
        <div style="font-size:3rem;" class="floating-icon">🎬</div>
        <h3 style="color:#667eea; margin:0.75rem 0;">Terima Kasih Telah Menggunakan Netflix AI Recommender!</h3>
        <p style="color:#718096; margin:0;">Built with Streamlit + Scikit-learn</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Run Instructions
# -----------------------------
# Save as app.py
# pip install streamlit pandas numpy scikit-learn
# streamlit run app.py
# Put netflix_titles.csv beside app.py or upload your CSV
