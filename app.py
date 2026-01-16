import io
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="🎬 Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# =========================================================
# CSS (Sidebar full + scroll, selectbox fix, UI rapih)
# =========================================================
NETFLIX_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;700;800;900&display=swap');

:root{
  --bg:#000000;
  --panel:#0F0F0F;
  --panel2:#141414;
  --border:#2D2D2D;
  --red:#E50914;
  --red2:#B81D24;
  --text:#F5F5F5;
  --muted:#BBBBBB;
  --good:#00B894;
  --warn:#FDCB6E;
}

*{
  font-family:'Netflix Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.stApp{
  background: var(--bg) !important;
  color: var(--text) !important;
  overflow-x:hidden !important;
}

/* Text aman (hindari target div global) */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stApp p, .stApp label, .stApp li {
  color: var(--text) !important;
}

h1,h2,h3,h4,h5,h6{
  color:#FFFFFF !important;
  font-weight:900 !important;
  text-shadow:1px 1px 3px rgba(0,0,0,0.8);
}

/* Header */
.netflix-header{
  background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.70), rgba(0,0,0,0.92)),
              url('https://assets.nflxext.com/ffe/siteui/vlv3/9c5457b8-9ab0-4a04-9fc1-e608d5670f1a/710d74e0-7158-408e-8d9b-23c219dee5df/IN-en-20210719-popsignuptwoweeks-perspective_alpha_website_small.jpg');
  background-size:cover;
  background-position:center 30%;
  padding:3rem 2rem 3.5rem 2rem;
  margin:-1rem -1rem 1.6rem -1rem;
  border-bottom:6px solid var(--red);
  box-shadow:0 15px 40px rgba(0,0,0,0.9);
  position:relative;
  overflow:hidden;
}
.netflix-header::before{
  content:'';
  position:absolute;
  inset:0;
  background: linear-gradient(to right, rgba(229,9,20,0.40) 0%, transparent 40%, transparent 60%, rgba(229,9,20,0.40) 100%);
  z-index:1;
}
.netflix-title{
  font-size:3rem !important;
  font-weight:900 !important;
  color:var(--red) !important;
  text-shadow:3px 3px 0 rgba(0,0,0,0.9), 0 0 30px rgba(229,9,20,0.7) !important;
  margin:0 0 0.4rem 0 !important;
  z-index:2;
  position:relative;
  letter-spacing:-0.5px;
}
.netflix-subtitle{
  font-size:1.25rem !important;
  font-weight:750 !important;
  color:#FFFFFF !important;
  text-shadow:2px 2px 4px rgba(0,0,0,0.8) !important;
  margin:0 0 1rem 0 !important;
  z-index:2;
  position:relative;
  opacity:0.95;
}

/* Panels */
.glass-panel{
  background: rgba(15,15,15,0.92) !important;
  border-radius: 12px;
  border: 1px solid rgba(229,9,20,0.40);
  padding: 1.25rem;
  box-shadow: 0 8px 25px rgba(0,0,0,0.7);
  margin-bottom: 1rem !important;
}
.stats-card{
  background: linear-gradient(145deg, var(--panel), #1A1A1A);
  border-radius: 12px;
  padding: 1.1rem;
  border: 1px solid var(--border);
  box-shadow: 0 8px 20px rgba(0,0,0,0.6);
  margin-bottom: 1rem !important;
  position: relative;
  overflow: hidden;
}
.stats-card::before{
  content:'';
  position:absolute;
  top:0; left:0; right:0;
  height:3px;
  background: linear-gradient(90deg, var(--red), #FF0000);
}

/* Badges */
.badge{
  display:inline-flex !important;
  align-items:center !important;
  justify-content:center !important;
  padding:0.48rem 0.95rem !important;
  border-radius: 20px !important;
  font-size:0.85rem !important;
  font-weight:900 !important;
  margin-right:0.55rem !important;
  margin-bottom:0.55rem !important;
  background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%) !important;
  color:#FFFFFF !important;
  box-shadow:0 4px 12px rgba(229,9,20,0.4) !important;
}
.badge-year{ background: linear-gradient(135deg, #333 0%, #666 100%) !important; }
.badge-rating{ background: linear-gradient(135deg, #F5C518 0%, #FFD700 100%) !important; color:#000 !important; }

/* Button */
.stButton > button{
  background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.85rem 1.2rem !important;
  font-weight: 900 !important;
  min-height: 48px !important;
  width:100% !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap:0.5rem !important;
  background: linear-gradient(135deg, var(--panel) 0%, #1A1A1A 100%) !important;
  padding:0.8rem !important;
  border-radius:12px !important;
  border:1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"]{
  border-radius:10px !important;
  padding:0.75rem 1.2rem !important;
  font-weight:900 !important;
  color:#AAAAAA !important;
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%) !important;
  color:#FFFFFF !important;
}

/* =========================================================
   SIDEBAR FIX + SCROLL
   ========================================================= */

/* Shell sidebar - jangan padding (hindari clip) */
section[data-testid="stSidebar"], [data-testid="stSidebar"]{
  background: linear-gradient(135deg, var(--panel) 0%, #1A1A1A 100%) !important;
  border-right: 3px solid var(--red) !important;
  box-shadow: 5px 0 20px rgba(0,0,0,0.8) !important;
  padding: 0 !important;
  overflow: hidden !important;
}

/* Konten sidebar - buat scroll */
div[data-testid="stSidebarContent"]{
  padding: 1.25rem 1rem 1.75rem 1rem !important;
  box-sizing: border-box !important;
  height: 100vh !important;         /* ✅ bikin full tinggi layar */
  overflow-y: auto !important;      /* ✅ bikin bisa scroll */
  overflow-x: hidden !important;
}

/* Lebarkan sidebar desktop */
@media (min-width: 992px){
  section[data-testid="stSidebar"]{
    min-width: 340px !important;
    max-width: 340px !important;
  }
}

.sidebar-title{
  color: #FFFFFF !important;
  font-weight: 950 !important;
  font-size: 1.05rem !important;
  margin: 0.85rem 0 0.5rem 0 !important;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.sidebar-title span{
  color: var(--red) !important;
}

/* radio group rapi */
section[data-testid="stSidebar"] div[role="radiogroup"]{
  background: rgba(20,20,20,0.85) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 0.5rem !important;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"]{
  width: 100% !important;
  padding: 0.65rem 0.75rem !important;
  border-radius: 10px !important;
  margin: 0.15rem 0 !important;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{
  background: rgba(229,9,20,0.14) !important;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"] span{
  color: #FFFFFF !important;
  font-weight: 900 !important;
}

/* uploader rapi */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"]{
  background: rgba(20,20,20,0.85) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 0.65rem !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section{
  background: var(--panel2) !important;
  border: 2px dashed rgba(229,9,20,0.60) !important;
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] *{
  color: #FFFFFF !important;
}

/* =========================================================
   SELECTBOX FIX
   ========================================================= */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  background: var(--panel2) !important;
  border: 2px solid var(--red) !important;
  border-radius: 10px !important;
  min-height: 48px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[data-baseweb="select"] input{
  color: #FFFFFF !important;
  font-weight: 900 !important;
}
div[data-baseweb="popover"]{ z-index: 999999 !important; }
div[data-baseweb="popover"] div[role="listbox"]{
  background: var(--panel2) !important;
  border: 1px solid rgba(229,9,20,0.75) !important;
  border-radius: 10px !important;
}
div[data-baseweb="popover"] div[role="option"]{
  color: #FFFFFF !important;
  font-weight: 850 !important;
}
div[data-baseweb="popover"] div[role="option"]:hover{
  background: rgba(229,9,20,0.25) !important;
}

/* Scrollbar */
::-webkit-scrollbar{ width:8px; height:8px; }
::-webkit-scrollbar-track{ background: var(--panel); border-radius:4px; }
::-webkit-scrollbar-thumb{
  background: linear-gradient(135deg, var(--red), var(--red2));
  border-radius:4px;
  border:1px solid var(--panel);
}
::-webkit-scrollbar-thumb:hover{
  background: linear-gradient(135deg, #FF0000, var(--red));
}
</style>
"""
st.markdown(NETFLIX_CSS, unsafe_allow_html=True)

# =========================================================
# UI Helpers
# =========================================================
def ui_alert(kind: str, text: str) -> None:
    palette = {
        "success": ("#00B894", "rgba(0, 184, 148, 0.16)", "✅"),
        "info": ("#E50914", "rgba(229, 9, 20, 0.16)", "ℹ️"),
        "warning": ("#FDCB6E", "rgba(253, 203, 110, 0.18)", "⚠️"),
        "error": ("#E50914", "rgba(229, 9, 20, 0.16)", "⛔"),
    }
    border, bg, icon = palette.get(kind, palette["info"])
    st.markdown(
        f"""
        <div style="
            border: 1px solid {border}55;
            background: {bg};
            border-left: 6px solid {border};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0 1rem 0;
            box-shadow: 0 10px 26px rgba(0,0,0,0.45);
        ">
          <div style="display:flex; gap:0.85rem; align-items:flex-start;">
            <div style="font-size:1.55rem; line-height:1; color:{border};">{icon}</div>
            <div style="flex:1; color:#FFF; font-weight:850; line-height:1.45;">
              {text}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sidebar_section(title: str, icon: str = "📌") -> None:
    st.markdown(
        f"""
        <div class="sidebar-title">{icon} <span>{title}</span></div>
        """,
        unsafe_allow_html=True,
    )

def metric_grid(total: int, unique_types: int, movies: int, tv: int) -> None:
    st.markdown(
        f"""
        <div class="stats-card" style="padding:1rem;">
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.9rem;">
            <div>
              <div style="color:var(--muted); font-weight:900;">TOTAL DATA</div>
              <div style="font-size:1.65rem; font-weight:950; color:#FFF;">{total:,}</div>
            </div>
            <div style="text-align:right;">
              <div style="color:var(--muted); font-weight:900;">JENIS UNIK</div>
              <div style="font-size:1.65rem; font-weight:950; color:#FFF;">{unique_types}</div>
            </div>
            <div>
              <div style="color:var(--muted); font-weight:900;">FILM</div>
              <div style="font-size:1.4rem; font-weight:950; color:#FFF;">{movies:,}</div>
            </div>
            <div style="text-align:right;">
              <div style="color:var(--muted); font-weight:900;">SERIAL TV</div>
              <div style="font-size:1.4rem; font-weight:950; color:#FFF;">{tv:,}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Text helpers
# =========================================================
def _normalize_text(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"unknown", "nan", "none", "null", ""}:
        return ""
    s = s.replace("&", " and ").lower()
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

# =========================================================
# Data loading + prep
# =========================================================
@st.cache_data(show_spinner=False)
def load_data_from_path(path_str: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path_str)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data_from_upload(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    mapping = {
        "show id": "show_id",
        "release year": "release_year",
        "listed in": "listed_in",
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    expected = ["show_id","type","title","director","cast","country","release_year","rating","duration","listed_in","description"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    df["type"] = df["type"].fillna("").astype(str).str.strip()
    df["type"] = df["type"].apply(lambda x: "TV Show" if x.lower() == "tv show" else x)
    df["type"] = df["type"].apply(lambda x: "Movie" if x.lower() == "movie" else x)

    text_cols = ["type","title","director","cast","country","rating","duration","listed_in","description"]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
        df[c] = df[c].replace({"unknown":"", "Unknown":"", "nan":"", "NaN":"", "None":"", "none":""})

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

    df["soup"] = (
        df["title"].map(_normalize_text) +
        " " + df["type"].map(_normalize_text) +
        " " + df["director"].map(_normalize_text) +
        " " + df["cast"].map(_normalize_text) +
        " " + df["country"].map(_normalize_text) +
        " " + df["listed_in"].map(_normalize_text) +
        " " + df["rating"].map(_normalize_text) +
        " " + df["description"].map(_normalize_text)
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

# =========================================================
# Model
# =========================================================
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
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> pd.DataFrame:
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = sims.argsort()[::-1]
    order = order[order != idx]

    recs = df.iloc[order].copy()
    recs["similarity"] = sims[order]

    if same_type:
        t = df.iloc[idx].get("type", "")
        if t:
            recs = recs[recs["type"] == t]

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
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> pd.DataFrame:
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

def display_recommendation_card(r: pd.Series, rank: int) -> None:
    similarity = float(r.get("similarity", 0.0))
    title = _safe_str(r.get("title", ""))
    content_type = _safe_str(r.get("type", ""))
    year = r.get("release_year", "")
    rating = _safe_str(r.get("rating", ""))
    genre = _safe_str(r.get("listed_in", ""))
    description = _safe_str(r.get("description", "Tidak ada deskripsi"))
    duration = _safe_str(r.get("duration", ""))

    st.markdown(
        f"""
        <div class="glass-panel">
          <div style="display:flex; justify-content:space-between; gap:0.8rem; flex-wrap:wrap;">
            <div style="font-size:1.25rem; font-weight:950; color:#FFFFFF;">
              #{rank} — {title}
            </div>
            <div class="badge">{similarity:.1%}</div>
          </div>

          <div style="margin-top:0.8rem;">
            <span class="badge">🎬 {content_type}</span>
            <span class="badge badge-year">📅 {year}</span>
            <span class="badge badge-rating">⭐ {rating or "N/A"}</span>
            <span class="badge">⏱️ {duration or "N/A"}</span>
          </div>

          {f"<div style='margin-top:0.6rem;'><b style='color:var(--red);'>🎭 Genre:</b> {genre}</div>" if genre else ""}

          <div style="margin-top:0.85rem; line-height:1.55;">
            {description}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="netflix-header">
      <div style="position:relative; z-index:2;">
        <div style="display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;">
          <div style="font-size:3.2rem;">🎬</div>
          <div style="flex:1; min-width:260px;">
            <h1 class="netflix-title">NETFLIX RECOMMENDER</h1>
            <p class="netflix-subtitle">Sistem Rekomendasi Berbasis Konten (TF-IDF + Cosine Similarity)</p>
          </div>
        </div>
        <div style="margin-top:1rem;">
          <span class="badge">🎯 Presisi</span>
          <span class="badge badge-year">⚡ Cepat</span>
          <span class="badge badge-rating">📌 Content-Based</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR (STEP 1): menu + dataset dulu (biar variabel ada)
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="stats-card" style="text-align:center; margin-bottom:0.6rem;">
          <div style="font-size:2.7rem;">🎬</div>
          <div style="font-size:1.25rem; font-weight:950; color:#FFFFFF; letter-spacing:0.6px;">NETFLIX</div>
          <div style="color:var(--muted); font-weight:850; margin-top:0.15rem;">Recommender System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar_section("MENU", "🧭")
    page = st.radio(
        "Menu",
        ["🎯 REKOMENDASI", "📊 DASHBOARD ANALITIK", "🤖 TENTANG SISTEM"],
        index=0,
        label_visibility="collapsed",
        key="nav_menu",
    )

    sidebar_section("DATASET", "📁")
    uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"], key="uploader_csv")
    use_local = st.checkbox("Gunakan dataset lokal (netflix_titles.csv)", value=True, key="use_local")

# =========================================================
# LOAD DATA
# =========================================================
raw_df = None
data_loaded = False
source_label = ""

if uploaded is not None:
    raw_df = load_data_from_upload(uploaded.getvalue())
    if raw_df is not None and not raw_df.empty:
        data_loaded = True
        source_label = "Upload CSV"
elif use_local and DEFAULT_DATA_PATH.exists():
    raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
    if raw_df is not None and not raw_df.empty:
        data_loaded = True
        source_label = "Dataset Lokal"

if not data_loaded:
    with st.sidebar:
        sidebar_section("STATUS SISTEM", "📊")
        ui_alert("warning", "Sistem belum aktif. Upload CSV atau gunakan dataset lokal.")
    st.stop()

# =========================================================
# PREP + MODEL
# =========================================================
df = prepare_data(raw_df)
vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])
if df.empty or vectorizer is None or tfidf_matrix is None:
    with st.sidebar:
        sidebar_section("STATUS SISTEM", "📊")
        ui_alert("error", "Data ada, tapi teks kosong. Model TF-IDF tidak bisa dibuat.")
    st.stop()

# stats
total = len(df)
movies = int((df["type"] == "Movie").sum())
tv = int((df["type"] == "TV Show").sum())
unique_types = int(df["type"].nunique())

valid_years = df["release_year"][df["release_year"] > 0]
min_year = int(valid_years.min()) if len(valid_years) else 1900
max_year = int(valid_years.max()) if len(valid_years) else datetime.now().year

type_options = ["All"] + sorted([t for t in df["type"].unique().tolist() if t])

# =========================================================
# SIDEBAR (STEP 2): status + filter cepat + setting rekom
# =========================================================
with st.sidebar:
    sidebar_section("STATUS SISTEM", "📊")
    ui_alert("success", f"SISTEM AKTIF — {source_label}<br>Dataset berhasil diproses ({total:,} baris)")

    metric_grid(total=total, unique_types=unique_types, movies=movies, tv=tv)

    sidebar_section("FILTER CEPAT", "🔎")
    quick_type = st.selectbox("Tipe konten", options=type_options, index=0, key="quick_type")
    quick_year = st.slider(
        "Pilih rentang tahun:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        key="quick_year",
    )
    year_min, year_max = quick_year

    sidebar_section("PENGATURAN REKOMENDASI", "⚙️")
    top_n = st.slider("Jumlah rekomendasi (Top-N)", 5, 20, 10, 1, key="top_n")
    same_type = st.checkbox("Hanya tipe yang sama (untuk rekomendasi judul)", value=True, key="same_type")

# =========================================================
# MAIN PAGE CONTENT
# =========================================================
if page == "🎯 REKOMENDASI":
    tabs = st.tabs(["🎬 Berdasarkan Judul", "🔍 Berdasarkan Kata Kunci", "⭐ Konten Populer"])

    # --- TAB 1: title-based
    with tabs[0]:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("## 🎯 Rekomendasi Berdasarkan Judul")

            # selector mengikuti filter cepat sidebar
            selector_df = df.copy()
            if quick_type != "All":
                selector_df = selector_df[selector_df["type"] == quick_type]
            selector_df = selector_df[(selector_df["release_year"] >= year_min) & (selector_df["release_year"] <= year_max)]

            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("### 📝 Pilih Judul")

            title_search = st.text_input(
                "Cari judul (opsional)",
                placeholder="contoh: naruto, money heist, avengers",
                key="title_search",
            )

            selector_df_view = selector_df
            if title_search.strip():
                mask = selector_df["display_title"].str.contains(title_search, case=False, na=False)
                selector_df_view = selector_df[mask]
                if selector_df_view.empty:
                    selector_df_view = selector_df
                    ui_alert("warning", "Tidak ada judul cocok — menampilkan semua judul sesuai filter.")

            options = selector_df_view["display_title"].tolist()
            if not options:
                ui_alert("error", "Tidak ada judul untuk filter ini. Longgarkan filter tahun/tipe di sidebar.")
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()

            selected_display = st.selectbox(
                "Judul",
                options=options,
                index=0,
                label_visibility="collapsed",
                key="title_selector",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🚀 Dapatkan Rekomendasi", type="primary", key="get_recs_btn"):
                matches = df[df["display_title"] == selected_display]
                idx = matches.index[0]

                with st.spinner("Mencari rekomendasi terbaik..."):
                    recs = recommend_by_index(
                        idx=idx,
                        df=df,
                        tfidf_matrix=tfidf_matrix,
                        top_n=top_n,
                        same_type=same_type,
                        year_min=year_min,
                        year_max=year_max,
                    )

                st.markdown("---")
                if recs.empty:
                    ui_alert("warning", "Tidak ada rekomendasi. Coba matikan 'Hanya tipe yang sama' atau longgarkan filter tahun.")
                else:
                    ui_alert("success", f"Menampilkan {len(recs)} rekomendasi teratas.")
                    for i, (_, r) in enumerate(recs.iterrows(), 1):
                        display_recommendation_card(r, i)

        with col2:
            st.markdown("## 📌 Ringkasan Filter Aktif")
            st.markdown(
                f"""
                <div class="stats-card">
                  <div style="font-weight:950; color:#FFF;">Filter Sidebar</div>
                  <div style="margin-top:0.6rem; line-height:1.6; color:var(--muted); font-weight:850;">
                    • Tipe: <span style="color:#FFF;">{quick_type}</span><br>
                    • Tahun: <span style="color:#FFF;">{year_min} - {year_max}</span><br>
                    • Top-N: <span style="color:#FFF;">{top_n}</span><br>
                    • Same Type: <span style="color:#FFF;">{str(same_type)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- TAB 2: query-based
    with tabs[1]:
        st.markdown("## 🔍 Rekomendasi Berdasarkan Kata Kunci")
        qcol1, qcol2 = st.columns([3, 1])
        with qcol1:
            query = st.text_input(
                "Kata kunci",
                placeholder="contoh: action adventure, crime drama, romantic comedy, sci-fi",
                label_visibility="collapsed",
                key="search_query",
            )
        with qcol2:
            search_btn = st.button("🔍 Cari", type="primary", key="search_btn")

        # hasil juga ikut filter cepat
        if search_btn:
            base_df = df.copy()
            if quick_type != "All":
                base_df = base_df[base_df["type"] == quick_type]
            base_df = base_df[(base_df["release_year"] >= year_min) & (base_df["release_year"] <= year_max)]

            recs_q = recommend_by_query(
                query=query,
                df=base_df,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,   # matrix full tetap OK karena vectorizer sama
                top_n=top_n,
                type_filter="All",           # sudah difilter via base_df
                year_min=None,
                year_max=None,
            )

            if recs_q.empty:
                ui_alert("error", "Tidak ada hasil. Coba keyword lebih umum (biasanya bahasa Inggris).")
            else:
                ui_alert("success", f"Ditemukan {len(recs_q)} hasil teratas.")
                for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                    display_recommendation_card(r, i)

    # --- TAB 3: popular
    with tabs[2]:
        st.markdown("## ⭐ Konten Populer (Sampling)")
        sample_size = min(8, len(df))
        sample_df = df.sample(sample_size)

        cols = st.columns(2)
        for i, (_, item) in enumerate(sample_df.iterrows()):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="glass-panel">
                      <div style="font-size:1.15rem; font-weight:950; color:#FFF;">
                        {_safe_str(item['title'])}
                      </div>
                      <div style="margin-top:0.7rem;">
                        <span class="badge">🎬 {_safe_str(item['type'])}</span>
                        <span class="badge badge-year">📅 {int(item['release_year'])}</span>
                      </div>
                      <div style="margin-top:0.75rem; color:var(--text); line-height:1.55;">
                        {_safe_str(item['description'])[:170]}{'...' if len(_safe_str(item['description'])) > 170 else ''}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif page == "📊 DASHBOARD ANALITIK":
    st.markdown("## 📊 Dashboard Analitik Netflix")
    st.bar_chart(df["type"].value_counts())

    st.markdown("### 🎬 Top 10 Genre")
    genres = df["listed_in"].fillna("").astype(str).str.split(",").explode().str.strip()
    genres = genres[genres != ""].value_counts().head(10)
    st.bar_chart(genres)

    st.markdown("### 📅 Tren Tahun Rilis")
    year_counts = df["release_year"][df["release_year"] > 0].value_counts().sort_index()
    st.line_chart(year_counts)

    st.markdown("### 🔍 Preview Data")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)

else:
    st.markdown("## 🤖 Tentang Sistem")
    st.markdown(
        """
        <div class="glass-panel">
          <p style="line-height:1.65;">
            Sistem menggunakan <b>Content-Based Filtering</b> berbasis fitur teks gabungan:
            judul, tipe, sutradara, cast, negara, genre, rating, dan deskripsi.
            Fitur dibuat dengan <b>TF-IDF</b> dan dihitung kemiripannya menggunakan <b>Cosine Similarity</b>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    """
    <div style="text-align:center; margin-top:2.2rem; padding:1.3rem 1rem;
                border-top:2px solid var(--red);
                background:linear-gradient(to bottom, transparent, rgba(229,9,20,0.12));">
      <div style="font-weight:950; color:#FFF;">🎬 Netflix Recommender System</div>
      <div style="color:var(--muted); font-weight:800; margin-top:0.35rem;">
        Built with Streamlit + Scikit-learn (Content-Based Filtering)
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
