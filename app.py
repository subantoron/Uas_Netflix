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
# CSS
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

/* Text aman */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stApp p, .stApp label, .stApp li {
  color: var(--text) !important;
}

h1,h2,h3,h4,h5,h6{
  color:#FFFFFF !important;
  font-weight:800 !important;
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
  background: linear-gradient(to right, rgba(229,9,20,0.38) 0%, transparent 40%, transparent 60%, rgba(229,9,20,0.38) 100%);
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
}
.netflix-subtitle{
  font-size:1.25rem !important;
  font-weight:700 !important;
  color:#FFFFFF !important;
  text-shadow:2px 2px 4px rgba(0,0,0,0.8) !important;
  margin:0 0 1rem 0 !important;
  z-index:2;
  position:relative;
  opacity:0.95;
}

/* Cards */
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
  font-weight:800 !important;
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
  font-weight:850 !important;
  color:#AAAAAA !important;
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%) !important;
  color:#FFFFFF !important;
}

/* =========================================================
   SIDEBAR FIX (ANTI KE-POTONG)
   ========================================================= */
section[data-testid="stSidebar"], [data-testid="stSidebar"]{
  background: linear-gradient(135deg, var(--panel) 0%, #1A1A1A 100%) !important;
  border-right: 3px solid var(--red) !important;
  box-shadow: 5px 0 20px rgba(0,0,0,0.8) !important;
  padding: 0 !important;
  overflow: visible !important;
}
div[data-testid="stSidebarContent"]{
  padding: 1.25rem 1rem !important;
  box-sizing: border-box !important;
  overflow: visible !important;
}
@media (min-width: 992px){
  section[data-testid="stSidebar"]{
    min-width: 320px !important;
    max-width: 320px !important;
  }
}
.sidebar-title{
  color: var(--red) !important;
  font-weight: 900 !important;
  font-size: 1.02rem !important;
  margin: 0.85rem 0 0.6rem 0 !important;
  letter-spacing: 0.4px;
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
  font-weight: 850 !important;
}
div[data-baseweb="popover"]{ z-index: 999999 !important; }
div[data-baseweb="popover"] div[role="listbox"]{
  background: var(--panel2) !important;
  border: 1px solid rgba(229,9,20,0.75) !important;
  border-radius: 10px !important;
}
div[data-baseweb="popover"] div[role="option"]{
  color: #FFFFFF !important;
  font-weight: 750 !important;
}
div[data-baseweb="popover"] div[role="option"]:hover{
  background: rgba(229,9,20,0.25) !important;
}
</style>
"""
st.markdown(NETFLIX_CSS, unsafe_allow_html=True)

# =========================================================
# UI HELPERS
# =========================================================
def ui_alert(kind: str, html: str) -> None:
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
            border: 1px solid {border}55;
            background: {bg};
            border-left: 5px solid {border};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.45);
        ">
          <div style="display:flex; gap:0.9rem; align-items:flex-start;">
            <div style="font-size:1.5rem; line-height:1; color:{border};">{icon}</div>
            <div style="flex:1; color:#FFF !important; font-weight:650; line-height:1.45;">
              {html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
# DATA LOADING
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
# MODEL
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

def recommend_by_index(idx: int, df: pd.DataFrame, tfidf_matrix, top_n: int = 10,
                       same_type: bool = True, year_min: Optional[int] = None, year_max: Optional[int] = None) -> pd.DataFrame:
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

def recommend_by_query(query: str, df: pd.DataFrame, vectorizer: TfidfVectorizer, tfidf_matrix,
                       top_n: int = 10, type_filter: str = "All",
                       year_min: Optional[int] = None, year_max: Optional[int] = None) -> pd.DataFrame:
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

# =========================================================
# DASHBOARD STATS
# =========================================================
def split_and_count(series: pd.Series, sep: str = ",", top_k: int = 10) -> pd.Series:
    s = series.fillna("").astype(str).replace({"unknown": "", "Unknown": "", "nan": "", "NaN": ""})
    exploded = s.str.split(sep).explode().astype(str).str.strip()
    exploded = exploded[exploded != ""]
    return exploded.value_counts().head(top_k)

def create_dashboard_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["total"] = len(df)
    stats["movies"] = int((df["type"] == "Movie").sum())
    stats["tv_shows"] = int((df["type"] == "TV Show").sum())

    valid_years = df["release_year"][df["release_year"] > 0]
    if len(valid_years) > 0:
        stats["min_year"] = int(valid_years.min())
        stats["max_year"] = int(valid_years.max())
        stats["avg_year"] = int(valid_years.mean())
    else:
        stats["min_year"] = 1900
        stats["max_year"] = datetime.now().year
        stats["avg_year"] = 2000
    return stats

def display_metric_card(title: str, value: str, subtitle: str = "", icon: str = "📊") -> None:
    st.markdown(
        f"""
        <div class="stats-card" style="text-align:center;">
          <div style="font-size:2.1rem; margin-bottom:0.55rem;">{icon}</div>
          <div style="font-size:2rem; font-weight:900; color:#FFFFFF; line-height:1;">{value}</div>
          <div style="margin-top:0.45rem; font-weight:900; color:#FFFFFF;">{title}</div>
          <div style="margin-top:0.35rem; color:var(--muted); font-weight:650;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            <div style="font-size:1.25rem; font-weight:900; color:#FFFFFF;">
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
# SIDEBAR -> DEFINISI uploaded & use_local DI SINI (FIX NameError)
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="stats-card" style="text-align:center; margin-bottom:0.9rem;">
          <div style="font-size:2.8rem;">🎬</div>
          <div style="font-size:1.35rem; font-weight:900; color:#FFFFFF; letter-spacing:0.6px;">NETFLIX</div>
          <div style="color:var(--muted); font-weight:800; margin-top:0.2rem;">Recommender System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">🧭 MENU</div>', unsafe_allow_html=True)
    page = st.radio(
        "Menu",
        ["🎯 REKOMENDASI", "📊 DASHBOARD ANALITIK", "🤖 TENTANG SISTEM"],
        index=0,
        label_visibility="collapsed",
        key="nav_menu",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">📁 DATASET</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"], key="uploader_csv")
    use_local = st.checkbox("Gunakan dataset lokal (netflix_titles.csv)", value=True, key="use_local")
    st.caption("Jika upload CSV → pakai upload. Jika tidak → pakai file lokal.")

# =========================================================
# LOAD DATA (SEKARANG uploaded SUDAH ADA)
# =========================================================
raw_df = None
data_loaded = False

if uploaded is not None:
    with st.spinner("Memuat dataset dari upload..."):
        raw_df = load_data_from_upload(uploaded.getvalue())
    if raw_df is not None and not raw_df.empty:
        data_loaded = True
        ui_alert("success", f"<b>Dataset berhasil dimuat</b> — {len(raw_df):,} baris")
    else:
        ui_alert("error", "Dataset upload kosong / tidak valid.")

elif use_local:
    if DEFAULT_DATA_PATH.exists():
        with st.spinner("Memuat dataset lokal..."):
            raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
        if raw_df is not None and not raw_df.empty:
            data_loaded = True
            ui_alert("success", f"<b>Dataset lokal berhasil dimuat</b> — {len(raw_df):,} baris")
        else:
            ui_alert("error", "Dataset lokal kosong / gagal dibaca.")
    else:
        ui_alert("warning", "File <code>netflix_titles.csv</code> tidak ditemukan di folder aplikasi.")

if not data_loaded:
    ui_alert(
        "info",
        "Upload CSV Netflix atau letakkan <code>netflix_titles.csv</code> di folder yang sama dengan <code>app.py</code>.",
    )
    st.stop()

# =========================================================
# BUILD MODEL
# =========================================================
with st.spinner("Memproses data & membangun model TF-IDF..."):
    df = prepare_data(raw_df)
    vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])

if df.empty or vectorizer is None or tfidf_matrix is None:
    ui_alert("error", "Model tidak bisa dibangun (data kosong atau teks kosong).")
    st.stop()

stats = create_dashboard_stats(df)
unique_types = sorted([t for t in df["type"].unique().tolist() if t and str(t) != "nan"])
type_options = ["All"] + unique_types

min_year = int(df["release_year"][df["release_year"] > 0].min()) if (df["release_year"] > 0).any() else 1900
max_year = int(df["release_year"].max()) if len(df) else datetime.now().year

# =========================================================
# PAGES
# =========================================================
if page == "🎯 REKOMENDASI":
    tabs = st.tabs(["🎬 Berdasarkan Judul", "🔍 Berdasarkan Kata Kunci"])

    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("## 🎯 Pilih Konten untuk Direkomendasikan")

            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("### 🎭 Filter Tipe Konten")
            filter_type_for_selector = st.selectbox(
                "Tipe Konten",
                options=type_options,
                index=0,
                label_visibility="collapsed",
                key="filter_type_title",
            )
            selector_df = df if filter_type_for_selector == "All" else df[df["type"] == filter_type_for_selector]

            st.markdown("### 📝 Pilih Judul")
            title_search = st.text_input(
                "Cari judul (opsional)",
                placeholder="contoh: naruto, money heist, avengers",
                key="title_search",
            )

            if title_search.strip():
                mask = selector_df["display_title"].str.contains(title_search, case=False, na=False)
                selector_df_view = selector_df[mask].copy()
                if selector_df_view.empty:
                    selector_df_view = selector_df.copy()
                    ui_alert("warning", "Tidak ada judul cocok, menampilkan semua judul sesuai filter.")
            else:
                selector_df_view = selector_df

            options = selector_df_view["display_title"].tolist()
            selected_display = st.selectbox(
                "Judul",
                options=options,
                index=0,
                label_visibility="collapsed",
                key="title_selector",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("## ⚙️ Pengaturan Rekomendasi")
            cA, cB = st.columns(2)
            with cA:
                top_n = st.slider("Jumlah rekomendasi", 5, 20, 10, 1, key="top_n_slider")
            with cB:
                same_type = st.checkbox("Hanya tipe yang sama", value=True, key="same_type_check")

            st.markdown("## 📅 Filter Tahun Rilis")
            year_range = st.slider(
                "Rentang tahun",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                label_visibility="collapsed",
                key="year_range_title",
            )
            year_min, year_max = year_range

            if st.button("🚀 Dapatkan Rekomendasi", type="primary", key="get_recs_btn"):
                matches = df[df["display_title"] == selected_display]
                idx = matches.index[0]
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
                    ui_alert("warning", "Tidak menemukan rekomendasi. Coba longgarkan filter.")
                else:
                    ui_alert("success", f"Menampilkan <b>{len(recs)}</b> rekomendasi teratas.")
                    for i, (_, r) in enumerate(recs.iterrows(), 1):
                        display_recommendation_card(r, i)

        with col2:
            st.markdown("## 📊 Statistik Dataset")
            display_metric_card("Total Konten", f"{stats['total']:,}", "Movies & TV Shows", "📈")
            display_metric_card("Movies", f"{stats['movies']:,}", "Jumlah film", "🎥")
            display_metric_card("TV Shows", f"{stats['tv_shows']:,}", "Jumlah serial", "📺")
            st.markdown("## 🎭 Genre Populer")
            st.bar_chart(split_and_count(df["listed_in"], sep=",", top_k=8))

    with tabs[1]:
        st.markdown("## 🔍 Pencarian Berdasarkan Kata Kunci")
        query = st.text_input("Kata kunci", placeholder="contoh: action adventure, romantic comedy, crime drama", label_visibility="collapsed")
        type_filter = st.selectbox("Filter tipe", options=type_options, index=0)
        top_n_q = st.slider("Jumlah hasil", 5, 20, 10)

        if st.button("🔍 Cari", type="primary"):
            recs_q = recommend_by_query(
                query=query,
                df=df,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                top_n=top_n_q,
                type_filter=type_filter if type_filter != "All" else "All",
            )
            if recs_q.empty:
                ui_alert("error", "Tidak ada hasil. Coba keyword yang lebih umum.")
            else:
                ui_alert("success", f"Ditemukan <b>{len(recs_q)}</b> hasil teratas.")
                for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                    display_recommendation_card(r, i)

elif page == "📊 DASHBOARD ANALITIK":
    st.markdown("## 📊 Dashboard Analitik Netflix")
    st.bar_chart(df["type"].value_counts())
    st.markdown("### Top 10 Genre")
    st.bar_chart(split_and_count(df["listed_in"], sep=",", top_k=10))
    st.markdown("### Tren Tahun Rilis")
    year_counts = df["release_year"][df["release_year"] > 0].value_counts().sort_index()
    st.line_chart(year_counts)
    st.markdown("### Preview Data")
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)

else:
    st.markdown("## 🤖 Tentang Sistem")
    st.markdown(
        """
        <div class="glass-panel">
          <p style="line-height:1.6;">
            Sistem menggunakan <b>Content-Based Filtering</b> dengan <b>TF-IDF</b> dan
            <b>Cosine Similarity</b>. Setiap konten dibuatkan fitur teks gabungan (judul, genre, cast,
            director, negara, rating, deskripsi), lalu dicari konten yang paling mirip.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center; margin-top:2.2rem; padding:1.3rem 1rem;
                border-top:2px solid var(--red);
                background:linear-gradient(to bottom, transparent, rgba(229,9,20,0.12));">
      <div style="font-weight:900; color:#FFF;">🎬 Netflix Recommender System</div>
      <div style="color:var(--muted); font-weight:700; margin-top:0.35rem;">
        Built with Streamlit + Scikit-learn (Content-Based Filtering)
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
