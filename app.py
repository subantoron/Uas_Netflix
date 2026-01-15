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
    page_title="Netflix Recommender (Content-Based)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"


# -----------------------------
# Helpers
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
    # keep letters/numbers, turn others to spaces
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

    # Ensure expected columns exist (dataset variants)
    expected = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added_iso",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    # Basic cleaning
    text_cols = ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"]
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)
        df[c] = df[c].replace({"Unknown": ""})

    # Robust numeric
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    else:
        df["release_year"] = 0

    # Build "soup" for content-based filtering
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

    # Create display title (unique)
    df["display_title"] = df["title"].astype(str) + " (" + df["type"].astype(str) + ", " + df["release_year"].astype(str) + ")"
    dup = df["display_title"].duplicated(keep=False)
    if dup.any():
        # Append show_id to make unique
        df.loc[dup, "display_title"] = df.loc[dup].apply(
            lambda r: f"{r['title']} ({r['type']}, {r['release_year']}) — {r.get('show_id','')}",
            axis=1,
        )

    # Ensure show_id exists and is unique-ish
    if df["show_id"].astype(str).duplicated().any():
        df["show_id"] = df.apply(lambda r: f"{r.get('show_id','')}_{r.name}", axis=1)

    return df


@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(corpus: pd.Series):
    """
    Build TF-IDF vectorizer + matrix.
    Cached as a resource (kept in memory across reruns).
    """
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
    # cosine similarity between 1 item vs all items
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
        # No known vocab terms in query
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
    s = series.fillna("").astype(str).replace({"Unknown": ""})
    exploded = s.str.split(sep).explode().astype(str).str.strip()
    exploded = exploded[exploded != ""]
    return exploded.value_counts().head(top_k)


def fmt_item_row(r: pd.Series) -> dict:
    return {
        "Judul": r.get("title", ""),
        "Tipe": r.get("type", ""),
        "Tahun": int(r.get("release_year", 0)) if str(r.get("release_year", "")).isdigit() else r.get("release_year", ""),
        "Rating": r.get("rating", ""),
        "Durasi": r.get("duration", ""),
        "Genre": r.get("listed_in", ""),
        "Negara": r.get("country", ""),
        "Similarity": float(r.get("similarity", 0.0)),
    }


def show_item_detail(r: pd.Series, header_prefix: str = "🎬"):
    title = _safe_str(r.get("title", ""))
    typ = _safe_str(r.get("type", ""))
    year = r.get("release_year", "")
    rating = _safe_str(r.get("rating", ""))
    duration = _safe_str(r.get("duration", ""))
    genres = _safe_str(r.get("listed_in", ""))
    country = _safe_str(r.get("country", ""))
    director = _safe_str(r.get("director", ""))
    cast = _safe_str(r.get("cast", ""))
    desc = _safe_str(r.get("description", ""))

    st.markdown(f"### {header_prefix} {title}")
    meta = []
    if typ:
        meta.append(typ)
    if year:
        meta.append(str(year))
    if rating:
        meta.append(f"Rating: {rating}")
    if duration:
        meta.append(duration)
    if country:
        meta.append(country)
    if genres:
        meta.append(genres)
    if meta:
        st.caption(" • ".join(meta))

    cols = st.columns(2)
    with cols[0]:
        if director:
            st.write(f"**Director:** {director}")
        if cast:
            st.write(f"**Cast:** {cast}")
    with cols[1]:
        if desc:
            st.write("**Deskripsi:**")
            st.write(desc)


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Sistem Rekomendasi Netflix Movies & TV Shows")
st.write(
    """
Aplikasi ini menggunakan **Content-Based Filtering** berbasis metadata (judul, genre, cast, director, deskripsi, dll).
Model mengubah teks menjadi **TF‑IDF** lalu menghitung kemiripan menggunakan **Cosine Similarity**.
"""
)

with st.sidebar:
    st.header("⚙️ Pengaturan")
    page = st.radio("Menu", ["Rekomendasi", "Eksplorasi Dataset", "Tentang"], index=0)
    st.divider()

    st.subheader("Dataset")
    uploaded = st.file_uploader("Upload dataset (.csv) jika tidak ada file lokal", type=["csv"])
    use_local = st.checkbox("Gunakan file lokal netflix_titles.csv (jika tersedia)", value=True)


# Load data
raw_df = None
if uploaded is not None:
    raw_df = load_data_from_upload(uploaded.getvalue())
else:
    if use_local and DEFAULT_DATA_PATH.exists():
        raw_df = load_data_from_path(str(DEFAULT_DATA_PATH))
    else:
        st.warning(
            "Dataset belum tersedia. Upload file CSV di sidebar, atau taruh file **netflix_titles.csv** di folder yang sama dengan app.py."
        )

if raw_df is None:
    st.stop()

df = prepare_data(raw_df)

# Build vectorizer/matrix once per dataset content
vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["soup"])

# Common limits for filters
min_year = int(df["release_year"].replace(0, np.nan).min(skipna=True) or 1900)
max_year = int(df["release_year"].max() or 2025)
type_options = ["All"] + sorted(df["type"].dropna().unique().tolist())


# -----------------------------
# Pages
# -----------------------------
if page == "Rekomendasi":
    tabs = st.tabs(["Berdasarkan Judul", "Berdasarkan Kata Kunci"])

    # ---- Tab 1: By title
    with tabs[0]:
        st.subheader("Rekomendasi berdasarkan judul yang dipilih")

        left, right = st.columns([1.2, 1.0])
        with left:
            filter_type_for_selector = st.selectbox("Filter pilihan judul (opsional)", options=type_options, index=0)
            if filter_type_for_selector == "All":
                selector_df = df
            else:
                selector_df = df[df["type"] == filter_type_for_selector]

            options = selector_df["display_title"].tolist()
            selected_display = st.selectbox("Pilih judul", options=options, index=0)

            top_n = st.slider("Jumlah rekomendasi", min_value=5, max_value=30, value=10, step=1)
            same_type = st.checkbox("Rekomendasikan hanya tipe yang sama (Movie ↔ Movie / TV ↔ TV)", value=True)

            year_range = st.slider(
                "Filter tahun rilis (opsional)",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                step=1,
            )
            year_min, year_max = year_range

            run = st.button("🔎 Cari Rekomendasi", type="primary")

        with right:
            # Show "newly added" list (if column exists)
            if "date_added_iso" in df.columns:
                st.caption("📌 Contoh judul yang relatif baru ditambahkan (berdasarkan date_added_iso)")
                tmp = df.copy()
                tmp["date_added_iso"] = pd.to_datetime(tmp["date_added_iso"], errors="coerce")
                tmp = tmp.sort_values("date_added_iso", ascending=False).head(10)
                st.dataframe(
                    tmp[["title", "type", "release_year", "date_added_iso"]].rename(
                        columns={"release_year": "tahun_rilis", "date_added_iso": "date_added"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        if run:
            # Map display -> idx
            idx = int(df.index[df["display_title"] == selected_display][0])

            st.divider()
            st.subheader("Judul yang dipilih")
            show_item_detail(df.iloc[idx], header_prefix="✅")

            st.divider()
            st.subheader("Hasil rekomendasi")

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
                st.info("Tidak ada rekomendasi yang cocok dengan filter kamu.")
            else:
                # Table view
                table_df = pd.DataFrame([fmt_item_row(r) for _, r in recs.iterrows()])
                table_df["Similarity"] = table_df["Similarity"].round(4)
                st.dataframe(table_df, use_container_width=True, hide_index=True)

                # Detail expanders
                st.markdown("#### Detail rekomendasi")
                for i, (_, r) in enumerate(recs.iterrows(), start=1):
                    with st.expander(f"{i}. {_safe_str(r.get('title',''))} — similarity {float(r.get('similarity',0.0)):.4f}", expanded=False):
                        show_item_detail(r, header_prefix="⭐")

    # ---- Tab 2: By keywords
    with tabs[1]:
        st.subheader("Rekomendasi berdasarkan kata kunci (query)")

        c1, c2, c3 = st.columns([2.0, 1.0, 1.0])
        with c1:
            query = st.text_input(
                "Masukkan kata kunci (mis: 'heist crime gang', 'romance korean', 'documentary nature')",
                value="",
            )
        with c2:
            type_filter = st.selectbox("Filter tipe", options=type_options, index=0)
        with c3:
            top_n_q = st.slider("Jumlah hasil", min_value=5, max_value=30, value=10, step=1, key="top_n_q")

        year_range_q = st.slider(
            "Filter tahun rilis (opsional)",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
            key="year_range_q",
        )
        year_min_q, year_max_q = year_range_q

        run_q = st.button("🔎 Cari berdasarkan kata kunci", type="primary", key="run_q")

        if run_q:
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
                st.warning(
                    "Query tidak menghasilkan rekomendasi. Coba gunakan kata kunci lain (bahasa Inggris biasanya lebih cocok), atau hilangkan filter."
                )
            else:
                table_df = pd.DataFrame([fmt_item_row(r) for _, r in recs_q.iterrows()])
                table_df["Similarity"] = table_df["Similarity"].round(4)
                st.dataframe(table_df, use_container_width=True, hide_index=True)

                st.markdown("#### Detail hasil")
                for i, (_, r) in enumerate(recs_q.iterrows(), start=1):
                    with st.expander(f"{i}. {_safe_str(r.get('title',''))} — similarity {float(r.get('similarity',0.0)):.4f}", expanded=False):
                        show_item_detail(r, header_prefix="🔎")

elif page == "Eksplorasi Dataset":
    st.header("📊 Eksplorasi Dataset")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total judul", f"{len(df):,}")
    c2.metric("Movie", f"{int((df['type']=='Movie').sum()):,}")
    c3.metric("TV Show", f"{int((df['type']=='Tv Show').sum()):,}")

    st.divider()

    st.subheader("Contoh data")
    st.dataframe(
        df[["show_id", "title", "type", "release_year", "rating", "duration", "listed_in", "country"]].head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Distribusi tipe")
    type_counts = df["type"].value_counts()
    st.bar_chart(type_counts)

    st.divider()
    st.subheader("Top genre")
    top_genres = split_and_count(df["listed_in"], sep=",", top_k=12)
    st.bar_chart(top_genres)

    st.divider()
    st.subheader("Top negara")
    top_countries = split_and_count(df["country"], sep=",", top_k=12)
    st.bar_chart(top_countries)

    st.divider()
    st.subheader("Tren tahun rilis")
    year_counts = df["release_year"].replace(0, np.nan).dropna().astype(int).value_counts().sort_index()
    st.line_chart(year_counts)

elif page == "Tentang":
    st.header("ℹ️ Tentang Proyek")

    st.markdown(
        """
### Ide Proyek
Membangun **Sistem Rekomendasi Netflix Movies & TV Shows** menggunakan pendekatan **Content-Based Filtering**.

### Kenapa Content-Based?
Karena dataset ini berisi metadata (genre, cast, director, deskripsi, dll) tanpa data interaksi user (rating/klik/riwayat tonton).
Jadi rekomendasi dibuat dengan mencari judul lain yang **paling mirip kontennya** dengan judul yang dipilih.

### Metode & Algoritma
- Representasi teks: **TF‑IDF (Term Frequency–Inverse Document Frequency)**  
- Ukuran kemiripan: **Cosine Similarity**  
- Skor similarity dihitung antara 1 item vs seluruh item, lalu diambil Top‑N.

### Catatan
- Ini adalah demo sistem rekomendasi, bukan sistem produksi Netflix.
- Kamu bisa menambahkan fitur lanjutan seperti:
  - pembobotan fitur (genre lebih penting daripada cast, dll),
  - hybrid (content + collaborative),
  - evaluasi (precision@k) jika punya ground-truth.
"""
    )
