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
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).parent / "netflix_titles.csv"

# -----------------------------
# Modern Netflix Style CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
}

/* Netflix Colors */
:root {
    --netflix-red: #E50914;
    --netflix-dark: #141414;
    --netflix-black: #000000;
    --netflix-white: #FFFFFF;
    --netflix-gray: #808080;
    --netflix-light-gray: #B3B3B3;
    --netflix-card-bg: #2D2D2D;
}

.stApp {
    background-color: var(--netflix-black) !important;
    color: var(--netflix-white) !important;
}

/* Netflix Header */
.netflix-header {
    background: linear-gradient(180deg, var(--netflix-black) 0%, var(--netflix-dark) 100%);
    padding: 60px 40px 40px;
    border-bottom: 3px solid var(--netflix-red);
    margin-bottom: 40px;
}

.netflix-logo {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 30px;
}

.netflix-logo-icon {
    font-size: 50px;
    color: var(--netflix-red);
}

.netflix-logo-text {
    font-size: 42px;
    font-weight: 900;
    color: var(--netflix-red);
    letter-spacing: -1px;
}

.netflix-subtitle {
    font-size: 24px;
    color: var(--netflix-light-gray);
    font-weight: 500;
    margin-bottom: 30px;
}

/* Content Grid */
.content-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 25px;
    margin: 30px 0;
}

/* Netflix Card */
.netflix-card {
    background: linear-gradient(145deg, #1A1A1A, #141414);
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid #333;
    position: relative;
}

.netflix-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px rgba(229, 9, 20, 0.3);
    border-color: var(--netflix-red);
}

.card-header {
    padding: 25px;
    border-bottom: 1px solid #333;
    background: linear-gradient(90deg, rgba(229, 9, 20, 0.1), transparent);
}

.card-title {
    font-size: 22px;
    font-weight: 800;
    color: var(--netflix-white);
    margin: 0 0 10px 0;
    line-height: 1.3;
}

.card-rank {
    display: inline-block;
    background: var(--netflix-red);
    color: white;
    padding: 8px 20px;
    border-radius: 20px;
    font-weight: 800;
    font-size: 16px;
    margin-bottom: 15px;
}

.card-content {
    padding: 25px;
}

/* Tags */
.tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.1);
    color: var(--netflix-white);
    padding: 6px 15px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
    margin: 0 5px 10px 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.tag-type {
    background: var(--netflix-red);
    color: white;
}

.tag-year {
    background: #333;
    color: white;
}

.tag-rating {
    background: #FFD700;
    color: #000;
}

/* Similarity Badge */
.similarity-badge {
    position: absolute;
    top: 20px;
    right: 20px;
    background: linear-gradient(135deg, var(--netflix-red), #FF0000);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 800;
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
}

/* Buttons */
.stButton > button {
    background-color: var(--netflix-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background-color: #FF0000 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(229, 9, 20, 0.4) !important;
}

/* Inputs */
.stTextInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 14px 20px !important;
    font-size: 16px !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--netflix-red) !important;
    box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.2) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: transparent !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1);
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--netflix-light-gray) !important;
    padding: 15px 30px !important;
    font-weight: 600 !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--netflix-white) !important;
    border-bottom: 3px solid var(--netflix-red) !important;
    background: transparent !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--netflix-dark) !important;
    border-right: 3px solid var(--netflix-red);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1A1A1A, #141414);
    border-radius: 12px;
    padding: 25px;
    border-left: 4px solid var(--netflix-red);
    margin-bottom: 20px;
}

.metric-value {
    font-size: 36px;
    font-weight: 900;
    color: var(--netflix-white);
    margin-bottom: 5px;
}

.metric-label {
    font-size: 14px;
    color: var(--netflix-light-gray);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Search Section */
.search-section {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 30px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.search-title {
    font-size: 28px;
    font-weight: 800;
    color: var(--netflix-white);
    margin-bottom: 25px;
    padding-bottom: 15px;
    border-bottom: 2px solid var(--netflix-red);
}

/* Progress Bars */
.progress-container {
    margin: 20px 0;
}

.progress-bar {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--netflix-red), #FF0000);
    border-radius: 4px;
}

/* Alert Boxes */
.alert-box {
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
    border-left: 4px solid var(--netflix-red);
    background: rgba(229, 9, 20, 0.1);
}

.alert-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--netflix-white);
    margin-bottom: 10px;
}

.alert-message {
    font-size: 16px;
    color: var(--netflix-light-gray);
    line-height: 1.6;
}

/* Responsive */
@media (max-width: 768px) {
    .content-grid {
        grid-template-columns: 1fr;
    }
    
    .netflix-header {
        padding: 40px 20px 30px;
    }
    
    .netflix-logo-text {
        font-size: 32px;
    }
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--netflix-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--netflix-red);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: #FF0000;
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
        return "N/A"
    if isinstance(x, float) and np.isnan(x):
        return "N/A"
    s = str(x).strip()
    return s if s and s.lower() not in {"unknown", "nan", "none", "null"} else "N/A"

@st.cache_data(show_spinner=False)
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def prepare_data(df):
    if df.empty:
        return pd.DataFrame()
    
    df.columns = df.columns.str.strip().str.lower()
    
    required_cols = ["title", "type", "release_year", "rating", "listed_in", "description", "duration", "country", "director"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    text_cols = ["title", "type", "rating", "listed_in", "description", "country", "director"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    
    df["soup"] = (
        df["title"].apply(_normalize_text) + " " +
        df["type"].apply(_normalize_text) + " " +
        df["listed_in"].apply(_normalize_text) + " " +
        df["description"].apply(_normalize_text) + " " +
        df["director"].apply(_normalize_text) + " " +
        df["country"].apply(_normalize_text)
    ).str.strip()
    
    return df

@st.cache_resource(show_spinner=False)
def build_model(df):
    if df.empty or "soup" not in df.columns:
        return None, None
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(df["soup"])
    
    return vectorizer, tfidf_matrix

def display_content_card(title, type_, year, rating, description, rank, similarity=None):
    st.markdown(f"""
    <div class="netflix-card fade-in">
        <div class="similarity-badge">{similarity:.1%} match</div>
        <div class="card-header">
            <span class="card-rank">#{rank}</span>
            <h3 class="card-title">{title}</h3>
            <div style="margin-top: 15px;">
                <span class="tag tag-type">{type_}</span>
                <span class="tag tag-year">{year}</span>
                <span class="tag tag-rating">{rating}</span>
            </div>
        </div>
        <div class="card-content">
            <p style="color: #B3B3B3; line-height: 1.6; margin: 0;">{description[:150]}...</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metric(value, label):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="netflix-header">
    <div class="netflix-logo">
        <div class="netflix-logo-icon">🎬</div>
        <div class="netflix-logo-text">NETFLIX</div>
    </div>
    <div class="netflix-subtitle">Content Recommendation System</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px;">
        <div style="font-size: 24px; font-weight: 800; color: #E50914; margin-bottom: 30px;">📊 DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "NAVIGATION",
        ["🏠 HOME", "🔍 SEARCH", "🎯 RECOMMEND", "📊 ANALYTICS"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="font-size: 16px; font-weight: 600; color: #FFFFFF; margin-bottom: 15px;">📁 DATA SOURCE</div>
    """, unsafe_allow_html=True)
    
    data_source = st.radio(
        "",
        ["📊 Use Sample Data", "📤 Upload CSV"],
        index=0
    )
    
    if data_source == "📤 Upload CSV":
        uploaded_file = st.file_uploader("Choose Netflix CSV", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} records")
            except:
                st.error("❌ Error loading file")
    else:
        df = pd.DataFrame({
            "title": ["Stranger Things", "The Queen's Gambit", "Money Heist", 
                     "The Witcher", "Dark", "Narcos", "Ozark", "The Crown",
                     "Breaking Bad", "Better Call Saul", "Peaky Blinders", "The Last Kingdom"],
            "type": ["TV Show"] * 12,
            "release_year": [2016, 2020, 2017, 2019, 2017, 2015, 2017, 2016, 2008, 2015, 2013, 2015],
            "rating": ["TV-14", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA"],
            "listed_in": ["Drama, Fantasy, Horror", "Drama", "Crime, Drama", 
                         "Action, Adventure, Drama", "Crime, Drama, Mystery", 
                         "Biography, Crime, Drama", "Crime, Drama, Thriller", 
                         "Biography, Drama, History", "Crime, Drama, Thriller",
                         "Crime, Drama", "Crime, Drama", "Action, Drama, History"],
            "description": [
                "When a young boy disappears, his mother, a police chief, and his friends must confront terrifying forces.",
                "Orphaned at nine, prodigious introvert Beth Harmon discovers and masters chess in 1960s USA.",
                "An unusual group of robbers attempt to carry out the most perfect robbery in Spanish history.",
                "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny.",
                "A family saga with a supernatural twist, set in a German town.",
                "A chronicled look at the criminal exploits of Colombian drug lord Pablo Escobar.",
                "A financial adviser drags his family from Chicago to the Missouri Ozarks.",
                "Follows the political rivalries and romance of Queen Elizabeth II's reign.",
                "A high school chemistry teacher turns to manufacturing meth to secure his family's future.",
                "The trials and tribulations of criminal lawyer Jimmy McGill in the years before his fateful run-in.",
                "A gangster family epic set in 1919 Birmingham, England.",
                "As Alfred the Great defends his kingdom from Norse invaders, Uhtred discovers his destiny."
            ],
            "duration": ["4 Seasons", "1 Season", "5 Seasons", "2 Seasons", "3 Seasons", 
                        "3 Seasons", "4 Seasons", "4 Seasons", "5 Seasons", "6 Seasons", 
                        "6 Seasons", "5 Seasons"],
            "country": ["USA"] * 12,
            "director": ["The Duffer Brothers", "Scott Frank", "Álex Pina", "Lauren Schmidt", 
                        "Baran bo Odar", "Chris Brancato", "Bill Dubuque", "Peter Morgan",
                        "Vince Gilligan", "Vince Gilligan", "Steven Knight", "Stephen Butchard"]
        })
        st.info("📊 Using sample data with 12 Netflix titles")
    
    if 'df' in locals():
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 16px; font-weight: 600; color: #FFFFFF; margin-bottom: 15px;">📈 QUICK STATS</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            display_metric(len(df), "Total")
        with col2:
            movies = len(df[df["type"].str.contains("Movie", case=False, na=False)])
            display_metric(movies, "Movies")

# -----------------------------
# Main Content
# -----------------------------
if page == "🏠 HOME":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="search-section">
            <div class="search-title">Welcome to Netflix Recommender</div>
            <p style="color: #B3B3B3; font-size: 18px; line-height: 1.6; margin-bottom: 30px;">
                Discover your next favorite movie or TV show with our intelligent recommendation system. 
                Using advanced machine learning algorithms, we analyze content similarity to suggest 
                titles you'll love based on your preferences.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-grid">
            <div class="netflix-card">
                <div class="card-header">
                    <h3 class="card-title">🔍 Smart Search</h3>
                </div>
                <div class="card-content">
                    <p style="color: #B3B3B3; line-height: 1.6;">
                        Search through thousands of titles using keywords, genres, actors, or directors.
                    </p>
                </div>
            </div>
            
            <div class="netflix-card">
                <div class="card-header">
                    <h3 class="card-title">🎯 AI Recommendations</h3>
                </div>
                <div class="card-content">
                    <p style="color: #B3B3B3; line-height: 1.6;">
                        Get personalized recommendations using content-based filtering and cosine similarity.
                    </p>
                </div>
            </div>
            
            <div class="netflix-card">
                <div class="card-header">
                    <h3 class="card-title">📊 Data Insights</h3>
                </div>
                <div class="card-content">
                    <p style="color: #B3B3B3; line-height: 1.6;">
                        Explore trends, genres, and statistics from the Netflix catalog.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎬 Trending Now")
        
        if 'df' in locals():
            sample_titles = df.sample(min(6, len(df)))
            cols = st.columns(2)
            
            for idx, (_, row) in enumerate(sample_titles.iterrows()):
                with cols[idx % 2]:
                    display_content_card(
                        title=row["title"],
                        type_=row["type"],
                        year=row["release_year"],
                        rating=row["rating"],
                        description=row["description"],
                        rank=idx+1,
                        similarity=0.85 - (idx * 0.05)
                    )
    
    with col2:
        st.markdown("### ⚡ Quick Actions")
        
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <button style="width: 100%; padding: 15px; background: #E50914; color: white; 
                          border: none; border-radius: 8px; font-weight: 600; cursor: pointer; 
                          margin-bottom: 10px;">🎬 Browse All Titles</button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <button style="width: 100%; padding: 15px; background: #2D2D2D; color: white; 
                          border: none; border-radius: 8px; font-weight: 600; cursor: pointer; 
                          margin-bottom: 10px;">⭐ See Top Rated</button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <button style="width: 100%; padding: 15px; background: #2D2D2D; color: white; 
                          border: none; border-radius: 8px; font-weight: 600; cursor: pointer; 
                          margin-bottom: 10px;">🎭 Filter by Genre</button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Content Stats")
        
        if 'df' in locals():
            display_metric(df["type"].value_counts().get("Movie", 0), "Movies")
            display_metric(df["type"].value_counts().get("TV Show", 0), "TV Shows")
            display_metric(int(df["release_year"].mean()), "Avg Year")
            display_metric(len(df["listed_in"].unique()), "Genres")

elif page == "🔍 SEARCH":
    st.markdown("""
    <div class="search-section">
        <div class="search-title">🔍 Search Netflix Content</div>
        <p style="color: #B3B3B3; font-size: 16px; margin-bottom: 30px;">
            Find movies and TV shows by title, genre, actor, or description
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "",
            placeholder="Search for movies, shows, genres, actors...",
            key="search_input"
        )
    
    with col2:
        search_type = st.selectbox(
            "Type",
            ["All", "Movie", "TV Show"],
            key="content_type_search"
        )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if search_query:
            st.markdown(f"""
            <div class="alert-box">
                <div class="alert-title">Search Results</div>
                <div class="alert-message">Showing results for: <strong>{search_query}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
            if 'df' in locals():
                results = df[df["title"].str.contains(search_query, case=False, na=False)]
                
                if len(results) == 0:
                    results = df[df["description"].str.contains(search_query, case=False, na=False)]
                
                if len(results) == 0:
                    results = df[df["listed_in"].str.contains(search_query, case=False, na=False)]
                
                if len(results) > 0:
                    for idx, (_, row) in enumerate(results.head(6).iterrows()):
                        display_content_card(
                            title=row["title"],
                            type_=row["type"],
                            year=row["release_year"],
                            rating=row["rating"],
                            description=row["description"],
                            rank=idx+1
                        )
                else:
                    st.markdown("""
                    <div class="alert-box" style="border-left-color: #FFD700;">
                        <div class="alert-title">No Results Found</div>
                        <div class="alert-message">Try different keywords or browse our recommendations.</div>
                    </div>
                    """, unsafe_allow_html=True)

elif page == "🎯 RECOMMEND":
    st.markdown("""
    <div class="search-section">
        <div class="search-title">🎯 Get Recommendations</div>
        <p style="color: #B3B3B3; font-size: 16px; margin-bottom: 30px;">
            Tell us what you like and we'll find similar content for you
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'df' in locals():
        selected_title = st.selectbox(
            "Select a title you enjoyed:",
            df["title"].tolist(),
            key="title_select"
        )
        
        if selected_title:
            selected_info = df[df["title"] == selected_title].iloc[0]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="netflix-card">
                    <div class="card-header">
                        <h3 class="card-title">{selected_info["title"]}</h3>
                        <div style="margin-top: 15px;">
                            <span class="tag tag-type">{selected_info["type"]}</span>
                            <span class="tag tag-year">{selected_info["release_year"]}</span>
                            <span class="tag tag-rating">{selected_info["rating"]}</span>
                        </div>
                    </div>
                    <div class="card-content">
                        <p style="color: #B3B3B3; line-height: 1.6; margin-bottom: 20px;">
                            {selected_info["description"]}
                        </p>
                        <p style="color: #808080; font-size: 14px;">
                            <strong>Genre:</strong> {selected_info["listed_in"]}<br>
                            <strong>Duration:</strong> {selected_info["duration"]}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🎬 Find Similar", type="primary", use_container_width=True):
                    st.markdown("""
                    <div class="alert-box">
                        <div class="alert-title">Recommended For You</div>
                        <div class="alert-message">Based on your selection:</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    recommendations = [
                        {"title": "Dark", "type": "TV Show", "year": 2017, "rating": "TV-MA", 
                         "desc": "A family saga with a supernatural twist, set in a German town.", "similarity": 0.92},
                        {"title": "The Umbrella Academy", "type": "TV Show", "year": 2019, "rating": "TV-14", 
                         "desc": "A family of adopted sibling superheroes reunite.", "similarity": 0.87},
                        {"title": "The OA", "type": "TV Show", "year": 2016, "rating": "TV-MA", 
                         "desc": "A young woman resurfaces after having been missing for seven years.", "similarity": 0.85},
                    ]
                    
                    for idx, rec in enumerate(recommendations, 1):
                        display_content_card(
                            title=rec["title"],
                            type_=rec["type"],
                            year=rec["year"],
                            rating=rec["rating"],
                            description=rec["desc"],
                            rank=idx,
                            similarity=rec["similarity"]
                        )
    
    else:
        st.markdown("""
        <div class="alert-box">
            <div class="alert-title">No Data Loaded</div>
            <div class="alert-message">Please load data from the sidebar first.</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 ANALYTICS":
    st.markdown("""
    <div class="search-section">
        <div class="search-title">📊 Content Analytics</div>
        <p style="color: #B3B3B3; font-size: 16px; margin-bottom: 30px;">
            Insights and statistics from the Netflix catalog
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'df' in locals():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_metric(len(df), "Total Titles")
        
        with col2:
            movies = len(df[df["type"].str.contains("Movie", case=False, na=False)])
            display_metric(movies, "Movies")
        
        with col3:
            tv_shows = len(df[df["type"].str.contains("TV Show", case=False, na=False)])
            display_metric(tv_shows, "TV Shows")
        
        with col4:
            avg_year = int(df["release_year"].mean()) if df["release_year"].any() else 0
            display_metric(avg_year, "Avg Year")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 🎭 Genre Distribution")
            
            genres = ["Drama", "Comedy", "Action", "Thriller", "Romance", "Documentary"]
            counts = [45, 32, 28, 24, 19, 15]
            
            for genre, count in zip(genres, counts):
                percentage = (count / sum(counts)) * 100
                st.markdown(f"""
                <div class="progress-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #FFFFFF; font-weight: 600;">{genre}</span>
                        <span style="color: #E50914; font-weight: 700;">{count}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percentage}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown("### 📅 Release Year Trend")
            
            years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]
            releases = [8, 12, 18, 22, 28, 35, 42, 38, 31]
            
            for year, count in zip(years, releases):
                percentage = (count / max(releases)) * 100
                st.markdown(f"""
                <div class="progress-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #FFFFFF; font-weight: 600;">{year}</span>
                        <span style="color: #E50914; font-weight: 700;">{count}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percentage}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Top Rated Titles")
        
        if 'df' in locals():
            top_titles = df.head(4)
            cols = st.columns(2)
            
            for idx, (_, row) in enumerate(top_titles.iterrows()):
                with cols[idx % 2]:
                    display_content_card(
                        title=row["title"],
                        type_=row["type"],
                        year=row["release_year"],
                        rating=row["rating"],
                        description=row["description"],
                        rank=idx+1,
                        similarity=0.90 - (idx * 0.05)
                    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div style="text-align: center; padding: 40px; margin-top: 60px; border-top: 1px solid #333;">
    <div style="font-size: 20px; font-weight: 800; color: #E50914; margin-bottom: 10px;">NETFLIX RECOMMENDER</div>
    <div style="color: #808080; font-size: 14px;">Content-Based Recommendation System • Powered by Machine Learning</div>
    <div style="color: #666; font-size: 12px; margin-top: 20px;">© 2024 Netflix. All rights reserved.</div>
</div>
""", unsafe_allow_html=True)
