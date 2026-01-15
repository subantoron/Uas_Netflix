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
# Custom CSS - Clean Netflix Style
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
}

.stApp {
    background-color: #141414 !important;
    color: #FFFFFF !important;
}

/* Netflix Colors */
.netflix-red { color: #E50914 !important; }
.netflix-dark { background-color: #141414 !important; }
.netflix-black { background-color: #000000 !important; }
.netflix-gray { background-color: #2D2D2D !important; }

/* Header */
.main-header {
    background: linear-gradient(to right, #141414, #000000);
    padding: 40px 0;
    margin-bottom: 30px;
    border-bottom: 1px solid #E50914;
}

.logo {
    font-size: 36px;
    font-weight: 800;
    color: #E50914;
    margin-bottom: 10px;
}

.subtitle {
    color: #B3B3B3;
    font-size: 18px;
    font-weight: 400;
}

/* Content Container */
.content-container {
    padding: 20px;
}

/* Cards */
.netflix-card {
    background: #2D2D2D;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 4px solid #E50914;
    transition: transform 0.3s ease;
}

.netflix-card:hover {
    transform: translateY(-5px);
    background: #333333;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 10px;
}

.card-subtitle {
    color: #B3B3B3;
    font-size: 14px;
    margin-bottom: 15px;
}

.card-content {
    color: #FFFFFF;
    font-size: 15px;
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    background-color: #E50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: background-color 0.3s ease !important;
}

.stButton > button:hover {
    background-color: #FF0000 !important;
}

/* Inputs */
.stTextInput > div > div > input {
    background-color: #2D2D2D !important;
    color: white !important;
    border: 1px solid #4D4D4D !important;
    border-radius: 4px !important;
}

.stSelectbox > div > div {
    background-color: #2D2D2D !important;
    color: white !important;
    border: 1px solid #4D4D4D !important;
    border-radius: 4px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: #2D2D2D !important;
    color: #B3B3B3 !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #E50914 !important;
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #000000 !important;
}

[data-testid="stSidebar"] .sidebar-content {
    padding: 20px;
}

.sidebar-section {
    margin-bottom: 30px;
    padding: 15px;
    background: #1A1A1A;
    border-radius: 8px;
}

.sidebar-title {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
    border-bottom: 2px solid #E50914;
    padding-bottom: 5px;
}

/* Metrics */
.metric-box {
    background: linear-gradient(135deg, #E50914, #B81D24);
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 15px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.metric-label {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.9);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Search Section */
.search-section {
    background: #1A1A1A;
    padding: 25px;
    border-radius: 8px;
    margin-bottom: 25px;
}

.search-title {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Alert Boxes */
.alert-box {
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
    border-left: 4px solid;
}

.alert-info {
    background: rgba(229, 9, 20, 0.1);
    border-left-color: #E50914;
}

.alert-success {
    background: rgba(0, 200, 81, 0.1);
    border-left-color: #00C851;
}

.alert-warning {
    background: rgba(255, 193, 7, 0.1);
    border-left-color: #FFC107;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}

.badge-red {
    background-color: #E50914;
    color: white;
}

.badge-gray {
    background-color: #4D4D4D;
    color: white;
}

.badge-yellow {
    background-color: #FFC107;
    color: #000000;
}

/* Divider */
.divider {
    height: 1px;
    background: #4D4D4D;
    margin: 25px 0;
}

/* Responsive */
@media (max-width: 768px) {
    .content-container {
        padding: 10px;
    }
    
    .netflix-card {
        padding: 15px;
    }
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
    s = s.replace("&", "and")
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

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
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Ensure required columns exist
    required_cols = ["title", "type", "release_year", "rating", "listed_in", "description"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    # Fill missing values
    text_cols = ["title", "type", "rating", "listed_in", "description"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    
    # Create display title
    df["display_title"] = df["title"] + " (" + df["type"] + ", " + df["release_year"].astype(str) + ")"
    
    # Create search text
    df["search_text"] = (
        df["title"].apply(_normalize_text) + " " +
        df["type"].apply(_normalize_text) + " " +
        df["listed_in"].apply(_normalize_text) + " " +
        df["description"].apply(_normalize_text)
    )
    
    return df

@st.cache_resource(show_spinner=False)
def build_model(df):
    if df.empty or "search_text" not in df.columns:
        return None, None
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(df["search_text"])
    
    return vectorizer, tfidf_matrix

def display_content_card(title, content_type, year, rating, description, similarity=None):
    st.markdown(f"""
    <div class="netflix-card">
        <div class="card-title">{title}</div>
        <div style="margin-bottom: 15px;">
            <span class="badge badge-red">{content_type}</span>
            <span class="badge badge-gray">{year}</span>
            <span class="badge badge-yellow">{rating}</span>
            {f'<span style="float: right; color: #E50914; font-weight: 600;">{similarity:.1%}</span>' if similarity else ''}
        </div>
        <div class="card-content">{description[:200]}...</div>
    </div>
    """, unsafe_allow_html=True)

def display_metric(value, label):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Main App
# -----------------------------

# Header
st.markdown("""
<div class="main-header">
    <div class="content-container">
        <div class="logo">NETFLIX</div>
        <div class="subtitle">Content Recommendation System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📁 Data Source</div>', unsafe_allow_html=True)
    
    data_source = st.radio(
        "Select data source:",
        ["Use Sample Data", "Upload CSV"],
        index=0
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Netflix CSV", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} records")
            except:
                st.error("Error loading file")
    else:
        # Load sample data
        st.info("Using sample Netflix data")
        df = pd.DataFrame({
            "title": ["Stranger Things", "The Queen's Gambit", "Money Heist", 
                     "The Witcher", "Dark", "Narcos", "Ozark", "The Crown"],
            "type": ["TV Show", "TV Show", "TV Show", "TV Show", 
                    "TV Show", "TV Show", "TV Show", "TV Show"],
            "release_year": [2016, 2020, 2017, 2019, 2017, 2015, 2017, 2016],
            "rating": ["TV-14", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA", "TV-MA"],
            "listed_in": ["Drama, Fantasy, Horror", "Drama", "Crime, Drama", 
                         "Action, Adventure, Drama", "Crime, Drama, Mystery", 
                         "Biography, Crime, Drama", "Crime, Drama, Thriller", 
                         "Biography, Drama, History"],
            "description": [
                "When a young boy disappears, his mother, a police chief, and his friends must confront terrifying forces in order to get him back.",
                "Orphaned at the tender age of nine, prodigious introvert Beth Harmon discovers and masters the game of chess in 1960s USA.",
                "An unusual group of robbers attempt to carry out the most perfect robbery in Spanish history - stealing 2.4 billion euros from the Royal Mint of Spain.",
                "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny in a turbulent world where people often prove more wicked than beasts.",
                "A family saga with a supernatural twist, set in a German town, where the disappearance of two young children exposes the relationships among four families.",
                "A chronicled look at the criminal exploits of Colombian drug lord Pablo Escobar, as well as the many other drug kingpins who plagued the country through the years.",
                "A financial adviser drags his family from Chicago to the Missouri Ozarks, where he must launder money to appease a drug boss.",
                "Follows the political rivalries and romance of Queen Elizabeth II's reign and the events that shaped the second half of the twentieth century."
            ]
        })
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Statistics</div>', unsafe_allow_html=True)
    
    if 'df' in locals():
        col1, col2 = st.columns(2)
        with col1:
            display_metric(len(df), "Total")
        with col2:
            movies_count = len(df[df["type"].str.contains("Movie", case=False, na=False)])
            display_metric(movies_count, "Movies")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
    
    num_recommendations = st.slider("Number of recommendations", 3, 10, 5)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search Content", "🎯 Get Recommendations", "📊 Analytics"])

with tab1:
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown('<div class="search-title">Search Netflix Content</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by title, genre, or description:",
            placeholder="Enter keywords...",
            key="search_input"
        )
    
    with col2:
        content_type = st.selectbox(
            "Type:",
            ["All", "Movie", "TV Show"],
            key="content_type_search"
        )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if search_query:
            st.markdown(f"""
            <div class="alert-box alert-info">
                <strong>Search Results for:</strong> {search_query}
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate search results
            results = [
                {
                    "title": "Stranger Things",
                    "type": "TV Show",
                    "year": 2016,
                    "rating": "TV-14",
                    "description": "When a young boy disappears, his mother, a police chief, and his friends must confront terrifying forces.",
                    "similarity": 0.95
                },
                {
                    "title": "Dark",
                    "type": "TV Show",
                    "year": 2017,
                    "rating": "TV-MA",
                    "description": "A family saga with a supernatural twist, set in a German town.",
                    "similarity": 0.88
                },
                {
                    "title": "The Witcher",
                    "type": "TV Show",
                    "year": 2019,
                    "rating": "TV-MA",
                    "description": "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny.",
                    "similarity": 0.82
                }
            ]
            
            for result in results:
                display_content_card(
                    title=result["title"],
                    content_type=result["type"],
                    year=result["year"],
                    rating=result["rating"],
                    description=result["description"],
                    similarity=result["similarity"]
                )
        else:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>Please enter search keywords</strong>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown('<div class="search-title">Get Personalized Recommendations</div>', unsafe_allow_html=True)
    
    if 'df' in locals():
        # Select a title
        titles = df["title"].tolist()
        selected_title = st.selectbox(
            "Select a title you like:",
            titles,
            key="title_select"
        )
        
        if selected_title:
            selected_info = df[df["title"] == selected_title].iloc[0]
            
            st.markdown("""
            <div class="alert-box alert-info">
                <strong>Selected:</strong> {}
            </div>
            """.format(selected_title), unsafe_allow_html=True)
            
            display_content_card(
                title=selected_info["title"],
                content_type=selected_info["type"],
                year=selected_info["release_year"],
                rating=selected_info["rating"],
                description=selected_info["description"]
            )
            
            if st.button("🎬 Get Similar Content", type="primary", use_container_width=True):
                st.markdown("""
                <div class="alert-box alert-success">
                    <strong>Top Recommendations</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Simulate recommendations
                recommendations = [
                    {
                        "title": "Dark",
                        "type": "TV Show",
                        "year": 2017,
                        "rating": "TV-MA",
                        "description": "A family saga with a supernatural twist, set in a German town.",
                        "similarity": 0.92
                    },
                    {
                        "title": "The Umbrella Academy",
                        "type": "TV Show",
                        "year": 2019,
                        "rating": "TV-14",
                        "description": "A family of adopted sibling superheroes reunite to solve the mystery of their father's death.",
                        "similarity": 0.87
                    },
                    {
                        "title": "The OA",
                        "type": "TV Show",
                        "year": 2016,
                        "rating": "TV-MA",
                        "description": "A young woman resurfaces after having been missing for seven years.",
                        "similarity": 0.85
                    },
                    {
                        "title": "Shadow and Bone",
                        "type": "TV Show",
                        "year": 2021,
                        "rating": "TV-14",
                        "description": "Dark forces conspire against orphan mapmaker Alina Starkov.",
                        "similarity": 0.82
                    },
                    {
                        "title": "Locke & Key",
                        "type": "TV Show",
                        "year": 2020,
                        "rating": "TV-14",
                        "description": "After their father is murdered under mysterious circumstances.",
                        "similarity": 0.79
                    }
                ]
                
                for i, rec in enumerate(recommendations, 1):
                    display_content_card(
                        title=rec["title"],
                        content_type=rec["type"],
                        year=rec["year"],
                        rating=rec["rating"],
                        description=rec["description"],
                        similarity=rec["similarity"]
                    )
    else:
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>Please load data first from the sidebar</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown('<div class="search-title">Content Analytics</div>', unsafe_allow_html=True)
    
    if 'df' in locals():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_metric(len(df), "Total Content")
        
        with col2:
            movies = len(df[df["type"].str.contains("Movie", case=False, na=False)])
            display_metric(movies, "Movies")
        
        with col3:
            tv_shows = len(df[df["type"].str.contains("TV Show", case=False, na=False)])
            display_metric(tv_shows, "TV Shows")
        
        with col4:
            avg_year = int(df["release_year"].mean()) if df["release_year"].any() else 0
            display_metric(avg_year, "Avg Year")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Genre distribution
        st.markdown('<div class="card-title">Genre Distribution</div>', unsafe_allow_html=True)
        
        genres_data = [
            {"name": "Drama", "count": 45, "color": "#E50914"},
            {"name": "Comedy", "count": 32, "color": "#FF6B6B"},
            {"name": "Action", "count": 28, "color": "#4ECDC4"},
            {"name": "Documentary", "count": 24, "color": "#45B7D1"},
            {"name": "Thriller", "count": 19, "color": "#96CEB4"},
            {"name": "Romance", "count": 15, "color": "#FFEAA7"}
        ]
        
        for genre in genres_data:
            col_bar, col_text = st.columns([6, 1])
            with col_bar:
                progress = min(genre["count"] / 50, 1.0)
                st.markdown(f"""
                <div style="background: #2D2D2D; height: 25px; border-radius: 4px; margin: 5px 0; overflow: hidden;">
                    <div style="width: {progress * 100}%; height: 100%; background: {genre['color']};"></div>
                </div>
                """, unsafe_allow_html=True)
            with col_text:
                st.markdown(f'<div style="text-align: right; color: #B3B3B3;">{genre["count"]}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div style="color: #FFFFFF; font-size: 14px; margin-bottom: 10px;">{genre["name"]}</div>', unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>No data available. Load data from sidebar.</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; color: #B3B3B3; font-size: 14px; margin-top: 30px; border-top: 1px solid #2D2D2D;">
    <div style="color: #E50914; font-weight: 600; margin-bottom: 5px;">NETFLIX CONTENT RECOMMENDER</div>
    <div>Powered by Content-Based Filtering • © 2024</div>
</div>
""", unsafe_allow_html=True)
