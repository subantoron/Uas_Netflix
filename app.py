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

# -----------------------------
# Sample Data (Fallback if no CSV found)
# -----------------------------
SAMPLE_DATA = {
    "show_id": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    "type": ["Movie", "Movie", "TV Show", "Movie", "TV Show", "Movie", "TV Show", "Movie", "TV Show", "Movie"],
    "title": [
        "The Dark Knight", 
        "Inception", 
        "Stranger Things", 
        "Pulp Fiction", 
        "Breaking Bad",
        "The Shawshank Redemption",
        "The Crown",
        "Forrest Gump",
        "Black Mirror",
        "The Godfather"
    ],
    "director": [
        "Christopher Nolan", 
        "Christopher Nolan", 
        "The Duffer Brothers", 
        "Quentin Tarantino", 
        "Vince Gilligan",
        "Frank Darabont",
        "Peter Morgan",
        "Robert Zemeckis",
        "Charlie Brooker",
        "Francis Ford Coppola"
    ],
    "cast": [
        "Christian Bale, Heath Ledger",
        "Leonardo DiCaprio, Joseph Gordon-Levitt",
        "Millie Bobby Brown, Finn Wolfhard",
        "John Travolta, Uma Thurman",
        "Bryan Cranston, Aaron Paul",
        "Tim Robbins, Morgan Freeman",
        "Claire Foy, Olivia Colman",
        "Tom Hanks, Robin Wright",
        "Various",
        "Marlon Brando, Al Pacino"
    ],
    "country": [
        "USA", "USA", "USA", "USA", "USA", "USA", "UK", "USA", "UK", "USA"
    ],
    "release_year": [2008, 2010, 2016, 1994, 2008, 1994, 2016, 1994, 2011, 1972],
    "rating": ["PG-13", "PG-13", "TV-14", "R", "TV-MA", "R", "TV-MA", "PG-13", "TV-MA", "R"],
    "duration": [
        "152 min", "148 min", "4 Seasons", "154 min", "5 Seasons", 
        "142 min", "4 Seasons", "142 min", "5 Seasons", "175 min"
    ],
    "listed_in": [
        "Action, Crime, Drama", 
        "Action, Sci-Fi, Thriller", 
        "Drama, Fantasy, Horror", 
        "Crime, Drama", 
        "Crime, Drama, Thriller",
        "Drama",
        "Biography, Drama, History",
        "Drama, Romance",
        "Drama, Sci-Fi, Thriller",
        "Crime, Drama"
    ],
    "description": [
        "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham...",
        "A thief who steals corporate secrets through the use of dream-sharing technology...",
        "When a young boy vanishes, a small town uncovers a mystery...",
        "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine...",
        "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing...",
        "Two imprisoned men bond over a number of years, finding solace and eventual redemption...",
        "Follows the political rivalries and romance of Queen Elizabeth II's reign...",
        "The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate...",
        "An anthology series exploring a twisted, high-tech multiverse...",
        "The aging patriarch of an organized crime dynasty transfers control to his son..."
    ],
}

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 1rem 0 2rem 0;
        color: white;
        text-align: center;
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .recommendation-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #667eea;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.5rem;
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
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x)
    if s.strip().lower() in {"unknown", "nan", "none", "null", ""}:
        return ""
    return s

@st.cache_data
def load_sample_data():
    return pd.DataFrame(SAMPLE_DATA)

@st.cache_data
def prepare_data(df: pd.DataFrame):
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ["type", "title", "director", "cast", "country", "release_year", 
                     "rating", "duration", "listed_in", "description"]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    # Clean text columns
    text_cols = ["type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    
    # Create soup for TF-IDF
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
    
    df["display_title"] = df["title"] + " (" + df["type"] + ", " + df["release_year"].astype(str) + ")"
    
    return df

@st.cache_resource
def build_model(df: pd.DataFrame):
    corpus = df["soup"].fillna("")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def recommend_similar(idx: int, df: pd.DataFrame, tfidf_matrix, top_n: int = 5):
    if idx < 0 or idx >= len(df):
        return pd.DataFrame()
    
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = sims.argsort()[::-1]
    order = order[order != idx]
    
    recs = df.iloc[order].copy()
    recs["similarity"] = sims[order]
    
    return recs.head(top_n)

def recommend_by_keywords(query: str, df: pd.DataFrame, vectorizer, tfidf_matrix, top_n: int = 5):
    q = _normalize_text(query)
    if not q:
        return pd.DataFrame()
    
    q_vec = vectorizer.transform([q])
    sims = linear_kernel(q_vec, tfidf_matrix).flatten()
    order = sims.argsort()[::-1]
    
    recs = df.iloc[order].copy()
    recs["similarity"] = sims[order]
    
    return recs.head(top_n)

# -----------------------------
# Initialize App
# -----------------------------

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 style="font-size: 3rem; margin-bottom: 1rem;">🎬 Netflix AI Recommender</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Discover your next favorite movie or TV show with AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)

# Load Data
with st.spinner("Loading recommendation engine..."):
    df = load_sample_data()
    df = prepare_data(df)
    vectorizer, tfidf_matrix = build_model(df)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Find Similar", "🔍 Search", "📊 Analytics"])

# Tab 1: Find Similar
with tab1:
    st.markdown("### Find Similar Content")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_title = st.selectbox(
            "Select a movie or TV show",
            df["display_title"].tolist(),
            key="similar_select"
        )
        
        if selected_title:
            idx = df[df["display_title"] == selected_title].index[0]
            selected_item = df.iloc[idx]
            
            st.markdown(f"""
            <div class="card">
                <h4>{selected_item['title']}</h4>
                <p><strong>Type:</strong> {selected_item['type']} | <strong>Year:</strong> {selected_item['release_year']}</p>
                <p><strong>Genre:</strong> {selected_item['listed_in']}</p>
                <p>{selected_item['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Find Similar", type="primary"):
                with st.spinner("Finding recommendations..."):
                    recs = recommend_similar(idx, df, tfidf_matrix, top_n=5)
                
                if not recs.empty:
                    st.markdown("### Recommended for You")
                    for i, (_, row) in enumerate(recs.iterrows(), 1):
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <h4>{row['title']} <span style="font-size: 0.9rem; color: #666;">({row['type']}, {row['release_year']})</span></h4>
                                    <p><span class="badge">⭐ {row['similarity']:.1%} Match</span></p>
                                    <p><strong>Genre:</strong> {row['listed_in']}</p>
                                    <p>{row['description'][:150]}...</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Quick Stats")
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df)}</h3>
            <p>Total Titles</p>
        </div>
        """, unsafe_allow_html=True)
        
        movies = len(df[df["type"] == "Movie"])
        shows = len(df[df["type"] == "TV Show"])
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{movies}</h3>
            <p>Movies</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{shows}</h3>
            <p>TV Shows</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Search
with tab2:
    st.markdown("### Search by Keywords")
    
    query = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g., action, comedy, drama, sci-fi",
        key="search_query"
    )
    
    if query:
        with st.spinner("Searching..."):
            recs = recommend_by_keywords(query, df, vectorizer, tfidf_matrix, top_n=8)
        
        if not recs.empty:
            cols = st.columns(2)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="card">
                        <h4>{row['title']}</h4>
                        <p><span class="badge">{row['type']}</span> <span class="badge" style="background: #4CAF50;">{row['release_year']}</span></p>
                        <p><strong>Match:</strong> {row['similarity']:.1%}</p>
                        <p><strong>Genre:</strong> {row['listed_in']}</p>
                        <p>{row['description'][:100]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No results found. Try different keywords.")

# Tab 3: Analytics
with tab3:
    st.markdown("### Data Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Content Type Distribution
        type_counts = df["type"].value_counts()
        st.bar_chart(type_counts)
    
    with col2:
        # Top Genres
        genres = []
        for g in df["listed_in"]:
            genres.extend([x.strip() for x in str(g).split(",")])
        genre_counts = pd.Series(genres).value_counts().head(10)
        st.bar_chart(genre_counts)
    
    with col3:
        # Year Distribution
        year_counts = df["release_year"].value_counts().sort_index()
        st.line_chart(year_counts)
    
    # Data Preview
    st.markdown("### Dataset Preview")
    st.dataframe(df[["title", "type", "release_year", "rating", "listed_in"]].head(10), 
                use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🎬</h2>
        <h3>Netflix AI</h3>
        <p>Recommender System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🔍 How It Works")
    st.markdown("""
    1. **Content Analysis**: Analyzes movie/show metadata
    2. **TF-IDF Vectorization**: Converts text to numerical vectors
    3. **Cosine Similarity**: Finds most similar content
    4. **Personalized Recommendations**: Shows you what you'll love
    """)
    
    st.divider()
    
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Titles", len(df))
    st.metric("Movies", len(df[df["type"] == "Movie"]))
    st.metric("TV Shows", len(df[df["type"] == "TV Show"]))
    
    st.divider()
    
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **Streamlit**: Frontend
    - **Scikit-learn**: ML Algorithms
    - **Pandas**: Data Processing
    - **TF-IDF**: Feature Extraction
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎬 Netflix AI Recommender | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
