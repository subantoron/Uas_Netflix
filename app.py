# -----------------------------
# Custom CSS - Netflix Style with Yellow Accent - DIPERBAIKI
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

/* TYPOGRAPHY - SUPER CLEAR & BOLD - DIPERBAIKI */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Netflix Sans', sans-serif !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    line-height: 1.2 !important;
    margin-bottom: 1.5rem !important;
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

/* INPUT & SELECT BOX - DIPERBAIKI JELAS */
.stTextInput > div > div > input {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    border: 3px solid #333333 !important;
    padding: 1.2rem 1.8rem !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
}

.stTextInput > div > div > input:focus {
    border-color: #FFD700 !important;
    box-shadow: 0 0 0 4px rgba(255, 215, 0, 0.2) !important;
    background: #1A1A1A !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: #999999 !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

/* SELECTBOX - DIPERBAIKI JELAS */
.stSelectbox > div > div {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    border: 3px solid #333333 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 1.2rem !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
}

.stSelectbox > div > div:hover {
    border-color: #FFD700 !important;
    background: #1A1A1A !important;
}

.stSelectbox > div > div > div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* OPTIONS DROPDOWN */
[data-baseweb="popover"] {
    background: #0F0F0F !important;
    border: 3px solid #FFD700 !important;
    border-radius: 12px !important;
    box-shadow: 0 15px 35px rgba(255, 215, 0, 0.2) !important;
}

[data-baseweb="menu"] li {
    background: #0F0F0F !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    padding: 1rem 1.5rem !important;
    border-bottom: 1px solid #333333 !important;
}

[data-baseweb="menu"] li:hover {
    background: #1A1A1A !important;
    color: #FFD700 !important;
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

/* TABS - DIPERBAIKI */
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
    background: rgba(20, 20, 20, 0.95) !important;
    backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 2px solid #333333;
    padding: 2.5rem;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    margin-bottom: 2rem;
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

/* LABEL UNTUK INPUT - DIPERBAIKI JELAS */
label {
    color: #FFD700 !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    margin-bottom: 0.8rem !important;
    display: block !important;
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

/* SPECIAL: INPUT GROUP YANG JELAS */
.input-group {
    margin-bottom: 2rem;
    padding: 2rem;
    background: rgba(15, 15, 15, 0.9);
    border-radius: 16px;
    border: 2px solid #333333;
}

.input-group label {
    display: block;
    color: #FFD700 !important;
    font-weight: 900;
    font-size: 1.3rem;
    margin-bottom: 1rem;
}

.input-group .stTextInput,
.input-group .stSelectbox {
    margin-bottom: 1.5rem;
}

/* SLIDER - DIPERBAIKI */
.stSlider > div > div {
    background: #1A1A1A !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.stSlider > div > div > div {
    color: #FFD700 !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(135deg, #E50914 0%, #FFD700 100%) !important;
}

/* CHECKBOX - DIPERBAIKI */
.stCheckbox > label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

.stCheckbox > label > div:first-child {
    background: #1A1A1A !important;
    border: 2px solid #333333 !important;
}

.stCheckbox > label > div:first-child:hover {
    border-color: #FFD700 !important;
}

/* SEARCH SECTION - KHUSUS DIPERBAIKI */
.search-section {
    background: linear-gradient(145deg, #141414, #0F0F0F);
    border-radius: 20px;
    padding: 3rem;
    margin: 2rem 0;
    border: 3px solid #333333;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
}

.search-title {
    color: #FFD700 !important;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    margin-bottom: 2rem !important;
    text-align: center;
}

.search-input-container {
    background: #0F0F0F;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 2px solid #444444;
}

.search-label {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.3rem !important;
    margin-bottom: 1rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Page: Recommendation - DIPERBAIKI BAGIAN PENCARIAN
# -----------------------------
if page == "🎯 REKOMENDASI":
    tabs = st.tabs(["🎬 BERDASARKAN JUDUL", "🔍 BERDASARKAN KATA KUNCI", "⭐ KONTEN POPULER"])

    # Tab 1: BERDASARKAN JUDUL - DIPERBAIKI
    with tabs[0]:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="search-section">', unsafe_allow_html=True)
            
            # HEADER YANG JELAS
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 3rem;">
                    <div style="font-size: 4rem; color: #FFD700; margin-bottom: 1rem;">🎯</div>
                    <h2 style="color: #FFD700 !important; margin: 0; font-weight: 900;">PILIH KONTEN UNTUK DIREKOMENDASIKAN</h2>
                    <p style="color: #CCCCCC !important; margin-top: 1rem; font-size: 1.2rem;">
                        Pilih judul film atau series favorit Anda untuk mendapatkan rekomendasi serupa
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # SECTION 1: FILTER TIPE KONTEN - DIPERBAIKI
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown(
                '<div class="search-label">🎭 FILTER TIPE KONTEN</div>',
                unsafe_allow_html=True
            )
            
            filter_type_for_selector = st.selectbox(
                "Pilih tipe konten yang ingin Anda rekomendasikan:",
                options=type_options,
                index=0,
                help="Filter berdasarkan Movie atau TV Show",
                key="filter_type_title",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if filter_type_for_selector == "All":
                selector_df = df
            else:
                selector_df = df[df["type"] == filter_type_for_selector]

            # SECTION 2: PILIH JUDUL - DIPERBAIKI JELAS
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown(
                '<div class="search-label">📝 PILIH JUDUL KONTEN</div>',
                unsafe_allow_html=True
            )
            
            if len(selector_df) > 0:
                options = selector_df["display_title"].tolist()
                selected_display = st.selectbox(
                    "Pilih satu judul dari daftar di bawah ini:",
                    options=options,
                    index=0,
                    help="Pilih judul untuk mendapatkan rekomendasi konten serupa",
                    key="title_selector",
                )
                
                # Tampilkan info jumlah konten
                st.markdown(
                    f'<div style="color: #CCCCCC; font-size: 0.9rem; margin-top: 0.5rem;">'
                    f'📊 {len(options)} konten tersedia</div>',
                    unsafe_allow_html=True
                )
            else:
                selected_display = None
                ui_alert("warning", f"Tidak ada konten dengan tipe: <b>{filter_type_for_selector}</b>")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # SECTION 3: PENGATURAN REKOMENDASI - DIPERBAIKI
            st.markdown("### ⚙️ PENGATURAN REKOMENDASI")
            
            col_settings1, col_settings2, col_settings3 = st.columns(3)
            with col_settings1:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size: 1.1rem; color: #FFD700 !important; margin-bottom: 1rem; font-weight: 800;">📊 JUMLAH REKOMENDASI</div>',
                    unsafe_allow_html=True
                )
                top_n = st.slider(
                    "Pilih jumlah rekomendasi:",
                    5, 20, 10, 1,
                    label_visibility="collapsed",
                    key="top_n_slider"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col_settings2:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size: 1.1rem; color: #FFD700 !important; margin-bottom: 1rem; font-weight: 800;">🎯 FILTER TIPE</div>',
                    unsafe_allow_html=True
                )
                same_type = st.checkbox(
                    "Hanya rekomendasi dengan tipe yang sama",
                    value=True,
                    help="Movie hanya direkomendasikan Movie, TV Show hanya TV Show",
                    key="same_type_check"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col_settings3:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size: 1.1rem; color: #FFD700 !important; margin-bottom: 1rem; font-weight: 800;">⚡ TINDAKAN</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
                
            # SECTION 4: FILTER TAHUN - DIPERBAIKI
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown(
                '<div class="search-label">📅 FILTER TAHUN RILIS</div>',
                unsafe_allow_html=True
            )
            
            year_range = st.slider(
                "Pilih rentang tahun rilis untuk rekomendasi:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                format="%d",
                key="year_range_title",
            )
            year_min, year_max = year_range
            
            st.markdown(
                f'<div style="color: #CCCCCC; font-size: 0.9rem; margin-top: 1rem;">'
                f'📌 Rentang tahun: {year_min} - {year_max}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # BUTTON DAPATKAN REKOMENDASI - DIPERBAIKI
            st.markdown('</div>', unsafe_allow_html=True)  # Tutup search-section
            
            if selected_display:
                st.markdown('<div style="margin: 3rem 0;">', unsafe_allow_html=True)
                if st.button("🚀 DAPATKAN REKOMENDASI", type="primary", use_container_width=True, key="get_recs_btn"):
                    matches = df[df["display_title"] == selected_display]
                    if len(matches) == 0:
                        ui_alert("error", "Judul tidak ditemukan!")
                    else:
                        idx = matches.index[0]

                        st.divider()
                        st.markdown("### 🎬 KONTEN YANG DIPILIH")
                        selected_item = df.loc[idx]

                        # TAMPILKAN KONTEN YANG DIPILIH - DIPERBAIKI
                        col_selected1, col_selected2 = st.columns([2, 1])
                        with col_selected1:
                            st.markdown(
                                f"""
                                <div class="glass-panel">
                                    <h3 style="color: #FFD700 !important; margin-bottom: 1.5rem; font-weight: 900; font-size: 2.2rem;">
                                        {_safe_str(selected_item['title'])}
                                    </h3>
                                    <div style="display: flex; gap: 1.2rem; margin-bottom: 2rem; flex-wrap: wrap;">
                                        <span class="badge badge-movie" style="font-size: 1.1rem;">
                                            🎬 {_safe_str(selected_item['type'])}
                                        </span>
                                        <span class="badge badge-year" style="font-size: 1.1rem;">
                                            📅 {int(selected_item['release_year'])}
                                        </span>
                                        <span class="badge badge-rating" style="font-size: 1.1rem;">
                                            ⭐ {_safe_str(selected_item['rating'])}
                                        </span>
                                        <span class="badge" style="font-size: 1.1rem; background: linear-gradient(135deg, #2D2D2D 0%, #4A4A4A 100%) !important;">
                                            ⏱️ {_safe_str(selected_item['duration'])}
                                        </span>
                                    </div>
                                    <div style="color: #F5F5F5 !important; line-height: 1.8; padding: 2rem; 
                                            background: rgba(20, 20, 20, 0.8); border-radius: 12px; 
                                            border-left: 5px solid #FFD700; font-size: 1.1rem;">
                                        <strong style="color: #FFD700;">📖 Deskripsi:</strong><br>
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
                                        <div style="font-size: 1rem; color: #FFD700 !important; font-weight: 800;">🎭 GENRE</div>
                                        <div style="font-weight: 700; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem; line-height: 1.6;">
                                            {_safe_str(selected_item['listed_in']) or "Tidak tersedia"}
                                        </div>
                                    </div>
                                    <div style="margin-bottom: 2rem;">
                                        <div style="font-size: 1rem; color: #FFD700 !important; font-weight: 800;">🌍 NEGARA</div>
                                        <div style="font-weight: 700; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem;">
                                            {_safe_str(selected_item['country']) or "Tidak tersedia"}
                                        </div>
                                    </div>
                                    <div style="margin-bottom: 2rem;">
                                        <div style="font-size: 1rem; color: #FFD700 !important; font-weight: 800;">🎬 DIRECTOR</div>
                                        <div style="font-weight: 700; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem;">
                                            {_safe_str(selected_item['director']) or "Tidak tersedia"}
                                        </div>
                                    </div>
                                    <div>
                                        <div style="font-size: 1rem; color: #FFD700 !important; font-weight: 800;">🆔 ID</div>
                                        <div style="font-weight: 700; color: #FFFFFF !important; font-size: 1.1rem; margin-top: 0.8rem;">
                                            {_safe_str(selected_item.get('show_id','N/A'))}
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # PROSES REKOMENDASI
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
                                            <div style="margin-top: 0.8rem; color: #F5F5F5 !important;">
                                                Coba sesuaikan filter (tahun, tipe konten) untuk hasil lebih baik.
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="text-align: center; padding: 3rem; 
                                        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(255, 215, 0, 0.1) 100%);
                                        border-radius: 16px; margin: 3rem 0; border: 3px solid #FFD700;">
                                    <div style="font-size: 5rem; color: #FFD700; margin-bottom: 1.5rem;" class="floating">🎯</div>
                                    <h2 style="color: #FFD700 !important; margin: 0; font-weight: 900; font-size: 2.5rem;">
                                        {len(recs)} REKOMENDASI TERBAIK
                                    </h2>
                                    <p style="color: #F5F5F5 !important; margin: 1rem 0 0 0; font-size: 1.2rem;">
                                        Berdasarkan kemiripan konten (TF-IDF + Cosine Similarity)
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            for i, (_, r) in enumerate(recs.iterrows(), 1):
                                display_recommendation_card(r, i)

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                ui_alert("info", "Pilih judul terlebih dahulu untuk mendapatkan rekomendasi.")

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
                            <span style="color: #FFFFFF !important; font-weight: 700; font-size: 1rem;">
                                {genre[:20]}{'...' if len(genre)>20 else ''}
                            </span>
                            <span class="badge" style="font-size: 0.9rem; padding: 0.5rem 1rem; min-width: 45px; justify-content: center;">
                                {count}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Tab 2: BERDASARKAN KATA KUNCI - DIPERBAIKI
    with tabs[1]:
        st.markdown('<div class="search-section">', unsafe_allow_html=True)
        
        # HEADER YANG JELAS
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 3rem;">
                <div style="font-size: 4rem; color: #FFD700; margin-bottom: 1rem;">🔍</div>
                <h2 style="color: #FFD700 !important; margin: 0; font-weight: 900;">PENCARIAN DENGAN KATA KUNCI</h2>
                <p style="color: #CCCCCC !important; margin-top: 1rem; font-size: 1.2rem;">
                    Masukkan kata kunci untuk menemukan konten yang sesuai dengan preferensi Anda
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # SEARCH INPUT - DIPERBAIKI JELAS
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown(
            '<div class="search-label">🔎 MASUKKAN KATA KUNCI</div>',
            unsafe_allow_html=True
        )
        
        query = st.text_input(
            "Kata kunci pencarian (gunakan bahasa Inggris untuk hasil terbaik):",
            placeholder="Contoh: action adventure, romantic comedy, sci-fi, crime drama, fantasy",
            help="Masukkan genre, kata kunci, atau deskripsi konten yang Anda cari",
            label_visibility="collapsed",
            key="search_query",
        )
        
        st.markdown(
            '<div style="color: #CCCCCC; font-size: 0.9rem; margin-top: 0.5rem;">'
            '💡 Tips: Gunakan kata kunci spesifik untuk hasil yang lebih akurat</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # FILTER SECTION - DIPERBAIKI
        st.markdown("### ⚙️ FILTER PENCARIAN")
        
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 1.1rem; color: #FFD700 !important; margin-bottom: 1rem; font-weight: 800;">🎭 FILTER TIPE</div>',
                unsafe_allow_html=True
            )
            type_filter = st.selectbox(
                "Pilih tipe konten:",
                options=type_options,
                index=0,
                label_visibility="collapsed",
                key="type_filter_search"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_filter2:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 1.1rem; color: #FFD700 !important; margin-bottom: 1rem; font-weight: 800;">📊 JUMLAH HASIL</div>',
                unsafe_allow_html=True
            )
            top_n_q = st.slider(
                "Jumlah hasil yang ditampilkan:",
                5, 20, 10,
                label_visibility="collapsed",
                key="top_n_search"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_filter3:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 1.1rem; color: #FFD700 !important; margin-bottom: 1rem; font-weight: 800;">📅 TAHUN RILIS</div>',
                unsafe_allow_html=True
            )
            year_range_q = st.slider(
                "Rentang tahun:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="year_range_search",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        year_min_q, year_max_q = year_range_q
        
        # SEARCH BUTTON - DIPERBAIKI
        st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
        search_btn = st.button("🔍 LAKUKAN PENCARIAN", type="primary", use_container_width=True, key="search_btn")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Tutup search-section
        
        if search_btn:
            if not query:
                ui_alert("warning", "Masukkan kata kunci terlebih dahulu untuk melakukan pencarian")
            else:
                with st.spinner("🔎 Menganalisis kata kunci dan mencari konten terbaik..."):
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
                            <div style="display: flex; align-items: center; gap: 2rem;">
                                <div style="font-size: 4rem; color: #E50914;">🔍</div>
                                <div>
                                    <h3 style="margin: 0; color: #FFD700 !important;">TIDAK MENEMUKAN HASIL</h3>
                                    <div style="margin-top: 0.8rem; color: #F5F5F5 !important;">
                                        Untuk pencarian: <b>"{_safe_str(query)}"</b>
                                    </div>
                                    <div style="margin-top: 1.5rem; color: #CCCCCC !important;">
                                        💡 Tips:<br>
                                        1. Gunakan kata kunci dalam bahasa Inggris<br>
                                        2. Coba kata kunci yang lebih umum<br>
                                        3. Kurangi filter yang diterapkan
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                    )
                else:
                    ui_alert(
                        "success",
                        f"""
                        <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 4rem; color: #00B894; margin-bottom: 1rem;">🎉</div>
                            <div style="font-weight: 900; color: #00B894 !important; font-size: 2rem;">
                                {len(recs_q)} HASIL DITEMUKAN!
                            </div>
                            <div style="color: #F5F5F5 !important; margin-top: 1rem; font-size: 1.2rem;">
                                Untuk pencarian: <b>"{_safe_str(query)}"</b>
                            </div>
                        </div>
                        """,
                    )
                    
                    st.markdown(
                        f"""
                        <div style="text-align: center; margin: 3rem 0;">
                            <h3 style="color: #FFD700 !important; margin: 0; font-weight: 900; font-size: 1.8rem;">
                                🎬 REKOMENDASI BERDASARKAN PENCARIAN
                            </h3>
                            <p style="color: #CCCCCC !important; margin-top: 0.5rem;">
                                Disusun berdasarkan kemiripan dengan kata kunci Anda
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    for i, (_, r) in enumerate(recs_q.iterrows(), 1):
                        display_recommendation_card(r, i)
