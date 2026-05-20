# styles.py

# ==========================================
# 1. CSS UTAMA (FONTS, METRICS, BUTTONS)
# ==========================================
GLOBAL_CSS = """
<style>
    /* --- IMPORT GOOGLE FONT (POPPINS) --- */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Terapkan Font ke Seluruh Aplikasi */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* 2. Kartu Metrik (Kotak Putih & Shadow) */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        padding: 15px !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
        transition: all 0.3s ease !important;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(255, 0, 0, 0.1) !important;
        border-color: #FF0000 !important;
    }
    
    /* 3. Info Tambahan (Delta) */
    div[data-testid="stMetricDelta"] {
        opacity: 0; height: 0; transition: all 0.3s ease; transform: translateY(10px);
        display: flex; align-items: center;
    }
    div[data-testid="stMetric"]:hover div[data-testid="stMetricDelta"] {
        opacity: 1; height: auto; transform: translateY(0px);
    }
    div[data-testid="stMetricDelta"] svg { display: none !important; } 
    div[data-testid="stMetricDelta"] > div { font-weight: 600; margin-left: 0 !important; }

    /* 4. Font Angka & Label */
    div[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800 !important; color: #333; }
    div[data-testid="stMetricLabel"] { font-size: 14px !important; color: #666; }

    /* 5. Tombol Merah */
    div.stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.3s ease; }
    button[kind="primary"] { background-color: #FF0000; color: white; border: none; box-shadow: 0 4px 6px rgba(255,0,0,0.2); }
    button[kind="primary"]:hover { background-color: #cc0000; }

    /* 6. Sidebar & Layout */
    [data-testid="stSidebar"] > div > div { gap: 0.5rem; }
    hr { margin: 10px 0 !important; border-top: 1px solid #e0e0e0; }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
</style>
"""

# ==========================================
# 2. HTML COMPONENT
# ==========================================
SIDEBAR_TITLE_HTML = """
    <div style="margin-top: 0px; margin-bottom: 0px;">
        <h2 style="font-size: 20px; line-height: 1.1; margin: 0; padding: 1;">
            Analisis Ulasan <span style="color: #FF0000;">YT Music</span>
        </h2>
    </div>
"""

SIDEBAR_MODULE_BOX = """
<div style="padding: 15px; background-color: #ffffff; border: 1px solid #f0f0f0; border-radius: 10px; font-size: 16px; color: #444; margin-bottom: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
    <strong style="color: #ff0000;">Analisis Bigdata</strong>
</div>
"""

SIDEBAR_DESC = """<div style="text-align: justify; margin-bottom: 30px">Selamat datang! Aplikasi ini menggunakan <strong>Machine Learning</strong> untuk menganalisis sentimen dan topik dari ulasan pengguna aplikasi Yt Music.</div>"""

NAV_HEADER_HTML = """<div style="margin-bottom: 5px;"><h3 style="margin: 0px; padding: 0px; font-size: 18px; font-weight: 700; color: #31333F;">Navigasi</h3><p style="margin: 2px 0px 0px 0px; padding: 0px; font-size: 14px; color: gray;">Pilih Menu:</p></div>"""

FOOTER_ABOUT_HTML = """<div style="padding: 15px; background-color: #ffffff; border: 1px solid #f0f0f0; border-radius: 10px; font-size: 15px; color: #444;"><strong style="color: #ff0000;">Dibuat oleh:</strong> PT. Tambang Teks Tbk.<br><strong style="color: #ff0000;">Metode:</strong> SVM (Sentimen) & LDA (Topik)<br><strong style="color: #ff0000;">Data:</strong> Google Play Store</div>"""

# ==========================================
# 3. CONFIG & COLORS
# ==========================================
MENU_STYLES = {
    "container": { "padding": "0!important", "background-color": "transparent", "margin": "0!important" },
    "nav-link": { "font-size": "16px", "text-align": "left", "margin": "5px 0px", "padding": "10px 15px", "--hover-color": "#f9f9f9", "color": "#555" },
    "nav-link-selected": { "background-color": "#ffffff", "color": "#FF0000", "font-weight": "600", "border-left": "4px solid #FF0000", "border-radius": "4px", "box-shadow": "0 2px 4px rgba(0,0,0,0.05)" },
}

CHART_SENTIMENT_COLORS = ['#FF6B6B', '#4ECDC4']
TABLE_HIGHLIGHT_COLOR = '#ffcccc'

# ==========================================
# 4. WORDCLOUD & PILLS STYLE (UPDATED)
# ==========================================
# styles.py (Update bagian ini saja)

WORDCLOUD_STYLE = """
<style>
    .wc-card { background-color: #ffffff; padding: 20px 30px; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; margin-top: 10px; margin-bottom: 20px; }
    div[data-testid="stPills"] { gap: 10px; margin-top: 10px; }
</style>
"""

RESULT_ANIMATION_CSS = """
<style>
    @keyframes pulse-green { 0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); } 70% { box-shadow: 0 0 20px 10px rgba(40, 167, 69, 0); } 100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); } }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 20px 10px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
    
    .result-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 20px; font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; min-height: 80px; border-width: 2px; border-style: solid; cursor: default; }
    .box-pos { background-color: #f0fff4; color: #28a745; border-color: #28a745; }
    .box-pos:hover { animation: pulse-green 1.5s infinite; transform: translateY(-3px); transition: 0.3s; }
    .box-neg { background-color: #fff5f5; color: #FF0000; border-color: #FF0000; }
    .box-neg:hover { animation: pulse-red 1.5s infinite; transform: translateY(-3px); transition: 0.3s; }
    .box-topic { background-color: #e7f3fe; color: #0c5460; border-color: #b8daff; }
    .box-topic:hover { transform: translateY(-5px); border-color: #007bff; box-shadow: 0 5px 15px rgba(0, 123, 255, 0.25); }
</style>
"""

# ==========================================
# 5. KPI CARD CSS (CUSTOM)
# ==========================================
KPI_CARD_CSS = """
<style>
.kpi-card {
    border-radius: 12px; padding: 20px 24px; text-align: left;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 100%;
}
.kpi-card:hover { transform: translateY(-5px) scale(1.01); box-shadow: 0 10px 25px rgba(0,0,0,0.15); z-index: 2; }
.kpi-label { font-size: 16px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.kpi-value { font-size: 26px; font-weight: 800; margin-bottom: 6px; line-height: 1.2; }
.kpi-desc { font-size: 16px; font-weight: 500; opacity: 0.9; }

.card-green { background-color: #E6F4EA; color: #137333; border: 2px solid #E6F4EA; }
.card-green:hover { border: 2px solid #34A853; }
.card-red { background-color: #FCE8E6; color: #C5221F; border: 2px solid #FCE8E6; }
.card-red:hover { border: 2px solid #EA4335; }
.card-blue { background-color: #E8F0FE; color: #1967D2; border: 2px solid #E8F0FE; }
.card-blue:hover { border: 2px solid #4285F4; }
</style>
"""