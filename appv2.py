import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt # Import Altair global agar aman
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
from PIL import Image
import warnings

# Menekan UserWarning dari Altair agar konsol bersih
warnings.simplefilter(action='ignore', category=UserWarning)

# Import custom modules
import styles
try:
    from preprocessing import clean_and_preprocess
except ImportError:
    st.error("🚨 File 'preprocessing.py' tidak ditemukan!")
    st.stop()

# ==========================================
# 1. KONFIGURASI HALAMAN & FUNGSI GLOBAL
# ==========================================
try:
    img_favicon = Image.open("logo.png")
except FileNotFoundError:
    img_favicon = "🎵"

st.set_page_config(
    page_title="Analisis Ulasan YT Music",
    page_icon=img_favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(styles.GLOBAL_CSS, unsafe_allow_html=True)

# --- FUNGSI HELPER (WAJIB ADA) ---

@st.cache_data(show_spinner=False)
def generate_cached_wordcloud(text_input, colormap="viridis"):
    wc = WordCloud(
        width=800, 
        height=450, 
        background_color="white", 
        colormap=colormap, 
        contour_width=0
    ).generate(text_input)
    return wc.to_array()

def highlight_max_sentiment_visual(s):
    # Ubah string persentase ke float untuk perbandingan numerik
    try:
        s_float = s.str.rstrip('%').astype(float)
    except:
        return [''] * len(s)
    
    LIGHT_GREEN = '#E6F4EA' 
    LIGHT_RED = '#FFEBEE'    
    
    if s.name == '% Positif':
        is_max = s_float == s_float.max()
        return [f'background-color: {LIGHT_GREEN}' if v else '' for v in is_max] 
    elif s.name == '% Negatif':
        is_max = s_float == s_float.max()
        return [f'background-color: {LIGHT_RED}' if v else '' for v in is_max]
    return [''] * len(s)

@st.cache_data(show_spinner=False)
def calculate_dashboard_metrics(_svm_model, _tfidf_vec, _lda_model, _cv_vec, corpus):
    corpus_list = corpus.astype(str).tolist()
    X_tfidf = _tfidf_vec.transform(corpus_list)
    y_pred = _svm_model.predict(X_tfidf)
    X_cv = _cv_vec.transform(corpus_list)
    topic_dist = _lda_model.transform(X_cv)
    dominant_topics = np.argmax(topic_dist, axis=1)
    return y_pred, dominant_topics

# ==========================================
# 2. LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        svm = joblib.load("model_svm.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        lda = joblib.load("model_lda.pkl")
        cv = joblib.load("vectorizer_lda.pkl") 

        try:
            df_hist = pd.read_csv("data_dashboard_with_date.csv")
            df_hist['Tanggal'] = pd.to_datetime(df_hist['Tanggal'])
        except:
            df_hist = None
        try:
            corpus = pd.read_pickle("corpus.pkl")
        except:
            corpus = None
        return svm, tfidf, lda, cv, df_hist, corpus

    except Exception as e:
        return None, None, None, None, None, str(e)

svm_model, tfidf_vec, lda_model, cv_vec, df_hist, corpus_data = load_assets()

if isinstance(corpus_data, str): 
    st.error(f"🚨 Gagal Memuat Aset Model: {corpus_data}")
    st.stop()

# ==========================================
# 3. DEFINISI TOPIK & DATA
# ==========================================
NUM_TOPICS = 4
TOPIC_LABELS = {
    0: "Kepuasan dan Akses",
    1: "Iklan Fitur Premium",
    2: "Apresiasi Kualitas Aplikasi",
    3: "Pengalaman Dengar Lagu"
}

TOPIC_KEYWORDS = {
    0: ["mantap", "music", "suka", "good", "musik", "youtube", "you", "hibur", "bantu", "mudah"], 
    1: ["musik", "lagu", "iklan", "premium", "keren", "kalo", "putar", "download", "pakai", "bayar"],
    2: ["bagus", "banget", "kasih", "coba", "bintang", "google", "terima", "enak", "lumayan", "muas"],
    3: ["oke", "lagu", "dengar", "puas", "banget", "musik", "lengkap", "the", "top", "nice"]
}

# ==========================================
# 4. ANTARMUKA STREAMLIT
# ==========================================
try:
    logo = Image.open("logo.png")
except FileNotFoundError:
    logo = "https://upload.wikimedia.org/wikipedia/commons/d/d8/YouTube_Music_icon_%282020%29.svg"

col_img, col_text = st.sidebar.columns([1, 3.5]) 
with col_img: st.image(logo, width=55)
with col_text: st.markdown(styles.SIDEBAR_TITLE_HTML, unsafe_allow_html=True)

st.sidebar.markdown(styles.SIDEBAR_MODULE_BOX, unsafe_allow_html=True)
st.sidebar.markdown(styles.SIDEBAR_DESC, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(styles.NAV_HEADER_HTML, unsafe_allow_html=True)
    menu = option_menu(
        menu_title=None,
        options=["Dashboard Evaluasi", "Prediksi Real-time"], 
        icons=['📊', '🔍'], 
        default_index=0,
        orientation="vertical",
        styles=styles.MENU_STYLES
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Tentang App")
st.sidebar.markdown(styles.FOOTER_ABOUT_HTML, unsafe_allow_html=True)

# ------------------------------------------
# HALAMAN 1: DASHBOARD EVALUASI
# ------------------------------------------
if menu == "Dashboard Evaluasi":
    st.title("📊 Dashboard Evaluasi Model")

    # Hitung Data (Cached)
    if corpus_data is not None:
        y_pred_all, dominant_topics = calculate_dashboard_metrics(
            svm_model, tfidf_vec, lda_model, cv_vec, corpus_data
        )
    else:
        y_pred_all, dominant_topics = [], []

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="🎯 Akurasi", value="79.19%", delta="High Performance")
    with col2: st.metric(label="⚖️ Precision", value="0.85", delta="Stabil")
    with col3: st.metric(label="📡 Recall", value="0.79", delta="Normal")
    with col4: st.metric(label="⭐ F1-Score", value="0.81", delta="Optimal")
    st.write("")

    # 2. Confusion Matrix & Pie Chart
    with st.container(border=True):
        st.subheader("📈 Performa Model & Distribusi")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🎯 Confusion Matrix")
            st.caption("Split Data (70:30)")
            cm_data = np.array([[3259, 1003], [3668, 14517]])
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], ax=ax_cm, cbar=False)
            st.pyplot(fig_cm); plt.close(fig_cm)
            
        with c2:
            st.markdown("##### 🥧 Proporsi Sentimen")
            if len(y_pred_all) > 0:
                unique, counts = np.unique(y_pred_all, return_counts=True)
                sentimen_counts = dict(zip(unique, counts))
                pos, neg = sentimen_counts.get(1, 0), sentimen_counts.get(0, 0)
                
                fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                ax_pie.pie([neg, pos], labels=['Negatif', 'Positif'], autopct='%1.1f%%', startangle=90, colors=styles.CHART_SENTIMENT_COLORS, explode=(0.05, 0))
                st.pyplot(fig_pie); plt.close(fig_pie)
    st.write("")

    # GRAFIK TREND (REAL DATA)
    with st.container(border=True):
        st.subheader("📈 Trend Volume Sentimen")
        
        if df_hist is not None:
            c_filter, c_blank = st.columns([1, 3]) 
            with c_filter:
                freq_opsi = st.pills(
                    "Pilih Rentang Waktu:", 
                    options=["Harian", "Bulanan"], 
                    default="Harian",
                    selection_mode="single",
                    label_visibility="collapsed",
                    key="pills_dash_trend"
                )
            if not freq_opsi: freq_opsi = "Harian"
            freq_code = 'D' if freq_opsi == "Harian" else 'ME'

            df_trend = df_hist.copy()
            df_trend['Sentimen'] = df_trend['Sentimen_Prediksi']
            
            trend_chart_data = df_trend.groupby([pd.Grouper(key='Tanggal', freq=freq_code), 'Sentimen']).size().unstack(fill_value=0)
            trend_chart_data = trend_chart_data.reindex(columns=['Negatif', 'Positif'], fill_value=0)
            st.line_chart(trend_chart_data, color=styles.CHART_SENTIMENT_COLORS)
            
            c_info1, c_info2 = st.columns(2)
            with c_info1:
                if not trend_chart_data.empty:
                    peak_idx = trend_chart_data.sum(axis=1).idxmax()
                    fmt = '%d %B %Y' if freq_opsi == "Harian" else '%B %Y'
                    lbl = "Tanggal Tersibuk" if freq_opsi == "Harian" else "Bulan Tersibuk"
                    st.info(f"📅 **{lbl}:** {peak_idx.strftime(fmt)}")
            with c_info2:
                if not trend_chart_data.empty:
                    avg_vol = int(trend_chart_data.sum(axis=1).mean())
                    satuan = "ulasan/hari" if freq_opsi == "Harian" else "ulasan/bulan"
                    st.info(f"📊 **Rata-rata Volume:** {avg_vol} {satuan}")
        else:
            st.warning("Data history (data_dashboard_with_date.csv) tidak ditemukan.")

    # ANALISIS TOPIK (DASHBOARD)
    with st.container(border=True):
        st.subheader("📉 Analisis Sentimen per Topik")
        if len(y_pred_all) > 0 and len(dominant_topics) > 0:
            df_viz = pd.DataFrame({'Topik Code': dominant_topics, 'Sentimen Code': y_pred_all})
            df_viz['Nama Topik'] = df_viz['Topik Code'].map(TOPIC_LABELS)
            df_viz['Sentimen'] = df_viz['Sentimen Code'].map({0: 'Negatif', 1: 'Positif'})
            
            cross_tab = pd.crosstab(df_viz['Nama Topik'], df_viz['Sentimen'])
            cross_tab = cross_tab.reindex([TOPIC_LABELS[i] for i in range(NUM_TOPICS)], fill_value=0)
            cross_tab_pct = cross_tab.div(cross_tab.sum(1), axis=0) * 100
            
            for col in ['Negatif', 'Positif']:
                if col not in cross_tab_pct.columns: cross_tab_pct[col] = 0
            
            df_display = cross_tab.copy()
            df_display['Total'] = df_display.sum(axis=1)
            df_display['% Negatif'] = (df_display['Negatif'] / df_display['Total'] * 100).fillna(0).replace([np.inf, -np.inf], 0).round(1).astype(str) + '%'
            df_display['% Positif'] = (df_display['Positif'] / df_display['Total'] * 100).fillna(0).replace([np.inf, -np.inf], 0).round(1).astype(str) + '%'

            col_chart, col_info = st.columns([2, 1])
            with col_chart:
                fig, ax = plt.subplots(figsize=(8, 5))
                cross_tab[['Negatif', 'Positif']].plot(kind='bar', stacked=True, color=styles.CHART_SENTIMENT_COLORS, ax=ax, width=0.7)
                for c in ax.containers: ax.bar_label(c, label_type='center', color='white', fontsize=9, fmt='%d')
                ax.set_title("Distribusi Sentimen Berdasarkan Topik"); ax.set_xlabel(""); plt.xticks(rotation=15, ha='right'); plt.legend(title='Sentimen', loc='upper right')
                st.pyplot(fig); plt.close(fig)
            
            with col_info:
                st.info("💡 **Insight:**")
                if 'Negatif' in cross_tab_pct.columns and cross_tab_pct['Negatif'].max() > 0:
                    most_neg = cross_tab_pct['Negatif'].idxmax()
                    val = cross_tab_pct.loc[most_neg, 'Negatif']
                    st.warning(f"Rawan Komplain:\n\n**{most_neg}**\n\n({val:.1f}% negatif)")
                else: st.success("Luar biasa! Tidak ada keluhan signifikan.")
                
                st.dataframe(df_display[['Negatif', 'Positif', 'Total', '% Negatif', '% Positif']].style.apply(highlight_max_sentiment_visual, axis=0, subset=['% Negatif', '% Positif']), use_container_width=True)
    st.write("")

    # WORDCLOUD (DASHBOARD)
    st.markdown(styles.WORDCLOUD_STYLE, unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("☁️ WordCloud Kata Kunci Topik")
        st.caption("Visualisasi kata-kata dominan per topik.")
        
        # CSS untuk Vertical Pills (Khusus di sini)
        st.markdown("""
        <style>
        div[data-testid="stPills"] > div { flex-direction: column !important; gap: 8px !important; }
        div[data-testid="stPills"] button { width: 100% !important; text-align: left !important; }
        </style>
        """, unsafe_allow_html=True)
        
        c_pills, c_wc = st.columns([1, 2.5], gap="large", vertical_alignment="top")
        with c_pills:
            st.markdown("##### 🎛️ Pilih Topik")
            options_labels = [TOPIC_LABELS[i] for i in range(NUM_TOPICS)]
            selected_label = st.pills("Filter Topik", options=options_labels, default=options_labels[0], selection_mode="single", label_visibility="collapsed", key="pills_wc_dash")
        
        with c_wc:
            if selected_label:
                found_ids = [k for k, v in TOPIC_LABELS.items() if v == selected_label]
                if found_ids:
                    sel_idx = found_ids[0]
                    kw = TOPIC_KEYWORDS.get(sel_idx, ["data"])
                    wc_array = generate_cached_wordcloud(" ".join(kw * 10), "viridis")
                    st.image(wc_array, use_container_width=True)

# ------------------------------------------
# HALAMAN 2: PREDIKSI REAL-TIME
# ------------------------------------------
elif menu == "Prediksi Real-time":
    st.title("🔍 Prediksi Sentimen & Topik")
    st.write("Analisis ulasan baru secara manual atau batch (CSV).")
    tab_manual, tab_csv = st.tabs(["✍️ Input Manual", "📂 Upload CSV"])

    with tab_manual:
        user_input = st.text_area("Masukkan ulasan:", height=100, label_visibility="collapsed", placeholder="Ketik ulasan di sini...")
        if st.button("Analisis", type="primary"):
            if user_input.strip():
                with st.spinner("Memproses..."):
                    clean = clean_and_preprocess(user_input)
                    with st.expander("Lihat Hasil Preprocessing"): st.text(clean)
                    if clean:
                        vec_s = tfidf_vec.transform([clean]); pred_s = svm_model.predict(vec_s)[0]
                        vec_l = cv_vec.transform([clean]); topic_d = lda_model.transform(vec_l)[0]
                        top_topic = np.argmax(topic_d); conf = topic_d[top_topic]
                        
                        st.markdown(styles.RESULT_ANIMATION_CSS, unsafe_allow_html=True)
                        st.write("") 
                        with st.container(border=True):
                            st.subheader("📝 Hasil Analisis")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("##### Sentimen")
                                if pred_s == 1: st.markdown('<div class="result-box box-pos">Positif 😊</div>', unsafe_allow_html=True)
                                else: st.markdown('<div class="result-box box-neg">Negatif 😡</div>', unsafe_allow_html=True)
                            with c2:
                                st.markdown("##### Topik Utama")
                                st.markdown(f'<div class="result-box box-topic">📂 {TOPIC_LABELS[top_topic]}</div>', unsafe_allow_html=True)
                                st.caption(f"Confidence: {conf:.1%}")
                                st.write(f"**Keywords:** {', '.join(TOPIC_KEYWORDS[top_topic][:5])}")
                            st.write(""); st.markdown("##### 📊 Probabilitas Topik")
                            st.dataframe(pd.DataFrame({"Nama Topik": [TOPIC_LABELS[i] for i in range(NUM_TOPICS)], "Probabilitas": topic_d}), 
                                       column_config={"Probabilitas": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1)}, hide_index=True, use_container_width=True)
                    else: st.error("Teks tidak valid.")
            else: st.warning("Mohon isi teks.")

    with tab_csv:
        st.subheader("Analisis Batch (CSV)")
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        
        if uploaded_file:
            file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            if "file_id" not in st.session_state or st.session_state["file_id"] != file_id:
                st.session_state["file_id"] = file_id; st.session_state["csv_results"] = None; st.session_state["csv_col"] = None
                if "filter_apps" in st.session_state: del st.session_state["filter_apps"]
                if "previous_selection" in st.session_state: del st.session_state["previous_selection"]

        if uploaded_file and st.session_state.get("csv_results") is None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            c1, c2, c3 = st.columns(3)
            with c1: text_col = st.selectbox("Pilih kolom ulasan:", options=df.columns.tolist())
            with c2: 
                dc = [c for c in df.columns if 'at' in c.lower() or 'date' in c.lower()]
                date_col = st.selectbox("Pilih kolom tanggal:", options=[None] + df.columns.tolist(), index=df.columns.get_loc(dc[0])+1 if dc else 0)
            with c3:
                ac = [c for c in df.columns if 'app' in c.lower() or 'name' in c.lower()]
                app_col = st.selectbox("Pilih kolom nama app:", options=[None] + df.columns.tolist(), index=df.columns.get_loc(ac[0])+1 if ac else 0)
            
            if st.button("Mulai Proses", type="primary"):
                progress_bar = st.progress(0); status_text = st.empty(); status_text.text("1/3 Preprocessing...")
                clean_texts = []
                total_rows = len(df)
                for i, txt in enumerate(df[text_col]):
                    clean_texts.append(clean_and_preprocess(str(txt)))
                    if i % max(1, int(total_rows*0.1)) == 0: progress_bar.progress(int((i/total_rows)*30))
                
                df['text_clean'] = clean_texts
                if date_col: df['Tanggal'] = pd.to_datetime(df[date_col], errors='coerce')
                if app_col: df['App_Name'] = df[app_col].astype(str)
                progress_bar.progress(30); status_text.text("2/3 Prediksi Sentimen...")
                df['Sentimen_Code'] = svm_model.predict(tfidf_vec.transform(df['text_clean']))
                df['Sentimen'] = df['Sentimen_Code'].map({1:'Positif', 0:'Negatif'})
                progress_bar.progress(60); status_text.text("3/3 Klasifikasi Topik...")
                tr = lda_model.transform(cv_vec.transform(df['text_clean']))
                df['Topik_Code'] = np.argmax(tr, axis=1)
                df['Nama_Topik'] = df['Topik_Code'].map(TOPIC_LABELS)
                progress_bar.progress(100); status_text.success("Selesai!"); st.session_state["csv_results"] = df; st.session_state["csv_col"] = text_col; st.rerun()

        if st.session_state.get("csv_results") is not None:
            df_master = st.session_state["csv_results"]; text_col = st.session_state["csv_col"]

            # Filter Logic
            if 'App_Name' in df_master.columns:
                st.markdown("##### 🕵️ Filter Aplikasi")
                app_list = sorted(df_master['App_Name'].unique().tolist()); all_opts = ["Semua"] + app_list 
                if "filter_apps" not in st.session_state: st.session_state["filter_apps"] = ["Semua"]
                if "previous_selection" not in st.session_state: st.session_state["previous_selection"] = ["Semua"]

                def update_filter_logic():
                    curr = st.session_state["filter_apps"]; prev = st.session_state["previous_selection"]
                    if "Semua" in curr and len(curr) > 1:
                        st.session_state["filter_apps"] = ["Semua"] if "Semua" not in prev else [x for x in curr if x != "Semua"]
                    elif not curr: st.session_state["filter_apps"] = ["Semua"]
                    st.session_state["previous_selection"] = st.session_state["filter_apps"]

                st.pills("Pilih Aplikasi:", options=all_opts, selection_mode="multi", key="filter_apps", on_change=update_filter_logic, label_visibility="collapsed")
                sel_apps = st.session_state["filter_apps"]
                df = df_master if "Semua" in sel_apps else df_master[df_master['App_Name'].isin(sel_apps)]
            else: df = df_master

            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            with c1: st.success(f"✅ Selesai! ({len(df)} data)")
            with c2: 
                if st.button("Hapus Analisis", use_container_width=True): st.session_state["csv_results"] = None; st.rerun()
            
            st.write(""); st.subheader("📊 Laporan Analisis")
            tot, pos, neg = len(df), len(df[df['Sentimen_Code']==1]), len(df[df['Sentimen_Code']==0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", f"{tot:,}"); m2.metric("Positif", f"{pos:,}", f"{(pos/tot)*100:.1f}%" if tot>0 else "0%"); m3.metric("Negatif", f"{neg:,}", f"{(neg/tot)*100:.1f}%" if tot>0 else "0%", delta_color="inverse")
            st.write("")

            with st.container(border=True):
                st.subheader("📈 Ringkasan Statistik")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Sentimen**")
                    if tot > 0: fig, ax = plt.subplots(figsize=(4,4)); ax.pie([neg, pos], labels=['Negatif','Positif'], autopct='%1.1f%%', colors=styles.CHART_SENTIMENT_COLORS, startangle=90); st.pyplot(fig); plt.close(fig)
                with c2:
                    st.write("**Topik Terbanyak**")
                    if tot > 0:
                        tc = df['Nama_Topik'].value_counts().reset_index(); tc.columns=['Topik','Jumlah']; tc['%'] = tc['Jumlah']/tc['Jumlah'].sum()*100
                        st.dataframe(tc, column_config={"%": st.column_config.ProgressColumn("Dominasi", format="%.1f%%", min_value=0, max_value=100)}, hide_index=True, use_container_width=True)
            st.write("")

            # KOMPARASI (ALTAIR)
            is_comp = 'App_Name' in df.columns and ("Semua" in st.session_state.get("filter_apps", []) or len(st.session_state.get("filter_apps", [])) > 1)
            if is_comp:
                with st.container(border=True):
                    st.subheader("📊 Komparasi Performa Aplikasi")
                    cross = pd.crosstab(df['App_Name'], df['Sentimen'])
                    if not cross.empty:
                        cp = cross.div(cross.sum(1), axis=0)*100; df_melt = cp.reset_index().melt(id_vars='App_Name', var_name='Sentimen', value_name='Persentase')
                        cts = df.groupby(['App_Name', 'Sentimen']).size().reset_index(name='Jumlah')
                        df_melt = pd.merge(df_melt, cts, on=['App_Name', 'Sentimen'], how='left').fillna(0)
                        df_melt['Order'] = df_melt['Sentimen'].map({'Negatif':1, 'Positif':2}); df_melt.sort_values(['App_Name','Order'], inplace=True)
                        df_melt['Cumsum'] = df_melt.groupby('App_Name')['Persentase'].cumsum()
                        df_melt['X_Text'] = df_melt['Cumsum'] - (df_melt['Persentase']/2)
                        df_melt['Label'] = df_melt['Persentase'].apply(lambda x: f"{x:.0f}%" if x>5 else "")

                        av = df['App_Name'].value_counts()
                        best = df_melt[df_melt['Sentimen']=='Positif'].sort_values('Persentase', ascending=False).iloc[0] if not df_melt[df_melt['Sentimen']=='Positif'].empty else {'App_Name':'-','Persentase':0}
                        worst = df_melt[df_melt['Sentimen']=='Negatif'].sort_values('Persentase', ascending=False).iloc[0] if not df_melt[df_melt['Sentimen']=='Negatif'].empty else {'App_Name':'-','Persentase':0}
                        
                        st.markdown(styles.KPI_CARD_CSS, unsafe_allow_html=True)
                        k1, k2, k3 = st.columns(3)
                        with k1: st.markdown(f'<div class="kpi-card card-green"><div class="kpi-label">🏆 Disukai</div><div class="kpi-value">{best["App_Name"]}</div><div class="kpi-desc">{best["Persentase"]:.1f}% positif</div></div>', unsafe_allow_html=True)
                        with k2: 
                            if worst['Persentase']>0: st.markdown(f'<div class="kpi-card card-red"><div class="kpi-label">⚠️ Keluhan</div><div class="kpi-value">{worst["App_Name"]}</div><div class="kpi-desc">{worst["Persentase"]:.1f}% negatif</div></div>', unsafe_allow_html=True)
                            else: st.markdown(f'<div class="kpi-card card-green"><div class="kpi-label">🛡️ Aman</div><div class="kpi-value">Zero Issue</div><div class="kpi-desc">0% negatif</div></div>', unsafe_allow_html=True)
                        with k3: st.markdown(f'<div class="kpi-card card-blue"><div class="kpi-label">📢 Populer</div><div class="kpi-value">{av.idxmax()}</div><div class="kpi-desc">{av.max()} ulasan</div></div>', unsafe_allow_html=True)
                        st.write("")

                        st.caption("Proporsi Sentimen (%):")
                        hvr = alt.selection_point(on='mouseover', nearest=False, empty=False)
                        bs = alt.Chart(df_melt).encode(y=alt.Y('App_Name', title=None, sort='-x'), order='Order')
                        brs = bs.mark_bar(size=40).encode(x=alt.X('Persentase', stack='zero', axis=None), color=alt.Color('Sentimen', scale=alt.Scale(domain=['Negatif','Positif'], range=styles.CHART_SENTIMENT_COLORS), legend=alt.Legend(orient='bottom')), stroke=alt.condition(hvr, alt.value('white'), alt.value('transparent')), strokeWidth=alt.condition(hvr, alt.value(3), alt.value(0)), tooltip=['App_Name','Sentimen',alt.Tooltip('Persentase',format='.1f'),'Jumlah']).add_params(hvr)
                        txt = bs.mark_text(dy=0, color='white', fontWeight='bold').encode(x='X_Text', text='Label')
                        st.altair_chart((brs+txt).properties(height=max(180, len(cross)*60)).configure_view(strokeWidth=0).configure_axis(grid=False), use_container_width=True)
            elif 'App_Name' in df.columns: st.info("Pilih opsi 'Semua' atau lebih dari satu aplikasi untuk komparasi.")
            st.write("")

            # TREND CHART
            if 'Tanggal' in df.columns and not df['Tanggal'].isnull().all():
                with st.container(border=True):
                    st.subheader("📅 Trend Volume")
                    c_flt, _ = st.columns([1, 3])
                    with c_flt: f_opsi = st.pills("Rentang:", ["Harian", "Bulanan"], default="Harian", key="p_tr_csv", selection_mode="single", label_visibility="collapsed") or "Harian"
                    fc = 'D' if f_opsi == "Harian" else 'ME'
                    dft = df.copy().set_index('Tanggal')
                    td = dft.groupby([pd.Grouper(freq=fc), 'Sentimen']).size().unstack(fill_value=0).reindex(columns=['Negatif','Positif'], fill_value=0)
                    st.line_chart(td, color=styles.CHART_SENTIMENT_COLORS)
                    
                    if not td.empty:
                        pidx = td.sum(axis=1).idxmax()
                        fmt = '%d %B %Y' if f_opsi=="Harian" else '%B %Y'
                        lbl = "Tanggal" if f_opsi=="Harian" else "Bulan"
                        c1, c2 = st.columns(2)
                        c1.info(f"📅 **{lbl} Tersibuk:** {pidx.strftime(fmt)}")
                        c2.info(f"📊 **Rata-rata:** {int(td.sum(axis=1).mean())} ulasan/{lbl.lower()}")
            st.write("")

            # TOPIC ANALYSIS (CSV)
            with st.container(border=True):
                st.subheader("📉 Sentimen per Topik")
                if tot > 0:
                    ct = pd.crosstab(df['Nama_Topik'], df['Sentimen']).reindex([TOPIC_LABELS[i] for i in range(NUM_TOPICS)], fill_value=0)
                    ct_pct = ct.div(ct.sum(1), axis=0)*100
                    df_d = ct.copy(); df_d['Total'] = df_d.sum(axis=1)
                    for c in ['Negatif','Positif']:
                        if c not in df_d: df_d[c]=0
                        if c not in ct_pct: ct_pct[c]=0
                    
                    df_d['% Negatif'] = (df_d['Negatif']/df_d['Total']*100).fillna(0).replace([np.inf, -np.inf], 0).round(1).astype(str)+'%'
                    df_d['% Positif'] = (df_d['Positif']/df_d['Total']*100).fillna(0).replace([np.inf, -np.inf], 0).round(1).astype(str)+'%'

                    c1, c2 = st.columns([2,1])
                    with c1:
                        fig, ax = plt.subplots(figsize=(8,5))
                        ct.plot(kind='bar', stacked=True, color=styles.CHART_SENTIMENT_COLORS, ax=ax, width=0.7)
                        for c in ax.containers: ax.bar_label(c, label_type='center', color='white', fontsize=9, fmt='%d')
                        ax.set_xlabel(""); plt.xticks(rotation=15, ha='right'); st.pyplot(fig); plt.close(fig)
                    with c2:
                        st.info("💡 **Insight:**")
                        if ct_pct['Negatif'].max()>0:
                            mn = ct_pct['Negatif'].idxmax(); vl = ct_pct.loc[mn, 'Negatif']
                            st.warning(f"Rawan Komplain:\n\n**{mn}**\n\n({vl:.1f}% negatif)")
                        else: st.success("Aman! Tidak ada keluhan.")
                        st.dataframe(df_d[['Negatif','Positif','Total','% Negatif','% Positif']].style.apply(highlight_max_sentiment_visual, axis=0, subset=['% Negatif','% Positif']), use_container_width=True)
            st.write("")

            # WORDCLOUD CSV (VERTIKAL)
            st.markdown(styles.WORDCLOUD_STYLE, unsafe_allow_html=True)
            with st.container(border=True):
                st.subheader("☁️ WordCloud Kata Kunci")
                st.markdown("""<style>div[data-testid="stPills"] > div {flex-direction: column !important; gap: 8px !important;} div[data-testid="stPills"] button {width: 100% !important; text-align: left !important;}</style>""", unsafe_allow_html=True)
                
                c_pills, c_wc = st.columns([1, 2.5], gap="large", vertical_alignment="top")
                with c_pills:
                    st.markdown("##### 🎛️ Pilih Topik")
                    tfile = set(df['Nama_Topik'].dropna().unique())
                    topt = [TOPIC_LABELS[i] for i in range(NUM_TOPICS) if TOPIC_LABELS[i] in tfile]
                    sel_tp = st.pills("Filter", options=topt, default=topt[0], selection_mode="single", label_visibility="collapsed", key="wc_pills_csv") if topt else None
                
                with c_wc:
                    if sel_tp:
                        fids = [k for k,v in TOPIC_LABELS.items() if v==sel_tp]
                        if fids:
                            kw = TOPIC_KEYWORDS.get(fids[0], ["data"])
                            st.image(generate_cached_wordcloud(" ".join(kw*10), "viridis"), use_container_width=True)
                    else: st.info("Menunggu data...")
            
            st.write("")
            with st.container(border=True):
                st.subheader("📥 Data Hasil")
                st.dataframe(df[[text_col, 'Sentimen', 'Nama_Topik']], use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "hasil.csv", "text/csv", type="primary")