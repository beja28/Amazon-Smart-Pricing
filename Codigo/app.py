import streamlit as st
import streamlit.components.v1 as components  # ← AÑADIDO (1/2)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import requests
import numpy as np
import os
from PIL import Image
from io import BytesIO
import base64 # Asegúrate de que esto se importa

# Configuración
API_URL = "http://127.0.0.1:8001/predict"
DATASET_PATH = "../Datasets/evaluacion4_produccion.csv"
TEST_SAMPLES_PATH = "dashboard_test_samples.csv"

st.set_page_config(page_title="Amazon Price Optimizer", layout="wide", initial_sidebar_state="collapsed")

# --- ESTILOS CUSTOM — PREMIUM DARK EDITORIAL ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Variables del sistema ── */
    :root {
        --bg-base:      #0a0c0f;
        --bg-surface:   #111418;
        --bg-card:      #161b22;
        --bg-card-hover:#1c2330;
        --border:       #21262d;
        --border-accent:#c9933a;
        --gold:         #c9933a;
        --gold-light:   #e8b96a;
        --gold-dim:     rgba(201,147,58,0.12);
        --text-primary: #e8e6e1;
        --text-secondary:#8b9099;
        --text-muted:   #4a5260;
        --success:      #2ea84c;
        --danger:       #e5534b;
        --info:         #388bfd;
        --warning:      #d29922;
        --font-display: 'Cormorant Garamond', Georgia, serif;
        --font-mono:    'DM Mono', monospace;
        --font-body:    'DM Sans', sans-serif;
    }

    /* ── Base global ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stHeader"]  { background: transparent !important; }

    /* Fondo general */
    .main .block-container {
        background-color: var(--bg-base) !important;
        padding-top: 2rem !important;
        max-width: 1400px !important;
    }

    /* ── Tipografía heading ── */
    h1, h2, h3 {
        font-family: var(--font-display) !important;
        font-weight: 300 !important;
        letter-spacing: 0.04em !important;
        color: var(--text-primary) !important;
    }
    h1 { font-size: 2.8rem !important; letter-spacing: 0.06em !important; }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.45rem !important; }
    h4, h5, h6, p, li, span, label {
        font-family: var(--font-body) !important;
        color: var(--text-primary) !important;
    }

    /* ── Divisores ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 2rem 0 !important;
    }

    /* ── Botones Streamlit ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-secondary) !important;
        border-radius: 4px !important;
        font-family: var(--font-body) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--gold) !important;
        color: var(--gold) !important;
        background: var(--gold-dim) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--gold) !important;
        border-color: var(--gold) !important;
        color: #0a0c0f !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--gold-light) !important;
        border-color: var(--gold-light) !important;
        color: #0a0c0f !important;
    }

    /* ── Métricas Streamlit ── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 1rem 1.25rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-body) !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        font-size: 1.55rem !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* ── Alerts / st.success, st.error, st.warning, st.info ── */
    [data-testid="stAlert"] {
        border-radius: 4px !important;
        border-left-width: 3px !important;
        background: var(--bg-card) !important;
        font-family: var(--font-body) !important;
        font-size: 0.88rem !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        font-family: var(--font-body) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.5rem !important;
        border-bottom: 2px solid transparent !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--gold) !important;
        border-bottom-color: var(--gold) !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    .dvn-scroller { background: var(--bg-card) !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        background: var(--bg-card) !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] { color: var(--gold) !important; }

    /* ── Caption / small text ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-family: var(--font-body) !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.06em !important;
    }

    /* ══════════════════════════════
       COMPONENTES CUSTOM
    ══════════════════════════════ */

    /* ── Header principal ── */
    .dash-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        padding: 0 0 2rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2.5rem;
    }
    .dash-header-title {
        font-family: var(--font-display);
        font-size: 3rem;
        font-weight: 300;
        color: var(--text-primary);
        letter-spacing: 0.08em;
        line-height: 1;
        margin: 0;
    }
    .dash-header-title span {
        color: var(--gold);
    }
    .dash-header-sub {
        font-family: var(--font-body);
        font-size: 0.75rem;
        font-weight: 400;
        color: var(--text-muted);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    .dash-header-version {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 0.12em;
        padding: 0.3rem 0.75rem;
        border: 1px solid var(--border);
        border-radius: 3px;
    }

    /* ── Section label ── */
    .section-label {
        font-family: var(--font-body);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ── Section heading ── */
    .section-heading {
        font-family: var(--font-display);
        font-size: 1.6rem;
        font-weight: 300;
        color: var(--text-primary);
        letter-spacing: 0.05em;
        margin: 0 0 0.3rem 0;
    }
    .section-heading-num {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--gold);
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
        display: block;
    }

    /* ── Product card ── */
    .product-card {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0;
        background: var(--bg-card);
        transition: all 0.25s ease;
        cursor: pointer;
        overflow: hidden;
        position: relative;
    }
    .product-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: transparent;
        transition: background 0.25s ease;
    }
    .product-card:hover::before,
    .product-card-selected::before {
        background: var(--gold);
    }
    .product-card:hover {
        border-color: var(--border-accent);
        background: var(--bg-card-hover);
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.4);
    }
    .product-card-selected {
        border-color: var(--border-accent) !important;
        background: var(--bg-card-hover) !important;
    }
    .product-card-body {
        padding: 1rem 1rem 0.75rem;
    }

    /* ── Product title ── */
    .product-title {
        font-family: var(--font-body);
        font-size: 0.82rem;
        font-weight: 400;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        height: 52px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        line-height: 1.4;
    }

    /* ── Product price ── */
    .product-price {
        font-family: var(--font-mono);
        font-size: 1.6rem;
        color: var(--gold);
        font-weight: 400;
        margin: 0.5rem 0 0.75rem;
        letter-spacing: -0.02em;
    }

    /* ── Badges ── */
    .product-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 2px;
        font-family: var(--font-body);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 2px 2px 2px 0;
        white-space: nowrap;
    }
    .badge-bestseller  { background: rgba(201,147,58,0.15); color: var(--gold); border: 1px solid rgba(201,147,58,0.3); }
    .badge-coupon      { background: rgba(56,139,253,0.12); color: #388bfd;     border: 1px solid rgba(56,139,253,0.25); }
    .badge-buybox      { background: rgba(46,168,76,0.12);  color: #2ea84c;     border: 1px solid rgba(46,168,76,0.25); }
    .badge-sustainable { background: rgba(63,185,80,0.12);  color: #3fb950;     border: 1px solid rgba(63,185,80,0.25); }
    .badge-premium     { background: rgba(163,113,247,0.12);color: #a371f7;     border: 1px solid rgba(163,113,247,0.25); }
    .badge-sponsored   { background: rgba(229,83,75,0.12);  color: #e5534b;     border: 1px solid rgba(229,83,75,0.25); }

    /* ── Category / Subtype tags ── */
    .category-tag {
        font-family: var(--font-body);
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .subtype-tag {
        font-family: var(--font-body);
        font-size: 0.62rem;
        font-weight: 400;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        display: inline-block;
        margin-bottom: 0.5rem;
        margin-left: 0.5rem;
    }
    .subtype-tag::before { content: '·  '; }

    /* ── Metric small ── */
    .metric-small {
        font-family: var(--font-body);
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin: 4px 0;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* ── Precio big display ── */
    .big-metric {
        font-family: var(--font-mono);
        font-size: 2.8rem;
        font-weight: 300;
        color: var(--gold);
        text-align: center;
        letter-spacing: -0.02em;
    }

    /* ── Imagen container ── */
    .product-image-container {
        width: 100%;
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border);
    }
    .product-image-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }

    /* ── Eliminar gaps de Streamlit dentro de cards ── */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] > div > div > div {
        gap: 0 !important;
    }
    /* Colapsar el margen superior del elemento vacío que Streamlit pone entre HTML y botón */
    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stButton"]) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* ── Salud del producto ── */
    .health-panel {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .health-panel::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
    }
    .health-score {
        font-family: var(--font-mono);
        font-size: 4rem;
        font-weight: 300;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .health-label {
        font-family: var(--font-body);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* ── Result comparison boxes ── */
    .result-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-box-label {
        font-family: var(--font-body);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }
    .result-box-value {
        font-family: var(--font-mono);
        font-size: 2.2rem;
        font-weight: 300;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    .result-box-value.gold { color: var(--gold); }
    .result-box-value.green { color: var(--success); }
    .result-box-value.red { color: var(--danger); }
    .result-arrow {
        font-family: var(--font-mono);
        font-size: 1.8rem;
        color: var(--text-muted);
        text-align: center;
        padding-top: 2.5rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold); }

    /* ── Footer ── */
    .dash-footer {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        color: var(--text-muted);
        letter-spacing: 0.12em;
        text-align: center;
        padding: 2rem 0 1rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. CARGA DE DATOS ---

# ── Plotly dark theme helper ──
PLOTLY_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,20,24,1)',
    font=dict(family='DM Sans', color='#8b9099', size=11),
    title_font=dict(family='DM Sans', color='#e8e6e1', size=13),
    xaxis=dict(gridcolor='#21262d', linecolor='#21262d', tickfont=dict(color='#4a5260')),
    yaxis=dict(gridcolor='#21262d', linecolor='#21262d', tickfont=dict(color='#4a5260')),
    legend=dict(bgcolor='rgba(17,20,24,0.8)', bordercolor='#21262d', borderwidth=1, font=dict(color='#8b9099')),
    margin=dict(l=10, r=10, t=40, b=10)
)

@st.cache_data
def load_image_from_url(url):
    """Carga imagen desde URL con caché"""
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

def load_data():
    try:
        if not os.path.exists(DATASET_PATH):
            st.error(f"No se encuentra el archivo en: {DATASET_PATH}")
            return None
        
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.lower()
        
        df['precio_real'] = np.exp(df['log_original_price'])
        df['ventas_mes_real'] = np.exp(df['log_purchased_last_month'])
        df['reviews_real'] = np.exp(df['log_total_reviews'])
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

@st.cache_data
def load_test_samples():
    """Carga los 120 productos de test desde el CSV dedicado"""
    try:
        if not os.path.exists(TEST_SAMPLES_PATH):
            st.error(f"No se encuentra el archivo de muestras en: {TEST_SAMPLES_PATH}")
            return None
        df_test = pd.read_csv(TEST_SAMPLES_PATH)
        df_test.columns = df_test.columns.str.lower()
        df_test['precio_real']     = np.exp(df_test['log_original_price'])
        df_test['ventas_mes_real'] = np.exp(df_test['log_purchased_last_month'])
        df_test['reviews_real']    = np.exp(df_test['log_total_reviews'])
        return df_test
    except Exception as e:
        st.error(f"Error cargando muestras de test: {e}")
        return None

@st.cache_resource
def get_knn_engine(df_market):
    """Construye el motor KNN en memoria para buscar competidores idénticos."""
    df_valid = df_market.dropna(subset=['original_title']).reset_index(drop=True)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_tfidf = vectorizer.fit_transform(df_valid['original_title'])
    
    # Buscamos hasta 50 vecinos para tener margen al filtrar por precio
    n_vecinos = min(50, len(df_valid))
    knn = NearestNeighbors(n_neighbors=n_vecinos, metric='cosine', n_jobs=1)
    knn.fit(X_tfidf)
    
    return vectorizer, knn, df_valid

def get_sample_products(n_products=5):
    """Selecciona 5 productos aleatorios de 5 categorías distintas del CSV de test"""
    df_test = load_test_samples()
    if df_test is None or len(df_test) == 0:
        return []

    categorias = df_test['category'].unique()
    if len(categorias) < n_products:
        n_products = len(categorias)

    categorias_seleccionadas = np.random.choice(categorias, size=n_products, replace=False)

    productos_muestra = []
    for categoria in categorias_seleccionadas:
        productos_categoria = df_test[df_test['category'] == categoria]
        if len(productos_categoria) > 0:
            producto = productos_categoria.sample(n=1).iloc[0]
            productos_muestra.append(producto)

    return productos_muestra

df = load_data()

if df is None:
    st.stop()

# Inicializar session state
if 'productos_muestra' not in st.session_state:
    st.session_state.productos_muestra = get_sample_products()
    st.session_state.producto_seleccionado_idx = None

if 'refresh' not in st.session_state:
    st.session_state.refresh = 0

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'resumen'

# --- 2. HEADER ---
st.markdown("""
    <div class="dash-header">
        <div>
            <p class="dash-header-sub">Intelligence Dashboard · Amazon Marketplace</p>
            <h1 class="dash-header-title">Smart <span>Pricing</span></h1>
        </div>
        <div class="dash-header-version">v3.0 · ML ENGINE</div>
    </div>
""", unsafe_allow_html=True)

col_refresh = st.columns([2, 3, 1])

with col_refresh[1]:
    df_test_search = load_test_samples()
    if df_test_search is not None:
        busqueda = st.text_input(
            "Buscar",
            placeholder="Buscar producto por nombre…",
            label_visibility="collapsed",
            key="product_search"
        )
        if busqueda and len(busqueda) >= 2:
            mask = df_test_search['original_title'].str.contains(busqueda, case=False, na=False)
            resultados = df_test_search[mask]

            if len(resultados) > 0:
                opciones = {
                    row['original_title'][:90]: idx
                    for idx, row in resultados.head(8).iterrows()
                }
                seleccion_nombre = st.selectbox(
                    "Resultados",
                    options=list(opciones.keys()),
                    label_visibility="collapsed",
                    key="product_search_select"
                )
                if st.button("Cargar producto", use_container_width=True, key="load_searched_product"):
                    producto_cargado = df_test_search.loc[opciones[seleccion_nombre]]
                    productos_actuales = list(st.session_state.productos_muestra)
                    if not productos_actuales:
                        productos_actuales = get_sample_products()
                    productos_actuales[0] = producto_cargado
                    st.session_state.productos_muestra = productos_actuales
                    st.session_state.producto_seleccionado_idx = 0
                    st.rerun()
            else:
                st.caption("Sin resultados para esa búsqueda.")

with col_refresh[2]:
    if st.button("Cambiar Productos", use_container_width=True, key="refresh_products"):
        st.session_state.productos_muestra = get_sample_products()
        st.session_state.producto_seleccionado_idx = None
        st.rerun()

# --- 3. CARRUSEL DE PRODUCTOS ---
st.markdown('<div class="section-label">Catálogo de Productos</div>', unsafe_allow_html=True)

cols = st.columns(5)

for idx, producto in enumerate(st.session_state.productos_muestra):
    with cols[idx]:
        is_selected = st.session_state.producto_seleccionado_idx == idx
        with st.container():
            img_container = st.container()
            with img_container:
                img = load_image_from_url(producto['product_image_url'])
                if img:
                    img_resized = img.copy()
                    img_resized.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    background = Image.new('RGB', (300, 300), (255, 255, 255))
                    offset = ((300 - img_resized.size[0]) // 2, (300 - img_resized.size[1]) // 2)
                    background.paste(img_resized, offset)
                    st.image(background, use_container_width=True)
                else:
                    placeholder = Image.new('RGB', (300, 300), (240, 240, 240))
                    st.image(placeholder, use_container_width=True)
            
            badges_html = ''
            if producto['is_best_seller'] == 'Yes':
                badges_html += '<span class="product-badge badge-bestseller">🏆 Best Seller</span>'
            if producto['has_coupon'] == 1:
                badges_html += '<span class="product-badge badge-coupon">🎟️ Cupón</span>'
            if producto['buy_box_availability'] == 1:
                badges_html += '<span class="product-badge badge-buybox">📦 Buy Box</span>'
            if producto['sustainability_tags'] == 1:
                badges_html += '<span class="product-badge badge-sustainable">🌱 Eco</span>'
            if producto['is_premium_brand']:
                badges_html += '<span class="product-badge badge-premium">👑 Premium</span>'
            if producto['is_sponsored'] == 'Yes':
                badges_html += '<span class="product-badge badge-sponsored">📢 Sponsored</span>'

            titulo_producto = producto['original_title']

            st.markdown(f'''
                <div style="padding: 0.75rem 0.75rem 0.5rem;">
                    <div style="height:32px;overflow:hidden;margin-bottom:0.5rem;">
                        <span class="category-tag">{producto["category"]}</span>
                        <span class="subtype-tag">{producto["subtype"]}</span>
                    </div>
                    <div class="product-title" style="height:55px;-webkit-line-clamp:3;">{titulo_producto}</div>
                    <div class="product-price">{producto["precio_real"]:.2f} €</div>
                    <div style="height:70px;overflow:hidden;">
                        <div class="metric-small">⭐ {producto["product_rating"]}/5.0 &nbsp;·&nbsp; {int(producto["reviews_real"])} reviews</div>
                        <div class="metric-small">📊 {int(producto["ventas_mes_real"])} ventas/mes</div>
                        <div class="metric-small">🏭 {producto["market_tier"]} &nbsp;·&nbsp; {producto["condition"]}</div>
                    </div>
                    <div style="height:60px;overflow:hidden;margin: 0.5rem 0 0.75rem;">
                        {badges_html if badges_html else '&nbsp;'}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
            if st.button(
                "Seleccionado" if is_selected else "Seleccionar",
                key=f"select_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.producto_seleccionado_idx = idx
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# BARRA DE TABS FIJA — siempre visible desde el inicio
# Resumen activo por defecto. Los tabs solo tienen efecto
# cuando hay un producto seleccionado.
# ══════════════════════════════════════════════════════════════
_tab_activo = st.session_state.selected_tab
_tiene_producto = st.session_state.producto_seleccionado_idx is not None
_tabs = [
    ('resumen',   'Resumen',            'Vista general'),
    ('mercado',   'Análisis de Mercado','¿Dónde estoy?'),
    ('optimizar', 'Optimizar Precio',   'IA y predicción'),
]
_items = ''.join(
    f'<div class="st-tab-item {"active" if k == _tab_activo else ""} {"enabled" if _tiene_producto else "disabled"}" data-tab="{k}">'
    f'<span class="st-tab-label">{lbl}</span>'
    f'<span class="st-tab-sub">{sub}</span>'
    f'</div>'
    for k, lbl, sub in _tabs
)
components.html(f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:transparent; overflow:hidden; }}
</style></head><body><script>
(function() {{
    var p = window.parent.document;

    // Inyectar estilos en el padre una sola vez
    if (!p.getElementById('_stTabBarStyle')) {{
        var s = p.createElement('style');
        s.id = '_stTabBarStyle';
        s.textContent = `
            #_stTabBar {{
                position: fixed;
                top: 0; left: 0; right: 0;
                z-index: 999999;
                height: 70px; /* ← AUMENTADO de 52px */
                background: rgba(10,12,15,0.96);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border-bottom: 1px solid #21262d;
                display: flex;
                align-items: stretch;
                justify-content: center; /* ← AÑADIDO para centrar los botones horizontalmente */
                padding: 0 2.5rem;
                font-family: 'DM Sans', sans-serif;
                box-shadow: 0 2px 20px rgba(0,0,0,0.4);
            }}
            .st-tab-item {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 0 3rem; /* ← AUMENTADO para separar más los botones entre sí */
                cursor: pointer;
                border-bottom: 4px solid transparent; /* ← AUMENTADO para que el borde activo se note más */
                transition: all .15s ease;
                user-select: none;
                gap: 4px; /* ← AUMENTADO para separar un poco el título del subtítulo */
            }}
            .st-tab-item.disabled {{
                opacity: 0.2;
                cursor: allowed;
            }}
            .st-tab-item.enabled:hover {{ background: rgba(255,255,255,0.04); }}
            .st-tab-item.active {{ border-bottom-color: #c9933a; }}
            .st-tab-label {{
                font-size: 0.9rem; /* ← AUMENTADO de 0.7rem */
                font-weight: 600;
                letter-spacing: .12em;
                text-transform: uppercase;
                color: #4a5260;
                white-space: nowrap;
                transition: color .15s ease;
            }}
            .st-tab-item.active .st-tab-label {{ color: #c9933a; }}
            .st-tab-sub {{
                font-size: 0.75rem; /* ← AUMENTADO de 0.57rem */
                color: #4a5260;
                opacity: .5;
                white-space: nowrap;
            }}
            .st-tab-item.active .st-tab-sub {{ color: #e8b96a; opacity: .75; }}
        `;
        p.head.appendChild(s);
    }}

    // Crear o reemplazar la barra
    var old = p.getElementById('_stTabBar');
    if (old) old.remove();
    var bar = p.createElement('div');
    bar.id = '_stTabBar';
    bar.innerHTML = `{_items}`;
    p.body.appendChild(bar);

    // Clicks: solo activos si hay producto seleccionado
    bar.querySelectorAll('.st-tab-item.enabled').forEach(function(item) {{
        item.addEventListener('click', function() {{
            var tab = this.getAttribute('data-tab');
            var keyword = tab === 'resumen'   ? 'resumen'   :
                          tab === 'mercado'   ? 'análisis'  : 'optimizar';
            var btns = p.querySelectorAll('button');
            for (var i = 0; i < btns.length; i++) {{
                if (btns[i].innerText.toLowerCase().indexOf(keyword) !== -1) {{
                    btns[i].click();
                    return;
                }}
            }}
        }});
    }});
}})();
</script></body></html>
""", height=0, scrolling=False)

# --- 4. ANÁLISIS DEL PRODUCTO SELECCIONADO ---
if st.session_state.producto_seleccionado_idx is not None:
    producto_seleccionado = st.session_state.productos_muestra[st.session_state.producto_seleccionado_idx]
    
    st.markdown('<div class="section-label">Análisis del Producto Seleccionado</div>', unsafe_allow_html=True)
    
    det_col1, det_col2, det_col3 = st.columns([1, 2, 1])
    
    with det_col1:
        img = load_image_from_url(producto_seleccionado['product_image_url'])
        if img:
            img_resized = img.copy()
            img_resized.thumbnail((300, 300), Image.Resampling.LANCZOS)
            background = Image.new('RGB', (300, 300), (17, 20, 24))
            offset = ((300 - img_resized.size[0]) // 2, (300 - img_resized.size[1]) // 2)
            background.paste(img_resized, offset)
            st.image(background, use_container_width=True)
        else:
            placeholder = Image.new('RGB', (300, 300), (22, 27, 34))
            st.image(placeholder, use_container_width=True)
    
    with det_col2:
        titulo_producto = f"{producto_seleccionado['original_title']}"
        st.markdown(f"### {titulo_producto}")
        st.markdown(
            f'<p style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5260;margin-bottom:1rem;">'
            f'{producto_seleccionado["category"]} &nbsp;·&nbsp; {producto_seleccionado["subtype"]} &nbsp;·&nbsp; '
            f'{producto_seleccionado["market_tier"]} &nbsp;·&nbsp; {producto_seleccionado["condition"]} &nbsp;·&nbsp; {producto_seleccionado["tech_generation"]}'
            f'</p>',
            unsafe_allow_html=True
        )
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(
                "Precio Actual",
                f"{producto_seleccionado['precio_real']:.2f} €",
                delta=f"-{producto_seleccionado['discount_percentage']*100:.0f}%" if producto_seleccionado['discount_percentage'] > 0 else None
            )
        with metric_cols[1]:
            st.metric("Rating", f"{producto_seleccionado['product_rating']}/5")
        with metric_cols[2]:
            st.metric("Ventas (mes)", f"{int(producto_seleccionado['ventas_mes_real'])}")
        with metric_cols[3]:
            st.metric("Reviews", f"{int(producto_seleccionado['reviews_real'])}")
    
    with det_col3:
        st.markdown(
            '<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;'
            'color:#4a5260;margin-bottom:1rem;">Características</p>',
            unsafe_allow_html=True
        )
        badges = []
        if producto_seleccionado['is_best_seller'] == 'Yes':
            badges.append('<span class="product-badge badge-bestseller">🏆 Best Seller</span>')
        if producto_seleccionado['has_coupon'] == 1:
            badges.append('<span class="product-badge badge-coupon">🎟️ Cupón Activo</span>')
        if producto_seleccionado['buy_box_availability'] == 1:
            badges.append('<span class="product-badge badge-buybox">📦 Buy Box</span>')
        if producto_seleccionado['is_sponsored'] == 'Yes':
            badges.append('<span class="product-badge badge-sponsored">📢 Patrocinado</span>')
        if producto_seleccionado['sustainability_tags'] == 1:
            badges.append('<span class="product-badge badge-sustainable">🌱 Sostenible</span>')
        if producto_seleccionado['is_premium_brand']:
            badges.append('<span class="product-badge badge-premium">👑 Marca Premium</span>')
        
        if badges:
            st.markdown('<div style="display:flex;flex-direction:column;gap:0.5rem;">' + ''.join(badges) + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4a5260;font-size:0.8rem;">Sin características especiales</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- BOTONES FANTASMA (LA SOLUCIÓN DEFINITIVA) ---
    # Los metemos en el sidebar. Como en tus estilos globales pusiste 
    # [data-testid="stSidebar"] { display: none; }, estos botones jamás se verán en pantalla, 
    # pero el JavaScript de la barra superior sí podrá hacerles clic.
    with st.sidebar:
        for _hkey, _htitle in [('resumen', 'Resumen'), ('mercado', 'Análisis de Mercado'), ('optimizar', 'Optimizar Precio')]:
            if st.button(_htitle, key=f"tab_{_hkey}"):
                st.session_state.selected_tab = _hkey
                st.rerun()
    
    # --- CONTENIDO DE CADA TAB ---
    
    # ==================== TAB 1: RESUMEN ====================
    if st.session_state.selected_tab == 'resumen':
        st.markdown('<div class="section-label">Resumen del Producto</div>', unsafe_allow_html=True)
        
        # KPIs comparativos - Filtrar por categoría Y subtipo
        df_categoria = df[df['category'] == producto_seleccionado['category']]
        df_subtipo = df[df['subtype'] == producto_seleccionado['subtype']]
        
        st.markdown('<p style="font-size:0.8rem;color:#8b9099;margin-bottom:1rem;">Tu producto vs promedio del subtipo</p>', unsafe_allow_html=True)
        
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            precio_promedio = df_subtipo['precio_real'].mean()
            diff_precio = producto_seleccionado['precio_real'] - precio_promedio
            st.metric(
                "Precio",
                f"{producto_seleccionado['precio_real']:.2f} €",
                delta=f"{diff_precio:+.2f} € vs promedio",
                delta_color="inverse"
            )
        
        with kpi_cols[1]:
            rating_promedio = df_subtipo['product_rating'].mean()
            diff_rating = producto_seleccionado['product_rating'] - rating_promedio
            st.metric(
                "Rating",
                f"{producto_seleccionado['product_rating']:.1f}/5",
                delta=f"{diff_rating:+.2f} vs promedio"
            )
        
        with kpi_cols[2]:
            ventas_promedio = df_subtipo['ventas_mes_real'].mean()
            diff_ventas = producto_seleccionado['ventas_mes_real'] - ventas_promedio
            st.metric(
                "Ventas/mes",
                f"{int(producto_seleccionado['ventas_mes_real'])}",
                delta=f"{int(diff_ventas):+} vs promedio"
            )
        
        with kpi_cols[3]:
            reviews_promedio = df_subtipo['reviews_real'].mean()
            diff_reviews = producto_seleccionado['reviews_real'] - reviews_promedio
            st.metric(
                "Reviews",
                f"{int(producto_seleccionado['reviews_real'])}",
                delta=f"{int(diff_reviews):+} vs promedio"
            )
        
        st.markdown("---")
        st.markdown('<div class="section-label">Diagnóstico Rápido</div>', unsafe_allow_html=True)
        
        # Calcular scores individuales (mismo método que Análisis de Mercado)
        # Score 1: Precio (mejor cerca de la mediana)
        banda_precio = df_subtipo[
            (df_subtipo['precio_real'] >= producto_seleccionado['precio_real'] * 0.85) &
            (df_subtipo['precio_real'] <= producto_seleccionado['precio_real'] * 1.15)
        ]


        if len(banda_precio) >= 3:
            base = banda_precio
        else:
            n = max(10, int(len(df_subtipo) * 0.15))
            base = (
                df_subtipo.assign(
                    diff=(df_subtipo["precio_real"] - producto_seleccionado["precio_real"]).abs()
                )
                .nsmallest(n, "diff")
            )


        score_precio = float((base["ventas_mes_real"] <= producto_seleccionado["ventas_mes_real"]).mean() * 100)
        score_precio = float(np.clip(score_precio, 0, 100))
        score_precio = float((base["ventas_mes_real"] <= producto_seleccionado["ventas_mes_real"]).mean() * 100)
        score_precio = float(np.clip(score_precio, 0, 100))



        # --- SCORE CALIDAD: rating bayesiano (suaviza productos con pocas reviews)
        # m= mediana siendo 20 el mínimo y 200 el máximo de reseñas
        m_raw = float(df_subtipo['reviews_real'].median())
        m = float(np.clip(m_raw, 20, 200))
        C = float(df_subtipo['product_rating'].mean())          # media del subtipo (prior)
        r = float(producto_seleccionado['product_rating'])
        v = float(producto_seleccionado['reviews_real'])
        bayes_rating  = (v / (v + m)) * r + (m / (v + m)) * C  # rating ajustado
        score_calidad = float(np.clip((bayes_rating - 1) / 4 * 100, 0, 100))  # escala 1-5 → 0-100


        # --- SCORE POPULARIDAD: percentil en log(ventas), menos sesgo por outliers
        ventas_log    = np.log1p(df_subtipo['ventas_mes_real'].clip(lower=0))
        ventas_log_ps = np.log1p(max(0.0, float(producto_seleccionado['ventas_mes_real'])))
        score_popularidad = float((ventas_log <= ventas_log_ps).mean() * 100)


        # --- SCORE GENERAL: 60% promedio + 40% peor dimensión
        # Si la peor dimensión está en rojo (<40), arrastra el resultado hacia abajo
        score_general = 0.6 * ((score_precio + score_calidad + score_popularidad) / 3) \
                    + 0.4 * min(score_precio, score_calidad, score_popularidad)
        score_general = float(np.clip(score_general, 0, 100))
        
        # Determinar color
        if min(score_precio, score_calidad, score_popularidad) >= 70:
            color = "green";  status = "EXCELENTE"; emoji = "🟢"
        elif score_general >= 55 and min(score_precio, score_calidad, score_popularidad) >= 40:
            color = "orange"; status = "BUENA";     emoji = "🟡"
        else:
            color = "red";    status = "NECESITA ATENCIÓN"; emoji = "🔴"

        # Convertir color lógico a HEX (para el estilo del panel)
        COLOR_MAP = {"green": "#2ea84c", "orange": "#d29922", "red": "#e5534b"}
        color_hex = COLOR_MAP[color]

        # Si tu HTML usa score_salud, asígnalo (o cambia el HTML a score_general)
        score_salud = score_general

        st.markdown(f"""
            <div class="health-panel" style="border-color:{color_hex}33;">
                <div class="health-label" style="color:{color_hex};">SALUD DEL PRODUCTO</div>
                <div class="health-score" style="color:{color_hex};">{score_salud:.0f}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4a5260;margin-top:0.25rem;">/100 &nbsp;·&nbsp; {status}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detalles
        detail_cols = st.columns(3)
        
        with detail_cols[0]:
            st.markdown(f"**Precio:** Score {score_precio:.0f}/100")
            if score_precio >= 70:
                st.success("Precio bien posicionado")
            elif score_precio >= 40:
                st.info("Precio en rango aceptable")
            else:
                st.warning("Precio muy alejado de la mediana de mercado")
        
        with detail_cols[1]:
            st.markdown(f"**Calidad:** Score {score_calidad:.0f}/100")
            if score_calidad >= 70:
                st.success("Buena calidad percibida")
            elif score_calidad >= 40:
                st.info("Calidad aceptable")
            else:
                st.warning("Necesitas mejorar rating y/o conseguir más reviews")
        
        with detail_cols[2]:
            st.markdown(f"**Popularidad:** Score {score_popularidad:.0f}/100")
            if score_popularidad >= 70:
                st.success("Vendes bien en tu subtipo")
            elif score_popularidad >= 40:
                st.info("Ventas en rango medio")
            else:
                st.warning("Tus ventas están por debajo del promedio")
    
    # ==================== TAB 2: ANÁLISIS DE MERCADO ====================
    elif st.session_state.selected_tab == 'mercado':
        st.markdown('<div class="section-label">Análisis de Mercado</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem;color:#8b9099;margin-bottom:2rem;">Entiende tu posición competitiva y oportunidades de mejora</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Obtener datos de la categoría y subtipo
        df_categoria = df[df['category'] == producto_seleccionado['category']]
        df_subtipo = df[df['subtype'] == producto_seleccionado['subtype']]
        
                # ========== SECCIÓN 1: TU POSICIÓN EN EL MERCADO ==========
        st.markdown('<span class="section-heading-num">01</span><div class="section-heading">Tu Posición en el Mercado</div>', unsafe_allow_html=True)
        st.caption("¿Dónde estoy yo en mi subtipo?")

        # Cuartiles P25 y P75
        precio_p25 = df_subtipo['precio_real'].quantile(0.25)
        precio_p75 = df_subtipo['precio_real'].quantile(0.75)
        ventas_p25 = df_subtipo['ventas_mes_real'].quantile(0.25)
        ventas_p75 = df_subtipo['ventas_mes_real'].quantile(0.75)

        x_min = df_subtipo['precio_real'].min() * 0.9
        x_max = df_subtipo['precio_real'].max() * 1.1
        y_min = df_subtipo['ventas_mes_real'].min() * 0.9
        y_max = df_subtipo['ventas_mes_real'].max() * 1.1

        fig1 = go.Figure()

        # 9 zonas (x0, y0, x1, y1, color_fondo, etiqueta, color_texto)
        zonas = [
            (x_min,      ventas_p75, precio_p25, y_max,      "rgba(33,150,243,0.12)",  "🔵 OPORTUNIDAD", "blue"),
            (precio_p25, ventas_p75, precio_p75, y_max,      "rgba(76,175,80,0.12)",   "🟢 EFICIENTE",   "green"),
            (precio_p75, ventas_p75, x_max,      y_max,      "rgba(255,215,0,0.20)",   "🏆 PREMIUM",     "goldenrod"),
            (x_min,      ventas_p25, precio_p25, ventas_p75, "rgba(255,193,7,0.12)",   "🟡 COMPETITIVA", "orange"),
            (precio_p25, ventas_p25, precio_p75, ventas_p75, "rgba(200,200,200,0.12)", "⚪ ESTÁNDAR",    "gray"),
            (precio_p75, ventas_p25, x_max,      ventas_p75, "rgba(255,87,34,0.12)",   "🟠 CARA",        "darkorange"),
            (x_min,      y_min,      precio_p25, ventas_p25, "rgba(244,67,54,0.12)",   "🔴 DESCARTADA",  "red"),
            (precio_p25, y_min,      precio_p75, ventas_p25, "rgba(244,67,54,0.12)",   "🔴 RIESGO",      "red"),
            (precio_p75, y_min,      x_max,      ventas_p25, "rgba(183,28,28,0.20)",   "🔴 RIESGO ALTO", "darkred"),
        ]

        for x0, y0, x1, y1, color, etiqueta, color_texto in zonas:
            fig1.add_shape(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor=color, line=dict(width=0), layer="below"
            )
            fig1.add_annotation(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=etiqueta,
                showarrow=False,
                font=dict(size=11, color=color_texto, family="Arial Black"),
                bgcolor="white", bordercolor=color_texto, borderwidth=1.5, opacity=0.85
            )

        # 4 líneas de cuartiles (punteadas)
        for val, eje, label, pos in [
            (precio_p25, "v", f"P25 precio: {precio_p25:.0f}€", "top"),
            (precio_p75, "v", f"P75 precio: {precio_p75:.0f}€", "top"),
            (ventas_p25, "h", f"P25 ventas: {ventas_p25:.0f}",  "right"),
            (ventas_p75, "h", f"P75 ventas: {ventas_p75:.0f}",  "right"),
        ]:
            if eje == "v":
                fig1.add_vline(x=val, line_dash="dot", line_color="gray",
                               annotation_text=label, annotation_position=pos)
            else:
                fig1.add_hline(y=val, line_dash="dot", line_color="gray",
                               annotation_text=label, annotation_position=pos)

        # Competidores
        fig1.add_trace(go.Scatter(
            x=df_subtipo['precio_real'], y=df_subtipo['ventas_mes_real'],
            mode='markers', marker=dict(size=8, color='lightgray', opacity=0.5),
            text=df_subtipo['brand'] + ' ' + df_subtipo['subtype'],
            hovertemplate='<b>%{text}</b><br>Precio: %{x:.2f}€<br>Ventas: %{y:.0f}<extra></extra>',
            name='Competidores', showlegend=True
        ))

        # Tu producto
        fig1.add_trace(go.Scatter(
            x=[producto_seleccionado['precio_real']],
            y=[producto_seleccionado['ventas_mes_real']],
            mode='markers+text',
            marker=dict(size=25, color='red', symbol='star', line=dict(width=2, color='white')),
            text=['TÚ'], textposition='top center',
            textfont=dict(size=14, color='red', family='Arial Black'),
            hovertemplate=f'<b>TU PRODUCTO</b><br>Precio: {producto_seleccionado["precio_real"]:.2f}€<br>Ventas: {int(producto_seleccionado["ventas_mes_real"])}<extra></extra>',
            name='Tu Producto', showlegend=True
        ))

        fig1.update_layout(
            title=f"Mapa de Posicionamiento: {producto_seleccionado['subtype']}",
            xaxis_title="Precio (€)",
            yaxis_title="Ventas último mes",
            height=600,
            hovermode='closest',
            showlegend=True,
            **PLOTLY_DARK
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Insight automático — 9 zonas con cuartiles
        precio_ps = producto_seleccionado['precio_real']
        ventas_ps = producto_seleccionado['ventas_mes_real']

        if precio_ps < precio_p25:    banda_precio = "bajo"
        elif precio_ps < precio_p75:  banda_precio = "medio"
        else:                         banda_precio = "alto"

        if ventas_ps >= ventas_p75:   banda_ventas = "alta"
        elif ventas_ps >= ventas_p25: banda_ventas = "media"
        else:                         banda_ventas = "baja"

        mensajes_zona = {
            ("bajo",  "alta"):  ("🔵 OPORTUNIDAD",  "info",    "Vendes muy bien con precio bajo. Tienes margen para subir precio y mejorar margen sin perder ventas."),
            ("medio", "alta"):  ("🟢 EFICIENTE",    "success", "Buena combinación precio-volumen. Posición sólida y sostenible en el mercado."),
            ("alto",  "alta"):  ("🏆 PREMIUM",      "success", "¡Máximo rendimiento! Precio alto y ventas altas. Protege este posicionamiento."),
            ("bajo",  "media"): ("🟡 COMPETITIVA",  "warning", "Precio bajo con ventas medias. Analiza si el margen cubre costes o si hay margen de subida."),
            ("medio", "media"): ("⚪ ESTÁNDAR",     "info",    "Posición media en todo. Diferenciarte (cupón, Buy Box, reviews) es clave para crecer."),
            ("alto",  "media"): ("🟠 CARA",         "warning", "Precio alto pero ventas mediocres. El mercado no está justificando el precio actual."),
            ("bajo",  "baja"):  ("🔴 DESCARTADA",   "error",   "Precio bajo y pocas ventas. Revisa visibilidad, título y ficha del producto."),
            ("medio", "baja"):  ("🔴 RIESGO",       "error",   "Ventas bajas sin ventaja de precio. Necesitas acción urgente en alguna dimensión."),
            ("alto",  "baja"):  ("🔴 RIESGO ALTO",  "error",   "Precio alto con ventas bajas: la peor combinación. Baja el precio inmediatamente."),
        }

        zona, tipo_msg, mensaje = mensajes_zona[(banda_precio, banda_ventas)]

        insight_cols = st.columns([2, 1])
        with insight_cols[0]:
            if tipo_msg == "success": st.success(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")
            elif tipo_msg == "info":  st.info(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")
            elif tipo_msg == "error": st.error(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")
            else:                     st.warning(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")

        with insight_cols[1]:
            rango_min = precio_ps * 0.9
            rango_max = precio_ps * 1.1
            competidores_rango = df_subtipo[
                (df_subtipo['precio_real'] >= rango_min) & (df_subtipo['precio_real'] <= rango_max)
            ]
            st.metric("Competidores en tu rango", len(competidores_rango),
                      help="Productos con precio ±10% del tuyo")
            percentil_ventas = (df_subtipo['ventas_mes_real'] < ventas_ps).sum() / len(df_subtipo) * 100
            st.metric("Percentil de ventas", f"{percentil_ventas:.0f}%",
                      help="% de productos que venden menos que tú")
        
        st.markdown("---")
        
               
        # ========== GRÁFICO ADICIONAL: DISTRIBUCIÓN POR CATEGORÍA ==========
        st.markdown("### Análisis por Categoría y Subtipo")
        
        # Tabs para diferentes vistas
        analisis_tabs = st.tabs(["Por Categoría", "Por Subtipo", "Productos Patrocinados"])
        
        with analisis_tabs[0]:
            st.markdown("#### Precios Promedio por Categoría")
            
            # Agrupar por categoría
            categoria_stats = df.groupby('category').agg({
                'precio_real': 'mean',
                'ventas_mes_real': 'mean',
                'product_rating': 'mean'
            }).reset_index()
            categoria_stats = categoria_stats.sort_values('precio_real', ascending=False)
            
            fig_cat = px.bar(
                categoria_stats,
                x='category',
                y='precio_real',
                title='Precio Promedio por Categoría',
                labels={'precio_real': 'Precio Promedio (€)', 'category': 'Categoría'},
                color='precio_real',
                color_continuous_scale='Viridis',
                height=400
            )
            fig_cat.update_layout(**PLOTLY_DARK)
            
            # Marcar tu categoría
            tu_categoria_idx = categoria_stats[categoria_stats['category'] == producto_seleccionado['category']].index
            if len(tu_categoria_idx) > 0:
                fig_cat.add_annotation(
                    x=producto_seleccionado['category'],
                    y=categoria_stats.loc[tu_categoria_idx[0], 'precio_real'],
                    text="TU CATEGORÍA",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#c9933a",
                    font=dict(color="#c9933a", size=11, family="DM Sans")
                )
            
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with analisis_tabs[1]:
            st.markdown("#### Precios Promedio por Subtipo en tu Categoría")
            
            # Filtrar por categoría y agrupar por subtipo
            subtipo_stats = df_categoria.groupby('subtype').agg({
                'precio_real': 'mean',
                'ventas_mes_real': 'mean',
                'product_rating': 'mean'
            }).reset_index()
            subtipo_stats = subtipo_stats.sort_values('precio_real', ascending=False)
            
            fig_sub = px.bar(
                subtipo_stats,
                x='subtype',
                y='precio_real',
                title=f'Precio Promedio por Subtipo en {producto_seleccionado["category"]}',
                labels={'precio_real': 'Precio Promedio (€)', 'subtype': 'Subtipo'},
                color='precio_real',
                color_continuous_scale='Cividis',
                height=400
            )
            fig_sub.update_layout(**PLOTLY_DARK)
            
            # Marcar tu subtipo
            tu_subtipo_idx = subtipo_stats[subtipo_stats['subtype'] == producto_seleccionado['subtype']].index
            if len(tu_subtipo_idx) > 0:
                fig_sub.add_annotation(
                    x=producto_seleccionado['subtype'],
                    y=subtipo_stats.loc[tu_subtipo_idx[0], 'precio_real'],
                    text="TU SUBTIPO",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#c9933a",
                    font=dict(color="#c9933a", size=11, family="DM Sans")
                )
            
            st.plotly_chart(fig_sub, use_container_width=True)
        
        with analisis_tabs[2]:
            st.markdown("#### Análisis de Productos Patrocinados en tu Categoría")
            
            # Comparar patrocinados vs no patrocinados
            sponsored_stats = df_categoria.groupby('is_sponsored').agg({
                'precio_real': 'mean',
                'ventas_mes_real': 'mean',
                'product_rating': 'mean'
            }).reset_index()
            
            sponsored_stats['is_sponsored'] = sponsored_stats['is_sponsored'].map({
                'Sponsored': 'Patrocinados',
                'Organic': 'No Patrocinados'
            })

            
            fig_sponsored = go.Figure()
            
            fig_sponsored.add_trace(go.Bar(
                name='Precio Promedio',
                x=sponsored_stats['is_sponsored'],
                y=sponsored_stats['precio_real'],
                text=[f"{p:.2f}€" for p in sponsored_stats['precio_real']],
                textposition='outside',
                marker_color=['#ff5722', '#2196F3']
            ))
            
            fig_sponsored.update_layout(
                title=f'Comparación: Productos Patrocinados vs No Patrocinados en {producto_seleccionado["category"]}',
                xaxis_title='Tipo',
                yaxis_title='Precio Promedio (€)',
                height=400,
                showlegend=False,
                **PLOTLY_DARK
            )
            
            st.plotly_chart(fig_sponsored, use_container_width=True)
            
            # Métricas comparativas
            col_sp1, col_sp2, col_sp3 = st.columns(3)
            
            patrocinados = df_categoria[df_categoria['is_sponsored'] == 'Sponsored']
            no_patrocinados = df_categoria[df_categoria['is_sponsored'] == 'Organic']
            
            with col_sp1:
                if len(patrocinados) > 0 and len(no_patrocinados) > 0:
                    diff_precio_sp = patrocinados['precio_real'].mean() - no_patrocinados['precio_real'].mean()
                    st.metric(
                        "Diferencia de Precio",
                        f"{diff_precio_sp:+.2f} €",
                        help="Los patrocinados cuestan más/menos en promedio"
                    )
            
            with col_sp2:
                if len(patrocinados) > 0 and len(no_patrocinados) > 0:
                    diff_ventas_sp = patrocinados['ventas_mes_real'].mean() - no_patrocinados['ventas_mes_real'].mean()
                    st.metric(
                        "Diferencia de Ventas",
                        f"{diff_ventas_sp:+.0f} uds/mes",
                        help="Los patrocinados venden más/menos en promedio"
                    )
            
            with col_sp3:
                pct_patrocinados = (df_categoria['is_sponsored'] == 'Sponsored').mean() * 100
                st.metric(
                    "% Patrocinados",
                    f"{pct_patrocinados:.1f}%",
                    help="Porcentaje de productos patrocinados en tu categoría"
                )
            
            # Insight
            if producto_seleccionado['is_sponsored'] == 'Sponsored':
                st.info("Tu producto está patrocinado. Esto puede aumentar tu visibilidad.")
            else:
                if pct_patrocinados > 30:
                    st.warning(f"{pct_patrocinados:.0f}% de tu categoría está patrocinada. Considera activar publicidad para competir.")
                else:
                    st.success("Bajo nivel de publicidad en tu categoría. Puedes competir orgánicamente.")
        
        st.markdown("---")
        
        # ========== SECCIÓN 2: TUS COMPETIDORES MÁS CERCANOS ==========
        st.markdown('<span class="section-heading-num">02</span><div class="section-heading">Competidores Más Cercanos</div>', unsafe_allow_html=True)
        
        # --- NUEVA LÓGICA DE COMPETIDORES: KNN (IA) ---
        vectorizer, knn_model, df_knn = get_knn_engine(df)
        titulo_actual = str(producto_seleccionado['original_title'])
        
        vec_entrada = vectorizer.transform([titulo_actual])
        distancias, indices = knn_model.kneighbors(vec_entrada)
        
        # Rango de precio: +/- 40% del precio ACTUAL
        rango_min = producto_seleccionado['precio_real'] * 0.60
        rango_max = producto_seleccionado['precio_real'] * 1.40
        
        indices_validos = []
        for d, idx in zip(distancias[0], indices[0]):
            if d < 0.70: # Filtro de similitud textual
                row_vecino = df_knn.iloc[idx]
                if rango_min <= row_vecino['precio_real'] <= rango_max:
                    indices_validos.append(idx)
                    
        competidores_directos = df_knn.iloc[indices_validos].copy()

        # Fallback de seguridad: Si hay menos de 4 competidores exactos, 
        # ampliamos la búsqueda al subtipo clásico en ese rango de precios.
        if len(competidores_directos) < 4:
            competidores_directos = df[(df['subtype'] == producto_seleccionado['subtype']) & 
                                       (df['precio_real'] >= rango_min) & 
                                       (df['precio_real'] <= rango_max)].copy()
        
        # Ordenar por ventas para mostrar los líderes primero en la tabla
        # (He quitado el .head(10) porque como tu tabla ya tiene scroll infinito,
        # es mejor mostrar todos los rivales que detecte la IA)
        competidores_directos = competidores_directos.sort_values('ventas_mes_real', ascending=False)
        
        # Crear filas HTML dinámicamente con la nueva estructura
        rows_html = ""
        for _, comp in competidores_directos.iterrows():
            diff_precio = comp.get('precio_real', 0) - producto_seleccionado['precio_real']
            color_diff = "#e5534b" if diff_precio < 0 else ("#2ea84c" if diff_precio > 0 else "#8b9099")
            
            img_url = comp.get('product_image_url', '')
            img_tag = f'<img src="{img_url}" style="width:40px;height:40px;object-fit:contain;border-radius:4px;background:#fff;padding:2px;">' if pd.notna(img_url) and img_url else 'N/A'
            
            titulo = str(comp.get('original_title', ''))
            titulo_corto = titulo[:55] + '...' if len(titulo) > 55 else titulo
            
            marca = str(comp.get('brand', ''))
            precio_str = f"{comp.get('precio_real', 0):.2f} €"
            diff_str = f'<span style="color:{color_diff}; font-weight:600;">{diff_precio:+.2f} €</span>'
            ventas_str = f"{int(comp.get('ventas_mes_real', 0))} uds"
            rating_str = f"{comp.get('product_rating', 0):.1f} ⭐"
            reviews_str = f"{int(comp.get('reviews_real', 0))}"
            
            badge_str = '🏆' if comp.get('is_best_seller') == 'Yes' else ''
            cupon_str = '🎟️' if comp.get('has_coupon') == 1 else ''
            buybox_str = '📦' if comp.get('buy_box_availability') == 1 else ''
            premium_str = '👑' if comp.get('is_premium_brand') else ''
            
            link_url = str(comp.get('product_url', ''))
            if link_url and link_url.lower() != 'nan':
                link_url = link_url.split(',202')[0]
                if link_url.startswith('/'):
                    link_url = f"https://www.amazon.es{link_url}"
                link_tag = f'<a href="{link_url}" target="_blank" style="color:#c9933a;text-decoration:none;font-weight:600;font-size:0.75rem;">Ver ↗️</a>'
            else:
                link_tag = ''
            
            cells = (
                f'<td style="padding:8px 12px;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{img_tag}</td>'
                f'<td style="padding:8px 12px;font-size:0.8rem;color:#e8e6e1;border-bottom:1px solid #21262d1a;vertical-align:middle;max-width:250px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{titulo}">{titulo_corto}</td>'
                f'<td style="padding:8px 12px;font-size:0.8rem;color:#b3b8c2;border-bottom:1px solid #21262d1a;vertical-align:middle;">{marca}</td>'
                f'<td style="padding:8px 12px;font-size:0.85rem;color:#e8e6e1;font-family:var(--font-mono);border-bottom:1px solid #21262d1a;vertical-align:middle;">{precio_str}</td>'
                f'<td style="padding:8px 12px;font-size:0.85rem;font-family:var(--font-mono);border-bottom:1px solid #21262d1a;vertical-align:middle;">{diff_str}</td>'
                f'<td style="padding:8px 12px;font-size:0.85rem;color:#e8e6e1;font-family:var(--font-mono);border-bottom:1px solid #21262d1a;vertical-align:middle;">{ventas_str}</td>'
                f'<td style="padding:8px 12px;font-size:0.8rem;color:#e8e6e1;border-bottom:1px solid #21262d1a;vertical-align:middle;">{rating_str}</td>'
                f'<td style="padding:8px 12px;font-size:0.8rem;color:#e8e6e1;border-bottom:1px solid #21262d1a;vertical-align:middle;">{reviews_str}</td>'
                f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{badge_str}</td>'
                f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{cupon_str}</td>'
                f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{buybox_str}</td>'
                f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{premium_str}</td>'
                f'<td style="padding:8px 12px;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{link_tag}</td>'
            )
            rows_html += f'<tr style="transition:background 0.15s;" onmouseover="this.style.background=\'#1c2330\'" onmouseout="this.style.background=\'transparent\'">{cells}</tr>'
        
        # Cabeceras pegajosas y Z-Index para el scroll
        headers_list = ['Img', 'Producto', 'Marca', 'Precio', 'Diff. Precio', 'Ventas', 'Rating', 'Reviews', 'Badge', 'Cupón', 'Buy Box', 'Premium', 'Link']
        headers_html = ''.join(f'<th style="position:sticky; top:0; background:#161b22; z-index:10; padding:10px 12px; text-align:center; font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#8b9099; border-bottom:1px solid #21262d; white-space:nowrap;">{col}</th>' for col in headers_list)

        html_completo = f"""
        <div style="overflow-x:auto; overflow-y:auto; max-height:400px; border:1px solid #21262d; border-radius:6px; margin-top:10px; position:relative; display:block;">
            <table style="width:100%; border-collapse:collapse; background:#161b22;">
                <thead><tr>{headers_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
        st.markdown(html_completo, unsafe_allow_html=True)
        
        # Insights de competidores
        st.markdown("#### Insights Clave")
        
        insight_comp_cols = st.columns(4)
        
        with insight_comp_cols[0]:
            num_con_cupon = (competidores_directos['has_coupon'] == 1).sum()
            pct_cupon = (num_con_cupon / len(competidores_directos)) * 100 if len(competidores_directos) > 0 else 0
            st.metric(
                "Competidores con cupón",
                f"{num_con_cupon}/{len(competidores_directos)}",
                delta=f"{pct_cupon:.0f}%"
            )
            if pct_cupon > 50 and producto_seleccionado['has_coupon'] == 0:
                st.warning("Considera activar un cupón")
        
        with insight_comp_cols[1]:
            precio_promedio_comp = competidores_directos['precio_real'].mean() if len(competidores_directos) > 0 else 0
            diff_vs_comp = producto_seleccionado['precio_real'] - precio_promedio_comp
            st.metric(
                "Tu precio vs promedio",
                f"{diff_vs_comp:+.2f} €",
                delta=f"{(diff_vs_comp/precio_promedio_comp)*100:+.1f}%" if precio_promedio_comp > 0 else None
            )
        
        with insight_comp_cols[2]:
            rating_promedio_comp = competidores_directos['product_rating'].mean() if len(competidores_directos) > 0 else 0
            diff_rating = producto_seleccionado['product_rating'] - rating_promedio_comp
            st.metric(
                "Tu rating vs promedio",
                f"{diff_rating:+.2f}",
                delta="⭐" if diff_rating > 0 else None
            )
            if diff_rating < 0:
                st.warning("Tu rating está por debajo")
        
        with insight_comp_cols[3]:
            num_premium = (competidores_directos['is_premium_brand'] == True).sum()
            pct_premium = (num_premium / len(competidores_directos)) * 100 if len(competidores_directos) > 0 else 0
            st.metric(
                "Marcas Premium",
                f"{num_premium}/{len(competidores_directos)}",
                delta=f"{pct_premium:.0f}%"
            )
        
        st.markdown("---")
        
        # ========== SECCIÓN 3: ¿QUÉ HACE LA DIFERENCIA? ==========
        st.markdown('<span class="section-heading-num">03</span><div class="section-heading">¿Qué Hace la Diferencia?</div>', unsafe_allow_html=True)
        st.caption("¿Qué características importan en mi subtipo?")
        
        # Análisis de características
        caracteristicas_analisis = {
            'Best Seller': ('is_best_seller', 'Best Seller'),
            'Cupón': ('has_coupon', 1),
            'Buy Box': ('buy_box_availability', 1),
            'Sostenible': ('sustainability_tags', 1),
            'Marca Premium': ('is_premium_brand', True),
            'Patrocinado': ('is_sponsored', 'Sponsored')
        }
        
        datos_comparacion = []
        
        for nombre_car, (columna, valor) in caracteristicas_analisis.items():
            # Con característica
            con_car = df_subtipo[df_subtipo[columna] == valor]
            sin_car = df_subtipo[df_subtipo[columna] != valor]
            
            if len(con_car) > 0 and len(sin_car) > 0:
                precio_con = con_car['precio_real'].mean()
                precio_sin = sin_car['precio_real'].mean()
                ventas_con = con_car['ventas_mes_real'].mean()
                ventas_sin = sin_car['ventas_mes_real'].mean()
                
                # Determinar si el producto tiene esta característica
                if columna == 'is_best_seller':
                    tienes = producto_seleccionado[columna] == valor
                elif columna == 'is_sponsored':
                    tienes = producto_seleccionado[columna] == valor
                elif columna == 'is_premium_brand':
                    tienes = producto_seleccionado[columna] == valor
                else:
                    tienes = producto_seleccionado[columna] == valor
                
                datos_comparacion.append({
                    'Característica': nombre_car,
                    'Precio CON': precio_con,
                    'Precio SIN': precio_sin,
                    'Ventas CON': ventas_con,
                    'Ventas SIN': ventas_sin,
                    'Diff Precio': precio_con - precio_sin,
                    'Diff Ventas': ventas_con - ventas_sin,
                    'Tienes': tienes
                })
        
        # Gráfica de barras comparativas
        if datos_comparacion:
            fig3 = go.Figure()
            
            caracteristicas = [d['Característica'] for d in datos_comparacion]
            precios_con = [d['Precio CON'] for d in datos_comparacion]
            precios_sin = [d['Precio SIN'] for d in datos_comparacion]
            
            fig3.add_trace(go.Bar(
                name='CON característica',
                x=caracteristicas,
                y=precios_con,
                marker_color='#4CAF50',
                text=[f"{p:.2f}€" for p in precios_con],
                textposition='outside',
            ))
            
            fig3.add_trace(go.Bar(
                name='SIN característica',
                x=caracteristicas,
                y=precios_sin,
                marker_color='#FF9800',
                text=[f"{p:.2f}€" for p in precios_sin],
                textposition='outside',
            ))
            
            fig3.update_layout(
                title='Precio Promedio: CON vs SIN Característica',
                xaxis_title='Característica',
                yaxis_title='Precio Promedio (€)',
                barmode='group',
                height=400,
                **PLOTLY_DARK
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Tabla de impacto
            st.markdown("#### Impacto de Características")
            
            impacto_cols = st.columns(2)
            
            with impacto_cols[0]:
                st.markdown("##### Impacto en Precio")
                for dato in datos_comparacion:
                    diff_pct = (dato['Diff Precio'] / dato['Precio SIN']) * 100 if dato['Precio SIN'] > 0 else 0
                    tiene_marca = "✅" if dato['Tienes'] else "❌"
                    
                    if dato['Diff Precio'] > 0:
                        st.success(f"{tiene_marca} **{dato['Característica']}**: +{dato['Diff Precio']:.2f}€ ({diff_pct:+.1f}%)")
                    else:
                        st.info(f"{tiene_marca} **{dato['Característica']}**: {dato['Diff Precio']:.2f}€ ({diff_pct:+.1f}%)")
            
            with impacto_cols[1]:
                st.markdown("##### Impacto en Ventas")
                for dato in datos_comparacion:
                    diff_ventas_pct = (dato['Diff Ventas'] / dato['Ventas SIN']) * 100 if dato['Ventas SIN'] > 0 else 0
                    tiene_marca = "✅" if dato['Tienes'] else "❌"
                    
                    if dato['Diff Ventas'] > 0:
                        st.success(f"{tiene_marca} **{dato['Característica']}**: +{int(dato['Diff Ventas'])} ventas ({diff_ventas_pct:+.1f}%)")
                    else:
                        st.info(f"{tiene_marca} **{dato['Característica']}**: {int(dato['Diff Ventas'])} ventas ({diff_ventas_pct:+.1f}%)")
            
            # Recomendaciones
            st.markdown("#### Recomendaciones")
            caracteristicas_faltantes = [d for d in datos_comparacion if not d['Tienes'] and d['Diff Precio'] > 0]
            
            if caracteristicas_faltantes:
                # Ordenar por impacto en precio
                caracteristicas_faltantes.sort(key=lambda x: x['Diff Precio'], reverse=True)
                
                st.info(f"**Características que podrías activar:**")
                for dato in caracteristicas_faltantes[:3]:  # Top 3
                    st.markdown(f"- **{dato['Característica']}**: Podría aumentar tu precio en ~{dato['Diff Precio']:.2f}€ y ventas en ~{int(dato['Diff Ventas'])} unidades/mes")
            else:
                st.success("Tienes todas las características premium activadas.")
        
        st.markdown("---")
        
        # ========== SECCIÓN 4: DIAGNÓSTICO DE SALUD ==========
        st.markdown('<span class="section-heading-num">04</span><div class="section-heading">Diagnóstico de Salud</div>', unsafe_allow_html=True)
        st.caption("¿Cómo está mi producto?")
        
        # Calcular scores individuales
        # Score 1: Precio (percentil de precio competitivo)
        banda_precio = df_subtipo[
            (df_subtipo['precio_real'] >= producto_seleccionado['precio_real'] * 0.85) &
            (df_subtipo['precio_real'] <= producto_seleccionado['precio_real'] * 1.15)
        ]


        if len(banda_precio) >= 3:
            base = banda_precio
        else:
            n = max(10, int(len(df_subtipo) * 0.15))
            base = (
                df_subtipo.assign(
                    diff=(df_subtipo["precio_real"] - producto_seleccionado["precio_real"]).abs()
                )
                .nsmallest(n, "diff")
            )


        score_precio = float((base["ventas_mes_real"] <= producto_seleccionado["ventas_mes_real"]).mean() * 100)
        score_precio = float(np.clip(score_precio, 0, 100))
        score_precio = float((base["ventas_mes_real"] <= producto_seleccionado["ventas_mes_real"]).mean() * 100)
        score_precio = float(np.clip(score_precio, 0, 100))


        # Añadir la siguiente línea solo en la SECCIÓN 4
        pct_precio = float((df_subtipo["precio_real"] <= producto_seleccionado["precio_real"]).mean() * 100)


        # Score 2: Calidad — rating bayesiano
        # m= mediana siendo 20 el mínimo y 200 el máximo de reseñas
        m_raw = float(df_subtipo['reviews_real'].median())
        m = float(np.clip(m_raw, 20, 200))
        C = float(df_subtipo['product_rating'].mean())          # media del subtipo (prior)
        r = float(producto_seleccionado['product_rating'])
        v = float(producto_seleccionado['reviews_real'])
        bayes_rating  = (v / (v + m)) * r + (m / (v + m)) * C  # rating ajustado
        score_calidad = float(np.clip((bayes_rating - 1) / 4 * 100, 0, 100))  # escala 1-5 → 0-100


        # Score 3: Popularidad — percentil en log(ventas)
        ventas_log    = np.log1p(df_subtipo['ventas_mes_real'].clip(lower=0))
        ventas_log_ps = np.log1p(max(0.0, float(producto_seleccionado['ventas_mes_real'])))
        score_popularidad = float((ventas_log <= ventas_log_ps).mean() * 100)
        
        # Crear gauges

        gauge_cols = st.columns(3)

        def crear_gauge(valor, titulo):
            """Crea un gauge chart"""
            if valor >= 70:
                color = "#2ea84c"
                estado = "BUENA"
            elif valor >= 40:
                color = "#d29922"
                estado = "MEDIA"
            else:
                color = "#e5534b"
                estado = "BAJA"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(valor),
                title={'text': titulo, 'font': {'size': 13, 'color': '#8b9099', 'family': 'DM Sans'}},
                number={'font': {'color': color, 'family': 'DM Mono', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#4a5260', 'tickfont': {'color': '#4a5260', 'size': 10}},
                    'bar': {'color': color},
                    'bgcolor': '#111418',
                    'bordercolor': '#21262d',
                    'steps': [
                        {'range': [0, 40],  'color': 'rgba(229,83,75,0.08)'},
                        {'range': [40, 70], 'color': 'rgba(210,153,34,0.08)'},
                        {'range': [70, 100],'color': 'rgba(46,168,76,0.08)'}
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 2},
                        'thickness': 0.75,
                        'value': float(valor)
                    }
                }
            ))

            fig.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b9099')
            )

            return fig, estado

        
        with gauge_cols[0]:
            fig_precio, estado_precio = crear_gauge(score_precio, "PRECIO")
            st.plotly_chart(fig_precio, use_container_width=True)
            st.markdown(f"<p style='text-align: center; font-weight: bold;'>{estado_precio}</p>", unsafe_allow_html=True)
        
        with gauge_cols[1]:
            fig_calidad, estado_calidad = crear_gauge(score_calidad, "CALIDAD")
            st.plotly_chart(fig_calidad, use_container_width=True)
            st.markdown(f"<p style='text-align: center; font-weight: bold;'>{estado_calidad}</p>", unsafe_allow_html=True)
        
        with gauge_cols[2]:
            fig_popularidad, estado_popularidad = crear_gauge(score_popularidad, "POPULARIDAD")
            st.plotly_chart(fig_popularidad, use_container_width=True)
            st.markdown(f"<p style='text-align: center; font-weight: bold;'>{estado_popularidad}</p>", unsafe_allow_html=True)
        
        # Score general
        score_general = 0.6 * ((score_precio + score_calidad + score_popularidad) / 3) \
                    + 0.4 * min(score_precio, score_calidad, score_popularidad)
        score_general = float(np.clip(score_general, 0, 100))
        
        st.markdown("---")
        
        # Semáforo general
        if min(score_precio, score_calidad, score_popularidad) >= 70:
            color_gen_hex = "#2ea84c"; estado_general = "EXCELENTE"
        elif score_general >= 55 and min(score_precio, score_calidad, score_popularidad) >= 40:
            color_gen_hex = "#d29922"; estado_general = "BUENA"
        else:
            color_gen_hex = "#e5534b"; estado_general = "NECESITA ATENCIÓN"
        
        st.markdown(f"""
            <div class="health-panel" style="border-color:{color_gen_hex}33;">
                <div class="health-label" style="color:{color_gen_hex};">SALUD GENERAL</div>
                <div class="health-score" style="color:{color_gen_hex};">{score_general:.0f}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4a5260;margin-top:0.25rem;">/100 &nbsp;·&nbsp; {estado_general}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Resumen de fortalezas y áreas de mejora
        fortaleza_mejora_cols = st.columns(2)
        
        with fortaleza_mejora_cols[0]:
            st.markdown('<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;color:#4a5260;">Fortalezas</p>', unsafe_allow_html=True)
            
            if score_precio >= 70:
                st.success(f"Precio competitivo (Score: {score_precio:.0f})")
            if score_calidad >= 70:
                st.success(f"Buena calidad percibida (Score: {score_calidad:.0f})")
            if score_popularidad >= 70:
                st.success(f"Ventas sólidas (Score: {score_popularidad:.0f})")
            
            if score_precio < 70 and score_calidad < 70 and score_popularidad < 70:
                st.info("Hay oportunidades de mejora en todas las áreas")
        
        with fortaleza_mejora_cols[1]:
            st.markdown('<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;color:#4a5260;">Áreas de Mejora</p>', unsafe_allow_html=True)
            
            if score_precio < 70:
                if pct_precio> 75:
                    st.warning(f"💰 Precio alto vs competencia (Percentil: {pct_precio:.0f})")
                else:
                    st.warning(f"Revisa tu estrategia de precio (Score: {score_precio:.0f})")
            
            if score_calidad < 70:
                if producto_seleccionado['product_rating'] < df_subtipo['product_rating'].mean():
                    st.warning(f"Rating por debajo del promedio ({producto_seleccionado['product_rating']:.1f} vs {df_subtipo['product_rating'].mean():.1f})")
                if producto_seleccionado['reviews_real'] < df_subtipo['reviews_real'].mean():
                    st.warning(f"Pocas reviews ({int(producto_seleccionado['reviews_real'])} vs promedio: {int(df_subtipo['reviews_real'].mean())})")
            
            if score_popularidad < 70:
                st.warning(f"📊 Ventas por debajo del promedio (Percentil: {score_popularidad:.0f})")
            
            if score_precio >= 70 and score_calidad >= 70 and score_popularidad >= 70:
                st.success("Todo en orden. Sigue así.")
    
    # ==================== TAB 3: OPTIMIZAR PRECIO ====================
    elif st.session_state.selected_tab == 'optimizar':
        st.markdown('<div class="section-label">Optimizar Precio con IA</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem;color:#8b9099;margin-bottom:2rem;">Predicción de precio óptimo basada en condiciones de mercado</p>', unsafe_allow_html=True)

        col_button = st.columns([1, 2, 1])
        with col_button[1]:
            analizar = st.button("ANALIZAR PRECIO ÓPTIMO CON IA", use_container_width=True, type="primary", key="analyze_button")

        if analizar:
            st.markdown("---")
            st.markdown("## Análisis de Precio Óptimo")

            COLS_EXCLUIR = {
                'product_image_url', # URL, no es feature del modelo
                'log_original_price',# TARGET — nunca enviarlo
                'precio_real',       # columna derivada del dashboard
                'ventas_mes_real',   # columna derivada del dashboard
                'reviews_real',      # columna derivada del dashboard
            }

            payload = {}
            for col, val in producto_seleccionado.items():
                if col in COLS_EXCLUIR:
                    continue
                if isinstance(val, (np.integer,)): payload[col] = int(val)
                elif isinstance(val, (np.floating,)): payload[col] = None if np.isnan(val) else float(val)
                elif isinstance(val, (np.bool_,)): payload[col] = bool(val)
                elif isinstance(val, float) and np.isnan(val): payload[col] = None
                else: payload[col] = val

            with st.spinner("Analizando mercado y competencia..."):
                try:
                    response = requests.post(API_URL, json=payload, timeout=15)

                    if response.status_code == 200:
                        data_api = response.json()
                        precio_predicho = data_api['predicted_price']

                        st.markdown('<div class="section-label">Resultado del Análisis</div>', unsafe_allow_html=True)
                        result_cols = st.columns([1, 0.3, 1, 1])
                        with result_cols[0]:
                            st.markdown(f'<div class="result-box"><div class="result-box-label">Precio Actual</div><div class="result-box-value">{producto_seleccionado["precio_real"]:.2f} €</div></div>', unsafe_allow_html=True)
                        with result_cols[1]:
                            st.markdown('<div class="result-arrow">→</div>', unsafe_allow_html=True)
                        with result_cols[2]:
                            st.markdown(f'<div class="result-box" style="border-color:rgba(201,147,58,0.3);"><div class="result-box-label">Precio Sugerido IA</div><div class="result-box-value gold">{precio_predicho:.2f} €</div></div>', unsafe_allow_html=True)
                        with result_cols[3]:
                            diferencia = precio_predicho - producto_seleccionado['precio_real']
                            porcentaje_cambio = (diferencia / producto_seleccionado['precio_real']) * 100
                            diff_color = "#2ea84c" if diferencia > 0 else "#e5534b" if diferencia < 0 else "#8b9099"
                            signo = "+" if diferencia > 0 else ""
                            st.markdown(f'<div class="result-box" style="border-color:{diff_color}33;"><div class="result-box-label">Diferencia</div><div class="result-box-value" style="color:{diff_color};font-size:1.8rem;">{signo}{diferencia:.2f} €</div><div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:{diff_color};margin-top:0.25rem;">{porcentaje_cambio:+.1f}%</div></div>', unsafe_allow_html=True)

                        # --- MOTOR DE CRECIMIENTO (GROWTH OPTIMIZATION) ---
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Motor de Crecimiento (Revenue Optimization)</div>', unsafe_allow_html=True)

                        # 1. Crear el "Competitive Bucket" usando KNN (Similitud Textual)
                        vectorizer, knn_model, df_knn = get_knn_engine(df)
                        titulo_actual = str(producto_seleccionado['original_title'])
                        
                        vec_entrada = vectorizer.transform([titulo_actual])
                        distancias, indices = knn_model.kneighbors(vec_entrada)
                        
                        # RED MÁS AMPLIA: +/- 40% del precio sugerido para captar más mercado
                        rango_min = precio_predicho * 0.60
                        rango_max = precio_predicho * 1.40
                        
                        indices_validos = []
                        for d, idx in zip(distancias[0], indices[0]):
                            # RED MÁS AMPLIA: Distancia < 0.70
                            if d < 0.70:
                                row_vecino = df_knn.iloc[idx]
                                if rango_min <= row_vecino['precio_real'] <= rango_max:
                                    indices_validos.append(idx)
                                    
                        df_bucket = df_knn.iloc[indices_validos]

                        # Fallback de seguridad: Si hay menos de 4 competidores exactos, 
                        # cogemos todo el subtipo en ese rango de precios.
                        if len(df_bucket) < 4:
                            df_bucket = df[(df['subtype'] == producto_seleccionado['subtype']) & 
                                           (df['precio_real'] >= rango_min) & 
                                           (df['precio_real'] <= rango_max)]

                        # 2. Definir Palancas de Acción (Levers)
                        levers = [
                            {'id': 'has_coupon', 'target': 1, 'current': producto_seleccionado['has_coupon'], 'name': 'Activar Cupón Promocional', 'icon': '🎟️', 'diff': 'Baja'},
                            {'id': 'is_sponsored', 'target': 'Yes', 'current': producto_seleccionado['is_sponsored'], 'name': 'Activar Campaña Sponsored', 'icon': '📢', 'diff': 'Baja'},
                            {'id': 'sustainability_tags', 'target': 1, 'current': producto_seleccionado['sustainability_tags'], 'name': 'Certificación Eco / Sostenible', 'icon': '🌱', 'diff': 'Media'},
                            {'id': 'buy_box_availability', 'target': 1, 'current': producto_seleccionado['buy_box_availability'], 'name': 'Asegurar Buy Box (Stock/Prime)', 'icon': '📦', 'diff': 'Alta'}
                        ]

                        recomendaciones = []
                        ventas_actuales = producto_seleccionado['ventas_mes_real']
                        
                        # Obtenemos datos macro (todo el subtipo) por si el bucket local falla
                        df_subtipo_entero = df[df['subtype'] == producto_seleccionado['subtype']]

                        # 3. Calcular el Incremento de Ventas (Lift Analysis Inteligente)
                        for lever in levers:
                            # Solo analizamos las cosas que el producto AÚN NO TIENE
                            if lever['current'] != lever['target']: 
                                col = lever['id']
                                val_target = lever['target']
                                
                                competidores_con = df_bucket[df_bucket[col] == val_target]['ventas_mes_real']
                                competidores_sin = df_bucket[df_bucket[col] != val_target]['ventas_mes_real']
                                
                                penetracion = (len(competidores_con) / len(df_bucket)) * 100 if len(df_bucket) > 0 else 0
                                
                                lift_pct = 0.0
                                motivo_estrategico = ""
                                
                                # ESTRATEGIA A: Lift puro en competidores directos
                                if len(competidores_con) >= 1 and len(competidores_sin) >= 1:
                                    m_con = competidores_con.median()
                                    m_sin = competidores_sin.median()
                                    if m_con > m_sin and m_sin > 0:
                                        lift_pct = (m_con - m_sin) / m_sin
                                        motivo_estrategico = f"El <b style='color:#c9933a; font-size:1.1em;'>{penetracion:.0f}%</b> de tus rivales directos lo usan y venden más."
                                
                                # ESTRATEGIA B: Fallback a la Macro-Categoría
                                if lift_pct == 0.0 and penetracion > 15:
                                    macro_con = df_subtipo_entero[df_subtipo_entero[col] == val_target]['ventas_mes_real']
                                    macro_sin = df_subtipo_entero[df_subtipo_entero[col] != val_target]['ventas_mes_real']
                                    if len(macro_con) > 3 and len(macro_sin) > 3:
                                        m_macro_con = macro_con.median()
                                        m_macro_sin = macro_sin.median()
                                        if m_macro_con > m_macro_sin and m_macro_sin > 0:
                                            lift_pct = min((m_macro_con - m_macro_sin) / m_macro_sin, 0.50)
                                            motivo_estrategico = f"Tendencia en tu categoría: venderás un <b style='color:#2ea84c; font-size:1.1em;'>+{lift_pct*100:.0f}%</b> más de media."

                                # ESTRATEGIA C: Recomendación Defensiva (Estándar de mercado)
                                if lift_pct == 0.0 and penetracion >= 45:
                                    lift_pct = 0.08 
                                    motivo_estrategico = f"Estándar de mercado: el <b style='color:#c9933a; font-size:1.1em;'>{penetracion:.0f}%</b> ya lo tiene. Estás perdiendo visibilidad."

                                # Si hemos logrado obtener un impacto positivo, lo guardamos
                                if lift_pct > 0:
                                    # Limitamos el optimismo extremo (nadie vende un +500% solo por un tag eco)
                                    lift_pct = min(lift_pct, 0.60) 
                                    
                                    ventas_extra = ventas_actuales * lift_pct
                                    ingreso_extra_mensual = ventas_extra * precio_predicho
                                    
                                    recomendaciones.append({
                                        'Accion': f"{lever['icon']} {lever['name']}",
                                        'Dificultad': lever['diff'],
                                        'Lift_PCT': lift_pct * 100,
                                        'Ingreso_Extra': ingreso_extra_mensual,
                                        'Motivo': motivo_estrategico
                                    })

                        # 4. Mostrar el Plan de Acción Visual
                        if recomendaciones:
                            recomendaciones.sort(key=lambda x: x['Ingreso_Extra'], reverse=True)
                            
                            ingreso_base = ventas_actuales * precio_predicho
                            mejor_accion = recomendaciones[0]
                            nuevo_ingreso = ingreso_base + mejor_accion['Ingreso_Extra']
                            
                            st.info(
                                f"💡 **Simulador de Revenue:** Ajustando el precio a **{precio_predicho:.2f}€** "
                                f"y aplicando la acción principal recomendada, podrías pasar de facturar "
                                f"**{ingreso_base:,.0f}€** a **{nuevo_ingreso:,.0f}€** mensuales."
                            )
                            
                            st.markdown(f"<p style='font-size:0.75rem; color:#8b9099;'>Análisis impulsado por KNN basado en competidores similares.</p>", unsafe_allow_html=True)
                            
                            for rec in recomendaciones:
                                # Colores exactos para la dificultad
                                diff_color = "#3fb950" if rec['Dificultad'] == "Baja" else "#d29922" if rec['Dificultad'] == "Media" else "#e5534b"
                                
                                st.markdown(f"""
                                <div style="background:#161b22; border:1px solid #21262d; border-radius:8px; padding:1.2rem; margin-bottom:0.8rem; display:flex; justify-content:space-between; align-items:center;">
                                    <div style="flex:2;">
                                        <div style="font-weight:600; font-size:1.15rem; color:#e8e6e1; letter-spacing:0.02em;">
                                            {rec['Accion']}
                                        </div>
                                        <div style="font-size:0.9rem; color:#b3b8c2; margin-top:0.35rem; line-height:1.4;">
                                            {rec['Motivo']}
                                        </div>
                                    </div>
                                    <div style="flex:1; text-align:center;">
                                        <span style="font-size:0.75rem; color:{diff_color}; border:1px solid {diff_color}55; background:{diff_color}11; padding:0.3rem 0.8rem; border-radius:15px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                                            Dificultad {rec['Dificultad']}
                                        </span>
                                    </div>
                                    <div style="flex:1.5; text-align:right;">
                                        <div style="color:#2ea84c; font-family:'DM Mono', monospace; font-size:1.4rem; font-weight:500;">
                                            +{rec['Lift_PCT']:.0f}% ventas
                                        </div>
                                        <div style="color:#c9933a; font-family:'DM Mono', monospace; font-size:0.95rem; margin-top:0.15rem;">
                                            + {rec['Ingreso_Extra']:,.0f} € / mes
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.success("🏆 **¡Posición de ventaja!** Tu producto ya cuenta con las palancas clave para tu segmento. Céntrate en mantener el precio sugerido y conseguir más reviews orgánicas.")

                        st.markdown("---")
                        
                        # --- NUEVOS TÍTULOS GIGANTES Y EXPLICATIVOS ---
                        st.markdown(f'''
                            <div style="margin-top:1.5rem; margin-bottom:2rem;">
                                <span style="font-family:var(--font-mono); font-size:0.85rem; color:var(--gold); letter-spacing:0.15em; text-transform:uppercase;">
                                    Simulación Estratégica
                                </span>
                                <h2 style="font-family:var(--font-display); font-size:2.6rem; font-weight:300; color:#ffffff; margin:0.3rem 0;">
                                    Tu Posicionamiento a <span style="color:var(--gold); font-weight:600;">{precio_predicho:.2f} €</span>
                                </h2>
                                <p style="color:#b3b8c2; font-size:1.05rem; margin-top:0.4rem; line-height:1.5;">
                                    Descubre cómo cambia tu mapa de batalla si adoptas el precio sugerido. 
                                    A partir de aquí, <b>comparamos a tus rivales directamente contra tu nuevo Objetivo IA</b>.
                                </p>
                            </div>
                        ''', unsafe_allow_html=True)

                        tab1, tab2 = st.tabs(["Productos Similares (Precio)", "Posicionamiento de Mercado"])

                        with tab1:
                            # 1. Recuperamos los verdaderos rivales (El bucket de KNN)
                            if 'df_bucket' in locals() and len(df_bucket) > 0:
                                df_plot = df_bucket.copy()
                            else:
                                df_plot = df[(df['subtype'] == producto_seleccionado['subtype'])]
                            
                            df_plot['short_title'] = df_plot['original_title'].apply(lambda x: str(x)[:55] + '...' if len(str(x)) > 55 else str(x))
                            
                            # Escala de burbujas
                            max_rev = df_plot['reviews_real'].max() if len(df_plot) > 0 and df_plot['reviews_real'].max() > 0 else 1
                            df_plot['bubble_size'] = 8 + np.sqrt(df_plot['reviews_real'] / max_rev) * 45

                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=df_plot['precio_real'], y=df_plot['ventas_mes_real'],
                                mode='markers',
                                marker=dict(size=df_plot['bubble_size'], color='#388bfd', opacity=0.5, line=dict(width=1, color='#8b9099')),
                                customdata=np.stack((df_plot['brand'], df_plot['short_title'], df_plot['product_rating'], df_plot['reviews_real']), axis=-1),
                                hovertemplate="<b style='color:#388bfd; font-size:1.1em;'>%{customdata[0]}</b><br><span style='color:#b3b8c2;'>%{customdata[1]}</span><br><br><b>Precio:</b> %{x:.2f} €<br><b>Ventas/mes:</b> %{y:.0f}<br><b>Rating:</b> %{customdata[2]:.1f} ⭐<br><b>Reviews:</b> %{customdata[3]}<br><extra></extra>",
                                name="Rivales Directos (KNN)"
                            ))

                            fig.add_annotation(
                                x=precio_predicho, y=producto_seleccionado['ventas_mes_real'],
                                ax=producto_seleccionado['precio_real'], ay=producto_seleccionado['ventas_mes_real'],
                                xref="x", yref="y", axref="x", ayref="y",
                                text="", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1, arrowcolor="#c9933a", opacity=0.6
                            )

                            fig.add_trace(go.Scatter(
                                x=[producto_seleccionado['precio_real']], y=[producto_seleccionado['ventas_mes_real']],
                                mode='markers', marker=dict(size=10, color='#e5534b', opacity=0.4), hoverinfo='skip', name="Posición Actual"
                            ))

                            fig.add_trace(go.Scatter(
                                x=[precio_predicho], y=[producto_seleccionado['ventas_mes_real']],
                                mode='markers+text', marker=dict(size=16, color='#c9933a', symbol='diamond', line=dict(width=2, color='#e8e6e1')),
                                text=['TARGET IA'], textposition='top center', textfont=dict(size=11, color='#c9933a', family='DM Mono', weight='bold'),
                                hovertemplate="<b style='color:#c9933a'>TU NUEVO POSICIONAMIENTO</b><br><b>Precio Óptimo:</b> %{x:.2f} €<br><b>Ventas Actuales:</b> %{y:.0f}<br><extra></extra>",
                                name='Objetivo IA'
                            ))

                            if len(df_plot) > 0:
                                median_price = df_plot['precio_real'].median()
                                median_sales = df_plot['ventas_mes_real'].median()
                                fig.add_hline(y=median_sales, line_dash="dot", line_color="#4a5260", opacity=0.5, annotation_text="Ventas Medias", annotation_font_color="#8b9099", annotation_position="bottom right")
                                fig.add_vline(x=median_price, line_dash="dot", line_color="#4a5260", opacity=0.5, annotation_text="Precio Medio", annotation_font_color="#8b9099", annotation_position="top left")

                            fig.update_layout(**PLOTLY_DARK)
                            fig.update_layout(
                                title=dict(text=f"Mapa de Batalla: {len(df_plot)} Rivales encontrados por IA (KNN)", font=dict(color='#e8e6e1', size=16, family='Cormorant Garamond')),
                                xaxis_title="Precio (€)", yaxis_title="Ventas último mes", height=550, showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(17,20,24,0.8)'),
                                hoverlabel=dict(bgcolor="#161b22", bordercolor="#21262d", font_size=13, font_family="DM Sans")
                            )
                            fig.add_annotation(text="* El tamaño de la burbuja representa el volumen de Reviews", xref="paper", yref="paper", x=0, y=-0.12, showarrow=False, font=dict(color="#4a5260", size=11, family="DM Sans"))
                            st.plotly_chart(fig, use_container_width=True)

                            # --- TITULAR GIGANTE PARA LA TABLA ---
                            st.markdown(f'''
                                <div style="margin-top:3.5rem; margin-bottom:1.5rem;">
                                    <h3 style="font-family:var(--font-display); font-size:1.8rem; font-weight:400; color:#ffffff; margin:0;">
                                        Radiografía de Competidores Directos
                                    </h3>
                                    <p style="color:#8b9099; font-size:0.95rem; margin-top:0.4rem;">
                                        La columna <b>"Diff. Precio"</b> compara a tus rivales contra tu nuevo <b>Precio Sugerido ({precio_predicho:.2f} €)</b>. 
                                        <span style="color:#2ea84c; font-weight:600;">
                                    </p>
                                </div>
                            ''', unsafe_allow_html=True)

                            # --- TABLA HTML ---
                            comparison_df = df_plot.copy()
                            if 'ventas_mes_real' in comparison_df.columns:
                                comparison_df = comparison_df.sort_values('ventas_mes_real', ascending=False)
                            
                            rows_html = ""
                            for _, comp in comparison_df.iterrows():
                                # AHORA LA MATEMÁTICA SE CALCULA CONTRA EL PRECIO DE LA IA, NO EL ACTUAL
                                diff_precio = comp.get('precio_real', 0) - precio_predicho
                                color_diff = "#e5534b" if diff_precio < 0 else ("#2ea84c" if diff_precio > 0 else "#8b9099")
                                
                                img_url = comp.get('product_image_url', '')
                                img_tag = f'<img src="{img_url}" style="width:40px;height:40px;object-fit:contain;border-radius:4px;background:#fff;padding:2px;">' if pd.notna(img_url) and img_url else 'N/A'
                                
                                titulo = str(comp.get('original_title', ''))
                                titulo_corto = titulo[:55] + '...' if len(titulo) > 55 else titulo
                                
                                marca = str(comp.get('brand', ''))
                                precio_str = f"{comp.get('precio_real', 0):.2f} €"
                                diff_str = f'<span style="color:{color_diff}; font-weight:600;">{diff_precio:+.2f} €</span>'
                                ventas_str = f"{int(comp.get('ventas_mes_real', 0))} uds"
                                rating_str = f"{comp.get('product_rating', 0):.1f} ⭐"
                                reviews_str = f"{int(comp.get('reviews_real', 0))}"
                                
                                badge_str = '🏆' if comp.get('is_best_seller') == 'Yes' else ''
                                cupon_str = '🎟️' if comp.get('has_coupon') == 1 else ''
                                buybox_str = '📦' if comp.get('buy_box_availability') == 1 else ''
                                premium_str = '👑' if comp.get('is_premium_brand') else ''
                                
                                link_url = str(comp.get('product_url', ''))
                                if link_url and link_url.lower() != 'nan':
                                    link_url = link_url.split(',202')[0]
                                    if link_url.startswith('/'):
                                        link_url = f"https://www.amazon.es{link_url}"
                                    link_tag = f'<a href="{link_url}" target="_blank" style="color:#c9933a;text-decoration:none;font-weight:600;font-size:0.75rem;">Ver ↗️</a>'
                                else:
                                    link_tag = ''
                                
                                cells = (
                                    f'<td style="padding:8px 12px;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{img_tag}</td>'
                                    f'<td style="padding:8px 12px;font-size:0.8rem;color:#e8e6e1;border-bottom:1px solid #21262d1a;vertical-align:middle;max-width:250px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{titulo}">{titulo_corto}</td>'
                                    f'<td style="padding:8px 12px;font-size:0.8rem;color:#b3b8c2;border-bottom:1px solid #21262d1a;vertical-align:middle;">{marca}</td>'
                                    f'<td style="padding:8px 12px;font-size:0.85rem;color:#e8e6e1;font-family:var(--font-mono);border-bottom:1px solid #21262d1a;vertical-align:middle;">{precio_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:0.85rem;font-family:var(--font-mono);border-bottom:1px solid #21262d1a;vertical-align:middle;">{diff_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:0.85rem;color:#e8e6e1;font-family:var(--font-mono);border-bottom:1px solid #21262d1a;vertical-align:middle;">{ventas_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:0.8rem;color:#e8e6e1;border-bottom:1px solid #21262d1a;vertical-align:middle;">{rating_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:0.8rem;color:#e8e6e1;border-bottom:1px solid #21262d1a;vertical-align:middle;">{reviews_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{badge_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{cupon_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{buybox_str}</td>'
                                    f'<td style="padding:8px 12px;font-size:1.1rem;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{premium_str}</td>'
                                    f'<td style="padding:8px 12px;border-bottom:1px solid #21262d1a;vertical-align:middle;text-align:center;">{link_tag}</td>'
                                )
                                rows_html += f'<tr style="transition:background 0.15s;" onmouseover="this.style.background=\'#1c2330\'" onmouseout="this.style.background=\'transparent\'">{cells}</tr>'
                            
                            headers_list = ['Img', 'Producto', 'Marca', 'Precio', 'Diff. Precio', 'Ventas', 'Rating', 'Reviews', 'Badge', 'Cupón', 'Buy Box', 'Premium', 'Link']
                            headers_html = ''.join(f'<th style="position:sticky; top:0; background:#161b22; z-index:10; padding:10px 12px; text-align:center; font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#8b9099; border-bottom:1px solid #21262d; white-space:nowrap;">{col}</th>' for col in headers_list)

                            html_completo = f"""
                            <div style="overflow-x:auto; overflow-y:auto; max-height:450px; border:1px solid #21262d; border-radius:6px; margin-top:10px; position:relative; display:block;">
                            <table style="width:100%; border-collapse:collapse; background:#161b22;">
                            <thead><tr>{headers_html}</tr></thead>
                            <tbody>{rows_html}</tbody>
                            </table>
                            </div>
                            """
                            st.markdown(html_completo, unsafe_allow_html=True)
                            
                                
                        with tab2:
                            # 1. Recuperamos tus rivales directos
                            if 'df_bucket' in locals() and len(df_bucket) > 0:
                                df_plot_t2 = df_bucket.copy()
                            else:
                                df_plot_t2 = df[(df['subtype'] == producto_seleccionado['subtype'])]
                                
                            df_plot_t2['short_title'] = df_plot_t2['original_title'].apply(lambda x: str(x)[:55] + '...' if len(str(x)) > 55 else str(x))
                            
                            st.markdown(f'''
                                <div style="margin-top:0.5rem; margin-bottom:1.5rem;">
                                    <h3 style="font-family:var(--font-display); font-size:1.6rem; font-weight:400; color:#ffffff; margin:0;">
                                        Relación Precio vs Calidad (Rating)
                                    </h3>
                                    <p style="color:#8b9099; font-size:0.95rem; margin-top:0.4rem;">
                                        Descubre si tu nuevo <b>Objetivo IA</b> te sitúa como una opción "Low Cost", "Premium" o "Calidad-Precio" frente a tus verdaderos rivales.
                                    </p>
                                </div>
                            ''', unsafe_allow_html=True)

                            # 2. Configurar la Gráfica de Cuadrantes
                            max_sales = df_plot_t2['ventas_mes_real'].max() if len(df_plot_t2) > 0 and df_plot_t2['ventas_mes_real'].max() > 0 else 1
                            df_plot_t2['bubble_size'] = 10 + np.sqrt(df_plot_t2['ventas_mes_real'] / max_sales) * 35

                            fig2 = go.Figure()

                            # Traza 1: Competidores
                            fig2.add_trace(go.Scatter(
                                x=df_plot_t2['precio_real'],
                                y=df_plot_t2['product_rating'],
                                mode='markers',
                                marker=dict(size=df_plot_t2['bubble_size'], color='#388bfd', opacity=0.5, line=dict(width=1, color='#8b9099')),
                                customdata=np.stack((df_plot_t2['brand'], df_plot_t2['short_title'], df_plot_t2['ventas_mes_real'], df_plot_t2['reviews_real']), axis=-1),
                                hovertemplate=(
                                    "<b style='color:#388bfd; font-size:1.1em;'>%{customdata[0]}</b><br>"
                                    "<span style='color:#b3b8c2;'>%{customdata[1]}</span><br><br>"
                                    "<b>Precio:</b> %{x:.2f} €<br>"
                                    "<b>Rating:</b> %{y:.1f} ⭐<br>"
                                    "<b>Ventas/mes:</b> %{customdata[2]}<br>"
                                    "<extra></extra>"
                                ),
                                name="Rivales Directos"
                            ))

                            # Traza 2: Línea de movimiento (Estrategia)
                            fig2.add_annotation(
                                x=precio_predicho, y=producto_seleccionado['product_rating'],
                                ax=producto_seleccionado['precio_real'], ay=producto_seleccionado['product_rating'],
                                xref="x", yref="y", axref="x", ayref="y",
                                text="", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#c9933a", opacity=0.6
                            )

                            # Traza 3: Posición Actual
                            fig2.add_trace(go.Scatter(
                                x=[producto_seleccionado['precio_real']], y=[producto_seleccionado['product_rating']],
                                mode='markers', marker=dict(size=10, color='#e5534b', opacity=0.4), hoverinfo='skip', name="Posición Actual"
                            ))

                            # Traza 4: Objetivo IA
                            fig2.add_trace(go.Scatter(
                                x=[precio_predicho], y=[producto_seleccionado['product_rating']],
                                mode='markers+text', marker=dict(size=18, color='#c9933a', symbol='diamond', line=dict(width=2, color='#e8e6e1')),
                                text=['TARGET IA'], textposition='top center', textfont=dict(size=11, color='#c9933a', family='DM Mono', weight='bold'),
                                hovertemplate="<b style='color:#c9933a'>NUEVO POSICIONAMIENTO</b><br><b>Precio IA:</b> %{x:.2f} €<br><b>Rating:</b> %{y:.1f} ⭐<br><extra></extra>",
                                name='Objetivo IA'
                            ))

                            # Medianas del nicho para dibujar los cuadrantes
                            if len(df_plot_t2) > 0:
                                median_price = df_plot_t2['precio_real'].median()
                                median_rating = df_plot_t2['product_rating'].median()
                                fig2.add_hline(y=median_rating, line_dash="dot", line_color="#4a5260", opacity=0.5, annotation_text=f"Rating Medio ({median_rating:.1f})", annotation_font_color="#8b9099", annotation_position="bottom right")
                                fig2.add_vline(x=median_price, line_dash="dot", line_color="#4a5260", opacity=0.5, annotation_text=f"Precio Medio ({median_price:.2f}€)", annotation_font_color="#8b9099", annotation_position="top left")

                            # Aplicamos estilos y corregimos el "undefined" forzando un título vacío
                            fig2.update_layout(**PLOTLY_DARK)
                            fig2.update_layout(
                                title=dict(text=""), # <--- ESTO ELIMINA EL "undefined"
                                xaxis_title="Precio (€)", 
                                yaxis_title="Calidad Percibida (Rating ⭐)", 
                                height=500, 
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(17,20,24,0.8)'),
                                hoverlabel=dict(bgcolor="#161b22", bordercolor="#21262d", font_size=13, font_family="DM Sans"),
                                yaxis=dict(range=[max(0, df_plot_t2['product_rating'].min() - 0.5), min(5.1, df_plot_t2['product_rating'].max() + 0.5)]),
                                margin=dict(t=30) # Reducimos el margen superior para que quede más compacto
                            )
                            
                            # Añadimos la leyenda del tamaño de la burbuja en la esquina inferior izquierda
                            fig2.add_annotation(
                                text="* El tamaño de la burbuja representa el volumen de Ventas Mensuales",
                                xref="paper", yref="paper", x=0, y=-0.14, showarrow=False,
                                font=dict(color="#4a5260", size=11, family="DM Sans")
                            )

                            st.plotly_chart(fig2, use_container_width=True)

                            # 3. Métricas Explicativas Nivel "Pro"
                            st.markdown('<p style="font-size:0.75rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;color:#8b9099;margin-top:1.5rem;margin-bottom:1rem;">Lectura Estratégica</p>', unsafe_allow_html=True)
                            
                            col_ins1, col_ins2, col_ins3 = st.columns(3)
                            
                            if len(df_plot_t2) > 0:
                                # Cálculos de impacto basados en el NUEVO precio
                                pct_precio_nuevo = (df_plot_t2['precio_real'] > precio_predicho).sum() / len(df_plot_t2) * 100
                                pct_precio_antiguo = (df_plot_t2['precio_real'] > producto_seleccionado['precio_real']).sum() / len(df_plot_t2) * 100
                                
                                pct_rating = (df_plot_t2['product_rating'] < producto_seleccionado['product_rating']).sum() / len(df_plot_t2) * 100
                                dif_pct = pct_precio_nuevo - pct_precio_antiguo
                                
                                label_diff = f"<span style='color:var(--success); font-weight:600;'>+{dif_pct:.0f}%</span> de mejora" if dif_pct > 0 else (f"<span style='color:var(--danger); font-weight:600;'>{dif_pct:.0f}%</span>" if dif_pct < 0 else "Sin cambio")

                                with col_ins1:
                                    st.markdown(f"""
                                    <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:6px; padding:1.2rem; height:100%;">
                                        <div style="font-family:var(--font-body); font-size:0.7rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Estrategia de Precio IA</div>
                                        <div style="font-family:var(--font-mono); font-size:1.8rem; color:var(--gold); line-height:1.2; margin-bottom:0.5rem;">{pct_precio_nuevo:.0f}%</div>
                                        <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.4;">
                                            Con el Precio IA, serás <b>más barato que el {pct_precio_nuevo:.0f}%</b> de tus rivales directos. ({label_diff} vs tu precio actual).
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_ins2:
                                    color_rating = "var(--success)" if pct_rating >= 50 else "var(--warning)"
                                    st.markdown(f"""
                                    <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:6px; padding:1.2rem; height:100%;">
                                        <div style="font-family:var(--font-body); font-size:0.7rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Autoridad (Rating)</div>
                                        <div style="font-family:var(--font-mono); font-size:1.8rem; color:{color_rating}; line-height:1.2; margin-bottom:0.5rem;">Top {100-pct_rating:.0f}%</div>
                                        <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.4;">
                                            Tu calificación ({producto_seleccionado['product_rating']:.1f} ⭐) <b>supera al {pct_rating:.0f}%</b> de los productos en tu anillo de competencia.
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_ins3:
                                    st.markdown(f"""
                                    <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:6px; padding:1.2rem; height:100%;">
                                        <div style="font-family:var(--font-body); font-size:0.7rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Volumen de Análisis</div>
                                        <div style="font-family:var(--font-mono); font-size:1.8rem; color:var(--info); line-height:1.2; margin-bottom:0.5rem;">{len(df_plot_t2)}</div>
                                        <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.4;">
                                            Competidores directos analizados para simular este escenario de mercado.
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                    else:
                        st.error(f"Error en la API: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Error de conexión. ¿Está encendida la API en el puerto 8001?")
                except Exception as e:
                    st.error(f"Error inesperado: {e}")
                    st.exception(e)

else:
    st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;'>
            <p style='font-family:"DM Mono",monospace;font-size:0.7rem;letter-spacing:0.2em;color:#4a5260;text-transform:uppercase;'>
                Selecciona un producto del catálogo para comenzar el análisis
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<div class="dash-footer">Powered by Machine Learning &nbsp;·&nbsp; Amazon Price Optimizer v3.0</div>', unsafe_allow_html=True)