import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
import os
from PIL import Image
from io import BytesIO

# Configuración
API_URL = "http://127.0.0.1:8001/predict"
DATASET_PATH = "data_dashboard.csv"

st.set_page_config(page_title="Amazon Price Optimizer", layout="wide", initial_sidebar_state="collapsed")

# --- ESTILOS CUSTOM ---
st.markdown("""
    <style>
    /* Ocultar sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Card de producto */
    .product-card {
        border: 3px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        border-color: #FF9900;
    }
    
    .product-card-selected {
        border: 4px solid #FF9900;
        background: linear-gradient(135deg, #fff5e6 0%, #ffe6cc 100%);
        box-shadow: 0 8px 20px rgba(255,153,0,0.3);
    }
    
    .product-title {
        font-size: 1.1em;
        font-weight: bold;
        color: #232f3e;
        margin-bottom: 10px;
        height: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    .product-price {
        font-size: 2em;
        color: #FF9900;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .product-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        margin: 3px;
        font-weight: 600;
        white-space: nowrap;
    }
    
    .badge-bestseller {
        background-color: #ff9900;
        color: white;
    }
    
    .badge-coupon {
        background-color: #00a8e1;
        color: white;
    }
    
    .badge-buybox {
        background-color: #067d62;
        color: white;
    }
    
    .badge-sustainable {
        background-color: #7fda89;
        color: #232f3e;
    }
    
    .badge-premium {
        background-color: #9c27b0;
        color: white;
    }
    
    .badge-sponsored {
        background-color: #ff5722;
        color: white;
    }
    
    .metric-small {
        font-size: 0.9em;
        color: #666;
        margin: 5px 0;
    }
    
    .category-tag {
        background-color: #232f3e;
        color: white;
        padding: 5px 12px;
        border-radius: 5px;
        font-size: 0.85em;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    .subtype-tag {
        background-color: #546e7a;
        color: white;
        padding: 5px 12px;
        border-radius: 5px;
        font-size: 0.75em;
        display: inline-block;
        margin-bottom: 10px;
        margin-left: 5px;
    }
    
    /* Scroll horizontal personalizado */
    .product-carousel {
        display: flex;
        overflow-x: auto;
        gap: 20px;
        padding: 20px 0;
        scroll-behavior: smooth;
    }
    
    .product-carousel::-webkit-scrollbar {
        height: 10px;
    }
    
    .product-carousel::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .product-carousel::-webkit-scrollbar-thumb {
        background: #FF9900;
        border-radius: 10px;
    }
    
    .product-carousel::-webkit-scrollbar-thumb:hover {
        background: #e88b00;
    }
    
    .big-metric {
        font-size: 3em;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
    }
    
    .product-image-container {
        width: 100%;
        height: 250px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background-color: #f8f8f8;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .product-image-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. CARGA DE DATOS ---
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
            st.error(f"⛔ No se encuentra el archivo en: {DATASET_PATH}")
            return None
        
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.lower()
        
        # Preprocesamiento: convertir logs a valores reales
        df['precio_real'] = np.exp(df['log_original_price'])
        df['ventas_mes_real'] = np.exp(df['log_purchased_last_month'])
        df['reviews_real'] = np.exp(df['log_total_reviews'])
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

def get_sample_products(df, n_products=5):
    """Selecciona 5 productos de 5 categorías diferentes"""
    categorias = df['category'].unique()
    
    # Asegurarnos de tener al menos 5 categorías
    if len(categorias) < n_products:
        st.warning(f"Solo hay {len(categorias)} categorías disponibles")
        n_products = len(categorias)
    
    # Seleccionar 5 categorías aleatorias
    categorias_seleccionadas = np.random.choice(categorias, size=n_products, replace=False)
    
    productos_muestra = []
    for categoria in categorias_seleccionadas:
        # De cada categoría, tomar un producto aleatorio con buenas métricas
        productos_categoria = df[df['category'] == categoria]
        
        # Priorizar productos con buen rating y ventas
        productos_categoria = productos_categoria[
            (productos_categoria['product_rating'] >= 3.5) &
            (productos_categoria['ventas_mes_real'] > 10)
        ]
        
        if len(productos_categoria) > 0:
            producto = productos_categoria.sample(n=1).iloc[0]
            productos_muestra.append(producto)
    
    return productos_muestra

df = load_data()

if df is None:
    st.stop()

# Inicializar session state
if 'productos_muestra' not in st.session_state:
    st.session_state.productos_muestra = get_sample_products(df)
    st.session_state.producto_seleccionado_idx = None

if 'refresh' not in st.session_state:
    st.session_state.refresh = 0

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'resumen'

# --- 2. HEADER ---
col_title, col_refresh = st.columns([4, 1])

with col_title:
    st.title("🛒 Amazon Smart Pricing Dashboard")
    st.markdown("### Selecciona un producto para analizar su precio óptimo")

with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Cambiar Productos", use_container_width=True):
        st.session_state.productos_muestra = get_sample_products(df)
        st.session_state.producto_seleccionado_idx = None
        st.rerun()

st.markdown("---")

# --- 3. CARRUSEL DE PRODUCTOS ---
st.markdown("#### 📦 Catálogo de Productos")

# Crear columnas para los productos
cols = st.columns(5)

for idx, producto in enumerate(st.session_state.productos_muestra):
    with cols[idx]:
        # Determinar si está seleccionado
        is_selected = st.session_state.producto_seleccionado_idx == idx
        
        # Container para el producto
        with st.container():
            # Imagen placeholder (ya no tenemos URLs de imágenes)
            # Imagen con contenedor de altura fija
            img_container = st.container()
            with img_container:
                img = load_image_from_url(producto['product_image_url'])
                if img:
                    # Redimensionar la imagen a un tamaño fijo antes de mostrarla
                    img_resized = img.copy()
                    img_resized.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    
                    # Crear imagen con fondo blanco del mismo tamaño
                    background = Image.new('RGB', (300, 300), (255, 255, 255))
                    # Centrar la imagen
                    offset = ((300 - img_resized.size[0]) // 2, (300 - img_resized.size[1]) // 2)
                    background.paste(img_resized, offset)
                    
                    st.image(background, use_container_width=True)
                else:
                    # Crear imagen placeholder del mismo tamaño
                    placeholder = Image.new('RGB', (300, 300), (240, 240, 240))
                    st.image(placeholder, use_container_width=True)
            
            # Categoría y Subtipo
            st.markdown(f'<div class="category-tag">{producto["category"]}</div><div class="subtype-tag">{producto["subtype"]}</div>', 
                       unsafe_allow_html=True)
            
            # Título (usamos brand + subtype como título)
            titulo_producto = f"{producto['original_title']}"
            st.markdown(f'<div class="product-title">{titulo_producto}</div>', 
                       unsafe_allow_html=True)
            
            # Precio
            st.markdown(f'<div class="product-price">{producto["precio_real"]:.2f} €</div>', 
                       unsafe_allow_html=True)
            
            # Métricas pequeñas
            st.markdown(f'''
                <div class="metric-small">⭐ {producto["product_rating"]}/5.0 ({int(producto["reviews_real"])} reviews)</div>
                <div class="metric-small">📊 {int(producto["ventas_mes_real"])} ventas/mes</div>
                <div class="metric-small">🏭 {producto["market_tier"]} | {producto["condition"]}</div>
            ''', unsafe_allow_html=True)
            
            # Badges - reservar espacio fijo
            badges_html = '<div style="min-height: 70px; margin: 10px 0;">'
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
            badges_html += '</div>'
            
            st.markdown(badges_html, unsafe_allow_html=True)
            
            # Botón de selección
            if st.button(
                "✅ Seleccionar" if is_selected else "👉 Seleccionar",
                key=f"select_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.producto_seleccionado_idx = idx
                st.rerun()

st.markdown("---")

# --- 4. ANÁLISIS DEL PRODUCTO SELECCIONADO ---
if st.session_state.producto_seleccionado_idx is not None:
    producto_seleccionado = st.session_state.productos_muestra[st.session_state.producto_seleccionado_idx]
    
    st.markdown("## 📊 Producto Seleccionado para Análisis")
    
    # Vista detallada del producto seleccionado
    det_col1, det_col2, det_col3 = st.columns([1, 2, 1])
    
    with det_col1:
        img = load_image_from_url(producto_seleccionado['product_image_url'])
        if img:
            # Redimensionar la imagen a un tamaño fijo antes de mostrarla
            img_resized = img.copy()
            img_resized.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Crear imagen con fondo blanco del mismo tamaño
            background = Image.new('RGB', (300, 300), (255, 255, 255))
            # Centrar la imagen
            offset = ((300 - img_resized.size[0]) // 2, (300 - img_resized.size[1]) // 2)
            background.paste(img_resized, offset)
            
            st.image(background, use_container_width=True)
        else:
            # Crear imagen placeholder del mismo tamaño
            placeholder = Image.new('RGB', (300, 300), (240, 240, 240))
            st.image(placeholder, use_container_width=True)
    
    with det_col2:
        titulo_producto = f"{producto_seleccionado['original_title']}"
        st.markdown(f"### {titulo_producto}")
        st.markdown(f"**📂 Categoría:** {producto_seleccionado['category']} | **🏷️ Subtipo:** {producto_seleccionado['subtype']}")
        st.markdown(f"**🏭 Tier:** {producto_seleccionado['market_tier']} | **📦 Estado:** {producto_seleccionado['condition']} | **🔧 Generación:** {producto_seleccionado['tech_generation']}")
        
        # Métricas principales
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
        st.markdown("### Características")
        if producto_seleccionado['is_best_seller'] == 'Yes':
            st.success("🏆 Best Seller")
        if producto_seleccionado['has_coupon'] == 1:
            st.info("🎟️ Cupón Activo")
        if producto_seleccionado['buy_box_availability'] == 1:
            st.success("📦 Buy Box")
        if producto_seleccionado['is_sponsored'] == 'Yes':
            st.warning("📢 Patrocinado")
        if producto_seleccionado['sustainability_tags'] == 1:
            st.success("🌱 Sostenible")
        if producto_seleccionado['is_premium_brand']:
            st.info("👑 Marca Premium")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- SELECTOR DE TABS ---
    tab_cols = st.columns(3)
    
    tab_options = {
        'resumen': {'icon': '📊', 'title': 'Resumen', 'subtitle': 'Vista general'},
        'mercado': {'icon': '🔍', 'title': 'Análisis de Mercado', 'subtitle': '¿Dónde estoy?'},
        'optimizar': {'icon': '💰', 'title': 'Optimizar Precio', 'subtitle': 'IA y predicción'}
    }
    
    for idx, (tab_key, tab_info) in enumerate(tab_options.items()):
        with tab_cols[idx]:
            is_active = st.session_state.selected_tab == tab_key
            
            if st.button(
                f"{tab_info['icon']}\n{tab_info['title']}\n{tab_info['subtitle']}",
                key=f"tab_{tab_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.selected_tab = tab_key
                st.rerun()
    
    st.markdown("---")
    
    # --- CONTENIDO DE CADA TAB ---
    
    # ==================== TAB 1: RESUMEN ====================
    if st.session_state.selected_tab == 'resumen':
        st.markdown("## 📊 Resumen del Producto")
        
        # KPIs comparativos - Filtrar por categoría Y subtipo
        df_categoria = df[df['category'] == producto_seleccionado['category']]
        df_subtipo = df[df['subtype'] == producto_seleccionado['subtype']]
        
        st.markdown("### 📍 Tu Producto vs Promedio de tu Subtipo")
        
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
        
        # Semáforo de salud
        st.markdown("### 🚦 Diagnóstico Rápido")
        
        # Calcular scores individuales (mismo método que Análisis de Mercado)
        # Score 1: Precio (mejor cerca de la mediana)
        percentil_precio = (df_subtipo['precio_real'] <= producto_seleccionado['precio_real']).sum() / len(df_subtipo) * 100
        score_precio = 100 - abs(percentil_precio - 50) * 2
        score_precio = max(0, min(100, score_precio))

        # Score 2: Calidad (rating ponderado con reviews)
        max_rating = df_subtipo['product_rating'].max()
        score_rating = (producto_seleccionado['product_rating'] / max_rating) * 100

        percentil_reviews = (df_subtipo['reviews_real'] <= producto_seleccionado['reviews_real']).sum() / len(df_subtipo) * 100
        bonus_reviews = (percentil_reviews / 100) * 10
        score_calidad = min(100, score_rating + bonus_reviews)

        # Score 3: Popularidad (ventas)
        percentil_ventas = (df_subtipo['ventas_mes_real'] <= producto_seleccionado['ventas_mes_real']).sum() / len(df_subtipo) * 100
        score_popularidad = percentil_ventas

        # Score general
        score_salud = (score_precio + score_calidad + score_popularidad) / 3
        
        # Determinar color
        if score_salud >= 70:
            color = "green"
            status = "EXCELENTE"
            emoji = "🟢"
        elif score_salud >= 40:
            color = "orange"
            status = "BUENA"
            emoji = "🟡"
        else:
            color = "red"
            status = "NECESITA ATENCIÓN"
            emoji = "🔴"
        
        st.markdown(f"""
            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;'>
                <h1 style='font-size: 4em; margin: 0;'>{emoji}</h1>
                <h2 style='color: {color}; margin: 10px 0;'>SALUD DEL PRODUCTO: {status}</h2>
                <p style='font-size: 2em; font-weight: bold; color: {color};'>{score_salud:.0f}/100</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detalles
        detail_cols = st.columns(3)
        
        with detail_cols[0]:
            st.markdown(f"**💰 Precio:** Score {score_precio:.0f}/100")
            if score_precio >= 70:
                st.success("✅ Precio bien posicionado")
            elif score_precio >= 40:
                st.info("ℹ️ Precio en rango aceptable")
            else:
                st.warning("⚠️ Precio muy alejado de la mediana de mercado")
        
        with detail_cols[1]:
            st.markdown(f"**⭐ Calidad:** Score {score_calidad:.0f}/100")
            if score_calidad >= 70:
                st.success("✅ Buena calidad percibida")
            elif score_calidad >= 40:
                st.info("ℹ️ Calidad aceptable")
            else:
                st.warning("⚠️ Necesitas mejorar rating y/o conseguir más reviews")
        
        with detail_cols[2]:
            st.markdown(f"**📊 Popularidad:** Score {score_popularidad:.0f}/100")
            if score_popularidad >= 70:
                st.success("✅ Vendes bien en tu subtipo")
            elif score_popularidad >= 40:
                st.info("ℹ️ Ventas en rango medio")
            else:
                st.warning("⚠️ Tus ventas están por debajo del promedio")
    
    # ==================== TAB 2: ANÁLISIS DE MERCADO ====================
    elif st.session_state.selected_tab == 'mercado':
        st.markdown("## 🔍 Análisis de Mercado")
        st.markdown("### Entiende tu posición competitiva y oportunidades de mejora")
        st.markdown("---")
        
        # Obtener datos de la categoría y subtipo
        df_categoria = df[df['category'] == producto_seleccionado['category']]
        df_subtipo = df[df['subtype'] == producto_seleccionado['subtype']]
        
        # ========== SECCIÓN 1: TU POSICIÓN EN EL MERCADO ==========
        st.markdown("### 1️⃣ Tu Posición en el Mercado")
        st.caption("¿Dónde estoy yo en mi subtipo?")
        
        # Crear zonas de posicionamiento (usando subtipo para más precisión)
        precio_medio_sub = df_subtipo['precio_real'].median()
        ventas_medio_sub = df_subtipo['ventas_mes_real'].median()
        
        # Gráfica scatter con zonas coloreadas
        fig1 = go.Figure()
        
        # Definir zonas con colores
        # Zona 1: Precio alto, Ventas altas (Verde - Ideal)
        fig1.add_shape(
            type="rect",
            x0=precio_medio_sub, y0=ventas_medio_sub,
            x1=df_subtipo['precio_real'].max() * 1.1, y1=df_subtipo['ventas_mes_real'].max() * 1.1,
            fillcolor="rgba(76, 175, 80, 0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # Zona 2: Precio bajo, Ventas altas (Azul - Oportunidad)
        fig1.add_shape(
            type="rect",
            x0=df_subtipo['precio_real'].min() * 0.9, y0=ventas_medio_sub,
            x1=precio_medio_sub, y1=df_subtipo['ventas_mes_real'].max() * 1.1,
            fillcolor="rgba(33, 150, 243, 0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # Zona 3: Precio bajo, Ventas bajas (Amarillo - Competitiva)
        fig1.add_shape(
            type="rect",
            x0=df_subtipo['precio_real'].min() * 0.9, y0=df_subtipo['ventas_mes_real'].min() * 0.9,
            x1=precio_medio_sub, y1=ventas_medio_sub,
            fillcolor="rgba(255, 193, 7, 0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # Zona 4: Precio alto, Ventas bajas (Rojo - Riesgo)
        fig1.add_shape(
            type="rect",
            x0=precio_medio_sub, y0=df_subtipo['ventas_mes_real'].min() * 0.9,
            x1=df_subtipo['precio_real'].max() * 1.1, y1=ventas_medio_sub,
            fillcolor="rgba(244, 67, 54, 0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # Añadir competidores
        fig1.add_trace(go.Scatter(
            x=df_subtipo['precio_real'],
            y=df_subtipo['ventas_mes_real'],
            mode='markers',
            marker=dict(
                size=8,
                color='lightgray',
                opacity=0.5
            ),
            text=df_subtipo['brand'] + ' ' + df_subtipo['subtype'],
            hovertemplate='<b>%{text}</b><br>Precio: %{x:.2f}€<br>Ventas: %{y:.0f}<extra></extra>',
            name='Competidores',
            showlegend=True
        ))
        
        # Añadir tu producto
        fig1.add_trace(go.Scatter(
            x=[producto_seleccionado['precio_real']],
            y=[producto_seleccionado['ventas_mes_real']],
            mode='markers+text',
            marker=dict(
                size=25,
                color='red',
                symbol='star',
                line=dict(width=2, color='white')
            ),
            text=['TÚ'],
            textposition='top center',
            textfont=dict(size=14, color='red', family='Arial Black'),
            hovertemplate=f'<b>TU PRODUCTO</b><br>Precio: {producto_seleccionado["precio_real"]:.2f}€<br>Ventas: {int(producto_seleccionado["ventas_mes_real"])}<extra></extra>',
            name='Tu Producto',
            showlegend=True
        ))
        
        # Líneas de promedio
        fig1.add_hline(
            y=ventas_medio_sub,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Ventas mediana: {int(ventas_medio_sub)}",
            annotation_position="right"
        )
        
        fig1.add_vline(
            x=precio_medio_sub,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Precio mediano: {precio_medio_sub:.2f}€",
            annotation_position="top"
        )
        
        # Añadir anotaciones de zonas
        fig1.add_annotation(
            x=df_subtipo['precio_real'].max() * 0.95,
            y=df_subtipo['ventas_mes_real'].max() * 0.95,
            text="🟢 ZONA IDEAL",
            showarrow=False,
            font=dict(size=12, color="green", family="Arial Black"),
            bgcolor="white",
            bordercolor="green",
            borderwidth=2
        )
        
        fig1.add_annotation(
            x=df_subtipo['precio_real'].min() * 1.05,
            y=df_subtipo['ventas_mes_real'].max() * 0.95,
            text="🔵 OPORTUNIDAD",
            showarrow=False,
            font=dict(size=12, color="blue", family="Arial Black"),
            bgcolor="white",
            bordercolor="blue",
            borderwidth=2
        )
        
        fig1.add_annotation(
            x=df_subtipo['precio_real'].max() * 0.95,
            y=df_subtipo['ventas_mes_real'].min() * 1.05,
            text="🔴 RIESGO",
            showarrow=False,
            font=dict(size=12, color="red", family="Arial Black"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=2
        )
        
        fig1.add_annotation(
            x=df_subtipo['precio_real'].min() * 1.05,
            y=df_subtipo['ventas_mes_real'].min() * 1.05,
            text="🟡 COMPETITIVA",
            showarrow=False,
            font=dict(size=12, color="orange", family="Arial Black"),
            bgcolor="white",
            bordercolor="orange",
            borderwidth=2
        )
        
        fig1.update_layout(
            title=f"Mapa de Posicionamiento: {producto_seleccionado['subtype']}",
            xaxis_title="Precio (€)",
            yaxis_title="Ventas último mes",
            height=600,
            plot_bgcolor='white',
            hovermode='closest',
            showlegend=True
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Insights automáticos
        if producto_seleccionado['precio_real'] >= precio_medio_sub and producto_seleccionado['ventas_mes_real'] >= ventas_medio_sub:
            zona = "🟢 ZONA IDEAL"
            mensaje = "¡Excelente! Estás en la zona premium con buen volumen de ventas."
            tipo = "success"
        elif producto_seleccionado['precio_real'] < precio_medio_sub and producto_seleccionado['ventas_mes_real'] >= ventas_medio_sub:
            zona = "🔵 ZONA OPORTUNIDAD"
            mensaje = "Vendes bien pero tu precio está por debajo. Considera subirlo gradualmente."
            tipo = "info"
        elif producto_seleccionado['precio_real'] >= precio_medio_sub and producto_seleccionado['ventas_mes_real'] < ventas_medio_sub:
            zona = "🔴 ZONA RIESGO"
            mensaje = "Precio alto pero ventas bajas. Considera bajar precio o mejorar el producto."
            tipo = "error"
        else:
            zona = "🟡 ZONA COMPETITIVA"
            mensaje = "Precio y ventas en rango medio. Hay oportunidad de diferenciarte."
            tipo = "warning"
        
        # Mostrar insights
        insight_cols = st.columns([2, 1])
        with insight_cols[0]:
            if tipo == "success":
                st.success(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")
            elif tipo == "info":
                st.info(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")
            elif tipo == "error":
                st.error(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")
            else:
                st.warning(f"📍 **Tu producto está en: {zona}**\n\n{mensaje}")
        
        with insight_cols[1]:
            # Calcular competidores cercanos (±10% precio)
            rango_min = producto_seleccionado['precio_real'] * 0.9
            rango_max = producto_seleccionado['precio_real'] * 1.1
            competidores_rango = df_subtipo[
                (df_subtipo['precio_real'] >= rango_min) & 
                (df_subtipo['precio_real'] <= rango_max)
            ]
            st.metric(
                "Competidores en tu rango",
                len(competidores_rango),
                help="Productos con precio ±10% del tuyo"
            )
            
            percentil_ventas = (df_subtipo['ventas_mes_real'] < producto_seleccionado['ventas_mes_real']).sum() / len(df_subtipo) * 100
            st.metric(
                "Percentil de ventas",
                f"{percentil_ventas:.0f}%",
                help="% de productos que venden menos que tú"
            )
        
        st.markdown("---")
        
        # ========== GRÁFICO ADICIONAL: DISTRIBUCIÓN POR CATEGORÍA ==========
        st.markdown("### 📊 Análisis por Categoría y Subtipo")
        
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
            
            # Marcar tu categoría
            tu_categoria_idx = categoria_stats[categoria_stats['category'] == producto_seleccionado['category']].index
            if len(tu_categoria_idx) > 0:
                fig_cat.add_annotation(
                    x=producto_seleccionado['category'],
                    y=categoria_stats.loc[tu_categoria_idx[0], 'precio_real'],
                    text="TU CATEGORÍA",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=12, family="Arial Black")
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
                color_continuous_scale='Blues',
                height=400
            )
            
            # Marcar tu subtipo
            tu_subtipo_idx = subtipo_stats[subtipo_stats['subtype'] == producto_seleccionado['subtype']].index
            if len(tu_subtipo_idx) > 0:
                fig_sub.add_annotation(
                    x=producto_seleccionado['subtype'],
                    y=subtipo_stats.loc[tu_subtipo_idx[0], 'precio_real'],
                    text="TU SUBTIPO",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=12, family="Arial Black")
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
                'Yes': 'Patrocinados',
                'No': 'No Patrocinados'
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
                showlegend=False
            )
            
            st.plotly_chart(fig_sponsored, use_container_width=True)
            
            # Métricas comparativas
            col_sp1, col_sp2, col_sp3 = st.columns(3)
            
            patrocinados = df_categoria[df_categoria['is_sponsored'] == 'Yes']
            no_patrocinados = df_categoria[df_categoria['is_sponsored'] == 'No']
            
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
                pct_patrocinados = (df_categoria['is_sponsored'] == 'Yes').sum() / len(df_categoria) * 100
                st.metric(
                    "% Patrocinados",
                    f"{pct_patrocinados:.1f}%",
                    help="Porcentaje de productos patrocinados en tu categoría"
                )
            
            # Insight
            if producto_seleccionado['is_sponsored'] == 'Yes':
                st.info("✅ Tu producto está patrocinado. Esto puede aumentar tu visibilidad.")
            else:
                if pct_patrocinados > 30:
                    st.warning(f"⚠️ {pct_patrocinados:.0f}% de tu categoría está patrocinada. Considera activar publicidad para competir.")
                else:
                    st.success("✅ Bajo nivel de publicidad en tu categoría. Puedes competir orgánicamente.")
        
        st.markdown("---")
        
        # ========== SECCIÓN 2: TUS COMPETIDORES MÁS CERCANOS ==========
        st.markdown("### 2️⃣ Tus Competidores Más Cercanos")
        st.caption("¿Contra quién compito en mi subtipo?")
        
        # Filtrar competidores: mismo subtipo + precio similar (±20%)
        rango_precio_min = producto_seleccionado['precio_real'] * 0.8
        rango_precio_max = producto_seleccionado['precio_real'] * 1.2
        
        competidores_directos = df_subtipo[
            (df_subtipo['precio_real'] >= rango_precio_min) & 
            (df_subtipo['precio_real'] <= rango_precio_max)
        ].copy()
        
        # Ordenar por ventas
        competidores_directos = competidores_directos.sort_values('ventas_mes_real', ascending=False).head(10)
        
        # Crear tabla interactiva
        tabla_competidores = []
        for idx, comp in competidores_directos.iterrows():
            diff_precio = comp['precio_real'] - producto_seleccionado['precio_real']
            
            tabla_competidores.append({
                'Marca': comp['brand'],
                'Market Tier': comp['market_tier'],
                'Estado': comp['condition'],
                'Precio': f"{comp['precio_real']:.2f} €",
                'Diff. Precio': f"{diff_precio:+.2f} €",
                'Rating': f"{comp['product_rating']:.1f} ⭐",
                'Ventas/mes': int(comp['ventas_mes_real']),
                'Reviews': int(comp['reviews_real']),
                'Best Seller': '🏆' if comp['is_best_seller'] == 'Yes' else '',
                'Cupón': '🎟️' if comp['has_coupon'] == 1 else '',
                'Buy Box': '📦' if comp['buy_box_availability'] == 1 else '',
                'Premium': '👑' if comp['is_premium_brand'] else '',
            })
        
        df_tabla = pd.DataFrame(tabla_competidores)
        
        # Mostrar tabla
        st.dataframe(
            df_tabla,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Insights de competidores
        st.markdown("#### 💡 Insights Clave")
        
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
                st.warning("⚠️ Considera activar un cupón")
        
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
                st.warning("⚠️ Tu rating está por debajo")
        
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
        st.markdown("### 3️⃣ ¿Qué Hace la Diferencia?")
        st.caption("¿Qué características importan en mi subtipo?")
        
        # Análisis de características
        caracteristicas_analisis = {
            'Best Seller': ('is_best_seller', 'Yes'),
            'Cupón': ('has_coupon', 1),
            'Buy Box': ('buy_box_availability', 1),
            'Sostenible': ('sustainability_tags', 1),
            'Marca Premium': ('is_premium_brand', True),
            'Patrocinado': ('is_sponsored', 'Yes')
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
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Tabla de impacto
            st.markdown("#### 📊 Impacto de Características")
            
            impacto_cols = st.columns(2)
            
            with impacto_cols[0]:
                st.markdown("##### 💰 Impacto en Precio")
                for dato in datos_comparacion:
                    diff_pct = (dato['Diff Precio'] / dato['Precio SIN']) * 100 if dato['Precio SIN'] > 0 else 0
                    tiene_marca = "✅" if dato['Tienes'] else "❌"
                    
                    if dato['Diff Precio'] > 0:
                        st.success(f"{tiene_marca} **{dato['Característica']}**: +{dato['Diff Precio']:.2f}€ ({diff_pct:+.1f}%)")
                    else:
                        st.info(f"{tiene_marca} **{dato['Característica']}**: {dato['Diff Precio']:.2f}€ ({diff_pct:+.1f}%)")
            
            with impacto_cols[1]:
                st.markdown("##### 📈 Impacto en Ventas")
                for dato in datos_comparacion:
                    diff_ventas_pct = (dato['Diff Ventas'] / dato['Ventas SIN']) * 100 if dato['Ventas SIN'] > 0 else 0
                    tiene_marca = "✅" if dato['Tienes'] else "❌"
                    
                    if dato['Diff Ventas'] > 0:
                        st.success(f"{tiene_marca} **{dato['Característica']}**: +{int(dato['Diff Ventas'])} ventas ({diff_ventas_pct:+.1f}%)")
                    else:
                        st.info(f"{tiene_marca} **{dato['Característica']}**: {int(dato['Diff Ventas'])} ventas ({diff_ventas_pct:+.1f}%)")
            
            # Recomendaciones
            st.markdown("#### 💡 Recomendaciones")
            caracteristicas_faltantes = [d for d in datos_comparacion if not d['Tienes'] and d['Diff Precio'] > 0]
            
            if caracteristicas_faltantes:
                # Ordenar por impacto en precio
                caracteristicas_faltantes.sort(key=lambda x: x['Diff Precio'], reverse=True)
                
                st.info(f"**Características que podrías activar:**")
                for dato in caracteristicas_faltantes[:3]:  # Top 3
                    st.markdown(f"- **{dato['Característica']}**: Podría aumentar tu precio en ~{dato['Diff Precio']:.2f}€ y ventas en ~{int(dato['Diff Ventas'])} unidades/mes")
            else:
                st.success("✅ ¡Tienes todas las características premium activadas!")
        
        st.markdown("---")
        
        # ========== SECCIÓN 4: DIAGNÓSTICO DE SALUD ==========
        st.markdown("### 4️⃣ Diagnóstico de Salud del Producto")
        st.caption("¿Cómo está mi producto?")
        
        # Calcular scores individuales
        # Score 1: Precio (percentil de precio competitivo)
        percentil_precio = (df_subtipo['precio_real'] <= producto_seleccionado['precio_real']).sum() / len(df_subtipo) * 100
        # Invertir: precio bajo = bueno
        score_precio = 100 - abs(percentil_precio - 50) * 2  # Mejor en el centro (percentil 50)
        score_precio = max(0, min(100, score_precio))
        
        # Score 2: Calidad (rating ponderado con reviews)
        max_rating = df_subtipo['product_rating'].max()
        score_rating = (producto_seleccionado['product_rating'] / max_rating) * 100 if max_rating > 0 else 0
        
        # Ajustar por número de reviews
        percentil_reviews = (df_subtipo['reviews_real'] <= producto_seleccionado['reviews_real']).sum() / len(df_subtipo) * 100
        bonus_reviews = (percentil_reviews / 100) * 10  # Bonus hasta 10 puntos
        score_calidad = min(100, score_rating + bonus_reviews)
        
        # Score 3: Popularidad (ventas)
        percentil_ventas_salud = (df_subtipo['ventas_mes_real'] <= producto_seleccionado['ventas_mes_real']).sum() / len(df_subtipo) * 100
        score_popularidad = percentil_ventas_salud
        
        # Crear gauges
        gauge_cols = st.columns(3)
        
        def crear_gauge(valor, titulo):
            """Crea un gauge chart"""
            if valor >= 70:
                color = "green"
                estado = "BUENA"
            elif valor >= 40:
                color = "orange"
                estado = "MEDIA"
            else:
                color = "red"
                estado = "BAJA"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=valor,
                title={'text': titulo, 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(244, 67, 54, 0.2)"},
                        {'range': [40, 70], 'color': "rgba(255, 193, 7, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(76, 175, 80, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': valor
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20)
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
        score_general = (score_precio + score_calidad + score_popularidad) / 3
        
        st.markdown("---")
        
        # Semáforo general
        if score_general >= 70:
            color_general = "green"
            estado_general = "EXCELENTE"
            emoji_general = "🟢"
        elif score_general >= 40:
            color_general = "orange"
            estado_general = "BUENA"
            emoji_general = "🟡"
        else:
            color_general = "red"
            estado_general = "NECESITA ATENCIÓN"
            emoji_general = "🔴"
        
        st.markdown(f"""
            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;'>
                <h2 style='font-size: 3em; margin: 0;'>{emoji_general}</h2>
                <h3 style='color: {color_general}; margin: 10px 0;'>SALUD GENERAL: {estado_general}</h3>
                <p style='font-size: 2.5em; font-weight: bold; color: {color_general};'>{score_general:.0f}/100</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Resumen de fortalezas y áreas de mejora
        fortaleza_mejora_cols = st.columns(2)
        
        with fortaleza_mejora_cols[0]:
            st.markdown("#### ✅ Fortalezas")
            
            if score_precio >= 70:
                st.success(f"💰 Precio competitivo (Score: {score_precio:.0f})")
            if score_calidad >= 70:
                st.success(f"⭐ Buena calidad percibida (Score: {score_calidad:.0f})")
            if score_popularidad >= 70:
                st.success(f"📈 Ventas sólidas (Score: {score_popularidad:.0f})")
            
            if score_precio < 70 and score_calidad < 70 and score_popularidad < 70:
                st.info("Hay oportunidades de mejora en todas las áreas")
        
        with fortaleza_mejora_cols[1]:
            st.markdown("#### ⚠️ Áreas de Mejora")
            
            if score_precio < 70:
                if percentil_precio > 75:
                    st.warning(f"💰 Precio alto vs competencia (Percentil: {percentil_precio:.0f})")
                else:
                    st.warning(f"💰 Revisa tu estrategia de precio (Score: {score_precio:.0f})")
            
            if score_calidad < 70:
                if producto_seleccionado['product_rating'] < df_subtipo['product_rating'].mean():
                    st.warning(f"⭐ Rating por debajo del promedio ({producto_seleccionado['product_rating']:.1f} vs {df_subtipo['product_rating'].mean():.1f})")
                if producto_seleccionado['reviews_real'] < df_subtipo['reviews_real'].mean():
                    st.warning(f"💬 Pocas reviews ({int(producto_seleccionado['reviews_real'])} vs promedio: {int(df_subtipo['reviews_real'].mean())})")
            
            if score_popularidad < 70:
                st.warning(f"📊 Ventas por debajo del promedio (Percentil: {percentil_ventas_salud:.0f})")
            
            if score_precio >= 70 and score_calidad >= 70 and score_popularidad >= 70:
                st.success("¡Todo en orden! Sigue así 🚀")
    
    # ==================== TAB 3: OPTIMIZAR PRECIO ====================
    elif st.session_state.selected_tab == 'optimizar':
        st.markdown("## 💰 Optimizar Precio con IA")

        col_button = st.columns([1, 2, 1])
        with col_button[1]:
            analizar = st.button(
                "🚀 ANALIZAR PRECIO ÓPTIMO CON IA",
                use_container_width=True,
                type="primary",
                key="analyze_button"
            )

        if analizar:
            st.markdown("---")
            st.markdown("## 🎯 Análisis de Precio Óptimo")

            # ----------------------------------------------------------
            # PAYLOAD — fila completa del producto
            # Excluimos solo las 2 columnas que no son features del modelo
            # más las columnas derivadas del dashboard
            # ----------------------------------------------------------
            COLS_EXCLUIR = {
                'original_title',    # texto libre, no es feature del modelo
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
                # Convertir tipos numpy a tipos Python nativos serializables
                if isinstance(val, (np.integer,)):
                    payload[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    payload[col] = None if np.isnan(val) else float(val)
                elif isinstance(val, (np.bool_,)):
                    payload[col] = bool(val)
                elif isinstance(val, float) and np.isnan(val):
                    payload[col] = None
                else:
                    payload[col] = val

            with st.spinner("🤖 Analizando mercado y competencia..."):
                try:
                    response = requests.post(API_URL, json=payload)

                    if response.status_code == 200:
                        data_api        = response.json()
                        precio_predicho = data_api['predicted_price']

                        # --- RESULTADOS PRINCIPALES ---
                        st.markdown("### 💰 Resultado del Análisis")

                        result_cols = st.columns([1, 1, 1, 1])

                        with result_cols[0]:
                            st.markdown("#### Precio Actual")
                            st.markdown(
                                f"<p style='font-size:2.5em;color:#666;text-align:center;'>"
                                f"{producto_seleccionado['precio_real']:.2f} €</p>",
                                unsafe_allow_html=True
                            )

                        with result_cols[1]:
                            st.markdown("####  ")
                            st.markdown(
                                "<p style='font-size:2.5em;text-align:center;'>→</p>",
                                unsafe_allow_html=True
                            )

                        with result_cols[2]:
                            st.markdown("#### Precio Sugerido IA")
                            st.markdown(
                                f"<p style='font-size:2.5em;color:#FF9900;font-weight:bold;"
                                f"text-align:center;'>{precio_predicho:.2f} €</p>",
                                unsafe_allow_html=True
                            )

                        with result_cols[3]:
                            diferencia        = precio_predicho - producto_seleccionado['precio_real']
                            porcentaje_cambio = (diferencia / producto_seleccionado['precio_real']) * 100
                            color_diff        = "green" if diferencia > 0 else "red" if diferencia < 0 else "gray"
                            signo             = "+" if diferencia > 0 else ""

                            st.markdown("#### Diferencia")
                            st.markdown(
                                f"<p style='font-size:2em;color:{color_diff};font-weight:bold;"
                                f"text-align:center;'>{signo}{diferencia:.2f} €</p>"
                                f"<p style='font-size:1.2em;color:{color_diff};"
                                f"text-align:center;'>({porcentaje_cambio:+.1f}%)</p>",
                                unsafe_allow_html=True
                            )

                        # --- RECOMENDACIÓN ---
                        st.markdown("<br>", unsafe_allow_html=True)
                        if diferencia > 0:
                            st.success(
                                f"📈 **Recomendación:** Hay margen para subir el precio. "
                                f"Podrías aumentarlo en {diferencia:.2f}€ según las condiciones del mercado."
                            )
                            ingresos_adicionales = diferencia * producto_seleccionado['ventas_mes_real']
                            st.info(
                                f"💰 **Ingresos adicionales proyectados:** "
                                f"+{ingresos_adicionales:.2f} €/mes (asumiendo ventas constantes)"
                            )
                        elif diferencia < 0:
                            st.error(
                                f"📉 **Recomendación:** Tu precio está por encima del mercado. "
                                f"Considera bajarlo en {abs(diferencia):.2f}€ para ser más competitivo."
                            )
                            st.warning("⚠️ Un precio más bajo podría incrementar tus ventas y visibilidad.")
                        else:
                            st.success("✅ **¡Perfecto!** Tu precio actual está alineado con el mercado.")

                        # --- ANÁLISIS DE COMPETENCIA ---
                        st.markdown("---")
                        st.markdown("## 📊 Análisis de Competencia")

                        tab1, tab2, tab3 = st.tabs([
                            "🎯 Productos Similares (Precio)",
                            "📈 Posicionamiento de Mercado",
                            "📋 Estadísticas de Categoría"
                        ])

                        with tab1:
                            df_context = df.copy()
                            df_context['diff_precio'] = (df_context['precio_real'] - precio_predicho).abs()
                            df_zoom = df_context.nsmallest(50, 'diff_precio')

                            fig = px.scatter(
                                df_zoom,
                                x="precio_real",
                                y="ventas_mes_real",
                                size="reviews_real",
                                color="category",
                                opacity=0.6,
                                hover_data={
                                    "brand":           True,
                                    "subtype":         True,
                                    "category":        True,
                                    "precio_real":     ":.2f €",
                                    "ventas_mes_real": ":.0f",
                                    "product_rating":  ":.1f"
                                },
                                title=f"50 productos con precio más cercano a {precio_predicho:.2f}€",
                                labels={
                                    "precio_real":     "Precio (€)",
                                    "ventas_mes_real": "Ventas último mes"
                                },
                                height=600
                            )
                            fig.add_scatter(
                                x=[precio_predicho],
                                y=[producto_seleccionado['ventas_mes_real']],
                                mode='markers+text',
                                marker=dict(size=35, color='red', symbol='star',
                                            line=dict(width=3, color='white')),
                                text=['TU PRODUCTO'],
                                textposition='top center',
                                textfont=dict(size=16, color='red', family='Arial Black'),
                                name='TU PRODUCTO',
                                showlegend=True
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(240,240,240,0.5)',
                                font=dict(size=12)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            with st.expander("📋 Ver detalles de productos similares"):
                                comparison_df = df_zoom[[
                                    'brand', 'subtype', 'category',
                                    'precio_real', 'ventas_mes_real',
                                    'product_rating', 'reviews_real'
                                ]].copy()
                                comparison_df.columns = [
                                    'Marca', 'Subtipo', 'Categoría',
                                    'Precio (€)', 'Ventas/mes', 'Rating', 'Reviews'
                                ]
                                comparison_df = comparison_df.sort_values('Precio (€)')
                                comparison_df['Precio (€)'] = comparison_df['Precio (€)'].apply(lambda x: f"{x:.2f}")
                                comparison_df['Ventas/mes'] = comparison_df['Ventas/mes'].apply(lambda x: f"{int(x)}")
                                comparison_df['Rating']     = comparison_df['Rating'].apply(lambda x: f"{x:.1f}")
                                comparison_df['Reviews']    = comparison_df['Reviews'].apply(lambda x: f"{int(x)}")
                                st.dataframe(comparison_df, use_container_width=True, height=400)

                        with tab2:
                            df_categoria = df[df['category'] == producto_seleccionado['category']]

                            fig2 = px.scatter(
                                df_categoria,
                                x="precio_real",
                                y="product_rating",
                                size="ventas_mes_real",
                                color="market_tier",
                                opacity=0.5,
                                hover_data={
                                    "brand":           True,
                                    "subtype":         True,
                                    "precio_real":     ":.2f €",
                                    "product_rating":  ":.2f",
                                    "ventas_mes_real": ":.0f"
                                },
                                title=f"Posicionamiento en categoría: {producto_seleccionado['category']}",
                                labels={
                                    "precio_real":    "Precio (€)",
                                    "product_rating": "Rating"
                                },
                                height=600
                            )
                            fig2.add_scatter(
                                x=[producto_seleccionado['precio_real']],
                                y=[producto_seleccionado['product_rating']],
                                mode='markers+text',
                                marker=dict(size=30, color='red', symbol='star',
                                            line=dict(width=3, color='white')),
                                text=['TU PRODUCTO'],
                                textposition='top center',
                                textfont=dict(size=16, color='red', family='Arial Black'),
                                name='TU PRODUCTO',
                                showlegend=True
                            )
                            precio_medio = df_categoria['precio_real'].mean()
                            rating_medio = df_categoria['product_rating'].mean()
                            fig2.add_hline(
                                y=rating_medio, line_dash="dash", line_color="gray",
                                annotation_text=f"Rating medio: {rating_medio:.2f}",
                                annotation_position="right"
                            )
                            fig2.add_vline(
                                x=precio_medio, line_dash="dash", line_color="gray",
                                annotation_text=f"Precio medio: {precio_medio:.2f}€",
                                annotation_position="top"
                            )
                            fig2.update_layout(
                                plot_bgcolor='rgba(240,240,240,0.5)',
                                font=dict(size=12)
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                            st.markdown("#### 📍 Tu Posicionamiento")
                            col_ins1, col_ins2, col_ins3, col_ins4 = st.columns(4)

                            with col_ins1:
                                pct_precio = (df_categoria['precio_real'] < producto_seleccionado['precio_real']).sum() / len(df_categoria) * 100
                                st.metric("Percentil de Precio", f"{pct_precio:.0f}%",
                                          help="% de productos en tu categoría con precio menor al tuyo")
                            with col_ins2:
                                pct_rating = (df_categoria['product_rating'] < producto_seleccionado['product_rating']).sum() / len(df_categoria) * 100
                                st.metric("Percentil de Rating", f"{pct_rating:.0f}%",
                                          help="% de productos en tu categoría con rating menor al tuyo")
                            with col_ins3:
                                pct_ventas = (df_categoria['ventas_mes_real'] < producto_seleccionado['ventas_mes_real']).sum() / len(df_categoria) * 100
                                st.metric("Percentil de Ventas", f"{pct_ventas:.0f}%",
                                          help="% de productos en tu categoría con menos ventas que tú")
                            with col_ins4:
                                st.metric("Competidores en Categoría", len(df_categoria),
                                          help="Total de productos en tu misma categoría")

                        with tab3:
                            df_categoria = df[df['category'] == producto_seleccionado['category']]
                            st.markdown("#### 📊 Estadísticas de tu Categoría")

                            stats_cols = st.columns(3)
                            with stats_cols[0]:
                                st.markdown("##### 💰 Precios")
                                st.metric("Precio Medio",   f"{df_categoria['precio_real'].mean():.2f} €")
                                st.metric("Precio Mediano", f"{df_categoria['precio_real'].median():.2f} €")
                                st.metric("Rango",
                                          f"{df_categoria['precio_real'].min():.2f} – "
                                          f"{df_categoria['precio_real'].max():.2f} €")
                            with stats_cols[1]:
                                st.markdown("##### ⭐ Ratings")
                                st.metric("Rating Medio",   f"{df_categoria['product_rating'].mean():.2f}/5")
                                st.metric("% Best Sellers",
                                          f"{(df_categoria['is_best_seller'] == 'Best Seller').sum() / len(df_categoria) * 100:.1f}%")
                            with stats_cols[2]:
                                st.markdown("##### 📈 Ventas")
                                st.metric("Ventas Medias/mes", f"{int(df_categoria['ventas_mes_real'].mean())}")
                                st.metric("Reviews Medias",    f"{int(df_categoria['reviews_real'].mean())}")

                            st.markdown("---")
                            st.markdown("##### 📊 Distribución de Precios en la Categoría")
                            fig_hist = px.histogram(
                                df_categoria,
                                x="precio_real",
                                nbins=30,
                                title=f"Distribución de precios en {producto_seleccionado['category']}",
                                labels={"precio_real": "Precio (€)", "count": "Nº productos"},
                                color_discrete_sequence=['#FF9900']
                            )
                            fig_hist.add_vline(
                                x=producto_seleccionado['precio_real'],
                                line_dash="dash", line_color="red",
                                annotation_text="Tu Precio", annotation_position="top"
                            )
                            fig_hist.add_vline(
                                x=precio_predicho,
                                line_dash="dash", line_color="green",
                                annotation_text="Precio Sugerido", annotation_position="top"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                    else:
                        st.error(f"❌ Error en la API: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("⛔ Error de conexión. ¿Está encendida la API en el puerto 8001?")
                except Exception as e:
                    st.error(f"Error inesperado: {e}")
                    st.exception(e)

else:
    # Mensaje cuando no hay producto seleccionado
    st.info("👆 Selecciona un producto del catálogo superior para comenzar el análisis")

# --- FOOTER ---
st.markdown("---")
st.caption("🤖 Powered by Machine Learning | Amazon Price Optimizer v3.0")