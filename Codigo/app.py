import streamlit as st
import pandas as pd
import plotly.express as px
import random
import math

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Amazon Smart Pricing - Prototipo", layout="wide")

st.title("🚀 Amazon Smart Pricing: Simulador de Predicción")
st.markdown("""
Esta es una versión de prueba del dashboard. Los datos y las predicciones son simulados 
para validar la estructura de la herramienta.
""")

# 2. CREACIÓN DE DATOS FICTICIOS (HARDCODED)
# Simulamos una base de datos de productos que ya existen en Amazon
data = {
    'product_name': [f"Producto {i}" for i in range(1, 101)],
    'category': random.choices(['Electrónica', 'Hogar', 'Ropa', 'Libros'], k=100),
    'actual_price': [random.uniform(10, 500) for _ in range(100)],
    'purchased_last_month': [random.randint(0, 1000) for _ in range(100)],
    'rating': [random.uniform(3.0, 5.0) for _ in range(100)]
}
df_mock = pd.DataFrame(data)

# 3. INTERFAZ DE ENTRADA (USER INPUT)
st.sidebar.header("📥 Configuración del Producto")
with st.sidebar:
    nombre = st.text_input("Nombre del producto", "Mi nuevo Gadget")
    # Categoría del producto (mantenemos la lista original como default)
    categoria = st.selectbox("Categoría", ['Electrónica', 'Hogar', 'Ropa', 'Libros'])

    # Características solicitadas por el usuario
    st.subheader("Características del producto")
    product_rating = st.slider("product_rating", 0.0, 5.0, 4.0, 0.1)
    is_best_seller = st.selectbox("is_best_seller", ['No', 'Sí'])
    is_sponsored = st.checkbox("is_sponsored", value=False)
    buy_box_availability = st.checkbox("buy_box_availability (buy box disponible)", value=True)
    sustainability_tags = st.checkbox("sustainability_tags (etiqueta sostenible)", value=False)
    has_coupon = st.checkbox("has_coupon", value=False)
    discount_percentage = st.number_input("discount_percentage (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    product_category = st.text_input("product_category (texto)", categoria)
    product_segment = st.selectbox("product_segment", ['basic', 'standard', 'premium'])

    st.markdown("---")
    log_original_price = st.number_input("log_original_price (ln(precio original))", value=math.log(50.0))
    log_purchased_last_month = st.number_input("log_purchased_last_month (ln(ventas mes pasado +1))", value=math.log(10.0))
    log_total_reviews = st.number_input("log_total_reviews (ln(total_reviews +1))", value=math.log(50.0))

    es_prime = st.checkbox("¿Incluye envío Prime?", value=True)
    boton_predecir = st.button("Calcular Precio Ideal")

# 4. LÓGICA DE PREDICCIÓN "FALSEADA"
# Generamos un precio aleatorio simplemente para mostrar la funcionalidad
if 'precio_falso' not in st.session_state:
    st.session_state.precio_falso = 0.0

if boton_predecir:
    # Construimos una predicción sencilla basada en las características ingresadas
    try:
        orig_price = math.exp(log_original_price)
    except Exception:
        orig_price = 50.0

    score = 0.0
    score += product_rating * 15.0
    score += 40.0 if is_best_seller == 'Sí' else 0.0
    score += -10.0 if is_sponsored else 0.0
    score += 30.0 if buy_box_availability else 0.0
    score += 5.0 if sustainability_tags else 0.0
    score += -10.0 if has_coupon else 0.0
    score += -0.4 * discount_percentage

    if product_segment == 'basic':
        score += -10.0
    elif product_segment == 'premium':
        score += 20.0

    # aporte por reseñas y ventas (convertimos logs)
    try:
        total_reviews = math.exp(log_total_reviews)
    except Exception:
        total_reviews = 50.0
    try:
        purchased_last_month = math.exp(log_purchased_last_month)
    except Exception:
        purchased_last_month = 10.0

    score += 0.02 * total_reviews
    score += 0.03 * purchased_last_month

    # Mezclamos con el precio original para obtener una predicción aproximada
    predicted = max(0.5, orig_price * 0.6 + score + random.uniform(-5, 15))
    st.session_state.precio_falso = predicted
    st.success("¡Predicción calculada con éxito!")

# 5. VISUALIZACIÓN DE RESULTADOS
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(label="Precio Predicho para tu producto", value=f"{st.session_state.precio_falso:.2f} €")
    st.info(f"Análisis para: {nombre} en la categoría {categoria}")
    
    # Resumen de características ingresadas
    st.subheader("Características ingresadas")
    features = {
        'product_rating': product_rating,
        'is_best_seller': is_best_seller,
        'is_sponsored': is_sponsored,
        'buy_box_availability': buy_box_availability,
        'sustainability_tags': sustainability_tags,
        'has_coupon': has_coupon,
        'discount_percentage': discount_percentage,
        'product_category': product_category,
        'product_segment': product_segment,
        'log_original_price': log_original_price,
        'log_purchased_last_month': log_purchased_last_month,
        'log_total_reviews': log_total_reviews,
        'es_prime': es_prime
    }
    st.write(features)

with col2:
    st.subheader(f"Ventas del último mes en {categoria}")
    
    # Filtramos los datos ficticios para mostrar productos similares al precio predicho
    # Definimos un rango de +- 20% sobre el precio predicho
    precio_min = st.session_state.precio_falso * 0.8
    precio_max = st.session_state.precio_falso * 1.2
    
    df_filtrado = df_mock[
        (df_mock['category'] == categoria) & 
        (df_mock['actual_price'] >= precio_min) & 
        (df_mock['actual_price'] <= precio_max)
    ].sort_values(by="purchased_last_month", ascending=False)

    if not df_filtrado.empty:
        fig = px.bar(
            df_filtrado, 
            x='product_name', 
            y='purchased_last_month',
            color='actual_price',
            labels={'purchased_last_month': 'Ventas Mes Pasado', 'product_name': 'Producto'},
            title=f"Productos entre {precio_min:.2f}€ y {precio_max:.2f}€"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay productos similares en este rango de precio para mostrar ventas.")

# 6. TABLA DE DETALLE
if st.checkbox("Mostrar tabla de productos de la competencia"):
    st.write(df_filtrado)