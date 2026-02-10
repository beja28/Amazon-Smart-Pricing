from openai import OpenAI
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Configura tu cliente (asegúrate de no subir tu API Key a GitHub)
# Puedes usar st.secrets["OPENAI_API_KEY"] si usas Streamlit Cloud
client = OpenAI(api_key="pedirmela (diego)")

import numpy as np

# Título complejo de ejemplo (Dell que pasaste antes)
titulo_test = 'HP 14" Chromebook | Intel N100 Processor | 8GB RAM | 128GB Flash Storage | Intel UHD Graphics | FHD Display | Up to 12.25 Hour Battery | Chrome OS | Dual Speakers |'

# Estos son los datos del "entorno" del producto en Amazon
contexto_venta = {
    'product_rating': 4.5,
    'is_best_seller': "No Badge",   # Asegúrate que coincida con cómo entrenaste (ej: 'Yes' o "Amazon'sChoice")
    'is_sponsored': 'Sponsored',
    'buy_box_availability': 1,
    'sustainability_tags': 0,
    'has_coupon': 0,
    'discount_percentage': 0.0,
    'category': 'Laptops & Chromebooks', # Ojo: Usa el nombre exacto de la categoría en tu dataset
    'log_purchased_last_month': np.log1p(0),
    'log_total_reviews': np.log1p(609)
}

def extraer_features_con_gpt(titulo_producto):
    """
    Envía el título a GPT para que extraiga las specs técnicas
    y las devuelva en formato JSON limpio.
    """
    
    prompt_sistema = """
    Eres un experto en hardware de ordenadores. Tu tarea es extraer especificaciones técnicas 
    de un título de producto de Amazon y devolverlas en formato JSON estricto.
    
    Debes inferir los siguientes campos basándote en el texto:
    - brand: La marca principal (ej: Dell, HP, Apple).
    - ram_gb: Cantidad de memoria RAM en GB (número flotante).
    - storage_gb: Cantidad de almacenamiento principal en GB (número flotante. 1TB = 1000).
    - size_value: Tamaño de pantalla en pulgadas (número flotante).
    - resolution_standard: "HD/HD+", "FHD (1080p)", "2K/QHD (1440p)", "4K/UHD (2160p)", "5K/8K+". Si ves 'Retina', 'Liquid Retina' o 'XDR' -> '2K/QHD (1440p)' o '4K/UHD (2160p)'.
    - cpu_gpu_tier: "Ej: i7, Ryzen 5, RTX 4060, M3". IMPORTANTE: "Integrated Graphics", "Intel UHD", "Radeon" NO son procesadores. Ignóralos.
    - market_tier: 'Mainstream', 'Premium' (Apple/Gaming gama alta), o 'Budget' (marcas chinas baratas/Chromebooks).
    - is_premium_brand: 'True' si es Apple, Razer, Alienware, Surface. 'False' para el resto.
    - tech_generation: 'Current-Gen' (DDR5, Ryzen 7000/8000, Intel 13/14 Gen), 'Cutting-Edge' (DDR4, Intel 11/12), 'Last-Gen' (Anterior).
    - condition: 'New' o 'Renewed/Refurbished' (si dice Refurbished/Restaurado).
    
    Si algún dato no aparece explícitamente no te lo inventes.
    
    Responde SOLO con el JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # O gpt-3.5-turbo (más barato)
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Título del producto: {titulo_producto}"}
            ],
            temperature=0, # Temperatura 0 para máxima precisión y determinismo
            response_format={"type": "json_object"} # Fuerza salida JSON
        )
        
        # Parsear la respuesta
        datos_json = json.loads(response.choices[0].message.content)
        return datos_json

    except Exception as e:
        print(f"Error en la API de OpenAI: {e}")
        # Retornar valores por defecto en caso de fallo de API para que no rompa el dashboard
        return {
            'brand': 'Generic', 'ram_gb': 8.0, 'storage_gb': 256.0, 
            'size_value': 15.6, 'resolution_standard': 'FHD',
            'cpu_gpu_tier': 'Mid', 'market_tier': 'Mainstream', 
            'is_premium_brand': 'No', 'tech_generation': 'Previous', 
            'condition': 'New'
        }

# ==========================================
# 2. INTEGRACIÓN CON TU MODELO CATBOOST
# ==========================================

def predecir_precio_inteligente(titulo_input, datos_contexto):
    """
    Combina specs extraídas por IA + contexto de mercado manual para predecir.
    """
    import pandas as pd
    import json
    from catboost import CatBoostRegressor

    print(f"🧠 Analizando Título: {titulo_input[:40]}...")
    
    # ---------------------------------------------------------
    # PASO 1: Extraer Specs Técnicas con la IA (RAM, SSD, Brand...)
    # ---------------------------------------------------------
    # (Llama a la función que creamos antes con OpenAI)
    features_ia = extraer_features_con_gpt(titulo_input)
    print("✅ Specs extraídas (IA):", features_ia)
    
    # ---------------------------------------------------------
    # PASO 2: Fusión de Datos (Contexto + Specs)
    # ---------------------------------------------------------
    # Unimos los dos diccionarios. 
    # Si hay claves repetidas, el segundo (**features_ia) sobrescribe al primero.
    datos_completos = {**datos_contexto, **features_ia}
    
    # Convertimos a DataFrame (una sola fila)
    df_pred = pd.DataFrame([datos_completos])

    # ---------------------------------------------------------
    # PASO 3: Alinear con el Modelo (Orden y Columnas)
    # ---------------------------------------------------------
    # Cargar metadata para saber qué columnas espera el modelo y en qué orden
    try:
        with open('modelos/laptops/catboost_laptops_metadata.json', 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        print("❌ Error: No se encuentra el archivo metadata.json")
        return 0, {}

    numeric_keywords = [
        'log_', 'value', 'gb', 'count', 'rating', 
        'availability', 'tags', 'coupon', 'discount', 
        'price', 'w', 'hz', 'inch'
    ]

    # --- BLOQUE DE SANITIZACIÓN ROBUSTA ---
    for col in meta['features']:
        # A. Si la columna no existe, la creamos vacía (NaN)
        if col not in df_pred.columns:
            df_pred[col] = np.nan
            
        # B. Detectar si la columna debería ser numérica
        is_numeric = any(key in col for key in numeric_keywords)
        
        # C. Limpieza específica por tipo
        if is_numeric:
            # 1. Rellenar Nulos con 0
            # 2. Convertir a float (Crucial para CatBoost)
            df_pred[col] = df_pred[col].fillna(0).astype(float)
        else:
            # 1. Rellenar Nulos con 'None' (string) o 'Unknown'
            # 2. Convertir a string (Crucial para que no haya objetos raros)
            df_pred[col] = df_pred[col].fillna('None').astype(str)
            
            # Limpieza extra: a veces entra la cadena "nan"
            df_pred[col] = df_pred[col].replace('nan', 'None')
            
    # FILTRO FINAL: Ordenar columnas exactamente como en el entrenamiento
    # (Esto descarta columnas extra que no use el modelo)
    df_pred = df_pred[meta['features']]
    
    # ---------------------------------------------------------
    # PASO 4: Predicción
    # ---------------------------------------------------------
    model = CatBoostRegressor()
    try:
        model.load_model('modelos/laptops/catboost_laptops_expert.cbm')
    except:
        print("❌ Error: No se encuentra el modelo .cbm")
        return 0, {}
    
    pred_log = model.predict(df_pred)[0]
    precio_final = np.expm1(pred_log) # Deshacer logaritmo
    
    return precio_final, features_ia

# ==========================================
# 3. PRUEBA DE FUEGO (EJECÚTAME)
# ==========================================

# Ejecutar
precio, datos_ia = predecir_precio_inteligente(titulo_test, contexto_venta)

print("\n" + "="*40)
print(f"💻 DATOS INTERPRETADOS: {json.dumps(datos_ia, indent=2)}")
print(f"💰 TASACIÓN FINAL: {precio:.2f} €")
print("="*40)