from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Dict
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import pickle
import os
import logging
import json
import time
from corrector_precio import CorrectorPorSimilitud 

# Configuración de Monitorización (Logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuración de Seguridad
API_KEY = "tfm-mioti-2026-key"
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Acceso denegado. API Key inválida.")

MODELS_DIR = "../Evaluaciones/Evaluacion_4/Entrenamiento_modelos/Modelos_produccion/"

GENERAL_MODEL_FILENAME = "modelo_general_lightgbm_produccion.pkl" 

# Diccionario de pesos: { "Nombre_Categoria": (Peso_General, Peso_Especifico) }
# Para el blending
CATEGORY_WEIGHTS = {
    "Office, Printing & Power": (1, 0),
    "Audio & Media Systems": (1, 0),
    "Accessories": (0.69, 0.31),
    "Peripherals & Input": (0.66, 0.34),
    "PC Components (Core)": (1, 0),
    "Computers & Gaming": (0.3, 0.7),
    "Cameras, Photography & Video": (0.55, 0.45),
    "Displays & Mounting": (1, 0),
    "Networking & Smart Home": (0.58, 0.42),
    "Mobile Devices": (0.46, 0.54)
}

# DICCIONARIO PARA ALMACENAR LOS MODELOS EN MEMORIA
models: Dict[str, Any] = {}
corrector_api = None 


@asynccontextmanager
async def lifespan(app: FastAPI):
    global corrector_api
    logger.info("Iniciando API y cargando modelos en memoria...")
    
    # Cargar el archivo de anclas estadísticas 
    # Usado para calibrar la prediccion devuelta por el modelo usando validacion knn para sacar sus competidores
    ruta_stats = "estadisticas_precio.json"
    if os.path.exists(ruta_stats):
        with open(ruta_stats, 'r') as f:
            diccionario_estadisticas = json.load(f)
        corrector_api = CorrectorPorSimilitud('motor_similitud_titulos.pkl')
        logger.info(f"Cargado con {len(diccionario_estadisticas)} anclas.")
    else:
        logger.warning(f"ADVERTENCIA: No se encontró '{ruta_stats}'. El post-procesamiento fallará.")
    
    # Cargar el Modelo General
    general_model_path = os.path.join(MODELS_DIR, GENERAL_MODEL_FILENAME)
    if not os.path.exists(general_model_path):
        raise FileNotFoundError(f"Modelo general no encontrado en: {general_model_path}")
    
    with open(general_model_path, 'rb') as f:
        models["general"] = pickle.load(f)
    logger.info(f"Modelo General (LightGBM) cargado exitosamente.")

    # Cargar los Modelos por Categoría 
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith(".cbm"):
                # los archivos se llaman "model_{Categoria}_ALIGNED.cbm"
                cat_name = file.replace(".cbm", "")
                
                cat_model = CatBoostRegressor()
                cat_model.load_model(os.path.join(MODELS_DIR, file))
                models[cat_name] = cat_model
                logger.info(f"Modelo Categórico (CatBoost) cargado: {cat_name}")
    else:
        logger.error(f"Directorio de modelos no encontrado: {MODELS_DIR}")

    yield 
    
    logger.info("Apagando API y liberando memoria...")
    models.clear()

app = FastAPI(title="Amazon Smart Pricing API - BLENDING", version="7.0", lifespan=lifespan)


# COLUMNAS NUMÉRICAS — las que NO son categóricas en el entrenamiento
NUMERIC_COLS = {
    "confidence", "ppm", "pack_count", "ups_capacity_value", "rack_u",
    "power_capacity_value", "total_power_w", "dpi", "read_speed_mbs",
    "node_count", "multi_gig_ports", "screen_size_in", "refresh_rate_hz",
    "num_screens", "max_weight_kg", "megapixels", "max_aperture_f",
    "vram_gb", "ram_gb", "storage_gb", "power_wattage", "nas_bays",
    "case_size_mm", "cable_length_m", "port_count", "pd_wattage",
    "product_rating", "discount_percentage", "log_purchased_last_month", 
    "log_total_reviews",
    "is_premium_brand",     
    "case_keyboard",        
    "buy_box_availability", 
    "sustainability_tags",  
    "has_coupon"            
}


# CONTRATO DE DATOS — data_dashboard.csv sin original_title,
# product_image_url, log_original_price y columnas derivadas del dashboard.
# Specs técnicos como Optional[Any] porque mezclan bool/str/float/None.
class ProductInput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # --- Columnas base (tipos fijos) ---
    category:         str
    subtype:          str
    market_tier:      str
    condition:        str
    is_premium_brand: bool
    tech_generation:  str
    brand:            str
    confidence:       Optional[float] = None

    # --- Specs técnicos: Optional[Any] ---
    printer_tech:         Optional[Any] = None
    is_color:             Optional[Any] = None
    is_multifunction:     Optional[Any] = None
    paper_size_max:       Optional[Any] = None
    ppm:                  Optional[Any] = None
    pack_count:           Optional[Any] = None
    is_xl:                Optional[Any] = None
    ups_capacity_value:   Optional[Any] = None
    ups_capacity_unit:    Optional[Any] = None
    is_rackmount:         Optional[Any] = None
    rack_u:               Optional[Any] = None
    power_capacity_value: Optional[Any] = None
    power_capacity_unit:  Optional[Any] = None
    audio_channels:       Optional[Any] = None
    total_power_w:        Optional[Any] = None
    has_dolby_atmos:      Optional[Any] = None
    has_subwoofer:        Optional[Any] = None
    has_anc:              Optional[Any] = None
    form_factor:          Optional[Any] = None
    mic_type:             Optional[Any] = None
    audio_resolution:     Optional[Any] = None
    video_output_res:     Optional[Any] = None
    switch_type:          Optional[Any] = None
    dpi:                  Optional[Any] = None
    has_screen:           Optional[Any] = None
    is_thunderbolt:       Optional[Any] = None
    max_resolution:       Optional[Any] = None
    read_speed_mbs:       Optional[Any] = None
    scan_engine:          Optional[Any] = None
    wifi_standard:        Optional[Any] = None
    speed_class:          Optional[Any] = None
    node_count:           Optional[Any] = None
    frequency_bands:      Optional[Any] = None
    has_poe:              Optional[Any] = None
    multi_gig_ports:      Optional[Any] = None
    power_source:         Optional[Any] = None
    smart_protocol:       Optional[Any] = None
    is_outdoor:           Optional[Any] = None
    screen_size_in:       Optional[Any] = None
    resolution_standard:  Optional[Any] = None
    panel_tech:           Optional[Any] = None
    refresh_rate_hz:      Optional[Any] = None
    is_smart:             Optional[Any] = None
    mount_type:           Optional[Any] = None
    adjustment_type:      Optional[Any] = None
    num_screens:          Optional[Any] = None
    max_weight_kg:        Optional[Any] = None
    is_heavy_duty:        Optional[Any] = None
    sensor_format:        Optional[Any] = None
    max_video_res:        Optional[Any] = None
    is_body_only:         Optional[Any] = None
    mount_system:         Optional[Any] = None
    megapixels:           Optional[Any] = None
    is_pro_line:          Optional[Any] = None
    focal_range:          Optional[Any] = None
    max_aperture_f:       Optional[Any] = None
    is_kit:               Optional[Any] = None
    component_series:     Optional[Any] = None
    vram_gb:              Optional[Any] = None
    has_x3d:              Optional[Any] = None
    ram_gb:               Optional[Any] = None
    storage_gb:           Optional[Any] = None
    tech_gen:             Optional[Any] = None
    power_wattage:        Optional[Any] = None
    efficiency_rating:    Optional[Any] = None
    chipset:              Optional[Any] = None
    is_aio:               Optional[Any] = None
    is_gaming:            Optional[Any] = None
    platform_family:      Optional[Any] = None
    is_digital_edition:   Optional[Any] = None
    nas_bays:             Optional[Any] = None
    model_name:           Optional[Any] = None
    cellular_type:        Optional[Any] = None
    has_stylus:           Optional[Any] = None
    case_size_mm:         Optional[Any] = None
    is_rugged:            Optional[Any] = None
    is_specialized:       Optional[Any] = None
    gps_activity:         Optional[Any] = None
    has_touchscreen:      Optional[Any] = None
    cable_length_m:       Optional[Any] = None
    interface_type:       Optional[Any] = None
    is_braided:           Optional[Any] = None
    material:             Optional[Any] = None
    brand_compatibility:  Optional[Any] = None
    is_active:            Optional[Any] = None
    is_original:          Optional[Any] = None
    connector_gender:     Optional[Any] = None
    is_wireless:          Optional[Any] = None
    port_count:           Optional[Any] = None
    storage_type:         Optional[Any] = None
    pd_wattage:           Optional[Any] = None
    case_keyboard:        Optional[Any] = None

    original_title:       Optional[str] = ""
    
    # --- Features de comportamiento (tipos fijos) ---
    product_rating:           float = Field(..., ge=0, le=5)
    is_best_seller:           str
    is_sponsored:             str
    buy_box_availability:     int
    sustainability_tags:      int
    has_coupon:               int
    discount_percentage:      float = Field(default=0, ge=0, le=100)
    log_purchased_last_month: float
    log_total_reviews:        float

    @field_validator('category')
    @classmethod
    def check_category_exists(cls, v):
        if v not in CATEGORY_WEIGHTS and v != "general":
            logger.warning(f"Categoría '{v}' no tiene pesos asignados. Se usará predicción general.")
        return v
    
    
# FUNCIONES DE PREPARACIÓN DE DATOS

def prepare_for_lightgbm(row, model_instance):
    df = pd.DataFrame([row])
    
    # Recuperar el orden exacto de features del modelo
    features = model_instance.feature_name_
    
    # Asegurar que todas las columnas existen (si no, poner NaN)
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
            
    # Reordenar columnas para que coincidan con el entrenamiento
    df = df[features]
    
    # 5. CASTING DE TIPOS
    for col in df.columns:
        if col in NUMERIC_COLS:
            # Forzar a float para que LightGBM lo vea como numérico
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        else:
            # Forzar a categoría de Pandas
            # Primero a string para evitar errores con nulos, luego a category
            df[col] = df[col].astype(str).replace('nan', np.nan).astype('category')
            
    return df

def prepare_for_catboost(row_dict: dict, model_instance: CatBoostRegressor) -> pd.DataFrame:
    """Tubería de datos para los modelos específicos (CatBoost)"""
    df = pd.DataFrame([row_dict])
    
    # Extraer nombres de las variables del CatBoost
    features = model_instance.feature_names_
    
    for col in features:
        if col not in df.columns:
            df[col] = None
            
    df = df[features] # Orden exacto
    
    # Nulos categóricos a "Missing" y tipo str
    for col in df.columns:
        if col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].apply(
                lambda x: "Missing" if (x is None or (isinstance(x, float) and np.isnan(x)) or x == "") else str(x)
            )
            
    return df


# ENDPOINTS
@app.get("/")
def root():
    return {"status": "online", "models_loaded": list(models.keys()), "version": "7.0"}

@app.get("/health")
def health():
    """
    Chequeo del estado de la API y de los modelos cargados en memoria.
    """
    if "general" not in models:
        logger.error("Health check fallido: Modelo general no cargado.")
        raise HTTPException(status_code=503, detail="Modelo general no cargado. La API no puede operar.")
    
    # Extraemos qué categorías tienen su modelo específico cargado
    categorias_disponibles = [k for k in models.keys() if k != "general"]
    
    return {
        "status": "healthy",
        "general_model": "Loaded (LightGBM)",
        "specific_models_loaded": len(categorias_disponibles),
        "available_categories": categorias_disponibles
    }


@app.get("/model/features", dependencies=[Depends(get_api_key)])
def get_features(category: Optional[str] = None):
    """
    Devuelve las features que espera un modelo específico.
    Si pasas 'category', te da las features que sobrevivieron al THRESHOLD en entrenamiento.
    Si no pasas nada, te da las del modelo general.
    """
    # Determinamos a qué modelo le vamos a preguntar
    target_model_name = category if category and category in models else "general"
    
    if target_model_name not in models:
        raise HTTPException(
            status_code=404, 
            detail=f"No hay un modelo específico cargado para la categoría: '{category}'"
        )

    model_instance = models[target_model_name]
    
    # 1. Caso Modelo General
    if target_model_name == "general":
        features = model_instance.feature_name_  
        motor = "LightGBM"
        info_extra = "Features completas del modelo global."
        
    # 2. Caso Modelo por Categoría
    else:
        features = model_instance.feature_names_
        motor = "CatBoost"
        info_extra = "Features pre-filtradas durante el entrenamiento (Threshold aplicado)."

    return {
        "model_requested": target_model_name,
        "engine": motor,
        "n_features": len(features),
        "note": info_extra,
        "feature_names": features
    }


@app.post("/predict", dependencies=[Depends(get_api_key)])
def predict_price(product: ProductInput):
    start_time = time.time()
    logger.info(f"Petición recibida: Categoría '{product.category}', Título '{product.original_title}'")
    
    if "general" not in models:
        logger.error("Modelo general no disponible para predicción.")
        raise HTTPException(status_code=503, detail="Modelo general no disponible.")

    try:
        row = product.model_dump()
        if row.get("confidence") is None:
            row["confidence"] = 0.95

        categoria_producto = product.category

        # PREDICCIÓN MODELO GENERAL
        df_general = prepare_for_lightgbm(row, models["general"])
        
        log_pred_general = float(models["general"].predict(df_general)[0])

        # LÓGICA DE BLENDING
        peso_general = 1.0
        peso_especifico = 0.0
        log_pred_especifico = 0.0
        modelo_usado = "Solo_General_(LightGBM)"

        # 3. PREDICCIÓN MODELO CATEGORÍA
        if categoria_producto in models and categoria_producto in CATEGORY_WEIGHTS:
            peso_general, peso_especifico = CATEGORY_WEIGHTS[categoria_producto]
            
            df_especifico = prepare_for_catboost(row, models[categoria_producto])
            log_pred_especifico = float(models[categoria_producto].predict(df_especifico)[0])
            modelo_usado = f"Blending_General(LGBM)+Especifico_{categoria_producto}(CatBoost)"

        # ENSAMBLADO MATEMÁTICO
        log_pred_modelos = (log_pred_general * peso_general) + (log_pred_especifico * peso_especifico)
        precio_modelos_eur = float(np.expm1(log_pred_modelos))

        # POST-PROCESAMIENTO: Corrector por Similitud Textual
        log_final_corregido = log_pred_modelos
        motivo_correccion = "Corrector no inicializado"
        
        if corrector_api is not None:
            try:
                # El corrector buscará el 'original_title' dentro de 'row'
                log_final_corregido, motivo_correccion = corrector_api.corregir_prediccion(log_pred_modelos, row)
            except Exception as e:
                logger.error(f"Error en corrector: {e}")
                motivo_correccion = "Error en corrección (Se usa modelo puro)"

        # TRANSFORMACIÓN FINAL A EUROS (Ya con el precio corregido)
        precio_final_eur = float(np.expm1(log_final_corregido))
        diferencia_log = log_final_corregido - log_pred_modelos
        
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"Predicción completada en {latency:.4f}s. Precio final: {round(precio_final_eur, 2)}€")

        # RESPUESTA JSON ENRIQUECIDA
        return {
            "predicted_price": round(precio_final_eur, 2),
            "predicted_log_price": round(log_final_corregido, 4),
            
            "prediction_details": {
                "strategy": modelo_usado,
                "general_log_pred": round(log_pred_general, 4),
                "specific_log_pred": round(log_pred_especifico, 4) if peso_especifico > 0 else None,
                "weights": {
                    "general": peso_general,
                    "specific": peso_especifico
                }
            },
            
            "correction_details": {
                "status": motivo_correccion,
                "original_model_price_eur": round(precio_modelos_eur, 2),
                "original_model_log_price": round(log_pred_modelos, 4),
                "log_difference": round(diferencia_log, 4)
            },
            
            "status": "success",
            "latency_seconds": round(latency, 4)
        }

    except Exception as e:
        import traceback
        logger.error(f"Error crítico en predicción: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")