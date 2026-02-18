from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import os

app = FastAPI(title="Amazon Smart Pricing API", version="5.0")

# ---------------------------------------------------------------------------
# CARGA DEL MODELO
# ---------------------------------------------------------------------------
MODEL_PATH = "../Evaluaciones/Evaluacion_4/Entrenamiento_modelos/Modelos/global_model_optimized.cbm"
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    print(f"✅ Modelo cargado desde {MODEL_PATH}")
    print(f"📊 Nº features: {len(model.feature_names_)}")

try:
    load_model()
except Exception as e:
    print(f"⚠️  Error cargando modelo: {e}")


# ---------------------------------------------------------------------------
# COLUMNAS NUMÉRICAS — las que NO son categóricas en el entrenamiento
# Las categóricas vacías deben ser "" (string vacío)
# Las numéricas vacías deben ser np.nan
# ---------------------------------------------------------------------------
NUMERIC_COLS = {
    "confidence", "ppm", "pack_count", "ups_capacity_value", "rack_u",
    "power_capacity_value", "total_power_w", "dpi", "read_speed_mbs",
    "node_count", "multi_gig_ports", "screen_size_in", "refresh_rate_hz",
    "num_screens", "max_weight_kg", "megapixels", "max_aperture_f",
    "vram_gb", "ram_gb", "storage_gb", "power_wattage", "nas_bays",
    "case_size_mm", "cable_length_m", "port_count", "pd_wattage",
    "product_rating", "buy_box_availability", "sustainability_tags",
    "has_coupon", "discount_percentage", "log_purchased_last_month",
    "log_total_reviews",
}


# ---------------------------------------------------------------------------
# CONTRATO DE DATOS — data_dashboard.csv sin original_title,
# product_image_url, log_original_price y columnas derivadas del dashboard.
# Specs técnicos como Optional[Any] porque mezclan bool/str/float/None.
# ---------------------------------------------------------------------------
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

    # --- Features de comportamiento (tipos fijos) ---
    product_rating:           float
    is_best_seller:           str
    is_sponsored:             str
    buy_box_availability:     int
    sustainability_tags:      int
    has_coupon:               int
    discount_percentage:      float
    log_purchased_last_month: float
    log_total_reviews:        float


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "online", "model_loaded": model is not None, "version": "5.0"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"status": "healthy", "model": MODEL_PATH, "n_features": len(model.feature_names_)}


@app.get("/model/features")
def get_features():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"feature_names": model.feature_names_, "n_features": len(model.feature_names_)}


@app.post("/predict")
def predict_price(product: ProductInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    try:
        # 1. Convertir a dict
        row = product.model_dump()

        # 2. Construir brand_subtype (no está en data_dashboard)
        row["brand_subtype"] = f"{product.brand}_{product.subtype}"

        # 3. confidence por defecto
        if row.get("confidence") is None:
            row["confidence"] = 0.95

        # 4. Crear DataFrame
        df_input = pd.DataFrame([row])

        # 5. Añadir columnas que el modelo espera y no vienen en el payload
        for col in model.feature_names_:
            if col not in df_input.columns:
                df_input[col] = None

        # 6. Reordenar al orden exacto del modelo
        df_input = df_input[model.feature_names_]

        # 7. Tratar valores vacíos según el tipo de columna:
        #    - NUMÉRICAS:    None/NaN → np.nan  (CatBoost acepta NaN numérico)
        #    - CATEGÓRICAS:  None/NaN → ""      (CatBoost NO acepta NaN en categóricas)
        for col in df_input.columns:
            if col in NUMERIC_COLS:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce")
            else:
                # Categórica: convertir None/NaN a string vacío
                df_input[col] = df_input[col].apply(
                    lambda x: "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
                )

        # 8. Predicción — modelo predice log_price → convertir a precio real
        log_pred    = float(model.predict(df_input)[0])
        precio_real = float(np.exp(log_pred))

        return {
            "predicted_price":     round(precio_real, 2),
            "predicted_log_price": round(log_pred, 4),
            "status":              "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")