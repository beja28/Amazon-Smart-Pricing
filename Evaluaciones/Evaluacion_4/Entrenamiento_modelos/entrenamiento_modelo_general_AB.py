import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import warnings
from tqdm import tqdm

# Suprimir warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Amazon_Global_Stacked_Minimalist"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- HIPERPARÁMETROS GANADORES (MODELO A) ---
BEST_PARAMS_A = {
    'iterations': 3176,
    'depth': 6,
    'learning_rate': 0.07006861077404392,
    'l2_leaf_reg': 5.888401099727424,
    'random_strength': 0.028880735614999892,
    'rsm': 0.30384518013113837,
    'subsample': 0.5271988520434844,
    'nan_mode': 'Min',
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

# --- CONFIGURACIÓN DEL MODELO B (CORRECTOR SELECTIVO) ---
PARAMS_B = {
    'iterations': 1200,
    'depth': 4,
    'learning_rate': 0.02,
    'rsm': 1.0,  # Usamos todas las (pocas) columnas que le damos
    'loss_function': 'MAE',
    'random_seed': 42,
    'verbose': False
}

# --- DEFINICIÓN DE COLUMNAS ---
# 1. Todas las columnas para el Modelo A
RAW_FEATURES_STRING = "category,subtype,market_tier,condition,is_premium_brand,tech_generation,brand,confidence,printer_tech,is_color,is_multifunction,paper_size_max,ppm,pack_count,is_xl,ups_capacity_value,ups_capacity_unit,is_rackmount,rack_u,power_capacity_value,power_capacity_unit,audio_channels,total_power_w,has_dolby_atmos,has_subwoofer,has_anc,form_factor,mic_type,audio_resolution,video_output_res,switch_type,dpi,has_screen,is_thunderbolt,max_resolution,read_speed_mbs,scan_engine,wifi_standard,speed_class,node_count,frequency_bands,has_poe,multi_gig_ports,power_source,smart_protocol,is_outdoor,screen_size_in,resolution_standard,panel_tech,refresh_rate_hz,is_smart,mount_type,adjustment_type,num_screens,max_weight_kg,is_heavy_duty,sensor_format,max_video_res,is_body_only,mount_system,megapixels,is_pro_line,focal_range,max_aperture_f,is_kit,component_series,vram_gb,has_x3d,ram_gb,storage_gb,tech_gen,power_wattage,efficiency_rating,chipset,is_aio,is_gaming,platform_family,is_digital_edition,nas_bays,model_name,cellular_type,has_stylus,case_size_mm,is_rugged,is_specialized,gps_activity,has_touchscreen,cable_length_m,interface_type,is_braided,material,brand_compatibility,is_active,is_original,connector_gender,is_wireless,port_count,storage_type,pd_wattage,case_keyboard,product_rating,is_best_seller,is_sponsored,buy_box_availability,sustainability_tags,has_coupon,discount_percentage,log_original_price,log_purchased_last_month,log_total_reviews,brand_subtype"
ALL_COLUMNS_A = [x.strip() for x in RAW_FEATURES_STRING.split(',')]
TARGET = 'log_original_price'
FEATURES_A = [c for c in ALL_COLUMNS_A if c != TARGET]

# 2. Solo columnas maestras para el Modelo B (Minimalista)
FEATURES_B_SELECT = ['category', 'subtype', 'brand', 'log_purchased_last_month'] 
# Nota: 'pred_A' se añade dinámicamente en el código

def prepare_global_data(csv_path):
    print("📂 Cargando dataset global...")
    df = pd.read_csv(csv_path)
    
    # Preparamos X con TODAS las features (para el Modelo A)
    available_features = [f for f in FEATURES_A if f in df.columns]
    X = df[available_features].copy()
    y = df[TARGET]
    
    # Tratamiento de Categóricas (Global)
    cat_features_indices = []
    for i, col in enumerate(X.columns):
        if X[col].dtype == 'object' or X[col].dtype.name == 'category' or X[col].dtype == 'bool':
            X[col] = X[col].fillna("Missing").astype(str)
            cat_features_indices.append(i)
            
    print(f"✅ Datos listos: {X.shape[0]} filas. Features Modelo A: {len(X.columns)}")
    return X, y, cat_features_indices

def get_oof_predictions(X, y, params, cat_indices):
    """Genera predicciones 'limpias' usando K-Fold con Barra de Progreso"""
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    print(f"   ... Generando OOF Preds ({n_splits} Folds)...")
    
    for train_idx, val_idx in tqdm(kf.split(X, y), total=n_splits, desc="Entrenando Folds", unit="fold"):
        X_t, y_t = X.iloc[train_idx], y.iloc[train_idx]
        X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(**params, cat_features=cat_indices)
        model.fit(X_t, y_t, eval_set=(X_v, y_v), verbose=False, early_stopping_rounds=50)
        oof_preds[val_idx] = model.predict(X_v)
        
    return oof_preds

def train_stacked_system(csv_path):
    with mlflow.start_run(run_name="Stacked_Minimalist_Training"):
        
        # 1. Preparar datos completos
        X, y, cat_indices_A = prepare_global_data(csv_path)
        
        # 2. Split Train/Validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- FASE 1: MODELO A (BASE) ---
        print("\n🏗️ FASE 1: Generando residuos del Modelo A...")
        oof_preds_train = get_oof_predictions(X_train, y_train, BEST_PARAMS_A, cat_indices_A)
        residuals_train = y_train - oof_preds_train
        
        # --- FASE 2: MODELO B (CORRECTOR MINIMALISTA) ---
        print("\n🔧 FASE 2: Entrenando Modelo B (Solo Features Maestras)...")
        
        # Crear dataset reducido para B
        # Solo usamos las columnas maestras disponibles + pred_A
        available_B_feats = [f for f in FEATURES_B_SELECT if f in X_train.columns]
        
        X_train_B = X_train[available_B_feats].copy()
        X_train_B['pred_A'] = oof_preds_train # Añadir la predicción como feature clave
        
        # Identificar categóricas para el Modelo B (recalcular índices porque el subset es diferente)
        cat_indices_B = []
        for i, col in enumerate(X_train_B.columns):
            if X_train_B[col].dtype == 'object' or X_train_B[col].dtype.name == 'category':
                cat_indices_B.append(i)
        
        # Entrenar B
        model_B = CatBoostRegressor(**PARAMS_B, cat_features=cat_indices_B)
        model_B.fit(X_train_B, residuals_train, verbose=False)
        
        # --- FASE 3: MODELO A FINAL ---
        print("\n🚀 FASE 3: Entrenando Modelo A Final...")
        model_A_final = CatBoostRegressor(**BEST_PARAMS_A, cat_features=cat_indices_A)
        model_A_final.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        # --- FASE 4: EVALUACIÓN ---
        print("\n⚖️ FASE 4: Evaluación Final (Comparativa)...")
        
        # Predicción A
        pred_val_A = model_A_final.predict(X_val)
        
        # Predicción B (Preparar datos reducidos)
        X_val_B = X_val[available_B_feats].copy()
        X_val_B['pred_A'] = pred_val_A
        pred_val_B = model_B.predict(X_val_B)
        
        # Combinar
        pred_final_log = pred_val_A + pred_val_B
        
        # Métricas (Euros)
        y_real_eur = np.expm1(y_val) # Usamos expm1 si log fue log1p, o exp si fue log normal. Asumo consistencia.
        pred_A_eur = np.expm1(pred_val_A)
        pred_final_eur = np.expm1(pred_final_log)
        
        mae_A = mean_absolute_error(y_real_eur, pred_A_eur)
        mae_final = mean_absolute_error(y_real_eur, pred_final_eur)
        r2_final = r2_score(y_real_eur, pred_final_eur)
        rmse_final = np.sqrt(mean_squared_error(y_real_eur, pred_final_eur))
        
        print(f"\n📊 RESULTADOS:")
        print(f"   -> Modelo A (Solo):    MAE = {mae_A:.2f}€")
        print(f"   -> Stacked (A+B):      MAE = {mae_final:.2f}€")
        print(f"   -> RMSE Final:         RMSE = {rmse_final:.2f}€")
        print(f"   -> R² Final:           R² = {r2_final:.4f}")
        
        if mae_final < mae_A:
            print("   ✅ ¡ÉXITO! El Modelo B Minimalista ayuda.")
        else:
            print("   ⚠️ El Modelo B sigue sin ayudar. Recomendación: Usar solo Modelo A.")
            
        # Log y Guardar
        mlflow.log_metric("mae_final_euro", mae_final)
        mlflow.log_metric("rmse_final_euro", rmse_final)
        mlflow.log_metric("r2_final", r2_final)
        model_A_final.save_model("model_A_global.cbm")
        model_B.save_model("model_B_minimalist.cbm")

if __name__ == "__main__":
    csv_path = "../../../Datasets/evaluacion4.csv" 
    train_stacked_system(csv_path)