import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import warnings
import json

# Suprimir warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Amazon_Global_Pricing_Optimized"
mlflow.set_experiment(EXPERIMENT_NAME)

# Tu lista exacta de columnas (raw string)
RAW_FEATURES_STRING = "category,subtype,market_tier,condition,is_premium_brand,tech_generation,brand,confidence,printer_tech,is_color,is_multifunction,paper_size_max,ppm,pack_count,is_xl,ups_capacity_value,ups_capacity_unit,is_rackmount,rack_u,power_capacity_value,power_capacity_unit,audio_channels,total_power_w,has_dolby_atmos,has_subwoofer,has_anc,form_factor,mic_type,audio_resolution,video_output_res,switch_type,dpi,has_screen,is_thunderbolt,max_resolution,read_speed_mbs,scan_engine,wifi_standard,speed_class,node_count,frequency_bands,has_poe,multi_gig_ports,power_source,smart_protocol,is_outdoor,screen_size_in,resolution_standard,panel_tech,refresh_rate_hz,is_smart,mount_type,adjustment_type,num_screens,max_weight_kg,is_heavy_duty,sensor_format,max_video_res,is_body_only,mount_system,megapixels,is_pro_line,focal_range,max_aperture_f,is_kit,component_series,vram_gb,has_x3d,ram_gb,storage_gb,tech_gen,power_wattage,efficiency_rating,chipset,is_aio,is_gaming,platform_family,is_digital_edition,nas_bays,model_name,cellular_type,has_stylus,case_size_mm,is_rugged,is_specialized,gps_activity,has_touchscreen,cable_length_m,interface_type,is_braided,material,brand_compatibility,is_active,is_original,connector_gender,is_wireless,port_count,storage_type,pd_wattage,case_keyboard,product_rating,is_best_seller,is_sponsored,buy_box_availability,sustainability_tags,has_coupon,discount_percentage,log_original_price,log_purchased_last_month,log_total_reviews,brand_subtype"

# Convertimos a lista y limpiamos espacios
ALL_COLUMNS = [x.strip() for x in RAW_FEATURES_STRING.split(',')]

# Definimos Target y Features
TARGET = 'log_original_price'
# Quitamos el target de la lista de features para no hacer trampa
FEATURES = [c for c in ALL_COLUMNS if c != TARGET]

def prepare_global_data(csv_path):
    print("📂 Cargando dataset global...")
    df = pd.read_csv(csv_path)
    
    # 1. Verificar columnas existentes
    available_features = [f for f in FEATURES if f in df.columns]
    missing = set(FEATURES) - set(df.columns)
    if missing:
        print(f"⚠️ {len(missing)} columnas no encontradas en el CSV (se ignorarán).")
    
    X = df[available_features].copy()
    y = df[TARGET]
    
    # 2. Tratamiento Inteligente de Tipos para CatBoost
    cat_features_indices = []
    
    for col in X.columns:
        # Detectar categóricas (object, category, bool)
        if X[col].dtype == 'object' or X[col].dtype.name == 'category' or X[col].dtype == 'bool':
            # Rellenamos nulos con "Missing" -> CatBoost aprenderá que "Missing" es un valor
            X[col] = X[col].fillna("Missing").astype(str)
            cat_features_indices.append(col)
        else:
            # Numéricas: NO rellenamos nulos. CatBoost maneja NaN nativamente.
            pass
            
    print(f"✅ Datos listos: {X.shape[0]} filas. Features usadas: {len(X.columns)}")
    return X, y, cat_features_indices

def objective(trial, X_train, y_train, X_val, y_val, cat_features):
    
    with mlflow.start_run(nested=True):
        # --- ESPACIO DE BÚSQUEDA "AGRESIVO" ---
        params = {
            # Arquitectura del árbol
            'iterations': trial.suggest_int('iterations', 1000, 4000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            
            # Regularización (Evitar Overfitting)
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            
            # --- AQUÍ ESTÁ LA MAGIA DE LA SELECCIÓN DE COLUMNAS ---
            # rsm (Colsample_bylevel): % de columnas que usa en cada split.
            # Si baja a 0.5, descarta la mitad de las columnas en cada decisión.
            'rsm': trial.suggest_float('rsm', 0.1, 1.0), 
            
            # Bagging (Subsample): % de filas que usa. Ayuda con el ruido.
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            
            # Manejo de Nulos (¿Los nulos valen mucho o poco?)
            'nan_mode': trial.suggest_categorical('nan_mode', ['Min', 'Max']),
            
            # Configuración fija
            'loss_function': 'RMSE',
            'cat_features': cat_features,
            'verbose': False,
            'early_stopping_rounds': 50,
            'random_seed': 42
        }

        # Entrenar
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        # Evaluar
        preds = model.predict(X_val)
        preds_euro = np.exp(preds)
        y_val_euro = np.exp(y_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_euro = np.sqrt(mean_squared_error(y_val_euro, preds_euro))
        r2 = r2_score(y_val_euro, preds_euro)
        
        # Loggear métricas
        mlflow.log_params(params)
        mlflow.log_metric("val_rmse_log", rmse)
        mlflow.log_metric("rmse_euro", rmse_euro)
        mlflow.log_metric("val_r2_euro", r2)
        
        return rmse

def run_global_optimization(csv_path, n_trials=30):
    
    # 1. Preparar datos
    X, y, cat_features = prepare_global_data(csv_path)
    
    # 2. Split (Como tenemos 6500 filas, 80/20 está bien)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"🔬 Iniciando optimización Optuna ({n_trials} trials)...")
    
    # 3. Optimización
    study = optuna.create_study(direction='minimize', study_name="Global_Pricing_Aggressive")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, cat_features), n_trials=n_trials, show_progress_bar=True)
    
    print("\n🏆 ¡Optimización Terminada!")
    print(f"Mejor RMSE (Log): {study.best_value:.4f}")
    print("Mejores Parámetros:", study.best_params)
    
    # 4. Entrenar Modelo Final con los ganadores
    print("\n🚀 Entrenando Modelo Final con todo el X_train...")
    best_params = study.best_params
    best_params['cat_features'] = cat_features
    best_params['loss_function'] = 'RMSE'
    best_params['verbose'] = False
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    # 5. Métricas Finales (Euros)
    preds_log = final_model.predict(X_val)
    preds_euro = np.expm1(preds_log) # Asumiendo log natural
    y_val_euro = np.expm1(y_val)
    
    mae_final = mean_absolute_error(y_val_euro, preds_euro)
    rmse_final = np.sqrt(mean_squared_error(y_val_euro, preds_euro))
    r2_final = r2_score(y_val_euro, preds_euro)
    
    print(f"\n📊 RESULTADOS FINALES (Validación):")
    print(f"   MAE:  {mae_final:.2f}€")
    print(f"   RMSE: {rmse_final:.2f}€")
    print(f"   R2:   {r2_final:.4f}")
    
    # Guardar
    mlflow.log_params(best_params)
    mlflow.log_metric("final_mae_euro", mae_final)
    mlflow.log_metric("final_rmse_euro", rmse_final)
    mlflow.log_metric("final_r2", r2_final)
    final_model.save_model("global_model_optimized.cbm")
    mlflow.log_artifact("global_model_optimized.cbm")
    
    # Feature Importance (Ver qué descartó)
    print("\n🔝 Top 5 Variables más importantes:")
    feature_importance = final_model.get_feature_importance()
    feature_names = X.columns
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance}).sort_values(by='importance', ascending=False)
    print(fi_df.head(5))
    
    return final_model

if __name__ == "__main__":
    csv_path = "../../../Datasets/evaluacion4.csv" 
    run_global_optimization(csv_path, n_trials=30)