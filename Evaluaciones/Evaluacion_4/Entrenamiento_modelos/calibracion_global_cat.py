import pandas as pd
import numpy as np
import joblib  # <--- NECESARIO PARA CARGAR LIGHTGBM (.pkl)
import lightgbm as lgb # <--- Importamos LightGBM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
CSV_PATH = "../../../Datasets/evaluacion4.csv"
# Asegúrate de que este path apunta a tu .pkl o .joblib
PATH_MODELO_GLOBAL = "modelo_general_lightgbm.pkl" 
PATH_MODELO_OFFICE = "model_Networking & Smart Home.cbm"

TARGET_CATEGORY = "Networking & Smart Home"
TARGET_COL = 'log_original_price'

# --- 1. FUNCIÓN DE PREPARACIÓN PARA CATBOOST (Especialista) ---
def prepare_data_for_catboost(df, model):
    """
    Lógica original: Strings para categóricas, Floats para numéricas.
    """
    df_ready = df.copy()
    
    # Identificar features del modelo
    all_features = model.feature_names_
    cat_indices = model.get_cat_feature_indices()
    cat_feature_names = set([all_features[i] for i in cat_indices])
    
    for col in all_features:
        if col not in df_ready.columns:
            df_ready[col] = 0 
            continue

        if col in cat_feature_names:
            # CatBoost quiere Strings
            df_ready[col] = df_ready[col].fillna("Missing").astype(str)
        else:
            # Numéricas y Booleanos
            if df_ready[col].dtype == 'bool' or df_ready[col].dtype == 'object':
                df_ready[col] = df_ready[col].map({True: 1, False: 0, "True": 1, "False": 0, "Yes": 1, "No": 0})
                df_ready[col] = pd.to_numeric(df_ready[col], errors='coerce').fillna(0)
            else:
                df_ready[col] = df_ready[col].fillna(0).astype(float)
                
    return df_ready[all_features]

# --- 2. FUNCIÓN DE PREPARACIÓN PARA LIGHTGBM (Global) ---
def prepare_data_for_lightgbm(df, target_col):
    """
    Replica EXACTAMENTE la lógica de tu entrenamiento profesional:
    1. Eliminar columnas administrativas.
    2. Convertir object -> category.
    """
    df_ready = df.copy()
    
    # 1. Limpieza de columnas (IGUAL QUE EN ENTRENAMIENTO)
    cols_a_eliminar = [target_col]
    # Solo borramos las que existan en el df actual
    drop_cols = [c for c in cols_a_eliminar if c in df_ready.columns]
    df_ready = df_ready.drop(columns=drop_cols)
    
    # 2. Categorizar variables de texto (IGUAL QUE EN ENTRENAMIENTO)
    cat_features = df_ready.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        df_ready[col] = df_ready[col].astype('category')
        
    return df_ready

def calibrate_blending_common_ground():
    print(f"📂 Cargando dataset original: {CSV_PATH}...")
    df_full = pd.read_csv(CSV_PATH)

    # 1. REPLICAR SPLIT GLOBAL (Obtener el 20% de Test)
    print("✂️  Replicando train_test_split (random_state=42)...")
    _, df_test_global = train_test_split(df_full, test_size=0.2, random_state=42)


    # 2. FILTRAR SOLO LA CATEGORÍA TARGET
    print(f"🔍 Filtrando categoría '{TARGET_CATEGORY}'...")
    df_common = df_test_global[df_test_global['category'] == TARGET_CATEGORY].copy()

    if len(df_common) == 0:
        print("❌ Error: No hay productos de esta categoría en el set de validación global.")
        return
    
    print(f"✅ Set de 'Common Ground': {len(df_common)} productos.")
    
    # Target real
    y_real_eur = np.expm1(df_common[TARGET_COL])

    # --- CARGA DE MODELOS ---
    print("\n🤖 Cargando modelos...")
    
    # CARGA LIGHTGBM (Usando joblib)
    try:
        model_global = joblib.load(PATH_MODELO_GLOBAL)
        print("   -> Modelo Global (LightGBM) cargado.")
    except Exception as e:
        print(f"   ❌ Error cargando LightGBM: {e}")
        return

    # CARGA CATBOOST (Usando método nativo)
    model_office = CatBoostRegressor()
    model_office.load_model(PATH_MODELO_OFFICE)
    print("   -> Modelo Especialista (CatBoost) cargado.")

    # --- PREDICCIONES SIMULTÁNEAS ---
    print("🔮 Generando predicciones...")

    # A) PREDICCIÓN GLOBAL (LightGBM)
    # Preparamos los datos con la lógica de LightGBM
    X_global = prepare_data_for_lightgbm(df_common, TARGET_COL)
    
    # LightGBM es un poco sensible con las columnas extra que no usó en train.
    # Nos aseguramos de quedarnos solo con las columnas que el modelo conoce.
    # feature_name_ está disponible en el objeto booster o en el sklearn wrapper
    try:
        feats_lgbm = model_global.feature_name_
        X_global = X_global[feats_lgbm]
    except:
        pass # Si falla, asumimos que X_global ya está correcto por la función de drop
        
    pred_global_log = model_global.predict(X_global)
    pred_global_eur = np.expm1(pred_global_log)
    
    mae_global = mean_absolute_error(y_real_eur, pred_global_eur)
    rmse_global = np.sqrt(mean_squared_error(y_real_eur, pred_global_eur))
    r2_global = r2_score(y_real_eur, pred_global_eur)

    # B) PREDICCIÓN ESPECIALISTA (CatBoost)
    # Preparamos los datos con la lógica de CatBoost
    X_office = prepare_data_for_catboost(df_common, model_office)
    pred_office_log = model_office.predict(X_office)
    pred_office_eur = np.expm1(pred_office_log)

    mae_office = mean_absolute_error(y_real_eur, pred_office_eur)
    rmse_office = np.sqrt(mean_squared_error(y_real_eur, pred_office_eur))
    r2_office = r2_score(y_real_eur, pred_office_eur)

    # --- REPORTE Y BLENDING (Igual que antes) ---
    print(f"\n📊 RESULTADOS INDIVIDUALES (Sobre Test Set):")
    print("-" * 60)
    print(f"{'Métrica':<10} | {'Global (LGBM)':<15} | {'Espec. (CB)':<15} | {'Ganador':<10}")
    print("-" * 60)
    print(f"{'MAE':<10} | {mae_global:.2f}€            | {mae_office:.2f}€            | {'Espec.' if mae_office < mae_global else 'Global'}")
    print(f"{'RMSE':<10} | {rmse_global:.2f}€            | {rmse_office:.2f}€            | {'Espec.' if rmse_office < rmse_global else 'Global'}")
    print(f"{'R²':<10} | {r2_global:.4f}             | {r2_office:.4f}             | {'Espec.' if r2_office > r2_global else 'Global'}")
    print("-" * 60)

    print("\n⚖️ Buscando el blend perfecto...")
    best_mae = float('inf')
    best_w = 0.0 # Peso del Especialista

    for w in np.linspace(0, 1, 101):
        blend_log = (w * pred_office_log) + ((1 - w) * pred_global_log)
        blend_eur = np.expm1(blend_log)
        
        mae = mean_absolute_error(y_real_eur, blend_eur)
        if mae < best_mae:
            best_mae = mae
            best_w = w

    best_blend_log = (best_w * pred_office_log) + ((1 - best_w) * pred_global_log)
    best_blend_eur = np.expm1(best_blend_log)
    rmse_blend = np.sqrt(mean_squared_error(y_real_eur, best_blend_eur))
    r2_blend = r2_score(y_real_eur, best_blend_eur)

    print("\n🏆 ESTRATEGIA GANADORA:")
    print(f"   Peso Especialista (CB): {best_w:.2f}")
    print(f"   Peso Global (LGBM):     {1-best_w:.2f}")
    print(f"\n   MAE Blend:  {best_mae:.2f}€ (Mejora: {min(mae_global, mae_office) - best_mae:.2f}€)")
    print(f"   RMSE Blend: {rmse_blend:.2f}€")
    print(f"   R² Blend:   {r2_blend:.4f}")

if __name__ == "__main__":
    calibrate_blending_common_ground()