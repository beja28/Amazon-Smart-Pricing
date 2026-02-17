import pandas as pd
import numpy as np
import re
import mlflow
import joblib # Necesario para guardar el modelo Ridge
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# ==========================================
# 1. PREPARACIÓN DE DATOS
# ==========================================
def prepare_data_for_category(df, category_name, threshold=0.005):
    """
    Prepara los datos incluyendo la interacción 'brand_subtype' y
    un umbral bajo para salvar variables técnicas.
    """
    print(f"   ... Limpiando datos para: {category_name}")
    df_cat = df[df['category'] == category_name].copy()
    

    # --- B. LIMPIEZA DINÁMICA ---
    # Umbral 0.5%: Si hay 1000 productos, salvamos columnas con al menos 5 datos
    min_samples = int(len(df_cat) * threshold)
    if min_samples < 2: min_samples = 2 
    df_cat = df_cat.dropna(axis=1, thresh=min_samples)
    
    # --- C. DEFINICIÓN DE TARGET ---
    target = 'log_original_price'
    if target not in df_cat.columns:
        raise ValueError(f"❌ Falta '{target}' en {category_name}")
    
    # --- D. SEPARAR X e y ---
    cols_to_drop = [target, 'category']
    existing_drop = [c for c in cols_to_drop if c in df_cat.columns]
    
    X = df_cat.drop(columns=existing_drop)
    y = df_cat[target]
    
    # --- E. TRATAMIENTO DE CATEGÓRICAS ---
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)
        
    return X, y, cat_features

# ==========================================
# 2. GENERACIÓN DE RESIDUOS (Anti-Leakage)
# ==========================================
def get_oof_predictions(model_params, X, y, cat_features, n_splits=10):
    """
    Entrena el Modelo A en K-Folds para generar predicciones 'limpias'
    (Out-of-Fold) sobre el set de entrenamiento. Vital para evitar Data Leakage.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    print(f"   ... Generando residuos OOF con {n_splits}-Fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_t, y_t = X.iloc[train_idx], y.iloc[train_idx]
        X_v = X.iloc[val_idx]
        
        # Modelo temporal (Clon del Modelo A)
        model = CatBoostRegressor(**model_params)
        model.fit(X_t, y_t, eval_set=(X_v, y.iloc[val_idx]), verbose=False)
        
        # Predecir sobre la validación del fold
        oof_preds[val_idx] = model.predict(X_v)
        
    return oof_preds

# ==========================================
# 3. ENTRENAMIENTO DEL PIPELINE COMPLETO
# ==========================================
def train_stacked_pipeline(X_train, y_train, X_val, y_val, cat_features, params_A):
    
    with mlflow.start_run(nested=True):
        
        # --- PASO 1: Generar Residuos Seguros (Modelo A) ---
        oof_preds_train = get_oof_predictions(params_A, X_train, y_train, cat_features, n_splits=10)
        
        # Residuo = Real - Predicho
        residuals_train = y_train - oof_preds_train
        
        # --- PASO 2: Entrenar Modelo B (XGBoost LINEAL) ---
        print("   ... Entrenando Modelo B (Target Encoding + XGBoost Linear)...")
        
        # Identificar numéricas
        num_features = [c for c in X_train.columns if c not in cat_features]
        
        # Preprocesador: SOLO codificamos categorías. Las numéricas pasan con sus NaNs.
        preprocessor = ColumnTransformer(transformers=[
            ('cat', TargetEncoder(smoothing=20.0), cat_features), # Smoothing alto
            ('num', 'passthrough', num_features) # <--- ¡Aquí pasan los NaNs tal cual!
        ])
        
        # Configuración de XGBoost para actuar como un Ridge robusto
        xgb_linear_params = {
            'booster': 'gblinear',      # <--- ESTO LO CONVIERTE EN LINEAL (NO ÁRBOLES)
            'n_estimators': 200,        # Iteraciones de descenso de gradiente
            'learning_rate': 0.1,
            'reg_lambda': 10.0,         # Equivalente al Alpha de Ridge (L2)
            'reg_alpha': 1.0,           # Añadimos un poco de L1 (Lasso) para limpieza
            'missing': np.nan,          # Le decimos explícitamente qué es un nulo
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model_B = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(**xgb_linear_params)) 
        ])
        
        # XGBoost maneja internamente los NaNs sin quejarse
        model_B.fit(X_train, residuals_train)
        
        # --- PASO 3: Re-Entrenar Modelo A Final ---
        print("   ... Re-entrenando Modelo A con todo el set...")
        model_A = CatBoostRegressor(**params_A)
        model_A.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        # --- PASO 4: Evaluación ---
        pred_val_A = model_A.predict(X_val)
        pred_val_B = model_B.predict(X_val)
        pred_val_final_log = pred_val_A + pred_val_B
        
        # Invertir a Euros
        y_real_euro = np.exp(y_val)
        pred_A_euro = np.exp(pred_val_A)
        pred_final_euro = np.exp(pred_val_final_log)
        
        # Métricas
        rmse_A = np.sqrt(mean_squared_error(y_real_euro, pred_A_euro))
        mae_A = mean_absolute_error(y_real_euro, pred_A_euro)
        
        rmse_final = np.sqrt(mean_squared_error(y_real_euro, pred_final_euro))
        mae_final = mean_absolute_error(y_real_euro, pred_final_euro)
        r2_final = r2_score(y_real_euro, pred_final_euro)
        
        print(f"\n📊 RESULTADOS COMPARATIVOS:")
        print(f"   -> Modelo A Solo (Base): RMSE={rmse_A:.2f}€ | MAE={mae_A:.2f}€")
        print(f"   -> Modelo A + B (Stack): RMSE={rmse_final:.2f}€ | MAE={mae_final:.2f}€")
        
        # --- LOGGING MLFLOW ---
        for k, v in params_A.items():
            mlflow.log_param(f"model_A_{k}", v)
        
        mlflow.log_param("model_B_type", "XGBoost_Linear_NativeNaN")
        
        mlflow.log_metric("base_rmse", rmse_A)
        mlflow.log_metric("base_mae", mae_A)
        mlflow.log_metric("final_rmse", rmse_final)
        mlflow.log_metric("final_mae", mae_final)
        mlflow.log_metric("final_r2", r2_final)
        
        # Guardar Modelos
        model_A.save_model("model_A.cbm")
        joblib.dump(model_B, "model_B.joblib")
        
        mlflow.log_artifact("model_A.cbm")
        mlflow.log_artifact("model_B.joblib")
        
        return model_A, model_B, mae_final

# ==========================================
# 4. ORQUESTADOR PRINCIPAL
# ==========================================
def run_stacked_experiment(df, category_name):
    print(f"\n🚀 INICIANDO PIPELINE STACKED (RIDGE): {category_name}")
    
    # 1. Preparar Datos
    X, y, cat_features = prepare_data_for_category(df, category_name)
    print(f" -> Features activas: {len(X.columns)}")
    
    if len(X) < 50:
        print("⚠️ Muy pocos datos. Saltando.")
        return

    # 2. Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Configurar Experimento
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', category_name)
    experiment_name = f"Pricing_Stacked_{safe_name}_Ridge"
    mlflow.set_experiment(experiment_name)
    
    # 4. Parámetros del Modelo A (Tus ganadores)
    params_A = {
        'iterations': 1323,
        'depth': 8,
        'learning_rate': 0.0489,
        'l2_leaf_reg': 5.86,
        'random_strength': 4.89,
        'loss_function': 'RMSE', 
        'cat_features': cat_features,
        'verbose': False,
        'allow_writing_files': False,
        'random_seed': 42
    }
    
    # 5. Ejecutar Pipeline
    with mlflow.start_run(run_name="Stacked_Training_Ridge"):
        mlflow.set_tag("features", ",".join(X.columns))
        train_stacked_pipeline(X_train, y_train, X_val, y_val, cat_features, params_A)

# --- EJECUCIÓN ---
if __name__ == "__main__":
    # Cargar CSV (Ajusta la ruta si es necesario)
    csv_path = "../../../Datasets/evaluacion4.csv" 
    print(f"📂 Cargando datos...")
    df = pd.read_csv(csv_path)
    
    # Ejecutar
    run_stacked_experiment(df, "Audio & Media Systems")