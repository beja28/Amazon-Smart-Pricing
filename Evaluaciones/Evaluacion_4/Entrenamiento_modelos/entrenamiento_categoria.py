import pandas as pd
import numpy as np
import re
import mlflow
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

# --- CONFIGURACIÓN GLOBAL ---
TARGET_COL = 'log_original_price'
THRESHOLD = 0.005  # 0.5% de presencia mínima

def prepare_aligned_data(df_full, category_name):
    """
    1. Hace el split GLOBAL (random_state=42) para alinearse con el modelo general.
    2. Filtra por categoría.
    3. Selecciona features basándose SOLO en el Train (evita leakage).
    4. Limpia tipos de datos (Booleanos -> Float, Categóricas -> String).
    """
    
    # 1. SPLIT GLOBAL (Crucial para alineación)
    print("✂️  Aplicando Split Global (random_state=42)...")
    train_full, val_full = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['subtype'])
    
    # 2. FILTRADO POR CATEGORÍA
    print(f"🔍 Filtrando categoría '{category_name}'...")
    df_train = train_full[train_full['category'] == category_name].copy()
    df_val = val_full[val_full['category'] == category_name].copy()
    
    if len(df_train) < 20 or len(df_val) < 5:
        print(f"⚠️ Saltando {category_name}: Muy pocos datos tras el split.")
        return None, None, None, None, None

    # 3. SELECCIÓN DE FEATURES (Basada en TRAIN)
    # Excluimos target y columnas administrativas
    exclude_cols = [TARGET_COL, 'category']
    potential_cols = [c for c in df_train.columns if c not in exclude_cols]
    
    selected_features = []
    for col in potential_cols:
        # Calculamos presencia solo en train
        if df_train[col].notna().mean() > THRESHOLD:
            selected_features.append(col)
            
    print(f"🎯 Features seleccionadas (Threshold > {THRESHOLD}): {len(selected_features)}")
    
    # 4. PREPARACIÓN FINAL DE X e y
    X_train = df_train[selected_features].copy()
    y_train = df_train[TARGET_COL]
    
    X_val = df_val[selected_features].copy()
    y_val = df_val[TARGET_COL]
    
    # 5. TRATAMIENTO DE CATEGÓRICAS (TU LÓGICA ORIGINAL)
    # Identificamos las columnas object/category en el set de entrenamiento
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Aplicamos la transformación a ambos conjuntos (Train y Val)
    print(f"🧹 Procesando {len(cat_features)} variables categóricas...")
    for col in cat_features:
        # Rellenar nulos y forzar string
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_val[col] = X_val[col].fillna("Missing").astype(str)

    return X_train, y_train, X_val, y_val, cat_features

def objective(trial, X_train, y_train, X_val, y_val, cat_features, model_type):
    with mlflow.start_run(nested=True): 
        
        # --- HIPERPARÁMETROS ---
        if model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'depth': trial.suggest_int('depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'loss_function': 'RMSE',
                'verbose': False,
                'cat_features': cat_features, # CatBoost usa nombres o indices
                'allow_writing_files': False,
                'random_seed': 42
            }
            model = CatBoostRegressor(**params, use_best_model=True)
        else:
            X_train_fit, X_val_fit = X_train, X_val

        # --- ENTRENAMIENTO ---
        if model_type == 'catboost':
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False, use_best_model=True)
        else:
            model.fit(X_train_fit, y_train, eval_set=[(X_val_fit, y_val)], verbose=False)


        preds_log = model.predict(X_val)
        
        preds_euros = np.expm1(preds_log)
        y_val_euros = np.expm1(y_val)
        
        rmse_euros = np.sqrt(mean_squared_error(y_val_euros, preds_euros))
        mae_euros = mean_absolute_error(y_val_euros, preds_euros)
        r2 = r2_score(y_val_euros, preds_euros)
        
        mlflow.log_params(trial.params)
        mlflow.log_metric("rmse_euros", rmse_euros)
        mlflow.log_metric("mae_euros", mae_euros)
        mlflow.log_metric("r2_real", r2)
        
        return rmse_euros 

def run_aligned_experiment(df_full, category_name, n_trials=20):
    print(f"\n🔬 PROCESANDO ALINEADO: {category_name}")
    
    try:
        # 1. Preparar datos ALINEADOS
        X_train, y_train, X_val, y_val, cat_features = prepare_aligned_data(df_full, category_name)
        
        if X_train is None: return

        print(f" -> Dimensiones Train: {len(X_train)} | Val: {len(X_val)}")
        
        # 2. Configurar MLflow
        safe_cat_name = re.sub(r'[^a-zA-Z0-9_]', '_', category_name)
        experiment_name = f"Pricing_{safe_cat_name}_Stratify"
        mlflow.set_experiment(experiment_name)
        
        active_cols_str = ",".join(list(X_train.columns))
        
        # --- OPTIMIZACIÓN CATBOOST ---
        print(" -> Optimizando CatBoost...")
        with mlflow.start_run(run_name="CatBoost_Optimization_Aligned"):
            mlflow.set_tag("features_used", active_cols_str)
            
            study_cb = optuna.create_study(direction='minimize', study_name=f"CB_{safe_cat_name}")
            study_cb.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val, cat_features, "catboost"), 
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            print(f"    🏆 Mejor RMSE CatBoost: {study_cb.best_value:.2f}€")
            
            # --- GUARDADO FINAL ---
            print("    💾 Guardando modelo alineado...")
            best_params = study_cb.best_params
            best_params.update({'cat_features': cat_features, 'loss_function': 'RMSE', 'verbose': False, 'random_seed': 42})
            
            final_model = CatBoostRegressor(**best_params)
            final_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            
            # --- AQUÍ ESTÁ LA MAGIA: EXTRAER EL ADN ---
            best_iteration = final_model.get_best_iteration()
            print(f"    ⭐ Iteración óptima encontrada: {best_iteration}")
            
            # Guardamos todo en el Run Padre de MLflow
            mlflow.log_params(study_cb.best_params) # Los params de Optuna
            mlflow.log_param("best_iteration", best_iteration) # El número mágico
            mlflow.log_param("feature_names", active_cols_str) # El orden exacto de columnas
            mlflow.log_param("cat_features_list", cat_features) # Qué columnas eran texto
            mlflow.log_metric("final_rmse_euros", study_cb.best_value)
            
            model_filename = f"model_{category_name}.cbm"
            final_model.save_model(model_filename)
            print(f"       -> Modelo guardado: {model_filename}")

        # (Puedes repetir el bloque para XGBoost si lo necesitas)
            
    except Exception as e:
        print(f"❌ Error crítico en {category_name}: {e}")
        import traceback
        traceback.print_exc()

# --- EJECUCIÓN ---
if __name__ == "__main__":
    csv_path = "../../../Datasets/evaluacion4.csv" 
    print(f"📂 Cargando datos COMPLETOS desde {csv_path}...")
    df_full = pd.read_csv(csv_path)
    
    # Probamos solo con la categoría target
    run_aligned_experiment(df_full, "Mobile Devices", n_trials=30)