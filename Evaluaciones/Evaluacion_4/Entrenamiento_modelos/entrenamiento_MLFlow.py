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

# Suprimir warnings de pandas/sklearn para mantener la consola limpia
warnings.filterwarnings('ignore')

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

def prepare_data_for_category(df, category_name, threshold=0.005):
    """
    1. Filtra por categoría.
    2. Elimina columnas vacías (NaN) específicas de esa categoría.
    3. Separa X e y usando 'log_original_price' que ya existe.
    """
    df_cat = df[df['category'] == category_name].copy()
    
    # 1. Calculamos el número mínimo de valores no nulos requeridos
    min_samples = int(len(df_cat) * threshold)
    
    # 2. Filtrado dinámico por umbral
    # Borramos columnas que tengan menos de 'min_samples' datos reales
    df_cat = df_cat.dropna(axis=1, thresh=min_samples)
    
    # 3. Definir Target
    target = 'log_original_price'
    
    if target not in df_cat.columns:
        raise ValueError(f"❌ La columna '{target}' no existe en el DataFrame para {category_name}.")
    
    # 4. Definir X e y
    # Eliminamos el target y la columna 'category' (ya que todas son iguales aquí)
    cols_to_drop = [target, 'category']

    # Aseguramos que solo borramos lo que existe
    existing_drop = [c for c in cols_to_drop if c in df_cat.columns]
    
    X = df_cat.drop(columns=existing_drop)
    y = df_cat[target]
    
    # 5. Tratamiento de Categóricas (Rellenar nulos con "Unknown")
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)
        
    return X, y, cat_features

def objective(trial, X_train, y_train, X_val, y_val, cat_features, model_type):
    """
    Entrena con LOG (y_train) pero evalúa en EUROS (invirtiendo el log).
    """
    with mlflow.start_run(nested=True): 
        
        # --- HIPERPARÁMETROS ---
        if model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'depth': trial.suggest_int('depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
                'loss_function': 'MAE',
                'verbose': False,
                'cat_features': cat_features,
                'allow_writing_files': False,
                'random_seed': 42
            }
            model = CatBoostRegressor(**params)
            
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'enable_categorical': True, 
                'n_jobs': -1,
                'random_state': 42
            }
            # XGBoost requiere tipo 'category' explícito
            X_train_xgb = X_train.copy()
            X_val_xgb = X_val.copy()
            for col in cat_features:
                X_train_xgb[col] = X_train_xgb[col].astype('category')
                X_val_xgb[col] = X_val_xgb[col].astype('category')
            
            model = XGBRegressor(**params)
            X_train_fit, X_val_fit = X_train_xgb, X_val_xgb
        else:
            X_train_fit, X_val_fit = X_train, X_val

        # --- ENTRENAMIENTO (Target es Log Precio) ---
        if model_type == 'catboost':
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_train_fit, y_train, eval_set=[(X_val_fit, y_val)], verbose=False)

        # --- EVALUACIÓN (Log -> Euros) ---
        # 1. Predecir (El resultado está en escala Logarítmica)
        if model_type == 'xgboost':
            preds_log = model.predict(X_val_fit)
        else:
            preds_log = model.predict(X_val)
        
        # 2. Invertir Logaritmo para tener Euros reales
        preds_euros = np.expm1(preds_log)
        y_val_euros = np.expm1(y_val)
        
        # 3. Métricas interpretables
        rmse_euros = np.sqrt(mean_squared_error(y_val_euros, preds_euros))
        mae_euros = mean_absolute_error(y_val_euros, preds_euros)
        r2 = r2_score(y_val_euros, preds_euros)
        
        # --- LOGGING ---
        mlflow.log_params(trial.params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("rmse_euros", rmse_euros)
        mlflow.log_metric("mae_euros", mae_euros)
        mlflow.log_metric("r2_real", r2)
        
        # Optuna minimizará el error en Euros
        return rmse_euros 

def run_category_experiment(df, category_name, n_trials=20):
    print(f"\n🔬 PROCESANDO CATEGORÍA: {category_name}")
    
    try:
        # 1. Preparar datos (Ahora mucho más simple)
        X, y, cat_features = prepare_data_for_category(df, category_name)
        
        # Check de seguridad
        if len(X) < 20:
            print(f"⚠️ Saltando {category_name}: Muy pocos datos ({len(X)} filas).")
            return

        print(f" -> Dimensiones: {len(X)} filas x {len(X.columns)} columnas")
        
        # 2. Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Configurar Experimento MLflow
        safe_cat_name = re.sub(r'[^a-zA-Z0-9_]', '_', category_name)
        experiment_name = f"Pricing_{safe_cat_name}_v2_MAE"
        mlflow.set_experiment(experiment_name)
        
        # Tag con las columnas usadas (Muy útil para debugear)
        active_cols_str = ",".join(list(X.columns))
        
        # --- OPTIMIZACIÓN CATBOOST ---
        print(" -> Optimizando CatBoost...")
        with mlflow.start_run(run_name="CatBoost_Optimization") as parent_run:
            mlflow.set_tag("features_used", active_cols_str)
            
            # Dentro de run_category_experiment
            study_cb = optuna.create_study(
                direction='minimize', 
                study_name=f"CB_Optimization_{category_name}" # <--- Nombre personalizado
            )
            study_cb.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val, cat_features, "catboost"), 
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            mlflow.log_params(study_cb.best_params)
            mlflow.log_metric("best_rmse_euros", study_cb.best_value)
            print(f"    🏆 Mejor RMSE CatBoost: {study_cb.best_value:.2f}€")
            
            # --- NUEVO: GUARDAR ARTEFACTOS PARA EL NOTEBOOK ---
            print("    💾 Guardando modelo y datos de validación para análisis...")

            # 1. Re-entrenar el mejor modelo con los mejores parámetros
            best_params = study_cb.best_params
            best_params['cat_features'] = cat_features
            best_params['loss_function'] = 'RMSE'
            best_params['verbose'] = False
            
            final_model = CatBoostRegressor(**best_params)
            final_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            
            # 2. Guardar el modelo en formato nativo de CatBoost
            model_filename = f"model_catboost_{safe_cat_name}_v2_MAE.cbm"
            final_model.save_model(model_filename)
            print(f"       -> Modelo guardado: {model_filename}")
            
            # 3. Guardar el X_val e y_val para no perder la referencia de qué productos eran
            # Guardamos también el índice para poder cruzarlo con el CSV original y sacar los títulos
            val_data = X_val.copy()
            val_data['target_log_real'] = y_val # Añadimos el precio real al csv
            
            val_filename = f"val_data_{safe_cat_name}_v2_MAE.csv"
            val_data.to_csv(val_filename, index=True) # IMPORTANTE: index=True para mantener el ID del producto
            print(f"       -> Datos validación guardados: {val_filename}")

            # (Opcional) Guardar en MLflow también
            mlflow.log_artifact(model_filename)
            mlflow.log_artifact(val_filename)

        # --- OPTIMIZACIÓN XGBOOST ---
        print(" -> Optimizando XGBoost...")
        with mlflow.start_run(run_name="XGBoost_Optimization") as parent_run:
            mlflow.set_tag("features_used", active_cols_str)
            
            # Dentro de run_category_experiment
            study_xgb = optuna.create_study(
                direction='minimize', 
                study_name=f"XGB_Optimization_{category_name}" # <--- Nombre personalizado
            )
            study_xgb.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val, cat_features, "xgboost"), 
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            mlflow.log_params(study_xgb.best_params)
            mlflow.log_metric("best_rmse_euros", study_xgb.best_value)
            print(f"    🏆 Mejor RMSE XGBoost: {study_xgb.best_value:.2f}€")
            
    except Exception as e:
        print(f"❌ Error crítico en {category_name}: {e}")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    
    # 1. Cargar CSV
    csv_path = "../../../Datasets/evaluacion4.csv" 
    print(f"📂 Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. Obtener categorías
    unique_categories = df['category'].unique()
    print(f"📋 Categorías encontradas: {len(unique_categories)}")
    
    # 3. Bucle principal
    for cat in unique_categories:
        if cat != "Office, Printing & Power": continue # probar solo con una categoría para acelerar el desarrollo
        
        run_category_experiment(df, cat, n_trials=30)