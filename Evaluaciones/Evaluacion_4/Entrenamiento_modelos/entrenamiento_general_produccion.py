import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import mlflow
import warnings
import pickle
import os

# Suprimir warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE MLFLOW ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Amazon_Global_Pricing_LIGHTGBM_PROD"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- HIPERPARÁMETROS ÓPTIMOS (Guardados previamente) ---
BEST_PARAMS = {
    'learning_rate': 0.04079953871130477, 
    'max_depth': 7, 
    'num_leaves': 48, 
    'min_child_samples': 17, 
    'feature_fraction': 0.7210199975854399, 
    'bagging_fraction': 0.7259381630822127, 
    'bagging_freq': 3, 
    'reg_alpha': 0.0031775077299006937, 
    'reg_lambda': 6.961418991463772
}
BEST_ITERATION = 904  # Iteración óptima encontrada con Early Stopping


def entrenar_modelo_general_produccion(df, best_params, best_iteration, target_col='log_original_price'):
    """
    Entrena el modelo global con el 100% de los datos disponibles, 
    usando los parámetros predefinidos y la iteración óptima exacta.
    """
    with mlflow.start_run(run_name="LGBM_General_100%_Produccion"):
        print("\n--- 1. PREPARACIÓN DE DATOS AL 100% ---")
        
        # 1. Limpieza de columnas
        cols_a_eliminar = [target_col, 'image_url', 'product_url', 'original_title']
        existing_cols = [c for c in cols_a_eliminar if c in df.columns]
        
        X = df.drop(columns=existing_cols)
        y = df[target_col]

        # 2. Categorizar variables de texto (Exigencia de LightGBM)
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype('category')

        # 3. Configurar parámetros fijos + óptimos
        params = best_params.copy()
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42
        })

        # IMPORTANTE: n_estimators es igual al best_iteration (No hay early stopping)
        model = lgb.LGBMRegressor(**params, n_estimators=best_iteration)
        
        # 4. Entrenamiento Final (Sin set de validación)
        print(f"\n🚀 Entrenando modelo final de Producción ({len(X)} filas, {best_iteration} iteraciones)...")
        model.fit(X, y)

        # 5. Loggear parámetros en MLflow (Omitimos métricas porque entrenamos con el 100%)
        mlflow.log_params(params)
        mlflow.log_param("final_n_estimators_used", best_iteration)
        mlflow.log_param("dataset_size", len(X))

        # 6. Guardado del modelo
        nombre_archivo = 'modelo_general_lightgbm_produccion.pkl'
        
        # Aseguramos que la carpeta existe (ajusta la ruta si lo necesitas)
        os.makedirs("Modelos_produccion", exist_ok=True)
        ruta_guardado = os.path.join("Modelos", nombre_archivo)
        
        with open(ruta_guardado, 'wb') as archivo:
             pickle.dump(model, archivo)

        print(f"💾 ¡Modelo GLOBAL guardado con éxito en '{ruta_guardado}'!")
        
        return model

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # Ruta a tu dataset limpio
    csv_path = "../../../Datasets/evaluacion4_produccion.csv"
    
    # 1. Separar el set de datos para el Dashboard y quedarnos con el de entrenamiento
    df_para_entrenar = pd.read_csv(csv_path)
    
    
    # 2. Entrenar directamente con el 100% de ese dataset restante
    modelo_final = entrenar_modelo_general_produccion(
        df=df_para_entrenar, 
        best_params=BEST_PARAMS, 
        best_iteration=BEST_ITERATION,
        target_col='log_original_price'
    )