import pandas as pd
import numpy as np
import re
import mlflow
from catboost import CatBoostRegressor
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN GLOBAL ---
mlflow.set_tracking_uri("http://127.0.0.1:5000") 
TARGET_COL = 'log_original_price'
THRESHOLD = 0.005  # 0.5% de presencia mínima

def prepare_category_data_100(df_full, category_name):
    """
    Prepara los datos usando el 100% del dataset disponible.
    Ya no hacemos split de Train/Val.
    """
    print(f"🔍 Filtrando categoría '{category_name}' al 100%...")
    
    # 1. Filtrar por categoría
    df_cat = df_full[df_full['category'] == category_name].copy()
    
    if len(df_cat) < 20:
        print(f"⚠️ Saltando {category_name}: Muy pocos datos para entrenar.")
        return None, None, None
        
    # 2. SELECCIÓN DE FEATURES (Basada en el 100% de la categoría)
    exclude_cols = [TARGET_COL, 'image_url', 'product_url', 'original_title']
    potential_cols = [c for c in df_cat.columns if c not in exclude_cols]
    
    selected_features = []
    for col in potential_cols:
        if df_cat[col].notna().mean() > THRESHOLD:
            selected_features.append(col)
            
    print(f"🎯 Features seleccionadas (Threshold > {THRESHOLD}): {len(selected_features)}")
    
    X_full = df_cat[selected_features].copy()
    y_full = df_cat[TARGET_COL]
    
    # 3. TRATAMIENTO DE CATEGÓRICAS (CatBoost Style)
    cat_features = X_full.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"🧹 Procesando {len(cat_features)} variables categóricas...")
    for col in cat_features:
        # Rellenar nulos y forzar string (como le gusta a CatBoost)
        X_full[col] = X_full[col].fillna("Missing").astype(str)

    return X_full, y_full, cat_features

def entrenar_categoria_produccion(df_full, category_name, params):
    print(f"\n🔬 INICIANDO ENTRENAMIENTO FINAL: {category_name}")
    
    try:
        # 1. Preparar datos
        X_full, y_full, cat_features = prepare_category_data_100(df_full, category_name)
        if X_full is None: return
        
        print(f" -> Dimensiones de entrenamiento (100%): {len(X_full)} filas")

        # 2. Configurar MLflow
        safe_cat_name = re.sub(r'[^a-zA-Z0-9_]', '_', category_name)
        experiment_name = "Amazon_Category_Pricing_PROD"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"CatBoost_PROD_{safe_cat_name}"):
            
            # 3. Preparar parámetros definitivos
            # IMPORTANTE: Reemplazamos 'iterations' por el 'best_iteration' histórico
            final_params = {
                'iterations': params['best_iteration'], # <--- LA MAGIA ESTÁ AQUÍ
                'depth': params['depth'],
                'learning_rate': params['learning_rate'],
                'l2_leaf_reg': params['l2_leaf_reg'],
                'loss_function': 'RMSE',
                'cat_features': cat_features,
                'verbose': 100, # Que nos imprima el progreso cada 100 rondas
                'allow_writing_files': False,
                'random_seed': 42 # Estabilidad total
            }
            
            # 4. Entrenar el modelo
            print(f"🚀 Entrenando modelo campeón con {params['best_iteration']} iteraciones...")
            model = CatBoostRegressor(**final_params)
            
            # Fíjate que YA NO HAY eval_set ni early_stopping_rounds
            model.fit(X_full, y_full)
            
            # 5. Guardado en MLflow y local
            mlflow.log_params(final_params)
            mlflow.log_param("dataset_size", len(X_full))
            mlflow.log_param("features_count", len(X_full.columns))
            
            model_filename = f"{category_name}.cbm"
            model.save_model(model_filename)
            print(f"✅ Modelo guardado exitosamente: {model_filename}")
            
    except Exception as e:
        print(f"❌ Error crítico en {category_name}: {e}")
        import traceback
        traceback.print_exc()

# --- EJECUCIÓN ---
if __name__ == "__main__":
    csv_path = "../../../Datasets/evaluacion4_produccion.csv"
    # 1. Separar el set de datos para el Dashboard y quedarnos con el de entrenamiento
    df_train_produccion = pd.read_csv(csv_path)
    
    # Los parámetros que rescataste de MLflow
    params_optimizados = {
        'depth': 8,
        'learning_rate': 0.10738858980199081,
        'l2_leaf_reg': 6.070371513951016,
        'best_iteration': 135  # Sustituye a las 782 iteraciones totales
    }
    
    # Lanzar el entrenamiento
    entrenar_categoria_produccion(df_train_produccion, "Mobile Devices", params_optimizados)