import pandas as pd
import numpy as np
import re
import mlflow
from catboost import CatBoostRegressor
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    exclude_cols = [TARGET_COL, 'product_image_url', 'product_url', 'original_title']
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
        experiment_name = "Amazon_Category_Pricing_PROD_v3"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"CatBoost_PROD_{safe_cat_name}_v3"):
            
            # 3. Preparar parámetros definitivos
            final_params = {
                'iterations': params['best_iteration'],
                'depth': params['depth'],
                'learning_rate': params['learning_rate'],
                'l2_leaf_reg': params['l2_leaf_reg'],
                'loss_function': 'RMSE',
                'cat_features': cat_features,
                'verbose': 100,
                'allow_writing_files': False,
                'random_seed': 42
            }
            
            # 4. Entrenar el modelo
            print(f"🚀 Entrenando modelo campeón con {params['best_iteration']} iteraciones...")
            model = CatBoostRegressor(**final_params)
            model.fit(X_full, y_full)
            
            # 5. Guardado en MLflow y local de métricas y modelo
            mlflow.log_params(final_params)
            mlflow.log_param("dataset_size", len(X_full))
            mlflow.log_param("features_count", len(X_full.columns))
            
            model_filename = f"{category_name}.cbm"
            model.save_model(model_filename)
            
            # 6. --- GENERAR Y GUARDAR FEATURE IMPORTANCES EN MLFLOW ---
            print("📊 Generando gráfico de Feature Importances...")
            # Extraer las importancias y los nombres de las features
            feature_importances = model.get_feature_importance()
            feature_names = X_full.columns
            
            # Crear un DataFrame para ordenarlas fácilmente
            df_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            
            # Tomar el top 20 para que el gráfico sea legible (opcional)
            top_n = 10
            df_top = df_importances.head(top_n)

            # Crear el gráfico
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=df_top, color='#e9437a')
            plt.title(f'Top {top_n} Feature Importances - {category_name}')
            plt.xlabel('Importancia (CatBoost)')
            plt.ylabel('Feature')
            plt.tight_layout()

            # Guardar el gráfico temporalmente
            plot_filename = f"feature_importances_{safe_cat_name}.png"
            plt.savefig(plot_filename)
            plt.close() # Cerrar la figura para liberar memoria

            # Registrar el gráfico en MLflow como un artefacto
            mlflow.log_artifact(plot_filename, "plots")
            
            # Eliminar el archivo temporal local
            os.remove(plot_filename)
            print(f"✅ Gráfico guardado en MLflow (carpeta 'plots').")

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
        'depth': 7,
        'learning_rate': 0.1408011263679544,
        'l2_leaf_reg': 5.708308909966304,
        'best_iteration': 562
    }
    
    # Lanzar el entrenamiento
    entrenar_categoria_produccion(df_train_produccion, "Computers & Gaming", params_optimizados)