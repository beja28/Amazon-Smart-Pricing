import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import mlflow
import optuna
import warnings
import pickle

# Suprimir warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Amazon_Global_Pricing_LIGHTGBM"
mlflow.set_experiment(EXPERIMENT_NAME)


def optimizar_con_optuna(df, target_col='log_original_price'):
    # --- 1. PREPARACIÓN IDÉNTICA ---
    cols_a_eliminar = [target_col, 'original_row_id', 'error_log', 'original_title', 'price']
    X = df.drop(columns=[c for c in cols_a_eliminar if c in df.columns])
    y = df[target_col]

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].astype('category')

    # Split de 3 vías
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.125, random_state=42)

    def objective(trial):
        # 2. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        # 3. ENTRENAMIENTO RÁPIDO PARA OPTUNA
        gbm = lgb.LGBMRegressor(**param, n_estimators=1000)
        gbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        preds = gbm.predict(X_val)
        mae_val = mean_absolute_error(np.expm1(y_val), np.expm1(preds))
        
        return mae_val # Queremos minimizar el MAE en Euros

    # 4. EJECUTAR LA OPTIMIZACIÓN
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # Prueba con 50 o 100 intentos

    print("\n Mejores parámetros encontrados:", study.best_params)
    
    # 5. ENTRENAR MODELO FINAL CON LOS MEJORES PARÁMETROS
    modelo_final = lgb.LGBMRegressor(**study.best_params, n_estimators=2000)
    modelo_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=100)])
    
    return modelo_final, X_test, y_test

def entrenar_modelo_precio_profesional(df, target_col='log_original_price'):
    """
    Entrena usando Train/Val/Test para evitar data leakage por early stopping.
    """
    with mlflow.start_run(nested=True):
        print("--- 1. PREPARACIÓN DE DATOS ---")
        
        # 1. Limpieza de columnas innecesarias
        cols_a_eliminar = [target_col, 'original_row_id', 'error_log', 'original_title', 'price']
        existing_cols = [c for c in cols_a_eliminar if c in df.columns]
        X = df.drop(columns=existing_cols)
        y = df[target_col]

        # 2. Categorizar variables de texto
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype('category')

        # --- 3. SPLIT DE 3 VÍAS (LA CLAVE) ---
        # Paso A: Separar el TEST FINAL (20%). Este es SAGRADO.
        # Usamos random_state=42 para que coincida con tus otros modelos.
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Paso B: Del 80% restante (Dev), sacamos un trozo para VALIDACIÓN (Early Stopping).
        # 0.125 de 0.80 es igual a 0.10 (10% del total).
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.125, random_state=42)
        
        print(f"📊 Dimensiones del Split:")
        print(f"   -> Train (Aprender):      {X_train.shape[0]} filas")
        print(f"   -> Val   (Early Stop):    {X_val.shape[0]} filas")
        print(f"   -> Test  (Eval Final):    {X_test.shape[0]} filas (Intocable)")

        # 4. Configuración LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.0678632566746439,
            'num_leaves': 251,
            'max_depth': 7, 'feature_fraction': 0.9339257464617899,
            'bagging_fraction': 0.7508640428820034,
            'bagging_freq': 5,
            'min_child_samples': 8,
            'verbosity': -1,
            'random_state': 42
        }

        model = lgb.LGBMRegressor(**params, n_estimators=2000)
        
        # 5. Entrenamiento
        print("\n🚀 Entrenando modelo...")
        # AQUÍ ESTÁ EL CAMBIO: Entrenamos con TRAIN, pero miramos VAL para parar.
        # El TEST ni lo toca.
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], # <--- Usamos Validación, NO Test
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        # 6. Predicción en escala LOG sobre TEST (Datos nuevos)
        preds_log = model.predict(X_test)

        # --- TRANSFORMACIÓN A EUROS ---
        y_test_eur = np.expm1(y_test)
        preds_eur = np.expm1(preds_log)

        # 7. Métricas en Euros
        mae_eur = mean_absolute_error(y_test_eur, preds_eur)
        r2_eur = r2_score(y_test_eur, preds_eur)
        rmse_eur = np.sqrt(mean_squared_error(y_test_eur, preds_eur))
        # rmse = np.sqrt(mean_squared_error(y_val, preds_eur))


        print(f"\n---  Resultados Honestos (Sobre Set de Test) ---")
        print(f"MAE:  {mae_eur:.2f}€")
        print(f"RMSE: {rmse_eur:.2f}€")
        print(f"R2:   {r2_eur:.4f}")

        # Loggear métricas
        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae_eur)
        # mlflow.log_metric("val_rmse_log", rmse)
        mlflow.log_metric("rmse_euro", rmse_eur)
        mlflow.log_metric("val_r2_euro", r2_eur)

        # Suponiendo que tu modelo se llama 'mi_modelo'
        # nombre_archivo = 'modelo_lightgbm.pkl'

        # with open(nombre_archivo, 'wb') as archivo:
        #     pickle.dump(model, archivo)

        # print("¡Modelo guardado con éxito!")
        
    
        return model, X_test

# EJECUTAMOS LA FUNCIÓN
# if _name_ == "_main_":
    # Asegúrate de poner la ruta correcta a tu CSV
csv_path = "../../../Datasets/evaluacion4.csv"
print(f" Cargando {csv_path}...")
df = pd.read_csv(csv_path)
# mejor_modelo, X_test_final, y_test_final = optimizar_con_optuna(df)
modelo_final, datos_test = entrenar_modelo_precio_profesional(df, target_col='log_original_price')