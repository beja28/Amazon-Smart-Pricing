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
    print("\n🔍 INICIANDO BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA...")
    
    # --- 1. PREPARACIÓN (Mismas columnas que en entrenamiento) ---
    cols_a_eliminar = [target_col, 'original_row_id', 'error_log', 'original_title', 'price']
    existing_cols = [c for c in cols_a_eliminar if c in df.columns]
    X = df.drop(columns=existing_cols)
    y = df[target_col]

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].astype('category')

    # Split de 3 vías (Igual que en evaluación final)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.125, random_state=42)

    def objective(trial):
        # 2. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS (AJUSTADO PARA 6.5k ROWS)
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            
            # Tasa de aprendizaje (usamos escala logarítmica para buscar mejor)
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            
            # Control de Complejidad (Reducido para evitar overfitting)
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'num_leaves': trial.suggest_int('num_leaves', 15, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 80),
            
            # Aleatoriedad para robustez
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            
            # REGULARIZACIÓN (NUEVO: Vital para datasets pequeños)
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),  # L1
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True) # L2
        }

        # Entrenamiento rápido para Optuna
        gbm = lgb.LGBMRegressor(**param, n_estimators=1000)
        gbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] # Reducido a 30 para agilizar Optuna
        )

        preds = gbm.predict(X_val)
        mae_val = mean_absolute_error(np.expm1(y_val), np.expm1(preds))
        
        return mae_val

    # 4. EJECUTAR LA OPTIMIZACIÓN
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30) # 30 intentos suele ser un buen balance

    print("\n✅ Mejores parámetros encontrados:", study.best_params)
    
    # IMPORTANTE: Ahora solo devolvemos el diccionario de parámetros. 
    # El entrenamiento final lo hará la otra función para que quede logueado en MLflow.
    return study.best_params


def entrenar_modelo_precio_profesional(df, best_params, target_col='log_original_price'):
    """
    Entrena usando Train/Val/Test y registra en MLflow usando los parámetros de Optuna.
    """
    with mlflow.start_run(run_name="LGBM_Optimizado_Final"):
        print("\n--- 1. PREPARACIÓN DE DATOS (FINAL) ---")
        
        # 1. Limpieza de columnas
        cols_a_eliminar = [target_col, 'original_row_id', 'error_log', 'original_title', 'price']
        existing_cols = [c for c in cols_a_eliminar if c in df.columns]
        X = df.drop(columns=existing_cols)
        y = df[target_col]

        # 2. Categorizar variables de texto
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype('category')

        # 3. SPLIT DE 3 VÍAS
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.125, random_state=42)
        
        print(f"📊 Dimensiones del Split:")
        print(f"   -> Train (Aprender):      {X_train.shape[0]} filas")
        print(f"   -> Val   (Early Stop):    {X_val.shape[0]} filas")
        print(f"   -> Test  (Eval Final):    {X_test.shape[0]} filas (Intocable)")

        # 4. CONFIGURAR PARÁMETROS (INYECTAR OPTUNA)
        # Tomamos los parámetros de Optuna y añadimos los fijos necesarios
        params = best_params.copy()
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42
        })

        model = lgb.LGBMRegressor(**params, n_estimators=2000)
        
        # En tu script de entrenamiento original
        print(X_train.dtypes)
        
        # 5. Entrenamiento Final
        print("\n🚀 Entrenando modelo final con mejores parámetros...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], 
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        # 6. Predicción en TEST (Datos nuevos)
        preds_log = model.predict(X_test)

        # --- TRANSFORMACIÓN A EUROS ---
        y_test_eur = np.expm1(y_test)
        preds_eur = np.expm1(preds_log)

        # 7. Métricas en Euros
        mae_eur = mean_absolute_error(y_test_eur, preds_eur)
        r2_eur = r2_score(y_test_eur, preds_eur)
        rmse_eur = np.sqrt(mean_squared_error(y_test_eur, preds_eur))

        print(f"\n--- 🏆 Resultados Honestos (Sobre Set de Test) ---")
        print(f"MAE:  {mae_eur:.2f}€")
        print(f"RMSE: {rmse_eur:.2f}€")
        print(f"R2:   {r2_eur:.4f}")

        # Loggear métricas en MLflow
        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae_eur)
        mlflow.log_metric("rmse_euro", rmse_eur)
        mlflow.log_metric("val_r2_euro", r2_eur)

        # Guardado del modelo
        nombre_archivo = 'modelo_general_lightgbm.pkl'
        with open(nombre_archivo, 'wb') as archivo:
             pickle.dump(model, archivo)

        print(f"💾 ¡Modelo guardado con éxito en '{nombre_archivo}'!")
        
        return model, X_test

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    csv_path = "../../../Datasets/evaluacion4.csv"
    print(f"📂 Cargando {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Paso 1: Sacar los mejores parámetros con Optuna
    mejores_parametros = optimizar_con_optuna(df, target_col='log_original_price')
    
    # Paso 2: Pasarle esos parámetros al entrenamiento profesional
    modelo_final, datos_test = entrenar_modelo_precio_profesional(df, mejores_parametros, target_col='log_original_price')