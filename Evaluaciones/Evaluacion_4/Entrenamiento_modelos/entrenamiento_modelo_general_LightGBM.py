import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import mlflow
import optuna
import warnings
import pickle

warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "Amazon_Global_Pricing_LIGHTGBM"
mlflow.set_experiment(EXPERIMENT_NAME)


def optimizar_con_optuna(df, target_col='log_original_price'):
    print("\nINICIANDO BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA...")
    
    cols_a_eliminar = [target_col, 'original_row_id', 'error_log', 'original_title', 'price']
    existing_cols = [c for c in cols_a_eliminar if c in df.columns]
    X = df.drop(columns=existing_cols)
    y = df[target_col]

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].astype('category')

    # Split de 3 vías
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.125, random_state=42)

    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'num_leaves': trial.suggest_int('num_leaves', 15, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 80),
            
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),  # L1
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True) # L2
        }

        gbm = lgb.LGBMRegressor(**param, n_estimators=1000)
        gbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] 
        )

        preds = gbm.predict(X_val)
        mae_val = mean_absolute_error(np.expm1(y_val), np.expm1(preds))
        
        return mae_val

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30) 

    print("\nMejores parámetros encontrados:", study.best_params)
    
    # solo devolvemos el diccionario de parámetros. 
    return study.best_params


def entrenar_modelo_precio_profesional(df, best_params, target_col='log_original_price'):
    """
    Entrena usando Train/Val/Test y registra en MLflow usando los parámetros de Optuna.
    """
    with mlflow.start_run(run_name="LGBM_Optimizado_Final"):
        print("\nPREPARACIÓN DE DATOS")
        
        cols_a_eliminar = [target_col, 'original_row_id', 'error_log', 'original_title', 'price']
        existing_cols = [c for c in cols_a_eliminar if c in df.columns]
        X = df.drop(columns=existing_cols)
        y = df[target_col]

        # Categorizar variables de texto
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype('category')

        # SPLIT DE 3 VÍAS
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.125, random_state=42)
        
        print(f"Dimensiones del Split:")
        print(f"   -> Train:      {X_train.shape[0]} filas")
        print(f"   -> Val:    {X_val.shape[0]} filas")
        print(f"   -> Test:    {X_test.shape[0]} filas")

        # parámetros de Optuna y añadimos los fijos necesarios
        params = best_params.copy()
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42
        })

        model = lgb.LGBMRegressor(**params, n_estimators=2000)
        
        print("\nEntrenando modelo final con mejores parámetros")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], 
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        # 6. Predicción en TEST
        preds_log = model.predict(X_test)

        #TRANSFORMACIÓN A EUROS
        y_test_eur = np.expm1(y_test)
        preds_eur = np.expm1(preds_log)

        mae_eur = mean_absolute_error(y_test_eur, preds_eur)
        r2_eur = r2_score(y_test_eur, preds_eur)
        rmse_eur = np.sqrt(mean_squared_error(y_test_eur, preds_eur))

        print(f"\nResultados sobre Set de Test")
        print(f"MAE:  {mae_eur:.2f}€")
        print(f"RMSE: {rmse_eur:.2f}€")
        print(f"R2:   {r2_eur:.4f}")

        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae_eur)
        mlflow.log_metric("rmse_euro", rmse_eur)
        mlflow.log_metric("val_r2_euro", r2_eur)

        nombre_archivo = 'modelo_general_lightgbm.pkl'
        with open(nombre_archivo, 'wb') as archivo:
             pickle.dump(model, archivo)

        print(f"Modelo guardado con éxito en '{nombre_archivo}'")
        
        return model, X_test

if __name__ == "__main__":
    csv_path = "../../../Datasets/evaluacion4.csv"
    print(f"Cargando {csv_path}")
    df = pd.read_csv(csv_path)
    
    mejores_parametros = optimizar_con_optuna(df, target_col='log_original_price')
    
    modelo_final, datos_test = entrenar_modelo_precio_profesional(df, mejores_parametros, target_col='log_original_price')