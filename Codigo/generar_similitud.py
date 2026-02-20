import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def construir_motor_similitud(csv_path):
    print("🧠 Construyendo Motor de Similitud de Títulos...")
    df = pd.read_csv(csv_path)
    
    # Nos quedamos solo con las filas que tienen título y precio válido
    df = df.dropna(subset=['original_title', 'log_original_price'])
    
    # 1. Vectorizador: Convierte el texto del título en números ponderados
    # Ignoramos palabras comunes en inglés (stop_words)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_tfidf = vectorizer.fit_transform(df['original_title'])
    
    # 2. Modelo KNN: Encuentra los "vecinos más cercanos"
    # metric='cosine' es perfecta para texto. Mide el ángulo entre vectores, no la distancia pura.
    knn = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=1)
    knn.fit(X_tfidf)
    
    # 3. Guardamos los precios exactos correspondientes a cada índice
    precios_log = df['log_original_price'].values
    titulos_reales = df['original_title'].values # Guardamos esto para poder hacer debug
    
    # Guardamos todo en un paquete para la API
    datos_exportar = {
        'vectorizer': vectorizer,
        'knn': knn,
        'precios_log': precios_log,
        'titulos_reales': titulos_reales
    }
    
    with open('motor_similitud_titulos.pkl', 'wb') as f:
        pickle.dump(datos_exportar, f)
        
    print(f"✅ Motor guardado con {len(df)} productos de referencia en 'motor_similitud_titulos.pkl'")

# Pon aquí la ruta a tu dataset de entrenamiento (sin los 12 del test)
construir_motor_similitud("../Datasets/evaluacion4_produccion.csv")