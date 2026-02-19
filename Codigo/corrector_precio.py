import numpy as np
import pickle

class CorrectorPorSimilitud:
    def __init__(self, ruta_pkl="motor_similitud_titulos.pkl"):
        # Cargamos el motor en memoria
        with open(ruta_pkl, 'rb') as f:
            datos = pickle.load(f)
            
        self.vectorizer = datos['vectorizer']
        self.knn = datos['knn']
        self.precios_log = datos['precios_log']
        self.titulos_reales = datos['titulos_reales']

    def corregir_prediccion(self, log_pred_modelo, row):
        titulo_entrada = str(row.get('original_title', ''))
        categoria = str(row.get('category', ''))
        
        # Si no hay título válido, confiamos 100% en la IA
        if not titulo_entrada or titulo_entrada == 'nan':
            return log_pred_modelo, "Sin título para comparar"

        # Convertimos el título a vector y buscamos vecinos
        vec_entrada = self.vectorizer.transform([titulo_entrada])
        distancias, indices = self.knn.kneighbors(vec_entrada)
        
        distancias = distancias[0]
        indices = indices[0]
        
        # ---------------------------------------------------------------------
        # 1. REGLAS POR CATEGORÍA (Modo Estricto vs Modo Flexible)
        # ---------------------------------------------------------------------
        if categoria in ['Computers & Gaming', 'Mobile Devices', 'PC Components (Core)']:
            UMBRAL_ESTRICTO = 0.20
            UMBRAL_EXTENDIDO = 0.35
        else:
            UMBRAL_ESTRICTO = 0.50
            UMBRAL_EXTENDIDO = 0.65

        precios_parecidos = []
        titulos_encontrados = []
        es_busqueda_extendida = False

        # Intento inicial: Filtro estricto
        for d, idx in zip(distancias, indices):
            if d < UMBRAL_ESTRICTO:
                precios_parecidos.append(self.precios_log[idx])
                titulos_encontrados.append(self.titulos_reales[idx])

        # Segunda oportunidad: Filtro extendido si no encontramos nada
        if len(precios_parecidos) == 0:
            for d, idx in zip(distancias, indices):
                if d < UMBRAL_EXTENDIDO:
                    precios_parecidos.append(self.precios_log[idx])
                    titulos_encontrados.append(self.titulos_reales[idx])
            if len(precios_parecidos) > 0:
                es_busqueda_extendida = True

        # Si seguimos sin encontrar nada, la IA manda sola
        if len(precios_parecidos) == 0:
            return log_pred_modelo, "Modelo IA (Sin equivalentes seguros)"

        # ---------------------------------------------------------------------
        # 2. CÁLCULO DE MERCADO Y FUERZA DE EMPUJE (Nudge)
        # ---------------------------------------------------------------------
        log_mercado = np.median(precios_parecidos)
        mejor_distancia = distancias[0]
        
        # Cuánto difiere la IA del mercado KNN
        diferencia_cruda = log_mercado - log_pred_modelo
        desviacion = abs(diferencia_cruda)

        # Asignamos la fuerza del "tirón" según lo idénticos que sean los títulos
        if es_busqueda_extendida:
            fuerza = 0.15  # Tira muy poquito (15% del camino)
            nivel = "Muy Suave"
        else:
            if mejor_distancia < 0.15:
                fuerza = 0.70  # Tira hasta la mitad del camino
                nivel = "Fuerte"
            elif mejor_distancia < 0.30:
                fuerza = 0.50  # Tira un tercio del camino
                nivel = "Moderada"
            else:
                fuerza = 0.30  # Tira un quinto del camino
                nivel = "Suave"

        empuje_calculado = diferencia_cruda * fuerza

        # ---------------------------------------------------------------------
        # 3. LA CORREA (LÍMITE MÁXIMO DE CORRECCIÓN)
        # ---------------------------------------------------------------------
        # Límite máximo que permitimos que el KNN altere el logaritmo original de la IA
        # 0.20 equivale a mover el precio un ±22% en euros como máximo.
        LIMITE_LOG = 0.4 
        
        # Recortamos el empuje para que nunca supere el límite por arriba o por abajo
        empuje_aplicado = max(min(empuje_calculado, LIMITE_LOG), -LIMITE_LOG)

        # ---------------------------------------------------------------------
        # 4. APLICACIÓN FINAL
        # ---------------------------------------------------------------------
        # Solo aplicamos el empuje si la desviación original valía la pena (> 0.10)
        if desviacion > 0.05: 
            log_corregido = log_pred_modelo + empuje_aplicado
            
            # Formateamos el texto para el Dashboard
            mejor_match_str = titulos_encontrados[0][:35] + "..."
            tope = " [LÍMITE ALCANZADO]" if abs(empuje_aplicado) == LIMITE_LOG else ""
            
            motivo = f"Nudge {nivel}: {empuje_aplicado:+.3f} log{tope}. Match: '{mejor_match_str}'"
            return log_corregido, motivo
            
        return log_pred_modelo, "Aprobado (IA alineada con mercado)"