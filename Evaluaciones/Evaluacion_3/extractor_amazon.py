import pandas as pd
import time
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
from tqdm import tqdm

# --- 1. CONFIGURACIÓN ---
# Reemplaza con tu API KEY real
API_KEY = "pedirmela (Diego)" 
INPUT_FILE = "ev3_productos.csv"  
OUTPUT_FILE = "amazon_specs_enriched.csv"
CHECKPOINT_INTERVAL = 50  # Guarda cada 50 productos

client = OpenAI(api_key=API_KEY)

# --- 2. EL SYSTEM PROMPT REFORZADO ---
SYSTEM_PROMPT = """
Eres un Analista de Datos Experto para Amazon. Tu trabajo es extraer especificaciones estructuradas de títulos de productos para un modelo XGBoost. La pureza del dato es tu prioridad absoluta.

REGLAS MAESTRAS DE CLASIFICACIÓN (Jerarquía Crítica):
1. DESKTOPS, WORKSTATIONS & SERVERS: Computadoras completas, Mini-PCs, Servidores NAS y Kits de IA (NVIDIA Jetson).
2. CAMERAS & PHOTOGRAPHY: Cámaras, Drones, Gimbals, Prismáticos, Luces profesionales y Monitores de campo.
3. SMARTPHONES & WEARABLES: SOLO teléfonos móviles y relojes/pulseras inteligentes. EXCLUIR: Rastreadores (AirTags/Tile), teléfonos fijos y GPS de mano.
4. NETWORKING & SMART HOME: Infraestructura (Routers, Mesh, Switches), Navegación (GPS mano/náutico, Sonares) y Seguridad (Cámaras, Timbres, Cerraduras).
5. DISPLAYS & MOUNTING: Monitores de PC, Smart TVs, Proyectores y Soportes/Racks de pared.
6. AUDIO & VIDEO EQUIPMENT: Auriculares, Altavoces, Soundbars, Reproductores Blu-ray, Mezcladores y Micrófonos.
7. PC COMPONENTS (CORE): Piezas INTERNAS (CPU, GPU, RAM, Placas Base, Fuentes, Discos NVMe/SATA internos, Refrigeración).
8. PERIPHERALS & INPUT DEVICES: Hardware externo ACTIVO: Impresoras, Escáneres, SAIs/UPS, Teclados, Ratones, Volantes y Docks.
9. ACCESSORIES & CONSUMABLES: Objetos PASIVOS o auxiliares: Tinta, Papel, Cables, Fundas, Cargadores, Lámparas, Pilas, Rastreadores (AirTag, SmartTag, Tile) y Teléfonos fijos.

REGLAS DE EXTRACCIÓN TÉCNICA (Features para ML):
1. PRECISIÓN DE ALMACENAMIENTO: Sé exacto. 1TB = 1024.0. Si hay dos discos, súmalos.
2. PROTECCIÓN DE BATERÍAS: Queda PROHIBIDO extraer 'storage_gb' de Power Banks o pilas. Los mAh NO son Gigabytes.
3. RESOLUCIÓN: Extrae solo si el producto tiene pantalla propia.
   - MAPEADO: 'Retina', 'Liquid Retina' o 'XDR' -> '2K/QHD (1440p)' o '4K/UHD (2160p)'.
   - EXCLUSIÓN: Ignóralo en tarjetas SD, cables, webcams, auriculares o AirTags.
4. PACK COUNT: Si el título indica un pack (ej. 2-Pack, 24-Count), extrae el número exacto.
5. MARKET TIER: Productos >500€ o marcas líderes (Apple, DJI, RTX 50xx, Garmin Fenix, Netgear Orbi) son 'Premium'.
6. REGLA DE COMPATIBILIDAD (CRÍTICA): Si el título dice "Compatible with iPad", "for iPhone", o "Setup with Galaxy", el producto es el ACCESORIO, no el dispositivo principal. NO extraigas specs (RAM, Pantalla) del dispositivo mencionado como compatible.
"""

# --- 3. ESQUEMA PYDANTIC COMPLETO ---
class AmazonFinalSpecs(BaseModel):
    # 1. Taxonomía de 12 categorías
    category: Literal[
        "Smartphones & Wearables", "Tablets & E-Readers", "Laptops & Chromebooks",
        "Desktops, Workstations & Servers", "PC Components (Core)", "Peripherals & Input Devices",
        "Displays & Mounting", "Audio & Video Equipment", "Cameras & Photography",
        "Networking & Smart Home", "Gaming Systems & VR", "Accessories & Consumables"
    ]

    # 2. Anclas de Valor
    market_tier: Literal["Budget", "Mainstream", "Premium", "Enterprise/Professional"]
    condition: Literal["New", "Renewed/Refurbished"]
    is_premium_brand: bool
    tech_generation: Literal["Cutting-Edge", "Current-Gen", "Last-Gen", "Legacy"]

    # 3. Especificaciones Densas
    ram_gb: Optional[float] = Field(None, description="RAM o VRAM del sistema en GB")
    storage_gb: Optional[float] = Field(None, description="Capacidad almacenamiento en GB")
    size_value: Optional[float] = Field(None, description="Pulgadas (Screens) o mm (Lentes/Audio)")
    performance_value: Optional[float] = Field(None, description="Hz (Monitor), DPI (Mouse), MP (Cámara), Read Speed (MB/s)")
    power_wattage: Optional[float] = Field(None, description="Watios (W)")

    # 4. Descriptores Categóricos
    resolution_standard: Optional[Literal["HD/HD+", "FHD (1080p)", "2K/QHD (1440p)", "4K/UHD (2160p)", "5K/8K+"]] = Field(
        None, description="Estándar de resolución visual. Solo para pantallas, cámaras (video), monitores o proyectores."
    )
    cpu_gpu_tier: Optional[str] = Field(None, description="Ej: i7, Ryzen 5, RTX 4060, M3")
    connectivity: Optional[str] = Field(None, description="WiFi 6, 5G, Bluetooth, Wired, PoE")
    
    brand: str
    pack_count: int = Field(1, description="Número de unidades en el paquete")
    confidence: float = Field(description="Confianza en la extracción (0-1)")

# --- 4. FUNCIÓN PRINCIPAL ---
def process_dataset():
    # Cargar datos
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: No encuentro el archivo '{INPUT_FILE}'. Asegúrate de poner el nombre correcto.")
        return

    # Leer CSV o Excel (detecta extensión)
    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
        
    print(f"✅ Dataset cargado: {len(df)} productos.")

    # Lógica de reanudación (Resume)
    start_index = 0
    results = []
    
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        start_index = len(df_existing)
        # Convertimos a lista de diccionarios para seguir añadiendo
        results = df_existing.to_dict('records')
        print(f"🔄 Archivo de progreso detectado. Reanudando desde la fila {start_index}...")
    
    if start_index >= len(df):
        print("🎉 ¡El proceso ya estaba completado!")
        return

    # Bucle de procesamiento
    print("🚀 Iniciando extracción masiva...")
    
    for i in tqdm(range(start_index, len(df)), initial=start_index, total=len(df)):
        row = df.iloc[i]
        # Asegúrate de que la columna se llame 'product_title' o cámbialo aquí
        title = row.get('product_title', row.get('title', '')) 
        
        # Saltamos filas vacías si las hay
        if not isinstance(title, str) or len(title) < 2:
            results.append({"original_title": str(title), "confidence": 0.0, "category": "Accessories & Consumables"}) # Fallback seguro
            continue

        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analiza este título: {title}"}
                ],
                response_format=AmazonFinalSpecs,
            )
            
            extracted_data = completion.choices[0].message.parsed.dict()
            extracted_data['original_title'] = title 
            extracted_data['original_row_id'] = i # Para trazar vuelta al original
            results.append(extracted_data)

        except Exception as e:
            # Error de API o Parseo
            # print(f"⚠️ Error en fila {i}: {e}") # Descomenta si quieres ver cada error
            error_record = {
                "original_title": title, 
                "confidence": 0.0, 
                "category": "ERROR_API_FAIL", 
                "error_log": str(e)
            }
            results.append(error_record)
            time.sleep(1) # Pequeña pausa si hay error

        # Guardado incremental (Checkpoint)
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    
    # Guardado final
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ ¡Proceso completado! Datos guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()