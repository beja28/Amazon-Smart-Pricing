import pandas as pd
import time
import os
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field
from typing import Literal, Optional
from tqdm import tqdm
import random

# --- 1. CONFIGURACIÓN ---
# Reemplaza con tu API KEY real
API_KEY = "" 
INPUT_FILE = "ev3_productos.csv"  
OUTPUT_FILE = "amazon_specs_enriched_final.csv"
CHECKPOINT_INTERVAL = 10  # Guarda cada 10 productos

client = OpenAI(api_key=API_KEY)

# --- 2. EL SYSTEM PROMPT REFORZADO ---
SYSTEM_PROMPT = """ 
You are a Data Analyst for Amazon. Your job is to extract structured specifications 
from product titles for a prediction model (CatBoost). Data purity is your absolute priority. 

TASK 1 (CRITICAL): Assign ONE category among these 11: 
1) Accessories
2) Office, Printing & Power 
3) Audio & Media Systems 
4) Peripherals & Input 
5) Networking & Smart Home 
6) Displays & Mounting 
7) Cameras, Photography & Video 
8) PC Components (Core) 
9) Computers & Gaming 
10) Mobile Devices 
11) Out of Scope 

TASK 2: Place each product in its corresponding category, explicitly following the MASTER CATEGORIZATION RULES.

======================================================== 
MASTER CATEGORIZATION RULES (DECISION ORDER) 
======================================================== 

STEP 0 — Out of Scope (ALWAYS FIRST): 
If the product belongs to these families, it MUST be "Out of Scope" (even if it has WiFi/Bluetooth):
- MARINE / FISHING / NAUTICAL: Sonar, fish finder, transducer, marine chartplotter, livescope, echomap, Humminbird, Lowrance, Navionics. 
- AUTOMOTIVE / ROAD: Radar detector, laser detector, car stereo head unit, CarPlay receiver, OTR truck navigator, driving coach/optimizer, dashcam bundles focused on driving. 
- PROFESSIONAL RADIO / SCANNER: Two-way business radio, handheld scanner (police scanner), SDS100, Motorola business radios, professional walkie-talkies. 
- INSTRUMENTS: Guitars, pianos, musical amplifiers, pedals.
- APPLIANCES / HEALTH: Coffee makers, air fryers, shavers, electric toothbrushes, blood pressure monitors, household fans.
- TOYS & PETS: Non-electronic toys, LEGO, accessories for dogs/cats.
- FURNITURE: Gaming chairs, desks, tables.
- All types of backpacks, Sleeve Bags, or handbags go here.
- Microscopes and telescopes go here.
- Other “services” or non-products: Protection plans, warranty plans, subscriptions/plans. 
If there is a real doubt between a category and Out of Scope, choose Out of Scope with low-to-medium confidence. 

STEP 1 — Mobile Devices: 
Main devices with a mobile screen/OS: Smartphones, tablets, e-readers, smartwatches, fitness bands. 
GPS Navigators (for cars, drones, and any electronic product) should go here.
EXCLUDE: Accessories “for iPhone/iPad/Galaxy” (these go to Accessories). 

STEP 2 — Computers & Gaming: 
Complete computers and gaming/VR: 
- Laptops, desktops, mini PCs, all-in-ones, workstations, servers, NAS/Storage appliances (Synology, DiskStation). 
- Consoles (Switch/Xbox/PlayStation), VR headsets (Quest/PSVR), handheld gaming devices (ROG Ally). 

STEP 3 — PC Components (Core): 
INTERNAL PC parts: CPU, GPU, RAM, internal SSD/HDD, motherboards, PSU, cooling systems, case fans, thermal paste. 
Include VRAM if it comes with the GPU. 
INCLUDE: Everything that goes inside the case, even if inexpensive (thermal paste, internal brackets, case fans, internal SATA cables).
EXCLUDE: Docks, hubs, cables, external enclosures -> Accessories / Peripherals depending on the case. 

STEP 4 — Cameras, Photography & Video: 
“Photo/video” cameras and their ecosystem: 
- Mirrorless/DSLR, lenses, flashes, tripods, gimbals, photo backgrounds/sets, drones, action cams, webcams.
INCLUDE: Home security cameras (Nest/Blink/Tapo).
INCLUDE: Video conferencing equipment (meeting bars, video bars, conference kits, room controllers). 

STEP 5 — Audio & Media Systems: 
Home Audio/AV and conferencing: 
- Headphones, speakers, soundbars, receivers, subwoofers, microphones, speakerphones, audio interfaces (Focusrite), Blu-ray players.

STEP 6 — Networking & Smart Home (EXCLUDING UPS/sonar/radar): 
- Routers, mesh systems, switches, PoE injectors, sensors, door/window alarms, video doorbells, etc.
Explicitly EXCLUDE: UPS/SAI, heavy racks, sonars/radars (see Out of Scope), TV antennas if there is no smart-home functionality (if doubtful, Accessories/Displays depending on the accessory type). 

STEP 7 — Office, Printing & Power: 
- Printers/plotters/scanners/label printers + consumables (ink/toner/paper/printhead). 
- Office: Shredders, laminating pouches, calculators (TI-84 etc.), clearly “office” supplies. 
- Power/Infra: UPS/SAI, professional surge protectors, power distribution boxes, IT rack elements (racks, rails, shelves) if they appear.
- Batteries: Any replacement battery or power accumulator goes here, NOT in Accessories.
- Chargers: Any charger for electronic watches, mobile phones, tablets, etc.

STEP 8 — Displays & Mounting: 
- TVs, monitors, projectors, signage/commercial displays. 
- VESA supports/arms/mounts, stands, brackets, mounts (TV/monitor/projector). 
EXCLUDE: Generic cables (HDMI/DP) and TV Remote Controls -> Accessories, unless the title is clearly “mount/stand/arm”. 

STEP 9 — Peripherals & Input (EXCLUDING printers/UPS): 
- Keyboards, mice, trackpads, mousepads, drawing tablets, retail-use barcode scanners (if NOT a “police scanner”).
- Active PC Accessories (e.g., USB audio adapters, external fans, etc.) when they are not “just a cable”. Remember, this applies as long as it is external. 
- Active Connectivity: Hubs/docks if they are the primary peripheral, video capture cards, KVM switches, and external storage devices (USB Flash Drives, SD cards, MicroSD, external drives).

STEP 10 — Accessories (Clean): 
PASSIVE or auxiliary accessories, typically low-to-medium price: 
- VERY IMPORTANT: All types of cables (HDMI/DP/USB/RCA/XLR), audio cables, ethernet, coaxial, filters for TV antennas, cam plugs, power cords, etc.
- Adapters (wifi, ethernet...), cases (for phones, keyboards...), sleeves, straps, screen protectors, small mounts (phone mount), simple remotes, Remote Controllers, SD adapters, etc. 
- Holders, phone grips, chargers (for watches or anything else), electronic pencils, and Styluses.
EXCLUDE: Printing consumables (these go to Office, Printing & Power). 
EXCLUDE: “for iPhone/iPad/Watch” (these remain Accessories, NOT Mobile Devices).
CRITICAL EXCLUSION LIST (DO NOT PLACE HERE):
- NO Storage: SD, MicroSD, USB Sticks (these go to Peripherals).
- NO Active Power: Power Banks, chargers (these go to Office/Power).
- NO Components: Internal fans or thermal paste (these go to PC Components).
- NO Chip Converters: Adapters that change the signal (e.g., HDMI to VGA, USB to HDMI) go to Peripherals.

======================================================== 
TECHNICAL EXTRACTION RULES (features) 
======================================================== 
1) STORAGE PRECISION: 1TB = 1024.0. If there are two drives, sum them up. 
2) FORBIDDEN: Do not extract 'storage_gb' from batteries/power banks: mAh is NOT GB. 
3) RESOLUTION: Only if the product has its own screen or is a camera (video). 
4) PACK COUNT: Extract the exact number if a pack is indicated (2-pack, 6-pack, etc.). 
5) MARKET TIER (CATEGORICAL):
Select ONLY ONE based on the following hierarchy (Top-down):
- Enterprise/Professional: High-end business/infrastructure equipment. Keywords: "Enterprise", "Managed", "Rackmount", "Workstation", "Business Class", "Commercial", "Smart-UPS X", "Pro Printing".
- Premium: High-end consumer/flagship products. Price > 500 EUR OR flagship series (e.g., Apple, DJI, NVIDIA RTX 50xx, Garmin Fenix/Descent, Netgear Orbi). 
- Budget: Explicitly "Entry-level", "Basic", "Value", or low-cost generic brands.
- Mainstream: Standard consumer products. This is the DEFAULT if the product doesn't fit the other tiers.
6) COMPATIBILITY RULE (CRITICAL): 
If it says "for iPhone/iPad/Galaxy", "compatible with", "works with", the product is the ACCESSORY. 
DO NOT extract specs from the device mentioned as compatible. 

======================================================== 
TECHNICAL EXTRACTION RULES (features by subtype) 
======================================================== 

GLOBAL FIELD:
- subtype: Assign one specific value from the allowed list within each category.

### 1. Office, Printing & Power
Allowed Subtypes: printer, plotter, scanner, label_printer, toner_ink, ups, rack_component, shredder, calculator, battery, charger, surge_protector.

* Printers/Scanners (printer, plotter, scanner, label_printer): Extract printer_tech (Laser, Inkjet, Tank, Thermal), is_color (Bool), is_multifunction (Bool), paper_size_max (A4, A3, Wide), and ppm.
* Consumables (toner_ink): Extract pack_count and is_xl (Bool).
* Energy/Infra (ups, rack_component, surge_protector): Extract ups_capacity_value, ups_capacity_unit (VA, Watt), is_rackmount (Bool), and rack_u.
* Power/Battery (battery, charger): Extract power_capacity_value, power_capacity_unit (mAh, Wh, W), and pd_wattage.

### 2. Audio & Media Systems
Allowed Subtypes: headphones, earbuds, speaker, soundbar, receiver, subwoofer, microphone, audio_interface, blu_ray_player, speakerphone.

* Output (soundbar, speaker, receiver, subwoofer): Extract audio_channels (e.g., 2.0, 2.1, 5.1, 7.1.2), total_power_w, has_dolby_atmos (Bool), and has_subwoofer (Bool).
* Personal (headphones, earbuds): Extract has_anc (Bool), form_factor (over-ear, on-ear, in-ear), and is_wireless (Bool).
* Pro/Recording (microphone, audio_interface, speakerphone): Extract mic_type (condenser, dynamic, lavalier), interface_type (USB, XLR), and audio_resolution.
* Media (blu_ray_player): Extract video_output_res (4K_UHD, 1080p).

### 3. Peripherals & Input
Allowed Subtypes: keyboard, mouse, trackpad, mousepad, drawing_tablet, barcode_scanner, audio_adapter, external_fan, dock_hub, video_capture_card, kvm_switch, external_drive, usb_flash_drive, sd_card.

* Input (keyboard, mouse, trackpad, drawing_tablet): Extract is_gaming (Bool), switch_type (Mechanical, Membrane, Optical), dpi (mice), has_screen (tablets), and is_wireless (Bool).
* Connectivity (dock_hub, kvm_switch, video_capture_card): Extract is_thunderbolt (Bool), port_count, max_resolution (4K60, 4K30, 1080p), and pd_wattage.
* External Storage (external_drive, usb_flash_drive, sd_card): Extract storage_type (SSD, HDD, NVMe, Flash), read_speed_mbs, and storage_gb.
* Retail/Other (barcode_scanner, audio_adapter, external_fan): Extract scan_engine (1D, 2D_QR) and interface_type (USB-A, USB-C, 3.5mm).

### 4. Networking & Smart Home
Allowed Subtypes: router, mesh_system, switch, poe_injector, smart_sensor, smart_alarm, video_doorbell, wifi_adapter, access_point, range_extender, hub_bridge.

* Infrastructure (router, mesh_system, access_point, wifi_adapter, range_extender): Extract wifi_standard (WiFi_5_AC to WiFi_7_BE), speed_class (AX1800, AX3000, BE19000), node_count, and frequency_bands (Dual-Band, Tri-Band, Quad-Band).
* Switches (switch, poe_injector): Extract port_count, has_poe (Bool), and multi_gig_ports (2.5G, 10G).
* Smart Home (video_doorbell, smart_sensor, smart_alarm, hub_bridge): Extract power_source (Battery, Wired), smart_protocol (WiFi, Zigbee, Z-Wave, Matter), and is_outdoor (Bool).

### 5. Displays & Mounting
Allowed Subtypes: tv, monitor, projector, digital_signage, mount_arm, mount_wall, mount_ceiling, mount_stand.

* Displays (tv, monitor, projector, digital_signage): Extract screen_size_in, resolution_standard (FHD, QHD, 4K, 8K), panel_tech (OLED, Mini-LED, QLED, LED, LCD), refresh_rate_hz, and is_smart (Bool).
* Mounting (mount_arm, mount_wall, mount_ceiling, mount_stand): Extract mount_type (wall, desk_arm, ceiling, floor_stand), adjustment_type (Fixed, Tilt, Full-Motion, Gas-Spring), num_screens, max_weight_kg, and is_heavy_duty (Bool).

### 6. Cameras, Photography & Video
Allowed Subtypes: mirrorless_dslr, cinema_camera, lens, drone, action_cam, security_cam, webcam, gimbal_tripod, video_conf_system, photo_flash.

* Cameras (mirrorless_dslr, cinema_camera, drone, action_cam): Extract sensor_format (Full-Frame, APS-C, Micro-Four-Thirds), max_video_res (8K, 6K, 4K, 1080p), is_body_only (Bool), mount_system (Canon_RF, Sony_E, Nikon_Z, L-Mount), and megapixels.
* Lenses (lens): Extract is_pro_line (Bool), focal_range (Prime, Zoom), and max_aperture_f.
* Support/Kits (gimbal_tripod, video_conf_system, photo_flash): Extract max_weight_kg and is_kit (Bool).

### 7. PC Components (Core)
Allowed Subtypes: cpu, gpu, ram, motherboard, ssd_internal, hdd_internal, psu, cpu_cooler, case_fan, thermal_paste.

* Processing (cpu, gpu): Extract component_series (RTX 4080, Ryzen 7, i9-14900K, etc.), vram_gb, and has_x3d (Bool).
* Memory/Storage (ram, ssd_internal, hdd_internal): Extract ram_gb, storage_gb, and tech_gen (DDR4, DDR5, Gen4, Gen5).
* PSU/Mobo (psu, motherboard): Extract power_wattage, efficiency_rating (80 Plus Gold, Platinum, Bronze), and chipset (Z790, B650, X670).
* Cooling/Other (cpu_cooler, case_fan, thermal_paste): Extract is_aio (Bool).

### 8. Computers & Gaming
Allowed Subtypes: laptop, desktop, mini_pc, all_in_one, workstation, server, nas, console, vr_headset, handheld_gaming.

* PC/Computing (laptop, desktop, mini_pc, all_in_one, workstation, server): Extract component_series (CPU+GPU), ram_gb, storage_gb, screen_size_in, and is_gaming (Bool).
* Entertainment (console, vr_headset, handheld_gaming): Extract platform_family (PlayStation_5, Xbox_Series, Nintendo_Switch, Meta_Quest, Steam_Deck/ROG_Ally) and is_digital_edition (Bool).
* Network Storage (nas): Extract nas_bays and storage_gb.

### 9. Mobile Devices
Allowed Subtypes: smartphone, tablet, e_reader, smartwatch, fitness_band, gps_navigator.

* Devices (smartphone, tablet, e_reader): Extract model_name, storage_gb, screen_size_in, cellular_type (WiFi, LTE/4G, 5G), and has_stylus (Bool).
* Wearables (smartwatch, fitness_band): Extract case_size_mm and is_rugged (Bool).
* Navigation (gps_navigator): Extract is_specialized (Bool), gps_activity (Cycling, Automotive, Hiking, Marine), and has_touchscreen (Bool).

### 10. Accessories (Clean)
Allowed Subtypes: cable, adapter, protection_case, screen_protector, stylus, remote_control, mount_small, strap_grip, charger_aux.

* Cables (cable): Extract cable_length_m, interface_type (identify specific protocol and version), and is_braided (Bool).
* Protection (protection_case, screen_protector, strap_grip): Extract material (Silicone, Leather, Plastic, Tempered_Glass) and brand_compatibility.
* Control/Input (stylus, remote_control): Extract is_active (Bool) and is_original (Bool).
* Connectivity/Power (adapter, mount_small, charger_aux): Extract power_wattage and connector_gender (M-to-M, M-to-F, F-to-F).

### 11. Out of Scope
Allowed Subtype: excluded.

* Rules: Assign subtype as excluded. Extract only brand and confidence. No specific technical features.

"""


# --- 2. ESQUEMA PYDANTIC (ADAPTADO A LAS NUEVAS REGLAS) ---
class AmazonFinalSpecs(BaseModel):
    # --- GLOBAL ATTRIBUTES ---
    category: Literal[
        "Accessories", "Office, Printing & Power", "Audio & Media Systems", 
        "Peripherals & Input", "Networking & Smart Home", "Displays & Mounting", 
        "Cameras, Photography & Video", "PC Components (Core)", 
        "Computers & Gaming", "Mobile Devices", "Out of Scope"
    ]
    subtype: str = Field(..., description="The specific subtype from the allowed list in each category (Allowed Subtypes). Use 'excluded' for Out of Scope.")
    market_tier: Literal["Budget", "Mainstream", "Premium", "Enterprise/Professional"]
    condition: Literal["New", "Renewed/Refurbished"]
    is_premium_brand: bool
    tech_generation: Literal["Cutting-Edge", "Current-Gen", "Last-Gen", "Legacy"]
    brand: str
    confidence: float = Field(..., description="Confidence score (0.0 - 1.0)")

    # --- 1. OFFICE, PRINTING & POWER ---
    printer_tech: Optional[Literal["Laser", "Inkjet", "Tank", "Thermal"]] = None
    is_color: Optional[bool] = None
    is_multifunction: Optional[bool] = None
    paper_size_max: Optional[Literal["A4", "A3", "Wide-Format/Plotter"]] = None
    ppm: Optional[float] = None
    pack_count: Optional[int] = Field(None, description="Critical for ink/toner packs")
    is_xl: Optional[bool] = None
    ups_capacity_value: Optional[float] = None
    ups_capacity_unit: Optional[Literal["VA", "Watt"]] = None
    is_rackmount: Optional[bool] = None
    rack_u: Optional[int] = None
    power_capacity_value: Optional[float] = None
    power_capacity_unit: Optional[Literal["mAh", "Wh", "W"]] = None

    # --- 2. AUDIO & MEDIA ---
    audio_channels: Optional[str] = Field(None, description="Ex: 2.1, 5.1, 7.1.2")
    total_power_w: Optional[float] = None
    has_dolby_atmos: Optional[bool] = None
    has_subwoofer: Optional[bool] = None
    has_anc: Optional[bool] = None
    form_factor: Optional[Literal["over-ear", "on-ear", "in-ear"]] = None
    mic_type: Optional[Literal["condenser", "dynamic", "lavalier"]] = None
    audio_resolution: Optional[str] = None
    video_output_res: Optional[Literal["4K_UHD", "1080p"]] = None

    # --- 3. PERIPHERALS & INPUT ---
    switch_type: Optional[Literal["Mechanical", "Membrane", "Optical"]] = None
    dpi: Optional[float] = None
    has_screen: Optional[bool] = None
    is_thunderbolt: Optional[bool] = None
    max_resolution: Optional[Literal["4K60", "4K30", "1080p"]] = None
    read_speed_mbs: Optional[float] = None
    scan_engine: Optional[Literal["1D", "2D_QR"]] = None
    
    # --- 4. NETWORKING & SMART HOME ---
    wifi_standard: Optional[Literal["WiFi_5_AC", "WiFi_6_AX", "WiFi_6E_AXE", "WiFi_7_BE"]] = None
    speed_class: Optional[str] = None
    node_count: Optional[int] = None
    frequency_bands: Optional[Literal["Dual-Band", "Tri-Band", "Quad-Band"]] = None
    has_poe: Optional[bool] = None
    multi_gig_ports: Optional[int] = None
    power_source: Optional[Literal["Battery", "Wired"]] = None
    smart_protocol: Optional[Literal["WiFi", "Zigbee", "Z-Wave", "Matter"]] = None
    is_outdoor: Optional[bool] = None

    # --- 5. DISPLAYS & MOUNTING ---
    screen_size_in: Optional[float] = None
    resolution_standard: Optional[Literal["FHD_1080p", "QHD_1440p", "4K_UHD", "8K"]] = None
    panel_tech: Optional[Literal["OLED", "QD-OLED", "Mini-LED", "QLED", "LED", "LCD"]] = None
    refresh_rate_hz: Optional[float] = None
    is_smart: Optional[bool] = None
    mount_type: Optional[Literal["wall", "desk_arm", "ceiling", "floor_stand"]] = None
    adjustment_type: Optional[Literal["Fixed", "Tilt", "Full-Motion", "Gas-Spring"]] = None
    num_screens: Optional[int] = None
    max_weight_kg: Optional[float] = None
    is_heavy_duty: Optional[bool] = None

    # --- 6. CAMERAS & VIDEO ---
    sensor_format: Optional[Literal["Full-Frame", "APS-C", "Micro-Four-Thirds"]] = None
    max_video_res: Optional[Literal["8K", "6K", "4K", "1080p"]] = None
    is_body_only: Optional[bool] = None
    mount_system: Optional[Literal["Canon_RF", "Sony_E", "Nikon_Z", "L-Mount", "Micro_43"]] = None
    megapixels: Optional[float] = None
    is_pro_line: Optional[bool] = None
    focal_range: Optional[Literal["Prime", "Zoom"]] = None
    max_aperture_f: Optional[float] = None
    is_kit: Optional[bool] = None

    # --- 7. PC COMPONENTS (CORE) ---
    component_series: Optional[str] = Field(None, description="Ex: RTX 4080, Ryzen 7, i5-13600K")
    vram_gb: Optional[float] = None
    has_x3d: Optional[bool] = None
    ram_gb: Optional[float] = None
    storage_gb: Optional[float] = None
    tech_gen: Optional[Literal["DDR4", "DDR5", "Gen3", "Gen4", "Gen5"]] = None
    power_wattage: Optional[float] = None
    efficiency_rating: Optional[Literal["80_Plus_Bronze", "80_Plus_Gold", "80_Plus_Platinum", "80_Plus_Titanium"]] = None
    chipset: Optional[str] = None
    is_aio: Optional[bool] = None

    # --- 8. COMPUTERS & GAMING ---
    is_gaming: Optional[bool] = None
    platform_family: Optional[Literal["PlayStation_5", "PlayStation_4", "Xbox_Series", "Xbox_One", "Nintendo_Switch", "Meta_Quest", "Steam_Deck/ROG_Ally"]] = None
    is_digital_edition: Optional[bool] = None
    nas_bays: Optional[int] = None

    # --- 9. MOBILE DEVICES ---
    model_name: Optional[str] = None
    cellular_type: Optional[Literal["WiFi", "LTE/4G", "5G"]] = None
    has_stylus: Optional[bool] = None
    case_size_mm: Optional[float] = None
    is_rugged: Optional[bool] = None
    is_specialized: Optional[bool] = None
    gps_activity: Optional[Literal["Cycling", "Automotive", "Hiking", "Marine"]] = None
    has_touchscreen: Optional[bool] = None

    # --- 10. ACCESSORIES ---
    cable_length_m: Optional[float] = None
    interface_type: Optional[str] = Field(None, description="Protocol + Version. Ex: HDMI 2.1, USB-C 3.2")
    is_braided: Optional[bool] = None
    material: Optional[Literal["Silicone", "Leather", "Plastic", "Tempered_Glass", "Metal"]] = None
    brand_compatibility: Optional[str] = None
    is_active: Optional[bool] = None
    is_original: Optional[bool] = None
    connector_gender: Optional[Literal["M-to-M", "M-to-F", "F-to-F"]] = None
    
    # Common reused fields
    is_wireless: Optional[bool] = None
    port_count: Optional[int] = None
    storage_type: Optional[Literal["SSD", "HDD", "NVMe", "Flash"]] = None
    pd_wattage: Optional[float] = None

# --- 3. FUNCIÓN DE LLAMADA API ---
def call_extraction_api_with_backoff(title: str, model: str = "gpt-4o-mini", max_retries: int = 5):
    """
    Intenta llamar a la API. Si da error de Rate Limit, espera y reintenta.
    """
    delay = 2  # Segundos iniciales de espera
    
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze product: {title}"}
                ],
                response_format=AmazonFinalSpecs,
                temperature=0,
            )
            return completion.choices[0].message.parsed
            
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            # Si es el último intento, lanzamos el error para que se guarde en el CSV
            if attempt == max_retries - 1:
                raise e
            
            # Cálculo de espera: Exponential Backoff + Jitter (ruido aleatorio)
            # El jitter es vital para que tus 4 PCs no golpeen la API a la vez de nuevo
            sleep_time = delay * (2 ** attempt) + random.uniform(0, 1)
            
            # Solo imprimimos si la espera es larga para no ensuciar la consola
            if sleep_time > 3:
                print(f"\n⏳ Rate Limit o Red detectado. Esperando {sleep_time:.1f}s antes de reintentar...")
            
            time.sleep(sleep_time)
            
        except Exception as e:
            # Si es otro error (ej: prompt inválido), no tiene sentido reintentar
            raise e

# --- 4. FUNCIÓN PRINCIPAL DE PROCESAMIENTO ---
def process_dataset():
    # 1. Cargar datos
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: No encuentro '{INPUT_FILE}'.")
        return

    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)
    
    print(f"✅ Dataset cargado: {len(df)} productos.")

    # 2. Lógica de reanudación (Resume)
    start_index = 0
    results = []
    
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        start_index = len(df_existing)
        results = df_existing.to_dict('records')
        print(f"🔄 Reanudando desde fila {start_index}...")
    
    if start_index >= len(df):
        print("🎉 ¡Proceso ya completado!")
        return

    # 3. Bucle principal
    print("🚀 Iniciando extracción inteligente...")
    
    for i in tqdm(range(start_index, len(df)), initial=start_index, total=len(df)):
            row = df.iloc[i]
            title = row.get('product_title', row.get('title', '')) 
            
            if not isinstance(title, str) or len(title) < 2:
                results.append({"original_title": str(title), "confidence": 0.0, "category": "ERROR_EMPTY"})
                continue

            try:
                # Usamos la nueva función con backoff
                extracted_obj = call_extraction_api_with_backoff(title, model="gpt-4o-mini")
                data = extracted_obj.dict()

                # Lógica de confianza (igual que antes)
                if data['confidence'] < 0.7:
                    tqdm.write(f"⚠️ Baja confianza ({data['confidence']:.2f}) -> Consultando GPT-4o")
                    try:
                        # También usamos backoff para el experto
                        expert_obj = call_extraction_api_with_backoff(title, model="gpt-4o")
                        data = expert_obj.dict()
                        data['model_used'] = "gpt-4o"
                    except Exception as e_expert:
                        tqdm.write(f"❌ Falló GPT-4o: {e_expert}")
                        data['model_used'] = "gpt-4o-mini" # Nos quedamos con el anterior
                else:
                    data['model_used'] = "gpt-4o-mini"

                data['original_title'] = title 
                data['original_row_id'] = i
                results.append(data)

            except Exception as e:
                # Este bloque solo se ejecuta si fallaron los 5 reintentos
                error_record = {
                    "original_title": title, 
                    "confidence": 0.0, 
                    "category": "ERROR_API_FAIL", 
                    "error_log": str(e)
                }
                results.append(error_record)

            # 4. Guardado incremental (Checkpoint)
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    
    # Guardado final
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ ¡Proceso finalizado! Datos en {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()