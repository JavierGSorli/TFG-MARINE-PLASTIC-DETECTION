# TFG — Detección de residuos plásticos marinos en el Estrecho de Gibraltar
## Documento de contexto para Claude Code

---

## 1. Descripción del proyecto

**Título:** Detección de residuos plásticos mediante procesado de imágenes Sentinel-2 y Machine Learning  
**Grado:** Ciencia de Datos e Inteligencia Artificial — UPM (ETSII)  
**Autor:** Javier González Sorlí  
**Tutora:** Estíbaliz Martínez Izquierdo  

**Objetivo:** Evaluar y comparar distintos métodos de detección de residuos plásticos flotantes en imágenes Sentinel-2 usando como referencia un catálogo de detecciones reales en el Mediterráneo. La zona de estudio es el **Estrecho de Gibraltar y Mar de Alborán occidental**, definida por un polígono KML.

---

## 2. Entorno técnico

- **SO:** Windows 11
- **Python:** entorno conda llamado `marida`
- **Ruta base del proyecto:** `C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\`
- **Librerías principales:** rasterio, numpy, pandas, torch, torchvision, openeo, xarray, scipy, pyproj, scikit-learn, xgboost, matplotlib, Pillow, tkinter, requests

---

## 3. Estructura de carpetas

```
tfg-marine-plastic-detection/
│
├── data/
│   ├── windrows_nature/
│   │   ├── general/
│   │   │   └── 41467_2024_48674_MOESM3_ESM.xlsx   # CSV Nature (14374 LW)
│   │   └── detallado/11045944/
│   │       └── WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc  # NC Nature
│   ├── mapa_estrecho.kml                           # Polígono de la zona de estudio
│   └── marida/marine-debris.github.io/
│       ├── semantic_segmentation/unet/             # UNet MARIDA
│       │   ├── unet.py
│       │   ├── dataloader.py
│       │   ├── predict_mask.py                     # Script de predicción UNet (llamado via subprocess)
│       │   └── trained_models/44/model.pth         # Pesos UNet
│       ├── semantic_segmentation/random_forest/    # RF MARIDA
│       │   └── rf_classifier.joblib
│       └── multi-label/resnet/                     # ResNet MARIDA
│           ├── resnet.py
│           ├── predict_resnet.py                   # Script de predicción ResNet (llamado via subprocess)
│           └── trained_models/18/model.pth
│
├── results/auto/                                   # Todas las salidas del pipeline
│   ├── gibraltar_candidatos.csv                    # 876 candidatos filtrados
│   ├── positive_download_failures.csv              # Positivos que fallaron la descarga
│   ├── test_patches_final/                         # Patches descargados (SI + NO)
│   │   ├── YYYYMMDD_SI_NPXPATCH_IDX.tif            # Patch positivo
│   │   ├── YYYYMMDD_SI_NPXPATCH_IDX_mask.tif       # Máscara GT (del NC)
│   │   └── YYYYMMDD_NO_000000_IDX_LABEL.tif        # Patch negativo validado
│   ├── test_masks_unet/                            # Máscaras predichas por UNet
│   ├── test_masks_rf/                              # Máscaras predichas por RF
│   ├── test_resnet_json/                           # JSONs con probs ResNet
│   ├── test_indices/                               # Rasters FDI, NDVI, FDI+NDVI y máscaras
│   ├── predictions_master.csv                      # CSV maestro (salida 06)
│   ├── xgboost_dataset.csv                         # Dataset tabular (salida 07)
│   ├── xgboost_model/                              # Modelo, métricas y cv_results.csv XGBoost
│   ├── evaluation/                                 # Tabla comparativa y figuras
│   └── error_analysis/                             # Paneles de errores FP/FN
│
├── deliverables/tfg_pipeline_unificado/            # Pipeline canónico de entrega
│   ├── scripts/                                    # ← AQUÍ están todos los scripts activos
│   │   ├── config.py
│   │   ├── geo_utils.py
│   │   ├── pipeline_utils.py
│   │   ├── 00_explore_candidates.py
│   │   ├── 01_download_dataset.py
│   │   ├── 02_predict_unet.py
│   │   ├── 03_predict_rf.py
│   │   ├── 04_predict_resnet.py
│   │   ├── 05_predict_indices.py
│   │   ├── 06_unify_predictions.py
│   │   ├── 07_build_xgboost_dataset.py
│   │   ├── 08_train_xgboost.py
│   │   ├── 09_evaluate.py
│   │   ├── 10_error_analysis.py
│   │   └── 11_visualizacion.py
│   └── docs/REVIEW.md                             # Problemas detectados en versión anterior
│
└── notebooks/                                      # Scripts antiguos (NO usar — referencia histórica)
```

---

## 4. Datasets

### 4.1 CSV Nature (Cózar et al. 2024, Nature Communications)
- **Archivo:** `41467_2024_48674_MOESM3_ESM.xlsx`
- **Contenido:** 14.374 detecciones de Litter Windows (LW) en el Mediterráneo (2015–2021), revisadas por humanos
- **Columnas:** `Latitude`, `Longitude`, `Dec_time`, `Str_time`, `Year`, `Month`, `Day`, `Date`, `CodeT`, `Pixels per LW`, `Distance to land (m)`
- `CodeT` = hora de la imagen Sentinel-2 (ej: `T105629`)
- **Gibraltar filtrado:** 876 candidatos (`Pixels per LW >= 5` AND `Distance to land > 1280m` AND dentro del KML)
- **Guardado en:** `results/auto/gibraltar_candidatos.csv`

### 4.2 NC Nature (WASP catalogue — BEC, ICM-CSIC)
- **Archivo:** `WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc`
- **Variables clave:**
  - `lat_centroid`, `lon_centroid` — coordenadas del centroide de cada LW
  - `x_centroid`, `y_centroid` — posición en píxeles de la imagen L1C original
  - `n_pixels_fil` — número de píxeles del filamento
  - `pixel_x`, `pixel_y` — coordenadas de píxel de cada pixel del filamento (shape: 14374 × 2563)
  - `pixel_spec` — espectros de reflectancia de cada píxel (13 bandas L1C)
  - `s2_product` — nombre del producto Sentinel-2 usado
- **Uso:** genera las **máscaras de ground truth pixel a pixel** para los patches positivos
- La conexión con `gibraltar_candidatos.csv` es por `lat_centroid`/`lon_centroid` (coinciden exactamente)
- El CSV tiene una columna `nc_idx` que guarda el índice en el NC para cada candidato

### 4.3 Marine Pollution Bulletin 2025 (Ramos-Alcántara et al.)
- Suplementario S3: concentración in situ de plástico (items/m²) medida con redes de arrastre en Mediterráneo occidental
- **Uso planificado:** validación física — correlación Spearman entre predicciones y concentración real

---

## 5. Modelos implementados

### 5.1 UNet (MARIDA)
- **Tipo:** Segmentación semántica pixel a pixel
- **Input:** patch (11, 256, 256) float32 normalizado con `bands_mean`/`bands_std`
- **Output:** máscara de clases 1–11 (clase 1 = Marine Debris)
- **Pesos:** `unet/trained_models/44/model.pth`
- **Script de predicción:** `predict_mask.py` en `data/marida/…/unet/` (llamado via subprocess desde `02_predict_unet.py`)
- **Notas:** requiere pad a múltiplo de 32 antes de inferencia; NO usar `--auto_scale` con patches de OpenEO (ya en 0–1)

### 5.2 Random Forest (MARIDA)
- **Tipo:** Clasificación pixel a pixel con features espectrales + índices + GLCM
- **Script batch:** `03_predict_rf.py`
- **Modelo:** `rf_classifier.joblib`
- **Features:** 11 bandas + NDVI, FAI, FDI, SI, NDWI, NRD, NDMI, BSI + 6 features GLCM

### 5.3 ResNet (MARIDA)
- **Tipo:** Clasificación multi-label por patch completo
- **Output:** probabilidad de presencia de cada clase (no máscara, sino label por patch)
- **Script batch:** `04_predict_resnet.py` → llama via subprocess a `predict_resnet.py` en `data/marida/…/resnet/`
- **JSON de salida:** `{"probabilities": {"Marine Debris": float, ...}, "active_labels": [...], "threshold": float}`
- **Clase de interés:** `Marine Debris` (probabilidad continua 0–1)

### 5.4 Índices espectrales (baseline)
- **FDI (Floating Debris Index)** — Biermann et al. 2020:
  ```
  FDI = B08 - (B06 + (B11-B06) * ((842-665)/(1610-665)) * 10)
  ```
  Índices MARIDA: B08=idx7, B06=idx5, B11=idx9
- **NDVI:**
  ```
  NDVI = (B08 - B04) / (B08 + B04)
  ```
- **FDI+NDVI:** máscara combinada (AND lógico de ambas)
- **Umbral adaptativo:** media + 3*std del patch
- **Script:** `05_predict_indices.py` → genera rasters FDI, NDVI, FDI+NDVI y sus máscaras binarias

### 5.5 XGBoost (nuevo)
- **Tipo:** Clasificación binaria por patch (SI/NO plástico)
- **Features:** estadísticas espectrales por banda (media, std, p95) + stats de índices + scores de los otros modelos
- **CV:** Leave-One-Out (dataset pequeño) o StratifiedKFold(5)
- **Scripts:** `07_build_xgboost_dataset.py` → `08_train_xgboost.py`

---

## 6. Pipeline de descarga de patches

### Fuente de datos
- **Sentinel-2 L2A** via OpenEO (dataspace.copernicus.eu)
- Mismo usuario que Copernicus Browser
- Bandas en orden MARIDA exacto: `["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]`

### Formato de los patches
- **Shape:** (11, 256, 256)
- **Dtype:** float32
- **Rango:** 0–1 (DN/10000, solo si DN > 10)
- **CRS:** EPSG:32630 (UTM 30N)
- **Resolución:** 10m/píxel
- **Descarga:** bbox de 3.84×3.84 km → recorte central a 256×256

### Naming convention
```
YYYYMMDD_SI_NPXPATCH_IDX.tif       # positivos (hay plástico)
YYYYMMDD_SI_NPXPATCH_IDX_mask.tif  # máscara GT (píxeles plástico del NC)
YYYYMMDD_NO_000000_IDX_LABEL.tif   # negativos validados visualmente
                                    # LABEL = CLARO | DUDOSO | DIFICIL
```
`NPXPATCH` = número de píxeles del filamento que caen dentro del patch

### Generación de máscaras GT (ground truth pixel a pixel)
La función `build_mask` en `01_download_dataset.py`:
1. Proyecta el centroide del filamento (lat/lon del NC → UTM del patch) para obtener el píxel central
2. Calcula el desplazamiento relativo de cada píxel del filamento respecto al centroide en coordenadas de escena L1C
3. Aplica ese desplazamiento al píxel central del patch
4. Refina con `find_shift()`: correlación cruzada vía FFT entre la máscara proyectada y el FDI del patch
5. El nombre del archivo incluye el número de píxeles que realmente caen en el patch

**Nota importante:** el NC usa L1C y coordenadas de píxel en el sistema de la imagen original. El patch descargado es L2A en UTM. El desplazamiento entre ambos sistemas se compensa con `find_shift()` usando el FDI como referencia espectral.

### Validación de negativos
Los patches negativos se muestran en una ventana tkinter con 4 botones:
- **NO** → descartar, buscar otra coordenada
- **DUDOSO** → aceptar con etiqueta `DUDOSO`
- **CLARO** → aceptar con etiqueta `CLARO`
- **DIFÍCIL** → aceptar con etiqueta `DIFICIL`

---

## 7. Parámetros MARIDA (normalización)

```python
bands_mean = [0.05197577, 0.04783991, 0.04056812, 0.03163572,
              0.02972606, 0.03457443, 0.03875053, 0.03436435,
              0.0392113,  0.02358126, 0.01588816]

bands_std  = [0.04725893, 0.04743808, 0.04699043, 0.04967381,
              0.04946782, 0.06458357, 0.07594915, 0.07120246,
              0.08251058, 0.05111466, 0.03524419]
```

### Clases MARIDA
```
1  = Marine Debris         7  = Marine Water
2  = Dense Sargassum       8  = Sediment-Laden Water
3  = Sparse Sargassum      9  = Foam
4  = Natural Organic Mat.  10 = Turbid Water
5  = Ship                  11 = Shallow Water
6  = Clouds
```

---

## 8. Scripts del pipeline — descripción y estado

Todos los scripts activos están en `deliverables/tfg_pipeline_unificado/scripts/`.

| Script | Función | Estado |
|--------|---------|--------|
| `config.py` | Rutas y parámetros centralizados (auto-detecta raíz via `.git`) | ✅ |
| `geo_utils.py` | Parseo KML, point-in-polygon, filtrado geográfico | ✅ |
| `pipeline_utils.py` | `iter_patch_files`, `run_command`, helpers comunes | ✅ |
| `00_explore_candidates.py` | Filtra CSV Nature por KML del Estrecho, genera `gibraltar_candidatos.csv` | ✅ |
| `01_download_dataset.py` | Descarga patches Sentinel-2 via OpenEO, genera máscaras GT, valida negativos | ✅ |
| `02_predict_unet.py` | Aplica UNet MARIDA en batch a todos los patches | ✅ |
| `03_predict_rf.py` | Aplica Random Forest MARIDA en batch | ✅ |
| `04_predict_resnet.py` | Aplica ResNet MARIDA en batch, guarda JSON con probs | ✅ |
| `05_predict_indices.py` | Calcula FDI, NDVI y FDI+NDVI con umbral adaptativo | ✅ |
| `06_unify_predictions.py` | Unifica todas las predicciones en `predictions_master.csv` | ✅ |
| `07_build_xgboost_dataset.py` | Construye dataset tabular para XGBoost | ✅ |
| `08_train_xgboost.py` | Entrena XGBoost con LOO-CV, guarda modelo y métricas | ✅ |
| `09_evaluate.py` | Tabla comparativa + curvas ROC + gráficos | ✅ |
| `10_error_analysis.py` | Análisis visual de FP y FN por método | ✅ |
| `11_visualizacion.py` | Visor interactivo matplotlib: RGB + todas las máscaras por patch | ✅ |

---

## 9. Orden de ejecución del pipeline completo

Ejecutar desde `deliverables/tfg_pipeline_unificado/scripts/`:

```bash
python 00_explore_candidates.py
python 01_download_dataset.py --n_positives 50

# Predicciones (pueden ejecutarse en paralelo)
python 02_predict_unet.py
python 03_predict_rf.py
python 04_predict_resnet.py
python 05_predict_indices.py

# Unificar y evaluar
python 06_unify_predictions.py
python 07_build_xgboost_dataset.py
python 08_train_xgboost.py
python 09_evaluate.py
python 10_error_analysis.py

# Visualización interactiva
python 11_visualizacion.py --limit 50 --only-label ALL
```

---

## 10. Problemas conocidos y limitaciones

### Diferencia radiométrica L2A vs ACOLITE
MARIDA fue entrenado con reflectancias Rayleigh-corregidas via ACOLITE (`rhos`). Los patches de OpenEO son L2A de ESA. Los valores son similares pero no idénticos, lo que causa que la UNet clasifique agua mediterránea limpia como `Shallow Water` en algunos casos. Esta es una limitación documentada que se menciona en la memoria.

### Coordenadas L1C vs L2A en las máscaras GT
El NC de Nature usa coordenadas de píxel en la imagen L1C original. Los patches descargados son L2A en UTM. La función `build_mask` compensa este desajuste con `find_shift()` (correlación cruzada FDI), pero el shift puede variar por patch (observado: 15–77 píxeles vertical, 0–30 horizontal).

### Rate limiting OpenEO
OpenEO limita a ~3 descargas simultáneas. Entre descargas se aplica una pausa. Errores 429 se reintentan automáticamente.

### Error cosmético GDAL
```
ERROR 4: Unable to open EPSG support file gcs.csv.
```
No afecta a los resultados. Solución opcional: añadir variable de entorno `GDAL_DATA`.

### Dataset pequeño
Con pocos patches positivos descargados, las métricas de evaluación no son estadísticamente robustas. El pipeline está diseñado para escalar a los 876 candidatos disponibles.

---

## 11. Próximos pasos pendientes

1. **Escalar la descarga** a 50–100 patches positivos + negativos del CSV de Gibraltar (los 876 candidatos están disponibles en `gibraltar_candidatos.csv`)
2. **Aplicar todos los modelos** al dataset completo
3. **Ejecutar pipeline de evaluación** (scripts 06–10)
4. **Mapa Leaflet interactivo** del Estrecho con predicciones del mejor modelo superpuestas a los puntos del CSV de Nature (`11_visualizacion.py` cubre la visualización local)
5. **Correlación Spearman** con concentraciones in situ del Marine Pollution Bulletin 2025
6. **Redactar la memoria** con los resultados comparativos

---

## 12. Notas para Claude Code

- **Los scripts activos están en `deliverables/tfg_pipeline_unificado/scripts/`** — la carpeta `notebooks/` contiene versiones antiguas y NO debe usarse como referencia
- `config.py` centraliza todas las rutas y se auto-detecta usando `.git` como ancla; si se cambia algo, solo hay que tocar ese archivo
- `geo_utils.py` y `pipeline_utils.py` son módulos de utilidades compartidas por todos los scripts
- Los scripts 02–05 (predicciones) se pueden ejecutar en paralelo entre sí
- El entorno conda `marida` tiene todas las dependencias instaladas
- Los patches están georeferenciados (GeoTIFF con CRS y transform) — se pueden abrir en QGIS directamente
- La visualización RGB correcta usa B04/B03/B02 (índices 3,2,1) con estiramiento `min=0, max=0.1`
- `11_visualizacion.py` muestra RGB + GT + UNet + RF + FDI + NDVI + FDI+NDVI + XGBoost en una sola ventana; teclas: flechas izquierda/derecha, `a`/`d`, `q` para salir
