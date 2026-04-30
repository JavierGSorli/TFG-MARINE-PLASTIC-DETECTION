# Instrucciones para Codex

Estás trabajando sobre un TFG de detección de residuos/plásticos flotantes en imágenes Sentinel-2 usando una pipeline ya existente. El objetivo no es rehacer todo el proyecto, sino añadir mejoras incrementales y revisables.

Scripts actuales esperados:

```text
scripts/
├── 00_explore_candidates.py
├── 01_download_dataset.py
├── 02_predict_unet.py
├── 03_predict_rf.py
├── 04_predict_resnet.py
├── 05_predict_indices.py
├── 06_unify_predictions.py
├── 07_build_xgboost_dataset.py
├── 08_train_xgboost.py
├── 09_evaluate.py
├── 10_error_analysis.py
├── 11_visualizacion.py
├── config.py
├── geo_utils.py
└── pipeline_utils.py
```

Reglas generales:

1. No rompas la compatibilidad con los scripts existentes.
2. Si modificas un script actual, conserva el comportamiento anterior siempre que sea posible.
3. Añade scripts nuevos antes que reescribir toda la pipeline.
4. Guarda resultados en `results/auto/...` o en rutas configurables desde `config.py`.
5. Documenta claramente las salidas generadas.
6. No entrenes/evalúes con random split si hay riesgo de leakage temporal, espacial o por evento.
7. No trates las etiquetas Nature-derived como ground truth perfecto; son weak labels/proxy ground truth.

# Bloque 7 — Modelo híbrido y mapas del Estrecho

## Objetivo

Crear una salida más operativa del TFG:

1. un modelo híbrido detección + segmentación;
2. mapas para visualizar el área de estudio, candidatos, patches y predicciones.

---

# Parte A — Modelo híbrido

## Script a crear

```text
scripts/24_hybrid_detector_segmenter.py
```

## Idea

Combinar:

```text
patch-level detection -> ResNet/XGBoost/classifier
pixel-level segmentation -> U-Net/RF/FDI+NDVI
```

La clasificación decide si merece la pena segmentar o si la predicción debe ser incierta.

## Lógica inicial recomendada

```python
if xgb_prob >= 0.75:
    confidence = "high"
    run_segmentation_decision = True
elif xgb_prob >= 0.45:
    confidence = "medium"
    run_segmentation_decision = True
else:
    confidence = "low"
    run_segmentation_decision = False
```

Los umbrales deben estar en `config.py` o en constantes al inicio del script.

## Máscaras híbridas posibles

Implementar al menos una opción configurable.

### Conservadora

```text
hybrid_mask = UNet & (RF | FDI_NDVI)
```

### Majority vote

```text
hybrid_mask = al menos 2 de 3 entre UNet, RF, FDI_NDVI
```

### Flexible

```text
hybrid_mask = UNet | RF | FDI_NDVI
```

Recomendación inicial: majority vote.

## Salidas esperadas

```text
results/auto/hybrid/hybrid_patch_predictions.csv
results/auto/hybrid/masks/*.tif
results/auto/hybrid/hybrid_summary.csv
```

Columnas:

```text
patch
y_true
xgb_prob
resnet_prob
hybrid_confidence
hybrid_decision
selected_mask_strategy
hybrid_pred_px
unet_px
rf_px
fdi_ndvi_px
usable_for_main_eval
confidence_level
```

## Evaluación

Si existe pseudo-máscara, calcular métricas pixel-level con cautela:

```text
IoU
Dice
precision_pixel
recall_pixel
```

Pero en documentación indicar que son métricas contra pseudo-máscaras, no ground truth perfecto.

---

# Parte B — Mapas del Estrecho

## Script a crear

```text
scripts/25_map_gibraltar_results.py
```

## Mapas recomendados

### Mapa 1 — Candidatos Nature

```text
results/auto/maps/gibraltar_nature_candidates.html
```

Mostrar:

```text
lat/lon
fecha
n_pixels_fil
```

### Mapa 2 — Patches del dataset

```text
results/auto/maps/gibraltar_dataset_patches.html
```

Mostrar positivos y negativos:

```text
SI/NO
confidence_level
scene_tags
```

### Mapa 3 — Predicciones

```text
results/auto/maps/gibraltar_predictions.html
```

Mostrar:

```text
xgb_prob
resnet_prob
hybrid_confidence
hybrid_pred_px
```

## Librerías sugeridas

```text
folium
geopandas
pandas
rasterio
shapely
```

Si alguna no está instalada, intentar fallback con `folium` + `pandas`.

## Importante para la memoria

No vender el mapa como cobertura exhaustiva de plástico en el Estrecho.

Usar esta formulación:

```text
El mapa representa las detecciones analizadas y las predicciones generadas por la pipeline sobre los patches disponibles. No constituye un producto operacional exhaustivo de cobertura completa, ya que la disponibilidad de imágenes Sentinel-2 está condicionada por nubosidad, fechas de adquisición y criterios de calidad.
```

## Criterios de aceptación

1. Se generan mapas HTML navegables.
2. Los mapas no fallan si faltan algunas columnas; deben mostrar lo disponible.
3. Cada punto incluye popup con información relevante.
4. Los mapas diferencian visualmente positivos, negativos y predicciones.
5. El script documenta que no es mapa exhaustivo.

## Qué NO hacer en este bloque

- No intentar mapear todo el Estrecho en todas las fechas.
- No afirmar que el mapa muestra todo el plástico existente.
- No descargar datos nuevos masivamente.
