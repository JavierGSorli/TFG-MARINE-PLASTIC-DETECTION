# Fase E — Modelo híbrido, mapas y outputs finales

## Objetivo

Construir una salida final interpretable para el TFG a partir de los modelos y análisis ya consolidados.

Esta fase debe producir:

1. un modelo híbrido detección + segmentación;
2. mapas del Estrecho;
3. figuras finales para la memoria;
4. tablas finales limpias;
5. resumen de limitaciones.

---

# Parte 1 — Modelo híbrido detección + segmentación

## Idea

Separar dos tareas:

```text
1. Detección patch-level:
   ¿Hay una estructura compatible con debris/plástico flotante en el patch?

2. Segmentación pixel-level:
   ¿Dónde está aproximadamente esa estructura dentro del patch?
```

Esto es importante porque tus resultados muestran que la detección patch-level es bastante mejor que la coincidencia pixel-level exacta.

---

## Script

Crear:

```text
src/models/26_hybrid_detector_segmenter.py
```

Entrada:

```text
dataset_metadata.csv
feature_table_v2.csv
predicciones U-Net/RF/FDI/NDVI/ResNet/XGBoost
máscaras disponibles
```

Salida:

```text
results/auto/hybrid/hybrid_predictions.csv
results/auto/hybrid/masks/
results/auto/hybrid/hybrid_summary.md
```

---

## Lógica recomendada

### Detección

Usar como detector principal el mejor modelo validado de forma conservadora entre:

```text
XGBoost grouped
modelo clásico grouped
combinación simple de U-Net/RF
```

No usar el XGBoost exploratorio si no está validado por fecha.

### Segmentación

Crear varias opciones de máscara:

```text
unet_mask
rf_mask
fdi_ndvi_mask
majority_vote_mask
conservative_mask
```

Ejemplos:

```python
majority_vote = at_least_two_of(unet, rf, fdi_ndvi)
conservative = unet & (rf | fdi_ndvi)
sensitive = unet | rf
```

Guardar para cada patch:

```text
patch
hybrid_prob
hybrid_label
confidence_level
selected_mask_type
hybrid_pred_px
```

---

## Niveles de confianza

Definir:

```text
high_confidence
medium_confidence
low_confidence
```

Ejemplo:

```text
high_confidence:
  hybrid_prob >= 0.75
  and image_quality != bad
  and model_agreement >= 2

medium_confidence:
  0.45 <= hybrid_prob < 0.75

low_confidence:
  hybrid_prob < 0.45
  or image_quality == bad
```

No vender `high_confidence` como plástico confirmado.  
Debe significar:

```text
alta confianza de estructura compatible con debris según la pipeline
```

---

# Parte 2 — Evaluación del modelo híbrido

Crear:

```text
src/evaluation/27_evaluate_hybrid.py
```

Salida:

```text
results/auto/hybrid/hybrid_patch_metrics.csv
results/auto/hybrid/hybrid_pixel_metrics.csv
results/auto/hybrid/hybrid_by_date.csv
results/auto/hybrid/hybrid_by_quality.csv
```

Si el detector híbrido se entrena con tus datos, debe evaluarse con GroupKFold por fecha.  
Si es una regla fija sin entrenamiento, no necesita GroupKFold, pero sí análisis por fecha/calidad.

---

# Parte 3 — Mapas del Estrecho

## Objetivo

Generar mapas visuales, pero no venderlos como “mapa exhaustivo de plásticos del Estrecho”.

Deben presentarse como:

```text
visualización espacial de detecciones analizadas y predicciones de la pipeline
```

## Script

Crear:

```text
src/visualization/28_map_gibraltar_results.py
```

Salidas:

```text
results/auto/maps/gibraltar_dataset_map.html
results/auto/maps/gibraltar_predictions_map.html
results/auto/maps/gibraltar_confidence_map.html
results/auto/maps/gibraltar_dataset_map.png   # opcional
results/auto/maps/gibraltar_predictions_map.png # opcional
```

## Mapas recomendados

### 1. Mapa del dataset

Mostrar:

```text
SI
NO
fecha
n_pixels_gt
image_quality
```

### 2. Mapa de predicciones

Mostrar:

```text
hybrid_prob
hybrid_label
confidence_level
model agreement
```

### 3. Mapa temporal

Opcional:

```text
color por año
tamaño por número de píxeles del filamento
```

---

# Parte 4 — Figuras finales para memoria

Crear:

```text
src/visualization/29_make_final_figures.py
```

Generar:

```text
results/final_figures/
```

Figuras recomendadas:

```text
01_dataset_temporal_distribution.png
02_patch_level_comparison.png
03_pixel_level_comparison.png
04_fp_by_difficulty_or_quality.png
05_noise_on_negatives.png
06_xgboost_exploratory_vs_grouped.png
07_ablation_results.png
08_hybrid_examples.png
09_gibraltar_map.png
```

## Ejemplos visuales

Seleccionar automáticamente ejemplos:

```text
true_positive_good
false_positive_difficult
false_negative_small_filament
high_confidence_prediction
low_confidence_prediction
```

Para cada ejemplo, crear panel:

```text
RGB
pseudo-mask
U-Net
RF
FDI/NDVI
Hybrid
```

---

# Parte 5 — Tablas finales para memoria

Crear:

```text
src/evaluation/30_export_final_tables.py
```

Salida:

```text
results/final_tables/table_01_dataset_summary.csv
results/final_tables/table_02_patch_metrics.csv
results/final_tables/table_03_pixel_metrics.csv
results/final_tables/table_04_by_date.csv
results/final_tables/table_05_ablation.csv
results/final_tables/table_06_hybrid.csv
```

También exportar versión Markdown:

```text
results/final_tables/final_tables.md
```

---

# Parte 6 — Resumen de limitaciones

Crear automáticamente:

```text
results/final_report_sections/limitations_summary.md
```

Debe incluir:

```text
# Limitaciones principales

## Ground truth débil
Las etiquetas derivan de litter windrows y pseudo-máscaras, no de verificación material directa de plástico.

## No independencia total de patches
Algunos patches pertenecen a las mismas fechas/productos Sentinel-2.

## Segmentación pixel-level limitada
Pequeños desplazamientos espaciales penalizan mucho IoU.

## FDI y circularidad
Si la construcción/alineación de pseudo-máscaras usa información relacionada con FDI, las métricas de FDI deben interpretarse con cautela.

## Domain shift
Modelos MARIDA pueden no adaptarse perfectamente a L2A del Estrecho.

## Mapas no exhaustivos
Los mapas muestran la zona/detecciones analizadas, no una cartografía completa de todo el plástico del Estrecho.
```

---

# Criterios de aceptación

Codex debe completar esta fase si:

1. Existe modelo híbrido con salidas CSV y máscaras.
2. El híbrido se evalúa correctamente.
3. Existen mapas HTML.
4. Existen figuras finales.
5. Existen tablas finales.
6. Existe resumen de limitaciones.
7. No se afirma que las predicciones sean plástico confirmado.

---

# No hacer en esta fase

- No modificar el dataset base.
- No reescribir la lógica de descarga.
- No introducir nuevos modelos externos.
- No hacer fine-tuning salvo que se haya decidido explícitamente.
