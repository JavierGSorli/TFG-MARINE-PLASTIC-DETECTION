# Instrucciones para refactorizar y mejorar el pipeline TFG Marine Plastic Detection

## Rol esperado

Actúa como un **desarrollador senior / research engineer**. Tu objetivo no es simplemente mover archivos, sino convertir el pipeline actual en una estructura más limpia, reproducible y defendible para un TFG de detección de residuos plásticos flotantes con Sentinel-2.

El proyecto ya tiene patches generados, máscaras y outputs previos. **No debes regenerar patches ni repetir procesos caros o manuales salvo que se indique explícitamente.**

---

# 0. Restricciones obligatorias

## No hacer

1. **No regenerar patches Sentinel-2.**
   - Los patches actuales son válidos.
   - No tocar el proceso de descarga salvo para dejar scripts organizados o documentados.
   - No lanzar de nuevo descargas openEO/Copernicus.

2. **No rehacer anotaciones manuales.**
   - El sistema de anotación puede quedar preparado, pero no debe ejecutarse automáticamente.
   - No modificar manualmente metadata anotada.

3. **No recalcular máscaras Random Forest.**
   - RF tarda mucho.
   - Las máscaras RF ya existentes deben reutilizarse.
   - Cualquier script de evaluación debe detectar y reutilizar outputs existentes.

4. **No borrar datos fuente.**
   - No borrar `data/`.
   - No borrar patches.
   - No borrar modelos/pesos.
   - No borrar NetCDF, Excel o dataset.h5.

5. **No asumir ground truth perfecto.**
   - Mantener siempre la terminología de `weak labels`, `pseudo-masks`, `Nature-derived masks` o equivalente.

---

# 1. Objetivo general de la refactorización

Reordenar el pipeline en fases metodológicas claras:

```text
Fase A — Dataset y metadata
Fase B — Baselines y descriptores
Fase C — Consolidación y evaluación
Fase D — Modelos tabulares y ablación
Fase E — Sistema híbrido, mapas, visualización y reports
```

La numeración actual histórica puede cambiarse. Se permite borrar/recrear carpetas `scripts/`, `outputs/`, `reports/` y `src/` si hace falta, pero **no borrar los datos fuente ni los patches ya generados**.


---

# 2. Estructura final recomendada

Crear o adaptar esta estructura:

```text
pipeline/
├── scripts/
│   ├── 00_utils/
│   ├── 01_dataset/
│   ├── 02_baselines/
│   ├── 03_evaluation/
│   ├── 04_features_and_models/
│   ├── 05_hybrid_and_maps/
│   └── 06_reports/
├── src/
│   ├── common/
│   ├── dataset/
│   ├── baselines/
│   ├── features/
│   ├── evaluation/
│   ├── models/
│   ├── visualization/
│   └── reporting/
├── outputs/
│   ├── 01_dataset/
│   ├── 02_baselines/
│   ├── 03_evaluation/
│   ├── 04_features_and_models/
│   ├── 05_hybrid_and_maps/
│   └── 06_reports/
└── reports/
    ├── tables/
    ├── figures/
    ├── visual_examples/
    └── methodology/
```

La carpeta de patches debe seguir apuntando a la ubicación actual definida en config. No mover los patches salvo que sea estrictamente necesario.

---

# 3. Configuración común

Crear/reforzar:

```text
pipeline/src/common/config.py
pipeline/src/common/io_utils.py
pipeline/src/common/raster_utils.py
pipeline/src/common/metrics_utils.py
pipeline/src/common/validation_utils.py
pipeline/src/common/feature_utils.py
```

## `config.py`

Debe centralizar rutas:

```python
PROJECT_ROOT
DATA_DIR
PATCHES_DIR
OUTPUTS_DIR
REPORTS_DIR
MODELS_DIR

DATASET_METADATA_PATH
DATASET_METADATA_GROUPED_PATH
CSV_MASTER
FEATURE_TABLE_V2_PATH

UNET_OUT
RF_OUT
RESNET_OUT
INDICES_OUT
SAM_OUT
GLCM_OUT
EXTERNAL_MODELS_OUT
CLASSICAL_MODELS_OUT
ABLATION_OUT
HYBRID_OUT
MAPS_OUT
VIZ_OUT
REPORTS_OUT
```

También constantes:

```python
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
MARIDA_NM_COLUMNS = ["nm440", "nm490", "nm560", "nm665", "nm705", "nm740", "nm783", "nm842", "nm865", "nm1600", "nm2200"]
DEBRIS_CLASS
PATCH_SIZE = 256
RANDOM_STATE = 42
```

Crear función:

```python
def ensure_output_dirs():
    ...
```

---

# 4. Fase A — Dataset y metadata

## Scripts finales

```text
scripts/01_dataset/01_select_nature_candidates.py
scripts/01_dataset/02_download_dataset_patches.py
scripts/01_dataset/03_annotate_patch_quality.py
scripts/01_dataset/04_build_grouped_metadata.py
scripts/01_dataset/05_build_grouped_splits.py
```

## Estado esperado

Los scripts deben existir y estar listos, pero:

- `02_download_dataset_patches.py` **no debe ejecutarse automáticamente**.
- `03_annotate_patch_quality.py` **no debe ejecutarse automáticamente**.

## 01_select_nature_candidates.py

Mantener funcionalidad actual:

- leer Excel Nature;
- filtrar por KML;
- distancia a costa;
- mínimo de píxeles;
- guardar candidatos.

Añadir si no existe:

```text
outputs/01_dataset/candidate_selection_summary.md
outputs/01_dataset/candidates_filtered.csv
outputs/01_dataset/rejected_candidates_by_filter.csv
```

Debe registrar:

```text
n_original
n_inside_bbox
n_inside_kml
n_after_distance
n_after_pixels
thresholds used
```

## 02_download_dataset_patches.py

No rehacer la descarga. Solo dejar script robusto.

Añadir log persistente de intentos, tanto negativos como positivos:

```text
outputs/01_dataset/download_attempts_log.csv
```

Columnas recomendadas:

```text
timestamp
kind                # SI / NO
candidate_id
lat
lon
date
status              # ok / download_error / cloud_reject / land_reject / manual_reject / duplicate / too_close / etc.
reason
cloud_frac
land_frac
patch_path
mask_path
```

Muy importante: actualmente muchos negativos fallidos se pierden en consola. Deben persistirse para evitar reintentos y para la memoria.

Pero no ejecutar descarga.

## 03_annotate_patch_quality.py

Dejar listo. No ejecutar.

Debe soportar:

```bash
python 03_annotate_patch_quality.py --only-unannotated
python 03_annotate_patch_quality.py --label SI
python 03_annotate_patch_quality.py --label NO
```

Mantener como está pero quitar la columna de accept, reject, uncertain y manual confidence en vez de 3,2,1 que sea manual_confidence        # high / medium / low


## 04_build_grouped_metadata.py

Crear o reforzar.


## 05_build_grouped_splits.py

Reutilizar lógica actual de grouped splits.

Debe generar:

```text
outputs/01_dataset/grouped_splits/folds.csv
outputs/01_dataset/grouped_splits/split_summary.md
```

Validaciones obligatorias:

- ningún `group_id` puede aparecer en train y test del mismo fold;
- contar SI/NO por fold;
- guardar warnings si algún fold queda desbalanceado.

---

# 5. Fase B — Baselines y descriptores

## Scripts finales

```text
scripts/02_baselines/10_run_unet_marida.py
scripts/02_baselines/11_run_rf_marida.py
scripts/02_baselines/12_run_resnet_marida.py
scripts/02_baselines/13_run_spectral_indices.py
scripts/02_baselines/14_compute_sam_features.py
scripts/02_baselines/15_run_sam_pixel_classifier.py
scripts/02_baselines/16_compute_glcm_features.py
scripts/02_baselines/17_run_external_models.py
```

---

## 10_run_unet_marida.py

Reutilizar script actual.

---

## 11_run_rf_marida.py

Importante: **no recalcular RF por defecto**.

---

## 12_run_resnet_marida.py

Mantener wrapper.

No limitarse a `resnet_active`; guardar siempre `resnet_prob`.

---

## 13_run_spectral_indices.py

Mantener FDI/NDVI baseline original, pero añadir variantes.

'''
--water-mask none
--water-mask simple
```

Outputs separados por configuración:

```text
outputs/02_baselines/indices/no_water_maks/
outputs/02_baselines/indices/water_mask/
```

---

## 14_compute_sam_features.py

Reforzar la versión actual.

Objetivo:

1. Crear o reutilizar CSV de firmas espectrales MARIDA desde `dataset.h5`.
2. Calcular SAM de cada patch contra todas las clases MARIDA.
3. Generar features robustas, no solo `min`.

### Firmas MARIDA

CSV cache:

```text
outputs/02_baselines/sam/marida_spectral_signatures_by_class.csv
```

Debe calcular por clase:

```text
class
class_safe
n_pixels
B01_mean, B01_std, B01_median, B01_p05, B01_p95
...
B12_mean, B12_std, B12_median, B12_p05, B12_p95
```


### Features SAM por patch

No centrar el análisis en el mejor píxel solamente.

Para cada clase guardar:

```text
sam_<class>_mean
sam_<class>_min
sam_<class>_p01
sam_<class>_p05
sam_<class>_p10
sam_<class>_top10_mean
sam_<class>_top50_mean
sam_<class>_count_lt_010
sam_<class>_count_lt_015
sam_<class>_count_lt_020
sam_<class>_pct_lt_010
sam_<class>_pct_lt_015
sam_<class>_pct_lt_020
```

Guardar también:

```text
sam_best_class_min
sam_best_class_p05
sam_best_class_top50
sam_debris_p05
sam_debris_top50_mean
sam_margin_debris_vs_water_p05
sam_margin_debris_vs_best_confuser_p05
sam_margin_debris_vs_best_confuser_top50
sam_debris_confidence_p05
```

No tratar `Dense Sargassum` como debris por defecto. 

---

## 15_run_sam_pixel_classifier.py

Crear script nuevo. que sea como 13_compute_sam_features2.py

Objetivo: baseline de segmentación interpretable SAM pixel-wise.

Para cada patch:

1. leer patch 11 bandas;
2. cargar firmas MARIDA;
3. calcular ángulo de cada píxel contra cada clase;
4. asignar clase de menor ángulo;
5. guardar mapas.

Outputs por patch:

```text
<stem>_sam_class.tif          # uint8 class id winner
<stem>_sam_debris_mask.tif    # binary robust debris mask

<stem>_sam_second_angle.tif   # mascara binaria pero marca 1 si marine debries ha salido como la primera o la segunda mejor ángulo
<stem>_sam_third_angle.tif   # mascara binaria pero marca 1 si marine debries ha salido como la primera, la segunda, o la tercera mejor ángulo
```

Reglas para debris robusto:

```text
class winner == Marine Debris
```

Importante: este script sí puede tardar, pero no tanto como RF. Debe ser vectorizado por patch.

---

## 16_compute_glcm_features.py

Mantener

---

## 17_run_external_models.py

Mantener modelo externo.

Muy importante: probar estrategias para B09:

```bash
--b09-mode zero
--b09-mode copy_b8a
--b09-mode interpolate_b8a_b11
```

Guardar outputs separados:

```text
external_models/marinedebrisdetector_b09_zero/
external_models/marinedebrisdetector_b09_copy_b8a/
external_models/marinedebrisdetector_b09_interp/
```

Default puede ser `zero`, pero documentar que es adaptación fuerte.

---

# 6. Fase C — Consolidación, calibración y evaluación

## Scripts finales

```text
scripts/03_evaluation/20_unify_predictions.py
scripts/03_evaluation/21_calibrate_thresholds.py
scripts/03_evaluation/22_evaluate_pretrained_baselines.py
scripts/03_evaluation/23_evaluate_segmentation_pixelwise.py
scripts/03_evaluation/24_error_analysis_and_diagnostics.py
```

## 20_unify_predictions.py

Unificar scores crudos y máscaras disponibles, sin aplicar thresholds finales.

Integrar:
- UNet
- RF existing
- ResNet
- Indices
- SAM pixel-wise
- External models
- GT masks
- metadata

Outputs:
- outputs/03_evaluation/predictions_master.csv
- outputs/03_evaluation/predictions_master_missing_report.md

## 21_calibrate_thresholds.py

Leer predictions_master.csv y calcular thresholds para:
- UNet (unet_pct)
- RF (rf_pct)
- FDI (fdi_pct)
- NDVI (ndvi_pct)
- FDI+NDVI (fdi_ndvi_pct)
- ResNet (resnet_prob)
- SAM (sam_pct o sam_score)
- External model (external_score)

Guardar:
- thresholds_selected.csv
- threshold_calibration_summary.md

## 22_evaluate_pretrained_baselines.py

Leer:
- predictions_master.csv
- thresholds_selected.csv

Evaluar:
- UNet
- RF
- FDI
- NDVI
- FDI+NDVI
- ResNet
- SAM
- External model

Métricas:
- AUC
- PR-AUC
- threshold usado
- F1
- Precision
- Recall
- Balanced Accuracy
- TP FP TN FN

Outputs:
- patch_level_metrics_calibrated.csv
- patch_level_metrics_default_thresholds.csv
- tabla_comparativa.csv

## 23_evaluate_segmentation_pixelwise.py

Evaluación pixel-wise real leyendo máscaras.

Métodos:
- UNet
- RF
- FDI
- NDVI
- FDI+NDVI
- SAM
- External model si tiene mask
- Hybrid futuro

Calcular:
- TP FP FN TN
- IoU
- Dice/F1
- precision
- recall
- pred_px
- gt_px

Outputs:
- pixelwise_metrics_by_patch.csv
- pixelwise_metrics_summary.csv
- segmentation_noise_on_negatives.csv

## 24_error_analysis_and_diagnostics.py

Fusiona error analysis y análisis temporal/calidad.

CLI:
```bash
--method UNet
--method RF
--method SAM
--method external
--errors FP
--errors FN
--limit 20
--by-date
--by-quality
--by-scene-tags
```

Generar overlays:
- RGB
- GT
- Prediction
- TP/FP/FN

Colores:
- TP verde
- FP rojo
- FN azul

Outputs:
- error_cases_patch_level.csv
- error_summary_by_method.csv
- error_summary_by_scene_tags.csv
- date_distribution.csv
- metrics_by_date.csv
- top_dates_summary.csv
- figures/*.png
- examples/*.png

---

# 7. Fase D — Feature table, modelos y ablación

## Scripts finales

```text
scripts/04_features_and_models/30_build_feature_table.py
scripts/04_features_and_models/31_train_xgboost_grouped.py
scripts/04_features_and_models/32_train_classical_models_grouped.py
scripts/04_features_and_models/33_run_ablation_grouped.py
scripts/04_features_and_models/34_feature_importance_and_shap.py
```

---

## 30_build_feature_table.py

Partir del script actual v2.

Debe crear una tabla con una fila por patch.

Feature families:

```text
spectral_
index_
marida_model_
sam_
glcm_
external_model_
quality_
```

Añadir validaciones:

```python
merge(..., validate="one_to_one")
```

Generar reporte:

```text
feature_table_report.md
```

Debe incluir:

```text
n_rows
n_columns
n_features_by_family
missing percentage by family
duplicate patches
columns flagged as potential leakage
```

Columnas potencialmente leakage:

```text
manual_confidence
has_possible_debris
quality_has_possible_debris
expected_gt_px
mask_gt_px
nc_px
```

No eliminarlas necesariamente, pero marcarlas.

---

## 31_train_xgboost_grouped.py

Reutilizar script XGBoost, pero separar claramente:

```text
exploratory random CV
main grouped CV
```

La métrica principal debe ser grouped OOF.

Outputs:

```text
out_of_fold_predictions.csv
summary_metrics.csv
fold_metrics.csv
feature_importance.csv
model_config.json
```

Threshold selection:

- seleccionar threshold en train fold;
- aplicar en test fold;
- no seleccionar threshold global para métrica principal.

---

## 32_train_classical_models_grouped.py

Mantener.

Pero añadir feature set selector:

```bash
--feature-set automatic_only
--feature-set all_without_quality
--feature-set all_features
```

Por defecto usar:

```text
automatic_only
```

No incluir features manuales/quality salvo flag explícito.

Modelos:

```text
logistic_regression
random_forest
hist_gradient_boosting
xgboost si instalado
```

Outputs actuales están bien.

---

## 33_run_ablation_grouped.py

Muy importante. Extender.

Experimentos mínimos:

```text
spectral_only
indices_only
marida_models_only
sam_only
glcm_only
external_only
quality_only
physics_only                 # indices + sam
spectral_plus_indices
marida_plus_indices
automatic_only               # todo menos quality/manual/GT
all_without_fdi
all_without_resnet
all_without_sam
all_without_quality
all_features
```

Importante:

- `all_features` puede contener leakage si incluye quality manual;
- usar `automatic_only` como resultado principal si se quiere evitar leakage.

Outputs:

```text
ablation_results.csv
ablation_summary.md
ablation_barplot.png
```

En summary, resaltar:

```text
best overall
best automatic-only
impact of removing FDI
impact of removing SAM
impact of quality features
```

---

## 34_feature_importance_and_shap.py

Crear script nuevo.

Objetivo:

- entrenar modelo final grouped-compatible o sobre todo dataset solo para interpretación;
- calcular feature importance;
- si SHAP está disponible, calcular SHAP.

Outputs:

```text
feature_importance.csv
top20_features.png
shap_summary.png si disponible
feature_family_importance.csv
```

Agrupar importancia por prefijo:

```text
spectral
index
marida_model
sam
glcm
external_model
quality
```

---

# 8. Fase E — Sistema final, mapas, visualización, reports

## Scripts finales

```text
scripts/05_hybrid_and_maps/40_build_hybrid_detector_segmenter.py
scripts/05_hybrid_and_maps/41_generate_gibraltar_maps.py
scripts/05_hybrid_and_maps/42_visualize_predictions.py
scripts/06_reports/50_export_reports.py
scripts/06_reports/51_build_tfg_assets.py
```

---

## 40_build_hybrid_detector_segmenter.py

Mantener idea actual, pero corregir puntos clave.

### Detector

Seleccionar mejor detector grouped entre:

```text
classical models
xgboost grouped
```

Pero usar threshold correcto:

- no usar fijo 0.5 si el modelo tiene threshold OOF/fold;
- si solo hay probabilidades, permitir:

```bash
--threshold 0.5
--threshold-source grouped_oof
```

### Segmentador híbrido

Si hay máscaras reales:

```text
majority_vote_mask.tif
conservative_mask.tif
sensitive_mask.tif
```

Si no hay máscaras, mantener conteos aproximados, pero escribir claramente:

```text
approximation_only = True
```

Reglas:

```text
majority_vote = al menos 2 de [UNet, RF, FDI+NDVI, SAM] positivos por píxel
conservative = UNet AND (RF OR SAM OR FDI+NDVI)
sensitive = UNet OR RF OR SAM
```

Si no existe SAM pixel mask, omitirlo con warning.

### Confidence

Reglas actuales aceptables, pero parametrizar:

```bash
--high-prob 0.75
--medium-prob 0.45
--min-agreement-high 2
```

Outputs:

```text
hybrid_predictions.csv
hybrid_patch_metrics.csv
hybrid_pixel_metrics.csv
hybrid_by_date.csv
hybrid_by_quality.csv
hybrid_summary.md
```

---

## 41_generate_gibraltar_maps.py

Mantener.

Mejoras:

1. Añadir capa satélite.
2. Diferenciar visualmente `approx_fallback`.
3. Añadir mapa operacional:

```text
gibraltar_operational_alert_map.html
```

Este mapa debe mostrar solo:

```text
high_confidence
hybrid_label = 1
prob >= 0.75
```

4. Añadir summary:

```text
n_points_by_latlon_source
n_high_confidence
n_positive_predictions
```

---

## 42_visualize_predictions.py

Unificar versiones 20 y 24.

Debe soportar:

```bash
--mode original
--mode extended
--only-label SI|NO|ALL
--patch PATCH_NAME
--errors fp|fn|all
--method UNet|RF|Hybrid|XGB|SAM
--sort uncertainty|prob_desc|agreement_low
--limit N
```

Añadir filtros reales. Actualmente algunos args existen pero no se usan.

Modo extended debe mostrar:

```text
GT
UNet
RF
FDI
NDVI
FDI+NDVI
SAM mask si existe
Hybrid mask si existe
XGB / Hybrid patch-level scores
```

Exportar PNG y log.

Añadir panel lateral más ordenado:

```text
Ground truth
Baselines
Meta-models
Hybrid
Quality
```

---

## 50_export_reports.py

Mantener versión actual, pero añadir:

```text
best_results_summary.csv
best_results_summary.md
```

Debe leer:

```text
baseline metrics
xgb grouped
classical models
ablation
external
hybrid
```

Y crear ranking final:

```text
best pretrained baseline
best tabular model
best automatic feature set
best hybrid detector
best external model
```

También generar warnings por outputs faltantes.

---

## 51_build_tfg_assets.py

Crear script nuevo para assets listos para memoria.

Outputs:

```text
reports/tfg_assets/tables_markdown/
reports/tfg_assets/tables_latex/
reports/tfg_assets/figures_selected/
reports/tfg_assets/captions.md
reports/tfg_assets/results_narrative.md
```

Debe generar:

1. Tabla comparativa baselines redondeada.
2. Tabla grouped models redondeada.
3. Tabla ablación redondeada.
4. Tabla híbrido.
5. Captions sugeridas.
6. Texto breve de resultados.

No inventar conclusiones; basarse en CSV existentes.

---

# 9. Reglas de reproducibilidad

Todos los scripts deben:

1. Tener `argparse`.
2. Usar rutas desde `config.py`.
3. Crear outputs en carpeta propia.
4. No sobrescribir outputs pesados salvo `--overwrite`.
5. Guardar CSV + Markdown summary.
6. Imprimir resumen final.
7. Capturar errores por patch cuando sea posible.
8. No depender de consola como único registro.

---

# 10. Orden recomendado de ejecución final

No ejecutar scripts de descarga/anotación/RF salvo indicación.

Orden seguro:

```bash
# Fase A — solo splits, no descarga
python scripts/01_dataset/05_build_grouped_splits.py

# Fase B — reutilizar outputs existentes y calcular features nuevas
python scripts/02_baselines/10_run_unet_marida.py
python scripts/02_baselines/11_run_rf_marida.py --reuse-existing-only
python scripts/02_baselines/12_run_resnet_marida.py
python scripts/02_baselines/13_run_spectral_indices.py 
python scripts/02_baselines/14_compute_sam_features.py
python scripts/02_baselines/15_run_sam_pixel_classifier.py
python scripts/02_baselines/16_compute_glcm_features.py
python scripts/02_baselines/17_run_external_models.py --b09-mode zero

# Fase C
python scripts/03_evaluation/20_unify_predictions.py
python scripts/03_evaluation/21_evaluate_pretrained_baselines.py
python scripts/03_evaluation/22_evaluate_segmentation_pixelwise.py
python scripts/03_evaluation/23_error_analysis_baselines.py
python scripts/03_evaluation/24_analysis_by_date.py
python scripts/03_evaluation/25_evaluate_external_models.py

# Fase D
python scripts/04_features_and_models/30_build_feature_table.py
python scripts/04_features_and_models/31_train_xgboost_grouped.py
python scripts/04_features_and_models/32_train_classical_models_grouped.py --feature-set automatic_only
python scripts/04_features_and_models/33_run_ablation_grouped.py
python scripts/04_features_and_models/34_feature_importance_and_shap.py

# Fase E
python scripts/05_hybrid_and_maps/40_build_hybrid_detector_segmenter.py
python scripts/05_hybrid_and_maps/41_generate_gibraltar_maps.py
python scripts/06_reports/50_export_reports.py
python scripts/06_reports/51_build_tfg_assets.py
```

Visualizador solo manual:

```bash
python scripts/05_hybrid_and_maps/42_visualize_predictions.py --mode extended
```

---

# 11. Criterios de éxito

La implementación se considerará correcta si:

1. No se regeneran patches.
2. No se recalcula RF salvo flag explícito.
3. Se preservan outputs existentes o se respaldan.
4. Existe estructura limpia por fases.
5. Se genera `predictions_master.csv` completo.
6. Se genera evaluación patch-level.
7. Se genera evaluación pixel-wise real si las máscaras existen.
8. Se genera feature table con reporte de missing/leakage.
9. Se genera ablación extendida.
10. Se genera reporte final en `reports/`.
11. El pipeline documenta claramente weak labels y limitaciones.

---

# 12. Comentario metodológico que debe mantenerse en summaries

Usar una redacción similar a esta en summaries/reportes:

> Las etiquetas positivas proceden del catálogo Nature de litter windrows y las máscaras utilizadas son reconstrucciones aproximadas basadas en geometría publicada y alineamiento sobre imágenes Sentinel-2. Por tanto, no constituyen ground truth pixel-perfect ni confirmación material directa de plástico. Las métricas pixel-level deben interpretarse como evaluación frente a weak labels/pseudo-masks y no como validación absoluta de residuos plásticos.

Y también:

> La validación agrupada por fecha/evento se emplea para los modelos entrenados con el dataset propio, con el fin de reducir leakage temporal o por evento. Los modelos preentrenados y reglas espectrales se evalúan como baselines externos/fijos, aunque sus umbrales pueden calibrarse internamente para análisis comparativo.

---

# 13. Prioridad de implementación

Si no hay tiempo para todo, implementar en este orden:

1. Reordenación de carpetas + config común.
2. RF reuse-only.
3. `20_unify_predictions.py` robusto.
4. `22_evaluate_segmentation_pixelwise.py` real.
5. SAM pixel-wise.
6. Feature table con leakage report.
7. Ablation extendida con `automatic_only` y `all_without_quality`.
8. Hybrid con threshold correcto.
9. Reports final + TFG assets.