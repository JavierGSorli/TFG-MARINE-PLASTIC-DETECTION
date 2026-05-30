# Fase E adaptada — Modelo híbrido, mapas, visualización y reportes finales

## Contexto de esta adaptación

Este documento adapta la Fase E al estado actual del proyecto.

Cambio principal respecto a versiones anteriores:

> La estructura del proyecto ya está correcta hasta el script `22`.  
> Por tanto, todo lo nuevo de la Fase E debe empezar a partir del `23`.

No renumerar ni modificar los scripts `00` a `22`, salvo que sea estrictamente necesario para leer sus outputs.

La Fase E debe añadirse como una capa final sobre la pipeline ya existente.

---

# Estructura actual asumida

El proyecto usa:

```text
scripts/   -> scripts ejecutables numerados
outputs/   -> salidas por script/fase
src/       -> funciones reutilizables
reports/   -> tablas, figuras y material para memoria
archive/   -> scripts antiguos, debug y versiones deprecated
```

La pipeline ya está organizada hasta:

```text
scripts/22_...
outputs/22_...
```

Por tanto, la Fase E debe comenzar en:

```text
scripts/23_build_hybrid_detector_segmenter.py
scripts/24_generate_gibraltar_maps.py
scripts/25_visualize_predictions.py
scripts/26_export_reports.py
```

---

# Objetivo general de la Fase E

Construir la capa final interpretable del TFG:

1. modelo híbrido detección + segmentación;
2. mapas del Estrecho;
3. visualización interactiva/flexible de resultados;
4. exportación de tablas, figuras y ejemplos a `reports/`;
5. resumen de limitaciones metodológicas.

Esta fase no debe cambiar el dataset base ni recalcular los modelos anteriores salvo que falten outputs necesarios.

---

# Nueva posición en la pipeline

La Fase E queda así:

```text
23_build_hybrid_detector_segmenter.py
24_generate_gibraltar_maps.py
25_visualize_predictions.py
26_export_reports.py
```

No modificar la numeración previa.

Si ya existen scripts de modelos externos, evaluación externa, XGBoost, modelos clásicos, SAM, GLCM, feature table o ablation antes del `23`, dejarlos como están.

La Fase E debe consumir sus outputs, no renumerarlos.

---

# Inputs esperados antes de Fase E

Usar los outputs existentes de la pipeline actual. Codex debe detectar rutas reales desde `src/common/config.py` o desde la estructura existente.

De forma orientativa, pueden existir salidas como:

```text
outputs/*build_dataset_metadata*/
outputs/*unify_predictions*/
outputs/*evaluate_pretrained_baselines*/
outputs/*error_analysis*/
outputs/*analysis_by_date*/
outputs/*build_grouped_splits*/
outputs/*run_xgboost_experiments*/
outputs/*compute_sam_features*/
outputs/*compute_glcm_features*/
outputs/*build_feature_table_v2*/
outputs/*train_classical_models_grouped*/
outputs/*run_ablation_grouped*/
outputs/*run_external_models*/
outputs/*evaluate_external_models*/
```

No asumir obligatoriamente una numeración concreta anterior al `23`, porque el usuario ya tiene esa parte organizada.

---

# Script 23 — Modelo híbrido detección + segmentación

## Crear

```text
scripts/23_build_hybrid_detector_segmenter.py
```

## Módulos reutilizables opcionales

Si hace falta, crear funciones auxiliares en:

```text
src/models/hybrid.py
src/evaluation/hybrid_metrics.py
```

No poner lógica reutilizable grande dentro del script si puede vivir en `src/`.

---

## Objetivo

Crear un sistema híbrido que separe dos tareas:

```text
1. Detección patch-level:
   ¿Hay una estructura compatible con debris/plástico flotante en el patch?

2. Segmentación pixel-level:
   ¿Dónde está aproximadamente esa estructura dentro del patch?
```

Esto es importante porque los resultados actuales muestran que la detección patch-level funciona mejor que la coincidencia pixel-level exacta.

---

## Entradas

Buscar y usar, si existen:

```text
dataset_metadata_with_groups.csv
predictions_master.csv
feature_table_v2.csv
patch_level_metrics.csv
summary_metrics.csv de XGBoost grouped
out_of_fold_predictions.csv de XGBoost grouped
summary_metrics.csv de modelos clásicos grouped
out_of_fold_predictions.csv de modelos clásicos grouped
ablation_results.csv
external_model_patch_metrics.csv
external_model_pixel_metrics.csv
```

Y máscaras/predicciones desde los outputs de:

```text
U-Net MARIDA
RF MARIDA
ResNet MARIDA
FDI / NDVI / FDI+NDVI
modelos externos si existen
```

Codex debe localizar estas rutas de forma robusta usando `src/common/config.py` o búsqueda controlada en `outputs/`.

---

## Salidas

Guardar todo en:

```text
outputs/23_build_hybrid_detector_segmenter/
```

Con estructura:

```text
outputs/23_build_hybrid_detector_segmenter/
├── hybrid_predictions.csv
├── hybrid_patch_metrics.csv
├── hybrid_pixel_metrics.csv
├── hybrid_by_date.csv
├── hybrid_by_quality.csv
├── hybrid_summary.md
└── masks/
    ├── majority_vote/
    ├── conservative/
    └── sensitive/
```

---

## Lógica recomendada

### A. Detección patch-level

Usar como detector principal el mejor modelo validado de forma conservadora.

Candidatos:

```text
XGBoost grouped
Logistic Regression grouped
Random Forest grouped
HistGradientBoosting grouped
combinación simple de U-Net/RF
```

No usar como resultado principal el XGBoost exploratorio.

El XGBoost exploratorio puede aparecer como comparación, pero no debe ser el detector principal si no controla leakage temporal.

---

### B. Selección automática del detector principal

Leer summaries disponibles de modelos grouped:

```text
summary_metrics.csv de modelos clásicos grouped
summary_metrics.csv de XGBoost grouped
```

Seleccionar el modelo principal por métrica configurable:

```text
primary_metric = F1
```

Opcional:

```text
primary_metric = AUC
```

Guardar en `hybrid_summary.md`:

```text
modelo seleccionado
métrica usada
valor de la métrica
razón de la selección
```

Ejemplo:

```text
Detector patch-level seleccionado: Random Forest grouped
Motivo: mayor F1 out-of-fold con validación agrupada por fecha.
```

---

### C. Segmentación pixel-level

Crear varias máscaras híbridas:

```text
unet_mask
rf_mask
fdi_ndvi_mask
majority_vote_mask
conservative_mask
sensitive_mask
```

Reglas sugeridas:

```python
majority_vote = at_least_two_of(unet_mask, rf_mask, fdi_ndvi_mask)
conservative = unet_mask & (rf_mask | fdi_ndvi_mask)
sensitive = unet_mask | rf_mask
```

Si `fdi_ndvi_mask` no existe o es demasiado ruidosa, permitir configuración:

```text
use_fdi_ndvi_in_hybrid = true/false
```

---

### D. Salida por patch

`hybrid_predictions.csv` debe incluir:

```text
patch
date
label
label_binary
hybrid_detector_name
hybrid_prob
hybrid_label
threshold
confidence_level
selected_mask_type
hybrid_pred_px
model_agreement
unet_pred_px
rf_pred_px
fdi_pred_px
fdi_ndvi_pred_px
xgboost_grouped_prob
best_classical_grouped_prob
external_model_prob
image_quality
scene_tags
```

Si alguna columna no existe, dejarla como `NaN` y documentarlo.

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

Si no hay `image_quality` anotado, no penalizar por calidad, pero indicar en el summary:

```text
No se usó image_quality porque las anotaciones manuales estaban vacías o incompletas.
```

---

## Advertencia metodológica obligatoria

En `hybrid_summary.md` incluir:

```text
El nivel high_confidence no implica confirmación material directa de plástico. Significa alta confianza de la pipeline en la presencia de una estructura compatible con debris/litter windrows según las fuentes y modelos disponibles.
```

---

## Evaluación del híbrido

Si el híbrido usa un modelo entrenado con el dataset propio, usar las predicciones out-of-fold ya generadas por la validación grouped.

No reentrenar dentro de este script salvo que sea estrictamente necesario.

Generar:

```text
hybrid_patch_metrics.csv
hybrid_pixel_metrics.csv
hybrid_by_date.csv
hybrid_by_quality.csv
```

Regla:

- Si el detector híbrido es una regla fija sin entrenamiento nuevo, no necesita GroupKFold.
- Si entrena algo nuevo, debe usar folds agrupados por fecha ya existentes.

---

# Script 24 — Mapas del Estrecho

## Crear

```text
scripts/24_generate_gibraltar_maps.py
```

## Módulos reutilizables opcionales

```text
src/visualization/maps.py
```

---

## Objetivo

Generar mapas visuales del Estrecho, pero sin venderlos como mapa exhaustivo de plástico.

Deben presentarse como:

```text
visualización espacial de detecciones analizadas y predicciones de la pipeline
```

No decir:

```text
mapa completo de plásticos del Estrecho
```

---

## Entradas

```text
dataset_metadata_with_groups.csv
outputs/23_build_hybrid_detector_segmenter/hybrid_predictions.csv
outputs/23_build_hybrid_detector_segmenter/hybrid_patch_metrics.csv
```

Opcional:

```text
date_distribution.csv
external_models_summary.md
```

El metadata debe contener o poder recuperar:

```text
lat
lon
date
label
label_binary
expected_gt_px
image_quality
```

Si `lat/lon` no están en metadata, intentar extraerlos de:

```text
georreferenciación del patch
CSV original de candidatos
outputs de descarga/generación del dataset
```

Si no se pueden recuperar, generar warning y no crear mapas.

---

## Salidas

```text
outputs/24_generate_gibraltar_maps/
├── gibraltar_dataset_map.html
├── gibraltar_predictions_map.html
├── gibraltar_confidence_map.html
├── gibraltar_temporal_map.html
├── maps_summary.md
└── static/
    ├── gibraltar_dataset_map.png          # opcional
    ├── gibraltar_predictions_map.png      # opcional
    └── gibraltar_confidence_map.png       # opcional
```

Copiar mapas finales útiles a:

```text
reports/figures/maps/
```

desde `26_export_reports.py`.

---

## Mapas recomendados

### 1. Dataset map

Mostrar:

```text
SI / NO
fecha
n_pixels_gt
image_quality
```

Uso:

```text
describir distribución espacial del dataset
```

### 2. Predictions map

Mostrar:

```text
hybrid_prob
hybrid_label
confidence_level
model_agreement
```

Uso:

```text
visualizar predicciones de la pipeline
```

### 3. Confidence map

Mostrar:

```text
high_confidence
medium_confidence
low_confidence
```

Uso:

```text
separar detecciones fuertes de casos inciertos
```

### 4. Temporal map

Mostrar:

```text
color por año
tamaño por expected_gt_px o hybrid_pred_px
```

Uso:

```text
mostrar distribución temporal de los casos analizados
```

---

## Popups de mapa

Cada punto debe incluir:

```text
patch
date
label
expected_gt_px
hybrid_prob
hybrid_label
confidence_level
image_quality
scene_tags
```

---

## Advertencia obligatoria

En `maps_summary.md` incluir:

```text
Estos mapas representan únicamente los patches y detecciones analizadas en el proyecto. No constituyen una cartografía exhaustiva de residuos plásticos en el Estrecho.
```

---

# Script 25 — Visualización interactiva/flexible

## Crear

```text
scripts/25_visualize_predictions.py
```

Este script sustituye al antiguo `11_visualizacion.py`, pero debe conservar su comportamiento útil.

El antiguo debe moverse a:

```text
archive/deprecated_scripts/11_visualizacion.py
```

---

## Objetivo

Permitir revisar visualmente los patches uno a uno.

Debe tener dos modos principales:

```bash
python scripts/25_visualize_predictions.py --mode original
python scripts/25_visualize_predictions.py --mode extended
```

Alias:

```bash
python scripts/25_visualize_predictions.py --mode 1
python scripts/25_visualize_predictions.py --mode 2
```

---

## Modo original

Debe comportarse igual que el antiguo `11_visualizacion.py`, salvo por rutas nuevas.

Debe permitir:

```text
ir patch por patch
avanzar
retroceder
salir
guardar figura actual
```

Debe mostrar los mismos modelos que el original, normalmente:

```text
RGB
pseudo-GT / máscara Nature-derived
U-Net MARIDA
RF MARIDA
FDI
NDVI
FDI+NDVI
```

Si el original mostraba algo más, conservarlo.

---

## Modo extended

Debe añadir modelos no comparados en el visualizador original.

Incluir si existen:

```text
XGBoost exploratory
XGBoost grouped
Logistic Regression grouped
Random Forest grouped
HistGradientBoosting grouped
mejor modelo clásico grouped
SAM score
GLCM features/score
modelo externo
híbrido
```

Como varios son patch-level y no tienen máscara, mostrar:

```text
probabilidad / score
label predicha
threshold
fold si es out-of-fold
confidence_level
```

---

## Requisito visual obligatorio

En ambos modos, para cada patch debe verse:

```text
RGB
pseudo-máscara / GT mask si existe
predicciones o scores principales
```

Para negativos:

```text
GT mask: empty / no positive mask
```

No crear visualizaciones donde solo se vea una máscara sin la imagen RGB.

---

## Navegación mínima

Implementar o conservar:

```text
n / flecha derecha   -> siguiente patch
p / flecha izquierda -> patch anterior
q / esc              -> salir
s                    -> guardar figura actual
```

Argumentos mínimos:

```text
--mode
--patch
--max-patches
```

Argumentos opcionales:

```text
--only SI
--only NO
--errors UNet
--difficulty DIFICIL
--confidence high
```

---

## Salidas

```text
outputs/25_visualize_predictions/
├── saved_original/
├── saved_extended/
└── visualization_log.csv
```

`visualization_log.csv`:

```text
timestamp
mode
patch
output_png
```

Las figuras guardadas deben copiarse después a:

```text
reports/visual_examples/
```

mediante `26_export_reports.py`.

---

# Script 26 — Exportación de reportes finales

## Crear

```text
scripts/26_export_reports.py
```

---

## Objetivo

Regenerar automáticamente la carpeta `reports/` con:

```text
reports/tables/
reports/figures/
reports/visual_examples/
reports/methodology/
reports/pipeline_summary.md
reports/final_results_index.md
```

Este script debe reproducir y mantener la utilidad de los reportes generados en el último ZIP reorganizado.

---

## Copiar a `reports/tables/`

Copiar si existen tablas relevantes de fases previas y de la Fase E.

Como la numeración anterior ya está cerrada hasta `22`, Codex debe detectar rutas reales existentes dentro de `outputs/`, pero como mínimo debe incluir si existen:

```text
patch_level_metrics.csv
pixel_level_segmentation_metrics.csv
false_positives_by_difficulty.csv
segmentation_noise_on_negatives.csv
date_distribution.csv
metrics_by_date.csv
top_dates_summary.csv
summary_metrics.csv de XGBoost grouped
fold_metrics.csv de XGBoost grouped
xgboost_comparison.csv
summary_metrics.csv de modelos clásicos
fold_metrics.csv de modelos clásicos
ablation_results.csv
external_model_patch_metrics.csv
external_model_pixel_metrics.csv
outputs/23_build_hybrid_detector_segmenter/hybrid_predictions.csv
outputs/23_build_hybrid_detector_segmenter/hybrid_patch_metrics.csv
outputs/23_build_hybrid_detector_segmenter/hybrid_pixel_metrics.csv
outputs/23_build_hybrid_detector_segmenter/hybrid_by_date.csv
outputs/23_build_hybrid_detector_segmenter/hybrid_by_quality.csv
```

Renombrar con prefijos para evitar colisiones.

Ejemplos:

```text
baseline_patch_level_metrics.csv
baseline_pixel_level_segmentation_metrics.csv
date_distribution.csv
xgboost_grouped_summary_metrics.csv
classical_models_summary_metrics.csv
ablation_results.csv
external_model_patch_metrics.csv
hybrid_predictions.csv
hybrid_patch_metrics.csv
hybrid_pixel_metrics.csv
```

---

## Copiar a `reports/figures/`

Copiar `.png`, `.jpg`, `.jpeg`, `.pdf` desde outputs relevantes, incluyendo:

```text
outputs/23_build_hybrid_detector_segmenter/
outputs/24_generate_gibraltar_maps/
outputs/25_visualize_predictions/saved_original/
outputs/25_visualize_predictions/saved_extended/
```

Y también desde outputs anteriores si existen:

```text
baseline/evaluation outputs
error analysis outputs
date analysis outputs
xgboost outputs
classical models outputs
ablation outputs
external model outputs
```

Organizar en:

```text
reports/figures/baselines/
reports/figures/errors/
reports/figures/date_analysis/
reports/figures/xgboost/
reports/figures/classical_models/
reports/figures/ablation/
reports/figures/external_models/
reports/figures/hybrid/
reports/figures/maps/
reports/figures/visualization/
```

---

## Copiar a `reports/visual_examples/`

Copiar desde:

```text
outputs/25_visualize_predictions/saved_original/
outputs/25_visualize_predictions/saved_extended/
error analysis examples si existen
```

Conservar nombres de patch en los nombres de archivo.

---

## Crear `reports/final_results_index.md`

Generar índice automático:

```markdown
# Final Results Index

## Tables

- `reports/tables/...`

## Figures

- `reports/figures/...`

## Visual examples

- `reports/visual_examples/...`

## Methodology

- `reports/methodology/...`
```

---

## Crear `reports/pipeline_summary.md`

Debe incluir:

```markdown
# Pipeline Summary

## Dataset

- Nº de patches
- SI / NO
- Fechas únicas
- Fechas más representadas

## Baseline models

- U-Net MARIDA
- RF MARIDA
- ResNet MARIDA
- FDI
- NDVI
- FDI+NDVI

## XGBoost

- Resultado exploratorio
- Resultado grouped-by-date
- Diferencia e interpretación sobre leakage

## Classical models

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost

## External models

- Modelos probados
- Estado de ejecución
- Métricas si existen

## Hybrid model

- Detector seleccionado
- Estrategia de segmentación
- Métricas patch-level
- Métricas pixel-level
- Limitaciones

## Maps

- Dataset map
- Predictions map
- Confidence map
- Temporal map

## Ablation

- Mejor familia de features
- Efecto de SAM
- Efecto de GLCM

## Limitations

- Weak ground truth
- Pseudo-masks
- Temporal grouping
- Pixel-level limitations
- FDI/circularity risk
- Maps are not exhaustive
```

---

# Resumen de limitaciones

Crear o actualizar:

```text
reports/methodology/limitations_summary.md
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

## Modelo híbrido

El nivel high_confidence no equivale a confirmación material directa de plástico; solo indica alta confianza de la pipeline.
```

---

# Criterios de aceptación de la Fase E adaptada

Codex debe completar esta fase si:

1. No modifica la estructura ya correcta hasta el `22`.
2. Existe `scripts/23_build_hybrid_detector_segmenter.py`.
3. Existe `outputs/23_build_hybrid_detector_segmenter/hybrid_predictions.csv`.
4. Existe evaluación patch-level y pixel-level del híbrido.
5. Existe `scripts/24_generate_gibraltar_maps.py`.
6. Existen mapas HTML en `outputs/24_generate_gibraltar_maps/`.
7. Existe `scripts/25_visualize_predictions.py`.
8. `25_visualize_predictions.py --mode original` reproduce el comportamiento del antiguo visualizador.
9. `25_visualize_predictions.py --mode extended` añade modelos nuevos/híbrido y permite navegar patch por patch.
10. En toda visualización aparece RGB + pseudo-máscara + predicciones/scores.
11. Existe `scripts/26_export_reports.py`.
12. `26_export_reports.py` genera `reports/tables/`, `reports/figures/`, `reports/visual_examples/`.
13. Existe `reports/pipeline_summary.md`.
14. Existe `reports/final_results_index.md`.
15. Existe `reports/methodology/limitations_summary.md`.
16. Los mapas no se describen como cartografía exhaustiva de plásticos.
17. El híbrido no afirma confirmación material directa de plástico.
18. No se modifica el dataset base.
19. No se reescribe la lógica de descarga.
20. No se introducen nuevos modelos externos en esta fase; solo se usan outputs ya generados por scripts anteriores.

---

# No hacer en esta fase

No hacer:

```text
modificar scripts 00-22 salvo ajustes mínimos de compatibilidad
regenerar todo el dataset base
cambiar las pseudo-máscaras Nature-derived
entrenar modelos nuevos no previstos
hacer fine-tuning
introducir nuevos repositorios externos
borrar outputs antiguos sin archivarlos
```

Sí se permite:

```text
leer outputs existentes
combinar predicciones
crear máscaras híbridas
generar mapas
generar visualizaciones
exportar reportes
crear summaries metodológicos
```

---

# Nota final para Codex

Esta Fase E debe ser una capa final de integración, interpretación y presentación.

No debe convertir el TFG en otro proyecto nuevo.

El objetivo es que, a partir de los resultados ya calculados, se pueda responder claramente:

```text
1. Qué modelos funcionan mejor.
2. Qué limitaciones tiene el ground truth.
3. Qué detecciones son más fiables.
4. Dónde están espacialmente los casos analizados.
5. Qué tablas y figuras se pueden usar directamente en la memoria.
```
