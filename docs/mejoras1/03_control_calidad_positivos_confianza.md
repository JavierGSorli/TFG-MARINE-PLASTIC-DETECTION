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

# Bloque 3 — Control de calidad de positivos y scoring de confianza

## Objetivo

Crear un sistema de calidad/confianza para positivos y negativos. Esto es clave porque las etiquetas son weak/proxy labels. La idea es poder evaluar por subconjuntos:

```text
alta confianza
media confianza
baja confianza
negativos claros
negativos difíciles
positivos grandes
positivos pequeños
```

## Script 1 a crear

```text
scripts/13_positive_quality_audit.py
```

## Funcionalidad

Para cada patch positivo, calcular métricas de calidad.

Columnas recomendadas:

```text
patch
date
lat
lon
n_pixels_fil
mask_pixels
cloud_frac
land_frac
water_frac
valid_px_frac
dark_frac
alignment_shift_row
alignment_shift_col
alignment_shift_magnitude
positive_confidence
quality_flags
```

Si alguna métrica no se puede calcular por falta de datos, no fallar; usar `NaN` y registrar warning.

## Posibles métricas

### De máscara

```text
mask_pixels
n_pixels_fil
```

### De escena

```text
cloud_frac
land_frac
water_frac
valid_px_frac
dark_frac
```

Usar SCL si está disponible. Si no hay SCL en los patches finales, documentar limitación.

### De alineación

```text
alignment_shift_magnitude
```

Debe venir del bloque 1, si se ha guardado.

## Regla inicial de confianza para positivos

Implementar una regla sencilla y editable:

```text
P3 = alta confianza
P2 = media confianza
P1 = baja confianza
```

Ejemplo inicial:

```python
if cloud_frac <= 0.10 and land_frac <= 0.20 and mask_pixels >= 10 and alignment_shift_magnitude <= 8:
    positive_confidence = "P3"
elif cloud_frac <= 0.20 and land_frac <= 0.35 and mask_pixels >= 3:
    positive_confidence = "P2"
else:
    positive_confidence = "P1"
```

Los umbrales deben estar en `config.py` o al inicio del script.

## Script 2 a crear

```text
scripts/14_build_patch_confidence.py
```

## Funcionalidad

Combinar:

1. calidad de positivos (`positive_quality.csv`);
2. anotaciones de negativos (`negative_annotations.csv`);
3. predicciones existentes si están disponibles (`unified_predictions.csv` o similar);
4. métricas de calidad ya generadas por otros scripts.

Salida:

```text
results/auto/confidence/patch_confidence.csv
```

Columnas recomendadas:

```text
patch
label
confidence_level
manual_confidence
quality_score
scene_tags
usable_for_main_eval
usable_for_stress_test
reason_excluded_main_eval
```

## Reglas sugeridas

### Para positivos

```text
P3 -> usable_for_main_eval = True
P2 -> usable_for_main_eval = True
P1 -> usable_for_main_eval = False, pero usable_for_stress_test = True
```

### Para negativos

```text
ACCEPT + confidence=3 + clean_water -> N3
ACCEPT + confidence=2 o escena difícil -> N2
UNCERTAIN -> N1, fuera de main eval
REJECT -> excluir de evaluación
```

## Script 3 a crear

```text
scripts/15_evaluate_by_confidence.py
```

## Evaluaciones necesarias

Generar métricas para:

```text
all_data
main_eval_only
high_confidence_only
positives_P3_vs_negatives_N3
hard_negatives_only
small_positives
large_positives
```

Métricas:

```text
AUC_ROC
Average Precision
F1
Precision
Recall
Balanced Accuracy
Confusion matrix
```

## Salidas esperadas

```text
results/auto/confidence/positive_quality.csv
results/auto/confidence/patch_confidence.csv
results/auto/confidence/evaluation_by_confidence.csv
results/auto/confidence/confusion_matrices_by_confidence.json
```

## Criterios de aceptación

1. Se puede saber qué patches se usan en evaluación principal y cuáles no.
2. Hay una columna clara `usable_for_main_eval`.
3. Las reglas de confianza son modificables.
4. Se generan métricas separadas por nivel de confianza.
5. El script no borra resultados previos.

## Qué NO hacer en este bloque

- No entrenar modelos nuevos.
- No cambiar aún XGBoost.
- No usar estas reglas para afirmar que un patch es plástico puro.
