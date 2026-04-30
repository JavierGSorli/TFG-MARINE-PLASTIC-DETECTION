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

# Bloque 4 — Validación sin leakage temporal, espacial o por evento

## Objetivo

Corregir la evaluación de modelos entrenados sobre features, especialmente XGBoost, para evitar resultados artificialmente optimistas.

## Riesgos a evitar

No usar random split como métrica principal porque puede introducir:

```text
leakage temporal: misma fecha en train y test
leakage espacial: patches cercanos en train y test
leakage por evento: variantes del mismo filamento en train y test
leakage por preprocesado: imputación/normalización antes del split
leakage por umbral: elegir el umbral usando todo el dataset
```

## Script 1 a crear

```text
scripts/16_build_validation_groups.py
```

## Funcionalidad

Crear un CSV con grupos de validación.

Salida:

```text
results/auto/splits/validation_groups.csv
```

Columnas recomendadas:

```text
patch
label
date
year
month
lat
lon
spatial_cell
event_group
date_group
final_group
usable_for_main_eval
```

## Reglas iniciales

Si no hay lat/lon fiables, usar fecha:

```python
final_group = date
```

Si hay lat/lon, crear celda espacial aproximada:

```python
spatial_cell = f"{round(lat, 1)}_{round(lon, 1)}"
```

Y un grupo combinado:

```python
final_group = f"{date}_{spatial_cell}"
```

Pero mantenerlo simple y documentado.

## Script 2 a crear

```text
scripts/17_train_xgboost_grouped_cv.py
```

## Cambios frente a `08_train_xgboost.py`

1. Usar `GroupKFold`, `StratifiedGroupKFold` si está disponible o estrategia equivalente.
2. Imputar valores dentro de cada fold, no antes.
3. Escalar dentro de cada fold si se usa un modelo que lo necesita.
4. Entrenar en train fold y predecir en test fold.
5. Guardar predicciones out-of-fold.
6. Elegir umbral usando solo train fold o usar umbral fijo previamente definido.
7. Reportar métricas agregadas fuera de muestra.

## Pipeline recomendado

Usar `sklearn.pipeline.Pipeline`:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(...))
])
```

Si XGBoost no está instalado, usar `HistGradientBoostingClassifier` o fallback documentado.

## Selección de umbral

No elegir el mejor umbral sobre todo el dataset para la métrica principal.

Opciones válidas:

### Opción A — Umbral fijo

```text
threshold = 0.5
```

### Opción B — Umbral elegido en train de cada fold

Elegir el umbral que maximiza F1 en train y aplicarlo al test fold.

Guardar el umbral usado por fold.

## Salidas esperadas

```text
results/auto/grouped_cv/xgb_oof_predictions.csv
results/auto/grouped_cv/xgb_grouped_cv_metrics.csv
results/auto/grouped_cv/xgb_grouped_cv_thresholds.csv
results/auto/grouped_cv/xgb_final_model.joblib
```

Columnas de predicciones:

```text
patch
label
group
fold
y_true
y_prob
y_pred
threshold
usable_for_main_eval
confidence_level
```

## Script 3 a crear o modificar

```text
scripts/18_evaluate_grouped_predictions.py
```

Debe evaluar las predicciones out-of-fold y generar:

```text
AUC_ROC
Average Precision
F1
Precision
Recall
Balanced Accuracy
Brier Score
confusion matrix
```

Separar:

```text
main_eval_only
all_data
high_confidence_only
```

## Criterios de aceptación

1. Existe CSV de grupos.
2. XGBoost se evalúa con grupos, no random split.
3. La imputación ocurre dentro del fold.
4. Hay predicciones out-of-fold por patch.
5. El umbral no se selecciona usando todo el dataset.
6. La evaluación principal usa `usable_for_main_eval=True` si existe.

## Qué NO hacer en este bloque

- No añadir 10 modelos clásicos todavía.
- No hacer fine-tuning.
- No cambiar las máscaras.
- No reportar random split como resultado principal.
