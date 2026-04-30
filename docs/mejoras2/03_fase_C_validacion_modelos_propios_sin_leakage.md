# Fase C — Validación sin leakage para modelos propios

## Objetivo

Implementar validación rigurosa solo para los modelos que se entrenan o ajustan usando nuestro dataset.

Esto aplica a:

- XGBoost;
- modelos clásicos propios;
- stacking;
- calibración aprendida;
- fine-tuning;
- selección de umbrales aprendida con los datos.

No aplica del mismo modo a:

- U-Net MARIDA preentrenada;
- RF MARIDA preentrenado;
- ResNet MARIDA preentrenado;
- FDI;
- NDVI;
- FDI+NDVI.

Para estos últimos, no hay leakage de entrenamiento porque no se entrenan con este dataset. Aun así, se recomienda reportar resultados por fecha/calidad para comprobar si una fecha domina la métrica global.

---

# Problema que se quiere evitar

Si se entrena XGBoost con features de los mismos patches que luego se evalúan, o si patches de la misma fecha aparecen en train y test, las métricas pueden ser demasiado optimistas.

Ejemplo:

```text
20190414_SI_000109_71.tif en train
20190414_SI_000119_68.tif en test
```

Aunque sean patches distintos, pueden compartir:

- misma escena Sentinel-2;
- mismas condiciones atmosféricas;
- mismo estado del mar;
- mismo producto;
- mismo tipo de ruido;
- posible evento relacionado.

Por eso, para modelos propios, se debe agrupar por fecha.

---

# Tarea 1 — Crear columna group_id

Crear:

```text
src/evaluation/16_build_group_ids.py
```

Entrada:

```text
data/processed/dataset_metadata.csv
```

Salida:

```text
data/processed/dataset_metadata_with_groups.csv
```

Añadir columnas:

```text
group_date
group_month
group_year
group_id
```

Para empezar:

```text
group_id = date
```

Si existe producto Sentinel-2, opcionalmente:

```text
group_id = s2_product
```

Pero no hacerlo obligatorio.

---

# Tarea 2 — Splitter para modelos propios

Crear:

```text
src/evaluation/17_grouped_splits.py
```

Debe implementar:

```text
GroupKFold por fecha
LeaveOneGroupOut opcional
```

Recomendación:

- si hay suficientes fechas únicas, usar `GroupKFold`;
- si hay pocas fechas, usar `LeaveOneGroupOut`;
- si algún fold queda sin positivos o sin negativos, emitir warning y saltar ese fold.

Guardar:

```text
results/auto/evaluation/grouped_splits/folds.csv
results/auto/evaluation/grouped_splits/split_summary.md
```

`folds.csv`:

```text
patch
date
group_id
fold
split
label
```

`split_summary.md`:

```text
# Grouped split summary

## Groups
n_groups
groups_used
groups_skipped

## Folds
fold
n_train
n_test
train_si
train_no
test_si
test_no
test_dates
```

---

# Tarea 3 — Reentrenar XGBoost con validación agrupada

Crear o modificar:

```text
src/models/18_train_xgboost_grouped_cv.py
```

Entrada:

```text
data/processed/dataset_metadata_with_groups.csv
results/auto/xgboost_dataset.csv
```

o el CSV actual de features.

## Reglas obligatorias

1. No imputar valores usando todo el dataset antes del split.
2. La imputación debe aprenderse solo en train.
3. Cualquier escalado debe aprenderse solo en train.
4. El umbral de decisión debe seleccionarse solo en train.
5. El test fold solo se usa para evaluar.
6. No mezclar misma fecha entre train y test.

Usar `Pipeline` de sklearn cuando sea posible.

## Outputs

```text
results/auto/evaluation/xgboost_grouped_cv/fold_metrics.csv
results/auto/evaluation/xgboost_grouped_cv/out_of_fold_predictions.csv
results/auto/evaluation/xgboost_grouped_cv/summary_metrics.csv
results/auto/evaluation/xgboost_grouped_cv/xgboost_grouped_cv_summary.md
```

`out_of_fold_predictions.csv`:

```text
patch
date
group_id
label
y_true
y_prob
y_pred
fold
threshold_used
```

`fold_metrics.csv`:

```text
fold
n_train
n_test
test_dates
auc
precision
recall
f1
balanced_accuracy
tp
fp
tn
fn
threshold
```

---

# Tarea 4 — Comparar XGBoost exploratorio vs XGBoost grouped

Crear:

```text
src/evaluation/19_compare_xgboost_exploratory_vs_grouped.py
```

Salida:

```text
results/auto/evaluation/xgboost_grouped_cv/comparison_exploratory_vs_grouped.md
```

Debe incluir:

```text
# XGBoost evaluation comparison

## Exploratory evaluation
Métricas originales.

## Grouped-by-date evaluation
Métricas out-of-fold.

## Interpretation
Si grouped baja respecto a exploratory, indicarlo como señal esperable de evaluación más conservadora.
```

---

# Tarea 5 — Aplicar lógica también a futuros modelos propios

Crear una utilidad común:

```text
src/evaluation/model_validation_utils.py
```

Funciones sugeridas:

```python
extract_date_from_patch_name(patch_name)
make_grouped_cv_splits(metadata, group_col="group_id")
select_threshold_on_train(y_train, prob_train, metric="f1")
evaluate_binary_predictions(y_true, y_prob, threshold)
```

Esto se reutilizará en Fase D para modelos clásicos, SAM/GLCM, stacking, etc.

---

# Importante sobre modelos preentrenados e índices

No aplicar GroupKFold a:

```text
UNet MARIDA
RF MARIDA
ResNet MARIDA
FDI
NDVI
FDI+NDVI
```

Para estos métodos solo generar:

```text
global_metrics
metrics_by_date
metrics_by_quality
```

La razón:

Estos métodos no se entrenan con tu dataset, por tanto no hay leakage de entrenamiento. El análisis por fecha es útil para interpretación, no como split de entrenamiento.

---

# Criterios de aceptación

Codex debe completar esta fase si:

1. Existe `dataset_metadata_with_groups.csv`.
2. Existe un sistema de splits agrupados por fecha.
3. XGBoost se evalúa con out-of-fold predictions.
4. El umbral se selecciona dentro de train.
5. Se compara resultado exploratorio vs grouped.
6. La lógica queda reutilizable para futuros modelos propios.
7. No se aplica GroupKFold a modelos preentrenados o índices.

---

# No hacer en esta fase

- No añadir modelos clásicos todavía.
- No añadir SAM/GLCM todavía.
- No cambiar generación de dataset.
- No fine-tuning.
- No mapas.
