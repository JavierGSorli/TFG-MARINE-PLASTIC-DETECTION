# Fase A — Consolidar resultados actuales

## Objetivo

Antes de mejorar el dataset o añadir nuevos modelos, consolidar los resultados actuales para que sean claros, reproducibles y fáciles de interpretar.

Ahora mismo ya existen resultados interesantes:

- U-Net MARIDA y RF MARIDA funcionan bien a nivel patch-level.
- FDI tiene alto recall, pero muchos falsos positivos.
- NDVI y FDI+NDVI parecen degenerar en clasificación positiva casi universal.
- XGBoost tiene resultados muy altos, pero todavía debe tratarse como resultado exploratorio si no se ha validado con cuidado.
- La segmentación pixel-level es bastante más débil que la clasificación patch-level.
- Hay posible confusión en el análisis de errores porque se exportan solo algunos ejemplos aunque existan más FP/FN totales.

Esta fase no debe modificar modelos. Solo debe mejorar el análisis y la organización de resultados.

---

# Tareas

## 1. Separar resultados patch-level y pixel-level

Crear o modificar un script:

```text
src/evaluation/11_consolidate_current_results.py
```

Debe leer los CSV actuales de evaluación y generar una carpeta:

```text
results/auto/evaluation/baseline_v1/
```

Con estos archivos:

```text
patch_level_metrics.csv
pixel_level_segmentation_metrics.csv
false_positives_by_difficulty.csv
segmentation_noise_on_negatives.csv
baseline_v1_summary.md
```

## Reglas

- No mezclar patch-level y pixel-level en la misma tabla final.
- Patch-level mide si el patch se clasifica como `SI` o `NO`.
- Pixel-level mide coincidencia espacial con la pseudo-máscara.
- Usar términos como `pseudo_mask` o `nature_derived_mask`, no `perfect_ground_truth`.

---

## 2. Corregir salida del análisis de errores

Modificar el script actual de error analysis o crear:

```text
src/evaluation/12_error_analysis_v2.py
```

Problema actual:

La tabla comparativa dice, por ejemplo:

```text
FDI: FP = 31
```

pero el export de imágenes muestra:

```text
[FDI] FP=10
```

Probablemente se están exportando solo los primeros 10 ejemplos.

## Cambio requerido

La salida debe distinguir:

```text
total_fp
total_fn
exported_fp_examples
exported_fn_examples
```

Ejemplo de salida correcta:

```text
[FDI]
Total FP = 31
Total FN = 8
Exported FP examples = 10
Exported FN examples = 8
```

Guardar un CSV:

```text
results/auto/evaluation/baseline_v1/error_examples_summary.csv
```

Columnas:

```text
method
total_fp
total_fn
exported_fp_examples
exported_fn_examples
export_dir
```

---

## 3. Análisis por fecha

Crear:

```text
src/evaluation/13_analysis_by_date.py
```

Objetivo:

Medir si el dataset está concentrado en pocas fechas y si algunos resultados están dominados por una fecha concreta.

Entrada:

- CSV de evaluación patch-level.
- Lista de patches.
- Nombres de archivos con fecha en formato `YYYYMMDD`.

Salida:

```text
results/auto/evaluation/baseline_v1/date_distribution.csv
results/auto/evaluation/baseline_v1/metrics_by_date.csv
results/auto/evaluation/baseline_v1/top_dates_summary.csv
```

`date_distribution.csv` debe incluir:

```text
date
n_total
n_si
n_no
pct_total
```

`metrics_by_date.csv` debe incluir, para cada método y fecha:

```text
date
method
n_total
n_si
n_no
tp
fp
tn
fn
precision
recall
f1
fp_rate
```

`top_dates_summary.csv` debe incluir:

```text
n_unique_dates
max_patches_same_date
top_5_dates_by_total
top_5_dates_by_si
top_5_dates_by_no
```

## Importante

Este análisis por fecha no implica todavía GroupKFold.  
Solo es un diagnóstico descriptivo.

---

## 4. Resumen interpretativo automático

Generar:

```text
results/auto/evaluation/baseline_v1/baseline_v1_summary.md
```

Debe incluir secciones:

```text
# Baseline v1 — Resumen de resultados actuales

## Dataset
- Número de patches.
- SI / NO.
- Fechas únicas.
- Fechas más representadas.

## Patch-level detection
- Tabla comparativa.
- Lectura de U-Net, RF, FDI, NDVI, ResNet, XGBoost.

## Pixel-level segmentation
- Tabla de segmentación.
- Diferencia entre detección patch-level y localización pixel-level.

## Error analysis
- Falsos positivos por dificultad.
- Ruido de segmentación en negativos.
- Nota sobre ejemplos exportados.

## Main observations
- U-Net/RF son fuertes como detectores patch-level.
- FDI tiene alto recall pero muchos FP.
- NDVI/FDI+NDVI no discriminan bien en la configuración actual.
- XGBoost es prometedor pero debe validarse mejor si está entrenado con estos datos.
- La segmentación pixel-level debe interpretarse con cautela por la naturaleza de las pseudo-máscaras.
```

---

# Criterios de aceptación

Codex debe completar esta fase si:

1. Existe una carpeta `baseline_v1`.
2. Se generan CSV separados para patch-level y pixel-level.
3. El análisis de errores distingue total de errores y ejemplos exportados.
4. Existe análisis por fecha.
5. Existe un `.md` resumen de baseline v1.
6. No se han modificado modelos ni generación de dataset.

---

# No hacer en esta fase

- No cambiar `01_download_dataset.py`.
- No reentrenar XGBoost.
- No añadir modelos nuevos.
- No aplicar GroupKFold todavía.
- No renombrar archivos.
