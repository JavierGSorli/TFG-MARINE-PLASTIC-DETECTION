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

# Bloque 2 — Reetiquetado de negativos

## Objetivo

Mejorar la calidad de las etiquetas negativas. Las etiquetas actuales tipo `CLARO`, `DIFICIL`, `DUDOSO` mezclan dos dimensiones distintas:

1. confianza de que el patch sea realmente negativo;
2. tipo de dificultad visual/espectral.

Hay que separar ambas cosas.

## Nueva estructura de anotación

Cada patch negativo debe tener:

### 1. Decisión

```text
ACCEPT      -> negativo válido
REJECT      -> descartar patch
UNCERTAIN   -> dudoso; no usar en evaluación principal
```

### 2. Tipo de escena

Permitir una o varias etiquetas:

```text
clean_water
very_dark_water
thin_cloud
cloud
coast
ship
wake
foam
turbid_water
sunglint
possible_debris
bad_quality
other
```

### 3. Confianza

```text
1 = baja confianza
2 = media confianza
3 = alta confianza
```

### 4. Notas libres

Campo de texto opcional.

## Script a crear

```text
scripts/12_relabel_negatives.py
```

## Funcionalidad mínima

El script debe:

1. Buscar patches negativos en la carpeta configurada.
2. Mostrar cada patch en RGB.
3. Si es posible, mostrar también FDI/NDVI o una vista auxiliar.
4. Permitir al usuario seleccionar:
   - decisión: `ACCEPT`, `REJECT`, `UNCERTAIN`;
   - una o varias etiquetas de escena;
   - confianza 1/2/3;
   - notas opcionales.
5. Guardar las anotaciones en CSV.
6. Permitir continuar si ya existe un CSV previo.
7. No modificar nombres de archivos TIFF originales.

## Salida esperada

```text
results/auto/annotations/negative_annotations.csv
```

Columnas recomendadas:

```text
patch
original_label_from_filename
decision
scene_tags
confidence
has_cloud
has_thin_cloud
has_coast
has_ship
has_wake
has_foam
has_turbid_water
has_sunglint
has_possible_debris
is_very_dark
is_bad_quality
notes
annotated_at
```

`scene_tags` puede ser una cadena separada por `;`.

## Interfaz

Puede ser Tkinter, matplotlib interactivo o una interfaz simple por consola si es más robusta. Priorizar que funcione.

Atajos recomendados si se usa teclado:

```text
a = ACCEPT
r = REJECT
u = UNCERTAIN
1/2/3 = confianza
s = guardar y siguiente
q = salir guardando
```

## Criterios de aceptación

1. El script puede ejecutarse varias veces sin perder anotaciones anteriores.
2. Genera `negative_annotations.csv`.
3. Permite diferenciar confianza y tipo de dificultad.
4. No depende de modificar los nombres de los TIFF.
5. Si un patch ya está anotado, permite saltarlo o editarlo.

## Qué NO hacer en este bloque

- No reentrenar modelos.
- No cambiar evaluación.
- No borrar patches originales.
- No mezclar positivos y negativos en este script.
