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

# Bloque 1 — Metodología: weak labels, proxy ground truth y circularidad con FDI

## Objetivo

Mejorar la solidez metodológica del proyecto sin cambiar todavía los modelos. El dataset positivo procede del catálogo Nature/Zenodo de litter windrows, pero no debe tratarse como ground truth perfecto de plástico puro ni como máscara pixel-perfect independiente.

## Problema que hay que reflejar

Las etiquetas positivas actuales significan aproximadamente:

```text
SI = existe un filamento/windrow detectado en el catálogo Nature y se ha construido una pseudo-máscara derivada de su geometría.
```

Pero NO significan necesariamente:

```text
SI = plástico puro confirmado en cada píxel.
```

Por tanto, en documentación, nombres de outputs y comentarios, usar preferentemente:

```text
weak labels
proxy ground truth
Nature-derived labels
Nature-derived pseudo-masks
positive candidates
high-confidence positives
```

Evitar `ground truth` sin matices.

## Cambio 1 — Añadir documentación metodológica

Crear o actualizar un documento:

```text
docs/methodology_labels.md
```

Contenido mínimo:

1. Explicar que las etiquetas positivas son weak/proxy labels.
2. Explicar que las pseudo-máscaras derivan del catálogo Nature.
3. Explicar que la evaluación mide detección de estructuras compatibles con acumulaciones flotantes, no identificación química absoluta de plástico.
4. Explicar que los resultados deben interpretarse con cautela.
5. Explicar el posible sesgo/circularidad de FDI.

Texto sugerido:

```text
Dado que el catálogo empleado no proporciona máscaras de segmentación independientes ni confirmación material directa de la composición de cada filamento, las etiquetas utilizadas en este trabajo se consideran weak labels o proxy ground truth. Por ello, la evaluación debe interpretarse como una estimación de la capacidad de los modelos para detectar estructuras espectrales y espaciales compatibles con acumulaciones flotantes de residuos, no como una validación absoluta de plástico marino.
```

## Cambio 2 — Documentar circularidad FDI

En `01_download_dataset.py` la máscara positiva se alinea usando una referencia basada en FDI:

```python
def build_alignment_reference(data):
    fdi = compute_fdi(data)
    thr_fdi = threshold_mean_plus_3std(fdi)
```

Esto es aceptable como solución práctica de alineación, pero implica que evaluar FDI contra esa pseudo-máscara no es completamente independiente.

Añadir comentario/docstring cerca de esa lógica:

```text
Nota metodológica: esta alineación usa una referencia espectral basada en FDI. Por tanto, las métricas pixel-level de métodos FDI frente a esta pseudo-máscara deben considerarse exploratorias y no completamente independientes.
```

## Cambio 3 — Guardar desplazamiento de alineación

Modificar `build_mask()` en `01_download_dataset.py` para que devuelva también:

```text
dr
dc
alignment_shift_magnitude
```

Ejemplo de retorno:

```python
return mask, int(mask.sum()), int(dr), int(dc), float((dr ** 2 + dc ** 2) ** 0.5)
```

Si actualmente se esperaba:

```python
mask, n_pixels = build_mask(...)
```

actualizar las llamadas sin romper la lógica.

## Cambio 4 — Guardar metadatos de alineación

Crear o actualizar un CSV:

```text
results/auto/quality/positive_alignment_quality.csv
```

Columnas mínimas:

```text
patch
source_product
date
lat
lon
n_pixels_fil
mask_pixels
dr
dc
alignment_shift_magnitude
```

Si no hay algunas columnas disponibles, dejar `NaN` y no fallar.

## Criterios de aceptación

1. Existe `docs/methodology_labels.md`.
2. `01_download_dataset.py` sigue funcionando.
3. Las máscaras se siguen generando igual que antes.
4. Se guarda un CSV con desplazamientos de alineación.
5. El código incluye una nota clara sobre la circularidad FDI.

## Qué NO hacer en este bloque

- No entrenar modelos nuevos.
- No cambiar la lógica de U-Net, RF, ResNet o XGBoost.
- No rehacer la evaluación completa.
- No afirmar que Nature equivale a plástico puro.
