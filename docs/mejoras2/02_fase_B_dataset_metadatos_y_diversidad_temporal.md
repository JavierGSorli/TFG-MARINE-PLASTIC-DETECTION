# Fase B — Dataset, metadatos, anotaciones y diversidad temporal

## Objetivo

Mejorar la calidad del dataset sin introducir una lógica compleja de “modos” de negativos.

La idea principal es:

1. dejar de depender del nombre del archivo como única fuente de información;
2. crear `dataset_metadata.csv`;
3. añadir anotaciones manuales y calidad de imagen;
4. ampliar el rango temporal y reducir la concentración de patches en pocas fechas;
5. mantener compatibilidad con los scripts existentes.

---

# Principios

## No implementar muchos modos de negativos

No crear una arquitectura con modos como:

```text
matched_same_date
matched_same_month
far_spatial
hard_negative_manual
random_different_date
```

Esto complica el TFG.

En su lugar, usar una estrategia sencilla:

```text
ampliar rango temporal
limitar máximo de patches por fecha
intentar aumentar número de fechas únicas
mantener negativos comparables a los positivos
```

---

## No renombrar todo de golpe

No cambiar masivamente los nombres de los `.tif`.

Crear primero un CSV central:

```text
data/processed/dataset_metadata.csv
```

Este CSV debe convertirse en la fuente de verdad.

Los nombres antiguos deben seguir funcionando.

---

# Tarea 1 — Crear dataset_metadata.csv

Crear script:

```text
src/data/14_build_dataset_metadata.py
```

Entrada:

```text
results/auto/test_patches_final/
results/auto/test_masks_unet/
results/auto/test_masks_rf/
results/auto/test_indices/
```

o las carpetas actuales equivalentes.

Salida:

```text
data/processed/dataset_metadata.csv
```

Columnas mínimas:

```text
patch
patch_path
mask_path
date
year
month
label
label_binary
expected_gt_px
mask_gt_px
name_mask_match
original_difficulty
is_positive
is_negative
```

Donde:

- `label` debe ser `SI` o `NO`.
- `label_binary` debe ser `1` para SI y `0` para NO.
- `expected_gt_px` se extrae del nombre si existe.
- `mask_gt_px` se calcula desde la máscara.
- `original_difficulty` se extrae de nombres antiguos tipo `CLARO`, `DIFICIL`, `DUDOSO`, si existe.

---

# Tarea 2 — Añadir campos manuales para anotación

Ampliar `dataset_metadata.csv` con columnas vacías o por defecto:

```text
manual_decision
manual_confidence
image_quality
scene_tags
has_cloud
has_thin_cloud
has_wake
has_ship
has_coast
has_sunglint
has_dark_water
has_turbid_water
has_possible_debris
notes
annotated_at
```

Valores recomendados:

## manual_decision

```text
accept
reject
uncertain
```

## manual_confidence

```text
1 = baja
2 = media
3 = alta
```

## image_quality

```text
good
medium
bad
```

## scene_tags

Lista separada por `;`, por ejemplo:

```text
clean_water;thin_cloud
wake;ship
dark_water;turbid_water
coast;sunglint
```

---

# Tarea 3 — Herramienta sencilla de anotación

Crear:

```text
src/data/15_annotate_patches_quality.py
```

Objetivo:

Permitir revisar patches y actualizar `dataset_metadata.csv`.

No hace falta una interfaz perfecta. Puede ser Tkinter, matplotlib interactivo o CLI visual simple.

Debe mostrar:

1. RGB del patch.
2. Opcionalmente FDI/NDVI.
3. Nombre del patch.
4. Label SI/NO.
5. Fecha.
6. Dificultad original, si existe.

Debe permitir seleccionar:

```text
manual_decision
manual_confidence
image_quality
scene_tags
notes
```

## Requisito importante

No modificar el nombre del archivo desde esta herramienta.

Solo actualizar `dataset_metadata.csv`.

---

# Tarea 4 — Cambio sencillo en 01_download_dataset.py para ampliar diversidad temporal

Modificar `01_download_dataset.py` de forma mínima.

Añadir parámetros configurables:

```python
TARGET_SI = 100
TARGET_NO = 100
MAX_PATCHES_PER_DATE = 4
MAX_SI_PER_DATE = 4
MAX_NO_PER_DATE = 4
MIN_UNIQUE_DATES_TARGET = 25
```

Estos valores deben poder ajustarse fácilmente al principio del script.

## Objetivo

Evitar que una sola fecha, por ejemplo `20190414`, domine el dataset.

## Lógica recomendada

Cuando se seleccionan candidatos positivos Nature:

1. Extraer `date`.
2. Agrupar por fecha.
3. Seleccionar candidatos intentando respetar `MAX_SI_PER_DATE`.
4. Priorizar diversidad temporal antes que coger demasiados de una sola fecha.
5. Si no hay suficientes fechas, permitir superar el máximo, pero emitir warning.

Pseudocódigo:

```python
selected = []
counts_by_date = {}

for candidate in sorted_candidates:
    date = candidate["date"]

    if counts_by_date.get(date, 0) >= MAX_SI_PER_DATE:
        continue

    selected.append(candidate)
    counts_by_date[date] = counts_by_date.get(date, 0) + 1

    if len(selected) >= TARGET_SI:
        break

if len(selected) < TARGET_SI:
    # segunda pasada permitiendo más por fecha
    # pero guardando warning
```

## Para negativos

Mantener la lógica actual de generación de negativos, pero añadir control de concentración:

```python
if no_counts_by_date[date] >= MAX_NO_PER_DATE:
    intentar otro candidato / otra fecha
```

No crear modos nuevos.

---

# Tarea 5 — Guardar resumen de distribución temporal tras descarga

Al final de `01_download_dataset.py`, guardar:

```text
results/auto/dataset_generation/date_distribution_after_download.csv
results/auto/dataset_generation/dataset_generation_summary.md
```

`date_distribution_after_download.csv`:

```text
date
n_si
n_no
n_total
```

`dataset_generation_summary.md`:

```text
# Dataset generation summary

## Target
TARGET_SI
TARGET_NO
MAX_PATCHES_PER_DATE
MAX_SI_PER_DATE
MAX_NO_PER_DATE

## Actual
n_si
n_no
n_unique_dates
max_patches_same_date

## Warnings
- Fechas que superan el máximo.
- No se alcanzó el target.
- Candidatos descartados por nube/tierra.
```

---

# Criterios de aceptación

Codex debe completar esta fase si:

1. Existe `dataset_metadata.csv`.
2. Los nombres antiguos siguen funcionando.
3. Existe herramienta de anotación.
4. `01_download_dataset.py` permite limitar concentración por fecha.
5. No se han creado modos complejos de negativos.
6. Se guarda resumen temporal tras generar dataset.

---

# No hacer en esta fase

- No entrenar XGBoost.
- No añadir modelos nuevos.
- No hacer GroupKFold.
- No hacer fine-tuning.
- No renombrar masivamente los `.tif`.
