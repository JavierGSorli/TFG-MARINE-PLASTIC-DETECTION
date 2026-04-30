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

# Bloque 6 — SAM y GLCM/texturas

## Objetivo

Añadir dos mejoras de prioridad media:

1. `SAM — Spectral Angle Mapper` como score auxiliar de confianza espectral.
2. `GLCM/texturas` como features adicionales para XGBoost/ablation.

No usar estas mejoras para afirmar plástico puro. Deben interpretarse como indicadores auxiliares.

---

# Parte A — SAM

## Script a crear

```text
scripts/21_sam_confidence.py
```

## Idea

Calcular similitud espectral entre píxeles/máscaras del proyecto y firmas de referencia.

Referencias posibles:

```text
MARIDA Marine Debris -> referencia más independiente, si está disponible.
Nature positives -> referencia local, pero menos independiente.
Water/background -> referencia negativa.
```

## Recomendación

Calcular al menos:

```text
sam_marida_debris_mean
sam_marida_debris_min
sam_water_mean
sam_margin_debris_vs_water
sam_confidence
```

Donde un menor ángulo SAM implica mayor similitud espectral.

## Entradas

```text
patches positivos con pseudo-máscara
patches negativos o agua limpia
MARIDA si está disponible
```

Si MARIDA no está disponible, el script debe fallar con mensaje claro o permitir modo Nature-only.

## Salidas

```text
results/auto/sam/sam_reference_marida.npy
results/auto/sam/sam_scores.csv
results/auto/sam/sam_filtered_positive_subset.csv
```

Columnas recomendadas:

```text
patch
label
sam_marida_debris_mean
sam_marida_debris_min
sam_water_mean
sam_margin_debris_vs_water
sam_confidence
```

## Uso recomendado

Usar SAM para:

```text
1. crear un score de confianza espectral;
2. filtrar positivos de alta confianza;
3. analizar si los modelos funcionan mejor en positivos espectralmente similares a MARIDA;
4. añadir features a XGBoost si procede.
```

No usar SAM para:

```text
decir que un filamento es plástico puro confirmado.
```

---

# Parte B — GLCM/texturas

## Script a crear

```text
scripts/22_build_texture_features.py
```

## Justificación

Los filamentos son estructuras espaciales. Las texturas pueden ayudar a distinguir agua homogénea, estelas, espuma, nubes y posibles residuos.

## Features sugeridas

Calcular GLCM sobre una imagen en escala de grises derivada del RGB o bandas visibles.

Features:

```text
glcm_contrast_mean
glcm_dissimilarity_mean
glcm_homogeneity_mean
glcm_energy_mean
glcm_correlation_mean
glcm_asm_mean
```

Opcionalmente percentiles:

```text
glcm_contrast_p95
glcm_homogeneity_p05
```

## Precauciones

1. No calcular texturas sobre zonas nodata.
2. Cuidar normalización a niveles discretos.
3. No hacer demasiado lento el script.
4. Añadir estas features como bloque separado en la ablación.

## Salida esperada

```text
results/auto/features/texture_features.csv
```

Columnas:

```text
patch
glcm_contrast_mean
glcm_dissimilarity_mean
glcm_homogeneity_mean
glcm_energy_mean
glcm_correlation_mean
glcm_asm_mean
```

## Integración con XGBoost

Modificar o crear una versión nueva de construcción de dataset:

```text
scripts/23_build_xgboost_dataset_v2.py
```

Debe unir:

```text
features actuales
sam_scores.csv si existe
texture_features.csv si existe
patch_confidence.csv si existe
```

Salida:

```text
results/auto/xgboost/xgboost_dataset_v2.csv
```

## Criterios de aceptación

1. SAM genera scores por patch.
2. GLCM genera features por patch.
3. Ambas integraciones son opcionales: si no existen, la pipeline principal no falla.
4. Las nuevas features se pueden incluir/excluir en ablation.

## Qué NO hacer en este bloque

- No entrenar U-Net nueva.
- No hacer fine-tuning.
- No afirmar que SAM valida materialmente el plástico.
- No mezclar referencias MARIDA y Nature sin dejar columnas separadas.
