# Fase D — Comparación de modelos, features, SAM y GLCM

## Objetivo

Una vez consolidados resultados, dataset y validación para modelos propios, ampliar la comparación de métodos.

Esta fase incluye:

1. modelos preentrenados de literatura;
2. modelos clásicos propios;
3. features SAM;
4. features GLCM;
5. ablation;
6. comparación final.

---

# Parte 1 — Modelos preentrenados de literatura

## Objetivo

Integrar solo modelos externos que sean razonablemente fáciles de ejecutar y comparables con Sentinel-2.

No invertir demasiado tiempo si un repositorio tiene dependencias rotas o no ofrece pesos.

## Candidatos recomendados

### 1. marinedebrisdetector

Repositorio:

```text
https://github.com/marccoru/marinedebrisdetector
```

Razón:

- paquete orientado a detección de marine debris en Sentinel-2;
- incluye inferencia por CLI;
- parece el candidato externo más interesante.

### 2. MADOS / MariNeXt

Repositorio:

```text
https://github.com/gkakogeorgiou/mados
```

Razón:

- dataset Sentinel-2 para marine debris y oil spills;
- incluye modelos y pesos de runs;
- puede ser más difícil de integrar.

### 3. POS2IDON

Repositorio:

```text
https://github.com/AIRCentre/POS2IDON
```

Razón:

- pipeline basado en firmas espectrales y Random Forest;
- puede ser interesante como baseline clásico externo.

### 4. SADMA / ResAttUNet

Repositorio:

```text
https://github.com/sheikhazhanmohammed/SADMA
```

Razón:

- arquitectura basada en MARIDA;
- probar solo si hay pesos e inferencia clara.

## Script wrapper recomendado

Crear:

```text
src/external_models/20_run_external_model.py
```

Debe soportar una interfaz común:

```bash
python src/external_models/20_run_external_model.py \
  --model marinedebrisdetector \
  --input-dir results/auto/test_patches_final \
  --output-dir results/auto/external_models/marinedebrisdetector
```

## Outputs comunes

Para cualquier modelo externo:

```text
results/auto/external_models/<model_name>/predictions.csv
results/auto/external_models/<model_name>/masks/
results/auto/external_models/<model_name>/run_summary.md
```

`predictions.csv`:

```text
patch
method
score
pred_px
has_prediction
status
error_message
```

## Evaluación

Como estos modelos son preentrenados, no usar GroupKFold.

Evaluar igual que U-Net/RF/ResNet:

```text
global metrics
metrics by date
metrics by image_quality
pixel-level metrics si producen máscaras
```

---

# Parte 2 — Modelos clásicos propios

## Objetivo

Entrenar pocos modelos clásicos, no una lista enorme.

Modelos recomendados:

```text
Logistic Regression
Random Forest
HistGradientBoosting
XGBoost
```

Opcional:

```text
ExtraTrees
```

No implementar por defecto:

```text
Naive Bayes
Discriminant Analysis
Bagging
AdaBoost
Stacking complejo
```

A menos que ya esté todo estable.

## Script

Crear:

```text
src/models/21_train_classical_models_grouped_cv.py
```

## Reglas obligatorias

Como estos modelos se entrenan con nuestro dataset, sí deben usar validación agrupada por fecha:

```text
GroupKFold / LeaveOneGroupOut usando group_id = date
```

También:

- imputación dentro del fold;
- escalado dentro del fold si aplica;
- umbral seleccionado solo en train;
- métricas out-of-fold.

## Outputs

```text
results/auto/evaluation/classical_models_grouped_cv/fold_metrics.csv
results/auto/evaluation/classical_models_grouped_cv/out_of_fold_predictions.csv
results/auto/evaluation/classical_models_grouped_cv/summary_metrics.csv
results/auto/evaluation/classical_models_grouped_cv/classical_models_summary.md
```

---

# Parte 3 — Features SAM

## Objetivo

Usar Spectral Angle Mapper como feature o score de confianza, no como ground truth definitivo.

SAM puede ayudar a medir similitud espectral con:

1. firmas MARIDA Marine Debris;
2. firmas de agua limpia;
3. firmas de posibles confusores si están disponibles.

## Script

Crear:

```text
src/features/22_compute_sam_features.py
```

Entrada:

```text
patches Sentinel-2
MARIDA reference spectra, si están disponibles
dataset_metadata.csv
```

Salida:

```text
results/auto/features/sam_features.csv
```

Columnas sugeridas:

```text
patch
sam_debris_mean
sam_debris_min
sam_debris_p05
sam_water_mean
sam_water_min
sam_margin_debris_vs_water
sam_confidence_score
```

## Uso correcto

SAM no debe usarse para afirmar:

```text
esto es plástico confirmado
```

Debe usarse como:

```text
score espectral de similitud
feature auxiliar
criterio de confianza
```

---

# Parte 4 — Features GLCM

## Objetivo

Añadir textura porque los filamentos no son solo una firma espectral, también tienen estructura espacial.

## Script

Crear:

```text
src/features/23_compute_glcm_features.py
```

Entrada:

```text
patches
```

Salida:

```text
results/auto/features/glcm_features.csv
```

Features sugeridas:

```text
glcm_contrast_mean
glcm_dissimilarity_mean
glcm_homogeneity_mean
glcm_energy_mean
glcm_correlation_mean
glcm_asm_mean
```

Opcional:

```text
mean
std
p95
max
```

Calcular preferiblemente sobre una imagen grayscale generada desde bandas RGB o bandas relevantes.

---

# Parte 5 — Dataset de features unificado

Crear:

```text
src/features/24_build_feature_table_v2.py
```

Entrada:

```text
dataset_metadata.csv
outputs de U-Net/RF/ResNet/FDI/NDVI
sam_features.csv
glcm_features.csv
external_model_predictions.csv
```

Salida:

```text
results/auto/features/feature_table_v2.csv
```

Columnas por familias:

```text
spectral_*
index_*
marida_model_*
external_model_*
sam_*
glcm_*
quality_*
```

---

# Parte 6 — Ablation

Crear:

```text
src/evaluation/25_ablation_grouped_cv.py
```

Como ablation entrena modelos propios, debe usar GroupKFold por fecha.

Experimentos recomendados:

```text
spectral_only
indices_only
marida_models_only
sam_only
glcm_only
spectral_plus_indices
marida_plus_indices
all_without_fdi
all_without_resnet
all_without_sam
all_features
```

Salida:

```text
results/auto/evaluation/ablation_grouped_cv/ablation_results.csv
results/auto/evaluation/ablation_grouped_cv/ablation_summary.md
```

Métricas:

```text
AUC
F1
Precision
Recall
Balanced Accuracy
Brier Score, si hay probabilidades
```

---

# Criterios de aceptación

Codex debe completar esta fase si:

1. Hay wrapper o documentación clara para modelos externos.
2. Los modelos externos se evalúan sin GroupKFold, pero con análisis por fecha/calidad.
3. Los modelos clásicos propios usan GroupKFold por fecha.
4. Existen features SAM.
5. Existen features GLCM.
6. Existe `feature_table_v2.csv`.
7. Existe ablation con validación agrupada para modelos propios.

---

# No hacer en esta fase

- No fine-tuning.
- No mapas finales.
- No cambiar `01_download_dataset.py`.
- No crear un sistema híbrido todavía.
