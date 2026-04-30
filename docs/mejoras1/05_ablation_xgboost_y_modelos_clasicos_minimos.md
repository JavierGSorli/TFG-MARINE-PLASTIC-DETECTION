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

# Bloque 5 — Ablation de XGBoost y modelos clásicos mínimos

## Objetivo

Medir qué aporta cada familia de variables y comparar XGBoost con algunos modelos clásicos, sin convertir el TFG en una colección excesiva de modelos.

Este bloque debe ejecutarse después de tener validación agrupada sin leakage.

## Script 1 a crear

```text
scripts/19_ablation_xgboost.py
```

## Familias de features recomendadas

Detectar columnas por prefijos/nombres. Si no existen algunas, ignorarlas con warning.

```text
spectral_bands      -> estadísticas de bandas Sentinel-2
indices             -> FDI, NDVI, NDWI, FAI, etc.
marida_models       -> unet_px, rf_px, resnet_prob, etc.
texture_glcm        -> GLCM si existe
quality_features    -> cloud_frac, land_frac, confidence, etc. solo si se decide usarlas
```

## Experimentos mínimos

```text
A_spectral_only
B_indices_only
C_marida_models_only
D_spectral_plus_indices
E_indices_plus_marida
F_all_without_fdi
G_all_without_resnet
H_all_features
```

No fallar si falta una familia de features; registrar que se omite.

## Validación

Usar el mismo enfoque del bloque 4:

```text
GroupKFold / StratifiedGroupKFold
imputación dentro del fold
predicción out-of-fold
threshold sin leakage
```

## Salidas esperadas

```text
results/auto/ablation/ablation_results.csv
results/auto/ablation/ablation_oof_predictions.csv
results/auto/ablation/ablation_feature_groups.json
results/auto/ablation/ablation_barplot.png
```

Métricas:

```text
AUC_ROC
Average Precision
F1
Precision
Recall
Balanced Accuracy
Brier Score
```

## Interpretación esperada

El output debe permitir responder preguntas como:

```text
¿Los modelos MARIDA aportan más que los índices?
¿Qué pasa si quitamos FDI?
¿Qué pasa si quitamos ResNet?
¿Las bandas solas son suficientes?
¿El modelo combinado mejora a los componentes individuales?
```

## Script 2 a crear

```text
scripts/20_train_classical_models_grouped_cv.py
```

## Modelos recomendados

No implementar una lista enorme. Usar solo:

```text
LogisticRegression
RandomForestClassifier
HistGradientBoostingClassifier
XGBoost si está disponible
```

Opcional:

```text
ExtraTreesClassifier
```

No implementar inicialmente:

```text
Naive Bayes
Discriminant Analysis
Bagging
AdaBoost
Stacking
```

Motivo: con pocas muestras y weak labels, añadir muchos modelos aumenta riesgo de sobreajuste y complica la defensa.

## Salidas esperadas

```text
results/auto/classical_models/classical_grouped_cv_metrics.csv
results/auto/classical_models/classical_oof_predictions.csv
```

Columnas de métricas:

```text
model
feature_set
n_samples
n_features
n_folds
auc_roc
average_precision
f1
precision
recall
balanced_accuracy
brier_score
```

## Criterios de aceptación

1. Ablation usa validación agrupada.
2. Cada experimento indica exactamente qué features usa.
3. Se guardan métricas y predicciones out-of-fold.
4. Los modelos clásicos son pocos y comparables.
5. No se reportan métricas de random split como resultado principal.

## Qué NO hacer en este bloque

- No crear stacking todavía.
- No hacer fine-tuning.
- No usar todas las features sin explicar su origen.
- No incluir variables que filtren directamente la etiqueta, como nombres de archivo que contengan `SI/NO`.
