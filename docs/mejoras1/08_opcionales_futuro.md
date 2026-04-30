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

# Bloque 8 — Opcionales y future work

Este bloque contiene ideas que pueden ser interesantes, pero no deberían hacerse antes de completar los bloques principales.

---

# 1. Calibración de ResNet/XGBoost

## Script opcional

```text
scripts/26_calibration_analysis.py
```

## Cuándo tiene sentido

Tiene sentido si las probabilidades se usan para:

```text
ranking
mapas de confianza
umbrales operativos
modelo híbrido
```

No es imprescindible si ResNet solo se usa como etiqueta binaria.

## Salidas

```text
results/auto/calibration/calibration_resnet.png
results/auto/calibration/calibration_xgb.png
results/auto/calibration/brier_scores.csv
```

## Precaución

Con pocas muestras y weak labels, la calibración debe presentarse como exploratoria.

---

# 2. Fine-tuning

## Recomendación

No hacer fine-tuning salvo que ya existan:

```text
positivos P3
negativos N3
hard negatives bien anotados
validación agrupada sin leakage
```

## Modelo recomendado si se hace

Empezar por ResNet patch-level, no por U-Net de segmentación.

Motivo: las máscaras positivas son pseudo-máscaras débiles. Fine-tuning de segmentación puede aprender ruido.

## Script opcional

```text
scripts/27_finetune_resnet_high_confidence.py
```

## Estrategia

```text
1. Congelar backbone.
2. Entrenar solo cabeza de clasificación.
3. Validar con GroupKFold o leave-one-date-out.
4. Comparar contra ResNet preentrenada sin fine-tuning.
```

---

# 3. Barcos y hard negatives

## Idea

Añadir barcos como clase de confusión/hard negatives.

## Recomendación

Priorizar:

```text
1. clases Ship/Wake/Foam de MARIDA si están disponibles;
2. patches Sentinel-2 propios con barcos visibles;
3. evitar datasets externos de otro sensor/resolución salvo justificación clara.
```

## Riesgo

Datasets como barcos en SAR o imágenes ópticas de alta resolución pueden introducir domain shift fuerte respecto a Sentinel-2 multiespectral a 10 m.

## Uso correcto

Usar barcos como:

```text
hard negatives
análisis de falsos positivos
stress test
```

No usarlos como nuevo ground truth principal.

---

# 4. Muchos modelos clásicos

No implementar una lista enorme de modelos salvo que sobre tiempo.

Modelos NO prioritarios:

```text
Naive Bayes
Discriminant Analysis
Bagging
AdaBoost
Stacking
```

Motivo:

```text
pocas muestras + weak labels + riesgo de leakage + poca mejora científica real
```

Mejor tener pocos modelos bien validados que muchos modelos mal defendidos.

---

# 5. Qué NO hacer por ahora

```text
No presentar Nature-derived labels como ground truth perfecto.
No afirmar detección química de plástico puro.
No entrenar modelos con random split y reportarlo como métrica principal.
No elegir umbrales usando todo el dataset y luego reportar esa métrica como final.
No mezclar sensores/resoluciones externas sin explicar domain shift.
No hacer fine-tuning de U-Net con pseudo-máscaras débiles sin validación muy cuidadosa.
```

---

# 6. Resultado esperado si se completan los bloques principales

El TFG debería poder defender una contribución como esta:

```text
Se desarrolla una pipeline híbrida para detectar estructuras compatibles con acumulaciones flotantes de residuos en Sentinel-2, usando etiquetas débiles derivadas del catálogo Nature y modelos preentrenados MARIDA. La contribución principal no es solo la predicción, sino el análisis crítico de la fiabilidad de las etiquetas, la comparación de métodos bajo validación sin leakage y la propuesta de un sistema de confianza para interpretar los resultados.
```
