# REESTRUCTURACIÓN COMPLETA — FASE 03_EVALUATION

## OBJETIVO GENERAL

Reestructurar completamente la fase `03_evaluation` para separar claramente:

1. calibración de thresholds,
2. generación de outputs calibrados,
3. unificación de predicciones,
4. evaluación patch-level,
5. evaluación pixel-wise,
6. evaluación pixel-wise tolerante,
7. análisis de errores y diagnóstico.

La filosofía nueva debe evitar mezclar:

* calibración,
* evaluación,
* y construcción del CSV maestro.

La evaluación final SIEMPRE debe hacerse únicamente sobre `test_final`.

---

# SPLITS

Actualmente:

* 60% → `train_val`
* 40% → `test_final`

REGLAS IMPORTANTES:

## train_val

Se utiliza para:

* calibrar thresholds,
* entrenar modelos tabulares propios,
* selección metodológica.

## test_final

Se utiliza SOLO para:

* evaluación final,
* métricas finales,
* análisis de errores.

NO puede utilizarse para:

* calibración,
* tuning,
* selección de thresholds,
* selección de features,
* selección de modelos.

---

# NUEVA ESTRUCTURA DE LA FASE

Reestructurar:

```text
03_evaluation/
│
├── 01_calibrate_thresholds.py
├── 02_generate_calibrated_outputs.py
├── 03_unify_predictions.py
├── 04_evaluate_patch_level.py
├── 05_evaluate_segmentation_pixelwise.py
├── 06_evaluate_segmentation_tolerant.py
├── 07_error_analysis_and_diagnostics.py
│
├── outputs/
│   ├── thresholds/
│   ├── calibrated_outputs/
│   ├── unified/
│   ├── patch_level/
│   ├── pixelwise/
│   ├── tolerant/
│   └── diagnostics/
```

---

# CAMBIO IMPORTANTE EN OUTPUTS

Actualmente los outputs están demasiado mezclados.

Debe reorganizarse completamente la carpeta `outputs/`.

Nueva estructura:

```text
outputs/
│
├── thresholds/
│   ├── thresholds_selected.csv
│   ├── threshold_curves.csv
│   └── threshold_calibration_summary.md
│
├── calibrated_outputs/
│   ├── unet/
│   ├── resnet/
│   ├── external/
│   └── sam/
│
├── unified/
│   ├── predictions_master.csv
│   └── predictions_master_missing_report.md
│
├── patch_level/
│   ├── patch_level_metrics.csv
│   ├── patch_level_by_patch.csv
│   └── patch_level_summary.md
│
├── pixelwise/
│   ├── pixelwise_metrics_by_patch.csv
│   ├── pixelwise_metrics_summary.csv
│   └── segmentation_noise_on_negatives.csv
│
├── tolerant/
│   ├── tolerant_metrics_by_patch.csv
│   ├── tolerant_metrics_summary.csv
│   └── tolerant_summary.md
│
└── diagnostics/
    ├── error_cases.csv
    ├── error_summary.csv
    ├── metrics_by_date.csv
    ├── metrics_by_quality.csv
    ├── metrics_by_scene_tags.csv
    ├── top_dates_summary.csv
    └── error_examples/
```

---

# 01_calibrate_thresholds.py

## OBJETIVO

Calibrar thresholds SOLO usando `train_val`.

NO usar `test_final`.

---

## MÉTODOS A CALIBRAR

ÚNICAMENTE:

### U-Net

* probabilidades continuas de Marine Debris

### ResNet

* `resnet_prob`

### MarineDebrisDetector

* scores/probabilidades continuas

### SAM

* scores continuos SAM

---

## NO CALIBRAR

NO calibrar:

* índices espectrales
* RF MARIDA
* máscaras binarias ya discretizadas

---

# ELIMINAR LÓGICA LEGACY DE SAM

Eliminar completamente:

* `sam_second_angle`
* `sam_third_angle`
* `sam_second_angle_pct`
* `sam_third_angle_pct`
* máscaras SAM top-2
* máscaras SAM top-3
* cualquier lógica basada en:

  * segunda clase más cercana
  * tercera clase más cercana

La nueva filosofía SAM debe basarse SOLO en:

* scores continuos
* distancias angulares
* features reales

Ejemplos válidos:

* `sam_score`
* `sam_debris_min`
* `sam_debris_p05`
* márgenes espectrales

---

# CÓMO CALIBRAR

Para cada método:

1. usar SOLO patches de `train_val`
2. usar SOLO etiquetas válidas
3. probar thresholds
4. maximizar F1 patch-level

---

## U-Net

Calibrar usando:

* mapa continuo de probabilidad Marine Debris

NO usar únicamente argmax.

IMPORTANTE:
Mantener SIEMPRE:

* versión original argmax
* versión calibrada

Queremos comparar:

* comportamiento original MARIDA
  vs
* comportamiento recalibrado

---

## ResNet

Eliminar threshold fijo 0.5.

Calibrar:

* `resnet_prob`

---

## MarineDebrisDetector

Calibrar:

* score/probabilidad continua

Mantener:

* versión original/default
* versión calibrada

---

## SAM

Calibrar:

* `sam_score`
* `sam_debris_min`
* `sam_debris_p05`

---

# OUTPUTS

Generar:

```text
outputs/thresholds/
```

con:

* `thresholds_selected.csv`
* `threshold_curves.csv`
* `threshold_calibration_summary.md`

---

# 02_generate_calibrated_outputs.py

## OBJETIVO

Aplicar thresholds seleccionados y generar outputs finales comparables.

---

# REGLA IMPORTANTE

Mantener SIEMPRE:

* versión original
* versión calibrada

---

# U-Net

Generar:

```text
unet_argmax_mask.tif
unet_debris_prob.tif
unet_thresholded_mask.tif
```

---

# MarineDebrisDetector

Generar:

```text
external_default_mask.tif
external_prob.tif
external_thresholded_mask.tif
```

---

# ResNet

Generar:

```text
resnet_prob
resnet_default_pred
resnet_thresholded_pred
```

---

# SAM

Generar outputs basados SOLO en:

* threshold sobre score
* NO top2/top3

---

# OUTPUTS

Guardar en:

```text
outputs/calibrated_outputs/
```

---

# 03_unify_predictions.py

## OBJETIVO

Construir:

```text
predictions_master.csv
```

---

# IMPORTANTE

Este script YA NO debe:

* calibrar thresholds
* decidir thresholds
* modificar outputs

Solo:

* leer outputs finales
* unificar información

---

# COLUMNAS RECOMENDADAS

## Metadata

```text
patch
date
group_id
label
nc_px
```

---

## U-Net

```text
unet_argmax_px
unet_argmax_pct

unet_thr_px
unet_thr_pct

unet_prob_mean
unet_prob_max
unet_prob_p95
```

---

## RF

```text
rf_full_px
rf_full_pct

rf_no_texture_px
rf_no_texture_pct

rf_indices_only_px
rf_indices_only_pct

rf_bands_only_px
rf_bands_only_pct
```

---

## ResNet

```text
resnet_prob
resnet_default_pred
resnet_thr_pred
```

---

## SAM

```text
sam_score
sam_debris_min
sam_debris_p05
```

Eliminar columnas legacy:

* top2
* top3

---

## External

```text
external_score
external_default_px
external_thr_px
```

---

# OUTPUTS

Guardar en:

```text
outputs/unified/
```

---

# 04_evaluate_patch_level.py

## OBJETIVO

Evaluar:

```text
¿el patch contiene acumulación?
```

---

# MUY IMPORTANTE

Evaluar SOLO:

* `test_final`

---

# MÉTODOS

Evaluar:

* outputs originales
* outputs calibrados

Ejemplos:

* U-Net argmax
* U-Net thresholded
* external default
* external thresholded
* ResNet default
* ResNet thresholded

---

# MÉTRICAS

Calcular:

* precision
* recall
* F1
* accuracy
* AUC-ROC
* TP
* FP
* TN
* FN

---

# OUTPUTS

Guardar en:

```text
outputs/patch_level/
```

---

# 05_evaluate_segmentation_pixelwise.py

## OBJETIVO

Evaluación pixel-wise estricta.

---

# IMPORTANTE

Evaluar SOLO:

* `test_final`

---

# MÉTODOS

Evaluar:

* U-Net argmax
* U-Net thresholded
* RF variants
* índices espectrales
* SAM binario
* MarineDebrisDetector default
* MarineDebrisDetector thresholded

---

# MÉTRICAS

Calcular:

* tp
* fp
* fn
* tn
* precision
* recall
* Dice/F1
* IoU
* gt_px
* pred_px

---

# IMPORTANTE

NO usar aproximación por conteos.

Usar máscaras reales píxel a píxel.

---

# OUTPUTS

Guardar en:

```text
outputs/pixelwise/
```

---

# 06_evaluate_segmentation_tolerant.py

## OBJETIVO

Crear evaluación tolerante espacialmente.

Esto es MUY IMPORTANTE porque:

* las máscaras GT son aproximadas,
* existen errores de alineamiento,
* puede haber pequeños desplazamientos geométricos.

---

# IDEA

Permitir tolerancia espacial:

* radio 1 px
* radio 2 px
* radio 3 px

---

# IMPLEMENTACIÓN SUGERIDA

Usar:

* dilatación morfológica
* distancia máxima aceptable

Ejemplo:
si la predicción cae muy cerca de la GT:

* NO penalizar igual que un error totalmente alejado.

---

# MÉTRICAS

Calcular:

* tolerant_precision
* tolerant_recall
* tolerant_f1
* tolerant_iou
* distance_to_gt

---

# OUTPUTS

Guardar en:

```text
outputs/tolerant/
```

---

# 07_error_analysis_and_diagnostics.py

## OBJETIVO

Analizar:

* dónde fallan los métodos,
* cómo fallan,
* cuándo fallan.

---

# INPUTS

Usar:

* predictions_master.csv
* thresholds_selected.csv
* patch_level_metrics
* pixelwise_metrics

---

# GENERAR

## FP/FN por método

---

## Casos visuales

Generar overlays:

* RGB
* GT
* predicción
* probability maps si existen

Guardar en:

```text
outputs/diagnostics/error_examples/
```

---

## Análisis por:

* fecha
* calidad
* cloud fraction
* land fraction
* scene type
* scene tags

---

# OUTPUTS

Guardar en:

```text
outputs/diagnostics/
```

---

# FILOSOFÍA FINAL

La nueva fase debe seguir estrictamente este orden:

```text
1. Calibrar thresholds
2. Generar outputs calibrados
3. Unificar resultados
4. Evaluar patch-level
5. Evaluar pixel-wise estricto
6. Evaluar pixel-wise tolerante
7. Analizar errores
```

La evaluación final SIEMPRE debe realizarse únicamente sobre `test_final`.
