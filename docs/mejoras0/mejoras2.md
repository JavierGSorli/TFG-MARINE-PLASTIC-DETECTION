# Plan de mejoras metodológicas y técnicas para el TFG

## Contexto general

Este documento describe las mejoras recomendadas para la versión actual del TFG sobre **detección de residuos plásticos flotantes mediante imágenes Sentinel-2 y machine learning**.

La pipeline actual ya contiene scripts para:

- exploración de candidatos Nature/windrows;
- descarga de patches Sentinel-2;
- creación de máscaras positivas derivadas del catálogo Nature;
- inferencia con modelos MARIDA: U-Net, Random Forest y ResNet;
- inferencia con índices espectrales FDI/NDVI;
- unificación de predicciones;
- construcción de dataset para XGBoost;
- entrenamiento/evaluación de XGBoost;
- evaluación comparativa;
- análisis de errores;
- visualización de resultados.

Scripts actuales detectados:

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

El objetivo de esta mejora no es simplemente añadir más modelos, sino fortalecer la **fiabilidad científica**, la **evaluación**, el **control de calidad de etiquetas** y la **interpretabilidad** de los resultados.

---

# 1. Problema metodológico principal

El dataset positivo usado como referencia procede del catálogo Nature/Zenodo de *Mediterranean Sentinel-2 Litter Windrows*. Sin embargo, este dataset no debe tratarse como un ground truth perfecto de plástico puro.

## Problema

Las detecciones positivas son filamentos o *litter windrows* detectados en Sentinel-2, pero:

- no existe confirmación material directa de que cada píxel sea plástico puro;
- no se proporcionan máscaras pixel-perfect independientes para segmentación;
- los filamentos pueden contener mezcla de plástico, materia orgánica, espuma, algas, restos flotantes u otros materiales;
- la máscara positiva generada en el proyecto es una pseudo-máscara derivada de la geometría del catálogo Nature;
- parte de la alineación espacial se apoya en una referencia basada en FDI, lo que puede introducir circularidad si luego se evalúa FDI contra esa misma máscara.

## Terminología recomendada

En la memoria y en los nombres de variables/documentación conviene evitar hablar de `ground truth` sin matices.

Usar preferentemente:

```text
proxy ground truth
weak labels
pseudo-labels
Nature-derived labels
Nature-derived pseudo-masks
positive candidates
high-confidence positives
```

Frase sugerida para la memoria:

> Dado que el catálogo empleado no proporciona máscaras de segmentación independientes ni confirmación material directa de la composición de cada filamento, las etiquetas utilizadas en este trabajo se consideran weak labels o proxy ground truth. Por ello, la evaluación debe interpretarse como una estimación de la capacidad de los modelos para detectar estructuras espectrales y espaciales compatibles con acumulaciones flotantes de residuos, no como una validación absoluta de plástico marino.

---

# 2. Riesgo de circularidad con FDI

En `01_download_dataset.py`, la función `build_mask()` construye la máscara positiva a partir de `pixel_x`, `pixel_y`, `x_centroid`, `y_centroid` del NetCDF. Después se usa una función de alineación espacial:

```python
def build_alignment_reference(data):
    fdi = compute_fdi(data)
    thr_fdi = threshold_mean_plus_3std(fdi)
    ...
```

Y posteriormente:

```python
dr, dc = find_shift(raw_mask, alignment_ref)
```

Esto es razonable como solución práctica para alinear el filamento, pero implica que las métricas de FDI deben interpretarse con cautela.

## Modificación recomendada

Añadir a la documentación y a la evaluación una advertencia metodológica:

> Las métricas pixel-level de métodos basados en FDI deben interpretarse de forma exploratoria, ya que la pseudo-máscara de referencia se ha refinado parcialmente usando una señal basada en FDI. Esto no invalida el uso de FDI, pero reduce la independencia entre referencia y método evaluado.

## Cambio técnico recomendado

Modificar `build_mask()` para que devuelva también los desplazamientos de alineación:

```python
return mask, int(mask.sum()), dr, dc
```

Guardar `dr`, `dc` y `alignment_shift_magnitude` en un CSV de calidad de positivos.

---

# 3. Prioridades generales

## Prioridad alta

1. Redefinir formalmente el ground truth como weak/proxy ground truth.
2. Mejorar las etiquetas de negativos.
3. Crear un sistema de control de calidad y confianza para positivos y negativos.
4. Corregir la evaluación para evitar leakage temporal, espacial o por evento.
5. Crear evaluación por subconjuntos de confianza.
6. Crear experimento de ablación de XGBoost.
7. Crear mapas del área de estudio y de las predicciones.

## Prioridad media

8. Añadir SAM como score de confianza espectral.
9. Añadir GLCM/texturas como features para XGBoost.
10. Crear modelo híbrido detección + segmentación.
11. Analizar calibración de ResNet/XGBoost.

## Prioridad baja/opcional

12. Añadir muchos modelos clásicos.
13. Fine-tuning de modelos profundos.
14. Incorporar datasets externos de barcos si no están en Sentinel-2 o no están bien alineados con el problema.

---

# 4. Mejora de etiquetas de negativos

## Problema actual

Actualmente los negativos parecen estar etiquetados con categorías como:

```text
CLARO
DIFICIL
DUDOSO
```

El problema es que estas etiquetas mezclan dos dimensiones distintas:

1. la confianza de que el patch sea realmente negativo;
2. el tipo de dificultad visual/espectral.

Por ejemplo, un negativo puede ser difícil por muchas razones diferentes:

- nube;
- nube fina;
- agua oscura;
- costa;
- espuma;
- estela de barco;
- barco;
- agua turbia;
- sunglint;
- posible debris no anotado.

## Nueva estructura recomendada

Separar la anotación en tres niveles.

### Nivel 1: decisión

```text
ACCEPT      -> negativo válido
REJECT      -> descartar patch
UNCERTAIN   -> dudoso, no usar para evaluación principal
```

### Nivel 2: tipo de escena

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

### Nivel 3: confianza

```text
1 = baja confianza
2 = confianza media
3 = alta confianza
```

Ejemplo de anotación:

```text
patch = 20170728_NO_000000_40_DIFICIL.tif
label = NO
decision = ACCEPT
scene_type = wake
confidence = 2
notes = Estela lineal brillante, posible confusión con filamento
```

## Script a crear

Crear:

```text
scripts/12_relabel_negatives.py
```

## Funcionalidad

El script debe:

1. recorrer todos los patches negativos de `PATCHES_DIR`;
2. abrir el patch `.tif`;
3. mostrar una visualización RGB;
4. opcionalmente mostrar FDI, NDVI y/o composición falsa color;
5. permitir seleccionar:
   - decisión;
   - tipo de escena;
   - confianza;
   - notas libres;
6. guardar o actualizar un CSV sin modificar los nombres de archivo.

## Salida esperada

```text
results/auto/annotations/negative_annotations.csv
```

Columnas recomendadas:

```text
patch
label
decision
scene_type
confidence
has_cloud
has_thin_cloud
has_ship
has_wake
has_coast
has_foam
is_dark
is_turbid
possible_debris
bad_quality
notes
annotated_at
```

## Criterios de aceptación

- El script permite reanudar una sesión anterior.
- Si un patch ya está anotado, se puede saltar o modificar.
- Las etiquetas se guardan en CSV, no en el nombre del fichero.
- El script no borra patches automáticamente.
- El CSV se puede usar después en evaluación y en XGBoost.

---

# 5. Control de calidad de positivos

## Objetivo

Crear un sistema para clasificar los positivos Nature-derived según su fiabilidad.

No todos los positivos tienen la misma calidad. Por ejemplo, un positivo con muchos píxeles, lejos de costa, sin nubes y con buena alineación es más fiable que un positivo pequeño, cerca de costa y con desplazamiento grande.

## Script a crear

Crear:

```text
scripts/13_positive_quality_audit.py
```

## Métricas a calcular

Para cada patch positivo:

```text
patch
date
lat
lon
n_pixels_fil
mask_px
cloud_frac
land_frac
zero_all_frac
key_valid_frac
b04_zero_frac
b08_zero_frac
alignment_shift_row
alignment_shift_col
alignment_shift_magnitude
positive_confidence
quality_flags
usable_for_main_eval
```

## Cambios necesarios en `01_download_dataset.py`

Modificar `build_mask()` para que devuelva:

```python
mask, mask_px, dr, dc
```

Después, guardar estos valores en un CSV, por ejemplo:

```text
results/auto/quality/positive_mask_alignment.csv
```

## Posible regla inicial de confianza

Ejemplo simple:

```text
P3 = positivo alta confianza
     n_pixels_fil >= 10
     cloud_frac <= 0.10
     land_frac <= 0.20
     alignment_shift_magnitude <= 80
     key_valid_frac >= 0.95

P2 = positivo confianza media
     no cumple P3, pero no está claramente degradado

P1 = positivo baja confianza
     pocos píxeles, mucha nube, mucha costa, mala calidad o alineación dudosa
```

No hace falta que estas reglas sean definitivas. Lo importante es que queden trazables en CSV.

## Criterios de aceptación

- El script genera un CSV de calidad de positivos.
- No cambia las máscaras existentes salvo que se solicite explícitamente.
- Permite filtrar positivos por `positive_confidence`.
- Permite justificar en la memoria qué positivos son más fiables.

---

# 6. Evitar leakage en entrenamiento y evaluación

## Problema

En `08_train_xgboost.py` actualmente se usa:

```python
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
```

Esto se hace antes de la validación cruzada. Es un leakage pequeño, porque la mediana se calcula usando todo el dataset, incluido test.

Además, si se usa `StratifiedKFold` aleatorio, puede haber:

- leakage temporal: patches de la misma fecha en train y test;
- leakage espacial: patches cercanos en train y test;
- leakage por evento: varios patches del mismo filamento o fecha separados entre train y test;
- leakage por selección de umbral: elegir el mejor umbral usando todo el dataset.

## Script a crear

Crear:

```text
scripts/14_build_validation_groups.py
```

## Objetivo

Construir grupos de validación para usar `GroupKFold` o `LeaveOneGroupOut`.

## Salida esperada

```text
results/auto/splits/validation_groups.csv
```

Columnas recomendadas:

```text
patch
label
date
year
month
spatial_cell
event_group
validation_group
```

## Reglas iniciales

Para una primera versión:

```text
validation_group = date
```

Si hay pocas fechas, usar:

```text
validation_group = year_month
```

Si se dispone de lat/lon:

```text
spatial_cell = celda geográfica aproximada, por ejemplo redondeando lat/lon a 0.1 grados
```

## Segundo script a crear

Crear:

```text
scripts/15_train_xgboost_grouped_cv.py
```

## Cambios respecto a `08_train_xgboost.py`

Usar un `Pipeline` de sklearn para que la imputación se haga dentro de cada fold:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(...))
])
```

Usar validación agrupada:

```python
GroupKFold(n_splits=...)
```

O, si hay pocos grupos:

```python
LeaveOneGroupOut()
```

## Selección de umbral

No elegir el umbral usando todo el dataset para la métrica principal.

Opción recomendada:

1. dentro de cada fold, dividir train en train interno/validación interna;
2. elegir el umbral en la parte de validación interna;
3. aplicar ese umbral al test externo;
4. guardar predicciones out-of-fold.

Si esto es demasiado complejo, usar inicialmente umbral fijo `0.5` y reportar el umbral óptimo global solo como análisis exploratorio.

## Salidas esperadas

```text
results/auto/xgboost_grouped/cv_predictions.csv
results/auto/xgboost_grouped/metrics.json
results/auto/xgboost_grouped/feature_importance.csv
results/auto/xgboost_grouped/grouped_cv_summary.csv
```

## Métricas mínimas

```text
AUC-ROC
Average Precision / PR-AUC
F1
Precision
Recall
Balanced Accuracy
Confusion Matrix
Brier Score
```

## Criterios de aceptación

- La imputación ocurre dentro del fold.
- No se usan datos de test para elegir umbral.
- Se guardan predicciones out-of-fold.
- Se reporta claramente el tipo de validación.

---

# 7. Evaluación por confianza

## Objetivo

Responder a la duda principal del TFG:

> ¿Son fiables los resultados si las etiquetas son débiles?

La respuesta debe ser matizada:

- los resultados son más fiables en positivos y negativos de alta confianza;
- bajan en escenas difíciles;
- las métricas globales deben interpretarse como aproximadas;
- las métricas por subconjunto explican mejor el comportamiento real.

## Script a crear

Crear:

```text
scripts/16_confidence_scoring.py
```

Este script debe combinar:

- calidad de positivos;
- anotaciones de negativos;
- acuerdo entre modelos;
- scores espectrales opcionales.

## Salida esperada

```text
results/auto/confidence/patch_confidence.csv
```

Columnas recomendadas:

```text
patch
label
manual_confidence
positive_confidence
negative_confidence
quality_score
model_agreement_score
spectral_confidence
final_confidence
confidence_group
usable_for_main_eval
usable_for_stress_test
```

## Grupos recomendados

```text
high_confidence
medium_confidence
low_confidence
hard_negative
uncertain
```

## Script de evaluación por confianza

Crear:

```text
scripts/17_evaluate_by_confidence.py
```

Evaluar métodos sobre:

```text
1. todos los patches
2. solo high_confidence
3. positivos high_confidence vs negativos high_confidence
4. negativos difíciles: wake, foam, ship, coast, cloud, dark_water
5. positivos pequeños vs positivos grandes
6. escenas con poca nube vs escenas con nube
```

## Salidas esperadas

```text
results/auto/evaluation_by_confidence/metrics_by_subset.csv
results/auto/evaluation_by_confidence/confusion_by_subset.csv
results/auto/evaluation_by_confidence/summary.md
```

## Criterios de aceptación

- Cada métrica indica claramente cuántas muestras usa.
- No se mezclan resultados exploratorios con resultados finales.
- Permite generar tablas para la memoria.

---

# 8. Ablation experiment de XGBoost

## Objetivo

Entender qué familias de variables aportan realmente valor.

El XGBoost no debe ser solo una “caja negra”. Hay que responder:

- ¿aportan más las bandas espectrales?
- ¿aportan más los índices FDI/NDVI?
- ¿aportan más las salidas de modelos MARIDA?
- ¿qué pasa si quitamos ResNet?
- ¿qué pasa si quitamos FDI?
- ¿qué pasa si añadimos GLCM?

## Script a crear

Crear:

```text
scripts/18_ablation_xgboost.py
```

## Grupos de features recomendados

A partir de `xgboost_dataset.csv`, definir grupos:

```text
spectral_bands:
    columnas tipo B01_mean, B01_std, B01_p95, etc.

indices:
    fdi_mean, fdi_max, fdi_p99, ndvi_mean, ndvi_max, ndvi_p99

model_outputs:
    unet_pct, rf_pct, resnet_prob, fdi_pct, ndvi_pct, fdi_ndvi_pct

texture_glcm:
    si se implementa después
```

## Experimentos mínimos

```text
A_spectral_only
B_indices_only
C_model_outputs_only
D_spectral_plus_indices
E_all_without_fdi
F_all_without_resnet
G_all_features
H_all_features_plus_glcm   # si se implementan texturas
```

## Validación

Usar los mismos grupos de validación que en `15_train_xgboost_grouped_cv.py`.

## Salidas esperadas

```text
results/auto/ablation_xgboost/ablation_results.csv
results/auto/ablation_xgboost/ablation_results.md
results/auto/ablation_xgboost/ablation_plot.png
```

## Métricas

```text
AUC-ROC
PR-AUC
F1
Precision
Recall
Balanced Accuracy
Brier Score
```

## Criterios de aceptación

- Todas las ablaciones usan la misma partición de validación.
- El CSV indica número de features y lista de features usadas.
- Los resultados están ordenados por métrica principal.

---

# 9. Añadir GLCM/texturas a XGBoost

## Justificación

Los filamentos no se diferencian solo por su firma espectral, sino también por su estructura espacial. Las features de textura pueden ayudar a distinguir:

- agua homogénea;
- filamentos lineales;
- espuma;
- nubes finas;
- estelas;
- zonas costeras.

MARIDA ya usa texturas GLCM en su Random Forest, por lo que añadir features texturales agregadas a XGBoost es metodológicamente coherente.

## Script recomendado

Crear una versión nueva, no romper el script actual:

```text
scripts/19_build_xgboost_dataset_v2_texture.py
```

O modificar `07_build_xgboost_dataset.py` añadiendo una opción `--include-texture`.

## Features sugeridas

Calcular sobre una imagen grayscale derivada de RGB o sobre bandas seleccionadas.

Features:

```text
glcm_contrast_mean
glcm_dissimilarity_mean
glcm_homogeneity_mean
glcm_energy_mean
glcm_correlation_mean
glcm_asm_mean
```

Opcionalmente:

```text
glcm_contrast_p95
glcm_homogeneity_p05
```

## Precauciones

- Las texturas pueden capturar nubes, costa o artefactos, no necesariamente plástico.
- Usarlas dentro del experimento de ablación.
- No afirmar que GLCM detecta plástico directamente.

## Salida esperada

```text
results/auto/xgboost_dataset_v2_texture.csv
```

---

# 10. SAM — Spectral Angle Mapper

## Objetivo

Usar SAM como score auxiliar de confianza espectral, no como ground truth definitivo.

SAM puede medir si los píxeles de un filamento Nature se parecen espectralmente a una firma de referencia de `Marine Debris`.

## Pregunta importante

¿Qué firma de referencia usar?

### Opción A: MARIDA

Ventaja:

- más independiente del catálogo Nature;
- tiene anotaciones pixel-level por clase.

Desventaja:

- puede haber diferencia de dominio/procesado con los patches Sentinel-2 descargados para Gibraltar.

### Opción B: Nature

Ventaja:

- más cercana al dominio local.

Desventaja:

- menos independiente, puede reforzar circularidad.

## Recomendación

Implementar dos scores distintos:

```text
sam_marida_debris
sam_nature_filament
```

Y, si es posible, comparar contra agua:

```text
sam_margin = similarity_to_debris - similarity_to_water
```

## Script a crear

Crear:

```text
scripts/20_sam_confidence.py
```

## Entradas

- patches MARIDA con máscaras/clases, si están disponibles;
- patches Nature-derived positivos con pseudo-máscaras;
- patches negativos claros para firma de agua.

## Salidas esperadas

```text
results/auto/sam/sam_reference_marida.npy
results/auto/sam/sam_reference_water.npy
results/auto/sam/sam_scores.csv
```

Columnas:

```text
patch
label
sam_marida_debris_mean
sam_marida_debris_min
sam_water_mean
sam_margin
sam_confidence
```

## Uso recomendado

No usar SAM para decir:

```text
esto es plástico seguro
```

Usarlo para decir:

```text
este filamento tiene mayor/menor compatibilidad espectral con la clase Marine Debris de MARIDA
```

## Experimento interesante

Crear subconjuntos:

```text
Nature positives con SAM alto
Nature positives con SAM medio
Nature positives con SAM bajo
```

Y evaluar si el rendimiento de los modelos mejora en los positivos con SAM alto.

---

# 11. Modelo híbrido: detección + segmentación

## Objetivo

Crear un sistema en dos etapas:

1. detección patch-level de presencia de posible debris;
2. segmentación pixel-level solo si el patch es suficientemente probable.

## Motivación

Actualmente hay métodos patch-level y pixel-level:

- ResNet: clasificación multi-label a nivel patch;
- XGBoost: clasificación patch-level;
- U-Net: segmentación pixel-level;
- RF: segmentación pixel-level;
- FDI/NDVI: detección pixel-level por índices.

Tiene sentido combinarlos.

## Script a crear

Crear:

```text
scripts/21_hybrid_detector_segmenter.py
```

## Lógica inicial recomendada

```text
if xgb_prob >= 0.75:
    confidence_level = high
    aplicar máscara híbrida
elif 0.45 <= xgb_prob < 0.75:
    confidence_level = medium
    marcar como incierto
else:
    confidence_level = low
    no generar alerta positiva
```

## Máscaras híbridas posibles

### Opción conservadora

```text
hybrid_mask = UNet AND (RF OR FDI_NDVI)
```

Reduce falsos positivos.

### Opción majority vote

```text
hybrid_mask = al menos 2 de 3 métodos positivos
```

Métodos:

```text
UNet
RF
FDI_NDVI
```

### Opción flexible

Permitir argumento:

```text
--fusion-mode conservative
--fusion-mode majority
--fusion-mode union
```

## Salidas esperadas

```text
results/auto/hybrid/hybrid_patch_predictions.csv
results/auto/hybrid/masks/*.tif
results/auto/hybrid/summary.json
```

Columnas:

```text
patch
label
xgb_prob
resnet_prob
confidence_level
fusion_mode
hybrid_pred_px
hybrid_pred_pct
hybrid_positive
```

## Evaluación

Evaluar el modelo híbrido en `17_evaluate_by_confidence.py`.

---

# 12. Mapa del Estrecho

## Objetivo

Crear visualizaciones geográficas del área de estudio.

No venderlo como mapa exhaustivo de plástico en el Estrecho, sino como:

> visualización espacial de los patches analizados, las detecciones Nature-derived y las predicciones del sistema.

## Script a crear

Crear:

```text
scripts/22_map_gibraltar_results.py
```

## Mapas recomendados

### Mapa 1: candidatos Nature

```text
Puntos/filamentos del catálogo Nature dentro del área de estudio.
Color por año.
Tamaño por n_pixels_fil.
```

### Mapa 2: patches del dataset

```text
Patches SI y NO usados en la evaluación.
Color por label.
Símbolo o borde por confidence_group.
```

### Mapa 3: predicciones

```text
Patches coloreados por xgb_prob o hybrid_prob.
Tamaño por número de píxeles predichos.
```

## Salidas esperadas

```text
results/auto/maps/gibraltar_nature_candidates.html
results/auto/maps/gibraltar_dataset_patches.html
results/auto/maps/gibraltar_predictions.html
```

Opcionalmente exportar PNG si es necesario para la memoria:

```text
results/auto/maps/gibraltar_predictions.png
```

## Librerías sugeridas

```text
folium
geopandas
shapely
pandas
```

Si no están instaladas, generar al menos CSV/GeoJSON.

## Criterios de aceptación

- El mapa permite ver SI/NO.
- El mapa permite ver confianza.
- El mapa no requiere conexión si se exporta como HTML con tiles básicos, o documentar si requiere internet para los tiles.

---

# 13. Calibración de ResNet/XGBoost

## Pregunta

¿Tiene sentido calibrar ResNet?

Sí, si se usa `resnet_prob` como probabilidad para ranking, mapas, XGBoost o umbrales.

No es imprescindible si solo se usa como activación binaria `> 0.5`.

## Precaución

Las probabilidades de ResNet pueden no estar calibradas en el dominio Gibraltar/Nature porque el modelo fue entrenado en MARIDA y se está aplicando a otro dominio.

## Script a crear

Crear:

```text
scripts/23_calibration_analysis.py
```

## Aplicar a

```text
ResNet
XGBoost
Hybrid model
```

## Salidas esperadas

```text
results/auto/calibration/calibration_resnet.png
results/auto/calibration/calibration_xgboost.png
results/auto/calibration/brier_scores.csv
```

## Métricas

```text
Brier Score
Reliability curve
Expected Calibration Error, opcional
```

## Interpretación recomendada

En la memoria, decir:

> La calibración se analiza de forma exploratoria debido al tamaño reducido del conjunto y a la naturaleza débil de las etiquetas.

---

# 14. Barcos y hard negatives

## Idea

Añadir barcos puede ser útil, pero no como dataset principal nuevo, sino como clase de confusión o hard negative.

## Recomendación

Priorizar este orden:

1. usar clases de MARIDA si incluyen ship/wake/foam/cloud/coast;
2. etiquetar manualmente patches Sentinel-2 propios con barcos/estelas;
3. solo después considerar datasets externos.

## Riesgo de datasets externos

Muchos datasets de barcos usan:

- SAR en vez de óptico;
- alta resolución en vez de Sentinel-2 10 m;
- RGB en vez de multiespectral;
- dominios geográficos distintos.

Eso puede introducir domain shift y complicar la memoria.

## Uso recomendado

Incluir barcos como:

```text
hard_negative_scene_type = ship / wake
```

No como nueva clase principal salvo que haya datos suficientes y comparables.

---

# 15. Añadir muchos modelos clásicos

## Modelos propuestos inicialmente

```text
Naive Bayes
Random Forest
Discriminant Analysis
Bagging
AdaBoost
Gradient Boosting
Stacking
XGBoost
```

## Recomendación

No añadir todos. Puede parecer más completo, pero aumenta el riesgo de:

- sobreajuste;
- leakage;
- resultados difíciles de justificar;
- comparación superficial.

## Modelos recomendados

Crear:

```text
scripts/24_train_classical_models_grouped.py
```

Incluir solo:

```text
LogisticRegression
RandomForestClassifier
HistGradientBoostingClassifier
XGBoost, si está instalado
```

Opcional:

```text
ExtraTreesClassifier
```

## Validación obligatoria

Usar los grupos creados por:

```text
scripts/14_build_validation_groups.py
```

No usar random split como métrica principal.

## Salida esperada

```text
results/auto/classical_models/model_comparison.csv
```

---

# 16. Fine-tuning

## Recomendación

No hacerlo como prioridad principal.

El fine-tuning puede mejorar resultados, pero con weak labels también puede enseñar al modelo sesgos del catálogo Nature, fechas, zona geográfica, nubes o artefactos.

## Si se hace

Hacer solo con subconjuntos de alta confianza:

```text
P3 positives
N3 negatives
hard negatives bien etiquetados
```

## Modelo recomendado

Fine-tuning patch-level de ResNet, no U-Net completa.

Estrategia:

```text
1. cargar modelo ResNet preentrenado;
2. congelar backbone;
3. entrenar solo cabeza/clasificador;
4. usar GroupKFold por fecha/evento;
5. comparar contra ResNet sin fine-tuning;
6. no usar test para selección de hiperparámetros.
```

## Script opcional

```text
scripts/25_finetune_resnet_high_confidence.py
```

## Criterio

Solo implementar si las fases anteriores están completadas.

---

# 17. Plan de implementación ordenado

## Fase 1: metodología y calidad de etiquetas

### Paso 1

Modificar documentación interna y nombres de outputs para hablar de:

```text
weak labels
proxy ground truth
pseudo-masks
Nature-derived positives
```

No hace falta cambiar nombres de todos los archivos existentes, pero sí los textos, README y outputs nuevos.

### Paso 2

Modificar `01_download_dataset.py`:

- hacer que `build_mask()` devuelva `dr`, `dc`;
- guardar alineación y calidad en CSV;
- no romper compatibilidad con scripts existentes.

### Paso 3

Crear:

```text
scripts/12_relabel_negatives.py
scripts/13_positive_quality_audit.py
```

---

## Fase 2: validación sin leakage

### Paso 4

Crear:

```text
scripts/14_build_validation_groups.py
```

### Paso 5

Crear:

```text
scripts/15_train_xgboost_grouped_cv.py
```

No borrar `08_train_xgboost.py`. Mantenerlo como baseline antiguo/exploratorio.

### Paso 6

Crear:

```text
scripts/16_confidence_scoring.py
scripts/17_evaluate_by_confidence.py
```

---

## Fase 3: análisis de valor añadido

### Paso 7

Crear:

```text
scripts/18_ablation_xgboost.py
```

### Paso 8

Añadir texturas:

```text
scripts/19_build_xgboost_dataset_v2_texture.py
```

---

## Fase 4: confianza espectral e híbrido

### Paso 9

Crear:

```text
scripts/20_sam_confidence.py
```

### Paso 10

Crear:

```text
scripts/21_hybrid_detector_segmenter.py
```

---

## Fase 5: mapas y visualización

### Paso 11

Crear:

```text
scripts/22_map_gibraltar_results.py
```

---

## Fase 6: opcionales

### Paso 12

Crear si hay tiempo:

```text
scripts/23_calibration_analysis.py
scripts/24_train_classical_models_grouped.py
scripts/25_finetune_resnet_high_confidence.py
```

---

# 18. Orden de ejecución recomendado

Una vez implementados los nuevos scripts, el orden lógico sería:

```bash
python scripts/00_explore_candidates.py
python scripts/01_download_dataset.py
python scripts/02_predict_unet.py
python scripts/03_predict_rf.py
python scripts/04_predict_resnet.py
python scripts/05_predict_indices.py
python scripts/06_unify_predictions.py
python scripts/07_build_xgboost_dataset.py

# Nuevos pasos de calidad
python scripts/12_relabel_negatives.py
python scripts/13_positive_quality_audit.py
python scripts/14_build_validation_groups.py
python scripts/16_confidence_scoring.py

# Entrenamiento/evaluación robusta
python scripts/15_train_xgboost_grouped_cv.py
python scripts/17_evaluate_by_confidence.py
python scripts/18_ablation_xgboost.py

# Opcionales avanzados
python scripts/19_build_xgboost_dataset_v2_texture.py
python scripts/20_sam_confidence.py
python scripts/21_hybrid_detector_segmenter.py
python scripts/22_map_gibraltar_results.py
python scripts/23_calibration_analysis.py
python scripts/24_train_classical_models_grouped.py
```

---

# 19. Cambios mínimos recomendados en `config.py`

Añadir nuevos directorios:

```python
ANNOTATIONS_OUT = RESULTS_DIR / "annotations"
QUALITY_OUT = RESULTS_DIR / "quality"
SPLITS_OUT = RESULTS_DIR / "splits"
CONFIDENCE_OUT = RESULTS_DIR / "confidence"
EVAL_CONFIDENCE_OUT = RESULTS_DIR / "evaluation_by_confidence"
XGB_GROUPED_OUT = RESULTS_DIR / "xgboost_grouped"
ABLATION_OUT = RESULTS_DIR / "ablation_xgboost"
SAM_OUT = RESULTS_DIR / "sam"
HYBRID_OUT = RESULTS_DIR / "hybrid"
MAPS_OUT = RESULTS_DIR / "maps"
CALIBRATION_OUT = RESULTS_DIR / "calibration"
CLASSICAL_OUT = RESULTS_DIR / "classical_models"
```

Añadir a `OUTPUT_DIRS`.

Añadir CSVs:

```python
CSV_NEGATIVE_ANNOTATIONS = ANNOTATIONS_OUT / "negative_annotations.csv"
CSV_POSITIVE_QUALITY = QUALITY_OUT / "positive_quality.csv"
CSV_VALIDATION_GROUPS = SPLITS_OUT / "validation_groups.csv"
CSV_PATCH_CONFIDENCE = CONFIDENCE_OUT / "patch_confidence.csv"
CSV_XGB_TEXTURE = RESULTS_DIR / "xgboost_dataset_v2_texture.csv"
```

---

# 20. Qué NO hacer por ahora

No implementar de golpe:

```text
Naive Bayes + LDA + Bagging + AdaBoost + GradientBoosting + Stacking + XGBoost + fine-tuning + barcos externos
```

Esto puede hacer que el TFG parezca más grande pero menos sólido.

Priorizar:

```text
calidad de etiquetas > evaluación robusta > confianza > ablación > mapas > modelos extra
```

---

# 21. Resultado esperado tras las mejoras

El TFG debería poder defender estas afirmaciones:

1. Las etiquetas positivas son weak labels derivadas del catálogo Nature, no ground truth absoluto.
2. Se ha creado una metodología para medir la confianza de positivos y negativos.
3. La evaluación principal evita leakage temporal/por evento mediante validación agrupada.
4. Los resultados se reportan globalmente y por subconjuntos de confianza.
5. XGBoost se analiza mediante ablación para justificar qué variables aportan valor.
6. SAM se usa como indicador auxiliar de compatibilidad espectral, no como prueba definitiva de plástico.
7. El sistema híbrido combina detección patch-level y segmentación pixel-level.
8. Los mapas muestran la distribución espacial de los datos y predicciones sin afirmar cobertura exhaustiva del Estrecho.

---

# 22. Mensaje metodológico final para la memoria

La idea científica central debería ser:

> Este trabajo no pretende demostrar la detección perfecta de plástico marino puro, sino desarrollar y evaluar una pipeline reproducible para detectar estructuras Sentinel-2 compatibles con acumulaciones flotantes de residuos, utilizando etiquetas débiles derivadas del catálogo Nature y modelos preentrenados MARIDA. La contribución principal reside en la integración de métodos espectrales, modelos de machine learning, análisis de confianza y evaluación crítica bajo incertidumbre de etiquetas.

