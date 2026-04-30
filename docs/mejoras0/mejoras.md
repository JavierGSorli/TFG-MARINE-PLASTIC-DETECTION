# Mejoras — TFG Detección de Plásticos Marinos
**Fecha de análisis:** 2026-04-28  
**Dataset en el momento del análisis:** 50 patches (25 SI + 25 NO)

---

## 1. Estado actual de resultados

### 1.1 Detección a nivel de patch (evaluación principal)

| Método | AUC-ROC | F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|
| RF (MARIDA) | **0.910** | 0.826 | 0.905 | 0.760 | 2 | 6 |
| XGBoost | 0.897 | **0.833** | 0.870 | 0.800 | 3 | 5 |
| UNet (MARIDA) | 0.893 | 0.833 | 0.870 | 0.800 | 3 | 5 |
| FDI | 0.714 | 0.754 | 0.639 | 0.920 | 13 | 2 |
| ResNet (MARIDA) | 0.570 | 0.478 | 0.524 | 0.440 | 10 | 14 |
| NDVI | 0.366 | 0.667 | 0.500 | 1.000 | 25 | 0 |

**Lecturas clave:**
- RF y UNet son equivalentes en F1. RF tiene mejor AUC y Precision.
- XGBoost es un meta-ensemble: su feature más importante es `unet_pct` (15.4%), luego `B12_std/mean` (SWIR, coherente con la física del plástico).
- ResNet rinde mal (AUC 0.57): el shift radiométrico L2A vs ACOLITE le afecta más que a UNet/RF.
- FDI tiene el mejor Recall (0.92) pero 13 FP sobre 25 negativos — inutilizable solo.
- NDVI y FDI+NDVI no discriminan a nivel de patch (AUC < 0.5).

### 1.2 Segmentación a nivel de píxel (evaluación secundaria)

| Método | IoU | Precision-px | Recall-px | GT cubierto |
|---|---|---|---|---|
| FDI | 23.3% | 28.5% | 56.1% | 56.1% |
| UNet | **15.7%** | 34.3% | 22.4% | 22.4% |
| RF | 14.9% | 34.7% | 20.6% | 20.6% |
| FDI+NDVI | 11.8% | 55.8% | 13.0% | 13.0% |
| NDVI | 12.9% | 23.2% | 22.5% | 22.5% |

FDI tiene mayor recall pero predice el doble de píxeles que el GT (10.844 vs 5.502). UNet y RF son similares entre sí (~34% precision, ~22% recall).

### 1.3 Falsos positivos en negativos por dificultad

| Método | FP CLARO (n=12) | FP DUDOSO (n=6) | FP DIFICIL (n=7) |
|---|---|---|---|
| UNet | 1 (8%) | 0 (0%) | 2 (29%) |
| RF | 2 (17%) | 0 (0%) | 0 (0%) |
| FDI | 5 (42%) | 5 (83%) | 3 (43%) |
| ResNet | 7 (58%) | 1 (17%) | 2 (29%) |

UNet y RF son los más conservadores. RF tiene 0 FP en DUDOSO y DIFICIL.

---

## 2. Problema fundamental: el ground truth

Este es el punto más crítico del proyecto antes de añadir más modelos.

### 2.1 Qué son realmente los positivos

Las detecciones WASP son **Litter Windows (LW)**: regiones con espectros tipo plástico detectadas algorítmicamente sobre imágenes L1C y validadas visualmente *a escala de escena*, no píxel a píxel. No son confirmaciones in situ de plástico real.

### 2.2 El problema de alineación L1C → L2A

Las máscaras GT se generan proyectando coordenadas de píxel L1C al sistema UTM del patch L2A mediante `find_shift()` (correlación cruzada FFT). El shift observado es **15–77 píxeles verticales** (150–770 metros), lo que significa que la máscara GT puede estar desplazada hasta medio kilómetro respecto a la detección real.

**Consecuencia directa:** la evaluación pixel-level no mide "qué modelos detectan mejor el plástico" sino "qué modelos se parecen más al catálogo WASP con posible error de registro". Los IoU bajos de UNet/RF no significan que sean malos segmentadores.

### 2.3 Qué es fiable y qué no

- **Fiable:** evaluación a nivel de patch (label SI/NO). No depende de la alineación pixel-a-pixel.
- **Orientativo con cautela:** evaluación de segmentación solo en patches con shift estimado < 5px y n_píxeles GT > 50.
- **Problemático:** comparar IoU entre métodos como si el GT fuera perfecto.

---

## 3. Problemas detectados en los scripts actuales

### 3.1 XGBoost: threshold de LOO demasiado bajo
- **Dónde:** `08_train_xgboost.py`, línea `if len(df) < 30`
- **Problema:** con n=50, StratifiedKFold(5) da 10 muestras de test por fold. El CI del AUC es enorme (~±0.08).
- **Fix:** cambiar el umbral a `n < 100` para usar LOO hasta datasets más grandes.

### 3.2 ResNet: umbral 0.5 no calibrado para el dominio
- **Dónde:** `09_evaluate.py`, bloque `ResNet (MARIDA)`
- **Problema:** ResNet usa `resnet_active` (umbral 0.5 fijo) en lugar de optimizar el umbral por F1 como hacen los demás métodos. Esto explica parte del bajo rendimiento (AUC 0.57 pero podría ser mayor con umbral optimizado).
- **Fix:** tratar ResNet igual que los demás métodos: usar `resnet_prob` como score continuo y optimizar umbral por F1.

### 3.3 XGBoost: features incompletas
- **Dónde:** `07_build_xgboost_dataset.py`
- **Problema:** solo incluye mean/std/p95 por banda. El RF usa 6 features GLCM (contrast, correlation, energy, homogeneity, ASM, dissimilarity) que capturan la textura filiforme del plástico.
- **Fix:** añadir GLCM calculado en ventana 5×5 sobre las bandas clave (B08, B11, B04).

### 3.4 Sin intervalos de confianza en métricas
- **Dónde:** `09_evaluate.py`
- **Problema:** con n=50 la diferencia entre UNet AUC=0.893 y RF AUC=0.910 puede ser estadísticamente nula. No hay CI ni tests estadísticos.
- **Fix:** añadir bootstrap CI (1000 iteraciones), test DeLong para comparar AUCs, test McNemar para comparar matrices de confusión.

### 3.5 Evaluación de segmentación sin compensar shift residual
- **Dónde:** `09_evaluate.py`, función `build_segmentation_evaluation`
- **Problema:** compara máscaras predichas directamente con GT sin compensar el shift L1C→L2A, penalizando sistemáticamente a todos los modelos.
- **Fix a corto plazo:** filtrar la evaluación de segmentación solo a patches con `gt_quality=ALTA` (shift < 5px).

### 3.6 Naming convention no contempla tipo de negativo externo
- **Dónde:** `01_download_dataset.py` y `09_evaluate.py`
- **Problema:** solo existe CLARO/DUDOSO/DIFICIL como categorías de negativos. No hay distinción por tipo de confundidor (barcos, algas, espuma).
- **Fix:** añadir `BARCO`, `ALGAS`, `ESPUMA` como etiquetas válidas en el naming convention y en el análisis de FP por dificultad.

### 3.7 XGBoost actúa como ensemble encubierto sin documentarlo
- **Dónde:** `07_build_xgboost_dataset.py`, `08_train_xgboost.py`
- **Problema:** el feature más importante de XGBoost es `unet_pct` (15.4%). XGBoost no es un nuevo modelo independiente, es un meta-aprendedor sobre los otros modelos. Esto no es leakage técnico (los modelos MARIDA no fueron entrenados en el dataset de Gibraltar) pero debe documentarse explícitamente.
- **Fix:** experimento de ablation (ver sección 4.1).

---

## 4. Plan de mejoras por fases

---

### FASE 0 — Fundaciones (hacer antes que todo lo demás)

#### 0.1 Ampliar el dataset a ~80 SI + ~80 NO

Con n=50 el margen de error en las métricas hace que los resultados sean poco robustos. La meta mínima para el TFG es 100 patches (50+50), idealmente 160 (80+80).

- **Positivos:** ejecutar `01_download_dataset.py --n_positives 80`. Priorizar candidatos con `n_pixels >= 100` (filamentos grandes, GT más fiable) y fechas 2019–2021 (L2A más estable).
- **Negativos CLARO adicionales:** agua limpia lejos de costa y rutas de barcos.
- **Negativos DIFICIL adicionales:** zonas con espuma, sedimentos del Guadalquivir, corrientes turbulentas.

#### 0.2 Clasificar positivos por calidad del GT

Para cada patch positivo, añadir columna `gt_quality` al CSV maestro:

| Nivel | Criterio |
|---|---|
| `ALTA` | shift estimado < 5px AND n_px_gt > 50 |
| `MEDIA` | shift 5–20px OR n_px_gt 20–50 |
| `BAJA` | shift > 20px OR n_px_gt < 20 |

Los positivos `BAJA` siguen siendo útiles para evaluación patch-level pero se excluyen de la evaluación de segmentación.

#### 0.3 Redefinir el marco de evaluación en dos niveles

Establecer explícitamente en la memoria:

- **Evaluación primaria (patch-level):** AUC-ROC, F1, Precision, Recall a nivel de patch. Métrica principal. Robusta al error de alineación GT.
- **Evaluación secundaria (segmentación):** IoU, Precision-px, Recall-px. Solo sobre positivos con `gt_quality=ALTA`. Presentarla como "coherencia espacial aproximada con el catálogo WASP", no como segmentación perfecta. Incluir nota metodológica sobre el problema L1C→L2A.
- **Evaluación terciaria (validación externa):** correlación Spearman con concentraciones in situ del Marine Pollution Bulletin 2025.

---

### FASE 1 — Auditoría y corrección de XGBoost

#### 1.1 Aclaración sobre leakage

**No hay leakage técnico.** Los modelos MARIDA (UNet, RF, ResNet) fueron entrenados en el dataset MARIDA (Grecia, Corea, Adriático), no en los patches de Gibraltar. Sus predicciones no están contaminadas por el GT de este proyecto. El CV que usa XGBoost es correcto: los scores MARIDA son constantes (no se recalculan por fold).

**El problema real es otro:** XGBoost actúa principalmente como ensemble ponderado de los modelos base, no como un modelo genuinamente nuevo. Esto debe documentarse y verificarse con el siguiente experimento.

#### 1.2 Experimento de ablation obligatorio

Entrenar tres versiones en paralelo con el mismo CV:

| Versión | Features | Propósito |
|---|---|---|
| `XGB-solo` | 11 bandas × (mean, std, p95) + FDI/NDVI stats | ¿Qué aportan los features espectrales solos? |
| `XGB-modelos` | unet_pct, rf_pct, resnet_prob, fdi_pct, fdi_ndvi_pct | ¿Qué aportan los scores de modelos solos? |
| `XGB-completo` | todo (actual) | Ensemble completo |

Interpretación esperada:
- Si `XGB-completo >> XGB-solo` → el ensemble aporta valor real.
- Si `XGB-completo ≈ XGB-modelos` → XGBoost no añade nada sobre los modelos base.
- Si `XGB-solo ≈ XGB-completo` → los features espectrales son suficientes y los scores de modelos son redundantes.

#### 1.3 Añadir features GLCM a XGBoost

El RF MARIDA usa 6 features GLCM que capturan la textura filiforme del plástico. XGBoost no las tiene actualmente. Añadirlas a `07_build_xgboost_dataset.py`:
- Features: contrast, correlation, energy, homogeneity, ASM, dissimilarity
- Calculadas en ventana 5×5 sobre B08 (NIR), B11 (SWIR), B04 (Rojo)
- Librería: `skimage.feature.graycomatrix` + `graycoprops`

#### 1.4 Tests estadísticos

Añadir a `09_evaluate.py`:
- **Bootstrap CI** (1000 iteraciones) para AUC, F1, Precision, Recall de cada método
- **Test DeLong** para comparar AUCs entre pares de métodos (especialmente UNet vs RF: diferencia de 0.017 puede ser no significativa)
- **Test McNemar** para comparar matrices de confusión entre pares
- Reportar resultados como: `AUC = 0.893 [IC95%: 0.821–0.951]`

---

### FASE 2 — Modelos adicionales

#### 2.1 SAM — Spectral Angle Mapper (baseline espectral puro)

**Qué es:** compara el espectro de cada píxel con una firma de referencia de plástico mediante el ángulo espectral. No requiere entrenamiento.

**Cómo implementarlo:**
1. Obtener firma de referencia: media de las 11 bandas sobre todos los píxeles GT de positivos con `gt_quality=ALTA`
2. Para cada píxel del patch: `SAM_score = arccos(v·ref / (|v|·|ref|))`
3. Umbral: ángulo < θ_opt → plástico (θ_opt optimizado por F1 igual que los demás)
4. Script: `12_predict_sam.py`

**Por qué añadirlo:** es el baseline espectral más honesto. Si SAM rinde mejor que FDI/NDVI confirma que usar las 11 bandas aporta frente a 2-3 bandas. Si rinde peor que UNet/RF justifica el uso de ML.

#### 2.2 Calibración de ResNet (mejora inmediata, bajo coste)

Las probabilidades de ResNet están mal calibradas para el dominio Gibraltar/L2A (entrenado con ACOLITE). Aplicar **Platt scaling** (regresión logística sobre los outputs de ResNet) usando los 50 patches actuales con LOO.

Esto no re-entrena ResNet, solo ajusta la escala de sus probabilidades al nuevo dominio. Impacto esperado: AUC puede subir de 0.57 a ~0.65–0.75.

#### 2.3 Isolation Forest (detección de anomalías no supervisada)

**Qué es:** detecta patches o píxeles que se alejan de la distribución de agua limpia. Solo necesita negativos para entrenarse — completamente independiente del GT problemático del WASP.

**Dos variantes:**
- **Patch-level:** features espectrales de cada patch. Entrena con patches NO, detecta los SI como anomalías.
- **Pixel-level:** espectros por píxel. Entrena con píxeles de patches NO, detecta píxeles anómalos.

**Por qué añadirlo:** si detecta bien los positivos sin haberlos visto en entrenamiento, confirma que los filamentos tienen firma espectral genuinamente anómala respecto al agua limpia. También es útil para el mapa de alertas donde no hay GT.

#### 2.4 Ensemble logístico ponderado

En lugar de (o además de) XGBoost, crear un ensemble logístico que combine los scores de todos los modelos:
- Input: (unet_pct, rf_pct, resnet_prob_calibrado, fdi_pct, sam_score, iso_forest_score)
- Output: probabilidad de SI via regresión logística con LOO-CV
- Más interpretable que XGBoost: los coeficientes indican el peso de cada modelo

---

### FASE 3 — XGBoost para segmentación pixel-level

#### 3.1 Motivación

El RF MARIDA hace segmentación pixel-level con features GLCM pero fue entrenado en otro dominio. XGBoost pixel-level se entrenaría directamente en el dataset de Gibraltar, potencialmente superando al RF en este dominio específico.

#### 3.2 Leakage real en segmentación pixel-level

**Aquí sí hay un riesgo de leakage que debe manejarse explícitamente.** Si en el fold de validación se mezclan píxeles del mismo patch en train y test, el modelo aprende características específicas del patch (condiciones de iluminación, estado del mar ese día) que no generalizan.

**Solución obligatoria:** `sklearn.model_selection.GroupKFold` donde los grupos son los nombres de patch. En cada fold, dejar fuera todos los píxeles de un patch completo como test.

#### 3.3 Construcción del dataset pixel-level

Para cada patch positivo con `gt_quality=ALTA/MEDIA`:
- **Píxeles positivos (label=1):** todos los píxeles de la máscara WASP
- **Píxeles negativos (label=0):** muestreo de N × n_px_gt píxeles fuera de la máscara (balance 1:3 recomendado)

Para negativos: muestreo aleatorio de píxeles de cada patch NO.

Features por píxel: 11 bandas + FDI, NDVI, NDWI, FAI, BSI + GLCM(contrast, correlation, energy, homogeneity, ASM, dissimilarity) en ventana 5×5.

#### 3.4 Comparación directa

Comparar XGBoost pixel-level con RF-MARIDA pixel-level usando el mismo group-CV y las mismas métricas (IoU, Precision-px, Recall-px). Si XGBoost ≥ RF en dominio Gibraltar → justifica el entrenamiento en el dominio local.

---

### FASE 4 — Negativos de bases de datos de barcos

#### 4.1 Por qué los barcos son el confundidor crítico

Los barcos en el Estrecho de Gibraltar (una de las rutas más transitadas del mundo) tienen:
- Alta reflectancia en NIR/SWIR → FDI positivo
- Firma espectral parcialmente similar a plástico compacto
- Alta densidad en la zona de estudio

Sin negativos de tipo barco, el sistema puede tener FP rate muy alto en aplicación real sobre el mapa de alertas.

#### 4.2 Fuentes de coordenadas

| Fuente | Acceso | Coste |
|---|---|---|
| **Global Fishing Watch** | API pública gratuita | Requiere registro |
| **Marine Traffic** | API con cuota (datos históricos) | Parcialmente gratuito |
| **Manual + Sentinel Hub Browser** | Identificación visual de barcos en imágenes ya descargadas | Gratuito, lento |

Estrategia recomendada: usar Global Fishing Watch para obtener posiciones AIS de barcos pesqueros en el Estrecho por fecha, cruzar con las fechas de imágenes Sentinel-2 ya descargadas, descargar el patch centrado en el barco y validar que hay señal visible (pico de FDI > umbral en el centro del patch).

#### 4.3 Implementación

Script: `01b_download_ship_negatives.py`  
Input: CSV con (lat, lon, fecha, tipo_barco) de AIS  
Output: `YYYYMMDD_NO_000000_IDX_BARCO.tif`

Validación automática: si FDI_central < umbral_ruido → descartar (barco no visible o demasiado pequeño).

**Meta:** 20–30 negativos de tipo BARCO.

Naming convention actualizado: añadir `BARCO`, `ALGAS`, `ESPUMA` como etiquetas válidas en el análisis de FP por dificultad.

---

### FASE 5 — Fine-tuning de un modelo

#### 5.1 Requisito previo

**No hacer fine-tuning hasta tener ≥ 60 patches positivos.** Con 25 positivos actuales el overfitting es prácticamente inevitable.

#### 5.2 Estrategia recomendada: fine-tuning parcial de UNet

Solo afinar el decoder (últimas capas decodificadoras) manteniendo el encoder congelado:

- **Ventaja:** ~1M parámetros entrenables en lugar de ~10M. Menor riesgo de overfitting.
- **Input:** mismo formato (11 bandas, 256×256, normalizado con `bands_mean/std` MARIDA).
- **Output:** salida binaria (plástico vs no-plástico) con BCE loss en lugar de 11 clases.
- **Datos:** píxeles GT de positivos con `gt_quality=ALTA/MEDIA` como label=1, píxeles de fondo de negativos como label=0.
- **Augmentation:** flips horizontal/vertical (filamentos sin orientación fija), rotación ±30°, jitter de brillo (±10%).
- **Holdout estratificado:** reservar 20% de patches (los más recientes, p.ej. 2021) completamente fuera del fine-tuning desde el inicio.

#### 5.3 Evaluación honesta del fine-tuning

Comparar en el mismo holdout:
- UNet original (sin fine-tuning)
- UNet fine-tuned (decoder)
- UNet fine-tuned (completo, si hay suficientes datos)

Métricas: IoU en holdout, AUC patch-level en holdout. Si fine-tuned > original en ambas métricas → el fine-tuning aporta valor en el dominio Gibraltar/L2A.

#### 5.4 Alternativa más rápida

Si el fine-tuning completo es demasiado costoso en tiempo de implementación: usar **Platt scaling / ensemble logístico** (ver FASE 2.4) como sustituto. Es más fácil de justificar metodológicamente y puede dar resultados similares.

---

### FASE 6 — Mapa de alertas del Estrecho

#### 6.1 Pipeline de inferencia sobre escena completa

Script: `12_inference_scene.py`

1. **Descargar escena completa** del Estrecho vía OpenEO (bbox del KML, ~120×80 km = ~12.000×8.000 px a 10m). Dividir en tiles de 2048×2048 si la memoria es limitada.
2. **Sliding window inference:** ventana 256×256 con stride 128 (50% overlap) → ~5.500 patches. Ejecutar el mejor modelo (UNet o RF) sobre cada uno.
3. **Mosaico:** en zonas de overlap, promediar probabilidades de los patches solapantes.
4. **Binarización y clustering:** umbral óptimo (del dataset de validación) + DBSCAN sobre píxeles positivos para agrupar objetos individuales.
5. **Métricas por objeto:** centroide (lat/lon), área (n_px × 100m²), probabilidad media, fecha.
6. **Exportar:** GeoTIFF del mapa de probabilidad + GeoJSON de alertas.

#### 6.2 Visualización Leaflet

Archivo: `13_mapa_leaflet.html` (HTML estático, sin servidor)

- Fondo: OpenStreetMap o ESRI World Imagery
- Capa 1: mapa de calor semitransparente del score de probabilidad
- Capa 2: puntos de alerta (GeoJSON) con popup (fecha, probabilidad, área estimada)
- Capa 3: puntos del catálogo WASP de Gibraltar (los 876 candidatos del CSV Nature)
- Control de fecha si hay varias escenas procesadas

#### 6.3 Selección de fecha para el mapa

Priorizar escenas con:
- Cobertura de nubes < 10% en la zona del Estrecho
- Fecha con detecciones en el catálogo WASP (hay filamentos conocidos ese día)
- Condiciones de viento moderado (Poniente/Levante fuerte genera espuma que confunde los modelos)

---

### FASE 7 — Arquitectura de dos etapas

#### 7.1 La idea

**Etapa 1 — Detector de patch** (¿hay plástico en este tile de 256×256?):
- Modelo: XGBoost calibrado o ResNet calibrada
- Umbral bajo (recall alto): aceptar si prob > 0.25
- Si NO → descartar el patch sin procesarlo más
- Si SÍ → pasar a Etapa 2

**Etapa 2 — Segmentador pixel-level** (¿dónde está el plástico?):
- Modelo: UNet (original o fine-tuned)
- Solo se ejecuta en patches que pasaron Etapa 1

#### 7.2 Ventajas prácticas

Con umbral bajo en Etapa 1 (recall 0.95), se descartan ~60–70% de patches vacíos antes de aplicar UNet. Sobre las ~5.500 ventanas del mapa de alertas, solo se procesa con UNet un subconjunto (~1.600), reduciendo el tiempo de inferencia en ~3×.

El FN del sistema completo está dominado por FN de Etapa 1. Bajar el umbral de Etapa 1 reduce FN a costa de más trabajo en Etapa 2.

#### 7.3 Evaluación end-to-end

- Curva: FP_patch vs IoU_final al variar el umbral de Etapa 1
- Comparar: UNet sola (sin filtro) vs Sistema dos etapas
- Métrica objetivo: minimizar tiempo_inferencia_por_escena manteniendo patch-Recall > 0.90

---

### FASE 8 — Marco de evaluación estadístico completo

#### 8.1 Tests estadísticos a incluir en la memoria

| Test | Propósito | Cuándo |
|---|---|---|
| **Bootstrap CI** (1000 iter.) | IC95% para todas las métricas | Siempre, en `09_evaluate.py` |
| **DeLong test** | Comparar AUCs entre pares de métodos | Para cualquier par con diferencia AUC < 0.05 |
| **McNemar test** | Comparar matrices de confusión entre pares | Para comparar métodos con F1 similar |
| **Spearman ρ** | Correlación con concentraciones in situ | Validación externa (MPB 2025) |

Con n=50 es muy probable que DeLong y McNemar den p > 0.05 entre los mejores modelos. Ese es un resultado válido: "con el dataset actual no hay evidencia estadística de diferencia significativa entre UNet y RF".

#### 8.2 Validación con datos in situ

Usando el dataset del Marine Pollution Bulletin 2025 (Ramos-Alcántara et al., Suplementario S3):
1. Para cada punto de muestreo in situ (lat/lon), descargar el patch Sentinel-2 de la fecha más cercana al muestreo (ventana ±7 días).
2. Aplicar el mejor modelo y registrar el score de probabilidad.
3. Calcular correlación de Spearman entre score del modelo y concentración in situ (items/m²).
4. Si ρ > 0 y p < 0.05 → validación independiente del GT. Si ρ ≈ 0 → el modelo no se correlaciona con concentración real (puede ser problema de escala temporal o espacial).

---

## 5. Resumen de prioridades y orden

```
PRIORIDAD CRÍTICA — hacer antes que nada:
  [0.1] Ampliar dataset a 80+80 patches
  [0.2] Clasificar positivos por calidad de GT (gt_quality)
  [0.3] Redefinir marco de evaluación en dos niveles en la memoria
  [1.4] Cambiar threshold LOO: n < 30 → n < 100

ALTA PRIORIDAD — semanas 1-2:
  [1.2] Experimento ablation XGBoost (3 versiones)
  [1.3] Añadir features GLCM a XGBoost
  [2.2] Calibrar ResNet con Platt scaling
  [2.1] Implementar SAM como baseline espectral
  [4.x] Añadir 20-30 negativos de barcos
  [1.4] Bootstrap CI y tests DeLong/McNemar en 09_evaluate.py

PRIORIDAD MEDIA — semanas 3-4:
  [2.3] Isolation Forest (patch-level y pixel-level)
  [2.4] Ensemble logístico ponderado
  [3.x] XGBoost pixel-level con GroupKFold
  [8.2] Correlación Spearman con in situ

PRIORIDAD BAJA — si hay tiempo:
  [5.x] Fine-tuning parcial UNet (requiere ≥60 positivos)
  [6.x] Mapa de alertas del Estrecho + Leaflet
  [7.x] Arquitectura de dos etapas
```

---

## 6. Scripts a crear

| Script | Fase | Descripción |
|---|---|---|
| `01b_download_ship_negatives.py` | 4 | Descarga negativos de barcos desde AIS |
| `12_predict_sam.py` | 2.1 | Spectral Angle Mapper batch |
| `13_xgboost_pixel.py` | 3 | XGBoost segmentación pixel-level con GroupKFold |
| `14_calibrate_resnet.py` | 2.2 | Platt scaling sobre outputs de ResNet |
| `15_isolation_forest.py` | 2.3 | Isolation Forest patch y pixel-level |
| `16_ensemble_logistic.py` | 2.4 | Ensemble logístico ponderado |
| `17_inference_scene.py` | 6 | Inferencia sliding-window sobre escena completa |
| `18_mapa_leaflet.py` | 6 | Generador de GeoJSON + HTML Leaflet |
| `19_finetune_unet.py` | 5 | Fine-tuning parcial UNet (cuando haya ≥60 positivos) |

---

## 7. Nota final sobre el enfoque del TFG

La contribución más valiosa del TFG no es añadir más modelos a la tabla comparativa — es hacer la comparación de forma **metodológicamente honesta** en un dominio donde el ground truth es imperfecto.

Aportes originales que distinguen este TFG:
1. **Análisis del problema de alineación L1C→L2A** y su impacto en la evaluación de segmentación (sección 2 de este documento).
2. **Experimento de ablation de XGBoost** que distingue entre "ensemble de modelos" y "nuevo modelo".
3. **Clasificación de negativos por tipo de confundidor** (barcos vs espuma vs agua limpia) y análisis de FP por categoría.
4. **Aplicación sobre la zona del Estrecho** con visualización de alertas — ningún estudio previo ha publicado un mapa de detección de plástico específico para esta zona.
5. **Correlación con datos in situ** del MPB 2025 — única validación independiente del catálogo WASP disponible para este dominio.
