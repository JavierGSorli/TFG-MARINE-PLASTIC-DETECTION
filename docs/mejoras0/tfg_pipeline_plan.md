# 🧠 Pipeline y Plan de Implementación — TFG Detección de Residuos Marinos

## 📌 Estado actual del proyecto

Actualmente el pipeline ya cubre tres bloques fundamentales:

### 1. Construcción del dataset
- Selección espacial con máscara KML
- Filtrado de candidatos (tamaño, distancia a costa)
- Descarga de patches Sentinel-2
- Generación de máscaras positivas (Nature)
- Generación de negativos (automático + validación manual)

👉 Resultado: dataset propio **muy sólido y realista**

---

### 2. Modelos implementados
- **U-Net (MARIDA)** → segmentación
- **Random Forest (MARIDA)** → clasificación pixel-level
- **ResNet (MARIDA)** → clasificación multi-label por patch

👉 Cubres:
- Deep Learning
- ML clásico
- Clasificación semántica

---

### 3. Índices espectrales (baseline)
- FDI
- NDVI
- FDI + NDVI (con threshold adaptativo)

👉 Esto equivale a un **baseline tipo Nature**

---

## ⚠️ Insight clave

El proyecto ya no está en fase de desarrollo inicial, sino en fase de:

👉 **integración, comparación y análisis**

---

# 🎯 Objetivo del TFG

Evaluar y comparar distintos métodos de detección de residuos marinos en Sentinel-2 utilizando como referencia el catálogo de windrows del Mediterráneo.

---

# 🧱 Arquitectura final del pipeline

Dataset (patches)
↓
Índices (FDI, NDVI)
↓
Modelos:
- U-Net
- RF
- ResNet
- XGBoost (nuevo)
↓
Unificación de resultados
↓
Evaluación comparativa
↓
Análisis de errores

---

# 📂 Archivos necesarios

## ✅ Ya existentes
- 00_explorar_gibraltar.py
- 01_download_dataset.py
- 02_predict_unet.py
- predict_mask_rf.py
- predict_resnet.py
- 03_predict_indices.py

---

## ⭐ Nuevos archivos a crear

### 04_unify_predictions.py
Unifica todas las salidas en un CSV maestro

### 05_build_xgboost_dataset.py
Construye dataset tabular por patch

### 06_train_xgboost.py
Entrena modelo XGBoost

### 07_evaluate.py
Evalúa y compara todos los métodos

### 08_error_analysis.py
Análisis cualitativo de errores

### config.py
Configuración centralizada (opcional)

---

# 🚀 Implementation Plan

## 🔵 Fase 1 — Unificación
Crear CSV maestro con todos los métodos

## 🔵 Fase 2 — Análisis inicial
Comparar SI vs NO

## 🔵 Fase 3 — XGBoost
Entrenar modelo patch-level

## 🔵 Fase 4 — Evaluación
Comparar métodos con métricas

## 🔵 Fase 5 — Error analysis
Visualización y análisis cualitativo

---

# 🧠 Contribución del TFG

Evaluación crítica de métodos:
- Índices espectrales
- Machine Learning
- Deep Learning

Aplicado a datos reales

---

# ⚠️ Qué NO hacer

- No reentrenar modelos complejos
- No usar GANs
- No complicar pipeline innecesariamente

---

# 💡 Insight final

El problema está limitado por la física del sensor, no solo por el modelo.

---

# 🎯 Prioridad inmediata

Implementar: **04_unify_predictions.py**
