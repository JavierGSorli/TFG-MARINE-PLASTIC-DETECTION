# Planteamiento y Plan de Trabajo: Detección de Plásticos en el Estrecho

Este documento responde a tus dudas metodológicas y propone un plan de acción estructurado para tu TFG. Se ha actualizado para incluir el uso de la **ResNet multi-etiqueta** y dar menor prioridad a re-evaluar los datasets de test de MARIDA.

## 1. Resolución de Dudas Metodológicas

### A. El problema del "Ground Truth" (Verdad Terreno) basado en Índices
Tienes mucha razón al dudar. El dataset de *Nature* no es una Verdad Terreno pura. Es una **Verdad Terreno semi-automática** obtenida buscando anomalías espectrales con índices (NDVI, FDI, etc.).
*   **¿Es trampa evaluar modelos de índices con esto?** Sí, en gran medida. Si evalúas tu script de índices (`03_predict_indices.py`) contra la máscara de *Nature*, en el fondo estás midiendo la "similitud entre algoritmos de cálculo de índices". Servirá como "Baseline" (línea base) para comprobar que replicas su método de extracción.
*   **¿Se pueden evaluar U-Net, Random Forest y ResNet con esto?** Sí. Al ser arquitecturas de *Machine/Deep Learning* entrenadas con un dataset distinto (*MARIDA*, etiquetado a mano píxel a píxel), evaluarlas contra las coordenadas de *Nature* sirve para medir su **capacidad de generalización**: comprobar si algoritmos entrenados de otra forma logran encontrar las acumulaciones reales.

### B. Diferencia de Diferencial (Segmentación vs Clasificación)
1.  **Segmentación Píxel a Píxel (U-Net, Random Forest, Índices):** Devuelven una máscara. Las clases de MARIDA (11-15) deben ser binarizadas. `Plástico` será la clase 1 (Marine Debris). `Fondo` será el resto. Como *Nature* engloba estela y espuma mientras MARIDA solo marca el píxel de plástico puro, evaluaremos mediante **detección orientada a objetos**: Si dentro del polígono detectado por *Nature* (Verdad Terreno) el modelo detecta un umbral mínimo de píxeles de plástico, lo consideraremos un acierto (Verdadero Positivo).
2.  **Clasificación de Imágen (ResNet Multi-Label):** Nos dice si en el parche entero existe o no plástico. Su evaluación es más limpia para este caso. Sabiendo si el parche lo descargaste como candidato positivo (centrado en un "Litter Window" de Nature) o como un parche negativo de mar limpio, cruzaremos la predicción de la categoría de Debris de la ResNet con la clase subyacente del parche.

### C. Replicar los resultados de MARIDA en su test (Opcional)
Como bien indicas, tener los modelos congelados del GitHub de MARIDA ya te da garantías de que funcionan con el rendimiento indicado en su paper (entorno a 0.70-0.80 en F1). Dejaremos esta validación como algo extra / opcional si la memoria requiere páginas de introducción a los modelos empleados, centrándonos en nuestro objetivo geoespacial.

### D. Entrenar un nuevo modelo con imágenes del Estrecho
*   Si entrenas una U-Net usando las máscaras de *Nature* como *Ground Truth*, la red simplemente aprenderá a emular la fórmula de los índices.
*   **Solución para aportar valor real:** Partir de los pesos pre-entrenados (*Transfer Learning*) de la U-Net o ResNet de MARIDA, procediendo a realizar un ajuste ligero (*Fine-Tuning*) sobre un conjunto de parches curados del Estrecho. Estos parches contendrán falsos positivos frecuentes de la zona como barcos particulares o estelas que engañan a MARIDA. De esta forma la adaptas estrictamente al Estrecho.

### E. Mapear todo el Estrecho
En teledetección de alta resolución y nubes, no tiene sentido generar una megafoto puntual. En su lugar, generaremos un **Mapa de Calor (Heatmap) Espaciotemporal**. Agregaremos las detecciones de tu mejor modelo a lo largo de los pases satelitales (ej: los pases limpios del Verano de 2021) resumiendo en marcadores o gradientes calientes las áreas con acumulación frecuente de residuos.

---

## 2. Revisión de tus Notebooks/Scripts actuales
*   `00_explorar_gibraltar.py` y `01_4.py`: Sólida base de descarga de candidatos y limpieza de nubes para conseguir el dataset local.
*   `ResNet_prediction` *(Faltante)*: Necesitamos crear el homólogo, por ejemplo `04_predict_resnet.py`, que procese los geotiffs con el dataloader del GitHub de ResNet para escupir si existen las etiquetas deseadas (Marine Debris).
*   `02_predict_unet.py` y `03_predict_indices.py`: Funcionan como inferencia ciega, debemos transicionarlos para que contrasten su resultado frente al `_mask.tif` en ese mismo instante calculando si acertó o no. Se aconseja centralizar esta lógica.

---

## 3. Plan de Trabajo Propuesto (Fases)

### Fase 1: Creación del Pipeline de Inferencia & Evaluación Unificados
1.  **Script para la ResNet:** Escribir el script que consume los patches CDSE listos y evalúa probabilidad de plástico.
2.  **Unificación de Evaluación:** Construir la lógica de comparación para todos los modelos contra los datos que salen de `01_4.py`.
    *   *Para ResNet:* Comparar Etiqueta Imagen (SI/NO plástico) vs Etiqueta Parche (SI/NO candidato).
    *   *Para U-Net / RF / Índices:* Comprobar superposición de píxeles clasificados como plástico sobre la forma de la `_mask.tif` u umbral orientativo. Extraer métricas tabuladas conjuntas.

### Fase 2: Análisis Comparativo Masivo
1.  Empleando `01_4.py`, almacenar un volumen de validación de por ejemplo 50 o 100 parches equilibrados (50 si, 50 no).
2.  Correr la evaluación de la *Fase 1* elaborando una tabla con resultados comparativos por modelo y falsos positivos / negativos representativos (con las cajitas y plots que adornen la memoria).

### Fase 3: Adaptación al Contexto del Estrecho (Transfer Learning)
1.  (De nuevo, opcional a medida que avances). Seleccionar en qué se equivoca repetidamente U-Net y corregirlo, añadiéndole una etapa de entreno con un ratio bajísimo de pérdida. 

### Fase 4: Despliegue en el Mapeo Geográfico
1.  Realizar un rastreo indiscriminado con el mejor modelo en un mes concreto.
2.  Visualizar resultados como polígonos/puntos vectoriales sobre base de QGIS o mapa folium, afrontando el factor nube vs pase del satélite en la justificación.

## > User Review Required
Por favor, lee el documento y confírmame:
1.  ¿Estamos de acuerdo en que la evaluación de ResNet será a nivel de parche completo (si la imagen es de un candidato validado de Nature, la ResNet debe clasificar "Contiene Debris", y si es un negativo corroborado por ti como el pipeline `01_4`, debe reportar "No Debris")?
2.  ¿Apruebas este plan adaptado para comenzar a codificar la inferencia cruzada y extracción de métricas de la **Fase 1**?
