# Plan de implementación TFG — Orden general por fases A-E

## Contexto

El proyecto actual evalúa detección de residuos/plásticos flotantes en Sentinel-2 usando:

- patches positivos derivados del catálogo Nature / litter windrows;
- negativos generados y revisados manualmente;
- modelos MARIDA preentrenados;
- índices espectrales;
- un ensemble propio tipo XGBoost;
- evaluación patch-level y pixel-level.

El objetivo de este plan es mejorar la solidez metodológica sin hacer que Codex implemente demasiadas cosas a la vez.

## Principios obligatorios

1. **No crear muchos modos de negativos.**  
   No implementar `matched_same_date`, `far_spatial`, `hard_negative`, etc. como modos separados. Mantener la generación sencilla.

2. **Mejorar principalmente la diversidad temporal.**  
   Lo importante no es cambiar completamente la lógica de negativos, sino evitar que unas pocas fechas dominen el dataset.

3. **No usar GroupKFold por fecha para modelos que no se entrenan con este dataset.**  
   Para U-Net MARIDA, RF MARIDA, ResNet MARIDA e índices espectrales, basta con análisis global y análisis por fecha/calidad.  
   La validación agrupada por fecha solo aplica a modelos entrenados o ajustados con nuestro dataset.

4. **Los modelos propios sí deben evaluarse evitando leakage temporal.**  
   Esto incluye:
   - XGBoost;
   - modelos clásicos entrenados sobre features del dataset;
   - stacking;
   - calibración aprendida;
   - fine-tuning;
   - selección de umbrales aprendida con los datos.

5. **Separar claramente patch-level y pixel-level.**  
   El rendimiento patch-level puede ser alto aunque la segmentación pixel-level sea limitada.

6. **No renombrar todos los archivos de golpe.**  
   Primero crear `dataset_metadata.csv` y usarlo como fuente de verdad. El nombre del fichero no debe seguir siendo la base de datos principal.

---

# Orden recomendado

## Fase A — Consolidar resultados actuales

Objetivo: dejar los resultados actuales limpios, reproducibles y bien separados.

Implementar:

1. corrección del análisis de errores;
2. resumen baseline v1;
3. análisis por fecha;
4. separación clara de clasificación patch-level y segmentación pixel-level.

Archivo para Codex:

- `01_fase_A_consolidar_resultados_actuales.md`

---

## Fase B — Mejorar dataset, metadatos, etiquetas y diversidad temporal

Objetivo: mejorar la base de datos sin cambiar todavía los modelos.

Implementar:

1. `dataset_metadata.csv`;
2. anotaciones manuales más ricas para negativos y calidad de imagen;
3. modificación sencilla de `01_download_dataset.py` para ampliar rango temporal y limitar concentración por fecha;
4. mantener compatibilidad con nombres antiguos.

Archivo para Codex:

- `02_fase_B_dataset_metadatos_y_diversidad_temporal.md`

---

## Fase C — Validación sólida para modelos propios

Objetivo: evitar leakage solo donde realmente aplica: modelos entrenados con nuestros datos.

Implementar:

1. splits agrupados por fecha para XGBoost/modelos propios;
2. selección de umbrales dentro de train;
3. evaluación conservadora;
4. comparación con resultados exploratorios.

Archivo para Codex:

- `03_fase_C_validacion_modelos_propios_sin_leakage.md`

---

## Fase D — Comparación de modelos y features

Objetivo: ampliar la comparación, pero una vez que dataset y validación estén más limpios.

Implementar:

1. integración de modelos preentrenados de literatura, si son fáciles de ejecutar;
2. modelos clásicos propios;
3. features SAM y GLCM;
4. ablation.

Archivo para Codex:

- `04_fase_D_comparacion_modelos_features_sam_glcm.md`

---

## Fase E — Producto final: modelo híbrido, mapas y outputs para memoria

Objetivo: generar un sistema final interpretable y visual.

Implementar:

1. modelo híbrido detección + segmentación;
2. mapas del Estrecho;
3. figuras y tablas finales;
4. outputs claros para memoria.

Archivo para Codex:

- `05_fase_E_modelo_hibrido_mapas_outputs_finales.md`

---

# Recomendación práctica

No pasar todos los archivos a Codex a la vez.

Orden recomendado:

1. Fase A.
2. Revisar outputs.
3. Fase B.
4. Revisar dataset.
5. Fase C.
6. Revisar XGBoost/modelos propios.
7. Fase D.
8. Fase E.

