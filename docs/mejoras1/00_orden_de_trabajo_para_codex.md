# Orden recomendado de trabajo para Codex

Este documento divide las mejoras del TFG en bloques pequeños. No conviene pasar todos a Codex de golpe. Ejecuta un bloque, revisa que funciona, mira las salidas y solo después pasa al siguiente.

## Orden recomendado

### Bloque 1 — Metodología y circularidad FDI
Archivo: `01_metodologia_ground_truth_y_fdi.md`

Objetivo: dejar claro en documentación y código que las etiquetas son weak/proxy labels, no ground truth absoluto. También guardar información de alineación de máscaras positivas (`dr`, `dc`, `alignment_shift_magnitude`).

Prioridad: muy alta.

No debería cambiar resultados de modelos, solo mejorar trazabilidad y solidez metodológica.

---

### Bloque 2 — Reetiquetado de negativos
Archivo: `02_reetiquetado_negativos.md`

Objetivo: sustituir la lógica ambigua de `CLARO/DIFICIL/DUDOSO` por una anotación más útil: decisión, tipo de escena y confianza.

Prioridad: muy alta.

Este bloque mejora mucho la calidad del dataset.

---

### Bloque 3 — Control de calidad de positivos y scoring de confianza
Archivo: `03_control_calidad_positivos_confianza.md`

Objetivo: crear métricas de calidad para positivos y negativos, generar `patch_confidence.csv` y permitir evaluar subconjuntos de alta/media/baja confianza.

Prioridad: muy alta.

Este bloque responde directamente a la duda metodológica principal del TFG: cuán fiables son las etiquetas.

---

### Bloque 4 — Validación sin leakage
Archivo: `04_validacion_sin_leakage.md`

Objetivo: crear grupos de validación por fecha/evento/zona y rehacer el entrenamiento/evaluación de XGBoost con `GroupKFold` o estrategia equivalente.

Prioridad: muy alta.

Este bloque debe hacerse antes de añadir muchos modelos clásicos.

---

### Bloque 5 — Ablation de XGBoost y modelos clásicos mínimos
Archivo: `05_ablation_xgboost_y_modelos_clasicos_minimos.md`

Objetivo: medir qué aporta cada familia de features: bandas, índices, modelos MARIDA, texturas, etc. Añadir pocos modelos clásicos, no una lista enorme.

Prioridad: alta.

---

### Bloque 6 — SAM y GLCM/texturas
Archivo: `06_sam_y_glcm_features.md`

Objetivo: añadir Spectral Angle Mapper como score de confianza espectral y features GLCM/texturales para evaluar si ayudan.

Prioridad: media.

No usar SAM para afirmar “plástico puro”, sino como indicador auxiliar de similitud espectral.

---

### Bloque 7 — Modelo híbrido y mapas
Archivo: `07_modelo_hibrido_y_mapas.md`

Objetivo: crear una salida operativa combinando detección patch-level y segmentación pixel-level. Crear mapas del Estrecho para visualizar candidatos, patches y predicciones.

Prioridad: media-alta.

---

### Bloque 8 — Opcionales / future work
Archivo: `08_opcionales_futuro.md`

Objetivo: dejar como opcional calibración formal, fine-tuning y datasets externos de barcos.

Prioridad: baja/media.

No hacer este bloque hasta haber completado los anteriores.

## Recomendación práctica

Si el tiempo es limitado, ejecutar solo estos bloques:

```text
1. 01_metodologia_ground_truth_y_fdi.md
2. 02_reetiquetado_negativos.md
3. 03_control_calidad_positivos_confianza.md
4. 04_validacion_sin_leakage.md
5. 05_ablation_xgboost_y_modelos_clasicos_minimos.md
```

Con esos cinco bloques el TFG queda metodológicamente mucho más defendible.
