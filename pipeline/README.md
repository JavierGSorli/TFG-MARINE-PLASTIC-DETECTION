# Pipeline final

Esta carpeta contiene el pipeline final reproducible del TFG. El flujo está organizado en cuatro fases:

1. construcción del dataset,
2. ejecución de baselines,
3. evaluación,
4. sistema híbrido y visualización geográfica.

## Estructura

```text
scripts/   pasos ejecutables por fase
src/       módulos reutilizables
outputs/   salidas generadas por fase
reports/   copias ligeras de resúmenes en formato markdown
```

## Dependencias externas usadas por el pipeline

Además del propio código dentro de `pipeline/`, este flujo utiliza:

- `data/area_estudio/mapa_estrecho.kml`
- `data/windrows_nature/...`
- `data/marida/marine-debris.github.io/...`
- `notebooks/randomforest/predict_mask_rf.py`
- `notebooks/indices/03_predict_indices.py`
- `notebooks/indices/03_predict_indices2.py`

Las rutas se resuelven desde `src/common/config.py`.

## Qué parte externa se publica

Aunque el flujo principal está en `pipeline/`, para que el repositorio sea reproducible también se publican algunos recursos externos mínimos:

- `data/windrows_nature/`: datos fuente usados para construir el dataset.
- `data/area_estudio/`: geometría del área de trabajo.
- `data/marida/marine-debris.github.io/`: solo la parte imprescindible del estudio MARIDA.
- `data/external_models/marinedebrisdetector/`: checkpoint del modelo externo.
- `notebooks/`: únicamente tres scripts auxiliares todavía invocados por el pipeline.

En particular, la carpeta de MARIDA no se conserva completa. Solo se mantienen:

- scripts de inferencia realmente usados,
- pesos entrenados necesarios,
- utilidades importadas por esos scripts,
- `dataset.h5` para las firmas espectrales SAM.

No se incluyen ejemplos, datos de prueba, material crudo, zips, cachés ni otros artefactos que no sean necesarios para ejecutar el flujo actual.

Dos dependencias necesarias no se versionan en GitHub por exceder el límite de 100 MB por archivo:

- `data/windrows_nature/detallado/11045944/WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc`
- `data/external_models/marinedebrisdetector/unetplusplus1.ckpt`

Ambos deben descargarse manualmente y colocarse en esas rutas exactas antes de ejecutar el flujo completo.

Referencias de descarga:

- catálogo detallado Nature / litter windrows:
  - registro Zenodo: `https://zenodo.org/records/11045944`
  - fichero esperado por el pipeline: `WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc`
  - descarga directa habitual de Zenodo:
    - `https://zenodo.org/records/11045944/files/WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc?download=1`
- pesos del modelo externo `MarineDebrisDetector`:
  - repositorio original: `https://github.com/MarcCoru/marinedebrisdetector`
  - página de modelos con los enlaces oficiales: `https://github.com/MarcCoru/marinedebrisdetector/blob/main/doc/models.md`
  - carpeta pública de pesos indicada en el README del repositorio original: `https://drive.google.com/drive/folders/1OBKr9G4zCP3X7fa8C7xBpJ8WNUyiajDL?usp=drive_link`
  - URL directa usada en el script:
    - `https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/checkpoints/unet%2B%2B1/epoch=54-val_loss=0.50-auroc=0.987.ckpt`

## Fases

### 1. Dataset

```text
scripts/01_dataset/
  01_select_nature_candidates.py
  02_download_dataset_patches.py
  03_annotate_patch_quality.py
  04_build_grouped_metadata.py
  05_build_grouped_splits.py
```

Qué hace:

- selecciona candidatos positivos y negativos
- descarga patches Sentinel-2
- permite revisión visual y anotación de calidad
- construye metadata agrupada
- genera `calibration_dev`, `selection_dev`, `test_final` y folds internos con `GroupKFold`

Salida principal:

- `outputs/01_dataset/`

### 2. Baselines

```text
scripts/02_baselines/
  01_run_unet_marida.py
  02_run_rf_marida.py
  03_run_resnet_marida.py
  04_run_spectral_indices.py
  05_build_sam_signatures.py
  06_run_sam_pixel_classifier.py
  07_run_external_models.py
```

Qué hace:

- ejecuta UNet, RF, ResNet, índices espectrales, SAM y el modelo externo MarineDebrisDetector
- genera máscaras `pixel-wise`, scores `patch-level` y salidas auxiliares por método

Salida principal:

- `outputs/02_baselines/`
- además de predicciones pesadas dentro de `data/sentinel2/`

### 3. Evaluación

```text
scripts/03_evaluation/
  01_calibrate_thresholds.py
  02_generate_calibrated_outputs.py
  03_unify_predictions.py
  04_evaluate_patch_level.py
  05_evaluate_segmentation_pixelwise.py
  06_evaluate_segmentation_tolerant.py
  07_error_analysis_and_diagnostics.py
```

Qué hace:

- calibra umbrales sobre `calibration_dev`
- genera salidas calibradas
- unifica todas las predicciones en una tabla maestra
- evalúa patch-level, pixel-wise estricta y segmentación tolerante
- produce análisis de errores y diagnósticos

Salida principal:

- `outputs/03_evaluation/`

Resúmenes `.md` relevantes:

- `outputs/03_evaluation/patch_level/patch_level_summary.md`
- `outputs/03_evaluation/pixelwise/pixelwise_summary.md`
- `outputs/03_evaluation/tolerant/tolerant_summary.md`
- `outputs/03_evaluation/diagnostics/diagnostics_summary.md`

### 4. Híbrido y mapas

```text
scripts/04_hybrid_and_maps/
  01_build_hybrid_detector_segmenter.py
  02_generate_gibraltar_maps.py
  03_visualize_all_mask_models.py
```

Qué hace:

- selecciona detector y combinación de máscaras usando validación interna sobre `selection_dev`
- construye los perfiles híbridos `sensitive`, `balanced` y `conservative`
- genera mapas de resultados sobre el área de estudio
- facilita visualizaciones comparativas de máscaras

Salida principal:

- `outputs/04_hybrid_and_maps/`

Resumen `.md` relevante:

- `outputs/04_hybrid_and_maps/hybrid_predictions/hybrid_summary.md`

## Orden recomendado de ejecución

Ejecuta las fases en orden. Si vuelves a lanzar una fase anterior:

- si cambias dataset o baselines, rehace evaluación e híbrido
- si cambias evaluación, rehace híbrido

## Notas prácticas

- `outputs/` contiene material generado; no todo se versiona en Git.
- `reports/` contiene solo copias ligeras de algunos resúmenes markdown útiles para la memoria.
- los scripts de descarga requieren credenciales y acceso funcional a Copernicus Data Space vía `openEO`.
