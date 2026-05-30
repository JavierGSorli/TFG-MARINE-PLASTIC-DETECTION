# TFG Marine Plastic Detection

Repositorio del Trabajo Fin de Grado sobre detección de residuos plásticos flotantes en Sentinel-2 en el área del Estrecho de Gibraltar.

El objetivo de este repositorio es permitir la **reconstrucción completa del flujo**:

1. selección de candidatos y descarga de imágenes,
2. ejecución de baselines,
3. calibración y evaluación,
4. construcción del sistema híbrido,
5. visualización geográfica de resultados.

## Estructura

```text
data/        datos fuente, modelos preentrenados y checkpoints necesarios
notebooks/   solo scripts auxiliares usados por el pipeline
pipeline/    pipeline final reproducible
```

## Requisitos

- Python 3.10 o 3.11
- Entorno con GDAL/Rasterio correctamente instalado
- Acceso a Copernicus Data Space / openEO para descargar Sentinel-2

Instalación básica:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Recursos incluidos necesarios

El pipeline depende de estos recursos versionados en el repo:

- `data/area_estudio/mapa_estrecho.kml`
- `data/windrows_nature/general/41467_2024_48674_MOESM3_ESM.xlsx`
- `data/marida/marine-debris.github.io/`
  - `semantic_segmentation/unet/`
  - `semantic_segmentation/random_forest/`
  - `multi-label/resnet/`
  - `utils/`
- scripts auxiliares todavía usados por el pipeline:
  - `notebooks/randomforest/predict_mask_rf.py`
  - `notebooks/indices/03_predict_indices.py`
  - `notebooks/indices/03_predict_indices2.py`

## Qué se conserva de `data/`

La carpeta `data/` no contiene salidas generadas por el pipeline, sino únicamente los recursos mínimos necesarios para poder reconstruirlo:

- `area_estudio/`: geometría del área de trabajo.
- `windrows_nature/`: datos fuente del estudio de Nature usados para construir el dataset.
- `marida/`: código y pesos mínimos tomados del estudio MARIDA para ejecutar los modelos preentrenados utilizados en este trabajo.
- `external_models/`: checkpoint del modelo externo `MarineDebrisDetector`.

No se versionan en `data/`:

- los patches Sentinel-2 descargados,
- las máscaras generadas,
- las predicciones raster,
- ni otros artefactos pesados producidos durante la ejecución.

Dos recursos necesarios para reproducir el flujo completo no se incluyen en GitHub por superar el límite de tamaño de archivos:

- `data/windrows_nature/detallado/11045944/WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc`
- `data/external_models/marinedebrisdetector/unetplusplus1.ckpt`
- `data/marida/marine-debris.github.io/data/dataset.h5`

El pipeline espera encontrarlos en esas rutas exactas. Deben descargarse manualmente antes de ejecutar:

- catálogo detallado WASP / Nature:
  - registro Zenodo: `https://zenodo.org/records/11045944`
  - fichero esperado por el pipeline: `WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc`
  - descarga directa habitual de Zenodo:
    - `https://zenodo.org/records/11045944/files/WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc?download=1`
- checkpoint externo `MarineDebrisDetector`:
  - repositorio original: `https://github.com/MarcCoru/marinedebrisdetector`
  - página de modelos con los enlaces oficiales: `https://github.com/MarcCoru/marinedebrisdetector/blob/main/doc/models.md`
  - carpeta pública de pesos indicada en el README del repositorio original:
    - `https://drive.google.com/drive/folders/1OBKr9G4zCP3X7fa8C7xBpJ8WNUyiajDL?usp=drive_link`
  - URL directa del checkpoint que usa este repositorio:
    - `https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/checkpoints/unet%2B%2B1/epoch=54-val_loss=0.50-auroc=0.987.ckpt`
- `dataset.h5` de MARIDA para construir las firmas SAM:
  - repositorio original de MARIDA: `https://github.com/marine-debris/marine-debris.github.io`
  - DOI principal del dataset MARIDA: `https://doi.org/10.5281/zenodo.5151941`
  - el propio README de MARIDA indica que `data/dataset.h5` puede generarse ejecutando `utils/spectral_extraction.py` tras descargar MARIDA completo, o descargarse aparte si se dispone de ese artefacto.

## Qué se conserva de `data/marida/`

La carpeta `data/marida/marine-debris.github.io/` corresponde al repositorio del estudio **MARIDA**. En este repositorio público no se incluye completa, sino solo la parte imprescindible para ejecutar el flujo actual:

- código de inferencia de `UNet`,
- código y modelos `Random Forest`,
- código de inferencia de `ResNet`,
- utilidades mínimas realmente importadas por esos scripts,
- pesos entrenados necesarios,
- y la estructura mínima para referenciar `dataset.h5`, usado para construir las firmas espectrales del clasificador SAM.

Se han excluido de esa carpeta:

- ejemplos,
- scripts de entrenamiento no usados por el pipeline,
- evaluaciones originales ajenas al flujo actual,
- datos de prueba,
- zips,
- cachés
- y material crudo que no aporta nada a la reproducibilidad del pipeline final.

## Papel de `notebooks/`

La carpeta `notebooks/` no se publica como material exploratorio general. Solo se conservan tres scripts auxiliares que el pipeline sigue invocando directamente:

- `notebooks/randomforest/predict_mask_rf.py`
- `notebooks/indices/03_predict_indices.py`
- `notebooks/indices/03_predict_indices2.py`

El resto del contenido histórico de `notebooks/` queda fuera del repositorio público.

## Datos generados

No se versionan:

- `data/sentinel2/` descargado o generado por el pipeline
- máscaras y predicciones `.tif`
- salidas pesadas dentro de `pipeline/outputs/`
- material auxiliar generado en `pipeline/reports/`

## Configuración de rutas

El pipeline ya no depende de rutas absolutas locales.

Por defecto resuelve la raíz del proyecto automáticamente. Si se quiere forzar otra ubicación, puede definirse:

```powershell
$env:TFG_PROJECT_ROOT="C:\ruta\al\repositorio"
```

## Descarga de datos Sentinel-2

La descarga de patches usa `openEO` sobre Copernicus Data Space. Antes de ejecutar la fase de dataset, debes autenticarte con el backend correspondiente.

El script implicado es:

- `pipeline/scripts/01_dataset/02_download_dataset_patches.py`

Si tu entorno requiere autenticación interactiva, hazla antes de lanzar el pipeline.

## Orden general de ejecución

Desde la raíz del repositorio:

```powershell
# 1. Construcción del dataset
python .\pipeline\scripts\01_dataset\01_select_nature_candidates.py
python .\pipeline\scripts\01_dataset\02_download_dataset_patches.py
python .\pipeline\scripts\01_dataset\03_annotate_patch_quality.py
python .\pipeline\scripts\01_dataset\04_build_grouped_metadata.py
python .\pipeline\scripts\01_dataset\05_build_grouped_splits.py

# 2. Baselines
python .\pipeline\scripts\02_baselines\01_run_unet_marida.py
python .\pipeline\scripts\02_baselines\02_run_rf_marida.py --all-modes
python .\pipeline\scripts\02_baselines\03_run_resnet_marida.py
python .\pipeline\scripts\02_baselines\04_run_spectral_indices.py
python .\pipeline\scripts\02_baselines\05_build_sam_signatures.py
python .\pipeline\scripts\02_baselines\06_run_sam_pixel_classifier.py
python .\pipeline\scripts\02_baselines\07_run_external_models.py --b09-mode zero
python .\pipeline\scripts\02_baselines\07_run_external_models.py --b09-mode copy_b8a
python .\pipeline\scripts\02_baselines\07_run_external_models.py --b09-mode interpolate_b8a_b11

# 3. Evaluación
python .\pipeline\scripts\03_evaluation\01_calibrate_thresholds.py
python .\pipeline\scripts\03_evaluation\02_generate_calibrated_outputs.py
python .\pipeline\scripts\03_evaluation\03_unify_predictions.py
python .\pipeline\scripts\03_evaluation\04_evaluate_patch_level.py
python .\pipeline\scripts\03_evaluation\05_evaluate_segmentation_pixelwise.py
python .\pipeline\scripts\03_evaluation\06_evaluate_segmentation_tolerant.py
python .\pipeline\scripts\03_evaluation\07_error_analysis_and_diagnostics.py

# 4. Híbrido y mapas
python .\pipeline\scripts\04_hybrid_and_maps\01_build_hybrid_detector_segmenter.py
python .\pipeline\scripts\04_hybrid_and_maps\02_generate_gibraltar_maps.py
python .\pipeline\scripts\04_hybrid_and_maps\03_visualize_all_mask_models.py
```

## Resultados principales

- evaluación: `pipeline/outputs/03_evaluation/`
- híbrido y mapas: `pipeline/outputs/04_hybrid_and_maps/`
- resúmenes para memoria:
  - `pipeline/outputs/tablas.md`
  - `pipeline/outputs/tablas2.md`

## Documentación específica del pipeline

La descripción detallada por fases está en:

- `pipeline/README.md`
