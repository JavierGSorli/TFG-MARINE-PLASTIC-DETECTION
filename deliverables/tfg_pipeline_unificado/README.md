# TFG Pipeline Unificado

Version simplificada y coherente del pipeline de deteccion de residuos marinos.

## Estructura

- `scripts/`: todos los scripts en el orden real de ejecucion.
- `docs/REVIEW.md`: problemas detectados en tu version actual.

## Requisitos

- Esta carpeta debe vivir dentro del repo `tfg-marine-plastic-detection`.
- Si la extraes fuera, define `TFG_PROJECT_ROOT` apuntando a la raiz del repo.
- Usa el mismo entorno Python que ya utilizas para `rasterio`, `openeo`, `xgboost`, `torch` y `scikit-learn`.

## Orden de ejecucion

Desde `deliverables/tfg_pipeline_unificado/scripts`:

```powershell
python 00_explore_candidates.py
python 01_download_dataset.py --n_positives 3
python 02_predict_unet.py
python 03_predict_rf.py
python 04_predict_resnet.py
python 05_predict_indices.py
python 06_unify_predictions.py
python 07_build_xgboost_dataset.py
python 08_train_xgboost.py
python 09_evaluate.py
python 10_error_analysis.py
```

## Salidas

Todos los resultados van a `results/auto/`:

- `test_patches_final/`
- `test_masks_unet/`
- `test_masks_rf/`
- `test_resnet_json/`
- `test_indices/`
- `predictions_master.csv`
- `xgboost_dataset.csv`
- `xgboost_model/`
- `evaluation/`
- `error_analysis/`

## Criterios de simplificacion

- Una sola configuracion central en `config.py`.
- Nombres consecutivos y consistentes.
- Scripts batch para UNet, RF y ResNet.
- Se excluyen siempre los `*_mask.tif` al recorrer patches.
- No se toca tu carpeta original `notebooks/predict_models`.
