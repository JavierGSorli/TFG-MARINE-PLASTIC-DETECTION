# Revision de tu estructura actual

## Lo que estaba mal o fragil

- Tenias duplicacion entre `notebooks/` y `notebooks/predict_models/`, lo que complica saber cual es la version buena.
- `config.py` existia, pero casi todos los scripts seguian usando rutas absolutas hardcodeadas.
- El plan habla de `01_download_dataset.py`, pero el script real era `01_5.py`.
- `02_predict_unet.py` guardaba en `test_masks_final`, mientras que la unificacion esperaba `test_masks_unet`.
- No habia scripts batch claros para RF y ResNet dentro del flujo final.
- Varios scripts recorrían `*.tif` y eso incluia tambien los `_mask.tif`, mezclando ground truth con patches reales.
- `01_5.py` tenia un bug importante: comprobaba la descarga con `if not download_and_convert(...)`, pero esa funcion devuelve strings; asi no detectaba fallos reales.
- El `predict_mask.py` copiado en `notebooks/predict_models` tiene un typo en el argumento `--autqo_scale` y despues usa `args.auto_scale`.
- Los scripts locales copiados de RF y ResNet dependian de imports relativos pensados para la estructura MARIDA y en esa ubicacion podian romperse.

## Lo que hace la version unificada

- Centraliza rutas y salidas.
- Renombra el pipeline con orden claro de `00` a `10`.
- Separa la revision de tu codigo original de la version de entrega.
- Mantiene la salida en `results/auto/`, para no romper tu trabajo ya generado.
