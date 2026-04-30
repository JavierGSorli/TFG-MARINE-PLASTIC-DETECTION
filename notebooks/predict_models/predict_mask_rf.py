#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import rasterio
from joblib import load
from skimage.color import rgb2gray
from skimage.feature import greycomatrix
from os.path import dirname as up

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = up(up(THIS_DIR))
UTILS_PATH = os.path.join(ROOT_PATH, "utils")

if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

from assets import cat_mapping_vec
from random_forest import bands_mean
from engineering_patches import ndvi, fai, fdi, si, ndwi, nrd, ndmi, bsi, glcm_feature

np.seterr(divide="ignore", invalid="ignore")


def compute_indices_engineering(patch_chw):
    ndvi_ = ndvi(patch_chw[3], patch_chw[7])
    fai_ = fai(patch_chw[3], patch_chw[7], patch_chw[9])
    fdi_ = fdi(patch_chw[5], patch_chw[7], patch_chw[9])
    si_ = si(patch_chw[1], patch_chw[2], patch_chw[3])
    ndwi_ = ndwi(patch_chw[2], patch_chw[7])
    nrd_ = nrd(patch_chw[3], patch_chw[7])
    ndmi_ = ndmi(patch_chw[7], patch_chw[9])
    bsi_ = bsi(patch_chw[1], patch_chw[3], patch_chw[7], patch_chw[9])

    return np.stack([ndvi_, fai_, fdi_, si_, ndwi_, nrd_, ndmi_, bsi_], axis=-1).astype(
        "float32"
    )


def compute_texture_engineering(patch_chw, window_size=13, max_value=16):
    dtype = patch_chw.dtype

    # Igual que engineering_patches.texture: usa src.read((2,3,4))
    img = patch_chw[[1, 2, 3]].astype(dtype).copy()
    img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])

    rgb_composite = img[:, :, [2, 1, 0]]
    rgb_composite[rgb_composite < 0.0] = 0.0
    rgb_composite[rgb_composite > 0.15] = 0.15
    rgb_composite = rgb_composite / 0.15
    gray = rgb2gray(rgb_composite)

    bins = np.linspace(0.00, 1.00, max_value)
    num_levels = max_value + 1

    assert (window_size - 1) % 2 == 0

    pad = (window_size - 1) // 2
    temp_gray = np.pad(gray, pad, mode="reflect")
    features_results = np.zeros((gray.shape[0], gray.shape[1], 6), dtype=dtype)

    # Se mantiene el mismo patrón que engineering_patches.py para máxima fidelidad.
    for col in range(pad, gray.shape[0] + pad):
        for row in range(pad, gray.shape[0] + pad):
            temp_gray_window = temp_gray[
                row - pad : row + pad + 1,
                col - pad : col + pad + 1,
            ]

            inds = np.digitize(temp_gray_window, bins)
            matrix_coocurrence = greycomatrix(
                inds,
                [1],
                [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=num_levels,
                normed=True,
                symmetric=True,
            )
            matrix_coocurrence = matrix_coocurrence.mean(3)[:, :, :, np.newaxis]

            con, dis, homo, ener, cor, asm = glcm_feature(matrix_coocurrence)
            features_results[row - pad, col - pad, 0] = con
            features_results[row - pad, col - pad, 1] = dis
            features_results[row - pad, col - pad, 2] = homo
            features_results[row - pad, col - pad, 3] = ener
            features_results[row - pad, col - pad, 4] = cor
            features_results[row - pad, col - pad, 5] = asm

    return features_results.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_tif", required=True, help="Patch multibanda (GeoTIFF).")
    ap.add_argument(
        "--model_path",
        default=os.path.join(THIS_DIR, "rf_classifier.joblib"),
        help="Ruta al rf_classifier.joblib",
    )
    ap.add_argument("--out_mask", required=True, help="GeoTIFF de salida con mascara.")
    ap.add_argument("--input_channels", type=int, default=11)
    ap.add_argument("--auto_scale", action="store_true", help="Si max>1.5, divide entre 10000.")
    ap.add_argument("--window_size", type=int, default=13)
    ap.add_argument("--max_value", type=int, default=16)
    args = ap.parse_args()

    print("Cargando modelo:", args.model_path)
    model = load(args.model_path)

    with rasterio.open(args.patch_tif, "r") as src:
        meta = src.meta.copy()
        tags = src.tags().copy()
        patch_raw = src.read().astype("float32")  # (C,H,W)
        dtype = src.read(1).dtype

    if patch_raw.shape[0] != args.input_channels:
        raise ValueError(f"Esperaba {args.input_channels} bandas pero hay {patch_raw.shape[0]}.")

    if args.auto_scale and patch_raw.max() > 1.5:
        patch_raw = patch_raw / 10000.0

    # Parte espectral: igual que train_eval.py, se imputan NaN con bands_mean.
    patch_hwc = np.moveaxis(patch_raw, (0, 1, 2), (2, 0, 1))
    h, w, _ = patch_hwc.shape
    impute_nan = np.tile(bands_mean, (h, w, 1))
    spectral_hwc = patch_hwc.copy()
    nan_mask = np.isnan(spectral_hwc)
    spectral_hwc[nan_mask] = impute_nan[nan_mask]
    spectral_features = spectral_hwc.reshape(h * w, -1)

    # Parte derivada: se calcula sobre el patch original, como en engineering_patches.py.
    print("Calculando indices con funciones de engineering_patches.py...")
    indices_hwc = compute_indices_engineering(patch_raw)
    indices_features = np.nan_to_num(indices_hwc).reshape(h * w, -1)

    print("Calculando textura GLCM con funciones de engineering_patches.py...")
    glcm_hwc = compute_texture_engineering(
        patch_raw,
        window_size=args.window_size,
        max_value=args.max_value,
    )
    glcm_features = np.nan_to_num(glcm_hwc).reshape(h * w, -1)

    x = np.concatenate([spectral_features, indices_features, glcm_features], axis=1)

    print("Feature matrix shape:", x.shape)
    print("Prediciendo...")
    pred_labels = model.predict(x)

    pred_numeric = cat_mapping_vec(pred_labels).astype(dtype).reshape(h, w)

    meta.update(count=1, dtype=dtype)
    os.makedirs(os.path.dirname(args.out_mask) or ".", exist_ok=True)

    with rasterio.open(args.out_mask, "w", **meta) as dst:
        dst.write_band(1, pred_numeric)
        dst.update_tags(**tags)

    print("OK ->", args.out_mask)


if __name__ == "__main__":
    main()
