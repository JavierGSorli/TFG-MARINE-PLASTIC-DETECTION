#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import rasterio
from joblib import load
from skimage.color import rgb2gray
try:
    from skimage.feature import graycomatrix as greycomatrix
except ImportError:
    from skimage.feature import greycomatrix
from pathlib import Path

_MARIDA_ROOT = Path(__file__).resolve().parents[2] / "data" / "marida" / "marine-debris.github.io"
_MARIDA_RF = _MARIDA_ROOT / "semantic_segmentation" / "random_forest"
for _p in [str(_MARIDA_ROOT / "utils"), str(_MARIDA_RF)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from assets import cat_mapping_vec
from random_forest import bands_mean
from engineering_patches import ndvi, fai, fdi, si, ndwi, nrd, ndmi, bsi, glcm_feature

np.seterr(divide="ignore", invalid="ignore")

SPECTRAL_FEATURES = [
    "nm440", "nm490", "nm560", "nm665", "nm705", "nm740",
    "nm783", "nm842", "nm865", "nm1600", "nm2200",
]
INDEX_FEATURES = ["NDVI", "FAI", "FDI", "SI", "NDWI", "NRD", "NDMI", "BSI"]
TEXTURE_FEATURES = ["CON", "DIS", "HOMO", "ENER", "COR", "ASM"]


def resolve_debris_class_index(model) -> int:
    if not hasattr(model, "classes_"):
        raise AttributeError("El modelo RF no expone classes_, necesario para guardar probabilidades.")
    classes = list(model.classes_)
    normalized = [str(cls).strip().lower() for cls in classes]
    for candidate in ("marine debris", "marine_debris"):
        if candidate in normalized:
            return normalized.index(candidate)
    raise ValueError(f"No se encontró la clase 'Marine Debris' en classes_: {classes}")


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


def build_feature_matrix(
    patch_raw: np.ndarray,
    feature_mode: str,
    window_size: int,
    max_value: int,
) -> np.ndarray:
    patch_hwc = np.moveaxis(patch_raw, (0, 1, 2), (2, 0, 1))
    h, w, _ = patch_hwc.shape
    feature_blocks = []

    if feature_mode in {"full", "no_texture", "bands_only"}:
        impute_nan = np.tile(bands_mean, (h, w, 1))
        spectral_hwc = patch_hwc.copy()
        nan_mask = np.isnan(spectral_hwc)
        spectral_hwc[nan_mask] = impute_nan[nan_mask]
        spectral_features = spectral_hwc.reshape(h * w, -1)
        feature_blocks.append(spectral_features)

    if feature_mode in {"full", "no_texture", "indices_only"}:
        print("Calculando indices con funciones de engineering_patches.py...")
        indices_hwc = compute_indices_engineering(patch_raw)
        indices_features = np.nan_to_num(indices_hwc).reshape(h * w, -1)
        feature_blocks.append(indices_features)

    if feature_mode == "full":
        print("Calculando textura GLCM con funciones de engineering_patches.py...")
        glcm_hwc = compute_texture_engineering(
            patch_raw,
            window_size=window_size,
            max_value=max_value,
        )
        glcm_features = np.nan_to_num(glcm_hwc).reshape(h * w, -1)
        feature_blocks.append(glcm_features)

    if not feature_blocks:
        raise ValueError(f"Modo de features no soportado: {feature_mode}")
    return np.concatenate(feature_blocks, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_tif", required=True, help="Patch multibanda (GeoTIFF).")
    ap.add_argument(
        "--model_path",
        default=str(_MARIDA_RF / "rf_classifier_full.joblib"),
        help="Ruta al modelo Random Forest.",
    )
    ap.add_argument("--out_mask", required=True, help="GeoTIFF de salida con mascara.")
    ap.add_argument(
        "--out_prob",
        default=None,
        help="GeoTIFF opcional de salida con probabilidad continua de Marine Debris por pixel.",
    )
    ap.add_argument("--input_channels", type=int, default=11)
    ap.add_argument("--auto_scale", action="store_true", help="Si max>1.5, divide entre 10000.")
    ap.add_argument("--window_size", type=int, default=13)
    ap.add_argument("--max_value", type=int, default=16)
    ap.add_argument(
        "--feature_mode",
        choices=["full", "no_texture", "indices_only", "bands_only"],
        default="full",
        help="Selecciona qué familias de features usa el RF.",
    )
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

    h, w = patch_raw.shape[1], patch_raw.shape[2]
    x = build_feature_matrix(
        patch_raw,
        feature_mode=args.feature_mode,
        window_size=args.window_size,
        max_value=args.max_value,
    )

    print("Feature matrix shape:", x.shape)
    print("Modo RF:", args.feature_mode)
    print("Prediciendo...")
    pred_labels = model.predict(x)
    debris_prob = None
    if args.out_prob:
        if not hasattr(model, "predict_proba"):
            raise AttributeError("El modelo RF no soporta predict_proba, necesario para --out_prob.")
        debris_class_idx = resolve_debris_class_index(model)
        pred_proba = model.predict_proba(x)
        debris_prob = pred_proba[:, debris_class_idx].astype("float32").reshape(h, w)

    pred_numeric = cat_mapping_vec(pred_labels).astype(dtype).reshape(h, w)

    meta.update(count=1, dtype=dtype)
    os.makedirs(os.path.dirname(args.out_mask) or ".", exist_ok=True)

    with rasterio.open(args.out_mask, "w", **meta) as dst:
        dst.write_band(1, pred_numeric)
        dst.update_tags(**tags)

    if args.out_prob:
        prob_meta = meta.copy()
        prob_meta.update(dtype="float32")
        os.makedirs(os.path.dirname(args.out_prob) or ".", exist_ok=True)
        with rasterio.open(args.out_prob, "w", **prob_meta) as dst:
            dst.write_band(1, debris_prob)
            dst.update_tags(**tags)
            dst.update_tags(probability_class="Marine Debris")

    print("OK ->", args.out_mask)
    if args.out_prob:
        print("OK ->", args.out_prob)


if __name__ == "__main__":
    main()
