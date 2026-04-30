from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import rasterio
from joblib import load
from skimage.color import rgb2gray

try:
    from skimage.feature import graycomatrix as greycomatrix
except ImportError:
    from skimage.feature import greycomatrix

from config import DEBRIS_CLASS, RF_DIR, RF_MODEL_PATH, RF_OUT, PATCHES_DIR, UTILS_DIR, ensure_output_dirs
from pipeline_utils import ensure_file, infer_label_from_name, iter_patch_files

if str(RF_DIR) not in sys.path:
    sys.path.insert(0, str(RF_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from assets import cat_mapping_vec
from engineering_patches import bsi, fai, fdi, glcm_feature, ndmi, ndvi, ndwi, nrd, si
from random_forest import bands_mean

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
    return np.stack([ndvi_, fai_, fdi_, si_, ndwi_, nrd_, ndmi_, bsi_], axis=-1).astype("float32")


def compute_texture_engineering(patch_chw, window_size=13, max_value=16):
    img = patch_chw[[1, 2, 3]].astype("float32").copy()
    img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])

    rgb_composite = img[:, :, [2, 1, 0]]
    rgb_composite = np.clip(rgb_composite, 0.0, 0.15) / 0.15
    gray = rgb2gray(rgb_composite)

    bins = np.linspace(0.0, 1.0, max_value)
    num_levels = max_value + 1
    pad = (window_size - 1) // 2
    temp_gray = np.pad(gray, pad, mode="reflect")

    height, width = gray.shape
    features = np.zeros((height, width, 6), dtype="float32")
    for row in range(pad, height + pad):
        for col in range(pad, width + pad):
            window = temp_gray[row - pad : row + pad + 1, col - pad : col + pad + 1]
            inds = np.digitize(window, bins)
            cooc = greycomatrix(
                inds,
                [1],
                [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=num_levels,
                normed=True,
                symmetric=True,
            )
            cooc = cooc.mean(3)[:, :, :, np.newaxis]
            con, dis, homo, ener, cor, asm = glcm_feature(cooc)
            features[row - pad, col - pad, 0] = con
            features[row - pad, col - pad, 1] = dis
            features[row - pad, col - pad, 2] = homo
            features[row - pad, col - pad, 3] = ener
            features[row - pad, col - pad, 4] = cor
            features[row - pad, col - pad, 5] = asm

    return features


def predict_mask(model, patch_raw, window_size=13, max_value=16):
    patch_hwc = np.moveaxis(patch_raw, (0, 1, 2), (2, 0, 1))
    height, width, _ = patch_hwc.shape

    impute_nan = np.tile(bands_mean, (height, width, 1))
    spectral_hwc = patch_hwc.copy()
    nan_mask = np.isnan(spectral_hwc)
    spectral_hwc[nan_mask] = impute_nan[nan_mask]
    spectral_features = spectral_hwc.reshape(height * width, -1)

    indices_hwc = compute_indices_engineering(patch_raw)
    indices_features = np.nan_to_num(indices_hwc).reshape(height * width, -1)

    glcm_hwc = compute_texture_engineering(
        patch_raw,
        window_size=window_size,
        max_value=max_value,
    )
    glcm_features = np.nan_to_num(glcm_hwc).reshape(height * width, -1)

    features = np.concatenate([spectral_features, indices_features, glcm_features], axis=1)
    pred_labels = model.predict(features)
    return cat_mapping_vec(pred_labels).astype("uint8").reshape(height, width)


def save_mask(mask_path, patch_path, pred_mask):
    with rasterio.open(patch_path) as src:
        meta = src.meta.copy()
    meta.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(mask_path, "w", **meta) as dst:
        dst.write(pred_mask, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--window_size", type=int, default=13)
    parser.add_argument("--max_value", type=int, default=16)
    args = parser.parse_args()

    ensure_output_dirs()
    ensure_file(RF_MODEL_PATH, "Modelo Random Forest")

    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    model = load(RF_MODEL_PATH)
    rows = []

    for patch_path in patches:
        out_mask = RF_OUT / f"{patch_path.stem}_mask.tif"
        label = infer_label_from_name(patch_path.name)

        if out_mask.exists() and not args.overwrite:
            with rasterio.open(out_mask) as src:
                pred_mask = src.read(1)
            plastic_px = int((pred_mask == DEBRIS_CLASS).sum())
            total_px = int(pred_mask.size)
            print(f"[SKIP] {patch_path.name} -> {out_mask.name}")
        else:
            with rasterio.open(patch_path) as src:
                patch_raw = src.read().astype("float32")
            pred_mask = predict_mask(
                model,
                patch_raw,
                window_size=args.window_size,
                max_value=args.max_value,
            )
            save_mask(out_mask, patch_path, pred_mask)
            plastic_px = int((pred_mask == DEBRIS_CLASS).sum())
            total_px = int(pred_mask.size)
            print(f"[OK] {patch_path.name} -> debris={plastic_px} px")

        rows.append(
            {
                "patch": patch_path.name,
                "label": label,
                "status": "ok",
                "plastic_px": plastic_px,
                "plastic_pct": round(100 * plastic_px / total_px, 4),
            }
        )

    summary_path = RF_OUT / "summary_rf.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
