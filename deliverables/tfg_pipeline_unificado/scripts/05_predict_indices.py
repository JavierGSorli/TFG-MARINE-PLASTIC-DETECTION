from __future__ import annotations

import csv

import numpy as np
import rasterio

from config import (
    FDI_WL_NIR,
    FDI_WL_RED,
    FDI_WL_SWIR,
    INDICES_OUT,
    PATCHES_DIR,
    ensure_output_dirs,
)
from pipeline_utils import iter_patch_files


def safe_div(a, b):
    out = np.full_like(a, np.nan, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def compute_fdi_ndvi(img_chw):
    b04 = img_chw[3]
    b06 = img_chw[5]
    b08 = img_chw[7]
    b11 = img_chw[9]

    factor = ((FDI_WL_NIR - FDI_WL_RED) / (FDI_WL_SWIR - FDI_WL_RED)) * 10.0
    fdi = b08 - (b06 + (b11 - b06) * factor)
    ndvi = safe_div(b08 - b04, b08 + b04)
    return fdi.astype(np.float32), ndvi.astype(np.float32)


def threshold_mean_plus_3std(arr):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals) + 3.0 * np.std(vals))


def save_raster(path, array, meta, dtype, nodata=None):
    meta2 = meta.copy()
    meta2.update(count=1, dtype=dtype)
    if nodata is not None:
        meta2.update(nodata=nodata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(array.astype(dtype), 1)


def process_one(tif_path):
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        img = src.read().astype(np.float32)

    if img.shape[0] != 11:
        raise ValueError(f"{tif_path.name}: esperaba 11 bandas, hay {img.shape[0]}")

    fdi, ndvi = compute_fdi_ndvi(img)
    thr_fdi = threshold_mean_plus_3std(fdi)
    thr_ndvi = threshold_mean_plus_3std(ndvi)
    if np.isnan(thr_fdi) or np.isnan(thr_ndvi):
        raise ValueError(f"{tif_path.name}: no se pudieron calcular umbrales")

    mask_fdi = np.isfinite(fdi) & (fdi > thr_fdi)
    mask_ndvi = np.isfinite(ndvi) & (ndvi > thr_ndvi)
    mask_both = mask_fdi & mask_ndvi

    stem = tif_path.stem
    save_raster(INDICES_OUT / f"{stem}_fdi.tif", fdi, meta, "float32", nodata=np.nan)
    save_raster(INDICES_OUT / f"{stem}_ndvi.tif", ndvi, meta, "float32", nodata=np.nan)
    save_raster(INDICES_OUT / f"{stem}_fdi_mask.tif", mask_fdi, meta, "uint8", nodata=0)
    save_raster(INDICES_OUT / f"{stem}_ndvi_mask.tif", mask_ndvi, meta, "uint8", nodata=0)
    save_raster(INDICES_OUT / f"{stem}_fdi_ndvi_mask.tif", mask_both, meta, "uint8", nodata=0)

    valid_both = np.isfinite(fdi) & np.isfinite(ndvi)
    return {
        "patch": tif_path.name,
        "thr_fdi": thr_fdi,
        "thr_ndvi": thr_ndvi,
        "fdi_px": int(mask_fdi.sum()),
        "ndvi_px": int(mask_ndvi.sum()),
        "both_px": int(mask_both.sum()),
        "valid_fdi_px": int(np.isfinite(fdi).sum()),
        "valid_ndvi_px": int(np.isfinite(ndvi).sum()),
        "fdi_pct": 100.0 * float(mask_fdi.sum()) / max(int(np.isfinite(fdi).sum()), 1),
        "ndvi_pct": 100.0 * float(mask_ndvi.sum()) / max(int(np.isfinite(ndvi).sum()), 1),
        "both_pct": 100.0 * float(mask_both.sum()) / max(int(valid_both.sum()), 1),
    }


def main():
    ensure_output_dirs()
    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    rows = []
    for patch_path in patches:
        print(f"Procesando: {patch_path.name}")
        row = process_one(patch_path)
        rows.append(row)
        print(
            f"  thr_fdi={row['thr_fdi']:.6f}  thr_ndvi={row['thr_ndvi']:.6f}  "
            f"FDI={row['fdi_px']} px  NDVI={row['ndvi_px']} px  BOTH={row['both_px']} px"
        )

    csv_path = INDICES_OUT / "summary_indices.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResumen guardado en: {csv_path}")


if __name__ == "__main__":
    main()
