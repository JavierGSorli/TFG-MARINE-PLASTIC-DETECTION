#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import rasterio


# Orden MARIDA:
# 0 B01, 1 B02, 2 B03, 3 B04, 4 B05, 5 B06, 6 B07, 7 B08, 8 B8A, 9 B11, 10 B12

WAVELENGTH_NIR = 842.0
WAVELENGTH_RED = 665.0
WAVELENGTH_SWIR1 = 1610.0


def safe_div(a, b):
    out = np.full_like(a, np.nan, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def compute_fdi_ndvi(img_chw: np.ndarray):
    """
    img_chw: (11, H, W) float32 reflectancias 0-1
    """
    b04 = img_chw[3]   # red
    b06 = img_chw[5]   # red edge 2
    b08 = img_chw[7]   # nir
    b11 = img_chw[9]   # swir1

    # Igual que el notebook DE Africa:
    # FDI = nir - (red_edge_2 + (swir1 - red_edge_2) * ((nir-red)/(swir1-red)) * 10)
    factor = ((WAVELENGTH_NIR - WAVELENGTH_RED) / (WAVELENGTH_SWIR1 - WAVELENGTH_RED)) * 10.0
    fdi = b08 - (b06 + (b11 - b06) * factor)

    # NDVI estándar
    ndvi = safe_div(b08 - b04, b08 + b04)

    return fdi.astype(np.float32), ndvi.astype(np.float32)


def threshold_mean_plus_3std(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    return mu + 3.0 * sigma


def save_raster(path, array, meta, dtype, nodata=None):
    meta2 = meta.copy()
    meta2.update(count=1, dtype=dtype)
    if nodata is not None:
        meta2.update(nodata=nodata)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(array.astype(dtype), 1)


def process_one(tif_path: Path, out_dir: Path):
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

    # Igual que el notebook: combinación AND
    mask_both = mask_fdi & mask_ndvi

    stem = tif_path.stem

    save_raster(out_dir / f"{stem}_fdi.tif", fdi, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndvi.tif", ndvi, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_fdi_mask.tif", mask_fdi, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_ndvi_mask.tif", mask_ndvi, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_fdi_ndvi_mask.tif", mask_both, meta, "uint8", nodata=0)

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
        "both_pct": 100.0 * float(mask_both.sum()) / max(int((np.isfinite(fdi) & np.isfinite(ndvi)).sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Ruta a un .tif o a una carpeta con patches .tif"
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Carpeta de salida"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        tif_files = sorted(in_path.glob("*.tif"))
    else:
        tif_files = [in_path]

    if not tif_files:
        raise FileNotFoundError("No se encontraron TIFFs de entrada")

    rows = []
    for tif_path in tif_files:
        print(f"Procesando: {tif_path.name}")
        row = process_one(tif_path, out_dir)
        rows.append(row)
        print(
            f"  thr_fdi={row['thr_fdi']:.6f}  thr_ndvi={row['thr_ndvi']:.6f}  "
            f"FDI={row['fdi_px']} px  NDVI={row['ndvi_px']} px  BOTH={row['both_px']} px"
        )

    csv_path = out_dir / "summary_indices.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResumen guardado en: {csv_path}")


if __name__ == "__main__":
    main()
