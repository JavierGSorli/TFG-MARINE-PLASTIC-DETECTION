#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


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


def save_raster(path, array, meta, dtype, nodata=None):
    meta2 = meta.copy()
    meta2.update(count=1, dtype=dtype)
    if nodata is not None:
        meta2.update(nodata=nodata)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(array.astype(dtype), 1)


def compute_raw_indices(img_chw: np.ndarray):
    """
    Índices sobre reflectancia original 0-1
    """
    b03 = img_chw[2]   # green
    b04 = img_chw[3]   # red
    b06 = img_chw[5]   # red-edge 2
    b08 = img_chw[7]   # nir
    b11 = img_chw[9]   # swir1

    ndwi = safe_div(b03 - b08, b03 + b08)
    ndvi = safe_div(b08 - b04, b08 + b04)

    factor = ((WAVELENGTH_NIR - WAVELENGTH_RED) / (WAVELENGTH_SWIR1 - WAVELENGTH_RED)) * 10.0
    fdi = b08 - (b06 + (b11 - b06) * factor)

    return fdi.astype(np.float32), ndvi.astype(np.float32), ndwi.astype(np.float32)


def build_water_mask(
    img_chw: np.ndarray,
    ndwi_raw: np.ndarray,
    ndwi_thr: float = 0.0,
    b11_max: float = 0.03,
    b08_max: float = 0.15,
    morph_open: int = 0,
    morph_close: int = 0,
    dilate_px: int = 0,
):
    """
    Máscara de agua sin usar NDVI, para no contaminar luego la evaluación de NDVI.
    Reglas simples:
    - NDWI alto
    - SWIR bajo
    - NIR no excesivo
    """
    b08 = img_chw[7]
    b11 = img_chw[9]

    water = (
        np.isfinite(ndwi_raw) &
        np.isfinite(b11) &
        np.isfinite(b08) &
        (ndwi_raw > ndwi_thr) &
        (b11 < b11_max) &
        (b08 < b08_max)
    )

    if morph_open > 0:
        water = ndimage.binary_opening(water, iterations=morph_open)

    if morph_close > 0:
        water = ndimage.binary_closing(water, iterations=morph_close)

    if dilate_px > 0:
        water = ndimage.binary_dilation(water, iterations=dilate_px)

    return water.astype(np.uint8)


def threshold_mean_plus_3std(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals) + 3.0 * np.std(vals))


def process_one(
    tif_path: Path,
    out_dir: Path,
    ndwi_thr: float,
    b11_max: float,
    b08_max: float,
    morph_open: int,
    morph_close: int,
    dilate_water: int,
    coast_buffer: int,
    output_coast_buffer: int,
):
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        img = src.read().astype(np.float32)

    if img.shape[0] != 11:
        raise ValueError(f"{tif_path.name}: esperaba 11 bandas, hay {img.shape[0]}")

    # 1) Índices RAW
    fdi_raw, ndvi_raw, ndwi_raw = compute_raw_indices(img)

    # 2) Máscara de agua SOLO con NDWI + SWIR + NIR
    water_mask = build_water_mask(
        img_chw=img,
        ndwi_raw=ndwi_raw,
        ndwi_thr=ndwi_thr,
        b11_max=b11_max,
        b08_max=b08_max,
        morph_open=morph_open,
        morph_close=morph_close,
        dilate_px=dilate_water,
    )

    wm_pred = water_mask > 0
    if wm_pred.sum() == 0:
        raise ValueError(f"{tif_path.name}: la máscara de agua quedó vacía")

    # Máscara de calibración: erosiona la costa para calcular umbrales sobre agua interior.
    wm_cal = wm_pred.copy()
    if coast_buffer > 0:
        wm_cal = ndimage.binary_erosion(wm_cal, iterations=coast_buffer)

    if wm_cal.sum() == 0:
        wm_cal = wm_pred.copy()

    # Máscara final de salida: opcionalmente también se erosiona para suprimir costa.
    wm_out = wm_pred.copy()
    if output_coast_buffer > 0:
        wm_out = ndimage.binary_erosion(wm_out, iterations=output_coast_buffer)

    if wm_out.sum() == 0:
        wm_out = wm_pred.copy()

    # 3) Versiones continuas ENMASCARADAS para visualizar bien
    fdi_water = np.where(wm_out, fdi_raw, np.nan).astype(np.float32)
    ndvi_water = np.where(wm_out, ndvi_raw, np.nan).astype(np.float32)
    ndwi_water = np.where(wm_out, ndwi_raw, np.nan).astype(np.float32)

    fdi_calibration = np.where(wm_cal, fdi_raw, np.nan).astype(np.float32)
    ndvi_calibration = np.where(wm_cal, ndvi_raw, np.nan).astype(np.float32)

    # 4) Thresholds SOLO sobre agua interior
    thr_fdi = threshold_mean_plus_3std(fdi_raw, wm_cal)
    thr_ndvi = threshold_mean_plus_3std(ndvi_raw, wm_cal)

    if np.isnan(thr_fdi) or np.isnan(thr_ndvi):
        raise ValueError(f"{tif_path.name}: no se pudieron calcular umbrales")

    # 5) Binarias finales sobre la máscara de predicción, más permisiva.
    mask_fdi = (fdi_raw > thr_fdi) & wm_out
    mask_ndvi = (ndvi_raw > thr_ndvi) & wm_out
    mask_both = mask_fdi & mask_ndvi

    stem = tif_path.stem

    # Guardar todo
    save_raster(out_dir / f"{stem}_water_mask.tif", water_mask, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_water_calibration_mask.tif", wm_cal, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_water_output_mask.tif", wm_out, meta, "uint8", nodata=0)

    save_raster(out_dir / f"{stem}_fdi_raw.tif", fdi_raw, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndvi_raw.tif", ndvi_raw, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndwi_raw.tif", ndwi_raw, meta, "float32", nodata=np.nan)

    save_raster(out_dir / f"{stem}_fdi_water.tif", fdi_water, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndvi_water.tif", ndvi_water, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndwi_water.tif", ndwi_water, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_fdi_calibration.tif", fdi_calibration, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndvi_calibration.tif", ndvi_calibration, meta, "float32", nodata=np.nan)

    save_raster(out_dir / f"{stem}_fdi_mask.tif", mask_fdi, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_ndvi_mask.tif", mask_ndvi, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_both_mask.tif", mask_both, meta, "uint8", nodata=0)

    return {
        "patch": tif_path.name,
        "water_px": int(wm_pred.sum()),
        "water_cal_px": int(wm_cal.sum()),
        "water_out_px": int(wm_out.sum()),
        "thr_fdi": thr_fdi,
        "thr_ndvi": thr_ndvi,
        "fdi_px": int(mask_fdi.sum()),
        "ndvi_px": int(mask_ndvi.sum()),
        "both_px": int(mask_both.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ruta a un .tif o a una carpeta")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--ndwi_thr", type=float, default=0.0, help="Umbral NDWI para agua")
    ap.add_argument("--b11_max", type=float, default=0.03, help="Umbral máximo B11 para agua")
    ap.add_argument("--b08_max", type=float, default=0.15, help="Umbral máximo B08 para agua")
    ap.add_argument("--morph_open", type=int, default=0, help="Apertura morfológica opcional")
    ap.add_argument("--morph_close", type=int, default=1, help="Cierre morfológico opcional")
    ap.add_argument("--dilate_water", type=int, default=1, help="Dilata la máscara de agua para no comerse la litter window")
    ap.add_argument("--coast_buffer", type=int, default=3, help="Erosiona la máscara para calibrar umbrales lejos de costa")
    ap.add_argument("--output_coast_buffer", type=int, default=0, help="Erosiona también la máscara final de salida para suprimir costa")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(in_path.glob("*.tif")) if in_path.is_dir() else [in_path]
    if not tif_files:
        raise FileNotFoundError("No se encontraron TIFFs")

    rows = []
    for tif_path in tif_files:
        print(f"Procesando: {tif_path.name}")
        row = process_one(
            tif_path=tif_path,
            out_dir=out_dir,
            ndwi_thr=args.ndwi_thr,
            b11_max=args.b11_max,
            b08_max=args.b08_max,
            morph_open=args.morph_open,
            morph_close=args.morph_close,
            dilate_water=args.dilate_water,
            coast_buffer=args.coast_buffer,
            output_coast_buffer=args.output_coast_buffer,
        )
        rows.append(row)
        print(
            f"  water={row['water_px']} px  water_cal={row['water_cal_px']} px  water_out={row['water_out_px']} px  "
            f"thr_fdi={row['thr_fdi']:.6f}  thr_ndvi={row['thr_ndvi']:.6f}  "
            f"FDI={row['fdi_px']} px  NDVI={row['ndvi_px']} px  BOTH={row['both_px']} px"
        )

    csv_path = out_dir / "summary_indices_watermask.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResumen guardado en: {csv_path}")


if __name__ == "__main__":
    main()
