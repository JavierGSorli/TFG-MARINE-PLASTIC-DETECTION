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


def compute_indices(img_chw: np.ndarray):
    b03 = img_chw[2]   # green
    b04 = img_chw[3]   # red
    b06 = img_chw[5]   # red edge 2
    b08 = img_chw[7]   # nir
    b11 = img_chw[9]   # swir1

    factor = ((WAVELENGTH_NIR - WAVELENGTH_RED) / (WAVELENGTH_SWIR1 - WAVELENGTH_RED)) * 10.0
    fdi = b08 - (b06 + (b11 - b06) * factor)
    ndvi = safe_div(b08 - b04, b08 + b04)
    ndwi = safe_div(b03 - b08, b03 + b08)

    return fdi.astype(np.float32), ndvi.astype(np.float32), ndwi.astype(np.float32)


def build_water_mask(
    img_chw: np.ndarray,
    ndwi: np.ndarray,
    ndwi_thr: float = 0.0,
    b11_max: float = 0.03,
    b08_max: float = 0.15,
    morph_open: int = 0,
    morph_close: int = 1,
):
    b08 = img_chw[7]
    b11 = img_chw[9]

    water = (
        np.isfinite(ndwi) &
        np.isfinite(b08) &
        np.isfinite(b11) &
        (ndwi > ndwi_thr) &
        (b11 < b11_max) &
        (b08 < b08_max)
    )

    if morph_open > 0:
        water = ndimage.binary_opening(water, iterations=morph_open)
    if morph_close > 0:
        water = ndimage.binary_closing(water, iterations=morph_close)

    return water


def threshold_mean_plus_3std(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals) + 3.0 * np.std(vals))


def build_disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return (xx * xx + yy * yy) <= (radius * radius)


def erode_without_killing_image_borders(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    Erosiona extendiendo virtualmente el raster con el valor del propio borde.
    Si un píxel de borde es agua, fuera del tile se asume agua.
    Si un píxel de borde es tierra, fuera del tile se asume tierra.
    """
    if iterations <= 0:
        return mask.copy()

    pad = iterations
    padded = np.pad(mask, pad_width=pad, mode="edge")
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(
        padded,
        structure=structure,
        iterations=iterations,
        border_value=0,
    )
    return eroded[pad:-pad, pad:-pad]


def recover_land_pixels_by_neighborhood(
    aggressive_water_mask: np.ndarray,
    reference_water_mask: np.ndarray,
    neighborhood_radius: int,
    min_water_ratio: float,
):
    """
    Revisa TODOS los píxeles clasificados como tierra tras la erosión agresiva.
    Si en su vecindad hay suficiente proporción de agua, los reincorpora a agua.
    """
    land_mask = ~aggressive_water_mask
    structure = build_disk(neighborhood_radius).astype(np.float32)
    kernel_area = float(structure.sum())

    pad = neighborhood_radius
    padded_reference = np.pad(reference_water_mask.astype(np.float32), pad_width=pad, mode="edge")
    local_water = ndimage.convolve(
        padded_reference,
        structure,
        mode="constant",
        cval=0.0,
    )
    local_water = local_water[pad:-pad, pad:-pad]
    local_ratio = local_water / max(kernel_area, 1.0)

    recovered_land = land_mask & (local_ratio >= min_water_ratio)
    final_water_mask = aggressive_water_mask | recovered_land

    return final_water_mask, recovered_land, local_ratio


def save_raster(path, array, meta, dtype, nodata=None):
    meta2 = meta.copy()
    meta2.update(count=1, dtype=dtype)
    if nodata is not None:
        meta2.update(nodata=nodata)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(array.astype(dtype), 1)


def process_one(
    tif_path: Path,
    out_dir: Path,
    ndwi_thr: float,
    b11_max: float,
    b08_max: float,
    morph_open: int,
    morph_close: int,
    aggressive_output_erosion: int,
    recovery_radius: int,
    recovery_water_ratio: float,
):
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        img = src.read().astype(np.float32)

    if img.shape[0] != 11:
        raise ValueError(f"{tif_path.name}: esperaba 11 bandas, hay {img.shape[0]}")

    fdi, ndvi, ndwi = compute_indices(img)

    water_mask = build_water_mask(
        img_chw=img,
        ndwi=ndwi,
        ndwi_thr=ndwi_thr,
        b11_max=b11_max,
        b08_max=b08_max,
        morph_open=morph_open,
        morph_close=morph_close,
    )
    fallback_no_water = int(water_mask.sum() == 0)

    if fallback_no_water:
        # If no water is detected, fall back to the same thresholding logic as no_water.
        aggressive_water_mask = water_mask.copy()
        recovered_land_mask = np.zeros_like(water_mask, dtype=bool)
        local_water_ratio = np.zeros_like(ndwi, dtype=np.float32)
        final_water_mask = np.isfinite(fdi) & np.isfinite(ndvi)
    else:
        aggressive_water_mask = water_mask.copy()
        if aggressive_output_erosion > 0:
            aggressive_water_mask = erode_without_killing_image_borders(
                aggressive_water_mask,
                iterations=aggressive_output_erosion,
            )
        if aggressive_water_mask.sum() == 0:
            aggressive_water_mask = water_mask.copy()

        final_water_mask, recovered_land_mask, local_water_ratio = recover_land_pixels_by_neighborhood(
            aggressive_water_mask=aggressive_water_mask,
            reference_water_mask=water_mask,
            neighborhood_radius=recovery_radius,
            min_water_ratio=recovery_water_ratio,
        )

    thr_fdi = threshold_mean_plus_3std(fdi, final_water_mask)
    thr_ndvi = threshold_mean_plus_3std(ndvi, final_water_mask)

    if np.isnan(thr_fdi) or np.isnan(thr_ndvi):
        raise ValueError(f"{tif_path.name}: no se pudieron calcular umbrales")

    mask_fdi = np.isfinite(fdi) & final_water_mask & (fdi > thr_fdi)
    mask_ndvi = np.isfinite(ndvi) & final_water_mask & (ndvi > thr_ndvi)
    mask_both = mask_fdi & mask_ndvi

    stem = tif_path.stem

    save_raster(out_dir / f"{stem}_water_mask.tif", water_mask, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_water_output_mask_aggressive.tif", aggressive_water_mask, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_water_recovered_land_mask.tif", recovered_land_mask, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_water_local_ratio.tif", local_water_ratio, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_water_output_mask.tif", final_water_mask, meta, "uint8", nodata=0)

    save_raster(out_dir / f"{stem}_fdi.tif", fdi, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndvi.tif", ndvi, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_ndwi.tif", ndwi, meta, "float32", nodata=np.nan)
    save_raster(out_dir / f"{stem}_fdi_mask.tif", mask_fdi, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_ndvi_mask.tif", mask_ndvi, meta, "uint8", nodata=0)
    save_raster(out_dir / f"{stem}_fdi_ndvi_mask.tif", mask_both, meta, "uint8", nodata=0)

    return {
        "patch": tif_path.name,
        "fallback_no_water": fallback_no_water,
        "water_px": int(water_mask.sum()),
        "aggressive_water_px": int(aggressive_water_mask.sum()),
        "recovered_land_px": int(recovered_land_mask.sum()),
        "output_water_px": int(final_water_mask.sum()),
        "thr_fdi": thr_fdi,
        "thr_ndvi": thr_ndvi,
        "fdi_px": int(mask_fdi.sum()),
        "ndvi_px": int(mask_ndvi.sum()),
        "both_px": int(mask_both.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ruta a un .tif o a una carpeta con patches .tif")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--ndwi_thr", type=float, default=0.0, help="Umbral NDWI mínimo para agua")
    ap.add_argument("--b11_max", type=float, default=0.03, help="SWIR1 máximo para agua")
    ap.add_argument("--b08_max", type=float, default=0.15, help="NIR máximo para agua")
    ap.add_argument("--morph_open", type=int, default=0, help="Apertura morfológica opcional")
    ap.add_argument("--morph_close", type=int, default=0, help="Cierre morfológico opcional")
    ap.add_argument(
        "--aggressive_output_erosion",
        type=int,
        default=6,
        help="Erosión agresiva de la máscara de agua para quitar costa",
    )
    ap.add_argument(
        "--recovery_radius",
        type=int,
        default=15,
        help="Radio de vecindad para revisar cada píxel de tierra de la máscara agresiva",
    )
    ap.add_argument(
        "--recovery_water_ratio",
        type=float,
        default=0.7,
        help="Si un píxel de tierra tiene esta fracción de agua alrededor, se recupera como agua",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = (
        sorted(
            tif_path
            for tif_path in in_path.glob("*.tif")
            if not tif_path.name.endswith("_mask.tif")
        )
        if in_path.is_dir()
        else [in_path]
    )
    if not tif_files:
        raise FileNotFoundError("No se encontraron TIFFs de entrada")

    rows = []
    failures = []
    for tif_path in tif_files:
        print(f"Procesando: {tif_path.name}")
        try:
            row = process_one(
                tif_path=tif_path,
                out_dir=out_dir,
                ndwi_thr=args.ndwi_thr,
                b11_max=args.b11_max,
                b08_max=args.b08_max,
                morph_open=args.morph_open,
                morph_close=args.morph_close,
                aggressive_output_erosion=args.aggressive_output_erosion,
                recovery_radius=args.recovery_radius,
                recovery_water_ratio=args.recovery_water_ratio,
            )
        except Exception as exc:
            failures.append({"patch": tif_path.name, "error": str(exc)})
            print(f"  ERROR: {exc}")
            continue

        rows.append(row)
        print(
            f"  water={row['water_px']} px  aggressive={row['aggressive_water_px']} px  "
            f"recovered_land={row['recovered_land_px']} px  output={row['output_water_px']} px  "
            f"thr_fdi={row['thr_fdi']:.6f}  thr_ndvi={row['thr_ndvi']:.6f}  "
            f"FDI={row['fdi_px']} px  NDVI={row['ndvi_px']} px  BOTH={row['both_px']} px"
        )
        print(f"  mascara_final={out_dir / f'{tif_path.stem}_water_output_mask.tif'}")

    csv_path = out_dir / "summary_indices2.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise RuntimeError("No se pudo procesar ningun patch en modo water.")

    if failures:
        failures_csv = out_dir / "summary_indices2_failures.csv"
        with open(failures_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["patch", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"\nFallos guardados en: {failures_csv}")

    print(f"\nResumen guardado en: {csv_path}")


if __name__ == "__main__":
    main()
