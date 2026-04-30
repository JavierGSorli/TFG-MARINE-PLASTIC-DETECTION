from __future__ import annotations

import json

import pandas as pd
import rasterio

from config import CSV_MASTER, DEBRIS_CLASS, INDICES_OUT, PATCHES_DIR, RESNET_OUT, RF_OUT, UNET_OUT
from pipeline_utils import infer_label_from_name, iter_patch_files


def read_debris_mask_px(mask_path):
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int((mask == DEBRIS_CLASS).sum())


def read_binary_mask_px(mask_path):
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int((mask > 0).sum())


def read_resnet_prob(json_path):
    if not json_path.exists():
        return None, None
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    prob = data.get("probabilities", {}).get("Marine Debris")
    active = int("Marine Debris" in data.get("active_labels", []))
    return prob, active


def main():
    rows = []
    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    print(f"Patches encontrados: {len(patches)}\n")
    for patch_path in patches:
        label = infer_label_from_name(patch_path.name)
        stem = patch_path.stem

        with rasterio.open(patch_path) as src:
            total_px = int(src.width * src.height)

        nc_mask_path = PATCHES_DIR / f"{stem}_mask.tif"
        nc_px = read_binary_mask_px(nc_mask_path) if label == "SI" else 0

        unet_px = read_debris_mask_px(UNET_OUT / f"{stem}_mask.tif")
        rf_px = read_debris_mask_px(RF_OUT / f"{stem}_mask.tif")
        resnet_prob, resnet_active = read_resnet_prob(RESNET_OUT / f"{stem}.json")
        fdi_px = read_binary_mask_px(INDICES_OUT / f"{stem}_fdi_mask.tif")
        ndvi_px = read_binary_mask_px(INDICES_OUT / f"{stem}_ndvi_mask.tif")
        fdi_ndvi_px = read_binary_mask_px(INDICES_OUT / f"{stem}_fdi_ndvi_mask.tif")

        rows.append(
            {
                "patch": patch_path.name,
                "label": label,
                "nc_px": nc_px,
                "unet_px": unet_px,
                "rf_px": rf_px,
                "resnet_prob": resnet_prob,
                "resnet_active": resnet_active,
                "fdi_px": fdi_px,
                "ndvi_px": ndvi_px,
                "fdi_ndvi_px": fdi_ndvi_px,
                "unet_pct": round(unet_px / total_px * 100, 4) if unet_px is not None else None,
                "rf_pct": round(rf_px / total_px * 100, 4) if rf_px is not None else None,
                "fdi_pct": round(fdi_px / total_px * 100, 4) if fdi_px is not None else None,
                "ndvi_pct": round(ndvi_px / total_px * 100, 4) if ndvi_px is not None else None,
                "fdi_ndvi_pct": round(fdi_ndvi_px / total_px * 100, 4)
                if fdi_ndvi_px is not None
                else None,
            }
        )

        print(
            f"[{label}] {patch_path.name}  "
            f"gt={nc_px} unet={unet_px} rf={rf_px} resnet={resnet_prob}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(CSV_MASTER, index=False)

    print(f"\nCSV maestro guardado: {CSV_MASTER}")
    print(f"{len(df)} patches  |  SI={(df.label == 'SI').sum()}  NO={(df.label == 'NO').sum()}")
    print("\nMissing values por columna:")
    print(df.isnull().sum().to_string())


if __name__ == "__main__":
    main()
