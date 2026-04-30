from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from config import BAND_NAMES, CSV_MASTER, CSV_XGB, INDICES_OUT, PATCHES_DIR
from pipeline_utils import infer_label_from_name, iter_patch_files


def spectral_features(data):
    feats = {}
    for index, name in enumerate(BAND_NAMES):
        band = data[index].ravel()
        band = band[band > 0]
        feats[f"{name}_mean"] = float(np.mean(band)) if len(band) > 0 else 0.0
        feats[f"{name}_std"] = float(np.std(band)) if len(band) > 0 else 0.0
        feats[f"{name}_p95"] = float(np.percentile(band, 95)) if len(band) > 0 else 0.0
    return feats


def index_features(stem):
    feats = {}
    for idx_name in ["fdi", "ndvi"]:
        path = INDICES_OUT / f"{stem}_{idx_name}.tif"
        if not path.exists():
            feats[f"{idx_name}_mean"] = None
            feats[f"{idx_name}_max"] = None
            feats[f"{idx_name}_p99"] = None
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
        vals = arr[np.isfinite(arr)]
        feats[f"{idx_name}_mean"] = float(np.mean(vals)) if len(vals) > 0 else None
        feats[f"{idx_name}_max"] = float(np.max(vals)) if len(vals) > 0 else None
        feats[f"{idx_name}_p99"] = float(np.percentile(vals, 99)) if len(vals) > 0 else None
    return feats


def main():
    master = pd.read_csv(CSV_MASTER).set_index("patch")
    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    rows = []
    print(f"Construyendo dataset XGBoost con {len(patches)} patches...\n")
    for patch_path in patches:
        stem = patch_path.stem
        label = 1 if infer_label_from_name(patch_path.name) == "SI" else 0

        with rasterio.open(patch_path) as src:
            data = src.read().astype("float32")

        row = {"patch": patch_path.name, "label": label}
        row.update(spectral_features(data))
        row.update(index_features(stem))

        if patch_path.name in master.index:
            info = master.loc[patch_path.name]
            row["unet_pct"] = info.get("unet_pct")
            row["rf_pct"] = info.get("rf_pct")
            row["resnet_prob"] = info.get("resnet_prob")
            row["fdi_pct"] = info.get("fdi_pct")
            row["ndvi_pct"] = info.get("ndvi_pct")
            row["fdi_ndvi_pct"] = info.get("fdi_ndvi_pct")
        else:
            row["unet_pct"] = None
            row["rf_pct"] = None
            row["resnet_prob"] = None
            row["fdi_pct"] = None
            row["ndvi_pct"] = None
            row["fdi_ndvi_pct"] = None

        rows.append(row)
        print(f"  {'SI' if label else 'NO'} {patch_path.name}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_XGB, index=False)

    print(f"\nDataset XGBoost guardado: {CSV_XGB}")
    print(f"{len(df)} muestras  |  SI={df.label.sum()}  NO={(df.label == 0).sum()}")
    print(f"Features: {len(df.columns) - 2}")
    print(f"Missing values totales: {int(df.isnull().sum().sum())}")


if __name__ == "__main__":
    main()
