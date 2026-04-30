# 05_build_xgboost_dataset.py
# Construye dataset tabular por patch para XGBoost.
# Features: estadísticas espectrales + índices + scores de modelos.
# Salida: results/auto/xgboost_dataset.csv

import numpy as np
import rasterio
import pandas as pd
from pathlib import Path

BASE        = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\results\auto")
PATCHES_DIR = BASE / "test_patches_final"
INDICES_DIR = BASE / "test_indices"
MASTER_CSV  = BASE / "predictions_master.csv"
OUT_CSV     = BASE / "xgboost_dataset.csv"

# Orden MARIDA
BAND_NAMES = ["B01","B02","B03","B04","B05","B06",
              "B07","B08","B8A","B11","B12"]

def spectral_features(data):
    """
    Features espectrales por patch: media y std de cada banda.
    data: (11, 256, 256) float32
    """
    feats = {}
    for i, name in enumerate(BAND_NAMES):
        b = data[i].ravel()
        b = b[b > 0]  # ignorar píxeles vacíos
        feats[f"{name}_mean"] = float(np.mean(b)) if len(b) > 0 else 0.0
        feats[f"{name}_std"]  = float(np.std(b))  if len(b) > 0 else 0.0
        feats[f"{name}_p95"]  = float(np.percentile(b, 95)) if len(b) > 0 else 0.0
    return feats

def index_features(stem):
    """Lee los rasters de índices y devuelve stats."""
    feats = {}
    for idx_name in ["fdi", "ndvi"]:
        p = INDICES_DIR / f"{stem}_{idx_name}.tif"
        if not p.exists():
            feats[f"{idx_name}_mean"] = None
            feats[f"{idx_name}_max"]  = None
            feats[f"{idx_name}_p99"]  = None
            continue
        with rasterio.open(p) as src:
            arr = src.read(1).astype("float32")
        vals = arr[np.isfinite(arr)]
        feats[f"{idx_name}_mean"] = float(np.mean(vals)) if len(vals) > 0 else None
        feats[f"{idx_name}_max"]  = float(np.max(vals))  if len(vals) > 0 else None
        feats[f"{idx_name}_p99"]  = float(np.percentile(vals, 99)) if len(vals) > 0 else None
    return feats

# Cargar CSV maestro para añadir scores de modelos como features
master = pd.read_csv(MASTER_CSV)
master_idx = master.set_index("patch")

rows = []

patches = sorted([
    p for p in PATCHES_DIR.glob("*.tif")
    if "mask" not in p.name and not p.name.startswith("_")
])

print(f"Construyendo dataset XGBoost con {len(patches)} patches...\n")

for patch_path in patches:
    stem  = patch_path.stem
    label = 1 if "SI" in stem else 0

    with rasterio.open(patch_path) as src:
        data = src.read().astype("float32")

    row = {"patch": patch_path.name, "label": label}

    # Features espectrales
    row.update(spectral_features(data))

    # Features de índices
    row.update(index_features(stem))

    # Scores de modelos (del CSV maestro) como features adicionales
    if patch_path.name in master_idx.index:
        m = master_idx.loc[patch_path.name]
        row["unet_pct"]      = m.get("unet_pct", None)
        row["rf_pct"]        = m.get("rf_pct", None)
        row["resnet_prob"]   = m.get("resnet_prob", None)
        row["fdi_pct"]       = m.get("fdi_pct", None)
        row["fdi_ndvi_pct"]  = m.get("fdi_ndvi_pct", None)
    else:
        row["unet_pct"] = row["rf_pct"] = row["resnet_prob"] = None
        row["fdi_pct"]  = row["fdi_ndvi_pct"] = None

    rows.append(row)
    print(f"  {'SI' if label else 'NO'} {patch_path.name[:45]}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"\n✓ Dataset XGBoost guardado: {OUT_CSV}")
print(f"  {len(df)} muestras  |  SI: {df.label.sum()}  NO: {(df.label==0).sum()}")
print(f"  Features: {len(df.columns)-2}")
print(f"  Missing values: {df.isnull().sum().sum()} en total")
