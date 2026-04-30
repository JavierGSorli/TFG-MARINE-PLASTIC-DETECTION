# 04_unify_predictions.py
# Unifica todas las predicciones en un CSV maestro.
# Ejecutar DESPUÉS de: 02_predict_unet.py, 03_predict_indices.py,
# predict_mask_rf.py y predict_resnet.py sobre todos los patches.
#
# Salida: results/auto/predictions_master.csv
# Una fila por patch con columnas:
#   patch, label, nc_px (ground truth), unet_px, rf_px, resnet_debris,
#   fdi_px, ndvi_px, fdi_ndvi_px, unet_score, rf_score, ...

import json
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────
BASE   = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\results\auto")
PATCHES_DIR  = BASE / "test_patches_final"
UNET_DIR     = BASE / "test_masks_unet"       # salida de 02_predict_unet.py
RF_DIR       = BASE / "test_masks_rf"         # salida de predict_mask_rf.py
RESNET_DIR   = BASE / "test_resnet_json"      # salida de predict_resnet.py
INDICES_DIR  = BASE / "test_indices"          # salida de 03_predict_indices.py
NC_MASKS_DIR = PATCHES_DIR                    # las _mask.tif están junto al patch

OUT_CSV = BASE / "predictions_master.csv"

# Clase "Marine Debris" en MARIDA = 1
DEBRIS_CLASS = 1

def read_mask_px(mask_path):
    """Cuenta píxeles de clase debris (valor=1) en una máscara uint8."""
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        m = src.read(1)
    return int((m == DEBRIS_CLASS).sum())

def read_index_mask_px(mask_path):
    """Cuenta píxeles positivos (valor=1) en máscara de índice."""
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        m = src.read(1)
    return int(m.sum())

def read_resnet_prob(json_path):
    """Extrae probabilidad de Marine Debris del JSON de ResNet."""
    if not json_path.exists():
        return None, None
    with open(json_path) as f:
        d = json.load(f)
    prob   = d.get("probabilities", {}).get("Marine Debris", None)
    active = int("Marine Debris" in d.get("active_labels", []))
    return prob, active

def read_nc_mask_px(nc_mask_path):
    """Lee la máscara generada por build_mask (ground truth NC)."""
    if not nc_mask_path.exists():
        return None
    with rasterio.open(nc_mask_path) as src:
        m = src.read(1)
    return int(m.sum())

rows = []

patches = sorted([
    p for p in PATCHES_DIR.glob("*.tif")
    if "mask" not in p.name and not p.name.startswith("_")
])

print(f"Patches encontrados: {len(patches)}\n")

for patch_path in patches:
    stem  = patch_path.stem
    parts = stem.split("_")
    label = "SI" if "SI" in stem else "NO"

    # Ground truth: máscara NC (solo patches SI)
    nc_mask_path = PATCHES_DIR / f"{stem}_mask.tif"
    nc_px = read_nc_mask_px(nc_mask_path) if label == "SI" else 0

    # UNet
    unet_mask = UNET_DIR / f"{stem}_mask.tif"
    unet_px   = read_mask_px(unet_mask)

    # RF
    rf_mask = RF_DIR / f"{stem}_mask.tif"
    rf_px   = read_mask_px(rf_mask)

    # ResNet
    resnet_json = RESNET_DIR / f"{stem}.json"
    resnet_prob, resnet_active = read_resnet_prob(resnet_json)

    # Índices
    fdi_mask      = INDICES_DIR / f"{stem}_fdi_mask.tif"
    ndvi_mask     = INDICES_DIR / f"{stem}_ndvi_mask.tif"
    both_mask     = INDICES_DIR / f"{stem}_fdi_ndvi_mask.tif"
    fdi_px        = read_index_mask_px(fdi_mask)
    ndvi_px       = read_index_mask_px(ndvi_mask)
    fdi_ndvi_px   = read_index_mask_px(both_mask)

    total_px = 256 * 256  # siempre 65536

    row = {
        "patch":          patch_path.name,
        "label":          label,
        "nc_px":          nc_px,           # GT: píxeles plástico en patch
        # Scores crudos (píxeles detectados)
        "unet_px":        unet_px,
        "rf_px":          rf_px,
        "resnet_prob":    resnet_prob,
        "resnet_active":  resnet_active,
        "fdi_px":         fdi_px,
        "ndvi_px":        ndvi_px,
        "fdi_ndvi_px":    fdi_ndvi_px,
        # Scores normalizados (% del patch)
        "unet_pct":       round(unet_px / total_px * 100, 4) if unet_px is not None else None,
        "rf_pct":         round(rf_px   / total_px * 100, 4) if rf_px   is not None else None,
        "fdi_pct":        round(fdi_px  / total_px * 100, 4) if fdi_px  is not None else None,
        "ndvi_pct":       round(ndvi_px / total_px * 100, 4) if ndvi_px is not None else None,
        "fdi_ndvi_pct":   round(fdi_ndvi_px / total_px * 100, 4) if fdi_ndvi_px is not None else None,
    }
    rows.append(row)

    avail = [
        f"unet={'✓' if unet_px is not None else '✗'}",
        f"rf={'✓' if rf_px is not None else '✗'}",
        f"resnet={'✓' if resnet_prob is not None else '✗'}",
        f"fdi={'✓' if fdi_px is not None else '✗'}",
    ]
    print(f"  [{label}] {patch_path.name[:40]}  "
          f"nc={nc_px}  {' '.join(avail)}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\n✓ CSV maestro guardado: {OUT_CSV}")
print(f"  {len(df)} patches  |  SI: {(df.label=='SI').sum()}  NO: {(df.label=='NO').sum()}")
print(f"\nColumnas disponibles:\n  {list(df.columns)}")
print(f"\nMissing values por columna:")
print(df.isnull().sum().to_string())
