# 08_error_analysis.py
# Análisis cualitativo de errores: FP y FN por método.
# Genera un panel visual por cada error notable.

import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE        = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\results\auto")
PATCHES_DIR = BASE / "test_patches_final"
UNET_DIR    = BASE / "test_masks_unet"
RF_DIR      = BASE / "test_masks_rf"
INDICES_DIR = BASE / "test_indices"
MASTER_CSV  = BASE / "predictions_master.csv"
EVAL_CSV    = BASE / "evaluation" / "tabla_comparativa.csv"
OUT_DIR     = BASE / "error_analysis"
OUT_DIR.mkdir(exist_ok=True)

df     = pd.read_csv(MASTER_CSV)
y_true = (df["label"] == "SI").astype(int).values

# ── Usar umbral óptimo de F1 guardado en tabla comparativa ────
tabla  = pd.read_csv(EVAL_CSV)
umbral = {}
for _, row in tabla.iterrows():
    umbral[row["Método"]] = float(row["Umbral"])

def make_rgb(data, vmax=0.1):
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    return np.clip(rgb / vmax, 0, 1)

def overlay_mask(rgb, mask, color=(1, 0, 0), alpha=0.6):
    out = rgb.copy()
    m   = mask.astype(bool)
    for c, v in enumerate(color):
        out[:,:,c] = np.where(m, alpha*v + (1-alpha)*rgb[:,:,c], rgb[:,:,c])
    return out

def read_mask(path):
    if not path or not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)

def read_index_mask(stem, index):
    p = INDICES_DIR / f"{stem}_{index}_mask.tif"
    return read_mask(p)

# ── Identificar errores por método ───────────────────────────
# Método UNet como ejemplo principal (repetir para los demás si quieres)
score_cols = {
    "UNet (MARIDA)":   ("unet_pct",    UNET_DIR,    "_mask.tif"),
    "FDI (umbral)":    ("fdi_pct",     INDICES_DIR, "_fdi_mask.tif"),
    "FDI+NDVI":        ("fdi_ndvi_pct",INDICES_DIR, "_fdi_ndvi_mask.tif"),
}

for method_name, (score_col, mask_dir, mask_suffix) in score_cols.items():
    if score_col not in df.columns:
        continue
    t = umbral.get(method_name, 0.5)
    scores = df[score_col].fillna(0).values
    preds  = (scores >= t).astype(int)

    fp_idx = np.where((preds==1) & (y_true==0))[0]
    fn_idx = np.where((preds==0) & (y_true==1))[0]

    cases = [(i, "FP") for i in fp_idx] + [(i, "FN") for i in fn_idx]
    if not cases:
        print(f"[{method_name}] Sin errores con este umbral.")
        continue

    print(f"\n[{method_name}] FP={len(fp_idx)}  FN={len(fn_idx)}")

    for idx, error_type in cases:
        row      = df.iloc[idx]
        patch_p  = PATCHES_DIR / row["patch"]
        stem     = patch_p.stem
        mask_p   = mask_dir / f"{stem}{mask_suffix}"
        nc_mask_p = PATCHES_DIR / f"{stem}_mask.tif"

        if not patch_p.exists():
            continue

        with rasterio.open(patch_p) as src:
            data = src.read().astype("float32")

        rgb      = make_rgb(data)
        pred_m   = read_mask(mask_p)
        nc_m     = read_mask(nc_mask_p)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].imshow(rgb)
        axes[0].set_title("RGB (B04/B03/B02)")
        axes[0].axis("off")

        if pred_m is not None:
            ov = overlay_mask(rgb, pred_m, color=(1,0,0))
            axes[1].imshow(ov)
        else:
            axes[1].imshow(rgb)
        axes[1].set_title(f"Predicción {method_name}\n(rojo = detectado)")
        axes[1].axis("off")

        if nc_m is not None:
            ov2 = overlay_mask(rgb, nc_m, color=(0,1,0))
            axes[2].imshow(ov2)
            axes[2].set_title("GT Nature (verde = plástico real)")
        else:
            axes[2].imshow(rgb)
            axes[2].set_title("GT Nature (no disponible)")
        axes[2].axis("off")

        label_str = row["label"]
        score_val = round(float(scores[idx]), 4)
        nc_px     = row.get("nc_px", "?")
        fig.suptitle(
            f"{error_type} | {method_name} | {row['patch']}\n"
            f"Label={label_str}  Score={score_val}  "
            f"Umbral={t:.4f}  NC_px={nc_px}",
            fontsize=9
        )
        plt.tight_layout()

        safe_name = method_name.replace(" ", "_").replace("(","").replace(")","")
        out_png = OUT_DIR / f"{error_type}_{safe_name}_{stem}.png"
        plt.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  → {out_png.name}")

print(f"\n✓ Análisis de errores guardado en: {OUT_DIR}")
