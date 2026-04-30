# 02_predict_unet.py
import sys
import subprocess
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path

PATCHES_DIR = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                   r"\results\auto\test_patches_final")
MASKS_DIR   = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                   r"\results\auto\test_masks_final")
UNET_DIR    = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                   r"\data\marida\marine-debris.github.io"
                   r"\semantic_segmentation\unet")
MASKS_DIR.mkdir(exist_ok=True)

MARIDA_CLASSES = {
    1: "Marine Debris",         2: "Dense Sargassum",
    3: "Sparse Sargassum",      4: "Natural Organic Material",
    5: "Ship",                  6: "Clouds",
    7: "Marine Water",          8: "Sediment-Laden Water",
    9: "Foam",                 10: "Turbid Water",
    11: "Shallow Water",
}

results = []

for tif_path in sorted(PATCHES_DIR.glob("*.tif")):
    mask_path = MASKS_DIR / tif_path.name.replace(".tif", "_mask.tif")
    label     = "SI" if "SI" in tif_path.name else "NO"

    print(f"── {tif_path.name}  [{label}]")

    # Llamar a predict_mask.py igual que lo hacías tú a mano
    cmd = [
        sys.executable,
        str(UNET_DIR / "predict_mask.py"),
        "--patch_tif",  str(tif_path),
        "--out_mask",   str(mask_path),
        # sin --auto_scale — ya está en 0-1
    ]

    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(UNET_DIR)   # necesario para que importe unet y dataloader
    )

    if proc.returncode != 0:
        print(f"   ✗ Error:\n{proc.stderr[-300:]}")
        results.append({"patch": tif_path.name, "label": label,
                        "status": "error", "plastic_px": None,
                        "plastic_pct": None})
        continue

    # Leer la máscara generada y calcular estadísticas
    with rasterio.open(mask_path) as src:
        pred = src.read(1).astype(int)

    total_px   = pred.size
    unique, counts = np.unique(pred, return_counts=True)

    print(f"   Predicción ({pred.shape[0]}×{pred.shape[1]}):")
    for cls, cnt in zip(unique, counts):
        pct    = 100 * cnt / total_px
        marker = " ← PLÁSTICO" if cls == 1 else ""
        print(f"     Clase {cls:2d} ({MARIDA_CLASSES.get(cls,'?'):30s}): "
              f"{cnt:5d} px  {pct:5.1f}%{marker}")

    plastic_px  = int((pred == 1).sum())
    plastic_pct = round(100 * plastic_px / total_px, 2)
    print(f"   → Plástico: {plastic_px} px ({plastic_pct}%)\n")

    results.append({
        "patch":       tif_path.name,
        "label":       label,
        "status":      "ok",
        "plastic_px":  plastic_px,
        "plastic_pct": plastic_pct,
        "total_px":    total_px,
    })

# ── Resumen ───────────────────────────────────────────────────
df = pd.DataFrame(results)
ok = df[df.status == "ok"]

print("=" * 55)
print("RESUMEN PREDICCIONES UNet-MARIDA")
print("=" * 55)
print(ok[["patch","label","plastic_px","plastic_pct"]]
      .to_string(index=False))

print(f"\nSI — plástico medio: "
      f"{ok[ok.label=='SI'].plastic_pct.mean():.2f}%")
print(f"NO — plástico medio: "
      f"{ok[ok.label=='NO'].plastic_pct.mean():.2f}%")