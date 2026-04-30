# check_patches_quality.py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

PATCHES_DIR = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                   r"\results\auto\test_patches_final")

# Rangos esperados para agua marina mediterránea en L2A (float32 0-1)
# Más permisivos que antes — agua limpia tiene NIR/SWIR muy bajo
EXPECTED = {
    # banda: (min_mean, max_mean)
    0:  (0.002, 0.08),   # B01 azul costero
    1:  (0.002, 0.08),   # B02 azul
    2:  (0.002, 0.08),   # B03 verde
    3:  (0.001, 0.06),   # B04 rojo
    4:  (0.000, 0.05),   # B05 red-edge
    5:  (0.000, 0.05),   # B06 red-edge
    6:  (0.000, 0.05),   # B07 red-edge
    7:  (0.000, 0.10),   # B08 NIR — puede ser alto si hay flotantes
    8:  (0.000, 0.05),   # B8A NIR estrecho
    9:  (0.000, 0.05),   # B11 SWIR
    10: (0.000, 0.03),   # B12 SWIR
}

MARIDA_NAMES = ["B01","B02","B03","B04","B05","B06",
                "B07","B08","B8A","B11","B12"]

patches = sorted(PATCHES_DIR.glob("*.tif"))
print(f"Patches encontrados: {len(patches)}\n")

resumen = []

for tif_path in patches:
    label = "SI" if "SI" in tif_path.name else "NO"
    print(f"── {tif_path.name}  [{label}]")

    with rasterio.open(tif_path) as src:
        data  = src.read().astype("float32")
        dtype = src.dtypes[0]
        shape = data.shape

    # Shape
    shape_ok = shape == (11, 256, 256)
    print(f"   shape: {shape} {'✓' if shape_ok else '✗'}")

    # Dtype
    dtype_ok = dtype == "float32"
    print(f"   dtype: {dtype} {'✓' if dtype_ok else '✗'}")

    # Píxeles vacíos (valor exactamente 0 en todas las bandas)
    all_zero = (data == 0).all(axis=0)
    zero_frac = all_zero.mean()
    zero_ok = zero_frac < 0.10  # máximo 10% de píxeles vacíos
    print(f"   píxeles vacíos (todas bandas=0): "
          f"{zero_frac*100:.1f}% {'✓' if zero_ok else '✗ PROBLEMA'}")

    # Rango global
    valid = data[data > 0]
    if len(valid) == 0:
        print(f"   ✗ PATCH COMPLETAMENTE VACÍO\n")
        resumen.append({"patch": tif_path.name, "ok": False,
                        "problema": "completamente vacío"})
        continue

    global_max = data.max()
    range_ok = global_max <= 1.5
    print(f"   rango global: [{data.min():.4f}, {global_max:.4f}] "
          f"{'✓' if range_ok else '✗ fuera de rango'}")

    # Verificación por banda
    band_issues = []
    print(f"   medias por banda:")
    for b in range(11):
        mean_val = data[b].mean()
        b_min, b_max = EXPECTED[b]
        ok = b_min <= mean_val <= b_max
        marker = "✓" if ok else "⚠"
        print(f"     {marker} {MARIDA_NAMES[b]:4s}: "
              f"mean={mean_val:.5f}  "
              f"[esperado {b_min:.3f}-{b_max:.3f}]")
        if not ok:
            band_issues.append(MARIDA_NAMES[b])

    patch_ok = shape_ok and dtype_ok and zero_ok and range_ok
    status = "✓ OK" if patch_ok else "✗ PROBLEMA"
    if band_issues:
        status += f" (bandas fuera de rango: {', '.join(band_issues)})"

    print(f"   → {status}\n")
    resumen.append({
        "patch":    tif_path.name,
        "label":    label,
        "ok":       patch_ok,
        "zero_pct": round(zero_frac * 100, 1),
        "max":      round(global_max, 4),
        "mean":     round(data.mean(), 5),
        "problema": ", ".join(band_issues) if band_issues else "ninguno"
    })

# ── Resumen final ─────────────────────────────────────────────
print("=" * 55)
print("RESUMEN")
print("=" * 55)
ok_count = sum(1 for r in resumen if r["ok"])
print(f"Patches válidos: {ok_count}/{len(resumen)}\n")

for r in resumen:
    marca = "✓" if r["ok"] else "✗"
    print(f"  {marca} {r['patch']}")
    if not r["ok"]:
        print(f"      problema: {r['problema']}")

# ── Visualización RGB ─────────────────────────────────────────
print("\nGenerando visualización RGB...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, tif_path in zip(axes, patches):
    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")

    # RGB = B04(idx3), B03(idx2), B02(idx1)
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)

    # Normalización percentil 2-98
    for c in range(3):
        p2, p98 = np.percentile(rgb[:,:,c][rgb[:,:,c] > 0]
                                if (rgb[:,:,c] > 0).any()
                                else rgb[:,:,c], (2, 98))
        rgb[:,:,c] = np.clip(
            (rgb[:,:,c] - p2) / (p98 - p2 + 1e-10), 0, 1
        )

    label  = "SI" if "SI" in tif_path.name else "NO"
    r_info = next((r for r in resumen
                   if r["patch"] == tif_path.name), {})
    color  = "green" if r_info.get("ok") else "red"

    ax.imshow(rgb)
    ax.set_title(
        f"{tif_path.name[:22]}\n"
        f"[{label}] mean={r_info.get('mean','?'):.4f} "
        f"zero={r_info.get('zero_pct','?')}%",
        fontsize=7, color=color
    )
    ax.axis("off")

# Apagar ejes sobrantes si hay menos de 6 patches
for ax in axes[len(patches):]:
    ax.axis("off")

plt.suptitle("Patches descargados — RGB (B04/B03/B02)", y=1.02)
plt.tight_layout()
out_fig = PATCHES_DIR.parent / "patches_rgb_check.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada: {out_fig}")