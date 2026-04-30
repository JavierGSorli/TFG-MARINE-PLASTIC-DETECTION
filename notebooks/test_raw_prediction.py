# check_all_si_patches.py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

PATCHES_DIR = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                   r"\results\auto\test_patches_final")
MASKS_DIR   = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                   r"\results\auto\test_masks_final")

MARIDA_CLASSES = {
    1:"Marine Debris",    2:"Dense Sargassum",
    3:"Sparse Sargassum", 4:"Natural Organic Material",
    5:"Ship",             6:"Clouds",
    7:"Marine Water",     8:"Sediment-Laden Water",
    9:"Foam",            10:"Turbid Water",
    11:"Shallow Water"
}

si_patches = sorted([p for p in PATCHES_DIR.glob("*SI*.tif")])
print(f"Patches SI encontrados: {len(si_patches)}")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, tif_path in enumerate(si_patches):
    mask_path = MASKS_DIR / tif_path.name.replace(".tif","_mask.tif")

    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")
    with rasterio.open(mask_path) as src:
        pred = src.read(1).astype(int)

    # RGB B04/B03/B02
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    for c in range(3):
        vals = rgb[:,:,c]
        p2, p98 = np.percentile(vals, (2, 98))
        rgb[:,:,c] = np.clip(
            (vals - p2) / (p98 - p2 + 1e-10), 0, 1
        )

    plastic_mask = (pred == 1)
    n_plastic    = plastic_mask.sum()

    # Col 0: RGB
    axes[row, 0].imshow(rgb)
    axes[row, 0].set_title(
        f"{tif_path.name[:24]}\nRGB (B04/B03/B02)",
        fontsize=8)
    axes[row, 0].axis("off")

    # Col 1: Predicción completa con colorbar
    cmap = plt.cm.get_cmap("tab20", 11)
    im = axes[row, 1].imshow(pred, cmap=cmap, vmin=1, vmax=11)
    # Leyenda de clases presentes
    unique = np.unique(pred)
    legend = [plt.Rectangle((0,0),1,1,
                color=cmap((c-1)/10),
                label=f"{c}:{MARIDA_CLASSES[c][:12]}")
              for c in unique]
    axes[row, 1].legend(handles=legend, fontsize=5,
                        loc="lower right")
    axes[row, 1].set_title(
        f"Predicción UNet\n{n_plastic} px plástico",
        fontsize=8)
    axes[row, 1].axis("off")

    # Col 2: RGB + overlay plástico en rojo
    overlay = rgb.copy()
    overlay[plastic_mask, 0] = 1.0
    overlay[plastic_mask, 1] = 0.0
    overlay[plastic_mask, 2] = 0.0
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(
        f"RGB + plástico (rojo)\n{n_plastic} px detectados",
        fontsize=8)
    axes[row, 2].axis("off")

    # Stats en consola
    print(f"\n── {tif_path.name}")
    print(f"   Plástico detectado: {n_plastic} px")
    unique_cls, counts = np.unique(pred, return_counts=True)
    for cls, cnt in zip(unique_cls, counts):
        pct    = 100 * cnt / pred.size
        marker = " ← PLÁSTICO" if cls == 1 else ""
        print(f"   Clase {cls:2d} "
              f"({MARIDA_CLASSES[cls]:25s}): "
              f"{cnt:5d} px  {pct:5.1f}%{marker}")

plt.suptitle("Patches SI — RGB / Predicción / Overlay",
             y=1.01, fontsize=12)
plt.tight_layout()

out = PATCHES_DIR.parent / "si_patches_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGuardado: {out}")