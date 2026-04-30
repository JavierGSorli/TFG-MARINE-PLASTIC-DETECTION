import numpy as np
import rasterio
import matplotlib.pyplot as plt

rgb_path = r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\data\marida\marine-debris.github.io\semantic_segmentation\unet\prueba_haiti\marida\S2_22-12-20_18QYF_0.tif"
fdi_path = r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\notebooks\pruebashaiti\S2_22-12-20_18QYF_0_fdi.tif"
ndvi_path = r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\notebooks\pruebashaiti\S2_22-12-20_18QYF_0_ndvi.tif"
fdi_mask_path = r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\notebooks\pruebashaiti\S2_22-12-20_18QYF_0_fdi_mask.tif"
ndvi_mask_path = r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\notebooks\pruebashaiti\S2_22-12-20_18QYF_0_ndvi_mask.tif"
both_mask_path = r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\notebooks\pruebashaiti\S2_22-12-20_18QYF_0_fdi_ndvi_mask.tif"

with rasterio.open(rgb_path) as src:
    data = src.read().astype("float32")

with rasterio.open(fdi_path) as src:
    fdi = src.read(1).astype("float32")

with rasterio.open(ndvi_path) as src:
    ndvi = src.read(1).astype("float32")

with rasterio.open(fdi_mask_path) as src:
    fdi_mask = src.read(1).astype("uint8")

with rasterio.open(ndvi_mask_path) as src:
    ndvi_mask = src.read(1).astype("uint8")

with rasterio.open(both_mask_path) as src:
    both_mask = src.read(1).astype("uint8")

# RGB = B04, B03, B02 -> índices 3,2,1
rgb = np.stack([data[3], data[2], data[1]], axis=-1)

# normalización visual robusta
for c in range(3):
    band = rgb[:, :, c]
    valid = band[np.isfinite(band) & (band > 0)]
    if valid.size > 0:
        p2, p98 = np.percentile(valid, (2, 98))
        rgb[:, :, c] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    else:
        rgb[:, :, c] = 0

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("RGB")
axes[0, 0].axis("off")

im1 = axes[0, 1].imshow(fdi, cmap="viridis")
axes[0, 1].set_title(f"FDI\nmin={np.nanmin(fdi):.4f} max={np.nanmax(fdi):.4f}")
axes[0, 1].axis("off")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im2 = axes[0, 2].imshow(ndvi, cmap="viridis")
axes[0, 2].set_title(f"NDVI\nmin={np.nanmin(ndvi):.4f} max={np.nanmax(ndvi):.4f}")
axes[0, 2].axis("off")
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

axes[1, 0].imshow(rgb)
axes[1, 0].imshow(np.ma.masked_where(fdi_mask == 0, fdi_mask), alpha=0.5, cmap="autumn")
axes[1, 0].set_title(f"FDI mask ({fdi_mask.sum()} px)")
axes[1, 0].axis("off")

axes[1, 1].imshow(rgb)
axes[1, 1].imshow(np.ma.masked_where(ndvi_mask == 0, ndvi_mask), alpha=0.5, cmap="autumn")
axes[1, 1].set_title(f"NDVI mask ({ndvi_mask.sum()} px)")
axes[1, 1].axis("off")

axes[1, 2].imshow(rgb)
axes[1, 2].imshow(np.ma.masked_where(both_mask == 0, both_mask), alpha=0.5, cmap="autumn")
axes[1, 2].set_title(f"FDI AND NDVI ({both_mask.sum()} px)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()