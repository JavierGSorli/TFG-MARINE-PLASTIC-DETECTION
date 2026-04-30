import rasterio
import numpy as np

patches = {
    "Cairo (manual, funciona)":
        r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
        r"\data\marida\marine-debris.github.io"
        r"\semantic_segmentation\unet\prueba_cairo\cairo.tif",
    "Gibraltar SI_001053 (OpenEO)":
        r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
        r"\results\auto\test_patches_final\20200329_SI_001053_00.tif",
}

BANDS = ["B01","B02","B03","B04","B05","B06",
         "B07","B08","B8A","B11","B12"]

for nombre, ruta in patches.items():
    with rasterio.open(ruta) as src:
        d    = src.read().astype("float32")
        crs  = src.crs
        res  = src.res
        dt   = src.dtypes[0]
    print(f"\n{'='*50}")
    print(f"{nombre}")
    print(f"{'='*50}")
    print(f"  dtype:  {dt}")
    print(f"  shape:  {d.shape}")
    print(f"  CRS:    {crs}")
    print(f"  res:    {res}")
    print(f"  global mean: {d.mean():.6f}")
    print(f"  global max:  {d.max():.6f}")
    print(f"  global min:  {d.min():.6f}")
    print(f"\n  Por banda:")
    for i, b in enumerate(BANDS):
        print(f"    {b}: "
              f"mean={d[i].mean():.6f}  "
              f"min={d[i].min():.6f}  "
              f"max={d[i].max():.6f}")