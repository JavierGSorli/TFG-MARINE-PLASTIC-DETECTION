#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage import measure


DEFAULT_MASK = Path(
    r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
    r"\notebooks\indices\pruebasghana\ghana_fdi_ndvi_mask.tif"
)


def read_mask(path):
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Mask shape no soportada: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Se esperaba una máscara 2D y se obtuvo {arr.shape}")

    return (arr > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="Visualiza una máscara binaria al estilo del notebook de Digital Earth Africa."
    )
    ap.add_argument("--mask_tif", default=str(DEFAULT_MASK), help="Ruta al TIFF binario")
    ap.add_argument("--min_area", type=int, default=1, help="Área mínima para mostrar contornos")
    ap.add_argument("--out_png", default="", help="Ruta opcional para guardar la figura")
    args = ap.parse_args()

    mask_path = Path(args.mask_tif)
    mask = read_mask(mask_path)

    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels)
    keep_labels = {r.label for r in regions if r.area >= args.min_area}
    mask_filtered = np.isin(labels, list(keep_labels)).astype(np.uint8)

    contours = measure.find_contours(mask_filtered, 0.5)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(mask, cmap="binary", vmin=0, vmax=1)
    axs[0].set_title("Predicted debris mask")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray", vmin=0, vmax=1, alpha=0.15)
    for contour in contours:
        axs[1].plot(contour[:, 1], contour[:, 0], color="black", linewidth=1.2)
    axs[1].set_title("Vectorized debris contours")
    axs[1].axis("off")

    fig.suptitle(mask_path.name)
    plt.tight_layout()

    if args.out_png:
        plt.savefig(args.out_png, dpi=180, bbox_inches="tight")
        print(f"Figura guardada en: {args.out_png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
