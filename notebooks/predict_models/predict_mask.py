#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict a segmentation mask from a multiband patch GeoTIFF using the MARIDA UNet model.
Preprocessing matches semantic_segmentation/unet/evaluation.py.

python predict_mask.py --patch_tif "" --out_mask "" --auto_scale
"""

import os
import argparse
import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms

from unet import UNet
from dataloader import bands_mean, bands_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_tif", required=True, help="Patch multibanda (GeoTIFF).")
    ap.add_argument("--out_mask", required=True, help="GeoTIFF de salida con máscara.")
    ap.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "trained_models", "44", "model.pth"))
    ap.add_argument("--input_channels", type=int, default=11)
    ap.add_argument("--output_channels", type=int, default=11)
    ap.add_argument("--hidden_channels", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu o cuda")
    ap.add_argument("--auto_scale", action="store_true", help="Si max>1.5, divide entre 10000.")
    args = ap.parse_args()

    device = torch.device(args.device)

    model = UNet(
        input_bands=args.input_channels,
        output_classes=args.output_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    with rasterio.open(args.patch_tif, "r") as src:
        meta = src.meta.copy()
        tags = src.tags().copy()
        img = src.read().astype("float32")  # (C,H,W)
        dtype = src.read(1).dtype

    if img.shape[0] != args.input_channels:
        raise ValueError(f"Esperaba {args.input_channels} bandas pero hay {img.shape[0]}.")

    img_hwc = np.moveaxis(img, (0, 1, 2), (2, 0, 1))

    if args.auto_scale and img_hwc.max() > 1.5:
        img_hwc = img_hwc / 10000.0

    H, W, _ = img_hwc.shape
    impute_nan = np.tile(bands_mean, (H, W, 1))
    nan_mask = np.isnan(img_hwc)
    img_hwc[nan_mask] = impute_nan[nan_mask]

    transform = transforms.ToTensor()
    standardization = transforms.Normalize(bands_mean, bands_std)

    x = transform(img_hwc)
    print("raw min/max:", float(x.min()), float(x.max()))
    x = standardization(x).unsqueeze(0).to(device)
    print("norm min/max:", float(x.min()), float(x.max()))
    print("norm mean/std:", float(x.mean()), float(x.std()))

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(1).squeeze() + 1  # 1..11 como evaluation.py

    meta.update(count=1, dtype=dtype)
    os.makedirs(os.path.dirname(args.out_mask) or ".", exist_ok=True)
    with rasterio.open(args.out_mask, "w", **meta) as dst:
        dst.write_band(1, pred.astype(dtype).copy())
        dst.update_tags(**tags)

    print("OK ->", args.out_mask)


if __name__ == "__main__":
    main()
