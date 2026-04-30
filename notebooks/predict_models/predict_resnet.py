#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a single multiband GeoTIFF patch with the MARIDA multi-label ResNet model.

This follows MARIDA evaluation.py + dataloader.py as closely as possible:
- np.moveaxis(img, [0,1,2], [2,0,1]).astype('float32')
- NaN imputation with tiled bands_mean
- transforms.ToTensor()
- transforms.Normalize(bands_mean, bands_std)
- logits = model(x)
- probs = torch.sigmoid(logits)
- thresholding
"""

import os
import sys
import json
import argparse
import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms
from os.path import dirname as up

# Same imports as MARIDA
sys.path.append(up(os.path.abspath(__file__)))
from resnet import ResNet
from dataloader import bands_mean, bands_std

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from assets import labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_tif", required=True, help="Patch multibanda (GeoTIFF).")
    ap.add_argument(
        "--model_path",
        default=r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\data\marida\marine-debris.github.io\multi-label\resnet\trained_models\18\model.pth",
        help="Ruta al modelo ResNet .pth"
    )
    ap.add_argument("--out_json", required=True, help="JSON de salida con probabilidades y etiquetas.")
    ap.add_argument("--input_channels", type=int, default=11, help="Número de bandas de entrada.")
    ap.add_argument("--output_channels", type=int, default=11, help="Número de salidas del modelo.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Umbral multilabel.")
    ap.add_argument("--agg_to_water", default=True, type=bool,
                    help="Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water")
    ap.add_argument("--device", default="cpu", help="cpu o cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Same label handling as evaluation.py
    out_labels = labels.copy()
    if args.agg_to_water:
        out_labels = out_labels[:-4]

    # Same transforms as evaluation.py
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    # Same model construction as evaluation.py
    model = ResNet(
        input_bands=args.input_channels,
        output_classes=args.output_channels
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Read patch as C,H,W exactly like gdal.ReadAsArray() would give
    with rasterio.open(args.patch_tif, "r") as src:
        img = src.read().astype("float32")  # (C,H,W)

    if img.shape[0] != args.input_channels:
        raise ValueError(f"Esperaba {args.input_channels} bandas pero hay {img.shape[0]}.")

    # EXACT dataloader.py preprocessing
    img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype("float32")  # CxWxH to WxHxC

    impute_nan = np.tile(bands_mean, (img.shape[0], img.shape[1], 1))
    nan_mask = np.isnan(img)
    img[nan_mask] = impute_nan[nan_mask]

    img = transform_test(img)
    img = standardization(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        pred = (probs >= args.threshold).astype(np.float32)

    if probs.ndim == 0:
        probs = np.array([float(probs)], dtype=np.float32)
        pred = np.array([float(pred)], dtype=np.float32)

    if len(out_labels) != len(probs):
        raise ValueError(
            f"Número de labels ({len(out_labels)}) y salidas del modelo ({len(probs)}) no coinciden."
        )

    result = {
        "patch_tif": args.patch_tif,
        "model_path": args.model_path,
        "threshold": args.threshold,
        "agg_to_water": args.agg_to_water,
        "probabilities": {label: float(prob) for label, prob in zip(out_labels, probs)},
        "predicted_labels": {label: int(v) for label, v in zip(out_labels, pred)},
        "active_labels": [label for label, v in zip(out_labels, pred) if v == 1],
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nProbabilities:")
    for label, prob in zip(out_labels, probs):
        print(f"{label:25s} {prob:.4f}")

    print("\nActive labels:")
    active = [label for label, v in zip(out_labels, pred) if v == 1]
    if active:
        for label in active:
            print(" -", label)
    else:
        print(" - none")

    print("\nOK ->", args.out_json)


if __name__ == "__main__":
    main()
