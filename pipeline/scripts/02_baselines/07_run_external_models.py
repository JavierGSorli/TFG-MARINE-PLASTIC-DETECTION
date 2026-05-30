from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import segmentation_models_pytorch as smp

from src.common.config import (
    DATA_DIR,
    EXTERNAL_B09_ZERO_OUT,
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
    PATCHES_DIR,
)


MODEL_NAME = "marinedebrisdetector"

CKPT_PATH = DATA_DIR / "external_models" / MODEL_NAME / "unetplusplus1.ckpt"
CKPT_URL = (
    "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/checkpoints/"
    "unet%2B%2B1/epoch=54-val_loss=0.50-auroc=0.987.ckpt"
)

B09_INSERT_IDX = 9
THRESHOLD = 0.5


def iter_patch_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.glob("*.tif")
        if "_mask" not in path.stem and not path.name.startswith("_")
    )


def load_model(device: torch.device) -> torch.nn.Module:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {CKPT_PATH}\n"
            f"Descargalo desde: {CKPT_URL}\n"
            f"y guardalo como: {CKPT_PATH}"
        )

    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    model = smp.UnetPlusPlus(in_channels=12, classes=1)
    state = {
        key[len("model."):]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_b09_channel(arr: np.ndarray, b09_mode: str) -> np.ndarray:
    """Construye el canal B09 (shape (1, H, W)) según el modo indicado."""
    h, w = arr.shape[1], arr.shape[2]
    if b09_mode == "zero":
        return np.zeros((1, h, w), dtype=np.float32)
    elif b09_mode == "copy_b8a":
        # B8A está en índice 8 del array de 11 bandas MARIDA
        return arr[8:9].copy()
    elif b09_mode == "interpolate_b8a_b11":
        # Interpolación lineal entre B8A (idx 8) y B11 (idx 9)
        return ((arr[8:9].astype(np.float32) + arr[9:10].astype(np.float32)) / 2.0)
    else:
        raise ValueError(f"b09_mode desconocido: {b09_mode}")


def read_patch_for_model(patch_path: Path, b09_mode: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(str(patch_path)) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()

    if arr.shape[0] != 11:
        raise ValueError(f"{patch_path.name}: se esperaban 11 bandas MARIDA, tiene {arr.shape[0]}")

    b09 = build_b09_channel(arr, b09_mode)
    arr = np.concatenate([arr[:B09_INSERT_IDX], b09, arr[B09_INSERT_IDX:]], axis=0)
    return arr, profile


def predict_patch(model: torch.nn.Module, patch_path: Path, device: torch.device, b09_mode: str) -> tuple[float, int, np.ndarray, dict]:
    arr, profile = read_patch_for_model(patch_path, b09_mode)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze().detach().cpu().numpy()

    score = float(probs.mean())
    pred_px = int((probs >= THRESHOLD).sum())
    return score, pred_px, probs, profile


def save_mask(probs: np.ndarray, profile: dict, mask_path: Path) -> None:
    profile = profile.copy()
    profile.update(count=1, dtype=rasterio.float32, nodata=None)
    prob_path = mask_path.parent / f"{mask_path.stem}_prob.tif"
    with rasterio.open(str(prob_path), "w", **profile) as dst:
        dst.write(probs[np.newaxis, :, :])

    profile.update(dtype=rasterio.uint8)
    with rasterio.open(str(mask_path), "w", **profile) as dst:
        dst.write((probs >= THRESHOLD).astype(np.uint8)[np.newaxis, :, :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-save-prob", action="store_true")
    parser.add_argument(
        "--b09-mode",
        choices=["zero", "copy_b8a", "interpolate_b8a_b11"],
        default="zero",
        help="Modo de generacion del canal B09: 'zero' (ceros), 'copy_b8a' (copia B8A), 'interpolate_b8a_b11' (interpolacion B8A-B11)",
    )
    args = parser.parse_args()
    if args.no_save_prob:
        print("WARN: --no-save-prob está deprecado e ignorado. Se guardará siempre el raster de probabilidades.")

    if args.b09_mode == "zero":
        output_dir = EXTERNAL_B09_ZERO_OUT
    elif args.b09_mode == "copy_b8a":
        output_dir = EXTERNAL_B09_COPY_B8A_OUT
    else:
        output_dir = EXTERNAL_B09_INTERP_OUT

    masks_dir = output_dir / "masks"

    if not PATCHES_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de patches: {PATCHES_DIR}")

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"{MODEL_NAME}: device={device}  b09_mode={args.b09_mode}")

    model = load_model(device)
    rows = []
    patches = iter_patch_files(PATCHES_DIR)
    existing = {}
    predictions_path = output_dir / "predictions.csv"
    if predictions_path.exists() and not args.overwrite:
        try:
            existing = pd.read_csv(predictions_path).set_index("patch").to_dict("index")
        except Exception:
            existing = {}

    for idx, patch in enumerate(patches, start=1):
        mask_path = masks_dir / f"{patch.stem}_mask.tif"
        if mask_path.exists() and patch.name in existing and not args.overwrite:
            row = existing[patch.name]
            rows.append(
                {
                    "patch": patch.name,
                    "method": row.get("method", MODEL_NAME),
                    "score": row.get("score", ""),
                    "pred_px": row.get("pred_px", ""),
                    "has_prediction": row.get("has_prediction", 1),
                    "status": "existing",
                    "error_message": "",
                }
            )
            print(f"[{idx}/{len(patches)}] SKIP {patch.name}")
            continue

        try:
            score, pred_px, probs, profile = predict_patch(model, patch, device, args.b09_mode)
            save_mask(probs, profile, mask_path)
            rows.append(
                {
                    "patch": patch.name,
                    "method": MODEL_NAME,
                    "score": round(score, 6),
                    "pred_px": pred_px,
                    "has_prediction": 1,
                    "status": "ok",
                    "error_message": "",
                }
            )
            print(f"[{idx}/{len(patches)}] OK {patch.name} score={score:.6f} pred_px={pred_px}")
        except Exception as exc:
            rows.append(
                {
                    "patch": patch.name,
                    "method": MODEL_NAME,
                    "score": "",
                    "pred_px": "",
                    "has_prediction": 0,
                    "status": "error",
                    "error_message": str(exc),
                }
            )
            print(f"[{idx}/{len(patches)}] ERROR {patch.name}: {exc}")

    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    n_ok = sum(row["has_prediction"] == 1 for row in rows)
    print(f"{MODEL_NAME}: {n_ok}/{len(rows)} predicciones OK -> {output_dir}")


if __name__ == "__main__":
    main()
