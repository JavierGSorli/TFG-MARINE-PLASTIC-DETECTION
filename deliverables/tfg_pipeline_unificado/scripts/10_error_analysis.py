from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from config import CSV_MASTER, DEBRIS_CLASS, EVAL_OUT, ERROR_OUT, INDICES_OUT, PATCHES_DIR, RF_OUT, UNET_OUT


def make_rgb(data, vmax=0.1):
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    return np.clip(rgb / vmax, 0, 1)


def overlay_mask(rgb, mask, color=(1, 0, 0), alpha=0.6):
    out = rgb.copy()
    active = mask.astype(bool)
    for channel, value in enumerate(color):
        out[:, :, channel] = np.where(
            active,
            alpha * value + (1 - alpha) * rgb[:, :, channel],
            rgb[:, :, channel],
        )
    return out


def read_mask(path):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def to_bool_mask(mask, mode):
    if mask is None:
        return None
    if mode == "debris_class":
        return mask == DEBRIS_CLASS
    return mask > 0


def expected_gt_px_from_patch_name(filename):
    stem = filename.replace(".tif", "")
    parts = stem.split("_")
    if len(parts) >= 3 and parts[1] == "SI":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_cases_per_type", type=int, default=10)
    args = parser.parse_args()

    ERROR_OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_MASTER)
    y_true = (df["label"] == "SI").astype(int).values
    table = pd.read_csv(EVAL_OUT / "tabla_comparativa.csv")
    thresholds = {row["Metodo"]: float(row["Umbral"]) for _, row in table.iterrows()}

    score_cols = {
        "UNet (MARIDA)": ("unet_pct", UNET_OUT, "_mask.tif", "debris_class"),
        "RF (MARIDA)": ("rf_pct", RF_OUT, "_mask.tif", "debris_class"),
        "FDI": ("fdi_pct", INDICES_OUT, "_fdi_mask.tif", "binary"),
        "NDVI": ("ndvi_pct", INDICES_OUT, "_ndvi_mask.tif", "binary"),
        "FDI+NDVI": ("fdi_ndvi_pct", INDICES_OUT, "_fdi_ndvi_mask.tif", "binary"),
    }

    for method_name, (score_col, mask_dir, mask_suffix, mask_mode) in score_cols.items():
        if score_col not in df.columns or df[score_col].notna().sum() == 0:
            continue

        threshold = thresholds.get(method_name, 0.5)
        scores = df[score_col].fillna(0).values
        preds = (scores >= threshold).astype(int)

        fp_idx = np.where((preds == 1) & (y_true == 0))[0][: args.max_cases_per_type]
        fn_idx = np.where((preds == 0) & (y_true == 1))[0][: args.max_cases_per_type]
        cases = [(idx, "FP") for idx in fp_idx] + [(idx, "FN") for idx in fn_idx]

        if not cases:
            print(f"[{method_name}] Sin errores.")
            continue

        print(f"\n[{method_name}] FP={len(fp_idx)}  FN={len(fn_idx)}")
        for idx, error_type in cases:
            row = df.iloc[idx]
            patch_path = PATCHES_DIR / row["patch"]
            stem = patch_path.stem
            pred_mask_raw = read_mask(mask_dir / f"{stem}{mask_suffix}")
            gt_mask_raw = read_mask(PATCHES_DIR / f"{stem}_mask.tif")
            pred_mask = to_bool_mask(pred_mask_raw, mask_mode)
            gt_mask = to_bool_mask(gt_mask_raw, "binary")

            with rasterio.open(patch_path) as src:
                data = src.read().astype("float32")
            rgb = make_rgb(data)

            pred_px = int(pred_mask.sum()) if pred_mask is not None else None
            gt_px = int(gt_mask.sum()) if gt_mask is not None else 0
            tp_px = int(np.logical_and(pred_mask, gt_mask).sum()) if pred_mask is not None and gt_mask is not None else None
            expected_gt_px = expected_gt_px_from_patch_name(row["patch"])

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            axes[1].imshow(overlay_mask(rgb, pred_mask) if pred_mask is not None else rgb)
            axes[1].set_title(f"Prediccion {method_name}")
            axes[1].axis("off")

            if gt_mask is not None:
                axes[2].imshow(overlay_mask(rgb, gt_mask, color=(0, 1, 0)))
                axes[2].set_title("GT Nature")
            else:
                axes[2].imshow(rgb)
                axes[2].set_title("GT no disponible")
            axes[2].axis("off")

            fig.suptitle(
                f"{error_type} | {method_name} | {row['patch']} | "
                f"score={float(scores[idx]):.4f} | umbral={threshold:.4f} | "
                f"GTexp={expected_gt_px} GTmask={gt_px} Pred={pred_px} TPpx={tp_px}",
                fontsize=9,
            )
            plt.tight_layout()

            safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
            out_png = ERROR_OUT / f"{error_type}_{safe_name}_{stem}.png"
            plt.savefig(out_png, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  -> {out_png.name}")

    print(f"\nAnalisis de errores guardado en: {ERROR_OUT}")


if __name__ == "__main__":
    main()
