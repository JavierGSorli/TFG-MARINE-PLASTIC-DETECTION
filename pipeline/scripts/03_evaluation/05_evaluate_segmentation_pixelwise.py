from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from src.common.config import (
    DATASET_METADATA_GROUPED_PATH,
    DEBRIS_CLASS,
    EVAL_PIXELWISE_OUT,
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
    EXTERNAL_B09_ZERO_OUT,
    INDICES_NO_WATER_OUT,
    INDICES_WATER_OUT,
    PATCHES_DIR,
    RF_MODE_DIRS,
    RF_MODE_NAMES,
    SAM_CALIBRATED_MASKS_OUT,
    SAM_PHASE_OUT,
    UNET_CALIBRATED_MASKS_OUT,
    UNET_OUT,
)
from src.evaluation.evaluation_split_utils import patches_for_subset


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin datos_\n"
    return df.to_markdown(index=False) + "\n"


def _method_mode(method_name: str) -> str:
    return "calibrated" if "calibrated" in method_name.lower() else "default"


def _unet_mask_path(stem: str) -> Path:
    return UNET_OUT / f"{stem}_mask.tif"


def _unet_thr_mask_path(stem: str) -> Path:
    return UNET_CALIBRATED_MASKS_OUT / f"{stem}_mask.tif"


def _path_factory(base: Path, suffix: str):
    return lambda stem: base / f"{stem}{suffix}"


def _external_mask_path(model_dir: Path, stem: str, calibrated: bool = False) -> Path:
    if calibrated:
        return model_dir / "calibrated_masks" / f"{stem}_mask.tif"
    mask_path = model_dir / "masks" / f"{stem}_mask.tif"
    return mask_path if mask_path.exists() else model_dir / f"{stem}_mask.tif"


def build_methods():
    methods = [
        ("UNet argmax", _unet_mask_path, lambda arr: arr == DEBRIS_CLASS),
        ("UNet calibrated", _unet_thr_mask_path, lambda arr: arr > 0),
        ("SAM binary", _path_factory(SAM_PHASE_OUT / "binario", "_sam_debris_mask.tif"), lambda arr: arr > 0),
        ("SAM calibrated", _path_factory(SAM_CALIBRATED_MASKS_OUT, "_sam_debris_mask.tif"), lambda arr: arr > 0),
    ]
    for mode in RF_MODE_NAMES:
        methods.append((f"RF {mode}", _path_factory(RF_MODE_DIRS[mode], "_mask.tif"), lambda arr: arr == DEBRIS_CLASS))
        methods.append((f"RF {mode} calibrated", _path_factory(RF_MODE_DIRS[mode] / "calibrated_masks", "_mask.tif"), lambda arr: arr > 0))
    methods.extend(
        [
            ("FDI", _path_factory(INDICES_NO_WATER_OUT, "_fdi_mask.tif"), lambda arr: arr > 0),
            ("NDVI", _path_factory(INDICES_NO_WATER_OUT, "_ndvi_mask.tif"), lambda arr: arr > 0),
            ("FDI+NDVI", _path_factory(INDICES_NO_WATER_OUT, "_fdi_ndvi_mask.tif"), lambda arr: arr > 0),
            ("FDI_mask", _path_factory(INDICES_WATER_OUT, "_fdi_mask.tif"), lambda arr: arr > 0),
            ("NDVI_mask", _path_factory(INDICES_WATER_OUT, "_ndvi_mask.tif"), lambda arr: arr > 0),
            ("FDI+NDVI_mask", _path_factory(INDICES_WATER_OUT, "_fdi_ndvi_mask.tif"), lambda arr: arr > 0),
        ]
    )
    for variant_name, model_dir in [("b09_zero", EXTERNAL_B09_ZERO_OUT), ("b09_copy_b8a", EXTERNAL_B09_COPY_B8A_OUT), ("b09_interpolate_b8a_b11", EXTERNAL_B09_INTERP_OUT)]:
        methods.append((f"External {variant_name} default", lambda stem, model_dir=model_dir: _external_mask_path(model_dir, stem, calibrated=False), lambda arr: arr > 0))
        methods.append((f"External {variant_name} calibrated", lambda stem, model_dir=model_dir: _external_mask_path(model_dir, stem, calibrated=True), lambda arr: arr > 0))
    return methods


def read_binary_mask(path: Path, positive_fn) -> np.ndarray | None:
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1)
    return positive_fn(arr).astype(bool)


def compute_pixel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "dice_f1": round(dice, 4),
        "iou": round(iou, 4),
        "gt_px": int(gt.sum()),
        "pred_px": int(pred.sum()),
    }


def main() -> None:
    EVAL_PIXELWISE_OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATASET_METADATA_GROUPED_PATH)
    test_patches = patches_for_subset("test_final")
    metadata = metadata[metadata["patch"].astype(str).isin(test_patches)].copy()
    positives = metadata[metadata["label"].astype(str).str.upper() == "SI"].copy()
    negatives = metadata[metadata["label"].astype(str).str.upper() == "NO"].copy()

    by_patch_rows = []
    methods = build_methods()
    for _, row in positives.iterrows():
        patch_name = str(row["patch"])
        stem = Path(patch_name).stem
        gt_path = PATCHES_DIR / f"{stem}_mask.tif"
        if not gt_path.exists():
            continue
        with rasterio.open(gt_path) as src:
            gt_mask = (src.read(1) > 0).astype(np.uint8)

        for method_name, path_fn, positive_fn in methods:
            pred_arr = read_binary_mask(path_fn(stem), positive_fn)
            if pred_arr is None:
                continue
            if pred_arr.shape != gt_mask.shape:
                h = min(gt_mask.shape[0], pred_arr.shape[0])
                w = min(gt_mask.shape[1], pred_arr.shape[1])
                gt_cmp = gt_mask[:h, :w]
                pred_cmp = pred_arr[:h, :w].astype(np.uint8)
            else:
                gt_cmp = gt_mask
                pred_cmp = pred_arr.astype(np.uint8)
            by_patch_rows.append({"patch": patch_name, "method": method_name, **compute_pixel_metrics(gt_cmp, pred_cmp)})

    by_patch_df = pd.DataFrame(by_patch_rows)
    summary_rows = []
    for method_name, group in by_patch_df.groupby("method"):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        gt_total_px = int(group["gt_px"].sum())
        pred_total_px = int(group["pred_px"].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        dice = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        summary_rows.append(
            {
                "method": method_name,
                "n_patches": int(len(group)),
                "tp_total": tp,
                "fp_total": fp,
                "fn_total": fn,
                "tn_total": tn,
                "micro_precision": round(precision, 4),
                "micro_recall": round(recall, 4),
                "micro_dice_f1": round(dice, 4),
                "micro_iou": round(iou, 4),
                "mean_iou": round(float(group["iou"].mean()), 4),
                "mean_dice": round(float(group["dice_f1"].mean()), 4),
                "gt_total_px": gt_total_px,
                "pred_total_px": pred_total_px,
            }
        )

    noise_rows = []
    negative_by_patch_rows = []
    for _, row in negatives.iterrows():
        patch_name = str(row["patch"])
        stem = Path(patch_name).stem
        patch_path = PATCHES_DIR / patch_name
        with rasterio.open(patch_path) as src:
            total_px = src.width * src.height
        for method_name, path_fn, positive_fn in methods:
            pred_arr = read_binary_mask(path_fn(stem), positive_fn)
            if pred_arr is None:
                continue
            fp_px = int(pred_arr.sum())
            noise_rows.append(
                {
                    "patch": patch_name,
                    "method": method_name,
                    "total_px": int(total_px),
                    "fp_px": fp_px,
                    "fp_rate": round(fp_px / total_px, 6) if total_px > 0 else 0.0,
                }
            )
            negative_by_patch_rows.append(
                {
                    "patch": patch_name,
                    "method": method_name,
                    "total_px": int(total_px),
                    "pred_px": fp_px,
                    "fp_px": fp_px,
                    "fp_rate": round(fp_px / total_px, 6) if total_px > 0 else 0.0,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("micro_dice_f1", ascending=False)
    noise_df = pd.DataFrame(noise_rows)
    negative_by_patch_df = pd.DataFrame(negative_by_patch_rows)
    if not noise_df.empty:
        noise_df = (
            noise_df.groupby("method")
            .agg(n_patches=("patch", "count"), mean_fp_rate=("fp_rate", "mean"), mean_fp_px=("fp_px", "mean"), total_fp_px=("fp_px", "sum"))
            .reset_index()
        )
        noise_df["mean_fp_rate"] = noise_df["mean_fp_rate"].round(6)
        noise_df["mean_fp_px"] = noise_df["mean_fp_px"].round(2)
        noise_df = noise_df.sort_values(["mean_fp_rate", "total_fp_px"], ascending=[False, False]).reset_index(drop=True)

    summary_df["mode"] = summary_df["method"].map(_method_mode)
    default_df = summary_df[summary_df["mode"] == "default"].copy().sort_values("micro_dice_f1", ascending=False)
    calibrated_df = summary_df[summary_df["mode"] == "calibrated"].copy().sort_values("micro_dice_f1", ascending=False)
    global_df = summary_df.copy().sort_values("micro_dice_f1", ascending=False)

    metric_cols = [
        "method",
        "gt_total_px",
        "pred_total_px",
        "micro_precision",
        "micro_recall",
        "micro_dice_f1",
        "micro_iou",
        "mean_iou",
        "mean_dice",
        "tp_total",
        "fp_total",
        "fn_total",
        "tn_total",
    ]

    by_patch_df.to_csv(EVAL_PIXELWISE_OUT / "pixelwise_metrics_by_patch.csv", index=False)
    summary_df.to_csv(EVAL_PIXELWISE_OUT / "pixelwise_metrics_summary.csv", index=False)
    noise_df.to_csv(EVAL_PIXELWISE_OUT / "segmentation_noise_on_negatives.csv", index=False)
    negative_by_patch_df.to_csv(EVAL_PIXELWISE_OUT / "pixelwise_negatives_by_patch.csv", index=False)
    summary_md = (
        "# Pixel-wise summary\n\n"
        "## Métodos sin calibrar (`test_final`, patches positivos)\n\n"
        f"{_df_to_markdown_table(default_df[metric_cols])}\n"
        "## Métodos calibrados (`test_final`, patches positivos)\n\n"
        f"{_df_to_markdown_table(calibrated_df[metric_cols])}\n"
        "## Comparativa global (`test_final`, patches positivos)\n\n"
        f"{_df_to_markdown_table(global_df[metric_cols])}\n"
        "## Ruido en negativos (`test_final`, patches negativos)\n\n"
        f"{_df_to_markdown_table(noise_df)}"
    )
    (EVAL_PIXELWISE_OUT / "pixelwise_summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
