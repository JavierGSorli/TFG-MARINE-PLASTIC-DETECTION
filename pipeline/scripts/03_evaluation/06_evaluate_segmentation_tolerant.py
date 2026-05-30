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
from scipy.ndimage import binary_dilation, distance_transform_edt

from src.common.config import (
    DATASET_METADATA_GROUPED_PATH,
    DEBRIS_CLASS,
    EVAL_TOLERANT_OUT,
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


def _path_factory(base: Path, suffix: str):
    return lambda stem: base / f"{stem}{suffix}"


def _external_mask_path(model_dir: Path, stem: str, calibrated: bool = False) -> Path:
    if calibrated:
        return model_dir / "calibrated_masks" / f"{stem}_mask.tif"
    path = model_dir / "masks" / f"{stem}_mask.tif"
    return path if path.exists() else model_dir / f"{stem}_mask.tif"


def build_methods():
    methods = [
        ("UNet argmax", lambda stem: UNET_OUT / f"{stem}_mask.tif", lambda arr: arr == DEBRIS_CLASS),
        ("UNet calibrated", lambda stem: UNET_CALIBRATED_MASKS_OUT / f"{stem}_mask.tif", lambda arr: arr > 0),
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
        methods.append((f"External {variant_name} default", lambda stem, model_dir=model_dir: _external_mask_path(model_dir, stem, False), lambda arr: arr > 0))
        methods.append((f"External {variant_name} calibrated", lambda stem, model_dir=model_dir: _external_mask_path(model_dir, stem, True), lambda arr: arr > 0))
    return methods


def _read_mask(path: Path, positive_fn):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1)
    return positive_fn(arr).astype(bool)


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _tolerant_metrics(gt: np.ndarray, pred: np.ndarray, radius: int) -> dict:
    struct = _disk(radius)
    gt_dil = binary_dilation(gt, structure=struct)
    pred_dil = binary_dilation(pred, structure=struct)
    tp = int((pred & gt_dil).sum())
    fp = int((pred & ~gt_dil).sum())
    fn = int((gt & ~pred_dil).sum())
    tn = int((~pred & ~gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    if pred.any():
        dist = distance_transform_edt(~gt)
        pred_distances = dist[pred]
        mean_distance = float(pred_distances.mean()) if pred_distances.size else np.nan
        p95_distance = float(np.percentile(pred_distances, 95)) if pred_distances.size else np.nan
    else:
        mean_distance = np.nan
        p95_distance = np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tolerant_precision": round(precision, 4),
        "tolerant_recall": round(recall, 4),
        "tolerant_f1": round(f1, 4),
        "tolerant_iou": round(iou, 4),
        "mean_distance_to_gt": round(mean_distance, 4) if np.isfinite(mean_distance) else np.nan,
        "p95_distance_to_gt": round(p95_distance, 4) if np.isfinite(p95_distance) else np.nan,
    }


def main() -> None:
    EVAL_TOLERANT_OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATASET_METADATA_GROUPED_PATH)
    test_patches = patches_for_subset("test_final")
    positives = metadata[
        (metadata["patch"].astype(str).isin(test_patches))
        & (metadata["label"].astype(str).str.upper() == "SI")
    ].copy()

    radii = [1, 2, 3]
    rows = []
    for _, row in positives.iterrows():
        patch_name = str(row["patch"])
        stem = Path(patch_name).stem
        gt_path = PATCHES_DIR / f"{stem}_mask.tif"
        if not gt_path.exists():
            continue
        with rasterio.open(gt_path) as src:
            gt = (src.read(1) > 0).astype(bool)

        for method_name, path_fn, positive_fn in build_methods():
            pred = _read_mask(path_fn(stem), positive_fn)
            if pred is None:
                continue
            if pred.shape != gt.shape:
                h = min(gt.shape[0], pred.shape[0])
                w = min(gt.shape[1], pred.shape[1])
                gt_cmp = gt[:h, :w]
                pred_cmp = pred[:h, :w]
            else:
                gt_cmp = gt
                pred_cmp = pred
            for radius in radii:
                rows.append({"patch": patch_name, "method": method_name, "radius_px": radius, **_tolerant_metrics(gt_cmp, pred_cmp, radius)})

    by_patch_df = pd.DataFrame(rows)
    summary_rows = []
    for (method_name, radius), group in by_patch_df.groupby(["method", "radius_px"]):
        summary_rows.append(
            {
                "method": method_name,
                "radius_px": int(radius),
                "n_patches": int(len(group)),
                "mean_tolerant_precision": round(float(group["tolerant_precision"].mean()), 4),
                "mean_tolerant_recall": round(float(group["tolerant_recall"].mean()), 4),
                "mean_tolerant_f1": round(float(group["tolerant_f1"].mean()), 4),
                "mean_tolerant_iou": round(float(group["tolerant_iou"].mean()), 4),
                "mean_distance_to_gt": round(float(group["mean_distance_to_gt"].mean()), 4),
                "mean_p95_distance_to_gt": round(float(group["p95_distance_to_gt"].mean()), 4),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["radius_px", "mean_tolerant_f1"], ascending=[True, False])
    by_patch_df.to_csv(EVAL_TOLERANT_OUT / "tolerant_metrics_by_patch.csv", index=False)
    summary_df.to_csv(EVAL_TOLERANT_OUT / "tolerant_metrics_summary.csv", index=False)
    metric_cols = [
        "method",
        "mean_tolerant_precision",
        "mean_tolerant_recall",
        "mean_tolerant_f1",
        "mean_tolerant_iou",
        "mean_distance_to_gt",
        "mean_p95_distance_to_gt",
    ]
    summary_md = "# Tolerant segmentation summary\n\n"
    for radius in sorted(summary_df["radius_px"].unique().tolist()):
        radius_df = summary_df[summary_df["radius_px"] == radius].copy().sort_values("mean_tolerant_f1", ascending=False)
        summary_md += f"## Radio {int(radius)} px\n\n"
        summary_md += _df_to_markdown_table(radius_df[metric_cols])
        summary_md += "\n"
    global_df = summary_df.copy().sort_values(["mean_tolerant_f1", "radius_px"], ascending=[False, True])
    summary_md += "## Comparativa global\n\n"
    summary_md += _df_to_markdown_table(global_df[["radius_px", *metric_cols]])
    (EVAL_TOLERANT_OUT / "tolerant_summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
