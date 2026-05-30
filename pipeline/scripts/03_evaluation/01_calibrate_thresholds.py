from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import numpy as np
import pandas as pd
import rasterio

from src.common.config import EVAL_THRESHOLDS_OUT, PATCHES_DIR, RF_MODE_NAMES, RF_MODE_DIRS, SAM_PROB_DIR, THRESHOLDS_PATH, UNET_OUT
from src.evaluation.raw_prediction_table import EXTERNAL_VARIANTS, build_raw_prediction_table


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _calibrate_scalar_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[dict, list[dict]]:
    thresholds = np.unique(scores[np.isfinite(scores)])
    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=float)
    best = None
    curve_rows = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        metrics = _safe_metrics(y_true, y_pred)
        curve_rows.append(
            {
                "threshold": float(thr),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
        if best is None or metrics["f1"] > best["f1"]:
            best = {"threshold": float(thr), **metrics}
    return best or {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}, curve_rows


def _calibrate_patch_px_thresholds(y_true: np.ndarray, px_counts: np.ndarray) -> tuple[dict, list[dict]]:
    thresholds = np.unique(px_counts[np.isfinite(px_counts)]).astype(float)
    if thresholds.size == 0:
        thresholds = np.array([1.0], dtype=float)
    best = None
    curve_rows = []
    for thr in thresholds:
        y_pred = (px_counts >= thr).astype(int)
        metrics = _safe_metrics(y_true, y_pred)
        curve_rows.append(
            {
                "threshold": float(thr),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
        if best is None or metrics["f1"] > best["f1"]:
            best = {"threshold": float(thr), **metrics}
    return best or {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, curve_rows


def _calibrate_raster_thresholds_pixelwise(gt_masks: list[np.ndarray], prob_rasters: list[np.ndarray]) -> tuple[dict, list[dict]]:
    thresholds = np.linspace(0.0, 1.0, 101, dtype=float)
    best = None
    curve_rows = []
    for thr in thresholds:
        tp = fp = fn = 0
        for gt_mask, prob_arr in zip(gt_masks, prob_rasters):
            pred_mask = np.isfinite(prob_arr) & (prob_arr >= thr)
            tp += int((pred_mask & gt_mask).sum())
            fp += int((pred_mask & ~gt_mask).sum())
            fn += int((~pred_mask & gt_mask).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
        curve_rows.append(
            {
                "threshold": float(thr),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
        if best is None or metrics["f1"] > best["f1"]:
            best = {"threshold": float(thr), **metrics}
    return best or {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}, curve_rows


def _load_raster_ground_truth_pairs(df: pd.DataFrame, prob_path_col: str) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    valid_df = df[df[prob_path_col].notna()].copy()
    gt_masks: list[np.ndarray] = []
    rasters: list[np.ndarray] = []
    for _, row in valid_df.iterrows():
        prob_path = _Path(row[prob_path_col])
        if not prob_path.exists():
            continue
        with rasterio.open(str(prob_path)) as src:
            prob_arr = src.read(1).astype("float32")
        gt_mask = np.zeros(prob_arr.shape, dtype=bool)
        if int(row.get("label_binary", 0)) == 1:
            gt_path = PATCHES_DIR / f"{_Path(str(row['patch'])).stem}_mask.tif"
            if not gt_path.exists():
                continue
            with rasterio.open(str(gt_path)) as src:
                gt_arr = (src.read(1) > 0)
            if gt_arr.shape != prob_arr.shape:
                h = min(gt_arr.shape[0], prob_arr.shape[0])
                w = min(gt_arr.shape[1], prob_arr.shape[1])
                gt_arr = gt_arr[:h, :w]
                prob_arr = prob_arr[:h, :w]
            gt_mask = gt_arr.astype(bool)
        gt_masks.append(gt_mask)
        rasters.append(prob_arr)
    return gt_masks, rasters, len(rasters)


def _prepare_raster_sources(master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    master["unet_prob_path"] = master["patch"].map(
        lambda patch: UNET_OUT / f"{_Path(patch).stem}_marine_debris_prob.tif"
    )
    for mode in RF_MODE_NAMES:
        master[f"rf_{mode}_prob_path"] = master["patch"].map(
            lambda patch, mode=mode: RF_MODE_DIRS[mode] / f"{_Path(patch).stem}_marine_debris_prob.tif"
        )
    master["sam_prob_path"] = master["patch"].map(
        lambda patch: SAM_PROB_DIR / f"{_Path(patch).stem}_sam_marine_debris_score.tif"
    )
    for variant_name, variant_dir in EXTERNAL_VARIANTS:
        master[f"external_{variant_name}_prob_path"] = master["patch"].map(
            lambda patch, variant_dir=variant_dir: variant_dir / "masks" / f"{_Path(patch).stem}_mask_prob.tif"
        )
    return master


def _pixel_count_from_prob_path(prob_path: _Path, threshold: float, label_binary: int) -> int | None:
    if not prob_path.exists():
        return None
    with rasterio.open(str(prob_path)) as src:
        prob_arr = src.read(1).astype("float32")
    pred_mask = np.isfinite(prob_arr) & (prob_arr >= threshold)
    return int(pred_mask.sum())


def main() -> None:
    EVAL_THRESHOLDS_OUT.mkdir(parents=True, exist_ok=True)

    master = build_raw_prediction_table()
    master = master[master["eval_subset"] == "calibration_dev"].copy()
    if master.empty:
        raise ValueError("No hay patches calibration_dev disponibles para calibración.")
    master = _prepare_raster_sources(master)

    result_rows: list[dict] = []
    curve_rows: list[dict] = []

    # U-Net raster probability threshold (pixel-wise against GT masks).
    gt_masks, prob_rasters, n_valid = _load_raster_ground_truth_pairs(master, "unet_prob_path")
    best, curves = _calibrate_raster_thresholds_pixelwise(gt_masks, prob_rasters)
    result_rows.append(
        {
            "method_key": "unet_prob",
            "method": "UNet probability threshold",
            "threshold_kind": "raster_prob_pixelwise",
            "score_col": "unet_prob_path",
            "threshold": round(best["threshold"], 4),
            "f1": round(best["f1"], 4),
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "n_valid": n_valid,
            "n_positive": int((master["label_binary"] == 1).sum()),
            "n_negative": int((master["label_binary"] == 0).sum()),
        }
    )
    curve_rows.extend({"method_key": "unet_prob", **row} for row in curves)

    # RF raster probability thresholds (pixel-wise against GT masks).
    for mode in RF_MODE_NAMES:
        prob_col = f"rf_{mode}_prob_path"
        valid_master = master[master[prob_col].map(lambda p: _Path(p).exists())].copy()
        if valid_master.empty:
            continue
        gt_masks, prob_rasters, n_valid = _load_raster_ground_truth_pairs(valid_master, prob_col)
        best, curves = _calibrate_raster_thresholds_pixelwise(gt_masks, prob_rasters)
        result_rows.append(
            {
                "method_key": f"rf_{mode}_prob",
                "method": f"RF {mode} probability threshold",
                "threshold_kind": "raster_prob_pixelwise",
                "score_col": prob_col,
                "threshold": round(best["threshold"], 4),
                "f1": round(best["f1"], 4),
                "precision": round(best["precision"], 4),
                "recall": round(best["recall"], 4),
                "n_valid": n_valid,
                "n_positive": int((valid_master["label_binary"] == 1).sum()),
                "n_negative": int((valid_master["label_binary"] == 0).sum()),
            }
        )
        curve_rows.extend({"method_key": f"rf_{mode}_prob", **row} for row in curves)

    # SAM raster similarity threshold (pixel-wise).
    valid_master = master[master["sam_prob_path"].map(lambda p: _Path(p).exists())].copy()
    if not valid_master.empty:
        gt_masks, prob_rasters, n_valid = _load_raster_ground_truth_pairs(valid_master, "sam_prob_path")
        best, curves = _calibrate_raster_thresholds_pixelwise(gt_masks, prob_rasters)
        result_rows.append(
            {
                "method_key": "sam_prob",
                "method": "SAM Marine Debris score threshold",
                "threshold_kind": "raster_prob_pixelwise",
                "score_col": "sam_prob_path",
                "threshold": round(best["threshold"], 4),
                "f1": round(best["f1"], 4),
                "precision": round(best["precision"], 4),
                "recall": round(best["recall"], 4),
                "n_valid": n_valid,
                "n_positive": int((valid_master["label_binary"] == 1).sum()),
                "n_negative": int((valid_master["label_binary"] == 0).sum()),
            }
        )
        curve_rows.extend({"method_key": "sam_prob", **row} for row in curves)

    # External variants raster thresholds.
    for variant_name, _ in EXTERNAL_VARIANTS:
        prob_col = f"external_{variant_name}_prob_path"
        valid_master = master[master[prob_col].map(lambda p: _Path(p).exists())].copy()
        if valid_master.empty:
            continue
        gt_masks, prob_rasters, n_valid = _load_raster_ground_truth_pairs(valid_master, prob_col)
        best, curves = _calibrate_raster_thresholds_pixelwise(gt_masks, prob_rasters)
        result_rows.append(
            {
                "method_key": f"external_{variant_name}_prob",
                "method": f"External {variant_name} probability threshold",
                "threshold_kind": "raster_prob_pixelwise",
                "score_col": prob_col,
                "threshold": round(best["threshold"], 4),
                "f1": round(best["f1"], 4),
                "precision": round(best["precision"], 4),
                "recall": round(best["recall"], 4),
                "n_valid": n_valid,
                "n_positive": int((valid_master["label_binary"] == 1).sum()),
                "n_negative": int((valid_master["label_binary"] == 0).sum()),
            }
        )
        curve_rows.extend({"method_key": f"external_{variant_name}_prob", **row} for row in curves)

    # Scalar methods.
    scalar_methods = [("resnet_prob", "ResNet probability")]
    for score_col, display_name in scalar_methods:
        if score_col not in master.columns:
            continue
        valid = master[score_col].notna().copy()
        if int(valid.sum()) == 0:
            continue
        y_true = master.loc[valid, "label_binary"].astype(int).to_numpy()
        scores = pd.to_numeric(master.loc[valid, score_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(scores)
        y_true = y_true[finite]
        scores = scores[finite]
        if len(scores) == 0 or np.unique(y_true).size < 2:
            continue
        best, curves = _calibrate_scalar_scores(y_true, scores)
        result_rows.append(
            {
                "method_key": score_col,
                "method": display_name,
                "threshold_kind": "scalar_score",
                "score_col": score_col,
                "threshold": round(best["threshold"], 4),
                "f1": round(best["f1"], 4),
                "precision": round(best["precision"], 4),
                "recall": round(best["recall"], 4),
                "n_valid": int(len(scores)),
                "n_positive": int(y_true.sum()),
                "n_negative": int(len(scores) - y_true.sum()),
            }
        )
        curve_rows.extend({"method_key": score_col, **row} for row in curves)

    # Patch-level thresholds using pixel counts instead of percentages.
    patch_px_methods = [
        ("unet_argmax_px", "UNet argmax patch px"),
        ("sam_binary_px", "SAM binary patch px"),
        ("rf_full_px", "RF full patch px"),
        ("rf_no_texture_px", "RF no_texture patch px"),
        ("rf_indices_only_px", "RF indices_only patch px"),
        ("rf_bands_only_px", "RF bands_only patch px"),
        ("fdi_no_water_px", "FDI patch px"),
        ("ndvi_no_water_px", "NDVI patch px"),
        ("fdi_ndvi_no_water_px", "FDI+NDVI patch px"),
        ("fdi_water_px", "FDI_mask patch px"),
        ("ndvi_water_px", "NDVI_mask patch px"),
        ("fdi_ndvi_water_px", "FDI+NDVI_mask patch px"),
        ("external_b09_zero_default_px", "External b09_zero default patch px"),
        ("external_b09_copy_b8a_default_px", "External b09_copy_b8a default patch px"),
        ("external_b09_interpolate_b8a_b11_default_px", "External b09_interpolate_b8a_b11 default patch px"),
    ]
    for score_col, display_name in patch_px_methods:
        if score_col not in master.columns:
            continue
        valid = master[score_col].notna().copy()
        if int(valid.sum()) == 0:
            continue
        y_true = master.loc[valid, "label_binary"].astype(int).to_numpy()
        scores = pd.to_numeric(master.loc[valid, score_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(scores)
        y_true = y_true[finite]
        scores = scores[finite]
        if len(scores) == 0 or np.unique(y_true).size < 2:
            continue
        best, curves = _calibrate_patch_px_thresholds(y_true, scores)
        result_rows.append(
            {
                "method_key": score_col,
                "method": display_name,
                "threshold_kind": "patch_px",
                "score_col": score_col,
                "threshold": round(best["threshold"], 4),
                "f1": round(best["f1"], 4),
                "precision": round(best["precision"], 4),
                "recall": round(best["recall"], 4),
                "n_valid": int(len(scores)),
                "n_positive": int(y_true.sum()),
                "n_negative": int(len(scores) - y_true.sum()),
            }
        )
        curve_rows.extend({"method_key": score_col, **row} for row in curves)

    # Patch-level thresholds for calibrated masks, derived from pixelwise-selected raster thresholds.
    calibrated_patch_specs = [
        ("unet_thr_px", "UNet calibrated patch px", "unet_prob_path", "unet_prob"),
        ("sam_thr_px", "SAM calibrated patch px", "sam_prob_path", "sam_prob"),
    ]
    for mode in RF_MODE_NAMES:
        calibrated_patch_specs.append(
            (
                f"rf_{mode}_thr_px",
                f"RF {mode} calibrated patch px",
                f"rf_{mode}_prob_path",
                f"rf_{mode}_prob",
            )
        )
    for variant_name, _ in EXTERNAL_VARIANTS:
        calibrated_patch_specs.append(
            (
                f"external_{variant_name}_thr_px",
                f"External {variant_name} calibrated patch px",
                f"external_{variant_name}_prob_path",
                f"external_{variant_name}_prob",
            )
        )
    for method_key, display_name, path_col, raster_method_key in calibrated_patch_specs:
        raster_thr = next((row["threshold"] for row in result_rows if row["method_key"] == raster_method_key), None)
        if raster_thr is None or path_col not in master.columns:
            continue
        scores = []
        y_true = []
        for _, row in master.iterrows():
            prob_path = _Path(row[path_col])
            px_count = _pixel_count_from_prob_path(prob_path, float(raster_thr), int(row["label_binary"]))
            if px_count is None:
                continue
            scores.append(float(px_count))
            y_true.append(int(row["label_binary"]))
        if not scores or len(set(y_true)) < 2:
            continue
        y_true_arr = np.asarray(y_true, dtype=int)
        scores_arr = np.asarray(scores, dtype=float)
        best, curves = _calibrate_patch_px_thresholds(y_true_arr, scores_arr)
        result_rows.append(
            {
                "method_key": method_key,
                "method": display_name,
                "threshold_kind": "patch_px",
                "score_col": method_key,
                "threshold": round(best["threshold"], 4),
                "f1": round(best["f1"], 4),
                "precision": round(best["precision"], 4),
                "recall": round(best["recall"], 4),
                "n_valid": int(len(scores_arr)),
                "n_positive": int(y_true_arr.sum()),
                "n_negative": int(len(scores_arr) - y_true_arr.sum()),
            }
        )
        curve_rows.extend({"method_key": method_key, **row} for row in curves)

    if not result_rows:
        raise ValueError("No se pudo calibrar ningún método continuo.")

    thresholds_df = pd.DataFrame(result_rows).sort_values(["threshold_kind", "method_key"]).reset_index(drop=True)
    curves_df = pd.DataFrame(curve_rows)
    thresholds_df.to_csv(THRESHOLDS_PATH, index=False)
    curves_df.to_csv(EVAL_THRESHOLDS_OUT / "threshold_curves.csv", index=False)

    lines = [
        "# Threshold Calibration Summary",
        "",
        "Fuente de calibración: subset `calibration_dev`.",
        "",
        "## Resultados",
        "",
        thresholds_df.to_string(index=False),
        "",
    ]
    (EVAL_THRESHOLDS_OUT / "threshold_calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Thresholds guardados: {THRESHOLDS_PATH}")
    print(f"Curvas guardadas: {EVAL_THRESHOLDS_OUT / 'threshold_curves.csv'}")
    print(thresholds_df.to_string(index=False))


if __name__ == "__main__":
    main()
