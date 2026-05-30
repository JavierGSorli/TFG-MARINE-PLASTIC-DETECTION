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
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
    EXTERNAL_B09_ZERO_OUT,
    GROUPKFOLD_FOLDS_PATH,
    HYBRID_MASKS_ROOT,
    HYBRID_PHASE_OUT,
    INDICES_NO_WATER_OUT,
    PATCHES_DIR,
    PREDICTIONS_MASTER_PATH,
    RF_MODE_DIRS,
    THRESHOLDS_PATH,
    UNET_CALIBRATED_MASKS_OUT,
    UNET_OUT,
)
from src.evaluation.evaluation_split_utils import build_patch_subset_map, load_grouped_eval_folds


VARIANT_DESCRIPTIONS = {
    "hybrid_simple": "UNet argmax",
    "hybrid_robust": "UNet argmax AND RF full",
    "hybrid_sensitive": "UNet calibrated OR RF full",
    "hybrid_or_no_texture": "UNet calibrated OR RF no_texture",
    "hybrid_or_indices_only": "UNet calibrated OR RF indices_only",
    "hybrid_or_bands_only": "UNet calibrated OR RF bands_only",
    "hybrid_union_argmax_no_texture": "UNet argmax OR RF no_texture",
    "hybrid_and_argmax_no_texture": "UNet argmax AND RF no_texture",
    "hybrid_majority_3": "Majority vote of UNet calibrated, RF full, RF no_texture",
    "hybrid_or_full_cal": "UNet calibrated OR RF full calibrated",
    "hybrid_or_no_texture_cal": "UNet calibrated OR RF no_texture calibrated",
    "hybrid_or_indices_cal": "UNet calibrated OR RF indices_only calibrated",
    "hybrid_or_bands_cal": "UNet calibrated OR RF bands_only calibrated",
    "hybrid_and_full_cal": "UNet argmax AND RF full calibrated",
    "hybrid_and_no_texture_cal": "UNet argmax AND RF no_texture calibrated",
    "hybrid_and_indices_cal": "UNet argmax AND RF indices_only calibrated",
    "hybrid_and_bands_cal": "UNet argmax AND RF bands_only calibrated",
    "hybrid_majority_cal_3": "Majority vote of UNet calibrated, RF full calibrated, RF no_texture calibrated",
    "hybrid_union_cal_3": "Union of UNet calibrated, RF full calibrated, RF no_texture calibrated",
    "hybrid_intersection_cal_3": "Intersection of UNet calibrated, RF full calibrated, RF no_texture calibrated",
    "hybrid_majority_mixed_4": "Majority vote of UNet argmax, UNet calibrated, RF full, RF full calibrated",
    "hybrid_union_cal_4": "Union of UNet calibrated, RF full calibrated, RF no_texture calibrated, RF indices_only calibrated",
    "hybrid_majority_cal_4": "Majority vote of UNet calibrated, RF full calibrated, RF no_texture calibrated, RF indices_only calibrated",
    "hybrid_intersection_cal_4": "Intersection of UNet calibrated, RF full calibrated, RF no_texture calibrated, RF indices_only calibrated",
    "hybrid_union_rf_cal_4": "Union of UNet calibrated and all four RF calibrated masks",
    "hybrid_majority_rf_cal_5": "Majority vote of UNet calibrated plus all four RF calibrated masks",
    "hybrid_intersection_rf_cal_5": "Intersection of UNet calibrated plus all four RF calibrated masks",
    "hybrid_union_default_rf_4": "Union of UNet calibrated, RF full, RF no_texture, RF indices_only",
    "hybrid_majority_default_rf_4": "Majority vote of UNet calibrated, RF full, RF no_texture, RF indices_only",
    "hybrid_or_indices_default_and_bands_cal": "UNet calibrated OR RF indices_only OR RF bands_only calibrated",
    "hybrid_majority_cross_5": "Majority vote of UNet argmax, UNet calibrated, RF full calibrated, RF no_texture calibrated, RF indices_only calibrated",
    "hybrid_or_fdi_sensitive": "UNet calibrated OR RF full OR FDI",
    "hybrid_union_fdi_rf_cal_4": "Union of UNet calibrated, RF full calibrated, RF no_texture calibrated and FDI",
    "hybrid_majority_fdi_rf_cal_4": "Majority vote of UNet calibrated, RF full calibrated, RF no_texture calibrated and FDI",
    "hybrid_or_ext_copy_cal": "UNet calibrated OR RF full calibrated OR External b09_copy_b8a calibrated",
    "hybrid_or_ext_interp_cal": "UNet calibrated OR RF full calibrated OR External b09_interpolate_b8a_b11 calibrated",
    "hybrid_majority_ext_copy_rf_cal_4": "Majority vote of UNet calibrated, RF full calibrated, RF no_texture calibrated and External b09_copy_b8a calibrated",
    "hybrid_majority_ext_interp_rf_cal_4": "Majority vote of UNet calibrated, RF full calibrated, RF no_texture calibrated and External b09_interpolate_b8a_b11 calibrated",
}
HYBRID_METHODS = list(VARIANT_DESCRIPTIONS.keys())
MASK_DIRS = {method_name: HYBRID_MASKS_ROOT / method_name.replace("hybrid_", "") for method_name in HYBRID_METHODS}
PROFILE_VARIANT_DIRS = {
    "sensitive": HYBRID_MASKS_ROOT / "profile_sensitive",
    "balanced": HYBRID_MASKS_ROOT / "profile_balanced",
    "conservative": HYBRID_MASKS_ROOT / "profile_conservative",
}
DETECTOR_CANDIDATES = [
    {"method": "UNet argmax", "score_col": "unet_argmax_px", "pred_col": "unet_argmax_px", "pred_rule": "px_threshold"},
    {"method": "UNet calibrated", "score_col": "unet_thr_px", "pred_col": "unet_thr_pred", "pred_rule": "binary_col"},
    {"method": "ResNet default", "score_col": "resnet_prob", "pred_col": "resnet_default_pred", "pred_rule": "binary_col"},
    {"method": "ResNet calibrated", "score_col": "resnet_prob", "pred_col": "resnet_thr_pred", "pred_rule": "binary_col"},
    {"method": "SAM binary", "score_col": "sam_binary_px", "pred_col": "sam_binary_px", "pred_rule": "px_threshold"},
    {"method": "SAM calibrated", "score_col": "sam_thr_px", "pred_col": "sam_thr_pred", "pred_rule": "binary_col"},
    {"method": "RF full", "score_col": "rf_full_px", "pred_col": "rf_full_px", "pred_rule": "px_threshold"},
    {"method": "RF full calibrated", "score_col": "rf_full_thr_px", "pred_col": "rf_full_thr_pred", "pred_rule": "binary_col"},
    {"method": "RF no_texture", "score_col": "rf_no_texture_px", "pred_col": "rf_no_texture_px", "pred_rule": "px_threshold"},
    {"method": "RF no_texture calibrated", "score_col": "rf_no_texture_thr_px", "pred_col": "rf_no_texture_thr_pred", "pred_rule": "binary_col"},
    {"method": "RF indices_only", "score_col": "rf_indices_only_px", "pred_col": "rf_indices_only_px", "pred_rule": "px_threshold"},
    {"method": "RF indices_only calibrated", "score_col": "rf_indices_only_thr_px", "pred_col": "rf_indices_only_thr_pred", "pred_rule": "binary_col"},
    {"method": "RF bands_only", "score_col": "rf_bands_only_px", "pred_col": "rf_bands_only_px", "pred_rule": "px_threshold"},
    {"method": "RF bands_only calibrated", "score_col": "rf_bands_only_thr_px", "pred_col": "rf_bands_only_thr_pred", "pred_rule": "binary_col"},
    {"method": "FDI", "score_col": "fdi_no_water_px", "pred_col": "fdi_no_water_px", "pred_rule": "px_threshold"},
    {"method": "NDVI", "score_col": "ndvi_no_water_px", "pred_col": "ndvi_no_water_px", "pred_rule": "px_threshold"},
    {"method": "FDI+NDVI", "score_col": "fdi_ndvi_no_water_px", "pred_col": "fdi_ndvi_no_water_px", "pred_rule": "px_threshold"},
    {"method": "FDI_mask", "score_col": "fdi_water_px", "pred_col": "fdi_water_px", "pred_rule": "px_threshold"},
    {"method": "NDVI_mask", "score_col": "ndvi_water_px", "pred_col": "ndvi_water_px", "pred_rule": "px_threshold"},
    {"method": "FDI+NDVI_mask", "score_col": "fdi_ndvi_water_px", "pred_col": "fdi_ndvi_water_px", "pred_rule": "px_threshold"},
    {"method": "External b09_zero default", "score_col": "external_b09_zero_default_px", "pred_col": "external_b09_zero_default_px", "pred_rule": "px_threshold"},
    {"method": "External b09_zero calibrated", "score_col": "external_b09_zero_thr_px", "pred_col": "external_b09_zero_thr_pred", "pred_rule": "binary_col"},
    {"method": "External b09_copy_b8a default", "score_col": "external_b09_copy_b8a_default_px", "pred_col": "external_b09_copy_b8a_default_px", "pred_rule": "px_threshold"},
    {"method": "External b09_copy_b8a calibrated", "score_col": "external_b09_copy_b8a_thr_px", "pred_col": "external_b09_copy_b8a_thr_pred", "pred_rule": "binary_col"},
    {"method": "External b09_interpolate_b8a_b11 default", "score_col": "external_b09_interpolate_b8a_b11_default_px", "pred_col": "external_b09_interpolate_b8a_b11_default_px", "pred_rule": "px_threshold"},
    {"method": "External b09_interpolate_b8a_b11 calibrated", "score_col": "external_b09_interpolate_b8a_b11_thr_px", "pred_col": "external_b09_interpolate_b8a_b11_thr_pred", "pred_rule": "binary_col"},
]


def _read_mask(path: Path, mode: str) -> tuple[np.ndarray | None, dict | None]:
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    if mode == "debris_class":
        mask = arr == DEBRIS_CLASS
    else:
        mask = arr > 0
    return mask.astype(bool), profile


def _align_mask(mask: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape == target_shape:
        return mask
    h = min(mask.shape[0], target_shape[0])
    w = min(mask.shape[1], target_shape[1])
    out = np.zeros(target_shape, dtype=bool)
    out[:h, :w] = mask[:h, :w]
    return out


def _write_binary_mask(path: Path, profile: dict, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    try:
        return round(float(roc_auc_score(y_true, y_score)), 6)
    except Exception:
        return None


def _patch_metrics_from_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_accuracy = (
        ((tp / (tp + fn)) if (tp + fn) > 0 else 0.0) +
        ((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    ) / 2.0
    return {
        "n": int(len(y_true)),
        "auc": _safe_auc(y_true, y_score) if y_score is not None else None,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _pixel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
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
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "dice_f1": round(dice, 6),
        "iou": round(iou, 6),
        "gt_px": int(gt.sum()),
        "pred_px": int(pred.sum()),
    }


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
        "tolerant_precision": round(precision, 6),
        "tolerant_recall": round(recall, 6),
        "tolerant_f1": round(f1, 6),
        "tolerant_iou": round(iou, 6),
        "mean_distance_to_gt": round(mean_distance, 6) if np.isfinite(mean_distance) else np.nan,
        "p95_distance_to_gt": round(p95_distance, 6) if np.isfinite(p95_distance) else np.nan,
    }


def _aggregate_pixel_summary(by_patch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_name, group in by_patch_df.groupby("method"):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        dice = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        rows.append(
            {
                "method": method_name,
                "n_patches": int(len(group)),
                "tp_total": tp,
                "fp_total": fp,
                "fn_total": fn,
                "tn_total": tn,
                "micro_precision": round(precision, 6),
                "micro_recall": round(recall, 6),
                "micro_dice_f1": round(dice, 6),
                "micro_iou": round(iou, 6),
                "mean_iou": round(float(group["iou"].mean()), 6),
                "mean_dice": round(float(group["dice_f1"].mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values("micro_dice_f1", ascending=False)


def _aggregate_tolerant_summary(by_patch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method_name, radius), group in by_patch_df.groupby(["method", "radius_px"]):
        rows.append(
            {
                "method": method_name,
                "radius_px": int(radius),
                "n_patches": int(len(group)),
                "mean_tolerant_precision": round(float(group["tolerant_precision"].mean()), 6),
                "mean_tolerant_recall": round(float(group["tolerant_recall"].mean()), 6),
                "mean_tolerant_f1": round(float(group["tolerant_f1"].mean()), 6),
                "mean_tolerant_iou": round(float(group["tolerant_iou"].mean()), 6),
                "mean_distance_to_gt": round(float(group["mean_distance_to_gt"].mean()), 6),
                "mean_p95_distance_to_gt": round(float(group["p95_distance_to_gt"].mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["radius_px", "mean_tolerant_f1"], ascending=[True, False])


def _detector_predictions(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if spec["pred_col"] not in df.columns:
        return None
    y_true = pd.to_numeric(df["label_binary"], errors="coerce")
    if spec["pred_rule"] == "px_threshold":
        pred_series = pd.to_numeric(df[spec["pred_col"]], errors="coerce").fillna(0)
        valid = y_true.notna()
        thresholds = _threshold_map()
        thr = float(thresholds.get(spec["score_col"], 1.0))
        y_pred = (pred_series[valid] >= thr).astype(int).to_numpy()
        scores = (
            pd.to_numeric(df.loc[valid, spec["score_col"]], errors="coerce").fillna(0).to_numpy(dtype=float)
            if spec["score_col"] in df.columns
            else np.zeros(int(valid.sum()), dtype=float)
        )
        return y_true[valid].astype(int).to_numpy(), y_pred, scores

    valid = y_true.notna() & df[spec["pred_col"]].notna()
    if not valid.any():
        return None
    y_pred = pd.to_numeric(df.loc[valid, spec["pred_col"]], errors="coerce").fillna(0).astype(int).to_numpy()
    scores = (
        pd.to_numeric(df.loc[valid, spec["score_col"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        if spec["score_col"] in df.columns
        else np.zeros(int(valid.sum()), dtype=float)
    )
    return y_true[valid].astype(int).to_numpy(), y_pred, scores


def _threshold_map() -> dict[str, float]:
    if not THRESHOLDS_PATH.exists():
        return {}
    df = pd.read_csv(THRESHOLDS_PATH)
    return dict(zip(df["method_key"].astype(str), pd.to_numeric(df["threshold"], errors="coerce")))


def _load_inner_fold_assignments() -> tuple[dict[int, set[str]], dict[str, int]]:
    folds = load_grouped_eval_folds(GROUPKFOLD_FOLDS_PATH)
    selection_dev = folds[folds["subset"] == "selection_dev"].copy()
    selection_dev["fold"] = pd.to_numeric(selection_dev["fold"], errors="coerce")
    selection_dev = selection_dev[selection_dev["fold"].notna()].copy()
    selection_dev["fold"] = selection_dev["fold"].astype(int)

    val_patch_sets: dict[int, set[str]] = {}
    patch_to_val_fold: dict[str, int] = {}
    for fold, group in selection_dev.groupby("fold"):
        val_patches = set(group.loc[group["split"] == "val", "patch"].astype(str))
        if not val_patches:
            continue
        val_patch_sets[int(fold)] = val_patches
        for patch in val_patches:
            patch_to_val_fold[patch] = int(fold)
    if not val_patch_sets:
        raise RuntimeError(f"No hay folds internos válidos en {GROUPKFOLD_FOLDS_PATH}")
    return val_patch_sets, patch_to_val_fold


def _select_best_detector(df: pd.DataFrame, val_patch_sets: dict[int, set[str]]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    selection_df = df[df["eval_subset"] == "selection_dev"].copy()
    by_fold_rows = []
    for spec in DETECTOR_CANDIDATES:
        for fold, val_patches in sorted(val_patch_sets.items()):
            fold_df = selection_df[selection_df["patch"].astype(str).isin(val_patches)].copy()
            result = _detector_predictions(fold_df, spec)
            if result is None:
                continue
            y_true, y_pred, scores = result
            if len(y_true) == 0:
                continue
            metrics = _patch_metrics_from_binary(y_true, y_pred, scores)
            by_fold_rows.append(
                {
                    "fold": int(fold),
                    "method": spec["method"],
                    "score_col": spec["score_col"],
                    "pred_col": spec["pred_col"],
                    "pred_rule": spec["pred_rule"],
                    **metrics,
                }
            )
    if not by_fold_rows:
        raise RuntimeError("No hay detectores candidatos válidos en los folds de selection_dev para construir el híbrido.")

    by_fold_df = pd.DataFrame(by_fold_rows).sort_values(["method", "fold"]).reset_index(drop=True)
    metrics_df = (
        by_fold_df.groupby(["method", "score_col", "pred_col", "pred_rule"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            n=("n", "sum"),
            auc=("auc", "mean"),
            auc_std=("auc", "std"),
            precision=("precision", "mean"),
            precision_std=("precision", "std"),
            recall=("recall", "mean"),
            recall_std=("recall", "std"),
            f1=("f1", "mean"),
            f1_std=("f1", "std"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            tn=("tn", "sum"),
            fn=("fn", "sum"),
        )
    )
    metrics_df["auc_sort"] = metrics_df["auc"].fillna(-1.0)
    metrics_df = metrics_df.sort_values(
        ["f1", "balanced_accuracy", "auc_sort", "precision", "recall", "method"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    best = metrics_df.iloc[0]
    chosen = {
        "method": str(best["method"]),
        "score_col": str(best["score_col"]),
        "pred_col": str(best["pred_col"]),
        "pred_rule": str(best["pred_rule"]),
    }
    return chosen, metrics_df.drop(columns=["auc_sort"]), by_fold_df


def _px_pred(px_value: object, method_key: str, threshold_map: dict[str, float]) -> int:
    if pd.isna(px_value):
        return 0
    return int(float(px_value) >= float(threshold_map.get(method_key, 1.0)))


def _majority_vote(*masks: np.ndarray) -> np.ndarray:
    stacked = np.stack([mask.astype(np.uint8) for mask in masks], axis=0)
    needed = (stacked.shape[0] // 2) + 1
    return stacked.sum(axis=0) >= needed


def _build_variant_masks(
    unet_argmax: np.ndarray,
    unet_calibrated: np.ndarray,
    rf_full: np.ndarray,
    rf_no_texture: np.ndarray,
    rf_indices_only: np.ndarray,
    rf_bands_only: np.ndarray,
    rf_full_calibrated: np.ndarray,
    rf_no_texture_calibrated: np.ndarray,
    rf_indices_only_calibrated: np.ndarray,
    rf_bands_only_calibrated: np.ndarray,
    fdi: np.ndarray,
    external_zero_calibrated: np.ndarray,
    external_copy_calibrated: np.ndarray,
    external_interp_calibrated: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "hybrid_simple": unet_argmax.copy(),
        "hybrid_robust": np.logical_and(unet_argmax, rf_full),
        "hybrid_sensitive": np.logical_or(unet_calibrated, rf_full),
        "hybrid_or_no_texture": np.logical_or(unet_calibrated, rf_no_texture),
        "hybrid_or_indices_only": np.logical_or(unet_calibrated, rf_indices_only),
        "hybrid_or_bands_only": np.logical_or(unet_calibrated, rf_bands_only),
        "hybrid_union_argmax_no_texture": np.logical_or(unet_argmax, rf_no_texture),
        "hybrid_and_argmax_no_texture": np.logical_and(unet_argmax, rf_no_texture),
        "hybrid_majority_3": _majority_vote(unet_calibrated, rf_full, rf_no_texture),
        "hybrid_or_full_cal": np.logical_or(unet_calibrated, rf_full_calibrated),
        "hybrid_or_no_texture_cal": np.logical_or(unet_calibrated, rf_no_texture_calibrated),
        "hybrid_or_indices_cal": np.logical_or(unet_calibrated, rf_indices_only_calibrated),
        "hybrid_or_bands_cal": np.logical_or(unet_calibrated, rf_bands_only_calibrated),
        "hybrid_and_full_cal": np.logical_and(unet_argmax, rf_full_calibrated),
        "hybrid_and_no_texture_cal": np.logical_and(unet_argmax, rf_no_texture_calibrated),
        "hybrid_and_indices_cal": np.logical_and(unet_argmax, rf_indices_only_calibrated),
        "hybrid_and_bands_cal": np.logical_and(unet_argmax, rf_bands_only_calibrated),
        "hybrid_majority_cal_3": _majority_vote(unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated),
        "hybrid_union_cal_3": np.logical_or(np.logical_or(unet_calibrated, rf_full_calibrated), rf_no_texture_calibrated),
        "hybrid_intersection_cal_3": np.logical_and(np.logical_and(unet_calibrated, rf_full_calibrated), rf_no_texture_calibrated),
        "hybrid_majority_mixed_4": _majority_vote(unet_argmax, unet_calibrated, rf_full, rf_full_calibrated),
        "hybrid_union_cal_4": np.logical_or.reduce(
            [unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, rf_indices_only_calibrated]
        ),
        "hybrid_majority_cal_4": _majority_vote(
            unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, rf_indices_only_calibrated
        ),
        "hybrid_intersection_cal_4": np.logical_and.reduce(
            [unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, rf_indices_only_calibrated]
        ),
        "hybrid_union_rf_cal_4": np.logical_or.reduce(
            [unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, rf_indices_only_calibrated, rf_bands_only_calibrated]
        ),
        "hybrid_majority_rf_cal_5": _majority_vote(
            unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, rf_indices_only_calibrated, rf_bands_only_calibrated
        ),
        "hybrid_intersection_rf_cal_5": np.logical_and.reduce(
            [unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, rf_indices_only_calibrated, rf_bands_only_calibrated]
        ),
        "hybrid_union_default_rf_4": np.logical_or.reduce([unet_calibrated, rf_full, rf_no_texture, rf_indices_only]),
        "hybrid_majority_default_rf_4": _majority_vote(unet_calibrated, rf_full, rf_no_texture, rf_indices_only),
        "hybrid_or_indices_default_and_bands_cal": np.logical_or.reduce(
            [unet_calibrated, rf_indices_only, rf_bands_only_calibrated]
        ),
        "hybrid_majority_cross_5": _majority_vote(
            unet_argmax, unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, rf_indices_only_calibrated
        ),
        "hybrid_or_fdi_sensitive": np.logical_or.reduce([unet_calibrated, rf_full, fdi]),
        "hybrid_union_fdi_rf_cal_4": np.logical_or.reduce(
            [unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, fdi]
        ),
        "hybrid_majority_fdi_rf_cal_4": _majority_vote(
            unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, fdi
        ),
        "hybrid_or_ext_copy_cal": np.logical_or.reduce(
            [unet_calibrated, rf_full_calibrated, external_copy_calibrated]
        ),
        "hybrid_or_ext_interp_cal": np.logical_or.reduce(
            [unet_calibrated, rf_full_calibrated, external_interp_calibrated]
        ),
        "hybrid_majority_ext_copy_rf_cal_4": _majority_vote(
            unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, external_copy_calibrated
        ),
        "hybrid_majority_ext_interp_rf_cal_4": _majority_vote(
            unet_calibrated, rf_full_calibrated, rf_no_texture_calibrated, external_interp_calibrated
        ),
    }


def _pick_profile_variant(
    selection_df: pd.DataFrame,
    profile: str,
) -> str:
    if selection_df.empty:
        raise RuntimeError("No hay combinaciones de máscara disponibles para seleccionar perfiles.")

    df = selection_df.copy()
    if profile == "sensitive":
        eligible = df[
            (df["pixel_micro_precision"].fillna(0.0) >= 0.20)
            & (df["tolerant_precision_r3"].fillna(0.0) >= 0.45)
        ].copy()
        if eligible.empty:
            eligible = df.copy()
        ordered = eligible.sort_values(
            ["tolerant_recall_r3", "tolerant_f1_r3", "pixel_micro_dice_f1", "method"],
            ascending=[False, False, False, True],
        )
    elif profile == "conservative":
        eligible = df[
            (df["pixel_micro_recall"].fillna(0.0) >= 0.18)
            & (df["tolerant_recall_r3"].fillna(0.0) >= 0.25)
        ].copy()
        if eligible.empty:
            eligible = df.copy()
        ordered = eligible.sort_values(
            ["pixel_micro_precision", "tolerant_precision_r3", "tolerant_f1_r3", "method"],
            ascending=[False, False, False, True],
        )
    elif profile == "balanced":
        ordered = df.sort_values(
            ["pixel_micro_dice_f1", "tolerant_f1_r3", "pixel_mean_dice", "method"],
            ascending=[False, False, False, True],
        )
    else:
        raise ValueError(f"Perfil no soportado: {profile}")
    return str(ordered.iloc[0]["method"])


def _select_best_mask_combination(
    train_pixel_df: pd.DataFrame,
    train_tolerant_df: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_pixel_df.empty:
        raise RuntimeError("No hay patches positivos en selection_dev con GT válida para seleccionar la combinación de máscaras.")

    pixel_fold_rows = []
    for (fold, method_name), group in train_pixel_df.groupby(["fold", "method"]):
        summary = _aggregate_pixel_summary(group)
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row["fold"] = int(fold)
        row["method"] = method_name
        pixel_fold_rows.append(row)
    pixel_by_fold_df = pd.DataFrame(pixel_fold_rows)
    pixel_summary_df = (
        pixel_by_fold_df.groupby("method", as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_patches=("n_patches", "sum"),
            tp_total=("tp_total", "sum"),
            fp_total=("fp_total", "sum"),
            fn_total=("fn_total", "sum"),
            tn_total=("tn_total", "sum"),
            micro_precision=("micro_precision", "mean"),
            micro_precision_std=("micro_precision", "std"),
            micro_recall=("micro_recall", "mean"),
            micro_recall_std=("micro_recall", "std"),
            micro_dice_f1=("micro_dice_f1", "mean"),
            micro_dice_f1_std=("micro_dice_f1", "std"),
            micro_iou=("micro_iou", "mean"),
            micro_iou_std=("micro_iou", "std"),
            mean_iou=("mean_iou", "mean"),
            mean_iou_std=("mean_iou", "std"),
            mean_dice=("mean_dice", "mean"),
            mean_dice_std=("mean_dice", "std"),
        )
        .sort_values("micro_dice_f1", ascending=False)
        .reset_index(drop=True)
    )

    tolerant_fold_rows = []
    for (fold, method_name, radius), group in train_tolerant_df.groupby(["fold", "method", "radius_px"]):
        row = {
            "fold": int(fold),
            "method": method_name,
            "radius_px": int(radius),
            "n_patches": int(len(group)),
            "mean_tolerant_precision": round(float(group["tolerant_precision"].mean()), 6),
            "mean_tolerant_recall": round(float(group["tolerant_recall"].mean()), 6),
            "mean_tolerant_f1": round(float(group["tolerant_f1"].mean()), 6),
            "mean_tolerant_iou": round(float(group["tolerant_iou"].mean()), 6),
            "mean_distance_to_gt": round(float(group["mean_distance_to_gt"].mean()), 6),
            "mean_p95_distance_to_gt": round(float(group["p95_distance_to_gt"].mean()), 6),
        }
        tolerant_fold_rows.append(row)
    tolerant_by_fold_df = pd.DataFrame(tolerant_fold_rows)
    tolerant_summary_df = (
        tolerant_by_fold_df.groupby(["method", "radius_px"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_patches=("n_patches", "sum"),
            mean_tolerant_precision=("mean_tolerant_precision", "mean"),
            mean_tolerant_precision_std=("mean_tolerant_precision", "std"),
            mean_tolerant_recall=("mean_tolerant_recall", "mean"),
            mean_tolerant_recall_std=("mean_tolerant_recall", "std"),
            mean_tolerant_f1=("mean_tolerant_f1", "mean"),
            mean_tolerant_f1_std=("mean_tolerant_f1", "std"),
            mean_tolerant_iou=("mean_tolerant_iou", "mean"),
            mean_tolerant_iou_std=("mean_tolerant_iou", "std"),
            mean_distance_to_gt=("mean_distance_to_gt", "mean"),
            mean_p95_distance_to_gt=("mean_p95_distance_to_gt", "mean"),
        )
        .sort_values(["radius_px", "mean_tolerant_f1"], ascending=[True, False])
        .reset_index(drop=True)
    )

    rows = []
    for method_name in HYBRID_METHODS:
        pixel_row = pixel_summary_df.loc[pixel_summary_df["method"] == method_name]
        if pixel_row.empty:
            continue
        tolerant_row = tolerant_summary_df.loc[
            (tolerant_summary_df["method"] == method_name) & (tolerant_summary_df["radius_px"] == 3)
        ]
        rows.append(
            {
                "method": method_name,
                "description": VARIANT_DESCRIPTIONS[method_name],
                "pixel_micro_dice_f1": float(pixel_row.iloc[0]["micro_dice_f1"]),
                "pixel_micro_iou": float(pixel_row.iloc[0]["micro_iou"]),
                "pixel_micro_precision": float(pixel_row.iloc[0]["micro_precision"]),
                "pixel_micro_recall": float(pixel_row.iloc[0]["micro_recall"]),
                "pixel_mean_dice": float(pixel_row.iloc[0]["mean_dice"]),
                "pixel_mean_iou": float(pixel_row.iloc[0]["mean_iou"]),
                "tolerant_precision_r3": float(tolerant_row.iloc[0]["mean_tolerant_precision"]) if not tolerant_row.empty else np.nan,
                "tolerant_recall_r3": float(tolerant_row.iloc[0]["mean_tolerant_recall"]) if not tolerant_row.empty else np.nan,
                "tolerant_f1_r3": float(tolerant_row.iloc[0]["mean_tolerant_f1"]) if not tolerant_row.empty else np.nan,
                "tolerant_iou_r3": float(tolerant_row.iloc[0]["mean_tolerant_iou"]) if not tolerant_row.empty else np.nan,
            }
        )
    selection_df = pd.DataFrame(rows).sort_values(
        ["pixel_micro_dice_f1", "tolerant_f1_r3", "pixel_mean_dice", "method"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selection_df.insert(0, "rank", range(1, len(selection_df) + 1))
    profile_variants = {
        "sensitive": _pick_profile_variant(selection_df, "sensitive"),
        "balanced": _pick_profile_variant(selection_df, "balanced"),
        "conservative": _pick_profile_variant(selection_df, "conservative"),
    }
    return profile_variants, selection_df, pixel_summary_df, tolerant_summary_df, pixel_by_fold_df, tolerant_by_fold_df


def main() -> None:
    HYBRID_PHASE_OUT.mkdir(parents=True, exist_ok=True)
    for path in MASK_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATASET_METADATA_GROUPED_PATH)
    val_patch_sets, patch_to_val_fold = _load_inner_fold_assignments()
    subset_map = build_patch_subset_map()
    master = pd.read_csv(PREDICTIONS_MASTER_PATH)
    df = metadata.merge(subset_map, on="patch", how="left").merge(master, on="patch", how="left", suffixes=("", "_master"))
    if "label_x" in df.columns:
        df["label"] = df["label_x"]
        df = df.drop(columns=["label_x", "label_y"], errors="ignore")

    detector_spec, detector_selection_df, detector_selection_by_fold_df = _select_best_detector(df, val_patch_sets)
    detector_selection_df.to_csv(HYBRID_PHASE_OUT / "detector_selection_train_val.csv", index=False)
    detector_selection_by_fold_df.to_csv(HYBRID_PHASE_OUT / "detector_selection_by_fold.csv", index=False)
    threshold_map = _threshold_map()

    patch_rows: list[dict] = []
    pixel_rows: list[dict] = []
    tolerant_rows: list[dict] = []
    train_combo_pixel_rows: list[dict] = []
    train_combo_tolerant_rows: list[dict] = []

    for _, row in df.iterrows():
        patch_name = str(row["patch"])
        stem = Path(patch_name).stem
        detector_score = float(row.get(detector_spec["score_col"], 0.0) or 0.0)
        detector_raw = row.get(detector_spec["pred_col"], 0)
        if pd.isna(detector_raw):
            detector_raw = 0
        if detector_spec["pred_rule"] == "px_threshold":
            detector_pred = _px_pred(detector_raw, detector_spec["score_col"], threshold_map)
        else:
            detector_pred = int(detector_raw)
        label_binary = int(row.get("label_binary", 1 if "_SI_" in patch_name else 0))
        eval_subset = row.get("eval_subset", "")

        unet_argmax, unet_profile = _read_mask(UNET_OUT / f"{stem}_mask.tif", "debris_class")
        unet_calibrated, unet_cal_profile = _read_mask(UNET_CALIBRATED_MASKS_OUT / f"{stem}_mask.tif", "binary")
        rf_full, rf_profile = _read_mask(RF_MODE_DIRS["full"] / f"{stem}_mask.tif", "debris_class")
        rf_no_texture, _ = _read_mask(RF_MODE_DIRS["no_texture"] / f"{stem}_mask.tif", "debris_class")
        rf_indices_only, _ = _read_mask(RF_MODE_DIRS["indices_only"] / f"{stem}_mask.tif", "debris_class")
        rf_bands_only, _ = _read_mask(RF_MODE_DIRS["bands_only"] / f"{stem}_mask.tif", "debris_class")
        rf_full_calibrated, _ = _read_mask(RF_MODE_DIRS["full"] / "calibrated_masks" / f"{stem}_mask.tif", "binary")
        rf_no_texture_calibrated, _ = _read_mask(RF_MODE_DIRS["no_texture"] / "calibrated_masks" / f"{stem}_mask.tif", "binary")
        rf_indices_only_calibrated, _ = _read_mask(RF_MODE_DIRS["indices_only"] / "calibrated_masks" / f"{stem}_mask.tif", "binary")
        rf_bands_only_calibrated, _ = _read_mask(RF_MODE_DIRS["bands_only"] / "calibrated_masks" / f"{stem}_mask.tif", "binary")
        fdi, _ = _read_mask(INDICES_NO_WATER_OUT / f"{stem}_fdi_mask.tif", "binary")
        external_zero_calibrated, _ = _read_mask(EXTERNAL_B09_ZERO_OUT / "calibrated_masks" / f"{stem}_mask.tif", "binary")
        external_copy_calibrated, _ = _read_mask(EXTERNAL_B09_COPY_B8A_OUT / "calibrated_masks" / f"{stem}_mask.tif", "binary")
        external_interp_calibrated, _ = _read_mask(EXTERNAL_B09_INTERP_OUT / "calibrated_masks" / f"{stem}_mask.tif", "binary")
        gt_mask, _ = _read_mask(PATCHES_DIR / f"{stem}_mask.tif", "binary")

        if unet_argmax is None or rf_full is None or rf_no_texture is None or rf_indices_only is None or rf_bands_only is None:
            continue
        base_shape = unet_argmax.shape
        rf_full = _align_mask(rf_full, base_shape)
        rf_no_texture = _align_mask(rf_no_texture, base_shape)
        rf_indices_only = _align_mask(rf_indices_only, base_shape)
        rf_bands_only = _align_mask(rf_bands_only, base_shape)
        rf_full_calibrated = _align_mask(rf_full_calibrated, base_shape) if rf_full_calibrated is not None else np.zeros(base_shape, dtype=bool)
        rf_no_texture_calibrated = _align_mask(rf_no_texture_calibrated, base_shape) if rf_no_texture_calibrated is not None else np.zeros(base_shape, dtype=bool)
        rf_indices_only_calibrated = _align_mask(rf_indices_only_calibrated, base_shape) if rf_indices_only_calibrated is not None else np.zeros(base_shape, dtype=bool)
        rf_bands_only_calibrated = _align_mask(rf_bands_only_calibrated, base_shape) if rf_bands_only_calibrated is not None else np.zeros(base_shape, dtype=bool)
        fdi = _align_mask(fdi, base_shape) if fdi is not None else np.zeros(base_shape, dtype=bool)
        external_zero_calibrated = _align_mask(external_zero_calibrated, base_shape) if external_zero_calibrated is not None else np.zeros(base_shape, dtype=bool)
        external_copy_calibrated = _align_mask(external_copy_calibrated, base_shape) if external_copy_calibrated is not None else np.zeros(base_shape, dtype=bool)
        external_interp_calibrated = _align_mask(external_interp_calibrated, base_shape) if external_interp_calibrated is not None else np.zeros(base_shape, dtype=bool)
        if unet_calibrated is not None:
            unet_calibrated = _align_mask(unet_calibrated, base_shape)
        else:
            unet_calibrated = np.zeros(base_shape, dtype=bool)
        if gt_mask is not None:
            gt_mask = _align_mask(gt_mask, base_shape)

        profile = unet_profile or rf_profile or unet_cal_profile
        if profile is None:
            continue

        raw_outputs = _build_variant_masks(
            unet_argmax,
            unet_calibrated,
            rf_full,
            rf_no_texture,
            rf_indices_only,
            rf_bands_only,
            rf_full_calibrated,
            rf_no_texture_calibrated,
            rf_indices_only_calibrated,
            rf_bands_only_calibrated,
            fdi,
            external_zero_calibrated,
            external_copy_calibrated,
            external_interp_calibrated,
        )

        val_fold = patch_to_val_fold.get(patch_name)
        if gt_mask is not None and label_binary == 1 and eval_subset == "selection_dev" and val_fold is not None:
            for method_name, mask in raw_outputs.items():
                train_combo_pixel_rows.append(
                    {"patch": patch_name, "fold": int(val_fold), "method": method_name, **_pixel_metrics(gt_mask.astype(np.uint8), mask.astype(np.uint8))}
                )
                for radius in (1, 2, 3):
                    train_combo_tolerant_rows.append(
                        {
                            "patch": patch_name,
                            "fold": int(val_fold),
                            "method": method_name,
                            "radius_px": radius,
                            **_tolerant_metrics(gt_mask, mask, radius),
                        }
                    )

        gated_outputs = {
            method_name: (mask.copy() if detector_pred == 1 else np.zeros(base_shape, dtype=bool))
            for method_name, mask in raw_outputs.items()
        }

        for method_name, mask in gated_outputs.items():
            out_dir = MASK_DIRS[method_name]
            _write_binary_mask(out_dir / f"{stem}_mask.tif", profile, mask.astype(np.uint8))

        patch_row = {
            "patch": patch_name,
            "label": row.get("label", ""),
            "label_binary": label_binary,
            "eval_subset": eval_subset,
            "eval_split": row.get("eval_split", ""),
            "date": row.get("date", ""),
            "group_id": row.get("group_id", ""),
            "detector_method": detector_spec["method"],
            "detector_score": round(detector_score, 6),
            "detector_pred": detector_pred,
            "external_detector_pred": detector_pred,
            "unet_argmax_px": int(unet_argmax.sum()),
            "unet_thr_pred": _px_pred(int(unet_calibrated.sum()), "unet_thr_px", threshold_map),
            "unet_thr_px": int(unet_calibrated.sum()),
            "rf_full_px": int(rf_full.sum()),
            "rf_no_texture_px": int(rf_no_texture.sum()),
            "rf_indices_only_px": int(rf_indices_only.sum()),
            "rf_bands_only_px": int(rf_bands_only.sum()),
        }
        for method_name, mask in gated_outputs.items():
            px = int(mask.sum())
            patch_row[f"{method_name}_px"] = px
            patch_row[f"{method_name}_pred"] = int(px > 0)
        patch_rows.append(patch_row)

        if gt_mask is None or label_binary != 1 or eval_subset != "test_final":
            continue

        method_masks = {
            "UNet argmax": unet_argmax,
            "UNet calibrated": unet_calibrated,
            "RF full": rf_full,
            "RF no_texture": rf_no_texture,
            "RF indices_only": rf_indices_only,
            "RF bands_only": rf_bands_only,
            "FDI": fdi,
            "External b09_zero calibrated": external_zero_calibrated,
            "External b09_copy_b8a calibrated": external_copy_calibrated,
            "External b09_interpolate_b8a_b11 calibrated": external_interp_calibrated,
            **gated_outputs,
        }
        for method_name, mask in method_masks.items():
            if mask is None:
                continue
            pixel_rows.append({"patch": patch_name, "method": method_name, **_pixel_metrics(gt_mask.astype(np.uint8), mask.astype(np.uint8))})
            for radius in (1, 2, 3):
                tolerant_rows.append({"patch": patch_name, "method": method_name, "radius_px": radius, **_tolerant_metrics(gt_mask, mask, radius)})

    train_combo_pixel_df = pd.DataFrame(train_combo_pixel_rows)
    train_combo_tolerant_df = pd.DataFrame(train_combo_tolerant_rows)
    (
        profile_variants,
        mask_selection_df,
        train_combo_pixel_summary_df,
        train_combo_tolerant_summary_df,
        train_combo_pixel_by_fold_df,
        train_combo_tolerant_by_fold_df,
    ) = _select_best_mask_combination(
        train_combo_pixel_df,
        train_combo_tolerant_df,
    )
    mask_selection_df.to_csv(HYBRID_PHASE_OUT / "mask_combination_selection_train_val.csv", index=False)
    train_combo_pixel_summary_df.to_csv(HYBRID_PHASE_OUT / "mask_combination_train_val_pixelwise.csv", index=False)
    train_combo_tolerant_summary_df.to_csv(HYBRID_PHASE_OUT / "mask_combination_train_val_tolerant.csv", index=False)
    train_combo_pixel_by_fold_df.to_csv(HYBRID_PHASE_OUT / "mask_combination_by_fold_pixelwise.csv", index=False)
    train_combo_tolerant_by_fold_df.to_csv(HYBRID_PHASE_OUT / "mask_combination_by_fold_tolerant.csv", index=False)

    hybrid_df = pd.DataFrame(patch_rows)
    hybrid_df["selected_mask_variant_sensitive"] = profile_variants["sensitive"]
    hybrid_df["selected_mask_description_sensitive"] = VARIANT_DESCRIPTIONS[profile_variants["sensitive"]]
    hybrid_df["selected_mask_variant_balanced"] = profile_variants["balanced"]
    hybrid_df["selected_mask_description_balanced"] = VARIANT_DESCRIPTIONS[profile_variants["balanced"]]
    hybrid_df["selected_mask_variant_conservative"] = profile_variants["conservative"]
    hybrid_df["selected_mask_description_conservative"] = VARIANT_DESCRIPTIONS[profile_variants["conservative"]]
    hybrid_df["selected_mask_variant"] = profile_variants["balanced"]
    hybrid_df["selected_mask_description"] = VARIANT_DESCRIPTIONS[profile_variants["balanced"]]
    for profile_name, variant_name in profile_variants.items():
        hybrid_df[f"hybrid_profile_{profile_name}_pred"] = (
            pd.to_numeric(hybrid_df.loc[:, f"{variant_name}_pred"], errors="coerce").fillna(0).astype(int).to_numpy()
        )
        hybrid_df[f"hybrid_profile_{profile_name}_px"] = (
            pd.to_numeric(hybrid_df.loc[:, f"{variant_name}_px"], errors="coerce").fillna(0).astype(int).to_numpy()
        )
    hybrid_df["final_hybrid_pred"] = hybrid_df["hybrid_profile_balanced_pred"]
    hybrid_df["final_hybrid_px"] = hybrid_df["hybrid_profile_balanced_px"]
    hybrid_df.to_csv(HYBRID_PHASE_OUT / "hybrid_predictions.csv", index=False)

    for profile_name, variant_name in profile_variants.items():
        profile_dir = PROFILE_VARIANT_DIRS[profile_name]
        profile_dir.mkdir(parents=True, exist_ok=True)
        source_dir = MASK_DIRS[variant_name]
        for mask_path in source_dir.glob("*_mask.tif"):
            target_path = profile_dir / mask_path.name
            if target_path.exists():
                target_path.unlink()
            target_path.write_bytes(mask_path.read_bytes())

    test_df = hybrid_df[hybrid_df["eval_subset"] == "test_final"].copy()
    y_true = test_df["label_binary"].astype(int).to_numpy()
    external_pred = test_df["detector_pred"].astype(int).to_numpy()
    external_score = test_df["detector_score"].astype(float).to_numpy()

    patch_metric_rows = [
        {"method": f"Detector only ({detector_spec['method']})", **_patch_metrics_from_binary(y_true, external_pred, external_score)},
        {"method": "UNet argmax", **_patch_metrics_from_binary(y_true, test_df["unet_argmax_px"].fillna(0).map(lambda v: _px_pred(v, "unet_argmax_px", threshold_map)).astype(int).to_numpy())},
        {"method": "UNet calibrated", **_patch_metrics_from_binary(y_true, test_df["unet_thr_pred"].fillna(0).astype(int).to_numpy())},
        {"method": "RF full", **_patch_metrics_from_binary(y_true, test_df["rf_full_px"].fillna(0).map(lambda v: _px_pred(v, "rf_full_px", threshold_map)).astype(int).to_numpy())},
        {"method": "RF no_texture", **_patch_metrics_from_binary(y_true, test_df["rf_no_texture_px"].fillna(0).map(lambda v: _px_pred(v, "rf_no_texture_px", threshold_map)).astype(int).to_numpy())},
        {"method": "RF indices_only", **_patch_metrics_from_binary(y_true, test_df["rf_indices_only_px"].fillna(0).map(lambda v: _px_pred(v, "rf_indices_only_px", threshold_map)).astype(int).to_numpy())},
        {"method": "RF bands_only", **_patch_metrics_from_binary(y_true, test_df["rf_bands_only_px"].fillna(0).map(lambda v: _px_pred(v, "rf_bands_only_px", threshold_map)).astype(int).to_numpy())},
        {"method": "Hybrid final", **_patch_metrics_from_binary(y_true, test_df["final_hybrid_pred"].astype(int).to_numpy(), external_score)},
        {"method": "Hybrid sensitive profile", **_patch_metrics_from_binary(y_true, test_df["hybrid_profile_sensitive_pred"].astype(int).to_numpy(), external_score)},
        {"method": "Hybrid balanced profile", **_patch_metrics_from_binary(y_true, test_df["hybrid_profile_balanced_pred"].astype(int).to_numpy(), external_score)},
        {"method": "Hybrid conservative profile", **_patch_metrics_from_binary(y_true, test_df["hybrid_profile_conservative_pred"].astype(int).to_numpy(), external_score)},
    ]
    for method_name in HYBRID_METHODS:
        patch_metric_rows.append(
            {"method": method_name, **_patch_metrics_from_binary(y_true, test_df[f"{method_name}_pred"].astype(int).to_numpy(), external_score)}
        )
    patch_metric_df = pd.DataFrame(patch_metric_rows).sort_values("f1", ascending=False)
    patch_metric_df.to_csv(HYBRID_PHASE_OUT / "hybrid_patch_metrics.csv", index=False)

    pixel_by_patch_df = pd.DataFrame(pixel_rows)
    if not pixel_by_patch_df.empty:
        for profile_name, variant_name in profile_variants.items():
            method_mask = pixel_by_patch_df["method"] == variant_name
            if method_mask.any():
                rows = pixel_by_patch_df.loc[method_mask].copy()
                rows["method"] = f"Hybrid {profile_name} profile"
                pixel_by_patch_df = pd.concat([pixel_by_patch_df, rows], ignore_index=True)
        final_method_mask = pixel_by_patch_df["method"] == profile_variants["balanced"]
        if final_method_mask.any():
            final_rows = pixel_by_patch_df.loc[final_method_mask].copy()
            final_rows["method"] = "Hybrid final"
            pixel_by_patch_df = pd.concat([pixel_by_patch_df, final_rows], ignore_index=True)
    pixel_summary_df = _aggregate_pixel_summary(pixel_by_patch_df)
    pixel_by_patch_df.to_csv(HYBRID_PHASE_OUT / "hybrid_pixelwise_by_patch.csv", index=False)
    pixel_summary_df.to_csv(HYBRID_PHASE_OUT / "hybrid_pixelwise_metrics.csv", index=False)

    tolerant_by_patch_df = pd.DataFrame(tolerant_rows)
    if not tolerant_by_patch_df.empty:
        for profile_name, variant_name in profile_variants.items():
            method_mask = tolerant_by_patch_df["method"] == variant_name
            if method_mask.any():
                rows = tolerant_by_patch_df.loc[method_mask].copy()
                rows["method"] = f"Hybrid {profile_name} profile"
                tolerant_by_patch_df = pd.concat([tolerant_by_patch_df, rows], ignore_index=True)
        final_method_mask = tolerant_by_patch_df["method"] == profile_variants["balanced"]
        if final_method_mask.any():
            final_rows = tolerant_by_patch_df.loc[final_method_mask].copy()
            final_rows["method"] = "Hybrid final"
            tolerant_by_patch_df = pd.concat([tolerant_by_patch_df, final_rows], ignore_index=True)
    tolerant_summary_df = _aggregate_tolerant_summary(tolerant_by_patch_df)
    tolerant_by_patch_df.to_csv(HYBRID_PHASE_OUT / "hybrid_tolerant_by_patch.csv", index=False)
    tolerant_summary_df.to_csv(HYBRID_PHASE_OUT / "hybrid_tolerant_metrics.csv", index=False)

    ablation_rows = []
    for method_name in [
        f"Detector only ({detector_spec['method']})",
        "UNet argmax",
        "UNet calibrated",
        "RF full",
        "RF no_texture",
        "RF indices_only",
        "RF bands_only",
        "Hybrid sensitive profile",
        "Hybrid balanced profile",
        "Hybrid conservative profile",
        "Hybrid final",
        *HYBRID_METHODS,
    ]:
        row = {"method": method_name}
        patch_row = patch_metric_df.loc[patch_metric_df["method"] == method_name]
        if not patch_row.empty:
            row["patch_f1"] = patch_row.iloc[0]["f1"]
            row["patch_precision"] = patch_row.iloc[0]["precision"]
            row["patch_recall"] = patch_row.iloc[0]["recall"]
        pixel_row = pixel_summary_df.loc[pixel_summary_df["method"] == method_name]
        if not pixel_row.empty:
            row["pixel_micro_dice_f1"] = pixel_row.iloc[0]["micro_dice_f1"]
            row["pixel_micro_iou"] = pixel_row.iloc[0]["micro_iou"]
        tolerant_row = tolerant_summary_df.loc[(tolerant_summary_df["method"] == method_name) & (tolerant_summary_df["radius_px"] == 3)]
        if not tolerant_row.empty:
            row["tolerant_f1_r3"] = tolerant_row.iloc[0]["mean_tolerant_f1"]
            row["tolerant_iou_r3"] = tolerant_row.iloc[0]["mean_tolerant_iou"]
        if method_name in VARIANT_DESCRIPTIONS:
            row["mask_formula"] = VARIANT_DESCRIPTIONS[method_name]
        elif method_name == "Hybrid sensitive profile":
            row["mask_formula"] = VARIANT_DESCRIPTIONS[profile_variants["sensitive"]]
        elif method_name == "Hybrid balanced profile":
            row["mask_formula"] = VARIANT_DESCRIPTIONS[profile_variants["balanced"]]
        elif method_name == "Hybrid conservative profile":
            row["mask_formula"] = VARIANT_DESCRIPTIONS[profile_variants["conservative"]]
        elif method_name == "Hybrid final":
            row["mask_formula"] = VARIANT_DESCRIPTIONS[profile_variants["balanced"]]
        ablation_rows.append(row)
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(HYBRID_PHASE_OUT / "hybrid_ablation.csv", index=False)

    summary_lines = [
        "# Hybrid detector + segmenter",
        "",
        "## Filosofía",
        "- la selección del detector y la selección de la máscara se hacen por separado",
        "- el detector se elige con los folds de validación interna de `selection_dev` (`folds.csv`) usando métricas patch-level",
        "- la combinación de máscaras se elige con los folds de validación interna de `selection_dev` usando solo segmentación, sin gating del detector",
        "- el híbrido final aplica: detector elegido + combinación de máscaras elegida",
        "",
        "## Detector seleccionado por validación interna en selection_dev",
        f"- detector: `{detector_spec['method']}`",
        f"- score_col: `{detector_spec['score_col']}`",
        f"- pred_col: `{detector_spec['pred_col']}`",
        "- si detector = negativo, la máscara final es vacía",
        "",
        "## Combinaciones de máscaras",
        *[f"- `{method}`: {desc}" for method, desc in VARIANT_DESCRIPTIONS.items()],
        "",
        "## Combinación seleccionada por validación interna en selection_dev",
        f"- sensitive: `{profile_variants['sensitive']}` -> `{VARIANT_DESCRIPTIONS[profile_variants['sensitive']]}`",
        f"- balanced: `{profile_variants['balanced']}` -> `{VARIANT_DESCRIPTIONS[profile_variants['balanced']]}`",
        f"- conservative: `{profile_variants['conservative']}` -> `{VARIANT_DESCRIPTIONS[profile_variants['conservative']]}`",
        "",
        "## Resultado final desplegado",
        f"- detector: `{detector_spec['method']}`",
        f"- máscara final por defecto: `{profile_variants['balanced']}` -> `{VARIANT_DESCRIPTIONS[profile_variants['balanced']]}`",
        "",
        "## Selección del detector en selection_dev",
        detector_selection_df.to_string(index=False),
        "",
        "## Selección del detector por fold",
        detector_selection_by_fold_df.to_string(index=False),
        "",
        "## Selección de combinación de máscaras en selection_dev",
        mask_selection_df.to_string(index=False),
        "",
        "## Patch-level en test_final",
        patch_metric_df.to_string(index=False),
        "",
        "## Pixel-wise estricto en test_final",
        pixel_summary_df.to_string(index=False),
        "",
        "## Pixel-wise tolerante en test_final",
        tolerant_summary_df.to_string(index=False),
    ]
    (HYBRID_PHASE_OUT / "hybrid_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        "Híbrido reconstruido con detector "
        f"`{detector_spec['method']}` | sensitive=`{profile_variants['sensitive']}` "
        f"| balanced=`{profile_variants['balanced']}` | conservative=`{profile_variants['conservative']}`"
    )
    print(patch_metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
