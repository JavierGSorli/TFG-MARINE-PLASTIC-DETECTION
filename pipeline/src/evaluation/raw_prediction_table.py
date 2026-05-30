from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.common.config import (
    DATASET_METADATA_GROUPED_PATH,
    DEBRIS_CLASS,
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
    EXTERNAL_B09_ZERO_OUT,
    INDICES_NO_WATER_OUT,
    INDICES_WATER_OUT,
    PATCHES_DIR,
    RESNET_OUT,
    RF_MODE_DIRS,
    RF_MODE_NAMES,
    SAM_CALIBRATED_MASKS_OUT,
    SAM_PHASE_OUT,
    SAM_PROB_DIR,
    UNET_OUT,
)
from src.common.pipeline_utils import infer_label_from_name, iter_patch_files
from src.evaluation.evaluation_split_utils import build_patch_subset_map


RF_VARIANTS = [(mode, RF_MODE_DIRS[mode]) for mode in RF_MODE_NAMES]
INDEX_VARIANTS = [("no_water", INDICES_NO_WATER_OUT), ("water", INDICES_WATER_OUT)]
EXTERNAL_VARIANTS = [
    ("b09_zero", EXTERNAL_B09_ZERO_OUT),
    ("b09_copy_b8a", EXTERNAL_B09_COPY_B8A_OUT),
    ("b09_interpolate_b8a_b11", EXTERNAL_B09_INTERP_OUT),
]


def read_debris_mask_px(mask_path: Path) -> int | None:
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int((mask == DEBRIS_CLASS).sum())


def read_binary_mask_px(mask_path: Path) -> int | None:
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int((mask > 0).sum())


def read_prob_raster_stats(prob_path: Path) -> dict[str, float | None]:
    if not prob_path.exists():
        return {"mean": None, "max": None, "p95": None}
    with rasterio.open(prob_path) as src:
        arr = src.read(1).astype("float32")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": None, "max": None, "p95": None}
    return {
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95)),
    }


def read_resnet_prob(json_path: Path) -> tuple[float | None, int | None]:
    if not json_path.exists():
        return None, None
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    prob = data.get("probabilities", {}).get("Marine Debris")
    active = int("Marine Debris" in data.get("active_labels", []))
    return prob, active


def read_external_prediction(model_dir: Path, stem: str) -> tuple[float | None, int | None]:
    predictions_path = model_dir / "predictions.csv"
    if predictions_path.exists():
        try:
            pred_df = pd.read_csv(predictions_path)
            patch_name = f"{stem}.tif"
            match = pred_df[(pred_df["patch"] == patch_name) & (pred_df["has_prediction"] == 1)]
            if not match.empty:
                score = match.iloc[0].get("score")
                pred_px = match.iloc[0].get("pred_px")
                score = float(score) if pd.notna(score) else None
                pred_px = int(pred_px) if pd.notna(pred_px) else None
                return score, pred_px
        except Exception:
            pass

    json_path = model_dir / f"{stem}.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        score = data.get("score", data.get("debris_prob", data.get("marine_debris_score")))
        return float(score) if score is not None else None, None

    mask_path = model_dir / "masks" / f"{stem}_mask.tif"
    if not mask_path.exists():
        mask_path = model_dir / f"{stem}_mask.tif"
    if mask_path.exists():
        with rasterio.open(mask_path) as src:
            total_px = src.width * src.height
            mask = src.read(1)
        pred_px = int((mask > 0).sum())
        score = float(pred_px / total_px) if total_px > 0 else 0.0
        return score, pred_px
    return None, None

def _load_metadata() -> pd.DataFrame:
    if DATASET_METADATA_GROUPED_PATH.exists():
        return pd.read_csv(DATASET_METADATA_GROUPED_PATH)
    rows = []
    for patch_path in iter_patch_files(PATCHES_DIR):
        rows.append(
            {
                "patch": patch_path.name,
                "date": patch_path.name[:8],
                "group_id": patch_path.name[:8],
                "label": infer_label_from_name(patch_path.name),
                "label_binary": 1 if infer_label_from_name(patch_path.name) == "SI" else 0,
            }
        )
    return pd.DataFrame(rows)


def build_raw_prediction_table() -> pd.DataFrame:
    metadata = _load_metadata()
    subset_map = build_patch_subset_map()
    metadata = metadata.merge(subset_map, on="patch", how="left")

    rows = []
    for _, meta_row in metadata.iterrows():
        patch_name = str(meta_row["patch"])
        patch_path = PATCHES_DIR / patch_name
        if not patch_path.exists():
            continue
        label = str(meta_row.get("label") or infer_label_from_name(patch_name))
        stem = patch_path.stem

        with rasterio.open(patch_path) as src:
            total_px = int(src.width * src.height)

        nc_mask_path = PATCHES_DIR / f"{stem}_mask.tif"
        nc_px = read_binary_mask_px(nc_mask_path) if label.upper() == "SI" else 0

        unet_argmax_px = read_debris_mask_px(UNET_OUT / f"{stem}_mask.tif")
        unet_prob_stats = read_prob_raster_stats(UNET_OUT / f"{stem}_marine_debris_prob.tif")
        resnet_prob, resnet_active = read_resnet_prob(RESNET_OUT / f"{stem}.json")

        row: dict[str, object] = {
            "patch": patch_name,
            "date": meta_row.get("date"),
            "group_id": meta_row.get("group_id"),
            "label": label,
            "label_binary": meta_row.get("label_binary", 1 if label.upper() == "SI" else 0),
            "eval_subset": meta_row.get("eval_subset"),
            "eval_split": meta_row.get("eval_split"),
            "nc_px": nc_px,
            "total_px": total_px,
            "unet_argmax_px": unet_argmax_px,
            "unet_prob_mean": unet_prob_stats["mean"],
            "unet_prob_max": unet_prob_stats["max"],
            "unet_prob_p95": unet_prob_stats["p95"],
            "resnet_prob": resnet_prob,
            "resnet_active": resnet_active,
        }

        for mode, variant_dir in RF_VARIANTS:
            rf_px = read_debris_mask_px(variant_dir / f"{stem}_mask.tif")
            rf_prob_stats = read_prob_raster_stats(variant_dir / f"{stem}_marine_debris_prob.tif")
            row[f"rf_{mode}_px"] = rf_px
            row[f"rf_{mode}_prob_mean"] = rf_prob_stats["mean"]
            row[f"rf_{mode}_prob_max"] = rf_prob_stats["max"]
            row[f"rf_{mode}_prob_p95"] = rf_prob_stats["p95"]

        for variant_name, indices_dir in INDEX_VARIANTS:
            for method in ["fdi", "ndvi", "fdi_ndvi"]:
                px = read_binary_mask_px(indices_dir / f"{stem}_{method}_mask.tif")
                row[f"{method}_{variant_name}_px"] = px

        sam_binary_px = read_binary_mask_px(SAM_PHASE_OUT / "binario" / f"{stem}_sam_debris_mask.tif")
        row["sam_binary_px"] = sam_binary_px
        sam_prob_stats = read_prob_raster_stats(SAM_PROB_DIR / f"{stem}_sam_marine_debris_score.tif")
        row["sam_prob_mean"] = sam_prob_stats["mean"]
        row["sam_prob_max"] = sam_prob_stats["max"]
        row["sam_prob_p95"] = sam_prob_stats["p95"]
        sam_thr_px = read_binary_mask_px(SAM_CALIBRATED_MASKS_OUT / f"{stem}_sam_debris_mask.tif")
        row["sam_thr_px"] = sam_thr_px

        for variant_name, variant_dir in EXTERNAL_VARIANTS:
            score, pred_px = read_external_prediction(variant_dir, stem)
            prob_stats = read_prob_raster_stats(variant_dir / "masks" / f"{stem}_mask_prob.tif")
            row[f"external_{variant_name}_score"] = score
            row[f"external_{variant_name}_default_px"] = pred_px
            row[f"external_{variant_name}_prob_mean"] = prob_stats["mean"]
            row[f"external_{variant_name}_prob_max"] = prob_stats["max"]
            row[f"external_{variant_name}_prob_p95"] = prob_stats["p95"]

        for col in metadata.columns:
            if col not in row and col not in {"patch"}:
                row[col] = meta_row.get(col)

        rows.append(row)

    return pd.DataFrame(rows)
