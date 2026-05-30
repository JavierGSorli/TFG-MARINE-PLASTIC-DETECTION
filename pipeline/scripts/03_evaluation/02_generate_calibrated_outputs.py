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

from src.common.config import (
    EVAL_CALIBRATED_OUT,
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
    EXTERNAL_B09_ZERO_OUT,
    PATCHES_DIR,
    RF_MODE_DIRS,
    RF_MODE_NAMES,
    SAM_CALIBRATED_MASKS_OUT,
    SAM_PROB_DIR,
    THRESHOLDS_PATH,
    UNET_CALIBRATED_MASKS_OUT,
    UNET_OUT,
)
from src.evaluation.raw_prediction_table import build_raw_prediction_table


EXTERNAL_VARIANT_DIRS = {
    "b09_zero": EXTERNAL_B09_ZERO_OUT,
    "b09_copy_b8a": EXTERNAL_B09_COPY_B8A_OUT,
    "b09_interpolate_b8a_b11": EXTERNAL_B09_INTERP_OUT,
}


def _write_binary_mask(prob_path: Path, out_path: Path, threshold: float) -> int:
    with rasterio.open(prob_path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
    binary = (arr >= threshold).astype("uint8")
    profile.update(count=1, dtype=rasterio.uint8, nodata=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(binary[np.newaxis, :, :])
        dst.update_tags(prob_threshold=threshold)
    return int(binary.sum())


def _threshold_map(df: pd.DataFrame) -> dict[str, float]:
    return {str(row["method_key"]): float(row["threshold"]) for _, row in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"No existe {THRESHOLDS_PATH}. Ejecuta 01_calibrate_thresholds.py primero.")

    EVAL_CALIBRATED_OUT.mkdir(parents=True, exist_ok=True)
    thresholds = pd.read_csv(THRESHOLDS_PATH)
    threshold_map = _threshold_map(thresholds)
    master = build_raw_prediction_table()
    sam_masks_written = 0

    rows: list[dict] = []
    for _, row in master.iterrows():
        patch = str(row["patch"])
        stem = Path(patch).stem
        total_px = int(row["total_px"])

        # U-Net calibrated mask
        unet_prob_path = UNET_OUT / f"{stem}_marine_debris_prob.tif"
        unet_thr = threshold_map.get("unet_prob")
        unet_thr_px = None
        if unet_thr is not None and unet_prob_path.exists():
            out_path = UNET_CALIBRATED_MASKS_OUT / f"{stem}_mask.tif"
            if args.overwrite or not out_path.exists():
                unet_thr_px = _write_binary_mask(unet_prob_path, out_path, unet_thr)
            else:
                with rasterio.open(out_path) as src:
                    unet_thr_px = int((src.read(1) > 0).sum())
        calibrated_row = {
            "patch": patch,
            "unet_thr_px": unet_thr_px,
            "unet_thr_pred": int(unet_thr_px >= threshold_map["unet_thr_px"]) if unet_thr_px is not None and "unet_thr_px" in threshold_map else None,
        }

        sam_prob_path = SAM_PROB_DIR / f"{stem}_sam_marine_debris_score.tif"
        sam_thr = threshold_map.get("sam_prob")
        sam_thr_px = None
        if sam_thr is not None and pd.notna(sam_thr) and sam_prob_path.exists():
            out_path = SAM_CALIBRATED_MASKS_OUT / f"{stem}_sam_debris_mask.tif"
            if args.overwrite or not out_path.exists():
                sam_thr_px = _write_binary_mask(sam_prob_path, out_path, sam_thr)
                sam_masks_written += 1
            else:
                with rasterio.open(out_path) as src:
                    sam_thr_px = int((src.read(1) > 0).sum())
        calibrated_row["sam_thr_px"] = sam_thr_px
        calibrated_row["sam_thr_pred"] = int(sam_thr_px >= threshold_map["sam_thr_px"]) if sam_thr_px is not None and "sam_thr_px" in threshold_map else None

        for mode in RF_MODE_NAMES:
            method_key = f"rf_{mode}_prob"
            thr = threshold_map.get(method_key)
            prob_path = RF_MODE_DIRS[mode] / f"{stem}_marine_debris_prob.tif"
            thr_px = None
            if thr is not None and prob_path.exists():
                out_path = RF_MODE_DIRS[mode] / "calibrated_masks" / f"{stem}_mask.tif"
                if args.overwrite or not out_path.exists():
                    thr_px = _write_binary_mask(prob_path, out_path, thr)
                else:
                    with rasterio.open(out_path) as src:
                        thr_px = int((src.read(1) > 0).sum())
            calibrated_row[f"rf_{mode}_thr_px"] = thr_px
            px_method_key = f"rf_{mode}_thr_px"
            calibrated_row[f"rf_{mode}_thr_pred"] = (
                int(thr_px >= threshold_map[px_method_key]) if thr_px is not None and px_method_key in threshold_map else None
            )

        for variant_name, variant_dir in EXTERNAL_VARIANT_DIRS.items():
            method_key = f"external_{variant_name}_prob"
            thr = threshold_map.get(method_key)
            prob_path = variant_dir / "masks" / f"{stem}_mask_prob.tif"
            thr_px = None
            if thr is not None and prob_path.exists():
                out_path = variant_dir / "calibrated_masks" / f"{stem}_mask.tif"
                if args.overwrite or not out_path.exists():
                    thr_px = _write_binary_mask(prob_path, out_path, thr)
                else:
                    with rasterio.open(out_path) as src:
                        thr_px = int((src.read(1) > 0).sum())
            calibrated_row[f"external_{variant_name}_thr_px"] = thr_px
            px_method_key = f"external_{variant_name}_thr_px"
            calibrated_row[f"external_{variant_name}_thr_pred"] = (
                int(thr_px >= threshold_map[px_method_key]) if thr_px is not None and px_method_key in threshold_map else None
            )

        resnet_prob = row.get("resnet_prob")
        calibrated_row["resnet_default_pred"] = int(pd.notna(resnet_prob) and float(resnet_prob) >= 0.5)
        calibrated_row["resnet_thr_pred"] = (
            int(pd.notna(resnet_prob) and float(resnet_prob) >= threshold_map["resnet_prob"])
            if "resnet_prob" in threshold_map
            else None
        )

        rows.append(calibrated_row)

    calibrated_df = pd.DataFrame(rows)
    calibrated_df.to_csv(EVAL_CALIBRATED_OUT / "calibrated_patch_predictions.csv", index=False)

    summary_lines = [
        "# Calibrated Outputs Summary",
        "",
        f"Patches procesados: {len(calibrated_df)}",
        "",
        "Máscaras calibradas guardadas en:",
        f"- U-Net: {UNET_CALIBRATED_MASKS_OUT}",
        f"- SAM: {SAM_CALIBRATED_MASKS_OUT}",
        *[f"- RF {mode}: {RF_MODE_DIRS[mode] / 'calibrated_masks'}" for mode in RF_MODE_NAMES],
        f"- External b09_zero: {EXTERNAL_B09_ZERO_OUT / 'calibrated_masks'}",
        f"- External b09_copy_b8a: {EXTERNAL_B09_COPY_B8A_OUT / 'calibrated_masks'}",
        f"- External b09_interpolate_b8a_b11: {EXTERNAL_B09_INTERP_OUT / 'calibrated_masks'}",
        "",
        f"Máscaras SAM escritas en esta ejecución: {sam_masks_written}",
    ]
    (EVAL_CALIBRATED_OUT / "calibrated_outputs_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Salidas calibradas guardadas en: {EVAL_CALIBRATED_OUT}")


if __name__ == "__main__":
    main()
