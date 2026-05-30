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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.common.config import EVAL_PATCH_LEVEL_OUT, PREDICTIONS_MASTER_PATH, THRESHOLDS_PATH


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2 or np.unique(scores).size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = _safe_auc(y_true, scores) if scores is not None else None
    return {
        "n": int(len(y_true)),
        "auc_roc": round(auc, 6) if auc is not None else np.nan,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _method_specs() -> list[dict]:
    specs = [
        {
            "method": "UNet argmax",
            "score_col": "unet_argmax_px",
            "pred_col": "unet_argmax_px",
            "pred_rule": "px_threshold",
            "mode": "default",
            "threshold": None,
            "min_positive_pixels": 1.0,
        },
        {
            "method": "UNet calibrated",
            "score_col": "unet_thr_px",
            "pred_col": "unet_thr_pred",
            "pred_rule": "binary_col",
            "mode": "calibrated",
            "threshold_key": "unet_prob",
            "min_pixels_key": "unet_thr_px",
        },
        {
            "method": "ResNet default",
            "score_col": "resnet_prob",
            "pred_col": "resnet_default_pred",
            "pred_rule": "binary_col",
            "mode": "default",
            "threshold": 0.5,
            "min_positive_pixels": np.nan,
        },
        {
            "method": "ResNet calibrated",
            "score_col": "resnet_prob",
            "pred_col": "resnet_thr_pred",
            "pred_rule": "binary_col",
            "mode": "calibrated",
            "threshold_key": "resnet_prob",
            "min_positive_pixels": np.nan,
        },
        {
            "method": "SAM binary",
            "score_col": "sam_binary_px",
            "pred_col": "sam_binary_px",
            "pred_rule": "px_threshold",
            "mode": "default",
            "threshold": None,
            "min_positive_pixels": 1.0,
        },
        {
            "method": "SAM calibrated",
            "score_col": "sam_thr_px",
            "pred_col": "sam_thr_pred",
            "pred_rule": "binary_col",
            "mode": "calibrated",
            "threshold_key": "sam_prob",
            "min_pixels_key": "sam_thr_px",
        },
    ]

    for rf_mode in ["full", "no_texture", "indices_only", "bands_only"]:
        specs.append(
            {
                "method": f"RF {rf_mode}",
                "score_col": f"rf_{rf_mode}_px",
                "pred_col": f"rf_{rf_mode}_px",
                "pred_rule": "px_threshold",
                "mode": "default",
                "threshold": None,
                "min_positive_pixels": 1.0,
            }
        )
        specs.append(
            {
                "method": f"RF {rf_mode} calibrated",
                "score_col": f"rf_{rf_mode}_thr_px",
                "pred_col": f"rf_{rf_mode}_thr_pred",
                "pred_rule": "binary_col",
                "mode": "calibrated",
                "threshold_key": f"rf_{rf_mode}_prob",
                "min_pixels_key": f"rf_{rf_mode}_thr_px",
            }
        )

    index_specs = [
        ("FDI", "fdi_no_water_px"),
        ("NDVI", "ndvi_no_water_px"),
        ("FDI+NDVI", "fdi_ndvi_no_water_px"),
        ("FDI_mask", "fdi_water_px"),
        ("NDVI_mask", "ndvi_water_px"),
        ("FDI+NDVI_mask", "fdi_ndvi_water_px"),
    ]
    for method_name, score_col in index_specs:
        specs.append(
            {
                "method": method_name,
                "score_col": score_col,
                "pred_col": score_col,
                "pred_rule": "px_threshold",
                "mode": "default",
                "threshold": None,
                "min_positive_pixels": 1.0,
            }
        )
        specs.append(
            {
                "method": f"{method_name} calibrated",
                "score_col": score_col,
                "pred_col": score_col,
                "pred_rule": "px_threshold",
                "mode": "calibrated",
                "threshold": np.nan,
                "min_pixels_key": score_col,
            }
        )

    for variant in ["b09_zero", "b09_copy_b8a", "b09_interpolate_b8a_b11"]:
        specs.append(
            {
                "method": f"External {variant} default",
                "score_col": f"external_{variant}_default_px",
                "pred_col": f"external_{variant}_default_px",
                "pred_rule": "px_threshold",
                "mode": "default",
                "threshold": None,
                "min_positive_pixels": 1.0,
            }
        )
        specs.append(
            {
                "method": f"External {variant} calibrated",
                "score_col": f"external_{variant}_thr_px",
                "pred_col": f"external_{variant}_thr_pred",
                "pred_rule": "binary_col",
                "mode": "calibrated",
                "threshold_key": f"external_{variant}_prob",
                "min_pixels_key": f"external_{variant}_thr_px",
            }
        )
    return specs


def _threshold_map() -> dict[str, float]:
    if not THRESHOLDS_PATH.exists():
        return {}
    df = pd.read_csv(THRESHOLDS_PATH)
    return dict(zip(df["method_key"].astype(str), pd.to_numeric(df["threshold"], errors="coerce")))


def _fmt_threshold(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _fmt_min_pixels(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(int(round(float(value))))


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin datos_\n"
    return df.to_markdown(index=False) + "\n"


def main() -> None:
    if not PREDICTIONS_MASTER_PATH.exists():
        raise FileNotFoundError(f"No existe {PREDICTIONS_MASTER_PATH}. Ejecuta 03_unify_predictions.py primero.")
    EVAL_PATCH_LEVEL_OUT.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(PREDICTIONS_MASTER_PATH)
    test_df = master[master["eval_subset"] == "test_final"].copy()
    y_true = test_df["label_binary"].astype(int).to_numpy()
    threshold_map = _threshold_map()

    summary_rows = []
    by_patch_rows = []
    for spec in _method_specs():
        if spec["pred_col"] not in test_df.columns:
            continue

        threshold = spec.get("threshold")
        if "threshold_key" in spec:
            threshold = float(threshold_map.get(spec["threshold_key"])) if spec["threshold_key"] in threshold_map else np.nan

        min_positive_pixels = spec.get("min_positive_pixels")
        if "min_pixels_key" in spec:
            min_positive_pixels = float(threshold_map.get(spec["min_pixels_key"])) if spec["min_pixels_key"] in threshold_map else np.nan

        if spec["pred_rule"] == "px_threshold":
            pred_series = pd.to_numeric(test_df[spec["pred_col"]], errors="coerce").fillna(0)
            px_thr = float(min_positive_pixels) if min_positive_pixels is not None and pd.notna(min_positive_pixels) else 1.0
            y_pred = (pred_series >= px_thr).astype(int).to_numpy()
            score_series = (
                pd.to_numeric(test_df[spec["score_col"]], errors="coerce").to_numpy(dtype=float)
                if spec["score_col"] in test_df.columns
                else None
            )
            metrics = _metrics(y_true, y_pred, score_series)
            summary_rows.append(
                {
                    "method": spec["method"],
                    "mode": spec["mode"],
                    "threshold": threshold,
                    "min_positive_pixels": min_positive_pixels,
                    **metrics,
                }
            )
            for local_idx, patch in enumerate(test_df["patch"]):
                by_patch_rows.append(
                    {
                        "method": spec["method"],
                        "mode": spec["mode"],
                        "patch": patch,
                        "y_true": int(y_true[local_idx]),
                        "y_pred": int(y_pred[local_idx]),
                        "score": float(score_series[local_idx]) if score_series is not None else np.nan,
                    }
                )
            continue

        valid = test_df[spec["pred_col"]].notna()
        pred_series = pd.to_numeric(test_df.loc[valid, spec["pred_col"]], errors="coerce").fillna(0)
        y_pred = pred_series.astype(int).to_numpy()
        y_true_valid = y_true[valid.to_numpy()]
        score_series = (
            pd.to_numeric(test_df.loc[valid, spec["score_col"]], errors="coerce").to_numpy(dtype=float)
            if spec["score_col"] in test_df.columns
            else None
        )
        if len(y_pred) == 0:
            continue
        metrics = _metrics(y_true_valid, y_pred, score_series)
        summary_rows.append(
            {
                "method": spec["method"],
                "mode": spec["mode"],
                "threshold": threshold,
                "min_positive_pixels": min_positive_pixels,
                **metrics,
            }
        )
        for local_idx, patch in enumerate(test_df.loc[valid, "patch"]):
            by_patch_rows.append(
                {
                    "method": spec["method"],
                    "mode": spec["mode"],
                    "patch": patch,
                    "y_true": int(y_true_valid[local_idx]),
                    "y_pred": int(y_pred[local_idx]),
                    "score": float(score_series[local_idx]) if score_series is not None else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["mode", "f1"], ascending=[True, False])
    by_patch_df = pd.DataFrame(by_patch_rows)
    summary_df.to_csv(EVAL_PATCH_LEVEL_OUT / "patch_level_metrics.csv", index=False)
    by_patch_df.to_csv(EVAL_PATCH_LEVEL_OUT / "patch_level_by_patch.csv", index=False)

    default_md_cols = ["method", "accuracy", "precision", "recall", "f1", "auc_roc", "tp", "fp", "tn", "fn"]
    calibrated_md_cols = ["method", "threshold", "min_positive_pixels", "accuracy", "precision", "recall", "f1", "auc_roc", "tp", "fp", "tn", "fn"]
    default_df = summary_df[summary_df["mode"] == "default"].copy().sort_values("f1", ascending=False)
    calibrated_df = summary_df[summary_df["mode"] == "calibrated"].copy().sort_values("f1", ascending=False)
    default_df["threshold"] = default_df["threshold"].map(_fmt_threshold)
    default_df["min_positive_pixels"] = default_df["min_positive_pixels"].map(_fmt_min_pixels)
    calibrated_df["threshold"] = calibrated_df["threshold"].map(_fmt_threshold)
    calibrated_df["min_positive_pixels"] = calibrated_df["min_positive_pixels"].map(_fmt_min_pixels)

    summary_md = (
        "# Patch-level summary\n\n"
        "## Métodos sin calibrar\n\n"
        f"{_markdown_table(default_df[default_md_cols])}\n"
        "## Métodos calibrados\n\n"
        "La columna `threshold` indica el umbral continuo usado cuando existe una salida probabilística o score global. "
        "La columna `min_positive_pixels` indica el número mínimo de píxeles positivos requerido para declarar el patch como positivo en métodos basados en máscara.\n\n"
        f"{_markdown_table(calibrated_df[calibrated_md_cols])}\n"
        "## Comparativa global\n\n"
        f"{_markdown_table(pd.concat([default_df, calibrated_df], ignore_index=True).sort_values('f1', ascending=False)[calibrated_md_cols])}"
    )
    (EVAL_PATCH_LEVEL_OUT / "patch_level_summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
