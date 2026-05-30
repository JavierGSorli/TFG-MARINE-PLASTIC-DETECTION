from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.common.config import (
    EVAL_ERROR_ANALYSIS_OUT,
    EVAL_PIXELWISE_OUT,
    INDICES_NO_WATER_OUT,
    INDICES_WATER_OUT,
    PATCHES_DIR,
    PREDICTIONS_MASTER_PATH,
    RF_MODE_DIRS,
    RF_MODE_NAMES,
    SAM_CALIBRATED_MASKS_OUT,
    SAM_PROB_DIR,
    THRESHOLDS_PATH,
    UNET_OUT,
    UNET_CALIBRATED_MASKS_OUT,
    EXTERNAL_B09_ZERO_OUT,
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
)


DIAGNOSTICS_SUMMARY_OUT = EVAL_ERROR_ANALYSIS_OUT / "diagnostics_summary.md"
MASK_PX_COLS = [
    "unet_argmax_px",
    "unet_thr_px",
    "sam_binary_px",
    "sam_thr_px",
    "rf_full_px",
    "rf_full_thr_px",
    "rf_no_texture_px",
    "rf_no_texture_thr_px",
    "rf_indices_only_px",
    "rf_indices_only_thr_px",
    "rf_bands_only_px",
    "rf_bands_only_thr_px",
    "fdi_no_water_px",
    "ndvi_no_water_px",
    "fdi_ndvi_no_water_px",
    "fdi_water_px",
    "ndvi_water_px",
    "fdi_ndvi_water_px",
    "external_b09_zero_default_px",
    "external_b09_zero_thr_px",
    "external_b09_copy_b8a_default_px",
    "external_b09_copy_b8a_thr_px",
    "external_b09_interpolate_b8a_b11_default_px",
    "external_b09_interpolate_b8a_b11_thr_px",
]
GT_PAGE_COLOR = (0.88, 0.18, 0.18)
GT_PAGE_BG = 0.78
PRED_PAGE_COLOR = (0.93, 0.22, 0.18)

ALL_METHODS = {
    "UNet_argmax": {"display": "UNet argmax", "pred_col": "unet_argmax_px", "kind": "px_threshold", "score_col": "unet_argmax_px", "threshold": 1.0, "prob_path": lambda stem: UNET_OUT / f"{stem}_marine_debris_prob.tif"},
    "UNet_calibrated": {"display": "UNet calibrated", "pred_col": "unet_thr_pred", "kind": "binary_col", "score_col": "unet_thr_px", "prob_path": lambda stem: UNET_OUT / f"{stem}_marine_debris_prob.tif"},
    "ResNet_default": {"display": "ResNet default", "pred_col": "resnet_default_pred", "kind": "binary_col", "score_col": "resnet_prob"},
    "ResNet_calibrated": {"display": "ResNet calibrated", "pred_col": "resnet_thr_pred", "kind": "binary_col", "score_col": "resnet_prob"},
    "SAM_binary": {"display": "SAM binary", "pred_col": "sam_binary_px", "kind": "px_threshold", "score_col": "sam_binary_px", "threshold": 1.0, "prob_path": lambda stem: SAM_PROB_DIR / f"{stem}_sam_marine_debris_score.tif"},
    "SAM_calibrated": {"display": "SAM calibrated", "pred_col": "sam_thr_pred", "kind": "binary_col", "score_col": "sam_thr_px", "prob_path": lambda stem: SAM_PROB_DIR / f"{stem}_sam_marine_debris_score.tif"},
    "External_zero_default": {"display": "External b09_zero default", "pred_col": "external_b09_zero_default_px", "kind": "px_threshold", "score_col": "external_b09_zero_default_px", "threshold": 1.0, "prob_path": lambda stem: EXTERNAL_B09_ZERO_OUT / "masks" / f"{stem}_mask_prob.tif"},
    "External_zero_calibrated": {"display": "External b09_zero calibrated", "pred_col": "external_b09_zero_thr_pred", "kind": "binary_col", "score_col": "external_b09_zero_thr_px", "prob_path": lambda stem: EXTERNAL_B09_ZERO_OUT / "masks" / f"{stem}_mask_prob.tif"},
    "External_copy_b8a_default": {"display": "External b09_copy_b8a default", "pred_col": "external_b09_copy_b8a_default_px", "kind": "px_threshold", "score_col": "external_b09_copy_b8a_default_px", "threshold": 1.0, "prob_path": lambda stem: EXTERNAL_B09_COPY_B8A_OUT / "masks" / f"{stem}_mask_prob.tif"},
    "External_copy_b8a_calibrated": {"display": "External b09_copy_b8a calibrated", "pred_col": "external_b09_copy_b8a_thr_pred", "kind": "binary_col", "score_col": "external_b09_copy_b8a_thr_px", "prob_path": lambda stem: EXTERNAL_B09_COPY_B8A_OUT / "masks" / f"{stem}_mask_prob.tif"},
    "External_interp_default": {"display": "External b09_interpolate_b8a_b11 default", "pred_col": "external_b09_interpolate_b8a_b11_default_px", "kind": "px_threshold", "score_col": "external_b09_interpolate_b8a_b11_default_px", "threshold": 1.0, "prob_path": lambda stem: EXTERNAL_B09_INTERP_OUT / "masks" / f"{stem}_mask_prob.tif"},
    "External_interp_calibrated": {"display": "External b09_interpolate_b8a_b11 calibrated", "pred_col": "external_b09_interpolate_b8a_b11_thr_pred", "kind": "binary_col", "score_col": "external_b09_interpolate_b8a_b11_thr_px", "prob_path": lambda stem: EXTERNAL_B09_INTERP_OUT / "masks" / f"{stem}_mask_prob.tif"},
    "FDI_default": {"display": "FDI", "pred_col": "fdi_no_water_px", "kind": "px_threshold", "score_col": "fdi_no_water_px", "threshold": 1.0},
    "FDI_calibrated": {"display": "FDI calibrated", "pred_col": "fdi_no_water_px", "kind": "px_threshold", "score_col": "fdi_no_water_px"},
    "FDI_mask_default": {"display": "FDI_mask", "pred_col": "fdi_water_px", "kind": "px_threshold", "score_col": "fdi_water_px", "threshold": 1.0},
    "FDI_mask_calibrated": {"display": "FDI_mask calibrated", "pred_col": "fdi_water_px", "kind": "px_threshold", "score_col": "fdi_water_px"},
    "NDVI_default": {"display": "NDVI", "pred_col": "ndvi_no_water_px", "kind": "px_threshold", "score_col": "ndvi_no_water_px", "threshold": 1.0},
    "NDVI_calibrated": {"display": "NDVI calibrated", "pred_col": "ndvi_no_water_px", "kind": "px_threshold", "score_col": "ndvi_no_water_px"},
    "NDVI_mask_default": {"display": "NDVI_mask", "pred_col": "ndvi_water_px", "kind": "px_threshold", "score_col": "ndvi_water_px", "threshold": 1.0},
    "NDVI_mask_calibrated": {"display": "NDVI_mask calibrated", "pred_col": "ndvi_water_px", "kind": "px_threshold", "score_col": "ndvi_water_px"},
    "FDI_NDVI_default": {"display": "FDI+NDVI", "pred_col": "fdi_ndvi_no_water_px", "kind": "px_threshold", "score_col": "fdi_ndvi_no_water_px", "threshold": 1.0},
    "FDI_NDVI_calibrated": {"display": "FDI+NDVI calibrated", "pred_col": "fdi_ndvi_no_water_px", "kind": "px_threshold", "score_col": "fdi_ndvi_no_water_px"},
    "FDI_NDVI_mask_default": {"display": "FDI+NDVI_mask", "pred_col": "fdi_ndvi_water_px", "kind": "px_threshold", "score_col": "fdi_ndvi_water_px", "threshold": 1.0},
    "FDI_NDVI_mask_calibrated": {"display": "FDI+NDVI_mask calibrated", "pred_col": "fdi_ndvi_water_px", "kind": "px_threshold", "score_col": "fdi_ndvi_water_px"},
}

for _mode in RF_MODE_NAMES:
    ALL_METHODS[f"RF_{_mode}"] = {
        "display": f"RF {_mode}",
        "pred_col": f"rf_{_mode}_px",
        "kind": "px_threshold",
        "score_col": f"rf_{_mode}_px",
        "threshold": 1.0,
        "prob_path": lambda stem, mode=_mode: RF_MODE_DIRS[mode] / f"{stem}_marine_debris_prob.tif",
    }
    ALL_METHODS[f"RF_{_mode}_calibrated"] = {
        "display": f"RF {_mode} calibrated",
        "pred_col": f"rf_{_mode}_thr_pred",
        "kind": "binary_col",
        "score_col": f"rf_{_mode}_thr_px",
        "prob_path": lambda stem, mode=_mode: RF_MODE_DIRS[mode] / f"{stem}_marine_debris_prob.tif",
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin datos_\n"
    return df.to_markdown(index=False) + "\n"


def split_scene_tags(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip().lower() for part in str(value).split(";") if part.strip()]


def make_rgb(data: np.ndarray, vmax: float = 0.1) -> np.ndarray:
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    return np.clip(rgb / vmax, 0, 1)


def read_patch_rgb(patch_name: str) -> np.ndarray | None:
    patch_path = PATCHES_DIR / patch_name
    if not patch_path.exists():
        return None
    with rasterio.open(patch_path) as src:
        return make_rgb(src.read().astype("float32"))


def read_prob_map(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype("float32")


def read_gt_mask(stem: str) -> np.ndarray | None:
    path = PATCHES_DIR / f"{stem}_mask.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(bool)


def read_water_mask(stem: str) -> np.ndarray | None:
    path = INDICES_WATER_OUT / f"{stem}_water_output_mask.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(bool)


def gt_only(mask: np.ndarray) -> np.ndarray:
    canvas = np.full((mask.shape[0], mask.shape[1], 3), GT_PAGE_BG, dtype=np.float32)
    active = mask.astype(bool)
    for ch, val in enumerate(GT_PAGE_COLOR):
        canvas[:, :, ch] = np.where(active, val, canvas[:, :, ch])
    return canvas


def pred_only(mask: np.ndarray) -> np.ndarray:
    canvas = np.full((mask.shape[0], mask.shape[1], 3), GT_PAGE_BG, dtype=np.float32)
    active = mask.astype(bool)
    for ch, val in enumerate(PRED_PAGE_COLOR):
        canvas[:, :, ch] = np.where(active, val, canvas[:, :, ch])
    return canvas


def mask_path_for_method(method_name: str, stem: str) -> Path | None:
    if method_name == "UNet argmax":
        return UNET_OUT / f"{stem}_mask.tif"
    if method_name == "UNet calibrated":
        return UNET_CALIBRATED_MASKS_OUT / f"{stem}_mask.tif"
    if method_name == "SAM binary":
        return SAM_PHASE_OUT / "binario" / f"{stem}_sam_debris_mask.tif"
    if method_name == "SAM calibrated":
        return SAM_CALIBRATED_MASKS_OUT / f"{stem}_sam_debris_mask.tif"
    if method_name.startswith("RF "):
        rf_name = method_name.replace("RF ", "", 1)
        calibrated = rf_name.endswith(" calibrated")
        mode = rf_name.replace(" calibrated", "")
        base = RF_MODE_DIRS[mode]
        if calibrated:
            return base / "calibrated_masks" / f"{stem}_mask.tif"
        return base / f"{stem}_mask.tif"
    if method_name == "FDI":
        return INDICES_NO_WATER_OUT / f"{stem}_fdi_mask.tif"
    if method_name == "NDVI":
        return INDICES_NO_WATER_OUT / f"{stem}_ndvi_mask.tif"
    if method_name == "FDI+NDVI":
        return INDICES_NO_WATER_OUT / f"{stem}_fdi_ndvi_mask.tif"
    if method_name == "FDI_mask":
        return INDICES_WATER_OUT / f"{stem}_fdi_mask.tif"
    if method_name == "NDVI_mask":
        return INDICES_WATER_OUT / f"{stem}_ndvi_mask.tif"
    if method_name == "FDI+NDVI_mask":
        return INDICES_WATER_OUT / f"{stem}_fdi_ndvi_mask.tif"
    if method_name == "External b09_zero default":
        return EXTERNAL_B09_ZERO_OUT / "masks" / f"{stem}_mask.tif"
    if method_name == "External b09_zero calibrated":
        return EXTERNAL_B09_ZERO_OUT / "calibrated_masks" / f"{stem}_mask.tif"
    if method_name == "External b09_copy_b8a default":
        return EXTERNAL_B09_COPY_B8A_OUT / "masks" / f"{stem}_mask.tif"
    if method_name == "External b09_copy_b8a calibrated":
        return EXTERNAL_B09_COPY_B8A_OUT / "calibrated_masks" / f"{stem}_mask.tif"
    if method_name == "External b09_interpolate_b8a_b11 default":
        return EXTERNAL_B09_INTERP_OUT / "masks" / f"{stem}_mask.tif"
    if method_name == "External b09_interpolate_b8a_b11 calibrated":
        return EXTERNAL_B09_INTERP_OUT / "calibrated_masks" / f"{stem}_mask.tif"
    return None


def read_method_mask(method_name: str, patch_name: str) -> np.ndarray | None:
    stem = Path(patch_name).stem
    path = mask_path_for_method(method_name, stem)
    if path is None or not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1)
    if method_name.startswith("RF ") and "calibrated" not in method_name:
        return (arr == DEBRIS_CLASS).astype(bool)
    if method_name == "UNet argmax":
        return (arr == DEBRIS_CLASS).astype(bool)
    return (arr > 0).astype(bool)


def load_thresholds() -> dict[str, float]:
    if not THRESHOLDS_PATH.exists():
        return {}
    df = pd.read_csv(THRESHOLDS_PATH)
    return dict(zip(df["method_key"], df["threshold"].astype(float)))


def classify_errors(master: pd.DataFrame, method_keys: list[str], error_types: str) -> pd.DataFrame:
    rows = []
    y_true = master["label_binary"].astype(int)
    thresholds = load_thresholds()
    for key in method_keys:
        cfg = ALL_METHODS[key]
        pred_col = cfg["pred_col"]
        if pred_col not in master.columns:
            continue
        if cfg["kind"] == "px_threshold":
            thr = float(cfg.get("threshold", thresholds.get(cfg["score_col"], 1.0)))
            preds = (pd.to_numeric(master[pred_col], errors="coerce").fillna(0) >= thr).astype(int)
        else:
            preds = pd.to_numeric(master[pred_col], errors="coerce").fillna(0).astype(int)
        for idx, pred in preds.items():
            truth = int(y_true.loc[idx])
            if pred == truth:
                continue
            error_type = "FP" if pred == 1 else "FN"
            if error_types != "all" and error_type.lower() != error_types.lower():
                continue
            row = master.loc[idx]
            rows.append(
                {
                    "method_key": key,
                    "method": cfg["display"],
                    "error_type": error_type,
                    "patch": row["patch"],
                    "label": row["label"],
                    "score": row.get(cfg["score_col"], np.nan),
                }
            )
    return pd.DataFrame(rows)


def render_error_examples(errors: pd.DataFrame, out_dir: Path, limit: int = 20) -> None:
    examples_dir = out_dir / "error_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for (method_key, error_type), group in errors.groupby(["method_key", "error_type"]):
        cfg = ALL_METHODS[method_key]
        for _, row in group.head(limit).iterrows():
            patch_name = row["patch"]
            stem = Path(patch_name).stem
            rgb = read_patch_rgb(patch_name)
            gt_mask = read_gt_mask(stem)
            if rgb is None or gt_mask is None:
                continue
            prob_path_fn = cfg.get("prob_path")
            prob_map = read_prob_map(prob_path_fn(stem)) if prob_path_fn is not None else None
            fig, axes = plt.subplots(1, 3 if prob_map is None else 4, figsize=(16, 4))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[1].imshow(gt_mask, cmap="Greens")
            axes[1].set_title("GT")
            if prob_map is not None:
                axes[2].imshow(prob_map, cmap="magma")
                axes[2].set_title("Probabilidad")
                axes[3].imshow(rgb)
                axes[3].set_title("Patch")
            else:
                axes[2].imshow(rgb)
                axes[2].set_title("Patch")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{error_type} | {cfg['display']} | {patch_name} | score={row['score']}")
            fig.tight_layout()
            out_path = examples_dir / f"{error_type}_{safe_name(cfg['display'])}_{stem}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close(fig)


def render_patch_tile(patch_name: str, out_path: Path, title: str, subtitle: str = "") -> None:
    rgb = read_patch_rgb(patch_name)
    if rgb is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_patch_tile_with_gt(patch_name: str, out_path: Path, title: str, subtitle: str = "") -> None:
    rgb = read_patch_rgb(patch_name)
    gt_mask = read_gt_mask(Path(patch_name).stem)
    if rgb is None or gt_mask is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(rgb)
    axes[0].axis("off")
    axes[0].set_title("RGB", fontsize=9)
    axes[1].imshow(gt_mask, cmap="Greens")
    axes[1].axis("off")
    axes[1].set_title("GT", fontsize=9)
    fig.suptitle(title, fontsize=10)
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_patch_gallery(items: list[dict], out_dir: Path, basename: str, title: str, include_gt_in_individual: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        return
    for idx, item in enumerate(items, start=1):
        patch_name = str(item["patch"])
        patch_stem = Path(patch_name).stem
        subtitle = item.get("subtitle", "")
        if include_gt_in_individual:
            render_patch_tile_with_gt(
                patch_name,
                out_dir / f"{idx:02d}_{patch_stem}.png",
                item.get("title", patch_stem),
                subtitle=subtitle,
            )
        else:
            render_patch_tile(
                patch_name,
                out_dir / f"{idx:02d}_{patch_stem}.png",
                item.get("title", patch_stem),
                subtitle=subtitle,
            )

    n = len(items)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, item in zip(axes, items):
        rgb = read_patch_rgb(str(item["patch"]))
        if rgb is None:
            continue
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(item.get("title", Path(str(item["patch"])).stem), fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{basename}_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_patch_gallery_pages_with_gt(items: list[dict], out_dir: Path, basename: str, title: str, page_size: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        return

    for idx, item in enumerate(items, start=1):
        patch_name = str(item["patch"])
        patch_stem = Path(patch_name).stem
        render_patch_tile_with_gt(
            patch_name,
            out_dir / f"{idx:02d}_{patch_stem}.png",
            item.get("title", patch_stem),
            subtitle=item.get("subtitle", ""),
        )

    pages = int(np.ceil(len(items) / page_size))
    for page_idx in range(pages):
        chunk = items[page_idx * page_size : (page_idx + 1) * page_size]
        fig, axes = plt.subplots(2, len(chunk), figsize=(max(3.0 * len(chunk), 12), 6.5))
        axes = np.atleast_2d(axes)
        for col_idx, item in enumerate(chunk):
            patch_name = str(item["patch"])
            rgb = read_patch_rgb(patch_name)
            gt_mask = read_gt_mask(Path(patch_name).stem)
            if rgb is not None:
                axes[0, col_idx].imshow(rgb)
            axes[0, col_idx].axis("off")
            axes[0, col_idx].set_title(item.get("title", Path(patch_name).stem), fontsize=8)
            if gt_mask is not None:
                axes[1, col_idx].imshow(gt_only(gt_mask))
            axes[1, col_idx].axis("off")
            axes[1, col_idx].set_title(item.get("subtitle", ""), fontsize=8)
        fig.suptitle(f"{title} | pagina {page_idx + 1}/{pages}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{basename}_page_{page_idx + 1:02d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def render_negative_method_gallery(items: list[dict], out_dir: Path, basename: str, title: str, page_size: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        return

    for idx, item in enumerate(items, start=1):
        patch_name = str(item["patch"])
        stem = Path(patch_name).stem
        rgb = read_patch_rgb(patch_name)
        pred_mask = read_method_mask(str(item["method"]), patch_name)
        if rgb is None or pred_mask is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
        axes[0].imshow(rgb)
        axes[0].axis("off")
        axes[0].set_title("RGB", fontsize=9)
        axes[1].imshow(pred_only(pred_mask))
        axes[1].axis("off")
        axes[1].set_title(f"{item['method']}\npred_px={int(item['pred_px'])}", fontsize=9)
        fig.suptitle(f"{idx}. {stem}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:02d}_{stem}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    pages = int(np.ceil(len(items) / page_size))
    for page_idx in range(pages):
        chunk = items[page_idx * page_size : (page_idx + 1) * page_size]
        fig, axes = plt.subplots(2, len(chunk), figsize=(max(3.2 * len(chunk), 13), 7.2))
        axes = np.atleast_2d(axes)
        for col_idx, item in enumerate(chunk):
            patch_name = str(item["patch"])
            rgb = read_patch_rgb(patch_name)
            pred_mask = read_method_mask(str(item["method"]), patch_name)
            if rgb is not None:
                axes[0, col_idx].imshow(rgb)
            axes[0, col_idx].axis("off")
            axes[0, col_idx].set_title(f"{Path(patch_name).stem}", fontsize=8)
            if pred_mask is not None:
                axes[1, col_idx].imshow(pred_only(pred_mask))
            axes[1, col_idx].axis("off")
            axes[1, col_idx].set_title(f"{item['method']}\npred_px={int(item['pred_px'])}", fontsize=7.5, pad=10)
        fig.suptitle(f"{title} | pagina {page_idx + 1}/{pages}", fontsize=12, y=0.98)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.05, wspace=0.08, hspace=0.30)
        fig.savefig(out_dir / f"{basename}_page_{page_idx + 1:02d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def render_water_mask_pages(master: pd.DataFrame, out_dir: Path, page_size: int = 6, max_patches: int = 18) -> int:
    mask_dir = out_dir / "mascaras"
    mask_dir.mkdir(parents=True, exist_ok=True)
    fixed_second_page_second_patch = "20171115_SI_000187_01.tif"

    subset = master.copy()
    subset["has_water_mask"] = subset["patch"].apply(lambda p: (INDICES_WATER_OUT / f"{Path(str(p)).stem}_water_output_mask.tif").exists())
    subset = subset[subset["has_water_mask"]].copy()
    if subset.empty:
        return 0

    ordered_groups: list[pd.DataFrame] = []
    used_patches: set[str] = set()
    for keyword in ["costa", "barco", "nube"]:
        group = subset[subset["scene_tags"].apply(lambda v, kw=keyword: kw in split_scene_tags(v))].copy()
        group = group[~group["patch"].isin(used_patches)].copy()
        if group.empty:
            continue
        group = group.sort_values(["image_quality", "label_binary", "patch"], ascending=[True, False, True])
        ordered_groups.append(group)
        used_patches.update(group["patch"].astype(str).tolist())

    if not ordered_groups:
        return 0

    ordered = pd.concat(ordered_groups, ignore_index=True)
    rows = ordered.head(max_patches).to_dict("records")
    if not rows:
        return 0

    target_idx = next((i for i, row in enumerate(rows) if str(row.get("patch", "")) == fixed_second_page_second_patch), None)
    desired_idx = page_size + 1  # pagina 2, posicion 2
    if target_idx is not None and desired_idx < len(rows) and target_idx != desired_idx:
        target_row = rows.pop(target_idx)
        rows.insert(desired_idx, target_row)

    pages = int(np.ceil(len(rows) / page_size))
    for page_idx in range(pages):
        chunk = rows[page_idx * page_size : (page_idx + 1) * page_size]
        fig, axes = plt.subplots(2, len(chunk), figsize=(max(2.6 * len(chunk), 10), 6.5))
        axes = np.atleast_2d(axes)
        for col_idx, row in enumerate(chunk):
            patch_name = str(row["patch"])
            stem = Path(patch_name).stem
            rgb = read_patch_rgb(patch_name)
            water_mask = read_water_mask(stem)
            if rgb is not None:
                axes[0, col_idx].imshow(rgb)
            axes[0, col_idx].axis("off")
            axes[0, col_idx].set_title(f"{stem}", fontsize=8)
            if water_mask is not None:
                axes[1, col_idx].imshow(water_mask, cmap="Blues")
            axes[1, col_idx].axis("off")
            tags = str(row.get("scene_tags", "") or "")
            axes[1, col_idx].set_title(tags, fontsize=8)
        fig.suptitle(
            "RGB arriba y mascara de agua abajo | prioridad: costa -> barco -> nube",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(mask_dir / f"mascaras_agua_page_{page_idx + 1:02d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    return len(rows)


def analysis_by_date(master: pd.DataFrame, errors: pd.DataFrame, out_dir: Path) -> None:
    if errors.empty:
        return
    date_rows = []
    master = master.copy()
    master["date"] = master["patch"].astype(str).str[:8]
    for (method, date), group in errors.merge(master[["patch", "date"]], on="patch", how="left").groupby(["method", "date"]):
        date_rows.append({"method": method, "date": date, "n_errors": int(len(group)), "n_fp": int((group["error_type"] == "FP").sum()), "n_fn": int((group["error_type"] == "FN").sum())})
    pd.DataFrame(date_rows).to_csv(out_dir / "metrics_by_date.csv", index=False)


def analysis_by_quality(master: pd.DataFrame, errors: pd.DataFrame, out_dir: Path) -> None:
    quality_cols = [c for c in ["patch_quality", "image_quality", "cloud_frac", "land_frac"] if c in master.columns]
    if not quality_cols or errors.empty:
        return
    rows = []
    merged = errors.merge(master[["patch", *quality_cols]], on="patch", how="left")
    for quality_col in quality_cols:
        for (method, qval), group in merged.groupby(["method", quality_col]):
            rows.append({"method": method, "quality_col": quality_col, "quality_value": qval, "n_errors": int(len(group)), "n_fp": int((group["error_type"] == "FP").sum()), "n_fn": int((group["error_type"] == "FN").sum())})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_quality.csv", index=False)


def analysis_by_scene_tags(master: pd.DataFrame, errors: pd.DataFrame, out_dir: Path) -> None:
    tag_cols = [c for c in ["scene_tag", "scene_tags", "tags", "scene_type"] if c in master.columns]
    if not tag_cols or errors.empty:
        return
    rows = []
    merged = errors.merge(master[["patch", *tag_cols]], on="patch", how="left")
    for tag_col in tag_cols:
        for (method, tval), group in merged.groupby(["method", tag_col]):
            rows.append({"method": method, "tag_col": tag_col, "tag_value": tval, "n_errors": int(len(group)), "n_fp": int((group["error_type"] == "FP").sum()), "n_fn": int((group["error_type"] == "FN").sum())})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_scene_tags.csv", index=False)


def _build_tag_rows(master: pd.DataFrame, errors: pd.DataFrame, methods: list[str], keyword: str) -> dict | None:
    subset = master[master["scene_tags"].apply(lambda v: keyword.lower() in split_scene_tags(v))].copy()
    if subset.empty:
        return None
    method_count = len(methods)
    total_decisions = len(subset) * method_count
    err = errors[errors["patch"].isin(subset["patch"])]
    fp = int((err["error_type"] == "FP").sum())
    fn = int((err["error_type"] == "FN").sum())
    total_errors = len(err)
    return {
        "factor": keyword,
        "n_patches": int(len(subset)),
        "positivos": int((subset["label_binary"] == 1).sum()),
        "negativos": int((subset["label_binary"] == 0).sum()),
        "errores": int(total_errors),
        "fp": fp,
        "fn": fn,
        "error_rate": round(total_errors / total_decisions, 4) if total_decisions else 0.0,
    }


def build_diagnostics_summary(master: pd.DataFrame, errors: pd.DataFrame, limit: int = 10, full_master: pd.DataFrame | None = None) -> None:
    pixel_neg_by_patch_path = EVAL_PIXELWISE_OUT / "pixelwise_negatives_by_patch.csv"
    pixel_neg_by_patch = pd.read_csv(pixel_neg_by_patch_path) if pixel_neg_by_patch_path.exists() else pd.DataFrame()
    pixel_by_patch_path = EVAL_PIXELWISE_OUT / "pixelwise_metrics_by_patch.csv"
    pixel_by_patch = pd.read_csv(pixel_by_patch_path) if pixel_by_patch_path.exists() else pd.DataFrame()
    methods = sorted(errors["method"].dropna().unique().tolist())
    n_methods = len(methods)

    err_patch = (
        errors.groupby(["patch", "label"])
        .agg(
            n_errors=("method", "count"),
            n_methods=("method", "nunique"),
            n_fp=("error_type", lambda s: int((s == "FP").sum())),
            n_fn=("error_type", lambda s: int((s == "FN").sum())),
        )
        .reset_index()
    )
    err_patch["error_rate_over_methods"] = (err_patch["n_errors"] / n_methods).round(4)
    top_err_patch = err_patch.sort_values(["n_errors", "n_fp", "n_fn"], ascending=[False, False, False]).head(limit)

    neg = master[master["label_binary"] == 0].copy()
    usable_px_cols = [c for c in MASK_PX_COLS if c in neg.columns]
    neg[usable_px_cols] = neg[usable_px_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    neg["sum_pred_px_all_masks"] = neg[usable_px_cols].sum(axis=1)
    neg["max_pred_px_single_method"] = neg[usable_px_cols].max(axis=1)
    neg["n_methods_with_positive_px"] = (neg[usable_px_cols] > 0).sum(axis=1)
    top_negative_px = neg[
        ["patch", "sum_pred_px_all_masks", "max_pred_px_single_method", "n_methods_with_positive_px", "scene_tags", "image_quality"]
    ].sort_values(["sum_pred_px_all_masks", "n_methods_with_positive_px", "max_pred_px_single_method"], ascending=[False, False, False]).head(limit)

    pos = master[master["label_binary"] == 1].copy()
    fn_errors = errors[errors["error_type"] == "FN"].copy()
    pos_hard = (
        fn_errors.groupby("patch")
        .agg(n_fn=("method", "count"), n_methods=("method", "nunique"))
        .reset_index()
        .merge(pos[["patch", "scene_tags", "image_quality", "nc_px"]], on="patch", how="left")
    )
    pos_hard["fn_rate_over_methods"] = (pos_hard["n_fn"] / n_methods).round(4)
    top_positive_hard = pos_hard.sort_values(["n_fn", "nc_px"], ascending=[False, False]).head(limit)

    quality_rows = []
    for quality, group in master.groupby(master["image_quality"].fillna("NA")):
        total_decisions = len(group) * n_methods
        err = errors[errors["patch"].isin(group["patch"])]
        quality_rows.append(
            {
                "image_quality": quality,
                "n_patches": int(len(group)),
                "positivos": int((group["label_binary"] == 1).sum()),
                "negativos": int((group["label_binary"] == 0).sum()),
                "errores": int(len(err)),
                "fp": int((err["error_type"] == "FP").sum()),
                "fn": int((err["error_type"] == "FN").sum()),
                "error_rate": round(len(err) / total_decisions, 4) if total_decisions else 0.0,
            }
        )
    quality_df = pd.DataFrame(quality_rows).sort_values("error_rate", ascending=False)

    tag_factor_rows = []
    for keyword in ["barco", "nube", "nube_fina", "espuma", "estela", "agua_oscura", "agua_turbia", "costa", "brillo_solar", "olas", "corte_patch", "sombra_nube"]:
        row = _build_tag_rows(master, errors, methods, keyword)
        if row is not None:
            tag_factor_rows.append(row)
    tag_factor_df = pd.DataFrame(tag_factor_rows).sort_values("error_rate", ascending=False)
    barco_nube_df = tag_factor_df[tag_factor_df["factor"].isin(["barco", "nube"])].copy()

    method_summary = pd.read_csv(EVAL_ERROR_ANALYSIS_OUT / "error_summary.csv").copy()
    method_summary["error_rate_over_test"] = (method_summary["total_errors"] / len(master)).round(4)
    best_methods = method_summary.sort_values("total_errors", ascending=True).head(min(limit, 8))
    worst_methods = method_summary.sort_values("total_errors", ascending=False).head(min(limit, 8))

    top_neg_pixel_method = pd.DataFrame()
    if not pixel_neg_by_patch.empty:
        top_neg_pixel_method = pixel_neg_by_patch.sort_values(["pred_px", "fp_rate"], ascending=[False, False])[
            ["method", "patch", "pred_px", "fp_px", "fp_rate"]
        ].head(limit)

    top_positive_fp = pd.DataFrame()
    if not pixel_by_patch.empty:
        pos_fp = (
            pixel_by_patch.groupby("patch")
            .agg(
                sum_fp=("fp", "sum"),
                max_fp_single_method=("fp", "max"),
                n_methods_with_fp=("fp", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
                mean_dice=("dice_f1", "mean"),
                gt_px=("gt_px", "max"),
            )
            .reset_index()
            .merge(master[["patch", "scene_tags", "image_quality"]], on="patch", how="left")
            .sort_values(["sum_fp", "n_methods_with_fp", "max_fp_single_method"], ascending=[False, False, False])
            .head(limit)
        )
        top_positive_fp = pos_fp

    gallery_root = EVAL_ERROR_ANALYSIS_OUT / "error_examples"
    water_mask_source = full_master if full_master is not None else master
    n_water_mask_examples = render_water_mask_pages(water_mask_source, EVAL_ERROR_ANALYSIS_OUT, page_size=6, max_patches=18)
    top_err_items = [
        {
            "patch": row["patch"],
            "title": f"{idx}. {Path(str(row['patch'])).stem}",
            "subtitle": f"errors={row['n_errors']} fp={row['n_fp']} fn={row['n_fn']}",
        }
        for idx, (_, row) in enumerate(top_err_patch.iterrows(), start=1)
    ]
    top_neg_items = [
        {
            "patch": row["patch"],
            "title": f"{idx}. {Path(str(row['patch'])).stem}",
            "subtitle": f"sum_px={int(row['sum_pred_px_all_masks'])} methods={int(row['n_methods_with_positive_px'])}",
        }
        for idx, (_, row) in enumerate(top_negative_px.iterrows(), start=1)
    ]
    top_pos_items = [
        {
            "patch": row["patch"],
            "title": f"{idx}. {Path(str(row['patch'])).stem}",
            "subtitle": f"fn={int(row['n_fn'])} nc_px={int(row['nc_px'])}",
        }
        for idx, (_, row) in enumerate(top_positive_hard.iterrows(), start=1)
    ]
    top_pos_fp_items = [
        {
            "patch": row["patch"],
            "title": f"{idx}. {Path(str(row['patch'])).stem}",
            "subtitle": f"sum_fp={int(row['sum_fp'])} methods_fp={int(row['n_methods_with_fp'])} gt_px={int(row['gt_px'])}",
        }
        for idx, (_, row) in enumerate(top_positive_fp.iterrows(), start=1)
    ]
    render_patch_gallery(top_err_items, gallery_root / "patches_clasificados_mal", "patches_clasificados_mal", "Top patches clasificados mal")
    render_patch_gallery(top_neg_items, gallery_root / "negativos_con_mas_ruido", "negativos_con_mas_ruido", "Top negativos con mas pixeles espurios")
    render_patch_gallery(
        top_pos_items,
        gallery_root / "positivos_menos_acertados",
        "positivos_menos_acertados",
        "Top positivos menos acertados",
        include_gt_in_individual=True,
    )
    top_neg_pixel_items = [
        {
            "patch": row["patch"],
            "method": row["method"],
            "pred_px": int(row["pred_px"]),
        }
        for _, row in top_neg_pixel_method.iterrows()
    ]
    render_negative_method_gallery(
        top_neg_pixel_items,
        gallery_root / "negativos_mas_problematicos_pixelwise",
        "negativos_mas_problematicos_pixelwise",
        "Top negativos mas problematicos en segmentacion pixel-wise",
        page_size=5,
    )
    render_patch_gallery_pages_with_gt(
        top_pos_fp_items,
        gallery_root / "positivos_con_mas_fp",
        "positivos_con_mas_fp",
        "Top positivos con mas falsos positivos",
        page_size=5,
    )

    md = []
    md.append("# Diagnostics summary\n")
    md.append(
        "Este resumen condensa los hallazgos más útiles del bloque de diagnósticos sobre `test_final`. "
        f"Los conteos de error proceden de `{n_methods}` métodos incluidos en `error_cases.csv`.\n"
    )
    md.append("## 1. Patches clasificados erróneamente más veces\n")
    md.append(markdown_table(top_err_patch))
    md.append("## 2. Patches negativos con más píxeles positivos espurios\n")
    md.append(markdown_table(top_negative_px))
    md.append("## 3. Patches positivos menos acertados\n")
    md.append(markdown_table(top_positive_hard))
    md.append("## 4. Patches positivos con más falsos positivos en segmentación\n")
    md.append(markdown_table(top_positive_fp))
    md.append("## 5. Error agregado por calidad de imagen\n")
    md.append(markdown_table(quality_df))
    md.append("## 6. Factores de escena más asociados a error\n")
    md.append(markdown_table(tag_factor_df))
    md.append("## 7. ¿Se clasifican bien los patches con barco o con nube?\n")
    md.append(markdown_table(barco_nube_df))
    md.append("## 8. Métodos con menos y más errores\n")
    md.append("### Menos errores\n")
    md.append(markdown_table(best_methods))
    md.append("### Más errores\n")
    md.append(markdown_table(worst_methods))
    md.append("## 9. Casos negativos más problemáticos en segmentación pixel-wise\n")
    md.append(markdown_table(top_neg_pixel_method))
    md.append("## 10. Qué son las imágenes PNG de `error_examples`\n")
    md.append(
        "La carpeta `error_examples/` contiene únicamente subcarpetas específicas con escenas resumidas para memoria:\n"
        "- `patches_clasificados_mal/`\n"
        "- `negativos_con_mas_ruido/`\n"
        "- `negativos_mas_problematicos_pixelwise/`\n"
        "- `positivos_menos_acertados/`\n"
        "- `positivos_con_mas_fp/`\n"
    )
    md.append(
        "Dentro de cada una de esas subcarpetas se guardan las 10 escenas individuales y una imagen global `*_grid.png` con la composición conjunta. "
        "En `positivos_menos_acertados/` y `positivos_con_mas_fp/` las imágenes individuales muestran `RGB` y `GT` lado a lado.\n"
    )
    md.append("## 11. Máscaras de agua priorizadas por tags\n")
    if n_water_mask_examples > 0:
        md.append(
            f"Se generó la carpeta `mascaras/` dentro de `diagnostics/` con páginas de 6 pares `RGB + máscara de agua`, "
            f"priorizando escenas con etiquetas `nube`, `costa` o `barco`. En total se incluyeron {n_water_mask_examples} patches.\n"
        )
    else:
        md.append("_No se encontraron patches con máscara de agua y tags priorizados para esta exportación._\n")
    DIAGNOSTICS_SUMMARY_OUT.write_text("\n".join(md), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", nargs="+", default=list(ALL_METHODS.keys()))
    parser.add_argument("--errors", choices=["fp", "fn", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--by-date", action="store_true")
    parser.add_argument("--by-quality", action="store_true")
    parser.add_argument("--by-scene-tags", action="store_true")
    parser.add_argument("--summary-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PREDICTIONS_MASTER_PATH.exists():
        raise FileNotFoundError(f"No existe {PREDICTIONS_MASTER_PATH}. Ejecuta 03_unify_predictions.py primero.")
    EVAL_ERROR_ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    master_all = pd.read_csv(PREDICTIONS_MASTER_PATH)
    master = master_all[master_all["eval_subset"] == "test_final"].copy()

    valid_keys = [key for key in args.method if key in ALL_METHODS]
    errors = classify_errors(master, valid_keys, args.errors)
    summary = (
        errors.pivot_table(index="method", columns="error_type", values="patch", aggfunc="count", fill_value=0)
        .reset_index()
        if not errors.empty
        else pd.DataFrame(columns=["method", "FP", "FN"])
    )
    if not summary.empty:
        for col in ["FP", "FN"]:
            if col not in summary.columns:
                summary[col] = 0
        summary["total_errors"] = summary["FP"] + summary["FN"]

    errors.to_csv(EVAL_ERROR_ANALYSIS_OUT / "error_cases.csv", index=False)
    summary.to_csv(EVAL_ERROR_ANALYSIS_OUT / "error_summary.csv", index=False)
    if args.by_date:
        analysis_by_date(master, errors, EVAL_ERROR_ANALYSIS_OUT)
    if args.by_quality:
        analysis_by_quality(master, errors, EVAL_ERROR_ANALYSIS_OUT)
    if args.by_scene_tags:
        analysis_by_scene_tags(master, errors, EVAL_ERROR_ANALYSIS_OUT)
    build_diagnostics_summary(master, errors, limit=args.summary_limit, full_master=master_all)
    print(summary.to_string(index=False) if not summary.empty else "Sin errores.")


if __name__ == "__main__":
    main()
