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

import pandas as pd
import rasterio
from rasterio.warp import transform

from src.common.config import (
    DATASET_METADATA_GROUPED_PATH,
    DATASET_METADATA_PATH,
    PATCHES_DIR,
)
from src.common.pipeline_utils import iter_patch_files


DIFFICULTIES = {"CLARO", "DIFICIL", "DUDOSO"}

BASE_METADATA_COLUMNS = [
    "patch", "patch_path", "mask_path", "date", "year", "month",
    "label", "label_binary", "expected_gt_px", "mask_gt_px",
    "name_mask_match", "original_difficulty", "is_positive", "is_negative",
    "center_x", "center_y", "center_lon", "center_lat",
]
ANNOTATION_COLUMNS = [
    "manual_decision", "manual_confidence", "image_quality",
    "scene_tags", "notes", "annotated_at",
]
TAG_COLUMNS = [
    "tag_nube", "tag_nube_fina", "tag_estela", "tag_barco", "tag_costa",
    "tag_brillo_solar", "tag_agua_oscura", "tag_agua_muy_oscura",
    "tag_agua_turbia", "tag_posible_residuo", "tag_agua_limpia",
    "tag_filamento_visible", "tag_olas", "tag_espuma",
    "tag_corte_patch", "tag_sombra_nube",
]
GROUP_COLUMNS = ["group_date", "group_month", "group_year", "group_id"]
PRESERVED_METADATA_COLUMNS = ANNOTATION_COLUMNS + TAG_COLUMNS


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "si", "s"}


def _clean_number_text(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def parse_patch_name(path: Path) -> dict:
    parts = path.stem.split("_")
    date = parts[0] if parts else ""
    label = parts[1] if len(parts) > 1 and parts[1] in {"SI", "NO"} else ""
    expected_gt_px = 0
    if label == "SI" and len(parts) > 2:
        match = re.search(r"\d+", parts[2])
        expected_gt_px = int(match.group(0)) if match else 0
    difficulty = ""
    for part in parts:
        if part in DIFFICULTIES:
            difficulty = part
            break
    return {
        "date": date,
        "year": int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else "",
        "month": int(date[4:6]) if len(date) >= 6 and date[4:6].isdigit() else "",
        "label": label,
        "label_binary": 1 if label == "SI" else 0,
        "expected_gt_px": expected_gt_px,
        "original_difficulty": difficulty,
    }


def mask_pixel_count(mask_path: Path) -> int:
    if not mask_path.exists():
        return 0
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int((mask > 0).sum())


def patch_center(path: Path) -> dict:
    with rasterio.open(path) as src:
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2.0
        center_y = (bounds.bottom + bounds.top) / 2.0
        lon, lat = transform(src.crs, "EPSG:4326", [center_x], [center_y])
    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_lon": float(lon[0]),
        "center_lat": float(lat[0]),
    }


def load_existing_annotations(metadata_path: Path, grouped_path: Path) -> pd.DataFrame:
    frames = []
    for path in [metadata_path, grouped_path]:
        if not path.exists():
            continue
        df = pd.read_csv(path).fillna("")
        if "patch" not in df.columns:
            continue
        keep = ["patch", *[c for c in PRESERVED_METADATA_COLUMNS if c in df.columns]]
        frames.append(df[keep].copy())
    if not frames:
        return pd.DataFrame(columns=["patch", *PRESERVED_METADATA_COLUMNS])
    annotations = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["patch"], keep="last")
    for col in PRESERVED_METADATA_COLUMNS:
        if col not in annotations.columns:
            annotations[col] = ""
    return annotations[["patch", *PRESERVED_METADATA_COLUMNS]]


def build_base_metadata() -> pd.DataFrame:
    rows = []
    for patch_path in iter_patch_files(PATCHES_DIR):
        parsed = parse_patch_name(patch_path)
        mask_path = patch_path.with_name(f"{patch_path.stem}_mask.tif")
        expected_gt_px = int(parsed["expected_gt_px"])
        mask_gt_px = mask_pixel_count(mask_path)
        label = parsed["label"]
        rows.append(
            {
                "patch": patch_path.name,
                "patch_path": str(patch_path),
                "mask_path": str(mask_path) if mask_path.exists() else "",
                "date": parsed["date"],
                "year": parsed["year"],
                "month": parsed["month"],
                "label": label,
                "label_binary": parsed["label_binary"],
                "expected_gt_px": expected_gt_px,
                "mask_gt_px": mask_gt_px,
                "name_mask_match": bool(expected_gt_px == mask_gt_px),
                "original_difficulty": parsed["original_difficulty"],
                "is_positive": bool(label == "SI"),
                "is_negative": bool(label == "NO"),
                **patch_center(patch_path),
            }
        )
    if not rows:
        return pd.DataFrame(columns=BASE_METADATA_COLUMNS)
    return pd.DataFrame(rows).sort_values(["label", "date", "patch"]).reset_index(drop=True)


def add_metadata_annotations(metadata: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.merge(annotations, on="patch", how="left")
    for col in PRESERVED_METADATA_COLUMNS:
        if col not in metadata.columns:
            metadata[col] = ""
        metadata[col] = metadata[col].fillna("")
    for col in TAG_COLUMNS:
        metadata[col] = metadata[col].map(
            lambda v: "1" if _as_bool(v) else ("0" if str(v).strip() else "")
        )
    metadata["manual_confidence"] = metadata["manual_confidence"].map(_clean_number_text)
    return metadata


def add_group_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    date = metadata["date"].astype(str).str.replace("-", "", regex=False).str[:8]
    metadata["group_date"] = date
    metadata["group_month"] = date.str[:6]
    metadata["group_year"] = date.str[:4]
    metadata["group_id"] = metadata["group_date"]
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir metadata del dataset agrupada por fecha")
    parser.add_argument("--metadata-out", type=Path, default=DATASET_METADATA_PATH)
    parser.add_argument("--grouped-out", type=Path, default=DATASET_METADATA_GROUPED_PATH)
    args = parser.parse_args()

    print(f"Escaneando patches en: {PATCHES_DIR}")
    base = build_base_metadata()
    if base.empty:
        print(f"WARN: No se encontraron patches en {PATCHES_DIR}.")
        return

    annotations = load_existing_annotations(args.metadata_out, args.grouped_out)
    metadata = add_metadata_annotations(base, annotations)

    base_out = metadata[[*BASE_METADATA_COLUMNS, *PRESERVED_METADATA_COLUMNS]].copy()
    grouped_out = add_group_columns(base_out)
    grouped_out = grouped_out[[*BASE_METADATA_COLUMNS, *PRESERVED_METADATA_COLUMNS, *GROUP_COLUMNS]]

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    base_out.to_csv(args.metadata_out, index=False)
    grouped_out.to_csv(args.grouped_out, index=False)

    n_si = int(base_out["is_positive"].sum())
    n_no = int(base_out["is_negative"].sum())
    n_dates = int(base_out["date"].nunique())
    print(f"Metadata guardada: {args.metadata_out}")
    print(f"Metadata con grupos: {args.grouped_out}")
    print(f"Patches={len(base_out)} | SI={n_si} | NO={n_no} | fechas={n_dates}")


if __name__ == "__main__":
    main()
