from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import argparse

import pandas as pd

from src.common.config import (
    CSV_CANDIDATES,
    DATASET_FILTERS_PREVIOUS_OUT,
    KML_PATH,
    XLSX_PATH,
    ensure_output_dirs,
)
from src.common.geo_utils import filter_points_to_kml, parse_kml_polygons


DEFAULT_MIN_DISTANCE_TO_LAND_M = 1280.0
DEFAULT_MIN_PIXELS = 6
REQUIRED_COLUMNS = [
    "Latitude",
    "Longitude",
    "Date",
    "Year",
    "Month",
    "CodeT",
    "Pixels per LW",
    "Distance to land (m)",
]
CANDIDATE_SELECTION_OUT = DATASET_FILTERS_PREVIOUS_OUT


def load_nature_excel(path) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel Nature: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_candidates(df: pd.DataFrame, min_distance_to_land_m: float, min_pixels: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    polygons = parse_kml_polygons(KML_PATH)
    df_bbox, in_area, _bbox = filter_points_to_kml(df, polygons)
    candidates = in_area[
        (in_area["Distance to land (m)"] >= min_distance_to_land_m)
        & (in_area["Pixels per LW"] >= min_pixels)
    ].copy()
    candidates["original_index"] = candidates.index
    candidates["datetime_str"] = candidates["Date"].dt.strftime("%Y%m%d") + candidates["CodeT"].astype(str)
    candidates = candidates.sort_values("Pixels per LW", ascending=False).reset_index(drop=True)
    return df_bbox, in_area, candidates


def write_summary(
    df: pd.DataFrame,
    df_bbox: pd.DataFrame,
    in_area: pd.DataFrame,
    candidates: pd.DataFrame,
    min_distance_to_land_m: float,
    min_pixels: int,
) -> None:
    CANDIDATE_SELECTION_OUT.mkdir(parents=True, exist_ok=True)

    n_original = len(df)
    n_inside_bbox = len(df_bbox)
    n_inside_kml = len(in_area)
    n_after_distance = len(in_area[in_area["Distance to land (m)"] >= min_distance_to_land_m])
    n_after_pixels = len(candidates)

    lines = [
        "# Candidate Selection Summary",
        "",
        f"- Excel: {XLSX_PATH}",
        f"- KML area: {KML_PATH}",
        "",
        "## Counts",
        f"- n_original: {n_original}",
        f"- n_inside_bbox: {n_inside_bbox}",
        f"- n_inside_kml: {n_inside_kml}",
        f"- n_after_distance: {n_after_distance}",
        f"- n_after_pixels: {n_after_pixels}",
        "",
        "## Thresholds Used",
        f"- min_distance_to_land_m: {min_distance_to_land_m:g}",
        f"- min_pixels_per_lw: {min_pixels}",
        "",
        "## Candidates By Year",
        candidates["Year"].value_counts().sort_index().to_string() if not candidates.empty else "No candidates",
        "",
        "## Pixels Per LW",
        candidates["Pixels per LW"].describe().to_string() if not candidates.empty else "No candidates",
        "",
        "## Distance To Land",
        candidates["Distance to land (m)"].describe().to_string() if not candidates.empty else "No candidates",
    ]
    (CANDIDATE_SELECTION_OUT / "candidate_selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rejected_by_filter(df: pd.DataFrame, df_bbox: pd.DataFrame, in_area: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame recording which rows were rejected and at which filter step."""
    all_indices = set(df.index)
    bbox_indices = set(df_bbox.index)
    kml_indices = set(in_area.index)
    candidate_indices = set(candidates["original_index"])

    rows = []
    for idx in all_indices:
        if idx not in bbox_indices:
            rows.append({"original_index": idx, "filter": "bbox"})
        elif idx not in kml_indices:
            rows.append({"original_index": idx, "filter": "kml_polygon"})
        elif idx not in candidate_indices:
            rows.append({"original_index": idx, "filter": "distance_or_pixels"})

    return pd.DataFrame(rows, columns=["original_index", "filter"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-distance-to-land-m", type=float, default=DEFAULT_MIN_DISTANCE_TO_LAND_M)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    args = parser.parse_args()

    ensure_output_dirs()
    df = load_nature_excel(XLSX_PATH)
    df_bbox, in_area, candidates = build_candidates(
        df,
        min_distance_to_land_m=args.min_distance_to_land_m,
        min_pixels=args.min_pixels,
    )

    candidates.to_csv(CANDIDATE_SELECTION_OUT / "candidates_filtered.csv", index=False)
    CSV_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CSV_CANDIDATES, index=False)

    rejected = build_rejected_by_filter(df, df_bbox, in_area, candidates)
    rejected.to_csv(CANDIDATE_SELECTION_OUT / "rejected_candidates_by_filter.csv", index=False)

    write_summary(df, df_bbox, in_area, candidates, args.min_distance_to_land_m, args.min_pixels)

    print("=== Nature candidates ===")
    print(f"Excel rows (n_original): {len(df)}")
    print(f"Inside KML bbox (n_inside_bbox): {len(df_bbox)}")
    print(f"Inside KML polygon (n_inside_kml): {len(in_area)}")
    print(f"After distance filter (n_after_distance): {len(in_area[in_area['Distance to land (m)'] >= args.min_distance_to_land_m])}")
    print(f"After pixels filter (n_after_pixels): {len(candidates)}")
    print(f"Saved: {CANDIDATE_SELECTION_OUT / 'candidates_filtered.csv'}")
    print(f"Rejected: {CANDIDATE_SELECTION_OUT / 'rejected_candidates_by_filter.csv'}")
    print(f"Summary: {CANDIDATE_SELECTION_OUT / 'candidate_selection_summary.md'}")


if __name__ == "__main__":
    main()
