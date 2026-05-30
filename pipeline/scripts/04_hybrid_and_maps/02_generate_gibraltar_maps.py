from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import hashlib
import io
import os
import sys
import base64
import math
from pathlib import Path


def _configure_gdal_env() -> None:
    candidates: list[Path] = []
    for prefix in [os.environ.get("CONDA_PREFIX"), sys.prefix, sys.base_prefix]:
        if not prefix:
            continue
        base = Path(prefix)
        candidates.extend(
            [
                base / "Library" / "share" / "gdal",
                base / "share" / "gdal",
            ]
        )
        candidates.extend(
            [
                base / "Library" / "share" / "proj",
                base / "share" / "proj",
            ]
        )

    if not os.environ.get("GDAL_DATA"):
        for path in candidates:
            if path.name == "gdal" and (path / "gcs.csv").exists():
                os.environ["GDAL_DATA"] = str(path)
                break

    if not os.environ.get("PROJ_LIB"):
        for path in candidates:
            if path.name == "proj" and path.exists():
                os.environ["PROJ_LIB"] = str(path)
                break


_configure_gdal_env()

import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window, bounds as window_bounds

from src.common.config import (
    CSV_CANDIDATES,
    DATASET_METADATA_GROUPED_PATH,
    HYBRID_PHASE_OUT,
    HYBRID_MASKS_ROOT,
    MAPS_PHASE_OUT,
)


OUT = MAPS_PHASE_OUT
HYBRID_PREDS = HYBRID_PHASE_OUT / "hybrid_predictions.csv"
GIBRALTAR_LAT = 35.95
GIBRALTAR_LON = -5.35
VARIANT_COLORS = {
    "hybrid_profile_sensitive": (46, 204, 113, 150),
    "hybrid_profile_balanced": (52, 152, 219, 160),
    "hybrid_profile_conservative": (230, 126, 34, 170),
    "final_hybrid": (142, 68, 173, 170),
}
BALANCED_PROBABILITY_STACK = {
    "hybrid_profile_sensitive": (255, 196, 79, 120),
    "hybrid_profile_balanced": (255, 127, 39, 170),
    "hybrid_profile_conservative": (214, 39, 40, 230),
}


def _variant_mask_dirs(df: pd.DataFrame) -> dict[str, Path]:
    variant_dirs: dict[str, Path] = {}
    profile_map = {
        "hybrid_profile_sensitive": HYBRID_MASKS_ROOT / "profile_sensitive",
        "hybrid_profile_balanced": HYBRID_MASKS_ROOT / "profile_balanced",
        "hybrid_profile_conservative": HYBRID_MASKS_ROOT / "profile_conservative",
        "final_hybrid": HYBRID_MASKS_ROOT / "profile_balanced",
    }
    for variant, mask_dir in profile_map.items():
        pred_col = "final_hybrid_pred" if variant == "final_hybrid" else f"{variant}_pred"
        if pred_col in df.columns and mask_dir.exists():
            variant_dirs[variant] = mask_dir
    return variant_dirs


def _stable_jitter(text: str, scale: float) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale


def _check_folium() -> bool:
    try:
        import folium  # noqa: F401
        return True
    except ImportError:
        return False


def _positive_rank_from_patch(patch_name: str):
    parts = str(patch_name).replace(".tif", "").split("_")
    if len(parts) >= 4 and parts[1] == "SI":
        try:
            return int(parts[3])
        except ValueError:
            return pd.NA
    return pd.NA


def _patch_center_lonlat(path_value):
    if pd.isna(path_value) or not str(path_value).strip():
        return pd.NA, pd.NA
    path = Path(str(path_value))
    if not path.exists():
        return pd.NA, pd.NA
    try:
        with rasterio.open(path) as src:
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.bottom + src.bounds.top) / 2
            if src.crs and str(src.crs).upper() != "EPSG:4326":
                lon, lat = transform(src.crs, "EPSG:4326", [cx], [cy])
                return float(lon[0]), float(lat[0])
            return float(cx), float(cy)
    except Exception:
        return pd.NA, pd.NA


def _mask_rgba(mask: np.ndarray, rgba: tuple[int, int, int, int]) -> np.ndarray:
    canvas = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    active = mask.astype(bool)
    if active.any():
        canvas[active, 0] = rgba[0]
        canvas[active, 1] = rgba[1]
        canvas[active, 2] = rgba[2]
        canvas[active, 3] = rgba[3]
    return canvas


def _load_mask_overlay(path: Path, rgba: tuple[int, int, int, int]):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1) > 0
    if not arr.any():
        return None
    rows, cols = np.where(arr)
    row_min = int(rows.min())
    row_max = int(rows.max()) + 1
    col_min = int(cols.min())
    col_max = int(cols.max()) + 1

    cropped_mask = arr[row_min:row_max, col_min:col_max]
    rgba_img = _mask_rgba(cropped_mask, rgba)

    with rasterio.open(path) as src:
        win = Window(col_off=col_min, row_off=row_min, width=col_max - col_min, height=row_max - row_min)
        crop_bounds_native = window_bounds(win, src.transform)
        if src.crs:
            crop_bounds = transform_bounds(src.crs, "EPSG:4326", *crop_bounds_native)
        else:
            crop_bounds = crop_bounds_native

    south, west, north, east = crop_bounds[1], crop_bounds[0], crop_bounds[3], crop_bounds[2]
    buffer = io.BytesIO()
    Image.fromarray(rgba_img, mode="RGBA").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return data_url, [[south, west], [north, east]]


def _read_binary_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1) > 0
    if not arr.any():
        return None
    return arr


def _mask_bounds_from_array(path: Path, arr: np.ndarray):
    rows, cols = np.where(arr)
    row_min = int(rows.min())
    row_max = int(rows.max()) + 1
    col_min = int(cols.min())
    col_max = int(cols.max()) + 1
    with rasterio.open(path) as src:
        win = Window(col_off=col_min, row_off=row_min, width=col_max - col_min, height=row_max - row_min)
        crop_bounds_native = window_bounds(win, src.transform)
        if src.crs:
            crop_bounds = transform_bounds(src.crs, "EPSG:4326", *crop_bounds_native)
        else:
            crop_bounds = crop_bounds_native
    south, west, north, east = crop_bounds[1], crop_bounds[0], crop_bounds[3], crop_bounds[2]
    return (row_min, row_max, col_min, col_max), [[south, west], [north, east]]


def _composite_balanced_overlay(df: pd.DataFrame, patch_name: str):
    variant_dirs = _variant_mask_dirs(df)
    base_variant = "hybrid_profile_balanced"
    base_dir = variant_dirs.get(base_variant)
    if base_dir is None:
        return None

    stem = Path(str(patch_name)).stem
    base_path = base_dir / f"{stem}_mask.tif"
    base_mask = _read_binary_mask(base_path)
    if base_mask is None:
        return None

    composite = np.zeros((*base_mask.shape, 4), dtype=np.uint8)
    any_active = np.zeros(base_mask.shape, dtype=bool)

    # Paint from low to high confidence so later overlays intensify the same area.
    for variant in ["hybrid_profile_sensitive", "hybrid_profile_balanced", "hybrid_profile_conservative"]:
        mask_dir = variant_dirs.get(variant)
        if mask_dir is None:
            continue
        mask = _read_binary_mask(mask_dir / f"{stem}_mask.tif")
        if mask is None:
            continue
        any_active |= mask
        rgba = BALANCED_PROBABILITY_STACK[variant]
        composite[mask, 0] = rgba[0]
        composite[mask, 1] = rgba[1]
        composite[mask, 2] = rgba[2]
        composite[mask, 3] = rgba[3]

    if not any_active.any():
        return None

    crop_idx, mask_bounds = _mask_bounds_from_array(base_path, any_active)
    row_min, row_max, col_min, col_max = crop_idx
    cropped = composite[row_min:row_max, col_min:col_max]
    buffer = io.BytesIO()
    Image.fromarray(cropped, mode="RGBA").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return data_url, mask_bounds


def _load_patch_bounds(path: Path):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        if src.crs:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        else:
            bounds = src.bounds
    south, west, north, east = bounds[1], bounds[0], bounds[3], bounds[2]
    return [[south, west], [north, east]]


def _load_data() -> pd.DataFrame:
    metadata = pd.read_csv(DATASET_METADATA_GROUPED_PATH)
    hybrid = pd.read_csv(HYBRID_PREDS)
    candidates = pd.read_csv(CSV_CANDIDATES)

    if "Date" in candidates.columns:
        candidates["date"] = pd.to_datetime(candidates["Date"]).dt.strftime("%Y%m%d")
    elif "datetime_str" in candidates.columns:
        candidates["date"] = candidates["datetime_str"].str[:8]

    candidates_by_rank = candidates.sort_values("Pixels per LW", ascending=False).reset_index(drop=True).copy()
    candidates_by_rank["positive_rank"] = candidates_by_rank.index.astype("Int64")
    candidates_by_rank = candidates_by_rank[
        ["positive_rank", "Latitude", "Longitude", "Distance to land (m)"]
    ].rename(
        columns={
            "Latitude": "lat_rank",
            "Longitude": "lon_rank",
            "Distance to land (m)": "distance_to_land_m",
        }
    )
    latlon_by_date = (
        candidates.groupby("date")[["Latitude", "Longitude"]]
        .mean()
        .reset_index()
        .rename(columns={"Latitude": "lat_date", "Longitude": "lon_date"})
    )

    metadata["date"] = metadata["date"].astype(str)
    if {"center_lat", "center_lon"}.issubset(metadata.columns):
        metadata["lat_metadata"] = pd.to_numeric(metadata["center_lat"], errors="coerce")
        metadata["lon_metadata"] = pd.to_numeric(metadata["center_lon"], errors="coerce")
    else:
        metadata["lat_metadata"] = pd.NA
        metadata["lon_metadata"] = pd.NA

    missing_metadata_centres = metadata["lat_metadata"].isna() | metadata["lon_metadata"].isna()
    metadata["lon_geotiff"] = pd.NA
    metadata["lat_geotiff"] = pd.NA
    if missing_metadata_centres.any() and "patch_path" in metadata.columns:
        centers = metadata.loc[missing_metadata_centres, "patch_path"].apply(_patch_center_lonlat)
        metadata.loc[missing_metadata_centres, "lon_geotiff"] = [lon for lon, _lat in centers]
        metadata.loc[missing_metadata_centres, "lat_geotiff"] = [lat for _lon, lat in centers]

    metadata["positive_rank"] = metadata["patch"].map(_positive_rank_from_patch).astype("Int64")
    df = metadata.merge(candidates_by_rank, on="positive_rank", how="left")
    df = df.merge(latlon_by_date, on="date", how="left")
    df["lat"] = (
        df["lat_metadata"]
        .combine_first(df["lat_geotiff"])
        .combine_first(df["lat_rank"])
        .combine_first(df["lat_date"])
    )
    df["lon"] = (
        df["lon_metadata"]
        .combine_first(df["lon_geotiff"])
        .combine_first(df["lon_rank"])
        .combine_first(df["lon_date"])
    )
    no_latlon = df["lat"].isna()
    if no_latlon.any():
        for idx, row in df.loc[no_latlon].iterrows():
            patch_name = str(row.get("patch", idx))
            df.at[idx, "lat"] = GIBRALTAR_LAT + _stable_jitter(patch_name + ":lat", 0.3)
            df.at[idx, "lon"] = GIBRALTAR_LON + _stable_jitter(patch_name + ":lon", 0.5)

    hybrid_cols = ["patch"] + [c for c in hybrid.columns if c not in df.columns]
    return df.merge(hybrid[hybrid_cols], on="patch", how="left")


def _selected_variant(df: pd.DataFrame) -> str | None:
    if "selected_mask_variant" not in df.columns or df.empty:
        return None
    series = df["selected_mask_variant"].dropna().astype(str)
    if series.empty:
        return None
    return series.iloc[0]


def _popup_html(row) -> str:
    lines = []
    for key in [
        "patch",
        "date",
        "label",
        "detector_score",
        "detector_pred",
        "selected_mask_variant_sensitive",
        "selected_mask_variant_balanced",
        "selected_mask_variant_conservative",
        "hybrid_profile_sensitive_pred",
        "hybrid_profile_balanced_pred",
        "hybrid_profile_conservative_pred",
        "hybrid_profile_sensitive_px",
        "hybrid_profile_balanced_px",
        "hybrid_profile_conservative_px",
        "final_hybrid_pred",
        "final_hybrid_px",
    ]:
        val = row.get(key, "")
        if pd.notna(val) and str(val).strip():
            lines.append(f"<b>{key}</b>: {val}")
    return "<br>".join(lines)


def _base_map():
    import folium

    return folium.Map(location=[GIBRALTAR_LAT, GIBRALTAR_LON], zoom_start=9, tiles="CartoDB positron")


def _detector_radius(score: float, score_min: float, score_max: float) -> float:
    score = max(0.0, float(score))
    score_min = max(0.0, float(score_min))
    score_max = max(score_min, float(score_max))
    if score_max <= score_min:
        return 6.0
    frac = (score - score_min) / (score_max - score_min)
    frac = min(1.0, max(0.0, frac))
    return 3.0 + 6.0 * frac


def _add_detector_layer(map_obj, df: pd.DataFrame):
    import folium

    pred_col = "detector_pred" if "detector_pred" in df.columns else "external_detector_pred"
    positives = df[df[pred_col].fillna(0).astype(int) == 1].copy()
    score_series = pd.to_numeric(positives.get("detector_score"), errors="coerce").fillna(0.0)
    score_min = float(score_series.min()) if not score_series.empty else 0.0
    score_max = float(score_series.max()) if not score_series.empty else 0.0
    for _, row in positives.iterrows():
        score = float(row.get("detector_score", 0.0) or 0.0)
        radius = _detector_radius(score, score_min, score_max)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="#d62728",
            fill=True,
            fill_opacity=0.18,
            weight=1,
            popup=folium.Popup(_popup_html(row), max_width=350),
            tooltip=f"detector={score:.3f}",
        ).add_to(map_obj)


def _add_hybrid_mask_layer(map_obj, df: pd.DataFrame, variant: str):
    import folium

    mask_dir = _variant_mask_dirs(df)[variant]
    rgba = VARIANT_COLORS.get(variant, (52, 73, 94, 170))
    pred_col = "final_hybrid_pred" if variant == "final_hybrid" else f"{variant}_pred"
    for _, row in df[df[pred_col].fillna(0).astype(int) == 1].iterrows():
        stem = Path(str(row["patch"])).stem
        mask_path = mask_dir / f"{stem}_mask.tif"
        overlay = _load_mask_overlay(mask_path, rgba)
        if overlay is None:
            continue
        patch_bounds = _load_patch_bounds(mask_path)
        if patch_bounds is None:
            continue
        image_data, mask_bounds = overlay
        folium.raster_layers.ImageOverlay(
            image=image_data,
            bounds=mask_bounds,
            opacity=1.0,
            interactive=True,
            cross_origin=False,
        ).add_to(map_obj)
        folium.Rectangle(
            bounds=patch_bounds,
            color="#333333",
            weight=1,
            fill=False,
            popup=folium.Popup(_popup_html(row), max_width=350),
        ).add_to(map_obj)


def _add_balanced_probability_layer(map_obj, df: pd.DataFrame):
    import folium

    pred_cols = [
        "hybrid_profile_sensitive_pred",
        "hybrid_profile_balanced_pred",
        "hybrid_profile_conservative_pred",
    ]
    usable_cols = [col for col in pred_cols if col in df.columns]
    if not usable_cols:
        return

    active_rows = df[df[usable_cols].fillna(0).astype(int).max(axis=1) == 1]
    for _, row in active_rows.iterrows():
        overlay = _composite_balanced_overlay(df, row["patch"])
        if overlay is None:
            continue
        mask_dir = _variant_mask_dirs(df).get("hybrid_profile_balanced")
        if mask_dir is None:
            continue
        stem = Path(str(row["patch"])).stem
        patch_bounds = _load_patch_bounds(mask_dir / f"{stem}_mask.tif")
        if patch_bounds is None:
            continue
        image_data, mask_bounds = overlay
        folium.raster_layers.ImageOverlay(
            image=image_data,
            bounds=mask_bounds,
            opacity=1.0,
            interactive=True,
            cross_origin=False,
        ).add_to(map_obj)
        folium.Rectangle(
            bounds=patch_bounds,
            color="#333333",
            weight=1,
            fill=False,
            popup=folium.Popup(_popup_html(row), max_width=350),
        ).add_to(map_obj)


def _make_detector_candidates_map(df: pd.DataFrame):
    import folium

    m = _base_map()
    _add_detector_layer(m, df)
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px;border:1px solid #ccc;font-size:13px;line-height:1.8">
      <b>Detector externo</b><br>
      <span style="color:#d62728">●</span> patch candidato<br>
      Tamaño del punto proporcional al score
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def _make_variant_map(df: pd.DataFrame, variant: str):
    import folium

    m = _base_map()
    _add_detector_layer(m, df)
    if variant == "final_hybrid":
        _add_balanced_probability_layer(m, df)
    else:
        _add_hybrid_mask_layer(m, df, variant)
    label = (
        variant.replace("hybrid_profile_", "profile_")
        .replace("final_hybrid", "final")
    )
    extra = ""
    if variant == "final_hybrid":
        extra = (
            "<br>Intensidad tipo probabilidad:"
            "<br><span style=\"color:#ffc44f\">■</span> sensitive"
            "<br><span style=\"color:#ff7f27\">■</span> balanced"
            "<br><span style=\"color:#d62728\">■</span> conservative"
        )
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px;border:1px solid #ccc;font-size:13px;line-height:1.8">
      <b>Mapa híbrido {label}</b><br>
      <span style="color:#d62728">●</span> detector positivo<br>
      <span style="color:#333333">▭</span> contorno del patch<br>
      Máscara híbrida insertada geográficamente
      {extra}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def main() -> None:
    if not _check_folium():
        print("ERROR: folium no está instalado. Ejecuta: pip install folium")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not HYBRID_PREDS.exists():
        print(f"ERROR: No se encontró {HYBRID_PREDS}. Ejecuta primero el script 01_build_hybrid_detector_segmenter.py")
        return

    df = _load_data()
    variant_mask_dirs = _variant_mask_dirs(df)
    selected_variant = "final_hybrid" if "final_hybrid_pred" in df.columns else _selected_variant(df)
    generated = []

    for old_map in OUT.glob("gibraltar_*_map.html"):
        old_map.unlink()

    detector_map = _make_detector_candidates_map(df)
    detector_path = OUT / "gibraltar_detector_candidates_map.html"
    detector_map.save(str(detector_path))
    generated.append(detector_path.name)

    for variant in variant_mask_dirs:
        if variant == "final_hybrid":
            continue
        out_name = f"gibraltar_{variant}_map.html"
        map_obj = _make_variant_map(df, variant)
        out_path = OUT / out_name
        map_obj.save(str(out_path))
        generated.append(out_name)

    if selected_variant in variant_mask_dirs and selected_variant != "final_hybrid":
        final_map = _make_variant_map(df, selected_variant)
        final_path = OUT / "gibraltar_hybrid_final_map.html"
        final_map.save(str(final_path))
        generated.append(final_path.name)
    elif "final_hybrid" in variant_mask_dirs:
        final_map = _make_variant_map(df, "final_hybrid")
        final_path = OUT / "gibraltar_hybrid_final_map.html"
        final_map.save(str(final_path))
        generated.append(final_path.name)

    summary = "\n".join(
        [
            "# Hybrid maps summary",
            "",
            "## Mapas generados",
            *[f"- `{name}`" for name in generated],
            "",
            "## Variante final",
            f"- `{selected_variant}`" if selected_variant else "- no disponible",
            "",
            "## Interpretación",
            "- capa 1: patches donde el detector externo da positivo",
            "- capa 2: máscara híbrida reinsertada dentro del bounding box del patch",
            "- la forma visible procede de la segmentación real, no de conteos agregados",
            "- en `gibraltar_hybrid_final_map.html`, la máscara se compone por intensidades: sensitive tenue, balanced media y conservative intensa",
        ]
    )
    (OUT / "maps_summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"Mapas híbridos generados en: {OUT}")


if __name__ == "__main__":
    main()
