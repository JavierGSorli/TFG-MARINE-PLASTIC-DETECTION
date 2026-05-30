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
import time
import tkinter as tk
from datetime import datetime
from math import ceil
from tkinter import ttk

import numpy as np
import openeo
import pandas as pd
import rasterio
import requests
import xarray as xr
from PIL import Image, ImageTk
from rasterio.warp import transform
from scipy.signal import fftconvolve

from src.common.config import (
    CSV_CANDIDATES,
    CSV_DOWNLOAD_FAILURES,
    DATASET_DOWNLOAD_OUT,
    DATASET_METADATA_GROUPED_PATH,
    DATASET_METADATA_PATH,
    HALF_DEG,
    KML_PATH,
    MARIDA_BANDS,
    MAX_ALL_BANDS_ZERO_FRAC,
    MAX_B04_ZERO_FRAC,
    MAX_CLOUD,
    MAX_B08_ZERO_FRAC,
    MAX_LAND,
    MIN_SEVERE_KEY_BANDS_VALID_FRAC,
    MIN_KEY_BANDS_VALID_FRAC,
    NC_PATH,
    PATCHES_DIR,
    TARGET_PX,
    ensure_output_dirs,
)
from src.common.geo_utils import kml_bbox, parse_kml_polygons, point_in_kml_mask
from src.common.pipeline_utils import iter_patch_files

DETERMINISTIC_POSITIVE_FAILURE_PREFIXES = (
    "too_dark",
    "low_quality:",
    "empty_patch",
    "bad_scale",
    "too_small",
    "conversion_error",
)
TEMPORARY_POSITIVE_FAILURES = {"download_error", "rate_limited"}
MAX_TEMPORARY_POSITIVE_FAILURE_ATTEMPTS = 3
DOWNLOAD_FAILURE_COLUMNS = [
    "kind",
    "key",
    "rank",
    "date",
    "lat",
    "lon",
    "nc_idx",
    "status",
    "attempts",
    "last_attempt_at",
]

TARGET_SI = 100
TARGET_NO = 100
MIN_PATCH_DISTANCE_KM = 4.0
BATCH_FACTOR = 1.5
DATASET_GENERATION_DIR = DATASET_DOWNLOAD_OUT
REJECTED_CANDIDATES_PATH = DATASET_DOWNLOAD_OUT / "rejected_candidates.csv"
REJECTED_COLUMNS = [
    "kind",
    "key",
    "date",
    "lat",
    "lon",
    "reason",
    "recorded_at",
]


def compute_fdi(data):
    b06, b08, b11 = data[5], data[7], data[9]
    return b08 - (b06 + (b11 - b06) * ((832.9 - 664.6) / (1613.7 - 664.6)) * 10)


def threshold_mean_plus_3std(arr):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals) + 3.0 * np.std(vals))


def build_alignment_reference(data):
    fdi = compute_fdi(data)
    thr_fdi = threshold_mean_plus_3std(fdi)
    if np.isfinite(thr_fdi):
        fdi_mask = np.isfinite(fdi) & (fdi > thr_fdi)
        if int(fdi_mask.sum()) >= 3:
            return fdi_mask.astype(np.float32)
    return fdi


def find_shift(mask, fdi, max_shift=225):
    fdi_norm = fdi - fdi.min()
    if fdi_norm.max() > 0:
        fdi_norm = fdi_norm / fdi_norm.max()
    corr = fftconvolve(fdi_norm, mask[::-1, ::-1], mode="same")
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    r0, r1 = max(0, cy - max_shift), min(height, cy + max_shift)
    c0, c1 = max(0, cx - max_shift), min(width, cx + max_shift)
    region = corr[r0:r1, c0:c1]
    best = np.unravel_index(region.argmax(), region.shape)
    dr = int(best[0] + r0 - cy)
    dc = int(best[1] + c0 - cx)
    return dr, dc


def build_mask(patch_path, nc_idx, ds):
    with rasterio.open(patch_path) as src:
        data = src.read().astype("float32")
        patch_crs = src.crs
        patch_tf = src.transform
        height, width = src.height, src.width

    cx_utm, cy_utm = transform(
        "EPSG:4326",
        patch_crs,
        [float(ds.lon_centroid.values[nc_idx])],
        [float(ds.lat_centroid.values[nc_idx])],
    )
    center_col, center_row = ~patch_tf * (cx_utm[0], cy_utm[0])

    n_pixels = int(ds.n_pixels_fil.values[nc_idx])
    px = ds.pixel_x.values[nc_idx][:n_pixels]
    py = ds.pixel_y.values[nc_idx][:n_pixels]
    cx_source = float(ds.x_centroid.values[nc_idx])
    cy_source = float(ds.y_centroid.values[nc_idx])

    cols = np.round(center_col + (py - cy_source)).astype(int)
    rows = np.round(center_row + (px - cx_source)).astype(int)

    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    raw_mask = np.zeros((height, width), dtype=np.uint8)
    raw_mask[rows[valid], cols[valid]] = 1

    alignment_ref = build_alignment_reference(data)
    dr, dc = find_shift(raw_mask, alignment_ref)

    ys, xs = np.where(raw_mask)
    rs = ys + dr
    cs = xs + dc
    valid_shifted = (rs >= 0) & (rs < height) & (cs >= 0) & (cs < width)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rs[valid_shifted], cs[valid_shifted]] = 1
    return mask, int(mask.sum())


def save_mask(patch_path, mask):
    out_path = patch_path.parent / patch_path.name.replace(".tif", "_mask.tif")
    with rasterio.open(patch_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    return out_path


def make_rgb(data):
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    rgb = np.clip(rgb / 0.1, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def estimate_land_cloud_fraction(data):
    b02 = data[1]
    b03 = data[2]
    b04 = data[3]
    b08 = data[7]
    b11 = data[9]

    finite = np.isfinite(data).all(axis=0)
    nonzero = np.any(data > 0, axis=0)
    valid = finite & nonzero
    if valid.sum() == 0:
        return 1.0, 1.0

    ndwi = np.full_like(b03, np.nan, dtype=np.float32)
    denom = b03 + b08
    ok = valid & (np.abs(denom) > 1e-12)
    ndwi[ok] = (b03[ok] - b08[ok]) / denom[ok]

    land = valid & np.isfinite(ndwi) & (ndwi < 0.0) & (b08 > 0.08) & (b11 > 0.03)
    cloud = valid & (b02 > 0.18) & (b03 > 0.18) & (b04 > 0.18) & (b11 > 0.10)

    land_frac = float(land.sum()) / float(valid.sum())
    cloud_frac = float(cloud.sum()) / float(valid.sum())
    return land_frac, cloud_frac


def estimate_spectral_quality(data):
    _, height, width = data.shape
    total_px = float(height * width)
    zero_all_frac = float(np.all(data == 0, axis=0).sum()) / total_px
    key_valid_frac = float(np.all(data[[1, 2, 3, 7, 9]] > 0, axis=0).sum()) / total_px
    b04_zero_frac = float((data[3] == 0).sum()) / total_px
    b08_zero_frac = float((data[7] == 0).sum()) / total_px
    return {
        "zero_all_frac": zero_all_frac,
        "key_valid_frac": key_valid_frac,
        "b04_zero_frac": b04_zero_frac,
        "b08_zero_frac": b08_zero_frac,
    }


def spectral_quality_rejection_reason(data):
    quality = estimate_spectral_quality(data)
    reasons = []
    if quality["zero_all_frac"] > MAX_ALL_BANDS_ZERO_FRAC:
        reasons.append(f"nodata_all>{MAX_ALL_BANDS_ZERO_FRAC:.0%}")
    if (
        quality["key_valid_frac"] < MIN_SEVERE_KEY_BANDS_VALID_FRAC
        and quality["b04_zero_frac"] > MAX_B04_ZERO_FRAC
        and quality["b08_zero_frac"] > MAX_B08_ZERO_FRAC
    ):
        reasons.append(
            f"severe_b04_b08_zero>{MAX_B04_ZERO_FRAC:.0%}/{MAX_B08_ZERO_FRAC:.0%}"
        )
    return ",".join(reasons)


def safe_unlink(path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def load_rejected_candidates():
    if not REJECTED_CANDIDATES_PATH.exists():
        return pd.DataFrame(columns=REJECTED_COLUMNS)
    df = pd.read_csv(REJECTED_CANDIDATES_PATH).fillna("")
    for column in REJECTED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[REJECTED_COLUMNS].copy()


def save_rejected_candidates(df):
    REJECTED_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[REJECTED_COLUMNS].drop_duplicates(subset=["kind", "key"], keep="last").to_csv(
        REJECTED_CANDIDATES_PATH,
        index=False,
    )


def record_rejected_candidate(df, kind, key, date, lat, lon, reason):
    row = pd.DataFrame(
        [
            {
                "kind": kind,
                "key": str(key),
                "date": date,
                "lat": float(lat) if lat != "" else "",
                "lon": float(lon) if lon != "" else "",
                "reason": str(reason),
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
        ],
        columns=REJECTED_COLUMNS,
    )
    df = df[~((df["kind"].astype(str) == kind) & (df["key"].astype(str) == str(key)))]
    df = pd.concat([df, row], ignore_index=True)
    save_rejected_candidates(df)
    return df


def rejected_keys(df, kind):
    if df.empty:
        return set()
    return set(df[df["kind"].astype(str) == kind]["key"].astype(str))


def download_failure_keys(df, kind):
    if df.empty:
        return set()
    return set(df[df["kind"].astype(str) == kind]["key"].astype(str))


def negative_candidate_key(date_str, lat, lon):
    return f"{date_str}_{float(lat):.6f}_{float(lon):.6f}"


def load_positive_failure_cache():
    if not CSV_DOWNLOAD_FAILURES.exists():
        return pd.DataFrame(columns=DOWNLOAD_FAILURE_COLUMNS)

    df = pd.read_csv(CSV_DOWNLOAD_FAILURES)
    for column in DOWNLOAD_FAILURE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[DOWNLOAD_FAILURE_COLUMNS].copy()
    if not df.empty:
        df["kind"] = df["kind"].replace("", "SI")
        df["key"] = df["key"].mask(df["key"].astype(str).str.strip() == "", df["rank"].astype(str))
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
        df["attempts"] = df["attempts"].fillna(0).astype(int)
    return df


def save_positive_failure_cache(df):
    CSV_DOWNLOAD_FAILURES.parent.mkdir(parents=True, exist_ok=True)
    df = df[DOWNLOAD_FAILURE_COLUMNS].copy()
    df.sort_values(["kind", "key"]).to_csv(CSV_DOWNLOAD_FAILURES, index=False)


def is_deterministic_positive_failure(status):
    return any(str(status).startswith(prefix) for prefix in DETERMINISTIC_POSITIVE_FAILURE_PREFIXES)


def should_skip_failed_positive(rank, failure_cache):
    if failure_cache.empty:
        return None

    matches = failure_cache[
        (failure_cache["kind"].astype(str) == "SI")
        & (failure_cache["key"].astype(str) == str(rank))
    ]
    if matches.empty:
        return None

    row = matches.sort_values("last_attempt_at").iloc[-1]
    status = str(row["status"])
    attempts = int(row["attempts"])

    if is_deterministic_positive_failure(status):
        return f"fallo previo {status}"
    if any(status.startswith(prefix) for prefix in TEMPORARY_POSITIVE_FAILURES) and attempts >= MAX_TEMPORARY_POSITIVE_FAILURE_ATTEMPTS:
        return f"fallo temporal repetido {status} ({attempts} intentos)"
    return None


def record_positive_failure(failure_cache, patch, status):
    return record_download_failure(
        failure_cache,
        kind="SI",
        key=patch["idx"],
        date=patch["date"],
        lat=patch["lat"],
        lon=patch["lon"],
        status=status,
        rank=patch["idx"],
        nc_idx=patch["nc_idx"],
    )


def record_download_failure(failure_cache, kind, key, date, lat, lon, status, rank="", nc_idx=""):
    now = datetime.now().isoformat(timespec="seconds")
    kind = str(kind)
    key = str(key)
    status = str(status)
    existing = failure_cache[
        (failure_cache["kind"].astype(str) == kind)
        & (failure_cache["key"].astype(str) == key)
    ]
    attempts = int(existing["attempts"].max()) + 1 if not existing.empty else 1

    new_row = pd.DataFrame(
        [
            {
                "kind": kind,
                "key": key,
                "rank": rank,
                "date": date,
                "lat": lat,
                "lon": lon,
                "nc_idx": nc_idx,
                "status": status,
                "attempts": attempts,
                "last_attempt_at": now,
            }
        ],
        columns=DOWNLOAD_FAILURE_COLUMNS,
    )

    failure_cache = failure_cache[
        ~(
            (failure_cache["kind"].astype(str) == kind)
            & (failure_cache["key"].astype(str) == key)
        )
    ]
    failure_cache = pd.concat([failure_cache, new_row], ignore_index=True)
    save_positive_failure_cache(failure_cache)
    return failure_cache


def clear_positive_failure(rank, failure_cache):
    if failure_cache.empty:
        return failure_cache
    updated = failure_cache[
        ~(
            (failure_cache["kind"].astype(str) == "SI")
            & (failure_cache["key"].astype(str) == str(rank))
        )
    ].copy()
    if len(updated) != len(failure_cache):
        save_positive_failure_cache(updated)
    return updated


def list_existing_patches_by_label(label):
    return [
        path
        for path in iter_patch_files(PATCHES_DIR)
        if f"_{label}_" in path.name
    ]


def patch_date(path):
    return path.name[:8]


def approx_distance_km(lat1, lon1, lat2, lon2):
    mean_lat = np.deg2rad((lat1 + lat2) / 2.0)
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * np.cos(mean_lat)
    return float(np.hypot(dlat, dlon))


def min_distance_to_coords_km(lat, lon, coords):
    if not coords:
        return float("inf")
    return min(approx_distance_km(lat, lon, other_lat, other_lon) for other_lat, other_lon in coords)


def patch_spatial_info(path):
    with rasterio.open(path) as src:
        bounds = src.bounds
        cx = (bounds.left + bounds.right) / 2.0
        cy = (bounds.bottom + bounds.top) / 2.0
        transform_key = tuple(round(value, 9) for value in src.transform[:6])
        lon, lat = transform(src.crs, "EPSG:4326", [cx], [cy])
        return {
            "lat": float(lat[0]),
            "lon": float(lon[0]),
            "transform_key": transform_key,
        }


def existing_patch_spatial_index(paths):
    coords = []
    transforms = set()
    for path in paths:
        try:
            info = patch_spatial_info(path)
        except Exception:
            continue
        coords.append((info["lat"], info["lon"]))
        transforms.add(info["transform_key"])
    return coords, transforms


def metadata_blocked_coords():
    for path in [DATASET_METADATA_GROUPED_PATH, DATASET_METADATA_PATH]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if {"center_lat", "center_lon"}.issubset(df.columns):
            coords = (
                df[["center_lat", "center_lon"]]
                .dropna()
                .apply(lambda row: (float(row["center_lat"]), float(row["center_lon"])), axis=1)
                .tolist()
            )
            if coords:
                return coords
    return []


def current_spatial_index():
    coords = metadata_blocked_coords()
    _, transforms = existing_patch_spatial_index(list(iter_patch_files(PATCHES_DIR)))
    if coords:
        return coords, transforms
    fallback_coords, transforms = existing_patch_spatial_index(list(iter_patch_files(PATCHES_DIR)))
    return fallback_coords, transforms


def too_close_to_existing(info, blocked_coords, min_distance_km=MIN_PATCH_DISTANCE_KM):
    nearest_km = min_distance_to_coords_km(info["lat"], info["lon"], blocked_coords)
    return nearest_km < min_distance_km, nearest_km


def ordered_positive_indices_by_distance(
    positives,
    target_remaining,
    existing_indices,
    rejected_positive_keys,
    blocked_coords,
    min_distance_km=MIN_PATCH_DISTANCE_KM,
    batch_factor=BATCH_FACTOR,
):
    selected = []
    local_blocked = list(blocked_coords)
    max_to_try = len(positives) if target_remaining > 0 else 0
    for idx, row in positives.iterrows():
        if idx in existing_indices:
            continue
        if str(idx) in rejected_positive_keys:
            continue
        lat = float(row.Latitude)
        lon = float(row.Longitude)
        if min_distance_to_coords_km(lat, lon, local_blocked) < min_distance_km:
            continue
        selected.append(idx)
        local_blocked.append((lat, lon))
        if len(selected) >= max_to_try:
            break
    return selected


def dataset_date_distribution():
    rows = []
    paths = list(iter_patch_files(PATCHES_DIR))
    dates = sorted({patch_date(path) for path in paths})
    for date in dates:
        n_si = sum(1 for path in paths if patch_date(path) == date and "_SI_" in path.name)
        n_no = sum(1 for path in paths if patch_date(path) == date and "_NO_" in path.name)
        rows.append({"date": date, "n_si": n_si, "n_no": n_no, "n_total": n_si + n_no})
    return pd.DataFrame(rows)


def extract_positive_rank_index(path):
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[1] == "SI":
        try:
            return int(parts[-1])
        except Exception:
            return None
    return None


def next_negative_index(base_offset=3):
    max_idx = base_offset - 1
    for tif in PATCHES_DIR.glob("*_NO_*_*.tif"):
        if "mask" in tif.name or tif.name.startswith("_"):
            continue
        parts = tif.stem.split("_")
        if len(parts) >= 4 and parts[1] == "NO":
            try:
                max_idx = max(max_idx, int(parts[3]))
            except Exception:
                continue
    return max_idx + 1


def build_unique_negative_path(date_str, idx_num, label):
    idx_str = f"{idx_num:02d}"
    base_name = f"{date_str.replace('-', '')}_NO_000000_{idx_str}_{label}.tif"
    path = PATCHES_DIR / base_name
    if not path.exists():
        return path

    suffix = 2
    while True:
        alt = PATCHES_DIR / f"{date_str.replace('-', '')}_NO_000000_{idx_str}_{label}_{suffix}.tif"
        if not alt.exists():
            return alt
        suffix += 1


def validate_patch_binary(patch_path, title, subtitle="Aceptar patch?"):
    with rasterio.open(patch_path) as src:
        data = src.read().astype("float32")
    rgb_arr = make_rgb(data)
    result = [None]

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)

    img = Image.fromarray(rgb_arr).resize((520, 520))
    photo = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=photo)
    panel.image = photo
    panel.pack(pady=8)
    tk.Label(root, text=f"{patch_path.name}\n{subtitle}", font=("Arial", 11)).pack()

    frame = ttk.Frame(root)
    frame.pack(pady=10)

    def on_click(value):
        result[0] = value
        root.destroy()

    tk.Button(
        frame,
        text="Aceptar",
        width=16,
        bg="#16a34a",
        fg="white",
        font=("Arial", 11, "bold"),
        command=lambda: on_click("accept"),
    ).pack(side=tk.LEFT, padx=8)
    tk.Button(
        frame,
        text="Rechazar",
        width=16,
        bg="#dc2626",
        fg="white",
        font=("Arial", 11, "bold"),
        command=lambda: on_click("reject"),
    ).pack(side=tk.LEFT, padx=8)

    root.mainloop()
    return result[0]


def download_and_convert(conn, lat, lon, date_str, out_path, reduce=True):
    bbox = {
        "west": round(lon - HALF_DEG, 6),
        "east": round(lon + HALF_DEG, 6),
        "south": round(lat - HALF_DEG, 6),
        "north": round(lat + HALF_DEG, 6),
        "crs": "EPSG:4326",
    }
    tmp = PATCHES_DIR / f"_tmp_bands_{out_path.stem}.tif"

    for attempt in range(3):
        try:
            s2 = conn.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox,
                temporal_extent=[date_str, date_str],
                bands=MARIDA_BANDS,
                max_cloud_cover=30,
            )
            result = s2.reduce_dimension(dimension="t", reducer="median") if reduce else s2
            result.download(tmp, format="GTiff")
            break
        except Exception as exc:
            message = str(exc)
            safe_unlink(tmp)
            if ("503" in message or "429" in message) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            kind = "rate_limited" if "429" in message else "download_error"
            return f"{kind}:{message}"
    else:
        return "download_error"

    try:
        with rasterio.open(tmp) as src:
            data = src.read().astype("float32")
            meta = src.meta.copy()
            transform = src.transform
        safe_unlink(tmp)

        if data.max() > 10:
            data = data / 10000.0

        data[data < -0.5] = np.nan
        data = np.clip(data, 0.0, None)
        data = np.nan_to_num(data, nan=0.0)

        valid = data[data > 0]
        if len(valid) == 0:
            return "empty_patch"
        if data.max() > 1.5:
            return "bad_scale"
        if data.mean() < 0.0005:
            return "too_dark"

        _, height, width = data.shape
        if height < TARGET_PX or width < TARGET_PX:
            return "too_small"

        r0 = (height - TARGET_PX) // 2
        c0 = (width - TARGET_PX) // 2
        data = data[:, r0 : r0 + TARGET_PX, c0 : c0 + TARGET_PX]
        quality_reason = spectral_quality_rejection_reason(data)
        if quality_reason:
            return f"low_quality:{quality_reason}"
        new_tf = rasterio.windows.transform(
            rasterio.windows.Window(c0, r0, TARGET_PX, TARGET_PX),
            transform,
        )
        meta.update(
            dtype="float32",
            count=11,
            width=TARGET_PX,
            height=TARGET_PX,
            transform=new_tf,
            nodata=0.0,
        )
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
        return "ok"
    except Exception as exc:
        safe_unlink(tmp)
        safe_unlink(out_path)
        return f"conversion_error:{exc}"


def find_negative_candidates(
    rng,
    all_coords,
    bbox,
    polygons,
    min_date,
    max_date,
    n_candidates=6,
    min_distance_km=MIN_PATCH_DISTANCE_KM,
    excluded_keys=None,
):
    candidates = []
    excluded_keys = set() if excluded_keys is None else set(excluded_keys)
    blocked_coords = [(float(lat), float(lon)) for lat, lon in all_coords]
    attempts = 0
    min_date = pd.Timestamp(min_date).normalize()
    max_date = pd.Timestamp(max_date).normalize()
    day_span = max(int((max_date - min_date).days), 0)
    while len(candidates) < n_candidates and attempts < 300:
        attempts += 1

        lat = rng.uniform(bbox["lat_min"], bbox["lat_max"])
        lon = rng.uniform(bbox["lon_min"], bbox["lon_max"])
        if not point_in_kml_mask(lon, lat, polygons):
            continue

        if min_distance_to_coords_km(lat, lon, blocked_coords) < min_distance_km:
            continue

        random_offset_days = int(rng.integers(0, day_span + 1)) if day_span > 0 else 0
        date = min_date + pd.Timedelta(days=random_offset_days)
        date_str = date.strftime("%Y-%m-%d")
        if negative_candidate_key(date_str, lat, lon) in excluded_keys:
            continue
        h = HALF_DEG
        d0 = date.strftime("%Y-%m-%dT00:00:00Z")
        d1 = date.strftime("%Y-%m-%dT23:59:59Z")
        url = (
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
            "?$filter=Collection/Name eq 'SENTINEL-2'"
            " and Attributes/OData.CSC.StringAttribute/any("
            "a:a/Name eq 'productType' and a/Value eq 'S2MSI2A')"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({lon-h} {lat-h},"
            f"{lon+h} {lat-h},{lon+h} {lat+h},{lon-h} {lat+h},{lon-h} {lat-h}))')"
            f" and ContentDate/Start gt {d0}"
            f" and ContentDate/Start lt {d1}"
            "&$top=1"
        )
        try:
            response = requests.get(url, timeout=10)
            if response.json().get("value"):
                candidates.append((lat, lon, date_str))
                blocked_coords.append((lat, lon))
        except Exception:
            continue

    return candidates


def download_negative_candidate(conn, lat, lon, date_str, tmp_id):
    tmp = PATCHES_DIR / f"_tmp_neg_{tmp_id}.tif"
    status = download_and_convert(conn, lat, lon, date_str, tmp)
    if status != "ok" or not tmp.exists():
        return lat, lon, date_str, tmp, False, None, None, status

    try:
        with rasterio.open(tmp) as src:
            data = src.read().astype("float32")
        land_frac, cloud_frac = estimate_land_cloud_fraction(data)
    except Exception:
        safe_unlink(tmp)
        return lat, lon, date_str, tmp, False, None, None, "read_failed"

    if land_frac > MAX_LAND or cloud_frac > MAX_CLOUD:
        safe_unlink(tmp)
        reasons = []
        if land_frac > MAX_LAND:
            reasons.append(f"land>{MAX_LAND:.0%}")
        if cloud_frac > MAX_CLOUD:
            reasons.append(f"cloud>{MAX_CLOUD:.0%}")
        return lat, lon, date_str, tmp, False, land_frac, cloud_frac, ",".join(reasons)

    return lat, lon, date_str, tmp, True, land_frac, cloud_frac, "accepted"


def gen_negatives(
    conn,
    rng,
    df_all,
    polygons,
    n_needed,
    rejected_df,
    failure_cache,
    min_distance_km=MIN_PATCH_DISTANCE_KM,
    batch_factor=BATCH_FACTOR,
    extra_blocked_coords=None,
    extra_patch_transforms=None,
):
    positive_coords = df_all[["Latitude", "Longitude"]].to_numpy()
    min_date = df_all["Date"].min()
    max_date = df_all["Date"].max()
    bbox = kml_bbox(polygons)

    accepted = []
    next_idx = next_negative_index(base_offset=n_needed)
    patch_coords, patch_transforms = current_spatial_index()
    if extra_patch_transforms:
        patch_transforms.update(extra_patch_transforms)
    blocked_coords = [*[(float(lat), float(lon)) for lat, lon in positive_coords], *patch_coords]
    if extra_blocked_coords:
        blocked_coords.extend(extra_blocked_coords)

    while len(accepted) < n_needed:
        remaining = n_needed - len(accepted)
        review_target = max(1, int(ceil(remaining * batch_factor)))
        reviewable = []

        while len(reviewable) < review_target:
            batch_size = max(1, review_target - len(reviewable))
            print(
                f"  Buscando candidatos negativos "
                f"({len(reviewable)}/{review_target} listos para revision; faltan {remaining})..."
            )
            candidates = find_negative_candidates(
                rng,
                all_coords=np.array(blocked_coords, dtype=float),
                bbox=bbox,
                polygons=polygons,
                min_date=min_date,
                max_date=max_date,
                n_candidates=batch_size,
                min_distance_km=min_distance_km,
                excluded_keys=(
                    download_failure_keys(failure_cache, "NO")
                    | rejected_keys(rejected_df, "NO")
                ),
            )
            if not candidates:
                print("  Sin candidatos validos, reintentando...")
                continue

            print(f"  Descargando {len(candidates)} candidatos...")
            for i, (lat, lon, date_str) in enumerate(candidates):
                if len(reviewable) >= review_target:
                    break
                candidate_key = negative_candidate_key(date_str, lat, lon)
                if candidate_key in download_failure_keys(failure_cache, "NO"):
                    print(f"  SKIP negativo {candidate_key}: fallo de descarga previo")
                    continue
                if candidate_key in rejected_keys(rejected_df, "NO"):
                    print(f"  SKIP negativo {candidate_key}: rechazado previamente")
                    continue
                lat, lon, date_str, tmp, ok, land_frac, cloud_frac, reason = download_negative_candidate(
                    conn,
                    lat,
                    lon,
                    date_str,
                    f"{len(accepted)}_{len(reviewable)}_{i}",
                )
                if not ok or not tmp.exists():
                    failure_cache = record_download_failure(
                        failure_cache,
                        kind="NO",
                        key=candidate_key,
                        date=date_str,
                        lat=lat,
                        lon=lon,
                        status=reason,
                    )
                    extra = f" motivo={reason}"
                    if land_frac is not None and cloud_frac is not None:
                        extra += f" land={land_frac:.2%} cloud={cloud_frac:.2%}"
                    print(f"  DESCARTADO ({lat:.3f}, {lon:.3f}){extra}")
                    time.sleep(2.5)
                    continue

                print(
                    f"  OK ({lat:.3f}, {lon:.3f}) {date_str}  "
                    f"land={land_frac:.2%} cloud={cloud_frac:.2%}"
                )

                try:
                    tmp_info = patch_spatial_info(tmp)
                except Exception:
                    safe_unlink(tmp)
                    print("  [NEG] descartado: no se pudo leer georreferenciacion")
                    time.sleep(2)
                    continue
                if tmp_info["transform_key"] in patch_transforms:
                    safe_unlink(tmp)
                    print("  [NEG] descartado: patch duplicado por coordenadas exactas")
                    time.sleep(2)
                    continue
                nearest_km = min_distance_to_coords_km(tmp_info["lat"], tmp_info["lon"], blocked_coords)
                if nearest_km < min_distance_km:
                    safe_unlink(tmp)
                    print(
                        f"  [NEG] descartado: demasiado cerca de otro patch "
                        f"({nearest_km:.2f} km < {min_distance_km:.2f} km)"
                    )
                    time.sleep(2)
                    continue

                rejection_key = f"{date_str}_{tmp_info['lat']:.6f}_{tmp_info['lon']:.6f}"
                if rejection_key in rejected_keys(rejected_df, "NO"):
                    safe_unlink(tmp)
                    print("  [NEG] descartado: candidato ya rechazado anteriormente")
                    time.sleep(2)
                    continue

                reviewable.append((date_str, tmp, tmp_info, rejection_key))
                print(f"  [NEG] listo para revision: {len(reviewable)}/{review_target}")
                time.sleep(2)

        if reviewable:
            print(f"\n  Revisando {len(reviewable)} negativos descargados...\n")

        for date_str, tmp, tmp_info, rejection_key in reviewable:
            if tmp_info["transform_key"] in patch_transforms:
                safe_unlink(tmp)
                print("  [NEG] descartado antes de mostrar: patch duplicado por coordenadas exactas")
                continue
            is_too_close, nearest_km = too_close_to_existing(tmp_info, blocked_coords, min_distance_km)
            if is_too_close:
                safe_unlink(tmp)
                print(
                    f"  [NEG] descartado antes de mostrar: demasiado cerca de otro patch "
                    f"({nearest_km:.2f} km < {min_distance_km:.2f} km)"
                )
                continue
            if len(accepted) >= n_needed:
                safe_unlink(tmp)
                print("  [NEG] sobrante sin revisar: el objetivo NO ya esta completo")
                continue

            decision = validate_patch_binary(tmp, "Validar negativo", "Aceptar este negativo?")
            if decision != "accept":
                safe_unlink(tmp)
                rejected_df = record_rejected_candidate(
                    rejected_df,
                    "NO",
                    rejection_key,
                    date_str,
                    tmp_info["lat"],
                    tmp_info["lon"],
                    "user_reject",
                )
                print("  [NEG] descartado por usuario")
                continue

            final_path = build_unique_negative_path(date_str, next_idx, "CLARO")
            tmp.rename(final_path)
            accepted.append(final_path)
            blocked_coords.append((tmp_info["lat"], tmp_info["lon"]))
            patch_transforms.add(tmp_info["transform_key"])
            next_idx += 1
            print(f"  [NEG] aceptado: {final_path.name}")
            write_dataset_metadata()

    for tmp in PATCHES_DIR.glob("_tmp_neg_*.tif"):
        safe_unlink(tmp)
    return accepted


def ensure_nc_idx(df_all, ds):
    if "nc_idx" in df_all.columns:
        return df_all

    print("Calculando nc_idx para el CSV...")

    def find_nc_idx(lat, lon):
        d = (ds.lat_centroid.values - lat) ** 2 + (ds.lon_centroid.values - lon) ** 2
        return int(d.argmin())

    df_all = df_all.copy()
    df_all["nc_idx"] = df_all.apply(lambda row: find_nc_idx(row.Latitude, row.Longitude), axis=1)
    df_all.to_csv(CSV_CANDIDATES, index=False)
    print(f"nc_idx anadido y guardado en {CSV_CANDIDATES}")
    return df_all


DIFFICULTIES = {"CLARO", "DIFICIL", "DUDOSO"}
BASE_METADATA_COLUMNS = [
    "patch",
    "patch_path",
    "mask_path",
    "date",
    "year",
    "month",
    "label",
    "label_binary",
    "expected_gt_px",
    "mask_gt_px",
    "name_mask_match",
    "original_difficulty",
    "is_positive",
    "is_negative",
    "center_x",
    "center_y",
    "center_lon",
    "center_lat",
]
ANNOTATION_COLUMNS = [
    "manual_decision",
    "manual_confidence",
    "image_quality",
    "scene_tags",
    "notes",
    "annotated_at",
]
TAG_COLUMNS = [
    "tag_nube",
    "tag_nube_fina",
    "tag_estela",
    "tag_barco",
    "tag_costa",
    "tag_brillo_solar",
    "tag_agua_oscura",
    "tag_agua_muy_oscura",
    "tag_agua_turbia",
    "tag_posible_residuo",
    "tag_agua_limpia",
    "tag_filamento_visible",
    "tag_olas",
    "tag_espuma",
    "tag_corte_patch",
    "tag_sombra_nube",
]
GROUP_COLUMNS = ["group_date", "group_month", "group_year", "group_id"]
PRESERVED_METADATA_COLUMNS = ANNOTATION_COLUMNS + TAG_COLUMNS


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "si", "s"}


def _clean_number_text(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def parse_patch_name(path):
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


def mask_pixel_count(mask_path):
    if not mask_path.exists():
        return 0
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int((mask > 0).sum())


def patch_center(path):
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


def load_existing_annotations():
    frames = []
    for path in [DATASET_METADATA_PATH, DATASET_METADATA_GROUPED_PATH]:
        if not path.exists():
            continue
        df = pd.read_csv(path).fillna("")
        if "patch" not in df.columns:
            continue
        keep = ["patch", *[column for column in PRESERVED_METADATA_COLUMNS if column in df.columns]]
        frames.append(df[keep].copy())
    if not frames:
        return pd.DataFrame(columns=["patch", *PRESERVED_METADATA_COLUMNS])
    annotations = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["patch"], keep="last")
    for column in PRESERVED_METADATA_COLUMNS:
        if column not in annotations.columns:
            annotations[column] = ""
    return annotations[["patch", *PRESERVED_METADATA_COLUMNS]]


def build_base_metadata():
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


def add_metadata_annotations(metadata, annotations):
    metadata = metadata.merge(annotations, on="patch", how="left")
    for column in PRESERVED_METADATA_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
        metadata[column] = metadata[column].fillna("")
    for column in TAG_COLUMNS:
        metadata[column] = metadata[column].map(lambda value: "1" if _as_bool(value) else "0" if str(value).strip() else "")
    metadata["manual_confidence"] = metadata["manual_confidence"].map(_clean_number_text)
    return metadata


def add_group_columns(metadata):
    metadata = metadata.copy()
    date = metadata["date"].astype(str).str.replace("-", "", regex=False).str[:8]
    metadata["group_date"] = date
    metadata["group_month"] = date.str[:6]
    metadata["group_year"] = date.str[:4]
    metadata["group_id"] = metadata["group_date"]
    return metadata


def write_dataset_metadata():
    metadata = add_metadata_annotations(build_base_metadata(), load_existing_annotations())
    if metadata.empty:
        print(f"WARN: No se encontraron patches en {PATCHES_DIR}.")
        return
    DATASET_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    base = metadata[[*BASE_METADATA_COLUMNS, *PRESERVED_METADATA_COLUMNS]].copy()
    grouped = add_group_columns(base)
    grouped = grouped[[*BASE_METADATA_COLUMNS, *PRESERVED_METADATA_COLUMNS, *GROUP_COLUMNS]]
    base.to_csv(DATASET_METADATA_PATH, index=False)
    grouped.to_csv(DATASET_METADATA_GROUPED_PATH, index=False)
    print(f"Metadata guardada: {DATASET_METADATA_PATH}")
    print(f"Metadata con grupos guardada: {DATASET_METADATA_GROUPED_PATH}")
    print(
        f"Patches={len(base)} | SI={int(base['is_positive'].sum())} | "
        f"NO={int(base['is_negative'].sum())} | fechas={int(base['date'].nunique())}"
    )


def save_dataset_generation_summary_simple(target_si, target_no):
    DATASET_GENERATION_DIR.mkdir(parents=True, exist_ok=True)
    distribution = dataset_date_distribution()
    distribution.to_csv(DATASET_GENERATION_DIR / "date_distribution_after_download.csv", index=False)
    n_si = int(distribution["n_si"].sum()) if not distribution.empty else 0
    n_no = int(distribution["n_no"].sum()) if not distribution.empty else 0
    summary = [
        "# Dataset generation summary",
        "",
        f"- target_si: {target_si}",
        f"- target_no: {target_no}",
        f"- current_si: {n_si}",
        f"- current_no: {n_no}",
        f"- min_patch_distance_km: {MIN_PATCH_DISTANCE_KM}",
        f"- rejected_cache: {REJECTED_CANDIDATES_PATH}",
    ]
    (DATASET_GENERATION_DIR / "dataset_generation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_positives", type=int, default=TARGET_SI)
    parser.add_argument("--n_negatives", type=int, default=None)
    parser.add_argument("--batch_factor", type=float, default=BATCH_FACTOR)
    args = parser.parse_args()

    ensure_output_dirs()
    target_negatives = args.n_negatives if args.n_negatives is not None else args.n_positives

    df_all = pd.read_csv(CSV_CANDIDATES)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    ds = xr.open_dataset(NC_PATH)
    df_all = ensure_nc_idx(df_all, ds)
    polygons = parse_kml_polygons(KML_PATH)
    positives = df_all.sort_values("Pixels per LW", ascending=False).reset_index(drop=True)

    existing_positive_files = list_existing_patches_by_label("SI")
    existing_negative_files = list_existing_patches_by_label("NO")
    positive_failure_cache = load_positive_failure_cache()
    rejected_df = load_rejected_candidates()
    existing_positive_indices = {
        idx
        for idx in (extract_positive_rank_index(path) for path in existing_positive_files)
        if idx is not None
    }
    patch_coords, patch_transforms = current_spatial_index()
    blocked_coords = [*patch_coords]
    positives_needed = max(args.n_positives - len(existing_positive_files), 0)
    negatives_needed = max(target_negatives - len(existing_negative_files), 0)
    if positives_needed == 0 and negatives_needed == 0:
        print(
            f"Ya hay suficientes patches: SI={len(existing_positive_files)}/{args.n_positives}, "
            f"NO={len(existing_negative_files)}/{target_negatives}. No se descarga nada."
        )
        save_dataset_generation_summary_simple(args.n_positives, target_negatives)
        write_dataset_metadata()
        return

    positive_indices_to_try = ordered_positive_indices_by_distance(
        positives=positives,
        target_remaining=positives_needed,
        existing_indices=existing_positive_indices,
        rejected_positive_keys=rejected_keys(rejected_df, "SI"),
        blocked_coords=blocked_coords,
        min_distance_km=MIN_PATCH_DISTANCE_KM,
        batch_factor=args.batch_factor,
    )

    print("\n=== CONECTANDO A OPENEO ===")
    conn = openeo.connect("openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()
    print("Autenticado\n")

    rng = np.random.default_rng(seed=777)

    print("=== POSITIVOS ===\n")
    print(
        f"Objetivo SI={args.n_positives}. "
        f"Ya existen {len(existing_positive_files)} positivos y {len(existing_negative_files)} negativos.\n"
    )

    current_positive_total = len(existing_positive_files)
    current_negative_total = len(existing_negative_files)
    negatives_needed = max(target_negatives - current_negative_total, 0)
    negatives_launched = False
    positive_cursor = 0
    while current_positive_total < args.n_positives:
        remaining = args.n_positives - current_positive_total
        review_target = max(1, int(ceil(remaining * args.batch_factor)))
        downloaded_positives = []

        while len(downloaded_positives) < review_target and positive_cursor < len(positive_indices_to_try):
            idx = positive_indices_to_try[positive_cursor]
            positive_cursor += 1
            row = positives.iloc[idx]

            patch = {
                "lat": float(row.Latitude),
                "lon": float(row.Longitude),
                "date": row.Date.strftime("%Y-%m-%d"),
                "nc_idx": int(row.nc_idx),
                "idx": idx,
            }

            skip_reason = should_skip_failed_positive(idx, positive_failure_cache)
            if skip_reason:
                print(f"  SKIP rank {idx}: {skip_reason}")
                continue
            if str(idx) in rejected_keys(rejected_df, "SI"):
                print(f"  SKIP rank {idx}: rechazado previamente por usuario")
                continue
            if min_distance_to_coords_km(patch["lat"], patch["lon"], blocked_coords) < MIN_PATCH_DISTANCE_KM:
                print(f"  SKIP rank {idx}: demasiado cerca de otro patch")
                continue

            tmp = PATCHES_DIR / f"_tmp_si_{patch['idx']:02d}.tif"
            print(
                f"  Descargando positivo rank {patch['idx']} "
                f"({patch['lat']:.3f}, {patch['lon']:.3f}) {patch['date']}..."
            )
            status = download_and_convert(conn, patch["lat"], patch["lon"], patch["date"], tmp)
            if status != "ok":
                safe_unlink(tmp)
                positive_failure_cache = record_positive_failure(positive_failure_cache, patch, status)
                print(f"  FALLO rank {patch['idx']}: {status}")
                continue

            try:
                tmp_info = patch_spatial_info(tmp)
            except Exception as exc:
                safe_unlink(tmp)
                positive_failure_cache = record_positive_failure(
                    positive_failure_cache,
                    patch,
                    f"georef_error:{exc}",
                )
                print(f"  FALLO rank {patch['idx']}: georef_error:{exc}")
                continue
            if tmp_info["transform_key"] in patch_transforms:
                safe_unlink(tmp)
                positive_failure_cache = record_positive_failure(positive_failure_cache, patch, "duplicate_transform")
                print(f"  FALLO rank {patch['idx']}: duplicate_transform")
                continue
            nearest_km = min_distance_to_coords_km(tmp_info["lat"], tmp_info["lon"], blocked_coords)
            if nearest_km < MIN_PATCH_DISTANCE_KM:
                safe_unlink(tmp)
                positive_failure_cache = record_positive_failure(positive_failure_cache, patch, f"too_close:{nearest_km:.2f}km")
                print(f"  FALLO rank {patch['idx']}: too_close {nearest_km:.2f} km")
                continue

            downloaded_positives.append((patch, tmp, tmp_info))
            print(
                f"  LISTO rank {patch['idx']} para revision "
                f"({len(downloaded_positives)}/{review_target}; faltan {remaining})"
            )
            time.sleep(3)

        if not downloaded_positives:
            print("  WARN: no quedan positivos validos para descargar.")
            break

        if len(downloaded_positives) < review_target:
            print(
                f"  WARN: solo hay {len(downloaded_positives)}/{review_target} "
                "positivos validos para revision."
            )

        if negatives_needed and not negatives_launched:
            pending_positive_coords = [
                (tmp_info["lat"], tmp_info["lon"])
                for _patch, _tmp, tmp_info in downloaded_positives
            ]
            pending_positive_transforms = {
                tmp_info["transform_key"]
                for _patch, _tmp, tmp_info in downloaded_positives
            }
            print("\n=== NEGATIVOS (prelanzado antes de revisar positivos) ===\n")
            print(
                f"Objetivo NO={target_negatives}. "
                f"Ya existen {current_negative_total}. "
                f"Faltan {negatives_needed}.\n"
            )
            gen_negatives(
                conn,
                rng,
                df_all=df_all,
                polygons=polygons,
                n_needed=negatives_needed,
                rejected_df=rejected_df,
                failure_cache=positive_failure_cache,
                min_distance_km=MIN_PATCH_DISTANCE_KM,
                batch_factor=args.batch_factor,
                extra_blocked_coords=pending_positive_coords,
                extra_patch_transforms=pending_positive_transforms,
            )
            negatives_launched = True

        print(f"\n  Revisando {len(downloaded_positives)} positivos descargados...\n")

        for patch, tmp, tmp_info in downloaded_positives:
            if tmp_info["transform_key"] in patch_transforms:
                safe_unlink(tmp)
                print(f"  SKIP rank {patch['idx']} antes de mostrar: patch duplicado por coordenadas exactas")
                continue
            is_too_close, nearest_km = too_close_to_existing(tmp_info, blocked_coords, MIN_PATCH_DISTANCE_KM)
            if is_too_close:
                safe_unlink(tmp)
                print(
                    f"  SKIP rank {patch['idx']} antes de mostrar: demasiado cerca de otro patch "
                    f"({nearest_km:.2f} km < {MIN_PATCH_DISTANCE_KM:.2f} km)"
                )
                continue
            if current_positive_total >= args.n_positives:
                safe_unlink(tmp)
                print(f"  SOBRANTE rank {patch['idx']}: el objetivo SI ya esta completo")
                continue

            decision = validate_patch_binary(tmp, "Validar positivo", "Aceptar este positivo?")
            if decision != "accept":
                safe_unlink(tmp)
                rejected_df = record_rejected_candidate(
                    rejected_df,
                    "SI",
                    patch["idx"],
                    patch["date"],
                    tmp_info["lat"],
                    tmp_info["lon"],
                    "user_reject",
                )
                print(f"  REJECT rank {patch['idx']} por usuario")
                continue

            mask, n_px = build_mask(tmp, patch["nc_idx"], ds)
            final_name = f"{patch['date'].replace('-', '')}_SI_{n_px:06d}_{patch['idx']:02d}.tif"
            final_path = PATCHES_DIR / final_name
            tmp.rename(final_path)
            save_mask(final_path, mask)
            positive_failure_cache = clear_positive_failure(patch["idx"], positive_failure_cache)
            existing_positive_indices.add(patch["idx"])
            blocked_coords.append((tmp_info["lat"], tmp_info["lon"]))
            patch_transforms.add(tmp_info["transform_key"])
            current_positive_total += 1
            print(f"  OK {final_name} ({n_px} px en patch) -> total SI={current_positive_total}")
            write_dataset_metadata()

    print("\n=== NEGATIVOS ===\n")
    current_negative_total = len(list_existing_patches_by_label("NO"))
    negatives_needed = max(target_negatives - current_negative_total, 0)
    print(
        f"Objetivo NO={target_negatives}. "
        f"Ya existen {current_negative_total}. "
        f"Faltan {negatives_needed}.\n"
    )
    if negatives_needed and not negatives_launched:
        gen_negatives(
            conn,
            rng,
            df_all=df_all,
            polygons=polygons,
            n_needed=negatives_needed,
            rejected_df=rejected_df,
            failure_cache=positive_failure_cache,
            min_distance_km=MIN_PATCH_DISTANCE_KM,
            batch_factor=args.batch_factor,
        )

    final_positive_total = len(list_existing_patches_by_label("SI"))
    final_negative_total = len(list_existing_patches_by_label("NO"))
    print(
        f"\nRESUMEN FINAL -> SI={final_positive_total} / objetivo {args.n_positives} | "
        f"NO={final_negative_total} / objetivo {target_negatives}"
    )
    save_dataset_generation_summary_simple(args.n_positives, target_negatives)
    write_dataset_metadata()


if __name__ == "__main__":
    main()
