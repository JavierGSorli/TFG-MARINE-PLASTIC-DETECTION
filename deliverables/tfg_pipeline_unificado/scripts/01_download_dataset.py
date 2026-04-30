from __future__ import annotations

import argparse
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import numpy as np
import openeo
import pandas as pd
import rasterio
import requests
import xarray as xr
from PIL import Image, ImageTk
from pyproj import Transformer
from scipy.signal import fftconvolve

from config import (
    CSV_CANDIDATES,
    CSV_POSITIVE_FAILURES,
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
from geo_utils import kml_bbox, parse_kml_polygons, point_in_kml_mask
from pipeline_utils import iter_patch_files

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
POSITIVE_FAILURE_COLUMNS = [
    "rank",
    "date",
    "lat",
    "lon",
    "nc_idx",
    "status",
    "attempts",
    "last_attempt_at",
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

    tr = Transformer.from_crs("EPSG:4326", patch_crs, always_xy=True)
    cx_utm, cy_utm = tr.transform(
        float(ds.lon_centroid.values[nc_idx]),
        float(ds.lat_centroid.values[nc_idx]),
    )
    center_col, center_row = ~patch_tf * (cx_utm, cy_utm)

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
    rs = np.clip(ys + dr, 0, height - 1)
    cs = np.clip(xs + dc, 0, width - 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rs, cs] = 1
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


def load_positive_failure_cache():
    if not CSV_POSITIVE_FAILURES.exists():
        return pd.DataFrame(columns=POSITIVE_FAILURE_COLUMNS)

    df = pd.read_csv(CSV_POSITIVE_FAILURES)
    for column in POSITIVE_FAILURE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[POSITIVE_FAILURE_COLUMNS].copy()
    if not df.empty:
        df["rank"] = df["rank"].astype(int)
        df["attempts"] = df["attempts"].fillna(0).astype(int)
    return df


def save_positive_failure_cache(df):
    CSV_POSITIVE_FAILURES.parent.mkdir(parents=True, exist_ok=True)
    df = df[POSITIVE_FAILURE_COLUMNS].copy()
    df.sort_values("rank").to_csv(CSV_POSITIVE_FAILURES, index=False)


def is_deterministic_positive_failure(status):
    return any(str(status).startswith(prefix) for prefix in DETERMINISTIC_POSITIVE_FAILURE_PREFIXES)


def should_skip_failed_positive(rank, failure_cache):
    if failure_cache.empty:
        return None

    matches = failure_cache[failure_cache["rank"] == int(rank)]
    if matches.empty:
        return None

    row = matches.sort_values("last_attempt_at").iloc[-1]
    status = str(row["status"])
    attempts = int(row["attempts"])

    if is_deterministic_positive_failure(status):
        return f"fallo previo {status}"
    if status in TEMPORARY_POSITIVE_FAILURES and attempts >= MAX_TEMPORARY_POSITIVE_FAILURE_ATTEMPTS:
        return f"fallo temporal repetido {status} ({attempts} intentos)"
    return None


def record_positive_failure(failure_cache, patch, status):
    now = datetime.now().isoformat(timespec="seconds")
    rank = int(patch["idx"])
    status = str(status)
    existing = failure_cache[failure_cache["rank"] == rank]
    attempts = int(existing["attempts"].max()) + 1 if not existing.empty else 1

    new_row = pd.DataFrame(
        [
            {
                "rank": rank,
                "date": patch["date"],
                "lat": patch["lat"],
                "lon": patch["lon"],
                "nc_idx": patch["nc_idx"],
                "status": status,
                "attempts": attempts,
                "last_attempt_at": now,
            }
        ],
        columns=POSITIVE_FAILURE_COLUMNS,
    )

    failure_cache = failure_cache[failure_cache["rank"] != rank]
    failure_cache = pd.concat([failure_cache, new_row], ignore_index=True)
    save_positive_failure_cache(failure_cache)
    return failure_cache


def clear_positive_failure(rank, failure_cache):
    if failure_cache.empty:
        return failure_cache
    updated = failure_cache[failure_cache["rank"] != int(rank)].copy()
    if len(updated) != len(failure_cache):
        save_positive_failure_cache(updated)
    return updated


def list_existing_patches_by_label(label):
    return [
        path
        for path in iter_patch_files(PATCHES_DIR)
        if f"_{label}_" in path.name
    ]


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


def validate_negative(patch_path):
    with rasterio.open(patch_path) as src:
        data = src.read().astype("float32")
    rgb_arr = make_rgb(data)
    result = [None]

    root = tk.Tk()
    root.title(f"Validar: {patch_path.name}")
    root.resizable(False, False)

    img = Image.fromarray(rgb_arr).resize((420, 420))
    photo = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=photo)
    panel.image = photo
    panel.pack(pady=8)
    tk.Label(root, text=f"{patch_path.name}\nQue ves?", font=("Arial", 10)).pack()

    frame = ttk.Frame(root)
    frame.pack(pady=10)

    def on_click(value):
        result[0] = value
        root.destroy()

    buttons = [
        ("NO - descartar", "NO", "#dc2626"),
        ("DUDOSO", "DUDOSO", "#d97706"),
        ("CLARO", "CLARO", "#16a34a"),
        ("DIFICIL", "DIFICIL", "#2563eb"),
    ]
    for text, value, color in buttons:
        tk.Button(
            frame,
            text=text,
            width=13,
            bg=color,
            fg="white",
            font=("Arial", 10, "bold"),
            command=lambda current=value: on_click(current),
        ).pack(side=tk.LEFT, padx=4)

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
                time.sleep(40 * (attempt + 1))
                continue
            return "rate_limited" if "429" in message else "download_error"
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
    except Exception:
        safe_unlink(tmp)
        safe_unlink(out_path)
        return "conversion_error"


def find_negative_candidates(rng, all_coords, bbox, polygons, all_dates, n_candidates=6):
    candidates = []
    attempts = 0
    while len(candidates) < n_candidates and attempts < 300:
        attempts += 1

        lat = rng.uniform(bbox["lat_min"], bbox["lat_max"])
        lon = rng.uniform(bbox["lon_min"], bbox["lon_max"])
        if not point_in_kml_mask(lon, lat, polygons):
            continue

        dists = np.sqrt((all_coords[:, 0] - lat) ** 2 + (all_coords[:, 1] - lon) ** 2)
        if dists.min() < 5 / 111:
            continue

        date_str = str(rng.choice(all_dates))
        h = HALF_DEG
        date = datetime.strptime(date_str, "%Y-%m-%d")
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


def gen_negatives(conn, rng, df_all, polygons, n_needed):
    all_coords = df_all[["Latitude", "Longitude"]].to_numpy()
    all_dates = df_all["Date"].dt.strftime("%Y-%m-%d").unique().tolist()
    bbox = kml_bbox(polygons)

    accepted = []
    next_idx = next_negative_index(base_offset=n_needed)

    while len(accepted) < n_needed:
        print("  Buscando candidatos negativos...")
        candidates = find_negative_candidates(
            rng,
            all_coords=all_coords,
            bbox=bbox,
            polygons=polygons,
            all_dates=all_dates,
            n_candidates=6,
        )
        if not candidates:
            print("  Sin candidatos validos, reintentando...")
            continue

        print(f"  Descargando {len(candidates)} candidatos...")
        downloaded = []
        for i, (lat, lon, date_str) in enumerate(candidates):
            lat, lon, date_str, tmp, ok, land_frac, cloud_frac, reason = download_negative_candidate(
                conn,
                lat,
                lon,
                date_str,
                i,
            )
            if ok and tmp.exists():
                downloaded.append((lat, lon, date_str, tmp))
                print(
                    f"  OK ({lat:.3f}, {lon:.3f}) {date_str}  "
                    f"land={land_frac:.2%} cloud={cloud_frac:.2%}"
                )
            else:
                extra = f" motivo={reason}"
                if land_frac is not None and cloud_frac is not None:
                    extra += f" land={land_frac:.2%} cloud={cloud_frac:.2%}"
                print(f"  DESCARTADO ({lat:.3f}, {lon:.3f}){extra}")
            time.sleep(2)

        for _, _, date_str, tmp in downloaded:
            if len(accepted) >= n_needed:
                safe_unlink(tmp)
                continue

            label = validate_negative(tmp)
            if label is None or label == "NO":
                safe_unlink(tmp)
                print("  [NEG] descartado por usuario")
                continue

            final_path = build_unique_negative_path(date_str, next_idx, label)
            tmp.rename(final_path)
            accepted.append(final_path)
            next_idx += 1
            print(f"  [NEG] aceptado: {final_path.name}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_positives", type=int, default=3)
    parser.add_argument("--n_negatives", type=int, default=None)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--skip_negatives", action="store_true")
    parser.add_argument("--retry_failed_positives", action="store_true")
    parser.add_argument("--clear_failed_positive_cache", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()

    if args.clear_failed_positive_cache:
        safe_unlink(CSV_POSITIVE_FAILURES)
        print(f"Cache de fallos positivos borrada: {CSV_POSITIVE_FAILURES}")
        return

    df_all = pd.read_csv(CSV_CANDIDATES)
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    ds = xr.open_dataset(NC_PATH)
    df_all = ensure_nc_idx(df_all, ds)
    polygons = parse_kml_polygons(KML_PATH)

    target_negatives = args.n_negatives if args.n_negatives is not None else args.n_positives

    positives = df_all.sort_values("Pixels per LW", ascending=False).reset_index(drop=True)

    existing_positive_files = list_existing_patches_by_label("SI")
    existing_negative_files = list_existing_patches_by_label("NO")
    positive_failure_cache = load_positive_failure_cache()
    existing_positive_indices = {
        idx
        for idx in (extract_positive_rank_index(path) for path in existing_positive_files)
        if idx is not None
    }

    print("\n=== CONECTANDO A OPENEO ===")
    conn = openeo.connect("openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()
    print("Autenticado\n")

    rng = np.random.default_rng(seed=args.seed)

    print("=== POSITIVOS ===\n")
    print(
        f"Objetivo SI={args.n_positives}. "
        f"Ya existen {len(existing_positive_files)} positivos y {len(existing_negative_files)} negativos.\n"
    )

    current_positive_total = len(existing_positive_files)
    for idx, row in positives.iterrows():
        if current_positive_total >= args.n_positives:
            break

        if idx in existing_positive_indices:
            existing = [
                path
                for path in existing_positive_files
                if extract_positive_rank_index(path) == idx
            ]
            if existing:
                print(f"  Ya existe: {existing[0].name}")
            continue

        patch = {
            "lat": float(row.Latitude),
            "lon": float(row.Longitude),
            "date": row.Date.strftime("%Y-%m-%d"),
            "nc_idx": int(row.nc_idx),
            "idx": idx,
        }

        if not args.retry_failed_positives:
            skip_reason = should_skip_failed_positive(idx, positive_failure_cache)
            if skip_reason:
                print(f"  SKIP rank {idx}: {skip_reason}")
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

        mask, n_px = build_mask(tmp, patch["nc_idx"], ds)
        final_name = f"{patch['date'].replace('-', '')}_SI_{n_px:06d}_{patch['idx']:02d}.tif"
        final_path = PATCHES_DIR / final_name
        tmp.rename(final_path)
        save_mask(final_path, mask)
        positive_failure_cache = clear_positive_failure(patch["idx"], positive_failure_cache)
        existing_positive_indices.add(idx)
        current_positive_total += 1
        print(f"  OK {final_name} ({n_px} px en patch) -> total SI={current_positive_total}")
        time.sleep(3)

    if current_positive_total < args.n_positives:
        print(
            f"\nWARN: se han conseguido {current_positive_total} positivos, "
            f"por debajo del objetivo {args.n_positives}."
        )
    else:
        print(f"\nObjetivo de positivos alcanzado: {current_positive_total}/{args.n_positives}")

    if not args.skip_negatives:
        print("\n=== NEGATIVOS (validacion visual) ===\n")
        current_negative_total = len(list_existing_patches_by_label("NO"))
        negatives_needed = max(target_negatives - current_negative_total, 0)
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
        )

    print("\n=== VERIFICACION FINAL ===")
    for tif in iter_patch_files(PATCHES_DIR):
        with rasterio.open(tif) as src:
            shape = src.read().shape
            dtype = src.dtypes[0]
        ok = shape == (11, 256, 256) and dtype == "float32"
        print(f"  {'OK' if ok else 'WARN'} {tif.name}  {shape}  {dtype}")

    final_positive_total = len(list_existing_patches_by_label("SI"))
    final_negative_total = len(list_existing_patches_by_label("NO"))
    print(
        f"\nRESUMEN FINAL -> SI={final_positive_total} / objetivo {args.n_positives} | "
        f"NO={final_negative_total} / objetivo {target_negatives}"
    )


if __name__ == "__main__":
    main()
