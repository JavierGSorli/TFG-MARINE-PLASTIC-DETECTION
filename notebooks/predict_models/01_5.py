# 01_download_final_v4.py
import openeo
import pandas as pd
import numpy as np
import rasterio
import xarray as xr
import requests
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pyproj import Transformer
from scipy.signal import fftconvolve
from pathlib import Path
from datetime import datetime
import importlib.util

MARIDA_BANDS = ["B01","B02","B03","B04","B05",
                "B06","B07","B08","B8A","B11","B12"]

HALF_DEG  = (3.84 / 2) / 111.0
TARGET_PX = 256
MAX_CLOUD = 0.30
MAX_LAND  = 0.50

OUT_DIR  = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                r"\results\auto\test_patches_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                r"\results\auto\gibraltar_candidatos.csv")
NC_PATH  = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\data\windrows_nature\detallado\11045944\WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc")   # ← ajusta
KML_MASK_PATH = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\data\mapa_estrecho.kml")

df_all = pd.read_csv(CSV_PATH)
df_all["Date"] = pd.to_datetime(df_all["Date"])
ds = xr.open_dataset(NC_PATH)

helpers_path = Path(__file__).with_name("00_explorar_gibraltar.py")
spec = importlib.util.spec_from_file_location("explorar_gibraltar_helpers", helpers_path)
explorar_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explorar_helpers)

KML_POLYGONS = explorar_helpers.parse_kml_polygons(KML_MASK_PATH)
BBOX_GIB = explorar_helpers.kml_bbox(KML_POLYGONS)

all_coords = df_all[["Latitude", "Longitude"]].values

top3 = (df_all.sort_values("Pixels per LW", ascending=False)
              .head(3).reset_index(drop=True))

# ── Añadir nc_idx al CSV si no existe ────────────────────────
# El CSV de Gibraltar es un subconjunto del .nc.
# Buscamos el índice en el .nc por lat+lon (únicos).
if "nc_idx" not in df_all.columns:
    print("Calculando nc_idx para el CSV...")
    def find_nc_idx(lat, lon):
        d = (ds.lat_centroid.values - lat)**2 + \
            (ds.lon_centroid.values - lon)**2
        return int(d.argmin())
    df_all["nc_idx"] = df_all.apply(
        lambda r: find_nc_idx(r.Latitude, r.Longitude), axis=1)
    df_all.to_csv(CSV_PATH, index=False)
    print("✓ nc_idx añadido y guardado en CSV")
    top3 = (df_all.sort_values("Pixels per LW", ascending=False)
                  .head(3).reset_index(drop=True))

# ── Funciones de máscara ──────────────────────────────────────
def compute_fdi(data):
    B06, B08, B11 = data[5], data[7], data[9]
    return B08 - (B06 + (B11-B06) *
                  ((832.9-664.6)/(1613.7-664.6)) * 10)

def find_shift(mask, fdi, max_shift=80):
    fdi_norm = fdi - fdi.min()
    if fdi_norm.max() > 0:
        fdi_norm = fdi_norm / fdi_norm.max()
    corr = fftconvolve(fdi_norm, mask[::-1,::-1], mode="same")
    H, W = corr.shape
    cy, cx = H//2, W//2
    r0, r1 = max(0,cy-max_shift), min(H,cy+max_shift)
    c0, c1 = max(0,cx-max_shift), min(W,cx+max_shift)
    region = corr[r0:r1, c0:c1]
    best   = np.unravel_index(region.argmax(), region.shape)
    return int(best[0]+r0-cy), int(best[1]+c0-cx)

def build_mask(patch_path, nc_idx):
    """Genera máscara alineada usando nc_idx directo."""
    with rasterio.open(patch_path) as src:
        data      = src.read().astype("float32")
        patch_crs = src.crs
        patch_tf  = src.transform
        H, W      = src.height, src.width

    tr = Transformer.from_crs("EPSG:4326", patch_crs, always_xy=True)
    cx_utm, cy_utm = tr.transform(
        float(ds.lon_centroid.values[nc_idx]),
        float(ds.lat_centroid.values[nc_idx])
    )
    center_col, center_row = ~patch_tf * (cx_utm, cy_utm)

    n  = int(ds.n_pixels_fil.values[nc_idx])
    px = ds.pixel_x.values[nc_idx][:n]
    py = ds.pixel_y.values[nc_idx][:n]
    cx_s = float(ds.x_centroid.values[nc_idx])
    cy_s = float(ds.y_centroid.values[nc_idx])

    cols = np.round(center_col + (py - cy_s)).astype(int)
    rows = np.round(center_row + (px - cx_s)).astype(int)

    valid = (rows>=0)&(rows<H)&(cols>=0)&(cols<W)
    mask_raw = np.zeros((H,W), dtype=np.uint8)
    mask_raw[rows[valid], cols[valid]] = 1

    fdi    = compute_fdi(data)
    dr, dc = find_shift(mask_raw, fdi)

    ys, xs = np.where(mask_raw)
    rs = np.clip(ys+dr, 0, H-1)
    cs = np.clip(xs+dc, 0, W-1)
    mask = np.zeros((H,W), dtype=np.uint8)
    mask[rs, cs] = 1

    return mask, int(mask.sum())

def save_mask(patch_path, mask, n_px):
    out = patch_path.parent / \
          patch_path.name.replace(".tif", "_mask.tif")
    with rasterio.open(patch_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mask[np.newaxis])
    return out

# ── RGB para visualización ────────────────────────────────────
def make_rgb(data):
    """B04/B03/B02, estiramiento fijo min=0 max=0.1"""
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    rgb = np.clip(rgb / 0.1, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)

def estimate_land_cloud_fraction(data):
    """
    Heurística rápida para filtrar candidatos negativos:
    - tierra: NDWI bajo y NIR/SWIR relativamente altos
    - nubes: visible muy brillante y SWIR también alto
    """
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

    land = valid & (
        np.isfinite(ndwi) &
        (ndwi < 0.0) &
        (b08 > 0.08) &
        (b11 > 0.03)
    )

    cloud = valid & (
        (b02 > 0.18) &
        (b03 > 0.18) &
        (b04 > 0.18) &
        (b11 > 0.10)
    )

    land_frac = float(land.sum()) / float(valid.sum())
    cloud_frac = float(cloud.sum()) / float(valid.sum())
    return land_frac, cloud_frac


def safe_unlink(path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def next_negative_index(out_dir, base_offset=3):
    max_idx = base_offset - 1
    for tif in out_dir.glob("*_NO_*_*.tif"):
        if "mask" in tif.name or tif.name.startswith("_"):
            continue
        parts = tif.stem.split("_")
        if len(parts) >= 4 and parts[1] == "NO":
            try:
                idx = int(parts[3])
                max_idx = max(max_idx, idx)
            except Exception:
                pass
    return max_idx + 1


def build_unique_negative_path(out_dir, date_str, idx_num, label):
    idx_str = f"{idx_num:02d}"
    name = f"{date_str.replace('-', '')}_NO_000000_{idx_str}_{label}.tif"
    path = out_dir / name
    if not path.exists():
        return path

    suffix = 2
    while True:
        alt = out_dir / f"{date_str.replace('-', '')}_NO_000000_{idx_str}_{label}_{suffix}.tif"
        if not alt.exists():
            return alt
        suffix += 1

# ── UI validación negativos ───────────────────────────────────
def validate_negative(patch_path):
    with rasterio.open(patch_path) as src:
        data = src.read().astype("float32")
    rgb_arr = make_rgb(data)
    result  = [None]

    root = tk.Tk()
    root.title(f"Validar: {patch_path.name}")
    root.resizable(False, False)

    img   = Image.fromarray(rgb_arr).resize((420, 420))
    photo = ImageTk.PhotoImage(img)
    tk.Label(root, image=photo).pack(pady=8)
    tk.Label(root,
             text=f"{patch_path.name}\n¿Qué ves?",
             font=("Arial", 10)).pack()

    frame = ttk.Frame(root)
    frame.pack(pady=10)

    def on_click(v):
        result[0] = v
        root.destroy()

    for text, val, bg in [
        ("NO — descartar",  "NO",      "#e74c3c"),
        ("DUDOSO",          "DUDOSO",  "#f39c12"),
        ("CLARO",           "CLARO",   "#27ae60"),
        ("DIFÍCIL",         "DIFICIL", "#2980b9"),
    ]:
        tk.Button(frame, text=text, width=13, bg=bg,
                  fg="white", font=("Arial", 10, "bold"),
                  command=lambda v=val: on_click(v)
                  ).pack(side=tk.LEFT, padx=4)

    root.mainloop()
    return result[0]

# ── Descarga y conversión ─────────────────────────────────────
def download_and_convert(conn, lat, lon, date_str, out_path,
                         reduce=True):
    bbox = {
        "west":  round(lon - HALF_DEG, 6),
        "east":  round(lon + HALF_DEG, 6),
        "south": round(lat - HALF_DEG, 6),
        "north": round(lat + HALF_DEG, 6),
        "crs":   "EPSG:4326"
    }
    tmp = OUT_DIR / f"_tmp_bands_{out_path.stem}.tif"

    for attempt in range(3):
        try:
            s2 = conn.load_collection(
                "SENTINEL2_L2A",
                spatial_extent  = bbox,
                temporal_extent = [date_str, date_str],
                bands           = MARIDA_BANDS,
                max_cloud_cover = 30
            )
            # Solo reducir si hay potencialmente más de 1 imagen
            # (Sentinel-2 puede pasar 2 veces el mismo día en zonas
            #  de overlap entre tiles)
            result = s2.reduce_dimension(
                dimension="t", reducer="median"
            ) if reduce else s2
            result.download(tmp, format="GTiff")
            break
        except Exception as e:
            msg = str(e)
            if tmp.exists(): tmp.unlink()
            if ("503" in msg or "429" in msg) and attempt < 2:
                time.sleep(40*(attempt+1))
                continue
            if "429" in msg:
                return "rate_limited"
            return "download_error"
    else:
        return "download_error"

    try:
        with rasterio.open(tmp) as src:
            data = src.read().astype("float32")
            meta = src.meta.copy()
            tf   = src.transform
        tmp.unlink()

        # Escalar solo si los valores son > 10 (están en DN 0-10000)
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

        C, H, W = data.shape
        if H < TARGET_PX or W < TARGET_PX:
            return "too_small"

        r0 = (H - TARGET_PX) // 2
        c0 = (W - TARGET_PX) // 2
        data   = data[:, r0:r0+TARGET_PX, c0:c0+TARGET_PX]
        new_tf = rasterio.windows.transform(
            rasterio.windows.Window(c0, r0, TARGET_PX, TARGET_PX),
            tf
        )
        meta.update(dtype="float32", count=11,
                    width=TARGET_PX, height=TARGET_PX,
                    transform=new_tf, nodata=0.0)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
        return "ok"

    except Exception as e:
        for f in [tmp, out_path]:
            if Path(str(f)).exists(): Path(str(f)).unlink()
        return "conversion_error"

# ── Buscar candidatos negativos  ───────────────────────────────
def find_negative_candidates(rng, n_candidates=5,
                              date_range=("2018-01-01","2021-12-31")):
    """
    Genera N coordenadas candidatas para negativos y verifica
    que existen en el catálogo CDSE. Sin descargar nada todavía.
    Fecha completamente libre dentro del rango.
    Devuelve lista de (lat, lon, date_str) válidas.
    """
    # Fechas disponibles en el CSV (variedad real)
    all_dates = df_all["Date"].dt.strftime("%Y-%m-%d").unique().tolist()

    candidates = []
    attempts   = 0

    while len(candidates) < n_candidates and attempts < 300:
        attempts += 1
        # Coordenada aleatoria en el bbox
        lat = rng.uniform(BBOX_GIB["lat_min"], BBOX_GIB["lat_max"])
        lon = rng.uniform(BBOX_GIB["lon_min"], BBOX_GIB["lon_max"])

        # Debe caer dentro de la máscara KML del Estrecho
        if not explorar_helpers.point_in_kml_mask(lon, lat, KML_POLYGONS):
            continue

        # Lejos de cualquier detección positiva
        dists = np.sqrt((all_coords[:,0]-lat)**2 +
                        (all_coords[:,1]-lon)**2)
        if dists.min() < 5/111:
            continue

        # Fecha aleatoria del pool de fechas reales
        date_str = str(rng.choice(all_dates))

        # Verificar catálogo (rápido, sin descargar)
        h  = HALF_DEG
        d  = datetime.strptime(date_str, "%Y-%m-%d")
        d0 = d.strftime("%Y-%m-%dT00:00:00Z")
        d1 = d.strftime("%Y-%m-%dT23:59:59Z")
        url = (
            "https://catalogue.dataspace.copernicus.eu"
            "/odata/v1/Products"
            f"?$filter=Collection/Name eq 'SENTINEL-2'"
            f" and Attributes/OData.CSC.StringAttribute/any("
            f"a:a/Name eq 'productType' and a/Value eq 'S2MSI2A')"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;"
            f"POLYGON(({lon-h} {lat-h},{lon+h} {lat-h},"
            f"{lon+h} {lat+h},{lon-h} {lat+h},{lon-h} {lat-h}))')"
            f" and ContentDate/Start gt {d0}"
            f" and ContentDate/Start lt {d1}"
            f"&$top=1"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.json().get("value"):
                candidates.append((lat, lon, date_str))
        except:
            continue

    return candidates

def download_negative_candidate(conn, lat, lon, date_str, tmp_id):
    """Descarga un candidato negativo a archivo temporal."""
    tmp = OUT_DIR / f"_tmp_neg_{tmp_id}.tif"
    status = download_and_convert(conn, lat, lon, date_str, tmp)
    if status != "ok" or not tmp.exists():
        return (lat, lon, date_str, tmp, False, None, None, status)

    try:
        with rasterio.open(tmp) as src:
            data = src.read().astype("float32")
        land_frac, cloud_frac = estimate_land_cloud_fraction(data)
    except Exception:
        safe_unlink(tmp)
        return (lat, lon, date_str, tmp, False, None, None, "read_failed")

    if land_frac > MAX_LAND or cloud_frac > MAX_CLOUD:
        safe_unlink(tmp)
        reason = []
        if land_frac > MAX_LAND:
            reason.append(f"land>{MAX_LAND:.0%}")
        if cloud_frac > MAX_CLOUD:
            reason.append(f"cloud>{MAX_CLOUD:.0%}")
        return (
            lat, lon, date_str, tmp, False, land_frac, cloud_frac,
            ",".join(reason) if reason else "filtered"
        )

    return (lat, lon, date_str, tmp, True, land_frac, cloud_frac, "accepted")

def gen_negatives(conn, rng, n_needed, filename_template):
    """
    Genera N negativos con validación visual. Para cada negativo:
    1. Encuentra N*3 candidatos (solo consulta catálogo, rápido)
    3. Muestra UI para cada uno — si NO, descarta y usa el siguiente
    """
    accepted = []
    next_idx = next_negative_index(OUT_DIR, base_offset=3)

    while len(accepted) < n_needed:
        print(f"  Buscando candidatos negativos...")
        candidates = find_negative_candidates(rng, n_candidates=6)
        if not candidates:
            print("  ⚠ Sin candidatos, reintentando...")
            continue

        # Descargar en secuencial para evitar límites de conexión del backend
        print(f"  Descargando {len(candidates)} candidatos...")
        downloaded = []
        for i, (lat, lon, date) in enumerate(candidates):
            lat, lon, date, tmp, ok, land_frac, cloud_frac, reason = (
                download_negative_candidate(conn, lat, lon, date, i)
            )
            if ok and tmp.exists():
                downloaded.append((lat, lon, date, tmp))
                print(
                    f"  ✓ descargado ({lat:.3f},{lon:.3f}) {date}  "
                    f"land={land_frac:.2%} cloud={cloud_frac:.2%}"
                )
            else:
                extra = f"  motivo={reason}"
                if land_frac is not None and cloud_frac is not None:
                    extra += f"  land={land_frac:.2%} cloud={cloud_frac:.2%}"
                print(f"  ✗ descartado ({lat:.3f},{lon:.3f}){extra}")
            time.sleep(2)

        # Mostrar UI para cada descargado
        for lat, lon, date, tmp in downloaded:
            if len(accepted) >= n_needed:
                safe_unlink(tmp)
                continue

            label = validate_negative(tmp)

            if label is None or label == "NO":
                safe_unlink(tmp)
                print(f"  [NEG] descartado por usuario")
                continue

            # Aceptado — renombrar con etiqueta
            final = build_unique_negative_path(
                OUT_DIR, date, next_idx, label
            )
            new_name = final.name
            tmp.rename(final)
            accepted.append(final)
            next_idx += 1
            print(f"  [NEG] aceptado: {new_name}")

    # Limpiar temporales sobrantes
    for f in OUT_DIR.glob("_tmp_neg_*.tif"):
        safe_unlink(f)

    return accepted

# ── Lista de patches ──────────────────────────────────────────
patches_si = []
for idx, row in top3.iterrows():
    patches_si.append({
        "lat":    row.Latitude,
        "lon":    row.Longitude,
        "date":   row.Date.strftime("%Y-%m-%d"),
        "nc_idx": int(row.nc_idx),
        "idx":    idx,
    })

# ── Conexión ──────────────────────────────────────────────────
print("\n=== CONECTANDO A OPENEO ===")
conn = openeo.connect("openeo.dataspace.copernicus.eu")
conn.authenticate_oidc()
print("✓ Autenticado\n")

rng = np.random.default_rng(seed=777)

# ── 1. Descargar positivos ──────────────────────────────────── 
print("=== POSITIVOS ===\n")
for p in patches_si:
    prefix   = f"{p['date'].replace('-','')}_{p['idx']:02d}"
    existing = list(OUT_DIR.glob(f"*SI*_{p['idx']:02d}.tif"))
    existing = [f for f in existing if "mask" not in f.name]
    if existing:
        print(f"  — Ya existe: {existing[0].name}")
        continue

    tmp = OUT_DIR / f"_tmp_si_{p['idx']:02d}.tif"
    print(f"  Descargando positivo {p['idx']} "
          f"({p['lat']:.3f},{p['lon']:.3f}) {p['date']}...")
    if not download_and_convert(
            conn, p["lat"], p["lon"], p["date"], tmp):
        print("  ✗ falló")
        continue

    # Máscara
    mask, n_px = build_mask(tmp, p["nc_idx"])
    final_name = (f"{p['date'].replace('-','')}_SI_"
                  f"{n_px:06d}_{p['idx']:02d}.tif")
    final = OUT_DIR / final_name
    tmp.rename(final)
    save_mask(final, mask, n_px)
    print(f"  ✓ {final_name}  ({n_px} px en patch)")

    time.sleep(3)   # pausa entre positivos

# ── 2. Generar negativos  ─────────────────────────────────────   
print("\n=== NEGATIVOS (validación visual) ===\n")
gen_negatives(
    conn, rng,
    n_needed          = len(patches_si),
    filename_template = "YYYYMMDD_NO_000000_IDX.tif"
)

# ── Verificación final ────────────────────────────────────────
print("\n=== VERIFICACIÓN FINAL ===")
for tif in sorted(OUT_DIR.glob("*.tif")):
    if "mask" in tif.name or tif.name.startswith("_"):
        continue
    with rasterio.open(tif) as src:
        shape = src.read().shape
        dt    = src.dtypes[0]
    ok = shape==(11,256,256) and dt=="float32"
    print(f"  {'✓' if ok else '✗'} {tif.name}  {shape}")
