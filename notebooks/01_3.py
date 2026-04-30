# 01_download_final_v2.py
import openeo
import pandas as pd
import numpy as np
import rasterio
import requests
import time
from pathlib import Path
from datetime import timedelta, datetime

MARIDA_BANDS = ["B01","B02","B03","B04","B05",
                "B06","B07","B08","B8A","B11","B12"]

HALF_DEG  = (3.84 / 2) / 111.0
TARGET_PX = 256
PAUSA_S   = 3
MAX_CLOUD = 0.15
MAX_LAND  = 0.30

OUT_DIR  = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                r"\results\auto\test_patches_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                r"\results\auto\gibraltar_candidatos.csv")

df_all     = pd.read_csv(CSV_PATH)
df_all["Date"] = pd.to_datetime(df_all["Date"])
all_coords = df_all[["Latitude","Longitude"]].values
BBOX_GIB   = dict(lon_min=-6.0, lon_max=-1.5,
                  lat_min=35.0, lat_max=36.5)

top3 = (df_all.sort_values("Pixels per LW", ascending=False)
              .head(3).reset_index(drop=True))

# ── Funciones ─────────────────────────────────────────────────
def check_cdse(lat, lon, date_str, days=1):
    h  = HALF_DEG
    d  = datetime.strptime(date_str, "%Y-%m-%d")
    d0 = (d - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    d1 = (d + timedelta(days=days)).strftime("%Y-%m-%dT23:59:59Z")
    url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
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
        r = requests.get(url, timeout=15)
        return len(r.json().get("value", [])) > 0
    except:
        return False

def check_scl_quality(conn, lat, lon, date_str, days=1):
    d  = datetime.strptime(date_str, "%Y-%m-%d")
    t0 = (d - timedelta(days=days)).strftime("%Y-%m-%d")
    t1 = (d + timedelta(days=days)).strftime("%Y-%m-%d")
    bbox = {
        "west":  round(lon - HALF_DEG, 6),
        "east":  round(lon + HALF_DEG, 6),
        "south": round(lat - HALF_DEG, 6),
        "north": round(lat + HALF_DEG, 6),
        "crs":   "EPSG:4326"
    }
    tmp = OUT_DIR / "_tmp_scl.tif"
    try:
        s2 = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent  = bbox,
            temporal_extent = [t0, t1],
            bands           = ["SCL"],
            max_cloud_cover = 30
        )
        s2.reduce_dimension(
            dimension="t", reducer="median"
        ).download(tmp, format="GTiff")

        with rasterio.open(tmp) as src:
            scl = src.read(1).astype(float)
        tmp.unlink()

        total      = scl.size
        cloud_frac = np.isin(scl, [8, 9, 10]).sum() / total
        land_frac  = np.isin(scl, [4, 5]).sum()     / total
        water_frac = np.isin(scl, [6, 7]).sum()     / total
        info = (f"agua={water_frac*100:.0f}% "
                f"nube={cloud_frac*100:.0f}% "
                f"tierra={land_frac*100:.0f}%")
        ok = cloud_frac <= MAX_CLOUD and land_frac <= MAX_LAND
        return ok, info
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return False, f"SCL error: {e}"

def download_and_convert(conn, lat, lon, date_str,
                         out_path, days=1):
    d  = datetime.strptime(date_str, "%Y-%m-%d")
    t0 = (d - timedelta(days=days)).strftime("%Y-%m-%d")
    t1 = (d + timedelta(days=days)).strftime("%Y-%m-%d")
    bbox = {
        "west":  round(lon - HALF_DEG, 6),
        "east":  round(lon + HALF_DEG, 6),
        "south": round(lat - HALF_DEG, 6),
        "north": round(lat + HALF_DEG, 6),
        "crs":   "EPSG:4326"
    }
    tmp = OUT_DIR / "_tmp_bands.tif"

    for attempt in range(3):
        try:
            s2 = conn.load_collection(
                "SENTINEL2_L2A",
                spatial_extent  = bbox,
                temporal_extent = [t0, t1],
                bands           = MARIDA_BANDS,
                max_cloud_cover = 30
            )
            s2.reduce_dimension(
                dimension="t", reducer="median"
            ).download(tmp, format="GTiff")
            break  # descarga OK

        except Exception as e:
            msg = str(e)
            if tmp.exists():
                tmp.unlink()
            if ("503" in msg or "429" in msg) and attempt < 2:
                wait = 40 * (attempt + 1)
                print(f"\n      ⟳ {msg[:50]} "
                      f"— reintentando en {wait}s")
                time.sleep(wait)
                continue
            print(f"\n      ✗ descarga: {e}")
            return False
    else:
        return False

    try:
        with rasterio.open(tmp) as src:
            data = src.read().astype("float32")
            meta = src.meta.copy()
            tf   = src.transform
        tmp.unlink()

        # Convertir INT16 → float32 0-1
        data = data / 10000.0

        # Limpiar nodata (-32768/10000 = -3.2768)
        data[data < -0.5] = np.nan
        data = np.clip(data, 0.0, None)

        # Imputar NaN con 0 (agua limpia)
        data = np.nan_to_num(data, nan=0.0)

        # Verificación simple: datos válidos en rango L2A
        valid = data[data > 0]
        if len(valid) == 0:
            print(f"\n      ✗ Sin píxeles válidos")
            return False
        if data.max() > 1.5:
            print(f"\n      ✗ Max fuera de rango: {data.max():.3f}")
            return False
        if data.mean() < 0.0005:
            print(f"\n      ✗ Imagen prácticamente vacía")
            return False

        print(f"\n      rango=[{data.min():.4f},{data.max():.4f}] "
              f"mean={data.mean():.5f} ✓")

        # Recortar centro 256×256
        C, H, W = data.shape
        if H < TARGET_PX or W < TARGET_PX:
            print(f"      ✗ Muy pequeño: {H}×{W}")
            return False

        r0   = (H - TARGET_PX) // 2
        c0   = (W - TARGET_PX) // 2
        data = data[:, r0:r0+TARGET_PX, c0:c0+TARGET_PX]
        new_tf = rasterio.windows.transform(
            rasterio.windows.Window(c0, r0, TARGET_PX, TARGET_PX),
            tf
        )

        meta.update(dtype="float32", count=11,
                    width=TARGET_PX, height=TARGET_PX,
                    transform=new_tf, nodata=0.0)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
        return True

    except Exception as e:
        for f in [tmp, out_path]:
            if Path(str(f)).exists():
                Path(str(f)).unlink()
        print(f"\n      ✗ conversión: {e}")
        return False

def gen_negative(pos_lat, pos_lon, date_str,
                 conn, rng, filename):
    out_path = OUT_DIR / filename
    attempt  = 0
    while True:
        attempt += 1
        angle   = rng.uniform(0, 2 * np.pi)
        dist    = rng.uniform(25/111, 45/111)
        lat_neg = pos_lat + dist * np.sin(angle)
        lon_neg = pos_lon + dist * np.cos(angle)

        if not (BBOX_GIB["lon_min"] < lon_neg < BBOX_GIB["lon_max"]
                and BBOX_GIB["lat_min"] < lat_neg < BBOX_GIB["lat_max"]):
            continue

        dists = np.sqrt((all_coords[:,0] - lat_neg)**2 +
                        (all_coords[:,1] - lon_neg)**2)
        if dists.min() < 5/111:
            continue

        print(f"  Intento {attempt}: "
              f"({lat_neg:.4f},{lon_neg:.4f})", end="")

        if not check_cdse(lat_neg, lon_neg, date_str):
            print(" — sin catálogo")
            continue

        print(" — SCL...", end=" ", flush=True)
        ok_scl, info = check_scl_quality(
            conn, lat_neg, lon_neg, date_str
        )
        print(f"[{info}]", end=" ")
        if not ok_scl:
            time.sleep(3)
            continue

        print("↓ descargando...", end=" ", flush=True)
        t_start = time.time()
        if download_and_convert(conn, lat_neg, lon_neg,
                                date_str, out_path):
            print(f"({time.time()-t_start:.0f}s)")
            return True
        time.sleep(10)

# ── Lista de patches ──────────────────────────────────────────
patches = []
for idx, row in top3.iterrows():
    patches.append({
        "lat":     row.Latitude,
        "lon":     row.Longitude,
        "date":    row.Date.strftime("%Y-%m-%d"),
        "label":   "SI",
        "pixels":  int(row["Pixels per LW"]),
        "filename": (f"{row.Date.strftime('%Y%m%d')}_SI_"
                     f"{int(row['Pixels per LW']):06d}_{idx:02d}.tif"),
        "is_neg":  False,
    })
for idx, row in top3.iterrows():
    patches.append({
        "date":    row.Date.strftime("%Y-%m-%d"),
        "label":   "NO",
        "pixels":  0,
        "filename": (f"{row.Date.strftime('%Y%m%d')}_NO_"
                     f"000000_{idx+3:02d}.tif"),
        "is_neg":  True,
        "pos_lat": row.Latitude,
        "pos_lon": row.Longitude,
    })

print("=== PLAN ===")
for p in patches:
    print(f"  {p['filename']}")

# ── Conexión ──────────────────────────────────────────────────
print("\n=== CONECTANDO A OPENEO ===")
conn = openeo.connect("openeo.dataspace.copernicus.eu")
conn.authenticate_oidc()
print("✓ Autenticado\n")

rng = np.random.default_rng(seed=777)

# ── Descarga ──────────────────────────────────────────────────
print("=== DESCARGANDO ===\n")
for i, p in enumerate(patches):
    out_path = OUT_DIR / p["filename"]

    if out_path.exists():
        with rasterio.open(out_path) as src:
            shape = src.read().shape
            dtype = src.dtypes[0]
        if shape == (11, 256, 256) and dtype == "float32":
            print(f"  — Ya existe: {p['filename']}")
            continue
        out_path.unlink()

    if i > 0:
        print(f"  ⏱ Pausa {PAUSA_S}s...", end=" ", flush=True)
        time.sleep(PAUSA_S)
        print("OK")

    if p["is_neg"]:
        print(f"Generando negativo para "
              f"({p['pos_lat']:.3f},{p['pos_lon']:.3f}) "
              f"fecha {p['date']}")
        gen_negative(p["pos_lat"], p["pos_lon"], p["date"],
                     conn, rng, p["filename"])
    else:
        print(f"  SCL: {p['filename']}...", end=" ", flush=True)
        ok_scl, info = check_scl_quality(
            conn, p["lat"], p["lon"], p["date"]
        )
        print(f"[{info}]")
        print(f"  ↓ descargando...", end=" ", flush=True)
        t_start = time.time()
        if download_and_convert(conn, p["lat"], p["lon"],
                                p["date"], out_path):
            print(f"({time.time()-t_start:.0f}s)")
        else:
            print("✗ falló")

# ── Verificación final ────────────────────────────────────────
print("\n=== VERIFICACIÓN FINAL ===")
todos_ok = True
for p in patches:
    path = OUT_DIR / p["filename"]
    if not path.exists():
        print(f"  ✗ {p['filename']} — falta")
        todos_ok = False
        continue
    with rasterio.open(path) as src:
        d     = src.read().astype("float32")
        dtype = src.dtypes[0]
    ok = d.shape == (11, 256, 256) and dtype == "float32"
    print(f"  {'✓' if ok else '✗'} {p['filename']}  "
          f"shape={d.shape}  dtype={dtype}  "
          f"rango=[{d.min():.4f},{d.max():.4f}]  "
          f"mean={d.mean():.5f}")
    if not ok:
        todos_ok = False

print()
print("✓ Listos" if todos_ok else "⚠ Revisar los marcados con ✗")