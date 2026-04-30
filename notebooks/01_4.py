# 01_download_final_v3.py
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

MARIDA_BANDS = ["B01","B02","B03","B04","B05",
                "B06","B07","B08","B8A","B11","B12"]

HALF_DEG  = (3.84 / 2) / 111.0
TARGET_PX = 256
PAUSA_S   = 3
MAX_CLOUD = 0.15
MAX_LAND  = 0.30

OUT_DIR  = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                r"\results\auto\test_patches_final3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
                r"\results\auto\gibraltar_candidatos.csv")
NC_PATH  = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\data\windrows_nature\detallado\11045944\WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc")  # ← ajusta

df_all     = pd.read_csv(CSV_PATH)
df_all["Date"] = pd.to_datetime(df_all["Date"])
all_coords = df_all[["Latitude","Longitude"]].values
BBOX_GIB   = dict(lon_min=-6.0, lon_max=-1.5,
                  lat_min=35.0, lat_max=36.5)

top3 = (df_all.sort_values("Pixels per LW", ascending=False)
              .head(3).reset_index(drop=True))

# ── NC dataset (cargado una sola vez) ─────────────────────────
ds = xr.open_dataset(NC_PATH)

# ── Funciones de máscara ──────────────────────────────────────
def get_nc_idx(lat, lon):
    d = (ds.lat_centroid.values - lat)**2 + \
        (ds.lon_centroid.values - lon)**2
    return int(d.argmin())

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

def build_mask(patch_path, lat, lon):
    """
    Genera máscara alineada para el filamento en (lat,lon).
    Devuelve (mask, n_pixels_en_patch).
    """
    nc_idx = get_nc_idx(lat, lon)

    with rasterio.open(patch_path) as src:
        data      = src.read().astype("float32")
        patch_crs = src.crs
        patch_tf  = src.transform
        H, W      = src.height, src.width

    # Centroide del filamento en píxeles del patch
    tr = Transformer.from_crs("EPSG:4326", patch_crs, always_xy=True)
    cx_utm, cy_utm = tr.transform(
        float(ds.lon_centroid.values[nc_idx]),
        float(ds.lat_centroid.values[nc_idx])
    )
    center_col, center_row = ~patch_tf * (cx_utm, cy_utm)

    # Píxeles del filamento — desplazamiento relativo al centroide
    n  = int(ds.n_pixels_fil.values[nc_idx])
    px = ds.pixel_x.values[nc_idx][:n]
    py = ds.pixel_y.values[nc_idx][:n]
    cx_scene = float(ds.x_centroid.values[nc_idx])
    cy_scene = float(ds.y_centroid.values[nc_idx])

    cols = np.round(center_col + (py - cy_scene)).astype(int)
    rows = np.round(center_row + (px - cx_scene)).astype(int)

    valid = (rows>=0)&(rows<H)&(cols>=0)&(cols<W)
    mask_raw = np.zeros((H,W), dtype=np.uint8)
    mask_raw[rows[valid], cols[valid]] = 1

    # Shift automático via FDI
    fdi    = compute_fdi(data)
    dr, dc = find_shift(mask_raw, fdi)

    # Aplicar shift
    ys, xs = np.where(mask_raw)
    rs = np.clip(ys + dr, 0, H-1)
    cs = np.clip(xs + dc, 0, W-1)
    mask = np.zeros((H,W), dtype=np.uint8)
    mask[rs, cs] = 1

    return mask, int(mask.sum())

def save_mask(patch_path, mask, n_px, label_suffix=""):
    """Guarda máscara como tif. Nombre incluye n_px en el patch."""
    # Nombre: YYYYMMDD_SI_NPXPATCH_IDX[_LABEL].tif
    stem  = patch_path.stem             # ej: 20200329_SI_001053_00
    parts = stem.split("_")             # [YYYY, SI, NPIXNATURE, IDX]
    parts[2] = f"{n_px:06d}"            # reemplazar por px en patch
    new_stem  = "_".join(parts)
    if label_suffix:
        new_stem += f"_{label_suffix}"
    out_path = patch_path.parent / f"{new_stem}_mask.tif"

    with rasterio.open(patch_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis])
    return out_path

# ── UI para validación de negativos ──────────────────────────
def make_rgb(data):
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    for c in range(3):
        p2, p98 = np.percentile(rgb[:,:,c], (2,98))
        rgb[:,:,c] = np.clip(
            (rgb[:,:,c]-p2)/(p98-p2+1e-10), 0, 1)
    return (rgb * 255).astype(np.uint8)

def validate_negative(patch_path):
    """
    Muestra el patch negativo y pide validación.
    Devuelve: 'NO' | 'DUDOSO' | 'CLARO' | 'DIFICIL'
    o None si el usuario cancela.
    """
    with rasterio.open(patch_path) as src:
        data = src.read().astype("float32")
    rgb_arr = make_rgb(data)

    result = [None]

    root = tk.Tk()
    root.title(f"Validar negativo: {patch_path.name}")
    root.resizable(False, False)

    # Imagen
    img    = Image.fromarray(rgb_arr).resize((400, 400))
    photo  = ImageTk.PhotoImage(img)
    lbl    = tk.Label(root, image=photo)
    lbl.pack(pady=10)

    # Título
    tk.Label(root,
             text=f"{patch_path.name}\n¿Qué ves en esta imagen?",
             font=("Arial", 11)).pack()

    # Botones
    frame = ttk.Frame(root)
    frame.pack(pady=10)

    def on_click(val):
        result[0] = val
        root.destroy()

    btn_cfg = [
        ("NO — descartar",   "NO",      "#e74c3c", "white"),
        ("DUDOSO",           "DUDOSO",  "#f39c12", "white"),
        ("CLARO",            "CLARO",   "#27ae60", "white"),
        ("DIFÍCIL",          "DIFICIL", "#2980b9", "white"),
    ]
    for text, val, bg, fg in btn_cfg:
        tk.Button(
            frame, text=text, width=14,
            bg=bg, fg=fg, font=("Arial", 10, "bold"),
            command=lambda v=val: on_click(v)
        ).pack(side=tk.LEFT, padx=5)

    root.mainloop()
    return result[0]

# ── Funciones de descarga (igual que antes) ───────────────────
def check_cdse(lat, lon, date_str):
    h  = HALF_DEG
    d  = datetime.strptime(date_str, "%Y-%m-%d")
    d0 = d.strftime("%Y-%m-%dT00:00:00Z")
    d1 = d.strftime("%Y-%m-%dT23:59:59Z")
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
        print(f"[CDSE] comprobando producto para fecha={date_str} lat={lat:.5f} lon={lon:.5f}")
        r = requests.get(url, timeout=15)
        n = len(r.json().get("value", []))
        if n > 0:
            print(f"[CDSE] OK: encontrado {n} producto(s)")
            return True
        else:
            print(f"[CDSE] FAIL: no hay producto")
            return False
    except Exception as e:
        print(f"[CDSE] ERROR: fecha={date_str} lat={lat:.5f} lon={lon:.5f} -> {e}")
        return False

def check_scl_quality(conn, lat, lon, date_str):
    bbox = {
        "west":  round(lon - HALF_DEG, 6),
        "east":  round(lon + HALF_DEG, 6),
        "south": round(lat - HALF_DEG, 6),
        "north": round(lat + HALF_DEG, 6),
        "crs":   "EPSG:4326"
    }
    tmp = OUT_DIR / "_tmp_scl.tif"
    try:
        print(f"[SCL] descargando SCL para fecha={date_str} lat={lat:.5f} lon={lon:.5f}")
        s2 = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent  = bbox,
            temporal_extent = [date_str, date_str],
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
        cloud_frac = np.isin(scl, [8,9,10]).sum() / total
        land_frac  = np.isin(scl, [4,5]).sum()    / total
        water_frac = np.isin(scl, [6,7]).sum()     / total
        info = (f"agua={water_frac*100:.0f}% "
                f"nube={cloud_frac*100:.0f}% "
                f"tierra={land_frac*100:.0f}%")
        return cloud_frac<=MAX_CLOUD and land_frac<=MAX_LAND, info
    except Exception as e:
        if tmp.exists(): tmp.unlink()
        print(f"[SCL] ERROR: fecha={date_str} lat={lat:.5f} lon={lon:.5f} -> {e}")
        return False, f"SCL error: {e}"

def download_and_convert(conn, lat, lon, date_str, out_path):
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
            print(f"[DOWNLOAD] intento {attempt+1}/3 -> fecha={date_str} lat={lat:.5f} lon={lon:.5f} out={out_path.name}")
            s2 = conn.load_collection(
                "SENTINEL2_L2A",
                spatial_extent  = bbox,
                temporal_extent = [date_str, date_str],
                bands           = MARIDA_BANDS,
                max_cloud_cover = 30
            )
            s2.reduce_dimension(
                dimension="t", reducer="median"
            ).download(tmp, format="GTiff")
            print(f"[DOWNLOAD] OK descarga bruta -> {tmp.name}")
            break
        except Exception as e:
            msg = str(e)
            if tmp.exists(): tmp.unlink()
            print(f"[DOWNLOAD] FAIL intento {attempt+1}/3 -> fecha={date_str} lat={lat:.5f} lon={lon:.5f} error={msg}")
            if ("503" in msg or "429" in msg) and attempt < 2:
                wait = 40 * (attempt+1)
                print(f"\n      ⟳ reintentando en {wait}s")
                time.sleep(wait)
                continue
            print(f"\n      ✗ {e}")
            return False
    else:
        return False

    try:
        with rasterio.open(tmp) as src:
            data = src.read().astype("float32")
            meta = src.meta.copy()
            tf   = src.transform
        tmp.unlink()

        data = data / 10000.0
        data[data < -0.5] = np.nan
        data = np.clip(data, 0.0, None)
        data = np.nan_to_num(data, nan=0.0)

        valid = data[data > 0]
        if len(valid)==0 or data.max()>1.5 or data.mean()<0.0005:
            return False

        print(f"\n      rango=[{data.min():.4f},{data.max():.4f}] "
              f"mean={data.mean():.5f} ✓")

        C, H, W = data.shape
        if H < TARGET_PX or W < TARGET_PX:
            return False

        r0 = (H - TARGET_PX) // 2
        c0 = (W - TARGET_PX) // 2
        data   = data[:, r0:r0+TARGET_PX, c0:c0+TARGET_PX]
        new_tf = rasterio.windows.transform(
            rasterio.windows.Window(c0, r0, TARGET_PX, TARGET_PX), tf
        )
        meta.update(dtype="float32", count=11,
                    width=TARGET_PX, height=TARGET_PX,
                    transform=new_tf, nodata=0.0)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
        return True

    except Exception as e:
        for f in [tmp, out_path]:
            if Path(str(f)).exists(): Path(str(f)).unlink()
        print(f"\n      ✗ conversión: {e}")
        return False

def gen_negative(pos_lat, pos_lon, date_str, conn, rng, filename):
    """Genera negativo verificado. Muestra UI para validación."""
    while True:
        # Generar coordenada candidata
        for _ in range(500):
            angle   = rng.uniform(0, 2*np.pi)
            dist    = rng.uniform(25/111, 45/111)
            lat_neg = pos_lat + dist * np.sin(angle)
            lon_neg = pos_lon + dist * np.cos(angle)

            if not (BBOX_GIB["lon_min"] < lon_neg < BBOX_GIB["lon_max"]
                    and BBOX_GIB["lat_min"] < lat_neg < BBOX_GIB["lat_max"]):
                continue
            dists = np.sqrt((all_coords[:,0]-lat_neg)**2 +
                            (all_coords[:,1]-lon_neg)**2)
            if dists.min() < 5/111:
                print(f"[NEG] descartado: demasiado cerca de un candidato ({dists.min()*111:.2f} km)")
                continue

            if not check_cdse(lat_neg, lon_neg, date_str):
                print("[NEG] descartado: no hay producto CDSE")
                continue

            ok_scl, info = check_scl_quality(
                conn, lat_neg, lon_neg, date_str
            )
            print(f"  ({lat_neg:.4f},{lon_neg:.4f}) [{info}]",
                  end=" ")
            if not ok_scl:
                print("[NEG] descartado: calidad SCL insuficiente")
                time.sleep(3)
                continue

            # Descarga temporal para mostrar en UI
            tmp_patch = OUT_DIR / "_tmp_neg_preview.tif"
            print("↓...", end=" ", flush=True)
            if not download_and_convert(
                    conn, lat_neg, lon_neg, date_str, tmp_patch):
                time.sleep(5)
                continue

            # Mostrar UI al usuario
            label = validate_negative(tmp_patch)

            if label == "NO" or label is None:
                # Descartar y buscar otra coordenada
                if tmp_patch.exists():
                    tmp_patch.unlink()
                print("[NEG] descartado por usuario")
                break  # romper el for interno → nueva iteración del while

            # Aceptado: renombrar con etiqueta
            out_path = OUT_DIR / filename
            # Insertar etiqueta en el nombre
            stem  = Path(filename).stem
            parts = stem.split("_")  # [YYYY, NO, 000000, IDX]
            new_name = f"{'_'.join(parts)}_{label}.tif"
            final_path = OUT_DIR / new_name
            tmp_patch.rename(final_path)
            print(f"✓ guardado como {new_name}")
            return final_path

        # Si el for terminó sin break (500 intentos fallidos)
        # continúa el while con nueva semilla
        print("  ⚠ 500 intentos fallidos, reintentando...")
        time.sleep(5)

# ── Lista de patches ──────────────────────────────────────────
patches = []
for idx, row in top3.iterrows():
    patches.append({
        "lat":    row.Latitude,
        "lon":    row.Longitude,
        "date":   row.Date.strftime("%Y-%m-%d"),
        "label":  "SI",
        "filename": (f"{row.Date.strftime('%Y%m%d')}_SI_"
                     f"XXXXXX_{idx:02d}.tif"),  # XXXXXX → px en patch
        "is_neg": False,
    })
for idx, row in top3.iterrows():
    patches.append({
        "date":   row.Date.strftime("%Y-%m-%d"),
        "label":  "NO",
        "filename": (f"{row.Date.strftime('%Y%m%d')}_NO_"
                     f"000000_{idx+3:02d}.tif"),
        "is_neg":  True,
        "pos_lat": row.Latitude,
        "pos_lon": row.Longitude,
    })

# ── Conexión ──────────────────────────────────────────────────
print("\n=== CONECTANDO A OPENEO ===")
conn = openeo.connect("openeo.dataspace.copernicus.eu")
conn.authenticate_oidc()
print("✓ Autenticado\n")

rng = np.random.default_rng(seed=777)

# ── Descarga principal ────────────────────────────────────────
print("=== DESCARGANDO ===\n")
for i, p in enumerate(patches):
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
        continue

    # ── Positivo ──────────────────────────────────────────────
    # Nombre provisional (sin saber aún los px en patch)
    tmp_path = OUT_DIR / p["filename"].replace("XXXXXX", "tmp")

    # Saltar si ya existe un archivo con este prefijo
    prefix = f"{p['date'].replace('-','')}_SI_"
    idx_str = f"_{i:02d}.tif"
    existing = list(OUT_DIR.glob(f"{prefix}*{idx_str}"))
    if existing:
        print(f"  — Ya existe: {existing[0].name}")
        continue

    print(f"  SCL: {p['filename']}...", end=" ", flush=True)
    ok_scl, info = check_scl_quality(
        conn, p["lat"], p["lon"], p["date"]
    )
    print(f"[{info}]")

    print(f"  ↓ descargando...", end=" ", flush=True)
    t_start = time.time()
    if not download_and_convert(
            conn, p["lat"], p["lon"], p["date"], tmp_path):
        print("✗ falló")
        continue
    print(f"✓ ({time.time()-t_start:.0f}s)")

    # Generar máscara y obtener n_px en patch
    print(f"  Generando máscara...", end=" ", flush=True)
    mask, n_px = build_mask(tmp_path, p["lat"], p["lon"])
    print(f"{n_px} px en patch")

    # Renombrar con n_px real
    final_name = p["filename"].replace("XXXXXX", f"{n_px:06d}")
    final_path = OUT_DIR / final_name
    tmp_path.rename(final_path)

    # Guardar máscara junto al patch
    save_mask(final_path, mask, n_px)
    print(f"  ✓ {final_name}")

# ── Verificación final ────────────────────────────────────────
print("\n=== VERIFICACIÓN FINAL ===")
for tif in sorted(OUT_DIR.glob("*.tif")):
    if "mask" in tif.name or tif.name.startswith("_"):
        continue
    with rasterio.open(tif) as src:
        shape = src.read().shape
        dtype = src.dtypes[0]
    ok = shape==(11,256,256) and dtype=="float32"
    print(f"  {'✓' if ok else '✗'} {tif.name}  {shape} {dtype}")