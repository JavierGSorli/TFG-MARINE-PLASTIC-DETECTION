#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage import measure


DEFAULT_MASK = Path(
    r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
    r"\notebooks\indices\pruebasghana3\ghana2_fdi_ndvi_mask.tif"
)


def read_mask(path):
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Mask shape no soportada: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Se esperaba una máscara 2D y se obtuvo {arr.shape}")

    return (arr > 0).astype(np.uint8)


def read_georef_from_tif(mask_path):
    with tiff.TiffFile(str(mask_path)) as tif:
        page = tif.pages[0]
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tie_tag = page.tags.get("ModelTiepointTag")
        geo_ascii = page.tags.get("GeoAsciiParamsTag")

        if scale_tag is None or tie_tag is None:
            raise ValueError("El TIFF no contiene ModelPixelScaleTag/ModelTiepointTag")

        scale = scale_tag.value
        tie = tie_tag.value
        crs_name = geo_ascii.value if geo_ascii is not None else ""

    pixel_width = float(scale[0])
    pixel_height = -float(scale[1])
    x_origin = float(tie[3])
    y_origin = float(tie[4])
    return x_origin, y_origin, pixel_width, pixel_height, crs_name


def utm_to_latlon(easting, northing, zone_number, northern=True):
    # WGS84
    a = 6378137.0
    e = 0.08181919084262149
    e1sq = 0.006739496742276434
    k0 = 0.9996

    x = easting - 500000.0
    y = northing
    if not northern:
        y -= 10000000.0

    m = y / k0
    mu = m / (
        a
        * (
            1
            - e**2 / 4.0
            - 3.0 * e**4 / 64.0
            - 5.0 * e**6 / 256.0
        )
    )

    e1 = (1.0 - math.sqrt(1.0 - e**2)) / (1.0 + math.sqrt(1.0 - e**2))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32.0
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32.0
    j3 = 151 * e1**3 / 96.0
    j4 = 1097 * e1**4 / 512.0

    fp = (
        mu
        + j1 * math.sin(2 * mu)
        + j2 * math.sin(4 * mu)
        + j3 * math.sin(6 * mu)
        + j4 * math.sin(8 * mu)
    )

    c1 = e1sq * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e**2) / ((1 - e**2 * math.sin(fp) ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - e**2 * math.sin(fp) ** 2)
    d = x / (n1 * k0)

    q1 = n1 * math.tan(fp) / r1
    q2 = d**2 / 2.0
    q3 = (
        (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e1sq)
        * d**4
        / 24.0
    )
    q4 = (
        (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * e1sq - 3 * c1**2)
        * d**6
        / 720.0
    )
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * d**3 / 6.0
    q7 = (
        (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e1sq + 24 * t1**2)
        * d**5
        / 120.0
    )
    lon = (q5 - q6 + q7) / math.cos(fp)

    lon0 = math.radians((zone_number - 1) * 6 - 180 + 3)
    lon = lon0 + lon

    return math.degrees(lat), math.degrees(lon)


def contours_to_lonlat(mask, x_origin, y_origin, pixel_width, pixel_height, zone_number, northern):
    contours = measure.find_contours(mask, 0.5)
    out = []
    for contour in contours:
        coords = []
        for row, col in contour:
            x = x_origin + col * pixel_width
            y = y_origin + row * pixel_height
            lat, lon = utm_to_latlon(x, y, zone_number, northern=northern)
            coords.append([lat, lon])  # Leaflet usa [lat, lon]
        if len(coords) >= 3:
            out.append(coords)
    return out


def html_template(center_lat, center_lon, polylines_js):
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Predicted debris map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      padding: 0;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([{center_lat}, {center_lon}], 11);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const polylines = {polylines_js};
    const group = L.featureGroup();

    for (const coords of polylines) {{
      const poly = L.polyline(coords, {{
        color: '#1f77ff',
        weight: 3,
        opacity: 0.95
      }}).addTo(map);
      group.addLayer(poly);
    }}

    if (polylines.length > 0) {{
      map.fitBounds(group.getBounds(), {{padding: [20, 20]}});
    }}
  </script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Visualiza una máscara binaria georreferenciada sobre OSM sin depender de geopandas/rasterio."
    )
    ap.add_argument("--mask_tif", default=str(DEFAULT_MASK), help="Ruta al TIFF binario")
    ap.add_argument("--out_html", default="", help="HTML de salida opcional")
    args = ap.parse_args()

    mask_path = Path(args.mask_tif)
    if not mask_path.exists():
        raise SystemExit(f"No existe: {mask_path}")

    mask = read_mask(mask_path)
    x_origin, y_origin, pixel_width, pixel_height, crs_name = read_georef_from_tif(mask_path)

    crs_name_lower = crs_name.lower()
    zone_number = None
    northern = True
    if "utm zone" in crs_name_lower:
        try:
            token = crs_name_lower.split("utm zone", 1)[1].split("|", 1)[0].strip()
            zone_number = int("".join(ch for ch in token if ch.isdigit()))
            northern = "n" in token
        except Exception:
            pass
    if zone_number is None:
        raise SystemExit(f"No pude inferir la zona UTM desde el CRS: {crs_name}")

    polylines = contours_to_lonlat(
        mask, x_origin, y_origin, pixel_width, pixel_height, zone_number, northern
    )

    if not polylines:
        raise SystemExit("No se encontraron contornos en la máscara.")

    all_lat = [lat for line in polylines for lat, _ in line]
    all_lon = [lon for line in polylines for _, lon in line]
    center_lat = float(np.mean(all_lat))
    center_lon = float(np.mean(all_lon))

    out_html = Path(args.out_html) if args.out_html else mask_path.with_name(mask_path.stem + "_light_map.html")
    html = html_template(center_lat, center_lon, json.dumps(polylines))
    out_html.write_text(html, encoding="utf-8")
    print(f"Mapa guardado en: {out_html}")


if __name__ == "__main__":
    main()
