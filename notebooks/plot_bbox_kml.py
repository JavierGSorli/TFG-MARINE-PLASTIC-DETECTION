#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import math
import os
import xml.etree.ElementTree as ET
import argparse

import matplotlib.pyplot as plt
import requests
from PIL import Image
from matplotlib.patches import Rectangle


DEFAULT_KML = (
    r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\data\mapa_estrecho.kml"
)
LON_MIN, LON_MAX = -6.0, -1.5
LAT_MIN, LAT_MAX = 35.0, 36.5
ZOOM = 8
TILE_SIZE = 256


def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def tile_to_lonlat(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def fetch_tile(z, x, y):
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    headers = {"User-Agent": "bbox-plotter/1.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def build_basemap(lon_min, lon_max, lat_min, lat_max, zoom):
    x0f, y1f = lonlat_to_tile(lon_min, lat_min, zoom)
    x1f, y0f = lonlat_to_tile(lon_max, lat_max, zoom)

    x0 = int(math.floor(min(x0f, x1f)))
    x1 = int(math.floor(max(x0f, x1f)))
    y0 = int(math.floor(min(y0f, y1f)))
    y1 = int(math.floor(max(y0f, y1f)))

    canvas = Image.new("RGB", ((x1 - x0 + 1) * TILE_SIZE, (y1 - y0 + 1) * TILE_SIZE))

    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = fetch_tile(zoom, x, y)
            canvas.paste(tile, ((x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE))

    west, north = tile_to_lonlat(x0, y0, zoom)
    east, south = tile_to_lonlat(x1 + 1, y1 + 1, zoom)
    return canvas, (west, east, south, north)


def parse_kml_geometries(kml_path):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(kml_path)
    root = tree.getroot()

    geometries = []
    for tag in ["Polygon", "LineString", "Point"]:
        for geom in root.findall(f".//kml:{tag}", ns):
            coords_node = geom.find(".//kml:coordinates", ns)
            if coords_node is None or not coords_node.text:
                continue

            coords = []
            for item in coords_node.text.strip().split():
                parts = item.split(",")
                if len(parts) < 2:
                    continue
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))

            if coords:
                geometries.append((tag, coords))

    return geometries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kml", default=DEFAULT_KML, help="Ruta al KML a visualizar")
    args = ap.parse_args()

    basemap, extent = build_basemap(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, ZOOM)
    geometries = []
    if args.kml and os.path.exists(args.kml):
        geometries = parse_kml_geometries(args.kml)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(basemap, extent=extent, origin="upper")

    rect = Rectangle(
        (LON_MIN, LAT_MIN),
        LON_MAX - LON_MIN,
        LAT_MAX - LAT_MIN,
        fill=False,
        edgecolor="crimson",
        linewidth=2.5,
        linestyle="--",
        label="BBOX actual",
    )
    ax.add_patch(rect)

    first_poly = True
    first_line = True
    first_point = True
    for geom_type, coords in geometries:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if geom_type == "Polygon":
            ax.plot(
                xs, ys,
                color="tab:cyan",
                linewidth=2,
                label="KML polígono" if first_poly else None,
            )
            first_poly = False
        elif geom_type == "LineString":
            ax.plot(
                xs, ys,
                color="yellow",
                linewidth=2,
                label="KML línea" if first_line else None,
            )
            first_line = False
        elif geom_type == "Point":
            ax.scatter(
                xs, ys,
                color="magenta",
                s=30,
                label="KML punto" if first_point else None,
            )
            first_point = False

    ax.set_xlim(LON_MIN - 0.2, LON_MAX + 0.2)
    ax.set_ylim(LAT_MIN - 0.2, LAT_MAX + 0.2)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("BBOX y KML del Estrecho sobre OpenStreetMap")
    ax.grid(alpha=0.2, color="white")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
