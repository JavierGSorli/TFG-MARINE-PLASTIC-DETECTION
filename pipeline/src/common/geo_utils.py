from __future__ import annotations

import xml.etree.ElementTree as ET


def parse_kml_polygons(kml_path):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(kml_path)
    root = tree.getroot()

    polygons = []
    for poly in root.findall(".//kml:Polygon", ns):
        coords_node = poly.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
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

        if len(coords) >= 3:
            polygons.append(coords)

    if not polygons:
        raise ValueError(f"No se encontraron poligonos validos en {kml_path}")
    return polygons


def point_in_polygon(lon, lat, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = (y1 > lat) != (y2 > lat)
        if intersects:
            xinters = (x2 - x1) * (lat - y1) / ((y2 - y1) + 1e-15) + x1
            if lon < xinters:
                inside = not inside
    return inside


def point_in_kml_mask(lon, lat, polygons):
    return any(point_in_polygon(lon, lat, poly) for poly in polygons)


def kml_bbox(polygons):
    all_lons = [lon for poly in polygons for lon, _ in poly]
    all_lats = [lat for poly in polygons for _, lat in poly]
    return {
        "lon_min": min(all_lons),
        "lon_max": max(all_lons),
        "lat_min": min(all_lats),
        "lat_max": max(all_lats),
    }


def filter_points_to_kml(df, polygons):
    bbox = kml_bbox(polygons)
    df_bbox = df[
        df.Longitude.between(bbox["lon_min"], bbox["lon_max"])
        & df.Latitude.between(bbox["lat_min"], bbox["lat_max"])
    ].copy()

    df_kml = df_bbox[
        df_bbox.apply(
            lambda row: point_in_kml_mask(float(row.Longitude), float(row.Latitude), polygons),
            axis=1,
        )
    ].copy()
    return df_bbox, df_kml, bbox
