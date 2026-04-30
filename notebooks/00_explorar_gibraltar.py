# 00_explorar_gibraltar.py
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET

DATA_XLSX_PATH = Path(
    r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
    r"\data\windrows_nature\general\41467_2024_48674_MOESM3_ESM.xlsx"
)
KML_PATH = Path(
    r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\data\mapa_estrecho.kml"
)
OUT_CSV_PATH = Path(
    r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection"
    r"\results\auto\gibraltar_candidatos.csv"
)


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
        raise ValueError(f"No se encontraron polígonos válidos en {kml_path}")
    return polygons


def point_in_polygon(lon, lat, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > lat) != (y2 > lat))
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
    return dict(
        lon_min=min(all_lons),
        lon_max=max(all_lons),
        lat_min=min(all_lats),
        lat_max=max(all_lats),
    )


def filter_points_to_kml(df, polygons):
    bbox = kml_bbox(polygons)
    df_bbox = df[
        df.Longitude.between(bbox["lon_min"], bbox["lon_max"]) &
        df.Latitude.between(bbox["lat_min"], bbox["lat_max"])
    ].copy()

    df_kml = df_bbox[
        df_bbox.apply(
            lambda r: point_in_kml_mask(float(r.Longitude), float(r.Latitude), polygons),
            axis=1,
        )
    ].copy()
    return df_bbox, df_kml, bbox


def main():
    df = pd.read_excel(DATA_XLSX_PATH)
    polygons = parse_kml_polygons(KML_PATH)
    df_bbox, gib, bbox = filter_points_to_kml(df, polygons)

    print(f"=== DETECCIONES EN GIBRALTAR ===")
    print(f"Total: {len(gib)} de {len(df)} ({100*len(gib)/len(df):.1f}%)")
    print(f"En bbox envolvente del KML: {len(df_bbox)}")

    print(f"\n--- Por año ---")
    print(gib.Year.value_counts().sort_index().to_string())

    print(f"\n--- Top 10 meses con más detecciones ---")
    print(gib.groupby(["Year", "Month"]).size()
            .sort_values(ascending=False).head(10).to_string())

    print(f"\n--- Valores únicos de CodeT (muestra) ---")
    print(gib.CodeT.value_counts().head(10).to_string())

    gib["datetime_str"] = gib.Date.dt.strftime("%Y%m%d") + gib.CodeT
    print(f"\n--- Ejemplos de datetime_str ---")
    print(gib["datetime_str"].head(5).tolist())

    print(f"\n--- Píxeles por LW (tamaño de la mancha) ---")
    print(gib["Pixels per LW"].describe().to_string())
    print(f"\nDetecciones con >= 5 píxeles: {(gib['Pixels per LW'] >= 5).sum()}")
    print(f"Detecciones con >= 10 píxeles: {(gib['Pixels per LW'] >= 10).sum()}")

    print(f"\n--- Distancia a tierra (m) ---")
    print(gib["Distance to land (m)"].describe().to_string())

    lejos = (gib["Distance to land (m)"] > 1280).sum()
    print(
        f"\nDetecciones a > 1.28 km de tierra "
        f"(patch mayormente agua): {lejos} ({100*lejos/len(gib):.0f}%)"
    )

    candidatos = gib[
        (gib["Pixels per LW"] >= 5) &
        (gib["Distance to land (m)"] > 1280)
    ].copy()

    print(f"\n=== CANDIDATOS PARA PATCHES POSITIVOS ===")
    print(f"Total candidatos: {len(candidatos)}")

    print(f"\n--- Top 10 fechas con más detecciones candidatas ---")
    top_fechas = (candidatos.groupby("datetime_str")
                            .size()
                            .sort_values(ascending=False)
                            .head(10))
    print(top_fechas.to_string())

    top5 = (candidatos.sort_values("Pixels per LW", ascending=False)
                      .head(5)
            [["Latitude", "Longitude", "Date", "CodeT",
              "datetime_str", "Pixels per LW", "Distance to land (m)"]])
    print(f"\n--- Top 5 candidatos por tamaño de mancha ---")
    print(top5.to_string())

    candidatos.to_csv(OUT_CSV_PATH, index=False)
    print(f"\n✓ Guardado: {OUT_CSV_PATH.name}")


if __name__ == "__main__":
    main()
