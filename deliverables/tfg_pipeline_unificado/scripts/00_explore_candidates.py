from __future__ import annotations

import pandas as pd

from config import CSV_CANDIDATES, KML_PATH, XLSX_PATH, ensure_output_dirs
from geo_utils import filter_points_to_kml, parse_kml_polygons


def main():
    ensure_output_dirs()

    df = pd.read_excel(XLSX_PATH)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    polygons = parse_kml_polygons(KML_PATH)
    df_bbox, gibraltar, _ = filter_points_to_kml(df, polygons)

    print("=== DETECCIONES EN GIBRALTAR ===")
    print(f"Total: {len(gibraltar)} de {len(df)} ({100 * len(gibraltar) / len(df):.1f}%)")
    print(f"En bbox envolvente del KML: {len(df_bbox)}")

    print("\n--- Por ano ---")
    print(gibraltar.Year.value_counts().sort_index().to_string())

    print("\n--- Top 10 meses con mas detecciones ---")
    print(
        gibraltar.groupby(["Year", "Month"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    gibraltar["datetime_str"] = gibraltar.Date.dt.strftime("%Y%m%d") + gibraltar.CodeT

    print("\n--- Pixeles por LW ---")
    print(gibraltar["Pixels per LW"].describe().to_string())

    print("\n--- Distancia a tierra (m) ---")
    print(gibraltar["Distance to land (m)"].describe().to_string())

    candidates = gibraltar[
        (gibraltar["Pixels per LW"] >= 5) & (gibraltar["Distance to land (m)"] > 1280)
    ].copy()

    print("\n=== CANDIDATOS PARA PATCHES POSITIVOS ===")
    print(f"Total candidatos: {len(candidates)}")

    top_dates = (
        candidates.groupby("datetime_str").size().sort_values(ascending=False).head(10)
    )
    print("\n--- Top 10 fechas candidatas ---")
    print(top_dates.to_string())

    top5 = (
        candidates.sort_values("Pixels per LW", ascending=False)
        .head(5)[
            [
                "Latitude",
                "Longitude",
                "Date",
                "CodeT",
                "datetime_str",
                "Pixels per LW",
                "Distance to land (m)",
            ]
        ]
    )
    print("\n--- Top 5 candidatos por tamano ---")
    print(top5.to_string(index=False))

    candidates.to_csv(CSV_CANDIDATES, index=False)
    print(f"\nGuardado: {CSV_CANDIDATES}")


if __name__ == "__main__":
    main()
