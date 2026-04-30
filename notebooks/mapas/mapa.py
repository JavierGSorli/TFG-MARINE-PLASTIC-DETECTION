import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

# ── 1. Cargar el CSV ──────────────────────────────────────────────────────────
# Ajusta el separador si tu CSV usa ";" en lugar de ","
df = pd.read_excel("C:\\CDIA_oficial\\tfg\\tfg-marine-plastic-detection\\data\\windrows_nature\\general\\41467_2024_48674_MOESM3_ESM.xlsx")

# Limpiar comas decimales europeas si las hubiera
for col in ["Latitude", "Longitude"]:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(",", ".").astype(float)

# Primeras 500 filas
df500 = df.head(14000).dropna(subset=["Latitude", "Longitude"])

# ── 2. Crear el mapa base centrado en el Mediterráneo ─────────────────────────
mapa = folium.Map(
    location=[38.0, 15.0],
    zoom_start=5,
    tiles="CartoDB dark_matter"
)

# ── 3. Añadir puntos con info al hacer clic ───────────────────────────────────
for _, row in df500.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4,
        color="#00d4ff",
        fill=True,
        fill_color="#00d4ff",
        fill_opacity=0.7,
        popup=folium.Popup(
            f"""
            <b>Fecha:</b> {row['Date']}<br>
            <b>Píxeles LW:</b> {row['Pixels per LW']}<br>
            <b>Distancia costa:</b> {row['Distance to land (m)']} m<br>
            <b>Coord:</b> {row['Latitude']:.4f}, {row['Longitude']:.4f}
            """,
            max_width=220
        ),
        tooltip=f"{row['Date']} | {row['Pixels per LW']} px"
    ).add_to(mapa)

# ── 4. Guardar ────────────────────────────────────────────────────────────────
mapa.save("mapa_litter_windrows.html")
print(f"✅ Mapa guardado con {len(df500)} puntos → mapa_litter_windrows.html")