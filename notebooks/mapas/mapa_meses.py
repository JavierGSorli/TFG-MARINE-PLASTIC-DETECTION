import pandas as pd
import folium
from folium.plugins import HeatMap
from folium import IFrame
import json

# ── 1. Cargar el Excel ────────────────────────────────────────────────────────
df = pd.read_excel("C:\\CDIA_oficial\\tfg\\tfg-marine-plastic-detection\\data\\windrows_nature\\general\\41467_2024_48674_MOESM3_ESM.xlsx")

for col in ["Latitude", "Longitude"]:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(",", ".").astype(float)

df500 = df.head(14000).dropna(subset=["Latitude", "Longitude"])
df500["Year"] = df500["Year"].astype(int)
df500["Month"] = df500["Month"].astype(int)

# ── 2. Convertir puntos a GeoJSON agrupados por año-mes ──────────────────────
features = []
for _, row in df500.iterrows():
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [row["Longitude"], row["Latitude"]]},
        "properties": {
            "year": int(row["Year"]),
            "month": int(row["Month"]),
            "date": str(row["Date"]),
            "pixels": int(row["Pixels per LW"]),
            "dist": int(row["Distance to land (m)"])
        }
    })

geojson_data = json.dumps({"type": "FeatureCollection", "features": features})
years = sorted(df500["Year"].unique().tolist())
months_names = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

# ── 3. Crear HTML completo ────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Marine Litter Windrows — Mediterráneo</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', sans-serif; background:#0a0e1a; color:#e0e6f0; display:flex; flex-direction:column; height:100vh; }}

  /* HEADER */
  header {{ background:#0d1224; padding:12px 24px; border-bottom:1px solid #1e2a4a;
            display:flex; align-items:center; gap:16px; flex-shrink:0; }}
  header h1 {{ font-size:16px; font-weight:600; color:#7eb8f7; letter-spacing:0.5px; }}
  header span {{ font-size:12px; color:#4a6080; }}

  /* PANEL FILTROS */
  #panel {{ background:#0d1224; padding:14px 24px; border-bottom:1px solid #1e2a4a;
            display:flex; align-items:center; gap:24px; flex-wrap:wrap; flex-shrink:0; }}
  .filter-group {{ display:flex; flex-direction:column; gap:6px; }}
  .filter-group label {{ font-size:11px; color:#4a7ab5; text-transform:uppercase; letter-spacing:1px; }}

  /* BOTONES AÑO */
  #year-btns {{ display:flex; gap:6px; flex-wrap:wrap; }}
  .year-btn {{
    padding:5px 14px; border-radius:20px; border:1px solid #1e3a5f;
    background:#0a1628; color:#7eb8f7; font-size:12px; cursor:pointer;
    transition:all 0.2s;
  }}
  .year-btn:hover {{ background:#1a2e4a; }}
  .year-btn.active {{ background:#1a4a8a; border-color:#4a8aca; color:#fff; }}

  /* SLIDER MESES */
  #month-slider-wrap {{ display:flex; align-items:center; gap:10px; }}
  #month-slider {{
    -webkit-appearance:none; width:220px; height:4px;
    background: linear-gradient(to right, #1a4a8a 0%, #1a4a8a var(--pct,0%), #1e3a5f var(--pct,0%), #1e3a5f 100%);
    border-radius:4px; outline:none; cursor:pointer;
  }}
  #month-slider::-webkit-slider-thumb {{
    -webkit-appearance:none; width:16px; height:16px; border-radius:50%;
    background:#4a8aca; border:2px solid #7eb8f7; cursor:pointer;
  }}
  #month-label {{ font-size:13px; color:#7eb8f7; min-width:90px; }}
  .month-all-btn {{
    padding:4px 10px; border-radius:20px; border:1px solid #1e3a5f;
    background:#0a1628; color:#4a8aca; font-size:11px; cursor:pointer;
    transition:all 0.2s;
  }}
  .month-all-btn:hover {{ background:#1a2e4a; }}

  /* CONTADOR */
  #counter {{ margin-left:auto; font-size:12px; color:#4a6080; }}
  #counter span {{ color:#7eb8f7; font-weight:600; }}

  /* MAPA */
  #map {{ flex:1; }}

  /* POPUP */
  .leaflet-popup-content-wrapper {{
    background:#0d1224 !important; color:#e0e6f0 !important;
    border:1px solid #1e3a5f; border-radius:8px;
  }}
  .leaflet-popup-tip {{ background:#0d1224 !important; }}
  .popup-row {{ display:flex; justify-content:space-between; gap:16px;
                font-size:12px; padding:2px 0; border-bottom:1px solid #1e2a4a; }}
  .popup-row:last-child {{ border-bottom:none; }}
  .popup-key {{ color:#4a8aca; }}
  .popup-val {{ color:#e0e6f0; font-weight:500; }}
</style>
</head>
<body>

<header>
  <h1>🌊 Marine Litter Windrows — Mediterráneo</h1>
  <span>Sentinel-2 · 2015–2021</span>
</header>

<div id="panel">
  <div class="filter-group">
    <label>Año</label>
    <div id="year-btns">
      <button class="year-btn active" data-year="all">Todos</button>
      {''.join(f'<button class="year-btn" data-year="{y}">{y}</button>' for y in years)}
    </div>
  </div>

  <div class="filter-group">
    <label>Mes</label>
    <div id="month-slider-wrap">
      <button class="month-all-btn" id="month-all-btn">Todos</button>
      <input type="range" id="month-slider" min="1" max="12" value="1">
      <span id="month-label">— Todos —</span>
    </div>
  </div>

  <div id="counter">Mostrando <span id="count">0</span> detecciones</div>
</div>

<div id="map"></div>

<script>
const ALL_DATA = {geojson_data};
const MONTHS = {json.dumps(months_names)};

const map = L.map('map', {{center:[38,15], zoom:5}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution:'&copy; OpenStreetMap &copy; CARTO', maxZoom:19
}}).addTo(map);

let markers = L.layerGroup().addTo(map);
let activeYear = 'all';
let activeMonth = 'all';
let sliderActive = false;

function makeIcon() {{
  return L.circleMarker;
}}

function render() {{
  markers.clearLayers();
  const features = ALL_DATA.features.filter(f => {{
    const p = f.properties;
    if (activeYear !== 'all' && p.year !== activeYear) return false;
    if (activeMonth !== 'all' && p.month !== activeMonth) return false;
    return true;
  }});

  features.forEach(f => {{
    const [lng, lat] = f.geometry.coordinates;
    const p = f.properties;
    const circle = L.circleMarker([lat, lng], {{
      radius: Math.min(3 + p.pixels * 0.3, 12),
      color: '#4a8aca',
      fillColor: '#00d4ff',
      fillOpacity: 0.75,
      weight: 1
    }});
    circle.bindPopup(`
      <div style="min-width:180px">
        <div style="font-size:13px;font-weight:600;color:#7eb8f7;margin-bottom:6px">
          📍 LW Detection
        </div>
        <div class="popup-row"><span class="popup-key">Fecha</span><span class="popup-val">${{p.date}}</span></div>
        <div class="popup-row"><span class="popup-key">Píxeles LW</span><span class="popup-val">${{p.pixels}}</span></div>
        <div class="popup-row"><span class="popup-key">Dist. costa</span><span class="popup-val">${{p.dist}} m</span></div>
        <div class="popup-row"><span class="popup-key">Coords</span><span class="popup-val">${{lat.toFixed(4)}}, ${{lng.toFixed(4)}}</span></div>
      </div>
    `);
    markers.addLayer(circle);
  }});

  document.getElementById('count').textContent = features.length;
}}

// ── Filtro año ────────────────────────────────────────────────────────────────
document.getElementById('year-btns').addEventListener('click', e => {{
  if (!e.target.classList.contains('year-btn')) return;
  document.querySelectorAll('.year-btn').forEach(b => b.classList.remove('active'));
  e.target.classList.add('active');
  const val = e.target.dataset.year;
  activeYear = val === 'all' ? 'all' : parseInt(val);
  render();
}});

// ── Filtro mes (slider) ───────────────────────────────────────────────────────
const slider = document.getElementById('month-slider');
const monthLabel = document.getElementById('month-label');

slider.addEventListener('input', () => {{
  sliderActive = true;
  const m = parseInt(slider.value);
  activeMonth = m;
  monthLabel.textContent = MONTHS[m-1];
  const pct = ((m-1)/11*100).toFixed(1);
  slider.style.setProperty('--pct', pct+'%');
  render();
}});

document.getElementById('month-all-btn').addEventListener('click', () => {{
  sliderActive = false;
  activeMonth = 'all';
  monthLabel.textContent = '— Todos —';
  slider.style.setProperty('--pct', '0%');
  render();
}});

render();
</script>
</body>
</html>"""

with open("mapa_litter_windrows_meses.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Mapa guardado con filtros → mapa_litter_windrows_meses.html")
print(f"   Abre el archivo en tu navegador")