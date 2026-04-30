# config.py
# Configuración centralizada del pipeline.
# Importar en todos los scripts con: from config import *

from pathlib import Path

# ── Rutas base ────────────────────────────────────────────────
ROOT = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection")

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "auto"

# Datos de entrada
XLSX_PATH   = DATA_DIR / "windrows_nature" / "general" / "41467_2024_48674_MOESM3_ESM.xlsx"
NC_PATH     = DATA_DIR / "windrows_nature" / "detallado" / "11045944" / \
              "WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc"
KML_PATH    = DATA_DIR / "mapa_estrecho.kml"

# Modelos MARIDA
MARIDA_ROOT  = DATA_DIR / "marida" / "marine-debris.github.io"
UNET_DIR     = MARIDA_ROOT / "semantic_segmentation" / "unet"
RESNET_DIR   = MARIDA_ROOT / "multi-label" / "resnet"

# Resultados
PATCHES_DIR  = RESULTS_DIR / "test_patches_final"
UNET_OUT     = RESULTS_DIR / "test_masks_unet"
RF_OUT       = RESULTS_DIR / "test_masks_rf"
RESNET_OUT   = RESULTS_DIR / "test_resnet_json"
INDICES_OUT  = RESULTS_DIR / "test_indices"
EVAL_OUT     = RESULTS_DIR / "evaluation"
XGB_OUT      = RESULTS_DIR / "xgboost_model"
ERROR_OUT    = RESULTS_DIR / "error_analysis"

CSV_CANDIDATOS = RESULTS_DIR / "gibraltar_candidatos.csv"
CSV_MASTER     = RESULTS_DIR / "predictions_master.csv"
CSV_XGB        = RESULTS_DIR / "xgboost_dataset.csv"

# ── Parámetros Sentinel-2 / MARIDA ───────────────────────────
MARIDA_BANDS = ["B01","B02","B03","B04","B05","B06",
                "B07","B08","B8A","B11","B12"]

BAND_NAMES = MARIDA_BANDS  # alias

MARIDA_CLASSES = {
    1: "Marine Debris",         2: "Dense Sargassum",
    3: "Sparse Sargassum",      4: "Natural Organic Material",
    5: "Ship",                  6: "Clouds",
    7: "Marine Water",          8: "Sediment-Laden Water",
    9: "Foam",                 10: "Turbid Water",
    11: "Shallow Water",
}

DEBRIS_CLASS = 1   # clase Marine Debris en MARIDA

# ── Parámetros de descarga ───────────────────────────────────
PATCH_KM  = 3.84
HALF_DEG  = (PATCH_KM / 2) / 111.0
TARGET_PX = 256
MAX_CLOUD = 0.30
MAX_LAND  = 0.50

# ── FDI ──────────────────────────────────────────────────────
FDI_WL_NIR  = 842.0
FDI_WL_RED  = 665.0
FDI_WL_SWIR = 1610.0
