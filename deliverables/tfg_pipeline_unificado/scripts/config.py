from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "data").exists() and (candidate / "results").exists():
            return candidate
    raise RuntimeError(
        "No se pudo localizar la raiz del proyecto. "
        "Extrae esta carpeta dentro del repo o define TFG_PROJECT_ROOT."
    )


_root_env = os.environ.get("TFG_PROJECT_ROOT")
ROOT = Path(_root_env).expanduser() if _root_env else _find_repo_root(Path(__file__).resolve())

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "auto"

XLSX_PATH = DATA_DIR / "windrows_nature" / "general" / "41467_2024_48674_MOESM3_ESM.xlsx"
NC_PATH = (
    DATA_DIR
    / "windrows_nature"
    / "detallado"
    / "11045944"
    / "WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc"
)
KML_PATH = DATA_DIR / "mapa_estrecho.kml"

MARIDA_ROOT = DATA_DIR / "marida" / "marine-debris.github.io"
UNET_DIR = MARIDA_ROOT / "semantic_segmentation" / "unet"
RF_DIR = MARIDA_ROOT / "semantic_segmentation" / "random_forest"
RESNET_DIR = MARIDA_ROOT / "multi-label" / "resnet"
UTILS_DIR = MARIDA_ROOT / "utils"

UNET_SCRIPT = UNET_DIR / "predict_mask.py"
RESNET_SCRIPT = RESNET_DIR / "predict_resnet.py"
RF_MODEL_PATH = RF_DIR / "rf_classifier.joblib"
RESNET_MODEL_PATH = RESNET_DIR / "trained_models" / "18" / "model.pth"

PATCHES_DIR = RESULTS_DIR / "test_patches_final"
UNET_OUT = RESULTS_DIR / "test_masks_unet"
RF_OUT = RESULTS_DIR / "test_masks_rf"
RESNET_OUT = RESULTS_DIR / "test_resnet_json"
INDICES_OUT = RESULTS_DIR / "test_indices"
EVAL_OUT = RESULTS_DIR / "evaluation"
XGB_OUT = RESULTS_DIR / "xgboost_model"
ERROR_OUT = RESULTS_DIR / "error_analysis"

CSV_CANDIDATES = RESULTS_DIR / "gibraltar_candidatos.csv"
CSV_MASTER = RESULTS_DIR / "predictions_master.csv"
CSV_XGB = RESULTS_DIR / "xgboost_dataset.csv"
CSV_POSITIVE_FAILURES = RESULTS_DIR / "positive_download_failures.csv"

MARIDA_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

BAND_NAMES = MARIDA_BANDS
DEBRIS_CLASS = 1
PATCH_SIZE = 256

PATCH_KM = 3.84
HALF_DEG = (PATCH_KM / 2) / 111.0
TARGET_PX = 256
MAX_CLOUD = 0.30
MAX_LAND = 0.50
MAX_ALL_BANDS_ZERO_FRAC = 0.20
MIN_KEY_BANDS_VALID_FRAC = 0.01
MIN_SEVERE_KEY_BANDS_VALID_FRAC = 0.005
MAX_B04_ZERO_FRAC = 0.99
MAX_B08_ZERO_FRAC = 0.98

FDI_WL_NIR = 842.0
FDI_WL_RED = 665.0
FDI_WL_SWIR = 1610.0

OUTPUT_DIRS = [
    RESULTS_DIR,
    PATCHES_DIR,
    UNET_OUT,
    RF_OUT,
    RESNET_OUT,
    INDICES_OUT,
    EVAL_OUT,
    XGB_OUT,
    ERROR_OUT,
]


def ensure_output_dirs() -> None:
    for path in OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
