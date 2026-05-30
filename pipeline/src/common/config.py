from __future__ import annotations

"""Configuracion central del pipeline.

Las rutas se resuelven de forma relativa al propio repositorio para que el
pipeline sea portable entre maquinas. Si se necesita forzar otra ubicacion,
puede definirse la variable de entorno ``TFG_PROJECT_ROOT``.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("TFG_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)).resolve()
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

OUTPUTS_DIR = PIPELINE_ROOT / "outputs"
REPORTS_DIR = PIPELINE_ROOT / "reports"
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
SRC_DIR = PIPELINE_ROOT / "src"


# ---------------------------------------------------------------------------
# Datos externos / entrada
# ---------------------------------------------------------------------------

SENTINEL2_DIR = DATA_DIR / "sentinel2"
PATCHES_DIR = SENTINEL2_DIR / "patches"
EXTERNAL_MODELS_DATA_DIR = DATA_DIR / "external_models"

AREA_ESTUDIO_DIR = DATA_DIR / "area_estudio"
KML_PATH = AREA_ESTUDIO_DIR / "mapa_estrecho.kml"

WINDROWS_DIR = DATA_DIR / "windrows_nature"
XLSX_PATH = WINDROWS_DIR / "general" / "41467_2024_48674_MOESM3_ESM.xlsx"
NC_PATH = (
    WINDROWS_DIR
    / "detallado"
    / "11045944"
    / "WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc"
)

MARIDA_ROOT = DATA_DIR / "marida" / "marine-debris.github.io"
UNET_DIR = MARIDA_ROOT / "semantic_segmentation" / "unet"
RF_DIR = MARIDA_ROOT / "semantic_segmentation" / "random_forest"
RESNET_DIR = MARIDA_ROOT / "multi-label" / "resnet"
UTILS_DIR = MARIDA_ROOT / "utils"

UNET_SCRIPT = UNET_DIR / "predict_mask.py"
UNET_MODEL_PATH = UNET_DIR / "trained_models" / "44" / "model.pth"
RESNET_SCRIPT = RESNET_DIR / "predict_resnet.py"
RF_MODEL_PATH = RF_DIR / "rf_classifier.joblib"
RF_MODEL_PATHS = {
    "full": RF_DIR / "rf_classifier_full.joblib",
    "no_texture": RF_DIR / "rf_classifier_no_texture.joblib",
    "indices_only": RF_DIR / "rf_classifier_indices_only.joblib",
    "bands_only": RF_DIR / "rf_classifier_bands_only.joblib",
}
RESNET_MODEL_PATH = RESNET_DIR / "trained_models" / "18" / "model.pth"
RF_PREDICT_SCRIPT = NOTEBOOKS_DIR / "randomforest" / "predict_mask_rf.py"
INDICES_PREDICT_SCRIPT = NOTEBOOKS_DIR / "indices" / "03_predict_indices.py"
INDICES_WATER_PREDICT_SCRIPT = NOTEBOOKS_DIR / "indices" / "03_predict_indices2.py"


# ---------------------------------------------------------------------------
# Estructura de outputs por fases
# ---------------------------------------------------------------------------

OUT_PHASE_DATASET    = OUTPUTS_DIR / "01_dataset"
OUT_PHASE_BASELINES  = OUTPUTS_DIR / "02_baselines"
OUT_PHASE_EVALUATION = OUTPUTS_DIR / "03_evaluation"
OUT_PHASE_HYBRID     = OUTPUTS_DIR / "04_hybrid_and_maps"

# Dataset (fase 01)
DATASET_FILTERS_PREVIOUS_OUT = OUT_PHASE_DATASET / "filtros_previos"
DATASET_DOWNLOAD_OUT         = OUT_PHASE_DATASET / "download"

CSV_CANDIDATES               = DATASET_FILTERS_PREVIOUS_OUT / "gibraltar_candidates.csv"
CSV_DOWNLOAD_FAILURES        = DATASET_DOWNLOAD_OUT / "candidate_download_failures.csv"
DATASET_METADATA_PATH        = DATASET_DOWNLOAD_OUT / "dataset_metadata.csv"
DATASET_METADATA_GROUPED_PATH = DATASET_DOWNLOAD_OUT / "dataset_metadata_with_groups.csv"
GROUPED_SPLITS_PHASE_OUT     = OUT_PHASE_DATASET / "grouped_splits"
GROUPKFOLD_SPLITS_OUT        = GROUPED_SPLITS_PHASE_OUT / "groupkfold"
GROUPKFOLD_FOLDS_PATH        = GROUPKFOLD_SPLITS_OUT / "folds.csv"
DOWNLOAD_LOG_PATH            = DATASET_DOWNLOAD_OUT / "download_attempts_log.csv"

# Baselines (fase 02) — outputs pesados/intermedios en data/, resúmenes evaluados en outputs/
UNET_OUT    = SENTINEL2_DIR / "unet_masks"
RF_OUT      = SENTINEL2_DIR / "rf_masks"
RESNET_OUT  = SENTINEL2_DIR / "resnet_predictions"
INDICES_BASE_OUT = SENTINEL2_DIR / "indices"
SAM_PHASE_OUT    = SENTINEL2_DIR / "sam"
SAM_PROB_DIR     = SAM_PHASE_OUT / "prob"
SAM_CALIBRATED_MASKS_OUT = SAM_PHASE_OUT / "calibrated_masks"
EXTERNAL_MODELS_OUT = EXTERNAL_MODELS_DATA_DIR / "marinedebrisdetector"

UNET_PHASE_OUT            = OUT_PHASE_BASELINES / "unet"
RF_PHASE_OUT              = OUT_PHASE_BASELINES / "rf"
RESNET_PHASE_OUT          = OUT_PHASE_BASELINES / "resnet"
INDICES_NO_WATER_OUT      = INDICES_BASE_OUT / "no_water_mask"
INDICES_WATER_OUT         = INDICES_BASE_OUT / "water_mask"
INDICES_OUT               = INDICES_NO_WATER_OUT
EXTERNAL_B09_ZERO_OUT     = EXTERNAL_MODELS_OUT / "b09_zero"
EXTERNAL_B09_COPY_B8A_OUT = EXTERNAL_MODELS_OUT / "b09_copy_b8a"
EXTERNAL_B09_INTERP_OUT   = EXTERNAL_MODELS_OUT / "b09_interpolate_b8a_b11"

# Evaluation (fase 03)
EVAL_THRESHOLDS_OUT        = OUT_PHASE_EVALUATION / "thresholds"
EVAL_CALIBRATED_OUT       = OUT_PHASE_EVALUATION / "calibrated_outputs"
EVAL_UNIFIED_OUT          = OUT_PHASE_EVALUATION / "unified"
EVAL_PATCH_LEVEL_OUT      = OUT_PHASE_EVALUATION / "patch_level"
EVAL_PIXELWISE_OUT        = OUT_PHASE_EVALUATION / "pixelwise"
EVAL_TOLERANT_OUT         = OUT_PHASE_EVALUATION / "tolerant"
EVAL_ERROR_ANALYSIS_OUT   = OUT_PHASE_EVALUATION / "diagnostics"

PREDICTIONS_MASTER_PATH = EVAL_UNIFIED_OUT / "predictions_master.csv"
THRESHOLDS_PATH         = EVAL_THRESHOLDS_OUT / "thresholds_selected.csv"
CSV_MASTER              = PREDICTIONS_MASTER_PATH
EVAL_OUT                = OUT_PHASE_EVALUATION
ERROR_OUT               = EVAL_ERROR_ANALYSIS_OUT / "error_examples"

UNET_CALIBRATED_MASKS_OUT = UNET_OUT / "calibrated_masks"
EXTERNAL_CALIBRATED_MASK_DIRNAME = "calibrated_masks"

# Hybrid/maps (fase 04)
HYBRID_PHASE_OUT = OUT_PHASE_HYBRID / "hybrid_predictions"
MAPS_PHASE_OUT   = OUT_PHASE_HYBRID / "maps"
VIZ_PHASE_OUT    = OUT_PHASE_HYBRID / "visualizations"
HYBRID_MASKS_ROOT = SENTINEL2_DIR / "hybrid_masks"
HYBRID_SIMPLE_MASKS_OUT = HYBRID_MASKS_ROOT / "simple"
HYBRID_ROBUST_MASKS_OUT = HYBRID_MASKS_ROOT / "robust"
HYBRID_SENSITIVE_MASKS_OUT = HYBRID_MASKS_ROOT / "sensitive"

# ---------------------------------------------------------------------------
# Parametros de datos / modelos
# ---------------------------------------------------------------------------

MARIDA_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
BAND_NAMES = MARIDA_BANDS
MARIDA_NM_COLUMNS = ["nm440", "nm490", "nm560", "nm665", "nm705", "nm740", "nm783", "nm842", "nm865", "nm1600", "nm2200"]
DEBRIS_CLASS = 1
PATCH_SIZE = 256
RANDOM_STATE = 42

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

RF_MODE_NAMES = ["full", "no_texture", "indices_only", "bands_only"]
RF_MODE_DIRS = {mode: RF_OUT / mode for mode in RF_MODE_NAMES}


OUTPUT_DIRS = [
    OUTPUTS_DIR,
    REPORTS_DIR,
    # Dataset
    OUT_PHASE_DATASET,
    DATASET_FILTERS_PREVIOUS_OUT,
    DATASET_DOWNLOAD_OUT,
    GROUPED_SPLITS_PHASE_OUT,
    # Baselines
    UNET_PHASE_OUT,
    RF_PHASE_OUT,
    *RF_MODE_DIRS.values(),
    RESNET_PHASE_OUT,
    INDICES_NO_WATER_OUT,
    INDICES_WATER_OUT,
    SAM_PHASE_OUT,
    SAM_PROB_DIR,
    SAM_CALIBRATED_MASKS_OUT,
    EXTERNAL_B09_ZERO_OUT,
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
    # Evaluation
    OUT_PHASE_EVALUATION,
    EVAL_THRESHOLDS_OUT,
    EVAL_CALIBRATED_OUT,
    EVAL_UNIFIED_OUT,
    EVAL_PATCH_LEVEL_OUT,
    EVAL_PIXELWISE_OUT,
    EVAL_TOLERANT_OUT,
    EVAL_ERROR_ANALYSIS_OUT,
    # Hybrid/maps
    HYBRID_PHASE_OUT,
    MAPS_PHASE_OUT,
    VIZ_PHASE_OUT,
    HYBRID_MASKS_ROOT,
    HYBRID_SIMPLE_MASKS_OUT,
    HYBRID_ROBUST_MASKS_OUT,
    HYBRID_SENSITIVE_MASKS_OUT,
]


def ensure_output_dirs() -> None:
    for path in OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
