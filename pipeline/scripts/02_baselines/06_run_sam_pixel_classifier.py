from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.common.config import PATCHES_DIR, SAM_CALIBRATED_MASKS_OUT, SAM_PHASE_OUT, SAM_PROB_DIR
from src.common.pipeline_utils import iter_patch_files

SAM_CLASES_DIR  = SAM_PHASE_OUT / "clases"
SAM_BINARIO_DIR = SAM_PHASE_OUT / "binario"

SAM_SIGNATURES_PATH = SAM_PHASE_OUT / "marida_spectral_signatures_by_class.csv"

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SIGNATURE_COLUMNS = [f"{band}_mean" for band in BAND_NAMES]

# Indices originales MARIDA (1-based) — coinciden con el QML de QGIS
MARIDA_CLASS_INDEX = {
    "Marine Debris":            1,
    "Dense Sargassum":          2,
    "Sparse Sargassum":         3,
    "Natural Organic Material": 4,
    "Ship":                     5,
    "Clouds":                   6,
    "Marine Water":             7,
    "Sediment-Laden Water":     8,
    "Foam":                     9,
    "Turbid Water":            10,
    "Shallow Water":           11,
    "Waves":                   12,
    "Cloud Shadows":           13,
    "Wakes":                   14,
    "Mixed Water":             15,
}

DEBRIS_CLASSES = {"marine debris", "dense sargassum"}  # → indices 1 y 2


def safe_class_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def spectral_angle_matrix(pixels: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """pixels: (N, 11), refs: (C, 11) -> angles: (N, C) en radianes"""
    pixel_norms = np.linalg.norm(pixels, axis=1, keepdims=True)
    ref_norms   = np.linalg.norm(refs,   axis=1, keepdims=True).T
    denom = pixel_norms * ref_norms
    dot   = pixels @ refs.T
    cosang = np.zeros_like(dot, dtype=np.float64)
    valid  = denom > 1e-12
    cosang[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles = np.full(dot.shape, np.nan, dtype=np.float32)
    angles[valid] = np.arccos(cosang[valid]).astype(np.float32)
    return angles


def classify_patch(
    patch_path: Path,
    refs: np.ndarray,
    class_names: list,
    marida_indices: np.ndarray,
    debris_class_idx: int,
) -> tuple:
    with rasterio.open(patch_path) as src:
        data    = src.read().astype("float32")
        profile = src.profile.copy()

    n_bands, h, w = data.shape
    if n_bands != 11:
        raise ValueError(f"{patch_path.name}: esperaba 11 bandas, hay {n_bands}")

    if np.nanmax(data) > 10:
        data = data / 10000.0

    pixels     = data.reshape(n_bands, -1).T          # (N, 11)
    valid_mask = np.isfinite(pixels).all(axis=1) & (np.linalg.norm(pixels, axis=1) > 1e-12)

    # class_map almacena el índice MARIDA (1-15), -1 = nodata
    class_map = np.full(h * w, -1, dtype=np.int16)
    debris_score_map = np.full(h * w, np.nan, dtype=np.float32)

    # second_angle_map: 1 si debris queda en rank 0 o 1 (top-2 ángulos menores)
    second_angle_map = np.zeros(h * w, dtype=np.uint8)
    # third_angle_map:  1 si debris queda en rank 0, 1 o 2 (top-3 ángulos menores)
    third_angle_map  = np.zeros(h * w, dtype=np.uint8)

    if valid_mask.any():
        angles_valid = spectral_angle_matrix(pixels[valid_mask], refs)  # (N_valid, C)

        # Winner (argmin del ángulo) → clase predicha
        row_indices = np.nanargmin(angles_valid, axis=1)
        class_map[valid_mask] = marida_indices[row_indices]

        # Similaridad normalizada [0,1] de la clase Marine Debris para threshold posterior.
        debris_angles = angles_valid[:, debris_class_idx]
        debris_similarity = 1.0 - np.clip(debris_angles / np.pi, 0.0, 1.0)
        debris_score_map[valid_mask] = debris_similarity.astype(np.float32)

        # Ranking por píxel: argsort de menor a mayor ángulo
        # ranks[i, j] = posición (0-based) de la clase j para el píxel i
        ranks = np.argsort(angles_valid, axis=1)  # (N_valid, C)

        # Posición de debris_class_idx en el ranking de cada píxel
        debris_rank = np.where(ranks == debris_class_idx)[1]  # (N_valid,)

        second_angle_map[valid_mask] = (debris_rank <= 1).astype(np.uint8)
        third_angle_map[valid_mask]  = (debris_rank <= 2).astype(np.uint8)

    # nodata: píxeles inválidos → 255
    second_angle_map[~valid_mask] = 255
    third_angle_map[~valid_mask]  = 255

    class_raster  = class_map.reshape(h, w)
    # Máscara binaria estricta: positiva solo si Marine Debris es la mejor clase.
    binary_raster = np.zeros((h, w), dtype=np.uint8)
    binary_raster[class_raster == 1] = 1
    binary_raster[class_raster == -1] = 255  # nodata
    debris_score_raster = debris_score_map.reshape(h, w)

    second_angle_raster = second_angle_map.reshape(h, w)
    third_angle_raster  = third_angle_map.reshape(h, w)

    return class_raster, binary_raster, debris_score_raster, second_angle_raster, third_angle_raster, profile


def main() -> None:
    SAM_CLASES_DIR.mkdir(parents=True, exist_ok=True)
    SAM_BINARIO_DIR.mkdir(parents=True, exist_ok=True)
    SAM_PROB_DIR.mkdir(parents=True, exist_ok=True)
    SAM_CALIBRATED_MASKS_OUT.mkdir(parents=True, exist_ok=True)

    if not SAM_SIGNATURES_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron firmas en {SAM_SIGNATURES_PATH}.\n"
            "Ejecuta primero: python pipeline/scripts/02_baselines/05_build_sam_signatures.py"
        )

    signatures = pd.read_csv(SAM_SIGNATURES_PATH)
    if "class_safe" not in signatures.columns:
        signatures["class_safe"] = signatures["class"].map(safe_class_name)

    missing = [c for c in SIGNATURE_COLUMNS if c not in signatures.columns]
    if missing:
        raise ValueError(f"Faltan columnas de firma: {missing}")

    class_names = signatures["class"].astype(str).tolist()
    refs        = signatures[SIGNATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Índice MARIDA para cada fila de signatures (fallback: 0 si clase desconocida)
    marida_indices = np.array(
        [MARIDA_CLASS_INDEX.get(n, 0) for n in class_names], dtype=np.int16
    )

    # Posición de "Marine Debris" en la lista de referencias (índice en refs/class_names)
    debris_class_idx = next(
        (i for i, n in enumerate(class_names) if n.strip() == "Marine Debris"),
        0,
    )

    debris_names = [n for n in class_names if n.lower().strip() in DEBRIS_CLASSES]
    print(f"[SAM2] Clases debris conocidas: {debris_names}")
    print("[SAM2] Máscara binaria estricta: 1 solo si 'Marine Debris' es la clase ganadora.")
    print(f"[SAM2] Total clases MARIDA: {len(class_names)}")
    print(f"[SAM2] Índice de 'Marine Debris' en refs: {debris_class_idx}")

    # Leyenda con índices MARIDA (compatibles con el QML de QGIS)
    pd.DataFrame({
        "marida_index": marida_indices,
        "class_name":   class_names,
        "is_debris":    [n.lower().strip() in DEBRIS_CLASSES for n in class_names],
    }).to_csv(SAM_PHASE_OUT / "sam_class_legend.csv", index=False)

    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    print(f"[SAM2] Procesando {len(patches)} patches ...\n")

    for patch_path in patches:
        stem = patch_path.stem
        print(f"  {patch_path.name}")

        class_raster, binary_raster, debris_score_raster, second_angle_raster, third_angle_raster, profile = classify_patch(
            patch_path, refs, class_names, marida_indices, debris_class_idx
        )

        base = {k: profile[k] for k in ("driver", "height", "width", "count", "transform", "crs")}
        base.update(count=1, compress="lzw")

        with rasterio.open(SAM_CLASES_DIR / f"{stem}_sam_class.tif",
                           "w", dtype="int16", nodata=-1, **base) as dst:
            dst.write(class_raster[np.newaxis])

        with rasterio.open(SAM_BINARIO_DIR / f"{stem}_sam_debris_mask.tif",
                           "w", dtype="uint8", nodata=255, **base) as dst:
            dst.write(binary_raster[np.newaxis])

        with rasterio.open(SAM_PROB_DIR / f"{stem}_sam_marine_debris_score.tif",
                           "w", dtype="float32", nodata=np.nan, **base) as dst:
            dst.write(debris_score_raster[np.newaxis])

        with rasterio.open(SAM_CLASES_DIR / f"{stem}_sam_second_angle.tif",
                           "w", dtype="uint8", nodata=255, **base) as dst:
            dst.write(second_angle_raster[np.newaxis])

        with rasterio.open(SAM_CLASES_DIR / f"{stem}_sam_third_angle.tif",
                           "w", dtype="uint8", nodata=255, **base) as dst:
            dst.write(third_angle_raster[np.newaxis])

    print(f"\n[SAM2] Máscaras de clase  -> {SAM_CLASES_DIR}")
    print(f"[SAM2] Máscaras binarias  -> {SAM_BINARIO_DIR}")
    print(f"[SAM2] Scores continuos   -> {SAM_PROB_DIR}")
    print(f"[SAM2] Leyenda de clases  -> {SAM_PHASE_OUT / 'sam_class_legend.csv'}")


if __name__ == "__main__":
    main()
