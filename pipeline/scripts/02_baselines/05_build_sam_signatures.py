from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import DATA_DIR, SAM_PHASE_OUT


MARIDA_DATASET_H5 = DATA_DIR / "marida" / "marine-debris.github.io" / "data" / "dataset.h5"
SAM_SIGNATURES_PATH = SAM_PHASE_OUT / "marida_spectral_signatures_by_class.csv"

MARIDA_NM_COLUMNS = [
    "nm440",
    "nm490",
    "nm560",
    "nm665",
    "nm705",
    "nm740",
    "nm783",
    "nm842",
    "nm865",
    "nm1600",
    "nm2200",
]

BAND_NAMES = [
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


def safe_class_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def decode_if_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def load_marida_table(h5_path: Path) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        table_path = f"{split}/table"
        try:
            df = pd.read_hdf(h5_path, key=table_path)
        except (KeyError, FileNotFoundError):
            continue
        df["split"] = split
        df["Class"] = df["Class"].map(decode_if_bytes)
        df["Confidence"] = df["Confidence"].map(decode_if_bytes)
        frames.append(df)

    if not frames:
        raise ValueError(f"No se encontraron tablas train/val/test en {h5_path}")
    return pd.concat(frames, ignore_index=True)


def build_marida_signatures(
    h5_path: Path,
    out_csv: Path,
    overwrite: bool = False,
    confidence_filter: str | None = None,
) -> pd.DataFrame:
    if out_csv.exists() and not overwrite:
        print(f"[SAM] Reutilizando firmas existentes: {out_csv}")
        return pd.read_csv(out_csv)

    if not h5_path.exists():
        raise FileNotFoundError(f"No existe dataset MARIDA H5: {h5_path}")

    print(f"[SAM] Construyendo firmas espectrales desde: {h5_path}")
    data = load_marida_table(h5_path)

    if confidence_filter:
        before = len(data)
        data = data[data["Confidence"].astype(str).str.lower() == confidence_filter.lower()].copy()
        print(f"[SAM] Filtro Confidence={confidence_filter}: {before} -> {len(data)} píxeles")

    rows = []
    for class_name, group in data.groupby("Class"):
        row = {
            "class": class_name,
            "class_safe": safe_class_name(class_name),
            "n_pixels": int(len(group)),
        }
        for nm_col, band in zip(MARIDA_NM_COLUMNS, BAND_NAMES):
            values = pd.to_numeric(group[nm_col], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            row[f"{band}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{band}_std"] = float(np.std(values)) if values.size else np.nan
            row[f"{band}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{band}_p05"] = float(np.percentile(values, 5)) if values.size else np.nan
            row[f"{band}_p95"] = float(np.percentile(values, 95)) if values.size else np.nan
        rows.append(row)

    signatures = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    signatures.to_csv(out_csv, index=False)

    print(f"[SAM] Firmas guardadas: {out_csv}")
    print(signatures[["class", "n_pixels"]].to_string(index=False))
    return signatures


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir firmas espectrales SAM por clase desde MARIDA.")
    parser.add_argument("--h5", type=Path, default=MARIDA_DATASET_H5)
    parser.add_argument("--out", type=Path, default=SAM_SIGNATURES_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confidence", type=str, default=None)
    args = parser.parse_args()

    build_marida_signatures(
        h5_path=args.h5,
        out_csv=args.out,
        overwrite=args.overwrite,
        confidence_filter=args.confidence,
    )


if __name__ == "__main__":
    main()
