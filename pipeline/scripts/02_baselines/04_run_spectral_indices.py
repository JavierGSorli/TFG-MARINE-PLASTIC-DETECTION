from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import argparse
import shutil
import subprocess
import sys

from src.common.config import (
    INDICES_NO_WATER_OUT,
    INDICES_PREDICT_SCRIPT,
    INDICES_WATER_OUT,
    INDICES_WATER_PREDICT_SCRIPT,
    PATCHES_DIR,
    ensure_output_dirs,
)
from src.common.pipeline_utils import ensure_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--water-mask",
        choices=["none", "simple"],
        default="none",
        help="Modo de máscara de agua: 'none' genera variantes sin mask y 'simple' genera variantes _mask.",
    )
    parser.add_argument("--ndwi_thr", type=float, default=0.0)
    parser.add_argument("--b11_max", type=float, default=0.03)
    parser.add_argument("--b08_max", type=float, default=0.15)
    parser.add_argument("--morph_open", type=int, default=0)
    parser.add_argument("--morph_close", type=int, default=0)
    parser.add_argument("--aggressive_output_erosion", type=int, default=6)
    parser.add_argument("--recovery_radius", type=int, default=15)
    parser.add_argument("--recovery_water_ratio", type=float, default=0.7)
    args = parser.parse_args()

    ensure_output_dirs()

    if args.water_mask == "none":
        ensure_file(INDICES_PREDICT_SCRIPT, "Script 03_predict_indices.py")
        indices_out = INDICES_NO_WATER_OUT
        cmd = [
            sys.executable,
            str(INDICES_PREDICT_SCRIPT),
            "--input",
            str(PATCHES_DIR),
            "--out_dir",
            str(indices_out),
        ]
    else:
        ensure_file(INDICES_WATER_PREDICT_SCRIPT, "Script 03_predict_indices2.py")
        indices_out = INDICES_WATER_OUT
        cmd = [
            sys.executable,
            str(INDICES_WATER_PREDICT_SCRIPT),
            "--input",
            str(PATCHES_DIR),
            "--out_dir",
            str(indices_out),
            "--ndwi_thr",
            str(args.ndwi_thr),
            "--b11_max",
            str(args.b11_max),
            "--b08_max",
            str(args.b08_max),
            "--morph_open",
            str(args.morph_open),
            "--morph_close",
            str(args.morph_close),
            "--aggressive_output_erosion",
            str(args.aggressive_output_erosion),
            "--recovery_radius",
            str(args.recovery_radius),
            "--recovery_water_ratio",
            str(args.recovery_water_ratio),
        ]

    indices_out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Predict indices failed").strip())

    water_summary = indices_out / "summary_indices2.csv"
    unified_summary = indices_out / "summary_indices.csv"
    if water_summary.exists():
        shutil.copyfile(water_summary, unified_summary)

    if proc.stdout.strip():
        print(proc.stdout.strip())
    print(f"Índices generados en: {indices_out}")


if __name__ == "__main__":
    main()
