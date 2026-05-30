from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import argparse
import subprocess
import sys

import pandas as pd
import rasterio

from src.common.config import (
    DEBRIS_CLASS,
    PATCHES_DIR,
    RF_MODE_DIRS,
    RF_MODE_NAMES,
    RF_MODEL_PATHS,
    RF_PHASE_OUT,
    RF_PREDICT_SCRIPT,
    ensure_output_dirs,
)
from src.common.pipeline_utils import ensure_file, infer_label_from_name, iter_patch_files


def run_predictor(
    patch_path: _Path,
    out_mask: _Path,
    out_prob: _Path,
    mode: str,
    auto_scale: bool,
    window_size: int,
    max_value: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(RF_PREDICT_SCRIPT),
        "--patch_tif",
        str(patch_path),
        "--model_path",
        str(RF_MODEL_PATHS[mode]),
        "--out_mask",
        str(out_mask),
        "--out_prob",
        str(out_prob),
        "--window_size",
        str(window_size),
        "--max_value",
        str(max_value),
        "--feature_mode",
        mode,
    ]
    if auto_scale:
        cmd.append("--auto_scale")
    return subprocess.run(cmd, text=True, capture_output=True)


def summarize_mask(mask_path: _Path) -> tuple[int, int]:
    with rasterio.open(mask_path) as src:
        pred_mask = src.read(1)
    return int((pred_mask == DEBRIS_CLASS).sum()), int(pred_mask.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--window_size", type=int, default=13)
    parser.add_argument("--max_value", type=int, default=16)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument(
        "--mode",
        choices=RF_MODE_NAMES,
        default="full",
        help="Modo RF: full, no_texture, indices_only o bands_only.",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Genera las cuatro variantes RF en una sola ejecución.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    ensure_file(RF_PREDICT_SCRIPT, "Script predict_mask_rf.py")

    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    modes = list(RF_MODE_NAMES) if args.all_modes else [args.mode]

    for mode in modes:
        ensure_file(RF_MODEL_PATHS[mode], f"Modelo Random Forest ({mode})")
        mode_dir = RF_MODE_DIRS[mode]
        mode_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        print(f"\n=== RF mode: {mode} ===")
        for patch_path in patches:
            out_mask = mode_dir / f"{patch_path.stem}_mask.tif"
            out_prob = mode_dir / f"{patch_path.stem}_marine_debris_prob.tif"
            label = infer_label_from_name(patch_path.name)

            if out_mask.exists() and out_prob.exists() and not args.overwrite:
                plastic_px, total_px = summarize_mask(out_mask)
                status = "existing"
                error_message = ""
                print(f"[SKIP:{mode}] {patch_path.name} -> {out_mask.name}")
            else:
                proc = run_predictor(
                    patch_path=patch_path,
                    out_mask=out_mask,
                    out_prob=out_prob,
                    mode=mode,
                    auto_scale=args.auto_scale,
                    window_size=args.window_size,
                    max_value=args.max_value,
                )
                if proc.returncode != 0 or not out_mask.exists() or not out_prob.exists():
                    status = "error"
                    error_message = (proc.stderr or proc.stdout or "predict_mask_rf failed").strip()
                    plastic_px = None
                    total_px = None
                    print(f"[ERROR:{mode}] {patch_path.name}: {error_message}")
                else:
                    plastic_px, total_px = summarize_mask(out_mask)
                    status = "ok"
                    error_message = ""
                    print(f"[OK:{mode}] {patch_path.name} -> debris={plastic_px} px")

            rows.append(
                {
                    "patch": patch_path.name,
                    "label": label,
                    "mode": mode,
                    "status": status,
                    "plastic_px": plastic_px,
                    "plastic_pct": round(100 * plastic_px / total_px, 4) if plastic_px is not None and total_px else None,
                    "error_message": error_message,
                }
            )

        summary_path = RF_PHASE_OUT / f"summary_rf_{mode}.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
