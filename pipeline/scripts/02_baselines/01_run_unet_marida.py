from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---


import argparse
import sys

import pandas as pd
import rasterio

from src.common.config import DEBRIS_CLASS, UNET_PHASE_OUT, PATCHES_DIR, UNET_DIR, UNET_MODEL_PATH, UNET_OUT, UNET_SCRIPT, ensure_output_dirs
from src.common.pipeline_utils import ensure_file, infer_label_from_name, iter_patch_files, run_command, tail_lines


def count_debris_pixels(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return int((mask == DEBRIS_CLASS).sum()), int(mask.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--auto_scale", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    ensure_file(UNET_SCRIPT, "Script de inferencia UNet")
    ensure_file(UNET_MODEL_PATH, "Pesos UNet MARIDA")

    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    rows = []
    for patch_path in patches:
        out_mask = UNET_OUT / f"{patch_path.stem}_mask.tif"
        out_prob = UNET_OUT / f"{patch_path.stem}_marine_debris_prob.tif"
        label = infer_label_from_name(patch_path.name)

        if out_mask.exists() and out_prob.exists() and not args.overwrite:
            plastic_px, total_px = count_debris_pixels(out_mask)
            print(f"[SKIP] {patch_path.name} -> {out_mask.name}")
            rows.append(
                {
                    "patch": patch_path.name,
                    "label": label,
                    "status": "existing",
                    "plastic_px": plastic_px,
                    "plastic_pct": round(100 * plastic_px / total_px, 4),
                    "prob_raster": out_prob.name,
                }
            )
            continue

        cmd = [
            sys.executable,
            str(UNET_SCRIPT),
            "--patch_tif",
            str(patch_path),
            "--out_mask",
            str(out_mask),
            "--out_prob",
            str(out_prob),
            "--model_path",
            str(UNET_MODEL_PATH),
            "--device",
            args.device,
        ]
        if args.auto_scale:
            cmd.append("--auto_scale")
        result = run_command(cmd, cwd=UNET_DIR)
        if result.returncode != 0 or not out_mask.exists() or not out_prob.exists():
            print(f"[ERROR] {patch_path.name}")
            err = tail_lines(result.stderr or result.stdout)
            if err:
                print(err)
            rows.append(
                {
                    "patch": patch_path.name,
                    "label": label,
                    "status": "error",
                    "plastic_px": None,
                    "plastic_pct": None,
                    "prob_raster": None,
                }
            )
            continue

        plastic_px, total_px = count_debris_pixels(out_mask)
        plastic_pct = round(100 * plastic_px / total_px, 4)
        print(f"[OK] {patch_path.name} -> debris={plastic_px} px ({plastic_pct}%)")
        rows.append(
            {
                "patch": patch_path.name,
                "label": label,
                "status": "ok",
                "plastic_px": plastic_px,
                "plastic_pct": plastic_pct,
                "prob_raster": out_prob.name,
            }
        )

    summary_path = UNET_PHASE_OUT / "summary_unet.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
