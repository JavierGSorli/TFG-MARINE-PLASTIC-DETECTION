from __future__ import annotations

import argparse
import sys

import pandas as pd
import rasterio

from config import DEBRIS_CLASS, PATCHES_DIR, UNET_DIR, UNET_OUT, UNET_SCRIPT, ensure_output_dirs
from pipeline_utils import ensure_file, infer_label_from_name, iter_patch_files, run_command, tail_lines


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

    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    rows = []
    for patch_path in patches:
        out_mask = UNET_OUT / f"{patch_path.stem}_mask.tif"
        label = infer_label_from_name(patch_path.name)

        if out_mask.exists() and not args.overwrite:
            plastic_px, total_px = count_debris_pixels(out_mask)
            print(f"[SKIP] {patch_path.name} -> {out_mask.name}")
            rows.append(
                {
                    "patch": patch_path.name,
                    "label": label,
                    "status": "existing",
                    "plastic_px": plastic_px,
                    "plastic_pct": round(100 * plastic_px / total_px, 4),
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
            "--device",
            args.device,
        ]
        if args.auto_scale:
            cmd.append("--auto_scale")
        result = run_command(cmd, cwd=UNET_DIR)
        if result.returncode != 0 or not out_mask.exists():
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
            }
        )

    summary_path = UNET_OUT / "summary_unet.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
