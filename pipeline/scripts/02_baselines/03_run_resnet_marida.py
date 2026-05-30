from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---


import argparse
import json
import sys

import pandas as pd

from src.common.config import RESNET_PHASE_OUT, PATCHES_DIR, RESNET_DIR, RESNET_MODEL_PATH, RESNET_OUT, RESNET_SCRIPT, ensure_output_dirs
from src.common.pipeline_utils import ensure_file, infer_label_from_name, iter_patch_files, run_command, tail_lines


def read_resnet_output(json_path):
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    prob = data.get("probabilities", {}).get("Marine Debris")
    active = int("Marine Debris" in data.get("active_labels", []))
    return prob, active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    ensure_output_dirs()
    ensure_file(RESNET_SCRIPT, "Script de inferencia ResNet")
    ensure_file(RESNET_MODEL_PATH, "Modelo ResNet")

    patches = iter_patch_files(PATCHES_DIR)
    if not patches:
        raise FileNotFoundError(f"No se encontraron patches en {PATCHES_DIR}")

    rows = []
    for patch_path in patches:
        out_json = RESNET_OUT / f"{patch_path.stem}.json"
        label = infer_label_from_name(patch_path.name)

        if out_json.exists() and not args.overwrite:
            prob, active = read_resnet_output(out_json)
            print(f"[SKIP] {patch_path.name} -> {out_json.name}")
            rows.append(
                {
                    "patch": patch_path.name,
                    "label": label,
                    "status": "existing",
                    "resnet_prob": prob,
                    "resnet_active": active,
                }
            )
            continue

        cmd = [
            sys.executable,
            str(RESNET_SCRIPT),
            "--patch_tif",
            str(patch_path),
            "--out_json",
            str(out_json),
            "--model_path",
            str(RESNET_MODEL_PATH),
            "--device",
            args.device,
        ]
        result = run_command(cmd, cwd=RESNET_DIR)
        if result.returncode != 0 or not out_json.exists():
            print(f"[ERROR] {patch_path.name}")
            err = tail_lines(result.stderr or result.stdout)
            if err:
                print(err)
            rows.append(
                {
                    "patch": patch_path.name,
                    "label": label,
                    "status": "error",
                    "resnet_prob": None,
                    "resnet_active": None,
                }
            )
            continue

        prob, active = read_resnet_output(out_json)
        prob_text = f"{prob:.4f}" if prob is not None else "None"
        print(f"[OK] {patch_path.name} -> prob={prob_text} active={active}")
        rows.append(
            {
                "patch": patch_path.name,
                "label": label,
                "status": "ok",
                "resnet_prob": prob,
                "resnet_active": active,
            }
        )

    summary_path = RESNET_PHASE_OUT / "summary_resnet.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
