from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import pandas as pd

from src.common.config import EVAL_UNIFIED_OUT, PREDICTIONS_MASTER_PATH
from src.evaluation.raw_prediction_table import build_raw_prediction_table


def _coalesce_suffix_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    suffix_x = [col for col in df.columns if col.endswith("_x")]
    for col_x in suffix_x:
        base = col_x[:-2]
        col_y = f"{base}_y"
        if col_y not in df.columns:
            continue
        df[base] = df[col_y].combine_first(df[col_x])
        df = df.drop(columns=[col_x, col_y])
    return df


def main() -> None:
    EVAL_UNIFIED_OUT.mkdir(parents=True, exist_ok=True)

    master = build_raw_prediction_table()
    calibrated_path = _Path(EVAL_UNIFIED_OUT.parent) / "calibrated_outputs" / "calibrated_patch_predictions.csv"
    if calibrated_path.exists():
        calibrated = pd.read_csv(calibrated_path)
        calibrated_by_patch = calibrated.drop_duplicates(subset=["patch"], keep="last").set_index("patch")
        master = master.merge(calibrated[["patch"]], on="patch", how="left")
        master = _coalesce_suffix_pairs(master)
        for col in calibrated.columns:
            if col == "patch":
                continue
            mapped = master["patch"].map(calibrated_by_patch[col])
            if col in master.columns:
                master[col] = mapped.combine_first(master[col])
            else:
                master[col] = mapped

    master.to_csv(PREDICTIONS_MASTER_PATH, index=False)

    missing = master.isnull().sum()
    lines = [
        "# predictions_master — Missing Values Report",
        "",
        f"Total patches: {len(master)}",
        f"SI: {int((master['label'].astype(str).str.upper() == 'SI').sum())}",
        f"NO: {int((master['label'].astype(str).str.upper() == 'NO').sum())}",
        "",
        "## NULL count por columna",
        "",
        "| Columna | NULLs | % |",
        "|---------|------:|--:|",
    ]
    for col, n_null in missing.items():
        pct = 100.0 * n_null / len(master) if len(master) else 0.0
        lines.append(f"| {col} | {int(n_null)} | {pct:.1f}% |")
    (EVAL_UNIFIED_OUT / "predictions_master_missing_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"CSV maestro guardado: {PREDICTIONS_MASTER_PATH}")
    print(master.head().to_string(index=False))


if __name__ == "__main__":
    main()
