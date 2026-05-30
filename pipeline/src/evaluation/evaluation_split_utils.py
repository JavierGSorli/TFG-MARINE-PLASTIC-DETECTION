from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import GROUPKFOLD_FOLDS_PATH


def load_grouped_eval_folds(folds_path: Path = GROUPKFOLD_FOLDS_PATH) -> pd.DataFrame:
    if not folds_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de folds agrupados: {folds_path}\n"
            "Ejecuta pipeline/scripts/01_dataset/05_build_grouped_splits.py primero."
        )
    folds = pd.read_csv(folds_path)
    required = {"patch", "split", "subset"}
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(f"{folds_path} no contiene columnas requeridas: {sorted(missing)}")
    return folds


def build_patch_subset_map(folds: pd.DataFrame | None = None) -> pd.DataFrame:
    folds = load_grouped_eval_folds() if folds is None else folds.copy()
    if "fold" in folds.columns:
        folds["_base_row"] = folds["fold"].isna()
        folds = folds.sort_values(["patch", "_base_row"], ascending=[True, False])
    subset_map = (
        folds[["patch", "subset", "split"]]
        .drop_duplicates()
        .groupby("patch", as_index=False)
        .agg(
            eval_subset=("subset", "first"),
            eval_split=("split", "first"),
        )
    )
    return subset_map


def patches_for_subset(subset: str, folds: pd.DataFrame | None = None) -> set[str]:
    subset_map = build_patch_subset_map(folds)
    return set(subset_map.loc[subset_map["eval_subset"] == subset, "patch"].astype(str))


def filter_dataframe_by_subset(
    df: pd.DataFrame,
    subset: str,
    patch_col: str = "patch",
    folds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    allowed = patches_for_subset(subset, folds)
    return df[df[patch_col].astype(str).isin(allowed)].copy()
