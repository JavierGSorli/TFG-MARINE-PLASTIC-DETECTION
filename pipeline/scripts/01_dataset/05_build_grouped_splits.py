from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---


import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.common.config import DATASET_METADATA_GROUPED_PATH, GROUPED_SPLITS_PHASE_OUT
from src.evaluation.model_validation_utils import make_grouped_cv_splits


METADATA_PATH = DATASET_METADATA_GROUPED_PATH
OUT_DIR = GROUPED_SPLITS_PHASE_OUT

STRATEGY_DIR_NAMES = {"groupkfold": "groupkfold"}


def split_train_val_vs_test_final_grouped(
    df: pd.DataFrame,
    test_fraction: float,
    random_state: int,
    n_splits: int,
    search_attempts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, object, object]], list[str]]:
    groups = df["group_id"].astype(str).values
    y = df["label_binary"].astype(int).values
    unique_groups = pd.unique(groups)

    if len(unique_groups) < 3:
        raise ValueError("Se necesitan al menos tres grupos para separar train_val y test_final.")

    best_candidate = None
    target_test_size = max(1, round(len(df) * test_fraction))

    for offset in range(search_attempts):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=random_state + offset,
        )
        train_val_idx, test_idx = next(splitter.split(df, y, groups))
        train_val = df.iloc[train_val_idx]
        test_final = df.iloc[test_idx]
        if train_val["label_binary"].nunique() < 2 or test_final["label_binary"].nunique() < 2:
            continue
        try:
            inner_splits, inner_warnings = make_grouped_cv_splits(
                train_val,
                group_col="group_id",
                n_splits=n_splits,
                strategy="groupkfold",
            )
        except ValueError:
            continue

        test_si = int((test_final["label_binary"] == 1).sum())
        test_ratio = test_si / max(len(test_final), 1)
        test_dev = abs(test_ratio - 0.5)
        worst_val_dev = 0.0
        avg_val_dev = 0.0
        for _, _, val_idx in inner_splits:
            val = train_val.iloc[val_idx]
            val_si = int((val["label_binary"] == 1).sum())
            val_ratio = val_si / max(len(val), 1)
            val_dev = abs(val_ratio - 0.5)
            worst_val_dev = max(worst_val_dev, val_dev)
            avg_val_dev += val_dev
        avg_val_dev /= max(len(inner_splits), 1)
        score = (
            round(worst_val_dev, 6),
            round(test_dev, 6),
            round(avg_val_dev, 6),
            abs(len(test_final) - target_test_size),
            abs(len(train_val) - (len(df) - target_test_size)),
        )
        candidate = (score, train_val, test_final, inner_splits, inner_warnings)
        if best_candidate is None or score < best_candidate[0]:
            best_candidate = candidate

    if best_candidate is not None:
        _, train_val, test_final, inner_splits, inner_warnings = best_candidate
        return train_val, test_final, inner_splits, inner_warnings

    outer_splits = GroupKFold(
        n_splits=min(max(2, round(1.0 / test_fraction)), len(unique_groups))
    ).split(df, y, groups)
    best_pair = None
    best_score = None
    for train_val_idx, test_idx in outer_splits:
        train_val = df.iloc[train_val_idx]
        test_final = df.iloc[test_idx]
        if train_val["label_binary"].nunique() < 2 or test_final["label_binary"].nunique() < 2:
            continue
        try:
            inner_splits, inner_warnings = make_grouped_cv_splits(
                train_val,
                group_col="group_id",
                n_splits=n_splits,
                strategy="groupkfold",
            )
        except ValueError:
            continue
        test_si = int((test_final["label_binary"] == 1).sum())
        test_ratio = test_si / max(len(test_final), 1)
        test_dev = abs(test_ratio - 0.5)
        worst_val_dev = 0.0
        avg_val_dev = 0.0
        for _, _, val_idx in inner_splits:
            val = train_val.iloc[val_idx]
            val_si = int((val["label_binary"] == 1).sum())
            val_ratio = val_si / max(len(val), 1)
            val_dev = abs(val_ratio - 0.5)
            worst_val_dev = max(worst_val_dev, val_dev)
            avg_val_dev += val_dev
        avg_val_dev /= max(len(inner_splits), 1)
        score = (
            round(worst_val_dev, 6),
            round(test_dev, 6),
            round(avg_val_dev, 6),
            abs(len(test_final) - target_test_size),
            abs(len(train_val) - (len(df) - target_test_size)),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (train_val, test_final, inner_splits, inner_warnings)

    if best_pair is None:
        raise ValueError("No se pudo construir un test_final agrupado con ambas clases.")
    return best_pair


def split_train_val_into_calibration_and_selection_grouped(
    train_val_df: pd.DataFrame,
    calibration_fraction: float,
    random_state: int,
    n_splits: int,
    search_attempts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, object, object]], list[str]]:
    groups = train_val_df["group_id"].astype(str).values
    y = train_val_df["label_binary"].astype(int).values
    unique_groups = pd.unique(groups)

    if len(unique_groups) < n_splits + 1:
        raise ValueError("No hay suficientes grupos en train_val para separar calibration_dev y selection_dev.")

    best_candidate = None
    target_cal_size = max(1, round(len(train_val_df) * calibration_fraction))

    for offset in range(search_attempts):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=(1.0 - calibration_fraction),
            random_state=random_state + 10000 + offset,
        )
        cal_idx, sel_idx = next(splitter.split(train_val_df, y, groups))
        calibration_dev = train_val_df.iloc[cal_idx]
        selection_dev = train_val_df.iloc[sel_idx]
        if calibration_dev["label_binary"].nunique() < 2 or selection_dev["label_binary"].nunique() < 2:
            continue
        try:
            inner_splits, inner_warnings = make_grouped_cv_splits(
                selection_dev,
                group_col="group_id",
                n_splits=n_splits,
                strategy="groupkfold",
            )
        except ValueError:
            continue

        cal_si = int((calibration_dev["label_binary"] == 1).sum())
        cal_ratio = cal_si / max(len(calibration_dev), 1)
        cal_dev = abs(cal_ratio - 0.5)
        sel_si = int((selection_dev["label_binary"] == 1).sum())
        sel_ratio = sel_si / max(len(selection_dev), 1)
        sel_dev = abs(sel_ratio - 0.5)
        worst_val_dev = 0.0
        avg_val_dev = 0.0
        for _, _, val_idx in inner_splits:
            val = selection_dev.iloc[val_idx]
            val_si = int((val["label_binary"] == 1).sum())
            val_ratio = val_si / max(len(val), 1)
            val_dev = abs(val_ratio - 0.5)
            worst_val_dev = max(worst_val_dev, val_dev)
            avg_val_dev += val_dev
        avg_val_dev /= max(len(inner_splits), 1)
        score = (
            round(worst_val_dev, 6),
            round(cal_dev, 6),
            round(sel_dev, 6),
            round(avg_val_dev, 6),
            abs(len(calibration_dev) - target_cal_size),
        )
        candidate = (score, calibration_dev, selection_dev, inner_splits, inner_warnings)
        if best_candidate is None or score < best_candidate[0]:
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError("No se pudo construir calibration_dev/selection_dev agrupados con ambas clases.")
    _, calibration_dev, selection_dev, inner_splits, inner_warnings = best_candidate
    return calibration_dev, selection_dev, inner_splits, inner_warnings


def build_strategy_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    strategy: str,
    n_splits: int,
    final_test_fraction: float,
    calibration_fraction: float,
    random_state: int,
    search_attempts: int,
) -> pd.DataFrame:
    train_val_df, test_final_df, splits, warnings = split_train_val_vs_test_final_grouped(
        df=df,
        test_fraction=final_test_fraction,
        random_state=random_state,
        n_splits=n_splits,
        search_attempts=search_attempts,
    )
    calibration_dev_df, selection_dev_df, selection_splits, selection_warnings = split_train_val_into_calibration_and_selection_grouped(
        train_val_df=train_val_df,
        calibration_fraction=calibration_fraction,
        random_state=random_state,
        n_splits=n_splits,
        search_attempts=search_attempts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    summary_rows = []
    all_groups = set(df["group_id"].astype(str))
    train_val_groups = set(train_val_df["group_id"].astype(str))
    test_final_groups = set(test_final_df["group_id"].astype(str))
    calibration_groups = set(calibration_dev_df["group_id"].astype(str))
    selection_groups = set(selection_dev_df["group_id"].astype(str))

    for _, row in test_final_df.iterrows():
        fold_rows.append(
            {
                "patch": row["patch"],
                "date": row["date"],
                "group_id": row["group_id"],
                "fold": pd.NA,
                "split": "test_final",
                "label": row["label"],
                "subset": "test_final",
                "strategy": strategy,
            }
        )

    for _, row in calibration_dev_df.iterrows():
        fold_rows.append(
            {
                "patch": row["patch"],
                "date": row["date"],
                "group_id": row["group_id"],
                "fold": pd.NA,
                "split": "calibration_dev",
                "label": row["label"],
                "subset": "calibration_dev",
                "strategy": strategy,
            }
        )

    for _, row in selection_dev_df.iterrows():
        fold_rows.append(
            {
                "patch": row["patch"],
                "date": row["date"],
                "group_id": row["group_id"],
                "fold": pd.NA,
                "split": "selection_dev",
                "label": row["label"],
                "subset": "selection_dev",
                "strategy": strategy,
            }
        )

    for fold, train_idx, test_idx in selection_splits:
        train = selection_dev_df.iloc[train_idx]
        val = selection_dev_df.iloc[test_idx]
        val_dates = sorted(val["group_id"].astype(str).unique().tolist())

        train_groups = set(train["group_id"].astype(str))
        val_groups = set(val["group_id"].astype(str))
        leaking = (
            (train_groups & val_groups)
            | (train_groups & test_final_groups)
            | (val_groups & test_final_groups)
            | (train_groups & calibration_groups)
            | (val_groups & calibration_groups)
        )
        if leaking:
            print(f"  WARN {strategy} fold {fold}: grupos repetidos entre splits: {leaking}")

        for split_name, split_df in [("train", train), ("val", val)]:
            for _, row in split_df.iterrows():
                fold_rows.append(
                    {
                        "patch": row["patch"],
                        "date": row["date"],
                        "group_id": row["group_id"],
                        "fold": fold,
                        "split": split_name,
                        "label": row["label"],
                        "subset": "selection_dev",
                        "strategy": strategy,
                    }
                )

        summary_rows.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_val": len(val),
                "train_si": int((train["label_binary"] == 1).sum()),
                "train_no": int((train["label_binary"] == 0).sum()),
                "val_si": int((val["label_binary"] == 1).sum()),
                "val_no": int((val["label_binary"] == 0).sum()),
                "val_dates": ";".join(val_dates),
                "strategy": strategy,
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_dir / "folds.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    all_warnings = list(warnings) + list(selection_warnings)
    warning_lines = "\n".join(f"- {w}" for w in all_warnings) if all_warnings else "- Sin warnings."

    balance_warnings = []
    test_final_si = int((test_final_df["label_binary"] == 1).sum())
    test_final_no = int((test_final_df["label_binary"] == 0).sum())
    test_final_ratio = test_final_si / max(len(test_final_df), 1)
    if test_final_ratio < 0.2 or test_final_ratio > 0.8:
        balance_warnings.append(
            f"- test_final: SI={test_final_si}, NO={test_final_no} (ratio SI={test_final_ratio:.2f})"
        )
    for _, row in summary.iterrows():
        si_ratio = row["val_si"] / max(row["n_val"], 1)
        if si_ratio < 0.2 or si_ratio > 0.8:
            balance_warnings.append(
            f"- Fold {row['fold']}: val_si={row['val_si']}, "
                f"val_no={row['val_no']} (ratio SI={si_ratio:.2f})"
            )

    calibration_si = int((calibration_dev_df["label_binary"] == 1).sum())
    calibration_no = int((calibration_dev_df["label_binary"] == 0).sum())
    selection_si = int((selection_dev_df["label_binary"] == 1).sum())
    selection_no = int((selection_dev_df["label_binary"] == 0).sum())

    summary_text = "# Grouped split summary (Final test + calibration + selection folds)\n\n"
    summary_text += "## Final test\n"
    summary_text += f"- strategy: {strategy}\n"
    summary_text += f"- n_groups: {len(all_groups)}\n"
    summary_text += f"- train_val_groups: {len(train_val_groups)}\n"
    summary_text += f"- test_final_groups: {len(test_final_groups)}\n"
    summary_text += f"- train_val_patches: {len(train_val_df)}\n"
    summary_text += f"- test_final_patches: {len(test_final_df)}\n"
    summary_text += f"- test_final_si: {test_final_si}\n"
    summary_text += f"- test_final_no: {test_final_no}\n"
    summary_text += (
        f"- test_final_dates: {';'.join(sorted(test_final_df['group_id'].astype(str).unique().tolist()))}\n\n"
    )
    summary_text += "## Development split inside train_val\n"
    summary_text += f"- calibration_dev_groups: {len(calibration_groups)}\n"
    summary_text += f"- selection_dev_groups: {len(selection_groups)}\n"
    summary_text += f"- calibration_dev_patches: {len(calibration_dev_df)}\n"
    summary_text += f"- calibration_dev_si: {calibration_si}\n"
    summary_text += f"- calibration_dev_no: {calibration_no}\n"
    summary_text += f"- selection_dev_patches: {len(selection_dev_df)}\n"
    summary_text += f"- selection_dev_si: {selection_si}\n"
    summary_text += f"- selection_dev_no: {selection_no}\n\n"
    summary_text += "## Inner folds (train/val sobre selection_dev)\n"
    summary_text += summary.to_string(index=False)
    summary_text += "\n\n## Balance warnings\n"
    summary_text += ("\n".join(balance_warnings) if balance_warnings else "- Sin folds desbalanceados.") + "\n"
    summary_text += "\n## CV Warnings\n"
    summary_text += warning_lines + "\n"
    (out_dir / "split_summary.md").write_text(summary_text, encoding="utf-8")

    print(f"Splits agrupados ({strategy}) guardados en: {out_dir}")
    print(
        f"test_final={len(test_final_df)} patches | "
        f"train_val={len(train_val_df)} patches | "
        f"calibration_dev={len(calibration_dev_df)} | "
        f"selection_dev={len(selection_dev_df)} | inner_folds={len(summary)}"
    )
    if balance_warnings:
        print("WARN: folds desbalanceados:")
        for w in balance_warnings:
            print(f"  {w}")

    return folds_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--strategy", choices=["auto", "groupkfold"], default="groupkfold")
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--final_test_fraction", type=float, default=0.2)
    parser.add_argument("--calibration_fraction", type=float, default=0.45)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--search_attempts", type=int, default=2000)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    strategies = ["groupkfold"] if args.strategy == "auto" else [args.strategy]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for strategy in strategies:
        strategy_dir = OUT_DIR / STRATEGY_DIR_NAMES[strategy]
        build_strategy_outputs(
            df=df,
            out_dir=strategy_dir,
            strategy=strategy,
            n_splits=args.n_splits,
            final_test_fraction=args.final_test_fraction,
            calibration_fraction=args.calibration_fraction,
            random_state=args.random_state,
            search_attempts=args.search_attempts,
        )


if __name__ == "__main__":
    main()
