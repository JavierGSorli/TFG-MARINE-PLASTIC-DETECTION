from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


def extract_date_from_patch_name(patch_name: str) -> str:
    return Path(str(patch_name)).name[:8]


def make_grouped_cv_splits(
    metadata: pd.DataFrame,
    group_col: str = "group_id",
    n_splits: int = 5,
    strategy: str = "auto",
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], List[str]]:
    y = metadata["label_binary"].astype(int).values
    groups = metadata[group_col].astype(str).values
    unique_groups = np.unique(groups)
    warnings = []

    if len(unique_groups) < 2:
        raise ValueError("Se necesitan al menos dos grupos para validacion agrupada.")

    if strategy == "auto":
        splitter_name = "groupkfold"
    else:
        splitter_name = strategy.lower()

    if splitter_name != "groupkfold":
        raise ValueError(f"Estrategia no soportada: {strategy}")

    raw_splits = GroupKFold(n_splits=min(n_splits, len(unique_groups))).split(metadata, y, groups)

    valid_splits = []
    for fold, (train_idx, test_idx) in enumerate(raw_splits):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2:
            warnings.append(f"Fold {fold} saltado: train sin ambas clases.")
            continue
        if len(np.unique(y_test)) < 2:
            warnings.append(f"Fold {fold} saltado: test sin ambas clases.")
            continue
        valid_splits.append((fold, train_idx, test_idx))

    if not valid_splits:
        raise ValueError("No queda ningun fold valido con ambas clases en train y test.")
    return valid_splits, warnings


def select_threshold_on_train(y_train: Iterable[int], prob_train: Iterable[float], metric: str = "f1") -> float:
    y_train = np.asarray(list(y_train)).astype(int)
    prob_train = np.asarray(list(prob_train)).astype(float)
    finite = np.isfinite(prob_train)
    if finite.sum() == 0:
        return 0.5

    candidates = np.unique(prob_train[finite])
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], candidates)))
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidates:
        pred = (prob_train >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_train, pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_train, pred)
        else:
            raise ValueError(f"Metrica no soportada para umbral: {metric}")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold


def evaluate_binary_predictions(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> dict:
    y_true = np.asarray(list(y_true)).astype(int)
    y_prob = np.asarray(list(y_prob)).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan

    return {
        "auc": auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
