from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from config import CSV_XGB, XGB_OUT


def cleanup_previous_outputs():
    for filename in ["cv_results.csv", "feature_importance.csv", "xgboost_model.json"]:
        path = XGB_OUT / filename
        if path.exists():
            path.unlink()


def main():
    XGB_OUT.mkdir(parents=True, exist_ok=True)
    try:
        import xgboost as xgb
    except ModuleNotFoundError:
        cleanup_previous_outputs()
        metrics = {
            "status": "skipped_missing_dependency",
            "reason": "xgboost no esta instalado en el entorno actual",
        }
        with open(XGB_OUT / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print("XGBoost no esta instalado. Se omite este paso y se continua sin resultados XGBoost.")
        return

    df = pd.read_csv(CSV_XGB)
    print(f"Dataset: {len(df)} patches  SI={df.label.sum()}  NO={(df.label == 0).sum()}")
    if df["label"].nunique() < 2:
        raise ValueError("Se necesitan muestras SI y NO para entrenar XGBoost.")

    feature_cols = [col for col in df.columns if col not in ("patch", "label")]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

    X = df[feature_cols].values.astype("float32")
    y = df["label"].values.astype(int)
    names = df["patch"].values

    if len(df) < 30:
        cv = LeaveOneOut()
        cv_name = "LOO"
        print("\nDataset pequeno -> Leave-One-Out CV")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_name = "StratifiedKFold5"
        print("\nDataset suficiente -> StratifiedKFold(5)")

    scale_pos = float((y == 0).sum()) / max(int((y == 1).sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    cv_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(pred.tolist())
        y_prob_all.extend(prob.tolist())

        for i, patch_index in enumerate(test_idx):
            cv_rows.append(
                {
                    "fold": fold,
                    "patch": names[patch_index],
                    "true": int(y_te[i]),
                    "pred": int(pred[i]),
                    "prob": round(float(prob[i]), 4),
                }
            )

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    print("\n=== RESULTADOS CV ===")
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
        print(f"AUC-ROC:   {auc:.3f}")
    except Exception:
        auc = None
        print("AUC-ROC:   no calculable")

    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)

    print(f"F1:        {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"Confusion matrix:\n{cm}")

    model.fit(X, y)
    model_path = XGB_OUT / "xgboost_model.json"
    model.save_model(str(model_path))

    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(XGB_OUT / "feature_importance.csv", index=False)

    pd.DataFrame(cv_rows).to_csv(XGB_OUT / "cv_results.csv", index=False)

    metrics = {
        "n_patches": len(df),
        "n_si": int(df.label.sum()),
        "n_no": int((df.label == 0).sum()),
        "n_features": int(X.shape[1]),
        "cv_method": cv_name,
        "auc_roc": round(float(auc), 4) if auc is not None else None,
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
    }
    with open(XGB_OUT / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nModelo final guardado: {model_path}")
    print(f"Resultados guardados en: {XGB_OUT}")


if __name__ == "__main__":
    main()
