# 06_train_xgboost.py
# Entrena XGBoost sobre el dataset tabular de patches.
# Con pocos datos (<<100 patches) usa leave-one-out CV.
# Salida: modelo .json + resultados CV

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
import xgboost as xgb

BASE    = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\results\auto")
IN_CSV  = BASE / "xgboost_dataset.csv"
OUT_DIR = BASE / "xgboost_model"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(IN_CSV)
print(f"Dataset: {len(df)} patches  SI={df.label.sum()}  NO={(df.label==0).sum()}")

# Features: todo excepto patch y label
feature_cols = [c for c in df.columns if c not in ("patch", "label")]

# Imputar missing con mediana de la columna
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

X = df[feature_cols].values.astype("float32")
y = df["label"].values.astype(int)
names = df["patch"].values

print(f"Features: {X.shape[1]}")

# ── Elegir CV según tamaño del dataset ────────────────────────
if len(df) < 30:
    print("\nDataset pequeño → Leave-One-Out CV")
    cv = LeaveOneOut()
else:
    print("\nDataset suficiente → StratifiedKFold(5)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── XGBoost con escala de pesos para clase desbalanceada ──────
scale_pos = int((y == 0).sum()) / max(int((y == 1).sum()), 1)

model = xgb.XGBClassifier(
    n_estimators      = 100,
    max_depth         = 3,
    learning_rate     = 0.1,
    scale_pos_weight  = scale_pos,
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = 42,
    verbosity         = 0,
)

# ── CV loop ───────────────────────────────────────────────────
y_true_all  = []
y_pred_all  = []
y_prob_all  = []
cv_results  = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)

    y_true_all.extend(y_te.tolist())
    y_pred_all.extend(pred.tolist())
    y_prob_all.extend(prob.tolist())

    for i, (true, p, pr) in enumerate(zip(y_te, pred, prob)):
        cv_results.append({
            "fold":  fold,
            "patch": names[test_idx[i]],
            "true":  int(true),
            "pred":  int(p),
            "prob":  round(float(pr), 4),
        })

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_prob_all = np.array(y_prob_all)

# ── Métricas globales ─────────────────────────────────────────
print("\n=== RESULTADOS CV ===")
try:
    auc = roc_auc_score(y_true_all, y_prob_all)
    print(f"AUC-ROC:   {auc:.3f}")
except Exception:
    auc = None
    print("AUC-ROC:   no calculable (una sola clase en algún fold)")

f1  = f1_score(y_true_all, y_pred_all, zero_division=0)
pre = precision_score(y_true_all, y_pred_all, zero_division=0)
rec = recall_score(y_true_all, y_pred_all, zero_division=0)
cm  = confusion_matrix(y_true_all, y_pred_all)

print(f"F1:        {f1:.3f}")
print(f"Precision: {pre:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"Confusion matrix:\n{cm}")

# ── Entrenar modelo final sobre todos los datos ───────────────
model.fit(X, y)
model_path = OUT_DIR / "xgboost_model.json"
model.save_model(str(model_path))
print(f"\n✓ Modelo final guardado: {model_path}")

# ── Feature importance ────────────────────────────────────────
importance = pd.DataFrame({
    "feature":    feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 15 features más importantes:")
print(importance.head(15).to_string(index=False))

importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)

# ── Guardar resultados CV ─────────────────────────────────────
df_cv = pd.DataFrame(cv_results)
df_cv.to_csv(OUT_DIR / "cv_results.csv", index=False)

metrics = {
    "n_patches":  len(df),
    "n_si":       int(df.label.sum()),
    "n_no":       int((df.label==0).sum()),
    "n_features": X.shape[1],
    "cv_method":  "LOO" if len(df) < 30 else "StratifiedKFold5",
    "auc_roc":    round(float(auc), 4) if auc else None,
    "f1":         round(float(f1), 4),
    "precision":  round(float(pre), 4),
    "recall":     round(float(rec), 4),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Resultados guardados en: {OUT_DIR}")
