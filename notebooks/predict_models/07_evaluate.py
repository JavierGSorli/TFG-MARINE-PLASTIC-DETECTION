# 07_evaluate.py
# Evaluación comparativa de todos los métodos.
# Lee el CSV maestro y calcula métricas para cada método.
# Salida: tabla comparativa + figuras

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score,
                             roc_curve, precision_recall_curve)

BASE      = Path(r"C:\CDIA_oficial\tfg\tfg-marine-plastic-detection\results\auto")
MASTER    = BASE / "predictions_master.csv"
XGB_CV    = BASE / "xgboost_model" / "cv_results.csv"
OUT_DIR   = BASE / "evaluation"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(MASTER)
print(f"Patches: {len(df)}  SI={( df.label=='SI').sum()}  NO={(df.label=='NO').sum()}\n")

y_true = (df["label"] == "SI").astype(int).values

# ── Definir métodos y sus scores ──────────────────────────────
# Para UNet, RF, FDI: score = % píxeles detectados (continuo)
# Para ResNet: probabilidad directa
# Para XGBoost: leer del CSV de CV

methods = {}

if df["unet_pct"].notna().all():
    methods["UNet (MARIDA)"]    = df["unet_pct"].fillna(0).values
if df["rf_pct"].notna().all():
    methods["RF (MARIDA)"]      = df["rf_pct"].fillna(0).values
if df["resnet_prob"].notna().all():
    methods["ResNet (MARIDA)"]  = df["resnet_prob"].fillna(0).values
if df["fdi_pct"].notna().all():
    methods["FDI (umbral)"]     = df["fdi_pct"].fillna(0).values
if df["fdi_ndvi_pct"].notna().all():
    methods["FDI+NDVI"]         = df["fdi_ndvi_pct"].fillna(0).values

# XGBoost: alinear con el CSV maestro
if XGB_CV.exists():
    df_cv = pd.read_csv(XGB_CV)
    # Tomar la predicción de cada patch (última aparición en LOO)
    df_cv_last = df_cv.groupby("patch").last().reset_index()
    xgb_scores = df.merge(
        df_cv_last[["patch","prob"]], on="patch", how="left"
    )["prob"].fillna(0).values
    methods["XGBoost"] = xgb_scores

# ── Calcular métricas para cada método ───────────────────────
def best_threshold_f1(scores, y_true):
    """Umbral que maximiza F1."""
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, scores >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

results = []
for name, scores in methods.items():
    scores = np.array(scores, dtype=float)
    t      = best_threshold_f1(scores, y_true)
    preds  = (scores >= t).astype(int)

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = None

    results.append({
        "Método":     name,
        "AUC-ROC":    round(auc, 3) if auc else "N/A",
        "F1":         round(f1_score(y_true, preds, zero_division=0), 3),
        "Precision":  round(precision_score(y_true, preds, zero_division=0), 3),
        "Recall":     round(recall_score(y_true, preds, zero_division=0), 3),
        "Umbral":     round(t, 5),
        "TP": int(((preds==1) & (y_true==1)).sum()),
        "FP": int(((preds==1) & (y_true==0)).sum()),
        "TN": int(((preds==0) & (y_true==0)).sum()),
        "FN": int(((preds==0) & (y_true==1)).sum()),
    })

df_res = pd.DataFrame(results)
print("=== TABLA COMPARATIVA ===")
print(df_res.to_string(index=False))
df_res.to_csv(OUT_DIR / "tabla_comparativa.csv", index=False)

# ── Figura 1: Curvas ROC ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

for (name, scores), color in zip(methods.items(), colors):
    scores = np.array(scores, dtype=float)
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=color, lw=2)
    except Exception:
        pass

ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curvas ROC — Comparativa de métodos")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curves.png", dpi=150)
plt.close()
print("\n✓ roc_curves.png guardado")

# ── Figura 2: Barras F1 / Precision / Recall ─────────────────
df_plot = df_res[df_res["F1"] != "N/A"].copy()
df_plot["F1"]        = df_plot["F1"].astype(float)
df_plot["Precision"] = df_plot["Precision"].astype(float)
df_plot["Recall"]    = df_plot["Recall"].astype(float)

x    = np.arange(len(df_plot))
w    = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w,   df_plot["Precision"], w, label="Precision", color="#3498db")
ax.bar(x,       df_plot["F1"],        w, label="F1",        color="#2ecc71")
ax.bar(x + w,   df_plot["Recall"],    w, label="Recall",    color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(df_plot["Método"], rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Comparativa de métodos — Precision / F1 / Recall")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "metrics_bar.png", dpi=150)
plt.close()
print("✓ metrics_bar.png guardado")

# ── Figura 3: Scores SI vs NO por método ─────────────────────
n_methods = len(methods)
fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 4),
                         sharey=False)
if n_methods == 1:
    axes = [axes]

for ax, (name, scores) in zip(axes, methods.items()):
    scores = np.array(scores, dtype=float)
    si_scores = scores[y_true == 1]
    no_scores = scores[y_true == 0]
    ax.boxplot([si_scores, no_scores],
               labels=["SI (plástico)", "NO (agua)"],
               patch_artist=True,
               boxprops=dict(facecolor="#ecf0f1"))
    ax.set_title(name, fontsize=9)
    ax.set_ylabel("Score / % píxeles")
    ax.grid(True, axis="y", alpha=0.3)

plt.suptitle("Distribución de scores — SI vs NO", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "scores_si_no.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ scores_si_no.png guardado")

print(f"\n✓ Todos los resultados en: {OUT_DIR}")
