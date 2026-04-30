from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from config import CSV_MASTER, DEBRIS_CLASS, EVAL_OUT, INDICES_OUT, PATCHES_DIR, RF_OUT, UNET_OUT, XGB_OUT


def best_threshold_f1(scores, y_true):
    if np.allclose(scores.min(), scores.max()):
        return float(scores.min())
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_t = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        current_f1 = f1_score(y_true, scores >= threshold, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_t = float(threshold)
    return best_t


def expected_gt_px_from_patch_name(filename):
    stem = filename.replace(".tif", "")
    parts = stem.split("_")
    if len(parts) >= 3 and parts[1] == "SI":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return 0


def negative_difficulty_from_patch_name(filename):
    stem = filename.replace(".tif", "")
    parts = stem.split("_")
    if len(parts) >= 5 and parts[1] == "NO":
        return parts[-1].upper()
    return "NA"


def read_bool_mask(path, mode):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        mask = src.read(1)
    if mode == "debris_class":
        return mask == DEBRIS_CLASS
    return mask > 0


def compute_segmentation_stats(gt_mask, pred_mask):
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    tp_px = int(np.logical_and(gt_mask, pred_mask).sum())
    fp_px = int(np.logical_and(~gt_mask, pred_mask).sum())
    fn_px = int(np.logical_and(gt_mask, ~pred_mask).sum())
    gt_px = int(gt_mask.sum())
    pred_px = int(pred_mask.sum())
    union_px = int(np.logical_or(gt_mask, pred_mask).sum())

    precision = tp_px / pred_px if pred_px > 0 else 0.0
    recall = tp_px / gt_px if gt_px > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp_px / union_px if union_px > 0 else 0.0

    return {
        "gt_px": gt_px,
        "pred_px": pred_px,
        "overlay_px": tp_px,
        "union_px": union_px,
        "tp_px": tp_px,
        "fp_px": fp_px,
        "fn_px": fn_px,
        "overlay_gt_pct": 100.0 * tp_px / gt_px if gt_px > 0 else 0.0,
        "overlay_pred_pct": 100.0 * tp_px / pred_px if pred_px > 0 else 0.0,
        "overlay_iou_pct": 100.0 * iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "count_error": pred_px - gt_px,
        "abs_count_error": abs(pred_px - gt_px),
    }


def build_segmentation_evaluation(df):
    positive_df = df[df["label"] == "SI"].copy()
    if positive_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    gt_check_rows = []
    for _, row in positive_df.iterrows():
        stem = row["patch"].replace(".tif", "")
        gt_mask = read_bool_mask(PATCHES_DIR / f"{stem}_mask.tif", "binary")
        gt_px = int(gt_mask.sum()) if gt_mask is not None else None
        expected_gt_px = expected_gt_px_from_patch_name(row["patch"])
        gt_check_rows.append(
            {
                "patch": row["patch"],
                "expected_gt_px": expected_gt_px,
                "mask_gt_px": gt_px,
                "match_name_vs_mask": expected_gt_px == gt_px,
            }
        )

    segmentation_methods = {
        "UNet (MARIDA)": (UNET_OUT, "_mask.tif", "debris_class"),
        "RF (MARIDA)": (RF_OUT, "_mask.tif", "debris_class"),
        "FDI": (INDICES_OUT, "_fdi_mask.tif", "binary"),
        "NDVI": (INDICES_OUT, "_ndvi_mask.tif", "binary"),
        "FDI+NDVI": (INDICES_OUT, "_fdi_ndvi_mask.tif", "binary"),
    }

    patch_rows = []
    for _, row in positive_df.iterrows():
        stem = row["patch"].replace(".tif", "")
        gt_mask = read_bool_mask(PATCHES_DIR / f"{stem}_mask.tif", "binary")
        if gt_mask is None:
            continue

        expected_gt_px = expected_gt_px_from_patch_name(row["patch"])
        for method_name, (mask_dir, suffix, mode) in segmentation_methods.items():
            pred_mask = read_bool_mask(mask_dir / f"{stem}{suffix}", mode)
            if pred_mask is None:
                continue

            stats = compute_segmentation_stats(gt_mask, pred_mask)
            patch_rows.append(
                {
                    "Metodo": method_name,
                    "patch": row["patch"],
                    "expected_gt_px": expected_gt_px,
                    **stats,
                }
            )

    df_gt_check = pd.DataFrame(gt_check_rows)
    df_seg_patch = pd.DataFrame(patch_rows)
    if df_seg_patch.empty:
        return df_gt_check, df_seg_patch, pd.DataFrame()

    summary_rows = []
    for method_name, group in df_seg_patch.groupby("Metodo"):
        tp_total = int(group["tp_px"].sum())
        fp_total = int(group["fp_px"].sum())
        fn_total = int(group["fn_px"].sum())
        gt_total = int(group["gt_px"].sum())
        pred_total = int(group["pred_px"].sum())
        union_total = int(group["union_px"].sum())

        micro_precision = tp_total / pred_total if pred_total > 0 else 0.0
        micro_recall = tp_total / gt_total if gt_total > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )
        micro_iou = tp_total / union_total if union_total > 0 else 0.0

        summary_rows.append(
            {
                "Metodo": method_name,
                "n_patches_SI": int(len(group)),
                "gt_px_total": gt_total,
                "pred_px_total": pred_total,
                "overlay_px_total": tp_total,
                "union_px_total": union_total,
                "tp_px_total": tp_total,
                "fp_px_total": fp_total,
                "fn_px_total": fn_total,
                "overlay_gt_pct": round(float(100.0 * tp_total / gt_total), 3) if gt_total > 0 else 0.0,
                "overlay_pred_pct": round(float(100.0 * tp_total / pred_total), 3) if pred_total > 0 else 0.0,
                "overlay_iou_pct": round(float(100.0 * micro_iou), 3),
                "micro_precision": round(float(micro_precision), 4),
                "micro_recall": round(float(micro_recall), 4),
                "micro_f1": round(float(micro_f1), 4),
                "micro_iou": round(float(micro_iou), 4),
                "mean_abs_count_error": round(float(group["abs_count_error"].mean()), 3),
                "mean_count_error": round(float(group["count_error"].mean()), 3),
            }
        )

    return df_gt_check, df_seg_patch, pd.DataFrame(summary_rows)


def build_false_positive_difficulty_evaluation(df, method_outputs):
    negative_df = df[df["label"] == "NO"].copy()
    if negative_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    difficulties = negative_df["patch"].map(negative_difficulty_from_patch_name).values
    rows = []
    for method_name, output in method_outputs.items():
        scores = output["scores"]
        preds = output["preds"]
        threshold = output["threshold"]
        for idx in negative_df.index:
            rows.append(
                {
                    "Metodo": method_name,
                    "patch": df.loc[idx, "patch"],
                    "dificultad": difficulties[list(negative_df.index).index(idx)],
                    "score": float(scores[idx]),
                    "threshold": float(threshold),
                    "pred": int(preds[idx]),
                    "is_false_positive": int(preds[idx] == 1),
                }
            )

    df_fp_patch = pd.DataFrame(rows)
    if df_fp_patch.empty:
        return df_fp_patch, pd.DataFrame()

    summary_rows = []
    for (method_name, dificultad), group in df_fp_patch.groupby(["Metodo", "dificultad"]):
        n_negatives = int(len(group))
        fp_count = int(group["is_false_positive"].sum())
        summary_rows.append(
            {
                "Metodo": method_name,
                "dificultad": dificultad,
                "n_negatives": n_negatives,
                "false_positives": fp_count,
                "true_negatives": n_negatives - fp_count,
                "fp_rate_pct": round(float(100.0 * fp_count / n_negatives), 3) if n_negatives > 0 else 0.0,
                "mean_score": round(float(group["score"].mean()), 6),
                "median_score": round(float(group["score"].median()), 6),
                "max_score": round(float(group["score"].max()), 6),
            }
        )

    return df_fp_patch, pd.DataFrame(summary_rows)


def build_negative_segmentation_noise_evaluation(df):
    negative_df = df[df["label"] == "NO"].copy()
    if negative_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    method_cols = {
        "UNet (MARIDA)": "unet_px",
        "RF (MARIDA)": "rf_px",
        "FDI": "fdi_px",
        "NDVI": "ndvi_px",
        "FDI+NDVI": "fdi_ndvi_px",
    }
    rows = []
    for _, row in negative_df.iterrows():
        dificultad = negative_difficulty_from_patch_name(row["patch"])
        for method_name, column in method_cols.items():
            if column not in row or pd.isna(row[column]):
                continue
            pred_px = int(row[column])
            rows.append(
                {
                    "Metodo": method_name,
                    "patch": row["patch"],
                    "dificultad": dificultad,
                    "pred_px": pred_px,
                    "has_predicted_pixels": int(pred_px > 0),
                }
            )

    df_noise_patch = pd.DataFrame(rows)
    if df_noise_patch.empty:
        return df_noise_patch, pd.DataFrame()

    summary_rows = []
    for (method_name, dificultad), group in df_noise_patch.groupby(["Metodo", "dificultad"]):
        n_negatives = int(len(group))
        patches_with_pixels = int(group["has_predicted_pixels"].sum())
        summary_rows.append(
            {
                "Metodo": method_name,
                "dificultad": dificultad,
                "n_negatives": n_negatives,
                "patches_with_predicted_pixels": patches_with_pixels,
                "patch_rate_with_pixels_pct": round(float(100.0 * patches_with_pixels / n_negatives), 3)
                if n_negatives > 0
                else 0.0,
                "total_pred_px": int(group["pred_px"].sum()),
                "mean_pred_px": round(float(group["pred_px"].mean()), 3),
                "median_pred_px": round(float(group["pred_px"].median()), 3),
                "max_pred_px": int(group["pred_px"].max()),
            }
        )

    return df_noise_patch, pd.DataFrame(summary_rows)


def main():
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_MASTER)
    print(f"Patches: {len(df)}  SI={(df.label == 'SI').sum()}  NO={(df.label == 'NO').sum()}\n")
    y_true = (df["label"] == "SI").astype(int).values

    method_defs = {}
    candidate_cols = {
        "UNet (MARIDA)": "unet_pct",
        "RF (MARIDA)": "rf_pct",
        "FDI": "fdi_pct",
        "NDVI": "ndvi_pct",
        "FDI+NDVI": "fdi_ndvi_pct",
    }
    for method_name, column in candidate_cols.items():
        if column in df.columns and df[column].notna().any():
            scores = df[column].fillna(0).values.astype(float)
            method_defs[method_name] = {
                "scores": scores,
                "preds": None,
                "threshold": None,
                "optimize_threshold": True,
            }

    if "resnet_active" in df.columns and df["resnet_active"].notna().any():
        resnet_preds = df["resnet_active"].fillna(0).astype(int).values
        if "resnet_prob" in df.columns and df["resnet_prob"].notna().any():
            resnet_scores = df["resnet_prob"].fillna(0).values.astype(float)
        else:
            resnet_scores = resnet_preds.astype(float)

        method_defs["ResNet (MARIDA)"] = {
            "scores": resnet_scores,
            "preds": resnet_preds,
            "threshold": 0.5,
            "optimize_threshold": False,
        }

    xgb_cv = XGB_OUT / "cv_results.csv"
    xgb_metrics = XGB_OUT / "metrics.json"
    xgb_enabled = True
    if xgb_metrics.exists():
        with open(xgb_metrics, encoding="utf-8") as handle:
            metrics_data = json.load(handle)
        if metrics_data.get("status") == "skipped_missing_dependency":
            xgb_enabled = False

    if xgb_enabled and xgb_cv.exists():
        df_cv = pd.read_csv(xgb_cv)
        if not df_cv.empty:
            df_cv_last = df_cv.groupby("patch").last().reset_index()
            method_defs["XGBoost"] = {
                "scores": (
                    df.merge(df_cv_last[["patch", "prob"]], on="patch", how="left")["prob"]
                    .fillna(0)
                    .values.astype(float)
                ),
                "preds": None,
                "threshold": None,
                "optimize_threshold": True,
            }

    if not method_defs:
        raise RuntimeError("No se encontraron metodos con scores para evaluar.")

    results = []
    method_outputs = {}
    for method_name, method_info in method_defs.items():
        scores = method_info["scores"]
        if method_info["optimize_threshold"]:
            threshold = best_threshold_f1(scores, y_true)
            preds = (scores >= threshold).astype(int)
        else:
            threshold = float(method_info["threshold"])
            preds = np.asarray(method_info["preds"], dtype=int)
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = None

        method_outputs[method_name] = {
            "scores": scores,
            "preds": preds,
            "threshold": threshold,
        }

        results.append(
            {
                "Metodo": method_name,
                "AUC_ROC": round(float(auc), 3) if auc is not None else None,
                "F1": round(float(f1_score(y_true, preds, zero_division=0)), 3),
                "Precision": round(float(precision_score(y_true, preds, zero_division=0)), 3),
                "Recall": round(float(recall_score(y_true, preds, zero_division=0)), 3),
                "Umbral": round(float(threshold), 5),
                "TP": int(((preds == 1) & (y_true == 1)).sum()),
                "FP": int(((preds == 1) & (y_true == 0)).sum()),
                "TN": int(((preds == 0) & (y_true == 0)).sum()),
                "FN": int(((preds == 0) & (y_true == 1)).sum()),
            }
        )

    df_res = pd.DataFrame(results)
    print("=== TABLA COMPARATIVA ===")
    print(df_res.to_string(index=False))
    df_res.to_csv(EVAL_OUT / "tabla_comparativa.csv", index=False)

    df_gt_check, df_seg_patch, df_seg_summary = build_segmentation_evaluation(df)
    if not df_gt_check.empty:
        df_gt_check.to_csv(EVAL_OUT / "gt_mask_check.csv", index=False)
        print("\n=== CHEQUEO GT MASK ===")
        print(df_gt_check.to_string(index=False))

    if not df_seg_patch.empty:
        df_seg_patch.to_csv(EVAL_OUT / "segmentacion_por_patch.csv", index=False)
        df_seg_summary.to_csv(EVAL_OUT / "segmentacion_resumen.csv", index=False)
        overlay_cols = [
            "Metodo",
            "patch",
            "expected_gt_px",
            "gt_px",
            "pred_px",
            "overlay_px",
            "union_px",
            "overlay_gt_pct",
            "overlay_pred_pct",
            "overlay_iou_pct",
            "pixel_f1",
        ]
        df_seg_patch[overlay_cols].sort_values(
            ["Metodo", "overlay_iou_pct", "patch"],
            ascending=[True, False, True],
        ).to_csv(EVAL_OUT / "overlay_por_patch.csv", index=False)
        print("\n=== RESUMEN SEGMENTACION (solo patches SI) ===")
        print(df_seg_summary.to_string(index=False))

    df_fp_patch, df_fp_summary = build_false_positive_difficulty_evaluation(df, method_outputs)
    if not df_fp_patch.empty:
        df_fp_patch.to_csv(EVAL_OUT / "falsos_positivos_por_patch_dificultad.csv", index=False)
        df_fp_summary.to_csv(EVAL_OUT / "falsos_positivos_por_dificultad.csv", index=False)
        print("\n=== FALSOS POSITIVOS POR DIFICULTAD (solo NO) ===")
        print(df_fp_summary.to_string(index=False))

    df_noise_patch, df_noise_summary = build_negative_segmentation_noise_evaluation(df)
    if not df_noise_patch.empty:
        df_noise_patch.to_csv(EVAL_OUT / "ruido_segmentacion_negativos_por_patch.csv", index=False)
        df_noise_summary.to_csv(EVAL_OUT / "ruido_segmentacion_negativos_por_dificultad.csv", index=False)
        print("\n=== RUIDO DE SEGMENTACION EN NEGATIVOS POR DIFICULTAD ===")
        print(df_noise_summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_defs)))
    for (method_name, method_info), color in zip(method_defs.items(), colors):
        scores = method_info["scores"]
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, label=f"{method_name} (AUC={auc:.2f})", color=color, lw=2)
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_OUT / "roc_curves.png", dpi=150)
    plt.close()

    x = np.arange(len(df_res))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, df_res["Precision"], width, label="Precision", color="#3b82f6")
    ax.bar(x, df_res["F1"], width, label="F1", color="#16a34a")
    ax.bar(x + width, df_res["Recall"], width, label="Recall", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["Metodo"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparativa de metricas")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_OUT / "metrics_bar.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, len(method_defs), figsize=(4 * len(method_defs), 4), sharey=False)
    if len(method_defs) == 1:
        axes = [axes]
    for ax, (method_name, method_info) in zip(axes, method_defs.items()):
        scores = method_info["scores"]
        si_scores = scores[y_true == 1]
        no_scores = scores[y_true == 0]
        ax.boxplot(
            [si_scores, no_scores],
            labels=["SI", "NO"],
            patch_artist=True,
            boxprops=dict(facecolor="#e5e7eb"),
        )
        ax.set_title(method_name, fontsize=9)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_OUT / "scores_si_no.png", dpi=150, bbox_inches="tight")
    plt.close()

    if not df_seg_summary.empty:
        x = np.arange(len(df_seg_summary))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, df_seg_summary["micro_precision"], width, label="Pixel Precision", color="#2563eb")
        ax.bar(x, df_seg_summary["micro_f1"], width, label="Pixel F1", color="#16a34a")
        ax.bar(x + width, df_seg_summary["micro_iou"], width, label="Pixel IoU", color="#ea580c")
        ax.set_xticks(x)
        ax.set_xticklabels(df_seg_summary["Metodo"], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Segmentacion sobre GT mask (solo positivos)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(EVAL_OUT / "segmentacion_metrics_bar.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, df_seg_summary["overlay_gt_pct"], width, label="GT cubierto (%)", color="#16a34a")
        ax.bar(x, df_seg_summary["overlay_pred_pct"], width, label="Pred sobre GT (%)", color="#2563eb")
        ax.bar(x + width, df_seg_summary["overlay_iou_pct"], width, label="Overlay IoU (%)", color="#ea580c")
        ax.set_xticks(x)
        ax.set_xticklabels(df_seg_summary["Metodo"], rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("%")
        ax.set_title("Overlay espacial sobre GT mask")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(EVAL_OUT / "overlay_metrics_bar.png", dpi=150)
        plt.close()

    if not df_fp_summary.empty:
        pivot = df_fp_summary.pivot(index="Metodo", columns="dificultad", values="false_positives").fillna(0)
        pivot = pivot.reindex(columns=[col for col in ["CLARO", "DUDOSO", "DIFICIL"] if col in pivot.columns])
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(pivot))
        colors = {"CLARO": "#16a34a", "DUDOSO": "#f59e0b", "DIFICIL": "#dc2626"}
        x = np.arange(len(pivot))
        for dificultad in pivot.columns:
            values = pivot[dificultad].values
            ax.bar(x, values, bottom=bottom, label=dificultad, color=colors.get(dificultad))
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20, ha="right")
        ax.set_ylabel("False positives")
        ax.set_title("Falsos positivos por dificultad del negativo")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(EVAL_OUT / "falsos_positivos_por_dificultad.png", dpi=150)
        plt.close()

    if not df_noise_summary.empty:
        pivot = df_noise_summary.pivot(index="Metodo", columns="dificultad", values="total_pred_px").fillna(0)
        pivot = pivot.reindex(columns=[col for col in ["CLARO", "DUDOSO", "DIFICIL"] if col in pivot.columns])
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(pivot))
        colors = {"CLARO": "#16a34a", "DUDOSO": "#f59e0b", "DIFICIL": "#dc2626"}
        x = np.arange(len(pivot))
        for dificultad in pivot.columns:
            values = pivot[dificultad].values
            ax.bar(x, values, bottom=bottom, label=dificultad, color=colors.get(dificultad))
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20, ha="right")
        ax.set_ylabel("Predicted pixels on NO patches")
        ax.set_title("Ruido de segmentacion por dificultad del negativo")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(EVAL_OUT / "ruido_segmentacion_negativos_por_dificultad.png", dpi=150)
        plt.close()

    print(f"\nResultados guardados en: {EVAL_OUT}")


if __name__ == "__main__":
    main()
