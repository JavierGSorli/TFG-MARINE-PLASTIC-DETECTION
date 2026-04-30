from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import rasterio

from config import DEBRIS_CLASS, INDICES_OUT, PATCHES_DIR, RESNET_OUT, RF_OUT, UNET_OUT, XGB_OUT
from pipeline_utils import iter_patch_files


COLORS = {
    "GT": (0.10, 0.75, 0.25),
    "UNet": (0.90, 0.20, 0.15),
    "RF": (0.15, 0.40, 0.95),
    "FDI": (0.95, 0.55, 0.10),
    "NDVI": (0.65, 0.20, 0.85),
    "FDI+NDVI": (0.15, 0.75, 0.75),
    "XGBoost_POS": (0.15, 0.75, 0.30),
    "XGBoost_NEG": (0.92, 0.20, 0.18),
}


def make_rgb(data, vmax=0.10):
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    return np.clip(rgb / vmax, 0, 1)


def overlay_mask(rgb, mask, color, alpha=0.65):
    out = rgb.copy()
    active = mask.astype(bool)
    for channel, value in enumerate(color):
        out[:, :, channel] = np.where(
            active,
            alpha * value + (1.0 - alpha) * rgb[:, :, channel],
            rgb[:, :, channel],
        )
    return out


def read_patch_rgb(path):
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
    return make_rgb(data)


def read_mask(path, mode):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        mask = src.read(1)
    if mode == "debris_class":
        return mask == DEBRIS_CLASS
    return mask > 0


def load_resnet_info(stem):
    path = RESNET_OUT / f"{stem}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "prob": data.get("probabilities", {}).get("Marine Debris"),
        "active": int("Marine Debris" in data.get("active_labels", [])),
        "threshold": data.get("threshold"),
    }


def load_xgboost_df():
    cv_path = XGB_OUT / "cv_results.csv"
    if not cv_path.exists():
        return None
    df = pd.read_csv(cv_path)
    if df.empty:
        return None
    return df.groupby("patch").last().reset_index().set_index("patch")


def expected_gt_px_from_patch_name(filename):
    parts = filename.replace(".tif", "").split("_")
    if len(parts) >= 3 and parts[1] == "SI":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return 0


def load_master_df():
    master_csv = PATCHES_DIR.parent / "predictions_master.csv"
    if not master_csv.exists():
        return None
    return pd.read_csv(master_csv).set_index("patch")


def build_patch_list(limit, only_label):
    patches = iter_patch_files(PATCHES_DIR)
    if only_label != "ALL":
        patches = [path for path in patches if f"_{only_label}_" in path.name]
    if limit is not None:
        patches = patches[:limit]
    return patches


class PatchViewer:
    def __init__(self, patches, master_df, xgboost_df):
        if not patches:
            raise ValueError("No hay patches para visualizar.")
        self.patches = patches
        self.master_df = master_df
        self.xgboost_df = xgboost_df
        self.index = 0

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("11_visualizacion")

        gs = gridspec.GridSpec(
            2,
            4,
            figure=self.fig,
            width_ratios=[1, 1, 1, 1.7],
            height_ratios=[1, 1],
        )

        self.mask_axes = {
            "GT": self.fig.add_subplot(gs[0, 0]),
            "UNet": self.fig.add_subplot(gs[0, 1]),
            "RF": self.fig.add_subplot(gs[0, 2]),
            "FDI": self.fig.add_subplot(gs[1, 0]),
            "NDVI": self.fig.add_subplot(gs[1, 1]),
            "FDI+NDVI": self.fig.add_subplot(gs[1, 2]),
        }
        self.xgb_ax = self.fig.add_subplot(gs[0, 3])
        self.rgb_ax = self.fig.add_subplot(gs[1, 3])

        for ax in list(self.mask_axes.values()) + [self.xgb_ax, self.rgb_ax]:
            ax.set_xticks([])
            ax.set_yticks([])

        self.prev_ax = self.fig.add_axes([0.32, 0.01, 0.12, 0.05])
        self.next_ax = self.fig.add_axes([0.46, 0.01, 0.12, 0.05])
        self.prev_btn = Button(self.prev_ax, "Anterior")
        self.next_btn = Button(self.next_ax, "Siguiente")
        self.prev_btn.on_clicked(lambda _event: self.step(-1))
        self.next_btn.on_clicked(lambda _event: self.step(1))

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.render()

    def on_key_press(self, event):
        if event.key in {"right", "d", "n", "space"}:
            self.step(1)
        elif event.key in {"left", "a", "p", "backspace"}:
            self.step(-1)
        elif event.key in {"q", "escape"}:
            plt.close(self.fig)

    def step(self, delta):
        self.index = (self.index + delta) % len(self.patches)
        self.render()

    def render_mask_panel(self, ax, rgb, mask, title, color):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        if mask is None:
            ax.imshow(rgb)
            ax.text(
                0.5,
                0.5,
                "No disponible",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                transform=ax.transAxes,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 6},
            )
            ax.set_title(title)
            return

        pred_px = int(mask.sum())
        ax.imshow(overlay_mask(rgb, mask, color=color))
        ax.set_title(f"{title}\npx={pred_px}")

    def render_xgboost_panel(self, ax, rgb, patch_name):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

        if self.xgboost_df is None or patch_name not in self.xgboost_df.index:
            ax.imshow(rgb)
            ax.text(
                0.5,
                0.5,
                "XGBoost no disponible",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                transform=ax.transAxes,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 6},
            )
            ax.set_title("XGBoost")
            return

        row = self.xgboost_df.loc[patch_name]
        pred = int(row.get("pred", 0))
        prob = float(row.get("prob", 0.0))
        color = COLORS["XGBoost_POS"] if pred == 1 else COLORS["XGBoost_NEG"]

        full_patch_mask = np.ones(rgb.shape[:2], dtype=bool)
        ax.imshow(overlay_mask(rgb, full_patch_mask, color=color, alpha=0.38))
        ax.set_title(f"XGBoost (patch-level)\npred={pred} prob={prob:.4f}")
        ax.text(
            0.02,
            0.02,
            "No hay mascara por pixel\nsolo clasificacion del patch",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 6},
        )

    def render(self):
        patch_path = self.patches[self.index]
        stem = patch_path.stem
        rgb = read_patch_rgb(patch_path)

        gt_mask = read_mask(PATCHES_DIR / f"{stem}_mask.tif", "binary")
        unet_mask = read_mask(UNET_OUT / f"{stem}_mask.tif", "debris_class")
        rf_mask = read_mask(RF_OUT / f"{stem}_mask.tif", "debris_class")
        fdi_mask = read_mask(INDICES_OUT / f"{stem}_fdi_mask.tif", "binary")
        ndvi_mask = read_mask(INDICES_OUT / f"{stem}_ndvi_mask.tif", "binary")
        fdi_ndvi_mask = read_mask(INDICES_OUT / f"{stem}_fdi_ndvi_mask.tif", "binary")
        resnet_info = load_resnet_info(stem)

        self.render_mask_panel(self.mask_axes["GT"], rgb, gt_mask, "GT", COLORS["GT"])
        self.render_mask_panel(self.mask_axes["UNet"], rgb, unet_mask, "UNet", COLORS["UNet"])
        self.render_mask_panel(self.mask_axes["RF"], rgb, rf_mask, "RF", COLORS["RF"])
        self.render_mask_panel(self.mask_axes["FDI"], rgb, fdi_mask, "FDI", COLORS["FDI"])
        self.render_mask_panel(self.mask_axes["NDVI"], rgb, ndvi_mask, "NDVI", COLORS["NDVI"])
        self.render_mask_panel(
            self.mask_axes["FDI+NDVI"],
            rgb,
            fdi_ndvi_mask,
            "FDI+NDVI",
            COLORS["FDI+NDVI"],
        )
        self.render_xgboost_panel(self.xgb_ax, rgb, patch_path.name)

        self.rgb_ax.clear()
        self.rgb_ax.set_xticks([])
        self.rgb_ax.set_yticks([])
        self.rgb_ax.imshow(rgb)

        label = "SI" if "_SI_" in patch_path.name else "NO"
        expected_gt_px = expected_gt_px_from_patch_name(patch_path.name)
        gt_px = int(gt_mask.sum()) if gt_mask is not None else 0

        info_lines = [
            f"[{self.index + 1}/{len(self.patches)}] {patch_path.name}",
            f"Label={label}",
            f"GT esperado={expected_gt_px}",
            f"GT mask px={gt_px}",
        ]

        if self.master_df is not None and patch_path.name in self.master_df.index:
            row = self.master_df.loc[patch_path.name]
            info_lines.extend(
                [
                    f"UNet px={row.get('unet_px', 'NA')}",
                    f"RF px={row.get('rf_px', 'NA')}",
                    f"FDI px={row.get('fdi_px', 'NA')}",
                    f"NDVI px={row.get('ndvi_px', 'NA')}",
                    f"FDI+NDVI px={row.get('fdi_ndvi_px', 'NA')}",
                ]
            )

        if self.xgboost_df is not None and patch_path.name in self.xgboost_df.index:
            xgb_row = self.xgboost_df.loc[patch_path.name]
            info_lines.extend(
                [
                    f"XGB pred={int(xgb_row.get('pred', 0))}",
                    f"XGB prob={float(xgb_row.get('prob', 0.0)):.4f}",
                ]
            )

        if resnet_info is not None:
            prob = resnet_info["prob"]
            active = resnet_info["active"]
            threshold = resnet_info["threshold"]
            prob_text = f"{prob:.4f}" if prob is not None else "NA"
            info_lines.extend(
                [
                    f"ResNet prob={prob_text}",
                    f"ResNet active={active}",
                    f"ResNet thr={threshold}",
                ]
            )

        self.rgb_ax.set_title("RGB (B04/B03/B02)")
        self.rgb_ax.text(
            0.02,
            0.02,
            "\n".join(info_lines),
            transform=self.rgb_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.62, "pad": 8},
        )

        self.fig.suptitle(
            "11_visualizacion  |  teclas: izquierda/derecha, a/d, n/p, q para salir",
            fontsize=14,
            y=0.98,
        )
        self.fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Numero maximo de patches a mostrar")
    parser.add_argument("--only-label", choices=["ALL", "SI", "NO"], default="ALL")
    args = parser.parse_args()

    patches = build_patch_list(args.limit, args.only_label)
    viewer = PatchViewer(patches, load_master_df(), load_xgboost_df())
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
