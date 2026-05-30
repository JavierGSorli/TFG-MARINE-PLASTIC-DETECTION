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

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import rasterio

from src.common.config import (
    DEBRIS_CLASS,
    EXTERNAL_B09_COPY_B8A_OUT,
    EXTERNAL_B09_INTERP_OUT,
    EXTERNAL_B09_ZERO_OUT,
    HYBRID_MASKS_ROOT,
    INDICES_NO_WATER_OUT,
    INDICES_WATER_OUT,
    PATCHES_DIR,
    RF_MODE_DIRS,
    SAM_CALIBRATED_MASKS_OUT,
    SAM_PHASE_OUT,
    UNET_CALIBRATED_MASKS_OUT,
    UNET_OUT,
    VIZ_PHASE_OUT,
)
from src.common.pipeline_utils import infer_label_from_name, iter_patch_files


OUT_DIR = VIZ_PHASE_OUT / "all_mask_models"
IMAGE_SIZE = 256
MASK_COLOR = (0.93, 0.22, 0.18)
MASK_BG_VALUE = 0.86
PANEL_TITLE_SIZE = 9
SUPTITLE_SIZE = 11
MODEL_COLORS = {
    "GT": (0.15, 0.85, 0.25),
}

PAGE_GROUPS = [
    "Base",
    "Calibrated",
    "Indices",
    "External",
    "Hybrid",
]


def make_rgb(data: np.ndarray, vmax: float = 0.10) -> np.ndarray:
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    return np.clip(rgb / vmax, 0, 1)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.65) -> np.ndarray:
    out = rgb.copy()
    active = mask.astype(bool)
    for ch, val in enumerate(color):
        out[:, :, ch] = np.where(active, alpha * val + (1.0 - alpha) * rgb[:, :, ch], rgb[:, :, ch])
    return out


def mask_only(mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    canvas = np.full((mask.shape[0], mask.shape[1], 3), MASK_BG_VALUE, dtype=np.float32)
    active = mask.astype(bool)
    for ch, val in enumerate(color):
        canvas[:, :, ch] = np.where(active, val, canvas[:, :, ch])
    return canvas


def read_patch_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
    return make_rgb(data)


def read_binary_mask(path: Path, mode: str) -> np.ndarray | None:
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1)
    if mode == "debris_class":
        return (arr == DEBRIS_CLASS).astype(bool)
    return (arr > 0).astype(bool)


def align_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape == shape:
        return mask
    h = min(mask.shape[0], shape[0])
    w = min(mask.shape[1], shape[1])
    out = np.zeros(shape, dtype=bool)
    out[:h, :w] = mask[:h, :w]
    return out


def build_model_specs() -> list[tuple[str, Path, str, str]]:
    specs: list[tuple[str, Path, str, str]] = [
        ("UNet argmax", UNET_OUT, "debris_class", "Base"),
        ("RF full", RF_MODE_DIRS["full"], "debris_class", "Base"),
        ("RF no_texture", RF_MODE_DIRS["no_texture"], "debris_class", "Base"),
        ("RF indices_only", RF_MODE_DIRS["indices_only"], "debris_class", "Base"),
        ("RF bands_only", RF_MODE_DIRS["bands_only"], "debris_class", "Base"),
        ("SAM binary", SAM_PHASE_OUT / "binario", "binary", "Base"),
        ("UNet calibrated", UNET_CALIBRATED_MASKS_OUT, "binary", "Calibrated"),
        ("RF full calibrated", RF_MODE_DIRS["full"] / "calibrated_masks", "binary", "Calibrated"),
        ("RF no_texture calibrated", RF_MODE_DIRS["no_texture"] / "calibrated_masks", "binary", "Calibrated"),
        ("RF indices_only calibrated", RF_MODE_DIRS["indices_only"] / "calibrated_masks", "binary", "Calibrated"),
        ("RF bands_only calibrated", RF_MODE_DIRS["bands_only"] / "calibrated_masks", "binary", "Calibrated"),
        ("SAM calibrated", SAM_CALIBRATED_MASKS_OUT, "binary", "Calibrated"),
        ("FDI", INDICES_NO_WATER_OUT, "binary", "Indices"),
        ("NDVI", INDICES_NO_WATER_OUT, "binary", "Indices"),
        ("FDI+NDVI", INDICES_NO_WATER_OUT, "binary", "Indices"),
        ("FDI_mask", INDICES_WATER_OUT, "binary", "Indices"),
        ("NDVI_mask", INDICES_WATER_OUT, "binary", "Indices"),
        ("FDI+NDVI_mask", INDICES_WATER_OUT, "binary", "Indices"),
        ("Water mask", INDICES_WATER_OUT, "binary", "External"),
        ("External b09_zero default", EXTERNAL_B09_ZERO_OUT / "masks", "binary", "External"),
        ("External b09_zero calibrated", EXTERNAL_B09_ZERO_OUT / "calibrated_masks", "binary", "External"),
        ("External b09_copy_b8a default", EXTERNAL_B09_COPY_B8A_OUT / "masks", "binary", "External"),
        ("External b09_copy_b8a calibrated", EXTERNAL_B09_COPY_B8A_OUT / "calibrated_masks", "binary", "External"),
        ("External b09_interpolate_b8a_b11 calibrated", EXTERNAL_B09_INTERP_OUT / "calibrated_masks", "binary", "External"),
        ("Hybrid sensitive", HYBRID_MASKS_ROOT / "profile_sensitive", "binary", "Hybrid"),
        ("Hybrid balanced", HYBRID_MASKS_ROOT / "profile_balanced", "binary", "Hybrid"),
        ("Hybrid conservative", HYBRID_MASKS_ROOT / "profile_conservative", "binary", "Hybrid"),
    ]
    return specs


def page_groups_from_specs(specs: list[tuple[str, Path, str, str]]) -> list[tuple[str, list[tuple[str, Path, str, str]]]]:
    grouped: list[tuple[str, list[tuple[str, Path, str, str]]]] = []
    for group_name in PAGE_GROUPS:
        group_specs = [spec for spec in specs if spec[3] == group_name]
        if group_specs:
            grouped.append((group_name, group_specs))
    return grouped


def lock_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, IMAGE_SIZE - 0.5)
    ax.set_ylim(IMAGE_SIZE - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")


def mask_path_for(model_name: str, base_dir: Path, stem: str) -> Path:
    if model_name.startswith("UNet"):
        return base_dir / f"{stem}_mask.tif"
    if model_name.startswith("RF"):
        return base_dir / f"{stem}_mask.tif"
    if model_name == "Water mask":
        return base_dir / f"{stem}_water_output_mask.tif"
    if model_name.startswith("FDI+NDVI"):
        return base_dir / f"{stem}_fdi_ndvi_mask.tif"
    if model_name.startswith("FDI"):
        return base_dir / f"{stem}_fdi_mask.tif"
    if model_name.startswith("NDVI"):
        return base_dir / f"{stem}_ndvi_mask.tif"
    if model_name == "SAM binary":
        return base_dir / f"{stem}_sam_debris_mask.tif"
    if model_name == "SAM calibrated":
        return base_dir / f"{stem}_sam_debris_mask.tif"
    return base_dir / f"{stem}_mask.tif"


def collect_patch_panels(
    patch_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, list[tuple[str, np.ndarray | None]]]]]:
    stem = patch_path.stem
    rgb = read_patch_rgb(patch_path)
    shape = rgb.shape[:2]

    gt_path = PATCHES_DIR / f"{stem}_mask.tif"
    gt_mask = align_mask(read_binary_mask(gt_path, "binary"), shape)
    if gt_mask is None:
        gt_mask = np.zeros(shape, dtype=bool)

    grouped_panels: list[tuple[str, list[tuple[str, np.ndarray | None]]]] = []
    for group_name, group_specs in page_groups_from_specs(build_model_specs()):
        page_panels: list[tuple[str, np.ndarray | None]] = []
        for model_name, base_dir, mode, _group in group_specs:
            mask = align_mask(read_binary_mask(mask_path_for(model_name, base_dir, stem), mode), shape)
            page_panels.append((model_name, mask))
        grouped_panels.append((group_name, page_panels))
    return rgb, gt_mask, grouped_panels


def save_patch_figure(patch_path: Path, save_dir: Path, page_index: int = 0) -> Path:
    rgb, gt_mask, grouped_panels = collect_patch_panels(patch_path)
    total_pages = max(1, len(grouped_panels))
    page_index = min(page_index, total_pages - 1)
    page_name, page_panels = grouped_panels[page_index]
    total_panels = 2 + len(page_panels)
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.5 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()

    axes_arr[0].imshow(rgb)
    axes_arr[0].set_title("RGB", fontsize=PANEL_TITLE_SIZE)
    axes_arr[1].imshow(mask_only(gt_mask, MODEL_COLORS["GT"]))
    axes_arr[1].set_title(f"GT real\npx={int(gt_mask.sum())}", fontsize=PANEL_TITLE_SIZE)
    for ax in axes_arr[:2]:
        lock_axis(ax)

    for idx, (model_name, mask) in enumerate(page_panels, start=2):
        ax = axes_arr[idx]
        if mask is None:
            ax.imshow(np.full_like(rgb, MASK_BG_VALUE))
            ax.text(0.5, 0.5, "No disponible", ha="center", va="center", color="black", fontsize=9, transform=ax.transAxes, bbox={"facecolor": "white", "alpha": 0.8, "pad": 6})
            ax.set_title(model_name, fontsize=PANEL_TITLE_SIZE)
        else:
            ax.imshow(mask_only(mask, MASK_COLOR))
            ax.set_title(f"{model_name}\npx={int(mask.sum())}", fontsize=PANEL_TITLE_SIZE)
        lock_axis(ax)

    for ax in axes_arr[total_panels:]:
        ax.axis("off")

    fig.suptitle(f"{patch_path.name} | grupo {page_name} | pagina {page_index + 1}/{total_pages}", fontsize=SUPTITLE_SIZE, y=0.965)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.08, wspace=0.08, hspace=0.24)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{patch_path.stem}_all_masks_page_{page_index + 1}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


class MaskGalleryViewer:
    def __init__(self, patches: list[Path], save_dir: Path):
        if not patches:
            raise ValueError("No hay patches para visualizar.")
        self.patches = patches
        self.save_dir = save_dir
        self.index = 0
        self.page_index = 0
        self.fig = plt.figure(figsize=(16, 11.5))
        self.fig.canvas.manager.set_window_title("03_visualize_all_mask_models")
        self.axes = [self.fig.add_subplot(2, 4, idx + 1) for idx in range(8)]
        self.fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.10, wspace=0.08, hspace=0.24)
        self.prev_ax = self.fig.add_axes([0.18, 0.01, 0.12, 0.045])
        self.next_ax = self.fig.add_axes([0.32, 0.01, 0.12, 0.045])
        self.prev_page_ax = self.fig.add_axes([0.48, 0.01, 0.12, 0.045])
        self.next_page_ax = self.fig.add_axes([0.62, 0.01, 0.12, 0.045])
        self.save_ax = self.fig.add_axes([0.76, 0.01, 0.12, 0.045])
        self.prev_btn = Button(self.prev_ax, "Anterior")
        self.next_btn = Button(self.next_ax, "Siguiente")
        self.prev_page_btn = Button(self.prev_page_ax, "Pag -")
        self.next_page_btn = Button(self.next_page_ax, "Pag +")
        self.save_btn = Button(self.save_ax, "Guardar")
        self.prev_btn.on_clicked(lambda _: self.step(-1))
        self.next_btn.on_clicked(lambda _: self.step(1))
        self.prev_page_btn.on_clicked(lambda _: self.step_page(-1))
        self.next_page_btn.on_clicked(lambda _: self.step_page(1))
        self.save_btn.on_clicked(lambda _: self.save_current())
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.render()

    def on_key_press(self, event):
        if event.key in {"right", "d", "n", "space"}:
            self.step(1)
        elif event.key in {"left", "a", "p", "backspace"}:
            self.step(-1)
        elif event.key in {"up", "w"}:
            self.step_page(-1)
        elif event.key in {"down", "x"}:
            self.step_page(1)
        elif event.key == "s":
            self.save_current()
        elif event.key in {"q", "escape"}:
            plt.close(self.fig)

    def step(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.patches)
        self.page_index = 0
        self.render()

    def step_page(self, delta: int) -> None:
        _, _, grouped_panels = collect_patch_panels(self.patches[self.index])
        total_pages = max(1, len(grouped_panels))
        self.page_index = (self.page_index + delta) % total_pages
        self.render()

    def save_current(self) -> None:
        out_path = save_patch_figure(self.patches[self.index], self.save_dir, self.page_index)
        print(f"[OK] {out_path}")

    def render(self) -> None:
        patch_path = self.patches[self.index]
        rgb, gt_mask, grouped_panels = collect_patch_panels(patch_path)
        total_pages = max(1, len(grouped_panels))
        self.page_index = min(self.page_index, total_pages - 1)
        page_name, page_panels = grouped_panels[self.page_index]
        total_panels = 2 + len(page_panels)
        for ax in self.axes:
            ax.clear()

        self.axes[0].imshow(rgb)
        self.axes[0].set_title("RGB", fontsize=PANEL_TITLE_SIZE)
        self.axes[1].imshow(mask_only(gt_mask, MODEL_COLORS["GT"]))
        self.axes[1].set_title(f"GT real\npx={int(gt_mask.sum())}", fontsize=PANEL_TITLE_SIZE)
        for ax in self.axes[:2]:
            lock_axis(ax)

        for idx, (model_name, mask) in enumerate(page_panels, start=2):
            ax = self.axes[idx]
            if mask is None:
                ax.imshow(np.full_like(rgb, MASK_BG_VALUE))
                ax.text(0.5, 0.5, "No disponible", ha="center", va="center", color="black", fontsize=8.5, transform=ax.transAxes, bbox={"facecolor": "white", "alpha": 0.8, "pad": 6})
                ax.set_title(model_name, fontsize=PANEL_TITLE_SIZE)
            else:
                ax.imshow(mask_only(mask, MASK_COLOR))
                ax.set_title(f"{model_name}\npx={int(mask.sum())}", fontsize=PANEL_TITLE_SIZE)
            lock_axis(ax)

        for ax in self.axes[total_panels:]:
            ax.axis("off")

        label = infer_label_from_name(patch_path.name)
        self.fig.suptitle(
            f"[{self.index + 1}/{len(self.patches)}] {patch_path.name} | {label} | grupo {page_name} | pagina {self.page_index + 1}/{total_pages}",
            fontsize=SUPTITLE_SIZE,
            y=0.965,
        )
        self.fig.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualizar RGB, RGB+GT y máscaras separadas de todos los modelos segmentadores.")
    parser.add_argument("--patch", default=None, help="Nombre exacto del patch o stem.")
    parser.add_argument("--only-label", choices=["SI", "NO", "ALL"], default="ALL")
    parser.add_argument("--limit", type=int, default=10, help="Máximo de patches a renderizar cuando no se indique --patch.")
    parser.add_argument("--output-subdir", default="all_models_masks", help="Subcarpeta dentro de visualizations para guardar PNGs.")
    parser.add_argument("--save-only", action="store_true", help="Guardar PNGs sin abrir visor interactivo.")
    return parser.parse_args()


def select_patches(args: argparse.Namespace) -> list[Path]:
    patches = iter_patch_files(PATCHES_DIR)
    if args.only_label != "ALL":
        patches = [p for p in patches if infer_label_from_name(p.name) == args.only_label]
    if args.patch:
        patches = [p for p in patches if p.name == args.patch or p.stem == args.patch]
    if args.patch:
        return patches
    return patches[: args.limit]


def main() -> None:
    args = parse_args()
    save_dir = OUT_DIR / args.output_subdir
    patches = select_patches(args)
    if not patches:
        print("No se encontraron patches para visualizar.")
        return

    if args.save_only:
        saved = []
        for patch_path in patches:
            _, _, grouped_panels = collect_patch_panels(patch_path)
            total_pages = max(1, len(grouped_panels))
            for page_index in range(total_pages):
                out_path = save_patch_figure(patch_path, save_dir, page_index)
                saved.append(out_path)
                print(f"[OK] {out_path}")
        print(f"\nGeneradas {len(saved)} figuras en: {save_dir}")
        return

    viewer = MaskGalleryViewer(patches, save_dir)
    plt.show()


if __name__ == "__main__":
    main()
