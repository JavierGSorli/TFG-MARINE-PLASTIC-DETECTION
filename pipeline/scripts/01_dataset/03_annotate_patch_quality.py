from __future__ import annotations

# --- project import bootstrap ---
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
# --- end project import bootstrap ---

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox

from src.common.config import DATASET_METADATA_GROUPED_PATH, DATASET_METADATA_PATH


BOOL_COLUMNS = [
    "tag_nube",
    "tag_nube_fina",
    "tag_estela",
    "tag_barco",
    "tag_costa",
    "tag_brillo_solar",
    "tag_agua_oscura",
    "tag_agua_muy_oscura",
    "tag_agua_turbia",
    "tag_posible_residuo",
    "tag_agua_limpia",
    "tag_filamento_visible",
    "tag_olas",
    "tag_espuma",
    "tag_corte_patch",
    "tag_sombra_nube",
]

TAG_LABELS = [
    ("nube", "tag_nube"),
    ("nube_fina", "tag_nube_fina"),
    ("estela", "tag_estela"),
    ("barco", "tag_barco"),
    ("costa", "tag_costa"),
    ("brillo_solar", "tag_brillo_solar"),
    ("agua_oscura", "tag_agua_oscura"),
    ("agua_muy_oscura", "tag_agua_muy_oscura"),
    ("agua_turbia", "tag_agua_turbia"),
    ("posible_residuo", "tag_posible_residuo"),
    ("agua_limpia", "tag_agua_limpia"),
    ("filamento_visible", "tag_filamento_visible"),
    ("olas", "tag_olas"),
    ("espuma", "tag_espuma"),
    ("corte_patch", "tag_corte_patch"),
    ("sombra_nube", "tag_sombra_nube"),
]

DECISIONS = ["", "accept", "reject", "uncertain"]
QUALITIES = ["", "good", "medium", "bad"]
CONFIDENCES = ["", "high", "medium", "low"]
IMAGE_SIZE = 256


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "si", "s"}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def make_rgb(data: np.ndarray, vmax: float = 0.1) -> np.ndarray:
    rgb = np.stack([data[3], data[2], data[1]], axis=-1)
    return np.clip(rgb / vmax, 0, 1)


def compute_fdi(data: np.ndarray) -> np.ndarray:
    b06, b08, b11 = data[5], data[7], data[9]
    return b08 - (b06 + (b11 - b06) * ((832.9 - 664.6) / (1613.7 - 664.6)) * 10)


def read_patch_data(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype("float32")


def lock_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, IMAGE_SIZE - 0.5)
    ax.set_ylim(IMAGE_SIZE - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")


class AnnotationViewer:
    def __init__(self, df: pd.DataFrame, indices: list[int], metadata_path: Path, show_indices: bool):
        if not indices:
            raise ValueError("No hay patches que anotar con los filtros indicados.")
        self.df = df
        self.indices = indices
        self.metadata_path = metadata_path
        self.show_indices = show_indices
        self.pos = 0
        self._loading = False

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("03_annotate_patch_quality")

        self.rgb_ax = self.fig.add_axes([0.04, 0.28, 0.29, 0.52])
        self.fdi_ax = self.fig.add_axes([0.38, 0.32, 0.24, 0.43])
        self.info_ax = self.fig.add_axes([0.66, 0.76, 0.31, 0.14])
        self.info_ax.axis("off")
        self.legend_ax = self.fig.add_axes([0.66, 0.58, 0.31, 0.15])
        self.legend_ax.axis("off")

        self.decision_ax = self.fig.add_axes([0.66, 0.44, 0.10, 0.10])
        self.quality_ax = self.fig.add_axes([0.78, 0.44, 0.12, 0.10])
        self.conf_ax = self.fig.add_axes([0.91, 0.44, 0.06, 0.10])
        self.tags_ax = self.fig.add_axes([0.66, 0.05, 0.31, 0.33])
        self.notes_ax = self.fig.add_axes([0.04, 0.10, 0.29, 0.04])
        self.prev_ax = self.fig.add_axes([0.38, 0.10, 0.08, 0.05])
        self.save_ax = self.fig.add_axes([0.48, 0.10, 0.08, 0.05])
        self.next_ax = self.fig.add_axes([0.58, 0.10, 0.08, 0.05])

        self.decision_radio = RadioButtons(self.decision_ax, DECISIONS)
        self.quality_radio = RadioButtons(self.quality_ax, QUALITIES)
        self.conf_radio = RadioButtons(self.conf_ax, CONFIDENCES)
        self.tags_check = CheckButtons(self.tags_ax, [label for label, _ in TAG_LABELS], [False] * len(TAG_LABELS))
        self.decision_ax.set_title("Decision", fontsize=10)
        self.quality_ax.set_title("Calidad imagen", fontsize=10)
        self.conf_ax.set_title("Confianza (high/med/low)", fontsize=10)
        self.tags_ax.set_title("Etiquetas", fontsize=10)
        self.notes_box = TextBox(self.notes_ax, "Notas", initial="")
        for text in self.tags_check.labels:
            text.set_fontsize(10)
        for text in [*self.quality_radio.labels, *self.decision_radio.labels]:
            text.set_fontsize(9)
        for text in self.conf_radio.labels:
            text.set_fontsize(10)
        self.prev_btn = Button(self.prev_ax, "Anterior")
        self.save_btn = Button(self.save_ax, "Guardar")
        self.next_btn = Button(self.next_ax, "Siguiente")

        self.decision_radio.on_clicked(lambda _label: self.save_current(silent=True))
        self.quality_radio.on_clicked(lambda _label: self.save_current(silent=True))
        self.conf_radio.on_clicked(lambda _label: self.save_current(silent=True))
        self.tags_check.on_clicked(lambda _label: self.save_current(silent=True))
        self.notes_box.on_submit(lambda _text: self.save_current(silent=True))
        self.prev_btn.on_clicked(lambda _event: self.step(-1))
        self.save_btn.on_clicked(lambda _event: self.save_current(silent=False))
        self.next_btn.on_clicked(lambda _event: self.step(1))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.render()

    @property
    def current_idx(self) -> int:
        return self.indices[self.pos]

    def current_row(self) -> pd.Series:
        return self.df.loc[self.current_idx]

    def _set_radio(self, radio: RadioButtons, labels: list[str], value: str) -> None:
        value = _clean_text(value)
        active = labels.index(value) if value in labels else 0
        if radio.value_selected != labels[active]:
            radio.set_active(active)

    def _set_checks(self, states: list[bool]) -> None:
        current = list(self.tags_check.get_status())
        for i, (want, have) in enumerate(zip(states, current)):
            if want != have:
                self.tags_check.set_active(i)

    def _load_controls_from_row(self, row: pd.Series) -> None:
        self._loading = True
        self._set_radio(self.decision_radio, DECISIONS, row.get("manual_decision", ""))
        self._set_radio(self.quality_radio, QUALITIES, row.get("image_quality", ""))
        self._set_radio(self.conf_radio, CONFIDENCES, row.get("manual_confidence", ""))
        self._set_checks([_as_bool(row.get(column, "")) for _label, column in TAG_LABELS])
        self.notes_box.set_val(_clean_text(row.get("notes", "")))
        self._loading = False

    def _selected_tags(self) -> tuple[list[str], dict[str, int]]:
        status = self.tags_check.get_status()
        tags = []
        values = {}
        for active, (label, column) in zip(status, TAG_LABELS):
            values[column] = int(active)
            if active:
                tags.append(label)
        return tags, values

    def save_current(self, silent: bool) -> None:
        if self._loading:
            return
        idx = self.current_idx
        tags, bool_values = self._selected_tags()
        self.df.loc[idx, "manual_decision"] = self.decision_radio.value_selected
        self.df.loc[idx, "image_quality"] = self.quality_radio.value_selected
        self.df.loc[idx, "manual_confidence"] = self.conf_radio.value_selected
        self.df.loc[idx, "scene_tags"] = ";".join(tags)
        self.df.loc[idx, "notes"] = self.notes_box.text
        for column, value in bool_values.items():
            self.df.loc[idx, column] = value
        self.df.loc[idx, "annotated_at"] = datetime.now().isoformat(timespec="seconds")
        self.df.to_csv(self.metadata_path, index=False)
        if not silent:
            print(f"Guardado: {self.current_row()['patch']} -> {self.metadata_path}")

    def step(self, delta: int) -> None:
        self.save_current(silent=True)
        self.pos = (self.pos + delta) % len(self.indices)
        self.render()

    def on_key_press(self, event) -> None:
        key = event.key
        if key in {"right", "n", "space"}:
            self.step(1)
        elif key in {"left", "p", "backspace"}:
            self.step(-1)
        elif key == "s":
            self.save_current(silent=False)
        elif key == "q" or key == "escape":
            self.save_current(silent=True)
            plt.close(self.fig)
        elif key in {"a", "r", "u"}:
            mapping = {"a": "accept", "r": "reject", "u": "uncertain"}
            self.decision_radio.set_active(DECISIONS.index(mapping[key]))
        elif key in {"g", "m", "b"}:
            mapping = {"g": "good", "m": "medium", "b": "bad"}
            self.quality_radio.set_active(QUALITIES.index(mapping[key]))
        elif key == "1":
            self.conf_radio.set_active(CONFIDENCES.index("low"))
        elif key == "2":
            self.conf_radio.set_active(CONFIDENCES.index("medium"))
        elif key == "3":
            self.conf_radio.set_active(CONFIDENCES.index("high"))

    def render(self) -> None:
        row = self.current_row()
        data = read_patch_data(row["patch_path"])
        rgb = make_rgb(data)

        self.rgb_ax.clear()
        lock_axis(self.rgb_ax)
        self.rgb_ax.imshow(rgb, extent=(-0.5, IMAGE_SIZE - 0.5, IMAGE_SIZE - 0.5, -0.5))
        self.rgb_ax.set_title("RGB B04/B03/B02")

        self.fdi_ax.clear()
        lock_axis(self.fdi_ax)
        self.fdi_ax.imshow(compute_fdi(data), cmap="magma", extent=(-0.5, IMAGE_SIZE - 0.5, IMAGE_SIZE - 0.5, -0.5))
        self.fdi_ax.set_title("FDI")

        if not self.show_indices:
            self.fdi_ax.set_visible(False)
        else:
            self.fdi_ax.set_visible(True)

        self.info_ax.clear()
        self.info_ax.axis("off")
        info = [
            f"[{self.pos + 1}/{len(self.indices)}]",
            str(row["patch"]),
            f"label={row.get('label', '')}  date={row.get('date', '')}",
            f"expected={row.get('expected_gt_px', '')}  mask={row.get('mask_gt_px', '')}",
            f"difficulty={row.get('original_difficulty', '')}",
        ]
        self.info_ax.text(0, 1, "\n".join(info), va="top", ha="left", fontsize=9)

        self.legend_ax.clear()
        self.legend_ax.axis("off")
        legend = [
            "Leyenda:",
            "Calidad imagen: good, medium, bad",
            "Confianza: high alta, medium media, low baja",
            "Etiquetas: nube, nube_fina, estela, barco, costa,",
            "brillo_solar, agua_oscura, agua_muy_oscura, agua_turbia,",
            "posible_residuo, agua_limpia, filamento_visible,",
            "olas, espuma, corte_patch, sombra_nube",
        ]
        self.legend_ax.text(0, 1, "\n".join(legend), va="top", ha="left", fontsize=8)

        self._load_controls_from_row(row)
        self.fig.suptitle("03_annotate_patch_quality", fontsize=14)
        self.fig.canvas.draw_idle()


def ensure_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "manual_decision": "",
        "manual_confidence": "",
        "image_quality": "",
        "scene_tags": "",
        "notes": "",
        "annotated_at": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    for column in BOOL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def main() -> None:
    default_metadata = DATASET_METADATA_GROUPED_PATH if DATASET_METADATA_GROUPED_PATH.exists() else DATASET_METADATA_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=default_metadata)
    parser.add_argument("--label", choices=["SI", "NO"], default=None)
    parser.add_argument("--only-unannotated", "--only_unannotated", action="store_true")
    parser.add_argument("--show-indices", "--show_indices", action="store_true", default=True)
    parser.add_argument("--rgb-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"No existe metadata: {args.metadata}")

    df = pd.read_csv(args.metadata).fillna("")
    df = ensure_annotation_columns(df)
    view = df.copy()
    if args.label:
        view = view[view["label"] == args.label]
    if args.only_unannotated:
        view = view[view["manual_decision"].astype(str).str.strip() == ""]
    if args.limit is not None:
        view = view.head(args.limit)

    viewer = AnnotationViewer(
        df=df,
        indices=list(view.index),
        metadata_path=args.metadata,
        show_indices=not args.rgb_only and args.show_indices,
    )
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
