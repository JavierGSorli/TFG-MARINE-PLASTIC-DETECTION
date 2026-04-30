from __future__ import annotations

import subprocess
from pathlib import Path


def iter_patch_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*.tif")
        if "_mask" not in path.stem and not path.name.startswith("_")
    )


def infer_label_from_name(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    if len(parts) >= 2 and parts[1] in {"SI", "NO"}:
        return parts[1]
    return "SI" if "_SI_" in filename else "NO"


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} no encontrado: {path}")


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
    )


def tail_lines(text: str, max_lines: int = 12) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])
