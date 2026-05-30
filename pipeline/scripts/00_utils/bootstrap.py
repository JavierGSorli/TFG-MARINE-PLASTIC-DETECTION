"""Bootstrap helper para scripts organizados en subdirectorios de pipeline/scripts/.

Uso en scripts dentro de pipeline/scripts/0N_xxx/:
    from scripts.00_utils.bootstrap import setup_project_root
    setup_project_root()

O directamente copiar el bloque bootstrap:
    import sys
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
"""
from __future__ import annotations

import sys
from pathlib import Path


def setup_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root
