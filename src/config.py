"""
Projektwurzel, config.yaml und die Namen, die alle Stufen teilen.

Jedes Notebook und jedes Skript beginnt mit denselben Zeilen: Wurzel suchen,
config.yaml lesen, Verfahren und Adapter aus der Umgebung uebernehmen, den
Embedding-Namen bilden. Hier stehen sie einmal.

Bootstrap in einem Notebook -- vier Zeilen, mehr braucht es nicht:

    import sys
    from pathlib import Path
    PROJECT_ROOT = next(d for d in (Path.cwd(), *Path.cwd().parents)
                        if (d / "config.yaml").exists())
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import load_config, paths
    CFG = load_config(PROJECT_ROOT)
    PATHS = paths(CFG, PROJECT_ROOT)
"""

import os
from pathlib import Path

import yaml


def find_project_root(start=None):
    """Erstes Verzeichnis aufwaerts, das eine config.yaml enthaelt."""
    start = Path(start or Path.cwd())
    for d in (start, *start.parents):
        if (d / "config.yaml").exists():
            return d
    raise FileNotFoundError("Projektroot nicht gefunden (keine config.yaml aufwaerts)")


def load_config(root=None):
    """
    config.yaml lesen.

    run.py setzt Verfahren und Adapter ueber VPR_METHOD und VPR_ADAPTER --
    frueher wurde dafuer die config.yaml ueberschrieben, eine versionierte
    Datei als Zustandsspeicher. Ohne gesetzte Variablen gilt die Datei.
    """
    root = Path(root) if root else find_project_root()
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    cfg["vpr"]["method"] = os.environ.get("VPR_METHOD", cfg["vpr"]["method"])
    cfg["vpr"]["adapter"] = os.environ.get(
        "VPR_ADAPTER", cfg["vpr"].get("adapter", "none")
    )
    return cfg


def paths(cfg, root=None):
    """Alle Ablageorte fuer die konfigurierte Stadt -- siehe src/paths.py."""
    from .paths import Paths

    return Paths(cfg, Path(root) if root else find_project_root())


def embedding_name(cfg):
    """`clip` ohne Adapter, `clip_linear` mit -- der Name jeder Artefaktdatei."""
    method = cfg["vpr"]["method"]
    adapter = cfg["vpr"].get("adapter", "none")
    return method if adapter in ("none", "None") else f"{method}_{adapter}"
