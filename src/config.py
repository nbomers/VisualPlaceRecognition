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
import sys
from pathlib import Path

import yaml


def find_project_root(start=None):
    """Erstes Verzeichnis aufwaerts, das eine config.yaml enthaelt."""
    start = Path(start or Path.cwd())
    for d in (start, *start.parents):
        if (d / "config.yaml").exists():
            return d
    raise FileNotFoundError("Projektroot nicht gefunden (keine config.yaml aufwaerts)")


def apply_env(cfg):
    """
    Umgebungsvariablen stechen die Datei: VPR_CITY, VPR_METHOD, VPR_ADAPTER.

    Frueher wurde dafuer die config.yaml ueberschrieben -- eine versionierte
    Datei als Zustandsspeicher, die nach jedem Lauf als geaendert dastand.
    Fuer Verfahren und Adapter macht run.py das laengst so; `city` fehlte,
    obwohl src/paths.py jede Stadt ohnehin in ihren eigenen Zweig legt.

    Praktischer Nutzen: eine zweite Stadt laeuft neben einem laufenden
    Durchgang im selben Klon. Die Notebooks lesen config.yaml bei JEDER
    Zellenausfuehrung neu -- die Datei mittendrin umzustellen wuerde einem
    laufenden run.py unter den Fuessen die Stadt wechseln.

        VPR_CITY="Würzburg, Germany" jupyter lab notebooks/01_mapillary_coverage.ipynb
    """
    cfg["city"] = os.environ.get("VPR_CITY", cfg["city"])
    cfg["vpr"]["method"] = os.environ.get("VPR_METHOD", cfg["vpr"]["method"])
    cfg["vpr"]["adapter"] = os.environ.get(
        "VPR_ADAPTER", cfg["vpr"].get("adapter", "none")
    )
    return cfg


# Aelteste Version, unter der die Testsuite hier durchlief. Darunter faellt
# es sonst irgendwo weiter unten auseinander -- bei einem Notebook, das
# run.py ohne Konsole ausfuehrt, mit einem SyntaxError aus einem Modul, das
# mit dem eigentlichen Problem nichts zu tun hat.
MIN_PYTHON = (3, 11)


def require_python(min_version=MIN_PYTHON):
    """Frueh und mit Namen abbrechen statt spaet und kryptisch."""
    if sys.version_info[:2] >= min_version:
        return
    ist = ".".join(str(x) for x in sys.version_info[:3])
    soll = ".".join(str(x) for x in min_version)
    raise RuntimeError(
        f"Python {ist} ist zu alt -- gebraucht wird mindestens {soll}.\n"
        f"  Interpreter: {sys.executable}\n"
        "  conda env create -f environment.yml && conda activate vpr\n"
        f"  oder:  uv venv --python {soll} && uv pip install -r requirements.txt"
    )


def load_config(root=None):
    """config.yaml lesen, dann die Umgebung darueberlegen (siehe apply_env).

    Jede Stufe geht hier durch -- deshalb steht die Versionspruefung hier und
    nicht in jedem Notebook einzeln.
    """
    require_python()
    root = Path(root) if root else find_project_root()
    return apply_env(yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")))


def paths(cfg, root=None):
    """Alle Ablageorte fuer die konfigurierte Stadt -- siehe src/paths.py."""
    from .paths import Paths

    return Paths(cfg, Path(root) if root else find_project_root())


def embedding_name(cfg):
    """`clip` ohne Adapter, `clip_linear` mit -- der Name jeder Artefaktdatei."""
    method = cfg["vpr"]["method"]
    adapter = cfg["vpr"].get("adapter", "none")
    return method if adapter in ("none", "None") else f"{method}_{adapter}"
