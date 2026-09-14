"""
Was jedes Experiment zuerst braucht: die Projektwurzel im Importpfad, die
config.yaml geladen, die Ablageorte der Stadt (src/paths.py) und den
Ergebnisordner darin. Ein Import statt sechs Zeilen je Skript:

    from _common import CFG, PATHS, RESULTS, ROOT
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, paths  # noqa: E402

CFG = load_config(ROOT)
PATHS = paths(CFG, ROOT)
RESULTS = PATHS.experiments
