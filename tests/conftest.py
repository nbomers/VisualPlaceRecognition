"""Projektwurzel im Importpfad, damit `from src ...` in jedem Test geht."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
