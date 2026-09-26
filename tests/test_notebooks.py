"""
Kein versioniertes Notebook traegt Zellausgaben.

Ausgaben entfernt nbstripout beim Staging -- aber nur, wenn der Filter auf
dem Rechner eingerichtet ist UND .gitattributes denselben Namen nennt. Genau
das passte einmal nicht zusammen: .gitattributes verlangte einen Filter, den
`nbstripout --install` gar nicht anlegt, und Git liess die Notebooks dann
ohne jede Warnung ungefiltert durch.

Geprueft wird die Fassung im Index (`git show :pfad`), nicht die
Arbeitskopie: ein in Jupyter gespeichertes Notebook hat lokal immer
Ausgaben, und das ist in Ordnung, solange nbstripout sie beim Staging
entfernt. Im CI sind Index und Checkout dasselbe.
"""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


pytestmark = pytest.mark.skipif(_git("rev-parse", "--git-dir").returncode != 0,
                                reason="kein Git-Repository")


def _notebooks():
    return [p for p in _git("ls-files", "*.ipynb").stdout.split("\n") if p]


def test_es_gibt_versionierte_notebooks():
    assert _notebooks(), "keine .ipynb im Index -- falsches Verzeichnis?"


@pytest.mark.parametrize("pfad", _notebooks())
def test_keine_zellausgaben(pfad):
    nb = json.loads(_git("show", f":{pfad}").stdout)
    mit_ausgabe = [i for i, c in enumerate(nb["cells"])
                   if c["cell_type"] == "code" and (c.get("outputs") or c.get("execution_count"))]
    assert not mit_ausgabe, (
        f"{pfad}: Zellen {mit_ausgabe} tragen Ausgaben im Index.\n"
        "nbstripout ist auf diesem Rechner nicht eingerichtet:\n"
        "  nbstripout --install --attributes .gitattributes\n"
        f"  git add {pfad}"
    )
