"""
Namensnennung je Bild: CC BY-SA 4.0 verlangt den Urheber jedes Bildes, und
QUELLEN.md ist die Stelle, an der das Repository ihn nennt -- als Link auf
die Bildseite. Geht dabei ein Eintrag verloren, faellt das niemandem auf.
"""

import json

from src.quellen import LINK, notiere_quellen


def test_eintraege_bleiben_und_werden_verlinkt(tmp_path):
    notiere_quellen(tmp_path, "b.png", [3, 1, 3])
    md = notiere_quellen(tmp_path, "a.png", [2]).read_text(encoding="utf-8")
    daten = json.loads((tmp_path / "quellen.json").read_text(encoding="utf-8"))
    # Beide Abbildungen, sortiert; Doppelte raus, Reihenfolge der Abbildung bleibt.
    assert daten == {"a.png": ["2"], "b.png": ["3", "1"]}
    assert md.index("## a.png") < md.index("## b.png")
    for i in ("1", "2", "3"):
        assert f"[{i}]({LINK.format(i)})" in md


def test_neu_speichern_ersetzt_nur_die_eigene_abbildung(tmp_path):
    notiere_quellen(tmp_path, "a.png", [1, 2])
    notiere_quellen(tmp_path, "b.png", [5])
    notiere_quellen(tmp_path, "a.png", [9])
    daten = json.loads((tmp_path / "quellen.json").read_text(encoding="utf-8"))
    assert daten == {"a.png": ["9"], "b.png": ["5"]}
