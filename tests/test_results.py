"""
Die versionierten Ergebnis-JSONs muessen zueinander passen: gleicher Split,
und der Bootstrap reproduziert jede 07-Zahl exakt.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "evaluation"
BOOT = ROOT / "experiments" / "results" / "bootstrap_ci.json"


def _laeufe():
    return [json.loads(p.read_text()) for p in sorted(EVAL.glob("*.json"))]


@pytest.mark.skipif(not EVAL.exists(), reason="keine Auswertungen")
def test_alle_laeufe_auf_demselben_split():
    laeufe = [r for r in _laeufe() if "auswertungen" in r]
    assert laeufe
    # Zwei Referenzen (database / database + train), je eine loesbare Menge
    for referenz in {r["n_database"] for r in laeufe}:
        gruppe = [r for r in laeufe if r["n_database"] == referenz]
        loesbar = {r["auswertungen"]["Alle Queries"]["schwellen"]["25"]["loesbar"] for r in gruppe}
        n_queries = {r["auswertungen"]["Alle Queries"]["n_queries"] for r in gruppe}
        assert len(loesbar) == 1 and len(n_queries) == 1, (referenz, loesbar, n_queries)
    for r in laeufe:
        for name in r["auswertungen"]:
            assert name.startswith(("Alle", "Hard", "Blickrichtung", "Nur Nicht")), name


@pytest.mark.skipif(not BOOT.exists(), reason="kein Bootstrap")
def test_bootstrap_reproduziert_07():
    boot = json.loads(BOOT.read_text())
    schwelle = str(int(boot["threshold_m"]))
    for r in _laeufe():
        if "auswertungen" not in r or r.get("variant") == "fullref":
            continue
        e = boot["encoder"].get(r["embedding_name"])
        assert e is not None, r["embedding_name"]
        soll = r["auswertungen"][boot["split"]]["schwellen"][schwelle]
        assert e["n_loesbar"] == soll["loesbar"]
        for k, wert in e["recall"].items():
            assert abs(wert["wert"] - soll["recall"][k]) < 1e-9, (r["embedding_name"], k)
            lo, hi = wert["ci"]
            assert lo <= wert["wert"] <= hi
