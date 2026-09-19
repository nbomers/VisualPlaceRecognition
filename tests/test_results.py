"""
Die versionierten Ergebnis-JSONs muessen zueinander passen: gleicher Split,
und der Bootstrap reproduziert jede 07-Zahl exakt.
"""

import json
import subprocess
from pathlib import Path

import pytest

from src.config import load_config, paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PATHS = paths(load_config(ROOT), ROOT)
EVAL = PATHS.evaluation
LOC = PATHS.localization
BOOT = PATHS.experiments / "bootstrap_ci.json"
BOOT_FULL = PATHS.experiments / "bootstrap_ci_fullref.json"


def _laeufe():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(EVAL.glob("*.json"))]


def _alle_ergebnis_jsons():
    """07 und 08, beide tragen code_version."""
    return sorted(EVAL.glob("*.json")) + sorted(LOC.glob("*.json"))


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True)


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
    boot = json.loads(BOOT.read_text(encoding="utf-8"))
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


@pytest.mark.skipif(_git("rev-parse", "--git-dir").returncode != 0,
                    reason="kein Git-Repository")
@pytest.mark.skipif(_git("rev-parse", "--is-shallow-repository").stdout.strip() == "true",
                    reason="flacher Klon -- aeltere Commits fehlen hier zu Recht")
def test_code_version_zeigt_auf_einen_auffindbaren_commit():
    """
    Jede Ergebnis-JSON nennt den Commit, auf dem sie entstanden ist. Zeigt der
    ins Leere, ist die Kennung wertlos -- man kann den Code, der die Zahlen
    erzeugt hat, nicht mehr auschecken.

    Das passiert lautlos durch `git pull --rebase`: der Rebase schreibt die
    Hashes um, nachdem die Zahlen gerechnet wurden. Deshalb: 07 und 08 auf dem
    Commit rechnen, der auch gepusht wird, und die JSONs mit ihm committen --
    nicht davor rebasen.
    """
    unauffindbar = {}
    for pfad in _alle_ergebnis_jsons():
        commit = (json.loads(pfad.read_text(encoding="utf-8")).get("code_version") or {}).get("commit")
        if not commit:
            continue
        if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            unauffindbar.setdefault(commit, []).append(pfad.name)

    assert not unauffindbar, (
        "Ergebnis-JSONs nennen Commits, die es im Repository nicht gibt:\n"
        + "\n".join(f"  {c}: {len(n)} Dateien, z.B. {n[0]}"
                     for c, n in sorted(unauffindbar.items()))
        + "\n\nDie Zahlen sind damit nicht mehr an ihren Code gebunden."
        "\n-> auf dem Rechner mit den Trefferlisten neu rechnen und committen:"
        "\n   python run.py --method all --adapter all --from 07"
        "\n   python run.py --method derived --adapter all --from 07"
        "\n   python experiments/full_reference.py"
    )


@pytest.mark.skipif(not BOOT_FULL.exists(), reason="kein Bootstrap fuer die volle Referenz")
def test_bootstrap_fullref_reproduziert_07():
    """Dasselbe fuer das zweite Protokoll -- bisher pruefte es niemand."""
    boot = json.loads(BOOT_FULL.read_text(encoding="utf-8"))
    schwelle = str(int(boot["threshold_m"]))
    geprueft = 0
    for r in _laeufe():
        if "auswertungen" not in r or r.get("variant") != "fullref":
            continue
        e = boot["encoder"].get(r["embedding_name"])
        assert e is not None, r["embedding_name"]
        soll = r["auswertungen"][boot["split"]]["schwellen"][schwelle]
        assert e["n_loesbar"] == soll["loesbar"], r["embedding_name"]
        for k, wert in e["recall"].items():
            assert abs(wert["wert"] - soll["recall"][k]) < 1e-9, (r["embedding_name"], k)
            lo, hi = wert["ci"]
            assert lo <= wert["wert"] <= hi
        geprueft += 1
    assert geprueft, "Keine fullref-Auswertung gefunden"
