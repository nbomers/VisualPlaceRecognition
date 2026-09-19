"""
Die versionierten Ergebnis-JSONs müssen gültiges JSON sein.

Klingt selbstverständlich, war es nicht: `json.dumps` schreibt für
float("nan") ein bares `NaN` in die Datei. Pythons Parser nimmt das, weil
`allow_nan` standardmäßig an ist -- jeder strenge Parser lehnt es ab, denn
RFC 8259 kennt weder NaN noch Infinity. Betroffen waren
recall_by_district_*.json (0/0 bei einem Stadtteil ohne Anfragen).

Der Test liest deshalb mit einem parse_constant, der wirft. Genau so würde
jq, JavaScript oder Go die Datei sehen.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _wirf(konstante):
    raise AssertionError(f"{konstante} ist nach RFC 8259 kein gültiges JSON")


def _alle_json():
    ordner = [ROOT / "results", ROOT / "experiments" / "results"]
    return sorted(p for o in ordner if o.is_dir() for p in o.rglob("*.json"))


@pytest.mark.skipif(not _alle_json(), reason="keine Ergebnis-JSONs")
@pytest.mark.parametrize("pfad", _alle_json(), ids=lambda p: p.name)
def test_streng_parsebar(pfad):
    json.loads(pfad.read_text(encoding="utf-8"), parse_constant=_wirf)


@pytest.mark.skipif(not _alle_json(), reason="keine Ergebnis-JSONs")
def test_kein_nan_im_klartext():
    """Zweiter Weg zum selben Befund -- unabhängig vom Parser."""
    treffer = [p.relative_to(ROOT) for p in _alle_json()
               if "NaN" in p.read_text(encoding="utf-8")]
    assert not treffer, (
        "NaN im Klartext:\n  " + "\n  ".join(str(t) for t in treffer)
        + "\n-> im erzeugenden Skript None statt float('nan') schreiben"
    )
