"""
Namensnennung fuer Abbildungen mit Mapillary-Bildinhalt.

CC BY-SA 4.0 verlangt, den Urheber JEDES Bildes zu nennen; "Bilder von
Mapillary" allein reicht nicht. Abschnitt 3(a)(2) der Lizenz laesst dafuer
einen Link auf eine Seite zu, die den Urheber nennt -- die Bildseite bei
Mapillary tut das. Wer eine solche Abbildung speichert, meldet hier, welche
Bilder darin stecken, in der Reihenfolge links nach rechts, oben nach unten:

    notiere_quellen(FIGURE_DIR, "eigenplaces_erfolg.png", [anfrage_id, *treffer_ids])

Daraus entstehen neben den Abbildungen quellen.json (die Daten) und
QUELLEN.md (dasselbe als Links, auf GitHub lesbar). Die creator_id steht
bewusst nicht darin -- siehe NOTICE.md, Personenbezug.
"""

import json
from pathlib import Path

LINK = "https://www.mapillary.com/app/?pKey={}"


def notiere_quellen(ordner, abbildung, image_ids):
    """Bilder einer Abbildung eintragen und QUELLEN.md neu schreiben."""
    ordner = Path(ordner)
    ordner.mkdir(parents=True, exist_ok=True)
    daten_pfad = ordner / "quellen.json"
    try:
        daten = json.loads(daten_pfad.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        daten = {}
    # Doppelte raus, Reihenfolge bleibt: dasselbe Bild zweimal in einer
    # Abbildung braucht nur einen Link.
    daten[abbildung] = list(dict.fromkeys(str(int(i)) for i in image_ids))
    daten = dict(sorted(daten.items()))
    daten_pfad.write_text(json.dumps(daten, indent=1) + "\n", encoding="utf-8")

    zeilen = [
        "# Bildquellen",
        "",
        "Die Abbildungen in diesem Ordner enthalten Bilder von "
        "[Mapillary](https://www.mapillary.com), lizenziert unter "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). "
        "Jeder Link führt zur Seite des Bildes; dort nennt Mapillary den Urheber. "
        "Reihenfolge wie in der Abbildung: links nach rechts, oben nach unten.",
        "",
        "Erzeugt von `src/quellen.py` beim Speichern der Abbildung — nicht von Hand ändern.",
        "",
    ]
    for name, ids in daten.items():
        zeilen += [f"## {name}", "", " · ".join(f"[{i}]({LINK.format(i)})" for i in ids), ""]
    (ordner / "QUELLEN.md").write_text("\n".join(zeilen), encoding="utf-8")
    return ordner / "QUELLEN.md"
