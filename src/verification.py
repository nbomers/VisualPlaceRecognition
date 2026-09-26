"""
Geometrische Verifikation: die gespeicherten Inlier in eine neue Reihenfolge
der Top-k uebersetzen.

experiments/geometric_verification.py zaehlt je Anfrage die RANSAC-Inlier
ihrer Top-k-Kandidaten und speichert NUR diese Zahlen, neben der
Trefferliste aus 06. Umsortiert wird hier -- dieselbe Funktion fuer das
Skript und fuer experiments/bootstrap_ci.py. So bleibt die gv-Zeile
nachrechenbar, ohne ein einziges Bild erneut zu matchen, und --min-inliers
laesst sich nachtraeglich pruefen.

    params = verification_from_record(record)        # None ausser bei gv-Zeilen
    indices = apply_verification(root, cfg, "megaloc", "none", indices, fingerprint, params)

Eigenes Modul und nicht Teil von retrieval.py: jene Datei geht in die
Code-Kennung der Auswertung ein (run_guard._CODE_DATEIEN), und jede
Aenderung dort erklaerte fuer run.py saemtliche 07-/08-Ergebnisse fuer
veraltet. Wie src/sequence_hmm.py ist das hier eine Nachbearbeitung, die
ihre Parameter in der eigenen JSON traegt.
"""

import numpy as np

from .paths import Paths
from .run_guard import require_fingerprint

# Was die Inlier-Zahlen selbst bestimmt. min_inliers gehoert NICHT dazu: es
# wirkt erst beim Umsortieren und kann sich aendern, ohne neu zu matchen.
GV_PARAMETER = ("top_k", "max_side", "max_keypoints", "ransac_px")

# Feld in der Ergebnis-JSON -> Name hier.
_FELDER = {"top_k": "verification_top_k", "min_inliers": "min_inliers",
           "max_side": "max_side", "max_keypoints": "max_keypoints",
           "ransac_px": "ransac_px"}


def verification_file(root, cfg, method, adapter="none", top_k=20):
    """Inlier-Matrix neben der Trefferliste aus 06 -- gitignored wie diese."""
    name = method if adapter in ("none", "None") else f"{method}_{adapter}"
    return Paths(cfg, root).retrieval / method / f"{name}_gv{int(top_k)}_inlier.npz"


def verification_fingerprint(fingerprint, params):
    """Embedding-Fingerabdruck plus die Parameter, die die Inlier bestimmen."""
    return {**fingerprint, "verification": {p: params[p] for p in GV_PARAMETER}}


def verification_from_record(record):
    """
    Die Verifikationsparameter einer gv-Zeile aus results/<stadt>/evaluation/,
    None fuer jede andere Zeile. Fehlt einer, stammt die JSON aus einer
    Version des Skripts, die ihre Inlier nicht gespeichert hat -- dann laut
    abbrechen statt mit geratenen Werten weiter.
    """
    if record.get("verification_top_k") is None:
        return None
    fehlt = [f for f in _FELDER.values() if record.get(f) is None]
    if fehlt:
        raise KeyError(
            f"{record.get('embedding_name')}: gv-Zeile ohne {', '.join(fehlt)} -- "
            "nicht nachrechenbar. experiments/geometric_verification.py neu laufen lassen.")
    return {p: record[f] for p, f in _FELDER.items()}


def rerank_by_inliers(indices, inliers, rows, min_inliers):
    """
    Die Top-k der Zeilen `rows` umsortieren: verifizierte Kandidaten
    (>= min_inliers) absteigend nach Inliern vorn, der Rest in alter
    Reihenfolge dahinter. Stabil -- Gleichstaende behalten die
    Deskriptor-Ordnung. Zeilen ausserhalb von `rows` und Plaetze hinter k
    bleiben unberuehrt.

    inliers: (len(rows), k), k <= indices.shape[1].
    """
    inliers = np.asarray(inliers, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    k = inliers.shape[1]
    neu = np.array(indices, copy=True)
    if not len(rows):
        return neu
    # Unverifizierte bekommen einen Schluessel hinter jedem verifizierten;
    # untereinander sind sie gleich und behalten so ihre Reihenfolge.
    schluessel = np.where(inliers >= min_inliers, -inliers, np.iinfo(np.int64).max)
    ordnung = np.argsort(schluessel, axis=1, kind="stable")
    neu[rows, :k] = np.take_along_axis(neu[rows, :k], ordnung, axis=1)
    return neu


def apply_verification(root, cfg, method, adapter, indices, fingerprint, params):
    """
    Die Trefferliste aus 06 so umsortieren, wie die gv-Zeile sie bewertet hat.
    fingerprint ist der Embedding-Fingerabdruck der Basis (load_retrieval
    prueft ihn fuer die Trefferliste, hier fuer die Inlier).
    """
    datei = verification_file(root, cfg, method, adapter, params["top_k"])
    require_fingerprint(datei, verification_fingerprint(fingerprint, params),
                        "Inlier der geometrischen Verifikation")
    with np.load(datei) as v:
        if int(v["fertig"]) != len(v["rows"]):
            raise RuntimeError(
                f"{datei.name}: Verifikation unvollstaendig "
                f"({int(v['fertig']):,} von {len(v['rows']):,} Anfragen).\n"
                "  experiments/geometric_verification.py mit denselben Parametern "
                "erneut starten -- es setzt dort fort.")
        return rerank_by_inliers(indices, v["inliers"], v["rows"], int(params["min_inliers"]))
