"""
Fuenf Anfragen, zehn Datenbankbilder, handgerechnete Erwartung -- fuer die
Standardauswertung, die Blickrichtung und "Hard". Rechnet ein Aenderung an
src/evaluation.py an dieser Rechnung vorbei, faellt es hier auf.
"""

import numpy as np
import pandas as pd

from src.evaluation import evaluate_retrieval, standard_evaluations

LAT0, LON0 = 52.0, 8.0
M_PER_DEG = 111_320.0            # Meter je Breitengrad, fuer Nord-Versaetze
TAG_MS = 86_400_000


def _lat(meter_nord):
    return LAT0 + meter_nord / M_PER_DEG


def _cfg():
    return {
        "vpr": {"split_seed": 1, "max_heading_diff_deg": 90.0},
        "retrieval": {"thresholds": [10, 25], "k_values": [1, 5], "min_days_apart": 180},
    }


def _daten():
    # Datenbank: zehn Bilder alle 100 m nach Norden, Kompass 0 -- ausser
    # db1 und db2, die nach Sueden schauen. db0 ist 400 Tage aelter,
    # db4 stammt von einem anderen Fotografen.
    db = pd.DataFrame({
        "image_id": np.arange(10),
        "lat": [_lat(100 * i) for i in range(10)],
        "lon": [LON0] * 10,
        "compass_angle": [0.0, 180.0, 180.0] + [0.0] * 7,
        "creator_id": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        "captured_at": [1000 * TAG_MS - 400 * TAG_MS] + [1000 * TAG_MS] * 9,
        "is_pano": [False] * 10,
        "sequence_id": ["d"] * 10,
    })
    # Anfragen: Abstand zum naechsten Datenbankbild 4 / 15 / 8 / 30 / 4 m.
    q = pd.DataFrame({
        "image_id": np.arange(100, 105),
        "lat": [_lat(4), _lat(115), _lat(208), _lat(330), _lat(404)],
        "lon": [LON0] * 5,
        "compass_angle": [0.0] * 5,
        "creator_id": [1] * 5,
        "captured_at": [1000 * TAG_MS] * 5,
        "is_pano": [False] * 5,
        "sequence_id": ["q"] * 5,
    })
    # Trefferlisten: q0, q2, q3 finden ihr Bild auf Platz 1; q1 auf Platz 2;
    # q4 erst auf Platz 4.
    indices = np.array([
        [0, 9, 8, 7, 6],
        [5, 1, 9, 8, 7],
        [2, 9, 8, 7, 6],
        [3, 9, 8, 7, 6],
        [9, 8, 7, 4, 6],
    ])
    return q, db, indices


def _recall(befund, schwelle, k):
    return befund["schwellen"][str(schwelle)]["recall"][str(k)]


def test_standard():
    q, db, idx = _daten()
    b = evaluate_retrieval(idx, q, db, _cfg(), verbose=False)
    # 10 m: loesbar q0, q2, q4 -- Treffer@1 bei q0 und q2, @5 bei allen dreien
    assert b["schwellen"]["10"]["loesbar"] == 3
    assert _recall(b, 10, 1) == 2 / 3
    assert _recall(b, 10, 5) == 1.0
    # 25 m: q1 kommt dazu (15 m), ihr Treffer steht auf Platz 2
    assert b["schwellen"]["25"]["loesbar"] == 4
    assert _recall(b, 25, 1) == 0.5
    assert _recall(b, 25, 5) == 1.0
    assert b["n_queries"] == 5


def test_blickrichtung_und_hard():
    q, db, idx = _daten()
    befunde = standard_evaluations(idx, q, db, _cfg(), verbose=False)
    blick = befunde["Blickrichtung: Treffer nur bei <= 90 Grad Abweichung"]
    # db1 und db2 schauen nach Sueden: q1 und q2 verlieren ihre einzige Referenz
    assert blick["schwellen"]["25"]["loesbar"] == 2
    assert _recall(blick, 25, 1) == 0.5
    assert _recall(blick, 25, 5) == 1.0

    hard = befunde["Hard: anderer creator_id ODER > 180 Tage Abstand"]
    # db0 ist 400 Tage alt, db4 von einem anderen Fotografen -- nur q0 und q4 bleiben
    assert hard["schwellen"]["25"]["loesbar"] == 2
    assert _recall(hard, 25, 1) == 0.5
    assert _recall(hard, 25, 5) == 1.0
    assert "Nur Nicht-Panorama-Queries" not in befunde


def test_zu_wenig_treffer_je_anfrage():
    q, db, idx = _daten()
    cfg = _cfg()
    cfg["retrieval"]["k_values"] = [1, 20]
    try:
        evaluate_retrieval(idx, q, db, cfg, verbose=False)
    except ValueError as e:
        assert "Recall@20" in str(e)
    else:
        raise AssertionError("k groesser als die Trefferliste muss abbrechen")


def test_panorama_auswertung():
    """Mit Panoramen im Datensatz kommt die Auswertung ohne sie dazu."""
    q, db, idx = _daten()
    q.loc[[1, 3], "is_pano"] = True          # q1 (loesbar bei 25 m) und q3 (nicht loesbar)
    befunde = standard_evaluations(idx, q, db, _cfg(), verbose=False)
    ohne = befunde["Nur Nicht-Panorama-Queries"]
    assert ohne["n_queries"] == 3
    # ohne q1: loesbar q0, q2, q4 -- Treffer@1 bei q0 und q2
    assert ohne["schwellen"]["25"]["loesbar"] == 3
    assert _recall(ohne, 25, 1) == 2 / 3
    # die Standardauswertung bleibt, wie sie war
    assert befunde["Alle Queries"]["schwellen"]["25"]["loesbar"] == 4
