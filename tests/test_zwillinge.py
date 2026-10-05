"""Zwillingsfahrten: Erkennung und R@1 ohne Zwillinge -- an einem Fall von Hand."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import zwillinge  # noqa: E402

CFG = {
    "retrieval": {"thresholds": [25], "k_values": [1, 2], "min_days_apart": 180},
    "vpr": {"split_seed": 0, "uncertain_radius_m": 25.0},
}
T0 = 1_526_740_341_177          # ms, wie captured_at bei Mapillary
GRAD_JE_M = 1 / 111_320


def _bild(konto, zeit_ms, nord_m):
    return {"creator_id": konto, "captured_at": zeit_ms,
            "lat": 52.27 + nord_m * GRAD_JE_M, "lon": 7.99}


def _faelle():
    # Anfrage 0: Zwilling (selbes Konto, 0,2 s, 0 m) auf Platz 1, ein echtes
    #            Referenzbild eines anderen Kontos 10 m daneben auf Platz 2.
    # Anfrage 1: nur ein Zwilling in Reichweite -- ohne ihn nicht loesbar.
    # Anfrage 2: selbes Konto, aber ein Jahr spaeter -- kein Zwilling.
    query = pd.DataFrame([_bild("a", T0, 0), _bild("b", T0, 5_000), _bild("c", T0, 10_000)])
    referenz = pd.DataFrame([
        _bild("a", T0 + 200, 0),                      # 0: Zwilling von Anfrage 0
        _bild("x", T0 - 86_400_000 * 30, 10),          # 1: echte Referenz fuer Anfrage 0
        _bild("b", T0 - 150, 5_000),                   # 2: Zwilling von Anfrage 1
        _bild("c", T0 + 86_400_000 * 365, 10_000),     # 3: selbes Konto, ein Jahr spaeter
        _bild("y", T0, 20_000),                        # 4: weit weg
    ])
    indices = np.array([[0, 1], [2, 4], [3, 4]])
    return query, referenz, indices


def test_zwilling_ist_konto_und_zeitfenster():
    query, referenz, _ = _faelle()
    z = zwillinge.zwilling_fn(query, referenz, 60_000)
    assert z(np.array([0, 0, 1, 2]), np.array([0, 1, 2, 3])).tolist() == [True, False, True, False]
    hat = zwillinge.anteil_mit_zwilling(query, referenz, 60_000, 25.0)
    assert hat.tolist() == [True, True, False]


def test_ohne_zwillinge_rueckt_auf_und_nimmt_die_loesbarkeit():
    query, referenz, indices = _faelle()
    z = zwillinge.auswerten(query, referenz, indices, CFG, 60_000)
    # Standard: alle drei loesbar, alle drei Top-1 richtig.
    assert z["loesbar"] == 3
    assert z["recall"]["1"] == 1.0
    # Ohne Zwillinge: Anfrage 1 hat keine Referenz mehr; Anfrage 0 rueckt auf
    # das echte Bild (10 m) vor; Anfrage 2 bleibt, wie sie war.
    assert z["loesbar_ohne"] == 2
    assert z["recall_ohne"]["1"] == 1.0
    assert abs(z["anteil_top1_ist_zwilling"] - 2 / 3) < 1e-12
