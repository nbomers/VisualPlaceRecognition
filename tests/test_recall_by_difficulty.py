"""Die Schwierigkeitsklassen muessen jede loesbare Anfrage genau einmal zaehlen."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import recall_by_difficulty as rb  # noqa: E402
from src.retrieval import localizable  # noqa: E402


def _rahmen(lat, lon, tage, erste_id):
    n = len(lat)
    return pd.DataFrame({
        "image_id": np.arange(erste_id, erste_id + n, dtype=np.int64),
        "lat": lat, "lon": lon,
        "captured_at": (np.asarray(tage) * 86_400_000).astype("int64"),
        "compass_angle": np.zeros(n), "creator_id": np.arange(n) % 3,
    })


def test_jede_loesbare_anfrage_in_genau_einer_klasse():
    rng = np.random.default_rng(3)
    db = _rahmen(52.279 + rng.normal(0, 0.003, 600), 8.047 + rng.normal(0, 0.004, 600),
                 rng.uniform(0, 800, 600), 10_000)
    # Anfragen um Datenbankbilder herum, bis knapp ueber die Schwelle, mit
    # gebrochenen Tagesabstaenden (7,5 Tage lagen frueher zwischen 0-7 und 8-30).
    q = _rahmen(db["lat"].to_numpy()[:300] + rng.normal(0, 0.00015, 300),
                db["lon"].to_numpy()[:300] + rng.normal(0, 0.0002, 300),
                db["captured_at"].to_numpy()[:300] / 86_400_000 + rng.uniform(0, 40, 300), 1)
    loesbar = localizable(q, db, 25.0)
    nachbarn, tage, _, _ = rb.query_properties(q, db, 25.0, 90.0)
    assert loesbar.sum() > 100
    for werte, klassen in ((nachbarn, rb.NACHBAR_KLASSEN), (tage, rb.TAGE_KLASSEN)):
        treffer = sum(((werte >= lo) & (werte <= hi)).astype(int) for lo, hi in klassen)
        assert (treffer[loesbar] == 1).all()
