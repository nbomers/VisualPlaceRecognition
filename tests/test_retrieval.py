"""
`localizable` entscheidet, welche Anfragen ueberhaupt zaehlen -- jede
Recall-Zahl der Experimente haengt daran. Die Funktion holt Kandidaten aus
einem KDTree statt aus der vollen Distanzmatrix; dieser Test nagelt fest,
dass beide Wege dasselbe ergeben.
"""

import numpy as np
import pandas as pd

from src.geo import haversine_distance
from src.retrieval import localizable


def _rahmen(lat, lon, erste_id):
    return pd.DataFrame({
        "image_id": np.arange(erste_id, erste_id + len(lat), dtype=np.int64),
        "lat": lat,
        "lon": lon,
    })


def _brute_force(query, database, threshold):
    """Die Rechnung, die localizable frueher gemacht hat: volle Matrix."""
    d = haversine_distance(
        query["lat"].to_numpy()[:, None], query["lon"].to_numpy()[:, None],
        database["lat"].to_numpy()[None, :], database["lon"].to_numpy()[None, :],
    )
    return (d <= threshold).any(axis=1)


def test_kdtree_ergibt_dieselbe_menge_wie_die_volle_matrix():
    rng = np.random.default_rng(7)
    # Rund um Osnabrueck, Streuung etwa 3 km -- genug Anfragen knapp
    # innerhalb und knapp ausserhalb jeder geprueften Schwelle.
    q = _rahmen(52.279 + rng.normal(0, 0.02, 400), 8.047 + rng.normal(0, 0.03, 400), 1)
    db = _rahmen(52.279 + rng.normal(0, 0.02, 900), 8.047 + rng.normal(0, 0.03, 900), 10_000)
    for schwelle in (5.0, 10.0, 25.0, 50.0, 100.0):
        assert np.array_equal(localizable(q, db, schwelle), _brute_force(q, db, schwelle)), schwelle


def test_randfaelle_genau_auf_der_schwelle():
    """Ein Nachbar exakt bei ~25 m darf nicht durch den Suchradius fallen."""
    # 25 m noerdlich entsprechen 25 / 111_195 Grad Breite.
    grad_je_meter = 1.0 / (6_371_000.0 * np.pi / 180.0)
    abstaende = np.array([0.0, 24.0, 24.9, 25.0, 25.1, 26.0, 200.0])
    q = _rahmen(np.full(len(abstaende), 52.0), np.full(len(abstaende), 8.0), 100)
    db = _rahmen(52.0 + abstaende * grad_je_meter, np.full(len(abstaende), 8.0), 20_000)
    # Je Anfrage derselbe Datenbanksatz -- loesbar ist deshalb ueberall True,
    # sobald irgendein Abstand unter der Schwelle liegt. Aussagekraeftiger ist
    # der Vergleich Zeile fuer Zeile gegen die volle Matrix.
    for schwelle in (5.0, 25.0, 25.05, 100.0):
        assert np.array_equal(localizable(q, db, schwelle), _brute_force(q, db, schwelle)), schwelle


def test_ohne_jeden_nachbarn():
    q = _rahmen(np.array([52.0, 52.1]), np.array([8.0, 8.1]), 200)
    db = _rahmen(np.array([53.0, 53.1]), np.array([9.0, 9.1]), 30_000)
    assert not localizable(q, db, 25.0).any()


# ----------------------------------------------------------------------
# Welche Dateien load_retrieval oeffnet
#
# Die fullref-Zeilen haben KEINE eigene Metadatendatei -- sie nutzen die des
# Basis-Encoders, nur die .npz traegt das Suffix. Wer das nachbaut statt
# retrieval_inputs zu fragen, sucht anyloc_fullref_metadata.parquet und
# findet sie nie. Genau so hat die Vorabpruefung in bootstrap_ci.py einmal
# alle 18 Encoder als "nicht vorhanden" gemeldet, obwohl alles da war.
# ----------------------------------------------------------------------

def _cfg_stadt():
    return {"city": "Osnabrück, Germany", "image_root": "/tmp"}


def test_fullref_nutzt_die_metadaten_des_basis_encoders():
    from src.retrieval import retrieval_inputs

    meta, npz = retrieval_inputs(".", _cfg_stadt(), "anyloc", "none")
    meta_full, npz_full = retrieval_inputs(".", _cfg_stadt(), "anyloc", "none",
                                           reference_splits=["database", "train"])
    assert meta_full == meta, "fullref hat keine eigene Metadatendatei"
    assert meta.name == "anyloc_metadata.parquet"
    assert npz.name == "anyloc_retrieval.npz"
    assert npz_full.name == "anyloc_fullref_retrieval.npz"


def test_adapter_im_namen_von_metadaten_und_trefferliste():
    from src.retrieval import retrieval_inputs

    meta, npz = retrieval_inputs(".", _cfg_stadt(), "megaloc", "linear")
    assert meta.name == "megaloc_linear_metadata.parquet"
    assert npz.name == "megaloc_linear_retrieval.npz"
    # Der Ordner bleibt der des echten Encoders, nicht der der Variante.
    assert meta.parent.name == "megaloc" and npz.parent.name == "megaloc"
