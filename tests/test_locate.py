"""
Die Suche des Locators -- ohne Torch und ohne FAISS.

Der Locator sucht mit numpy statt mit FAISS, weil Torch und FAISS auf macOS
nicht in denselben Prozess passen (tests/blockwise_check.py). Hier wird die
Suche gegen die vollstaendige Sortierung aller Aehnlichkeiten geprueft --
das ist genau, was faiss.IndexFlatIP liefert. FAISS selbst wird hier bewusst
nicht geladen: laeuft vorher tests/test_adapter_training.py mit Torch, riss
genau diese Kombination auf dem Mac frueher die ganze Testsuite mit.
"""

import numpy as np
import pytest

from src.locate import Locator


def _locator(vektoren):
    """Ein Locator ohne __init__ -- der wuerde Embeddings von der Platte und Torch laden."""
    loc = Locator.__new__(Locator)
    loc._vektoren = np.ascontiguousarray(vektoren, dtype=np.float32)
    return loc


def _normiert(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize("k", [1, 5, 10, 50])
def test_gleich_der_vollstaendigen_sortierung(k):
    rng = np.random.default_rng(0)
    db = _normiert(rng.normal(size=(3000, 64)))
    anfragen = _normiert(rng.normal(size=(7, 64)))
    idx, sims = _locator(db).search(anfragen, k)

    alle = anfragen.astype(np.float32) @ db.astype(np.float32).T
    erwartet = np.argsort(-alle, axis=1, kind="stable")[:, :k]
    assert idx.shape == sims.shape == (7, k)
    assert np.array_equal(idx, erwartet)
    assert np.allclose(sims, np.take_along_axis(alle, erwartet, axis=1))
    assert (np.diff(sims, axis=1) <= 0).all(), "je Zeile absteigend"


def test_k_groesser_als_die_datenbank():
    rng = np.random.default_rng(1)
    idx, sims = _locator(_normiert(rng.normal(size=(4, 8)))).search(
        _normiert(rng.normal(size=(2, 8))), 10)
    assert idx.shape == (2, 4)
    assert sorted(idx[0]) == [0, 1, 2, 3]


@pytest.mark.parametrize("teilmenge", [False, True])
def test_blockweise_gleich_im_speicher(monkeypatch, teilmenge):
    # Referenz zu gross fuer den Speicher: Block fuer Block aus der memmap,
    # die besten k laufend gemischt -- muss exakt dasselbe liefern.
    import src.locate as modul

    rng = np.random.default_rng(2)
    alle = _normiert(rng.normal(size=(2500, 32))).astype(np.float32)
    zeilen = np.sort(rng.choice(2500, 1700, replace=False)) if teilmenge else np.arange(2500)
    anfragen = _normiert(rng.normal(size=(5, 32)))

    monkeypatch.setattr(modul, "BLOCK", 300)
    loc = Locator.__new__(Locator)
    loc._vektoren, loc._zeilen = alle, zeilen
    idx, sims = loc.search(anfragen, 7)

    idx_ref, sims_ref = _locator(alle[zeilen]).search(anfragen, 7)
    assert np.array_equal(idx, idx_ref)
    assert np.allclose(sims, sims_ref)


def test_unbekannte_referenz():
    with pytest.raises(ValueError):
        Locator({"vpr": {}}, ".", referenz="irgendwas")


def test_locate_many_eine_suche_und_kaputte_fotos():
    import pandas as pd

    rng = np.random.default_rng(3)
    db = _normiert(rng.normal(size=(50, 16)))
    loc = _locator(db)
    loc.database = pd.DataFrame({"image_id": np.arange(50), "lat": 52.27 + np.arange(50) * 1e-4,
                                 "lon": np.full(50, 8.0), "split": ["database"] * 50})

    def embed(pfade):
        if pfade[0] == "kaputt.jpg":
            raise OSError("kein Bild")
        return db[[int(pfade[0].split(".")[0])]]           # "7.jpg" ist Referenzbild 7

    loc.embed = embed
    suchen = []
    original = loc.search
    loc.search = lambda v, k: (suchen.append(len(v)), original(v, k))[1]
    antworten, fehler = loc.locate_many(["7.jpg", "kaputt.jpg", "31.jpg"], k=3)
    assert suchen == [2], "eine Suche fuer alle Fotos"
    assert list(fehler) == ["kaputt.jpg"]
    assert antworten["7.jpg"]["treffer"][0]["image_id"] == 7
    assert antworten["31.jpg"]["treffer"][0]["image_id"] == 31
    assert antworten["7.jpg"]["treffer"][0]["split"] == "database"
