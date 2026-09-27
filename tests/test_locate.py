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
