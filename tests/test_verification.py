"""
Die geometrische Verifikation speichert nur Inlier-Zahlen; umsortiert wird in
src.verification.rerank_by_inliers -- vom Skript nach dem Lauf und vom Bootstrap
beim Nachrechnen. Stimmt die Funktion nicht mit der Regel ueberein, die das
Skript frueher je Anfrage in einer Schleife anwandte, rechnen beide Seiten
dieselbe falsche Zahl und die Kontrolle gegen 07 merkt nichts. Deshalb hier
gegen genau diese Schleife.
"""

import numpy as np
import pytest

from src.verification import rerank_by_inliers, verification_from_record


def _schleife(indices, inliers, rows, min_inliers):
    """Die Umsortierung, wie sie geometric_verification.py je Anfrage machte."""
    neu = indices.copy()
    k = inliers.shape[1]
    for pos, qi in enumerate(rows):
        werte = inliers[pos]
        ok = werte >= min_inliers
        vorn = np.flatnonzero(ok)[np.argsort(-werte[ok], kind="stable")]
        neu[qi, :k] = indices[qi, np.r_[vorn, np.flatnonzero(~ok)]]
    return neu


@pytest.mark.parametrize("min_inliers", [0, 1, 15, 60])
def test_gleich_der_schleife(min_inliers):
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 10_000, size=(300, 50))
    rows = np.sort(rng.choice(300, 120, replace=False))
    # Viele Gleichstaende und viele Nullen -- genau da entscheidet die Stabilitaet.
    inliers = rng.choice([0, 0, 0, 12, 15, 15, 40, 40, 80], size=(len(rows), 20))
    assert np.array_equal(rerank_by_inliers(indices, inliers, rows, min_inliers),
                          _schleife(indices, inliers, rows, min_inliers))


def test_nur_die_verifizierten_zeilen_und_plaetze_aendern_sich():
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 1_000, size=(40, 50))
    rows = np.array([3, 7, 20])
    inliers = rng.integers(0, 100, size=(3, 10))
    neu = rerank_by_inliers(indices, inliers, rows, 15)
    andere = np.setdiff1d(np.arange(40), rows)
    assert np.array_equal(neu[andere], indices[andere])
    assert np.array_equal(neu[:, 10:], indices[:, 10:])
    # Umsortiert, nicht ersetzt.
    for qi in rows:
        assert sorted(neu[qi, :10]) == sorted(indices[qi, :10])


def test_ohne_verifizierten_kandidaten_bleibt_die_reihenfolge():
    indices = np.arange(60).reshape(2, 30)
    neu = rerank_by_inliers(indices, np.full((2, 20), 14), np.array([0, 1]), 15)
    assert np.array_equal(neu, indices)


def test_parameter_aus_der_json():
    assert verification_from_record({"embedding_name": "megaloc", "variant": "none"}) is None
    voll = {"embedding_name": "megaloc_gv20", "verification_top_k": 20, "min_inliers": 15,
            "max_side": 640, "max_keypoints": 1024, "ransac_px": 3.0}
    assert verification_from_record(voll) == {
        "top_k": 20, "min_inliers": 15, "max_side": 640,
        "max_keypoints": 1024, "ransac_px": 3.0}
    # Eine gv-Zeile ohne die Parameter, die die Inlier bestimmen, ist nicht
    # nachrechenbar -- laut abbrechen statt mit falschen Annahmen weiter.
    with pytest.raises(KeyError, match="max_side"):
        verification_from_record({k: v for k, v in voll.items() if k != "max_side"})
