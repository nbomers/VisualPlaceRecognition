"""
Das Sequenz-HMM gegen eine unabhaengige Rechnung.

Forward-Backward und Viterbi sind Abkuerzungen fuer eine Summe bzw. ein
Maximum ueber ALLE Pfade durch die Kandidaten. Bei drei Frames mit drei
Kandidaten sind das 27 Pfade -- die zaehlt der Test einzeln durch und
vergleicht. Ein Vorzeichenfehler in der Rueckwaertsrekursion faellt so auf,
ein stiller Indexdreher auch.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from src.geo import haversine_distance
from src.sequence_hmm import (
    _uebergang,
    hmm_rerank,
    hmm_sequenz,
    schaetze_geschwindigkeit,
)

BETA, SIGMA = 30.0, 25.0


def _aufbau(rng, T=3, k=3):
    sim = rng.uniform(0.3, 0.9, (T, k))
    lat = 52.0 + rng.uniform(0, 0.01, (T, k))
    lon = 8.0 + rng.uniform(0, 0.01, (T, k))
    zeit = np.arange(T, dtype="int64") * 1000        # 1 s je Frame
    return sim, lat, lon, zeit


def _alle_pfade(sim, lat, lon, zeit, v_hat):
    """log-Gewicht jedes einzelnen Pfades -- die Definition, ohne Rekursion."""
    T, k = sim.shape
    dt = np.diff(zeit) / 1000.0
    A = [_uebergang(lat[t], lon[t], lat[t + 1], lon[t + 1], dt[t], v_hat, SIGMA)
         for t in range(T - 1)]
    for pfad in itertools.product(range(k), repeat=T):
        wert = sum(BETA * sim[t, pfad[t]] for t in range(T))
        wert += sum(A[t][pfad[t], pfad[t + 1]] for t in range(T - 1))
        yield pfad, wert


def test_posterior_ist_die_summe_ueber_alle_pfade():
    rng = np.random.default_rng(0)
    sim, lat, lon, zeit = _aufbau(rng)
    v_hat = 8.0
    pfade = list(_alle_pfade(sim, lat, lon, zeit, v_hat))

    T, k = sim.shape
    erwartet = np.empty((T, k))
    for t in range(T):
        for j in range(k):
            erwartet[t, j] = logsumexp([w for p, w in pfade if p[t] == j])
    erwartet -= logsumexp(erwartet, axis=1, keepdims=True)

    post, _ = hmm_sequenz(sim, lat, lon, zeit, v_hat, BETA, SIGMA)
    assert np.allclose(post, erwartet, atol=1e-9)


def test_viterbi_ist_der_beste_einzelne_pfad():
    rng = np.random.default_rng(1)
    for _ in range(20):
        sim, lat, lon, zeit = _aufbau(rng, T=4, k=3)
        v_hat = 8.0
        bester = max(_alle_pfade(sim, lat, lon, zeit, v_hat), key=lambda x: x[1])[0]
        _, pfad = hmm_sequenz(sim, lat, lon, zeit, v_hat, BETA, SIGMA)
        assert tuple(pfad) == bester


def test_ein_frame_bleibt_bei_der_emission():
    sim = np.array([[0.4, 0.9, 0.1]])
    lat, lon = np.full((1, 3), 52.0), np.full((1, 3), 8.0)
    post, pfad = hmm_sequenz(sim, lat, lon, np.array([0], dtype="int64"), 8.0, BETA, SIGMA)
    assert np.argmax(post[0]) == 1 and pfad[0] == 1
    assert np.isclose(logsumexp(post[0]), 0.0)


def test_geometrisch_unmoeglicher_ausreisser_wird_zurueckgestuft():
    # Drei Frames auf einer Linie, 8 m Abstand. In Frame 1 ist der
    # aehnlichste Kandidat sechs Kilometer entfernt -- die Verwechslung, die
    # 88 % der Fehlgriffe ausmacht. Der richtige Ort steht dort auf Platz 2.
    lat = np.array([[52.00000, 52.05000],
                    [52.00007, 52.05000],
                    [52.00014, 52.05000]])
    lon = np.full((3, 2), 8.0)
    sim = np.array([[0.80, 0.50],
                    [0.60, 0.75],      # Ausreisser fuehrt bei Einzelbetrachtung
                    [0.80, 0.50]])
    zeit = np.arange(3, dtype="int64") * 1000

    assert np.argmax(sim[1]) == 1                     # einzeln: falsch
    _, pfad = hmm_sequenz(sim, lat, lon, zeit, 8.0, BETA, SIGMA)
    assert tuple(pfad) == (0, 0, 0)                   # im Pfad: richtig
    post, _ = hmm_sequenz(sim, lat, lon, zeit, 8.0, BETA, SIGMA)
    assert np.argmax(post[1]) == 0


def test_kohaerenter_fehler_wird_nicht_repariert():
    # Alle drei Frames zeigen geschlossen auf die falsche Strasse. Der Pfad
    # dorthin ist genauso konsistent wie der richtige -- das HMM kann das
    # nicht sehen, und der Test haelt fest, dass es auch nichts erfindet.
    lat = np.array([[52.00000, 52.05000],
                    [52.00007, 52.05007],
                    [52.00014, 52.05014]])
    lon = np.full((3, 2), 8.0)
    sim = np.full((3, 2), 0.5)
    sim[:, 1] = 0.7                                   # die falsche Spur ist aehnlicher
    _, pfad = hmm_sequenz(sim, lat, lon, np.arange(3, dtype="int64") * 1000,
                          8.0, BETA, SIGMA)
    assert tuple(pfad) == (1, 1, 1)


def test_geschwindigkeit_aus_den_sequenzen():
    # Zwei Fahrten, 10 m je Sekunde, dazwischen eine Luecke von einer Stunde,
    # die nicht als 100-km/h-Sprung durchgehen darf.
    n = 20
    schritt = 10.0 / 111_320.0                        # 10 m in Grad Breite
    lat = np.r_[52.0 + np.arange(n) * schritt, 53.0 + np.arange(n) * schritt]
    zeit = np.r_[np.arange(n), 3600 + np.arange(n)].astype("int64") * 1000
    meta = pd.DataFrame({
        "sequence_id": ["a"] * n + ["b"] * n,
        "captured_at": zeit,
        "lat": lat,
        "lon": np.full(2 * n, 8.0),
    })
    assert schaetze_geschwindigkeit(meta) == pytest.approx(10.0, rel=0.01)


def test_rerank_sortiert_nur_um():
    rng = np.random.default_rng(2)
    n_db, n_q, k = 200, 30, 5
    database = pd.DataFrame({
        "lat": 52.0 + rng.uniform(0, 0.02, n_db),
        "lon": 8.0 + rng.uniform(0, 0.02, n_db),
    })
    query = pd.DataFrame({
        "sequence_id": np.repeat(["a", "b", "c"], 10),
        "captured_at": np.tile(np.arange(10, dtype="int64") * 1000, 3),
    })
    indices = rng.integers(0, n_db, (n_q, k))
    similarities = rng.uniform(0.3, 0.9, (n_q, k)).astype(np.float32)

    neu_idx, neu_sim, viterbi = hmm_rerank(indices, similarities, query, database, 8.0)

    for zeile in range(n_q):
        # Dieselbe Kandidatenmenge, nur andere Reihenfolge -- und der
        # Viterbi-Treffer muss einer der Kandidaten dieses Frames sein.
        assert set(neu_idx[zeile]) == set(indices[zeile])
        assert viterbi[zeile] in set(indices[zeile])
    assert np.all(np.diff(neu_sim, axis=1) <= 1e-6)    # absteigend sortiert


def test_rerank_faellt_bei_gleichstand_auf_die_suche_zurueck():
    # Alle Kandidaten am selben Ort und gleich aehnlich: das HMM hat keine
    # Praeferenz, dann darf es die Reihenfolge der Suche nicht durcheinander-
    # bringen.
    n_db, k = 10, 4
    database = pd.DataFrame({"lat": np.full(n_db, 52.0), "lon": np.full(n_db, 8.0)})
    query = pd.DataFrame({"sequence_id": ["a"] * 5,
                          "captured_at": np.arange(5, dtype="int64") * 1000})
    indices = np.tile(np.arange(k), (5, 1))
    similarities = np.full((5, k), 0.5, dtype=np.float32)

    neu_idx, _, _ = hmm_rerank(indices, similarities, query, database, 8.0)
    assert np.array_equal(neu_idx, indices)


def test_query_position_geht_nicht_ein():
    # Die Ground Truth darf das Ergebnis nicht beeinflussen: dieselbe Rechnung
    # mit voellig anderen Query-Koordinaten muss dasselbe liefern.
    rng = np.random.default_rng(3)
    n_db, k = 50, 4
    database = pd.DataFrame({
        "lat": 52.0 + rng.uniform(0, 0.02, n_db),
        "lon": 8.0 + rng.uniform(0, 0.02, n_db),
    })
    indices = rng.integers(0, n_db, (12, k))
    similarities = rng.uniform(0.3, 0.9, (12, k)).astype(np.float32)
    basis = pd.DataFrame({"sequence_id": ["a"] * 12,
                          "captured_at": np.arange(12, dtype="int64") * 1000})

    a = hmm_rerank(indices, similarities, basis.assign(lat=52.0, lon=8.0),
                   database, 8.0)[0]
    b = hmm_rerank(indices, similarities, basis.assign(lat=-33.0, lon=151.0),
                   database, 8.0)[0]
    assert np.array_equal(a, b)


def test_uebergang_belohnt_die_plausible_strecke():
    # 8 m in 1 s bei 8 m/s: Abweichung 0, das Maximum. Der zweite Kandidat
    # liegt 800 m weg und wird entsprechend abgestraft.
    lat_i, lon_i = np.array([52.0]), np.array([8.0])
    lat_j = np.array([52.0 + 8.0 / 111_320.0, 52.0 + 800.0 / 111_320.0])
    lon_j = np.array([8.0, 8.0])
    A = _uebergang(lat_i, lon_i, lat_j, lon_j, dt_s=1.0, v_hat=8.0, sigma_m=SIGMA)
    assert A[0, 0] > A[0, 1]
    assert A[0, 0] == pytest.approx(0.0, abs=1e-3)
    d = haversine_distance(52.0, 8.0, lat_j[1], 8.0)
    assert A[0, 1] == pytest.approx(-(d - 8.0) / SIGMA, rel=1e-6)


def test_hmm_greift_wo_das_aufsummieren_nichts_findet():
    """
    Der Grund, warum dieses Modul neben aggregate_sequence steht.

    `aggregate_sequence` summiert die Trefferlisten der Nachbarn ueber
    IDENTISCHE Datenbankzeilen. Wenn aufeinanderfolgende Frames andere
    Datenbankbilder finden -- bei dichter Referenz und kleinem k der
    Normalfall --, ueberlappt nichts, und die Summe aendert die Reihenfolge
    nicht. Das HMM verlangt keine Ueberlappung, sondern Geometrie.
    """
    from src.retrieval import aggregate_sequence, sequence_windows

    grad = 1.0 / 111_320.0
    T, k = 12, 2
    # Referenz alle 2 m entlang der Strasse, dazu eine Kopie 5 km weiter.
    n = 100
    db_lat = np.r_[52.0 + np.arange(n) * 2 * grad,
                   52.0 + 5000 * grad + np.arange(n) * 2 * grad]
    database = pd.DataFrame({"lat": db_lat, "lon": np.full(2 * n, 8.0)})
    query = pd.DataFrame({"sequence_id": ["a"] * T,
                          "captured_at": (np.arange(T) * 1000).astype("int64")})

    # Frame t steht bei 3,3 m * t -- jeder Frame trifft eine andere Zeile.
    echt = np.round(np.arange(T) * 3.3 / 2).astype(int)
    decoy = n + echt
    indices = np.stack([echt, decoy], axis=1)
    similarities = np.full((T, k), 0.80, dtype=np.float32)
    similarities[:, 1] = 0.70
    similarities[::4, 1] = 0.95                  # jeder vierte Frame kippt

    kaputt = np.flatnonzero(np.argmax(similarities, axis=1) == 1)
    assert len(kaputt) == 3                      # 0, 4, 8

    # Aufsummieren: keine gemeinsame Zeile, also keine Stuetze.
    agg_idx, _ = aggregate_sequence(indices, similarities,
                                    sequence_windows(query), 1, k)
    assert list(agg_idx[kaputt, 0]) == list(decoy[kaputt])

    # HMM: der Sprung ueber 5 km ist in einer Sekunde nicht zu fahren.
    neu_idx, _, viterbi = hmm_rerank(indices, similarities, query, database,
                                     v_hat=3.3, beta=BETA, sigma_m=SIGMA)
    assert list(neu_idx[kaputt, 0]) == list(echt[kaputt])
    assert np.array_equal(viterbi, echt)
