"""
Sequenz-HMM: eine Fahrt als Pfad statt als Folge unabhaengiger Entscheidungen.

`aggregate_sequence` in retrieval.py summiert die Trefferlisten benachbarter
Frames auf. Das setzt voraus, dass DASSELBE Datenbankbild in mehreren Listen
auftaucht -- bei einer duennen Datenbank passiert das selten, und gemessen
hat es in Osnabrueck nichts gebracht (R@1 0.507 einzeln gegen 0.498 bei +-3).

Hier ein anderer Mechanismus: die Kandidaten duerfen je Frame verschieden
sein, sie muessen nur geometrisch zueinander passen. Zustaende sind die
Top-k eines Frames, Uebergaenge bewerten, ob der Abstand zweier Kandidaten
zu der Zeit passt, die zwischen den Frames vergangen ist. Ein Kandidat sechs
Kilometer abseits ist dann nicht deshalb unwahrscheinlich, weil ihn der
Nachbarframe nicht auch gefunden haette, sondern weil man in 0,17 s keine
sechs Kilometer faehrt.

    Emission   E[t, j] = beta * Aehnlichkeit(t, j)
    Uebergang  A[i, j] = -|d(i, j) - v * dt| / sigma
    Posterior  Forward-Backward ueber beide, je Frame normiert
    Pfad       Viterbi ueber dieselben Groessen

Die Query-Position wird NICHT benutzt -- sie ist die Ground Truth. Die
Geschwindigkeit v schaetzt `schaetze_geschwindigkeit` aus den
Datenbanksequenzen; deren Positionen sind Referenzmaterial, kein Wissen
ueber die Anfrage.

Das Ergebnis ist eine Umsortierung derselben Top-k, wie bei der
geometrischen Verifikation: R@k fuer das volle k bleibt unveraendert, R@1
bis R@(k-1) koennen sich aendern.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from src.geo import haversine_distance

# Grenzen fuer die Geschwindigkeitsschaetzung: was darueber liegt, ist ein
# Sprung im Zeitstempel oder ein GPS-Ausreisser, keine Fahrt.
MAX_LUECKE_S = 60.0
MAX_TEMPO_MS = 50.0        # 180 km/h


def schaetze_geschwindigkeit(metadata):
    """
    Median der Frame-zu-Frame-Geschwindigkeit in m/s aus den Sequenzen einer
    Metadatentabelle. Gedacht fuer die Datenbank -- nicht fuer die Anfragen,
    deren Positionen die Ground Truth sind.
    """
    seq = metadata["sequence_id"].to_numpy()
    zeit = metadata["captured_at"].to_numpy().astype("int64")
    lat = metadata["lat"].to_numpy().astype("float64")
    lon = metadata["lon"].to_numpy().astype("float64")

    ordnung = np.lexsort((zeit, seq))
    seq, zeit, lat, lon = seq[ordnung], zeit[ordnung], lat[ordnung], lon[ordnung]

    gleich = seq[1:] == seq[:-1]
    dt = (zeit[1:] - zeit[:-1]) / 1000.0
    d = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])

    gueltig = gleich & (dt > 0) & (dt <= MAX_LUECKE_S)
    if not gueltig.any():
        raise ValueError("Keine zusammenhaengenden Frames -- Geschwindigkeit nicht schaetzbar.")
    tempo = d[gueltig] / dt[gueltig]
    tempo = tempo[tempo <= MAX_TEMPO_MS]
    if not len(tempo):
        raise ValueError("Alle Frame-Abstaende unplausibel -- Geschwindigkeit nicht schaetzbar.")
    return float(np.median(tempo))


def _uebergang(lat_i, lon_i, lat_j, lon_j, dt_s, v_hat, sigma_m):
    """log-Gewicht je Kandidatenpaar: Abweichung von der plausiblen Strecke."""
    d = haversine_distance(lat_i[:, None], lon_i[:, None], lat_j[None, :], lon_j[None, :])
    return -np.abs(d - v_hat * dt_s) / sigma_m


def hmm_sequenz(sim, lat, lon, zeit_ms, v_hat, beta, sigma_m):
    """
    Forward-Backward und Viterbi ueber eine Sequenz.

    sim       (T, k) Aehnlichkeit je Frame und Kandidat
    lat, lon  (T, k) Position des Kandidaten (aus der Datenbank)
    zeit_ms   (T,)   Aufnahmezeit der Frames, aufsteigend

    Rueckgabe (posterior (T, k) in log, je Zeile normiert; pfad (T,) Spalte
    je Frame nach Viterbi).
    """
    T, k = sim.shape
    E = beta * np.asarray(sim, dtype=np.float64)
    if T == 1:
        # Eine Fahrt aus einem Bild ist kein Pfad -- dann bleibt die Emission.
        post = E - logsumexp(E, axis=1, keepdims=True)
        return post, np.argmax(E, axis=1)

    dt = np.diff(np.asarray(zeit_ms, dtype="int64")) / 1000.0
    A = [_uebergang(lat[t], lon[t], lat[t + 1], lon[t + 1], dt[t], v_hat, sigma_m)
         for t in range(T - 1)]

    # Forward
    alpha = np.empty((T, k))
    alpha[0] = E[0]
    for t in range(1, T):
        alpha[t] = E[t] + logsumexp(alpha[t - 1][:, None] + A[t - 1], axis=0)

    # Backward
    bwd = np.zeros((T, k))
    for t in range(T - 2, -1, -1):
        bwd[t] = logsumexp(A[t] + (E[t + 1] + bwd[t + 1])[None, :], axis=1)

    post = alpha + bwd
    post -= logsumexp(post, axis=1, keepdims=True)

    # Viterbi ueber dieselben Groessen
    delta = np.empty((T, k))
    zurueck = np.zeros((T, k), dtype=np.int64)
    delta[0] = E[0]
    for t in range(1, T):
        kombi = delta[t - 1][:, None] + A[t - 1]
        zurueck[t] = np.argmax(kombi, axis=0)
        delta[t] = E[t] + kombi[zurueck[t], np.arange(k)]

    pfad = np.empty(T, dtype=np.int64)
    pfad[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 1, 0, -1):
        pfad[t - 1] = zurueck[t, pfad[t]]
    return post, pfad


def hmm_rerank(indices, similarities, query, database, v_hat, beta=30.0, sigma_m=25.0):
    """
    Alle Query-Sequenzen durch das HMM. Gibt (neu_idx, neu_sim, viterbi_idx):
    die Top-k nach Posterior umsortiert, die zugehoerigen log-Posterior und
    je Anfrage die eine Zeile, die der Viterbi-Pfad waehlt.
    """
    from src.retrieval import sequence_windows

    db_lat = database["lat"].to_numpy().astype("float64")
    db_lon = database["lon"].to_numpy().astype("float64")
    q_zeit = query["captured_at"].to_numpy().astype("int64")

    neu_idx = np.empty_like(indices)
    neu_sim = np.empty(indices.shape, dtype=np.float32)
    viterbi = np.empty(len(indices), dtype=indices.dtype)

    fenster = sequence_windows(query)
    # sequence_windows gibt je Zeile (Sequenzzeilen, eigene Position); die
    # Position 0 trifft jede Sequenz genau einmal.
    for eintrag in fenster:
        zeilen, pos = eintrag
        if pos != 0:
            continue
        kand = indices[zeilen]                       # (T, k)
        post, pfad = hmm_sequenz(
            similarities[zeilen], db_lat[kand], db_lon[kand], q_zeit[zeilen],
            v_hat, beta, sigma_m)
        # stabil sortieren: bei gleichem Posterior bleibt die Reihenfolge der
        # Suche stehen, das HMM verschlechtert dann nichts.
        ordnung = np.argsort(-post, axis=1, kind="stable")
        neu_idx[zeilen] = np.take_along_axis(kand, ordnung, axis=1)
        neu_sim[zeilen] = np.take_along_axis(post, ordnung, axis=1).astype(np.float32)
        viterbi[zeilen] = kand[np.arange(len(zeilen)), pfad]
    return neu_idx, neu_sim, viterbi
